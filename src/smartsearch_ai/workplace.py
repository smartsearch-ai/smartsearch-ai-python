"""Workplace: search, answers and chat over a workspace, plus member and source-grant administration."""

from __future__ import annotations

import codecs
import copy
import json
import math
import re
import threading
import weakref
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Iterator, Mapping

import httpx
from pydantic import BaseModel, ConfigDict, Field, field_validator

from ._auth import Credential
from ._util import (
    MAX_CODE_LENGTH, MAX_MESSAGE_LENGTH, auth_secrets, canonical_uuid, is_canonical_uuid, is_iso_control,
    safe_raw_body, safe_request_id, safe_text, url_encode, visible_ascii,
)
from .errors import (
    AuthenticationError, RequestTimeoutError, SmartSearchError, TransportError, error_type,
)

_REQUEST_ID_HEADER = "X-Request-Id"


# --- request models ---------------------------------------------------------------------------

class Mode(str, Enum):
    """Buffered mode: ranked sources only, or a generated answer."""

    RETRIEVAL_ONLY = "retrieval_only"
    ANSWER = "answer"


class MemoryMode(str, Enum):
    """``standard`` stores no long-term memory; ``agentic`` keeps memory for the calling identity."""

    STANDARD = "standard"
    AGENTIC = "agentic"


class Output(str, Enum):
    FULL = "full"
    COMPACT = "compact"
    MINIMAL = "minimal"


class CachePolicy(str, Enum):
    USE_AND_WRITE = "use_and_write"
    USE_ONLY = "use_only"
    BYPASS = "bypass"


class _Model(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", use_enum_values=False)

    def to_dict(self) -> dict[str, Any]:
        """The JSON body sent to the server."""
        return self.model_dump(mode="json", exclude_none=True)


class QueryOptions(_Model):
    """Options for Search, Query and Chat. Defaults mirror the server's; streaming is set by the
    ``stream_*`` methods, never here."""

    output: Output | None = None
    include_sources: bool = True
    max_subqueries: int = Field(4, gt=0)
    max_iterations: int = Field(4, gt=0)
    max_context_documents: int = Field(12, gt=0)
    cache_policy: CachePolicy = CachePolicy.USE_AND_WRITE


class WorkspaceQueryRequest(_Model):
    """Request body for Workplace Search, Query and Chat, buffered or streaming. Immutable.

    Server-owned attributes (tenant, groups, roles, security keys, guardrails, embeddings, memory
    contents, impersonation) do not exist here: the server derives them from the caller.

    Args:
        query: required natural-language query.
        mode: :class:`Mode`, default ``RETRIEVAL_ONLY``.
        session_id: Chat session to continue.
        agent_id, source_ids, project_ids: narrow the run to existing configuration.
        filters: a direct clause or ``{"all": [clauses]}``; clause types ``term``, ``terms``,
            ``match``, ``match_phrase`` and ``range`` over business fields (``title``, ``content``,
            ``source_type``, ``published_at``... or ``metadata.*``). Protected fields are rejected.
        options: :class:`QueryOptions`.
        memory_mode: :class:`MemoryMode`, or None for the agent's default. Memory is keyed by the
            authenticated identity, so one service key shares one memory.
    """

    query: str
    mode: Mode = Mode.RETRIEVAL_ONLY
    filters: dict[str, Any] | None = None
    options: QueryOptions = Field(default_factory=QueryOptions)
    session_id: str | None = None
    agent_id: str | None = None
    source_ids: tuple[str, ...] | None = None
    project_ids: tuple[str, ...] | None = None
    memory_mode: MemoryMode | None = None

    @field_validator("query")
    @classmethod
    def _query(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("Workspace query must not be blank")
        return value

    @field_validator("filters", mode="before")
    @classmethod
    def _filters(cls, value: Any) -> Any:
        return None if value is None else canonical_filters(value)


class MemberRole(str, Enum):
    MEMBER = "MEMBER"
    OWNER = "OWNER"


class WorkspaceMemberRequest(_Model):
    """One membership assignment; use :meth:`canonical` with the principal ID SmartSearch returned."""

    role: MemberRole
    principal_id: str | None = None
    user_id: str | None = None

    @classmethod
    def canonical(cls, principal_id: str, role: MemberRole = MemberRole.MEMBER) -> WorkspaceMemberRequest:
        return cls(role=role, principal_id=canonical_uuid(principal_id, "principal_id"))

    @classmethod
    def legacy(cls, user_id: str, role: MemberRole = MemberRole.MEMBER) -> WorkspaceMemberRequest:
        """Sends ``user_id`` as given, without deriving a principal from it."""
        if not isinstance(user_id, str) or not user_id.strip() or any(is_iso_control(c) for c in user_id):
            raise ValueError("user_id is required and must not contain control characters")
        return cls(role=role, user_id=user_id)


class WorkspacePrincipalGrantRequest(_Model):
    """Complete desired source grants for a principal at ``expected_revision``. Empty lists remove
    grants. A 409 means the revision moved: read the current grant and reconcile explicitly."""

    display_label: str | None = None
    groups: tuple[str, ...]
    roles: tuple[str, ...]
    security_keys: tuple[str, ...]
    expected_revision: int = Field(ge=0)


# --- filter grammar (mirrors the server's caller-filter support) ------------------------------

_SEARCH_TYPES = {"term", "terms", "match", "match_phrase", "range"}
_BUSINESS_FIELDS = {"source_type", "title", "content", "snippet", "source_url", "author", "published_at", "updated_at"}
_RANGE_FIELDS = {"published_at", "updated_at"}
_PROTECTED = (
    "workspaceid", "tenantid", "sourceid", "sourceids", "documentid", "projectid", "projectids", "hidden",
    "deleted", "groups", "roles", "owners", "securitykeys", "ssdocumentsecurity", "documentsecurity",
    "disabledocumentsecurity", "overridedocumentsecuritykeys",
)


def canonical_filters(filters: Any) -> dict[str, Any]:
    if not isinstance(filters, Mapping):
        raise ValueError("filters must be an object")
    if len(filters) == 1 and "all" in filters:
        clauses = filters["all"]
        if not isinstance(clauses, (list, tuple)) or not clauses:
            raise ValueError("filters.all must be a non-empty clause list")
        return {"all": [_clause(c) for c in clauses]}
    return {"all": [_clause(filters)]}


def _protected(field: str) -> bool:
    if not field.strip():
        return True
    compact = re.sub(r"[^a-z0-9]", "", field.strip().lower())
    return "security" in compact or any(p in compact for p in _PROTECTED)


def _clause(candidate: Any) -> dict[str, Any]:
    if not isinstance(candidate, Mapping):
        raise ValueError("Filter clause must be an object")
    raw_field = candidate.get("field")
    if isinstance(raw_field, str) and _protected(raw_field):
        raise ValueError(f"Protected filter field is not caller-selectable: {raw_field}")
    if not isinstance(raw_field, str) or not (
        raw_field.strip() in _BUSINESS_FIELDS
        or (raw_field.strip().startswith("metadata.") and len(raw_field.strip()) > len("metadata."))
    ):
        raise ValueError("Unsupported filter field")
    field = raw_field.strip()
    kind = candidate.get("search_type")
    if kind not in _SEARCH_TYPES:
        raise ValueError("Unsupported filter search_type")
    keys = set(candidate.keys())
    if kind in ("term", "match", "match_phrase"):
        _exact_keys(keys, {"search_type", "field", "value"})
        return {"search_type": kind, "field": field, "value": _scalar(candidate["value"], False)}
    if kind == "terms":
        if ("values" in keys) == ("valuess" in keys):
            raise ValueError("terms filter requires exactly one values property")
        key = "valuess" if "valuess" in keys else "values"
        _exact_keys(keys, {"search_type", "field", key})
        values = candidate[key]
        if not isinstance(values, (list, tuple)) or not values:
            raise ValueError("terms filter values must be a non-empty list")
        return {"search_type": "terms", "field": field, "valuess": [_scalar(v, False) for v in values]}
    bounds = {"gt", "gte", "lt", "lte"}
    if not keys <= {"search_type", "field"} | bounds:
        raise ValueError("Filter clause has unsupported properties")
    if not keys & bounds:
        raise ValueError("range filter requires at least one bound")
    if field not in _RANGE_FIELDS and not field.startswith("metadata."):
        raise ValueError("range filter field is not range-capable")
    if {"gt", "gte"} <= keys or {"lt", "lte"} <= keys:
        raise ValueError("range filter contains ambiguous bounds")
    out: dict[str, Any] = {"search_type": "range", "field": field}
    for bound in ("gt", "gte", "lt", "lte"):
        if bound in keys:
            out[bound] = _scalar(candidate[bound], True)
    return out


def _exact_keys(keys: set[Any], expected: set[str]) -> None:
    if not keys <= expected:
        raise ValueError("Filter clause has unsupported properties")
    if keys != expected:
        raise ValueError("Filter clause has missing properties")


def _scalar(value: Any, range_bound: bool) -> Any:
    if isinstance(value, str):
        if not value.strip():
            raise ValueError("Filter value must not be blank")
        return value
    if isinstance(value, bool):
        if range_bound:
            raise ValueError("Filter value must be scalar")
        return value
    if isinstance(value, (int, float)):
        if not math.isfinite(value):
            raise ValueError("Filter value must be finite")
        return value
    raise ValueError("Filter value must be scalar")


# --- context, results, events -----------------------------------------------------------------

@dataclass(frozen=True)
class WorkspaceRequestContext:
    """Per-call context: an optional end-user bearer JWT (used for that call only, never stored or
    retried) and an optional canonical request UUID sent as ``X-Request-Id``."""

    end_user_jwt: str | None = None
    request_id: str | None = None

    def __post_init__(self) -> None:
        jwt = self.end_user_jwt
        if jwt is not None:
            if not jwt.strip():
                raise ValueError("end_user_jwt must not be blank")
            if jwt != jwt.strip():
                raise ValueError("end_user_jwt must not have surrounding whitespace")
            if any(is_iso_control(c) for c in jwt):
                raise ValueError("end_user_jwt must not contain ISO control characters")
        if self.request_id is not None and not is_canonical_uuid(self.request_id):
            raise ValueError("request_id must be a canonical lowercase UUID")

    @classmethod
    def bearer(cls, end_user_jwt: str) -> WorkspaceRequestContext:
        if end_user_jwt is None:
            raise ValueError("end_user_jwt must not be blank")
        return cls(end_user_jwt=end_user_jwt)

    def with_request_id(self, request_id: str) -> WorkspaceRequestContext:
        return WorkspaceRequestContext(self.end_user_jwt, request_id)

    def __repr__(self) -> str:
        jwt = "<redacted>" if self.end_user_jwt else None
        return f"WorkspaceRequestContext(end_user_jwt={jwt}, request_id={self.request_id!r})"


@dataclass(frozen=True)
class WorkspaceResult:
    """HTTP status, the parsed JSON body (None when the body was not JSON) and the request ID."""

    status_code: int
    body: Any
    request_id: str | None = None


@dataclass(frozen=True)
class WorkspaceStreamEvent:
    """One server-sent event. Unknown event types and fields are preserved."""

    event: str
    id: str | None
    data: dict[str, Any]

    @property
    def payload(self) -> Any:
        return copy.deepcopy(self.data.get("payload", {}))

    def _text(self, key: str) -> str | None:
        value = self.data.get(key)
        return value if isinstance(value, str) else None

    @property
    def run_id(self) -> str | None:
        return self._text("run_id")

    @property
    def request_id(self) -> str | None:
        return self._text("request_id")

    @property
    def timestamp(self) -> str | None:
        return self._text("ts")

    @property
    def sequence(self) -> int | None:
        """Absent on heartbeats; duplicates and out-of-order values are preserved."""
        value = self.data.get("seq")
        return value if isinstance(value, int) and not isinstance(value, bool) else None


# --- transport --------------------------------------------------------------------------------

class _HttpFailure(Exception):
    def __init__(self, status: int, body: str | None, request_id: str | None) -> None:
        self.status, self.body, self.request_id = status, body, request_id


def _parse(text: str | bytes | None) -> Any:
    if text is None or not (text.strip() if isinstance(text, (str, bytes)) else text):
        return None
    try:
        return json.loads(text)
    except ValueError:
        return None


def _workspace_error(failure: _HttpFailure, authorization: str | None, override: tuple[str, str] | None = None
                     ) -> SmartSearchError:
    secrets = auth_secrets(authorization)
    node = _parse(failure.body)
    code, message = override or ("WORKSPACE_ERROR", "Workspace request failed")
    body_request_id = None
    error = node.get("error") if isinstance(node, dict) else None
    if isinstance(error, dict):
        if override is None and isinstance(error.get("code"), str):
            code = error["code"]
        if override is None and isinstance(error.get("message"), str):
            message = error["message"]
        if isinstance(error.get("request_id"), str):
            body_request_id = error["request_id"]
    request_id = failure.request_id if failure.request_id and failure.request_id.strip() else body_request_id
    return error_type(failure.status)(
        safe_text(message, "Workspace request failed", MAX_MESSAGE_LENGTH, secrets),
        status_code=failure.status,
        code=safe_text(code, "WORKSPACE_ERROR", MAX_CODE_LENGTH, secrets),
        request_id=safe_request_id(request_id),
        raw_body=safe_raw_body(failure.body, secrets) if override is None else None,
    )


def _provisioning_error(failure: _HttpFailure, authorization: str | None, override: str | None = None
                        ) -> SmartSearchError:
    code = override or "WORKSPACE_ERROR"
    retryable = False
    request_id = safe_request_id(failure.request_id)
    node = _parse(failure.body)
    if isinstance(node, dict):
        if override is None:
            error = node.get("error")
            if isinstance(error, dict) and isinstance(error.get("code"), str):
                code = error["code"]
            if isinstance(node.get("code"), str):
                code = node["code"]
            if isinstance(node.get("retryable"), bool):
                retryable = node["retryable"]
        if request_id is None:
            request_id = safe_request_id(node.get("request_id"))
        if request_id is None and isinstance(node.get("error"), dict):
            request_id = safe_request_id(node["error"].get("request_id"))
    if not re.fullmatch(r"[A-Za-z0-9_.:-]{1,64}", code):
        code = "PROVISIONING_ERROR"
    code = safe_text(code, "PROVISIONING_ERROR", MAX_CODE_LENGTH, auth_secrets(authorization)) or "PROVISIONING_ERROR"
    return error_type(failure.status)("Provisioning request failed", status_code=failure.status, code=code,
                                      retryable=retryable, request_id=request_id)


class _Transport:
    """Authenticated JSON and SSE requests against one base URL. Thread-safe."""

    def __init__(self, http: httpx.Client, base_url: str, credential: Credential, read_timeout: float) -> None:
        self._http = http
        self.base_url = base_url
        self._credential = credential
        self._read_timeout = read_timeout
        self._streams: weakref.WeakSet[WorkspaceStream] = weakref.WeakSet()
        self._lock = threading.Lock()
        self._closed = False

    def _timeout(self) -> httpx.Timeout:
        return httpx.Timeout(self._read_timeout, connect=self._http.timeout.connect)

    def _timeout_error(self) -> RequestTimeoutError:
        return RequestTimeoutError(
            f"Workspace request timed out after {self._read_timeout:g} s; answer and chat requests can take a "
            "minute or more, so raise read_timeout", code="WORKSPACE_REQUEST_TIMEOUT")

    def require_open(self) -> None:
        if self._closed:
            raise RuntimeError("Workplace client is closed")

    def with_authorization(self, context: WorkspaceRequestContext | None, operation: Callable[[str], Any],
                           provisioning: bool = False) -> Any:
        relayed = context is not None and context.end_user_jwt is not None
        retry = not relayed and self._credential.refreshable
        if relayed:
            authorization, generation = "Bearer " + context.end_user_jwt, 0  # type: ignore[union-attr]
        else:
            authorization, generation = self._credential.authorization()
        _require_header(authorization)
        try:
            return operation(authorization)
        except _HttpFailure as failure:
            if retry and failure.status == 401:
                self._credential.invalidate(generation)
                try:
                    retried, _ = self._credential.authorization()
                except AuthenticationError:
                    raise self._error(failure, authorization, provisioning, refresh_failed=True) from None
                _require_header(retried)
                try:
                    return operation(retried)
                except _HttpFailure as again:
                    raise self._error(again, retried, provisioning) from None
            raise self._error(failure, authorization, provisioning) from None

    @staticmethod
    def _error(failure: _HttpFailure, authorization: str, provisioning: bool, refresh_failed: bool = False
               ) -> SmartSearchError:
        if provisioning:
            return _provisioning_error(failure, authorization, "AUTHENTICATION_REFRESH_FAILED" if refresh_failed else None)
        override = ("AUTHENTICATION_REFRESH_FAILED", "Workspace authentication refresh failed") if refresh_failed else None
        return _workspace_error(failure, authorization, override)

    def execute(self, method: str, path: str, body: Any, context: WorkspaceRequestContext | None,
                idempotency_key: str | None = None, provisioning: bool = False) -> WorkspaceResult:
        self.require_open()
        payload = None if body is None else json.dumps(body, separators=(",", ":"), ensure_ascii=False).encode()

        def send(authorization: str) -> WorkspaceResult:
            headers = {"Content-Type": "application/json", "Accept": "application/json",
                       "Authorization": authorization}
            if context is not None and context.request_id is not None:
                headers[_REQUEST_ID_HEADER] = context.request_id
            if idempotency_key is not None:
                headers["Idempotency-Key"] = idempotency_key
            try:
                response = self._http.request(method, self.base_url + path, content=payload, headers=headers,
                                              timeout=self._timeout(), follow_redirects=False)
            except httpx.TimeoutException:
                raise self._timeout_error() from None
            except httpx.HTTPError:
                raise TransportError("Workspace request failed", code="WORKSPACE_CLIENT_ERROR") from None
            header_id = response.headers.get(_REQUEST_ID_HEADER)
            if not 200 <= response.status_code <= 299:
                raise _HttpFailure(response.status_code, response.text, header_id)
            data = _parse(response.content)
            request_id = header_id if header_id and header_id.strip() else _body_request_id(data)
            return WorkspaceResult(response.status_code, data, request_id if is_canonical_uuid(request_id) else None)

        return self.with_authorization(context, send, provisioning)

    def stream(self, path: str, request: WorkspaceQueryRequest, context: WorkspaceRequestContext | None,
               chat: bool) -> WorkspaceStream:
        self.require_open()
        body = _require_request(request).to_dict()
        body["options"]["stream"] = True
        payload = json.dumps(body, separators=(",", ":"), ensure_ascii=False).encode()

        def open_stream(authorization: str) -> WorkspaceStream:
            headers = {"Content-Type": "application/json", "Accept": "text/event-stream",
                       "Authorization": authorization}
            if context is not None and context.request_id is not None:
                headers[_REQUEST_ID_HEADER] = context.request_id
            req = self._http.build_request("POST", self.base_url + path, content=payload, headers=headers,
                                           timeout=self._timeout())
            try:
                response = self._http.send(req, stream=True, follow_redirects=False)
            except httpx.TimeoutException:
                raise self._timeout_error() from None
            except httpx.HTTPError:
                raise TransportError("Workspace request failed", code="WORKSPACE_CLIENT_ERROR") from None
            request_id = response.headers.get(_REQUEST_ID_HEADER)
            content_type = response.headers.get("Content-Type", "")
            is_sse = content_type.split(";", 1)[0].strip().lower() == "text/event-stream"
            if not 200 <= response.status_code <= 299:
                raise _HttpFailure(response.status_code, _read_error_body(response, is_sse), request_id)
            if not is_sse:
                response.close()
                raise SmartSearchError("Expected a text/event-stream response", status_code=response.status_code,
                                       code="WORKSPACE_STREAM_CONTENT_TYPE", request_id=safe_request_id(request_id))
            stream = WorkspaceStream(response, request_id, authorization,
                                     request.session_id if chat else None, self._read_timeout)
            with self._lock:
                self._streams.add(stream)
            return stream

        return self.with_authorization(context, open_stream)

    def close_streams(self) -> None:
        with self._lock:
            self._closed = True
            streams = list(self._streams)
        for stream in streams:
            stream.close()


def _require_header(value: str) -> None:
    if not value or not visible_ascii(value, allow_space=True):
        raise ValueError("Workspace authorization must contain only visible ASCII characters")


def _require_request(request: object) -> WorkspaceQueryRequest:
    if not isinstance(request, WorkspaceQueryRequest):
        raise ValueError("Workspace query request is required")
    return request


def _body_request_id(data: Any) -> str | None:
    if not isinstance(data, dict):
        return None
    if isinstance(data.get("request_id"), str):
        return data["request_id"]
    error = data.get("error")
    if isinstance(error, dict) and isinstance(error.get("request_id"), str):
        return error["request_id"]
    return None


def _read_error_body(response: httpx.Response, is_sse: bool) -> str | None:
    """Bounded diagnostics from a failed stream request."""
    try:
        if is_sse:
            event = _SseParser(response.iter_bytes()).next_event()
            return None if event is None else json.dumps(event.data)
        data = b""
        for chunk in response.iter_bytes():
            data += chunk
            if len(data) >= 65_536:
                break
        return data[:65_536].decode("utf-8", errors="replace")
    except Exception:
        return None
    finally:
        response.close()


# --- server-sent events -----------------------------------------------------------------------

MAX_EVENT_CHARACTERS = 1_048_576
_EOL = re.compile(r"\r\n|\r|\n")


class _EventTooLarge(Exception):
    pass


class _InvalidEvent(Exception):
    pass


class _SseParser:
    """Incremental SSE framing with a bounded frame size; never buffers the whole stream."""

    def __init__(self, chunks: Iterator[bytes]) -> None:
        self._chunks = iter(chunks)
        self._decoder = codecs.getincrementaldecoder("utf-8")("strict")
        self._buf = ""
        self._eof = False
        self._first = True
        self._chars = 0

    def _fill(self) -> bool:
        if self._eof:
            return False
        chunk = next(self._chunks, None)
        if chunk is None:
            self._eof = True
            self._buf += self._decoder.decode(b"", final=True)
            return False
        text = self._decoder.decode(chunk)
        if self._first and text:
            self._first = False
            if text.startswith("﻿"):
                text = text[1:]
        self._buf += text
        return True

    def _line(self) -> str | None:
        while True:
            match = _EOL.search(self._buf)
            # A trailing CR needs the next character to tell CR from CRLF, unless the stream ended.
            if match and not (match.group() == "\r" and match.end() == len(self._buf) and not self._eof):
                line = self._buf[:match.start()]
                self._buf = self._buf[match.end():]
                self._chars += len(line) + 1
                if self._chars > MAX_EVENT_CHARACTERS:
                    raise _EventTooLarge()
                return line
            if self._chars + len(self._buf) > MAX_EVENT_CHARACTERS:
                raise _EventTooLarge()
            if not self._fill() and not (match and self._eof):
                if not self._buf:
                    return None
                line, self._buf = self._buf, ""
                return line

    def next_event(self) -> WorkspaceStreamEvent | None:
        event, event_id, data, has_data = "message", None, [], False
        self._chars = 0
        while True:
            line = self._line()
            if line is None:
                return None  # an unterminated final frame is discarded
            if line == "":
                self._chars = 0
                if not has_data:
                    event, event_id = "message", None
                    continue
                try:
                    parsed = json.loads("\n".join(data))
                except ValueError:
                    raise _InvalidEvent() from None
                if not isinstance(parsed, dict):
                    raise _InvalidEvent()
                return WorkspaceStreamEvent(event, event_id, parsed)
            if line.startswith(":"):
                continue
            field, _, value = line.partition(":")
            if value.startswith(" "):
                value = value[1:]
            if field == "event":
                event = value or "message"
            elif field == "id":
                if "\0" not in value:
                    event_id = value
            elif field == "data":
                data.append(value)
                has_data = True


class WorkspaceStream:
    """A streaming Query or Chat run. Iterate for events; iteration ends after ``run.completed``.

    ``run.failed`` / ``error`` events, truncated or invalid input and read timeouts raise
    :class:`SmartSearchError`. The final ``run.completed`` payload is :attr:`result` and is
    authoritative over answer deltas. Use as a context manager; closing stops local reading only
    (there is no server cancellation or automatic reconnect). Single consumer; not thread-safe.
    """

    def __init__(self, response: httpx.Response, request_id: str | None, authorization: str,
                 session_id: str | None, read_timeout: float) -> None:
        self._response = response
        self._status = response.status_code
        self._request_id = request_id
        self._secrets = auth_secrets(authorization)
        self._timeout = read_timeout
        self._parser = _SseParser(response.iter_bytes())
        self.session_id: str | None = session_id
        self._result: WorkspaceResult | None = None
        self._failure: SmartSearchError | None = None
        self._closed = False

    @property
    def status_code(self) -> int:
        return self._status

    @property
    def result(self) -> WorkspaceResult | None:
        """The final ``run.completed`` payload with the retained session ID; None while streaming."""
        if self._failure is not None:
            raise self._failure
        return self._result

    def __iter__(self) -> WorkspaceStream:
        return self

    def __next__(self) -> WorkspaceStreamEvent:
        if self._failure is not None:
            raise self._failure
        if self._result is not None:
            raise StopIteration
        if self._closed:
            raise RuntimeError("Workspace stream is closed")
        try:
            event = self._parser.next_event()
        except httpx.TimeoutException:
            raise self._fail(RequestTimeoutError, "WORKSPACE_STREAM_TIMEOUT",
                             "Timed out waiting for a Workspace stream event") from None
        except _EventTooLarge:
            raise self._fail(SmartSearchError, "WORKSPACE_STREAM_EVENT_TOO_LARGE",
                             "Unable to read a complete Workspace event") from None
        except (_InvalidEvent, UnicodeDecodeError, httpx.HTTPError):
            raise self._fail(SmartSearchError, "WORKSPACE_STREAM_READ_ERROR",
                             "Unable to read a complete Workspace event") from None
        if event is None:
            raise self._fail(SmartSearchError, "WORKSPACE_STREAM_INCOMPLETE",
                             "Workspace stream ended before run completion")
        payload = event.data.get("payload", {})
        if event.event in ("run.failed", "error"):
            error = event.data.get("error") if event.event == "error" else payload
            error = error if isinstance(error, dict) else {}
            code = error.get("code") if isinstance(error.get("code"), str) else "WORKSPACE_STREAM_FAILED"
            message = error.get("message") if isinstance(error.get("message"), str) else "Workspace stream failed"
            raise self._fail(SmartSearchError, code, message, self._effective_request_id(event))
        if event.event == "run.started" and isinstance(payload, dict) and isinstance(payload.get("session_id"), str):
            self.session_id = payload["session_id"]
        if event.event == "run.completed":
            if not isinstance(payload, dict) or payload.get("status") != "COMPLETED":
                raise self._fail(SmartSearchError, "WORKSPACE_STREAM_INVALID_EVENT", "Invalid Workspace completion event")
            final = copy.deepcopy(payload)
            if self.session_id is not None and "session_id" not in final:
                final["session_id"] = self.session_id
            request_id = self._effective_request_id(event)
            self._result = WorkspaceResult(self._status, final, request_id if is_canonical_uuid(request_id) else None)
            self._release()
        return event

    def _effective_request_id(self, event: WorkspaceStreamEvent) -> str | None:
        return self._request_id if self._request_id and self._request_id.strip() else event.request_id

    def _fail(self, kind: type[SmartSearchError], code: str, message: str, request_id: str | None = None
              ) -> SmartSearchError:
        self._failure = kind(
            safe_text(message, "Workspace request failed", MAX_MESSAGE_LENGTH, self._secrets) or "Workspace request failed",
            status_code=self._status,
            code=safe_text(code, "WORKSPACE_ERROR", MAX_CODE_LENGTH, self._secrets),
            request_id=safe_request_id(request_id or self._request_id))
        self._release()
        return self._failure

    def _release(self) -> None:
        self._response.close()

    def close(self) -> None:
        """Stops reading and releases the connection. Idempotent."""
        self._closed = True
        self._release()

    def __enter__(self) -> WorkspaceStream:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def __repr__(self) -> str:
        return f"WorkspaceStream(status_code={self._status}, completed={self._result is not None})"


# --- facade -----------------------------------------------------------------------------------

_ROOT = "/workspace/v1/workspaces"


def _workspace(workspace_id: str) -> str:
    return f"{_ROOT}/{canonical_uuid(workspace_id, 'workspace_id')}"


def _grant(workspace_id: str, source_id: str, principal_id: str) -> str:
    return (f"{_workspace(workspace_id)}/data-sources/{canonical_uuid(source_id, 'source_id')}"
            f"/principal-grants/{canonical_uuid(principal_id, 'principal_id')}")


def _payload(model: object, kind: type[_Model]) -> dict[str, Any]:
    if not isinstance(model, kind):
        raise ValueError(f"{kind.__name__} is required")
    return model.to_dict()


class Workplace:
    """Workplace search, answers and chat plus member and source-grant administration.

    Every method takes an optional ``context`` (:class:`WorkspaceRequestContext`). Workspace, source
    and agent setup, ingestion and readiness are not part of this API. Thread-safe. Streams from
    ``stream_*`` must be closed by the caller (use ``with``).
    """

    def __init__(self, transport: _Transport) -> None:
        self._t = transport

    def _run(self, workspace_id: str, request: WorkspaceQueryRequest, context: WorkspaceRequestContext | None,
             endpoint: str) -> WorkspaceResult:
        body = _require_request(request).to_dict()
        return self._t.execute("POST", f"{_workspace(workspace_id)}/{endpoint}", body, context)

    def search(self, workspace_id: str, request: WorkspaceQueryRequest,
               context: WorkspaceRequestContext | None = None) -> WorkspaceResult:
        """``POST /workspace/v1/workspaces/{id}/search``: ranked sources."""
        return self._run(workspace_id, request, context, "search")

    def query(self, workspace_id: str, request: WorkspaceQueryRequest,
              context: WorkspaceRequestContext | None = None) -> WorkspaceResult:
        """``POST .../query``: retrieval or a generated answer, depending on ``request.mode``."""
        return self._run(workspace_id, request, context, "query")

    def chat(self, workspace_id: str, request: WorkspaceQueryRequest,
             context: WorkspaceRequestContext | None = None) -> WorkspaceResult:
        """``POST .../chat``: one chat turn; the body carries ``session_id`` for the next turn."""
        return self._run(workspace_id, request, context, "chat")

    def stream_query(self, workspace_id: str, request: WorkspaceQueryRequest,
                     context: WorkspaceRequestContext | None = None) -> WorkspaceStream:
        """``POST .../query`` as ``text/event-stream``."""
        return self._t.stream(f"{_workspace(workspace_id)}/query", request, context, chat=False)

    def stream_chat(self, workspace_id: str, request: WorkspaceQueryRequest,
                    context: WorkspaceRequestContext | None = None) -> WorkspaceStream:
        """``POST .../chat`` as ``text/event-stream``; ``stream.session_id`` continues the chat."""
        return self._t.stream(f"{_workspace(workspace_id)}/chat", request, context, chat=True)

    def list_members(self, workspace_id: str, context: WorkspaceRequestContext | None = None) -> WorkspaceResult:
        """``GET .../members``: members with identity, status and assignment metadata."""
        return self._t.execute("GET", f"{_workspace(workspace_id)}/members", None, context)

    def list_members_page(self, workspace_id: str, query: str | None = None, page: int = 0, size: int = 20,
                          context: WorkspaceRequestContext | None = None) -> WorkspaceResult:
        """``GET .../members/page``; ``page`` is zero-based and ``size`` is 1..100."""
        if isinstance(page, bool) or isinstance(size, bool) or page < 0 or not 1 <= size <= 100:
            raise ValueError("page must be nonnegative and size must be 1..100")
        path = (f"{_workspace(workspace_id)}/members/page?query={url_encode(query or '')}"
                f"&page={int(page)}&size={int(size)}")
        return self._t.execute("GET", path, None, context)

    def add_member(self, workspace_id: str, request: WorkspaceMemberRequest,
                   context: WorkspaceRequestContext | None = None) -> WorkspaceResult:
        """``POST .../members``: adds or updates one member. Source grants are separate."""
        return self._t.execute("POST", f"{_workspace(workspace_id)}/members",
                               _payload(request, WorkspaceMemberRequest), context)

    def get_principal_grant(self, workspace_id: str, source_id: str, principal_id: str,
                            context: WorkspaceRequestContext | None = None) -> WorkspaceResult:
        """Reads a principal's source grant and its revision."""
        return self._t.execute("GET", _grant(workspace_id, source_id, principal_id), None, context)

    def put_principal_grant(self, workspace_id: str, source_id: str, principal_id: str,
                            request: WorkspacePrincipalGrantRequest,
                            context: WorkspaceRequestContext | None = None) -> WorkspaceResult:
        """Replaces grants at the expected revision; a 409 is never reconciled automatically."""
        return self._t.execute("PUT", _grant(workspace_id, source_id, principal_id),
                               _payload(request, WorkspacePrincipalGrantRequest), context)

    def revoke_principal_grant(self, workspace_id: str, source_id: str, principal_id: str, expected_revision: int,
                               context: WorkspaceRequestContext | None = None) -> WorkspaceResult:
        """Revokes source grants at the exact expected revision; membership is unchanged."""
        if isinstance(expected_revision, bool) or not isinstance(expected_revision, int) or expected_revision < 0:
            raise ValueError("expected_revision must be nonnegative")
        path = f"{_grant(workspace_id, source_id, principal_id)}?expected_revision={expected_revision}"
        return self._t.execute("DELETE", path, None, context)

    def __repr__(self) -> str:
        return "Workplace"


class UserWorkplace(Workplace):
    """A :class:`Workplace` acting as one end user with that user's exchanged access token.

    The token is not refreshed: get a new one from ``SmartSearchAi.as_user*`` before
    :attr:`expires_at`. Close it (or use ``with``) when finished; that closes its open streams.
    """

    def __init__(self, transport: _Transport, expires_at: datetime) -> None:
        super().__init__(transport)
        self.expires_at = expires_at

    def close(self) -> None:
        self._t.close_streams()

    def __enter__(self) -> UserWorkplace:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def __repr__(self) -> str:
        return f"UserWorkplace(expires_at={self.expires_at.isoformat()})"


__all__ = [
    "Workplace", "UserWorkplace", "WorkspaceQueryRequest", "QueryOptions", "Mode", "MemoryMode", "Output",
    "CachePolicy", "WorkspaceRequestContext", "WorkspaceResult", "WorkspaceStream", "WorkspaceStreamEvent",
    "WorkspaceMemberRequest", "MemberRole", "WorkspacePrincipalGrantRequest",
]
