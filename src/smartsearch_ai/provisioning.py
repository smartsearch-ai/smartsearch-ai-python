"""User provisioning: jobs that register and onboard users, and complete membership snapshots.

Calls go to the Search Admin provisioning API (``/search-admin/api/provisioning/v1`` on ``admin_url``).
"""

from __future__ import annotations

import hashlib
import re
from enum import Enum
from typing import Any, Literal, Sequence, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ._util import canonical_uuid, is_iso_control, url_encode, visible_ascii
from .workplace import WorkspaceRequestContext, WorkspaceResult, _Transport

_ROOT = "/search-admin/api/provisioning/v1"


class ScopeKind(str, Enum):
    TENANT = "TENANT"
    DEPLOYMENT = "DEPLOYMENT"


class JobKind(str, Enum):
    UPSERT_USERS = "UPSERT_USERS"
    ONBOARD_USERS = "ONBOARD_USERS"
    DISABLE_USERS = "DISABLE_USERS"
    SET_MEMBERSHIPS = "SET_MEMBERSHIPS"
    REMOVE_MEMBERSHIPS = "REMOVE_MEMBERSHIPS"
    REPLACE_SOURCE_GRANTS = "REPLACE_SOURCE_GRANTS"


class JobState(str, Enum):
    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    WAITING_FOR_IDENTITY = "WAITING_FOR_IDENTITY"
    SUCCEEDED = "SUCCEEDED"
    PARTIAL = "PARTIAL"
    FAILED = "FAILED"
    RECONCILIATION_REQUIRED = "RECONCILIATION_REQUIRED"
    CANCEL_REQUESTED = "CANCEL_REQUESTED"
    CANCELLED = "CANCELLED"


def _text(value: Any, name: str, max_length: int) -> str:
    if (not isinstance(value, str) or not value.strip() or value != value.strip() or len(value) > max_length
            or any(is_iso_control(c) for c in value)):
        raise ValueError(f"{name} is invalid")
    return value


def _utf8_text(value: Any, name: str, max_bytes: int) -> str:
    text = _text(value, name, max(max_bytes, 1))
    if len(text.encode("utf-8")) > max_bytes:
        raise ValueError(f"{name} is too long")
    return text


def _labels(values: Any, name: str, max_items: int = 1000, max_length: int = 4096) -> tuple[str, ...]:
    if values is None or isinstance(values, str) or len(values) > max_items:
        raise ValueError(f"{name} size is invalid")
    copy = tuple(_text(v, name, max_length) for v in values)
    if len(set(copy)) != len(copy):
        raise ValueError(f"{name} contains duplicates")
    return copy


def _bounded(values: Any, name: str, low: int, high: int) -> tuple[Any, ...]:
    if values is None or isinstance(values, str) or not low <= len(values) <= high or any(v is None for v in values):
        raise ValueError(f"{name} size is invalid")
    return tuple(values)


class _Model(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    def to_dict(self) -> dict[str, Any]:
        """The JSON body sent to the server."""
        return self.model_dump(mode="json", exclude_none=True)


class Scope(_Model):
    kind: ScopeKind
    id: str

    @field_validator("id")
    @classmethod
    def _id(cls, v: str) -> str:
        return _text(v, "scope.id", 256)


class Grants(_Model):
    groups: tuple[str, ...] = ()
    roles: tuple[str, ...] = ()
    security_keys: tuple[str, ...] = ()

    @field_validator("groups", "roles", "security_keys", mode="before")
    @classmethod
    def _values(cls, v: Any, info: Any) -> tuple[str, ...]:
        return _labels(v, info.field_name)


class SourceGrant(_Model):
    source_id: str
    expected_revision: int = Field(ge=0)
    grants: Grants

    @field_validator("source_id")
    @classmethod
    def _source(cls, v: str) -> str:
        return canonical_uuid(v, "source_id")


class Membership(_Model):
    """Workspace membership, optionally with source grants (``expected_revision`` guards updates)."""

    workspace_id: str
    expected_revision: int | None = Field(None, ge=0)
    source_grants: tuple[SourceGrant, ...] | None = None

    @field_validator("workspace_id")
    @classmethod
    def _workspace(cls, v: str) -> str:
        return canonical_uuid(v, "workspace_id")

    @field_validator("source_grants", mode="before")
    @classmethod
    def _grants(cls, v: Any) -> Any:
        return None if v is None else _bounded(v, "source_grants", 0, 100)


class Profile(_Model):
    email: str
    first_name: str | None = None
    last_name: str | None = None
    display_name: str | None = None

    @field_validator("email")
    @classmethod
    def _email(cls, v: str) -> str:
        return _text(v, "email", 320)


class ExternalIdentity(_Model):
    """The user's identity at your identity provider (``subject`` and ``username``).

    Required on every user of a federated integration (``KEYCLOAK_FEDERATED``): the SmartSearch user is
    linked to this account, which must already exist there. Must be absent for an invite integration
    (``KEYCLOAK_INVITE``): the user gets an email to set a SmartSearch password. Otherwise the item fails
    with :data:`~smartsearch_ai.errors.EXTERNAL_IDENTITY_REQUIRED` or
    :data:`~smartsearch_ai.errors.EXTERNAL_IDENTITY_NOT_SUPPORTED`.
    """

    subject: str
    username: str

    @field_validator("subject", "username")
    @classmethod
    def _identity(cls, v: str, info: Any) -> str:
        if not v.strip() or len(v.encode("utf-8")) > 256 or any(is_iso_control(c) for c in v):
            raise ValueError(f"{info.field_name} is invalid")
        return v


class EnsureItem(_Model):
    """One user to register (``UPSERT_USERS``) or register and onboard (``ONBOARD_USERS``)."""

    item_key: str
    external_user_id: str
    profile: Profile
    platform_role: Literal["GUEST"] = "GUEST"
    permissions: tuple[str, ...] = ()
    workspaces: tuple[Membership, ...] = ()
    external_identity: ExternalIdentity | None = None

    @field_validator("item_key")
    @classmethod
    def _key(cls, v: str) -> str:
        return _text(v, "item_key", 128)

    @field_validator("external_user_id")
    @classmethod
    def _external(cls, v: str) -> str:
        return _utf8_text(v, "external_user_id", 256)

    @field_validator("permissions", mode="before")
    @classmethod
    def _permissions(cls, v: Any) -> tuple[str, ...]:
        if v is None or isinstance(v, str) or len(v) > 100:
            raise ValueError("permissions size is invalid")
        return tuple(sorted({_text(p, "permissions", 4096) for p in v}))

    @field_validator("workspaces", mode="before")
    @classmethod
    def _workspaces(cls, v: Any) -> tuple[Any, ...]:
        return _bounded(v, "workspaces", 0, 100)


class AssignmentItem(_Model):
    item_key: str
    principal_id: str
    workspaces: tuple[Membership, ...]

    @field_validator("item_key")
    @classmethod
    def _key(cls, v: str) -> str:
        return _text(v, "item_key", 128)

    @field_validator("principal_id")
    @classmethod
    def _principal(cls, v: str) -> str:
        return canonical_uuid(v, "principal_id")

    @field_validator("workspaces", mode="before")
    @classmethod
    def _workspaces(cls, v: Any) -> tuple[Any, ...]:
        return _bounded(v, "workspaces", 1, 100)


class DisableItem(_Model):
    item_key: str
    principal_id: str

    @field_validator("item_key")
    @classmethod
    def _key(cls, v: str) -> str:
        return _text(v, "item_key", 128)

    @field_validator("principal_id")
    @classmethod
    def _principal(cls, v: str) -> str:
        return canonical_uuid(v, "principal_id")


JobItem = Union[EnsureItem, AssignmentItem, DisableItem]

_ITEM_TYPES: dict[JobKind, type[_Model]] = {
    JobKind.UPSERT_USERS: EnsureItem, JobKind.ONBOARD_USERS: EnsureItem, JobKind.DISABLE_USERS: DisableItem,
    JobKind.SET_MEMBERSHIPS: AssignmentItem, JobKind.REMOVE_MEMBERSHIPS: AssignmentItem,
    JobKind.REPLACE_SOURCE_GRANTS: AssignmentItem,
}


class SubmitJob(_Model):
    """A provisioning job of 1..100 items; the item type must match ``kind``. Identity jobs
    (``UPSERT_USERS``, ``ONBOARD_USERS``, ``DISABLE_USERS``) require ``integration_id``."""

    scope: Scope
    integration_id: str | None = None
    kind: JobKind
    items: tuple[JobItem, ...]

    @field_validator("integration_id")
    @classmethod
    def _integration(cls, v: str | None) -> str | None:
        return None if v is None else _text(v, "integration_id", 256)

    @field_validator("items", mode="before")
    @classmethod
    def _items(cls, v: Any) -> tuple[Any, ...]:
        items = _bounded(v, "items", 1, 100)
        if not all(isinstance(i, (EnsureItem, AssignmentItem, DisableItem)) for i in items):
            raise ValueError("items must be EnsureItem, AssignmentItem or DisableItem values")
        return items

    @model_validator(mode="after")
    def _matches(self) -> SubmitJob:
        if not all(type(i) is _ITEM_TYPES[self.kind] for i in self.items):
            raise ValueError("items do not match job kind")
        if self.kind in (JobKind.UPSERT_USERS, JobKind.ONBOARD_USERS, JobKind.DISABLE_USERS) and self.integration_id is None:
            raise ValueError("integration_id is required for identity jobs")
        return self


class RetryJob(_Model):
    item_keys: tuple[str, ...]

    @field_validator("item_keys", mode="before")
    @classmethod
    def _keys(cls, v: Any) -> tuple[str, ...]:
        keys = _labels(v, "item_keys", 100, 128)
        if not keys:
            raise ValueError("item_keys is required")
        return keys


class SnapshotRow(_Model):
    principal_id: str
    workspace_id: str

    @field_validator("principal_id", "workspace_id")
    @classmethod
    def _ids(cls, v: str, info: Any) -> str:
        return canonical_uuid(v, info.field_name)


def snapshot_digest(rows: Sequence[SnapshotRow]) -> str:
    """SHA-256 of the compact rows sorted by workspace then principal, keys in lexicographic order."""
    ordered = sorted(_bounded(rows, "rows", 0, 5000), key=lambda r: (r.workspace_id, r.principal_id))
    text = "[" + ",".join(
        f'{{"principal_id":"{r.principal_id}","workspace_id":"{r.workspace_id}"}}' for r in ordered) + "]"
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


_DIGEST = re.compile(r"[a-f0-9]{64}")


class SnapshotRequest(_Model):
    scope: Scope
    integration_id: str
    workspace_ids: tuple[str, ...]
    generation: int = Field(ge=1)
    expected_count: int = Field(ge=0, le=5000)
    content_digest: str

    @field_validator("integration_id")
    @classmethod
    def _integration(cls, v: str) -> str:
        return _text(v, "integration_id", 256)

    @field_validator("workspace_ids", mode="before")
    @classmethod
    def _workspaces(cls, v: Any) -> tuple[str, ...]:
        if v is None or isinstance(v, str) or not 1 <= len(v) <= 100:
            raise ValueError("workspace_ids size is invalid")
        return tuple(sorted({canonical_uuid(w, "workspace_ids") for w in v}))

    @field_validator("content_digest")
    @classmethod
    def _digest(cls, v: str) -> str:
        if not _DIGEST.fullmatch(v):
            raise ValueError("content_digest must be lowercase SHA-256")
        return v

    @classmethod
    def for_rows(cls, scope: Scope, integration_id: str, workspace_ids: Sequence[str], generation: int,
                 rows: Sequence[SnapshotRow]) -> SnapshotRequest:
        """Builds the count and canonical digest from the exact rows to upload."""
        rows = _bounded(rows, "rows", 0, 5000)
        return cls(scope=scope, integration_id=integration_id, workspace_ids=workspace_ids,
                   generation=generation, expected_count=len(rows), content_digest=snapshot_digest(rows))


class SnapshotApply(_Model):
    expected_generation: int = Field(ge=1)
    preview_digest: str

    @field_validator("preview_digest")
    @classmethod
    def _digest(cls, v: str) -> str:
        if not _DIGEST.fullmatch(v):
            raise ValueError("preview_digest must be lowercase SHA-256")
        return v


def _key(value: str | None) -> str:
    if value is None:
        raise ValueError("idempotency_key is required")
    if not isinstance(value, str) or len(value) > 256 or not visible_ascii(value):
        raise ValueError("idempotency_key must be 1..256 visible ASCII characters")
    return value


def _id(value: str, name: str) -> str:
    try:
        return canonical_uuid(value, name)
    except ValueError:
        raise ValueError(f"{name} must be a canonical UUID") from None


def _query(value: str, name: str, max_length: int) -> str:
    if (not isinstance(value, str) or not value.strip() or len(value) > max_length or value != value.strip()
            or any(is_iso_control(c) for c in value)):
        raise ValueError(f"{name} is invalid")
    return url_encode(value)


def _limit(limit: int) -> int:
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 100:
        raise ValueError("limit must be 1..100")
    return limit


def _model(value: object, kind: type[_Model]) -> dict[str, Any]:
    if not isinstance(value, kind):
        raise ValueError(f"{kind.__name__} is required")
    return value.to_dict()


class Users:
    """Provisioning as the service key. Thread-safe.

    Jobs run asynchronously: :meth:`submit_job` returns the accepted job, then poll :meth:`get_job`.
    Mutating calls take an ``idempotency_key``: reuse it to retry the same request safely.
    Failures raise :class:`~smartsearch_ai.errors.SmartSearchError` subclasses with the server code
    (for example :data:`~smartsearch_ai.errors.INTEGRATION_NOT_FOUND`) and ``retryable``.

    Identity jobs name an integration that an administrator set up in SmartSearch (this SDK cannot create
    one). Users are always created in SmartSearch's sign-in realm; an invite integration emails each user
    to set a password (send no ``external_identity``), a federated one links each user to their existing
    account at your identity provider (``external_identity`` required).
    """

    def __init__(self, transport: _Transport) -> None:
        self._t = transport

    def _call(self, method: str, suffix: str, body: Any = None, context: WorkspaceRequestContext | None = None,
              idempotency_key: str | None = None) -> WorkspaceResult:
        return self._t.execute(method, _ROOT + suffix, body, context, idempotency_key, provisioning=True)

    def capabilities(self, context: WorkspaceRequestContext | None = None) -> WorkspaceResult:
        """``GET .../capabilities``: what this service key may provision."""
        return self._call("GET", "/capabilities", context=context)

    def submit_job(self, job: SubmitJob, idempotency_key: str,
                   context: WorkspaceRequestContext | None = None) -> WorkspaceResult:
        """``POST .../jobs`` with an ``Idempotency-Key`` header."""
        body = _model(job, SubmitJob)
        return self._call("POST", "/jobs", body, context, _key(idempotency_key))

    def get_job(self, job_id: str, context: WorkspaceRequestContext | None = None) -> WorkspaceResult:
        """``GET .../jobs/{job_id}``."""
        return self._call("GET", f"/jobs/{_id(job_id, 'job_id')}", context=context)

    def list_job_items(self, job_id: str, *, after: str | None = None, limit: int = 100,
                       state: JobState | None = None, context: WorkspaceRequestContext | None = None
                       ) -> WorkspaceResult:
        """``GET .../jobs/{job_id}/items``: per-item outcomes, paged with ``after``."""
        path = f"/jobs/{_id(job_id, 'job_id')}/items?limit={_limit(limit)}"
        if after is not None:
            path += f"&after={_query(after, 'after', 512)}"
        if state is not None:
            path += f"&state={JobState(state).value}"
        return self._call("GET", path, context=context)

    def retry_job(self, job_id: str, request: RetryJob, idempotency_key: str,
                  context: WorkspaceRequestContext | None = None) -> WorkspaceResult:
        body = _model(request, RetryJob)
        return self._call("POST", f"/jobs/{_id(job_id, 'job_id')}/retry", body, context, _key(idempotency_key))

    def cancel_job(self, job_id: str, context: WorkspaceRequestContext | None = None) -> WorkspaceResult:
        return self._call("POST", f"/jobs/{_id(job_id, 'job_id')}/cancel", context=context)

    def list_principals(self, integration_id: str, scope: Scope, *, after: str | None = None,
                        limit: int = 100) -> WorkspaceResult:
        """Pages the integration's external user IDs and their canonical principal IDs."""
        if not isinstance(scope, Scope):
            raise ValueError("scope is required")
        if not isinstance(integration_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", integration_id):
            raise ValueError("Invalid integration_id")
        path = (f"/integrations/{integration_id}/principals?scope_kind={scope.kind.value}"
                f"&scope_id={_query(scope.id, 'scope.id', 256)}&limit={_limit(limit)}")
        if after is not None:
            path += f"&after={_query(after, 'after', 512)}"
        return self._call("GET", path)

    def create_snapshot(self, request: SnapshotRequest, context: WorkspaceRequestContext | None = None
                        ) -> WorkspaceResult:
        return self._call("POST", "/snapshots", _model(request, SnapshotRequest), context)

    def upload_snapshot_page(self, snapshot_id: str, page_number: int, rows: Sequence[SnapshotRow],
                             context: WorkspaceRequestContext | None = None) -> WorkspaceResult:
        """Uploads one immutable zero-based page of 1..100 rows."""
        if isinstance(page_number, bool) or not isinstance(page_number, int) or page_number < 0:
            raise ValueError("page_number must be nonnegative")
        if not rows or len(rows) > 100 or not all(isinstance(r, SnapshotRow) for r in rows):
            raise ValueError("rows must contain 1..100 values")
        return self._call("PUT", f"/snapshots/{_id(snapshot_id, 'snapshot_id')}/pages/{page_number}",
                          [r.to_dict() for r in rows], context)

    def seal_snapshot(self, snapshot_id: str, context: WorkspaceRequestContext | None = None) -> WorkspaceResult:
        return self._call("POST", f"/snapshots/{_id(snapshot_id, 'snapshot_id')}/seal", context=context)

    def get_snapshot(self, snapshot_id: str, context: WorkspaceRequestContext | None = None) -> WorkspaceResult:
        return self._call("GET", f"/snapshots/{_id(snapshot_id, 'snapshot_id')}", context=context)

    def preview_snapshot(self, snapshot_id: str, *, after: int = 0, limit: int = 100,
                         context: WorkspaceRequestContext | None = None) -> WorkspaceResult:
        if isinstance(after, bool) or not isinstance(after, int) or after < 0:
            raise ValueError("after must be nonnegative")
        return self._call("GET", f"/snapshots/{_id(snapshot_id, 'snapshot_id')}/preview?after={after}"
                                 f"&limit={_limit(limit)}", context=context)

    def apply_snapshot(self, snapshot_id: str, request: SnapshotApply, idempotency_key: str,
                       context: WorkspaceRequestContext | None = None) -> WorkspaceResult:
        body = _model(request, SnapshotApply)
        return self._call("POST", f"/snapshots/{_id(snapshot_id, 'snapshot_id')}/apply", body, context,
                          _key(idempotency_key))

    def __repr__(self) -> str:
        return "Users"


__all__ = [
    "Users", "ScopeKind", "JobKind", "JobState", "Scope", "Grants", "SourceGrant", "Membership", "Profile",
    "ExternalIdentity", "EnsureItem", "AssignmentItem", "DisableItem", "SubmitJob", "RetryJob", "SnapshotRow",
    "SnapshotRequest", "SnapshotApply", "snapshot_digest",
]
