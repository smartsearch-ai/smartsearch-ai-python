"""Internal credentials: the service key (OAuth2 client credentials) and user token exchanges."""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Callable, Sequence

import httpx

from ._util import form_encode, visible_ascii
from .errors import AuthenticationError

_MIN_SAFETY_MS = 30_000
_MAX_SAFETY_MS = 45_000
_TOKEN_TIMEOUT = httpx.Timeout(30.0, connect=5.0)
_ACCESS_TOKEN_TYPE = "urn:ietf:params:oauth:token-type:access_token"


def _token_failure() -> AuthenticationError:
    # Never include the response, secret, assertion or token.
    return AuthenticationError("Token acquisition failed", code="TOKEN_ACQUISITION_FAILED")


@dataclass(frozen=True)
class _Snapshot:
    header: str
    generation: int
    expires_ms: int

    def fresh(self, now_ms: int) -> bool:
        return bool(self.header) and now_ms < self.expires_ms


_EMPTY = _Snapshot("", 0, 0)


class Credential:
    """Produces an ``Authorization`` value. ``refreshable`` credentials retry once on a 401."""

    refreshable = False

    def authorization(self) -> tuple[str, int]:
        raise NotImplementedError

    def invalidate(self, generation: int) -> None:  # pragma: no cover - static credentials
        pass


class ServiceCredential(Credential):
    """Client-credentials token, cached per instance and refreshed single-flight.

    Refreshes 30-45 s before expiry (half the lifetime for very short tokens). A 401 clears only the
    rejected generation, so a late 401 from an old token cannot discard a newer one. Thread-safe.
    """

    refreshable = True

    def __init__(
        self,
        http: httpx.Client,
        token_url: str,
        client_id: str,
        client_secret: str,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._http = http
        self._token_url = token_url
        self._client_id = client_id
        self._client_secret = client_secret
        self._clock = clock
        self._lock = threading.Lock()
        self._snapshot = _EMPTY
        self._generations = 0

    def _now_ms(self) -> int:
        return int(self._clock() * 1000)

    def authorization(self) -> tuple[str, int]:
        current = self._snapshot
        if not current.fresh(self._now_ms()):
            current = self._refresh()
        return current.header, current.generation

    def invalidate(self, generation: int) -> None:
        with self._lock:
            if self._snapshot.generation == generation:
                self._snapshot = _EMPTY

    def _refresh(self) -> _Snapshot:
        with self._lock:
            now = self._now_ms()
            if self._snapshot.fresh(now):
                return self._snapshot
            body = form_encode(
                [("grant_type", "client_credentials"), ("client_id", self._client_id),
                 ("client_secret", self._client_secret)]
            )
            try:
                response = self._http.post(
                    self._token_url,
                    content=body.encode("utf-8"),
                    headers={"Content-Type": "application/x-www-form-urlencoded", "Accept": "application/json"},
                    timeout=_TOKEN_TIMEOUT,
                    follow_redirects=False,
                )
                if response.status_code != 200:
                    raise _token_failure()
                data = json.loads(response.content or b"{}")
            except AuthenticationError:
                raise
            except Exception:
                raise _token_failure() from None
            token = data.get("access_token") if isinstance(data, dict) else None
            expires_in = data.get("expires_in") if isinstance(data, dict) else None
            if (
                not isinstance(token, str)
                or not visible_ascii(token)
                or isinstance(expires_in, bool)
                or not isinstance(expires_in, int)
                or expires_in <= 0
            ):
                raise _token_failure()
            lifetime = expires_in * 1000
            safety = min(min(_MAX_SAFETY_MS, max(_MIN_SAFETY_MS, lifetime // 10)), lifetime // 2)
            snapshot = _Snapshot("Bearer " + token, self._generations + 1, now + lifetime - safety)
            if not snapshot.fresh(self._now_ms()):
                raise _token_failure()
            self._generations += 1
            self._snapshot = snapshot
            return snapshot

    def __repr__(self) -> str:
        return "ServiceCredential"


class BearerCredential(Credential):
    """A fixed access token (an exchanged user token). Never refreshed; a 401 is terminal."""

    def __init__(self, token: str) -> None:
        if not visible_ascii(token):
            raise ValueError("A nonblank token without whitespace is required")
        self._header = "Bearer " + token

    def authorization(self) -> tuple[str, int]:
        return self._header, 1

    def __repr__(self) -> str:
        return "BearerCredential"


def _require_token(token: object) -> str:
    if not isinstance(token, str) or not visible_ascii(token):
        raise ValueError("A nonblank token without whitespace is required")
    return token


def _require_value(value: object) -> str:
    if not isinstance(value, str) or not visible_ascii(value):
        raise ValueError("Invalid OAuth value")
    return value


class TokenExchange:
    """One-shot user token exchanges with the service key's client. No cache, retry or fallback."""

    def __init__(self, http: httpx.Client, token_url: str, client_id: str, client_secret: str,
                 clock: Callable[[], float] = time.time) -> None:
        self._http = http
        self._token_url = token_url
        self._client_id = client_id
        self._client_secret = client_secret
        self._clock = clock

    def jwt_authorization_grant(self, assertion: str, scopes: Sequence[str] = ()) -> tuple[BearerCredential, datetime]:
        form = self._form("urn:ietf:params:oauth:grant-type:jwt-bearer")
        form.append(("assertion", _require_token(assertion)))
        self._scopes(form, scopes)
        return self._exchange(form)

    def standard_token_exchange(self, access_token: str, audiences: Sequence[str],
                                scopes: Sequence[str] = ()) -> tuple[BearerCredential, datetime]:
        form = self._form("urn:ietf:params:oauth:grant-type:token-exchange")
        form.append(("subject_token", _require_token(access_token)))
        form.append(("subject_token_type", _ACCESS_TOKEN_TYPE))
        form.append(("requested_token_type", _ACCESS_TOKEN_TYPE))
        targets = list(audiences)
        if not targets:
            raise ValueError("At least one target audience is required")
        for target in targets:
            form.append(("audience", _require_value(target)))
        self._scopes(form, scopes)
        return self._exchange(form)

    def _form(self, grant: str) -> list[tuple[str, str]]:
        return [("grant_type", grant), ("client_id", self._client_id), ("client_secret", self._client_secret)]

    @staticmethod
    def _scopes(form: list[tuple[str, str]], scopes: Sequence[str]) -> None:
        requested = [_require_value(scope) for scope in scopes]
        if requested:
            form.append(("scope", " ".join(requested)))

    def _exchange(self, form: list[tuple[str, str]]) -> tuple[BearerCredential, datetime]:
        started = self._clock()
        try:
            response = self._http.post(
                self._token_url,
                content=form_encode(form).encode("utf-8"),
                headers={"Content-Type": "application/x-www-form-urlencoded", "Accept": "application/json"},
                timeout=_TOKEN_TIMEOUT,
                follow_redirects=False,
            )
            if response.status_code != 200:
                raise _token_failure()
            data = json.loads(response.content)
            token = data.get("access_token")
            expires_in = data.get("expires_in")
            if (
                not isinstance(token, str)
                or isinstance(expires_in, bool)
                or not isinstance(expires_in, int)
                or expires_in <= 0
                or not isinstance(data.get("token_type"), str)
                or data["token_type"].lower() != "bearer"
            ):
                raise _token_failure()
            credential = BearerCredential(token)
        except AuthenticationError:
            raise
        except Exception:
            raise _token_failure() from None
        expires_at = datetime.fromtimestamp(started, tz=timezone.utc) + timedelta(seconds=expires_in)
        if datetime.fromtimestamp(self._clock(), tz=timezone.utc) >= expires_at:
            raise _token_failure()
        return credential, expires_at

    def __repr__(self) -> str:
        return "TokenExchange"
