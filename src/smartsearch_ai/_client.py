"""The SDK entry point."""

from __future__ import annotations

import threading
from typing import Sequence

import httpx

from ._auth import ServiceCredential, TokenExchange
from ._util import normalize_base_url, token_endpoint
from ._version import __version__
from .provisioning import Users
from .search import ProjectSearch
from .workplace import UserWorkplace, Workplace, _Transport

DEFAULT_EXCHANGE_AUDIENCES = ("workspace-api", "cloud-gateway", "search-admin")


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must not be blank")
    return value.strip()


def _timeout(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 < value <= 2_147_483:
        raise ValueError(f"{name} must be a positive number of seconds")
    return float(value)


class SmartSearchAi:
    """Entry point to the SmartSearch AI SDK.

    Authenticates with your service key (OAuth2 client credentials against ``auth_url``/``realm``)
    and can act as one of your end users through a token exchange. Thread-safe: create one per
    configuration and close it on shutdown, preferably with ``with``::

        with SmartSearchAi(api_url="https://api.example.com", admin_url="https://admin.example.com",
                           auth_url="https://auth.example.com", realm="example",
                           client_id=client_id, client_secret=client_secret) as ss:
            hits = ss.search().search(project_id, SearchQuery("shoes"))

    Args:
        api_url: gateway URL for project search and Workplace.
        admin_url: Search Admin URL for user provisioning.
        auth_url: identity-provider base URL (the part before ``/realms/...``). HTTPS required,
            except plain HTTP to a loopback host.
        realm: identity-provider realm.
        client_id, client_secret: your service key. Also used for user token exchange.
        connect_timeout: seconds to establish a connection (default 5).
        read_timeout: seconds to wait for a response or the next stream event (default 60;
            answer and chat runs can take a minute).
        exchange_audiences: audiences requested by :meth:`as_user_from_token`.

    Raises:
        ValueError: a setting is missing or invalid.
    """

    def __init__(
        self,
        *,
        api_url: str,
        admin_url: str,
        auth_url: str,
        realm: str,
        client_id: str,
        client_secret: str,
        connect_timeout: float = 5.0,
        read_timeout: float = 60.0,
        exchange_audiences: Sequence[str] = DEFAULT_EXCHANGE_AUDIENCES,
    ) -> None:
        self._api_url = normalize_base_url(api_url, "api_url")
        self._admin_url = normalize_base_url(admin_url, "admin_url")
        self._realm = _text(realm, "realm")
        token_url = token_endpoint(_text(auth_url, "auth_url"), self._realm)
        client_id = _text(client_id, "client_id")
        client_secret = _text(client_secret, "client_secret")
        connect = _timeout(connect_timeout, "connect_timeout")
        self._read_timeout = _timeout(read_timeout, "read_timeout")
        audiences = tuple(exchange_audiences) if not isinstance(exchange_audiences, str) else ()
        if not audiences or any(not isinstance(a, str) or not a.strip() for a in audiences):
            raise ValueError("exchange_audiences must be a non-empty list of nonblank values")
        self._audiences = audiences

        # One connection pool for every call; redirects are never followed (a redirect could
        # carry the Authorization header or the token form to another host).
        self._http = httpx.Client(
            timeout=httpx.Timeout(self._read_timeout, connect=connect),
            follow_redirects=False,
            headers={"User-Agent": f"smartsearch-ai-python/{__version__}"},
        )
        self._credential = ServiceCredential(self._http, token_url, client_id, client_secret)
        self._exchange = TokenExchange(self._http, token_url, client_id, client_secret)
        self._api = _Transport(self._http, self._api_url, self._credential, self._read_timeout)
        admin = self._api if self._admin_url == self._api_url else _Transport(
            self._http, self._admin_url, self._credential, self._read_timeout)
        self._admin = admin
        self._search = ProjectSearch(self._http, self._api_url, self._credential, self._read_timeout)
        self._workplace = Workplace(self._api)
        self._users = Users(admin)
        self._closed = False
        self._lock = threading.Lock()

    def search(self) -> ProjectSearch:
        """Project search as the service key."""
        self._require_open()
        return self._search

    def workplace(self) -> Workplace:
        """Workplace as the service key (service-only search)."""
        self._require_open()
        return self._workplace

    def users(self) -> Users:
        """User provisioning (Search Admin provisioning API on ``admin_url``) as the service key."""
        self._require_open()
        return self._users

    def as_user(self, assertion: str) -> UserWorkplace:
        """Acts as the user identified by an assertion (a JWT) from your identity provider, using
        the JWT authorization grant. Use a fresh assertion per call; close the result when done.

        Raises:
            AuthenticationError: the identity provider refused the assertion.
        """
        self._require_open()
        credential, expires_at = self._exchange.jwt_authorization_grant(assertion)
        return UserWorkplace(_Transport(self._http, self._api_url, credential, self._read_timeout), expires_at)

    def as_user_from_token(self, access_token: str) -> UserWorkplace:
        """Acts as the user holding ``access_token`` (standard token exchange to the configured
        audiences). Close the result when done.

        Raises:
            AuthenticationError: the exchange was refused.
        """
        self._require_open()
        credential, expires_at = self._exchange.standard_token_exchange(access_token, self._audiences)
        return UserWorkplace(_Transport(self._http, self._api_url, credential, self._read_timeout), expires_at)

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("SmartSearchAi is closed")

    def close(self) -> None:
        """Closes open streams and the connection pool. User workplaces stop working. Idempotent."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
        self._api.close_streams()
        self._admin.close_streams()
        self._http.close()

    def __enter__(self) -> SmartSearchAi:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def __repr__(self) -> str:
        return f"SmartSearchAi(api_url={self._api_url!r}, admin_url={self._admin_url!r}, realm={self._realm!r})"
