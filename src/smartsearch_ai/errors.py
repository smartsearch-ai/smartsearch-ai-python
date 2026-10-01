"""Exceptions raised by the SDK.

Every failure the server reports, or that stops a request from completing, is a
:class:`SmartSearchError`. Subclasses narrow it by cause. Exceptions carry only the HTTP status,
the server's error code and message, and a request ID. They never carry tokens, client secrets,
assertions or request bodies. Invalid arguments raise :class:`ValueError` before anything is sent.
"""

from __future__ import annotations


class SmartSearchError(Exception):
    """Base class for SDK errors.

    Attributes:
        status_code: HTTP status, or 0 when no response was received.
        code: server error code (for example ``FEATURE_DISABLED``), or an SDK code such as
            ``WORKSPACE_REQUEST_TIMEOUT``. For project search it is the API's numeric ``code`` as text.
        request_id: server request ID, when the response had a canonical one.
        retryable: whether the server marked the failure retryable (provisioning only).
        server_message / error_message: project-search ``message`` and ``errorMessage``.
        raw_body: a redacted, capped excerpt of the Workplace error body, for diagnostics.
    """

    def __init__(
        self,
        message: str,
        *,
        status_code: int = 0,
        code: str | None = None,
        request_id: str | None = None,
        retryable: bool = False,
        server_message: str | None = None,
        error_message: str | None = None,
        raw_body: str | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.code = code
        self.request_id = request_id
        self.retryable = retryable
        self.server_message = server_message
        self.error_message = error_message
        self.raw_body = raw_body

    def __str__(self) -> str:
        details = []
        if self.status_code:
            details.append(f"HTTP {self.status_code}")
        if self.code:
            details.append(self.code)
        return f"{self.message} ({', '.join(details)})" if details else self.message

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(status_code={self.status_code}, code={self.code!r}, "
            f"request_id={self.request_id!r})"
        )


class AuthenticationError(SmartSearchError):
    """HTTP 401, or a token could not be obtained (service key or user exchange)."""


class PermissionDeniedError(SmartSearchError):
    """HTTP 403: authenticated, but not allowed."""


class NotFoundError(SmartSearchError):
    """HTTP 404."""


class ConflictError(SmartSearchError):
    """HTTP 409, for example a stale ``expected_revision``. Never reconciled automatically."""


class RequestTimeoutError(SmartSearchError):
    """No complete response or stream event within ``read_timeout``."""


class TransportError(SmartSearchError):
    """The request could not be sent or the connection failed (``status_code`` 0)."""


# Named provisioning codes: ``SmartSearchError.code`` of a failed call, and the ``code`` of a failed job
# item's ``failure`` stage (``Users.list_job_items``), where identity checks run per user.

INTEGRATION_NOT_FOUND = "INTEGRATION_NOT_FOUND"
"""No integration with this ID in the scope (HTTP 404). An administrator creates integrations."""

EXTERNAL_IDENTITY_REQUIRED = "EXTERNAL_IDENTITY_REQUIRED"
"""Federated integration: the user was sent without ``external_identity`` (subject + username)."""

EXTERNAL_IDENTITY_NOT_SUPPORTED = "EXTERNAL_IDENTITY_NOT_SUPPORTED"
"""Invite integration: the user was sent with an ``external_identity``; invite users carry none."""


def error_type(status_code: int) -> type[SmartSearchError]:
    return {
        401: AuthenticationError,
        403: PermissionDeniedError,
        404: NotFoundError,
        409: ConflictError,
    }.get(status_code, SmartSearchError)


__all__ = [
    "SmartSearchError",
    "AuthenticationError",
    "PermissionDeniedError",
    "NotFoundError",
    "ConflictError",
    "RequestTimeoutError",
    "TransportError",
    "INTEGRATION_NOT_FOUND",
    "EXTERNAL_IDENTITY_REQUIRED",
    "EXTERNAL_IDENTITY_NOT_SUPPORTED",
]
