"""Internal helpers: URL and identifier validation, Java-compatible encoding, error-text sanitization."""

from __future__ import annotations

import re
import uuid
from urllib.parse import urlsplit

_UNRESERVED = frozenset(b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789.-*_")


def url_encode(value: str, *, space: str = "%20") -> str:
    """Percent-encodes like the server-facing Java SDK (URLEncoder): only ``A-Za-z0-9.-*_`` stay literal."""
    out = []
    for byte in value.encode("utf-8"):
        if byte in _UNRESERVED:
            out.append(chr(byte))
        elif byte == 0x20:
            out.append(space)
        else:
            out.append("%%%02X" % byte)
    return "".join(out)


def form_encode(pairs: list[tuple[str, str]]) -> str:
    """``application/x-www-form-urlencoded`` body, spaces as ``+``."""
    return "&".join(f"{name}={url_encode(value, space='+')}" for name, value in pairs)


def is_iso_control(ch: str) -> bool:
    code = ord(ch)
    return code <= 0x1F or 0x7F <= code <= 0x9F


def strip_controls(value: str) -> str:
    return "".join(ch for ch in value if not is_iso_control(ch))


def visible_ascii(value: str, *, allow_space: bool = False) -> bool:
    low = 32 if allow_space else 33
    return bool(value) and all(low <= ord(ch) <= 126 for ch in value)


def normalize_base_url(url: str | None, name: str) -> str:
    """An HTTP(S) base URL without credentials, query or fragment; trailing slashes removed."""
    if url is None or not str(url).strip():
        raise ValueError(f"{name} is required")
    trimmed = str(url).strip()
    while trimmed.endswith("/"):
        trimmed = trimmed[:-1].strip()
    try:
        parts = urlsplit(trimmed)
        host = parts.hostname
    except ValueError:
        raise ValueError(f"{name} must be an HTTP(S) URL") from None
    if (
        parts.scheme.lower() not in ("http", "https")
        or not host
        or "@" in parts.netloc
        or parts.query
        or parts.fragment
        or "?" in trimmed
        or "#" in trimmed
    ):
        raise ValueError(f"{name} must be an HTTP(S) URL without credentials, query or fragment")
    return trimmed


def token_endpoint(auth_url: str, realm: str) -> str:
    """``{auth_url}/realms/{realm}/protocol/openid-connect/token``; HTTPS, or HTTP to loopback only."""
    base = normalize_base_url(auth_url, "auth_url")
    endpoint = f"{base}/realms/{url_encode(realm.strip())}/protocol/openid-connect/token"
    parts = urlsplit(endpoint)
    loopback = (parts.hostname or "").lower() in ("localhost", "127.0.0.1", "::1")
    if not (parts.scheme.lower() == "https" or (parts.scheme.lower() == "http" and loopback)):
        raise ValueError("auth_url must use HTTPS (plain HTTP is accepted for loopback hosts only)")
    return endpoint


def canonical_uuid(value: object, name: str) -> str:
    """A canonical lowercase UUID; rejects encoded, slash-, query- or dot-bearing values."""
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} is required")
    if value != value.strip():
        raise ValueError(f"{name} must not have surrounding whitespace")
    if any(is_iso_control(ch) or ch in "%/?#." for ch in value):
        raise ValueError(f"{name} must be a canonical UUID; got unsafe character")
    try:
        canonical = str(uuid.UUID(value))
    except ValueError:
        raise ValueError(f"{name} must be a canonical UUID") from None
    if canonical != value:
        raise ValueError(f"{name} must be a canonical lowercase UUID")
    return canonical


def is_canonical_uuid(value: object) -> bool:
    if not isinstance(value, str) or len(value) != 36:
        return False
    try:
        return str(uuid.UUID(value)) == value
    except ValueError:
        return False


def path_segment(value: object, name: str) -> str:
    """A nonblank path segment other than ``.``/``..``, percent-encoded."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} is required")
    if value in (".", ".."):
        raise ValueError(f"{name} is invalid")
    return url_encode(value)


# --- error-text sanitization (never echo credentials) ---------------------------------------

MAX_CODE_LENGTH = 64
MAX_MESSAGE_LENGTH = 256
MAX_RAW_BODY_LENGTH = 512
REDACTED = "[REDACTED]"

_SECRET_PATTERNS = re.compile(
    r"(?i)(authorization\s*[:=]\s*['\"]?(?:bearer|apikey)\s+[^\s'\"]+)"
    r"|(bearer\s+[A-Za-z0-9._\-+/=]+)"
    r"|(apikey\s+[A-Za-z0-9._\-+/=]+)"
    r"|(client_secret\s*[:=]\s*['\"]?[^\s'\"]+)"
    r"|(access_token\s*[:=]\s*['\"]?[^\s'\"]+)"
)
_JSON_SECRET_FIELDS = re.compile(
    r"(?i)\"(authorization|token|access_token|client_secret|api_key|apikey)\"\s*:\s*\"[^\"]*\""
)


def redact(value: str, secrets: tuple[str, ...] = ()) -> str | None:
    """Redacts known secrets and secret-shaped text. ``None`` when a short secret would remain."""
    cleaned = value
    for secret in secrets:
        if not secret:
            continue
        if len(secret) >= 6:
            cleaned = cleaned.replace(secret, REDACTED)
        elif secret in cleaned:
            return None
    cleaned = _JSON_SECRET_FIELDS.sub(lambda m: f'"{m.group(1)}":"{REDACTED}"', cleaned)
    cleaned = _SECRET_PATTERNS.sub(REDACTED, cleaned)
    return cleaned


def auth_secrets(authorization: str | None) -> tuple[str, ...]:
    """The Authorization value and its credential part, for redaction."""
    if not authorization:
        return ()
    _, _, credential = authorization.partition(" ")
    return (authorization, credential) if credential else (authorization,)


def safe_text(value: object, fallback: str | None, max_length: int, secrets: tuple[str, ...] = ()) -> str | None:
    if not isinstance(value, str) or not value:
        return fallback
    cleaned = redact(strip_controls(value).strip(), secrets)
    if not cleaned:
        return fallback
    return cleaned[:max_length]


def safe_request_id(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    cleaned = strip_controls(value).strip()
    return cleaned if is_canonical_uuid(cleaned) else None


def safe_raw_body(value: str | None, secrets: tuple[str, ...] = ()) -> str | None:
    if value is None:
        return None
    cleaned = strip_controls(value).strip()
    if not cleaned:
        return None
    cleaned = redact(cleaned, secrets)
    return None if cleaned is None else cleaned[:MAX_RAW_BODY_LENGTH]
