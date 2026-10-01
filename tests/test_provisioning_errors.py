"""Named provisioning codes: same names and values as the public Java SDK, and server codes reach callers as them.

``fixtures/java_named_codes.json`` was printed from the public Java jar (``ProvisioningClientException``'s
public static final String fields).
"""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

import smartsearch_ai as ss
from smartsearch_ai import errors

JAVA = json.loads((Path(__file__).parent / "fixtures" / "java_named_codes.json").read_text())
JOB = "00000000-0000-4000-8000-000000000004"
NAMES = ("INTEGRATION_NOT_FOUND", "EXTERNAL_IDENTITY_REQUIRED", "EXTERNAL_IDENTITY_NOT_SUPPORTED")


def test_named_codes_match_java():
    assert {name: getattr(ss, name) for name in NAMES} == JAVA["ProvisioningClientException"]


@pytest.mark.parametrize("status,wire,error", [
    (404, "INTEGRATION_NOT_FOUND", ss.NotFoundError),
    (400, "EXTERNAL_IDENTITY_REQUIRED", ss.SmartSearchError),
    (400, "EXTERNAL_IDENTITY_NOT_SUPPORTED", ss.SmartSearchError),
])
def test_server_code_reaches_caller_as_named_code(status, wire, error):
    def answer(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/token"):
            return httpx.Response(200, json={"access_token": "t", "expires_in": 300, "token_type": "Bearer"})
        return httpx.Response(status, json={"code": wire, "retryable": False})

    c = ss.SmartSearchAi(api_url="http://127.0.0.1", admin_url="http://127.0.0.1", auth_url="http://127.0.0.1", realm="r",
                         client_id="c", client_secret="s")
    c._http._transport = httpx.MockTransport(answer)  # test seam: no network
    try:
        with pytest.raises(error) as raised:
            c.users().get_job(JOB)
    finally:
        c.close()
    assert raised.value.status_code == status
    assert raised.value.code == getattr(errors, wire)
