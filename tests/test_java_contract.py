"""Wire parity with the Java SDK.

``fixtures/java_wire_contract.json`` was produced by the Java SDK (a throwaway program that printed
``SearchQuery.toJson()``, serialized the Workplace and provisioning models, and captured every HTTP
request the Java client sent to a local server). Each test builds the same request in Python and
asserts the JSON (parsed, so object key order is not significant), path, query string and headers
are identical.
"""

from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import urlsplit

import httpx
import pytest

import smartsearch_ai as ss
from smartsearch_ai import (
    AnyOf, AssignmentItem, CardinalityAgg, Combination, DisableItem, EnsureItem, ExternalIdentity, Filter, Grants,
    JobKind, JobState, Membership, MemberRole, MemoryMode, Mode, NeuralMode, Normalization, Output, Profile,
    QueryOptions, CachePolicy, RetryJob, Scope, ScopeKind, SearchQuery, SnapshotApply, SnapshotRequest,
    SnapshotRow, Sort, SortOrder, SourceGrant, SubmitJob, TermBoost, TermsAgg, WorkspaceMemberRequest,
    WorkspacePrincipalGrantRequest, WorkspaceQueryRequest, WorkspaceRequestContext,
)

FIXTURES = json.loads((Path(__file__).parent / "fixtures" / "java_wire_contract.json").read_text())

WS = "00000000-0000-4000-8000-000000000001"
SRC = "00000000-0000-4000-8000-000000000002"
PRN = "00000000-0000-4000-8000-000000000003"
JOB = "00000000-0000-4000-8000-000000000004"
SNAP = "00000000-0000-4000-8000-000000000005"
RID = "00000000-0000-4000-8000-000000000006"


def every_public_method() -> SearchQuery:
    return SearchQuery(
        "trail shoes", from_=10, size=20, fields=["title", "body"], response_fields=["id", "title"],
        sort=[Sort("price", SortOrder.ASC), ("rating", SortOrder.DESC)],
        filters=[
            Filter.term("brand", "acme"), Filter.terms("color", "red", "blue"), Filter.match("title", "running"),
            Filter.match_phrase("title", "trail shoe"), Filter.exists("stock"),
            Filter.range("price", gte=10, lte=100), Filter.range("weight", gt=0.5, lt=9.5),
            AnyOf(Filter.term("x", "1"), Filter.terms("y", "2", "3"), Filter.match("z", "4"),
                  Filter.match_phrase("z", "5 6"), Filter.exists("w")),
        ],
        exclude=[Filter.term("status", "retired"), Filter.terms("tag", "a", "b"), Filter.match("title", "kids"),
                 Filter.match_phrase("title", "gift card"), Filter.exists("recalled"), Filter.range("price", gt=1000)],
        boosts=[TermBoost("brand", "acme", 2.5)],
        aggs=[TermsAgg("brands", "brand"), TermsAgg("colors", "color", 25), CardinalityAgg("skus", "sku"),
              CardinalityAgg("skus2", "sku", 3000)],
        highlight=True, wildcard=True, auto_correct=False, trim_query=True, remove_special_chars=True,
        disable_query_expansion=True, precision=11, neural_mode=NeuralMode.EXACT_AND_BM25_FUSED,
        neural_matches=(10, 100), normalization=Normalization.DISTRIBUTION_BASED,
        combination=Combination.ARITHMETIC_MEAN, rank_window=60, rank_constant=100,
        rerank=False, rerank_first_stage_size=50, rerank_min_score=0.1,
    )


SEARCH_QUERIES = {
    "minimal": lambda: SearchQuery("shoes"),
    "defaults_only": lambda: SearchQuery(),
    "every_public_method": every_public_method,
    "paging_sort": lambda: SearchQuery("star", from_=20, size=5,
                                       sort=[Sort("year", SortOrder.DESC), Sort("title", SortOrder.ASC)]),
    # Java: fields() replaces (so the last call wins), responseFields() appends.
    "fields_response_fields": lambda: SearchQuery("star", fields=["title", "overview"],
                                                  response_fields=["title", "year", "genres"]),
    "range_int_and_double": lambda: SearchQuery("x", filters=[
        Filter.range("year", gte=1990, lte=1999), Filter.range("rating", gt=7.5, lt=10.0),
        Filter.range("votes", lte=100.0)]),
    "terms_required_and_excluded": lambda: SearchQuery("x", filters=[Filter.terms("genre", "Drama", "Comedy")],
                                                       exclude=[Filter.terms("genre", "Horror", "Thriller")]),
    "any_of": lambda: SearchQuery("x", filters=[AnyOf(Filter.term("a", "1"), Filter.terms("b", "2", "3")),
                                                AnyOf(Filter.exists("c"))]),
    "hybrid_rrf": lambda: SearchQuery("x", neural_mode=NeuralMode.A_KNN_AND_BM25, normalization=Normalization.RANK,
                                      combination=Combination.RRF, rank_window=10, rank_constant=1),
    "hybrid_min_max": lambda: SearchQuery("x", neural_mode=NeuralMode.EXACT_AND_BM25_RANKING,
                                          normalization=Normalization.MIN_MAX, combination=Combination.HARMONIC_MEAN,
                                          neural_matches=(1, 10000)),
    "rerank_null_after_true": lambda: SearchQuery("x", rerank=None),
    "rerank_null_only": lambda: SearchQuery("x", rerank=None),
    "rerank_tuning": lambda: SearchQuery("x", rerank=True, rerank_first_stage_size=30, rerank_min_score=0.25),
    "aggs": lambda: SearchQuery("x", aggs=[TermsAgg("g", "genres"), TermsAgg("y", "year", 5),
                                           CardinalityAgg("c", "director"), CardinalityAgg("c2", "director", 0)]),
    "flags_false": lambda: SearchQuery("x", highlight=False, wildcard=False, auto_correct=True, trim_query=False,
                                       remove_special_chars=False, precision=1),
    "boost_integral_weight": lambda: SearchQuery("x", boosts=[TermBoost("genre", "Drama", 3),
                                                              TermBoost("genre", "Comedy", 0.5)]),
    "trimmed_text": lambda: SearchQuery("  padded query  ", filters=[Filter.term(" genre ", " Drama ")]),
    "unicode": lambda: SearchQuery("café ü中 \"quoted\" \\ back"),
}


def test_every_search_fixture_is_covered():
    assert set(SEARCH_QUERIES) == set(FIXTURES["search_queries"])


@pytest.mark.parametrize("name", sorted(SEARCH_QUERIES))
def test_search_query_json_matches_java(name):
    ours = SEARCH_QUERIES[name]()
    assert json.loads(ours.to_json()) == FIXTURES["search_queries"][name]
    # Numbers keep their JSON type: 3 -> 3.0 for a weight, 1999 stays integral for a bound.
    assert _types(json.loads(ours.to_json())) == _types(FIXTURES["search_queries"][name])


def _types(node):
    if isinstance(node, dict):
        return {k: _types(v) for k, v in node.items()}
    if isinstance(node, list):
        return [_types(v) for v in node]
    return type(node).__name__


WORKSPACE_REQUESTS = {
    "minimal": lambda: WorkspaceQueryRequest(query="what is the refund policy"),
    "answer_full": lambda: WorkspaceQueryRequest(
        query="q", mode=Mode.ANSWER, session_id="sess-1", agent_id="agent-1", source_ids=[SRC], project_ids=["p1"],
        memory_mode=MemoryMode.STANDARD,
        options=QueryOptions(include_sources=False, max_subqueries=2, max_iterations=3, max_context_documents=5,
                             cache_policy=CachePolicy.BYPASS, output=Output.COMPACT)),
    "filter_direct_term": lambda: WorkspaceQueryRequest(
        query="q", filters={"search_type": "term", "field": "source_type", "value": "pdf"}),
    "filter_all": lambda: WorkspaceQueryRequest(query="q", filters={"all": [
        {"search_type": "terms", "field": "metadata.team", "values": ["a", "b"]},
        {"search_type": "match_phrase", "field": "title", "value": "Refund Policy"},
        {"search_type": "range", "field": "published_at", "lte": "2026-01-01", "gte": "2025-01-01"},
        {"search_type": "match", "field": " content ", "value": 5}]}),
    "memory_agentic": lambda: WorkspaceQueryRequest(query="q", memory_mode=MemoryMode.AGENTIC),
}

WORKSPACE_OTHER = {
    "member_canonical": lambda: WorkspaceMemberRequest.canonical(PRN),
    "member_owner": lambda: WorkspaceMemberRequest.canonical(PRN, MemberRole.OWNER),
    "member_legacy": lambda: WorkspaceMemberRequest.legacy("user@example.com", MemberRole.MEMBER),
    "grant": lambda: WorkspacePrincipalGrantRequest(display_label="Label A", groups=["g1"], roles=[],
                                                    security_keys=["k1", "k2"], expected_revision=3),
    "grant_null_label": lambda: WorkspacePrincipalGrantRequest(groups=[], roles=["r"], security_keys=[],
                                                               expected_revision=0),
}


@pytest.mark.parametrize("name", sorted(WORKSPACE_REQUESTS))
def test_workspace_request_matches_java(name):
    assert set(WORKSPACE_REQUESTS) == set(FIXTURES["workspace_requests"])
    assert WORKSPACE_REQUESTS[name]().to_dict() == FIXTURES["workspace_requests"][name]


@pytest.mark.parametrize("name", sorted(WORKSPACE_OTHER))
def test_workspace_models_match_java(name):
    assert set(WORKSPACE_OTHER) == set(FIXTURES["workspace_other"])
    assert WORKSPACE_OTHER[name]().to_dict() == FIXTURES["workspace_other"][name]


def _membership():
    return Membership(workspace_id=WS, source_grants=[
        SourceGrant(source_id=SRC, expected_revision=2, grants=Grants(groups=["g"], roles=[], security_keys=["k"]))])


def _item():
    return EnsureItem(item_key="item-1", external_user_id="ext-1", profile=Profile(email="u1@example.com",
                      first_name="Ada"), permissions=["zeta", "alpha"], workspaces=[_membership()])


SCOPE = Scope(kind=ScopeKind.TENANT, id="tenant-a")
ROWS = [SnapshotRow(principal_id=PRN, workspace_id=SRC), SnapshotRow(principal_id=JOB, workspace_id=WS),
        SnapshotRow(principal_id=PRN, workspace_id=WS)]

PROVISIONING = {
    "upsert_users": lambda: SubmitJob(scope=SCOPE, integration_id="integration-a", kind=JobKind.UPSERT_USERS, items=[
        _item(),
        EnsureItem(item_key="item-2", external_user_id="ext-2",
                   profile=Profile(email="u2@example.com", display_name="U Two"), permissions=[],
                   workspaces=[Membership(workspace_id=WS, expected_revision=7)],
                   external_identity=ExternalIdentity(subject="sub-2", username="user2"))]),
    "onboard_users": lambda: SubmitJob(scope=Scope(kind=ScopeKind.DEPLOYMENT, id="dep"), integration_id="integration-a",
                                       kind=JobKind.ONBOARD_USERS, items=[_item()]),
    "disable_users": lambda: SubmitJob(scope=SCOPE, integration_id="integration-a", kind=JobKind.DISABLE_USERS,
                                       items=[DisableItem(item_key="d1", principal_id=PRN)]),
    "set_memberships": lambda: SubmitJob(scope=SCOPE, kind=JobKind.SET_MEMBERSHIPS, items=[
        AssignmentItem(item_key="a1", principal_id=PRN, workspaces=[_membership()])]),
    "retry": lambda: RetryJob(item_keys=["item-1", "item-2"]),
    "snapshot_request": lambda: SnapshotRequest.for_rows(SCOPE, "integration-a", [SRC, WS], 4, ROWS),
    "snapshot_rows": lambda: [r.to_dict() for r in ROWS],
    "snapshot_apply": lambda: SnapshotApply(expected_generation=4, preview_digest="a" * 64),
    "digest_rows": lambda: ss.snapshot_digest(ROWS),
    "digest_empty": lambda: ss.snapshot_digest([]),
}


@pytest.mark.parametrize("name", sorted(PROVISIONING))
def test_provisioning_models_match_java(name):
    assert set(PROVISIONING) == set(FIXTURES["provisioning"])
    value = PROVISIONING[name]()
    assert (value.to_dict() if hasattr(value, "to_dict") else value) == FIXTURES["provisioning"][name]


# --- full HTTP requests -------------------------------------------------------------------------

BASE = "http://127.0.0.1:9"
HEADERS = ("content-type", "accept", "authorization", "idempotency-key", "x-request-id")


class Recorder:
    """Answers like the Java capture server and records each request."""

    def __init__(self):
        self.requests = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        body = request.content.decode()
        self.requests.append(request)
        if request.url.path.endswith("/token"):
            token = "user-token-1" if ("jwt-bearer" in body or "token-exchange" in body) else "svc-token-1"
            return httpx.Response(200, json={"access_token": token, "expires_in": 300, "token_type": "Bearer"})
        if '"stream":true' in body:
            sse = ('event: run.started\ndata: {"payload":{"session_id":"s1"}}\n\n'
                   'event: run.completed\ndata: {"payload":{"status":"COMPLETED"}}\n\n')
            return httpx.Response(200, content=sse.encode(), headers={"Content-Type": "text/event-stream"})
        return httpx.Response(200, json={"code": 1})


def drain(stream):
    with stream:
        for _ in stream:
            pass


ANSWER = WorkspaceQueryRequest(query="refunds", mode=Mode.ANSWER, session_id="s0")

CALLS = {
    "search": lambda c: c.search().search("movies project/1", SearchQuery("star wars")),
    "search_by_usecase": lambda c: c.search().search_by_usecase("uc-1", SearchQuery("x", size=3)),
    "multi_search": lambda c: c.search().multi_search("p*~1", [SearchQuery("a"), SearchQuery("b", from_=10)]),
    "workplace_search": lambda c: c.workplace().search(WS, WorkspaceQueryRequest(query="refunds")),
    "workplace_query_ctx": lambda c: c.workplace().query(WS, ANSWER, WorkspaceRequestContext(request_id=RID)),
    "workplace_chat": lambda c: c.workplace().chat(WS, ANSWER),
    "workplace_stream_query": lambda c: drain(c.workplace().stream_query(WS, ANSWER)),
    "workplace_stream_chat": lambda c: drain(c.workplace().stream_chat(WS, ANSWER)),
    "list_members": lambda c: c.workplace().list_members(WS),
    "list_members_page": lambda c: c.workplace().list_members_page(WS, "ada lovelace&x", 2, 50),
    "list_members_page_null_query": lambda c: c.workplace().list_members_page(WS, None, 0, 1),
    "add_member": lambda c: c.workplace().add_member(WS, WorkspaceMemberRequest.canonical(PRN)),
    "get_grant": lambda c: c.workplace().get_principal_grant(WS, SRC, PRN),
    "put_grant": lambda c: c.workplace().put_principal_grant(WS, SRC, PRN, WorkspacePrincipalGrantRequest(
        groups=["g"], roles=[], security_keys=[], expected_revision=1)),
    "revoke_grant": lambda c: c.workplace().revoke_principal_grant(WS, SRC, PRN, 5),
    "capabilities": lambda c: c.users().capabilities(),
    "submit_job": lambda c: c.users().submit_job(SubmitJob(
        scope=Scope(kind=ScopeKind.TENANT, id="t"), integration_id="integration-a", kind=JobKind.DISABLE_USERS,
        items=[DisableItem(item_key="d1", principal_id=PRN)]), "idem-key-1"),
    "get_job": lambda c: c.users().get_job(JOB),
    "list_job_items": lambda c: c.users().list_job_items(JOB, after="cursor a", limit=10, state=JobState.FAILED),
    "retry_job": lambda c: c.users().retry_job(JOB, RetryJob(item_keys=["k"]), "idem-2"),
    "cancel_job": lambda c: c.users().cancel_job(JOB),
    "list_principals": lambda c: c.users().list_principals(
        "integration-a", Scope(kind=ScopeKind.TENANT, id="tenant a/b"), after="after+1", limit=25),
    "list_principals_no_after": lambda c: c.users().list_principals(
        "integration-a", Scope(kind=ScopeKind.DEPLOYMENT, id="d"), limit=100),
    "create_snapshot": lambda c: c.users().create_snapshot(SnapshotRequest.for_rows(
        Scope(kind=ScopeKind.TENANT, id="t"), "integration-a", [WS], 1, [SnapshotRow(principal_id=PRN, workspace_id=WS)])),
    "upload_snapshot_page": lambda c: c.users().upload_snapshot_page(
        SNAP, 0, [SnapshotRow(principal_id=PRN, workspace_id=WS)]),
    "seal_snapshot": lambda c: c.users().seal_snapshot(SNAP),
    "get_snapshot": lambda c: c.users().get_snapshot(SNAP),
    "preview_snapshot": lambda c: c.users().preview_snapshot(SNAP, after=0, limit=100),
    "apply_snapshot": lambda c: c.users().apply_snapshot(SNAP, SnapshotApply(expected_generation=1,
                                                                            preview_digest="b" * 64), "idem-3"),
    "as_user": lambda c: _with_user(c.as_user("header.payload.sig"),
                                    lambda u: u.search(WS, WorkspaceQueryRequest(query="mine"))),
    "as_user_from_token": lambda c: _with_user(c.as_user_from_token("human.access.token"),
                                               lambda u: u.list_members(WS)),
}


def _with_user(user, call):
    with user:
        call(user)


def client(recorder) -> ss.SmartSearchAi:
    c = ss.SmartSearchAi(api_url=BASE + "/", admin_url=BASE + "/admin-root", auth_url=BASE + "/auth/",
                         realm="demo realm", client_id="svc-client", client_secret="s3cr3t+/=&x")
    c._http._transport = httpx.MockTransport(recorder)  # test seam: no network
    return c


def test_every_http_fixture_is_covered():
    assert set(CALLS) == set(FIXTURES["http"])


@pytest.mark.parametrize("name", sorted(CALLS))
def test_http_requests_match_java(name):
    recorder = Recorder()
    c = client(recorder)
    # The Java run fetched the service token once, during its first call ("search").
    if name != "search":
        c._credential.authorization()
        recorder.requests.clear()
    try:
        CALLS[name](c)
    finally:
        c.close()
    expected = FIXTURES["http"][name]
    assert len(recorder.requests) == len(expected)
    for ours, theirs in zip(recorder.requests, expected):
        uri = urlsplit(str(ours.url))
        assert ours.method == theirs["method"]
        assert uri.path + (f"?{uri.query}" if uri.query else "") == theirs["uri"]
        got = {k: ours.headers[k] for k in HEADERS if k in ours.headers}
        want = dict(theirs["headers"])
        if uri.path.endswith("/token") and "client_credentials" in theirs["body"]:
            # Java's RestTemplate sends a generic Accept list here; both accept the JSON token response.
            want["accept"] = "application/json"
        assert got == want
        body = ours.content.decode()
        if want.get("content-type", "").startswith("application/json") and theirs["body"]:
            assert json.loads(body) == json.loads(theirs["body"])
        else:
            assert body == theirs["body"]
