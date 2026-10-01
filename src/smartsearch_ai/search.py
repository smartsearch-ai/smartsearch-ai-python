"""Project search: typed SmartSearch queries (SSPL) sent to ``/core`` on the API gateway."""

from __future__ import annotations

import copy
import json
import math
from dataclasses import dataclass
from enum import Enum
from typing import Any, Iterable, Sequence, Union

import httpx

from ._auth import Credential
from ._util import path_segment
from .errors import RequestTimeoutError, SmartSearchError, TransportError, error_type


class SortOrder(str, Enum):
    ASC = "ASC"
    DESC = "DESC"


class NeuralMode(str, Enum):
    BM25 = "BM25"
    A_KNN = "A_KNN"
    A_KNN_AND_BM25 = "A_KNN_AND_BM25"
    EXACT_AND_BM25_FUSED = "EXACT_AND_BM25_FUSED"
    EXACT_AND_BM25_RANKING = "EXACT_AND_BM25_RANKING"


class Normalization(str, Enum):
    MIN_MAX = "MIN_MAX"
    L2 = "L2"
    Z_SCORE = "Z_SCORE"
    RANK = "RANK"
    DISTRIBUTION_BASED = "DISTRIBUTION_BASED"
    NONE = "NONE"


class Combination(str, Enum):
    ARITHMETIC_MEAN = "ARITHMETIC_MEAN"
    GEOMETRIC_MEAN = "GEOMETRIC_MEAN"
    HARMONIC_MEAN = "HARMONIC_MEAN"
    RRF = "RRF"


def _text(name: str, value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} is required")
    return value.strip()


def _int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    return value


def _finite(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return float(value)


def _bound(name: str, value: object) -> int | float:
    # Integral bounds stay integral, others become floats (as the Java SDK sends them).
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"range bound {name} must be a number")
    return value if isinstance(value, int) else _finite(name, value)


def _enum(enum_type: type[Enum], name: str, value: object) -> Any:
    try:
        return enum_type(value)
    except ValueError:
        raise ValueError(f"{name} must be one of {[e.value for e in enum_type]}") from None


class Filter:
    """One typed filter condition for ``SearchQuery(filters=..., exclude=...)`` and :class:`AnyOf`. Immutable."""

    __slots__ = ("_clauses", "_any_of_values")

    def __init__(self, clauses: list[dict[str, Any]], any_of_values: bool = False) -> None:
        self._clauses = tuple(clauses)
        self._any_of_values = any_of_values

    @staticmethod
    def _clause(kind: str, field: str, value: str) -> dict[str, Any]:
        return {"search_type": kind, "field": _text("field", field), "value": _text("value", value)}

    @classmethod
    def term(cls, field: str, value: str) -> Filter:
        """Exact value match (``term``)."""
        return cls([cls._clause("term", field, value)])

    @classmethod
    def terms(cls, field: str, *values: str) -> Filter:
        """Field equals any of the values. Under ``exclude`` every value is excluded."""
        if not values:
            raise ValueError("values must contain at least one value")
        return cls([cls._clause("term", field, v) for v in values], any_of_values=True)

    @classmethod
    def match(cls, field: str, value: str) -> Filter:
        """Analyzed full-text match (``match``)."""
        return cls([cls._clause("match", field, value)])

    @classmethod
    def match_phrase(cls, field: str, value: str) -> Filter:
        """Analyzed phrase match (``match_phrase``)."""
        return cls([cls._clause("match_phrase", field, value)])

    @classmethod
    def exists(cls, field: str) -> Filter:
        """Field has a value (``exists``)."""
        return cls([{"search_type": "exists", "field": _text("field", field)}])

    @classmethod
    def range(cls, field: str, *, gte: float | None = None, lte: float | None = None,
              gt: float | None = None, lt: float | None = None) -> Filter:
        """Numeric range; at least one bound is required."""
        clause: dict[str, Any] = {"search_type": "range", "field": _text("field", field)}
        condition = {k: _bound(k, v) for k, v in (("gte", gte), ("lte", lte), ("gt", gt), ("lt", lt)) if v is not None}
        if not condition:
            raise ValueError("range filter requires at least one bound")
        clause["condition"] = condition
        return cls([clause])

    def _required_entry(self) -> dict[str, Any]:
        if not self._any_of_values:
            return copy.deepcopy(self._clauses[0])
        return {"any": copy.deepcopy(list(self._clauses))}

    def _excluded(self) -> list[dict[str, Any]]:
        return copy.deepcopy(list(self._clauses))

    def __repr__(self) -> str:
        return f"Filter{list(self._clauses)}"


class AnyOf:
    """Hits must match at least one of the filters (an ``any`` group in ``filters.all``)."""

    __slots__ = ("_filters",)

    def __init__(self, *filters: Filter) -> None:
        if not filters:
            raise ValueError("AnyOf requires at least one filter")
        for f in filters:
            if not isinstance(f, Filter):
                raise ValueError("AnyOf accepts Filter values only")
        self._filters = filters

    def _required_entry(self) -> dict[str, Any]:
        return {"any": [c for f in self._filters for c in f._excluded()]}

    def __repr__(self) -> str:
        return f"AnyOf{list(self._filters)}"


@dataclass(frozen=True)
class Sort:
    field: str
    order: SortOrder = SortOrder.ASC


@dataclass(frozen=True)
class TermBoost:
    """Scores hits whose ``field`` equals ``value`` higher by ``weight``."""

    field: str
    value: str
    weight: float


@dataclass(frozen=True)
class TermsAgg:
    """Terms aggregation (facet counts); ``size`` None uses the server default."""

    name: str
    field: str
    size: int | None = None


@dataclass(frozen=True)
class CardinalityAgg:
    """Approximate distinct count."""

    name: str
    field: str
    precision_threshold: int | None = None


Agg = Union[TermsAgg, CardinalityAgg]


class SearchQuery:
    """Immutable, typed project-search request. Only these options exist: there is no raw query,
    free-form flag or other escape hatch. Invalid values raise :class:`ValueError` before sending.

    Args:
        q: query text.
        from_: zero-based result offset (default 0). size: page size (default 10).
        fields: fields searched by ``q``. response_fields: fields returned per hit; name each field
            (no ``*`` and no internal ``_ss_ml_`` fields).
        sort: :class:`Sort` values (or ``(field, SortOrder)`` pairs), applied in order.
        filters: :class:`Filter` or :class:`AnyOf` values every hit must match.
        exclude: :class:`Filter` values no hit may match.
        boosts: :class:`TermBoost` values. aggs: :class:`TermsAgg` / :class:`CardinalityAgg` values.
        highlight, wildcard, auto_correct, trim_query, remove_special_chars: per-request switches;
            None leaves the project setting.
        disable_query_expansion: True turns query expansion off for this request.
        precision: match precision 1 (broad) to 11 (strict).
        neural_mode: keyword, vector or hybrid retrieval.
        neural_matches: ``(top, total)`` vector candidates, 1 <= top <= total <= 10000.
        normalization, combination: hybrid score fusion; give both. RRF pairs only with RANK, and
            NONE cannot combine with GEOMETRIC_MEAN or HARMONIC_MEAN.
        rank_window (>= 10), rank_constant (>= 1): hybrid rank window and RRF constant.
        rerank: True/False, or None for the project default. rerank_first_stage_size (> 0),
            rerank_min_score: reranking tuning.
    """

    __slots__ = ("_body",)

    def __init__(
        self,
        q: str | None = None,
        *,
        from_: int = 0,
        size: int = 10,
        fields: Sequence[str] | None = None,
        response_fields: Sequence[str] | None = None,
        sort: Iterable[Sort | tuple[str, SortOrder]] = (),
        filters: Iterable[Filter | AnyOf] = (),
        exclude: Iterable[Filter] = (),
        boosts: Iterable[TermBoost] = (),
        aggs: Iterable[Agg] = (),
        highlight: bool | None = None,
        wildcard: bool | None = None,
        auto_correct: bool | None = None,
        trim_query: bool | None = None,
        remove_special_chars: bool | None = None,
        disable_query_expansion: bool = False,
        precision: int | None = None,
        neural_mode: NeuralMode | str | None = None,
        neural_matches: tuple[int, int] | None = None,
        normalization: Normalization | str | None = None,
        combination: Combination | str | None = None,
        rank_window: int | None = None,
        rank_constant: int | None = None,
        rerank: bool | None = None,
        rerank_first_stage_size: int | None = None,
        rerank_min_score: float | None = None,
    ) -> None:
        body: dict[str, Any] = {"from": _int("from_", from_), "size": _int("size", size)}
        if body["from"] < 0:
            raise ValueError("from must be >= 0")
        if body["size"] <= 0:
            raise ValueError("size must be > 0")
        if q is not None:
            body["q"] = _text("q", q)
        if fields is not None:
            body["fields"] = _names("fields", fields)
        if response_fields is not None:
            names = _names("response_fields", response_fields)
            for name in names:
                if "*" in name or name.lower().startswith("_ss_ml_"):
                    raise ValueError(
                        "response_fields must name fields explicitly (no '*' or internal _ss_ml_ fields): " + name)
            body["response_fields"] = names
        sorts = []
        for entry in sort:
            s = entry if isinstance(entry, Sort) else Sort(*entry)
            order = _enum(SortOrder, "order", s.order)
            sorts.append({"field": _text("field", s.field), "order": order.value.lower()})
        if sorts:
            body["sorts"] = sorts

        all_entries, not_entries = [], []
        for f in filters:
            if not isinstance(f, (Filter, AnyOf)):
                raise ValueError("filters accepts Filter and AnyOf values")
            all_entries.append(f._required_entry())
        for f in exclude:
            if not isinstance(f, Filter):
                raise ValueError("exclude accepts Filter values only")
            not_entries.extend(f._excluded())
        if all_entries or not_entries:
            body["filters"] = {}
            if all_entries:
                body["filters"]["all"] = all_entries
            if not_entries:
                body["filters"]["not"] = not_entries

        boost_list = []
        for b in boosts:
            if not isinstance(b, TermBoost):
                raise ValueError("boosts accepts TermBoost values")
            boost_list.append({"search_type": "term", "field": _text("field", b.field),
                               "value": _text("value", b.value), "weight": _finite("weight", b.weight)})
        if boost_list:
            body["boosts"] = boost_list

        agg_list = []
        for a in aggs:
            if isinstance(a, TermsAgg):
                agg = {"type": "terms", "name": _text("name", a.name), "field": _text("field", a.field)}
                if a.size is not None:
                    if _int("size", a.size) <= 0:
                        raise ValueError("terms aggregation size must be > 0")
                    agg["size"] = a.size
            elif isinstance(a, CardinalityAgg):
                agg = {"type": "cardinality", "name": _text("name", a.name), "field": _text("field", a.field)}
                if a.precision_threshold is not None:
                    if _int("precision_threshold", a.precision_threshold) < 0:
                        raise ValueError("precision_threshold must be >= 0")
                    agg["precisionThreshold"] = a.precision_threshold
            else:
                raise ValueError("aggs accepts TermsAgg and CardinalityAgg values")
            agg_list.append(agg)
        if agg_list:
            body["aggs"] = agg_list

        flags: dict[str, Any] = {}
        for key, value in (("highlight_enabled", highlight), ("wild_card_search", wildcard),
                           ("auto_correct", auto_correct), ("trim_query", trim_query),
                           ("remove_special_chars", remove_special_chars)):
            if value is not None:
                flags[key] = _bool(key, value)
        if disable_query_expansion:
            flags["query_expansion_enable"] = False
        if precision is not None:
            if not 1 <= _int("precision", precision) <= 11:
                raise ValueError("precision must be 1..11")
            flags["precision"] = precision
        if neural_mode is not None:
            flags["neural_mode"] = _enum(NeuralMode, "neural_mode", neural_mode).value
        if neural_matches is not None:
            top, total = (_int("neural_matches", v) for v in neural_matches)
            if top < 1 or total < 1 or top > total or total > 10_000:
                raise ValueError("neural matches require 1 <= top <= total <= 10000")
            flags["neural_top_matches"] = top
            flags["neural_total_matches"] = total
        if normalization is not None or combination is not None:
            if normalization is None or combination is None:
                raise ValueError("normalization and combination are required together")
            n = _enum(Normalization, "normalization", normalization)
            c = _enum(Combination, "combination", combination)
            if n is Normalization.NONE and c in (Combination.GEOMETRIC_MEAN, Combination.HARMONIC_MEAN):
                raise ValueError(f"NONE normalization cannot be combined with {c.value}")
            # RRF fuses rank positions; the means fuse scores (same rule as the server).
            if c is Combination.RRF and n is not Normalization.RANK:
                raise ValueError(f"RRF requires RANK normalization; use a mean combination with {n.value}")
            if n is Normalization.RANK and c is not Combination.RRF:
                raise ValueError(f"RANK normalization requires RRF; use a score normalization with {c.value}")
            flags["normalization_technique"] = n.value
            flags["normalization_combination_technique"] = c.value
        if rank_window is not None:
            if _int("rank_window", rank_window) < 10:
                raise ValueError("rank_window must be >= 10")
            flags["neural_rank_window_size"] = rank_window
        if rank_constant is not None:
            if _int("rank_constant", rank_constant) < 1:
                raise ValueError("rank_constant must be >= 1")
            flags["neural_rank_constant"] = rank_constant
        if rerank is not None:
            flags["rerank_enabled"] = _bool("rerank", rerank)
        if rerank_first_stage_size is not None:
            if _int("rerank_first_stage_size", rerank_first_stage_size) <= 0:
                raise ValueError("rerank_first_stage_size must be > 0")
            flags["rerank_first_stage_size"] = rerank_first_stage_size
        if rerank_min_score is not None:
            flags["rerank_min_score"] = _finite("rerank_min_score", rerank_min_score)
        if flags:
            body["ssapi_flags"] = flags
        self._body = body

    def to_dict(self) -> dict[str, Any]:
        """A copy of the JSON body sent to the search API."""
        return copy.deepcopy(self._body)

    def to_json(self) -> str:
        """The exact JSON body sent to the search API."""
        return _dumps(self._body)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, SearchQuery) and other._body == self._body

    def __hash__(self) -> int:
        return hash(self.to_json())

    def __repr__(self) -> str:
        return f"SearchQuery{self.to_json()}"


def _bool(name: str, value: object) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be True or False")
    return value


def _names(name: str, values: Sequence[str]) -> list[str]:
    if isinstance(values, str):
        values = [values]
    names = [_text("field", v) for v in values]
    if not names:
        raise ValueError(f"{name} requires at least one field")
    return names


def _dumps(body: Any) -> str:
    return json.dumps(body, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


class SearchResult:
    """Curated view of a search response. Unknown response fields are ignored."""

    __slots__ = ("_body",)

    def __init__(self, body: dict[str, Any]) -> None:
        self._body = body

    def _text(self, key: str) -> str | None:
        value = self._body.get(key)
        return None if value is None else (value if isinstance(value, str) else json.dumps(value))

    @property
    def code(self) -> int:
        """1 on success, 0 on failure."""
        value = self._body.get("code")
        return value if isinstance(value, int) and not isinstance(value, bool) else 0

    @property
    def message(self) -> str | None:
        return self._text("message")

    @property
    def error_message(self) -> str | None:
        return self._text("errorMessage")

    @property
    def warning(self) -> str | None:
        return self._text("warning")

    @property
    def status_code(self) -> int:
        """Status the server reported inside the body, 0 when absent."""
        value = self._body.get("status_code")
        return value if isinstance(value, int) and not isinstance(value, bool) else 0

    @property
    def search_id(self) -> str | None:
        return self._text("search_id")

    @property
    def ranking_id(self) -> str | None:
        return self._text("ranking_id")

    @property
    def effective_neural_mode(self) -> str | None:
        """Neural mode the server actually ran, which can differ from the one requested."""
        return self._text("effective_neural_mode")

    @property
    def result(self) -> dict[str, Any]:
        """A copy of the ``result`` object (hits, aggregations); empty when absent."""
        value = self._body.get("result")
        return copy.deepcopy(value) if isinstance(value, dict) else {}

    def __repr__(self) -> str:
        return f"SearchResult(code={self.code}, search_id={self.search_id!r})"


class ProjectSearch:
    """Project search over the gateway's ``/core`` API. Thread-safe; does not own its transport.

    A 401 with the refreshable service key invalidates that token and retries once; a second 401,
    or a 401 with an exchanged user token, is terminal.
    """

    def __init__(self, http: httpx.Client, api_url: str, credential: Credential, read_timeout: float) -> None:
        self._http = http
        self._api_url = api_url
        self._credential = credential
        self._timeout = read_timeout

    def search(self, project_id: str, query: SearchQuery) -> SearchResult:
        """``POST /core/projects/{project_id}/search``."""
        path = f"/core/projects/{path_segment(project_id, 'project_id')}/search"
        return self._post(path, _require(query)._body)

    def search_by_usecase(self, usecase_id: str, query: SearchQuery) -> SearchResult:
        """``POST /core/usecases/{usecase_id}/search``."""
        path = f"/core/usecases/{path_segment(usecase_id, 'usecase_id')}/search"
        return self._post(path, _require(query)._body)

    def multi_search(self, project_id: str, queries: Sequence[SearchQuery]) -> SearchResult:
        """``POST /core/projects/{project_id}/mSearch`` with a JSON array of queries."""
        path = f"/core/projects/{path_segment(project_id, 'project_id')}/mSearch"
        if not queries:
            raise ValueError("queries must not be empty")
        return self._post(path, [_require(q)._body for q in queries])

    def _post(self, path: str, body: Any) -> SearchResult:
        payload = _dumps(body).encode("utf-8")
        header, generation = self._credential.authorization()
        response = self._send(path, payload, header)
        if response.status_code == 401 and self._credential.refreshable:
            self._credential.invalidate(generation)
            header, _ = self._credential.authorization()
            response = self._send(path, payload, header)
        data = _object(response.content)
        status = response.status_code
        if not 200 <= status <= 299:
            code = data.get("code")
            message = data.get("message") if isinstance(data.get("message"), str) else None
            error_message = data.get("errorMessage") if isinstance(data.get("errorMessage"), str) else None
            text = f"Search request failed with HTTP {status}"
            if message:
                text += f": {message}"
            if error_message:
                text += f" ({error_message})"
            raise error_type(status)(
                text, status_code=status,
                code=str(code) if isinstance(code, int) and not isinstance(code, bool) else None,
                server_message=message, error_message=error_message)
        return SearchResult(data)

    def _send(self, path: str, payload: bytes, authorization: str) -> httpx.Response:
        try:
            return self._http.post(
                self._api_url + path, content=payload,
                headers={"Content-Type": "application/json", "Accept": "application/json",
                         "Authorization": authorization},
                timeout=httpx.Timeout(self._timeout, connect=self._http.timeout.connect),
                follow_redirects=False)
        except httpx.TimeoutException:
            raise RequestTimeoutError(
                f"Search request timed out after {self._timeout:g} s", code="SEARCH_REQUEST_TIMEOUT") from None
        except httpx.HTTPError:
            # No cause: transport errors can carry URLs and headers.
            raise TransportError("Search request failed", code="SEARCH_CLIENT_ERROR") from None

    def __repr__(self) -> str:
        return f"ProjectSearch({self._api_url})"


def _require(query: object) -> SearchQuery:
    if not isinstance(query, SearchQuery):
        raise ValueError("query is required")
    return query


def _object(content: bytes) -> dict[str, Any]:
    if not content or not content.strip():
        return {}
    try:
        data = json.loads(content)
    except ValueError:
        return {}
    return data if isinstance(data, dict) else {}


__all__ = [
    "SearchQuery", "Filter", "AnyOf", "Sort", "SortOrder", "TermBoost", "TermsAgg", "CardinalityAgg",
    "NeuralMode", "Normalization", "Combination", "SearchResult", "ProjectSearch",
]
