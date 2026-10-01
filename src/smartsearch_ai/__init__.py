"""SmartSearch AI SDK for Python: project search, Workplace and user provisioning.

Start with :class:`SmartSearchAi`.
"""

from ._client import SmartSearchAi
from ._version import __version__
from .errors import (
    EXTERNAL_IDENTITY_NOT_SUPPORTED,
    EXTERNAL_IDENTITY_REQUIRED,
    INTEGRATION_NOT_FOUND,
    AuthenticationError,
    ConflictError,
    NotFoundError,
    PermissionDeniedError,
    RequestTimeoutError,
    SmartSearchError,
    TransportError,
)
from .provisioning import (
    AssignmentItem,
    DisableItem,
    EnsureItem,
    ExternalIdentity,
    Grants,
    JobKind,
    JobState,
    Membership,
    Profile,
    RetryJob,
    Scope,
    ScopeKind,
    SnapshotApply,
    SnapshotRequest,
    SnapshotRow,
    SourceGrant,
    SubmitJob,
    Users,
    snapshot_digest,
)
from .search import (
    AnyOf,
    CardinalityAgg,
    Combination,
    Filter,
    NeuralMode,
    Normalization,
    ProjectSearch,
    SearchQuery,
    SearchResult,
    Sort,
    SortOrder,
    TermBoost,
    TermsAgg,
)
from .workplace import (
    CachePolicy,
    MemberRole,
    MemoryMode,
    Mode,
    Output,
    QueryOptions,
    UserWorkplace,
    Workplace,
    WorkspaceMemberRequest,
    WorkspacePrincipalGrantRequest,
    WorkspaceQueryRequest,
    WorkspaceRequestContext,
    WorkspaceResult,
    WorkspaceStream,
    WorkspaceStreamEvent,
)

__all__ = [
    "__version__", "SmartSearchAi",
    # errors
    "SmartSearchError", "AuthenticationError", "PermissionDeniedError", "NotFoundError", "ConflictError",
    "RequestTimeoutError", "TransportError", "INTEGRATION_NOT_FOUND", "EXTERNAL_IDENTITY_REQUIRED",
    "EXTERNAL_IDENTITY_NOT_SUPPORTED",
    # search
    "ProjectSearch", "SearchQuery", "SearchResult", "Filter", "AnyOf", "Sort", "SortOrder", "TermBoost",
    "TermsAgg", "CardinalityAgg", "NeuralMode", "Normalization", "Combination",
    # workplace
    "Workplace", "UserWorkplace", "WorkspaceQueryRequest", "QueryOptions", "Mode", "MemoryMode", "Output",
    "CachePolicy", "WorkspaceRequestContext", "WorkspaceResult", "WorkspaceStream", "WorkspaceStreamEvent",
    "WorkspaceMemberRequest", "MemberRole", "WorkspacePrincipalGrantRequest",
    # provisioning
    "Users", "ScopeKind", "JobKind", "JobState", "Scope", "Grants", "SourceGrant", "Membership", "Profile",
    "ExternalIdentity", "EnsureItem", "AssignmentItem", "DisableItem", "SubmitJob", "RetryJob", "SnapshotRow",
    "SnapshotRequest", "SnapshotApply", "snapshot_digest",
]
