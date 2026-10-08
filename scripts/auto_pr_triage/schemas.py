"""Typed records for every Auto PR Triage stage boundary.

Sections follow the pipeline:

  assess_intake -> intake.json -> build_ownership_input -> LLM input
  -> LLM answer -> validate_ownership -> ownership.json -> plan_actions
  -> action plan (job output) -> apply_actions (live only)
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from types import UnionType
from typing import (
    Any,
    ClassVar,
    get_args,
    get_origin,
    get_type_hints,
    Literal,
    TYPE_CHECKING,
)

from identifiers import CODEPATH_OWNER_RE, TEAM_OWNER_ID_RE, USER_HANDLE_RE


if TYPE_CHECKING:
    from typing import Self


# =============================================================================
# Record machinery
# Every boundary record derives from _Record: exact field type checks, JSON
# round trips, and JSON Schema generation from annotations and field bounds.
# =============================================================================


def _matches_type(*, value: Any, hint: Any) -> bool:
    """Return whether a value exactly matches a supported field annotation."""

    origin, args = get_origin(hint), get_args(hint)
    if hint is Any:
        return True
    if origin is UnionType:
        return any(_matches_type(value=value, hint=arg) for arg in args)
    if origin is Literal:
        return any(type(value) is type(arg) and value == arg for arg in args)
    if origin is tuple:
        return type(value) is tuple and all(
            _matches_type(value=v, hint=args[0]) for v in value
        )
    if origin is dict:
        return type(value) is dict and all(
            _matches_type(value=key, hint=args[0])
            and _matches_type(value=item, hint=args[1])
            for key, item in value.items()
        )
    if is_dataclass(hint):
        return isinstance(value, hint)
    return type(value) is hint


def _from_json(*, value: Any, hint: Any) -> Any:
    """Restore nested records and tuples; leave other values for type checks."""

    if isinstance(hint, type) and issubclass(hint, _Record):
        return hint.from_dict(value)
    if get_origin(hint) is tuple and isinstance(value, list | tuple):
        return tuple(_from_json(value=item, hint=get_args(hint)[0]) for item in value)
    if get_origin(hint) is dict and isinstance(value, dict):
        return {
            key: _from_json(value=item, hint=get_args(hint)[1])
            for key, item in value.items()
        }
    if get_origin(hint) is UnionType:
        options = [arg for arg in get_args(hint) if arg is not type(None)]
        if value is None:
            return None
        for arg in options:
            if isinstance(arg, type) and issubclass(arg, _Action):
                if isinstance(value, dict) and value.get("kind") == arg.default_kind():
                    return arg.from_dict(value)
        # An optional value, such as X | None, decodes as its one non-None type.
        if len(options) == 1:
            return _from_json(value=value, hint=options[0])
    return value


class _Record:
    """Share exact field type checks and JSON round-trips across schema records."""

    error: ClassVar[type[Exception]] = RuntimeError
    max_json_bytes: ClassVar[int | None] = None

    def __post_init__(self) -> None:
        # Every field matches its annotation, recursively (tuples, dicts, nested
        # records, optionals). Subclasses add their semantic checks after this.
        hints = get_type_hints(type(self))
        for item in fields(self):
            value = getattr(self, item.name)
            if not _matches_type(value=value, hint=hints[item.name]):
                name = f"{type(self).__name__}.{item.name}"
                detail = f"({type(value).__name__}) does not match {item.type}"
                raise self.error(f"{name} {detail}")

    @classmethod
    def _fields_from_json(cls, value: Any) -> dict[str, Any]:
        if not isinstance(value, dict):
            raise cls.error(f"{cls.__name__} fields are invalid: not an object")
        expected = {item.name for item in fields(cls)}
        if set(value) != expected:
            missing, unexpected = expected - set(value), set(value) - expected
            detail = f"missing {sorted(missing)}, unexpected {sorted(unexpected)}"
            raise cls.error(f"{cls.__name__} fields are invalid: {detail}")
        hints = get_type_hints(cls)
        return {
            name: _from_json(value=item, hint=hints[name])
            for name, item in value.items()
        }

    @classmethod
    def from_dict(cls, value: Any) -> Self:
        """Restore one record from its exact JSON-compatible form."""

        return cls(**cls._fields_from_json(value))

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-compatible form of this record."""

        return json.loads(json.dumps(asdict(self)))

    @classmethod
    def from_json(cls, raw: str) -> Self:
        """Parse one bounded single-line record."""

        if cls.max_json_bytes is not None and len(raw.encode()) > cls.max_json_bytes:
            raise cls.error(f"{cls.__name__} exceeds the size limit")
        try:
            value = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise cls.error(f"{cls.__name__} is not valid JSON") from exc
        return cls.from_dict(value)

    def to_json(self) -> str:
        """Return the compact single-line form, bounded for job outputs."""

        value = json.dumps(self.to_dict(), separators=(",", ":"))
        limit = self.max_json_bytes
        if limit is not None and len(value.encode()) > limit:
            raise self.error(f"{type(self).__name__} exceeds the size limit")
        return value


def bounds(**keywords: Any) -> dict[str, Any]:
    """Return field metadata holding JSON Schema keywords for json_schema."""

    return {"schema": keywords}


def json_schema(*, hint: Any, bounds: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return the JSON Schema for an annotation plus its field bounds.

    Records become closed objects with every field required, and tuples become
    arrays whose "items" bounds apply to each element.
    """

    bounds = dict(bounds or {})
    if isinstance(hint, type) and issubclass(hint, _Record):
        hints = get_type_hints(hint)
        return {
            "type": "object",
            "properties": {
                item.name: json_schema(
                    hint=hints[item.name], bounds=item.metadata.get("schema")
                )
                for item in fields(hint)
            },
            "required": [item.name for item in fields(hint)],
            "additionalProperties": False,
        }
    if get_origin(hint) is tuple:
        items = json_schema(hint=get_args(hint)[0], bounds=bounds.pop("items", None))
        return {"type": "array", **bounds, "items": items}
    if get_origin(hint) is UnionType:
        options = [arg for arg in get_args(hint) if arg is not type(None)]
        if len(options) != 1 or len(get_args(hint)) != 2:
            raise TypeError(f"only optional values are supported: {hint}")
        return {
            "anyOf": [json_schema(hint=options[0], bounds=bounds), {"type": "null"}]
        }
    kinds = {str: "string", int: "integer", bool: "boolean"}
    return {"type": kinds[hint], **bounds}


# =============================================================================
# Stage 1 -> later stages: intake facts (intake.json)
# assess_intake.py records the gate facts and the one PR snapshot; planning
# makes the final admission decision from them. Read by build_ownership_input,
# validate_ownership, and plan_actions; the identity and facts are also embedded
# in the action plan.
# =============================================================================

MAX_HANDOFF_REVIEWERS = 3
MAX_PART_OF_ISSUES = 5


def are_canonical_logins(logins: list[str] | tuple[str, ...]) -> bool:
    """Return whether logins are unique, sorted, bounded GitHub logins."""

    return (
        len(logins) <= MAX_HANDOFF_REVIEWERS
        and all(
            isinstance(login, str) and USER_HANDLE_RE.fullmatch(f"@{login}")
            for login in logins
        )
        and len({login.casefold() for login in logins}) == len(logins)
        and list(logins) == sorted(logins, key=str.casefold)
    )


def passes_intake(
    *,
    is_open_non_draft_pr_against_main: bool,
    is_already_handled: bool,
    author_has_triage_permission: bool,
    has_actionable_linked_issue: bool,
    has_maintainer_activity: bool,
    has_supporter: bool = False,
    has_related_actionable_issue: bool = False,
) -> bool:
    """Return whether an intake signal admits the PR without a bypass."""

    return (
        is_open_non_draft_pr_against_main
        and not is_already_handled
        and (
            author_has_triage_permission
            or has_actionable_linked_issue
            or has_maintainer_activity
            or has_supporter
            or has_related_actionable_issue
        )
    )


@dataclass(frozen=True)
class PullRequestIdentity(_Record):
    """Identify the analyzed PR revision and the workflow revision of this run."""

    repository: str
    number: int
    head_sha: str
    workflow_sha: str

    def __post_init__(self) -> None:
        # Sanity: the PR number is positive.
        super().__post_init__()
        if self.number <= 0:
            raise RuntimeError("pull request number is not positive")


@dataclass(frozen=True)
class IntakeFacts(_Record):
    """Hold intake's gate facts and handoff reviewers; the LLM never sees them."""

    is_open_non_draft_pr_against_main: bool
    is_already_handled: bool
    author_has_triage_permission: bool
    has_actionable_linked_issue: bool
    has_maintainer_activity: bool
    has_supporter: bool
    has_related_actionable_issue: bool
    supporters: tuple[str, ...]
    actionable_labelers: tuple[str, ...]
    maintainer_requested_reviewers: tuple[str, ...]

    def __post_init__(self) -> None:
        # Sanity: the gate facts agree with each other.
        # - An inactive or handled PR carries no intake signals.
        # - has_supporter is true exactly when supporters is non-empty.
        # - Actionable labelers imply an actionable issue.
        # - Handoff reviewers exist only on a PR that passes intake.
        super().__post_init__()
        for name, logins in (
            ("supporters", self.supporters),
            ("actionable_labelers", self.actionable_labelers),
            ("maintainer_requested_reviewers", self.maintainer_requested_reviewers),
        ):
            if not are_canonical_logins(logins):
                raise RuntimeError(f"intake facts {name} are not canonical logins")
        has_actionable_issue = (
            self.has_actionable_linked_issue or self.has_related_actionable_issue
        )
        signals = (
            self.author_has_triage_permission,
            self.has_actionable_linked_issue,
            self.has_maintainer_activity,
            self.has_supporter,
        )
        active = self.is_open_non_draft_pr_against_main and not self.is_already_handled
        if not active and (any(signals) or self.has_related_actionable_issue):
            raise RuntimeError("intake facts: signals on an inactive or handled PR")
        if self.has_supporter != bool(self.supporters):
            raise RuntimeError("intake facts: has_supporter mismatches supporters")
        if self.actionable_labelers and not has_actionable_issue:
            raise RuntimeError("intake facts: actionable labelers without an issue")
        if (
            self.actionable_labelers or self.maintainer_requested_reviewers
        ) and not self.passes_intake:
            raise RuntimeError("intake facts: handoff reviewers without intake")

    @property
    def is_active(self) -> bool:
        """Return whether the PR is an open, unhandled PR against main."""

        return self.is_open_non_draft_pr_against_main and not self.is_already_handled

    @property
    def passes_intake(self) -> bool:
        """Return whether an intake signal admits the PR without a bypass."""

        return passes_intake(
            is_open_non_draft_pr_against_main=self.is_open_non_draft_pr_against_main,
            is_already_handled=self.is_already_handled,
            author_has_triage_permission=self.author_has_triage_permission,
            has_actionable_linked_issue=self.has_actionable_linked_issue,
            has_maintainer_activity=self.has_maintainer_activity,
            has_supporter=self.has_supporter,
            has_related_actionable_issue=self.has_related_actionable_issue,
        )


@dataclass(frozen=True)
class IntakeResult(_Record):
    """Hold the intake decision and the one PR snapshot later stages reuse.

    Title and body are attacker-controlled; intake carries them so the ownership
    stage never reads the PR a second time.
    """

    identity: PullRequestIdentity
    facts: IntakeFacts
    author_login: str
    title: str
    body: str


# =============================================================================
# Trusted configuration: checked-in ownership artifacts
# Codepath resolution and owner metadata loaded from the workflow revision and
# embedded in the LLM's trusted context.
# =============================================================================


@dataclass(frozen=True)
class PathGroup(_Record):
    """Hold changed files whose codepath-owner rules resolved to the same owners.

    If files 0 and 2 both resolve to @soulitzer and the autograd team, the group
    is PathGroup(owners=("@soulitzer", "autograd"), file_indices=(0, 2)). Indices
    point into untrusted_context.files.
    """

    owners: tuple[str, ...]
    file_indices: tuple[int, ...]

    def __post_init__(self) -> None:
        # Sanity: a group has at least one owner and one file, no repeats.
        super().__post_init__()
        if not self.owners or len(set(self.owners)) != len(self.owners):
            raise RuntimeError("path group owners are empty or repeated")
        indices = self.file_indices
        if not indices or len(set(indices)) != len(indices) or min(indices) < 0:
            raise RuntimeError("path group file indices are invalid")


@dataclass(frozen=True)
class CodepathOwners(_Record):
    """Hold the codepath-owner resolution for every changed file, by index.

    files_without_owners lists files with no codepath owner, whether no rule
    matched or the last matching rule listed no owners; the two are the same here.
    """

    owners: tuple[str, ...]
    matched_path_groups: tuple[PathGroup, ...]
    files_without_owners: tuple[int, ...]

    def __post_init__(self) -> None:
        # Sanity: each changed file has exactly one resolution, either in one
        # path group or in files_without_owners, and owners is exactly the union
        # of the group owners, canonically sorted.
        super().__post_init__()
        if not all(CODEPATH_OWNER_RE.fullmatch(owner) for owner in self.owners):
            raise RuntimeError("codepath owners contain an invalid owner")
        owner_keys = [owner.casefold() for owner in self.owners]
        if len(set(owner_keys)) != len(owner_keys) or owner_keys != sorted(owner_keys):
            raise RuntimeError("codepath owners are not unique and sorted")
        grouped = [i for group in self.matched_path_groups for i in group.file_indices]
        if len(set(grouped)) != len(grouped):
            raise RuntimeError("changed file appears in multiple codepath-owner groups")
        unowned = self.files_without_owners
        if len(set(unowned)) != len(unowned) or any(i < 0 for i in unowned):
            raise RuntimeError("files without owners are repeated or invalid")
        if set(grouped) & set(unowned):
            raise RuntimeError("changed file has conflicting codepath-owner resolution")
        grouped_owners = {o for group in self.matched_path_groups for o in group.owners}
        if grouped_owners != set(self.owners):
            raise RuntimeError("codepath owners do not match path groups")

    @property
    def file_indices(self) -> set[int]:
        """Return every changed-file index this resolution accounts for."""

        grouped = {i for group in self.matched_path_groups for i in group.file_indices}
        return grouped | set(self.files_without_owners)


@dataclass(frozen=True)
class OwnerMetadata(_Record):
    """Describe one team owner and the PRs it takes without intake gates."""

    description: str
    bypass_intake_criteria: str | None

    def __post_init__(self) -> None:
        # Sanity: the texts shown to the LLM are non-empty and bounded.
        super().__post_init__()
        for text in (self.description, self.bypass_intake_criteria):
            if text is not None and (not text.strip() or len(text) > 2_000):
                raise RuntimeError("extra ownership metadata entry is invalid")


# =============================================================================
# Stage 2a -> LLM: LLM input (llm_input.json and the prompt)
# build_ownership_input.py partitions trusted and untrusted input; the prompt is
# LLMInput.to_dict(). The trusted context refers to changed files only by
# index, so it contains no attacker-controlled text. LLMInput also crosses to
# validate_ownership as llm_input.json, which checks evidence against the
# real patches.
# =============================================================================


@dataclass(frozen=True)
class ChangedFile(_Record):
    """Hold one changed file exactly as collected from the pull request."""

    path: str
    status: str
    additions: int
    deletions: int
    patch: str | None
    patch_truncated_or_unavailable: bool

    def __post_init__(self) -> None:
        # Sanity: the path is non-empty.
        super().__post_init__()
        if not self.path:
            raise RuntimeError("triage input changed file has an empty path")


@dataclass(frozen=True)
class UntrustedContext(_Record):
    """Hold attacker-controlled pull-request content shown to the LLM."""

    title: str
    body: str
    files: tuple[ChangedFile, ...]

    def __post_init__(self) -> None:
        # Sanity: no file path repeats, so indices are unambiguous.
        super().__post_init__()
        if len({file.path for file in self.files}) != len(self.files):
            raise RuntimeError("triage input untrusted context repeats a file path")


@dataclass(frozen=True)
class TrustedContext(_Record):
    """Hold workflow-owned policy and ownership, free of attacker-controlled text."""

    worker_policy: str
    codepath_owners: CodepathOwners
    extra_ownership_metadata: dict[str, OwnerMetadata]
    diff_truncated_or_unavailable: bool

    def __post_init__(self) -> None:
        # Sanity: the LLM gets a description for every team owner ID it is
        # shown, and every metadata key is a valid team owner ID.
        super().__post_init__()
        if not self.worker_policy.strip():
            raise RuntimeError("triage input has an empty worker policy")
        metadata = self.extra_ownership_metadata
        if not metadata or not all(TEAM_OWNER_ID_RE.fullmatch(o) for o in metadata):
            raise RuntimeError("extra ownership metadata owners are invalid")

    @property
    def teams_with_intake_bypass(self) -> frozenset[str]:
        """Return the owners that configured bypass-intake criteria."""

        metadata = self.extra_ownership_metadata
        return frozenset(o for o, e in metadata.items() if e.bypass_intake_criteria)


@dataclass(frozen=True)
class LLMInput(_Record):
    """Hold the ownership stage's LLM input with an explicit trust partition."""

    trusted_context: TrustedContext
    untrusted_context: UntrustedContext

    def __post_init__(self) -> None:
        # Sanity: the codepath resolution covers exactly the changed files, so
        # the LLM and validate_ownership see the same file indices.
        super().__post_init__()
        expected = set(range(len(self.untrusted_context.files)))
        if self.trusted_context.codepath_owners.file_indices != expected:
            raise RuntimeError("codepath artifact does not cover all changed paths")

    @classmethod
    def create(
        cls,
        *,
        worker_policy: str,
        ownership: dict[str, Any],
        diff_truncated_or_unavailable: bool,
        untrusted_context: UntrustedContext,
    ) -> LLMInput:
        """Build an input from the resolver's paths and the loaded owner metadata.

        Paths are replaced by indices into untrusted_context.files here, so no
        trusted record ever holds a PR-controlled path.
        """

        index = {file.path: i for i, file in enumerate(untrusted_context.files)}

        def file_index(path: str) -> int:
            if path not in index:
                raise RuntimeError("codepath artifact names a file outside the PR")
            return index[path]

        codepath = ownership["codepath_owners"]
        codepath_owners = CodepathOwners(
            tuple(codepath["owners"]),
            tuple(
                PathGroup(tuple(g["owners"]), tuple(file_index(p) for p in g["paths"]))
                for g in codepath["matched_path_groups"]
            ),
            tuple(file_index(path) for path in codepath["paths_without_owners"]),
        )
        metadata = {
            owner: OwnerMetadata.from_dict(entry)
            for owner, entry in ownership["extra_ownership_metadata"]["owners"].items()
        }
        trusted_context = TrustedContext(
            worker_policy, codepath_owners, metadata, diff_truncated_or_unavailable
        )
        return cls(trusted_context, untrusted_context)


# =============================================================================
# LLM -> stage 2c: the LLM's answer (RESULT_SCHEMA)
# The action enforces RESULT_SCHEMA, generated from these records;
# validate_ownership.py parses the answer into LLMResult.
# =============================================================================

MAX_CODEPATH_OWNERS = 100
MAX_ADDITIONAL_OWNERS = 8
MAX_UNCOVERED_CONCERNS = 8
MAX_CODEPATH_OWNER_CONCERNS = 16
MAX_OWNER_EVIDENCE_ITEMS = 3
MAX_DIFF_EXCERPT_CHARS = 1_200
TEXT_BOUNDS = {"minLength": 20, "maxLength": 800}
PATH_BOUNDS = {"minLength": 1, "maxLength": 500}
FILE_LIST_BOUNDS = {
    "minItems": 1,
    "maxItems": 16,
    "uniqueItems": True,
    "items": PATH_BOUNDS,
}


@dataclass(frozen=True)
class Evidence(_Record):
    """Quote one changed hunk that supports an owner, bypass, or concern."""

    file: str = field(metadata=bounds(**PATH_BOUNDS))
    diff_excerpt: str = field(
        metadata=bounds(minLength=1, maxLength=MAX_DIFF_EXCERPT_CHARS)
    )
    relevance: str = field(metadata=bounds(**TEXT_BOUNDS))


EVIDENCE_BOUNDS = {
    "minItems": 1,
    "maxItems": MAX_OWNER_EVIDENCE_ITEMS,
    "uniqueItems": True,
}


@dataclass(frozen=True)
class BypassIntakeMatch(_Record):
    """Justify that a change matches a team's bypass-intake criteria.

    criteria_quote is copied verbatim from that team's bypass_intake_criteria;
    the rationale and diff evidence show how the change meets it.
    """

    criteria_quote: str = field(metadata=bounds(minLength=1, maxLength=2_000))
    rationale: tuple[str, ...] = field(
        metadata=bounds(minItems=1, maxItems=3, items=TEXT_BOUNDS)
    )
    evidence: tuple[Evidence, ...] = field(metadata=bounds(**EVIDENCE_BOUNDS))


@dataclass(frozen=True)
class Concern(_Record):
    """Describe one distinct, material change and the diff evidence for it."""

    description: str = field(metadata=bounds(**TEXT_BOUNDS))
    files: tuple[str, ...] = field(metadata=bounds(**FILE_LIST_BOUNDS))
    evidence: tuple[Evidence, ...] = field(metadata=bounds(**EVIDENCE_BOUNDS))


# Every concern the LLM finds lands in exactly one of the next three records,
# according to who handles it: the existing codepath owners, an additional owner
# from the metadata, or no configured owner.


@dataclass(frozen=True)
class CodepathOwnerConcern(_Record):
    """Record a concern the existing codepath owners already handle; no action."""

    concern: Concern
    codepath_owners: tuple[str, ...] = field(
        metadata=bounds(
            minItems=1,
            uniqueItems=True,
            items={"pattern": f"^{CODEPATH_OWNER_RE.pattern}$"},
        )
    )
    reason: str = field(metadata=bounds(**TEXT_BOUNDS))


@dataclass(frozen=True)
class AdditionalOwnerConcern(_Record):
    """Route a concern to one additional owner from the metadata."""

    concern: Concern
    owner_id: str = field(metadata=bounds(pattern=f"^{TEAM_OWNER_ID_RE.pattern}$"))
    rationale: tuple[str, ...] = field(
        metadata=bounds(minItems=3, maxItems=4, items=TEXT_BOUNDS)
    )
    confidence: str = field(metadata=bounds(enum=["high", "medium", "low"]))
    bypass_intake_match: BypassIntakeMatch | None = field(metadata=bounds())


@dataclass(frozen=True)
class UncoveredConcern(_Record):
    """Record a concern that no configured owner fits."""

    concern: Concern
    reason: str = field(metadata=bounds(**TEXT_BOUNDS))


@dataclass(frozen=True)
class LLMResult(_Record):
    """Hold the LLM's structured answer; the action enforces RESULT_SCHEMA.

    OwnershipResult describes how validation turns this into the stage result.
    """

    codepath_owner_concerns: tuple[CodepathOwnerConcern, ...] = field(
        metadata=bounds(maxItems=MAX_CODEPATH_OWNER_CONCERNS, uniqueItems=True)
    )
    additional_owner_concerns: tuple[AdditionalOwnerConcern, ...] = field(
        metadata=bounds(maxItems=MAX_ADDITIONAL_OWNERS, uniqueItems=True)
    )
    uncovered_concerns: tuple[UncoveredConcern, ...] = field(
        metadata=bounds(maxItems=MAX_UNCOVERED_CONCERNS, uniqueItems=True)
    )
    security_flags: tuple[str, ...] = field(
        metadata=bounds(maxItems=20, items={"maxLength": 500})
    )


RESULT_SCHEMA = json_schema(hint=LLMResult)


# =============================================================================
# Stage 2c -> planning: ownership result (ownership.json)
# validate_ownership.py records the trusted codepath owners and the LLM's
# validated concerns; planning reads the part it needs.
# =============================================================================


@dataclass(frozen=True)
class OwnershipResult(_Record):
    """Record everything the ownership stage concluded about the PR's owners.

    This is the validated LLMResult plus the trusted codepath owners. It differs
    from LLMResult in four ways:
    - llm_run_status is added. The result also exists when the LLM was skipped
      (an inactive PR: no owners) or failed (any validation error: codepath
      owners only). Only a succeeded run carries concerns.
    - codepath_owners is added from the trusted path rules, not the LLM, and
      maps each owner to the changed files it matched.
    - The LLM's additional_owner_concerns are split into accepted and
      discarded_additional_owner_concerns. A concern is discarded if its
      confidence is low or it cites a file whose patch was truncated or
      unavailable.
    - security_flags is dropped; nothing downstream reads it.
    The concern records themselves are the LLM's, unchanged.
    """

    error = ValueError

    llm_run_status: Literal["skipped", "failed", "succeeded"]
    codepath_owners: dict[str, tuple[str, ...]]
    codepath_owner_concerns: tuple[CodepathOwnerConcern, ...] = ()
    additional_owner_concerns: tuple[AdditionalOwnerConcern, ...] = ()
    discarded_additional_owner_concerns: tuple[AdditionalOwnerConcern, ...] = ()
    uncovered_concerns: tuple[UncoveredConcern, ...] = ()

    def __post_init__(self) -> None:
        # Sanity: this result is built by validate_ownership from already
        # validated LLM output, and planning trusts it.
        # - Owners are valid, canonically ordered, unique, and within limits.
        # - llm_run_status bounds the other fields: "skipped" has no owners,
        #   and only "succeeded" carries concerns.
        # - No accepted concern has low confidence.
        super().__post_init__()
        codepath = tuple(self.codepath_owners)
        if len(codepath) > MAX_CODEPATH_OWNERS:
            raise ValueError("ownership result has too many codepath owners")
        if not all(CODEPATH_OWNER_RE.fullmatch(owner) for owner in codepath):
            raise ValueError("ownership result has an invalid codepath owner")
        keys = [owner.casefold() for owner in codepath]
        if len(set(keys)) != len(keys) or keys != sorted(keys):
            raise ValueError("ownership result codepath owners are not canonical")
        if not all(self.codepath_owners.values()):
            raise ValueError("ownership result codepath owner has no files")
        additional = self.additional_owners
        if len(additional) > MAX_ADDITIONAL_OWNERS:
            raise ValueError("ownership result has too many additional owners")
        if not all(TEAM_OWNER_ID_RE.fullmatch(owner) for owner in additional):
            raise ValueError("ownership result has an invalid additional owner")
        if len(set(additional)) != len(additional) or list(additional) != sorted(
            additional
        ):
            raise ValueError("ownership result additional owners are not canonical")
        if any(c.confidence == "low" for c in self.additional_owner_concerns):
            raise ValueError("ownership result accepted a low-confidence owner")

        if self.llm_run_status == "skipped" and codepath:
            raise ValueError("a skipped LLM run cannot carry codepath owners")
        concerns = (
            self.codepath_owner_concerns,
            self.additional_owner_concerns,
            self.discarded_additional_owner_concerns,
            self.uncovered_concerns,
        )
        if self.llm_run_status != "succeeded" and any(concerns):
            raise ValueError("only a succeeded LLM run carries concerns")

    @property
    def additional_owners(self) -> tuple[str, ...]:
        return tuple(concern.owner_id for concern in self.additional_owner_concerns)

    @property
    def bypass_intake_matches(self) -> tuple[str, ...]:
        """Return the accepted owners whose concern matched bypass intake."""

        return tuple(
            concern.owner_id
            for concern in self.additional_owner_concerns
            if concern.bypass_intake_match is not None
        )

    @property
    def has_uncovered_concerns(self) -> bool:
        return bool(self.uncovered_concerns)

    @property
    def has_discarded_bypass_intake_match(self) -> bool:
        """Return whether validation dropped a concern that claimed a bypass.

        The planner then keeps a PR that fails intake open instead of closing
        it, since a real bypass may have been lost.
        """

        return any(
            concern.bypass_intake_match is not None
            for concern in self.discarded_additional_owner_concerns
        )

    @classmethod
    def create(
        cls,
        *,
        llm_run_status: Literal["skipped", "failed", "succeeded"],
        codepath_owners: dict[str, Any] | None = None,
        codepath_owner_concerns: list[CodepathOwnerConcern] | tuple = (),
        additional_owner_concerns: list[AdditionalOwnerConcern] | tuple = (),
        discarded_additional_owner_concerns: list[AdditionalOwnerConcern] | tuple = (),
        uncovered_concerns: list[UncoveredConcern] | tuple = (),
    ) -> OwnershipResult:
        """Canonicalize one result built by validation."""

        codepath = codepath_owners or {}
        return cls(
            llm_run_status,
            {
                owner: tuple(sorted(codepath[owner]))
                for owner in sorted(codepath, key=str.casefold)
            },
            tuple(codepath_owner_concerns),
            tuple(sorted(additional_owner_concerns, key=lambda c: c.owner_id)),
            tuple(discarded_additional_owner_concerns),
            tuple(uncovered_concerns),
        )


# =============================================================================
# Stage results and GitHub -> planning: the planner input (planner_input.json)
# plan_actions.py reads all live GitHub state once into a ReviewerSnapshot,
# bundles it with the stage results, and decides the plan from that alone, so
# every plan can be replayed from planner_input.json.
# =============================================================================


@dataclass(frozen=True)
class ReviewerSnapshot(_Record):
    """Hold every live GitHub read planning made for one PR.

    A None field was not needed or could not be read; errors says which reads
    failed and why.
    """

    existing_labels: tuple[str, ...]
    missing_labels: tuple[str, ...]
    requested_reviewers: tuple[str, ...] | None
    submitted_reviewers: tuple[str, ...] | None
    rosters: dict[str, tuple[str, ...]] | None
    errors: dict[str, str]


@dataclass(frozen=True)
class PlannerInput(_Record):
    """Hold everything the planner reads; plan.json is a function of this."""

    error = ValueError

    intake: IntakeResult
    ownership: OwnershipResult
    reviewers: ReviewerSnapshot
    run_attempt: int

    def __post_init__(self) -> None:
        # Sanity: the stage results describe one consistent PR.
        # - The LLM was skipped exactly when intake found the PR inactive.
        # - Every codepath owner team is in the repository's own org.
        super().__post_init__()
        check_stage_results(intake=self.intake, ownership=self.ownership)
        if self.run_attempt < 1:
            raise ValueError("planner input has an invalid run attempt")


def check_stage_results(*, intake: IntakeResult, ownership: OwnershipResult) -> None:
    """Reject an ownership result that does not fit its intake result."""

    if intake.facts.is_active == (ownership.llm_run_status == "skipped"):
        raise ValueError("ownership result does not match the intake decision")


# =============================================================================
# Planning -> live apply: the action plan (the analyze job output)
# plan_actions.py emits an ordered list of typed actions bounded by a context;
# apply_actions.py executes them without re-deriving anything.
# =============================================================================

MAX_ACTION_PLAN_BYTES = 16_000
MAX_REVIEW_REQUESTS = 15
TRIAGED_LABEL = "triaged"
BOT_TRIAGED_LABEL = "bot-triaged"
BOT_TRIAGE_ERROR_LABEL = "bot-triage-error"
MISSING_ACTIONABLE_ISSUE_LABEL = "missing actionable issue"
# The only labels a plan may add, by decision; missing_actionable_issue plans
# must instead be exactly MISSING_ACTIONABLE_ISSUE_ACTIONS.
DECISION_LABELS = {
    "triage": frozenset({TRIAGED_LABEL, BOT_TRIAGED_LABEL}),
    "incomplete": frozenset({BOT_TRIAGE_ERROR_LABEL}),
}
# Why a reviewer is requested, in the order requests appear in a plan.
REQUEST_REASONS = ("supporter", "actionable_labeler", "owner_roster")
TRIAGE_DECISIONS = frozenset(
    {
        "kept_open",
        "missing_actionable_issue",
        "triage",
        "incomplete",
        "routed_untriaged",
    }
)


class _Action(_Record):
    """Name one GitHub effect by its kind, which is fixed for each action record."""

    @classmethod
    def default_kind(cls) -> str:
        return cls.__dataclass_fields__["kind"].default

    def __post_init__(self) -> None:
        # Sanity: kind is fixed per action type, so decoding is unambiguous.
        super().__post_init__()
        if self.kind != self.default_kind():
            raise RuntimeError(f"{type(self).__name__} has the wrong kind")


@dataclass(frozen=True)
class RequestReviewers(_Action):
    """Request users who share one reason.

    - supporter: named as a supporter in the PR description and verified.
    - actionable_labeler: labeled a linked or related issue `actionable`.
    - owner_roster: picked from a team owner's roster.

    Codepath owners are never requested here: GitHub requests them through
    CODEOWNERS.
    """

    users: tuple[str, ...]
    reason: Literal["supporter", "actionable_labeler", "owner_roster"]
    kind: str = "request_reviewers"


@dataclass(frozen=True)
class AddLabels(_Action):
    """Add labels to the pull request."""

    labels: tuple[str, ...]
    kind: str = "add_labels"


Action = RequestReviewers | AddLabels


MISSING_ACTIONABLE_ISSUE_LABELS = (
    TRIAGED_LABEL,
    BOT_TRIAGED_LABEL,
    MISSING_ACTIONABLE_ISSUE_LABEL,
)
MISSING_ACTIONABLE_ISSUE_ACTIONS = (AddLabels(MISSING_ACTIONABLE_ISSUE_LABELS),)


@dataclass(frozen=True)
class PlanContext(_Record):
    """Carry the identity, facts, and owners that bound a plan's actions."""

    identity: PullRequestIdentity
    facts: IntakeFacts
    run_attempt: int
    codepath_owners: tuple[str, ...]
    additional_owners: tuple[str, ...]
    bypass_intake_matches: tuple[str, ...]

    def __post_init__(self) -> None:
        # Security: the apply job trusts the context to bound the actions.
        # - Every bypass match is an additional owner.
        # - run_attempt is valid; ActionPlan uses it to forbid marking a PR as
        #   missing an actionable issue on a rerun.
        super().__post_init__()
        if self.run_attempt < 1:
            raise RuntimeError("plan context has an invalid run attempt")
        if not all(CODEPATH_OWNER_RE.fullmatch(o) for o in self.codepath_owners):
            raise RuntimeError("plan context has an invalid codepath owner")
        if not all(
            TEAM_OWNER_ID_RE.fullmatch(owner) for owner in self.additional_owners
        ):
            raise RuntimeError("plan context has an invalid additional owner")
        if not set(self.bypass_intake_matches) <= set(self.additional_owners):
            raise RuntimeError("plan context bypass intake match is not an owner")

    @property
    def admitted(self) -> bool:
        """Return whether intake or a bypass intake match admits the PR."""

        return self.facts.passes_intake or bool(self.bypass_intake_matches)


@dataclass(frozen=True)
class ActionPlan(_Record):
    """Hold the ordered GitHub actions live mode applies after one analysis.

    The context only bounds the actions; apply executes the actions in order
    without re-deriving them.
    """

    max_json_bytes = MAX_ACTION_PLAN_BYTES

    context: PlanContext
    decision: str
    actions: tuple[Action, ...]

    def __post_init__(self) -> None:
        # Security: this is the analyze job's output, and the apply job, which
        # has pull-requests: write, executes it as is. These checks cap what a
        # faulty or manipulated analysis can do:
        # - An inactive or handled PR gets no actions.
        # - Only an unadmitted PR on the first attempt can be marked as missing
        #   an actionable issue, and only with exactly
        #   MISSING_ACTIONABLE_ISSUE_ACTIONS (triaged, bot-triaged, missing
        #   actionable issue).
        # - An unadmitted PR is never routed to reviewers.
        # - Only this bot's own labels can be added: those above, triaged and
        #   bot-triaged for triage, and bot-triage-error for incomplete. No other
        #   label (e.g. ciflow/*) can be added.
        # - Nothing can close a PR or post a comment: there is no action for it.
        # - Every requested user is justified by its request's reason: a verified
        #   supporter or an actionable labeler. owner_roster users are checked
        #   against rosters by apply. Nobody is requested twice, no team is ever
        #   requested, and each request is capped at MAX_REVIEW_REQUESTS.
        # The remaining checks (sorted logins, action order) keep the format canonical.
        super().__post_init__()
        context = self.context
        facts = context.facts
        if self.decision not in TRIAGE_DECISIONS:
            raise RuntimeError("action plan has an unknown decision")
        if not facts.is_active and (self.decision != "kept_open" or self.actions):
            raise RuntimeError("action plan acts on an inactive or handled PR")
        if self.decision == "missing_actionable_issue":
            if context.admitted:
                raise RuntimeError("action plan marks an admitted PR")
            if context.run_attempt != 1:
                raise RuntimeError("action plan marks a PR on a rerun")
            if self.actions != MISSING_ACTIONABLE_ISSUE_ACTIONS:
                raise RuntimeError("action plan labels are not the fixed set")
            return

        requests = [a for a in self.actions if isinstance(a, RequestReviewers)]
        label_actions = [a for a in self.actions if isinstance(a, AddLabels)]
        if self.actions != (*requests, *label_actions) or len(label_actions) > 1:
            raise RuntimeError("action plan must request reviewers, then add labels")
        reasons = [request.reason for request in requests]
        if reasons != sorted(set(reasons), key=REQUEST_REASONS.index):
            raise RuntimeError(
                "action plan reviewer requests are repeated or unordered"
            )
        if not context.admitted and (
            self.decision not in {"kept_open", "incomplete"} or requests
        ):
            raise RuntimeError("action plan routes a PR that was not admitted")

        allowed_labels = DECISION_LABELS.get(self.decision, frozenset())
        for action in label_actions:
            if not action.labels or not set(action.labels) <= allowed_labels:
                raise RuntimeError("action plan has a label outside its decision")

        # Each reason's users must come from that reason's own source. Roster
        # membership for owner_roster is checked by apply, which loads rosters.
        sources = {
            "supporter": {login.casefold() for login in facts.supporters},
            "actionable_labeler": {
                login.casefold() for login in facts.actionable_labelers
            },
        }
        requested: list[str] = []
        for request in requests:
            keys = [login.casefold() for login in request.users]
            requested += keys
            if (
                not request.users
                or not all(USER_HANDLE_RE.fullmatch(f"@{u}") for u in request.users)
                or keys != sorted(keys)
            ):
                raise RuntimeError("action plan reviewers are not canonical logins")
            source = sources.get(request.reason)
            if source is not None and not set(keys) <= source:
                raise RuntimeError(f"action plan has an unverified {request.reason}")
            if len(request.users) > MAX_REVIEW_REQUESTS:
                raise RuntimeError("action plan exceeds the review request limit")
        if len(set(requested)) != len(requested):
            raise RuntimeError("action plan requests one reviewer twice")
