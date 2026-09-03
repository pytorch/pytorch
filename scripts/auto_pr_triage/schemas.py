"""Data schemas and shared identity constraints for Auto PR Triage."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from codepath_owners import (
    CODEPATH_OWNERS_PATH,
    OWNER_RE as CODEPATH_OWNER_RE,
    REPOSITORY_RE,
)
from ownership import (
    EXTRA_OWNERSHIP_METADATA_PATH,
    OWNER_ID_RE,
    SHA_RE,
    TARGET_BASE_REF,
)


TRIAGE_INPUT_SCHEMA_VERSION = 10
MAX_CODEPATH_OWNERS = 100
MAX_ADDITIONAL_OWNERS = 8
MAX_UNCOVERED_CONCERNS = 8
MAX_OWNER_PROVENANCE_FILES = 16
MAX_OWNER_EVIDENCE_ITEMS = 3
MAX_DIFF_EXCERPT_CHARS = 1_200
MAX_ANALYSIS_RESULT_BYTES = 64_000


def should_run_ownership_analysis(
    *,
    is_open_non_draft_pr_against_main: bool,
    is_already_handled: bool,
    author_has_triage_permission: bool,
    has_actionable_linked_issue: bool,
    has_maintainer_activity: bool,
) -> bool:
    """Return whether ownership analysis can affect the controller decision."""

    return (
        is_open_non_draft_pr_against_main
        and not is_already_handled
        and (
            author_has_triage_permission
            or has_actionable_linked_issue
            or has_maintainer_activity
        )
    )


RESULT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "analyzed_file_indices": {
            "type": "array",
            "maxItems": 2_999,
            "uniqueItems": True,
            "items": {"type": "integer", "minimum": 0, "maximum": 2_998},
        },
        "additional_owners": {
            "type": "array",
            "maxItems": MAX_ADDITIONAL_OWNERS,
            "uniqueItems": True,
            "items": {
                "type": "object",
                "properties": {
                    "owner_id": {
                        "type": "string",
                        "pattern": f"^{OWNER_ID_RE.pattern}$",
                    },
                    "owned_concern": {
                        "type": "string",
                        "minLength": 20,
                        "maxLength": 800,
                    },
                    "rationale": {
                        "type": "array",
                        "minItems": 3,
                        "maxItems": 4,
                        "items": {
                            "type": "string",
                            "minLength": 20,
                            "maxLength": 800,
                        },
                    },
                    "files": {
                        "type": "array",
                        "minItems": 1,
                        "maxItems": 16,
                        "uniqueItems": True,
                        "items": {
                            "type": "string",
                            "minLength": 1,
                            "maxLength": 500,
                        },
                    },
                    "evidence": {
                        "type": "array",
                        "minItems": 1,
                        "maxItems": MAX_OWNER_EVIDENCE_ITEMS,
                        "uniqueItems": True,
                        "items": {
                            "type": "object",
                            "properties": {
                                "file": {
                                    "type": "string",
                                    "minLength": 1,
                                    "maxLength": 500,
                                },
                                "diff_excerpt": {
                                    "type": "string",
                                    "minLength": 1,
                                    "maxLength": MAX_DIFF_EXCERPT_CHARS,
                                },
                                "relevance": {
                                    "type": "string",
                                    "minLength": 20,
                                    "maxLength": 800,
                                },
                            },
                            "required": ["file", "diff_excerpt", "relevance"],
                            "additionalProperties": False,
                        },
                    },
                },
                "required": [
                    "owner_id",
                    "owned_concern",
                    "rationale",
                    "files",
                    "evidence",
                ],
                "additionalProperties": False,
            },
        },
        "uncovered_concerns": {
            "type": "array",
            "maxItems": MAX_UNCOVERED_CONCERNS,
            "uniqueItems": True,
            "items": {
                "type": "object",
                "properties": {
                    "description": {
                        "type": "string",
                        "minLength": 20,
                        "maxLength": 800,
                    },
                    "reason": {
                        "type": "string",
                        "minLength": 20,
                        "maxLength": 800,
                    },
                    "files": {
                        "type": "array",
                        "minItems": 1,
                        "maxItems": 16,
                        "uniqueItems": True,
                        "items": {
                            "type": "string",
                            "minLength": 1,
                            "maxLength": 500,
                        },
                    },
                },
                "required": ["description", "reason", "files"],
                "additionalProperties": False,
            },
        },
        "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
        "security_flags": {
            "type": "array",
            "maxItems": 20,
            "items": {"type": "string", "maxLength": 500},
        },
        "rationale": {"type": "string", "maxLength": 2000},
    },
    "required": [
        "analyzed_file_indices",
        "additional_owners",
        "uncovered_concerns",
        "confidence",
        "security_flags",
        "rationale",
    ],
    "additionalProperties": False,
}


@dataclass(frozen=True)
class AnalysisResult:
    """Carry validated decision facts and log-only ownership provenance."""

    is_open_non_draft_pr_against_main: bool
    is_already_handled: bool
    author_has_triage_permission: bool
    has_actionable_linked_issue: bool
    has_maintainer_activity: bool
    ownership_analysis: str
    codepath_owners: tuple[str, ...]
    additional_owners: tuple[str, ...]
    analyzed_head_sha: str
    owner_provenance: dict[str, dict[str, Any]]
    owner_provenance_truncated: bool = False
    has_uncovered_concerns: bool = False

    def __post_init__(self) -> None:
        gate_facts = (
            self.is_open_non_draft_pr_against_main,
            self.is_already_handled,
            self.author_has_triage_permission,
            self.has_actionable_linked_issue,
            self.has_maintainer_activity,
        )
        if any(type(value) is not bool for value in gate_facts):
            raise ValueError("analysis result has invalid triage facts")
        if self.has_maintainer_activity and (
            not self.is_open_non_draft_pr_against_main
            or self.is_already_handled
            or self.author_has_triage_permission
            or self.has_actionable_linked_issue
        ):
            raise ValueError("analysis result has inconsistent maintainer activity")
        if not isinstance(
            self.ownership_analysis, str
        ) or self.ownership_analysis not in {
            "not_run",
            "completed",
            "incomplete",
        }:
            raise ValueError("analysis result has an invalid ownership analysis")
        if len(self.codepath_owners) > MAX_CODEPATH_OWNERS:
            raise ValueError("analysis result has too many codepath owners")
        if len(self.additional_owners) > MAX_ADDITIONAL_OWNERS:
            raise ValueError("analysis result has too many additional owners")
        if not isinstance(self.analyzed_head_sha, str) or not SHA_RE.fullmatch(
            self.analyzed_head_sha
        ):
            raise ValueError("analysis result has an invalid analyzed head SHA")
        if type(self.owner_provenance_truncated) is not bool:
            raise ValueError("analysis result has invalid owner provenance")
        if type(self.has_uncovered_concerns) is not bool:
            raise ValueError("analysis result has invalid uncovered-concern state")

        owner_keys: list[str] = []
        for owner in self.codepath_owners:
            if not isinstance(owner, str) or not CODEPATH_OWNER_RE.fullmatch(owner):
                raise ValueError("analysis result has an invalid codepath owner")
            owner_keys.append(owner.casefold())
        if (
            len(set(owner_keys)) != len(owner_keys)
            or tuple(sorted(self.codepath_owners, key=str.casefold))
            != self.codepath_owners
        ):
            raise ValueError("analysis result codepath owners are not canonical")

        if any(
            not isinstance(owner, str) or not OWNER_ID_RE.fullmatch(owner)
            for owner in self.additional_owners
        ):
            raise ValueError("analysis result has an invalid additional owner")
        if (
            len(set(self.additional_owners)) != len(self.additional_owners)
            or tuple(sorted(self.additional_owners)) != self.additional_owners
        ):
            raise ValueError("analysis result additional owners are not canonical")
        internal_codepath_owners = {
            owner for owner in self.codepath_owners if not owner.startswith("@")
        }
        if internal_codepath_owners & set(self.additional_owners):
            raise ValueError("analysis result repeats a codepath owner")

        expected_provenance = {
            **{owner: "codepath" for owner in self.codepath_owners},
            **{owner: "semantic" for owner in self.additional_owners},
        }
        if (
            not isinstance(self.owner_provenance, dict)
            or (self.owner_provenance_truncated and self.owner_provenance)
            or (
                not self.owner_provenance_truncated
                and set(self.owner_provenance) != set(expected_provenance)
            )
            or tuple(self.owner_provenance) != tuple(sorted(self.owner_provenance))
        ):
            raise ValueError("analysis result has invalid owner provenance")
        for owner, provenance in self.owner_provenance.items():
            if not isinstance(provenance, dict) or set(provenance) != {
                "source",
                "files",
                "total_file_count",
                "llm_justification",
            }:
                raise ValueError("analysis result has invalid owner provenance")
            files = provenance["files"]
            total_file_count = provenance["total_file_count"]
            if (
                provenance["source"] != expected_provenance[owner]
                or not isinstance(files, list)
                or not files
                or len(files) > MAX_OWNER_PROVENANCE_FILES
                or len(set(files)) != len(files)
                or sorted(files) != files
                or any(not isinstance(path, str) or not path for path in files)
                or type(total_file_count) is not int
                or total_file_count < len(files)
            ):
                raise ValueError("analysis result has invalid owner provenance")
            justification = provenance["llm_justification"]
            if provenance["source"] == "codepath":
                if justification is not None:
                    raise ValueError("codepath provenance has an LLM justification")
                continue
            if total_file_count != len(files):
                raise ValueError("semantic provenance has an invalid file count")
            if not isinstance(justification, dict) or set(justification) != {
                "owned_concern",
                "rationale",
                "evidence",
            }:
                raise ValueError("semantic provenance has an invalid justification")
            concern = justification["owned_concern"]
            rationale = justification["rationale"]
            evidence = justification["evidence"]
            if (
                not isinstance(concern, str)
                or not 20 <= len(concern) <= 800
                or not isinstance(rationale, list)
                or not 3 <= len(rationale) <= 4
                or any(
                    not isinstance(reason, str) or not 20 <= len(reason) <= 800
                    for reason in rationale
                )
                or not isinstance(evidence, list)
                or not 1 <= len(evidence) <= MAX_OWNER_EVIDENCE_ITEMS
            ):
                raise ValueError("semantic provenance has an invalid justification")
            evidence_keys: list[tuple[str, str, str]] = []
            for item in evidence:
                if not isinstance(item, dict) or set(item) != {
                    "file",
                    "diff_excerpt",
                    "relevance",
                }:
                    raise ValueError("semantic provenance has invalid evidence")
                path = item["file"]
                excerpt = item["diff_excerpt"]
                relevance = item["relevance"]
                if (
                    not isinstance(path, str)
                    or path not in files
                    or not isinstance(excerpt, str)
                    or not 1 <= len(excerpt) <= MAX_DIFF_EXCERPT_CHARS
                    or not isinstance(relevance, str)
                    or not 20 <= len(relevance) <= 800
                ):
                    raise ValueError("semantic provenance has invalid evidence")
                evidence_keys.append((path, excerpt, relevance))
            if len(set(evidence_keys)) != len(evidence_keys):
                raise ValueError("semantic provenance has duplicate evidence")

        if should_run_ownership_analysis(
            is_open_non_draft_pr_against_main=self.is_open_non_draft_pr_against_main,
            is_already_handled=self.is_already_handled,
            author_has_triage_permission=self.author_has_triage_permission,
            has_actionable_linked_issue=self.has_actionable_linked_issue,
            has_maintainer_activity=self.has_maintainer_activity,
        ):
            if self.ownership_analysis == "not_run":
                raise ValueError("eligible triage facts require ownership analysis")
        elif (
            self.ownership_analysis != "not_run"
            or self.codepath_owners
            or self.additional_owners
        ):
            raise ValueError("ineligible triage facts carry ownership state")
        if (
            self.ownership_analysis in {"not_run", "incomplete"}
            and self.additional_owners
        ):
            raise ValueError("ownership analysis state cannot carry additional owners")
        if self.ownership_analysis != "completed" and self.has_uncovered_concerns:
            raise ValueError("ownership analysis state cannot carry uncovered concerns")

    @classmethod
    def create(
        cls,
        *,
        is_open_non_draft_pr_against_main: bool,
        is_already_handled: bool,
        author_has_triage_permission: bool,
        has_actionable_linked_issue: bool,
        has_maintainer_activity: bool,
        ownership_analysis: str,
        codepath_owners: list[str] | tuple[str, ...] = (),
        additional_owners: list[str] | tuple[str, ...] = (),
        analyzed_head_sha: str,
        owner_provenance: dict[str, dict[str, Any]] | None = None,
        owner_provenance_truncated: bool = False,
        has_uncovered_concerns: bool = False,
    ) -> AnalysisResult:
        """Canonicalize one controller-produced result."""

        provenance = {} if owner_provenance is None else owner_provenance
        if not isinstance(provenance, dict) or any(
            not isinstance(owner, str) for owner in provenance
        ):
            raise ValueError("analysis result has invalid owner provenance")

        return cls(
            is_open_non_draft_pr_against_main,
            is_already_handled,
            author_has_triage_permission,
            has_actionable_linked_issue,
            has_maintainer_activity,
            ownership_analysis,
            tuple(sorted(codepath_owners, key=str.casefold)),
            tuple(sorted(additional_owners)),
            analyzed_head_sha,
            dict(sorted(provenance.items())),
            owner_provenance_truncated,
            has_uncovered_concerns,
        )

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> AnalysisResult:
        """Parse and validate the exact cross-job result shape."""

        if not isinstance(value, dict) or set(value) != {
            "is_open_non_draft_pr_against_main",
            "is_already_handled",
            "author_has_triage_permission",
            "has_actionable_linked_issue",
            "has_maintainer_activity",
            "ownership_analysis",
            "codepath_owners",
            "additional_owners",
            "analyzed_head_sha",
            "owner_provenance",
            "owner_provenance_truncated",
            "has_uncovered_concerns",
        }:
            raise ValueError("analysis result has invalid fields")
        if (
            not isinstance(value["codepath_owners"], list)
            or not isinstance(value["additional_owners"], list)
            or not isinstance(value["owner_provenance"], dict)
        ):
            raise ValueError("analysis result owners must be arrays")
        codepath_owners = value["codepath_owners"]
        additional_owners = value["additional_owners"]
        provenance = value["owner_provenance"]
        if (
            any(not isinstance(owner, str) for owner in codepath_owners)
            or codepath_owners != sorted(codepath_owners, key=str.casefold)
            or any(not isinstance(owner, str) for owner in additional_owners)
            or additional_owners != sorted(additional_owners)
        ):
            raise ValueError("analysis result is not canonical")
        return cls.create(
            is_open_non_draft_pr_against_main=value[
                "is_open_non_draft_pr_against_main"
            ],
            is_already_handled=value["is_already_handled"],
            author_has_triage_permission=value["author_has_triage_permission"],
            has_actionable_linked_issue=value["has_actionable_linked_issue"],
            has_maintainer_activity=value["has_maintainer_activity"],
            ownership_analysis=value["ownership_analysis"],
            codepath_owners=codepath_owners,
            additional_owners=additional_owners,
            analyzed_head_sha=value["analyzed_head_sha"],
            owner_provenance=provenance,
            owner_provenance_truncated=value["owner_provenance_truncated"],
            has_uncovered_concerns=value["has_uncovered_concerns"],
        )

    @classmethod
    def from_json(cls, raw: str) -> AnalysisResult:
        """Parse a bounded single-line cross-job result."""

        if len(raw.encode()) > MAX_ANALYSIS_RESULT_BYTES:
            raise ValueError("analysis result exceeds the size limit")
        try:
            value = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError("analysis result is not valid JSON") from exc
        return cls.from_dict(value)

    def to_dict(self) -> dict[str, Any]:
        """Return the compact JSON-compatible cross-job record."""

        return {
            "is_open_non_draft_pr_against_main": self.is_open_non_draft_pr_against_main,
            "is_already_handled": self.is_already_handled,
            "author_has_triage_permission": self.author_has_triage_permission,
            "has_actionable_linked_issue": self.has_actionable_linked_issue,
            "has_maintainer_activity": self.has_maintainer_activity,
            "ownership_analysis": self.ownership_analysis,
            "codepath_owners": list(self.codepath_owners),
            "additional_owners": list(self.additional_owners),
            "analyzed_head_sha": self.analyzed_head_sha,
            "owner_provenance": json.loads(json.dumps(self.owner_provenance)),
            "owner_provenance_truncated": self.owner_provenance_truncated,
            "has_uncovered_concerns": self.has_uncovered_concerns,
        }

    def to_json(self) -> str:
        """Return the compact single-line cross-job representation."""

        value = json.dumps(self.to_dict(), separators=(",", ":"))
        if len(value.encode()) > MAX_ANALYSIS_RESULT_BYTES:
            raise ValueError("analysis result exceeds the size limit")
        return value


def _validate_source(
    source: Any,
    *,
    repository: str,
    ref: str,
    paths: set[str],
) -> None:
    if not isinstance(source, dict) or set(source) != {
        "repository",
        "path",
        "ref",
        "blob_sha",
    }:
        raise RuntimeError("ownership source is invalid")
    if (
        source["repository"] != repository
        or source["ref"] != ref
        or source["path"] not in paths
        or not SHA_RE.fullmatch(source["blob_sha"])
    ):
        raise RuntimeError("ownership source does not match the workflow revision")


def _canonicalize_ownership(config: dict[str, Any]) -> dict[str, Any]:
    """Copy analysis ownership artifacts into deterministic in-memory forms."""

    codepath = config["codepath_owners"]
    extra_metadata = config["extra_ownership_metadata"]
    return {
        "codepath_owners": {
            "source": dict(codepath["source"]),
            "owners": tuple(codepath["owners"]),
            "matched_path_groups": tuple(
                {
                    "owners": tuple(group["owners"]),
                    "paths": tuple(group["paths"]),
                }
                for group in codepath["matched_path_groups"]
            ),
            "paths_without_owners": tuple(
                {"path": item["path"], "reason": item["reason"]}
                for item in codepath["paths_without_owners"]
            ),
        },
        "extra_ownership_metadata": {
            "source": dict(extra_metadata["source"]),
            "owners": dict(sorted(extra_metadata["owners"].items())),
        },
    }


def _validate_ownership(
    ownership: dict[str, Any],
    *,
    repository: str,
    ref: str,
    changed_paths: set[str],
) -> None:
    if not isinstance(ownership, dict) or set(ownership) != {
        "codepath_owners",
        "extra_ownership_metadata",
    }:
        raise RuntimeError("ownership artifacts are incomplete")

    codepath = ownership["codepath_owners"]
    extra_metadata = ownership["extra_ownership_metadata"]
    if not all(isinstance(section, dict) for section in (codepath, extra_metadata)):
        raise RuntimeError("ownership artifacts are invalid")
    if set(codepath) != {
        "source",
        "owners",
        "matched_path_groups",
        "paths_without_owners",
    }:
        raise RuntimeError("codepath-owner artifact is invalid")
    if set(extra_metadata) != {"source", "owners"}:
        raise RuntimeError("extra ownership metadata is invalid")

    _validate_source(
        codepath["source"],
        repository=repository,
        ref=ref,
        paths={CODEPATH_OWNERS_PATH},
    )
    _validate_source(
        extra_metadata["source"],
        repository=repository,
        ref=ref,
        paths={EXTRA_OWNERSHIP_METADATA_PATH},
    )
    codepath_owners = codepath["owners"]
    if (
        not isinstance(codepath_owners, (list, tuple))
        or any(
            not isinstance(owner, str) or not CODEPATH_OWNER_RE.fullmatch(owner)
            for owner in codepath_owners
        )
        or len({owner.casefold() for owner in codepath_owners}) != len(codepath_owners)
        or tuple(sorted(codepath_owners, key=str.casefold)) != tuple(codepath_owners)
    ):
        raise RuntimeError("codepath owners are invalid")
    target_org = repository.split("/", 1)[0].casefold()
    if any(
        owner.startswith("@")
        and "/" in owner
        and owner[1:].split("/", 1)[0].casefold() != target_org
        for owner in codepath_owners
    ):
        raise RuntimeError("codepath owners contain a foreign team")

    grouped_paths: set[str] = set()
    grouped_owners: set[str] = set()
    groups = codepath["matched_path_groups"]
    if not isinstance(groups, (list, tuple)):
        raise RuntimeError("matched codepath-owner path groups are invalid")
    for group in groups:
        if not isinstance(group, dict) or set(group) != {"owners", "paths"}:
            raise RuntimeError("matched codepath-owner path group is invalid")
        owners = group["owners"]
        paths = group["paths"]
        if (
            not isinstance(owners, (list, tuple))
            or not owners
            or len(set(owners)) != len(owners)
            or any(owner not in codepath_owners for owner in owners)
            or not isinstance(paths, (list, tuple))
            or not paths
            or len(set(paths)) != len(paths)
            or any(not isinstance(path, str) or not path for path in paths)
        ):
            raise RuntimeError("matched codepath-owner path group is invalid")
        if grouped_paths & set(paths):
            raise RuntimeError("changed path appears in multiple codepath-owner groups")
        grouped_owners.update(owners)
        grouped_paths.update(paths)

    unmatched_paths: set[str] = set()
    unmatched = codepath["paths_without_owners"]
    if not isinstance(unmatched, (list, tuple)):
        raise RuntimeError("unmatched codepath-owner paths are invalid")
    for item in unmatched:
        if (
            not isinstance(item, dict)
            or set(item) != {"path", "reason"}
            or not isinstance(item["path"], str)
            or not item["path"]
            or item["reason"] not in {"no_matching_rule", "ownerless_override"}
            or item["path"] in unmatched_paths
        ):
            raise RuntimeError("unmatched codepath-owner path is invalid")
        unmatched_paths.add(item["path"])

    if grouped_owners != set(codepath_owners):
        raise RuntimeError("codepath owners do not match path groups")
    if grouped_paths & unmatched_paths:
        raise RuntimeError("changed path has conflicting codepath-owner resolution")
    if grouped_paths | unmatched_paths != changed_paths:
        raise RuntimeError("codepath-owner artifact does not cover all changed paths")

    metadata_owners = extra_metadata["owners"]
    if not isinstance(metadata_owners, dict) or not metadata_owners:
        raise RuntimeError("extra ownership metadata owners are invalid")
    for owner, description in metadata_owners.items():
        if (
            not isinstance(owner, str)
            or not OWNER_ID_RE.fullmatch(owner)
            or not isinstance(description, str)
            or not description.strip()
            or len(description) > 2_000
        ):
            raise RuntimeError("extra ownership metadata entry is invalid")
    internal_codepath_owners = {
        owner for owner in codepath_owners if not owner.startswith("@")
    }
    if not internal_codepath_owners <= set(metadata_owners):
        raise RuntimeError("codepath owner ID is absent from extra ownership metadata")


@dataclass(frozen=True)
class TriageInput:
    """Hold one-pass triage inputs with an explicit model trust partition."""

    trusted_context: dict[str, Any]
    untrusted_pr: dict[str, Any]

    def __post_init__(self) -> None:
        """Reject incomplete inputs and mismatched ownership artifacts."""

        try:
            if set(self.trusted_context) != {
                "worker_policy",
                "codepath_owners",
                "extra_ownership_metadata",
                "analysis_metadata",
            }:
                raise RuntimeError("triage input trusted context is invalid")
            metadata = self.trusted_context["analysis_metadata"]
            if set(metadata) != {
                "target_repository",
                "target_base_ref",
                "workflow_sha",
                "diff_truncated_or_unavailable",
                "is_open_non_draft_pr_against_main",
                "is_already_handled",
                "has_actionable_linked_issue",
                "author_has_triage_permission",
                "has_maintainer_activity",
            }:
                raise RuntimeError("triage input analysis metadata is invalid")
            files = self.untrusted_pr["files"]
            changed_paths = {file["path"] for file in files}
            _validate_ownership(
                {
                    "codepath_owners": self.trusted_context["codepath_owners"],
                    "extra_ownership_metadata": self.trusted_context[
                        "extra_ownership_metadata"
                    ],
                },
                repository=metadata["target_repository"],
                ref=metadata["workflow_sha"],
                changed_paths=changed_paths,
            )
            valid = (
                isinstance(self.trusted_context["worker_policy"], str)
                and bool(self.trusted_context["worker_policy"].strip())
                and isinstance(metadata["target_repository"], str)
                and REPOSITORY_RE.fullmatch(metadata["target_repository"])
                and metadata["target_base_ref"] == TARGET_BASE_REF
                and SHA_RE.fullmatch(metadata["workflow_sha"])
                and isinstance(metadata["diff_truncated_or_unavailable"], bool)
                and isinstance(metadata["is_open_non_draft_pr_against_main"], bool)
                and isinstance(metadata["is_already_handled"], bool)
                and isinstance(metadata["has_actionable_linked_issue"], bool)
                and isinstance(metadata["author_has_triage_permission"], bool)
                and isinstance(metadata["has_maintainer_activity"], bool)
                and not (
                    metadata["has_maintainer_activity"]
                    and (
                        metadata["is_already_handled"]
                        or metadata["author_has_triage_permission"]
                        or metadata["has_actionable_linked_issue"]
                    )
                )
                and isinstance(self.untrusted_pr["number"], int)
                and self.untrusted_pr["number"] > 0
                and isinstance(self.untrusted_pr["title"], str)
                and isinstance(self.untrusted_pr["body"], str)
                and SHA_RE.fullmatch(self.untrusted_pr["head_sha"])
                and isinstance(files, list)
                and len(changed_paths) == len(files)
                and all(
                    isinstance(file, dict)
                    and isinstance(file.get("path"), str)
                    and bool(file["path"])
                    for file in files
                )
            )
        except (KeyError, TypeError, AttributeError) as exc:
            raise RuntimeError("triage input is incomplete") from exc
        if not valid:
            raise RuntimeError("triage input is inconsistent")

    @classmethod
    def create(
        cls,
        worker_policy: str,
        ownership: dict[str, Any],
        pull_request: dict[str, Any],
    ) -> TriageInput:
        """Classify collected inputs and preserve immutable routing artifacts."""

        canonical = _canonicalize_ownership(ownership)
        trusted_context = {
            "worker_policy": worker_policy,
            **canonical,
            "analysis_metadata": {
                "target_repository": pull_request["repository"],
                "target_base_ref": pull_request["base_ref"],
                "workflow_sha": pull_request["workflow_sha"],
                "diff_truncated_or_unavailable": pull_request[
                    "diff_truncated_or_unavailable"
                ],
                "is_open_non_draft_pr_against_main": pull_request[
                    "is_open_non_draft_pr_against_main"
                ],
                "is_already_handled": pull_request["is_already_handled"],
                "has_actionable_linked_issue": pull_request[
                    "has_actionable_linked_issue"
                ],
                "author_has_triage_permission": pull_request[
                    "author_has_triage_permission"
                ],
                "has_maintainer_activity": pull_request["has_maintainer_activity"],
            },
        }
        trusted_pr_fields = {
            "repository",
            "base_ref",
            "workflow_sha",
            "diff_truncated_or_unavailable",
            "is_open_non_draft_pr_against_main",
            "is_already_handled",
            "has_actionable_linked_issue",
            "author_has_triage_permission",
            "has_maintainer_activity",
        }
        return cls(
            trusted_context,
            {
                key: value
                for key, value in pull_request.items()
                if key not in trusted_pr_fields
            },
        )

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> TriageInput:
        """Restore and validate one schema-versioned triage input."""

        try:
            if value["schema_version"] != TRIAGE_INPUT_SCHEMA_VERSION:
                raise RuntimeError("triage input has an unsupported schema")
            trusted = value["trusted_context"]
            metadata = trusted["analysis_metadata"]
            return cls(
                {
                    "worker_policy": trusted["worker_policy"],
                    **_canonicalize_ownership(
                        {
                            "codepath_owners": trusted["codepath_owners"],
                            "extra_ownership_metadata": trusted[
                                "extra_ownership_metadata"
                            ],
                        }
                    ),
                    "analysis_metadata": {
                        "target_repository": metadata["target_repository"],
                        "target_base_ref": metadata["target_base_ref"],
                        "workflow_sha": metadata["workflow_sha"],
                        "diff_truncated_or_unavailable": metadata[
                            "diff_truncated_or_unavailable"
                        ],
                        "is_open_non_draft_pr_against_main": metadata[
                            "is_open_non_draft_pr_against_main"
                        ],
                        "is_already_handled": metadata["is_already_handled"],
                        "has_actionable_linked_issue": metadata[
                            "has_actionable_linked_issue"
                        ],
                        "author_has_triage_permission": metadata[
                            "author_has_triage_permission"
                        ],
                        "has_maintainer_activity": metadata["has_maintainer_activity"],
                    },
                },
                value["untrusted_pr"],
            )
        except (KeyError, TypeError) as exc:
            raise RuntimeError("triage input is invalid") from exc

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-compatible input preserving the trust partition."""

        return {
            "schema_version": TRIAGE_INPUT_SCHEMA_VERSION,
            "trusted_context": json.loads(json.dumps(self.trusted_context)),
            "untrusted_pr": json.loads(json.dumps(self.untrusted_pr)),
        }

    def to_worker_dict(self) -> dict[str, Any]:
        """Return the compact additive-only model input."""

        value = self.to_dict()
        del value["schema_version"]
        trusted = value["trusted_context"]
        path_indices = {
            file["path"]: index
            for index, file in enumerate(value["untrusted_pr"]["files"])
        }
        codepath = trusted["codepath_owners"]
        codepath["matched_path_groups"] = [
            {
                "owners": group["owners"],
                "file_indices": [path_indices[path] for path in group["paths"]],
            }
            for group in codepath["matched_path_groups"]
        ]
        codepath["paths_without_owners"] = [
            {
                "file_index": path_indices[item["path"]],
                "reason": item["reason"],
            }
            for item in codepath["paths_without_owners"]
        ]
        metadata = trusted["analysis_metadata"]
        for field in (
            "is_open_non_draft_pr_against_main",
            "is_already_handled",
            "author_has_triage_permission",
            "has_actionable_linked_issue",
            "has_maintainer_activity",
        ):
            del metadata[field]
        return value

    @property
    def number(self) -> int:
        """Return the positive PR number validated when the input was created."""

        return self.untrusted_pr["number"]

    @property
    def repository(self) -> str:
        """Return the target repository bound to the trusted analysis input."""

        return self.trusted_context["analysis_metadata"]["target_repository"]
