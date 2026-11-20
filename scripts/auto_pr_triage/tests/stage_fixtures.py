"""Shared fixtures for the intake, ownership, and schema stage tests."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from build_ownership_input import WORKER_POLICY_PATH
from schemas import (
    ActionPlan,
    AdditionalOwnerConcern,
    AddLabels,
    ChangedFile,
    IntakeFacts,
    IntakeResult,
    LLMInput,
    MISSING_ACTIONABLE_ISSUE_ACTIONS,
    OwnershipResult,
    PlanContext,
    PullRequestIdentity,
    RequestReviewers,
    UncoveredConcern,
    UntrustedContext,
)
from trusted_config import EXTRA_OWNERSHIP_METADATA_PATH


WORKER_POLICY = WORKER_POLICY_PATH.read_text().strip()
REPOSITORY = "pytorch/ciforge"
HEAD_SHA = "c" * 40


def ownership_config(
    *,
    rosters: dict[str, list[str]] | None = None,
    codepath_owners: list[str] | None = None,
    changed_paths: list[str] | None = None,
    repository: str = REPOSITORY,
    bypass: dict[str, str] | None = None,
) -> dict[str, object]:
    rosters = rosters or {"owner": ["@owner"]}
    bypass = bypass or {}
    owners = (
        [f"@{repository.split('/', 1)[0]}/baseline"]
        if codepath_owners is None
        else codepath_owners
    )
    paths = changed_paths or ["torch/file.py"]
    codepath = {
        "source": {
            "repository": repository,
            "path": "CODEOWNERS",
            "ref": "a" * 40,
            "blob_sha": "c" * 40,
        },
        "owners": sorted(set(owners), key=str.casefold),
        "matched_path_groups": (
            [{"owners": sorted(set(owners), key=str.casefold), "paths": paths}]
            if owners
            else []
        ),
        "paths_without_owners": ([] if owners else list(paths)),
    }

    def source(*, path: str, blob_sha: str) -> dict[str, str]:
        return {
            "repository": repository,
            "path": path,
            "ref": "a" * 40,
            "blob_sha": blob_sha,
        }

    return {
        "codepath_owners": codepath,
        "extra_ownership_metadata": {
            "source": source(path=EXTRA_OWNERSHIP_METADATA_PATH, blob_sha="b" * 40),
            "owners": {
                owner: {
                    "description": f"Owns {owner}.",
                    "bypass_intake_criteria": bypass.get(owner),
                }
                for owner in rosters
            },
        },
    }


def pr_identity(repository: str = REPOSITORY) -> PullRequestIdentity:
    return PullRequestIdentity(repository, 123, HEAD_SHA, "a" * 40)


def intake_facts(
    *,
    is_open_non_draft_pr_against_main: bool = True,
    is_already_handled: bool = False,
    author_has_triage_permission: bool = False,
    has_actionable_linked_issue: bool | None = None,
    has_maintainer_activity: bool = False,
    has_related_actionable_issue: bool = False,
    supporters: list[str] | tuple[str, ...] = (),
    actionable_labelers: list[str] | tuple[str, ...] = (),
    maintainer_requested_reviewers: list[str] | tuple[str, ...] = (),
) -> IntakeFacts:
    if has_actionable_linked_issue is None:
        # Facts are gathered only for an active, unhandled PR.
        has_actionable_linked_issue = (
            is_open_non_draft_pr_against_main and not is_already_handled
        )
    return IntakeFacts(
        is_open_non_draft_pr_against_main=is_open_non_draft_pr_against_main,
        is_already_handled=is_already_handled,
        author_has_triage_permission=author_has_triage_permission,
        has_actionable_linked_issue=has_actionable_linked_issue,
        has_maintainer_activity=has_maintainer_activity,
        has_supporter=bool(supporters),
        has_related_actionable_issue=has_related_actionable_issue,
        supporters=tuple(supporters),
        actionable_labelers=tuple(actionable_labelers),
        maintainer_requested_reviewers=tuple(maintainer_requested_reviewers),
    )


def intake_result(
    *, title: str = "fixture", body: str = "", **facts: Any
) -> IntakeResult:
    return IntakeResult(
        pr_identity(), intake_facts(**facts), "external-author", title, body
    )


def llm_input(
    *,
    repository: str = REPOSITORY,
    team_rosters: dict[str, list[str]] | None = None,
    codepath_owners: list[str] | None = None,
    title: str = "fixture",
    body: str = "",
    diff_truncated_or_unavailable: bool = False,
    bypass: dict[str, str] | None = None,
    files: tuple[ChangedFile, ...] | None = None,
) -> LLMInput:
    rosters = team_rosters or {"owner": ["@owner"]}
    files = files or (changed_file(),)
    ownership = ownership_config(
        rosters=rosters,
        codepath_owners=codepath_owners,
        changed_paths=[file.path for file in files],
        repository=repository,
        bypass=bypass,
    )
    return LLMInput.create(
        worker_policy=WORKER_POLICY,
        ownership=ownership,
        diff_truncated_or_unavailable=diff_truncated_or_unavailable,
        untrusted_context=UntrustedContext(title=title, body=body, files=files),
    )


PATCH = "@@ -1,2 +1,2 @@\n unchanged context\n-old behavior\n+new behavior"


def changed_file(
    *, patch: str | None = PATCH, path: str = "torch/file.py", truncated: bool = False
) -> ChangedFile:
    return ChangedFile(
        path=path,
        status="modified",
        additions=1,
        deletions=1,
        patch=patch,
        patch_truncated_or_unavailable=truncated or patch is None,
    )


BYPASS_CRITERIA = "PRs that fix incorrect gradients."


def evidence(file: str) -> dict[str, str]:
    return {
        "file": file,
        "diff_excerpt": "+new behavior",
        "relevance": "The changed line implements the owned behavior.",
    }


def bypass_match(*, file: str, quote: str = BYPASS_CRITERIA) -> dict[str, object]:
    return {
        "criteria_quote": quote,
        "rationale": ["The change fixes a gradient formula the team asked to see."],
        "evidence": [evidence(file)],
    }


def concern(*, file: str, description: str) -> dict[str, object]:
    return {"description": description, "files": [file], "evidence": [evidence(file)]}


def llm_result(
    *,
    additional_owners: list[str] | None = None,
    uncovered_concerns: list[dict[str, object]] | None = None,
    codepath_owner_concerns: list[dict[str, object]] | None = None,
    confidence: str = "high",
    confidences: dict[str, str] | None = None,
    bypass: list[str] | tuple[str, ...] = (),
    bypass_quote: str = BYPASS_CRITERIA,
    owner_files: dict[str, str] | None = None,
) -> dict[str, object]:
    """Build a raw LLM answer; uncovered_concerns take description, reason, files."""

    confidences = confidences or {}
    owner_files = owner_files or {}
    uncovered = [
        {
            "concern": {
                "description": item["description"],
                "files": item["files"],
                "evidence": item.get("evidence", [evidence(item["files"][0])]),
            },
            "reason": item["reason"],
        }
        for item in uncovered_concerns or []
    ]
    return {
        "codepath_owner_concerns": codepath_owner_concerns or [],
        "additional_owner_concerns": [
            {
                "concern": concern(
                    file=owner_files.get(owner, "torch/file.py"),
                    description=f"{owner} owns a distinct changed contract.",
                ),
                "owner_id": owner,
                "rationale": [
                    "Changed behavior requires this team's technical review.",
                    "The configured metadata assigns this contract to the team.",
                    "The concern is distinct from supporting or mechanical edits.",
                ],
                "confidence": confidences.get(owner, confidence),
                "bypass_intake_match": (
                    bypass_match(
                        file=owner_files.get(owner, "torch/file.py"), quote=bypass_quote
                    )
                    if owner in bypass
                    else None
                ),
            }
            for owner in additional_owners or []
        ],
        "uncovered_concerns": uncovered,
        "security_flags": [],
    }


def additional_owner_concern(
    owner: str, *, bypass: bool = False, file: str = "torch/file.py"
) -> AdditionalOwnerConcern:
    return AdditionalOwnerConcern.from_dict(
        {
            "concern": concern(
                file=file, description=f"{owner} owns a distinct changed contract."
            ),
            "owner_id": owner,
            "rationale": [
                "Changed behavior requires this team's technical review.",
                "The configured metadata assigns this contract to the team.",
                "The concern is distinct from supporting or mechanical edits.",
            ],
            "confidence": "high",
            "bypass_intake_match": bypass_match(file=file) if bypass else None,
        }
    )


def uncovered_concern(file: str = "torch/file.py") -> UncoveredConcern:
    return UncoveredConcern.from_dict(
        {
            "concern": concern(
                file=file, description="An unowned changed contract needs review."
            ),
            "reason": "No configured owner describes this changed contract.",
        }
    )


def discarded_bypass_concern(owner: str = "discarded") -> AdditionalOwnerConcern:
    return replace(additional_owner_concern(owner, bypass=True), confidence="low")


def make_ownership_result(
    *,
    llm_run_status: str = "succeeded",
    codepath_owners: list[str] | tuple[str, ...] = (),
    additional_owners: list[str] | tuple[str, ...] = (),
    has_uncovered_concerns: bool = False,
    bypass_intake_matches: list[str] | tuple[str, ...] = (),
    has_discarded_bypass_intake_match: bool = False,
) -> OwnershipResult:
    return OwnershipResult.create(
        llm_run_status=llm_run_status,
        codepath_owners={owner: ("torch/file.py",) for owner in codepath_owners},
        additional_owner_concerns=[
            additional_owner_concern(owner, bypass=owner in bypass_intake_matches)
            for owner in additional_owners
        ],
        discarded_additional_owner_concerns=(
            [discarded_bypass_concern()] if has_discarded_bypass_intake_match else []
        ),
        uncovered_concerns=[uncovered_concern()] if has_uncovered_concerns else [],
    )


def make_action_plan(facts: IntakeFacts, **overrides: Any) -> ActionPlan:
    """Build a plan from context fields and planner-style effect fields.

    supporter_reviewers, roster_reviewers, and labels become actions in the
    planner's order; pass actions= to supply the action list directly.
    """

    values = {
        "run_attempt": 1,
        "codepath_owners": ("@codepath-owner",),
        "additional_owners": ("autograd",),
        "bypass_intake_matches": (),
        "decision": "triage",
        "supporter_reviewers": (),
        "roster_reviewers": (),
        "labels": ("triaged", "bot-triaged"),
    }
    identity = overrides.pop("identity", pr_identity())
    values.update(overrides)
    context = PlanContext(
        identity,
        facts,
        values["run_attempt"],
        values["codepath_owners"],
        values["additional_owners"],
        values["bypass_intake_matches"],
    )
    if "actions" in values:
        actions = values["actions"]
    elif values["decision"] == "missing_actionable_issue":
        actions = MISSING_ACTIONABLE_ISSUE_ACTIONS
    else:
        actions = []
        if values["supporter_reviewers"]:
            supporters = values["supporter_reviewers"]
            actions.append(RequestReviewers(supporters, "supporter"))
        if values["roster_reviewers"]:
            roster = values["roster_reviewers"]
            actions.append(RequestReviewers(roster, "owner_roster"))
        if values["labels"]:
            actions.append(AddLabels(values["labels"]))
    return ActionPlan(context, values["decision"], tuple(actions))
