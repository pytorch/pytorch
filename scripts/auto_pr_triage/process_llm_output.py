#!/usr/bin/env python3
"""Validate additive owner suggestions and publish one normalized result."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

from schemas import (
    AnalysisResult,
    MAX_OWNER_PROVENANCE_FILES,
    should_run_ownership_analysis,
    TriageInput,
)


def load_action_execution(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Extract the action-validated result and bounded model metadata."""

    messages = json.loads(path.read_text())
    if not isinstance(messages, list):
        raise RuntimeError("Claude action execution log is not a JSON array")
    result_messages = [
        message
        for message in messages
        if isinstance(message, dict) and message.get("type") == "result"
    ]
    if not result_messages:
        raise RuntimeError("Claude action execution log has no result")
    final = result_messages[-1]
    if final.get("subtype") != "success" or final.get("is_error") is True:
        raise RuntimeError("Claude action did not produce a successful result")
    structured = final.get("structured_output")
    if isinstance(structured, str):
        structured = json.loads(structured)
    if not isinstance(structured, dict):
        raise RuntimeError("Claude action result has no structured output")
    metadata = {
        key: final.get(key)
        for key in ["duration_ms", "num_turns", "total_cost_usd", "usage", "modelUsage"]
    }
    return structured, metadata


def write_json(path: Path, value: Any) -> None:
    """Overwrite a path with deterministic, newline-terminated JSON."""

    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def log_analysis_record(record: dict[str, Any]) -> None:
    """Print model reasoning safely without allowing workflow commands."""

    for line in json.dumps(record, indent=2, sort_keys=True).splitlines():
        safe_line = line.replace("::", r"\u003a\u003a").replace("##[", r"\u0023\u0023[")
        print(f"Auto PR Triage result | {safe_line}")


def validate_result(triage_input: TriageInput, result: dict[str, Any]) -> list[str]:
    """Return policy violations in one additional-owner recommendation."""

    errors: list[str] = []
    metadata_owners = set(
        triage_input.trusted_context["extra_ownership_metadata"]["owners"]
    )
    changed_files = {item["path"]: item for item in triage_input.untrusted_pr["files"]}
    changed_paths = set(changed_files)
    expected_indices = set(range(len(triage_input.untrusted_pr["files"])))
    raw_indices = result["analyzed_file_indices"]
    valid_indices = (
        isinstance(raw_indices, list)
        and all(type(index) is int and index >= 0 for index in raw_indices)
        and len(raw_indices) == len(set(raw_indices))
    )
    if not valid_indices or set(raw_indices) != expected_indices:
        errors.append("analysis does not account for every changed file")

    suggestions = result["additional_owners"]
    suggested_owners = [suggestion["owner_id"] for suggestion in suggestions]
    duplicates = sorted(
        {owner for owner in suggested_owners if suggested_owners.count(owner) > 1}
    )
    if duplicates:
        errors.append(f"duplicate additional owners: {duplicates}")
    unknown_owners = sorted(set(suggested_owners) - metadata_owners)
    if unknown_owners:
        errors.append(
            f"extra ownership metadata contains unknown owners: {unknown_owners}"
        )
    codepath_owner_ids = {
        owner
        for owner in triage_input.trusted_context["codepath_owners"]["owners"]
        if not owner.startswith("@")
    }
    repeated_owners = sorted(set(suggested_owners) & codepath_owner_ids)
    if repeated_owners:
        errors.append(f"additional owners repeat codepath owners: {repeated_owners}")
    reported_paths = {
        path for suggestion in suggestions for path in suggestion["files"]
    }
    unknown_paths = sorted(reported_paths - changed_paths)
    if unknown_paths:
        errors.append(f"reported paths absent from PR: {unknown_paths}")
    for suggestion in suggestions:
        owner = suggestion["owner_id"]
        supporting_paths = set(suggestion["files"])
        for evidence in suggestion["evidence"]:
            path = evidence["file"]
            excerpt = evidence["diff_excerpt"]
            if path not in changed_files:
                errors.append(f"evidence path for {owner} is absent from PR: {path}")
                continue
            if path not in supporting_paths:
                errors.append(
                    f"evidence path for {owner} is not a supporting file: {path}"
                )
                continue
            patch = changed_files[path].get("patch")
            if not isinstance(patch, str):
                errors.append(f"evidence patch for {owner} is unavailable: {path}")
                continue
            patch_lines = patch.splitlines()
            excerpt_lines = excerpt.splitlines()
            if not any(
                patch_lines[index : index + len(excerpt_lines)] == excerpt_lines
                for index in range(len(patch_lines) - len(excerpt_lines) + 1)
            ):
                errors.append(f"evidence excerpt for {owner} is not in patch: {path}")
                continue
            if not any(line.startswith(("+", "-")) for line in excerpt.splitlines()):
                errors.append(
                    f"evidence excerpt for {owner} has no changed line: {path}"
                )
    uncovered_paths = {
        path for concern in result["uncovered_concerns"] for path in concern["files"]
    }
    unknown_uncovered_paths = sorted(uncovered_paths - changed_paths)
    if unknown_uncovered_paths:
        errors.append(
            f"uncovered concern paths absent from PR: {unknown_uncovered_paths}"
        )
    return errors


def triage_facts(triage_input: TriageInput) -> dict[str, bool]:
    """Return the explicit trusted facts carried into the apply job."""

    metadata = triage_input.trusted_context["analysis_metadata"]
    return {
        "is_open_non_draft_pr_against_main": metadata[
            "is_open_non_draft_pr_against_main"
        ],
        "is_already_handled": metadata["is_already_handled"],
        "author_has_triage_permission": metadata["author_has_triage_permission"],
        "has_actionable_linked_issue": metadata["has_actionable_linked_issue"],
        "has_maintainer_activity": metadata["has_maintainer_activity"],
    }


def bound_analysis_result(result: AnalysisResult) -> AnalysisResult:
    """Drop oversized log provenance without changing decision fields."""

    try:
        result.to_json()
        return result
    except ValueError as exc:
        if str(exc) != "analysis result exceeds the size limit":
            raise
    return replace(
        result,
        owner_provenance={},
        owner_provenance_truncated=True,
    )


def build_analysis_result(
    triage_input: TriageInput,
    recommendation: dict[str, Any] | None = None,
    *,
    incomplete: bool = False,
) -> AnalysisResult:
    """Normalize triage and ownership facts without choosing a GitHub action."""

    facts = triage_facts(triage_input)
    analyzed_head_sha = triage_input.untrusted_pr["head_sha"]
    if not should_run_ownership_analysis(**facts):
        return AnalysisResult.create(
            **facts,
            ownership_analysis="not_run",
            analyzed_head_sha=analyzed_head_sha,
        )

    codepath_owners = triage_input.trusted_context["codepath_owners"]["owners"]
    codepath_files: dict[str, set[str]] = {owner: set() for owner in codepath_owners}
    for group in triage_input.trusted_context["codepath_owners"]["matched_path_groups"]:
        for owner in group["owners"]:
            codepath_files[owner].update(group["paths"])
    owner_provenance = {
        owner: {
            "source": "codepath",
            "files": sorted(paths)[:MAX_OWNER_PROVENANCE_FILES],
            "total_file_count": len(paths),
            "llm_justification": None,
        }
        for owner, paths in codepath_files.items()
    }
    if incomplete:
        return bound_analysis_result(
            AnalysisResult.create(
                **facts,
                ownership_analysis="incomplete",
                codepath_owners=codepath_owners,
                analyzed_head_sha=analyzed_head_sha,
                owner_provenance=owner_provenance,
            )
        )
    if recommendation is None:
        raise ValueError("eligible triage facts are missing a recommendation")

    metadata = triage_input.trusted_context["analysis_metadata"]
    accept_additional = (
        not metadata["diff_truncated_or_unavailable"]
        and recommendation["confidence"] != "low"
    )
    additional_owners = (
        [suggestion["owner_id"] for suggestion in recommendation["additional_owners"]]
        if accept_additional
        else []
    )
    if accept_additional:
        for suggestion in recommendation["additional_owners"]:
            files = sorted(suggestion["files"])
            owner_provenance[suggestion["owner_id"]] = {
                "source": "semantic",
                "files": files[:MAX_OWNER_PROVENANCE_FILES],
                "total_file_count": len(files),
                "llm_justification": {
                    "owned_concern": suggestion["owned_concern"],
                    "rationale": suggestion["rationale"],
                    "evidence": suggestion["evidence"],
                },
            }
    return bound_analysis_result(
        AnalysisResult.create(
            **facts,
            ownership_analysis="completed",
            codepath_owners=codepath_owners,
            additional_owners=additional_owners,
            analyzed_head_sha=analyzed_head_sha,
            owner_provenance=owner_provenance,
            has_uncovered_concerns=bool(recommendation["uncovered_concerns"]),
        )
    )


def write_github_output(path: Path, result: AnalysisResult) -> None:
    """Publish the single-line normalized result consumed by apply."""

    value = result.to_json()
    with path.open("a") as output:
        output.write(f"analysis-result-json={value}\n")


def write_github_step_summary(
    path: Path,
    triage_input: TriageInput,
    result: AnalysisResult,
    recommendation: dict[str, Any] | None,
    validation_errors: list[str],
) -> None:
    """Append a compact human-readable analysis summary."""

    def joined(values: tuple[str, ...]) -> str:
        return ", ".join(values) or "none"

    rows = [
        ("PR", f"{triage_input.repository}#{triage_input.number}"),
        (
            "Open non-draft PR against main",
            str(result.is_open_non_draft_pr_against_main).lower(),
        ),
        ("Already handled", str(result.is_already_handled).lower()),
        (
            "Author has triage permission",
            str(result.author_has_triage_permission).lower(),
        ),
        (
            "Actionable issue linked",
            str(result.has_actionable_linked_issue).lower(),
        ),
        ("Maintainer activity", str(result.has_maintainer_activity).lower()),
        ("Ownership analysis", result.ownership_analysis),
        ("Has uncovered concerns", str(result.has_uncovered_concerns).lower()),
        ("Codepath owners", joined(result.codepath_owners)),
        ("Additional owners", joined(result.additional_owners)),
        (
            "Model confidence",
            recommendation.get("confidence") if recommendation else "not run",
        ),
        ("Validation errors", str(len(validation_errors))),
    ]
    with path.open("a") as output:
        output.write("## Auto PR Triage\n\n")
        output.write("| Field | Value |\n|---|---|\n")
        for label, value in rows:
            output.write(f"| {label} | {value} |\n")
        output.write("\nApply derives the bounded GitHub action from this record.\n")


def load_triage_input(output_dir: Path, number: int) -> TriageInput:
    """Load and bind the prepared input to the triggering pull request."""

    try:
        serialized = json.loads((output_dir / "triage_input.json").read_text())
    except json.JSONDecodeError as exc:
        raise RuntimeError("triage input is invalid") from exc
    if not isinstance(serialized, dict):
        raise RuntimeError("triage input is invalid")
    triage_input = TriageInput.from_dict(serialized)
    if triage_input.number != number:
        raise RuntimeError("triage input belongs to another pull request")
    return triage_input


def parse_args() -> argparse.Namespace:
    """Parse the PR identity, execution paths, and Actions output paths."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pr", type=int, help="pull request number")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--execution-file", type=Path, required=True)
    parser.add_argument("--github-output", type=Path, required=True)
    parser.add_argument("--github-step-summary", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    """Publish one validated, action-free analysis result."""

    args = parse_args()
    if args.pr < 1:
        raise SystemExit("PR number must be positive")
    if not args.output_dir.is_dir():
        raise SystemExit("--output-dir must be an existing directory")

    try:
        triage_input = load_triage_input(args.output_dir, args.pr)
    except Exception as exc:
        write_json(
            args.output_dir / "error.json",
            {
                "error": str(exc)[:4_000],
                "stage": "process",
                "type": type(exc).__name__,
            },
        )
        print(
            f"PR #{args.pr}: invalid prepared input; no result emitted",
            file=sys.stderr,
        )
        return 1

    recommendation: dict[str, Any] | None = None
    validation_errors: list[str] = []
    model_metadata: dict[str, Any]
    facts = triage_facts(triage_input)
    if not should_run_ownership_analysis(**facts):
        if not facts["is_open_non_draft_pr_against_main"]:
            reason = "outside_active_target"
        elif facts["is_already_handled"]:
            reason = "already_handled"
        else:
            reason = "missing_actionable_issue"
        model_metadata = {
            "status": "skipped",
            "reason": reason,
        }
        analysis_result = build_analysis_result(triage_input)
    else:
        try:
            recommendation, model_metadata = load_action_execution(args.execution_file)
            validation_errors = validate_result(triage_input, recommendation)
            analysis_result = build_analysis_result(
                triage_input,
                recommendation,
                incomplete=bool(validation_errors),
            )
        except Exception as exc:
            write_json(
                args.output_dir / "error.json",
                {
                    "error": str(exc)[:4_000],
                    "stage": "process",
                    "type": type(exc).__name__,
                },
            )
            model_metadata = {
                "status": "failed",
                "type": type(exc).__name__,
                "error": str(exc)[:500],
            }
            analysis_result = build_analysis_result(triage_input, incomplete=True)

    result_record = {
        "target_repository": triage_input.repository,
        "analysis_result": analysis_result.to_dict(),
        "recommendation": recommendation,
        "validation_errors": validation_errors,
        "model_metadata": model_metadata,
    }
    try:
        write_json(args.output_dir / "result.json", result_record)
        write_github_output(args.github_output, analysis_result)
        write_github_step_summary(
            args.github_step_summary,
            triage_input,
            analysis_result,
            recommendation,
            validation_errors,
        )
    except Exception as exc:
        write_json(
            args.output_dir / "error.json",
            {
                "error": str(exc)[:4_000],
                "stage": "publish",
                "type": type(exc).__name__,
            },
        )
        print(
            f"{triage_input.repository}#{args.pr}: result publication failed",
            file=sys.stderr,
        )
        return 1

    log_analysis_record(
        {
            "analysis_result": analysis_result.to_dict(),
            "processing_error": model_metadata.get("error"),
            "recommendation": recommendation,
            "validation_errors": validation_errors,
        }
    )
    codepath_owners = ", ".join(analysis_result.codepath_owners) or "none"
    additional_owners = ", ".join(analysis_result.additional_owners) or "none"
    print(
        f"{triage_input.repository}#{args.pr}: "
        "open_non_draft_pr_against_main="
        f"{analysis_result.is_open_non_draft_pr_against_main}; "
        f"already_handled={analysis_result.is_already_handled}; "
        f"author_has_triage_permission={analysis_result.author_has_triage_permission}; "
        f"has_actionable_linked_issue={analysis_result.has_actionable_linked_issue}; "
        f"has_maintainer_activity={analysis_result.has_maintainer_activity}; "
        f"ownership={analysis_result.ownership_analysis}; "
        f"codepath owners={codepath_owners}; additional owners={additional_owners}; "
        f"validator_errors={len(validation_errors)}",
        flush=True,
    )
    print(f"Run artifacts: {args.output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
