#!/usr/bin/env python3
"""Stage 2c: validate the LLM's owner suggestions into an ownership result.

An inactive or handled PR gets a skipped result without LLM output. Any
LLM, schema, or validation failure yields a failed result that keeps the
deterministic codepath owners. Each suggested owner is accepted on its own
confidence and on its own supporting files having complete patches.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from schemas import (
    ChangedFile,
    Concern,
    Evidence,
    IntakeResult,
    LLMInput,
    LLMResult,
    OwnershipResult,
)
from step_summary import summary_prose


def load_action_execution(path: Path) -> tuple[LLMResult, dict[str, Any]]:
    """Extract the action-validated LLM result and bounded LLM metadata."""

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
    return LLMResult.from_dict(structured), metadata


def write_json(*, path: Path, value: Any) -> None:
    """Overwrite a path with deterministic, newline-terminated JSON."""

    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def log_analysis_record(record: dict[str, Any]) -> None:
    """Print LLM reasoning safely without allowing workflow commands."""

    for line in json.dumps(record, indent=2, sort_keys=True).splitlines():
        safe_line = line.replace("::", r"\u003a\u003a").replace("##[", r"\u0023\u0023[")
        print(f"Auto PR Triage result | {safe_line}")


def evidence_errors(
    *,
    subject: str,
    evidence: tuple[Evidence, ...],
    supporting_paths: tuple[str, ...],
    changed_files: dict[str, ChangedFile],
) -> list[str]:
    """Check that each excerpt is a verbatim changed hunk of a supporting file."""

    errors: list[str] = []
    for item in evidence:
        path = item.file
        if path not in changed_files:
            errors.append(f"evidence path for {subject} is absent from PR: {path}")
            continue
        if path not in supporting_paths:
            errors.append(f"evidence file for {subject} is not supporting: {path}")
            continue
        patch = changed_files[path].patch
        if patch is None:
            errors.append(f"evidence patch for {subject} is unavailable: {path}")
            continue
        patch_lines = patch.splitlines()
        excerpt_lines = item.diff_excerpt.splitlines()
        if not any(
            patch_lines[index : index + len(excerpt_lines)] == excerpt_lines
            for index in range(len(patch_lines) - len(excerpt_lines) + 1)
        ):
            errors.append(f"evidence excerpt for {subject} is not in patch: {path}")
            continue
        if not any(line.startswith(("+", "-")) for line in excerpt_lines):
            errors.append(f"evidence excerpt for {subject} has no changed line: {path}")
    return errors


def validate_result(*, llm_input: LLMInput, result: LLMResult) -> list[str]:
    """Return policy violations in one concern-based LLM result."""

    errors: list[str] = []
    trusted = llm_input.trusted_context
    metadata = trusted.extra_ownership_metadata
    metadata_owners = set(metadata)
    changed_files = {item.path: item for item in llm_input.untrusted_context.files}
    changed_paths = set(changed_files)

    suggestions = result.additional_owner_concerns
    suggested_owners = [suggestion.owner_id for suggestion in suggestions]
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
    unconfigured_bypass = sorted(
        suggestion.owner_id
        for suggestion in suggestions
        if suggestion.bypass_intake_match
        and suggestion.owner_id not in trusted.teams_with_intake_bypass
    )
    if unconfigured_bypass:
        errors.append(f"bypass claimed without criteria: {unconfigured_bypass}")
    reported_paths = {p for suggestion in suggestions for p in suggestion.concern.files}
    unknown_paths = sorted(reported_paths - changed_paths)
    if unknown_paths:
        errors.append(f"reported paths absent from PR: {unknown_paths}")
    for suggestion in suggestions:
        owner = suggestion.owner_id
        concern = suggestion.concern
        errors += evidence_errors(
            subject=owner,
            evidence=concern.evidence,
            supporting_paths=concern.files,
            changed_files=changed_files,
        )
        match = suggestion.bypass_intake_match
        if match is None or owner not in trusted.teams_with_intake_bypass:
            continue
        criteria = " ".join(str(metadata[owner].bypass_intake_criteria).split())
        if " ".join(match.criteria_quote.split()) not in criteria:
            errors.append(f"bypass criteria quote for {owner} is not in its criteria")
        errors += evidence_errors(
            subject=f"{owner} bypass",
            evidence=match.evidence,
            supporting_paths=concern.files,
            changed_files=changed_files,
        )
    codepath_owners = set(trusted.codepath_owners.owners)
    for item in result.codepath_owner_concerns:
        foreign = sorted(set(item.codepath_owners) - codepath_owners)
        if foreign:
            errors.append(
                f"concern names owners that are not codepath owners: {foreign}"
            )
    for label, items in (
        ("codepath owner concern", result.codepath_owner_concerns),
        ("uncovered concern", result.uncovered_concerns),
    ):
        for item in items:
            concern = item.concern
            unknown = sorted(set(concern.files) - changed_paths)
            if unknown:
                errors.append(f"{label} paths absent from PR: {unknown}")
            errors += evidence_errors(
                subject=label,
                evidence=concern.evidence,
                supporting_paths=concern.files,
                changed_files=changed_files,
            )
    return errors


def build_ownership_result(
    *,
    llm_input: LLMInput,
    llm_result: LLMResult | None = None,
    llm_failed: bool = False,
) -> OwnershipResult:
    """Normalize the LLM's owners without choosing a GitHub action."""

    codepath = llm_input.trusted_context.codepath_owners
    paths = [file.path for file in llm_input.untrusted_context.files]
    codepath_owners: dict[str, set[str]] = {owner: set() for owner in codepath.owners}
    for group in codepath.matched_path_groups:
        for owner in group.owners:
            codepath_owners[owner].update(paths[i] for i in group.file_indices)
    if llm_failed:
        return OwnershipResult.create(
            llm_run_status="failed", codepath_owners=codepath_owners
        )
    if llm_result is None:
        raise ValueError("a succeeded LLM run is missing its LLM result")

    incomplete_patches = {
        file.path
        for file in llm_input.untrusted_context.files
        if file.patch_truncated_or_unavailable
    }
    accepted, discarded = [], []
    for suggestion in llm_result.additional_owner_concerns:
        concern_files = set(suggestion.concern.files)
        if suggestion.confidence == "low" or incomplete_patches & concern_files:
            discarded.append(suggestion)
        else:
            accepted.append(suggestion)
    return OwnershipResult.create(
        llm_run_status="succeeded",
        codepath_owners=codepath_owners,
        codepath_owner_concerns=llm_result.codepath_owner_concerns,
        additional_owner_concerns=accepted,
        discarded_additional_owner_concerns=discarded,
        uncovered_concerns=llm_result.uncovered_concerns,
    )


def evidence_files(concern: Concern) -> str:
    """Render a concern's evidence files as inert Markdown."""

    return ", ".join(summary_prose(item.file) for item in concern.evidence)


def write_github_step_summary(
    *,
    path: Path,
    intake: IntakeResult,
    result: OwnershipResult,
    validation_errors: list[str],
) -> None:
    """Append a compact human-readable analysis summary."""

    def joined(values: tuple[str, ...]) -> str:
        return ", ".join(values) or "none"

    identity = intake.identity

    facts = intake.facts
    rows = [
        ("PR", f"{identity.repository}#{identity.number}"),
        (
            "Open non-draft PR against main",
            str(facts.is_open_non_draft_pr_against_main).lower(),
        ),
        ("Already handled", str(facts.is_already_handled).lower()),
        (
            "Author has triage permission",
            str(facts.author_has_triage_permission).lower(),
        ),
        (
            "Actionable issue linked",
            str(facts.has_actionable_linked_issue).lower(),
        ),
        ("Maintainer activity", str(facts.has_maintainer_activity).lower()),
        ("Verified supporters", joined(facts.supporters)),
        ("Actionable labelers", joined(facts.actionable_labelers)),
        (
            "Maintainer-requested reviewers",
            joined(facts.maintainer_requested_reviewers),
        ),
        (
            "Related actionable issue",
            str(facts.has_related_actionable_issue).lower(),
        ),
        ("LLM run status", result.llm_run_status),
        ("Has uncovered concerns", str(result.has_uncovered_concerns).lower()),
        ("Codepath owners", joined(tuple(result.codepath_owners))),
        ("Additional owners", joined(result.additional_owners)),
        ("Bypass intake owners", joined(result.bypass_intake_matches)),
        ("Validation errors", str(len(validation_errors))),
    ]
    with path.open("a") as output:
        output.write("## Auto PR Triage\n\n")
        output.write("| Field | Value |\n|---|---|\n")
        for label, value in rows:
            output.write(f"| {label} | {value} |\n")
        if result.llm_run_status == "succeeded":
            output.write("\n### LLM concerns\n\n")
            for item in result.codepath_owner_concerns:
                owners = ", ".join(f"`{owner}`" for owner in item.codepath_owners)
                output.write(
                    f"- Codepath owner concern ({owners}): "
                    f"{summary_prose(item.concern.description)} "
                    f"Why covered: {summary_prose(item.reason)} "
                    f"Evidence: {evidence_files(item.concern)}\n"
                )
            for status, owners in (
                ("accepted", result.additional_owner_concerns),
                ("discarded", result.discarded_additional_owner_concerns),
            ):
                for owner in owners:
                    match = owner.bypass_intake_match
                    bypass = ", bypass intake" if match else ""
                    output.write(
                        f"- Additional owner `{owner.owner_id}` ({status}, "
                        f"{owner.confidence} confidence{bypass}): "
                        f"{summary_prose(owner.concern.description)} "
                        f"Evidence: {evidence_files(owner.concern)}\n"
                    )
                    if match:
                        quote = summary_prose(match.criteria_quote)
                        reasons = " ".join(summary_prose(r) for r in match.rationale)
                        output.write(
                            f'  - Matches bypass intake "{quote}": {reasons}\n'
                        )
            for item in result.uncovered_concerns:
                output.write(
                    f"- Uncovered concern: {summary_prose(item.concern.description)} "
                    f"Why uncovered: {summary_prose(item.reason)} "
                    f"Evidence: {evidence_files(item.concern)}\n"
                )
        output.write("\nPlanning derives the GitHub effects from this record.\n")


def parse_args() -> argparse.Namespace:
    """Parse the stage directory, LLM execution log, and summary path."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--execution-file", type=Path, required=True)
    parser.add_argument("--github-step-summary", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    """Write ownership.json and the reviewable result record."""

    args = parse_args()
    try:
        intake = IntakeResult.from_json((args.output_dir / "intake.json").read_text())
        identity = intake.identity
        facts = intake.facts
        input_path = args.output_dir / "llm_input.json"
        llm_input = (
            LLMInput.from_json(input_path.read_text()) if facts.is_active else None
        )
    except Exception as exc:
        write_json(
            path=args.output_dir / "error.json",
            value={
                "error": str(exc),
                "stage": "ownership",
                "type": type(exc).__name__,
            },
        )
        print("Invalid stage input; no ownership result emitted", file=sys.stderr)
        return 1

    llm_result: LLMResult | None = None
    validation_errors: list[str] = []
    llm_metadata: dict[str, Any]
    if llm_input is None:
        llm_metadata = {"status": "skipped", "reason": "inactive_or_handled"}
        result = OwnershipResult.create(llm_run_status="skipped")
    else:
        try:
            llm_result, llm_metadata = load_action_execution(args.execution_file)
            validation_errors = validate_result(llm_input=llm_input, result=llm_result)
            result = build_ownership_result(
                llm_input=llm_input,
                llm_result=llm_result,
                llm_failed=bool(validation_errors),
            )
        except Exception as exc:
            write_json(
                path=args.output_dir / "error.json",
                value={
                    "error": str(exc),
                    "stage": "ownership",
                    "type": type(exc).__name__,
                },
            )
            llm_metadata = {
                "status": "failed",
                "type": type(exc).__name__,
                "error": str(exc),
            }
            result = build_ownership_result(llm_input=llm_input, llm_failed=True)

    result_record = {
        "target_repository": identity.repository,
        "ownership_result": result.to_dict(),
        "llm_result": llm_result.to_dict() if llm_result else None,
        "validation_errors": validation_errors,
        "llm_metadata": llm_metadata,
    }
    try:
        write_json(path=args.output_dir / "ownership.json", value=result.to_dict())
        write_json(path=args.output_dir / "result.json", value=result_record)
        write_github_step_summary(
            path=args.github_step_summary,
            intake=intake,
            result=result,
            validation_errors=validation_errors,
        )
    except Exception as exc:
        write_json(
            path=args.output_dir / "error.json",
            value={"error": str(exc), "stage": "publish", "type": type(exc).__name__},
        )
        print(
            f"{identity.repository}#{identity.number}: result publication failed",
            file=sys.stderr,
        )
        return 1

    log_analysis_record(
        {
            "ownership_result": result.to_dict(),
            "processing_error": llm_metadata.get("error"),
            "llm_result": llm_result.to_dict() if llm_result else None,
            "validation_errors": validation_errors,
        }
    )
    codepath_owners = ", ".join(result.codepath_owners) or "none"
    additional_owners = ", ".join(result.additional_owners) or "none"
    print(
        f"{identity.repository}#{identity.number}: "
        f"llm_run_status={result.llm_run_status}; "
        f"codepath owners={codepath_owners}; additional owners={additional_owners}; "
        f"validator_errors={len(validation_errors)}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
