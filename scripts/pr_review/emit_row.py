#!/usr/bin/env python3
"""Build the ClickHouse telemetry row for one hardened PR review attempt.

Two rows are written per attempt, and the pair is what makes the record
readable:

  * ``started`` — written by the prepare job before any model call. It exists so
    that a run which is cancelled, times out, or dies on the runner still leaves
    a trace.
  * a terminal row — usually written by the publish job under ``if: always()``,
    carrying the real outcome. The prepare job writes one itself when it finds
    the request superseded, since publish never runs in that case.

That gives three distinguishable states, which a single row cannot express:

  | rows present            | meaning                                        |
  |-------------------------|------------------------------------------------|
  | none                    | no review was ever triggered, or prepare failed|
  | started, no terminal    | cancelled, superseded, or the runner died —    |
  |                         | reconcilable from the Actions API              |
  | started + terminal      | the attempt ran; `status` says how it ended    |

``verdict`` is only ever populated when ``status == 'succeeded'``. A failed
review must never surface as ``changes_requested``, because a reader cannot tell
a real objection from an infrastructure hiccup.

The empty row is the one a consumer must render explicitly. Stage 1's workflow
file comes from the PR's own ref, so a PR author can rename its ``name:``, leave
``workflow_run`` with nothing to match, and produce no review and no rows at all.
A surface that shows only what it finds renders that suppression as "clean".

The row is deliberately keyed and versioned for reuse: the interim GitHub
Actions harness and the credential-less sandbox that replaces it write the SAME
shape, distinguished by ``harness``, so the two eras stay comparable. ``extra``
is the escape hatch for anything we learn we need later without re-cutting the
table. Two keys today, both only on a succeeded review: ``findings``, the
sanitized findings as a JSON array, which Dr.CI renders under the verdict; and
``findings_dropped_at_publish``, present only when the publish-side re-check
dropped some, so ``findings_count`` can exceed the array's length.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

from extract_verdict import is_neutral_prose, MAX_SUMMARY, published_findings, VERDICTS


SCHEMA_VERSION = 1

# `model` is the one string in this row that we do not construct ourselves. It
# comes from the usage file's modelUsage keys, which the review job extracts
# with jq from the action's execution log — in a job that reads untrusted PR
# code. Influence over it is marginal, but every other string crossing this
# boundary is validated, and an unvalidated one reaching a ClickHouse row is
# worth less than the two lines it costs to bound. Out-of-charset or over-length
# becomes empty rather than failing: telemetry never breaks a review.
_MODEL_CHARS = re.compile(r"^[A-Za-z0-9._:-]{1,128}$")

# verdict.json reaches this job as an artifact of the job that read untrusted
# PR code, so every string it puts in the row — verdict, summary, findings and
# failure_detail — is re-checked here with extract_verdict.py's own predicates
# rather than trusted to have come through that module. `build` holds its
# output to the same predicates, so validate_findings.py sees what this drops.
_DETAIL_REJECTED = "failure_detail failed the publish-side re-check"

# Terminal statuses. Anything not in this set is a bug in the caller.
TERMINAL = {
    "succeeded",
    "schema_invalid",
    "sanitizer_rejected",
    "model_error",
    # RESERVED, and unreachable from the GitHub Actions harness: a step timeout
    # surfaces as outcome `failure`, which `extract_verdict.downgrade()` maps to
    # model_error. It is kept for the sandbox harness, which can tell the two
    # apart. Nothing may read its absence as "reviews never time out".
    "timed_out",
    "skipped_stale",
    "skipped_too_large",
    "blocked",
}


def env(name: str, default: str = "") -> str:
    return (os.environ.get(name) or default).strip()


def as_int(value: str, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def as_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value or 0)
    except (TypeError, ValueError):
        return default


def safe_model(value: object) -> str:
    # fullmatch, not match: `$` also matches before a final newline, so
    # `.match` accepted "claude\n" and a 128-char name plus one at 129.
    text = value if isinstance(value, str) else ""
    return text if _MODEL_CHARS.fullmatch(text) else ""


def safe_detail(value: object) -> str:
    """`failure_detail` if it is text the sanitizer could have written."""
    if value in (None, ""):
        return ""
    return value if is_neutral_prose(value, MAX_SUMMARY) else _DETAIL_REJECTED


def base_row(phase: str) -> dict:
    """Fields common to both rows — the identity of the attempt."""
    return {
        "schema_version": SCHEMA_VERSION,
        "phase": phase,
        # Which execution substrate produced this row. The sandbox will write
        # 'sandbox' here; without it the two eras' data cannot be compared,
        # which is the whole reason this table outlives the interim workflow.
        "harness": env("HARNESS", "gha-interim"),
        "harness_version": env("HARNESS_VERSION"),
        "repo": env("REPO"),
        "pr_number": as_int(env("PR_NUMBER")),
        "head_sha": env("HEAD_SHA"),
        "base_sha": env("BASE_SHA"),
        "is_fork": env("IS_FORK", "false") == "true",
        "trigger_event": env("TRIGGER_EVENT"),
        "trigger_label": env("TRIGGER_LABEL"),
        "trigger_run_id": as_int(env("TRIGGER_RUN_ID")),
        "review_run_id": as_int(env("GITHUB_RUN_ID")),
        "review_run_attempt": as_int(env("GITHUB_RUN_ATTEMPT"), 1),
        # Content hash of the trusted prompt + skill files, not a hand-maintained
        # version string: those rot within a week of the first hotfix.
        "prompt_hash": env("PROMPT_HASH"),
        "trusted_sha": env("TRUSTED_SHA"),
        "timestamp": env("ROW_TIMESTAMP"),
    }


def read_json(path: str) -> dict:
    if not path:
        return {}
    try:
        loaded = json.loads(Path(path).read_text())
    except (OSError, ValueError) as exc:
        print(f"warning: could not read {path}: {exc}", file=sys.stderr)
        return {}
    return loaded if isinstance(loaded, dict) else {}


# (row column, modelUsage field) for each token count.
_TOKEN_FIELDS = (
    ("input_tokens", "inputTokens"),
    ("output_tokens", "outputTokens"),
    ("cache_read_input_tokens", "cacheReadInputTokens"),
    ("cache_creation_input_tokens", "cacheCreationInputTokens"),
)


def token_counts(usage: dict, model_usage: object) -> dict:
    """Token columns, summed over every model entry in `modelUsage`.

    The result's `usage` covers only the top-level session; sub-agent tokens
    appear only in `modelUsage`. Checked on Claude Code 2.1.280: with one
    sub-agent, `usage` had 156 output tokens and `modelUsage` 2730, matching
    `total_cost_usd`; without sub-agents the two agree, so rows stay comparable.
    `usage` is the fallback when `modelUsage` carries no token fields.
    """
    entries = (
        [e for e in model_usage.values() if isinstance(e, dict)]
        if isinstance(model_usage, dict)
        else []
    )
    if any(field in e for e in entries for _, field in _TOKEN_FIELDS):
        return {
            column: sum(as_int(str(e.get(field, 0))) for e in entries)
            for column, field in _TOKEN_FIELDS
        }
    return {column: as_int(str(usage.get(column, 0))) for column, _ in _TOKEN_FIELDS}


def usage_metrics(path: str) -> dict:
    """Pull cost/latency from the usage file, if present.

    Accepts BOTH shapes on purpose:
      * a bare result object — what the review job extracts and hands over in
        the artifact, so that only numbers cross a world-readable boundary;
      * claude-code-action's full execution file (a list of turns) — still the
        shape if this is ever pointed at the raw file directly.

    Best-effort by design: telemetry must never be the reason a review fails.
    """
    try:
        loaded = json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return {}
    if isinstance(loaded, dict):
        last = loaded
    elif isinstance(loaded, list):
        results = [
            t for t in loaded if isinstance(t, dict) and t.get("type") == "result"
        ]
        if not results:
            return {}
        last = results[-1]
    else:
        return {}
    if not last:
        return {}
    usage = last.get("usage") if isinstance(last.get("usage"), dict) else {}
    model_usage = last.get("modelUsage")
    return {
        "duration_ms": as_int(str(last.get("duration_ms", 0))),
        "num_turns": as_int(str(last.get("num_turns", 0))),
        # as_float, not float(): this line sits OUTSIDE the try above, so a
        # non-numeric cost would raise and take the whole row down — breaking
        # the "telemetry must never be the reason a review fails" promise in
        # this function's own docstring.
        "total_cost_usd": as_float(last.get("total_cost_usd")),
        **token_counts(usage, model_usage),
        # Alphabetically first when several models appear, which is arbitrary
        # but deterministic. Practically there is one: the workflow forces
        # sub-agents onto the review model (CLAUDE_CODE_SUBAGENT_MODEL_FORCE).
        "model": safe_model(
            sorted(model_usage)[0]
            if isinstance(model_usage, dict) and model_usage
            else ""
        ),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True, choices=["started", "terminal"])
    ap.add_argument(
        "--verdict-file", default="", help="verdict.json from extract_verdict.py"
    )
    ap.add_argument(
        "--usage-file", default="", help="claude-code-action execution file"
    )
    ap.add_argument(
        "--status",
        default="",
        help="override the terminal status (e.g. skipped_stale) when no verdict file exists",
    )
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    row = base_row(args.phase)
    row["extra"] = {}
    findings: list[dict] = []

    if args.phase == "started":
        row["status"] = "started"
        row["verdict"] = None
    else:
        verdict = read_json(args.verdict_file)
        status = args.status or verdict.get("status") or "model_error"
        if status not in TERMINAL:
            print(
                f"warning: unknown terminal status {status!r}, recording as model_error",
                file=sys.stderr,
            )
            status = "model_error"
        row["status"] = status
        # Only a succeeded review carries a verdict. A failed one must not look
        # like an objection to the change.
        if status == "succeeded" and (
            not isinstance(verdict.get("verdict"), str)
            or verdict.get("verdict") not in VERDICTS
            or not is_neutral_prose(verdict.get("summary"), MAX_SUMMARY)
        ):
            # A verdict extract_verdict.py could not have written. Record the
            # run without publishing anything it says. The workflow reads the
            # status and verdict back out of this row, so the label step
            # follows this decision.
            print(
                "::warning::verdict.json failed the publish-side re-check",
                file=sys.stderr,
            )
            status = "sanitizer_rejected"
            row["status"] = status
            verdict = {"failure_detail": "verdict failed the publish-side re-check"}
        row["verdict"] = verdict.get("verdict") if status == "succeeded" else None
        row["summary"] = verdict.get("summary", "") if status == "succeeded" else ""
        raw_findings = verdict.get("findings")
        row["findings_count"] = (
            len(raw_findings) if isinstance(raw_findings, list) else 0
        )
        if status == "succeeded":
            try:
                findings = published_findings(verdict)
            except Exception as exc:  # noqa: BLE001 - telemetry never breaks a review
                print(f"warning: findings not recorded: {exc!r}", file=sys.stderr)
            dropped_at_publish = row["findings_count"] - len(findings)
            if dropped_at_publish:
                print(
                    f"::warning::{dropped_at_publish} finding(s) failed the"
                    " publish-side re-check and were not recorded",
                    file=sys.stderr,
                )
                row["extra"]["findings_dropped_at_publish"] = str(dropped_at_publish)
        row["findings_dropped"] = as_int(str(verdict.get("findings_dropped", 0)))
        row["failure_detail"] = safe_detail(verdict.get("failure_detail"))
        row["reasoning_uri"] = env("REASONING_URI")
        row.update(usage_metrics(args.usage_file))

    if findings:
        row["extra"]["findings"] = json.dumps(
            findings, ensure_ascii=True, separators=(",", ":")
        )

    Path(args.out).write_text(json.dumps(row, ensure_ascii=True) + "\n")
    print(
        f"{args.phase} row: status={row['status']} verdict={row.get('verdict')}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
