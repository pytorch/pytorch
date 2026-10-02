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
table. Its one key today is ``findings``: the sanitized findings of a succeeded
review as a JSON array, which Dr.CI renders under the verdict.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

from extract_verdict import (
    _is_repo_path,
    _ISSUE_REF,
    _MENTION,
    _URL,
    _XREF,
    MAX_FINDINGS,
    MAX_LINE,
    MAX_MESSAGE,
    MAX_PATH,
    MAX_SUMMARY,
    neutralize_path,
    SEVERITIES,
    VERDICTS,
)


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
# PR code, so the strings it publishes are re-checked here against what
# extract_verdict.py's neutralize() and neutralize_path() can emit, rather than
# trusted to have come through them.
_PUBLISHED_CHARS = re.compile(r"[\x09\x0a\x20-\x7e]*")
_FINDING_KEYS = ("path", "line", "severity", "message")
# neutralize() strips every backslash, then adds them only before `[`, `]` and
# the `#` of a defused reference; neutralize_path() only before `[]()@#*_`.
_STRAY_BACKSLASH = re.compile(r"\\(?![\[\]#])")
_UNESCAPED_BRACKET = re.compile(r"(?<!\\)[\[\]]")
# neutralize() caps AFTER escaping, so a capped string can end in a cut entity.
_BARE_AMPERSAND = re.compile(r"&(?!amp;|lt;|gt;|(?:a(?:mp?)?|lt?|gt?)?\Z)")
_PATH_ESCAPE = re.compile(r"\\([\[\]()@#*_])")

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


def _published_text(value: object, cap: int) -> bool:
    return (
        isinstance(value, str)
        and 0 < len(value) <= cap
        and _PUBLISHED_CHARS.fullmatch(value) is not None
    )


def neutral_prose(value: object, cap: int) -> bool:
    """Whether `value` is text neutralize() could have produced."""
    return (
        _published_text(value, cap)
        and "<" not in value
        and ">" not in value
        and not _BARE_AMPERSAND.search(value)
        and not _STRAY_BACKSLASH.search(value)
        and not _UNESCAPED_BRACKET.search(value)
        and not any(p.search(value) for p in (_URL, _XREF, _MENTION, _ISSUE_REF))
    )


def neutral_path(value: object) -> bool:
    """Whether `value` is neutralize_path() of a valid repo-relative path."""
    if not _published_text(value, MAX_PATH):
        return False
    raw = _PATH_ESCAPE.sub(r"\1", value)
    return _is_repo_path(raw) and neutralize_path(raw) == value


def published_findings(verdict: dict) -> list[dict]:
    """The findings Dr.CI may render, in extract_verdict.py's output shape.

    A finding that does not match that shape is dropped, not repaired, and the
    list is capped at the sanitizer's own bound.
    """
    raw = verdict.get("findings")
    if not isinstance(raw, list):
        return []
    kept: list[dict] = []
    for item in raw:
        if len(kept) >= MAX_FINDINGS:
            break
        if not isinstance(item, dict) or set(item) != set(_FINDING_KEYS):
            continue
        line = item["line"]
        if (
            not neutral_path(item["path"])
            or not neutral_prose(item["message"], MAX_MESSAGE)
            or not isinstance(item["severity"], str)
            or item["severity"] not in SEVERITIES
            or not isinstance(line, int)
            or isinstance(line, bool)
            or not 0 < line <= MAX_LINE
        ):
            continue
        kept.append({key: item[key] for key in _FINDING_KEYS})
    return kept


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
        "input_tokens": as_int(str(usage.get("input_tokens", 0))),
        "output_tokens": as_int(str(usage.get("output_tokens", 0))),
        "cache_read_input_tokens": as_int(str(usage.get("cache_read_input_tokens", 0))),
        "cache_creation_input_tokens": as_int(
            str(usage.get("cache_creation_input_tokens", 0))
        ),
        # Alphabetically first when several models appear, which is arbitrary
        # but deterministic. Practically there is one: the review makes a single
        # model call. If sub-agents ever land, this needs a real rule.
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
            or not neutral_prose(verdict.get("summary"), MAX_SUMMARY)
        ):
            # A verdict extract_verdict.py could not have written. Record the
            # run without publishing anything it says.
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
            if len(findings) != row["findings_count"]:
                print(
                    f"::warning::{row['findings_count'] - len(findings)} finding(s)"
                    " failed the publish-side re-check and were not recorded",
                    file=sys.stderr,
                )
        row["findings_dropped"] = as_int(str(verdict.get("findings_dropped", 0)))
        row["failure_detail"] = verdict.get("failure_detail", "")
        row["reasoning_uri"] = env("REASONING_URI")
        row.update(usage_metrics(args.usage_file))

    row["extra"] = {}
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
