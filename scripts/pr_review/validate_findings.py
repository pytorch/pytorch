"""Validate the findings file the review model writes, and say what is wrong.

The model writes its verdict to a JSON file instead of returning it as the
action's structured output, so that it can be checked WHILE the session is still
running and corrected. Two hooks call this: a PostToolUse hook after every Write
(advisory — the model reads the errors and rewrites) and a Stop hook (blocking —
the session cannot end on an invalid file).

THE POINT OF SHARING CODE WITH `extract_verdict.py`. Every check below runs
through that module's own `load_structured` / `parse_diff` / `build`, so the
answer the model is given is produced by the code that will actually publish.
A separate reimplementation would drift, and the direction it drifts in is the
bad one: the model is told its file is fine, and the sanitizer silently drops
findings anyway. That is exactly the failure this exists to end — the first live
run published two findings anchored to blank lines and dropped the most valuable
one, and nothing in the loop knew.

WHAT IT DOES NOT DO. It is not a security boundary and must never be treated as
one. It runs inside the model's own session, on a file the model controls, and a
prompt-injected model can simply not call it. The boundary is still the separate
`extract_verdict.py` step that runs after the action exits, plus the upstream
role scoping. This only makes the honest failure mode — a model that miscounts —
visible and fixable while there is still a turn left to fix it.

Exit 0 when the file would publish exactly what it says. Exit 1 otherwise, with
every problem on stderr in the order a reader would fix them.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from extract_verdict import (
    build,
    load_structured,
    parse_diff,
    Rejected,
    sanitize_findings,
)


MAX_REPORTED_LINES = 12


def _ranges(lines: set[int]) -> str:
    """Render a line set as compact ranges: {1,2,3,7} -> '1-3, 7'."""
    if not lines:
        return "(none — this file has no anchorable lines)"
    out: list[str] = []
    ordered = sorted(lines)
    start = prev = ordered[0]
    for n in ordered[1:]:
        if n == prev + 1:
            prev = n
            continue
        out.append(f"{start}-{prev}" if start != prev else f"{start}")
        start = prev = n
    out.append(f"{start}-{prev}" if start != prev else f"{start}")
    if len(out) > MAX_REPORTED_LINES:
        return ", ".join(out[:MAX_REPORTED_LINES]) + f", … ({len(out)} ranges total)"
    return ", ".join(out)


def _report_drops(dropped: list[dict], touched: dict[str, set[int]]) -> None:
    """Explain each dropped finding, and for an anchor miss say what WOULD work."""
    print(
        f"{len(dropped)} finding(s) would be DISCARDED before publication:",
        file=sys.stderr,
    )
    for d in dropped:
        path = d.get("path")
        reason = d.get("reason", "unknown")
        where = f"{path}:{d['line']}" if d.get("line") is not None else str(path)
        print(f"  - {where} — {reason}", file=sys.stderr)
        if reason == "line_not_in_diff" and path in touched:
            print(
                f"      `line` must be a line number in the file at the head commit.\n"
                f"      Lines this PR touches in {path}: {_ranges(touched[path])}",
                file=sys.stderr,
            )
        elif reason == "path_not_in_diff":
            known = ", ".join(sorted(touched)) or "(the diff touches no files)"
            print(f"      Files this PR changed: {known}", file=sys.stderr)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--findings-file", required=True)
    ap.add_argument("--diff-file", required=True)
    args = ap.parse_args()

    findings_path = Path(args.findings_file)
    if not findings_path.exists():
        print(f"INVALID: {findings_path} does not exist yet.", file=sys.stderr)
        print(
            "Write your verdict there with the Write tool before finishing.",
            file=sys.stderr,
        )
        return 1

    obj, problem = load_structured(findings_path)
    if obj is None:
        status, _, detail = problem.partition(":")
        print(f"INVALID ({status}): {detail}", file=sys.stderr)
        return 1

    try:
        diff_text = Path(args.diff_file).read_bytes().decode("utf-8", "replace")
    except OSError as exc:
        # The diff is workflow-provided. If it is unreadable the model cannot
        # act on anything reported here, so say so rather than blaming the file
        # the model just wrote — and do NOT fall back to an empty diff, which
        # would report every finding as unanchorable.
        print(
            f"CANNOT VALIDATE: the diff at {args.diff_file} is unreadable: {exc}",
            file=sys.stderr,
        )
        print(
            "This is a workflow problem, not a problem with your file.", file=sys.stderr
        )
        return 1

    touched = parse_diff(diff_text)

    # ANCHORING IS REPORTED BEFORE ANYTHING ELSE, and the ordering is the point.
    # `build()` raises `Rejected` when a `changes_requested` verdict is left with
    # no publishable finding — which is exactly what happens when the model's
    # only finding is mis-anchored. Running `build()` first therefore answers the
    # single most common failure with "the whole review is discarded" and not one
    # word about WHICH line was wrong or which would work. Draining the drops
    # first turns that back into an actionable message.
    raw = obj.get("findings")
    if isinstance(raw, list):
        try:
            _, dropped_first = sanitize_findings(raw, touched)
        except (ValueError, TypeError):
            dropped_first = []  # a shape problem; `build()` below names it properly
        if dropped_first:
            _report_drops(dropped_first, touched)
            print(
                "Fix these and write the file again. A discarded finding is not "
                "published at all — it is not downgraded, it is lost.",
                file=sys.stderr,
            )
            return 1

    try:
        result = build(obj, touched)
    except Rejected as exc:
        print(f"INVALID (sanitizer_rejected): {exc}", file=sys.stderr)
        print(
            "The WHOLE review is discarded when this fires, not one finding.",
            file=sys.stderr,
        )
        return 1
    except (ValueError, TypeError) as exc:
        print(f"INVALID (schema_invalid): {exc}", file=sys.stderr)
        return 1
    except RecursionError:
        print(
            "INVALID (schema_invalid): the file nests too deeply to parse.",
            file=sys.stderr,
        )
        return 1

    dropped = result.get("dropped_detail") or []
    if dropped:
        _report_drops(dropped, touched)
        print(
            "Fix these and write the file again. A discarded finding is not "
            "published at all — it is not downgraded, it is lost.",
            file=sys.stderr,
        )
        return 1

    kept = len(result.get("findings", []))
    print(
        f"VALID: verdict={result.get('verdict')}, {kept} finding(s), none discarded.",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
