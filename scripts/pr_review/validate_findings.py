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
    _is_repo_path,
    build,
    load_structured,
    MAX_PATH,
    neutralize,
    neutralize_path,
    parse_diff,
    Rejected,
    sanitize_findings,
)


MAX_REPORTED_LINES = 12
# Everything this module prints to stderr is folded into the model's
# `additionalContext` by `validate-post-write.sh`, which that script's header
# identifies as a system message. The file names come from the PR's own diff, so
# a fork contributor chooses them: `_is_repo_path` admits any printable ASCII
# except `|`, backtick, `\`, `<`, `>` and `&`, which leaves spaces, colons,
# quotes and whole sentences legal, up to MAX_PATH each. Bounding the COUNT is
# what keeps a hundred 400-character names from arriving as 40KB of
# attacker-chosen prose wearing trusted framing.
MAX_REPORTED_FILES = 20
MAX_REPORTED_KEY = 60
# The same argument for the other direction. A dropped record's `path` is what
# the MODEL wrote, already neutralized by `sanitize_findings` but bounded only
# per-name: `drop()` tracks up to 200 of them, so an injected model emitting
# junk findings could still push ~80KB back through the same channel.
MAX_REPORTED_DROPS = 20


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


def _safe_path(path: object) -> str:
    """Render one RAW diff path for the system-message channel.

    For keys of `touched` only. Those come straight off the diff's `+++` lines,
    so they are PR-authored and have had nothing done to them beyond
    `_is_repo_path`. `neutralize_path` rather than `neutralize`, deliberately:
    the model is being told which file to anchor to, so the name has to stay
    exactly recoverable, and escaping preserves that where rewriting would not.

    NOT for a dropped record's `path`. Those have already been through
    `sanitize_findings`, so they arrive escaped — `pkg/[click].py` is already
    `pkg/\\[click\\].py` — and `_is_repo_path` refuses a backslash, so running
    them through here would report a perfectly publishable name as unusable.
    """
    text = str(path)
    if not _is_repo_path(text):
        return "(not a usable repo path)"
    return neutralize_path(text)


def _sanitized_path(path: object) -> str:
    """Render a path that `sanitize_findings` has already neutralized.

    Nothing to escape — the publisher did it — so this only bounds the length,
    since a dropped record is diagnostic and never passed the publisher's caps.
    """
    return str(path)[:MAX_PATH]


def _safe_file_list(touched: dict[str, set[int]]) -> str:
    """The changed-file list, bounded in count as well as in per-name length."""
    names = sorted(touched)
    if not names:
        return "(the diff touches no files)"
    shown = ", ".join(_safe_path(n) for n in names[:MAX_REPORTED_FILES])
    if len(names) > MAX_REPORTED_FILES:
        return f"{shown}, … ({len(names)} files total)"
    return shown


def _report_drops(dropped: list[dict], touched: dict[str, set[int]]) -> None:
    """Explain each dropped finding, and for an anchor miss say what WOULD work."""
    print(
        f"{len(dropped)} finding(s) would be DISCARDED before publication:",
        file=sys.stderr,
    )
    if len(dropped) > MAX_REPORTED_DROPS:
        print(
            f"  (showing the first {MAX_REPORTED_DROPS}; fixing those usually "
            f"fixes the rest)",
            file=sys.stderr,
        )
    anchor_misses = False
    for d in dropped[:MAX_REPORTED_DROPS]:
        path = d.get("path")
        reason = d.get("reason", "unknown")
        if path is None:
            where = "(finding with no usable path)"
        elif d.get("line") is not None:
            where = f"{_sanitized_path(path)}:{d['line']}"
        else:
            where = _sanitized_path(path)
        # `keys` names which fields were rejected; without it the model is
        # told only "unexpected_keys" and cannot tell which one to drop. They
        # are keys from the model's own file, so they get the prose treatment
        # rather than the path one.
        keys = d.get("keys")
        detail = (
            f" ({', '.join(neutralize(str(k), MAX_REPORTED_KEY) for k in keys)})"
            if keys
            else ""
        )
        print(f"  - {where} — {reason}{detail}", file=sys.stderr)
        if reason == "line_not_in_diff" and path in touched:
            print(
                f"      `line` must be a line number in the file at the head commit.\n"
                # `path in touched` is the guard: it is a raw diff key here, so
                # `_safe_path` is the right one and cannot double-escape.
                f"      Lines this PR touches in {_safe_path(path)}: "
                f"{_ranges(touched[path])}",
                file=sys.stderr,
            )
        elif reason == "path_not_in_diff":
            anchor_misses = True
    # ONCE, not once per drop. Printing it inside the loop multiplied the capped
    # list by the drop count: 100 names at MAX_PATH with 20 drops measured 161KB
    # of advisory output, which defeats the per-item caps above.
    if anchor_misses:
        print(
            f"      Files this PR changed: {_safe_file_list(touched)}",
            file=sys.stderr,
        )


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
            _, dropped_first, _ = sanitize_findings(raw, touched)
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
