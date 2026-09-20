"""Validate the findings file the review model writes, and say what is wrong.

The model writes its verdict to a JSON file instead of returning it as the
action's structured output, so that it can be checked WHILE the session is still
running and corrected. Two hooks call this: a PostToolUse hook after every Write
(advisory — the model reads the errors and rewrites) and a Stop hook, which
blocks ONCE. Claude Code sets `stop_hook_active` on the retry, and
`validate-on-stop.sh` lets that second stop through, so the model gets one
forced turn to fix the file rather than being held until it succeeds. That is
deliberate: an unfixable file would otherwise loop the session to its limit.

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

Exit 0 when every finding in the file SURVIVES to publication — anchored to a
line the PR touched, with a severity and a path the sanitizer accepts. Exit 1
otherwise, with every problem on stderr in the order a reader would fix them.

A pass is about WHICH findings publish, not about their text arriving character
for character. `neutralize` runs on the verdict summary (capped at
`MAX_SUMMARY`) and on every finding's `message` (capped at `MAX_MESSAGE`), and
in both it silently drops out-of-charset codepoints and backslashes. That is by
design: each is a bounded rewrite of one field, none loses a finding, and the
alternative to dropping a stray em dash was discarding the whole review. So a
passing file can publish a shortened summary or a shortened finding message; it
cannot publish fewer findings than it names. A message that neutralizes to
nothing IS a lost finding, and is reported as `message_empty_after_neutralize`.

A kept finding's `path` is rewritten too, by `neutralize_path`, which
backslash-escapes the markdown-active characters — so `torch/_dynamo/x.py`
publishes as ``torch/\\_dynamo/x.py``. Same file, different text, and in this
repository it fires on most findings.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from extract_verdict import (
    build,
    load_structured,
    MAX_PATH,
    neutralize,
    parse_diff,
    Rejected,
    sanitize_findings,
)


MAX_REPORTED_LINES = 12
MAX_REPORTED_KEY = 60
# Everything this module prints to stderr is folded into the model's
# `additionalContext` by `validate-post-write.sh`, which that script's header
# identifies as a system message. So the question for every line below is: how
# much text of a fork contributor's choosing does it carry?
#
# NO CHANGED-FILE LIST. The diff's `+++` paths are chosen by the pull request
# author, `_is_repo_path` admits any printable ASCII except ``|`\<>&`` — spaces,
# colons, quotes and whole sentences are legal, up to MAX_PATH each — and the
# report used to print up to twenty of them. The model is pointed at the file
# the workflow already wrote instead; see `_report_drops`.
DEFAULT_FILES_FILE = "/tmp/pr-files.txt"
# WHAT STILL GETS THROUGH, stated because "none" would be wrong. Each dropped
# record's `path` is printed. It is the MODEL's own text — it named that file
# in its own findings file, and `sanitize_findings` has already neutralized it
# — but the model chose it after reading the pull request, so its CONTENT is
# not independent of the pull request. Note the path need not even NAME a real
# file: a `path_not_in_diff` drop echoes back whatever string the model wrote,
# so this is a round trip for arbitrary text, not only for filenames. It stays
# because it is the only way to say WHICH finding was discarded. Bounded to one
# name per discarded finding, MAX_PATH each, and `drop()` tracks up to 200
# records, so the COUNT is capped here too.
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


def _sanitized_path(path: object) -> str:
    """Render a path that `sanitize_findings` has already neutralized.

    Nothing to escape — the publisher did it — so this only bounds the length,
    since a dropped record is diagnostic and never passed the publisher's caps.
    """
    return str(path)[:MAX_PATH]


def _where_the_file_list_is(
    files_file: str, diff_file: str, touched: dict[str, set[int]]
) -> str:
    """Where to READ the changed-file set, rather than the set itself.

    An anchor miss means the model named a file the diff does not touch, so it
    needs the real set to correct itself. Naming the file it is already granted
    `Read` on does that without putting a fork contributor's choice of file
    names into a system message, which is what printing them did — capped and
    escaped, but still by design.

    Both paths here are workflow-authored, at the same trust level as this
    script. They are length-bounded anyway, so a mis-set variable degrades the
    sentence rather than the channel.
    """
    # NO ANCHORABLE LINE ANYWHERE. "Go and read the list" is then advice the
    # model cannot act on — it would fetch the file and come back no better
    # off. The old changed-file list said this for free by rendering as
    # "(the diff touches no files)"; a pointer has to say it deliberately.
    #
    # `any(...)`, not `if not touched`: a hunk that only REMOVES lines yields
    # `{"x.py": set()}` — a key with nothing anchorable under it — so the
    # mapping is non-empty while the answer to "can anything be anchored" is
    # still no. Measured against `parse_diff`.
    #
    # AND IT IS NOT AUTOMATICALLY A WORKFLOW FAULT, which an earlier draft of
    # this sentence asserted. A pull request of pure renames, deletions,
    # mode changes or binary files legitimately anchors nothing. The model's
    # finding is still the model's to withdraw either way, so say what was
    # observed and what to do, and do not assign blame the validator cannot
    # establish.
    if not any(touched.values()):
        return (
            "This diff has no anchorable lines at all — a pull request of pure "
            "renames, deletions, mode changes or binary files looks like this. "
            "No finding can be anchored, so remove the ones that cannot be."
        )
    # THE LIST IS A SUPERSET OF THE ANCHORABLE SET, and saying otherwise sends
    # the model round a loop it cannot exit. The workflow builds that file with
    # `git diff --name-only`, which names renames, deletions, binary files and
    # mode-only changes; `parse_diff` admits none of them, because none has a
    # new-side hunk line to anchor to (its own docstring says so). So "anchor
    # to one of those paths" was advice that is FALSE for some entries: the
    # model re-anchors to a deleted file, is dropped again, and — with no
    # shell to tell the two kinds of entry apart — repeats until `--max-turns`
    # ends the session with no valid file at all. The rule has to travel with
    # the pointer.
    # WHAT "CANNOT BE ANCHORED" COVERS, and what to do about it. Two earlier
    # drafts of this sentence were each wrong in their own way:
    #
    #   * the first listed only deleted/renamed/mode-only/binary files. A file
    #     whose hunks ONLY REMOVE lines is a fifth case — `parse_diff` returns
    #     `{"x.py": set()}` for it, a key with an empty set — so it is present
    #     in every enumeration and still anchors nothing.
    #   * the second said such a finding "must be REMOVED, not re-anchored".
    #     Too strong, and it loses real findings: a defect in a module this PR
    #     DELETED is often still reportable at the line that now imports it.
    #     Unanchorable FILE does not mean unanchorable FINDING.
    # POSITIVE RULE FIRST, examples second. An enumeration invites the reader
    # to treat it as exhaustive and it is not — a newly added EMPTY file is a
    # sixth case with no new-side hunk line, and there is no reason to think
    # that is the last one. State what DOES anchor, then illustrate.
    rule = (
        "Only an added or unchanged line inside a hunk can take an anchor, so "
        "not every entry can be anchored — a file this PR only deleted, "
        "renamed, changed the mode of, or that is binary, and a file whose "
        "hunks only remove lines, have no such line. Re-anchor the finding to "
        "a line that does locate the defect; remove it only if no such line "
        "exists."
    )
    if Path(files_file).is_file():
        return (
            f"Read {str(files_file)[:MAX_PATH]} for the files this PR changed. {rule}"
        )
    # NEVER send the model to a file that is not there: it cannot run a shell,
    # so a dead pointer would leave it no way to enumerate the changed files at
    # all. The diff is the better fallback than it looks — `parse_diff` reads
    # the diff and nothing else, so a path there WITH a hunk is exactly an
    # anchorable path — but the same four kinds of entry appear in it without
    # one, so the rule above still applies.
    # "a path that has a hunk" was wrong for the same fifth case above: a
    # removal-only hunk is a hunk. What anchors is a NEW-SIDE line, which is
    # what `parse_diff` collects and the only thing a `line` can refer to.
    return (
        f"The files this PR changed are named in the diff at "
        f"{str(diff_file)[:MAX_PATH]}; anchor to a path whose hunk there "
        f"contains an added or unchanged line. {rule}"
    )


def _report_drops(
    dropped: list[dict],
    touched: dict[str, set[int]],
    files_file: str = DEFAULT_FILES_FILE,
    diff_file: str = "/tmp/pr-diff.txt",
) -> None:
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
            # NO FILE NAME HERE. The line above already identifies the finding,
            # and it does so with the MODEL's own `path` — so repeating the raw
            # diff key would add a second, differently-escaped rendering of a
            # name a fork contributor chose, for no information at all. Which
            # file it is about was never in doubt.
            print(
                f"      `line` must be a line number in the file at the head "
                f"commit.\n"
                f"      Lines this PR touches in that file: "
                f"{_ranges(touched[path])}",
                file=sys.stderr,
            )
        elif reason == "path_not_in_diff":
            anchor_misses = True
    # ONCE, not once per drop — and a POINTER, not the names. Printing the list
    # inside the loop multiplied it by the drop count: 100 names at MAX_PATH
    # with 20 drops measured 161KB of advisory output. Capping the list fixed
    # the size; pointing at the file is what addresses the actual finding
    # (#196844 — `additionalContext` is a system message, and the names in it
    # come from the pull request).
    if anchor_misses:
        print(
            f"      {_where_the_file_list_is(files_file, diff_file, touched)}",
            file=sys.stderr,
        )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--findings-file", required=True)
    ap.add_argument("--diff-file", required=True)
    # Not required: an anchor-miss report names this path, and the default is
    # what hardened-pr-review-run.yml writes, so a hand-run still says
    # something true. `_where_the_file_list_is` checks it before pointing there.
    ap.add_argument("--files-file", default=DEFAULT_FILES_FILE)
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
            _report_drops(dropped_first, touched, args.files_file, args.diff_file)
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
        _report_drops(dropped, touched, args.files_file, args.diff_file)
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
