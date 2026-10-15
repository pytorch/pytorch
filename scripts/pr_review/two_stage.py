#!/usr/bin/env python3
"""Trusted glue between the two stages of the hardened PR review.

Stage A runs the pr-review skill the way an interactive `/pr-review` does and
ends with a markdown review whose last line is `Recommendation: Approve`,
`Recommendation: Request Changes` or `Recommendation: Needs Discussion`. Stage
B copies that review's findings into the findings file the sanitizer reads. The VERDICT is not stage B's to choose: this script derives
it from stage A's Recommendation, so the automated review agrees with
pr-review by construction.

  collect-guides  copy the REVIEW.md guides that cover the changed paths from
                  the merge base, so a PR cannot rewrite its own rules, and
                  list them in INDEX.md beside them
  extract-review  pull the review out of stage A's execution file, write it for
                  stage B, and print the verdict it maps to
  pin-verdict     after stage B, overwrite the findings file's verdict with the
                  mapped one; on a ready verdict, demote `major` to `minor`

Every input here except the merge-base SHA is model- or PR-influenced. None of
it is executed, and the only decision taken from it is the three-way mapping
below, which an injected stage A could already steer by writing "Approve".
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

# The sanitizer's own loader: pinning must accept exactly the files the
# sanitizer accepts, or one it skips could still be published.
from extract_verdict import load_structured


# pr-review's three recommendations and the verdict each maps to. Needs
# Discussion asks for a human, which is what `ready_for_human_review` means.
VERDICT_FOR = {
    "Approve": "ready_for_human_review",
    "Needs Discussion": "ready_for_human_review",
    "Request Changes": "changes_requested",
}
# Stage A is told to end its final message with exactly this line, and only
# that line is read: quoted PR text, earlier drafts and the review's own prose
# can say anything. Emphasis around it is tolerated, nothing else.
_TERMINAL = re.compile(
    r"[*_`]*Recommendation[*_`]*:[*_` ]*"
    r"(Approve|Request Changes|Needs Discussion)[*_`.]*"
)
MIN_REVIEW_CHARS = 200
MAX_REVIEW_BYTES = 512 * 1024
# Written beside the guides. Never a guide's own name: guides are REVIEW.md.
INDEX_NAME = "INDEX.md"


class NoVerdict(ValueError):
    pass


def recommendation(review: str) -> str:
    """The recommendation on the review's last non-blank line, which must be
    `Recommendation: <choice>`; anything else raises NoVerdict."""
    lines = [ln.strip() for ln in review.split("\n") if ln.strip()]
    m = _TERMINAL.fullmatch(lines[-1]) if lines else None
    if not m:
        raise NoVerdict("the review does not end with a `Recommendation: ...` line")
    return m.group(1)


def review_text(execution: list) -> str:
    """The review stage A published: its final result. Never an earlier
    message, which a later turn may have withdrawn."""
    results = [
        t for t in execution if isinstance(t, dict) and t.get("type") == "result"
    ]
    text = results[-1].get("result") if results else None
    return text if isinstance(text, str) else ""


def extract_review(execution_file: Path, review_out: Path) -> tuple[str, str]:
    """Return (recommendation, verdict); write the review for stage B."""
    raw = execution_file.read_bytes()
    try:
        execution = json.loads(raw)
    except ValueError as exc:
        raise NoVerdict(f"stage A's execution file is not JSON: {exc}") from exc
    if not isinstance(execution, list):
        raise NoVerdict("stage A's execution file is not a list of turns")
    text = review_text(execution)
    if len(text.strip()) < MIN_REVIEW_CHARS:
        raise NoVerdict(f"stage A produced no usable review ({len(text)} chars)")
    if len(text.encode()) > MAX_REVIEW_BYTES:
        raise NoVerdict(f"stage A's review is over {MAX_REVIEW_BYTES} bytes")
    rec = recommendation(text)
    review_out.write_text(text)
    return rec, VERDICT_FOR[rec]


def pin_verdict(findings_file: Path, verdict: str) -> str:
    """Set the findings file's verdict to the mapped one. Returns what changed.

    Reads the file with the sanitizer's loader and leaves one it rejects
    untouched: the sanitizer then rejects the same bytes and reports why.
    """
    if verdict not in set(VERDICT_FOR.values()):
        raise ValueError(f"not a verdict: {verdict!r}")
    obj, problem = load_structured(findings_file)
    if obj is None:
        return (
            f"findings file left for the sanitizer to reject ({problem.split(':')[0]})"
        )
    notes = []
    if obj.get("verdict") != verdict:
        # Stage B's value is never printed: the log would run any workflow
        # command it carries.
        notes.append(f"verdict overwritten with {verdict!r}")
        obj["verdict"] = verdict
    findings = obj.get("findings")
    if verdict == "ready_for_human_review" and isinstance(findings, list):
        demoted = 0
        for f in findings:
            if (
                isinstance(f, dict)
                and isinstance(f.get("severity"), str)
                and f["severity"].strip().lower() == "major"
            ):
                f["severity"] = "minor"
                demoted += 1
        if demoted:
            notes.append(f"{demoted} major finding(s) -> minor")
    if notes:
        findings_file.write_text(json.dumps(obj, indent=2))
    return "; ".join(notes) or "unchanged"


def _printable(name: str) -> str:
    """A file name on one line: control characters escaped, so a crafted name
    cannot forge an index entry."""
    return "".join(
        c if c.isprintable() else c.encode("unicode_escape").decode() for c in name
    )


def guide_paths(changed: list[str]) -> dict[str, list[str]]:
    """Every REVIEW.md that could cover a changed path: one per ancestor dir."""
    cover: dict[str, list[str]] = {}
    for p in changed:
        d = os.path.dirname(p)
        while True:
            g = f"{d}/REVIEW.md" if d else "REVIEW.md"
            cover.setdefault(g, []).append(p)
            if not d:
                break
            d = os.path.dirname(d)
    return cover


def collect_guides(repo: Path, base: str, paths_file: Path, out_dir: Path) -> str:
    """Copy covering guides from the merge base into out_dir; return an index."""
    changed = sorted({p for p in paths_file.read_text().split("\0") if p})
    lines = []
    for g, files in sorted(guide_paths(changed).items()):
        r = subprocess.run(
            ["git", "-C", str(repo), "show", f"{base}:{g}"],
            capture_output=True,
            check=False,
        )
        if r.returncode != 0:
            continue
        dst = out_dir / g
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(r.stdout)
        names = ", ".join(_printable(f) for f in sorted(set(files)))
        lines.append(f"- {dst} (from the base branch) covers: {names}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("collect-guides")
    g.add_argument("--repo", required=True, type=Path)
    g.add_argument("--base-commit", required=True)
    g.add_argument("--paths-file", required=True, type=Path)
    g.add_argument("--out-dir", required=True, type=Path)
    e = sub.add_parser("extract-review")
    e.add_argument("--execution-file", required=True, type=Path)
    e.add_argument("--review-out", required=True, type=Path)
    e.add_argument("--github-output", type=Path)
    p = sub.add_parser("pin-verdict")
    p.add_argument("--findings-file", required=True, type=Path)
    p.add_argument("--verdict", required=True)
    args = ap.parse_args(argv)

    if args.cmd == "collect-guides":
        if not re.fullmatch(r"[0-9a-f]{40}", args.base_commit):
            print(f"::error::not a commit SHA: {args.base_commit!r}", file=sys.stderr)
            return 1
        args.out_dir.mkdir(parents=True, exist_ok=True)
        index = collect_guides(
            args.repo, args.base_commit, args.paths_file, args.out_dir
        )
        (args.out_dir / INDEX_NAME).write_text(
            (index or "No REVIEW.md covers any of this PR's changed files.") + "\n"
        )
        # A count only: the index holds PR file names, and printing them would
        # let a name carrying "\n::add-mask::..." issue workflow commands.
        print(f"{index.count(chr(10)) + 1 if index else 0} guide(s) collected")
        return 0
    if args.cmd == "extract-review":
        try:
            rec, verdict = extract_review(args.execution_file, args.review_out)
        except (OSError, NoVerdict) as exc:
            print(f"::error::{exc}", file=sys.stderr)
            return 1
        print(f"stage A recommendation: {rec} -> verdict {verdict}")
        if args.github_output:
            with args.github_output.open("a") as f:
                f.write(f"verdict={verdict}\n")
        return 0
    print(pin_verdict(args.findings_file, args.verdict))
    return 0


if __name__ == "__main__":
    sys.exit(main())
