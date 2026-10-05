#!/usr/bin/env python3
"""Post the hardened PR review's verdict as a native GitHub pull-request review.

Until this existed, the verdict reached the PR only as a collapsed section of the
Dr.CI comment, and an EDIT to that comment sends no notification. Authors whose
PR was blocked on them never found out. A submitted review notifies the author,
and its findings sit on the lines they are about.

Reads the TERMINAL ROW (terminal.json), never the review job's artifact: the row
is what emit_row.py re-checked, so the review cannot say something the row and
the label step did not agree to. Every string is re-checked here again with the
same predicates, and anything that fails is dropped rather than repaired.

ONE AUTOMATED REVIEW PER PR. A new review is posted first; only once that has
succeeded are the earlier automated reviews taken down, so a failed post never
leaves the PR with no review at all. Taking one down means deleting each of its
inline comments, then deleting the review itself. GitHub may refuse to delete a
SUBMITTED review; when it does, the body is replaced by a pointer to the new
review and the review is minimized as outdated, which is the closest the API
allows.

`@pytorchbot` IS DEFUSED IN EVERYTHING POSTED. pytorch-bot parses submitted and
edited review bodies for commands (torchci/lib/bot/pytorchBot.ts) and only skips
its own accounts, so a review body line starting `@pytorchbot merge` would run.
The sanitizer already defuses every mention in model prose; `defuse_bot_commands`
is the last pass over the text this script assembles, paths and framing included.

Never fails the job: the telemetry row is already written, and a red publish job
on every GitHub hiccup would teach people to ignore it. Problems are `::warning::`.
"""

from __future__ import annotations

import argparse
import http.client
import json
import os
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path

from extract_verdict import (
    is_neutral_prose,
    MAX_FINDINGS,
    MAX_SUMMARY,
    published_findings,
    VERDICTS,
)


# The first line of every review this script posts. `neutralize` escapes `<` in
# model prose, so no summary or finding can forge it; the author check below is
# what stops a human pasting it into their own review.
MARKER = "<!-- hardened-pr-review -->"
SUPERSEDED_MARKER = "<!-- hardened-pr-review superseded -->"
DEFAULT_AUTHOR = "github-actions[bot]"
API = "https://api.github.com"

# pytorch-bot's command pattern is `^ *@pytorch(merge|)bot .+$`. Matched more
# widely here (any case, the `pytorch-bot` login too) because defusing a harmless
# mention costs nothing and missing a live one runs a command.
_BOT_MENTION = re.compile(r"@(?=pytorch(?:merge)?bot\b|pytorch-bot\b)", re.IGNORECASE)

_SEVERITY_LABEL = {"major": "Major", "minor": "Minor", "info": "Info"}
_VERDICT_LINE = {
    "changes_requested": "changes requested before human review",
    "ready_for_human_review": "nothing blocking; ready for human review",
}


def warn(msg: str) -> None:
    print(f"::warning::{msg}")


def defuse_bot_commands(text: str) -> str:
    return _BOT_MENTION.sub("@ ", text)


def unescape_path(path: str) -> str:
    """Invert `neutralize_path`, for use where GitHub wants the real file name."""
    return re.sub(r"\\(.)", r"\1", path)


class GitHubError(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(f"HTTP {status}: {message}")
        self.status = status


class GitHub:
    """The handful of REST and GraphQL calls this script makes."""

    def __init__(self, token: str, repo: str):
        self.token = token
        self.repo = repo

    def request(self, method: str, path: str, body: dict | None = None):
        url = path if path.startswith("https://") else f"{API}{path}"
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(url, data=data, method=method)
        req.add_header("Authorization", f"Bearer {self.token}")
        req.add_header("Accept", "application/vnd.github+json")
        req.add_header("X-GitHub-Api-Version", "2022-11-28")
        if data is not None:
            req.add_header("Content-Type", "application/json")
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                raw = resp.read()
        except urllib.error.HTTPError as exc:
            try:
                detail = exc.read().decode("utf-8", "replace")[:500]
            except (OSError, http.client.HTTPException):
                detail = repr(exc)
            raise GitHubError(exc.code, detail) from exc
        # URLError, timeouts, resets and truncated bodies carry no HTTP status.
        # They must still be a GitHubError, which is what callers isolate.
        except (OSError, http.client.HTTPException) as exc:
            raise GitHubError(0, repr(exc)) from exc
        try:
            return json.loads(raw) if raw else None
        except ValueError as exc:
            raise GitHubError(0, f"unparsable response: {exc}") from exc

    def graphql(self, query: str, variables: dict) -> dict:
        out = self.request("POST", "/graphql", {"query": query, "variables": variables})
        if not isinstance(out, dict) or out.get("errors"):
            raise GitHubError(200, json.dumps((out or {}).get("errors"))[:500])
        return out.get("data") or {}

    def paged(self, path: str) -> list:
        items: list = []
        page = 1
        while True:
            sep = "&" if "?" in path else "?"
            batch = self.request("GET", f"{path}{sep}per_page=100&page={page}") or []
            items.extend(batch)
            if len(batch) < 100:
                return items
            page += 1


def load_review(row: dict) -> tuple[str, str, list[dict]] | None:
    """The (verdict, summary, findings) a review may publish, or None."""
    if row.get("status") != "succeeded":
        return None
    verdict, summary = row.get("verdict"), row.get("summary")
    if verdict not in VERDICTS or not is_neutral_prose(summary, MAX_SUMMARY):
        return None
    try:
        raw = json.loads((row.get("extra") or {}).get("findings") or "[]")
    except (TypeError, ValueError):
        raw = []
    return verdict, summary, published_findings({"findings": raw})


def comment_body(finding: dict) -> str:
    label = _SEVERITY_LABEL[finding["severity"]]
    return defuse_bot_commands(f"**{label}** (automated review): {finding['message']}")


def review_body(
    verdict: str, summary: str, findings: list[dict], sha: str, inline: bool
) -> str:
    lines = [
        MARKER,
        f"**Automated review** of {sha[:12]}: {_VERDICT_LINE[verdict]}.",
        "",
        summary,
    ]
    if findings and inline:
        lines += ["", f"{len(findings)} finding(s) are attached to the code below."]
    elif findings:
        lines += [""]
        for f in findings:
            label = _SEVERITY_LABEL[f["severity"]]
            path = unescape_path(f["path"])
            lines.append(f"- `{path}` line {f['line']} ({label}): {f['message']}")
    if verdict == "changes_requested":
        lines += [
            "",
            "Please address the findings and push; a new commit normally gets "
            "a fresh automated review.",
        ]
    lines += [
        "",
        "This review is AI-generated and advisory. Each new automated review "
        "replaces the previous one.",
    ]
    return defuse_bot_commands("\n".join(lines))


def post(gh: GitHub, pr: int, sha: str, verdict, summary, findings) -> dict:
    """Create the review with inline comments, or with findings in the body.

    GitHub refuses the WHOLE review (422) if any comment's line is not in its
    diff. The sanitizer anchored every finding to our own diff, which should
    match, but a refusal must cost the anchoring, not the review.
    """
    path = f"/repos/{gh.repo}/pulls/{pr}/reviews"
    base = {"commit_id": sha, "event": "COMMENT"}
    if findings:
        comments = [
            {
                "path": unescape_path(f["path"]),
                "line": f["line"],
                "side": "RIGHT",
                "body": comment_body(f),
            }
            for f in findings
        ]
        body = review_body(verdict, summary, findings, sha, inline=True)
        try:
            return gh.request(
                "POST", path, {**base, "body": body, "comments": comments}
            )
        except GitHubError as exc:
            if exc.status != 422:
                raise
            warn(
                f"GitHub refused the inline comments ({exc}); posting them in the body"
            )
    body = review_body(verdict, summary, findings, sha, inline=False)
    return gh.request("POST", path, {**base, "body": body})


def earlier_reviews(gh: GitHub, pr: int, author: str, new_id: int) -> list[dict]:
    """This workflow's reviews posted BEFORE `new_id`, superseded ones included.

    OLDER, not merely other: review ids increase, so when two publish runs
    overlap the later review survives both cleanups instead of each run taking
    down the other's. Superseded reviews stay in the set so that a comment whose
    deletion failed last time is retried rather than abandoned.
    """
    return [
        r
        for r in gh.paged(f"/repos/{gh.repo}/pulls/{pr}/reviews")
        if isinstance(r.get("id"), int)
        and r["id"] < new_id
        and (r.get("user") or {}).get("login") == author
        and (r.get("body") or "").startswith((MARKER, SUPERSEDED_MARKER))
    ]


_DELETE = "mutation($id: ID!) { deletePullRequestReview(input: {pullRequestReviewId: $id}) { clientMutationId } }"
_MINIMIZE = "mutation($id: ID!) { minimizeComment(input: {subjectId: $id, classifier: OUTDATED}) { clientMutationId } }"


def take_down(gh: GitHub, pr: int, review: dict, new_url: str) -> None:
    rid = review["id"]
    for c in gh.paged(f"/repos/{gh.repo}/pulls/{pr}/reviews/{rid}/comments"):
        try:
            gh.request("DELETE", f"/repos/{gh.repo}/pulls/comments/{c['id']}")
        except GitHubError as exc:
            if exc.status != 404:
                warn(f"could not delete comment {c['id']} of review {rid}: {exc}")
    try:
        gh.graphql(_DELETE, {"id": review["node_id"]})
        print(f"deleted earlier automated review {rid}")
        return
    except GitHubError as exc:
        print(f"review {rid} cannot be deleted ({exc}); marking it superseded")
    body = f"{SUPERSEDED_MARKER}\nSuperseded by a newer automated review: {new_url}"
    try:
        if not review.get("body", "").startswith(SUPERSEDED_MARKER):
            gh.request(
                "PUT", f"/repos/{gh.repo}/pulls/{pr}/reviews/{rid}", {"body": body}
            )
        gh.graphql(_MINIMIZE, {"id": review["node_id"]})
    except GitHubError as exc:
        warn(f"could not mark review {rid} superseded: {exc}")


def head_sha(gh: GitHub, pr: int):
    return (
        (gh.request("GET", f"/repos/{gh.repo}/pulls/{pr}") or {}).get("head") or {}
    ).get("sha")


def run(gh: GitHub, row: dict, pr: int, sha: str, opt_out: str, author: str) -> int:
    review = load_review(row)
    if review is None:
        print(f"status={row.get('status')}; no review to post")
        return 0
    current = head_sha(gh, pr)
    if current != sha:
        print(f"head moved {sha} -> {current}; not posting a stale review")
        return 0
    labels = (
        gh.request("GET", f"/repos/{gh.repo}/issues/{pr}/labels?per_page=100") or []
    )
    if any(str(lb.get("name", "")).lower() == opt_out.lower() for lb in labels):
        print(f"PR carries '{opt_out}'; not posting")
        return 0
    verdict, summary, findings = review
    new = post(gh, pr, sha, verdict, summary, findings[:MAX_FINDINGS])
    print(f"posted automated review {new.get('id')}: {new.get('html_url')}")
    if not isinstance(new.get("id"), int):
        warn("GitHub returned no review id; leaving earlier reviews in place")
        return 0
    # CHECKED AGAIN AFTER POSTING. If the head moved between the check above and
    # the post, a run for the newer commit may already have published, with a
    # SMALLER review id, and the id rule below would take the current review
    # down in favour of this stale one. So a moved head retracts this review
    # instead. If the head moves after this read, the newer commit's run has
    # not posted yet; its review gets the larger id and takes this one down.
    current = head_sha(gh, pr)
    if current != sha:
        print(f"head moved {sha} -> {current} while posting; retracting this review")
        take_down(gh, pr, new, f"https://github.com/{gh.repo}/pull/{pr}")
        return 0
    # Newest first, and each one isolated: the review being replaced matters
    # more than retrying leftovers on long-superseded ones, and one failed call
    # must not stop the rest.
    olds = earlier_reviews(gh, pr, author, new["id"])
    for old in sorted(olds, key=lambda r: r["id"], reverse=True):
        try:
            take_down(gh, pr, old, new.get("html_url", ""))
        except GitHubError as exc:
            warn(f"could not take down review {old['id']}: {exc}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--row-file", required=True, help="terminal.json from emit_row.py")
    args = ap.parse_args()
    try:
        row = json.loads(Path(args.row_file).read_text())
        gh = GitHub(os.environ["GH_TOKEN"], os.environ["REPO"])
        return run(
            gh,
            row if isinstance(row, dict) else {},
            int(os.environ["PR_NUMBER"]),
            os.environ["REVIEWED_SHA"],
            os.environ.get("OPT_OUT_LABEL", "no automated review"),
            os.environ.get("REVIEW_AUTHOR", DEFAULT_AUTHOR),
        )
    except Exception as exc:  # noqa: BLE001 - never fail the publish job
        warn(f"automated review not posted: {exc!r}")
        return 0


if __name__ == "__main__":
    sys.exit(main())
