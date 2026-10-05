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
leaves the PR with no review at all. GitHub cannot delete a SUBMITTED review
(GraphQL `deletePullRequestReview` answers "Can not delete a non-pending pull
request review"; measured on pytorch/ciforge, 2026-10-05), so taking one down is
the most the API allows: dismiss it if it requested changes, delete its inline
comments, replace its body with a pointer to the new review, and minimize it as
outdated.

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
import urllib.parse
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


def blocking_findings(verdict: str, findings: list[dict]) -> list[dict]:
    """The findings that hold the PR back, posted inline; the rest are folded.

    Normally the `major` ones. A `changes_requested` verdict with no major
    finding is resting on its minor ones, so then every finding blocks, rather
    than requesting changes while calling all of them non-blocking.
    """
    major = [f for f in findings if f["severity"] == "major"]
    if verdict == "changes_requested" and not major:
        return list(findings)
    return major


def quoted(text: str) -> list[str]:
    """`text` as a blockquote, which contains whatever markdown it holds.

    Messages keep their markdown structure, and an unclosed code fence in one
    would otherwise swallow every finding, link and tag after it. A fence or
    heading inside a blockquote ends where the quote ends.
    """
    return [f"> {line}" if line else ">" for line in text.split("\n")]


def permalink(repo: str, sha: str, finding: dict) -> str:
    """A blob link to the finding's line, built on the POSTING repo.

    Alone on its own line, GitHub renders it as the code it points at; it does
    so only when the link's repo is the one the comment is posted in. Every
    character outside `/` is percent-encoded, which also keeps `@` out of it.
    """
    path = urllib.parse.quote(unescape_path(finding["path"]), safe="/")
    return f"https://github.com/{repo}/blob/{sha}/{path}#L{finding['line']}"


def finding_block(repo: str, sha: str, finding: dict) -> list[str]:
    label = _SEVERITY_LABEL[finding["severity"]]
    return [
        permalink(repo, sha, finding),
        "",
        f"**{label}**",
        *quoted(finding["message"]),
        "",
    ]


# GitHub refuses a review body over 65536 characters, and the sanitizer's caps
# do not bound this one: percent-encoding and quote prefixes expand the text.
MAX_BODY = 60_000


def review_body(
    verdict: str, summary: str, findings: list[dict], sha: str, repo: str, inline: bool
) -> str:
    """The review body: verdict, summary, and every finding not posted inline.

    Blocking findings are inline comments when GitHub accepts them; the rest
    are folded into one collapsed section, each led by a permalink so it still
    shows its code. `<summary>` text is plain: GitHub renders no markdown there.
    A body over MAX_BODY drops findings from the end, folded ones first, and
    says how many; the Dr.CI comment still lists them all.
    """
    blocking = blocking_findings(verdict, findings)
    in_body = [] if inline else blocking
    folded = [f for f in findings if f not in blocking]
    keep_body, keep_folded = len(in_body), len(folded)
    while True:
        body = _compose(
            verdict,
            summary,
            sha,
            repo,
            blocking,
            inline,
            in_body[:keep_body],
            folded[:keep_folded],
            len(in_body) - keep_body + len(folded) - keep_folded,
        )
        if len(body) <= MAX_BODY or keep_body + keep_folded == 0:
            return body
        if keep_folded:
            keep_folded -= 1
        else:
            keep_body -= 1


def _compose(
    verdict, summary, sha, repo, blocking, inline, in_body, folded, omitted
) -> str:
    lines = [
        MARKER,
        f"**Automated review** of {sha[:12]}: {_VERDICT_LINE[verdict]}.",
        "",
        *quoted(summary),
        "",
    ]
    if blocking and inline:
        lines += [
            f"{len(blocking)} blocking finding(s) are attached to the code below.",
            "",
        ]
    for f in in_body:
        lines += finding_block(repo, sha, f)
    if folded:
        lines += [
            "<details>",
            f"<summary>{len(folded)} non-blocking finding(s)</summary>",
            "",
        ]
        for f in folded:
            lines += finding_block(repo, sha, f)
        lines += ["</details>", ""]
    if omitted:
        lines += [
            f"{omitted} more finding(s) did not fit in this review; the Dr.CI "
            "comment lists them.",
            "",
        ]
    if verdict == "changes_requested":
        lines += [
            "Please address the findings and push; a new commit normally gets "
            "a fresh automated review.",
            "",
        ]
    lines += [
        "This review is AI-generated and advisory. Each new automated review "
        "replaces the previous one.",
    ]
    return defuse_bot_commands("\n".join(lines))


def post(gh: GitHub, pr: int, sha: str, verdict, summary, findings) -> dict:
    """Create the review: blocking findings inline, the rest folded in the body.

    A `changes_requested` verdict is a REQUEST_CHANGES review, so the PR shows
    it the way it shows a human's; anything else is a COMMENT. Never APPROVE: a
    bot approval would read as a maintainer's sign-off.

    GitHub refuses the WHOLE review (422) if any comment's line is not in its
    diff. The sanitizer anchored every finding to our own diff, which should
    match, but a refusal must cost the anchoring, not the review: the blocking
    findings then move into the body too. If a REQUEST_CHANGES review is
    refused outright (the token cannot request changes on this PR), it is
    posted as a COMMENT rather than not at all.
    """
    path = f"/repos/{gh.repo}/pulls/{pr}/reviews"
    event = "REQUEST_CHANGES" if verdict == "changes_requested" else "COMMENT"
    base = {"commit_id": sha, "event": event}
    comments = [
        {
            "path": unescape_path(f["path"]),
            "line": f["line"],
            "side": "RIGHT",
            "body": comment_body(f),
        }
        for f in blocking_findings(verdict, findings)
    ]
    if comments:
        body = review_body(verdict, summary, findings, sha, gh.repo, inline=True)
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
    body = review_body(verdict, summary, findings, sha, gh.repo, inline=False)
    try:
        return gh.request("POST", path, {**base, "body": body})
    except GitHubError as exc:
        if exc.status != 422 or event == "COMMENT":
            raise
        warn(f"GitHub refused {event} ({exc}); posting the review as a COMMENT")
    return gh.request("POST", path, {**base, "event": "COMMENT", "body": body})


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


_MINIMIZE = "mutation($id: ID!) { minimizeComment(input: {subjectId: $id, classifier: OUTDATED}) { clientMutationId } }"


def take_down(gh: GitHub, pr: int, review: dict, new_url: str) -> None:
    """Remove an earlier automated review from the PR as far as GitHub allows.

    A REQUEST_CHANGES review is DISMISSED first, so the PR stops showing changes
    requested even if a later step fails; dismissal is what GitHub offers for
    taking back a review, and the timeline keeps a "dismissed" entry.
    """
    rid = review["id"]
    if review.get("state") == "CHANGES_REQUESTED":
        message = f"Superseded by a newer automated review: {new_url}"
        try:
            gh.request(
                "PUT",
                f"/repos/{gh.repo}/pulls/{pr}/reviews/{rid}/dismissals",
                {"message": message, "event": "DISMISS"},
            )
        except GitHubError as exc:
            # Leave it whole: hiding a change request that still stands would
            # make the PR claim a verdict nobody can read. The next
            # publication retries it.
            warn(f"could not dismiss review {rid}; leaving it in place: {exc}")
            return
    for c in gh.paged(f"/repos/{gh.repo}/pulls/{pr}/reviews/{rid}/comments"):
        try:
            gh.request("DELETE", f"/repos/{gh.repo}/pulls/comments/{c['id']}")
        except GitHubError as exc:
            if exc.status != 404:
                warn(f"could not delete comment {c['id']} of review {rid}: {exc}")
    body = f"{SUPERSEDED_MARKER}\nSuperseded by a newer automated review: {new_url}"
    try:
        if not review.get("body", "").startswith(SUPERSEDED_MARKER):
            gh.request(
                "PUT", f"/repos/{gh.repo}/pulls/{pr}/reviews/{rid}", {"body": body}
            )
        gh.graphql(_MINIMIZE, {"id": review["node_id"]})
        print(f"superseded earlier automated review {rid}")
    except GitHubError as exc:
        warn(f"could not mark review {rid} superseded: {exc}")


def head_sha(gh: GitHub, pr: int):
    return (
        (gh.request("GET", f"/repos/{gh.repo}/pulls/{pr}") or {}).get("head") or {}
    ).get("sha")


def withdraw(gh: GitHub, pr: int, author: str, message: str, keep_commit=None):
    """Dismiss this workflow's standing change requests, except on `keep_commit`.

    A REQUEST_CHANGES review stays until it is dismissed, and only a write user
    can dismiss it, so a request the pipeline will not replace has to be taken
    back here. Dismissal only: the findings stay readable under the dismissal.
    """
    standing = [
        r
        for r in gh.paged(f"/repos/{gh.repo}/pulls/{pr}/reviews")
        if r.get("state") == "CHANGES_REQUESTED"
        and (r.get("user") or {}).get("login") == author
        and (r.get("body") or "").startswith((MARKER, SUPERSEDED_MARKER))
        and not (keep_commit and r.get("commit_id") == keep_commit)
    ]
    # LIST, THEN RE-READ THE HEAD. A run for a newer head may have posted a
    # change request on that head; if it did so before this read, the head has
    # moved and nothing is dismissed. If it posts after, it is not in the list.
    if keep_commit and standing and head_sha(gh, pr) != keep_commit:
        print("head moved while withdrawing; leaving change requests alone")
        return
    for r in standing:
        try:
            gh.request(
                "PUT",
                f"/repos/{gh.repo}/pulls/{pr}/reviews/{r['id']}/dismissals",
                {"message": message, "event": "DISMISS"},
            )
            print(f"withdrew automated change request {r['id']}")
        except GitHubError as exc:
            warn(f"could not dismiss review {r['id']}: {exc}")


def run(gh: GitHub, row: dict, pr: int, sha: str, opt_out: str, author: str) -> int:
    review = load_review(row)
    current = head_sha(gh, pr)
    if current != sha:
        # A run for the newer head is coming; anything done here could act on
        # that run's review.
        print(f"head moved {sha} -> {current}; not posting a stale review")
        return 0
    labels = (
        gh.request("GET", f"/repos/{gh.repo}/issues/{pr}/labels?per_page=100") or []
    )
    if any(str(lb.get("name", "")).lower() == opt_out.lower() for lb in labels):
        print(f"PR carries '{opt_out}'; not posting")
        withdraw(gh, pr, author, "Withdrawn: this PR opted out of automated review.")
        return 0
    if review is None:
        # The head is `sha`, so a change request on any other commit is about
        # code that has since changed, and no review is replacing it.
        print(f"status={row.get('status')}; no review to post")
        withdraw(
            gh,
            pr,
            author,
            f"Withdrawn: the automated review of {sha[:12]} did not complete, "
            "and this request was about an earlier commit.",
            keep_commit=sha,
        )
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
