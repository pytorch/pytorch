#!/usr/bin/env python3
"""Post the hardened PR review's verdict as a native GitHub pull-request review.

A submitted review notifies the PR author, and its findings sit on the lines
they are about.

Reads the TERMINAL ROW (terminal.json), never the review job's artifact: the row
is what emit_row.py re-checked, so the review cannot say something the row and
the label step did not agree to. Every string is re-checked here again with the
same predicates, and anything that fails is dropped rather than repaired.

EARLIER AUTOMATED REVIEWS ARE HIDDEN, NEVER EDITED. A new review is posted
first; only once that has succeeded are the earlier ones hidden, so a failed
post never leaves the PR with no review at all. Hiding resolves the review's
inline threads, then minimizes it as outdated; a review whose threads cannot
be resolved stays visible until a later publication manages it. A change
request is hidden only once it no longer stands. Nothing is
deleted, and GitHub cannot delete a submitted review anyway.

A CHANGE REQUEST STANDS UNTIL A NEWER VERDICT REPLACES IT. A newer
REQUEST_CHANGES review takes its place in the PR's review decision. A clean
verdict is a COMMENT, which does not clear an earlier request, so after one
the earlier requests are dismissed. Apart from a run retracting its own review
for a commit that is no longer the head, that is the only dismissal. A run
that fails, is skipped or is opted out of leaves the last verdict as it is.

`@pytorchbot` IS DEFUSED IN EVERYTHING POSTED. pytorch-bot parses submitted and
edited review bodies for commands (torchci/lib/bot/pytorchBot.ts) and only skips
its own accounts, so a review body line starting `@pytorchbot merge` would run.
The sanitizer already defuses every mention in model prose; `defuse_bot_commands`
is the last pass over the text this script assembles, paths and framing included.

An API failure never fails the job: the telemetry row is already written, and a
red publish job on every GitHub hiccup would teach people to ignore it. Problems
are `::warning::`, and idempotent calls are retried. The one exit 1 is a change
request that may still stand after a clean verdict, which nothing else would
clear.
"""

from __future__ import annotations

import argparse
import http.client
import json
import os
import re
import sys
import time
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
DEFAULT_AUTHOR = "github-actions[bot]"
RETRIES = 3
MAX_WAIT = 60  # longest rate-limit wait honoured; the job has a 10-minute budget
_sleep = time.sleep
API = "https://api.github.com"

# pytorch-bot's command pattern is `^ *@pytorch(merge|)bot .+$`. Matched more
# widely here (any case, the `pytorch-bot` login too) because defusing a harmless
# mention costs nothing and missing a live one runs a command.
_BOT_MENTION = re.compile(r"@(?=pytorch(?:merge)?bot\b|pytorch-bot\b)", re.IGNORECASE)

BLOCKING_MARK = "\U0001f534"  # red circle
FOLDED_MARK = "\u26aa"  # grey circle
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
    def __init__(self, status: int, message: str, retry_after: int | None = None):
        super().__init__(f"HTTP {status}: {message}")
        self.status = status
        # Seconds GitHub asked us to wait; set only on a rate-limit response.
        self.retry_after = retry_after


def _retry_after(headers) -> int | None:
    """The wait a rate-limit response asks for, or None if it is not one."""
    if headers is None:
        return None
    value = headers.get("Retry-After")
    if value is not None:
        try:
            return max(0, int(value))
        except ValueError:
            return None
    if headers.get("X-RateLimit-Remaining") == "0":
        try:
            return max(0, int(headers.get("X-RateLimit-Reset", "")) - int(time.time()))
        except ValueError:
            return None
    return None


class GitHub:
    """The handful of REST and GraphQL calls this script makes."""

    def __init__(self, token: str, repo: str):
        self.token = token
        self.repo = repo

    def request(
        self, method: str, path: str, body: dict | None = None, idempotent=None
    ):
        """One API call. Idempotent calls are retried on transient failures.

        GET, PUT and DELETE are idempotent, and so is every GraphQL call made
        here; creating a review is not, so a failed POST is never repeated.
        """
        if idempotent is None:
            idempotent = method in ("GET", "PUT", "DELETE")
        for attempt in range(RETRIES if idempotent else 1):
            try:
                return self._once(method, path, body)
            except GitHubError as exc:
                # A secondary rate limit arrives as a 403 carrying Retry-After;
                # a plain 403 is a permission answer and is not retried.
                throttled = exc.retry_after is not None
                transient = exc.status in (0, 429) or exc.status >= 500 or throttled
                if not transient or attempt == RETRIES - 1 or not idempotent:
                    raise
                wait = min(exc.retry_after, MAX_WAIT) if throttled else 0
                _sleep(max(2**attempt, wait))

    def _once(self, method: str, path: str, body: dict | None):
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
            raise GitHubError(exc.code, detail, _retry_after(exc.headers)) from exc
        # URLError, timeouts, resets and truncated bodies carry no HTTP status.
        # They must still be a GitHubError, which is what callers isolate.
        except (OSError, http.client.HTTPException) as exc:
            raise GitHubError(0, repr(exc)) from exc
        try:
            return json.loads(raw) if raw else None
        except ValueError as exc:
            raise GitHubError(0, f"unparsable response: {exc}") from exc

    def graphql(self, query: str, variables: dict) -> dict:
        """One GraphQL call, retried like an idempotent REST call.

        GraphQL reports most failures inside an HTTP 200, which `request`'s
        retry never sees; every query and mutation sent here is idempotent.
        """
        for attempt in range(RETRIES):
            out = self.request(
                "POST", "/graphql", {"query": query, "variables": variables}, True
            )
            if isinstance(out, dict) and not out.get("errors"):
                return out.get("data") or {}
            if attempt == RETRIES - 1:
                raise GitHubError(200, json.dumps((out or {}).get("errors"))[:500])
            _sleep(2**attempt)
        return {}

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


# A line that starts markdown block syntax: heading, list, quote, table row,
# indented code, or an (escaped) fence run. Prefixing a mark would turn it into
# plain text, so the mark then goes on a line of its own.
_BLOCK_START = re.compile(r"^(?: {4}|\s*(?:#|[-*+>|]|\d+[.)]|\\[`~]))")


def marked(mark: str, lines: list[str]) -> list[str]:
    if _BLOCK_START.match(lines[0]):
        return [mark, "", *lines]
    return [f"{mark} {lines[0]}", *lines[1:]]


def comment_body(finding: dict) -> str:
    message = finding["message"].split("\n")
    return defuse_bot_commands("\n".join(marked(BLOCKING_MARK, message)))


def count(n: int, noun: str) -> str:
    return f"{n} {noun}" if n == 1 else f"{n} {noun}s"


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


_FENCE_RUN = re.compile(r"`{3,}|~{3,}")


def contained(text: str) -> list[str]:
    """`text` as lines, with every possible code-fence run escaped.

    The sanitizer has already removed links, images, HTML, mentions and issue
    references, so what a message can still carry is markdown STRUCTURE. Nearly
    all of it ends at the blank line after the message: a heading, list, table,
    setext underline or quote cannot reach the next finding. A fenced code block
    is the one construct that runs on until it is closed, and an open one would
    turn every later finding, link and `</details>` into code.

    Escaping, not balancing: whether a run opens or closes a fence depends on
    the list or quote it sits in, and a balancer that misreads a container
    adds the very fence it meant to close. A backslash-escaped backtick or
    tilde is never a fence character, in any container. The cost is that a
    model's fenced block shows as literal backticks around plain text; inline
    code still renders.
    """
    escaped = _FENCE_RUN.sub(lambda m: "".join("\\" + c for c in m.group()), text)
    return escaped.split("\n")


def permalink(repo: str, sha: str, finding: dict) -> str:
    """A blob link to the finding's line, built on the POSTING repo.

    Alone on its own line, GitHub renders it as the code it points at; it does
    so only when the link's repo is the one the comment is posted in. Every
    character outside `/` is percent-encoded, which also keeps `@` out of it.
    """
    path = urllib.parse.quote(unescape_path(finding["path"]), safe="/")
    return f"https://github.com/{repo}/blob/{sha}/{path}#L{finding['line']}"


def finding_blocks(
    repo: str, sha: str, findings: list[dict], mark: str = ""
) -> list[str]:
    """Each finding as its code link with the message under it, rule-separated.

    The blank line after the link is required: GitHub renders the code only
    for a link alone in its paragraph. `mark` leads the message's first line.
    """
    lines: list[str] = []
    for i, f in enumerate(findings):
        if i:
            lines += ["---", ""]
        message = contained(f["message"])
        if mark:
            message = marked(mark, message)
        lines += [permalink(repo, sha, f), "", *message, ""]
    return lines


# GitHub refuses a review body over 65536 characters, and the sanitizer's caps
# do not bound this one: percent-encoding and the layout expand the text.
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
        *contained(summary),
        "",
    ]
    if blocking and inline:
        lines += [
            f"{BLOCKING_MARK} {count(len(blocking), 'blocking finding')} "
            f"{'is' if len(blocking) == 1 else 'are'} attached to the code below.",
            "",
        ]
    lines += finding_blocks(repo, sha, in_body, BLOCKING_MARK)
    if folded:
        lines += [
            "<details>",
            f"<summary>{FOLDED_MARK} {count(len(folded), 'non-blocking finding')}"
            "</summary>",
            "",
        ]
        lines += finding_blocks(repo, sha, folded)
        lines += ["</details>", ""]
    if omitted:
        lines += [
            f"{count(omitted, 'more finding')} did not fit in this review; the Dr.CI "
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
        "<sub>This review is AI-generated and advisory. Each new automated "
        "review replaces the previous one.</sub>",
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


_MINIMIZED = "query($ids: [ID!]!) { nodes(ids: $ids) { ... on PullRequestReview { id isMinimized } } }"


def earlier_reviews(gh: GitHub, pr: int, author: str, new_id: int) -> list[dict]:
    """This workflow's reviews posted BEFORE `new_id` that are not yet hidden.

    OLDER, not merely other: review ids increase, so when two publish runs
    overlap the later review survives both cleanups instead of each run hiding
    the other's. A minimized review is done: its threads were resolved before it
    was minimized, and leaving it alone keeps a thread someone reopened open.
    """
    mine = [
        r
        for r in gh.paged(f"/repos/{gh.repo}/pulls/{pr}/reviews")
        if isinstance(r.get("id"), int)
        and r["id"] < new_id
        and (r.get("user") or {}).get("login") == author
        and (r.get("body") or "").startswith(MARKER)
    ]
    # Only a review GitHub explicitly reports as visible is touched; one whose
    # state did not come back is left for the next publication.
    visible: set[str] = set()
    for i in range(0, len(mine), 100):
        ids = [r["node_id"] for r in mine[i : i + 100]]
        nodes = gh.graphql(_MINIMIZED, {"ids": ids}).get("nodes") or []
        visible |= {n["id"] for n in nodes if n and n.get("isMinimized") is False}
    return [r for r in mine if r["node_id"] in visible]


_MINIMIZE = "mutation($id: ID!) { minimizeComment(input: {subjectId: $id, classifier: OUTDATED}) { clientMutationId } }"


def retract(gh: GitHub, pr: int, review: dict) -> None:
    """Take back this run's own review, posted for a commit that is no longer the head.

    A change request is dismissed as well as hidden: a newer run may already
    have posted its clean verdict and finished, and nothing after that would
    clear a request this run left standing. If the dismissal fails the review
    stays visible, so the request it makes is not hidden.
    """
    if review.get("state") == "CHANGES_REQUESTED":
        try:
            gh.request(
                "PUT",
                f"/repos/{gh.repo}/pulls/{pr}/reviews/{review['id']}/dismissals",
                {
                    "message": "Retracted: the PR changed while this review was posted.",
                    "event": "DISMISS",
                },
            )
        except GitHubError as exc:
            print(
                f"::error::could not dismiss retracted review {review['id']}; "
                f"it still requests changes on an old commit: {exc}"
            )
            return
    take_down(gh, pr, review)


def take_down(gh: GitHub, pr: int, review: dict) -> None:
    """Hide an automated review as outdated, leaving its text and state alone.

    Its inline threads are resolved, then the review is minimized. A reader can
    expand either one. A change request is not dismissed here (see the module
    docstring). If a thread cannot be resolved the review stays visible, and the
    next publication tries again.
    """
    rid = review["id"]
    try:
        ids = {
            c["id"]
            for c in gh.paged(f"/repos/{gh.repo}/pulls/{pr}/reviews/{rid}/comments")
        }
        resolved = resolve_threads(gh, pr, ids)
    except GitHubError as exc:
        warn(f"could not resolve the threads of review {rid}: {exc}")
        resolved = False
    if not resolved:
        warn(f"leaving review {rid} visible until its threads are resolved")
        return
    try:
        gh.graphql(_MINIMIZE, {"id": review["node_id"]})
        print(f"hid earlier automated review {rid} as outdated")
    except GitHubError as exc:
        warn(f"could not minimize review {rid}: {exc}")


_THREADS = """query($o: String!, $r: String!, $n: Int!, $c: String) {
  repository(owner: $o, name: $r) { pullRequest(number: $n) {
    reviewThreads(first: 100, after: $c) {
      pageInfo { hasNextPage endCursor }
      nodes { id isResolved comments(first: 1) { nodes { databaseId } } }
    } } } }"""
_RESOLVE = "mutation($id: ID!) { resolveReviewThread(input: {threadId: $id}) { clientMutationId } }"


def resolve_threads(gh: GitHub, pr: int, comment_ids: set[int]) -> bool:
    """Resolve every open thread started by one of `comment_ids`.

    True when each of those threads was found and is now resolved. Threads are
    listed once per call, not once per comment.
    """
    if not comment_ids:
        return True
    owner, name = gh.repo.split("/")
    found: set[int] = set()
    ok = True
    cursor = None
    while True:
        data = gh.graphql(_THREADS, {"o": owner, "r": name, "n": pr, "c": cursor})
        threads = data["repository"]["pullRequest"]["reviewThreads"]
        for node in threads["nodes"]:
            first = node["comments"]["nodes"]
            if not first or first[0]["databaseId"] not in comment_ids:
                continue
            found.add(first[0]["databaseId"])
            if node["isResolved"]:
                continue
            try:
                gh.graphql(_RESOLVE, {"id": node["id"]})
            except GitHubError as exc:
                warn(f"could not resolve thread {node['id']}: {exc}")
                ok = False
        if not threads["pageInfo"]["hasNextPage"]:
            break
        cursor = threads["pageInfo"]["endCursor"]
    return ok and found == comment_ids


def head_sha(gh: GitHub, pr: int):
    return (
        (gh.request("GET", f"/repos/{gh.repo}/pulls/{pr}") or {}).get("head") or {}
    ).get("sha")


def standing_requests(gh: GitHub, pr: int, author: str) -> list[dict]:
    """This workflow's reviews that still show changes requested."""
    return [
        r
        for r in gh.paged(f"/repos/{gh.repo}/pulls/{pr}/reviews")
        if r.get("state") == "CHANGES_REQUESTED"
        and (r.get("user") or {}).get("login") == author
        and (r.get("body") or "").startswith(MARKER)
    ]


def in_scope(review: dict, keep_commit=None, older_than=None) -> bool:
    """Whether a standing request is one this run may take back.

    `keep_commit` spares requests on that commit; `older_than` spares any
    review at or above that id, so an overlapping run's newer request on the
    same commit is never touched.
    """
    if keep_commit and review.get("commit_id") == keep_commit:
        return False
    return older_than is None or review["id"] < older_than


def withdraw(
    gh: GitHub,
    pr: int,
    author: str,
    message: str,
    head=None,
    keep_commit=None,
    older_than=None,
):
    """Dismiss this workflow's standing change requests that are in scope.

    Dismissal only: the findings stay readable under the dismissal.
    """
    standing = [
        r
        for r in standing_requests(gh, pr, author)
        if in_scope(r, keep_commit, older_than)
    ]
    # LIST, THEN RE-READ THE HEAD. A run for a newer head may have posted a
    # change request on that head; if it did so before this read, the head has
    # moved and nothing is dismissed. If it posts after, it is not in the list.
    if head and standing and head_sha(gh, pr) != head:
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
        return 0
    if review is None:
        print(f"status={row.get('status')}; no review to post")
        return 0
    verdict, summary, findings = review
    clean = verdict == "ready_for_human_review"
    try:
        new_id = publish(gh, pr, sha, author, verdict, summary, findings)
    except GitHubError as exc:
        if not clean:
            raise
        warn(f"automated review not posted: {exc}")
        # No review of ours to measure age against: spare this commit's
        # requests, which an overlapping run may have just posted.
        return clear_after_clean(gh, pr, sha, author, keep_commit=sha)
    if new_id is RETRACTED or not clean:
        return 0
    if new_id is None:
        return clear_after_clean(gh, pr, sha, author, keep_commit=sha)
    rc = clear_after_clean(gh, pr, sha, author, older_than=new_id)
    # Now hide the requests that were just dismissed; any still standing stay
    # visible (see `hide_earlier`).
    try:
        hide_earlier(gh, pr, author, {"id": new_id, "state": "COMMENTED"})
    except GitHubError as exc:
        warn(f"could not hide the withdrawn reviews: {exc}")
    return rc


def clear_after_clean(
    gh: GitHub, pr: int, sha: str, author: str, keep_commit=None, older_than=None
) -> int:
    """After a clean verdict, no earlier change request from this workflow may remain.

    The clean review is a COMMENT, which leaves an earlier request standing, and
    the PR has moved to `ready for review`, after which pushes trigger no
    review. So the earlier requests are dismissed here; if one still stands,
    fail the step naming it, so it reaches a maintainer. Only requests in scope
    (see `in_scope`) count, both times. An API error here fails the step too,
    because the earlier requests are already hidden and may still stand.
    """
    try:
        return _clear_after_clean(gh, pr, sha, author, keep_commit, older_than)
    except GitHubError as exc:
        print(
            "::error::could not confirm that earlier automated change requests "
            f"were withdrawn after a clean verdict; check the PR's reviews: {exc}"
        )
        return 1


def _clear_after_clean(gh, pr, sha, author, keep_commit, older_than) -> int:
    withdraw(
        gh,
        pr,
        author,
        f"Withdrawn: the automated review of {sha[:12]} found nothing blocking.",
        head=sha,
        keep_commit=keep_commit,
        older_than=older_than,
    )
    left = [
        r
        for r in standing_requests(gh, pr, author)
        if in_scope(r, keep_commit, older_than)
    ]
    # Same order as `withdraw`: list, then re-read the head. A moved head means
    # a newer run owns the PR now, and any request it posted is not stale.
    if not left or head_sha(gh, pr) != sha:
        return 0
    for r in left:
        print(
            f"::error::change request {r.get('html_url', r['id'])} still stands "
            "after a clean verdict; dismiss it by hand"
        )
    return 1 if left else 0


RETRACTED = object()


def publish(gh, pr, sha, author, verdict, summary, findings):
    """Post the review and hide earlier ones.

    Returns the new review's id, None if GitHub returned none, or RETRACTED
    when the head moved while posting and the review was taken back.
    """
    new = post(gh, pr, sha, verdict, summary, findings[:MAX_FINDINGS])
    print(f"posted automated review {new.get('id')}: {new.get('html_url')}")
    if not isinstance(new.get("id"), int):
        warn("GitHub returned no review id; leaving earlier reviews in place")
        return None
    # Once the review exists, its id is what scopes the clean-verdict check,
    # so nothing after this point may lose it by raising.
    try:
        return replace_earlier(gh, pr, sha, author, new)
    except GitHubError as exc:
        warn(f"posted review {new['id']}, but cleanup failed: {exc}")
        return new["id"]


def replace_earlier(gh, pr, sha, author, new):
    """Hide `new` if the head moved, else hide earlier reviews."""
    # CHECKED AGAIN AFTER POSTING. If the head moved between the check above and
    # the post, a run for the newer commit may already have published, with a
    # SMALLER review id, and the id rule below would hide the current review
    # in favour of this stale one. So a moved head retracts this review
    # instead. If the head moves after this read, the newer commit's run has
    # not posted yet; its review gets the larger id and hides this one.
    current = head_sha(gh, pr)
    if current != sha:
        print(f"head moved {sha} -> {current} while posting; retracting this review")
        retract(gh, pr, new)
        return RETRACTED
    hide_earlier(gh, pr, author, new)
    return new["id"]


def hide_earlier(gh, pr, author, new):
    """Hide this workflow's reviews older than `new`.

    A REQUEST HIDDEN ONLY ONCE IT NO LONGER STANDS. An earlier change request is
    hidden here only when `new` itself requests changes and so replaces it.
    Otherwise it stays visible until `clear_after_clean` has dismissed it, and
    `run` calls this again after that.
    """
    replaced = new.get("state") == "CHANGES_REQUESTED"
    # Newest first, and each one isolated: the review being replaced matters
    # more than retrying leftovers on long-superseded ones, and one failed call
    # must not stop the rest.
    olds = earlier_reviews(gh, pr, author, new["id"])
    for old in sorted(olds, key=lambda r: r["id"], reverse=True):
        if old.get("state") == "CHANGES_REQUESTED" and not replaced:
            continue
        try:
            take_down(gh, pr, old)
        except GitHubError as exc:
            warn(f"could not hide review {old['id']}: {exc}")


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
    except Exception as exc:  # noqa: BLE001 - an API failure never fails the job
        warn(f"automated review not posted: {exc!r}")
        return 0


if __name__ == "__main__":
    sys.exit(main())
