#!/usr/bin/env python3
"""Post the hardened PR review's verdict as a native GitHub pull-request review.

A review notifies the PR author and puts findings on their lines.

Input is the terminal row (terminal.json) that emit_row.py already checked.
Every string is checked again with the sanitizer's predicates; anything that
fails is dropped, never repaired.

Invariants:
- A `changes_requested` verdict is REQUEST_CHANGES; anything else is COMMENT,
  never APPROVE.
- A new review is posted before earlier ones are hidden. Hiding resolves the
  review's inline threads and minimizes it as outdated. Nothing is edited or
  deleted. A change request is hidden only once it no longer stands.
- A newer REQUEST_CHANGES replaces an older one in the review decision. A
  COMMENT does not, so after a clean verdict earlier requests are dismissed.
  The only other dismissal is a run retracting its own review because the
  head moved. Failed, skipped and opted-out runs change nothing.
- `@pytorchbot` is escaped in all posted text, because pytorch-bot reads
  commands from review bodies.
- API errors are warnings. The step fails only when an earlier change request
  may still stand after a clean verdict.
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
        """One API call. Idempotent calls (GET, PUT, DELETE, our GraphQL) retry
        on transient errors; creating a review never does."""
        if idempotent is None:
            idempotent = method in ("GET", "PUT", "DELETE")
        for attempt in range(RETRIES if idempotent else 1):
            try:
                return self._once(method, path, body)
            except GitHubError as exc:
                # A rate limit can be a 403 with Retry-After; a plain 403 is not.
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
        # Transport errors have no HTTP status but must still be GitHubError.
        except (OSError, http.client.HTTPException) as exc:
            raise GitHubError(0, repr(exc)) from exc
        try:
            return json.loads(raw) if raw else None
        except ValueError as exc:
            raise GitHubError(0, f"unparsable response: {exc}") from exc

    def graphql(self, query: str, variables: dict) -> dict:
        """One GraphQL call, retried also on errors reported inside an HTTP 200."""
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


# A line starting markdown block syntax; a mark in front would break it, so the
# mark goes on its own line.
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
    """The `major` findings, or all of them if changes are requested without one."""
    major = [f for f in findings if f["severity"] == "major"]
    if verdict == "changes_requested" and not major:
        return list(findings)
    return major


_FENCE_RUN = re.compile(r"`{3,}|~{3,}")


def contained(text: str) -> list[str]:
    """`text` as lines with every run of 3+ backticks or tildes escaped.

    A fenced code block is the only markdown that runs past the blank line
    after a message, so an unclosed one would swallow the rest of the body.
    Escaping is used rather than balancing, which depends on list and quote
    context and can get it wrong.
    """
    escaped = _FENCE_RUN.sub(lambda m: "".join("\\" + c for c in m.group()), text)
    return escaped.split("\n")


def permalink(repo: str, sha: str, finding: dict) -> str:
    """A blob link to the finding's line in the posting repo, which GitHub
    renders as the code. The path is percent-encoded, so it cannot carry `@`."""
    path = urllib.parse.quote(unescape_path(finding["path"]), safe="/")
    return f"https://github.com/{repo}/blob/{sha}/{path}#L{finding['line']}"


def finding_blocks(
    repo: str, sha: str, findings: list[dict], mark: str = ""
) -> list[str]:
    """Each finding as its code link, a blank line, then the message."""
    lines: list[str] = []
    for i, f in enumerate(findings):
        if i:
            lines += ["---", ""]
        message = contained(f["message"])
        if mark:
            message = marked(mark, message)
        lines += [permalink(repo, sha, f), "", *message, ""]
    return lines


# GitHub's limit is 65536; encoding and layout can exceed the sanitizer's caps.
MAX_BODY = 60_000


def review_body(
    verdict: str, summary: str, findings: list[dict], sha: str, repo: str, inline: bool
) -> str:
    """Verdict, summary, and the findings not posted inline (non-blocking ones
    folded). Over MAX_BODY, findings are dropped from the end and counted."""
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
    """Create the review. If GitHub refuses the inline comments (422), they go
    into the body; if it refuses REQUEST_CHANGES, the review is a COMMENT."""
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
    """This workflow's visible reviews older than `new_id`.

    Older, not just other, so overlapping runs never hide the newer review.
    Minimized reviews are skipped, so a thread someone reopened stays open.
    """
    mine = [
        r
        for r in gh.paged(f"/repos/{gh.repo}/pulls/{pr}/reviews")
        if isinstance(r.get("id"), int)
        and r["id"] < new_id
        and (r.get("user") or {}).get("login") == author
        and (r.get("body") or "").startswith(MARKER)
    ]
    # A review whose state did not come back is left alone.
    visible: set[str] = set()
    for i in range(0, len(mine), 100):
        ids = [r["node_id"] for r in mine[i : i + 100]]
        nodes = gh.graphql(_MINIMIZED, {"ids": ids}).get("nodes") or []
        visible |= {n["id"] for n in nodes if n and n.get("isMinimized") is False}
    return [r for r in mine if r["node_id"] in visible]


_MINIMIZE = "mutation($id: ID!) { minimizeComment(input: {subjectId: $id, classifier: OUTDATED}) { clientMutationId } }"


def retract(gh: GitHub, pr: int, review: dict) -> None:
    """Take back this run's review of a commit that is no longer the head.

    A change request is dismissed first, since a newer clean run may already
    have finished. If that fails, the review stays visible.
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
    """Resolve the review's threads, then minimize it as outdated.

    Its text and state are unchanged. If a thread cannot be resolved, the
    review stays visible for the next run to retry.
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
    """Resolve the open threads started by `comment_ids`; True if all are."""
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
    """Whether this run may dismiss a request: not on `keep_commit`, and older
    than `older_than`, so an overlapping run's newer request is never touched."""
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
    """Dismiss this workflow's standing change requests that are in scope."""
    standing = [
        r
        for r in standing_requests(gh, pr, author)
        if in_scope(r, keep_commit, older_than)
    ]
    # List, then re-read the head: a newer run's request is either not in the
    # list or makes the head check fail.
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
        # A run for the newer head will handle it.
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
        # Without our review id, spare this commit's requests (overlapping run).
        return clear_after_clean(gh, pr, sha, author, keep_commit=sha)
    if new_id is RETRACTED or not clean:
        return 0
    if new_id is None:
        return clear_after_clean(gh, pr, sha, author, keep_commit=sha)
    rc = clear_after_clean(gh, pr, sha, author, older_than=new_id)
    # Hide the requests just dismissed; any still standing stay visible.
    try:
        hide_earlier(gh, pr, author, {"id": new_id, "state": "COMMENTED"})
    except GitHubError as exc:
        warn(f"could not hide the withdrawn reviews: {exc}")
    return rc


def clear_after_clean(
    gh: GitHub, pr: int, sha: str, author: str, keep_commit=None, older_than=None
) -> int:
    """Dismiss earlier change requests after a clean verdict; 1 if any may remain.

    A COMMENT does not clear them, and after `ready for review` no later push
    is reviewed, so a leftover request fails the step, as does an API error.
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
    # A moved head means a newer run owns the PR.
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
    """Post the review and hide earlier ones. Returns the new id, None if GitHub
    gave none, or RETRACTED if the head moved while posting."""
    new = post(gh, pr, sha, verdict, summary, findings[:MAX_FINDINGS])
    print(f"posted automated review {new.get('id')}: {new.get('html_url')}")
    if not isinstance(new.get("id"), int):
        warn("GitHub returned no review id; leaving earlier reviews in place")
        return None
    # The id scopes the clean-verdict check, so cleanup errors must not lose it.
    try:
        return replace_earlier(gh, pr, sha, author, new)
    except GitHubError as exc:
        warn(f"posted review {new['id']}, but cleanup failed: {exc}")
        return new["id"]


def replace_earlier(gh, pr, sha, author, new):
    """Hide `new` if the head moved, else hide earlier reviews."""
    # Re-checked after posting: a newer run may already have posted a review
    # with a smaller id, which hiding by id would wrongly take down.
    current = head_sha(gh, pr)
    if current != sha:
        print(f"head moved {sha} -> {current} while posting; retracting this review")
        retract(gh, pr, new)
        return RETRACTED
    hide_earlier(gh, pr, author, new)
    return new["id"]


def hide_earlier(gh, pr, author, new):
    """Hide this workflow's reviews older than `new`.

    An earlier change request is hidden only if `new` requests changes too;
    otherwise it waits until `clear_after_clean` has dismissed it.
    """
    replaced = new.get("state") == "CHANGES_REQUESTED"
    # Newest first; one failure does not stop the rest.
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
