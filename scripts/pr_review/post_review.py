#!/usr/bin/env python3
"""Post the hardened PR review's verdict as a native GitHub pull-request review.

A review notifies the PR author and puts findings on their lines.

`--row-file` posts the terminal row (terminal.json) that emit_row.py already
checked. Every string is checked again with the sanitizer's predicates; anything
that fails is dropped, never repaired. `--mark-pending` and `--restore-pending`
take no row: they mark the previous review in progress for RUN_ID, or put it
back (see Invariants).

Invariants:
- A `changes_requested` verdict is REQUEST_CHANGES; anything else is COMMENT,
  never APPROVE.
- A new review is posted before earlier ones are hidden. Hiding folds the
  review's body into a collapsed section, deletes its inline comments that
  nobody replied to, resolves the replied threads where it can, and minimizes
  it as outdated. Minimizing alone leaves the inline comments listed on the
  Files tab and in GitHub Mobile, hence the deleting. A change request is
  hidden only once it no longer stands.
- A blocking finding next to an earlier inline comment someone replied to
  goes in the body with a link to that thread, not on the line again. The
  match is by position only, so the link says where, not that it is the same
  finding.
- A newer REQUEST_CHANGES replaces an older one in the review decision. A
  COMMENT does not, so after a clean verdict earlier requests are dismissed.
  The only other dismissal is a run retracting its own review because the
  head moved.
- While a run reviews, the newest visible review is marked in progress, its
  body collapsed under a status line (`--mark-pending`). A run that ends
  without replacing it, failed, skipped or opted out, puts the body back
  (`--restore-pending`). Another run's mark is left alone, though with no
  compare-and-set on review edits two runs writing at once can still clash.
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


class OutOfTime(GitHubError):
    """`GitHub.deadline` passed; the step stops rather than skipping a call."""

    def __init__(self):
        super().__init__(0, "out of time for this step")


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
        # Every comment any listed reply in this run answers: such a comment is
        # never deleted, even if a later listing misses the reply. One of the
        # deletion gates, with reactions and the thread's comment count.
        self.replies_seen: set[int] = set()
        # When set (time.monotonic()), no call starts or waits past it.
        self.deadline: float | None = None

    def request(
        self, method: str, path: str, body: dict | None = None, idempotent=None
    ):
        """One API call. Idempotent calls (GET, PUT, DELETE, our GraphQL) retry
        on transient errors; creating a review never does."""
        if idempotent is None:
            idempotent = method in ("GET", "PUT", "DELETE")
        for attempt in range(RETRIES if idempotent else 1):
            if self.deadline is not None and time.monotonic() > self.deadline:
                raise OutOfTime()
            try:
                return self._once(method, path, body)
            except GitHubError as exc:
                # A rate limit can be a 403 with Retry-After; a plain 403 is not.
                throttled = exc.retry_after is not None
                transient = exc.status in (0, 429) or exc.status >= 500 or throttled
                if not transient or attempt == RETRIES - 1 or not idempotent:
                    raise
                wait = max(
                    2**attempt, min(exc.retry_after, MAX_WAIT) if throttled else 0
                )
                if (
                    self.deadline is not None
                    and time.monotonic() + wait > self.deadline
                ):
                    raise OutOfTime() from exc
                _sleep(wait)

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
            if (
                self.deadline is not None
                and time.monotonic() + 2**attempt > self.deadline
            ):
                raise OutOfTime()
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
    """Each finding as its code link, a blank line, then the message, and the
    earlier discussion of it if there is one."""
    lines: list[str] = []
    for i, f in enumerate(findings):
        if i:
            lines += ["---", ""]
        message = contained(f["message"])
        if mark:
            message = marked(mark, message)
        lines += [permalink(repo, sha, f), "", *message, ""]
        if f.get("discussion"):
            lines += [
                f"An earlier comment near this line has replies: {f['discussion']}",
                "",
            ]
    return lines


# GitHub's limit is 65536; encoding and layout can exceed the sanitizer's caps.
MAX_BODY = 60_000


def review_body(
    verdict: str, summary: str, findings: list[dict], sha: str, repo: str, inline: bool
) -> str:
    """Verdict, summary, and the findings not posted inline (non-blocking ones
    folded). Over MAX_BODY, findings are dropped from the end and counted."""
    blocking = blocking_findings(verdict, findings)
    in_body = [f for f in blocking if f.get("discussion")] if inline else blocking
    folded = [f for f in findings if f not in blocking]
    keep_body, keep_folded = len(in_body), len(folded)
    while True:
        body = _compose(
            verdict,
            summary,
            sha,
            repo,
            len(blocking) - len(in_body) if inline else 0,
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


def _compose(verdict, summary, sha, repo, n_inline, in_body, folded, omitted) -> str:
    lines = [
        MARKER,
        f"**Automated review** of {sha[:12]}: {_VERDICT_LINE[verdict]}.",
        "",
        *contained(summary),
        "",
    ]
    if n_inline:
        lines += [
            f"{BLOCKING_MARK} {count(n_inline, 'blocking finding')} "
            f"{'is' if n_inline == 1 else 'are'} attached to the code below.",
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
        if not f.get("discussion")
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


def earlier_reviews(
    gh: GitHub, pr: int, author: str, new_id: int, listing=None
) -> tuple[list[dict], list[dict]]:
    """This workflow's reviews older than `new_id`: (visible, minimized).

    Older, not just other, so overlapping runs never hide the newer review.
    A review whose state did not come back is in neither list.
    """
    if listing is None:
        listing = gh.paged(f"/repos/{gh.repo}/pulls/{pr}/reviews")
    mine = [
        r
        for r in listing
        if isinstance(r.get("id"), int)
        and r["id"] < new_id
        and (r.get("user") or {}).get("login") == author
        and (r.get("body") or "").startswith(MARKER)
    ]
    state: dict[str, bool] = {}
    for i in range(0, len(mine), 100):
        ids = [r["node_id"] for r in mine[i : i + 100]]
        nodes = gh.graphql(_MINIMIZED, {"ids": ids}).get("nodes") or []
        state |= {
            n["id"]: n["isMinimized"]
            for n in nodes
            if n and isinstance(n.get("isMinimized"), bool)
        }
    visible = [r for r in mine if state.get(r["node_id"]) is False]
    hidden = [r for r in mine if state.get(r["node_id"]) is True]
    return visible, hidden


def pr_comments(gh: GitHub, pr: int) -> list[dict] | None:
    """Every inline comment on the PR, or None if they could not be listed.

    The comment each reply answers is recorded in `gh.replies_seen` as its
    page arrives, so a reply seen before a later page fails still protects it.
    """
    comments: list[dict] = []
    page = 1
    try:
        while True:
            batch = (
                gh.request(
                    "GET",
                    f"/repos/{gh.repo}/pulls/{pr}/comments?per_page=100&page={page}",
                )
                or []
            )
            gh.replies_seen |= {
                c["in_reply_to_id"] for c in batch if c.get("in_reply_to_id")
            }
            comments.extend(batch)
            if len(batch) < 100:
                return comments
            page += 1
    except OutOfTime:
        raise
    except GitHubError as exc:
        warn(f"could not list the PR's inline comments: {exc}")
        return None


# How far a new finding may sit from an earlier discussed one and still be the
# same place.
NEARBY = 3


def ours(gh: GitHub, pr: int, author: str) -> set[int]:
    """The ids of every review this workflow posted on the PR."""
    return {
        r["id"]
        for r in gh.paged(f"/repos/{gh.repo}/pulls/{pr}/reviews")
        if (r.get("user") or {}).get("login") == author
        and (r.get("body") or "").startswith(MARKER)
    }


def with_discussions(gh: GitHub, pr: int, author: str, findings: list[dict]):
    """`findings`, each linked to an earlier finding of ours next to it that
    someone replied to, so the reply is not answered by a fresh comment on the
    line. Unchanged if the PR could not be read."""
    if not findings:
        return findings
    comments = pr_comments(gh, pr)
    if not comments:
        return findings
    try:
        reviews = ours(gh, pr, author)
    except GitHubError as exc:
        warn(f"could not list earlier reviews: {exc}")
        return findings
    return note_discussions(findings, comments, reviews, gh.replies_seen)


def note_discussions(findings: list[dict], comments, reviews: set[int], replied):
    discussed = [
        c
        for c in comments
        if c.get("id") in replied
        and c.get("pull_request_review_id") in reviews
        and isinstance(c.get("line"), int)  # None once its line has changed
    ]
    out = []
    for f in findings:
        near = [
            c
            for c in discussed
            if c.get("path") == unescape_path(f["path"])
            and abs(c["line"] - f["line"]) <= NEARBY
        ]
        url = (near[0].get("html_url") or "") if near else ""
        if url.startswith("https://github.com/"):
            f = {**f, "discussion": url}
        out.append(f)
    return out


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
    take_down(gh, pr, review, reason="withdrawn: the PR changed while it was posted")


SUPERSEDED = "a newer one replaces it"


def take_down(
    gh: GitHub, pr: int, review: dict, comments=None, reason=SUPERSEDED
) -> None:
    """Fold the review, delete its unreplied comments, resolve its replied
    threads, then minimize it as outdated. Its state is unchanged.

    Each step is attempted even if an earlier one fails. GITHUB_TOKEN is
    refused resolveReviewThread on pytorch/pytorch, so replied threads stay
    open, and once the review is minimized no later run retries them.
    """
    rid = review["id"]
    fold(gh, pr, review, reason)
    if comments is None:
        comments = pr_comments(gh, pr)
    kept = delete_unreplied(gh, pr, rid, comments)
    try:
        resolved = kept is not None and resolve_threads(gh, pr, kept)
    except GitHubError as exc:
        warn(f"could not resolve the threads of review {rid}: {exc}")
        resolved = False
    if not resolved:
        warn(f"hiding review {rid} with some of its threads still open")
    try:
        gh.graphql(_MINIMIZE, {"id": review["node_id"]})
        print(f"hid earlier automated review {rid} as outdated")
    except GitHubError as exc:
        warn(f"could not minimize review {rid}: {exc}")


FOLDED_REVIEW = "<details><summary>Outdated automated review"


def folded(review: dict, reason: str = SUPERSEDED) -> str:
    """The review's body inside a collapsed section, marker still first."""
    rest = unwrapped(review.get("body") or "")[len(MARKER) :].lstrip("\n")
    sha = str(review.get("commit_id") or "")[:12]
    # Defused again: the body may come from an older version of this script,
    # and pytorch-bot reads edited review bodies.
    return defuse_bot_commands(
        "\n".join(
            [
                MARKER,
                f"{FOLDED_REVIEW} of {sha}; {reason}</summary>",
                "",
                rest,
                "",
                "</details>",
            ]
        )
    )


# A review marked in progress keeps its own body collapsed under a status line.
# The tag names the run that marked it: only that run puts the body back.
# `neutralize` escapes `<` in model prose, so no summary can forge the tag.
PENDING_TAG = "<!-- hardened-pr-review-pending run="
_PENDING_LINE = re.compile(r"<!-- hardened-pr-review-pending run=(\d+) -->")
PREVIOUS_REVIEW = "<details><summary>Previous automated review"


def pending_run(body: str) -> str | None:
    """The run a review is marked in progress for, or None."""
    lines = (body or "").replace("\r\n", "\n").split("\n")
    m = len(lines) > 1 and lines[0] == MARKER and _PENDING_LINE.fullmatch(lines[1])
    return m.group(1) if m else None


def unwrapped(body: str) -> str:
    """The body as it was before it was marked in progress. A marked body that
    does not have the shape `pending` writes comes back unchanged."""
    if pending_run(body) is None:
        return body
    lines = body.replace("\r\n", "\n").split("\n")
    if (
        len(lines) < 9
        or not lines[4].startswith(PREVIOUS_REVIEW)
        or lines[5:6] != [""]
        or lines[-2:] != ["", "</details>"]
    ):
        return body
    return "\n".join([MARKER, *lines[6:-2]])


def pending(review: dict, run_id: str, sha: str, repo: str) -> str:
    """`review` marked in progress for `run_id`, which reviews `sha`."""
    rest = unwrapped(review.get("body") or "")[len(MARKER) :].lstrip("\n")
    old = str(review.get("commit_id") or "")[:12]
    return defuse_bot_commands(
        "\n".join(
            [
                MARKER,
                f"{PENDING_TAG}{run_id} -->",
                f"\u23f3 **Automated review in progress** for {sha[:12]} "
                f"([run](https://github.com/{repo}/actions/runs/{run_id})); "
                "its result will replace this review.",
                "",
                f"{PREVIOUS_REVIEW} of {old}</summary>",
                "",
                rest,
                "",
                "</details>",
            ]
        )
    )


def mark_pending(
    gh: GitHub, pr: int, sha: str, author: str, run_id: str, opt_out: str
) -> None:
    """Mark the newest visible review of this workflow in progress for `run_id`.

    A mark by a newer run (run ids only grow) is left alone, so a delayed older
    run cannot take it over and then put the body back while the newer runs.
    """
    if head_sha(gh, pr) != sha:
        print("head moved; the run for the newer head marks the review")
        return
    if opted_out(gh, pr, opt_out):
        print(f"PR carries '{opt_out}'; not marking")
        return
    visible, _ = earlier_reviews(gh, pr, author, sys.maxsize)
    if not visible:
        print("no visible automated review to mark")
        return
    newest = max(visible, key=lambda r: r["id"])
    body = newest.get("body") or ""
    owner = pending_run(body)
    if body.startswith(f"{MARKER}\n{FOLDED_REVIEW}") or (
        owner is not None and int(owner) >= int(run_id)
    ):
        return
    gh.request(
        "PUT",
        f"/repos/{gh.repo}/pulls/{pr}/reviews/{newest['id']}",
        {"body": pending(newest, run_id, sha, gh.repo)},
    )
    print(f"marked automated review {newest['id']} in progress")


def restore_pending(gh: GitHub, pr: int, author: str, run_id: str) -> None:
    """Put back the body of every review still marked in progress for `run_id`.

    A review this run replaced was folded, which drops the mark, so what is
    left is a review this run did not replace.
    """
    for r in gh.paged(f"/repos/{gh.repo}/pulls/{pr}/reviews"):
        body = r.get("body") or ""
        if (r.get("user") or {}).get("login") != author or pending_run(body) != run_id:
            continue
        original = unwrapped(body)
        if original == body:
            warn(f"review {r['id']} is marked in progress in a shape not ours")
            continue
        try:
            gh.request(
                "PUT",
                f"/repos/{gh.repo}/pulls/{pr}/reviews/{r['id']}",
                {"body": defuse_bot_commands(original)},
            )
            print(f"put back automated review {r['id']}")
        except GitHubError as exc:
            warn(f"could not put back review {r['id']}: {exc}")


def fold(
    gh: GitHub, pr: int, review: dict, reason: str = SUPERSEDED, budget=None
) -> None:
    """Fold the review's body unless it is folded already."""
    body = review.get("body") or ""
    if not body.startswith(MARKER) or body.startswith(f"{MARKER}\n{FOLDED_REVIEW}"):
        return
    if budget is not None and not budget.take():
        return
    try:
        gh.request(
            "PUT",
            f"/repos/{gh.repo}/pulls/{pr}/reviews/{review['id']}",
            {"body": folded(review, reason)},
        )
    except OutOfTime:
        raise
    except GitHubError as exc:
        warn(f"could not fold review {review['id']}: {exc}")


def reacted(comment: dict) -> bool:
    """Whether anyone reacted to the comment: feedback that deleting it would lose."""
    return bool((comment.get("reactions") or {}).get("total_count"))


class Budget:
    """API writes a step may still start; None is unlimited. With `pause`,
    each write after the first waits that long, as GitHub asks of clients
    making many mutating requests."""

    def __init__(self, left: int | None = None, pause: float = 0):
        self.left = left
        self.pause = pause
        self.taken = 0

    def take(self) -> bool:
        if self.left is not None:
            if self.left <= 0:
                return False
            self.left -= 1
        if self.pause and self.taken:
            _sleep(self.pause)
        self.taken += 1
        return True

    def spent(self) -> bool:
        return self.left is not None and self.left <= 0


def delete_unreplied(
    gh: GitHub,
    pr: int,
    rid: int,
    comments,
    budget: Budget | None = None,
    refresh=True,
    sizes=None,
) -> set[int] | None:
    """Delete review `rid`'s inline comments nobody replied to.

    Returns the ids of the comments kept, or None if the PR's comments could
    not be listed. Replies stay, and so does a comment someone answered or
    reacted to; a reply any listing in this run saw protects its comment. A
    reply still pending in someone's unsubmitted review is invisible to the
    API and protects nothing. A comment is deleted only if its thread, read
    by cursor (`sizes`, from `thread_sizes`), holds that one comment. With
    `refresh` the comments are listed again just before deleting, which
    narrows the window for a reply landing first but cannot close it. Each
    delete spends `budget`.
    """
    budget = budget or Budget()

    def roots_of(listing):
        return [
            c
            for c in listing
            if c.get("pull_request_review_id") == rid and not c.get("in_reply_to_id")
        ]

    if comments is None:
        return None
    roots = roots_of(comments)
    if all(c["id"] in gh.replies_seen or reacted(c) for c in roots):
        return {c["id"] for c in roots}
    fresh = pr_comments(gh, pr) if refresh else comments
    if fresh is None:
        return None
    if sizes is None:
        sizes = thread_sizes(gh, pr)
        if sizes is None:
            return {c["id"] for c in roots_of(fresh)}
    kept: set[int] = set()
    for c in roots_of(fresh):
        alone = sizes.get(c["id"]) == 1
        if not alone or c["id"] in gh.replies_seen or reacted(c) or not budget.take():
            kept.add(c["id"])
            continue
        try:
            gh.request("DELETE", f"/repos/{gh.repo}/pulls/comments/{c['id']}")
        except OutOfTime:
            raise
        except GitHubError as exc:
            if exc.status != 404:  # an overlapping run deleted it first
                warn(f"could not delete comment {c['id']} of review {rid}: {exc}")
                kept.add(c["id"])
    return kept


# Writes the backfill may start in one run, and for how long; the rest waits
# for the next run. Retries of a throttled call can outlast the count, hence
# the clock, which every call (and its retry waits) honours.
MAX_BACKFILL = 50
BACKFILL_SECONDS = 90
BACKFILL_PAUSE = 1.0


def backfill(gh: GitHub, pr: int, author: str, new_id: int) -> None:
    """Fold and empty this workflow's reviews hidden before folding and
    deleting existed, or by a run whose deletes failed.

    Runs last, after any change request has been dealt with, and spends at
    most MAX_BACKFILL writes and BACKFILL_SECONDS. The comments are listed
    once, just before the first delete. Threads are not touched: one reopened
    stays open. A change request that may still stand is skipped.
    """
    gh.deadline = time.monotonic() + BACKFILL_SECONDS
    try:
        _backfill(gh, pr, author, new_id)
    except GitHubError as exc:
        warn(f"stopped tidying earlier reviews: {exc}")
    finally:
        gh.deadline = None


def _backfill(gh: GitHub, pr: int, author: str, new_id: int) -> None:
    listing = gh.paged(f"/repos/{gh.repo}/pulls/{pr}/reviews")
    _, hidden = earlier_reviews(gh, pr, author, new_id, listing)
    replaced = latest_is_request(listing, author)
    budget = Budget(MAX_BACKFILL, BACKFILL_PAUSE)
    comments = sizes = None
    for old in sorted(hidden, key=lambda r: r["id"], reverse=True):
        if old.get("state") == "CHANGES_REQUESTED" and not replaced:
            continue
        if budget.spent():
            print("more earlier reviews to tidy; the next run continues")
            return
        fold(gh, pr, old, budget=budget)
        if comments is None:
            comments = pr_comments(gh, pr)
            sizes = thread_sizes(gh, pr)
            if comments is None or sizes is None:
                return
        delete_unreplied(gh, pr, old["id"], comments, budget, False, sizes)


def latest_is_request(listing: list[dict], author: str) -> bool:
    """Whether this workflow's newest review requests changes, so that it, not
    any earlier one, is what stands."""
    mine = [
        r
        for r in listing
        if (r.get("user") or {}).get("login") == author
        and (r.get("body") or "").startswith(MARKER)
        and isinstance(r.get("id"), int)
    ]
    return bool(mine) and max(mine, key=lambda r: r["id"]).get("state") == (
        "CHANGES_REQUESTED"
    )


_THREADS = """query($o: String!, $r: String!, $n: Int!, $c: String) {
  repository(owner: $o, name: $r) { pullRequest(number: $n) {
    reviewThreads(first: 100, after: $c) {
      pageInfo { hasNextPage endCursor }
      nodes { id isResolved comments(first: 1) { totalCount nodes { databaseId } } }
    } } } }"""
_RESOLVE = "mutation($id: ID!) { resolveReviewThread(input: {threadId: $id}) { clientMutationId } }"


def thread_sizes(gh: GitHub, pr: int) -> dict[int, int] | None:
    """Comments per review thread, keyed by the thread's first comment, or None
    if the threads could not be read. Cursor pagination, unlike the REST
    listing's page offsets, does not skip a thread when an overlapping run
    deletes comments meanwhile."""
    owner, name = gh.repo.split("/")
    sizes: dict[int, int] = {}
    cursor = None
    try:
        while True:
            data = gh.graphql(_THREADS, {"o": owner, "r": name, "n": pr, "c": cursor})
            threads = data["repository"]["pullRequest"]["reviewThreads"]
            for node in threads["nodes"]:
                comments = node["comments"]
                if comments["nodes"]:
                    sizes[comments["nodes"][0]["databaseId"]] = comments["totalCount"]
            if not threads["pageInfo"]["hasNextPage"]:
                return sizes
            cursor = threads["pageInfo"]["endCursor"]
    except OutOfTime:
        raise
    except (GitHubError, KeyError, TypeError) as exc:
        warn(f"could not read the PR's review threads: {exc!r}")
        return None


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


def opted_out(gh: GitHub, pr: int, opt_out: str) -> bool:
    labels = (
        gh.request("GET", f"/repos/{gh.repo}/issues/{pr}/labels?per_page=100") or []
    )
    return any(str(lb.get("name", "")).lower() == opt_out.lower() for lb in labels)


def run(gh: GitHub, row: dict, pr: int, sha: str, opt_out: str, author: str) -> int:
    review = load_review(row)
    current = head_sha(gh, pr)
    if current != sha:
        # A run for the newer head will handle it.
        print(f"head moved {sha} -> {current}; not posting a stale review")
        return 0
    if opted_out(gh, pr, opt_out):
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
    if new_id is RETRACTED:
        return 0
    if new_id is None:
        return clear_after_clean(gh, pr, sha, author, keep_commit=sha) if clean else 0
    rc = 0
    if clean:
        rc = clear_after_clean(gh, pr, sha, author, older_than=new_id)
        # Hide the requests just dismissed; any still standing stay visible.
        try:
            hide_earlier(gh, pr, author, {"id": new_id, "state": "COMMENTED"})
        except GitHubError as exc:
            warn(f"could not hide the withdrawn reviews: {exc}")
    backfill(gh, pr, author, new_id)
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
    findings = with_discussions(gh, pr, author, findings[:MAX_FINDINGS])
    new = post(gh, pr, sha, verdict, summary, findings)
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
    listing = gh.paged(f"/repos/{gh.repo}/pulls/{pr}/reviews")
    # Its state now, not when posted: an overlapping run may have dismissed it.
    now = next((r for r in listing if r.get("id") == new["id"]), new)
    replaced = now.get("state") == "CHANGES_REQUESTED"
    visible, _ = earlier_reviews(gh, pr, author, new["id"], listing)
    visible = [r for r in visible if r.get("state") != "CHANGES_REQUESTED" or replaced]
    comments = pr_comments(gh, pr) if visible else None
    # Newest first; one failure does not stop the rest.
    for old in sorted(visible, key=lambda r: r["id"], reverse=True):
        try:
            take_down(gh, pr, old, comments)
        except GitHubError as exc:
            warn(f"could not hide review {old['id']}: {exc}")


def main() -> int:
    ap = argparse.ArgumentParser()
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--row-file", help="terminal.json from emit_row.py")
    mode.add_argument(
        "--mark-pending",
        action="store_true",
        help="mark the newest automated review in progress for RUN_ID",
    )
    mode.add_argument(
        "--restore-pending",
        action="store_true",
        help="put back reviews still marked in progress for RUN_ID",
    )
    args = ap.parse_args()
    try:
        gh = GitHub(os.environ["GH_TOKEN"], os.environ["REPO"])
        if args.mark_pending or args.restore_pending:
            run_id = os.environ["RUN_ID"]
            if not run_id.isdigit():
                raise ValueError(f"RUN_ID is not a run id: {run_id!r}")
            pr = int(os.environ["PR_NUMBER"])
            author = os.environ.get("REVIEW_AUTHOR", DEFAULT_AUTHOR)
            if args.mark_pending:
                mark_pending(
                    gh,
                    pr,
                    os.environ["REVIEWED_SHA"],
                    author,
                    run_id,
                    os.environ.get("OPT_OUT_LABEL", "no automated review"),
                )
            else:
                restore_pending(gh, pr, author, run_id)
            return 0
        row = json.loads(Path(args.row_file).read_text())
        return run(
            gh,
            row if isinstance(row, dict) else {},
            int(os.environ["PR_NUMBER"]),
            os.environ["REVIEWED_SHA"],
            os.environ.get("OPT_OUT_LABEL", "no automated review"),
            os.environ.get("REVIEW_AUTHOR", DEFAULT_AUTHOR),
        )
    except Exception as exc:  # noqa: BLE001 - an API failure never fails the job
        what = (
            "posted" if args.row_file else "marked" if args.mark_pending else "put back"
        )
        warn(f"automated review not {what}: {exc!r}")
        return 0


if __name__ == "__main__":
    sys.exit(main())
