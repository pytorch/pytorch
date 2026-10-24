#!/usr/bin/env python3
"""Collect a pull request's title, description and conversation for the review.

Runs in the trusted `prepare` job, which holds GITHUB_TOKEN; the review job has
neither the token nor network access to the API. The file is handed over as a
`prepare` job output and rewritten in the review job from an environment
variable, so it is one line of compact JSON capped at ``MAX_TOTAL_BYTES``.

Kept: issue comments, reviews and inline review comments written by the PR
author or by a user with triage access or above, and this workflow's own
earlier findings (role ``automated_review``): its inline findings still on the
PR and the body of its latest review, which holds the findings it did not post
inline. post_review.py deletes superseded inline findings nobody replied or
reacted to, so those are normally not shown. Each inline comment's
``original_commit_id`` is the commit it was written against, and ``head_sha``,
passed in by the caller, is the commit under review. Other bots and other users
are dropped. A comment's ``reactions`` are the emoji reactions of the author
and maintainers, read for up to ``MAX_REACTION_LOOKUPS`` reacted comments,
earlier findings first and newest first within each (reactions to a review
body have no REST endpoint and are not read), and
``reactions_truncated`` is set when a limit or a failed lookup may have left
some out. Bodies are capped. If the conversation is too long, ``truncated`` is
set and items are selected in two phases. First, people's comments alone,
newest first; the first that does not fit ends selection, and older comments
and every finding are left out. Then, only if all of them fit, earlier
findings are considered in this order, each kept if it fits and skipped if
not: those a kept reply answers (in the order of the replies), then those with
a reaction seen and the latest review body, then the rest, each group newest
first. Kept items are written in time order.

Every failure is fatal except a 404 on a permission check (that user is not a
maintainer) and any failure while reading a comment's reactions, including the
reactors' permission checks: the comment is kept with ``reactions_truncated``
set. Reactions are context, while a review that silently saw no comments would
approve a PR whose maintainer feedback it was never shown.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable
from urllib.parse import quote


PER_PAGE = 100
MAX_PAGES = 30
MAX_BODY = 4000
# Of the latest automated review's body, which lists its non-inline findings.
MAX_REVIEW_BODY = 12_000
# Of the description once JSON-escaped; leaves most of MAX_TOTAL_BYTES to comments.
MAX_PR_BODY_BYTES = 40_000
MAX_HUNK = 1500
MAX_ITEMS = 200
# Of the whole serialized file. One environment variable is limited to 128 KiB
# (MAX_ARG_STRLEN) and a job output to 1 MB.
MAX_TOTAL_BYTES = 100_000
MAINTAINER_PERMISSIONS = ("triage", "push", "maintain", "admin")
# Machine accounts registered as plain users, some with write access.
NON_HUMAN_LOGINS = frozenset(
    {"facebook-github-bot", "pytorch-bot", "pytorchbot", "pytorchmergebot"}
)
# How post_review.py marks this workflow's reviews: their author and the first
# line of their body. Pinned to post_review.py's values by a test; the author
# can be overridden by the same REVIEW_AUTHOR environment variable.
REVIEW_AUTHOR = "github-actions[bot]"
MARKER = "<!-- hardened-pr-review -->"
AUTOMATED = "automated_review"
REACTIONS = {"issue_comment": "issues/comments", "review_comment": "pulls/comments"}
# Reactions are read for a bounded number of reacted comments, one page each: every
# lookup is an API call on a token the repository's other workflows share.
MAX_REACTION_LOOKUPS = 50
# Permission checks for reactors not already known from the conversation; a
# reactor beyond them is left out.
MAX_REACTOR_CHECKS = 30

Api = Callable[[str], Any]


class NotFound(RuntimeError):
    pass


def dumps(obj: Any) -> str:
    return json.dumps(obj, separators=(",", ":"))


def gh_api(endpoint: str) -> Any:
    proc = subprocess.run(
        ["gh", "api", endpoint], capture_output=True, text=True, check=False
    )
    if proc.returncode != 0:
        err = proc.stderr.strip()
        if "(HTTP 404)" in err:
            raise NotFound(f"{endpoint}: {err}")
        raise RuntimeError(f"gh api {endpoint} failed: {err}")
    return json.loads(proc.stdout)


def paginate(api: Api, endpoint: str) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for page in range(1, MAX_PAGES + 1):
        batch = api(f"{endpoint}?per_page={PER_PAGE}&page={page}")
        if not isinstance(batch, list):
            raise ValueError(f"{endpoint} page {page} is not a list")
        items += batch
        if len(batch) < PER_PAGE:
            return items
    raise ValueError(f"{endpoint} has more than {MAX_PAGES} pages")


def human_login(obj: dict[str, Any]) -> str | None:
    user = obj.get("user") or {}
    login = user.get("login")
    return (
        login if user.get("type") == "User" and login not in NON_HUMAN_LOGINS else None
    )


def is_maintainer(api: Api, repo: str, login: str) -> bool:
    try:
        resp = api(f"repos/{repo}/collaborators/{quote(login)}/permission")
    except NotFound:
        return False
    perms = (resp.get("user") or {}).get("permissions")
    if not isinstance(perms, dict):
        raise ValueError(f"no permissions in the response for {login}")
    return any(perms.get(p) is True for p in MAINTAINER_PERMISSIONS)


def cap(text: str | None, limit: int) -> str:
    text = text or ""
    return text if len(text) <= limit else text[:limit] + " [truncated]"


def cap_encoded(text: str | None, limit: int) -> str:
    """Cut by its encoded size: an astral character escapes to 12 bytes."""
    text, keep = text or "", len(text or "")
    while len(dumps(text[:keep])) > limit:
        keep = keep * 9 // 10
    return text if keep == len(text) else text[:keep] + " [truncated]"


def reacted(kind: str, obj: dict[str, Any]) -> bool:
    return kind in REACTIONS and bool((obj.get("reactions") or {}).get("total_count"))


def reactions_of(
    api: Api, repo: str, kind: str, obj: dict[str, Any], role_of
) -> tuple[list[dict[str, Any]], bool]:
    """The author's and maintainers' reactions to a comment, from its first
    page of reactions, and whether that page was full, so more may exist."""
    page = api(
        f"repos/{repo}/{REACTIONS[kind]}/{obj.get('id')}/reactions?per_page={PER_PAGE}"
    )
    if not isinstance(page, list):
        raise ValueError(f"reactions of {kind} {obj.get('id')} are not a list")
    out = []
    for r in page:
        login = human_login(r)
        role = role_of(login) if login else None
        if role:
            out.append({"author": login, "role": role, "content": r.get("content")})
    return out, len(page) >= PER_PAGE


def collect(api: Api, repo: str, pr: int, head_sha: str = "") -> dict[str, Any]:
    pull = api(f"repos/{repo}/pulls/{pr}")
    author = human_login(pull)
    body = cap_encoded(pull.get("body"), MAX_PR_BODY_BYTES)
    head = {
        "pr_author": author,
        "head_sha": head_sha,
        "title": pull.get("title") or "",
        "body": body,
    }
    sources = {
        "issue_comment": paginate(api, f"repos/{repo}/issues/{pr}/comments"),
        "review": paginate(api, f"repos/{repo}/pulls/{pr}/reviews"),
        "review_comment": paginate(api, f"repos/{repo}/pulls/{pr}/comments"),
    }
    bot = os.environ.get("REVIEW_AUTHOR", REVIEW_AUTHOR)
    marked = [
        r
        for r in sources["review"]
        if (r.get("user") or {}).get("login") == bot
        and (r.get("body") or "").startswith(MARKER)
    ]
    ours = {r.get("id") for r in marked}
    # By id, as post_review.py orders them: submitted_at ties within a second.
    latest = max(marked, key=lambda r: r.get("id") or 0, default=None)
    roles: dict[str, str | None] = {}

    def role_of(login: str) -> str | None:
        if login not in roles:
            if login == author:
                roles[login] = "author"
            else:
                roles[login] = "maintainer" if is_maintainer(api, repo, login) else None
        return roles[login]

    checks_left = MAX_REACTOR_CHECKS
    unchecked = False

    def reactor_role(login: str) -> str | None:
        nonlocal checks_left, unchecked
        if login in roles or login == author:
            return role_of(login)
        if checks_left == 0:
            unchecked = True
            return None
        checks_left -= 1
        return role_of(login)

    items = []
    lookups: dict[int, tuple[str, dict[str, Any]]] = {}
    for kind, raw in sources.items():
        for obj in raw:
            limit = MAX_BODY
            if obj is latest:
                login, role, limit = bot, AUTOMATED, MAX_REVIEW_BODY
            elif (
                kind == "review_comment"
                and (obj.get("user") or {}).get("login") == bot
                and obj.get("pull_request_review_id") in ours
            ):
                login, role = bot, AUTOMATED
            elif kind == "review" and obj.get("id") in ours:
                continue  # an older verdict, superseded by the latest
            else:
                login = human_login(obj)
                if login is None:
                    continue
                if (
                    kind == "review"
                    and not obj.get("body")
                    and obj.get("state") != "CHANGES_REQUESTED"
                ):
                    continue
                role = role_of(login)
                if role is None:
                    continue
            item = {
                "id": obj.get("id"),
                "kind": kind,
                "author": login,
                "role": role,
                "created_at": obj.get(
                    "submitted_at" if kind == "review" else "created_at"
                )
                or "",
                "body": cap(obj.get("body"), limit),
            }
            if kind == "review":
                item["state"] = obj.get("state")
            elif kind == "review_comment":
                item |= {
                    "path": obj.get("path"),
                    "line": obj.get("line"),
                    "original_line": obj.get("original_line"),
                    "in_reply_to_id": obj.get("in_reply_to_id"),
                    "pull_request_review_id": obj.get("pull_request_review_id"),
                    "original_commit_id": obj.get("original_commit_id"),
                    "diff_hunk": cap(obj.get("diff_hunk"), MAX_HUNK),
                }
            if reacted(kind, obj):
                lookups[id(item)] = (kind, obj)
            items.append(item)

    items.sort(key=lambda i: i["created_at"])
    # Earlier findings first, so a reaction to one is never crowded out by
    # people's comments; newest first within each.
    newest = sorted(
        (i for i in reversed(items) if id(i) in lookups),
        key=lambda i: i["role"] != AUTOMATED,
    )
    # Items whose reactions some limit may have left out.
    partial = {id(i) for i in newest[MAX_REACTION_LOOKUPS:]}
    for item in newest[:MAX_REACTION_LOOKUPS]:
        unchecked = False
        try:
            reactions, full = reactions_of(api, repo, *lookups[id(item)], reactor_role)
        except (RuntimeError, ValueError):
            # Deleted since it was listed, rate-limited, or any other failure:
            # keep the comment and report its reactions as unknown.
            reactions, full = [], True
        if full or unchecked:
            partial.add(id(item))
        if reactions:
            item["reactions"] = reactions
    # Answered: a finding with reactions or a reply people wrote, and the latest
    # review's body, whose findings are not inline and so cannot be answered.
    replied = {i.get("in_reply_to_id") for i in items if i["role"] != AUTOMATED}
    answered = {
        id(i)
        for i in items
        if i["role"] == AUTOMATED
        and (i.get("reactions") or i["id"] in replied or i["kind"] == "review")
    }
    newest_first = list(reversed(range(len(items))))
    groups = [
        [n for n in newest_first if items[n]["role"] != AUTOMATED],
        [n for n in newest_first if id(items[n]) in answered],
        [
            n
            for n in newest_first
            if items[n]["role"] == AUTOMATED and id(items[n]) not in answered
        ],
    ]
    kept: list[int] = []
    # The widest envelope: `false` and the largest possible count; +1 per comma.
    envelope = {
        **head,
        "truncated": False,
        "reactions_truncated": False,
        "omitted_items": len(items),
        "items": [],
    }
    size = len(dumps(envelope))
    findings = {
        i["id"]: n
        for n, i in enumerate(items)
        if i["role"] == AUTOMATED and i["kind"] == "review_comment"
    }

    def take(n: int) -> bool:
        nonlocal size
        if n in kept:
            return True
        grow = len(dumps(items[n])) + 1
        if len(kept) == MAX_ITEMS or size + grow > MAX_TOTAL_BYTES:
            return False
        size += grow
        kept.append(n)
        return True

    # People's comments alone, cut strictly oldest first, so a finding never
    # takes a person's place. Earlier findings follow only if every person's
    # comment fit: first those a kept reply answers, then the groups above, and
    # one that does not fit is skipped so that smaller ones can still be kept.
    if all(take(n) for n in groups[0]):
        replied_to = [findings.get(items[n].get("in_reply_to_id")) for n in kept]
        for n in [m for m in replied_to if m is not None] + groups[1] + groups[2]:
            take(n)
    omitted = len(items) - len(kept)
    return {
        **head,
        "truncated": omitted > 0,
        "reactions_truncated": any(id(items[n]) in partial for n in kept),
        "omitted_items": omitted,
        "items": [items[n] for n in sorted(kept)],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--pr", required=True, type=int)
    parser.add_argument("--head-sha", required=True, help="the commit under review")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        result = collect(gh_api, args.repo, args.pr, args.head_sha)
    except (RuntimeError, ValueError) as e:
        print(f"::error::could not fetch the PR conversation: {e}", file=sys.stderr)
        return 1
    args.out.write_text(dumps(result))
    print(f"kept {len(result['items'])} items, omitted {result['omitted_items']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
