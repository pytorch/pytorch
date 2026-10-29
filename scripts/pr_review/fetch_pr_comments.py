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
passed in by the caller, is the commit under review, and ``previous_review_sha`` is the commit
the latest automated review saw (empty without one, or when it is ``head_sha``). Other bots and other users
are dropped. A comment's ``reactions`` are the emoji reactions of the author
and maintainers. They are read with GraphQL, one call per 100 comments, for
reacted comments and the latest automated review's body, up to
``MAX_REACTION_LOOKUPS`` of them, earlier findings first and newest first
within each; the reactors' permission checks are spent only on items
selection considers. ``reactions_truncated`` is set when a limit or a failed lookup
may have left some out of a kept item. Bodies are capped; the latest review's
body at ``MAX_REVIEW_BODY``. If the conversation is too long, ``truncated`` is
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
approve a PR whose maintainer feedback it was never shown. A reactions call is
tried ``REACTION_ATTEMPTS`` times, and each failure is logged as a workflow
warning.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable
from urllib.parse import quote


PER_PAGE = 100
MAX_PAGES = 30
MAX_BODY = 4000
# Of the latest automated review's body, which lists its non-inline findings.
MAX_REVIEW_BODY = 40_000
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
# REST listings carry a reaction count for these kinds only; a review's does not.
COUNTED = ("issue_comment", "review_comment")
# Items whose reactions are read, at most 100 per GraphQL call. Every call is
# on a token the repository's other workflows share.
MAX_REACTION_LOOKUPS = 300
REACTION_ATTEMPTS = 3
# GraphQL's ReactionContent, as REST spells it.
CONTENT = {
    "THUMBS_UP": "+1",
    "THUMBS_DOWN": "-1",
    "LAUGH": "laugh",
    "HOORAY": "hooray",
    "CONFUSED": "confused",
    "HEART": "heart",
    "ROCKET": "rocket",
    "EYES": "eyes",
}
_REACTIONS = "reactions(first: 100) { totalCount nodes { content user { login } } }"
REACTIONS_QUERY = (
    "query($ids: [ID!]!) { nodes(ids: $ids) { "
    f"... on IssueComment {{ id {_REACTIONS} }} "
    f"... on PullRequestReviewComment {{ id {_REACTIONS} }} "
    f"... on PullRequestReview {{ id {_REACTIONS} }} }} }}"
)
# Permission checks for reactors not already known from the conversation; a
# reactor beyond them is left out.
MAX_REACTOR_CHECKS = 30

# api(endpoint) is a REST GET; api("graphql", body) is a GraphQL call.
Api = Callable[..., Any]


class NotFound(RuntimeError):
    pass


def dumps(obj: Any) -> str:
    return json.dumps(obj, separators=(",", ":"))


def gh_api(endpoint: str, body: dict[str, Any] | None = None) -> Any:
    args = ["gh", "api", endpoint]
    if body is not None:
        args += ["--method", "POST", "--input", "-"]
    proc = subprocess.run(
        args,
        input=None if body is None else dumps(body),
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        err = proc.stderr.strip()
        if body is not None:
            # GraphQL reports per-node errors next to the data that resolved.
            try:
                out = json.loads(proc.stdout)
            except ValueError:
                out = None
            if isinstance(out, dict) and isinstance(out.get("data"), dict):
                return out
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


def warn(message: str) -> None:
    print(f"::warning::{message}", file=sys.stderr)


def read_reactions(
    api: Api, node_ids: list[str]
) -> tuple[dict[str, dict[str, Any]], set[str]]:
    """Each node's reaction connection, by node id, from one GraphQL call per
    100 ids, and the ids of nodes an error in the response may have left
    incomplete. A node whose call or lookup failed is absent."""
    out: dict[str, dict[str, Any]] = {}
    flawed: set[str] = set()
    for i in range(0, len(node_ids), 100):
        ids = node_ids[i : i + 100]
        for attempt in range(REACTION_ATTEMPTS):
            try:
                resp = api(
                    "graphql", {"query": REACTIONS_QUERY, "variables": {"ids": ids}}
                )
                nodes = ((resp or {}).get("data") or {}).get("nodes")
                if not isinstance(nodes, list):
                    raise ValueError("no nodes in the reactions response")
            except (RuntimeError, ValueError) as e:
                warn(f"could not read reactions (attempt {attempt + 1}): {e}")
                if attempt + 1 < REACTION_ATTEMPTS:
                    time.sleep(2**attempt)
                continue
            for err in resp.get("errors") or []:
                path = (err or {}).get("path") if isinstance(err, dict) else None
                n = path[1] if isinstance(path, list) and len(path) > 1 else None
                if path and path[0] == "nodes" and isinstance(n, int) and n < len(ids):
                    flawed.add(ids[n])
                else:  # an error not tied to one node may touch any of them
                    flawed.update(ids)
            if resp.get("errors"):
                warn(f"reactions response had errors: {dumps(resp['errors'])[:500]}")
            for n in nodes:
                if isinstance(n, dict) and isinstance(n.get("reactions"), dict):
                    out[n.get("id")] = n["reactions"]
            break
    return out, flawed


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
    # The commit the latest automated review saw, from which the review job builds the interdiff.
    # Empty without one, or when it is this head: a re-run on the same commit has nothing to compare.
    seen = (latest or {}).get("commit_id") or ""
    head["previous_review_sha"] = (
        seen if re.fullmatch(r"[0-9a-f]{40}", seen) and seen != head_sha else ""
    )
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
    lookups: dict[int, str | None] = {}
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
            reacted = (obj.get("reactions") or {}).get("total_count")
            if obj is latest or (kind in COUNTED and reacted):
                lookups[id(item)] = obj.get("node_id")
            items.append(item)

    items.sort(key=lambda i: i["created_at"])
    # Earlier findings first, so a reaction to one is never crowded out by
    # people's comments; newest first within each.
    wanted = sorted(
        (i for i in reversed(items) if id(i) in lookups),
        key=lambda i: i["role"] != AUTOMATED,
    )
    # Items whose reactions some limit or failure may have left out.
    partial = {id(i) for i in wanted[MAX_REACTION_LOOKUPS:]}
    wanted = wanted[:MAX_REACTION_LOOKUPS]
    found, flawed = read_reactions(
        api, [lookups[id(i)] for i in wanted if lookups[id(i)]]
    )
    resolved = {id(i) for i in items} - {id(i) for i in wanted}

    def resolve(item: dict[str, Any]) -> None:
        """Set the reactions of the author and maintainers on `item`. Called only
        once the item is a candidate for keeping, so the reactors' permission
        checks are not spent on items selection would leave out anyway."""
        nonlocal unchecked
        if id(item) in resolved:
            return
        resolved.add(id(item))
        node_id = lookups[id(item)]
        conn = found.get(node_id)
        if conn is None or node_id in flawed:
            partial.add(id(item))
        if conn is None:
            return
        nodes = conn.get("nodes") or []
        unchecked = False
        reactions = []
        try:
            for r in nodes:
                if not isinstance(r, dict):
                    partial.add(id(item))
                    continue
                login = (r.get("user") or {}).get("login")
                if not login or login in NON_HUMAN_LOGINS or login.endswith("[bot]"):
                    continue
                role = reactor_role(login)
                if role:
                    content = r.get("content")
                    reactions.append(
                        {
                            "author": login,
                            "role": role,
                            "content": CONTENT.get(content, str(content).lower()),
                        }
                    )
        except (RuntimeError, ValueError) as e:
            warn(f"could not check who reacted to {item['kind']} {item['id']}: {e}")
            partial.add(id(item))
            return
        if unchecked or (conn.get("totalCount") or 0) > len(nodes):
            partial.add(id(item))
        if reactions:
            item["reactions"] = reactions

    findings = {
        i["id"]: n
        for n, i in enumerate(items)
        if i["role"] == AUTOMATED and i["kind"] == "review_comment"
    }
    # The widest envelope: `false` and the largest possible count; +1 per comma.
    envelope = {
        **head,
        "truncated": False,
        "reactions_truncated": False,
        "omitted_items": len(items),
        "items": [],
    }
    size = len(dumps(envelope))
    kept: list[int] = []

    def take(n: int) -> bool:
        nonlocal size
        if n in kept:
            return True
        resolve(items[n])
        grow = len(dumps(items[n])) + 1
        if len(kept) == MAX_ITEMS or size + grow > MAX_TOTAL_BYTES:
            return False
        size += grow
        kept.append(n)
        return True

    newest_first = list(reversed(range(len(items))))
    # People's comments alone, cut strictly oldest first: the first that does not
    # fit ends selection, so a finding never takes a person's place.
    for n in newest_first:
        if items[n]["role"] != AUTOMATED and not take(n):
            break
    else:
        # Findings are candidates only now that every person's comment fit.
        for n in newest_first:
            if items[n]["role"] == AUTOMATED:
                resolve(items[n])
        # Answered: a finding with reactions or a reply people wrote, and the latest
        # review's body, whose findings are not inline and so cannot be replied to.
        replied = {i.get("in_reply_to_id") for i in items if i["role"] != AUTOMATED}
        answered = {
            id(i)
            for i in items
            if i["role"] == AUTOMATED
            and (i.get("reactions") or i["id"] in replied or i["kind"] == "review")
        }
        # Then earlier findings: first those a kept reply answers, then the
        # answered, then the rest, each newest first. One that does not fit is
        # skipped so that smaller ones can still be kept.
        replied_to = [findings.get(items[n].get("in_reply_to_id")) for n in kept]
        for n in [m for m in replied_to if m is not None] + [
            n for n in newest_first if id(items[n]) in answered
        ]:
            take(n)
        for n in newest_first:
            if items[n]["role"] == AUTOMATED:
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
