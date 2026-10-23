#!/usr/bin/env python3
"""Collect a pull request's title, description and conversation for the review.

Runs in the trusted `prepare` job, which holds GITHUB_TOKEN; the review job has
neither the token nor network access to the API. The file is handed over as a
`prepare` job output and rewritten in the review job from an environment
variable, so it is one line of compact JSON capped at ``MAX_TOTAL_BYTES``.

Kept: issue comments, reviews and inline review comments written by the PR
author or by a user with triage access or above, and this workflow's own
earlier inline findings still on the PR (role ``automated_review``), so the
replies to them have something to answer. post_review.py deletes superseded
findings nobody replied or reacted to, so those are not shown. Other bots and
other users are dropped. A comment's ``reactions`` are the emoji reactions of
the author and maintainers. Bodies are capped and, if the conversation is too
long, the oldest items are dropped first, earlier automated findings before
anything people wrote, and ``truncated`` is set.

Every failure is fatal. A review that silently saw no comments would approve a
PR whose maintainer feedback it was never shown.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable
from urllib.parse import quote


PER_PAGE = 100
MAX_PAGES = 30
MAX_BODY = 4000
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
# line of their body. Pinned to post_review.py's values by a test.
REVIEW_AUTHOR = "github-actions[bot]"
MARKER = "<!-- hardened-pr-review -->"
AUTOMATED = "automated_review"
REACTIONS = {"issue_comment": "issues/comments", "review_comment": "pulls/comments"}

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


def reactions_of(
    api: Api, repo: str, kind: str, obj: dict[str, Any], role_of
) -> list[dict[str, Any]]:
    """The author's and maintainers' reactions to a comment."""
    if kind not in REACTIONS or not (obj.get("reactions") or {}).get("total_count"):
        return []
    out = []
    for r in paginate(api, f"repos/{repo}/{REACTIONS[kind]}/{obj.get('id')}/reactions"):
        login = human_login(r)
        role = role_of(login) if login else None
        if role:
            out.append({"author": login, "role": role, "content": r.get("content")})
    return out


def collect(api: Api, repo: str, pr: int) -> dict[str, Any]:
    pull = api(f"repos/{repo}/pulls/{pr}")
    author = human_login(pull)
    body = cap_encoded(pull.get("body"), MAX_PR_BODY_BYTES)
    head = {"pr_author": author, "title": pull.get("title") or "", "body": body}
    sources = {
        "issue_comment": paginate(api, f"repos/{repo}/issues/{pr}/comments"),
        "review": paginate(api, f"repos/{repo}/pulls/{pr}/reviews"),
        "review_comment": paginate(api, f"repos/{repo}/pulls/{pr}/comments"),
    }
    ours = {
        r.get("id")
        for r in sources["review"]
        if (r.get("user") or {}).get("login") == REVIEW_AUTHOR
        and (r.get("body") or "").startswith(MARKER)
    }
    roles: dict[str, str | None] = {}

    def role_of(login: str) -> str | None:
        if login not in roles:
            if login == author:
                roles[login] = "author"
            else:
                roles[login] = "maintainer" if is_maintainer(api, repo, login) else None
        return roles[login]

    items = []
    for kind, raw in sources.items():
        for obj in raw:
            if (
                kind == "review_comment"
                and (obj.get("user") or {}).get("login") == REVIEW_AUTHOR
                and obj.get("pull_request_review_id") in ours
            ):
                login, role = REVIEW_AUTHOR, AUTOMATED
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
                "body": cap(obj.get("body"), MAX_BODY),
            }
            if kind == "review":
                item["state"] = obj.get("state")
            elif kind == "review_comment":
                item |= {
                    "path": obj.get("path"),
                    "line": obj.get("line"),
                    "original_line": obj.get("original_line"),
                    "in_reply_to_id": obj.get("in_reply_to_id"),
                    "diff_hunk": cap(obj.get("diff_hunk"), MAX_HUNK),
                }
            reactions = reactions_of(api, repo, kind, obj, role_of)
            if reactions:
                item["reactions"] = reactions
            items.append(item)

    items.sort(key=lambda i: i["created_at"])
    kept: list[int] = []
    # The widest envelope: `false` and the largest possible count; +1 per comma.
    envelope = {
        **head,
        "truncated": False,
        "omitted_items": len(items),
        "items": [],
    }
    size = len(dumps(envelope))
    # Newest first, what people wrote before earlier automated findings.
    for automated in (False, True):
        for n in reversed(range(len(items))):
            if (items[n]["role"] == AUTOMATED) != automated:
                continue
            grow = len(dumps(items[n])) + 1
            if len(kept) == MAX_ITEMS or size + grow > MAX_TOTAL_BYTES:
                break
            size += grow
            kept.append(n)
    omitted = len(items) - len(kept)
    return {
        **head,
        "truncated": omitted > 0,
        "omitted_items": omitted,
        "items": [items[n] for n in sorted(kept)],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--pr", required=True, type=int)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        result = collect(gh_api, args.repo, args.pr)
    except (RuntimeError, ValueError) as e:
        print(f"::error::could not fetch the PR conversation: {e}", file=sys.stderr)
        return 1
    args.out.write_text(dumps(result))
    print(f"kept {len(result['items'])} items, omitted {result['omitted_items']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
