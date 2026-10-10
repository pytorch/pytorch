#!/usr/bin/env python3
"""Collect a pull request's maintainer conversation for the hardened review.

Runs in the trusted `prepare` job, which holds GITHUB_TOKEN; the review job has
neither the token nor network access to the API. The file is handed over as a
`prepare` job output and rewritten in the review job from an environment
variable, so it is one line of compact JSON capped at ``MAX_TOTAL_BYTES``.

Kept: issue comments, reviews and inline review comments written by the PR
author or by a user with triage access or above. Bots and other users are
dropped. Bodies are capped and, if the conversation is too long, the oldest
items are dropped first and ``truncated`` is set.

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


def collect(api: Api, repo: str, pr: int) -> dict[str, Any]:
    author = human_login(api(f"repos/{repo}/pulls/{pr}"))
    sources = {
        "issue_comment": paginate(api, f"repos/{repo}/issues/{pr}/comments"),
        "review": paginate(api, f"repos/{repo}/pulls/{pr}/reviews"),
        "review_comment": paginate(api, f"repos/{repo}/pulls/{pr}/comments"),
    }
    roles: dict[str, str | None] = {}
    items = []
    for kind, raw in sources.items():
        for obj in raw:
            login = human_login(obj)
            if login is None:
                continue
            if (
                kind == "review"
                and not obj.get("body")
                and obj.get("state") != "CHANGES_REQUESTED"
            ):
                continue
            if login not in roles:
                if login == author:
                    roles[login] = "author"
                else:
                    roles[login] = (
                        "maintainer" if is_maintainer(api, repo, login) else None
                    )
            if roles[login] is None:
                continue
            item = {
                "id": obj.get("id"),
                "kind": kind,
                "author": login,
                "role": roles[login],
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
            items.append(item)

    items.sort(key=lambda i: i["created_at"])
    kept: list[dict[str, Any]] = []
    # The widest envelope: `false` and the largest possible count; +1 per comma.
    envelope = {
        "pr_author": author,
        "truncated": False,
        "omitted_items": len(items),
        "items": [],
    }
    size = len(dumps(envelope))
    for item in reversed(items):
        size += len(dumps(item)) + 1
        if len(kept) == MAX_ITEMS or size > MAX_TOTAL_BYTES:
            break
        kept.append(item)
    kept.reverse()
    omitted = len(items) - len(kept)
    return {
        "pr_author": author,
        "truncated": omitted > 0,
        "omitted_items": omitted,
        "items": kept,
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
