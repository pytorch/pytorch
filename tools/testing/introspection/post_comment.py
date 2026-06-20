"""Render a test-diff result and post/update the sticky PR comment.

Shared by both comment paths (the workflow_run job for forks, and the temporary
same-run job used to test on branch PRs before landing), so the comment logic lives
in one tested place and the workflows only differ in how they obtain the artifact.

    python -m tools.testing.introspection.post_comment test-diff-result.json

Env: GITHUB_TOKEN, GITHUB_REPOSITORY, optional GITHUB_API_URL. Reads the PR number
from the result JSON (written by ci_run), so it works for workflow_run too where the
event carries no PR number. Uses urllib (no `gh` CLI dependency).
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from urllib.request import Request, urlopen

from tools.testing.introspection import render_comment


def _api(method: str, url: str, token: str, data: dict | None = None) -> object:
    body = json.dumps(data).encode() if data is not None else None
    req = Request(url, data=body, method=method)
    req.add_header("Authorization", f"token {token}")
    req.add_header("Accept", "application/vnd.github+json")
    if data is not None:
        req.add_header("Content-Type", "application/json")
    with urlopen(req) as resp:
        return json.loads(resp.read().decode())


def _find_sticky(api: str, repo: str, pr: int, token: str) -> int | None:
    page = 1
    while True:
        url = f"{api}/repos/{repo}/issues/{pr}/comments?per_page=100&page={page}"
        comments = _api("GET", url, token)
        if not comments:
            return None
        for c in comments:
            if render_comment.MARKER in (c.get("body") or ""):
                return int(c["id"])
        page += 1


def post(result: dict) -> None:
    if result.get("skipped"):
        print("compute stage reported no test-relevant changes; nothing to post")
        return
    token = os.environ["GITHUB_TOKEN"]
    repo = os.environ["GITHUB_REPOSITORY"]
    api = os.environ.get("GITHUB_API_URL", "https://api.github.com")
    pr = int(result["pr"])
    body = render_comment.render(result)

    existing = _find_sticky(api, repo, pr, token)
    if existing is not None:
        _api(
            "PATCH",
            f"{api}/repos/{repo}/issues/comments/{existing}",
            token,
            {"body": body},
        )
        print(f"updated comment {existing} on PR #{pr}")
    else:
        _api("POST", f"{api}/repos/{repo}/issues/{pr}/comments", token, {"body": body})
        print(f"posted new comment on PR #{pr}")


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    post(json.loads(Path(argv[0]).read_text()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
