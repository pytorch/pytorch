"""GitHub Utilities"""

import json
import os

from dataclasses import dataclass
from typing import Any, Callable, cast, Dict, List, Optional, Tuple
from urllib.error import HTTPError
from urllib.parse import quote
from urllib.request import Request, urlopen


@dataclass
class GitHubComment:
    body_text: str
    created_at: str
    author_login: str
    author_association: str
    editor_login: Optional[str]
    database_id: int
    url: str


def gh_fetch_url_and_headers(
    url: str,
    *,
    headers: Optional[Dict[str, str]] = None,
    data: Optional[Dict[str, Any]] = None,
    method: Optional[str] = None,
    reader: Callable[[Any], Any] = lambda x: x.read(),
) -> Tuple[Any, Any]:
    if headers is None:
        headers = {}
    token = os.environ.get("GITHUB_TOKEN")
    if token is not None and url.startswith("https://api.github.com/"):
        headers["Authorization"] = f"token {token}"
    data_ = json.dumps(data).encode() if data is not None else None
    try:
        with urlopen(Request(url, headers=headers, data=data_, method=method)) as conn:
            return conn.headers, reader(conn)
    except HTTPError as err:
        if err.code == 403 and all(
            key in err.headers for key in ["X-RateLimit-Limit", "X-RateLimit-Used"]
        ):
            print(
                f"""Rate limit exceeded:
                Used: {err.headers['X-RateLimit-Used']}
                Limit: {err.headers['X-RateLimit-Limit']}
                Remaining: {err.headers['X-RateLimit-Remaining']}
                Resets at: {err.headers['x-RateLimit-Reset']}"""
            )
        raise


def gh_fetch_url(
    url: str,
    *,
    headers: Optional[Dict[str, str]] = None,
    data: Optional[Dict[str, Any]] = None,
    method: Optional[str] = None,
    reader: Callable[[Any], Any] = lambda x: x.read(),
) -> Any:
    return gh_fetch_url_and_headers(
        url, headers=headers, data=data, reader=json.load, method=method
    )[1]


def gh_fetch_json(
    url: str,
    params: Optional[Dict[str, Any]] = None,
    data: Optional[Dict[str, Any]] = None,
    method: Optional[str] = None,
) -> List[Dict[str, Any]]:
    headers = {"Accept": "application/vnd.github.v3+json"}
    if params is not None and len(params) > 0:
        url += "?" + "&".join(
            f"{name}={quote(str(val))}" for name, val in params.items()
        )
    return cast(
        List[Dict[str, Any]],
        gh_fetch_url(url, headers=headers, data=data, reader=json.load, method=method),
    )


def _gh_fetch_json_any(
    url: str,
    params: Optional[Dict[str, Any]] = None,
    data: Optional[Dict[str, Any]] = None,
) -> Any:
    headers = {"Accept": "application/vnd.github.v3+json"}
    if params is not None and len(params) > 0:
        url += "?" + "&".join(
            f"{name}={quote(str(val))}" for name, val in params.items()
        )
    return gh_fetch_url(url, headers=headers, data=data, reader=json.load)


def gh_fetch_json_list(
    url: str,
    params: Optional[Dict[str, Any]] = None,
    data: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    return cast(List[Dict[str, Any]], _gh_fetch_json_any(url, params, data))


def gh_fetch_json_dict(
    url: str,
    params: Optional[Dict[str, Any]] = None,
    data: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    return cast(Dict[str, Any], _gh_fetch_json_any(url, params, data))


def _gh_post_comment(
    url: str, comment: str, dry_run: bool = False
) -> List[Dict[str, Any]]:
    if dry_run:
        print(comment)
        return []
    return gh_fetch_json_list(url, data={"body": comment})


def gh_post_pr_comment(
    org: str, repo: str, pr_num: int, comment: str, dry_run: bool = False
) -> List[Dict[str, Any]]:
    return _gh_post_comment(
        f"https://api.github.com/repos/{org}/{repo}/issues/{pr_num}/comments",
        comment,
        dry_run,
    )


def gh_post_commit_comment(
    org: str, repo: str, sha: str, comment: str, dry_run: bool = False
) -> List[Dict[str, Any]]:
    return _gh_post_comment(
        f"https://api.github.com/repos/{org}/{repo}/commits/{sha}/comments",
        comment,
        dry_run,
    )


def gh_delete_comment(org: str, repo: str, comment_id: int) -> None:
    url = f"https://api.github.com/repos/{org}/{repo}/issues/comments/{comment_id}"
    gh_fetch_url(url, method="DELETE")
