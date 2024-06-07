"""GitHub Utilities"""

import json
import os

from typing import Any, Callable, cast, Dict, Optional, Tuple

from urllib.error import HTTPError
from urllib.parse import quote
from urllib.request import Request, urlopen


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


def gh_fetch_json_dict(
    url: str,
    params: Optional[Dict[str, Any]] = None,
    data: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    return cast(Dict[str, Any], _gh_fetch_json_any(url, params, data))


def gh_fetch_commit(org: str, repo: str, sha: str) -> Dict[str, Any]:
    return gh_fetch_json_dict(
        f"https://api.github.com/repos/{org}/{repo}/commits/{sha}"
    )
