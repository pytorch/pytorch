"""GitHub Utilities"""

import json
import os
import warnings
from dataclasses import dataclass
from typing import Any, Callable, cast, Dict, List, Optional, Tuple, Union
from urllib.error import HTTPError
from urllib.parse import quote
from urllib.request import Request, urlopen


GITHUB_API_URL = "https://api.github.com"


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
    data: Union[Optional[Dict[str, Any]], str] = None,
    method: Optional[str] = None,
    reader: Callable[[Any], Any] = lambda x: x.read(),
) -> Tuple[Any, Any]:
    if headers is None:
        headers = {}
    token = os.environ.get("GITHUB_TOKEN")
    if token is not None and url.startswith(f"{GITHUB_API_URL}/"):
        headers["Authorization"] = f"token {token}"

    data_ = None
    if data is not None:
        data_ = data.encode() if isinstance(data, str) else json.dumps(data).encode()

    try:
        with urlopen(Request(url, headers=headers, data=data_, method=method)) as conn:
            return conn.headers, reader(conn)
    except HTTPError as err:
        if (
            err.code == 403
            and all(
                key in err.headers
                for key in ["X-RateLimit-Limit", "X-RateLimit-Remaining"]
            )
            and int(err.headers["X-RateLimit-Remaining"]) == 0
        ):
            print(
                f"""{url}
                Rate limit exceeded:
                Used: {err.headers['X-RateLimit-Used']}
                Limit: {err.headers['X-RateLimit-Limit']}
                Remaining: {err.headers['X-RateLimit-Remaining']}
                Resets at: {err.headers['x-RateLimit-Reset']}"""
            )
        else:
            print(f"Error fetching {url} {err}")
        raise


def gh_fetch_url(
    url: str,
    *,
    headers: Optional[Dict[str, str]] = None,
    data: Union[Optional[Dict[str, Any]], str] = None,
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


def gh_graphql(query: str, **kwargs: Any) -> Dict[str, Any]:
    rc = gh_fetch_url(
        "https://api.github.com/graphql",
        data={"query": query, "variables": kwargs},
        reader=json.load,
    )
    if "errors" in rc:
        raise RuntimeError(
            f"GraphQL query {query}, args {kwargs} failed: {rc['errors']}"
        )
    return cast(Dict[str, Any], rc)


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
        f"{GITHUB_API_URL}/repos/{org}/{repo}/issues/{pr_num}/comments",
        comment,
        dry_run,
    )


def gh_post_commit_comment(
    org: str, repo: str, sha: str, comment: str, dry_run: bool = False
) -> List[Dict[str, Any]]:
    return _gh_post_comment(
        f"{GITHUB_API_URL}/repos/{org}/{repo}/commits/{sha}/comments",
        comment,
        dry_run,
    )


def gh_close_pr(org: str, repo: str, pr_num: int, dry_run: bool = False) -> None:
    url = f"{GITHUB_API_URL}/repos/{org}/{repo}/pulls/{pr_num}"
    if dry_run:
        print(f"Dry run closing PR {pr_num}")
    else:
        gh_fetch_url(url, method="PATCH", data={"state": "closed"})


def gh_delete_comment(org: str, repo: str, comment_id: int) -> None:
    url = f"{GITHUB_API_URL}/repos/{org}/{repo}/issues/comments/{comment_id}"
    gh_fetch_url(url, method="DELETE")


def gh_fetch_merge_base(org: str, repo: str, base: str, head: str) -> str:
    merge_base = ""
    # Get the merge base using the GitHub REST API. This is the same as using
    # git merge-base without the need to have git. The API doc can be found at
    # https://docs.github.com/en/rest/commits/commits?apiVersion=2022-11-28#compare-two-commits
    try:
        json_data = gh_fetch_url(
            f"{GITHUB_API_URL}/repos/{org}/{repo}/compare/{base}...{head}",
            headers={"Accept": "application/vnd.github.v3+json"},
            reader=json.load,
        )
        if json_data:
            merge_base = json_data.get("merge_base_commit", {}).get("sha", "")
        else:
            warnings.warn(
                f"Failed to get merge base for {base}...{head}: Empty response"
            )
    except Exception as error:
        warnings.warn(f"Failed to get merge base for {base}...{head}: {error}")

    return merge_base


def gh_update_pr_state(org: str, repo: str, pr_num: int, state: str = "open") -> None:
    url = f"{GITHUB_API_URL}/repos/{org}/{repo}/pulls/{pr_num}"
    try:
        gh_fetch_url(url, method="PATCH", data={"state": state})
    except HTTPError as err:
        # When trying to open the pull request, error 422 means that the branch
        # has been deleted and the API couldn't re-open it
        if err.code == 422 and state == "open":
            warnings.warn(
                f"Failed to open {pr_num} because its head branch has been deleted: {err}"
            )
        else:
            raise


def gh_query_issues_by_labels(
    org: str, repo: str, labels: List[str], state: str = "open"
) -> List[Dict[str, Any]]:
    url = f"{GITHUB_API_URL}/repos/{org}/{repo}/issues"
    return gh_fetch_json(
        url, method="GET", params={"labels": ",".join(labels), "state": state}
    )
