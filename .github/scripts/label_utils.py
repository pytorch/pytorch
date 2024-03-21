"""GitHub Label Utilities."""

import json

from functools import lru_cache
from typing import Any, List, Tuple, TYPE_CHECKING, Union

from github_utils import gh_fetch_url_and_headers, GitHubComment

# TODO: this is a temp workaround to avoid circular dependencies,
#       and should be removed once GitHubPR is refactored out of trymerge script.
if TYPE_CHECKING:
    from trymerge import GitHubPR

BOT_AUTHORS = ["github-actions", "pytorchmergebot", "pytorch-bot"]

LABEL_ERR_MSG_TITLE = "This PR needs a `release notes:` label"
LABEL_ERR_MSG = f"""# {LABEL_ERR_MSG_TITLE}
If your changes are user facing and intended to be a part of release notes, please use a label starting with `release notes:`.

If not, please add the `topic: not user facing` label.

To add a label, you can comment to pytorchbot, for example
`@pytorchbot label "topic: not user facing"`

For more information, see
https://github.com/pytorch/pytorch/wiki/PyTorch-AutoLabel-Bot#why-categorize-for-release-notes-and-how-does-it-work.
"""


def request_for_labels(url: str) -> Tuple[Any, Any]:
    headers = {"Accept": "application/vnd.github.v3+json"}
    return gh_fetch_url_and_headers(
        url, headers=headers, reader=lambda x: x.read().decode("utf-8")
    )


def update_labels(labels: List[str], info: str) -> None:
    labels_json = json.loads(info)
    labels.extend([x["name"] for x in labels_json])


def get_last_page_num_from_header(header: Any) -> int:
    # Link info looks like: <https://api.github.com/repositories/65600975/labels?per_page=100&page=2>;
    # rel="next", <https://api.github.com/repositories/65600975/labels?per_page=100&page=3>; rel="last"
    link_info = header["link"]
    # Docs does not specify that it should be present for projects with just few labels
    # And https://github.com/malfet/deleteme/actions/runs/7334565243/job/19971396887 it's not the case
    if link_info is None:
        return 1
    prefix = "&page="
    suffix = ">;"
    return int(
        link_info[link_info.rindex(prefix) + len(prefix) : link_info.rindex(suffix)]
    )


@lru_cache
def gh_get_labels(org: str, repo: str) -> List[str]:
    prefix = f"https://api.github.com/repos/{org}/{repo}/labels?per_page=100"
    header, info = request_for_labels(prefix + "&page=1")
    labels: List[str] = []
    update_labels(labels, info)

    last_page = get_last_page_num_from_header(header)
    assert (
        last_page > 0
    ), "Error reading header info to determine total number of pages of labels"
    for page_number in range(2, last_page + 1):  # skip page 1
        _, info = request_for_labels(prefix + f"&page={page_number}")
        update_labels(labels, info)

    return labels


def gh_add_labels(
    org: str, repo: str, pr_num: int, labels: Union[str, List[str]], dry_run: bool
) -> None:
    if dry_run:
        print(f"Dryrun: Adding labels {labels} to PR {pr_num}")
        return
    gh_fetch_url_and_headers(
        url=f"https://api.github.com/repos/{org}/{repo}/issues/{pr_num}/labels",
        data={"labels": labels},
    )


def gh_remove_label(
    org: str, repo: str, pr_num: int, label: str, dry_run: bool
) -> None:
    if dry_run:
        print(f"Dryrun: Removing {label} from PR {pr_num}")
        return
    gh_fetch_url_and_headers(
        url=f"https://api.github.com/repos/{org}/{repo}/issues/{pr_num}/labels/{label}",
        method="DELETE",
    )


def get_release_notes_labels(org: str, repo: str) -> List[str]:
    return [
        label
        for label in gh_get_labels(org, repo)
        if label.lstrip().startswith("release notes:")
    ]


def has_required_labels(pr: "GitHubPR") -> bool:
    pr_labels = pr.get_labels()
    # Check if PR is not user facing
    is_not_user_facing_pr = any(
        label.strip() == "topic: not user facing" for label in pr_labels
    )
    return is_not_user_facing_pr or any(
        label.strip() in get_release_notes_labels(pr.org, pr.project)
        for label in pr_labels
    )


def is_label_err_comment(comment: GitHubComment) -> bool:
    # comment.body_text returns text without markdown
    no_format_title = LABEL_ERR_MSG_TITLE.replace("`", "")
    return (
        comment.body_text.lstrip(" #").startswith(no_format_title)
        and comment.author_login in BOT_AUTHORS
    )
