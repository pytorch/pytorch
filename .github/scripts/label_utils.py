import re
from typing import Pattern, List
from gitutils import (
    GitHubPR,
    GitHubComment,
    delete_comment,
    get_pytorch_labels,
    gh_post_pr_comment,
)

CIFLOW_LABEL = re.compile(r"^ciflow/.+")
CIFLOW_TRUNK_LABEL = re.compile(r"^ciflow/trunk")

BOT_AUTHORS = ["github-actions", "pytorchmergebot", "pytorch-bot"]

LABEL_ERR_MSG_TITLE = "This PR needs a label"
LABEL_ERR_MSG = (
    f"# {LABEL_ERR_MSG_TITLE}\n"
    "If your changes are user facing and intended to be a part of release notes, please use a label starting with `release notes:`.\n\n"  # noqa: E501  pylint: disable=line-too-long
    "If not, please add the `topic: not user facing` label.\n\n"
    "For more information, see https://github.com/pytorch/pytorch/wiki/PyTorch-AutoLabel-Bot#why-categorize-for-release-notes-and-how-does-it-work."  # noqa: E501  pylint: disable=line-too-long
)

def has_label(labels: List[str], pattern: Pattern[str] = CIFLOW_LABEL) -> bool:
    return len(list(filter(pattern.match, labels))) > 0


def get_release_notes_labels() -> List[str]:
    return [label for label in get_pytorch_labels() if label.lstrip().startswith("release notes:")]


def has_required_labels(pr: GitHubPR) -> bool:
    pr_labels = pr.get_labels()
    # Check if PR is not user facing
    is_not_user_facing_pr = any(label.strip() == "topic: not user facing" for label in pr_labels)
    return is_not_user_facing_pr or any(label.strip() in get_release_notes_labels() for label in pr_labels)


def is_label_err_comment(comment: GitHubComment) -> bool:
    return comment.body_text.lstrip(" #").startswith(LABEL_ERR_MSG_TITLE) and comment.author_login in BOT_AUTHORS


def delete_all_label_err_comments(pr: GitHubPR) -> None:
    for comment in pr.get_comments():
        if is_label_err_comment(comment):
            delete_comment(comment.database_id)
     
             
def add_label_err_comment(pr: GitHubPR) -> None:
    # Only make a comment if one doesn't exist already
    if not any(is_label_err_comment(comment) for comment in pr.get_comments()):
        gh_post_pr_comment(pr.org, pr.project, pr.pr_num, LABEL_ERR_MSG)
        