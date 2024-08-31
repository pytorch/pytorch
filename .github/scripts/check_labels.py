#!/usr/bin/env python3
"""Check whether a PR has required labels."""

import sys
from typing import Any

from github_utils import gh_delete_comment, gh_post_pr_comment
from gitutils import get_git_remote_name, get_git_repo_dir, GitRepo
from label_utils import has_required_labels, is_label_err_comment, LABEL_ERR_MSG
from trymerge import GitHubPR


def delete_all_label_err_comments(pr: "GitHubPR") -> None:
    for comment in pr.get_comments():
        if is_label_err_comment(comment):
            gh_delete_comment(pr.org, pr.project, comment.database_id)


def add_label_err_comment(pr: "GitHubPR") -> None:
    # Only make a comment if one doesn't exist already
    if not any(is_label_err_comment(comment) for comment in pr.get_comments()):
        gh_post_pr_comment(pr.org, pr.project, pr.pr_num, LABEL_ERR_MSG)


def parse_args() -> Any:
    from argparse import ArgumentParser

    parser = ArgumentParser("Check PR labels")
    parser.add_argument("pr_num", type=int)
    # add a flag to return a non-zero exit code if the PR does not have the required labels
    parser.add_argument(
        "--exit-non-zero",
        action="store_true",
        help="Return a non-zero exit code if the PR does not have the required labels",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo = GitRepo(get_git_repo_dir(), get_git_remote_name())
    org, project = repo.gh_owner_and_name()
    pr = GitHubPR(org, project, args.pr_num)

    try:
        if not has_required_labels(pr):
            print(LABEL_ERR_MSG)
            add_label_err_comment(pr)
            if args.exit_non_zero:
                sys.exit(1)
        else:
            delete_all_label_err_comments(pr)
    except Exception as e:
        if args.exit_non_zero:
            sys.exit(1)

    sys.exit(0)


if __name__ == "__main__":
    main()
