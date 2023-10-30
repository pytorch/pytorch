#!/usr/bin/env python3
"""Check whether a PR has required labels."""

import sys
from typing import Any

from github_utils import gh_delete_comment, gh_post_pr_comment
from gitutils import get_git_remote_name, get_git_repo_dir, GitRepo
from label_utils import has_required_labels, is_label_err_comment, LABEL_ERR_MSG, gh_add_labels
from trymerge import GitHubPR
import re


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

    return parser.parse_args()

def check_docathon_label(pr: "GitHubPR") -> None:
    """
    A check used to propagate labels from the docathon issues to the
    corresponding PRs.
    Check if there is an issue mentioned in the body of the PR and if yes,
    check that that issue has a docathon label. If it does, propagate all the
    labels from the issue to the PR.
    """
    # Get the pull request body
    pull_request_body = pr.get_body()
    pull_request_labels = pr.get_labels()

    # PR without description
    if pull_request_body is None:
        print("The pull request does not have a description.")
        return

    # Check for the docathon label
    if not re.search(r'#\d{1,6}', pull_request_body):  # search for six digit issue number
        print("The pull request does not mention an issue.")
        return
    issue_number = int(re.findall(r'#(\d{1,6})', pull_request_body)[0])
    issue_labels = pr.get_labels(issue_number)
    docathon_label_present = any(label == 'docathon-h2-2023' for label in issue_labels)  # the label updated for each docathon

    # If the issue has a docathon label, add all labels from the issue to the PR.
    if not docathon_label_present:
        print("The 'docathon-h2-2023' label is not present in the issue.")
        return
    issue_label_names = [label for label in issue_labels]
    labels_to_add = [label for label in issue_label_names if label not in pull_request_labels]
    if labels_to_add:
        gh_add_labels(pr.org, pr.project, pr.pr_num, labels_to_add)
    else:
        print("The pull request already has the same labels.")
        return

def main() -> None:
    args = parse_args()
    repo = GitRepo(get_git_repo_dir(), get_git_remote_name())
    org, project = repo.gh_owner_and_name()
    pr = GitHubPR(org, project, args.pr_num)

    try:
        if not has_required_labels(pr):
            print(LABEL_ERR_MSG)
            add_label_err_comment(pr)
        else:
            delete_all_label_err_comments(pr)
        check_docathon_label(pr)
    except Exception as e:
        pass

    sys.exit(0)


if __name__ == "__main__":
    main()
