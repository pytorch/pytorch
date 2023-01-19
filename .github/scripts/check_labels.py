#!/usr/bin/env python3

from typing import Any
from gitutils import (
    get_git_remote_name,
    get_git_repo_dir,
    GitRepo,
)
from trymerge import (
    LABEL_ERR_MSG,
    has_required_labels,
    add_label_err_comment,
    delete_all_label_err_comments,
    GitHubPR,
)

def parse_args() -> Any:
    from argparse import ArgumentParser
    parser = ArgumentParser("Check PR labels")
    parser.add_argument("pr_num", type=int)

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
        else:
            delete_all_label_err_comments(pr)
    except Exception as e:
        pass


if __name__ == "__main__":
    main()
