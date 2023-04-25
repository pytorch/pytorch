import argparse
from typing import Any

from gitutils import get_git_remote_name, get_git_repo_dir, GitRepo
from trymerge import GitHubPR, MERGE_IN_PROGRESS_LABEL


def parse_args() -> Any:
    parser = argparse.ArgumentParser(
        description="Perform actions when a merge workflow is cancelled"
    )
    parser.add_argument(
        "--pr-num",
        type=int,
        required=True,
        help="The PR number to cancel the merge for",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo = GitRepo(get_git_repo_dir(), get_git_remote_name(), debug=True)
    org, project = repo.gh_owner_and_name()
    pr_num = args.pr_num

    GitHubPR(org, project, pr_num).remove_label(MERGE_IN_PROGRESS_LABEL)


if __name__ == "__main__":
    main()
