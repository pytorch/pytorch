import json
from argparse import ArgumentParser
from typing import Any

from github import Auth, Github
from github.Issue import Issue


WORKFLOW_TYPE_LABEL = "label"
WORKFLOW_TYPE_RG = "rg"
WORKFLOW_TYPE_BOTH = "both"


def parse_args() -> Any:
    parser = ArgumentParser("Get dynamic rollout settings")
    parser.add_argument("--github-token", type=str, required=True, help="GitHub token")
    parser.add_argument(
        "--github-repo",
        type=str,
        required=False,
        default="pytorch/test-infra",
        help="GitHub repo to get the issue",
    )
    parser.add_argument(
        "--github-issue", type=int, required=True, help="GitHub issue umber"
    )
    parser.add_argument(
        "--github-user", type=str, required=True, help="GitHub username"
    )
    parser.add_argument(
        "--github-branch", type=str, required=True, help="Current GitHub branch"
    )

    return parser.parse_args()


def get_gh_client(github_token: str) -> Github:
    auth = Auth.Token(github_token)
    return Github(auth=auth)


def get_issue(gh: Github, repo: str, issue_num: int) -> Issue:
    repo = gh.get_repo(repo)
    return repo.get_issue(number=issue_num)


def is_exception_branch(branch: str) -> bool:
    return branch.split("/")[0] in {"main", "nightly", "release", "landchecks"}


def get_workflow_type(issue: Issue, username: str) -> str:
    user_list = issue.get_comments()[0].body.split("\r\n")
    try:
        run_option = issue.get_comments()[1].body.split("\r\n")[0]
    except Exception as e:
        run_option = "single"

    if user_list[0] == "!":
        # Use old runners for everyone
        return WORKFLOW_TYPE_LABEL
    elif user_list[1] == "*":
        if run_option == WORKFLOW_TYPE_BOTH:
            # Use ARC runners and old runners for everyone
            return WORKFLOW_TYPE_BOTH
        else:
            # Use only ARC runners for everyone
            return WORKFLOW_TYPE_RG
    elif username in user_list:
        if run_option == WORKFLOW_TYPE_BOTH:
            # Use ARC runners and old runners for a specific user
            return WORKFLOW_TYPE_BOTH
        else:
            # Use only ARC runners for a specific user
            return WORKFLOW_TYPE_RG
    else:
        # Use old runners by default
        return WORKFLOW_TYPE_LABEL


def main() -> None:
    args = parse_args()

    if is_exception_branch(args.github_branch):
        output = {"workflow_type": WORKFLOW_TYPE_LABEL}
    else:
        try:
            gh = get_gh_client(args.github_token)
            issue = get_issue(gh, args.github_repo, args.github_issue)

            output = {"workflow_type": get_workflow_type(issue, args.github_user)}
        except Exception as e:
            output = {"workflow_type": WORKFLOW_TYPE_LABEL}

    json_output = json.dumps(output)
    print(json_output)


if __name__ == "__main__":
    main()
