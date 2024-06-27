import json
from argparse import ArgumentParser
from typing import Any, Iterable, Tuple

from github import Auth, Github
from github.Issue import Issue


WORKFLOW_LABEL_META = ""  # use meta runners
WORKFLOW_LABEL_LF = "lf."  # use runners from the linux foundation
LABEL_TYPE_KEY = "label_type"
MESSAGE_KEY = "message"
MESSAGE = ""  # Debug message to return to the caller


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
        "--github-issue", type=int, required=True, help="GitHub issue number"
    )
    parser.add_argument(
        "--github-actor", type=str, required=True, help="GitHub triggering_actor"
    )
    parser.add_argument(
        "--github-issue-owner", type=str, required=True, help="GitHub issue owner"
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


def get_workflow_type(issue: Issue, username_check: Iterable[str]) -> Tuple[str, str]:
    try:
        first_comment = issue.get_comments()[0].body.strip("\n\t ")

        if first_comment[0] == "!":
            MESSAGE = "LF Workflows are disabled for everyone. Using meta runners."
            return WORKFLOW_LABEL_META, MESSAGE
        elif first_comment[0] == "*":
            MESSAGE = "LF Workflows are enabled for everyone. Using LF runners."
            return WORKFLOW_LABEL_LF, MESSAGE
        else:
            user_set = {
                usr
                for usr in (
                    usr_raw.strip("\n\t@ ") for usr_raw in first_comment.split()
                )
            }
            if any(map(lambda uc: uc in user_set, username_check)):
                MESSAGE = f"LF Workflows are enabled for {', '.join(username_check)}. Using LF runners."
                return WORKFLOW_LABEL_LF, MESSAGE
            else:
                MESSAGE = f"LF Workflows are disabled for {', '.join(username_check)}. Using meta runners."
                return WORKFLOW_LABEL_META, MESSAGE

    except Exception as e:
        MESSAGE = f"Failed to get determine workflow type. Falling back to meta runners. Exception: {e}"
        return WORKFLOW_LABEL_META, MESSAGE


def main() -> None:
    args = parse_args()

    if is_exception_branch(args.github_branch):
        output = {
            LABEL_TYPE_KEY: WORKFLOW_LABEL_META,
            MESSAGE_KEY: f"Exception branch: '{args.github_branch}', using meta runners",
        }
    else:
        try:
            gh = get_gh_client(args.github_token)
            # The default issue we use - https://github.com/pytorch/test-infra/issues/5132
            issue = get_issue(gh, args.github_repo, args.github_issue)
            label_type, message = get_workflow_type(
                issue,
                (
                    args.github_issue_owner,
                    args.github_actor,
                ),
            )
            output = {
                LABEL_TYPE_KEY: label_type,
                MESSAGE_KEY: message,
            }
        except Exception as e:
            output = {
                LABEL_TYPE_KEY: WORKFLOW_LABEL_META,
                MESSAGE_KEY: f"Failed to get issue. Falling back to meta runners. Exception: {e}",
            }

    json_output = json.dumps(output)
    print(json_output)


if __name__ == "__main__":
    main()
