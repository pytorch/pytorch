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
        "--github-issue-repo",
        type=str,
        required=False,
        default="pytorch/test-infra",
        help="GitHub repo to get the issue",
    )
    parser.add_argument(
        "--github-repo",
        type=str,
        required=True,
        help="GitHub repo where CI is running",
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
        "--github-branch", type=str, required=True, help="Current GitHub branch or tag"
    )
    parser.add_argument(
        "--github-ref-type",
        type=str,
        required=True,
        help="Current GitHub ref type, branch or tag",
    )

    return parser.parse_args()


def get_gh_client(github_token: str) -> Github:
    auth = Auth.Token(github_token)
    return Github(auth=auth)


def get_issue(gh: Github, repo: str, issue_num: int) -> Issue:
    repo = gh.get_repo(repo)
    return repo.get_issue(number=issue_num)


def get_potential_pr_author(
    gh: Github, repo: str, username: str, ref_type: str, ref_name: str
) -> str:
    # If the trigger was a new tag added by a bot, this is a ciflow case
    # Fetch the actual username from the original PR. The PR number is
    # embedded in the tag name: ciflow/<name>/<pr-number>
    if username == "pytorch-bot[bot]" and ref_type == "tag":
        split_tag = ref_name.split("/")
        if (
            len(split_tag) == 3
            and split_tag[0] == "ciflow"
            and split_tag[2].isnumeric()
        ):
            pr_number = split_tag[2]
            try:
                repository = gh.get_repo(repo)
                pull = repository.get_pull(number=int(pr_number))
            except Exception as e:
                raise Exception(  # noqa: TRY002
                    f"issue with pull request {pr_number} from repo {repository}"
                ) from e
            return pull.user.login
    # In all other cases, return the original input username
    return username


def is_exception_branch(branch: str) -> bool:
    return branch.split("/")[0] in {"main", "nightly", "release", "landchecks"}


def get_workflow_type(
    issue: Issue, workflow_requestors: Iterable[str]
) -> Tuple[str, str]:
    try:
        first_comment = issue.get_comments()[0].body.strip("\n\t ")

        if first_comment[0] == "!":
            MESSAGE = "LF Workflows are disabled for everyone. Using meta runners."
            return WORKFLOW_LABEL_META, MESSAGE
        elif first_comment[0] == "*":
            MESSAGE = "LF Workflows are enabled for everyone. Using LF runners."
            return WORKFLOW_LABEL_LF, MESSAGE
        else:
            all_opted_in_users = {
                usr_raw.strip("\n\t@ ") for usr_raw in first_comment.split()
            }
            opted_in_requestors = {
                usr for usr in workflow_requestors if usr in all_opted_in_users
            }
            if opted_in_requestors:
                MESSAGE = f"LF Workflows are enabled for {', '.join(opted_in_requestors)}. Using LF runners."
                return WORKFLOW_LABEL_LF, MESSAGE
            else:
                MESSAGE = f"LF Workflows are disabled for {', '.join(workflow_requestors)}. Using meta runners."
                return WORKFLOW_LABEL_META, MESSAGE

    except Exception as e:
        MESSAGE = f"Failed to get determine workflow type. Falling back to meta runners. Exception: {e}"
        return WORKFLOW_LABEL_META, MESSAGE


def main() -> None:
    args = parse_args()

    if args.github_ref_type == "branch" and is_exception_branch(args.github_branch):
        output = {
            LABEL_TYPE_KEY: WORKFLOW_LABEL_META,
            MESSAGE_KEY: f"Exception branch: '{args.github_branch}', using meta runners",
        }
    else:
        try:
            gh = get_gh_client(args.github_token)
            # The default issue we use - https://github.com/pytorch/test-infra/issues/5132
            issue = get_issue(gh, args.github_issue_repo, args.github_issue)
            username = get_potential_pr_author(
                gh,
                args.github_repo,
                args.github_actor,
                args.github_ref_type,
                args.github_branch,
            )
            label_type, message = get_workflow_type(
                issue,
                (
                    args.github_issue_owner,
                    username,
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
