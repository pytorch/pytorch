# flake8: noqa: G004

"""
This runner determinator is used to determine which set of runners to run a
GitHub job on. It uses the first comment of a GitHub issue (by default
https://github.com/pytorch/test-infra/issues/5132) to define the configuration
of which runners should be used to run which job.

The configuration has two parts, the rollout settings and a user list,
separated by a line containing "---".  If the line is not present, the
configuration is considered to be empty with only the second part, the user
list, defined.

The first part is a YAML block that defines the rollout settings. This can be
used to define any settings that are needed to determine which runners to use.
It's fields are defined by the RolloutSettings class below.

The second part is a list of users who are explicitly opted in to the LF fleet.
The user list is also a comma separated list of additional features or
experiments which the user could be opted in to.

The user list has the following rules:

- Users are GitHub usernames with the @ prefix
- Each user is also a comma-separated list of features/experiments to enable
- A "#" prefix indicates the user is opted out of the new runners but is opting
  into features/experiments.

Example config:
    lf_fleet_rollout_percentage = 25

    ---

    @User1
    @User2,amz2023
    #@UserOptOutOfNewRunner,amz2023
"""

import logging
import os
import random
from argparse import ArgumentParser
from logging import LogRecord
from typing import Any, Iterable, NamedTuple

import yaml
from github import Auth, Github
from github.Issue import Issue


WORKFLOW_LABEL_META = ""  # use meta runners
WORKFLOW_LABEL_LF = "lf."  # use runners from the linux foundation
WORKFLOW_LABEL_LF_CANARY = "lf.c."  # use canary runners from the linux foundation

RUNNER_AMI_LEGACY = ""
RUNNER_AMI_AMZ2023 = "amz2023"

GITHUB_OUTPUT = os.getenv("GITHUB_OUTPUT", "")
GH_OUTPUT_KEY_AMI = "runner-ami"
GH_OUTPUT_KEY_LABEL_TYPE = "label-type"


class RolloutSettings(NamedTuple):
    # Percentage of jobs to run on the LF fleet.
    # For users not opted in, this is the percentage of jobs that will run on the LF fleet.
    lf_fleet_rollout_percentage: int = 0

    # Add more fields as needed


class ColorFormatter(logging.Formatter):
    """Color codes the log messages based on the log level"""

    COLORS = {
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",  # Red
        "CRITICAL": "\033[31m",  # Red
        "INFO": "\033[0m",  # Reset
        "DEBUG": "\033[0m",  # Reset
    }

    def format(self, record: LogRecord) -> str:
        log_color = self.COLORS.get(record.levelname, "\033[0m")  # Default to reset
        record.msg = f"{log_color}{record.msg}\033[0m"
        return super().format(record)


handler = logging.StreamHandler()
handler.setFormatter(ColorFormatter(fmt="%(levelname)-8s: %(message)s"))

log = logging.getLogger(os.path.basename(__file__))
log.addHandler(handler)
log.setLevel(logging.INFO)


def set_github_output(key: str, value: str) -> None:
    """
    Defines outputs of the github action that invokes this script
    """
    if not GITHUB_OUTPUT:
        # See https://github.blog/changelog/2022-10-11-github-actions-deprecating-save-state-and-set-output-commands/ for deprecation notice
        log.warning(
            "No env var found for GITHUB_OUTPUT, you must be running this code locally. Falling back to the deprecated print method."
        )
        print(f"::set-output name={key}::{value}")
        return

    with open(GITHUB_OUTPUT, "a") as f:
        log.info(f"Setting output: {key}='{value}'")
        f.write(f"{key}={value}\n")


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
    github_token: str, repo: str, username: str, ref_type: str, ref_name: str
) -> str:
    # If the trigger was a new tag added by a bot, this is a ciflow case
    # Fetch the actual username from the original PR. The PR number is
    # embedded in the tag name: ciflow/<name>/<pr-number>

    gh = get_gh_client(github_token)

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
    """
    Branches that get opted out of all experiments and should always use Meta runners
    """
    return branch.split("/")[0] in {"main", "nightly", "release", "landchecks"}


def load_yaml(yaml_text: str) -> Any:
    try:
        data = yaml.safe_load(yaml_text)
        return data
    except yaml.YAMLError as exc:
        log.exception("Error loading YAML")
        raise


def extract_user_opt_in_text(rollout_state: str) -> str:
    """
    Returns just the portion of the rollout_state that defines the opted in users
    """
    rollout_state_parts = rollout_state.split("---")
    if len(rollout_state_parts) == 2:
        return rollout_state_parts[1]
    else:
        return rollout_state


def parse_rollout_settings_text(rollout_state: str) -> str:
    """
    Returns just the portion of the rollout_state that defines the settings
    """
    rollout_state_parts = rollout_state.split("---")
    if len(rollout_state_parts) == 2:
        return rollout_state_parts[0]
    else:
        return ""


def parse_settings(rollout_state: str) -> RolloutSettings:
    """
    Parse settings, if any, from the rollout state.

    If the issue body contains "---" then the text above that is the settings
    and the text below is the list of opted in users.

    If it doesn't contain "---" then the settings are empty and the default values are used.
    """
    try:
        raw_settings = load_yaml(parse_rollout_settings_text(rollout_state))
        if raw_settings:
            # Filter out any unexpected fields and log a warning for each one
            settings = {}
            for setting in raw_settings:
                if setting in RolloutSettings._fields:
                    settings[setting] = raw_settings[setting]
                else:
                    log.warning(
                        f"Unexpected setting in rollout state: {setting} = {raw_settings[setting]}"
                    )
        else:
            user_list = rollout_state

        return RolloutSettings(**settings)

    except Exception as e:
        log.error(f"Failed to parse rollout state. Exception: {e}")
        raise


def get_fleet(rollout_state: str, workflow_requestors: Iterable[str]) -> str:
    """
    Determines if the job should run on the LF fleet or the Meta fleet

    Returns:
        The appropriate label prefix for the runner, corresponding to the fleet to use.
        This gets prefixed to the very start of the runner label.
    """

    try:
        user_optin = extract_user_opt_in_text(rollout_state)
        settings = parse_settings(rollout_state)

        all_opted_in_users = {
            usr_raw.strip("\n\t@ ").split(",")[0] for usr_raw in user_optin.split()
        }
        opted_in_requestors = {
            usr for usr in workflow_requestors if usr in all_opted_in_users
        }

        if opted_in_requestors:
            log.info(
                f"LF Workflows are enabled for {', '.join(opted_in_requestors)}. Using LF runners."
            )
            return WORKFLOW_LABEL_LF

        log.info(f"{', '.join(workflow_requestors)} have not opted into LF Workflows.")

        if settings.lf_fleet_rollout_percentage > 0:
            r = random.randint(1, 100)
            if r <= settings.lf_fleet_rollout_percentage:
                log.info(
                    f"Based on fleet rollout percentage of {settings.lf_fleet_rollout_percentage}%, using LF runners for this workflow."
                )
                return WORKFLOW_LABEL_LF

        return WORKFLOW_LABEL_META

    except Exception as e:
        log.error(
            f"Failed to get determine workflow type. Falling back to meta runners. Exception: {e}"
        )
        return WORKFLOW_LABEL_META


def get_optin_feature(
    rollout_state: str, workflow_requestors: Iterable[str], feature: str, fallback: str
) -> str:
    """
    Used to dynamically opt in jobs to specific runner-type variants.

    Returns:
        The runner-type's variant name if the user has opted in to the feature, otherwise returns an empty string.
        This variant name is prefixed to the runner-type in the label.
    """
    try:
        userlist = {u.lstrip("#").strip("\n\t@ ") for u in rollout_state.split()}
        all_opted_in_users = set()
        for user in userlist:
            for i in user.split(","):
                if i == feature:
                    all_opted_in_users.add(user.split(",")[0])
        opted_in_requestors = {
            usr for usr in workflow_requestors if usr in all_opted_in_users
        }

        if opted_in_requestors:
            log.info(
                f"Feature {feature} is enabled for {', '.join(opted_in_requestors)}. Using feature {feature}."
            )
            return feature
        else:
            log.info(
                f"Feature {feature} is disabled for {', '.join(workflow_requestors)}. Using fallback \"{fallback}\"."
            )
            return fallback

    except Exception as e:
        log.error(
            f'Failed to determine if user has opted-in to feature {feature}. Using fallback "{fallback}". Exception: {e}'
        )
        return fallback


def get_rollout_state_from_issue(github_token: str, repo: str, issue_num: int) -> str:
    """
    Gets the first comment of the issue, which contains the desired rollout state.

    The default issue we use - https://github.com/pytorch/test-infra/issues/5132
    """
    gh = get_gh_client(github_token)
    issue = get_issue(gh, repo, issue_num)
    return str(issue.get_comments()[0].body.strip("\n\t "))


def main() -> None:
    args = parse_args()

    if args.github_ref_type == "branch" and is_exception_branch(args.github_branch):
        log.info(f"Exception branch: '{args.github_branch}', using meta runners")
        label_type = WORKFLOW_LABEL_META
        runner_ami = RUNNER_AMI_LEGACY
    else:
        try:
            rollout_state = get_rollout_state_from_issue(
                args.github_token, args.github_issue_repo, args.github_issue
            )

            username = get_potential_pr_author(
                args.github_token,
                args.github_repo,
                args.github_actor,
                args.github_ref_type,
                args.github_branch,
            )

            label_type = get_fleet(
                rollout_state,
                (
                    args.github_issue_owner,
                    username,
                ),
            )
            runner_ami = get_optin_feature(
                rollout_state=rollout_state,
                workflow_requestors=(
                    args.github_issue_owner,
                    username,
                ),
                feature=RUNNER_AMI_AMZ2023,
                fallback=RUNNER_AMI_LEGACY,
            )
        except Exception as e:
            log.error(
                f"Failed to get issue. Falling back to meta runners. Exception: {e}"
            )
            label_type = WORKFLOW_LABEL_META
            runner_ami = RUNNER_AMI_LEGACY

    # For Canary builds use canary runners
    if args.github_repo == "pytorch/pytorch-canary" and label_type == WORKFLOW_LABEL_LF:
        label_type = WORKFLOW_LABEL_LF_CANARY

    set_github_output(GH_OUTPUT_KEY_LABEL_TYPE, label_type)
    set_github_output(GH_OUTPUT_KEY_AMI, runner_ami)


if __name__ == "__main__":
    main()
