# flake8: noqa: G004

# Note: Copies of this script in runner_determinator.py and _runner-determinator.yml
#       must be kept in sync. You can do it easily by running the following command:
#           python .github/scripts/update_runner_determinator.py

"""
This runner determinator is used to determine which set of runners to run a
GitHub job on. It uses the first comment of a GitHub issue (by default
https://github.com/pytorch/test-infra/issues/5132) to define the configuration
of which runners should be used to run which job.

The configuration has two parts, the settings and a list of opted-in users,
separated by a line containing "---".  If the line is not present, the
settings are considered to be empty with only the second part, the user
list, defined.

The first part is a YAML block that defines the rollout settings. This can be
used to define any settings that are needed to determine which runners to use.
It's fields are defined by the RolloutSettings class below.

The second part is a list of users who are explicitly opted in to the LF fleet.
The user list is also a comma separated list of additional features or
experiments which the user could be opted in to.

The user list has the following rules:

- Users are GitHub usernames, which must start with the @ prefix
- Each user is also a comma-separated list of features/experiments to enable
- A "#" prefix opts the user out of all experiments

Example config:
    # A list of experiments that can be opted into.
    # This defines the behavior they'll induce when opted into.
    # Expected syntax is:
    #   [experiment_name]: # Name of the experiment. Also used for the label prefix.
    #      rollout_perc: [int] # % of workflows to run with this experiment when users are not opted in.

    experiments:
      lf:
        rollout_percent: 25
        all_branches: false
        default: true
    ---

    # Opt-ins:
    # Users can opt into the LF fleet by adding their GitHub username to this list
    # and specifying experiments to enable in a comma-separated list.
    # To always opt out of an experiment, prefix it with a "-".
    # Experiments should be from the above list.

    @User1,-lf,split_build
    @User2,lf
    @User3,split_build
"""

import json
import logging
import os
import random
import re
import sys
from argparse import ArgumentParser
from collections.abc import Iterable
from functools import cache
from logging import LogRecord
from typing import Any, NamedTuple
from urllib.request import Request, urlopen

import yaml
from github import Auth, Github
from github.Issue import Issue


DEFAULT_LABEL_PREFIX = ""  # use meta runners
WORKFLOW_LABEL_LF = "lf."  # use runners from the linux foundation
WORKFLOW_LABEL_LF_CANARY = "lf.c."  # use canary runners from the linux foundation

GITHUB_OUTPUT = os.getenv("GITHUB_OUTPUT", "")
GH_OUTPUT_KEY_AMI = "runner-ami"
GH_OUTPUT_KEY_LABEL_TYPE = "label-type"
OPT_OUT_LABEL = "no-runner-experiments"

SETTING_EXPERIMENTS = "experiments"

LF_FLEET_EXPERIMENT = "lf"
CANARY_FLEET_SUFFIX = ".c"


class Experiment(NamedTuple):
    rollout_perc: float = (
        0  # Percentage of workflows to experiment on when user is not opted-in.
    )
    all_branches: bool = (
        False  # If True, the experiment is also enabled on the exception branches
    )
    default: bool = (
        True  # If True, the experiment is enabled by default for all queries
    )

    # Add more fields as needed


class Settings(NamedTuple):
    """
    Settings for the experiments that can be opted into.
    """

    experiments: dict[str, Experiment] = {}


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


def _str_comma_separated_to_set(value: str) -> frozenset[str]:
    return frozenset(
        filter(lambda itm: itm != "", map(str.strip, value.strip(" \n\t").split(",")))
    )


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
    parser.add_argument(
        "--eligible-experiments",
        type=_str_comma_separated_to_set,
        required=False,
        default="",
        help="comma separated list of experiments to check, if omitted all experiments marked with default=True are checked",
    )
    parser.add_argument(
        "--opt-out-experiments",
        type=_str_comma_separated_to_set,
        required=False,
        default="",
        help=(
            "comma separated list of experiments to opt-out of. If unset, no opt-outs will occur. "
            "If the same experiment is listed both here and in '--eligible-experiments' opt-out will take priority."
        ),
    )
    parser.add_argument(
        "--pr-number",
        type=str,
        required=False,
        default="",
        help="the optional PR number where this is run",
    )

    return parser.parse_args()


def get_gh_client(github_token: str) -> Github:  # type: ignore[no-any-unimported]
    auth = Auth.Token(github_token)
    return Github(auth=auth)


def get_issue(gh: Github, repo: str, issue_num: int) -> Issue:  # type: ignore[no-any-unimported]
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
            return pull.user.login  # type: ignore[no-any-return]
    # In all other cases, return the original input username
    return username


def is_exception_branch(branch: str) -> bool:
    """
    Branches that get opted out of experiments by default, until they're explicitly enabled.
    """
    return branch.split("/")[0] in {"main", "nightly", "release", "landchecks"}


def load_yaml(yaml_text: str) -> Any:
    try:
        data = yaml.safe_load(yaml_text)
        return data
    except yaml.YAMLError:
        log.exception("Error loading YAML")
        raise


def extract_settings_user_opt_in_from_text(rollout_state: str) -> tuple[str, str]:
    """
    Extracts the text with settings, if any, and the opted in users from the rollout state.

    If the issue body contains "---" then the text above that is the settings
    and the text below is the list of opted in users.

    If it doesn't contain "---" then the settings are empty and the rest is the users.
    """
    rollout_state_parts = rollout_state.split("---")
    if len(rollout_state_parts) >= 2:
        return rollout_state_parts[0], rollout_state_parts[1]
    else:
        return "", rollout_state


class UserOptins(dict[str, list[str]]):
    """
    Dictionary of users with a list of features they have opted into
    """


def parse_user_opt_in_from_text(user_optin_text: str) -> UserOptins:
    """
    Parse the user opt-in text into a key value pair of username and the list of features they have opted into

    Users are GitHub usernames with the @ prefix. Each user is also a comma-separated list of features/experiments to enable.
        - Example line: "@User1,lf,split_build"
        - A "#" prefix indicates the user is opted out of all experiments


    """
    optins = UserOptins()
    for user in user_optin_text.split("\n"):
        user = user.strip("\r\n\t -")
        if not user or not user.startswith("@"):
            # Not a valid user. Skip
            continue

        if user:
            usr_name = user.split(",")[0].strip("@")
            optins[usr_name] = [exp.strip(" ") for exp in user.split(",")[1:]]

    return optins


def is_valid_experiment_name(experiment_name: str) -> bool:
    """
    Check if the experiment name is valid.
    A valid name:
        - Contains only alphanumeric characters and the special characters "_" & "-"
        - The special characters "_" & "-" shouldn't be the first or last characters
        - Cannot contain spaces
    """

    valid_char_regex = r"^[a-zA-Z0-9]([\w-]*[a-zA-Z0-9])?$"
    valid = bool(re.match(valid_char_regex, experiment_name))

    if valid:
        return True

    log.error(
        f"Invalid experiment name: {experiment_name}. Experiment names should only contain alphanumeric characters, '_', and '-'. They cannot contain spaces, and the special characters '_' and '-' cannot be the first or last characters."
    )
    return False


def parse_settings_from_text(settings_text: str) -> Settings:
    """
    Parse the experiments from the issue body into a list of ExperimentSettings
    """
    try:
        if settings_text:
            # Escape the backtick as well so that we can have the settings in a code block on the GH issue
            # for easy reading
            # Note: Using ascii for the backtick so that the cat step in _runner-determinator.yml doesn't choke on
            #       the backtick character in shell commands.
            backtick = chr(96)  # backtick character
            settings_text = settings_text.strip(f"\r\n\t{backtick} ")
            settings = load_yaml(settings_text)

            # For now we just load experiments. We can expand this if/when we add more settings
            experiments = {}

            for exp_name, exp_settings in settings.get(SETTING_EXPERIMENTS).items():
                if not is_valid_experiment_name(exp_name):
                    # Exclude invalid experiments from the list. We log an error, but don't raise an exception so that other experiments can still be processed.
                    continue

                valid_settings = {}
                for setting in exp_settings:
                    if setting not in Experiment._fields:
                        log.warning(
                            f"Unexpected setting in experiment: {setting} = {exp_settings[setting]}"
                        )
                    else:
                        valid_settings[setting] = exp_settings[setting]

                experiments[exp_name] = Experiment(**valid_settings)
            return Settings(experiments)

    except Exception:
        log.exception("Failed to parse settings")

    return Settings()


def parse_settings(rollout_state: str) -> Settings:
    """
    Parse settings, if any, from the rollout state.

    If the issue body contains "---" then the text above that is the settings
    and the text below is the list of opted in users.

    If it doesn't contain "---" then the settings are empty and the default values are used.
    """
    settings_text, _ = extract_settings_user_opt_in_from_text(rollout_state)
    return parse_settings_from_text(settings_text)


def parse_users(rollout_state: str) -> UserOptins:
    """
    Parse users from the rollout state.

    """
    _, users_text = extract_settings_user_opt_in_from_text(rollout_state)
    return parse_user_opt_in_from_text(users_text)


def is_user_opted_in(user: str, user_optins: UserOptins, experiment_name: str) -> bool:
    """
    Check if a user is opted into an experiment
    """
    return experiment_name in user_optins.get(user, [])


def is_user_opted_out(user: str, user_optins: UserOptins, experiment_name: str) -> bool:
    """
    Check if a user explicitly opted out of an experiment
    """
    # if the experiment is prefixed with a "-", then it's an opt-out
    experiment_optout = "-" + experiment_name
    if experiment_optout not in user_optins.get(user, []):
        return False

    if is_user_opted_in(user, user_optins, experiment_name):
        log.warning(
            f"User {user} is opted into experiment {experiment_name}, but also opted out of it. Defaulting to opting out"
        )

    return True


def get_runner_prefix(
    rollout_state: str,
    workflow_requestors: Iterable[str],
    branch: str,
    eligible_experiments: frozenset[str] = frozenset(),
    opt_out_experiments: frozenset[str] = frozenset(),
    is_canary: bool = False,
) -> str:
    settings = parse_settings(rollout_state)
    user_optins = parse_users(rollout_state)

    fleet_prefix = ""
    prefixes = []
    for experiment_name, experiment_settings in settings.experiments.items():
        if not experiment_settings.all_branches and is_exception_branch(branch):
            log.info(
                f"Branch {branch} is an exception branch. Not enabling experiment {experiment_name}."
            )
            continue

        if opt_out_experiments:
            if experiment_name in opt_out_experiments:
                opt_out_exp_list = ", ".join(opt_out_experiments)
                log.info(
                    f"Skipping experiment '{experiment_name}', as this workflow has opted-out (opted out experiments are: {opt_out_exp_list})"
                )
                continue

        if eligible_experiments:
            if experiment_name not in eligible_experiments:
                exp_list = ", ".join(eligible_experiments)
                log.info(
                    f"Skipping experiment '{experiment_name}', as it is not in the eligible_experiments list: {exp_list}"
                )
                continue
        elif not experiment_settings.default:
            log.info(
                f"Skipping experiment '{experiment_name}', as it is not a default experiment"
            )
            continue

        # Is any workflow_requestor opted out to this experiment?
        opted_out_users = [
            requestor
            for requestor in workflow_requestors
            if is_user_opted_out(requestor, user_optins, experiment_name)
        ]

        if opted_out_users:
            log.info(
                f"{', '.join(opted_out_users)} have opted out of experiment {experiment_name}."
            )
            continue

        # Is any workflow_requestor opted in to this experiment?
        opted_in_users = [
            requestor
            for requestor in workflow_requestors
            if is_user_opted_in(requestor, user_optins, experiment_name)
        ]

        enabled = False
        if opted_in_users:
            log.info(
                f"{', '.join(opted_in_users)} have opted into experiment {experiment_name}."
            )
            enabled = True

        elif experiment_settings.rollout_perc:
            # If no user is opted in, then we randomly enable the experiment based on the rollout percentage
            if random.uniform(0, 100) <= experiment_settings.rollout_perc:
                log.info(
                    f"Based on rollout percentage of {experiment_settings.rollout_perc}%, enabling experiment {experiment_name}."
                )
                enabled = True

        if enabled:
            label = experiment_name
            if experiment_name == LF_FLEET_EXPERIMENT:
                # We give some special treatment to the "lf" experiment since determines the fleet we use
                #  - If it's enabled, then we always list it's prefix first
                #  - If we're in the canary branch, then we append ".c" to the lf prefix
                if is_canary:
                    label += CANARY_FLEET_SUFFIX
                fleet_prefix = label
            else:
                prefixes.append(label)

    if len(prefixes) > 1:
        log.error(
            f"Only a fleet and one other experiment can be enabled for a job at any time. Enabling {prefixes[0]} and ignoring the rest, which are {', '.join(prefixes[1:])}"
        )
        prefixes = prefixes[:1]

    # Fleet always comes first
    if fleet_prefix:
        prefixes.insert(0, fleet_prefix)

    return ".".join(prefixes) + "." if prefixes else ""


def get_rollout_state_from_issue(github_token: str, repo: str, issue_num: int) -> str:
    """
    Gets the first comment of the issue, which contains the desired rollout state.

    The default issue we use - https://github.com/pytorch/test-infra/issues/5132
    """
    gh = get_gh_client(github_token)
    issue = get_issue(gh, repo, issue_num)
    return str(issue.get_comments()[0].body.strip("\n\t "))


def download_json(url: str, headers: dict[str, str], num_retries: int = 3) -> Any:
    for _ in range(num_retries):
        try:
            req = Request(url=url, headers=headers)
            content = urlopen(req, timeout=5).read().decode("utf-8")
            return json.loads(content)
        except Exception as e:
            log.warning(f"Could not download {url}: {e}")

    log.warning(f"All {num_retries} retries exhausted, downloading {url} failed")
    return {}


@cache
def get_pr_info(github_repo: str, github_token: str, pr_number: int) -> dict[str, Any]:
    """
    Dynamically get PR information
    """
    github_api = f"https://api.github.com/repos/{github_repo}"
    headers = {
        "Accept": "application/vnd.github.v3+json",
        "Authorization": f"token {github_token}",
    }
    json_response: dict[str, Any] = download_json(
        url=f"{github_api}/issues/{pr_number}",
        headers=headers,
    )

    if not json_response:
        log.warning(f"Failed to get the labels for #{pr_number}")
        return {}

    return json_response


def get_labels(github_repo: str, github_token: str, pr_number: int) -> set[str]:
    """
    Dynamically get the latest list of labels from the pull request
    """
    pr_info = get_pr_info(github_repo, github_token, pr_number)
    return {
        label.get("name") for label in pr_info.get("labels", []) if label.get("name")
    }


def main() -> None:
    args = parse_args()

    runner_label_prefix = DEFAULT_LABEL_PREFIX

    # Check if the PR is opt-out
    if args.pr_number:
        labels = get_labels(args.github_repo, args.github_token, int(args.pr_number))
        if OPT_OUT_LABEL in labels:
            log.info(
                f"Opt-out runner determinator because #{args.pr_number} has {OPT_OUT_LABEL} label"
            )
            set_github_output(GH_OUTPUT_KEY_LABEL_TYPE, runner_label_prefix)
            sys.exit()

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

        is_canary = args.github_repo == "pytorch/pytorch-canary"

        runner_label_prefix = get_runner_prefix(
            rollout_state,
            (args.github_issue_owner, username),
            args.github_branch,
            args.eligible_experiments,
            args.opt_out_experiments,
            is_canary,
        )

    except Exception as e:
        log.error(
            f"Failed to get issue. Defaulting to Meta runners and no experiments. Exception: {e}"
        )

    set_github_output(GH_OUTPUT_KEY_LABEL_TYPE, runner_label_prefix)


if __name__ == "__main__":
    main()
