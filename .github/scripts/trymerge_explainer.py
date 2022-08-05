import os
import re
from typing import List, Pattern, Tuple, Optional

from trymerge import BOT_COMMANDS_WIKI, GitHubPR, gh_post_pr_comment

CIFLOW_LABEL = re.compile(r"^ciflow/.+")
CIFLOW_TRUNK_LABEL = re.compile(r"^ciflow/trunk")


def has_label(labels: List[str], pattern: Pattern[str] = CIFLOW_LABEL) -> bool:
    return len(list(filter(pattern.match, labels))) > 0


class TryMergeExplainer(object):
    force: bool
    on_green: bool
    land_checks: bool
    pr: GitHubPR
    org: str
    project: str

    has_trunk_label: bool
    has_ciflow_label: bool

    def __init__(
        self,
        force: bool,
        on_green: bool,
        land_checks: bool,
        pr: GitHubPR,
        org: str,
        project: str,
    ):
        self.force = force
        self.on_green = on_green
        self.land_checks = land_checks
        self.pr = pr
        self.org = org
        self.project = project

    def get_flags(self) -> Tuple[bool, bool]:
        self.has_trunk_label = has_label(self.pr.get_labels(), CIFLOW_TRUNK_LABEL)
        self.has_ciflow_label = has_label(self.pr.get_labels(), CIFLOW_LABEL)
        should_check_land_branch = self.land_checks and not self.has_trunk_label
        should_check_green = self.on_green or self.has_ciflow_label

        return (should_check_green, should_check_land_branch)

    def _get_flag_msg(self) -> str:
        if self.force:
            return " the force (-f)"
        elif self.on_green:
            return " the green (-g)"
        elif self.land_checks:
            return " the land check (-l)"
        else:
            return "out a "

    def _get_land_check_progress(self, commit: Optional[str]) -> str:
        if commit is not None:
            return (
                f" and [land check]({BOT_COMMANDS_WIKI}) "
                + f"progress [here](https://hud.pytorch.org/{self.org}/{self.project}/commit/{commit})"
            )
        else:
            return ""

    def _get_flag_explaination_message(self) -> str:
        if self.force:
            return (
                "This means your PR will be merged immediately, bypassing any checks."
            )
        elif self.on_green:
            return "This means that your PR will be merged once all signals have passed."
        elif self.land_checks:
            if self.has_trunk_label:
                land_check_msg_suffix = (
                    f"have run since you have the {CIFLOW_TRUNK_LABEL}."
                )
            else:
                land_check_msg_suffix = (
                    "and the land checks branch have run (ETA 4 Hours)."
                )
            return (
                "This means that your PR will be merged once all signals on your PR "
                + land_check_msg_suffix
            )

        else:
            if self.has_ciflow_label:
                return "Since your PR has a ciflow label, we will wait for all checks to be."
            else:
                return "This means only we will only wait for mandatory checks to be green."

    def print_merge_message(self, commit: Optional[str], dry_run: bool) -> None:
        message_prefix = "@pytorchbot successfully started a merge job."
        progress_links = f"Check the current status [here]({os.getenv('GH_RUN_URL')}){self._get_land_check_progress(commit)}."
        flag_message = f"The merge job was triggered with{self._get_flag_msg()} flag."
        explaination_message = self._get_flag_explaination_message()

        msg = message_prefix + " "
        msg += progress_links + "\n"
        msg += flag_message + " "
        msg += explaination_message
        gh_post_pr_comment(self.org, self.project, self.pr.pr_num, msg, dry_run)


def print_revert_message(org: str, project: str, pr_num: int, dry_run: bool) -> None:
    msg = (
        "@pytorchbot successfully started a revert job."
        + f"Check the current status [here]({os.getenv('GH_RUN_URL')})"
    )
    gh_post_pr_comment(org, project, pr_num, msg, dry_run)


def get_land_check_troubleshooting_message() -> str:
    return (
        " If you believe this is an error, you can use the old behavior with `@pytorchbot merge -g`"
        + ' (optionally with the "ciflow/trunk" to get land signals)'
        + ' or use `@pytorchbot merge -f "some reason here"`.'
        + f" For more information, see the [bot wiki]({BOT_COMMANDS_WIKI})."
    )
