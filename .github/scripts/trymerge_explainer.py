import os
import re
from typing import List, Pattern, Tuple, Optional


BOT_COMMANDS_WIKI = "https://github.com/pytorch/pytorch/wiki/Bot-commands"

CIFLOW_LABEL = re.compile(r"^ciflow/.+")
CIFLOW_TRUNK_LABEL = re.compile(r"^ciflow/trunk")

OFFICE_HOURS_LINK = "https://github.com/pytorch/pytorch/wiki/Dev-Infra-Office-Hours"
CONTACT_US = f"Please reach out to the [PyTorch DevX Team]({OFFICE_HOURS_LINK}) with feedback or questions!"
ALTERNATIVES = (
    "If this is not the intended behavior, feel free to use some "
    + f"of the other merge options in the [wiki]({BOT_COMMANDS_WIKI})."
)
LAND_CHECK_ROLLOUT = "https://github.com/pytorch/test-infra/blob/main/torchci/lib/bot/rolloutUtils.ts#L1-L34"


def has_label(labels: List[str], pattern: Pattern[str] = CIFLOW_LABEL) -> bool:
    return len(list(filter(pattern.match, labels))) > 0


class TryMergeExplainer(object):
    force: bool
    on_green: bool
    land_checks: bool
    labels: List[str]
    pr_num: int
    org: str
    project: str

    has_trunk_label: bool
    has_ciflow_label: bool

    def __init__(
        self,
        force: bool,
        on_green: bool,
        land_checks: bool,
        labels: List[str],
        pr_num: int,
        org: str,
        project: str,
    ):
        self.force = force
        self.on_green = on_green
        self.land_checks = land_checks
        self.labels = labels
        self.pr_num = pr_num
        self.org = org
        self.project = project
        self.get_flags()

    def get_flags(self) -> Tuple[bool, bool]:
        self.has_trunk_label = has_label(self.labels, CIFLOW_TRUNK_LABEL)
        self.has_ciflow_label = has_label(self.labels, CIFLOW_LABEL)
        should_check_land_branch = self.land_checks and not self.has_trunk_label
        should_check_green = self.on_green or self.has_ciflow_label

        return (should_check_green, should_check_land_branch)

    def _get_flag_msg(self) -> str:
        if self.force:
            return " the force (-f) flag."
        elif self.on_green:
            return " the green (-g) flag."
        elif self.land_checks:
            return (
                " the land checks (-l) flag."
                + " If you did not specify this flag yourself, "
                + f" you are likely enrolled in the [land checks rollout]({LAND_CHECK_ROLLOUT})."
            )
        else:
            return "out a flag."

    def _get_land_check_progress(self, commit: Optional[str]) -> str:
        if commit is not None:
            return (
                " and land check "
                + f"progress [here](https://hud.pytorch.org/{self.org}/{self.project}/commit/{commit})"
            )
        else:
            return ""

    def _get_flag_explanation_message(self) -> str:
        if self.force:
            return "This means your change will be merged **immediately**, bypassing any CI checks (ETA: 1-5 minutes)."
        elif self.on_green:
            return "This means that your change will be merged once all checks on your PR have passed (ETA: 0-4 Hours)."
        elif self.land_checks:
            if self.has_trunk_label:
                land_check_msg_suffix = "have passed since you have added the `ciflow/trunk` label to your PR (ETA 0-4 Hours)."
            else:
                land_check_msg_suffix = (
                    "and the land checks have passed (**ETA 4 Hours**). "
                )
                land_check_msg_suffix += "If you need to coordinate lands between different changes and cannot risk a land race, "
                land_check_msg_suffix += "please add the `ciflow/trunk` label to your PR and wait for signal to complete, "
                land_check_msg_suffix += "and then land your changes in proper order."
                land_check_msg_suffix += (
                    " Having `trunk`, `pull`, and `Lint` pre-run on a "
                )
                land_check_msg_suffix += (
                    "PR will bypass land checks and the ETA should be immediate."
                )

            return (
                "This means that your change will be merged once all checks on your PR "
                + land_check_msg_suffix
            )
        else:
            return "This means that your change will be merged once all checks on your PR have passed (ETA: 0-4 Hours)."

    def get_merge_message(self, commit: Optional[str] = None) -> str:
        message_prefix = "@pytorchbot successfully started a merge job."
        progress_links = f"Check the current status [here]({os.getenv('GH_RUN_URL')}){self._get_land_check_progress(commit)}."
        flag_message = f"The merge job was triggered with{self._get_flag_msg()}"
        explanation_message = self._get_flag_explanation_message()

        msg = message_prefix + " "
        msg += progress_links + "\n"
        msg += flag_message + " "
        msg += explanation_message + " "
        msg += ALTERNATIVES + "\n"
        msg += CONTACT_US
        return msg


def get_revert_message(org: str, project: str, pr_num: int) -> str:
    msg = (
        "@pytorchbot successfully started a revert job."
        + f" Check the current status [here]({os.getenv('GH_RUN_URL')}).\n"
    )
    msg += CONTACT_US
    return msg


def get_land_check_troubleshooting_message() -> str:
    return (
        " If you believe this is an error, you can use the old behavior with `@pytorchbot merge -g`"
        + " (optionally with the `ciflow/trunk` to get land checks)"
        + ' or use `@pytorchbot merge -f "some reason here"`.'
        + f" For more information, see the [bot wiki]({BOT_COMMANDS_WIKI}). \n\n"
        + CONTACT_US
    )
