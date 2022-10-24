import os
import re
from typing import List, Pattern, Tuple, Optional


BOT_COMMANDS_WIKI = "https://github.com/pytorch/pytorch/wiki/Bot-commands"

CIFLOW_LABEL = re.compile(r"^ciflow/.+")
CIFLOW_TRUNK_LABEL = re.compile(r"^ciflow/trunk")

OFFICE_HOURS_LINK = "https://github.com/pytorch/pytorch/wiki/Dev-Infra-Office-Hours"
CONTACT_US = f"Questions? Feedback? Please reach out to the [PyTorch DevX Team]({OFFICE_HOURS_LINK})"
ALTERNATIVES = (
    f"Learn more about merging in the [wiki]({BOT_COMMANDS_WIKI})."
)


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
            return "Your change will be merged immediately since you used the force (-f) flag, " + \
                "**bypassing any CI checks** (ETA: 1-5 minutes)."
        elif self.on_green:
            return "Your change will be merged once all checks on your PR pass since you used the green (-g) flag (ETA: 0-4 Hours)."
        elif self.land_checks:
            flag_msg = \
                "**The `-l` land checks flag is deprecated and no longer needed.** Instead we now automatically " + \
                "add the `ciflow\\trunk` label to your PR once it's approved\n\n"

            if self.has_trunk_label:
                flag_msg += "Your change will be merged once all checks on your PR pass (ETA 0-4 Hours)."
            else:
                flag_msg += "Your change will be merged once the land checks pass (**ETA 4 Hours**)."

            return flag_msg
        else:
            return "Your change will be merged once all checks pass (ETA 0-4 Hours)."

    def _get_land_check_progress(self, commit: Optional[str]) -> str:
        if commit is not None:
            return (
                " and land check "
                + f"progress <a href=\"https://hud.pytorch.org/{self.org}/{self.project}/commit/{commit}\">here</a>"
            )
        else:
            return ""

    def get_merge_message(self, commit: Optional[str] = None) -> str:
        title = "### Merge started"
        main_message = self._get_flag_msg()

        advanced_debugging = "\n".join((
            "<details><summary>Advanced Debugging</summary>",
            "Check the merge workflow status ",
            f"<a href=\"{os.getenv('GH_RUN_URL')}\">here</a>{self._get_land_check_progress(commit)}",
            "</details>"
        ))

        msg = title + "\n"
        msg += main_message + "\n\n"
        msg += ALTERNATIVES + "\n\n"
        msg += CONTACT_US
        msg += advanced_debugging
        return msg


def get_revert_message(org: str, project: str, pr_num: int) -> str:
    msg = (
        "@pytorchbot successfully started a revert job."
        + f" Check the current status [here]({os.getenv('GH_RUN_URL')}).\n"
    )
    msg += CONTACT_US
    return msg
