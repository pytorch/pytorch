import os
import re
from typing import List, Pattern, Optional


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
    labels: List[str]
    pr_num: int
    org: str
    project: str

    has_trunk_label: bool
    has_ciflow_label: bool

    def __init__(
        self,
        force: bool,
        labels: List[str],
        pr_num: int,
        org: str,
        project: str,
    ):
        self.force = force
        self.labels = labels
        self.pr_num = pr_num
        self.org = org
        self.project = project

    def _get_flag_msg(self) -> str:
        if self.force:
            return "Your change will be merged immediately since you used the force (-f) flag, " + \
                "**bypassing any CI checks** (ETA: 1-5 minutes)."
        else:
            return "Your change will be merged once all checks pass (ETA 0-4 Hours)."


    def get_merge_message(self, commit: Optional[str] = None) -> str:
        title = "### Merge started"
        main_message = self._get_flag_msg()

        advanced_debugging = "\n".join((
            "<details><summary>Advanced Debugging</summary>",
            "Check the merge workflow status ",
            f"<a href=\"{os.getenv('GH_RUN_URL')}\">here</a>",
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
