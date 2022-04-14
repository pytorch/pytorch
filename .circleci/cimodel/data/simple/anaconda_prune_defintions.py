from collections import OrderedDict

from cimodel.data.simple.util.branch_filters import gen_filter_dict
from cimodel.lib.miniutils import quote


CHANNELS_TO_PRUNE = ["pytorch-nightly", "pytorch-test"]
PACKAGES_TO_PRUNE = "pytorch torchvision torchaudio torchtext ignite torchcsprng"


def gen_workflow_job(channel: str):
    return OrderedDict(
        {
            "anaconda_prune": OrderedDict(
                {
                    "name": f"anaconda-prune-{channel}",
                    "context": quote("org-member"),
                    "packages": quote(PACKAGES_TO_PRUNE),
                    "channel": channel,
                    "filters": gen_filter_dict(branches_list=["postnightly"]),
                }
            )
        }
    )


def get_workflow_jobs():
    return [gen_workflow_job(channel) for channel in CHANNELS_TO_PRUNE]
