# Keep old package for BC purposes, this file should be removed once
# everything moves to the `torch.distributed.checkpoint` package.
import sys
import warnings

import torch
from torch.distributed.checkpoint import *  # noqa: F403


with warnings.catch_warnings():
    warnings.simplefilter("always")
    warnings.warn(
        "`torch.distributed._shard.checkpoint` will be deprecated, "
        "use `torch.distributed.checkpoint` instead",
        DeprecationWarning,
        stacklevel=2,
    )

sys.modules["torch.distributed._shard.checkpoint"] = torch.distributed.checkpoint
