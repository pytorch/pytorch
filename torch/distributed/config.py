# Copyright (c) Meta Platforms, Inc. and affiliates

"""
Global configuration flags for torch.distributed
"""

import os
import sys
from typing import TYPE_CHECKING

from torch.utils._config_module import Config, install_config_module


__all__ = ["compile_on_one_rank", "use_torchcomms", "pipeline_per_direction_p2p"]

# When enabled, coordinates are computed at runtime via a custom op rather
# than being baked in at compile time. This allows compiling on one rank
# and running on multiple ranks.
compile_on_one_rank: bool = bool(
    os.environ.get("TORCH_DISTRIBUTED_COMPILE_ON_ONE_RANK", False)
)

# When enabled, uses TorchComms for communication backend instead of the
# traditional ProcessGroup backends (NCCL, Gloo, etc.).
use_torchcomms: bool = Config(
    default=False,
    env_name_default="TORCH_DISTRIBUTED_USE_TORCHCOMMS",
)

# When enabled, pipeline stages carry downstream (r -> r+1, forward activations)
# and upstream (r -> r-1, backward gradients) P2P on two separate communicators
# instead of sharing one. A single PP communicator serializes all send/recv in one
# FIFO: coalescing makes a single mixed batch deadlock-free, but across batches
# (pipeline skew, looped / V schedules, skip connections) the shared FIFO can
# still form a dependency cycle and deadlock. Splitting by direction removes that
# hazard and restores full-duplex bandwidth. Requires a device-bound default
# process group.
#
# This flag force-enables the behavior; it is auto-enabled when TorchComms is in
# use regardless of this flag (see PipelineStage), so it mainly matters for the
# non-TorchComms backends.
pipeline_per_direction_p2p: bool = Config(
    default=False,
    env_name_default="TORCH_DISTRIBUTED_PIPELINE_PER_DIRECTION_P2P",
)


if TYPE_CHECKING:
    from torch.utils._config_typing import *  # noqa: F403


# adds patch, save_config, invalid config checks, etc
install_config_module(sys.modules[__name__])
