# Copyright (c) Meta Platforms, Inc. and affiliates

"""
Global configuration flags for torch.distributed
"""

import sys
from typing import TYPE_CHECKING

from torch.utils._config_module import Config, install_config_module


__all__ = ["compile_on_one_rank", "use_torchcomms", "pipeline_per_direction_p2p"]

# Deprecated alias. The canonical flag now lives in torch.compiler.config -- it is read
# across the compiler stack (make_fx, inductor) not just by distributed. Kept here for
# back-compat (reads, writes, and .patch forward to the canonical flag).
compile_on_one_rank: bool = Config(
    alias="torch.compiler.config.compile_on_one_rank",
    deprecated=True,
    deprecation_message="use torch.compiler.config.compile_on_one_rank instead",
)

# When enabled, uses TorchComms for communication backend instead of the
# traditional ProcessGroup backends (NCCL, Gloo, etc.).
use_torchcomms: bool = Config(
    default=False,
    env_name_default="TORCH_DISTRIBUTED_USE_TORCHCOMMS",
)

# When enabled, pipeline stages carry each adjacent directed physical rank edge
# on a separate communicator instead of sharing one FIFO. This preserves P2P
# ordering across looped schedules where distinct virtual-stage edges can reach
# the same ranks in different orders. The schedule initializes the actual PP
# parent before deriving child communicators.
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
