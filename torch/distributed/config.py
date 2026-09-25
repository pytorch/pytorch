# Copyright (c) Meta Platforms, Inc. and affiliates

"""
Global configuration flags for torch.distributed
"""

import sys
from typing import TYPE_CHECKING

from torch.utils._config_module import Config, install_config_module


__all__ = [
    "compile_on_one_rank",
    "use_torchcomms",
    "pipeline_per_edge_p2p",
]

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

# When enabled, each adjacent directed physical-rank edge uses a separate
# communicator. Opposite directions and distinct rank pairs are isolated;
# logical-stage edges mapped to the same directed rank pair share one FIFO.
#
# This flag force-enables the behavior; it is auto-enabled when TorchComms is in
# use regardless of this flag (see PipelineStage), so it mainly matters for the
# non-TorchComms backends. Schedule initialization creates one child
# communicator per directed physical-rank edge, then preconnects its
# send/receive path before execution or graph capture.
# Setup cost is proportional to the stage assignment's disjoint edge rounds;
# children are cached until full process-group teardown. A lazy NCCL parent still
# creates eager two-rank split children; pipeline P2P submits their operations in
# batches.
pipeline_per_edge_p2p: bool = Config(
    default=False,
    env_name_default="TORCH_DISTRIBUTED_PIPELINE_PER_EDGE_P2P",
)


if TYPE_CHECKING:
    from torch.utils._config_typing import *  # noqa: F403


# adds patch, save_config, invalid config checks, etc
install_config_module(sys.modules[__name__])
