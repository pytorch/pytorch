# Copyright (c) Meta Platforms, Inc. and affiliates

"""
Global configuration flags for torch.distributed
"""

import os
import sys
from typing import TYPE_CHECKING

from torch.utils._config_module import install_config_module


__all__ = ["compile_on_one_rank"]

# When enabled, coordinates are computed at runtime via a custom op rather
# than being baked in at compile time. This allows compiling on one rank
# and running on multiple ranks.
compile_on_one_rank: bool = bool(
    os.environ.get("TORCH_DISTRIBUTED_COMPILE_ON_ONE_RANK", False)
)


if TYPE_CHECKING:
    from torch.utils._config_typing import *  # noqa: F401, F403


# adds patch, save_config, invalid config checks, etc
install_config_module(sys.modules[__name__])
