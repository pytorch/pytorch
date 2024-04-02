# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Global flags for aot autograd
"""
import os
import sys
from typing import TYPE_CHECKING

# Converts torch rng ops to their functional philox rng equivalents. Note that
# we functionalize only CUDA rng ops today.
functionalize_rng_ops = False

# can be useful for debugging if we are incorrectly creating meta fake tensors
fake_tensor_allow_meta = os.environ.get("FAKE_ALLOW_META", True)

# Enables optional asserts in hotpath code to check for errors.  If
# you are seeing weird accuracy problems, try turning this on.
# This is currently off by default as it will harm tracing time,
# but it is on by default for aot_eager.
debug_assert = False

debug_partitioner = os.environ.get("AOT_PARTITIONER_DEBUG", False)

static_weight_shapes = True

# Applies CSE to the graph before partitioning
cse = True

# Restricts the amount of computation AOTAutograd can do.
max_dist_from_bw = 3

if TYPE_CHECKING:
    from torch.utils._config_typing import *  # noqa: F401, F403

from torch.utils._config_module import install_config_module

# adds patch, save_config, invalid config checks, etc
install_config_module(sys.modules[__name__])
