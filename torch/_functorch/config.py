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
from typing import Union

# Converts torch rng ops to their functional philox rng equivalents. Note that
# we functionalize only CUDA rng ops today.
functionalize_rng_ops: bool = False

# can be useful for debugging if we are incorrectly creating meta fake tensors
fake_tensor_allow_meta: Union[str, bool] = os.environ.get("FAKE_ALLOW_META", True)

# Enables optional asserts in hotpath code to check for errors.  If
# you are seeing weird accuracy problems, try turning this on.
# This is currently off by default as it will harm tracing time,
# but it is on by default for aot_eager.
debug_assert: bool = False

debug_partitioner: Union[str, bool] = os.environ.get("AOT_PARTITIONER_DEBUG", False)

static_weight_shapes: bool = True

# Applies CSE to the graph before partitioning
cse: bool = True

# Restricts the amount of computation AOTAutograd can do.
max_dist_from_bw: int = 3

from torch._config_utils import install_config_module
install_config_module('FunctorchConfig', sys.modules[__name__])
