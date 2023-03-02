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
import logging

use_functionalize = True

use_fake_tensor = True

# can be useful for debugging if we are incorrectly creating meta fake tensors
fake_tensor_allow_meta = os.environ.get("FAKE_ALLOW_META", True)

# Enables optional asserts in hotpath code to check for errors.  If
# you are seeing weird accuracy problems, try turning this on.
# For now, to more easily identify bugs, this is turned on by default.
debug_assert = True

debug_fake_cross_ref = os.environ.get("AOT_FAKE_CROSSREF", False)

debug_partitioner = os.environ.get("AOT_PARTITIONER_DEBUG", False)
# Prints out forward + backwards FX graphs
debug_graphs = os.environ.get("AOT_FX_GRAPHS", False)
# Prints out joint graph traced, before partitioning
debug_joint = os.environ.get("AOT_FX_GRAPHS_JOINT", False)

use_dynamic_shapes = os.getenv("AOT_DYNAMIC_SHAPES", False)

static_weight_shapes = True

# Applies CSE to the graph before partitioning
cse = True

# Restricts the amount of computation AOTAutograd can do.
max_dist_from_bw = 3

log_level = (
    logging.DEBUG if debug_partitioner or debug_graphs or debug_joint else logging.INFO
)

from .._dynamo.config_utils import install_config_module

# adds patch, save_config, invalid config checks, etc
install_config_module(sys.modules[__name__])
