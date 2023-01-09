# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Global flags for aot autograd
"""
import os
import logging

use_functionalize = True

use_fake_tensor = True

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

log_level = (
    logging.DEBUG if debug_partitioner or debug_graphs or debug_joint else logging.INFO
)
