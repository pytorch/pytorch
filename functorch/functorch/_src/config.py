# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Global flags for aot autograd
"""

use_functionalize = False

# TODO: flip this to true by default
# Waiting on
#   https://github.com/pytorch/pytorch/pull/81617
#   https://github.com/pytorch/pytorch/pull/81609
#   https://github.com/pytorch/pytorch/pull/81604
#   fix for test_aot_autograd_exhaustive_sgn_cpu_float32 _efficientzerotensor
#   fix for complex numbers
use_fake_tensor = False
