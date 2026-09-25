#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -eux

echo "====: Working directory: $(pwd)"

python -m pip install --upgrade pip
echo "====: Installed pip version: $(python -m pip --version)"

# Ensure cmake is at least the max version needed by PyTorch and Kineto.
python -m pip install --upgrade 'cmake>=3.27'
echo "====: Installed cmake version: $(cmake --version)"
