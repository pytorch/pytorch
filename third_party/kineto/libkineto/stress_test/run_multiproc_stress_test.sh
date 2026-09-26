#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

build_cmd="buck build --show-output @mode/opt -c fbcode.nvcc_arch=b200a -m ovr_config//third-party/cuda/constraints:12.8 :kineto_stress_test"
build_out=$($build_cmd)

mpi_path=/usr/local/fbcode/platform010-aarch64/bin//mpirun
binary_path="$HOME/fbsource/${build_out#* }"
num_ranks=2

echo "Binary Path:"
echo "$binary_path"

$mpi_path -np $num_ranks "$binary_path" ./stress_test_dense_mp.json
