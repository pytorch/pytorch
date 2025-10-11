#!/bin/bash

set -ex -o pipefail

# Suppress ANSI color escape sequences
export TERM=vt100

# shellcheck source=./common.sh
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
# shellcheck source=./common-build.sh
source "$(dirname "${BASH_SOURCE[0]}")/common-build.sh"

echo "Environment variables"
env

echo "Testing FA3 stable wheel still works with currently built torch"

echo "Installing ABI Stable FA3 wheel"
# The wheel was built on https://github.com/Dao-AILab/flash-attention/commit/b3846b059bf6b143d1cd56879933be30a9f78c81
# on torch nightly torch==2.9.0.dev20250830+cu129
$MAYBE_SUDO pip -q install https://s3.amazonaws.com/ossci-linux/wheels/flash_attn_3-3.0.0b1-cp39-abi3-linux_x86_64.whl

pushd flash-attention/hopper
export PYTHONPATH=$PWD
pytest -v -s \
  "test_flash_attn.py::test_flash_attn_output[1-1-192-False-False-False-0.0-False-False-mha-dtype0]" \
  "test_flash_attn.py::test_flash_attn_varlen_output[511-1-64-True-False-False-0.0-False-False-gqa-dtype2]" \
  "test_flash_attn.py::test_flash_attn_kvcache[1-128-128-False-False-True-None-0.0-False-False-True-False-True-False-gqa-dtype0]" \
  "test_flash_attn.py::test_flash_attn_race_condition[97-97-192-True-dtype0]" \
  "test_flash_attn.py::test_flash_attn_combine[2-3-64-dtype1]" \
  "test_flash_attn.py::test_flash3_bw_compatibility"
popd
