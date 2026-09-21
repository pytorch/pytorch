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
$MAYBE_SUDO pip -q install einops ninja
$MAYBE_SUDO pip -q install flash-attn-3 --index-url https://download.pytorch.org/whl/cu130

# To be as robust against upstream changes as possible, minimally only stage
# the relevant test files in a clean dir
HARNESS_DIR="$(mktemp -d)"
cp flash-attention/hopper/test_flash_attn.py \
   flash-attention/hopper/test_util.py \
   flash-attention/hopper/padding.py \
   "${HARNESS_DIR}/"

# Everything that imports torch must run from a dir other than the pytorch
# source root: from the root, `import torch` resolves to the unbuilt ./torch
# source package (no generated torch/version.py) and shadows the installed build.
pushd "${HARNESS_DIR}"

# Smoke check that the wheel is importable the way upstream intends: as an
# installed package. Prefer the packaged flash_attn_3.flash_attn_interface
# (see #2458); fall back to the legacy top-level module for older wheels.
python3 - <<'EOF'
try:
    from flash_attn_3.flash_attn_interface import flash_attn_func  # packaged (new)
    print("FA3 import: packaged flash_attn_3.flash_attn_interface")
except ImportError:
    from flash_attn_interface import flash_attn_func  # loose top-level (legacy)
    print("FA3 import: legacy top-level flash_attn_interface")
EOF

export FLASH_ATTENTION_ENABLE_OPCHECK=TRUE  # Enable testing for compile on the smoke tests
pytest -v -s \
  "test_flash_attn.py::test_flash_attn_output[1-1-192-False-False-False-0.0-False-False-mha-dtype0]" \
  "test_flash_attn.py::test_flash_attn_varlen_output[511-1-64-True-False-False-0.0-False-False-gqa-dtype2]" \
  "test_flash_attn.py::test_flash_attn_kvcache[1-128-128-False-False-True-None-0.0-False-False-True-False-True-False-gqa-dtype0]" \
  "test_flash_attn.py::test_flash_attn_race_condition[97-97-192-True-dtype0]" \
  "test_flash_attn.py::test_flash_attn_combine[2-3-64-dtype1]"
popd
