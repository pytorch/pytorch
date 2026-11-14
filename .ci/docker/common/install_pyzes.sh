#!/bin/bash

set -ex

source "$(dirname "${BASH_SOURCE[0]}")/common_utils.sh"

# Lets torch.xpu.device_count() count devices without initializing the XPU
# runtime, which poisons fork. Pinned to match generate_binary_build_matrix.py.
pip_install pyzes==0.1.3
# torch silently falls back to the runtime if pyzes can't load libze_loader.so.1
python -c "import pyzes"
