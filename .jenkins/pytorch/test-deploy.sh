#!/bin/bash
set -eux

# shellcheck source=./common.sh
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

# shellcheck source=./test-common.sh
source "$(dirname "${BASH_SOURCE[0]}")/test-common.sh"

python torch/csrc/deploy/example/generate_examples.py
ln -sf "$TORCH_LIB_DIR"/libtorch* "$TORCH_BIN_DIR"
ln -sf "$TORCH_LIB_DIR"/libshm* "$TORCH_BIN_DIR"
ln -sf "$TORCH_LIB_DIR"/libc10* "$TORCH_BIN_DIR"
"$TORCH_BIN_DIR"/test_deploy
"$TORCH_BIN_DIR"/test_deploy_gpu
assert_git_not_dirty
