#!/bin/bash

# shellcheck source=./common.sh
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

# Use to retry ONNX test, only retry it twice
retry () {
    "$@" || (sleep 60 && "$@")
}

if [[ "$BUILD_ENVIRONMENT" == *onnx* ]]; then
  # TODO: This can be removed later once vision is also part of the Docker image
  pip install -q --user --no-use-pep517 "git+https://github.com/pytorch/vision.git@$(cat .github/ci_commit_pins/vision.txt)"
  # JIT C++ extensions require ninja, so put it into PATH.
  export PATH="/var/lib/jenkins/.local/bin:$PATH"
  # NB: ONNX test is fast (~15m) so it's ok to retry it few more times to avoid any flaky issue, we
  # need to bring this to the standard PyTorch run_test eventually. The issue will be tracked in
  # https://github.com/pytorch/pytorch/issues/98626
  retry "$ROOT_DIR/scripts/onnx/test.sh"
fi
