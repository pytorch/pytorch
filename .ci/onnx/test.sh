#!/bin/bash

# shellcheck source=./common.sh
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

if [[ "$BUILD_ENVIRONMENT" == *onnx* ]]; then
  pip -q install --user "file:///var/lib/jenkins/workspace/third_party/onnx#egg=onnx"
  # TODO: This can be removed later once vision is also part of the Docker image
  pip install -q --user --no-use-pep517 "git+https://github.com/pytorch/vision.git@$(cat .github/ci_commit_pins/vision.txt)"

  # JIT C++ extensions require ninja, so put it into PATH.
  export PATH="/var/lib/jenkins/.local/bin:$PATH"
  "$ROOT_DIR/scripts/onnx/test.sh"
fi
