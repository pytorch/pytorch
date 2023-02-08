#!/bin/bash -ex

# shellcheck source=./common.sh
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

if [[ ${BUILD_ENVIRONMENT} == *onnx* ]]; then
  pip install click mock tabulate networkx==2.2
  pip -q install --user "file:///var/lib/jenkins/workspace/third_party/onnx#egg=onnx"
fi

# Skip tests in environments where they are not built/applicable
if [[ "${BUILD_ENVIRONMENT}" == *-android* ]]; then
  echo 'Skipping tests'
  exit 0
fi
if [[ "${BUILD_ENVIRONMENT}" == *-rocm* ]]; then
  # temporary to locate some kernel issues on the CI nodes
  export HSAKMT_DEBUG_LEVEL=4
fi
# These additional packages are needed for circleci ROCm builds.
if [[ $BUILD_ENVIRONMENT == *rocm* ]]; then
    # Need networkx 2.0 because bellmand_ford was moved in 2.1 . Scikit-image by
    # defaults installs the most recent networkx version, so we install this lower
    # version explicitly before scikit-image pulls it in as a dependency
    pip install networkx==2.0
    # click - onnx
    pip install --progress-bar off click protobuf tabulate virtualenv mock typing-extensions
fi

################################################################################
# Python tests #
################################################################################
if [[ "$BUILD_ENVIRONMENT" == *cmake* ]]; then
  exit 0
fi

# If pip is installed as root, we must use sudo.
# CircleCI docker images could install conda as jenkins user, or use the OS's python package.
PIP=$(which pip)
PIP_USER=$(stat --format '%U' $PIP)
CURRENT_USER=$(id -u -n)
if [[ "$PIP_USER" = root && "$CURRENT_USER" != root ]]; then
  MAYBE_SUDO=sudo
fi

##############
# ONNX tests #
##############
if [[ "$BUILD_ENVIRONMENT" == *onnx* ]]; then
  pip install -q --user --no-use-pep517 "git+https://github.com/pytorch/vision.git@$(cat .github/ci_commit_pins/vision.txt)"
  pip install -q --user ninja flatbuffers==2.0 numpy==1.22.4 beartype==0.10.4 onnx==1.12.0 coloredlogs==15.0.1
  pip install -q --user -i https://test.pypi.org/simple/ ort-nightly==1.14.0.dev20230207006
  # TODO: change this when onnx-script is on testPypi
  pip install 'onnx-script @ git+https://github.com/microsoft/onnx-script@4f3ff0d806d0d0f30cecdfd3e8b094b1e492d44a'
  # numba requires numpy <= 1.20, onnxruntime requires numpy >= 1.21.
  # We don't actually need it for our tests, but it's imported if it's present, so uninstall.
  pip uninstall -q --yes numba
  # JIT C++ extensions require ninja, so put it into PATH.
  export PATH="/var/lib/jenkins/.local/bin:$PATH"
  "$ROOT_DIR/scripts/onnx/test.sh"
fi
