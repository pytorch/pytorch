#!/bin/bash

# shellcheck source=./common.sh
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

if [[ ${BUILD_ENVIRONMENT} == *onnx* ]]; then
  pip install click mock tabulate networkx==2.0
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

# Uninstall pre-installed hypothesis and coverage to use an older version as newer
# versions remove the timeout parameter from settings which ideep/conv_transpose_test.py uses
$MAYBE_SUDO pip -q uninstall -y hypothesis
$MAYBE_SUDO pip -q uninstall -y coverage

# "pip install hypothesis==3.44.6" from official server is unreliable on
# CircleCI, so we host a copy on S3 instead
$MAYBE_SUDO pip -q install attrs==18.1.0 -f https://s3.amazonaws.com/ossci-linux/wheels/attrs-18.1.0-py2.py3-none-any.whl
$MAYBE_SUDO pip -q install coverage==4.5.1 -f https://s3.amazonaws.com/ossci-linux/wheels/coverage-4.5.1-cp36-cp36m-macosx_10_12_x86_64.whl
$MAYBE_SUDO pip -q install hypothesis==4.57.1

##############
# ONNX tests #
##############
if [[ "$BUILD_ENVIRONMENT" == *onnx* ]]; then
  pip install -q --user --no-use-pep517 "git+https://github.com/pytorch/vision.git@$(cat .github/ci_commit_pins/vision.txt)"
  pip install -q --user transformers==4.25.1
  pip install -q --user ninja flatbuffers==2.0 numpy==1.22.4 onnxruntime==1.14.0 beartype==0.10.4
  pip install -q --user onnx==1.13.1
  # TODO: change this when onnx-script is on testPypi
  pip install 'onnx-script @ git+https://github.com/microsoft/onnx-script@29241e15f5182be1384f1cf6ba203d7e2e125196'
  # numba requires numpy <= 1.20, onnxruntime requires numpy >= 1.21.
  # We don't actually need it for our tests, but it's imported if it's present, so uninstall.
  pip uninstall -q --yes numba
  # JIT C++ extensions require ninja, so put it into PATH.
  export PATH="/var/lib/jenkins/.local/bin:$PATH"
  "$ROOT_DIR/scripts/onnx/test.sh"
fi
