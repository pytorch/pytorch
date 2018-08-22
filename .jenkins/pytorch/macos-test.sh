#!/bin/bash

COMPACT_JOB_NAME="${BUILD_ENVIRONMENT}-test"
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

export PATH="/usr/local/bin:$PATH"

# Set up conda environment
export PYTORCH_ENV_DIR="${HOME}/pytorch-ci-env"
# If a local installation of conda doesn't exist, we download and install conda
if [ ! -d "${PYTORCH_ENV_DIR}/miniconda3" ]; then
  mkdir -p ${PYTORCH_ENV_DIR}
  curl https://repo.continuum.io/miniconda/Miniconda3-latest-MacOSX-x86_64.sh -o ${PYTORCH_ENV_DIR}/miniconda3.sh
  bash ${PYTORCH_ENV_DIR}/miniconda3.sh -b -p ${PYTORCH_ENV_DIR}/miniconda3
fi
export PATH="${PYTORCH_ENV_DIR}/miniconda3/bin:$PATH"
source ${PYTORCH_ENV_DIR}/miniconda3/bin/activate
conda install -y mkl mkl-include numpy pyyaml setuptools cmake cffi ninja
rm -rf ${PYTORCH_ENV_DIR}/miniconda3/lib/python3.6/site-packages/torch*

git submodule update --init --recursive
export CMAKE_PREFIX_PATH=${PYTORCH_ENV_DIR}/miniconda3/

# Test PyTorch
if [[ "${JOB_BASE_NAME}" == *cuda9.2* ]]; then
  # Eigen gives "explicit specialization of class must precede its first use" error
  # when compiling with Xcode 9.1 toolchain, so we have to use Xcode 8.2 toolchain instead.
  export DEVELOPER_DIR=/Library/Developer/CommandLineTools
else
  export DEVELOPER_DIR=/Applications/Xcode9.app/Contents/Developer
fi
export MACOSX_DEPLOYMENT_TARGET=10.9
export CXX=clang++
export CC=clang
# If we run too many parallel jobs, we will OOM
export MAX_JOBS=2

export IMAGE_COMMIT_TAG=${BUILD_ENVIRONMENT}-${IMAGE_COMMIT_ID}

# Download torch binaries in the test jobs
rm -rf ${PYTORCH_ENV_DIR}/miniconda3/lib/python3.6/site-packages/torch*
aws s3 cp s3://ossci-macos-build/pytorch/${IMAGE_COMMIT_TAG}.7z ${IMAGE_COMMIT_TAG}.7z
7z x ${IMAGE_COMMIT_TAG}.7z -o"${PYTORCH_ENV_DIR}/miniconda3/lib/python3.6/site-packages"

test_python_all() {
  echo "Ninja version: $(ninja --version)"
  python test/run_test.py --verbose
}

test_cpp_api() {
  # C++ API

  # NB: Install outside of source directory (at the same level as the root
  # pytorch folder) so that it doesn't get cleaned away prior to docker push.
  # But still clean it before we perform our own build.
  #
  CPP_BUILD="$PWD/../cpp-build"
  rm -rf $CPP_BUILD
  mkdir -p $CPP_BUILD/caffe2

  BUILD_LIBTORCH_PY=$PWD/tools/build_libtorch.py
  pushd $CPP_BUILD/caffe2
  WERROR=1 VERBOSE=1 DEBUG=1 python $BUILD_LIBTORCH_PY
  popd

  python tools/download_mnist.py --quiet -d test/cpp/api/mnist

  # Unfortunately it seems like the test can't load from miniconda3
  # without these paths being set
  export DYLD_LIBRARY_PATH="$DYLD_LIBRARY_PATH:$PWD/miniconda3/lib"
  export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$PWD/miniconda3/lib"
  "$CPP_BUILD"/caffe2/bin/test_api
}

if [ -z "${JOB_BASE_NAME}" ] || [[ "${JOB_BASE_NAME}" == *-test ]]; then
  test_python_all
  test_cpp_api
else
  if [[ "${JOB_BASE_NAME}" == *-test1 ]]; then
    test_python_all
  elif [[ "${JOB_BASE_NAME}" == *-test2 ]]; then
    test_cpp_api
  fi
fi
