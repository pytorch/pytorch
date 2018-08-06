#!/bin/bash

COMPACT_JOB_NAME="${BUILD_ENVIRONMENT}-test"
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

# Required environment variable: $BUILD_ENVIRONMENT
# (This is set by default in the Docker images we build, so you don't
# need to set it yourself.

echo "Testing pytorch"

test_python_nn() {
  time python test/run_test.py --include nn --verbose
}

test_python_all_except_nn() {
  time python test/run_test.py --exclude nn --verbose
}

test_aten() {
  # Test ATen
  if [[ "$BUILD_ENVIRONMENT" != *asan* ]]; then
    echo "Running ATen tests with pytorch lib"
    TORCH_LIB_PATH=$(python -c "import site; print(site.getsitepackages()[0])")/torch/lib
    # NB: the ATen test binaries don't have RPATH set, so it's necessary to
    # put the dynamic libraries somewhere were the dynamic linker can find them.
    # This is a bit of a hack.
    ln -s "$TORCH_LIB_PATH"/libcaffe2* build/bin
    ln -s "$TORCH_LIB_PATH"/libnccl* build/bin
    ls build/bin
    aten/tools/run_tests.sh build/bin
  fi
}

test_torchvision() {
  rm -rf ninja

  echo "Installing torchvision at branch master"
  rm -rf vision
  # TODO: This git clone is bad, it means pushes to torchvision can break
  # PyTorch CI
  git clone https://github.com/pytorch/vision --quiet
  pushd vision
  # python setup.py install with a tqdm dependency is broken in the
  # Travis Python nightly (but not in latest Python nightlies, so
  # this should be a transient requirement...)
  # See https://github.com/pytorch/pytorch/issues/7525
  #time python setup.py install
  pip install .
  popd
}

test_libtorch() {
  if [[ "$BUILD_TEST_LIBTORCH" == "1" ]]; then
     echo "Testing libtorch"
     CPP_BUILD="$PWD/../cpp-build"
     if [[ "$BUILD_ENVIRONMENT" == *cuda* ]]; then
       "$CPP_BUILD"/caffe2/bin/test_jit
     else
       "$CPP_BUILD"/caffe2/bin/test_jit "[cpu]"
     fi
     python tools/download_mnist.py --quiet -d test/cpp/api/mnist
     OMP_NUM_THREADS=2 "$CPP_BUILD"/caffe2/bin/test_api
  fi
}

if [ -z "${JOB_BASE_NAME}" ] || [[ "${JOB_BASE_NAME}" == *-test ]]; then
  test_python_nn
  test_python_all_except_nn
else
  if [[ "${JOB_BASE_NAME}" == *-test1 ]]; then
    test_python_nn
  elif [[ "${JOB_BASE_NAME}" == *-test2 ]]; then
    test_python_all_except_nn
  fi
fi
