#!/bin/bash

# Required environment variable: $BUILD_ENVIRONMENT
# (This is set by default in the Docker images we build, so you don't
# need to set it yourself.

# shellcheck disable=SC2034
COMPACT_JOB_NAME="${BUILD_ENVIRONMENT}"

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

echo "Testing pytorch"

if [ -n "${IN_CIRCLECI}" ]; then
  if [[ "$BUILD_ENVIRONMENT" == *-xenial-cuda9-* ]]; then
    # TODO: move this to Docker
    sudo apt-get -qq update
    sudo apt-get -qq install --allow-downgrades --allow-change-held-packages libnccl-dev=2.2.13-1+cuda9.0 libnccl2=2.2.13-1+cuda9.0
  fi

  if [[ "$BUILD_ENVIRONMENT" == *-xenial-cuda8-* ]] || [[ "$BUILD_ENVIRONMENT" == *-xenial-cuda9-cudnn7-py2* ]]; then
    # TODO: move this to Docker
    sudo apt-get -qq update
    sudo apt-get -qq install --allow-downgrades --allow-change-held-packages openmpi-bin libopenmpi-dev
    sudo apt-get -qq install --no-install-recommends openssh-client openssh-server
    sudo mkdir -p /var/run/sshd
  fi

  if [[ "$BUILD_ENVIRONMENT" == *-slow-* ]]; then
    export PYTORCH_TEST_WITH_SLOW=1
    export PYTORCH_TEST_SKIP_FAST=1
  fi
fi

# --user breaks ppc64le builds and these packages are already in ppc64le docker
if [[ "$BUILD_ENVIRONMENT" != *ppc64le* ]]; then
  # JIT C++ extensions require ninja.
  pip_install --user ninja
  # ninja is installed in /var/lib/jenkins/.local/bin
  export PATH="/var/lib/jenkins/.local/bin:$PATH"

  # TODO: move this to Docker
  pip_install --user hypothesis

  # TODO: move this to Docker
  PYTHON_VERSION=$(python -c 'import platform; print(platform.python_version())'|cut -c1)
  echo $PYTHON_VERSION
  # if [[ $PYTHON_VERSION == "2" ]]; then
  #   pip_install --user https://s3.amazonaws.com/ossci-linux/wheels/tensorboard-1.14.0a0-py2-none-any.whl
  # else
  #   pip_install --user https://s3.amazonaws.com/ossci-linux/wheels/tensorboard-1.14.0a0-py3-none-any.whl
  # fi
  pip_install --user tb-nightly
  # mypy will fail to install on Python <3.4.  In that case,
  # we just won't run these tests.

  # Temporarily skip mypy-0.720, with that our type hints checks failed with
  # ```torch/__init__.pyi:1307: error: Name 'pstrf' already defined (possibly by an import)```
  # https://circleci.com/api/v1.1/project/github/pytorch/pytorch/2185641/output/105/0?file=true
  pip_install --user mypy!=0.720 || true
fi

# faulthandler become built-in since 3.3
if [[ ! $(python -c "import sys; print(int(sys.version_info >= (3, 3)))") == "1" ]]; then
  pip_install --user faulthandler
fi

# DANGER WILL ROBINSON.  The LD_PRELOAD here could cause you problems
# if you're not careful.  Check this if you made some changes and the
# ASAN test is not working
if [[ "$BUILD_ENVIRONMENT" == *asan* ]]; then
    export ASAN_OPTIONS=detect_leaks=0:symbolize=1:strict_init_order=true
    # We suppress the vptr volation, since we have separate copies of
    # libprotobuf in both libtorch.so and libcaffe2.so, and it causes
    # the following problem:
    #    test_cse (__main__.TestJit) ... torch/csrc/jit/export.cpp:622:38:
    #        runtime error: member call on address ... which does not point
    #        to an object of type 'google::protobuf::MessageLite'
    #        ...: note: object is of type 'onnx_torch::ModelProto'
    #
    # This problem should be solved when libtorch.so and libcaffe2.so are
    # merged.
    export UBSAN_OPTIONS=print_stacktrace=1:suppressions=$PWD/ubsan.supp
    export PYTORCH_TEST_WITH_ASAN=1
    export PYTORCH_TEST_WITH_UBSAN=1
    # TODO: Figure out how to avoid hard-coding these paths
    export ASAN_SYMBOLIZER_PATH=/usr/lib/llvm-5.0/bin/llvm-symbolizer
    export LD_PRELOAD=/usr/lib/llvm-5.0/lib/clang/5.0.0/lib/linux/libclang_rt.asan-x86_64.so
    # Increase stack size, because ASAN red zones use more stack
    ulimit -s 81920

    (cd test && python -c "import torch")
    echo "The next three invocations are expected to crash; if they don't that means ASAN/UBSAN is misconfigured"
    (cd test && ! get_exit_code python -c "import torch; torch._C._crash_if_csrc_asan(3)")
    (cd test && ! get_exit_code python -c "import torch; torch._C._crash_if_csrc_ubsan(0)")
    (cd test && ! get_exit_code python -c "import torch; torch._C._crash_if_aten_asan(3)")
fi

if [[ "$BUILD_ENVIRONMENT" == *rocm* ]]; then
  export PYTORCH_TEST_WITH_ROCM=1
fi

if [[ "${BUILD_ENVIRONMENT}" == *-NO_AVX-* ]]; then
  export ATEN_CPU_CAPABILITY=default
elif [[ "${BUILD_ENVIRONMENT}" == *-NO_AVX2-* ]]; then
  export ATEN_CPU_CAPABILITY=avx
fi

test_python_nn() {
  time python test/run_test.py --include nn --verbose
  assert_git_not_dirty
}

test_python_all_except_nn() {
  time python test/run_test.py --exclude nn --verbose
  assert_git_not_dirty
}

test_aten() {
  # Test ATen
  # The following test(s) of ATen have already been skipped by caffe2 in rocm environment:
  # scalar_tensor_test, basic, native_test
  if ([[ "$BUILD_ENVIRONMENT" != *asan* ]] && [[ "$BUILD_ENVIRONMENT" != *rocm* ]]); then
    echo "Running ATen tests with pytorch lib"
    TORCH_LIB_PATH=$(python -c "import site; print(site.getsitepackages()[0])")/torch/lib
    # NB: the ATen test binaries don't have RPATH set, so it's necessary to
    # put the dynamic libraries somewhere were the dynamic linker can find them.
    # This is a bit of a hack.
    if [[ "$BUILD_ENVIRONMENT" == *ppc64le* ]]; then
      SUDO=sudo
    fi

    ${SUDO} ln -s "$TORCH_LIB_PATH"/libc10* build/bin
    ${SUDO} ln -s "$TORCH_LIB_PATH"/libcaffe2* build/bin
    ${SUDO} ln -s "$TORCH_LIB_PATH"/libmkldnn* build/bin
    ${SUDO} ln -s "$TORCH_LIB_PATH"/libnccl* build/bin

    ls build/bin
    aten/tools/run_tests.sh build/bin
    assert_git_not_dirty
  fi
}

test_torchvision() {
  pip_install --user git+https://github.com/pytorch/vision.git@2b73a4846773a670632b29fb2fc2ac57df7bce5d
}

test_libtorch() {
  if [[ "$BUILD_TEST_LIBTORCH" == "1" ]]; then
    echo "Testing libtorch"
    python test/cpp/jit/tests_setup.py setup
    CPP_BUILD="$PWD/../cpp-build"
    if [[ "$BUILD_ENVIRONMENT" == *cuda* ]]; then
      "$CPP_BUILD"/caffe2/build/bin/test_jit
    else
      "$CPP_BUILD"/caffe2/build/bin/test_jit "[cpu]"
    fi
    python test/cpp/jit/tests_setup.py shutdown
    python tools/download_mnist.py --quiet -d test/cpp/api/mnist
    OMP_NUM_THREADS=2 TORCH_CPP_TEST_MNIST_PATH="test/cpp/api/mnist" "$CPP_BUILD"/caffe2/build/bin/test_api
    assert_git_not_dirty
  fi
}

test_custom_script_ops() {
  if [[ "$BUILD_TEST_LIBTORCH" == "1" ]]; then
    echo "Testing custom script operators"
    CUSTOM_OP_BUILD="$PWD/../custom-op-build"
    pushd test/custom_operator
    cp -a "$CUSTOM_OP_BUILD" build
    # Run tests Python-side and export a script module.
    python test_custom_ops.py -v
    python model.py --export-script-module=model.pt
    # Run tests C++-side and load the exported script module.
    build/test_custom_ops ./model.pt
    popd
    assert_git_not_dirty
  fi
}

test_xla() {
  export XLA_USE_XRT=1 XRT_DEVICE_MAP="CPU:0;/job:localservice/replica:0/task:0/device:XLA_CPU:0"
  export XRT_WORKERS="localservice:0;grpc://localhost:40934"
  pushd xla
  python test/test_operations.py
  python test/test_train_mnist.py --tidy
  popd
  assert_git_not_dirty
}

(cd test && python -c "import torch; print(torch.__config__.show())")
(cd test && python -c "import torch; print(torch.__config__.parallel_info())")

if [[ "${BUILD_ENVIRONMENT}" == *xla* ]]; then
  test_torchvision
  test_xla
elif [[ "${BUILD_ENVIRONMENT}" == *-test1 ]]; then
  test_torchvision
  test_python_nn
elif [[ "${BUILD_ENVIRONMENT}" == *-test2 ]]; then
  test_python_all_except_nn
  test_aten
  test_libtorch
  test_custom_script_ops
else
  test_torchvision
  test_python_nn
  test_python_all_except_nn
  test_aten
  test_libtorch
  test_custom_script_ops
fi
