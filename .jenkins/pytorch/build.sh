#!/bin/bash

set -ex

# Required environment variable: $BUILD_ENVIRONMENT
# (This is set by default in the Docker images we build, so you don't
# need to set it yourself.

# shellcheck disable=SC2034
COMPACT_JOB_NAME="${BUILD_ENVIRONMENT}"

# shellcheck source=./common.sh
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

if [[ "$BUILD_ENVIRONMENT" == *-linux-xenial-py3-clang5-asan* ]]; then
  exec "$(dirname "${BASH_SOURCE[0]}")/build-asan.sh" "$@"
fi

if [[ "$BUILD_ENVIRONMENT" == *-mobile-*build* ]]; then
  exec "$(dirname "${BASH_SOURCE[0]}")/build-mobile.sh" "$@"
fi

if [[ "$BUILD_ENVIRONMENT" == *-mobile-code-analysis* ]]; then
  exec "$(dirname "${BASH_SOURCE[0]}")/build-mobile-code-analysis.sh" "$@"
fi

if [[ "$BUILD_ENVIRONMENT" == pytorch-linux-xenial-cuda11.1-cudnn8-py3-gcc7* ]]; then
  # Enabling DEPLOY build (embedded torch python interpreter, experimental)
  # only on one config for now, can expand later
  export USE_DEPLOY=ON

  # Deploy feature builds cpython. It requires these packages.
  # TODO move this to dockerfile?
  sudo apt-get -qq update
  sudo apt-get -qq install libffi-dev libbz2-dev libreadline-dev libncurses5-dev libncursesw5-dev libgdbm-dev libsqlite3-dev uuid-dev tk-dev
fi

echo "Python version:"
python --version

echo "GCC version:"
gcc --version

echo "CMake version:"
cmake --version

if [[ "$BUILD_ENVIRONMENT" == *cuda* ]]; then
  echo "NVCC version:"
  nvcc --version
fi

if [[ "$BUILD_ENVIRONMENT" == *coverage* ]]; then
  # enable build option in CMake
  export USE_CPP_CODE_COVERAGE=ON
fi

if [[ "$BUILD_ENVIRONMENT" == *cuda11* ]]; then
  # enable split torch_cuda build option in CMake
  export BUILD_SPLIT_CUDA=ON
fi

if [[ ${BUILD_ENVIRONMENT} == *"pure_torch"* ]]; then
  export BUILD_CAFFE2=OFF
fi

if [[ ${BUILD_ENVIRONMENT} == *"paralleltbb"* ]]; then
  export ATEN_THREADING=TBB
  export USE_TBB=1
elif [[ ${BUILD_ENVIRONMENT} == *"parallelnative"* ]]; then
  export ATEN_THREADING=NATIVE
fi

# TODO: Don't run this...
pip_install -r requirements.txt || true

# Enable LLVM dependency for TensorExpr testing
export USE_LLVM=/opt/llvm
export LLVM_DIR=/opt/llvm/lib/cmake/llvm

# TODO: Don't install this here
if ! which conda; then
  # In ROCm CIs, we are doing cross compilation on build machines with
  # intel cpu and later run tests on machines with amd cpu.
  # Also leave out two builds to make sure non-mkldnn builds still work.
  if [[ "$BUILD_ENVIRONMENT" != *rocm* && "$BUILD_ENVIRONMENT" != *-trusty-py3.5-* && "$BUILD_ENVIRONMENT" != *-xenial-cuda10.1-cudnn7-py3-* ]]; then
    pip_install mkl mkl-devel
    export USE_MKLDNN=1
  else
    export USE_MKLDNN=0
  fi
fi

if [[ "$BUILD_ENVIRONMENT" == *libtorch* ]]; then
  POSSIBLE_JAVA_HOMES=()
  POSSIBLE_JAVA_HOMES+=(/usr/local)
  POSSIBLE_JAVA_HOMES+=(/usr/lib/jvm/java-8-openjdk-amd64)
  POSSIBLE_JAVA_HOMES+=(/Library/Java/JavaVirtualMachines/*.jdk/Contents/Home)
  # Add the Windows-specific JNI
  POSSIBLE_JAVA_HOMES+=("$PWD/.circleci/windows-jni/")
  for JH in "${POSSIBLE_JAVA_HOMES[@]}" ; do
    if [[ -e "$JH/include/jni.h" ]] ; then
      # Skip if we're not on Windows but haven't found a JAVA_HOME
      if [[ "$JH" == "$PWD/.circleci/windows-jni/" && "$OSTYPE" != "msys" ]] ; then
        break
      fi
      echo "Found jni.h under $JH"
      export JAVA_HOME="$JH"
      export BUILD_JNI=ON
      break
    fi
  done
  if [ -z "$JAVA_HOME" ]; then
    echo "Did not find jni.h"
  fi
fi

# Use special scripts for Android builds
if [[ "${BUILD_ENVIRONMENT}" == *-android* ]]; then
  export ANDROID_NDK=/opt/ndk
  build_args=()
  if [[ "${BUILD_ENVIRONMENT}" == *-arm-v7a* ]]; then
    build_args+=("-DANDROID_ABI=armeabi-v7a")
  elif [[ "${BUILD_ENVIRONMENT}" == *-arm-v8a* ]]; then
    build_args+=("-DANDROID_ABI=arm64-v8a")
  elif [[ "${BUILD_ENVIRONMENT}" == *-x86_32* ]]; then
    build_args+=("-DANDROID_ABI=x86")
  elif [[ "${BUILD_ENVIRONMENT}" == *-x86_64* ]]; then
    build_args+=("-DANDROID_ABI=x86_64")
  fi
  if [[ "${BUILD_ENVIRONMENT}" == *vulkan* ]]; then
    build_args+=("-DUSE_VULKAN=ON")
  fi
  exec ./scripts/build_android.sh "${build_args[@]}" "$@"
fi

if [[ "$BUILD_ENVIRONMENT" != *android* && "$BUILD_ENVIRONMENT" == *vulkan-linux* ]]; then
  export USE_VULKAN=1
  # shellcheck disable=SC1091
  source /var/lib/jenkins/vulkansdk/setup-env.sh
fi

if [[ "$BUILD_ENVIRONMENT" == *rocm* ]]; then
  # hcc used to run out of memory, silently exiting without stopping
  # the build process, leaving undefined symbols in the shared lib,
  # causing undefined symbol errors when later running tests.
  # We used to set MAX_JOBS to 4 to avoid, but this is no longer an issue.
  if [ -z "$MAX_JOBS" ]; then
    export MAX_JOBS=$(($(nproc) - 1))
  fi

  if [[ -n "$IN_CI" && -z "$PYTORCH_ROCM_ARCH" ]]; then
      # Set ROCM_ARCH to gfx900 and gfx906 for CI builds, if user doesn't override.
      echo "Limiting PYTORCH_ROCM_ARCH to gfx90[06] for CI builds"
      export PYTORCH_ROCM_ARCH="gfx900;gfx906"
  fi

  python tools/amd_build/build_amd.py
  python setup.py install

  # remove sccache wrappers post-build; runtime compilation of MIOpen kernels does not yet fully support them
  sudo rm -f /opt/cache/bin/cc
  sudo rm -f /opt/cache/bin/c++
  sudo rm -f /opt/cache/bin/gcc
  sudo rm -f /opt/cache/bin/g++
  pushd /opt/rocm/llvm/bin
  if [[ -d original ]]; then
    sudo mv original/clang .
    sudo mv original/clang++ .
  fi
  sudo rm -rf original
  popd

  exit 0
fi

# sccache will fail for CUDA builds if all cores are used for compiling
# gcc 7 with sccache seems to have intermittent OOM issue if all cores are used
if [ -z "$MAX_JOBS" ]; then
  if { [[ "$BUILD_ENVIRONMENT" == *cuda* ]] || [[ "$BUILD_ENVIRONMENT" == *gcc7* ]]; } && which sccache > /dev/null; then
    export MAX_JOBS=$(($(nproc) - 1))
  fi
fi

# Target only our CI GPU machine's CUDA arch to speed up the build
export TORCH_CUDA_ARCH_LIST="5.2"

# Add sm_75 support for the Linux CUDA 11.1 cuDNN 8 CircleCI build
if [[ "$BUILD_ENVIRONMENT" == *xenial-cuda11.1-cudnn8*build ]]; then
  export TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST";7.5"
fi

if [[ "$BUILD_ENVIRONMENT" == *ppc64le* ]]; then
  export TORCH_CUDA_ARCH_LIST="6.0"
fi

if [[ "${BUILD_ENVIRONMENT}" == *clang* ]]; then
  export CC=clang
  export CXX=clang++
fi

# Patch required to build xla
if [[ "${BUILD_ENVIRONMENT}" == *xla* ]]; then
  clone_pytorch_xla
  # shellcheck disable=SC1091
  source "xla/.circleci/common.sh"
  apply_patches
fi

if [[ "${BUILD_ENVIRONMENT}" == pytorch-linux-xenial-py3.6-gcc7-build || "${BUILD_ENVIRONMENT}" == pytorch-linux-xenial-py3.6-gcc5.4-build ]]; then
  export USE_GLOO_WITH_OPENSSL=ON
fi

if [[ "$BUILD_ENVIRONMENT" == *-bazel-* ]]; then
  set -e

  get_bazel

  tools/bazel build :torch
else
  # check that setup.py would fail with bad arguments
  echo "The next three invocations are expected to fail with invalid command error messages."
  ( ! get_exit_code python setup.py bad_argument )
  ( ! get_exit_code python setup.py clean] )
  ( ! get_exit_code python setup.py clean bad_argument )

  if [[ "$BUILD_ENVIRONMENT" != *libtorch* ]]; then

    # ppc64le build fails when WERROR=1
    # set only when building other architectures
    # only use for "python setup.py install" line
    if [[ "$BUILD_ENVIRONMENT" != *ppc64le* && "$BUILD_ENVIRONMENT" != *clang* ]]; then
      WERROR=1 python setup.py bdist_wheel
      python -mpip install dist/*.whl
    else
      python setup.py bdist_wheel
      python -mpip install dist/*.whl
    fi

    # TODO: I'm not sure why, but somehow we lose verbose commands
    set -x

    if which sccache > /dev/null; then
      echo 'PyTorch Build Statistics'
      sccache --show-stats
    fi

    assert_git_not_dirty
    # Copy ninja build logs to dist folder
    mkdir -p dist
    if [ -f build/.ninja_log ]; then
      cp build/.ninja_log dist
    fi

    CUSTOM_TEST_ARTIFACT_BUILD_DIR=${CUSTOM_TEST_ARTIFACT_BUILD_DIR:-${PWD}/../}
    mkdir -pv "${CUSTOM_TEST_ARTIFACT_BUILD_DIR}"

    # Build custom operator tests.
    CUSTOM_OP_BUILD="${CUSTOM_TEST_ARTIFACT_BUILD_DIR}/custom-op-build"
    CUSTOM_OP_TEST="$PWD/test/custom_operator"
    python --version
    SITE_PACKAGES="$(python -c 'from distutils.sysconfig import get_python_lib; print(get_python_lib())')"
    mkdir -p "$CUSTOM_OP_BUILD"
    pushd "$CUSTOM_OP_BUILD"
    cmake "$CUSTOM_OP_TEST" -DCMAKE_PREFIX_PATH="$SITE_PACKAGES/torch" -DPYTHON_EXECUTABLE="$(which python)"
    make VERBOSE=1
    popd
    assert_git_not_dirty

    # Build jit hook tests
    JIT_HOOK_BUILD="${CUSTOM_TEST_ARTIFACT_BUILD_DIR}/jit-hook-build"
    JIT_HOOK_TEST="$PWD/test/jit_hooks"
    python --version
    SITE_PACKAGES="$(python -c 'from distutils.sysconfig import get_python_lib; print(get_python_lib())')"
    mkdir -p "$JIT_HOOK_BUILD"
    pushd "$JIT_HOOK_BUILD"
    cmake "$JIT_HOOK_TEST" -DCMAKE_PREFIX_PATH="$SITE_PACKAGES/torch" -DPYTHON_EXECUTABLE="$(which python)"
    make VERBOSE=1
    popd
    assert_git_not_dirty

    # Build custom backend tests.
    CUSTOM_BACKEND_BUILD="${CUSTOM_TEST_ARTIFACT_BUILD_DIR}/custom-backend-build"
    CUSTOM_BACKEND_TEST="$PWD/test/custom_backend"
    python --version
    mkdir -p "$CUSTOM_BACKEND_BUILD"
    pushd "$CUSTOM_BACKEND_BUILD"
    cmake "$CUSTOM_BACKEND_TEST" -DCMAKE_PREFIX_PATH="$SITE_PACKAGES/torch" -DPYTHON_EXECUTABLE="$(which python)"
    make VERBOSE=1
    popd
    assert_git_not_dirty
  else
    # Test standalone c10 build
    if [[ "$BUILD_ENVIRONMENT" == *xenial-cuda10.1-cudnn7-py3* ]]; then
      mkdir -p c10/build
      pushd c10/build
      cmake ..
      make -j
      popd
      assert_git_not_dirty
    fi

    # Test no-Python build
    echo "Building libtorch"
    # NB: Install outside of source directory (at the same level as the root
    # pytorch folder) so that it doesn't get cleaned away prior to docker push.
    BUILD_LIBTORCH_PY=$PWD/tools/build_libtorch.py
    mkdir -p ../cpp-build/caffe2
    pushd ../cpp-build/caffe2
    WERROR=1 VERBOSE=1 DEBUG=1 python "$BUILD_LIBTORCH_PY"
    popd
  fi
fi

# Test XLA build
if [[ "${BUILD_ENVIRONMENT}" == *xla* ]]; then
  XLA_DIR=xla
  # These functions are defined in .circleci/common.sh in pytorch/xla repo
  install_deps_pytorch_xla $XLA_DIR
  build_torch_xla $XLA_DIR
  assert_git_not_dirty
fi

if [[ "$BUILD_ENVIRONMENT" != *libtorch* && "$BUILD_ENVIRONMENT" != *bazel* ]]; then
  # export test times so that potential sharded tests that'll branch off this build will use consistent data
  # don't do this for libtorch as libtorch is C++ only and thus won't have python tests run on its build
  python test/run_test.py --export-past-test-times
fi
