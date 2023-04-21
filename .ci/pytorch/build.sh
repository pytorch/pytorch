#!/bin/bash

set -ex

# Required environment variable: $BUILD_ENVIRONMENT
# (This is set by default in the Docker images we build, so you don't
# need to set it yourself.

# shellcheck source=./common.sh
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
# shellcheck source=./common-build.sh
source "$(dirname "${BASH_SOURCE[0]}")/common-build.sh"

if [[ "$BUILD_ENVIRONMENT" == *-clang7-asan* ]]; then
  exec "$(dirname "${BASH_SOURCE[0]}")/build-asan.sh" "$@"
fi

if [[ "$BUILD_ENVIRONMENT" == *-mobile-*build* ]]; then
  exec "$(dirname "${BASH_SOURCE[0]}")/build-mobile.sh" "$@"
fi

echo "Python version:"
python --version

echo "GCC version:"
gcc --version

echo "CMake version:"
cmake --version

echo "Environment variables:"
env

if [[ "$BUILD_ENVIRONMENT" == *cuda* ]]; then
  echo "NVCC version:"
  nvcc --version
fi

if [[ "$BUILD_ENVIRONMENT" == *cuda11* ]]; then
  if [[ "$BUILD_ENVIRONMENT" != *cuda11.3* && "$BUILD_ENVIRONMENT" != *clang* ]]; then
    # TODO: there is a linking issue when building with UCC using clang,
    # disable it for now and to be fix later.
    export USE_UCC=1
    export USE_SYSTEM_UCC=1
  fi
fi

if [[ ${BUILD_ENVIRONMENT} == *"caffe2"* ]]; then
  echo "Caffe2 build is ON"
  export BUILD_CAFFE2=ON
fi

if [[ ${BUILD_ENVIRONMENT} == *"paralleltbb"* ]]; then
  export ATEN_THREADING=TBB
  export USE_TBB=1
elif [[ ${BUILD_ENVIRONMENT} == *"parallelnative"* ]]; then
  export ATEN_THREADING=NATIVE
fi

# Enable LLVM dependency for TensorExpr testing
if [[ "$BUILD_ENVIRONMENT" == *rocm* ]]; then
  export USE_LLVM=/opt/rocm/llvm
  export LLVM_DIR=/opt/rocm/llvm/lib/cmake/llvm
else
  export USE_LLVM=/opt/llvm
  export LLVM_DIR=/opt/llvm/lib/cmake/llvm
fi

if ! which conda; then
  # In ROCm CIs, we are doing cross compilation on build machines with
  # intel cpu and later run tests on machines with amd cpu.
  # Also leave out two builds to make sure non-mkldnn builds still work.
  if [[ "$BUILD_ENVIRONMENT" != *rocm* ]]; then
    export USE_MKLDNN=1
  else
    export USE_MKLDNN=0
  fi
else
  export CMAKE_PREFIX_PATH=/opt/conda
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
  build_args+=("-DUSE_LITE_INTERPRETER_PROFILER=OFF")
  exec ./scripts/build_android.sh "${build_args[@]}" "$@"
fi

if [[ "$BUILD_ENVIRONMENT" != *android* && "$BUILD_ENVIRONMENT" == *vulkan* ]]; then
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

  if [[ -n "$CI" && -z "$PYTORCH_ROCM_ARCH" ]]; then
      # Set ROCM_ARCH to gfx906 for CI builds, if user doesn't override.
      echo "Limiting PYTORCH_ROCM_ARCH to gfx906 for CI builds"
      export PYTORCH_ROCM_ARCH="gfx906"
  fi

  # hipify sources
  python tools/amd_build/build_amd.py
fi

# sccache will fail for CUDA builds if all cores are used for compiling
# gcc 7 with sccache seems to have intermittent OOM issue if all cores are used
if [ -z "$MAX_JOBS" ]; then
  if { [[ "$BUILD_ENVIRONMENT" == *cuda* ]] || [[ "$BUILD_ENVIRONMENT" == *gcc7* ]]; } && which sccache > /dev/null; then
    export MAX_JOBS=$(($(nproc) - 1))
  fi
fi

# TORCH_CUDA_ARCH_LIST must be passed from an environment variable
if [[ "$BUILD_ENVIRONMENT" == *cuda* && -z "$TORCH_CUDA_ARCH_LIST" ]]; then
  echo "TORCH_CUDA_ARCH_LIST must be defined"
  exit 1
fi

if [[ "${BUILD_ENVIRONMENT}" == *clang* ]]; then
  export CC=clang
  export CXX=clang++
fi

if [[ "${BUILD_ENVIRONMENT}" == *no-ops* ]]; then
  export USE_PER_OPERATOR_HEADERS=0
fi

if [[ "${BUILD_ENVIRONMENT}" == *-pch* ]]; then
    export USE_PRECOMPILED_HEADERS=1
fi

if [[ "${BUILD_ENVIRONMENT}" == *linux-focal-py3.7-gcc7-build*  ]]; then
  export USE_GLOO_WITH_OPENSSL=ON
fi

if [[ "${BUILD_ENVIRONMENT}" != *android* && "${BUILD_ENVIRONMENT}" != *cuda* ]]; then
  export BUILD_STATIC_RUNTIME_BENCHMARK=ON
fi

if [[ "${BUILD_ENVIRONMENT}" == *sm86* ]]; then
  export BUILD_AOT_INDUCTOR_TEST=ON
fi

if [[ "$BUILD_ENVIRONMENT" == *-bazel-* ]]; then
  set -e

  get_bazel
  install_sccache_nvcc_for_bazel

  # Leave 1 CPU free and use only up to 80% of memory to reduce the change of crashing
  # the runner
  BAZEL_MEM_LIMIT="--local_ram_resources=HOST_RAM*.8"
  BAZEL_CPU_LIMIT="--local_cpu_resources=HOST_CPUS-1"

  if [[ "$CUDA_VERSION" == "cpu" ]]; then
    # Build torch, the Python module, and tests for CPU-only
    tools/bazel build --config=no-tty "${BAZEL_MEM_LIMIT}" "${BAZEL_CPU_LIMIT}" --config=cpu-only :torch :_C.so :all_tests
  else
    tools/bazel build --config=no-tty "${BAZEL_MEM_LIMIT}" "${BAZEL_CPU_LIMIT}" //...
  fi
else
  # check that setup.py would fail with bad arguments
  echo "The next three invocations are expected to fail with invalid command error messages."
  ( ! get_exit_code python setup.py bad_argument )
  ( ! get_exit_code python setup.py clean] )
  ( ! get_exit_code python setup.py clean bad_argument )

  if [[ "$BUILD_ENVIRONMENT" != *libtorch* ]]; then

    # rocm builds fail when WERROR=1
    # XLA test build fails when WERROR=1
    # set only when building other architectures
    # or building non-XLA tests.
    if [[ "$BUILD_ENVIRONMENT" != *rocm*  &&
          "$BUILD_ENVIRONMENT" != *xla* ]]; then
      WERROR=1 python setup.py bdist_wheel
    else
      python setup.py bdist_wheel
    fi
    pip_install_whl "$(echo dist/*.whl)"

    # TODO: I'm not sure why, but somehow we lose verbose commands
    set -x

    assert_git_not_dirty
    # Copy ninja build logs to dist folder
    mkdir -p dist
    if [ -f build/.ninja_log ]; then
      cp build/.ninja_log dist
    fi

    if [[ "$BUILD_ENVIRONMENT" == *rocm* ]]; then
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
    fi

    CUSTOM_TEST_ARTIFACT_BUILD_DIR=${CUSTOM_TEST_ARTIFACT_BUILD_DIR:-"build/custom_test_artifacts"}
    CUSTOM_TEST_USE_ROCM=$([[ "$BUILD_ENVIRONMENT" == *rocm* ]] && echo "ON" || echo "OFF")
    CUSTOM_TEST_MODULE_PATH="${PWD}/cmake/public"
    mkdir -pv "${CUSTOM_TEST_ARTIFACT_BUILD_DIR}"

    # Build custom operator tests.
    CUSTOM_OP_BUILD="${CUSTOM_TEST_ARTIFACT_BUILD_DIR}/custom-op-build"
    CUSTOM_OP_TEST="$PWD/test/custom_operator"
    python --version
    SITE_PACKAGES="$(python -c 'from distutils.sysconfig import get_python_lib; print(get_python_lib())')"
    mkdir -p "$CUSTOM_OP_BUILD"
    pushd "$CUSTOM_OP_BUILD"
    cmake "$CUSTOM_OP_TEST" -DCMAKE_PREFIX_PATH="$SITE_PACKAGES/torch" -DPYTHON_EXECUTABLE="$(which python)" \
          -DCMAKE_MODULE_PATH="$CUSTOM_TEST_MODULE_PATH" -DUSE_ROCM="$CUSTOM_TEST_USE_ROCM"
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
    cmake "$JIT_HOOK_TEST" -DCMAKE_PREFIX_PATH="$SITE_PACKAGES/torch" -DPYTHON_EXECUTABLE="$(which python)" \
          -DCMAKE_MODULE_PATH="$CUSTOM_TEST_MODULE_PATH" -DUSE_ROCM="$CUSTOM_TEST_USE_ROCM"
    make VERBOSE=1
    popd
    assert_git_not_dirty

    # Build custom backend tests.
    CUSTOM_BACKEND_BUILD="${CUSTOM_TEST_ARTIFACT_BUILD_DIR}/custom-backend-build"
    CUSTOM_BACKEND_TEST="$PWD/test/custom_backend"
    python --version
    mkdir -p "$CUSTOM_BACKEND_BUILD"
    pushd "$CUSTOM_BACKEND_BUILD"
    cmake "$CUSTOM_BACKEND_TEST" -DCMAKE_PREFIX_PATH="$SITE_PACKAGES/torch" -DPYTHON_EXECUTABLE="$(which python)" \
          -DCMAKE_MODULE_PATH="$CUSTOM_TEST_MODULE_PATH" -DUSE_ROCM="$CUSTOM_TEST_USE_ROCM"
    make VERBOSE=1
    popd
    assert_git_not_dirty
  else
    # Test no-Python build
    echo "Building libtorch"

    # This is an attempt to mitigate flaky libtorch build OOM error. By default, the build parallelization
    # is set to be the number of CPU minus 2. So, let's try a more conservative value here. A 4xlarge has
    # 16 CPUs
    MAX_JOBS=$(nproc --ignore=4)
    export MAX_JOBS

    # NB: Install outside of source directory (at the same level as the root
    # pytorch folder) so that it doesn't get cleaned away prior to docker push.
    BUILD_LIBTORCH_PY=$PWD/tools/build_libtorch.py
    mkdir -p ../cpp-build/caffe2
    pushd ../cpp-build/caffe2
    WERROR=1 VERBOSE=1 DEBUG=1 python "$BUILD_LIBTORCH_PY"
    popd
  fi
fi

if [[ "$BUILD_ENVIRONMENT" != *libtorch* && "$BUILD_ENVIRONMENT" != *bazel* ]]; then
  # export test times so that potential sharded tests that'll branch off this build will use consistent data
  # don't do this for libtorch as libtorch is C++ only and thus won't have python tests run on its build
  python tools/stats/export_test_times.py
fi

print_sccache_stats
