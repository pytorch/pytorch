#!/bin/bash

set -ex -o pipefail

# Required environment variable: $BUILD_ENVIRONMENT
# (This is set by default in the Docker images we build, so you don't
# need to set it yourself.

# shellcheck source=./common.sh
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
# shellcheck source=./common-build.sh
source "$(dirname "${BASH_SOURCE[0]}")/common-build.sh"

echo "Python version:"
python --version

echo "GCC version:"
gcc --version

echo "CMake version:"
cmake --version

echo "Environment variables:"
env

# The sccache wrapped version of nvcc gets put in /opt/cache/lib in docker since
# there are some issues if it is always wrapped, so we need to add it to PATH
# during CI builds.
# https://github.com/pytorch/pytorch/blob/0b6c0898e6c352c8ea93daec854e704b41485375/.ci/docker/common/install_cache.sh#L97
export PATH="/opt/cache/lib:$PATH"

if [[ "$BUILD_ENVIRONMENT" == *cuda* ]]; then
  # Use jemalloc during compilation to mitigate https://github.com/pytorch/pytorch/issues/116289
  export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libjemalloc.so.2
  echo "NVCC version:"
  nvcc --version
fi

if [[ "$BUILD_ENVIRONMENT" == *cuda11* ]]; then
  if [[ "$BUILD_ENVIRONMENT" != *clang* ]]; then
    # TODO: there is a linking issue when building with UCC using clang,
    # disable it for now and to be fix later.
    # TODO: disable UCC temporarily to enable CUDA 12.1 in CI
    export USE_UCC=1
    export USE_SYSTEM_UCC=1
  fi
fi

if [[ ${BUILD_ENVIRONMENT} == *"parallelnative"* ]]; then
  export ATEN_THREADING=NATIVE
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
  # CMAKE_PREFIX_PATH precedences
  # 1. $CONDA_PREFIX, if defined. This follows the pytorch official build instructions.
  # 2. /opt/conda/envs/py_${ANACONDA_PYTHON_VERSION}, if ANACONDA_PYTHON_VERSION defined.
  #    This is for CI, which defines ANACONDA_PYTHON_VERSION but not CONDA_PREFIX.
  # 3. $(conda info --base). The fallback value of pytorch official build
  #    instructions actually refers to this.
  #    Commonly this is /opt/conda/
  if [[ -v CONDA_PREFIX ]]; then
    export CMAKE_PREFIX_PATH=${CONDA_PREFIX}
  elif [[ -v ANACONDA_PYTHON_VERSION ]]; then
    export CMAKE_PREFIX_PATH="/opt/conda/envs/py_${ANACONDA_PYTHON_VERSION}"
  else
    # already checked by `! which conda`
    CMAKE_PREFIX_PATH="$(conda info --base)"
    export CMAKE_PREFIX_PATH
  fi

  # Workaround required for MKL library linkage
  # https://github.com/pytorch/pytorch/issues/119557
  if [[ "$ANACONDA_PYTHON_VERSION" = "3.12" || "$ANACONDA_PYTHON_VERSION" = "3.13" ]]; then
    export CMAKE_LIBRARY_PATH="/opt/conda/envs/py_$ANACONDA_PYTHON_VERSION/lib/"
    export CMAKE_INCLUDE_PATH="/opt/conda/envs/py_$ANACONDA_PYTHON_VERSION/include/"
  fi
fi

if [[ "$BUILD_ENVIRONMENT" == *aarch64* ]]; then
  export USE_MKLDNN=1
  export USE_MKLDNN_ACL=1
  export ACL_ROOT_DIR=/acl
fi

if [[ "$BUILD_ENVIRONMENT" == *riscv64* ]]; then
  if [[ -f /opt/riscv-cross-env/bin/activate ]]; then
    # shellcheck disable=SC1091
    source /opt/riscv-cross-env/bin/activate
  else
    echo "Activation file not found"
    exit 1
  fi

  export CMAKE_CROSSCOMPILING=TRUE
  export CMAKE_SYSTEM_NAME=Linux
  export CMAKE_SYSTEM_PROCESSOR=riscv64

  export USE_CUDA=0
  export USE_MKLDNN=0

  export SLEEF_TARGET_EXEC_USE_QEMU=ON
  sudo chown -R jenkins /var/lib/jenkins/workspace /opt

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

if [[ "$BUILD_ENVIRONMENT" == *vulkan* ]]; then
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

if [[ "$BUILD_ENVIRONMENT" == *xpu* ]]; then
  # shellcheck disable=SC1091
  source /opt/intel/oneapi/compiler/latest/env/vars.sh
  # shellcheck disable=SC1091
  source /opt/intel/oneapi/ccl/latest/env/vars.sh
  # shellcheck disable=SC1091
  source /opt/intel/oneapi/mpi/latest/env/vars.sh
  # Enable XCCL build
  export USE_XCCL=1
  export USE_MPI=0
  # XPU kineto feature dependencies are not fully ready, disable kineto build as temp WA
  export USE_KINETO=0
  export TORCH_XPU_ARCH_LIST=pvc
fi

# sccache will fail for CUDA builds if all cores are used for compiling
# gcc 7 with sccache seems to have intermittent OOM issue if all cores are used
if [ -z "$MAX_JOBS" ]; then
  if { [[ "$BUILD_ENVIRONMENT" == *cuda* ]]; } && which sccache > /dev/null; then
    export MAX_JOBS=$(($(nproc) - 1))
  fi
fi

# TORCH_CUDA_ARCH_LIST must be passed from an environment variable
if [[ "$BUILD_ENVIRONMENT" == *cuda* && -z "$TORCH_CUDA_ARCH_LIST" ]]; then
  echo "TORCH_CUDA_ARCH_LIST must be defined"
  exit 1
fi

# We only build FlashAttention files for CUDA 8.0+, and they require large amounts of
# memory to build and will OOM

if [[ "$BUILD_ENVIRONMENT" == *cuda* ]] && echo "${TORCH_CUDA_ARCH_LIST}" | tr ' ' '\n' | sed 's/$/>= 8.0/' | bc | grep -q 1; then
  J=2  # default to 2 jobs
  case "$RUNNER" in
    linux.12xlarge.memory|linux.24xlarge.memory)
      J=24
      ;;
  esac
  echo "Building FlashAttention with job limit $J"
  export BUILD_CUSTOM_STEP="ninja -C build flash_attention -j ${J}"
fi

if [[ "${BUILD_ENVIRONMENT}" == *clang* ]]; then
  export CC=clang
  export CXX=clang++
fi

if [[ "$BUILD_ENVIRONMENT" == *-clang*-asan* ]]; then
  if [[ "$BUILD_ENVIRONMENT" == *cuda* ]]; then
    export USE_CUDA=1
  fi
  export USE_ASAN=1
  export REL_WITH_DEB_INFO=1
  export UBSAN_FLAGS="-fno-sanitize-recover=all"
fi

if [[ "${BUILD_ENVIRONMENT}" == *no-ops* ]]; then
  export USE_PER_OPERATOR_HEADERS=0
fi

if [[ "${BUILD_ENVIRONMENT}" == *-pch* ]]; then
    export USE_PRECOMPILED_HEADERS=1
fi

if [[ "${BUILD_ENVIRONMENT}" != *cuda* ]]; then
  export BUILD_STATIC_RUNTIME_BENCHMARK=ON
fi

if [[ "$BUILD_ENVIRONMENT" == *-debug* ]]; then
  export CMAKE_BUILD_TYPE=RelWithAssert
fi

# Do not change workspace permissions for ROCm and s390x CI jobs
# as it can leave workspace with bad permissions for cancelled jobs
if [[ "$BUILD_ENVIRONMENT" != *rocm* && "$BUILD_ENVIRONMENT" != *s390x* && "$BUILD_ENVIRONMENT" != *riscv64* && -d /var/lib/jenkins/workspace ]]; then
  # Workaround for dind-rootless userid mapping (https://github.com/pytorch/ci-infra/issues/96)
  WORKSPACE_ORIGINAL_OWNER_ID=$(stat -c '%u' "/var/lib/jenkins/workspace")
  cleanup_workspace() {
    echo "sudo may print the following warning message that can be ignored. The chown command will still run."
    echo "    sudo: setrlimit(RLIMIT_STACK): Operation not permitted"
    echo "For more details refer to https://github.com/sudo-project/sudo/issues/42"
    sudo chown -R "$WORKSPACE_ORIGINAL_OWNER_ID" /var/lib/jenkins/workspace
  }
  # Disable shellcheck SC2064 as we want to parse the original owner immediately.
  # shellcheck disable=SC2064
  trap_add cleanup_workspace EXIT
  sudo chown -R jenkins /var/lib/jenkins/workspace
  git config --global --add safe.directory /var/lib/jenkins/workspace
fi

if [[ "$BUILD_ENVIRONMENT" == *-bazel-* ]]; then
  set -e -o pipefail

  get_bazel
  python3 tools/optional_submodules.py checkout_eigen

  # Leave 1 CPU free and use only up to 80% of memory to reduce the change of crashing
  # the runner
  BAZEL_MEM_LIMIT="--local_ram_resources=HOST_RAM*.8"
  BAZEL_CPU_LIMIT="--local_cpu_resources=HOST_CPUS-1"

  if [[ "$CUDA_VERSION" == "cpu" ]]; then
    # Build torch, the Python module, and tests for CPU-only
    tools/bazel build --config=no-tty "${BAZEL_MEM_LIMIT}" "${BAZEL_CPU_LIMIT}" --config=cpu-only :torch :torch/_C.so :all_tests
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
    if [[ "$BUILD_ENVIRONMENT" != *rocm*  && "$BUILD_ENVIRONMENT" != *xla* && "$BUILD_ENVIRONMENT" != *riscv64* ]]; then
      # Install numpy-2.0.2 for builds which are backward compatible with 1.X
      python -mpip install numpy==2.0.2

      WERROR=1 python setup.py clean

      WERROR=1 python -m build --wheel --no-isolation
    else
      python setup.py clean
      if [[ "$BUILD_ENVIRONMENT" == *xla* ]]; then
        source .ci/pytorch/install_cache_xla.sh
      fi
      python -m build --wheel --no-isolation
    fi
    pip_install_whl "$(echo dist/*.whl)"

    if [[ "${BUILD_ADDITIONAL_PACKAGES:-}" == *vision* ]]; then
      install_torchvision
    fi

    if [[ "${BUILD_ADDITIONAL_PACKAGES:-}" == *audio* ]]; then
      install_torchaudio
    fi

    if [[ "${BUILD_ADDITIONAL_PACKAGES:-}" == *torchrec* || "${BUILD_ADDITIONAL_PACKAGES:-}" == *fbgemm* ]]; then
      install_torchrec_and_fbgemm
    fi

    if [[ "${BUILD_ADDITIONAL_PACKAGES:-}" == *torchao* ]]; then
      install_torchao
    fi

    if [[ "$BUILD_ENVIRONMENT" == *xpu* ]]; then
      echo "Checking that xpu is compiled"
      pushd dist/
      if python -c 'import torch; exit(0 if torch.xpu._is_compiled() else 1)'; then
        echo "XPU support is compiled in."
      else
        echo "XPU support is NOT compiled in."
        exit 1
      fi
      popd
    fi

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
    SITE_PACKAGES="$(python -c 'import site; print(";".join([x for x in site.getsitepackages()] + [x + "/torch" for x in site.getsitepackages()]))')"

    mkdir -p "$CUSTOM_OP_BUILD"
    pushd "$CUSTOM_OP_BUILD"
    cmake "$CUSTOM_OP_TEST" -DCMAKE_PREFIX_PATH="$SITE_PACKAGES" -DPython_EXECUTABLE="$(which python)" \
          -DCMAKE_MODULE_PATH="$CUSTOM_TEST_MODULE_PATH" -DUSE_ROCM="$CUSTOM_TEST_USE_ROCM"
    make VERBOSE=1
    popd
    assert_git_not_dirty

    # Build jit hook tests
    JIT_HOOK_BUILD="${CUSTOM_TEST_ARTIFACT_BUILD_DIR}/jit-hook-build"
    JIT_HOOK_TEST="$PWD/test/jit_hooks"
    python --version
    SITE_PACKAGES="$(python -c 'import site; print(";".join([x for x in site.getsitepackages()] + [x + "/torch" for x in site.getsitepackages()]))')"
    mkdir -p "$JIT_HOOK_BUILD"
    pushd "$JIT_HOOK_BUILD"
    cmake "$JIT_HOOK_TEST" -DCMAKE_PREFIX_PATH="$SITE_PACKAGES" -DPython_EXECUTABLE="$(which python)" \
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
    cmake "$CUSTOM_BACKEND_TEST" -DCMAKE_PREFIX_PATH="$SITE_PACKAGES" -DPython_EXECUTABLE="$(which python)" \
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
# don't do this for bazel or s390x or riscv64 as they don't use sccache
if [[ "$BUILD_ENVIRONMENT" != *s390x* && "$BUILD_ENVIRONMENT" != *riscv64* && "$BUILD_ENVIRONMENT" != *-bazel-* ]]; then
  print_sccache_stats
fi
