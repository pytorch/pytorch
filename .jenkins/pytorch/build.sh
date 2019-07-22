#!/bin/bash

# Required environment variable: $BUILD_ENVIRONMENT
# (This is set by default in the Docker images we build, so you don't
# need to set it yourself.

# shellcheck disable=SC2034
COMPACT_JOB_NAME="${BUILD_ENVIRONMENT}"

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

# For distributed, four environmental configs:
# (1) build with only NCCL
# (2) build with NCCL and MPI
# (3) build with only MPI
# (4) build with neither
if [[ "$BUILD_ENVIRONMENT" == *-xenial-cuda9-* ]]; then
  # TODO: move this to Docker
  sudo apt-get -qq update
  sudo apt-get -qq install --allow-downgrades --allow-change-held-packages libnccl-dev=2.2.13-1+cuda9.0 libnccl2=2.2.13-1+cuda9.0
fi

if [[ "$BUILD_ENVIRONMENT" == *-xenial-cuda9*gcc7* ]] || [[ "$BUILD_ENVIRONMENT" == *-xenial-cuda9-* ]] || [[ "$BUILD_ENVIRONMENT" == *-trusty-py2.7.9* ]]; then
  # TODO: move this to Docker
  sudo apt-get -qq update
  if [[ "$BUILD_ENVIRONMENT" == *-trusty-py2.7.9* ]]; then
    sudo apt-get -qq install openmpi-bin libopenmpi-dev
  else
    sudo apt-get -qq install --allow-downgrades --allow-change-held-packages openmpi-bin libopenmpi-dev
  fi
  sudo apt-get -qq install --no-install-recommends openssh-client openssh-server
  sudo mkdir -p /var/run/sshd
fi

if [[ "$BUILD_ENVIRONMENT" == *-linux-xenial-py3-clang5-asan* ]]; then
  exec "$(dirname "${BASH_SOURCE[0]}")/build-asan.sh" "$@"
fi

echo "Python version:"
python --version

echo "GCC version:"
gcc --version

echo "CMake version:"
cmake --version

# TODO: Don't run this...
pip_install -r requirements.txt || true

# TODO: Don't install this here
if ! which conda; then
  # In ROCm CIs, we are doing cross compilation on build machines with
  # intel cpu and later run tests on machines with amd cpu.
  # Also leave out two builds to make sure non-mkldnn builds still work.
  if [[ "$BUILD_ENVIRONMENT" != *rocm* && "$BUILD_ENVIRONMENT" != *-trusty-py3.5-* && "$BUILD_ENVIRONMENT" != *-xenial-cuda9-cudnn7-py3-* ]]; then
    pip_install mkl mkl-devel
    export USE_MKLDNN=1
  else
    export USE_MKLDNN=0
  fi
fi

# Use special scripts for Android builds
if [[ "${BUILD_ENVIRONMENT}" == *-android* ]]; then
  export ANDROID_NDK=/opt/ndk
  build_args=()
  build_args+=("-DBUILD_CAFFE2_MOBILE=OFF")
  build_args+=("-DCMAKE_PREFIX_PATH=$(python -c 'from distutils.sysconfig import get_python_lib; print(get_python_lib())')")
  build_args+=("-DPYTHON_EXECUTABLE=$(python -c 'import sys; print(sys.executable)')")
  exec ./scripts/build_android.sh "${build_args[@]}" "$@"
fi

if [[ "$BUILD_ENVIRONMENT" == *rocm* ]]; then
  # When hcc runs out of memory, it silently exits without stopping
  # the build process, leaving undefined symbols in the shared lib
  # which will cause undefined symbol errors when later running
  # tests. Setting MAX_JOBS to smaller number to make CI less flaky.
  export MAX_JOBS=4

  # ROCm CI is using Caffe2 docker images, which needs these wrapper
  # scripts to correctly use sccache.
  if [ -n "${SCCACHE_BUCKET}" ]; then
    mkdir -p ./sccache

    SCCACHE="$(which sccache)"
    if [ -z "${SCCACHE}" ]; then
      echo "Unable to find sccache..."
      exit 1
    fi

    # Setup wrapper scripts
    for compiler in cc c++ gcc g++; do
      (
        echo "#!/bin/sh"
        echo "exec $SCCACHE $(which $compiler) \"\$@\""
      ) > "./sccache/$compiler"
      chmod +x "./sccache/$compiler"
    done

    export CACHE_WRAPPER_DIR="$PWD/sccache"

    # CMake must find these wrapper scripts
    export PATH="$CACHE_WRAPPER_DIR:$PATH"
  fi

  python tools/amd_build/build_amd.py
  # OPENCV is needed to enable ImageInput operator in caffe2 resnet5_trainer
  # LMDB is needed to read datasets from https://download.caffe2.ai/databases/resnet_trainer.zip
  USE_ROCM=1 USE_LMDB=1 USE_OPENCV=1 python setup.py install --user

  ORIG_COMP=/opt/rocm/hcc/bin/clang-*_original
  if [ -e $ORIG_COMP ]; then
    # runtime compilation of MIOpen kernels manages to crash sccache - hence undo the wrapping
    # note that the wrapping always names the compiler "clang-7.0_original"
    WRAPPED=/opt/rocm/hcc/bin/clang-[0-99]
    sudo mv $ORIG_COMP $WRAPPED

  fi
  exit 0
fi

# sccache will fail for CUDA builds if all cores are used for compiling
# gcc 7 with sccache seems to have intermittent OOM issue if all cores are used
if [ -z "$MAX_JOBS" ]; then
  if ([[ "$BUILD_ENVIRONMENT" == *cuda* ]] || [[ "$BUILD_ENVIRONMENT" == *gcc7* ]]) && which sccache > /dev/null; then
    export MAX_JOBS=$(($(nproc) - 1))
  fi
fi

# Target only our CI GPU machine's CUDA arch to speed up the build
export TORCH_CUDA_ARCH_LIST="5.2"

if [[ "$BUILD_ENVIRONMENT" == *ppc64le* ]]; then
  export TORCH_CUDA_ARCH_LIST="6.0"
fi

if [[ "$BUILD_ENVIRONMENT" == *trusty-py3.6-gcc5.4* ]]; then
  export DEBUG=1
fi

# Patch required to build xla
if [[ "${BUILD_ENVIRONMENT}" == *xla* ]]; then
  git clone --recursive https://github.com/pytorch/xla.git
  ./xla/scripts/apply_patches.sh
fi


# check that setup.py would fail with bad arguments
echo "The next three invocations are expected to fail with invalid command error messages."
( ! get_exit_code python setup.py bad_argument )
( ! get_exit_code python setup.py clean] )
( ! get_exit_code python setup.py clean bad_argument )

# ppc64le build fails when WERROR=1
# set only when building other architectures
# only use for "python setup.py install" line
if [[ "$BUILD_ENVIRONMENT" != *ppc64le* ]]; then
  WERROR=1 python setup.py install
else
  python setup.py install
fi

assert_git_not_dirty

# Test documentation build
if [[ "$BUILD_ENVIRONMENT" == *xenial-cuda9-cudnn7-py3* ]]; then
  pushd docs
  # TODO: Don't run this here
  pip_install -r requirements.txt || true
  LC_ALL=C make html
  popd
  assert_git_not_dirty
fi

# Test standalone c10 build
if [[ "$BUILD_ENVIRONMENT" == *xenial-cuda9-cudnn7-py3* ]]; then
  mkdir -p c10/build
  pushd c10/build
  cmake ..
  make -j
  popd
  assert_git_not_dirty
fi

# Test no-Python build
if [[ "$BUILD_TEST_LIBTORCH" == "1" ]]; then
  echo "Building libtorch"
  # NB: Install outside of source directory (at the same level as the root
  # pytorch folder) so that it doesn't get cleaned away prior to docker push.
  BUILD_LIBTORCH_PY=$PWD/tools/build_libtorch.py
  mkdir -p ../cpp-build/caffe2
  pushd ../cpp-build/caffe2
  WERROR=1 VERBOSE=1 DEBUG=1 python $BUILD_LIBTORCH_PY
  popd

  # Build custom operator tests.
  CUSTOM_OP_BUILD="$PWD/../custom-op-build"
  CUSTOM_OP_TEST="$PWD/test/custom_operator"
  SITE_PACKAGES="$(python -c 'from distutils.sysconfig import get_python_lib; print(get_python_lib())')"
  mkdir "$CUSTOM_OP_BUILD"
  pushd "$CUSTOM_OP_BUILD"
  CMAKE_PREFIX_PATH="$SITE_PACKAGES/torch" cmake "$CUSTOM_OP_TEST"
  make VERBOSE=1
  popd
  assert_git_not_dirty
fi

# Test XLA build
if [[ "${BUILD_ENVIRONMENT}" == *xla* ]]; then
  # TODO: Move this to Dockerfile.

  pip_install lark-parser

  # Bazel doesn't work with sccache gcc. https://github.com/bazelbuild/bazel/issues/3642
  sudo add-apt-repository "deb http://apt.llvm.org/trusty/ llvm-toolchain-trusty-7 main"
  wget -O - https://apt.llvm.org/llvm-snapshot.gpg.key|sudo apt-key add -
  sudo apt-get -qq update

  # Install clang-7 clang++-7 for xla
  sudo apt-get -qq install clang-7 clang++-7

  # Bazel dependencies
  sudo apt-get -qq install pkg-config zip zlib1g-dev unzip
  # XLA build requires Bazel
  wget https://github.com/bazelbuild/bazel/releases/download/0.24.1/bazel-0.24.1-installer-linux-x86_64.sh
  chmod +x bazel-*.sh
  sudo ./bazel-*.sh
  BAZEL="$(which bazel)"
  if [ -z "${BAZEL}" ]; then
    echo "Unable to find bazel..."
    exit 1
  fi

  # Install bazels3cache for cloud cache
  sudo apt-get -qq install npm
  npm config set strict-ssl false
  curl -sL https://deb.nodesource.com/setup_6.x | sudo -E bash -
  sudo apt-get install -qq nodejs
  sudo npm install -g bazels3cache
  BAZELS3CACHE="$(which bazels3cache)"
  if [ -z "${BAZELS3CACHE}" ]; then
    echo "Unable to find bazels3cache..."
    exit 1
  fi

  bazels3cache --bucket=ossci-compiler-cache-circleci-xla --maxEntrySizeBytes=0
  pushd xla
  export CC=clang-7 CXX=clang++-7
  # Use cloud cache to build when available.
  sed -i '/bazel build/ a --remote_http_cache=http://localhost:7777 \\' build_torch_xla_libs.sh

  python setup.py install
  popd
  assert_git_not_dirty
fi
