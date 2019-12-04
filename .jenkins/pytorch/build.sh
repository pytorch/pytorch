#!/bin/bash

# Required environment variable: $BUILD_ENVIRONMENT
# (This is set by default in the Docker images we build, so you don't
# need to set it yourself.
# Example value: py3.6-devtoolset7-rocmrpm-centos7.5
# to build using python3, devtoolset7, ROCm CentOS RPMs
# And "venv" to BUILD_ENVIRONMENT string to use python3 venv,
# for example, py3.6-venv-devtoolset7-rocmrpm-centos7.5.
# Print a message and exit if environment variable is not set
${BUILD_ENVIRONMENT:?"Environment variable BUILD_ENVIRONMENT must be set"}

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

if [[ "$BUILD_ENVIRONMENT" == *-linux-xenial-py3-clang5-mobile* ]]; then
  # Use linux host toolchain + mobile build options in order to build & test
  # mobile libtorch without having to setup Android/iOS toolchain/simulator.
  exec ./scripts/build_mobile.sh -DBUILD_BINARY=ON "$@"
fi

echo "Python version:"
python --version

echo "GCC version:"
gcc --version

echo "CMake version:"
cmake --version

# TODO: Don't run this...
# pip_install -r requirements.txt || true

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

if [[ "$BUILD_ENVIRONMENT" == *libtorch* ]]; then
  POSSIBLE_JAVA_HOMES=()
  POSSIBLE_JAVA_HOMES+=(/usr/local)
  POSSIBLE_JAVA_HOMES+=(/usr/lib/jvm/java-8-openjdk-amd64)
  POSSIBLE_JAVA_HOMES+=(/Library/Java/JavaVirtualMachines/*.jdk/Contents/Home)
  for JH in "${POSSIBLE_JAVA_HOMES[@]}" ; do
    if [[ -e "$JH/include/jni.h" ]] ; then
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
  export BUILD_PYTORCH_MOBILE=1
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
    for compiler in cc c++ gcc g++ clang clang++; do
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

  if [[ "$BUILD_ENVIRONMENT" == *py3*venv* ]]; then
    echo "Using python3 virtual environment"
    LANG=en_US.UTF-8 python3 -m venv build
    source build/bin/activate
    pip3 install -r requirements.txt || true
    LANG=en_US.UTF-8 python3 tools/amd_build/build_amd.py
    LANG=en_US.UTF-8 USE_CUDA=0 python3 setup.py install --user
  elif [[ "$BUILD_ENVIRONMENT" == *py3* ]]; then
    echo "Using python3 systemn environment - sudo required"
    pip3 install -r requirements.txt || true
    LANG=en_US.UTF-8 python3 tools/amd_build/build_amd.py
    LANG=en_US.UTF-8 USE_CUDA=0 python3 setup.py install --user
  else
    echo "Using python2 system environment - sudo required"
    pip2 install -r requirements.txt || true
    python2 tools/amd_build/build_amd.py
    USE_CUDA=0 python2 setup.py install --user
  fi

  # runtime compilation of MIOpen kernels manages to crash sccache - hence undo the wrapping
  bash tools/amd_build/unwrap_clang.sh

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

# Patch required to build xla
if [[ "${BUILD_ENVIRONMENT}" == *xla* ]]; then
  git clone --recursive https://github.com/pytorch/xla.git
  ./xla/scripts/apply_patches.sh
fi

if [[ "${BUILD_ENVIRONMENT}" == *clang* ]]; then
  export CC=clang
  export CXX=clang++
fi

echo "DEBUG: call setup.py"

# check that setup.py would fail with bad arguments
echo "The next three invocations are expected to fail with invalid command error messages."
( ! get_exit_code python setup.py bad_argument )
( ! get_exit_code python setup.py clean] )
( ! get_exit_code python setup.py clean bad_argument )

if [[ "$BUILD_ENVIRONMENT" != *libtorch* ]]; then

  # ppc64le build fails when WERROR=1
  # set only when building other architectures
  # only use for "python setup.py install" line
  if [[ "$BUILD_ENVIRONMENT" != *ppc64le*  && "$BUILD_ENVIRONMENT" != *clang* ]]; then
    WERROR=1 python setup.py install
  else
    python setup.py install
  fi

  # TODO: I'm not sure why, but somehow we lose verbose commands
  set -x

  if which sccache > /dev/null; then
    echo 'PyTorch Build Statistics'
    sccache --show-stats
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

  # Build custom operator tests.
  CUSTOM_OP_BUILD="$PWD/../custom-op-build"
  CUSTOM_OP_TEST="$PWD/test/custom_operator"
  python --version
  SITE_PACKAGES="$(python -c 'from distutils.sysconfig import get_python_lib; print(get_python_lib())')"
  mkdir "$CUSTOM_OP_BUILD"
  pushd "$CUSTOM_OP_BUILD"
  cmake "$CUSTOM_OP_TEST" -DCMAKE_PREFIX_PATH="$SITE_PACKAGES/torch" -DPYTHON_EXECUTABLE="$(which python)"
  make VERBOSE=1
  popd
  assert_git_not_dirty
else
  # Test standalone c10 build
  if [[ "$BUILD_ENVIRONMENT" == *xenial-cuda9-cudnn7-py3* ]]; then
    mkdir -p c10/build
    pushd c10/build
    cmake -DCMAKE_VERBOSE_MAKEFILE=1 ..
    make VERBOSE=1 -j
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
  WERROR=1 VERBOSE=1 DEBUG=1 python $BUILD_LIBTORCH_PY
  popd
fi

# Test XLA build
if [[ "${BUILD_ENVIRONMENT}" == *xla* ]]; then
  # TODO: Move this to Dockerfile.

  pip_install lark-parser

  # Bazel doesn't work with sccache gcc. https://github.com/bazelbuild/bazel/issues/3642
  sudo add-apt-repository "deb http://apt.llvm.org/xenial/ llvm-toolchain-xenial-7 main"
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

  bazels3cache --bucket=${XLA_CLANG_CACHE_S3_BUCKET_NAME} --maxEntrySizeBytes=0
  pushd xla
  export CC=clang-7 CXX=clang++-7
  # Use cloud cache to build when available.
  sed -i '/bazel build/ a --remote_http_cache=http://localhost:7777 \\' build_torch_xla_libs.sh

  python setup.py install
  popd
  assert_git_not_dirty
fi
