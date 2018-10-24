#!/bin/bash

# For distributed, four environmental configs:
# (1) build with only NCCL
# (2) build with NCCL and MPI
# (3) build with only MPI
# (4) build with neither
if [[ "$BUILD_ENVIRONMENT" == *-xenial-cuda9-* ]]; then
  # TODO: move this to Docker
  sudo apt-get update
  sudo apt-get install -y --allow-downgrades --allow-change-held-packages libnccl-dev=2.2.13-1+cuda9.0 libnccl2=2.2.13-1+cuda9.0
fi

if [[ "$BUILD_ENVIRONMENT" == *-xenial-cuda8-* ]] || [[ "$BUILD_ENVIRONMENT" == *-xenial-cuda9-cudnn7-py2* ]] || [[ "$BUILD_ENVIRONMENT" == *-trusty-py2.7.9* ]]; then
  # TODO: move this to Docker
  sudo apt-get update
  sudo apt-get install -y --allow-downgrades --allow-change-held-packages openmpi-bin libopenmpi-dev
  sudo apt-get install -y --no-install-recommends openssh-client openssh-server
  sudo mkdir -p /var/run/sshd
fi

if [[ "$BUILD_ENVIRONMENT" == "pytorch-linux-xenial-py3-clang5-asan" ]]; then
  exec "$(dirname "${BASH_SOURCE[0]}")/build-asan.sh" $*
fi

# Required environment variable: $BUILD_ENVIRONMENT
# (This is set by default in the Docker images we build, so you don't
# need to set it yourself.

COMPACT_JOB_NAME="${BUILD_ENVIRONMENT}-build"
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

echo "Python version:"
python --version

echo "GCC version:"
gcc --version

echo "CMake version:"
cmake --version

# TODO: Don't run this...
pip install -r requirements.txt || true

if [[ "$BUILD_ENVIRONMENT" == *rocm* ]]; then
  # This is necessary in order to cross compile (or else we'll have missing GPU device).
  export HCC_AMDGPU_TARGET=gfx900

  # These environment variables are not set on CI when we were running as the Jenkins user.
  # The HIP Utility scripts require these environment variables to be set in order to run without error.
  export LANG=C.UTF-8
  export LC_ALL=C.UTF-8

  # This environment variable enabled HCC Optimizations that speed up the linking stage.
  # https://github.com/RadeonOpenCompute/hcc#hcc-with-thinlto-linking
  export KMTHINLTO=1

  # Need the libc++1 and libc++abi1 libraries to allow torch._C to load at runtime
  sudo apt-get install libc++1
  sudo apt-get install libc++abi1

  # When hcc runs out of memory, it silently exits without stopping
  # the build process, leaving undefined symbols in the shared lib
  # which will cause undefined symbol errors when later running
  # tests. Setting MAX_JOBS to smaller number to make CI less flaky.
  export MAX_JOBS=4

  python tools/amd_build/build_pytorch_amd.py
  python tools/amd_build/build_caffe2_amd.py
  USE_ROCM=1 python setup.py install --user
  exit 0
fi

# TODO: Don't install this here
if ! which conda; then
  pip install mkl mkl-devel
fi

# sccache will fail for CUDA builds if all cores are used for compiling
# gcc 7 with sccache seems to have intermittent OOM issue if all cores are used
if [ -z "$MAX_JOBS" ]; then
  if ([[ "$BUILD_ENVIRONMENT" == *cuda* ]] || [[ "$BUILD_ENVIRONMENT" == *gcc7* ]]) && which sccache > /dev/null; then
    export MAX_JOBS=`expr $(nproc) - 1`
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

# ppc64le build fails when WERROR=1
# set only when building other architectures
# only use for "python setup.py install" line
if [[ "$BUILD_ENVIRONMENT" != *ppc64le* ]]; then
  WERROR=1 python setup.py install
elif [[ "$BUILD_ENVIRONMENT" == *ppc64le* ]]; then
  python setup.py install
fi


# Add the test binaries so that they won't be git clean'ed away
git add -f build/bin

# Test documentation build
if [[ "$BUILD_ENVIRONMENT" == *xenial-cuda8-cudnn6-py3* ]]; then
  pushd docs
  # TODO: Don't run this here
  pip install -r requirements.txt || true
  LC_ALL=C make html
  popd
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
fi
