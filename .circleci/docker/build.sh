#!/bin/bash

set -ex

image="$1"
shift

if [ -z "${image}" ]; then
  echo "Usage: $0 IMAGE"
  exit 1
fi

# TODO: Generalize
OS="ubuntu"
DOCKERFILE="${OS}/Dockerfile"
if [[ "$image" == *-cuda* ]]; then
  DOCKERFILE="${OS}-cuda/Dockerfile"
fi

if [[ "$image" == *-trusty* ]]; then
  UBUNTU_VERSION=14.04
elif [[ "$image" == *-xenial* ]]; then
  UBUNTU_VERSION=16.04
elif [[ "$image" == *-artful* ]]; then
  UBUNTU_VERSION=17.10
elif [[ "$image" == *-bionic* ]]; then
  UBUNTU_VERSION=18.04
fi

TRAVIS_DL_URL_PREFIX="https://s3.amazonaws.com/travis-python-archives/binaries/ubuntu/14.04/x86_64"

# It's annoying to rename jobs every time you want to rewrite a
# configuration, so we hardcode everything here rather than do it
# from scratch
case "$image" in
  pytorch-linux-bionic-clang9-thrift-llvmdev)
    CLANG_VERSION=9
    THRIFT=yes
    LLVMDEV=yes
    PROTOBUF=yes
    ;;
  pytorch-linux-xenial-py2.7.9)
    TRAVIS_PYTHON_VERSION=2.7.9
    GCC_VERSION=7
    # Do not install PROTOBUF, DB, and VISION as a test
    ;;
  pytorch-linux-xenial-py2.7)
    TRAVIS_PYTHON_VERSION=2.7
    GCC_VERSION=7
    PROTOBUF=yes
    DB=yes
    VISION=yes
    ;;
  pytorch-linux-xenial-py3.5)
    TRAVIS_PYTHON_VERSION=3.5
    GCC_VERSION=7
    # Do not install PROTOBUF, DB, and VISION as a test
    ;;
  pytorch-linux-xenial-py3.8)
    # TODO: This is a hack, get rid of this as soon as you get rid of the travis downloads
    TRAVIS_DL_URL_PREFIX="https://s3.amazonaws.com/travis-python-archives/binaries/ubuntu/16.04/x86_64"
    TRAVIS_PYTHON_VERSION=3.8
    GCC_VERSION=7
    # Do not install PROTOBUF, DB, and VISION as a test
    ;;
  pytorch-linux-xenial-py3.6-gcc4.8)
    ANACONDA_PYTHON_VERSION=3.6
    GCC_VERSION=4.8
    PROTOBUF=yes
    DB=yes
    VISION=yes
    ;;
  pytorch-linux-xenial-py3.6-gcc5.4)
    ANACONDA_PYTHON_VERSION=3.6
    GCC_VERSION=5
    PROTOBUF=yes
    DB=yes
    VISION=yes
    KATEX=yes
    ;;
  pytorch-linux-xenial-py3.6-gcc7.2)
    ANACONDA_PYTHON_VERSION=3.6
    GCC_VERSION=7
    # Do not install PROTOBUF, DB, and VISION as a test
    ;;
  pytorch-linux-xenial-py3.6-gcc7)
    ANACONDA_PYTHON_VERSION=3.6
    GCC_VERSION=7
    PROTOBUF=yes
    DB=yes
    VISION=yes
    ;;
  pytorch-linux-xenial-pynightly)
    TRAVIS_PYTHON_VERSION=nightly
    GCC_VERSION=7
    PROTOBUF=yes
    DB=yes
    VISION=yes
    ;;
  pytorch-linux-xenial-cuda9-cudnn7-py2)
    CUDA_VERSION=9.0
    CUDNN_VERSION=7
    ANACONDA_PYTHON_VERSION=2.7
    PROTOBUF=yes
    DB=yes
    VISION=yes
    ;;
  pytorch-linux-xenial-cuda9-cudnn7-py3)
    CUDA_VERSION=9.0
    CUDNN_VERSION=7
    ANACONDA_PYTHON_VERSION=3.6
    PROTOBUF=yes
    DB=yes
    VISION=yes
    ;;
  pytorch-linux-xenial-cuda9.2-cudnn7-py3-gcc7)
    CUDA_VERSION=9.2
    CUDNN_VERSION=7
    ANACONDA_PYTHON_VERSION=3.6
    GCC_VERSION=7
    PROTOBUF=yes
    DB=yes
    VISION=yes
    ;;
  pytorch-linux-xenial-cuda10-cudnn7-py3-gcc7)
    CUDA_VERSION=10.0
    CUDNN_VERSION=7
    ANACONDA_PYTHON_VERSION=3.6
    GCC_VERSION=7
    PROTOBUF=yes
    DB=yes
    VISION=yes
    ;;
  pytorch-linux-xenial-cuda10.1-cudnn7-py3-gcc7)
    CUDA_VERSION=10.1
    CUDNN_VERSION=7
    ANACONDA_PYTHON_VERSION=3.6
    GCC_VERSION=7
    PROTOBUF=yes
    DB=yes
    VISION=yes
    KATEX=yes
    ;;
  pytorch-linux-xenial-cuda10.2-cudnn7-py3-gcc7)
    CUDA_VERSION=10.2
    CUDNN_VERSION=7
    ANACONDA_PYTHON_VERSION=3.6
    GCC_VERSION=7
    PROTOBUF=yes
    DB=yes
    VISION=yes
    KATEX=yes
    ;;
  pytorch-linux-xenial-py3-clang5-asan)
    ANACONDA_PYTHON_VERSION=3.6
    CLANG_VERSION=5.0
    PROTOBUF=yes
    DB=yes
    VISION=yes
    ;;
  pytorch-linux-xenial-py3-clang5-android-ndk-r19c)
    ANACONDA_PYTHON_VERSION=3.6
    CLANG_VERSION=5.0
    LLVMDEV=yes
    PROTOBUF=yes
    ANDROID=yes
    ANDROID_NDK_VERSION=r19c
    GRADLE_VERSION=4.10.3
    CMAKE_VERSION=3.7.0
    NINJA_VERSION=1.9.0
    ;;
  pytorch-linux-xenial-py3.6-clang7)
    ANACONDA_PYTHON_VERSION=3.6
    CLANG_VERSION=7
    PROTOBUF=yes
    DB=yes
    VISION=yes
    ;;
esac

# Set Jenkins UID and GID if running Jenkins
if [ -n "${JENKINS:-}" ]; then
  JENKINS_UID=$(id -u jenkins)
  JENKINS_GID=$(id -g jenkins)
fi

tmp_tag="tmp-$(cat /dev/urandom | tr -dc 'a-z' | fold -w 32 | head -n 1)"

# Build image
docker build \
       --no-cache \
       --build-arg "TRAVIS_DL_URL_PREFIX=${TRAVIS_DL_URL_PREFIX}" \
       --build-arg "BUILD_ENVIRONMENT=${image}" \
       --build-arg "PROTOBUF=${PROTOBUF:-}" \
       --build-arg "THRIFT=${THRIFT:-}" \
       --build-arg "LLVMDEV=${LLVMDEV:-}" \
       --build-arg "DB=${DB:-}" \
       --build-arg "VISION=${VISION:-}" \
       --build-arg "EC2=${EC2:-}" \
       --build-arg "JENKINS=${JENKINS:-}" \
       --build-arg "JENKINS_UID=${JENKINS_UID:-}" \
       --build-arg "JENKINS_GID=${JENKINS_GID:-}" \
       --build-arg "UBUNTU_VERSION=${UBUNTU_VERSION}" \
       --build-arg "CLANG_VERSION=${CLANG_VERSION}" \
       --build-arg "ANACONDA_PYTHON_VERSION=${ANACONDA_PYTHON_VERSION}" \
       --build-arg "TRAVIS_PYTHON_VERSION=${TRAVIS_PYTHON_VERSION}" \
       --build-arg "GCC_VERSION=${GCC_VERSION}" \
       --build-arg "CUDA_VERSION=${CUDA_VERSION}" \
       --build-arg "CUDNN_VERSION=${CUDNN_VERSION}" \
       --build-arg "ANDROID=${ANDROID}" \
       --build-arg "ANDROID_NDK=${ANDROID_NDK_VERSION}" \
       --build-arg "GRADLE_VERSION=${GRADLE_VERSION}" \
       --build-arg "CMAKE_VERSION=${CMAKE_VERSION:-}" \
       --build-arg "NINJA_VERSION=${NINJA_VERSION:-}" \
       --build-arg "KATEX=${KATEX:-}" \
       -f $(dirname ${DOCKERFILE})/Dockerfile \
       -t "$tmp_tag" \
       "$@" \
       .

function drun() {
  docker run --rm "$tmp_tag" $*
}

if [[ "$OS" == "ubuntu" ]]; then
  if !(drun lsb_release -a 2>&1 | grep -qF Ubuntu); then
    echo "OS=ubuntu, but:"
    drun lsb_release -a
    exit 1
  fi
  if !(drun lsb_release -a 2>&1 | grep -qF "$UBUNTU_VERSION"); then
    echo "UBUNTU_VERSION=$UBUNTU_VERSION, but:"
    drun lsb_release -a
    exit 1
  fi
fi

if [ -n "$TRAVIS_PYTHON_VERSION" ]; then
  if [[ "$TRAVIS_PYTHON_VERSION" != nightly ]]; then
    if !(drun python --version 2>&1 | grep -qF "Python $TRAVIS_PYTHON_VERSION"); then
      echo "TRAVIS_PYTHON_VERSION=$TRAVIS_PYTHON_VERSION, but:"
      drun python --version
      exit 1
    fi
  else
    echo "Please manually check nightly is OK:"
    drun python --version
  fi
fi

if [ -n "$ANACONDA_PYTHON_VERSION" ]; then
  if !(drun python --version 2>&1 | grep -qF "Python $ANACONDA_PYTHON_VERSION"); then
    echo "ANACONDA_PYTHON_VERSION=$ANACONDA_PYTHON_VERSION, but:"
    drun python --version
    exit 1
  fi
fi

if [ -n "$GCC_VERSION" ]; then
  if !(drun gcc --version 2>&1 | grep -q " $GCC_VERSION\\W"); then
    echo "GCC_VERSION=$GCC_VERSION, but:"
    drun gcc --version
    exit 1
  fi
fi

if [ -n "$CLANG_VERSION" ]; then
  if !(drun clang --version 2>&1 | grep -qF "clang version $CLANG_VERSION"); then
    echo "CLANG_VERSION=$CLANG_VERSION, but:"
    drun clang --version
    exit 1
  fi
fi

if [ -n "$KATEX" ]; then
  if !(drun katex --version); then
    echo "KATEX=$KATEX, but:"
    drun katex --version
    exit 1
  fi
fi
