#!/bin/bash

set -ex

image="$1"
shift

if [ -z "${image}" ]; then
  echo "Usage: $0 IMAGE"
  exit 1
fi

UBUNTU_VERSION="$(echo "${image}" | perl -n -e'/ubuntu(\d+\.\d+)/ && print $1')"
CENTOS_VERSION="$(echo "${image}" | perl -n -e'/centos(\d+)/ && print $1')"

if [ -n "${UBUNTU_VERSION}" ]; then
  OS="ubuntu"
  DOCKERFILE="ubuntu/Dockerfile"
elif [ -n "${CENTOS_VERSION}" ]; then
  OS="centos"
  DOCKERFILE="centos/Dockerfile"
else
  echo "Unable to derive operating system base..."
  exit 1
fi

if [[ "$image" == py* ]]; then
  PYTHON_VERSION="$(echo "${image}" | perl -n -e'/py(\d+(\.\d+)?)/ && print $1')"
fi

if [[ "$image" == *cuda* ]]; then
  CUDA_VERSION="$(echo "${image}" | perl -n -e'/cuda(\d+\.\d+)/ && print $1')"
  CUDNN_VERSION="$(echo "${image}" | perl -n -e'/cudnn(\d+)/ && print $1')"
  DOCKERFILE="${OS}-cuda/Dockerfile"
fi

# TODO: the version number here actually doesn't do anything at the
# moment
if [[ "$image" == *rocm* ]]; then
  ROCM_VERSION="$(echo "${image}" | perl -n -e'/rocm(\d+\.\d+\.\d+|nightly)/ && print $1')"
  DOCKERFILE="${OS}-rocm/Dockerfile"
  # newer cmake version needed
  CMAKE_VERSION=3.6.3
fi

if [[ "$image" == *conda* ]]; then
  # Unlike python version, Anaconda version is either 2 or 3
  ANACONDA_VERSION="$(echo "${image}" | perl -n -e'/conda(\d)/ && print $1')"
fi

if [[ "$image" == *-mkl-* ]]; then
  MKL=yes
fi

if [[ "$image" == *-android-* ]]; then
  ANDROID=yes

  # The Android NDK requires CMake 3.6 or higher.
  # See https://github.com/caffe2/caffe2/pull/1740 for more info.
  CMAKE_VERSION=3.6.3
fi

if [[ "$image" == *-gcc* ]]; then
  GCC_VERSION="$(echo "${image}" | perl -n -e'/gcc(\d+(\.\d+)?)/ && print $1')"
fi

if [[ "$image" == *-clang* ]]; then
  CLANG_VERSION="$(echo "${image}" | perl -n -e'/clang(\d+(\.\d+)?)/ && print $1')"
fi


if [[ "$image" == *-devtoolset* ]]; then
  DEVTOOLSET_VERSION="$(echo "${image}" | perl -n -e'/devtoolset(\d+(\.\d+)?)/ && print $1')"
fi

# Copy over common scripts to directory containing the Dockerfile to build
cp -a common/* "$(dirname ${DOCKERFILE})"

# Set Jenkins UID and GID if running Jenkins
if [ -n "${JENKINS:-}" ]; then
  JENKINS_UID=$(id -u jenkins)
  JENKINS_GID=$(id -g jenkins)
fi

# Build image
docker build \
       --build-arg "BUILD_ENVIRONMENT=${image}" \
       --build-arg "EC2=${EC2:-}" \
       --build-arg "JENKINS=${JENKINS:-}" \
       --build-arg "JENKINS_UID=${JENKINS_UID:-}" \
       --build-arg "JENKINS_GID=${JENKINS_GID:-}" \
       --build-arg "UBUNTU_VERSION=${UBUNTU_VERSION}" \
       --build-arg "CENTOS_VERSION=${CENTOS_VERSION}" \
       --build-arg "DEVTOOLSET_VERSION=${DEVTOOLSET_VERSION}" \
       --build-arg "PYTHON_VERSION=${PYTHON_VERSION}" \
       --build-arg "ANACONDA_VERSION=${ANACONDA_VERSION}" \
       --build-arg "CUDA_VERSION=${CUDA_VERSION}" \
       --build-arg "CUDNN_VERSION=${CUDNN_VERSION}" \
       --build-arg "MKL=${MKL}" \
       --build-arg "ANDROID=${ANDROID}" \
       --build-arg "GCC_VERSION=${GCC_VERSION}" \
       --build-arg "CLANG_VERSION=${CLANG_VERSION}" \
       --build-arg "CMAKE_VERSION=${CMAKE_VERSION:-}" \
       --build-arg "ROCM_VERSION=${ROCM_VERSION}" \
       "$@" \
       "$(dirname ${DOCKERFILE})"
