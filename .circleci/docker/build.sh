#!/bin/bash

set -ex

image="$1"
shift

if [ -z "${image}" ]; then
  echo "Usage: $0 IMAGE"
  exit 1
fi

function extract_version_from_image_name() {
  eval export $2=$(echo "${image}" | perl -n -e"/$1(\d+(\.\d+)?(\.\d+)?)/ && print \$1")
  if [ "x${!2}" = x ]; then
    echo "variable '$2' not correctly parsed from image='$image'"
    exit 1
  fi
}

function extract_all_from_image_name() {
  # parts $image into array, splitting on '-'
  keep_IFS="$IFS"
  IFS="-"
  declare -a parts=($image)
  IFS="$keep_IFS"
  unset keep_IFS

  for part in "${parts[@]}"; do
    name=$(echo "${part}" | perl -n -e"/([a-zA-Z]+)\d+(\.\d+)?(\.\d+)?/ && print \$1")
    vername="${name^^}_VERSION"
    # "py" is the odd one out, needs this special case
    if [ "x${name}" = xpy ]; then
      vername=ANACONDA_PYTHON_VERSION
    fi
    # skip non-conforming fields such as "pytorch", "linux" or "xenial" without version string
    if [ -n "${name}" ]; then
      extract_version_from_image_name "${name}" "${vername}"
    fi
  done
}

if [[ "$image" == *-xenial* ]]; then
  UBUNTU_VERSION=16.04
elif [[ "$image" == *-artful* ]]; then
  UBUNTU_VERSION=17.10
elif [[ "$image" == *-bionic* ]]; then
  UBUNTU_VERSION=18.04
elif [[ "$image" == *-focal* ]]; then
  UBUNTU_VERSION=20.04
elif [[ "$image" == *ubuntu* ]]; then
  extract_version_from_image_name ubuntu UBUNTU_VERSION
elif [[ "$image" == *centos* ]]; then
  extract_version_from_image_name centos CENTOS_VERSION
fi

if [ -n "${UBUNTU_VERSION}" ]; then
  OS="ubuntu"
elif [ -n "${CENTOS_VERSION}" ]; then
  OS="centos"
else
  echo "Unable to derive operating system base..."
  exit 1
fi

DOCKERFILE="${OS}/Dockerfile"
if [[ "$image" == *cuda* ]]; then
  DOCKERFILE="${OS}-cuda/Dockerfile"
elif [[ "$image" == *rocm* ]]; then
  DOCKERFILE="${OS}-rocm/Dockerfile"
fi

TRAVIS_DL_URL_PREFIX="https://s3.amazonaws.com/travis-python-archives/binaries/ubuntu/14.04/x86_64"

# It's annoying to rename jobs every time you want to rewrite a
# configuration, so we hardcode everything here rather than do it
# from scratch
case "$image" in
  pytorch-linux-xenial-py3.8)
    ANACONDA_PYTHON_VERSION=3.8
    GCC_VERSION=7
    # Do not install PROTOBUF, DB, and VISION as a test
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
  pytorch-linux-xenial-cuda9.2-cudnn7-py3-gcc5.4)
    CUDA_VERSION=9.2
    CUDNN_VERSION=7
    ANACONDA_PYTHON_VERSION=3.6
    GCC_VERSION=5
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
  pytorch-linux-xenial-cuda11.1-cudnn8-py3-gcc7)
    CUDA_VERSION=11.1
    CUDNN_VERSION=8
    ANACONDA_PYTHON_VERSION=3.6
    GCC_VERSION=7
    PROTOBUF=yes
    DB=yes
    VISION=yes
    KATEX=yes
    ;;
  pytorch-linux-xenial-cuda11.2-cudnn8-py3-gcc7)
    CUDA_VERSION=11.2.0 # Deviating from major.minor to conform to nvidia's Docker image names
    CUDNN_VERSION=8
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
  pytorch-linux-xenial-py3-clang7-onnx)
    ANACONDA_PYTHON_VERSION=3.6
    CLANG_VERSION=7
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
    GRADLE_VERSION=6.8.3
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
  pytorch-linux-bionic-py3.6-clang9)
    ANACONDA_PYTHON_VERSION=3.6
    CLANG_VERSION=9
    PROTOBUF=yes
    DB=yes
    VISION=yes
    VULKAN_SDK_VERSION=1.2.162.1
    SWIFTSHADER=yes
    ;;
  pytorch-linux-bionic-py3.8-gcc9)
    ANACONDA_PYTHON_VERSION=3.8
    GCC_VERSION=9
    PROTOBUF=yes
    DB=yes
    VISION=yes
    ;;
  pytorch-linux-bionic-cuda10.2-cudnn7-py3.6-clang9)
    CUDA_VERSION=10.2
    CUDNN_VERSION=7
    ANACONDA_PYTHON_VERSION=3.6
    CLANG_VERSION=9
    PROTOBUF=yes
    DB=yes
    VISION=yes
    ;;
  pytorch-linux-bionic-cuda10.2-cudnn7-py3.8-gcc9)
    CUDA_VERSION=10.2
    CUDNN_VERSION=7
    ANACONDA_PYTHON_VERSION=3.8
    GCC_VERSION=9
    PROTOBUF=yes
    DB=yes
    VISION=yes
    ;;
  pytorch-linux-bionic-rocm3.9-py3.6)
    ANACONDA_PYTHON_VERSION=3.6
    GCC_VERSION=9
    PROTOBUF=yes
    DB=yes
    VISION=yes
    ROCM_VERSION=3.9
    ;;
  pytorch-linux-bionic-rocm3.10-py3.6)
    ANACONDA_PYTHON_VERSION=3.6
    GCC_VERSION=9
    PROTOBUF=yes
    DB=yes
    VISION=yes
    ROCM_VERSION=3.10
    ;;
  pytorch-linux-bionic-rocm4.0.1-py3.6)
    ANACONDA_PYTHON_VERSION=3.6
    GCC_VERSION=9
    PROTOBUF=yes
    DB=yes
    VISION=yes
    ROCM_VERSION=4.0.1
    ;;
  *)
    # Catch-all for builds that are not hardcoded.
    PROTOBUF=yes
    DB=yes
    VISION=yes
    echo "image '$image' did not match an existing build configuration"
    if [[ "$image" == *py* ]]; then
      extract_version_from_image_name py ANACONDA_PYTHON_VERSION
    fi
    if [[ "$image" == *cuda* ]]; then
      extract_version_from_image_name cuda CUDA_VERSION
      extract_version_from_image_name cudnn CUDNN_VERSION
    fi
    if [[ "$image" == *rocm* ]]; then
      extract_version_from_image_name rocm ROCM_VERSION
    fi
    if [[ "$image" == *gcc* ]]; then
      extract_version_from_image_name gcc GCC_VERSION
    fi
    if [[ "$image" == *clang* ]]; then
      extract_version_from_image_name clang CLANG_VERSION
    fi
    if [[ "$image" == *devtoolset* ]]; then
      extract_version_from_image_name devtoolset DEVTOOLSET_VERSION
    fi
    if [[ "$image" == *glibc* ]]; then
      extract_version_from_image_name glibc GLIBC_VERSION
    fi
    if [[ "$image" == *cmake* ]]; then
      extract_version_from_image_name cmake CMAKE_VERSION
    fi
  ;;
esac

# Set Jenkins UID and GID if running Jenkins
if [ -n "${JENKINS:-}" ]; then
  JENKINS_UID=$(id -u jenkins)
  JENKINS_GID=$(id -g jenkins)
fi

tmp_tag="tmp-$(cat /dev/urandom | tr -dc 'a-z' | fold -w 32 | head -n 1)"

# Build image
# TODO: build-arg THRIFT is not turned on for any image, remove it once we confirm
# it's no longer needed.
docker build \
       --no-cache \
       --progress=plain \
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
       --build-arg "CENTOS_VERSION=${CENTOS_VERSION}" \
       --build-arg "DEVTOOLSET_VERSION=${DEVTOOLSET_VERSION}" \
       --build-arg "GLIBC_VERSION=${GLIBC_VERSION}" \
       --build-arg "CLANG_VERSION=${CLANG_VERSION}" \
       --build-arg "ANACONDA_PYTHON_VERSION=${ANACONDA_PYTHON_VERSION}" \
       --build-arg "GCC_VERSION=${GCC_VERSION}" \
       --build-arg "CUDA_VERSION=${CUDA_VERSION}" \
       --build-arg "CUDNN_VERSION=${CUDNN_VERSION}" \
       --build-arg "ANDROID=${ANDROID}" \
       --build-arg "ANDROID_NDK=${ANDROID_NDK_VERSION}" \
       --build-arg "GRADLE_VERSION=${GRADLE_VERSION}" \
       --build-arg "VULKAN_SDK_VERSION=${VULKAN_SDK_VERSION}" \
       --build-arg "SWIFTSHADER=${SWIFTSHADER}" \
       --build-arg "CMAKE_VERSION=${CMAKE_VERSION:-}" \
       --build-arg "NINJA_VERSION=${NINJA_VERSION:-}" \
       --build-arg "KATEX=${KATEX:-}" \
       --build-arg "ROCM_VERSION=${ROCM_VERSION:-}" \
       -f $(dirname ${DOCKERFILE})/Dockerfile \
       -t "$tmp_tag" \
       "$@" \
       .

# NVIDIA dockers for RC releases use tag names like `11.0-cudnn8-devel-ubuntu18.04-rc`,
# for this case we will set UBUNTU_VERSION to `18.04-rc` so that the Dockerfile could
# find the correct image. As a result, here we have to replace the
#   "$UBUNTU_VERSION" == "18.04-rc"
# with
#   "$UBUNTU_VERSION" == "18.04"
UBUNTU_VERSION=$(echo ${UBUNTU_VERSION} | sed 's/-rc$//')

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
