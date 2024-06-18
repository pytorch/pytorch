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
    # skip non-conforming fields such as "pytorch", "linux" or "bionic" without version string
    if [ -n "${name}" ]; then
      extract_version_from_image_name "${name}" "${vername}"
    fi
  done
}

# Use the same pre-built XLA test image from PyTorch/XLA
if [[ "$image" == *xla* ]]; then
  echo "Using pre-built XLA test image..."
  exit 0
fi

if [[ "$image" == *-focal* ]]; then
  UBUNTU_VERSION=20.04
elif [[ "$image" == *-jammy* ]]; then
  UBUNTU_VERSION=22.04
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
# When using ubuntu - 22.04, start from Ubuntu docker image, instead of nvidia/cuda docker image.
if [[ "$image" == *cuda* && "$UBUNTU_VERSION" != "22.04" ]]; then
  DOCKERFILE="${OS}-cuda/Dockerfile"
elif [[ "$image" == *rocm* ]]; then
  DOCKERFILE="${OS}-rocm/Dockerfile"
elif [[ "$image" == *xpu* ]]; then
  DOCKERFILE="${OS}-xpu/Dockerfile"
elif [[ "$image" == *cuda*linter* ]]; then
  # Use a separate Dockerfile for linter to keep a small image size
  DOCKERFILE="linter-cuda/Dockerfile"
elif [[ "$image" == *linter* ]]; then
  # Use a separate Dockerfile for linter to keep a small image size
  DOCKERFILE="linter/Dockerfile"
fi

# CMake 3.18 is needed to support CUDA17 language variant
CMAKE_VERSION=3.18.5

_UCX_COMMIT=7bb2722ff2187a0cad557ae4a6afa090569f83fb
_UCC_COMMIT=20eae37090a4ce1b32bcce6144ccad0b49943e0b

# It's annoying to rename jobs every time you want to rewrite a
# configuration, so we hardcode everything here rather than do it
# from scratch
case "$image" in
  pytorch-linux-focal-cuda12.4-cudnn9-py3-gcc9)
    CUDA_VERSION=12.4.0
    CUDNN_VERSION=9
    ANACONDA_PYTHON_VERSION=3.10
    GCC_VERSION=9
    PROTOBUF=yes
    DB=yes
    VISION=yes
    KATEX=yes
    UCX_COMMIT=${_UCX_COMMIT}
    UCC_COMMIT=${_UCC_COMMIT}
    CONDA_CMAKE=yes
    TRITON=yes
    ;;
  pytorch-linux-focal-cuda12.1-cudnn9-py3-gcc9)
    CUDA_VERSION=12.1.1
    CUDNN_VERSION=9
    ANACONDA_PYTHON_VERSION=3.10
    GCC_VERSION=9
    PROTOBUF=yes
    DB=yes
    VISION=yes
    KATEX=yes
    UCX_COMMIT=${_UCX_COMMIT}
    UCC_COMMIT=${_UCC_COMMIT}
    CONDA_CMAKE=yes
    TRITON=yes
    ;;
  pytorch-linux-focal-cuda12.4-cudnn9-py3-gcc9-inductor-benchmarks)
    CUDA_VERSION=12.4.0
    CUDNN_VERSION=9
    ANACONDA_PYTHON_VERSION=3.10
    GCC_VERSION=9
    PROTOBUF=yes
    DB=yes
    VISION=yes
    KATEX=yes
    UCX_COMMIT=${_UCX_COMMIT}
    UCC_COMMIT=${_UCC_COMMIT}
    CONDA_CMAKE=yes
    TRITON=yes
    INDUCTOR_BENCHMARKS=yes
    ;;
  pytorch-linux-focal-cuda12.1-cudnn9-py3-gcc9-inductor-benchmarks)
    CUDA_VERSION=12.1.1
    CUDNN_VERSION=9
    ANACONDA_PYTHON_VERSION=3.10
    GCC_VERSION=9
    PROTOBUF=yes
    DB=yes
    VISION=yes
    KATEX=yes
    UCX_COMMIT=${_UCX_COMMIT}
    UCC_COMMIT=${_UCC_COMMIT}
    CONDA_CMAKE=yes
    TRITON=yes
    INDUCTOR_BENCHMARKS=yes
    ;;
  pytorch-linux-focal-cuda12.1-cudnn9-py3.12-gcc9-inductor-benchmarks)
    CUDA_VERSION=12.1.1
    CUDNN_VERSION=9
    ANACONDA_PYTHON_VERSION=3.12
    GCC_VERSION=9
    PROTOBUF=yes
    DB=yes
    VISION=yes
    KATEX=yes
    UCX_COMMIT=${_UCX_COMMIT}
    UCC_COMMIT=${_UCC_COMMIT}
    CONDA_CMAKE=yes
    TRITON=yes
    INDUCTOR_BENCHMARKS=yes
    ;;
  pytorch-linux-focal-cuda12.4-cudnn9-py3.12-gcc9-inductor-benchmarks)
    CUDA_VERSION=12.4.0
    CUDNN_VERSION=9
    ANACONDA_PYTHON_VERSION=3.12
    GCC_VERSION=9
    PROTOBUF=yes
    DB=yes
    VISION=yes
    KATEX=yes
    UCX_COMMIT=${_UCX_COMMIT}
    UCC_COMMIT=${_UCC_COMMIT}
    CONDA_CMAKE=yes
    TRITON=yes
    INDUCTOR_BENCHMARKS=yes
    ;;
  pytorch-linux-focal-cuda11.8-cudnn9-py3-gcc9)
    CUDA_VERSION=11.8.0
    CUDNN_VERSION=9
    ANACONDA_PYTHON_VERSION=3.10
    GCC_VERSION=9
    PROTOBUF=yes
    DB=yes
    VISION=yes
    KATEX=yes
    UCX_COMMIT=${_UCX_COMMIT}
    UCC_COMMIT=${_UCC_COMMIT}
    CONDA_CMAKE=yes
    TRITON=yes
    ;;
  pytorch-linux-focal-cuda12.4-cudnn9-py3-gcc9)
    CUDA_VERSION=12.4.0
    CUDNN_VERSION=9
    ANACONDA_PYTHON_VERSION=3.10
    GCC_VERSION=9
    PROTOBUF=yes
    DB=yes
    VISION=yes
    KATEX=yes
    UCX_COMMIT=${_UCX_COMMIT}
    UCC_COMMIT=${_UCC_COMMIT}
    CONDA_CMAKE=yes
    TRITON=yes
    ;;
  pytorch-linux-focal-cuda12.1-cudnn9-py3-gcc9)
    CUDA_VERSION=12.1.1
    CUDNN_VERSION=9
    ANACONDA_PYTHON_VERSION=3.10
    GCC_VERSION=9
    PROTOBUF=yes
    DB=yes
    VISION=yes
    KATEX=yes
    UCX_COMMIT=${_UCX_COMMIT}
    UCC_COMMIT=${_UCC_COMMIT}
    CONDA_CMAKE=yes
    TRITON=yes
    ;;
  pytorch-linux-focal-cuda12.4-cudnn9-py3-gcc9)
    CUDA_VERSION=12.4.0
    CUDNN_VERSION=9
    ANACONDA_PYTHON_VERSION=3.10
    GCC_VERSION=9
    PROTOBUF=yes
    DB=yes
    VISION=yes
    KATEX=yes
    UCX_COMMIT=${_UCX_COMMIT}
    UCC_COMMIT=${_UCC_COMMIT}
    CONDA_CMAKE=yes
    TRITON=yes
    ;;
  pytorch-linux-focal-py3-clang10-onnx)
    ANACONDA_PYTHON_VERSION=3.8
    CLANG_VERSION=10
    PROTOBUF=yes
    DB=yes
    VISION=yes
    CONDA_CMAKE=yes
    ONNX=yes
    ;;
  pytorch-linux-focal-py3-clang9-android-ndk-r21e)
    ANACONDA_PYTHON_VERSION=3.8
    CLANG_VERSION=9
    LLVMDEV=yes
    PROTOBUF=yes
    ANDROID=yes
    ANDROID_NDK_VERSION=r21e
    GRADLE_VERSION=6.8.3
    NINJA_VERSION=1.9.0
    ;;
  pytorch-linux-focal-py3.8-clang10)
    ANACONDA_PYTHON_VERSION=3.8
    CLANG_VERSION=10
    PROTOBUF=yes
    DB=yes
    VISION=yes
    VULKAN_SDK_VERSION=1.2.162.1
    SWIFTSHADER=yes
    CONDA_CMAKE=yes
    TRITON=yes
    ;;
  pytorch-linux-focal-py3.11-clang10)
    ANACONDA_PYTHON_VERSION=3.11
    CLANG_VERSION=10
    PROTOBUF=yes
    DB=yes
    VISION=yes
    VULKAN_SDK_VERSION=1.2.162.1
    SWIFTSHADER=yes
    CONDA_CMAKE=yes
    TRITON=yes
    ;;
  pytorch-linux-focal-py3.8-gcc9)
    ANACONDA_PYTHON_VERSION=3.8
    GCC_VERSION=9
    PROTOBUF=yes
    DB=yes
    VISION=yes
    CONDA_CMAKE=yes
    TRITON=yes
    ;;
  pytorch-linux-focal-rocm-n-1-py3)
    ANACONDA_PYTHON_VERSION=3.8
    GCC_VERSION=9
    PROTOBUF=yes
    DB=yes
    VISION=yes
    ROCM_VERSION=6.0
    NINJA_VERSION=1.9.0
    CONDA_CMAKE=yes
    TRITON=yes
    ;;
  pytorch-linux-focal-rocm-n-py3)
    ANACONDA_PYTHON_VERSION=3.8
    GCC_VERSION=9
    PROTOBUF=yes
    DB=yes
    VISION=yes
    ROCM_VERSION=6.1
    NINJA_VERSION=1.9.0
    CONDA_CMAKE=yes
    TRITON=yes
    ;;
  pytorch-linux-jammy-xpu-2024.0-py3)
    ANACONDA_PYTHON_VERSION=3.8
    GCC_VERSION=11
    PROTOBUF=yes
    DB=yes
    VISION=yes
    XPU_VERSION=0.5
    NINJA_VERSION=1.9.0
    CONDA_CMAKE=yes
    TRITON=yes
    ;;
    pytorch-linux-jammy-py3.8-gcc11-inductor-benchmarks)
    ANACONDA_PYTHON_VERSION=3.8
    GCC_VERSION=11
    PROTOBUF=yes
    DB=yes
    VISION=yes
    KATEX=yes
    CONDA_CMAKE=yes
    TRITON=yes
    DOCS=yes
    INDUCTOR_BENCHMARKS=yes
    ;;
  pytorch-linux-jammy-cuda11.8-cudnn9-py3.8-clang12)
    ANACONDA_PYTHON_VERSION=3.8
    CUDA_VERSION=11.8
    CUDNN_VERSION=9
    CLANG_VERSION=12
    PROTOBUF=yes
    DB=yes
    VISION=yes
    TRITON=yes
    ;;
  pytorch-linux-jammy-py3-clang12-asan)
    ANACONDA_PYTHON_VERSION=3.9
    CLANG_VERSION=12
    PROTOBUF=yes
    DB=yes
    VISION=yes
    CONDA_CMAKE=yes
    TRITON=yes
    ;;
  pytorch-linux-jammy-py3-clang15-asan)
    ANACONDA_PYTHON_VERSION=3.10
    CLANG_VERSION=15
    CONDA_CMAKE=yes
    VISION=yes
    ;;
  pytorch-linux-jammy-py3.8-gcc11)
    ANACONDA_PYTHON_VERSION=3.8
    GCC_VERSION=11
    PROTOBUF=yes
    DB=yes
    VISION=yes
    KATEX=yes
    CONDA_CMAKE=yes
    TRITON=yes
    DOCS=yes
    UNINSTALL_DILL=yes
    ;;
  pytorch-linux-jammy-py3-clang12-executorch)
    ANACONDA_PYTHON_VERSION=3.10
    CLANG_VERSION=12
    CONDA_CMAKE=yes
    EXECUTORCH=yes
    ;;
  pytorch-linux-focal-linter)
    # TODO: Use 3.9 here because of this issue https://github.com/python/mypy/issues/13627.
    # We will need to update mypy version eventually, but that's for another day. The task
    # would be to upgrade mypy to 1.0.0 with Python 3.11
    ANACONDA_PYTHON_VERSION=3.9
    CONDA_CMAKE=yes
    ;;
  pytorch-linux-jammy-cuda11.8-cudnn9-py3.9-linter)
    ANACONDA_PYTHON_VERSION=3.9
    CUDA_VERSION=11.8
    CONDA_CMAKE=yes
    ;;
  pytorch-linux-jammy-aarch64-py3.10-gcc11)
    ANACONDA_PYTHON_VERSION=3.10
    GCC_VERSION=11
    ACL=yes
    PROTOBUF=yes
    DB=yes
    VISION=yes
    CONDA_CMAKE=yes
    # snadampal: skipping sccache due to the following issue
    # https://github.com/pytorch/pytorch/issues/121559
    SKIP_SCCACHE_INSTALL=yes
    # snadampal: skipping llvm src build install because the current version
    # from pytorch/llvm:9.0.1 is x86 specific
    SKIP_LLVM_SRC_BUILD_INSTALL=yes
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
      NINJA_VERSION=1.9.0
      TRITON=yes
      # To ensure that any ROCm config will build using conda cmake
      # and thus have LAPACK/MKL enabled
      CONDA_CMAKE=yes
    fi
    if [[ "$image" == *centos7* ]]; then
      NINJA_VERSION=1.10.2
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

tmp_tag=$(basename "$(mktemp -u)" | tr '[:upper:]' '[:lower:]')

#when using cudnn version 8 install it separately from cuda
if [[ "$image" == *cuda*  && ${OS} == "ubuntu" ]]; then
  IMAGE_NAME="nvidia/cuda:${CUDA_VERSION}-cudnn${CUDNN_VERSION}-devel-ubuntu${UBUNTU_VERSION}"
  if [[ ${CUDNN_VERSION} == 9 ]]; then
    IMAGE_NAME="nvidia/cuda:${CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION}"
  fi
fi

# Build image
docker build \
       --no-cache \
       --progress=plain \
       --build-arg "BUILD_ENVIRONMENT=${image}" \
       --build-arg "PROTOBUF=${PROTOBUF:-}" \
       --build-arg "LLVMDEV=${LLVMDEV:-}" \
       --build-arg "DB=${DB:-}" \
       --build-arg "VISION=${VISION:-}" \
       --build-arg "UBUNTU_VERSION=${UBUNTU_VERSION}" \
       --build-arg "CENTOS_VERSION=${CENTOS_VERSION}" \
       --build-arg "DEVTOOLSET_VERSION=${DEVTOOLSET_VERSION}" \
       --build-arg "GLIBC_VERSION=${GLIBC_VERSION}" \
       --build-arg "CLANG_VERSION=${CLANG_VERSION}" \
       --build-arg "ANACONDA_PYTHON_VERSION=${ANACONDA_PYTHON_VERSION}" \
       --build-arg "GCC_VERSION=${GCC_VERSION}" \
       --build-arg "CUDA_VERSION=${CUDA_VERSION}" \
       --build-arg "CUDNN_VERSION=${CUDNN_VERSION}" \
       --build-arg "TENSORRT_VERSION=${TENSORRT_VERSION}" \
       --build-arg "ANDROID=${ANDROID}" \
       --build-arg "ANDROID_NDK=${ANDROID_NDK_VERSION}" \
       --build-arg "GRADLE_VERSION=${GRADLE_VERSION}" \
       --build-arg "VULKAN_SDK_VERSION=${VULKAN_SDK_VERSION}" \
       --build-arg "SWIFTSHADER=${SWIFTSHADER}" \
       --build-arg "CMAKE_VERSION=${CMAKE_VERSION:-}" \
       --build-arg "NINJA_VERSION=${NINJA_VERSION:-}" \
       --build-arg "KATEX=${KATEX:-}" \
       --build-arg "ROCM_VERSION=${ROCM_VERSION:-}" \
       --build-arg "PYTORCH_ROCM_ARCH=${PYTORCH_ROCM_ARCH:-gfx906;gfx90a}" \
       --build-arg "IMAGE_NAME=${IMAGE_NAME}" \
       --build-arg "UCX_COMMIT=${UCX_COMMIT}" \
       --build-arg "UCC_COMMIT=${UCC_COMMIT}" \
       --build-arg "CONDA_CMAKE=${CONDA_CMAKE}" \
       --build-arg "TRITON=${TRITON}" \
       --build-arg "ONNX=${ONNX}" \
       --build-arg "DOCS=${DOCS}" \
       --build-arg "INDUCTOR_BENCHMARKS=${INDUCTOR_BENCHMARKS}" \
       --build-arg "EXECUTORCH=${EXECUTORCH}" \
       --build-arg "XPU_VERSION=${XPU_VERSION}" \
       --build-arg "ACL=${ACL:-}" \
       --build-arg "SKIP_SCCACHE_INSTALL=${SKIP_SCCACHE_INSTALL:-}" \
       --build-arg "SKIP_LLVM_SRC_BUILD_INSTALL=${SKIP_LLVM_SRC_BUILD_INSTALL:-}" \
       -f $(dirname ${DOCKERFILE})/Dockerfile \
       -t "$tmp_tag" \
       "$@" \
       .

# NVIDIA dockers for RC releases use tag names like `11.0-cudnn9-devel-ubuntu18.04-rc`,
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
