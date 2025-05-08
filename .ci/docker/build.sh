#!/bin/bash
# The purpose of this script is to:
# 1. Extract the set of parameters to be used for a docker build based on the provided image name.
# 2. Run docker build with the parameters found in step 1.
# 3. Run the built image and print out the expected and actual versions of packages installed.

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
if [[ "$image" == *rocm* ]]; then
  _UCX_COMMIT=cc312eaa4655c0cc5c2bcd796db938f90563bcf6
  _UCC_COMMIT=0c0fc21559835044ab107199e334f7157d6a0d3d
fi

tag=$(echo $image | awk -F':' '{print $2}')

# It's annoying to rename jobs every time you want to rewrite a
# configuration, so we hardcode everything here rather than do it
# from scratch
case "$tag" in
  pytorch-linux-focal-cuda12.6-cudnn9-py3-gcc11)
    CUDA_VERSION=12.6.3
    CUDNN_VERSION=9
    ANACONDA_PYTHON_VERSION=3.10
    GCC_VERSION=11
    VISION=yes
    KATEX=yes
    UCX_COMMIT=${_UCX_COMMIT}
    UCC_COMMIT=${_UCC_COMMIT}
    CONDA_CMAKE=yes
    TRITON=yes
    ;;
  pytorch-linux-focal-cuda12.4-cudnn9-py3-gcc9-inductor-benchmarks)
    CUDA_VERSION=12.4.1
    CUDNN_VERSION=9
    ANACONDA_PYTHON_VERSION=3.10
    GCC_VERSION=9
    VISION=yes
    KATEX=yes
    UCX_COMMIT=${_UCX_COMMIT}
    UCC_COMMIT=${_UCC_COMMIT}
    CONDA_CMAKE=yes
    TRITON=yes
    INDUCTOR_BENCHMARKS=yes
    ;;
  pytorch-linux-focal-cuda12.4-cudnn9-py3.12-gcc9-inductor-benchmarks)
    CUDA_VERSION=12.4.1
    CUDNN_VERSION=9
    ANACONDA_PYTHON_VERSION=3.12
    GCC_VERSION=9
    VISION=yes
    KATEX=yes
    UCX_COMMIT=${_UCX_COMMIT}
    UCC_COMMIT=${_UCC_COMMIT}
    CONDA_CMAKE=yes
    TRITON=yes
    INDUCTOR_BENCHMARKS=yes
    ;;
  pytorch-linux-focal-cuda12.4-cudnn9-py3.13-gcc9-inductor-benchmarks)
    CUDA_VERSION=12.4.1
    CUDNN_VERSION=9
    ANACONDA_PYTHON_VERSION=3.13
    GCC_VERSION=9
    VISION=yes
    KATEX=yes
    UCX_COMMIT=${_UCX_COMMIT}
    UCC_COMMIT=${_UCC_COMMIT}
    CONDA_CMAKE=yes
    TRITON=yes
    INDUCTOR_BENCHMARKS=yes
    ;;
  pytorch-linux-focal-cuda12.6-cudnn9-py3-gcc9)
    CUDA_VERSION=12.6.3
    CUDNN_VERSION=9
    ANACONDA_PYTHON_VERSION=3.10
    GCC_VERSION=9
    VISION=yes
    KATEX=yes
    UCX_COMMIT=${_UCX_COMMIT}
    UCC_COMMIT=${_UCC_COMMIT}
    CONDA_CMAKE=yes
    TRITON=yes
    ;;
  pytorch-linux-focal-cuda12.6-cudnn9-py3-gcc9-inductor-benchmarks)
    CUDA_VERSION=12.6.3
    CUDNN_VERSION=9
    ANACONDA_PYTHON_VERSION=3.10
    GCC_VERSION=9
    VISION=yes
    KATEX=yes
    UCX_COMMIT=${_UCX_COMMIT}
    UCC_COMMIT=${_UCC_COMMIT}
    CONDA_CMAKE=yes
    TRITON=yes
    INDUCTOR_BENCHMARKS=yes
    ;;
  pytorch-linux-focal-cuda12.6-cudnn9-py3.12-gcc9-inductor-benchmarks)
    CUDA_VERSION=12.6.3
    CUDNN_VERSION=9
    ANACONDA_PYTHON_VERSION=3.12
    GCC_VERSION=9
    VISION=yes
    KATEX=yes
    UCX_COMMIT=${_UCX_COMMIT}
    UCC_COMMIT=${_UCC_COMMIT}
    CONDA_CMAKE=yes
    TRITON=yes
    INDUCTOR_BENCHMARKS=yes
    ;;
  pytorch-linux-focal-cuda12.6-cudnn9-py3.13-gcc9-inductor-benchmarks)
    CUDA_VERSION=12.6.3
    CUDNN_VERSION=9
    ANACONDA_PYTHON_VERSION=3.13
    GCC_VERSION=9
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
    VISION=yes
    KATEX=yes
    UCX_COMMIT=${_UCX_COMMIT}
    UCC_COMMIT=${_UCC_COMMIT}
    CONDA_CMAKE=yes
    TRITON=yes
    ;;
  pytorch-linux-focal-py3-clang10-onnx)
    ANACONDA_PYTHON_VERSION=3.9
    CLANG_VERSION=10
    VISION=yes
    CONDA_CMAKE=yes
    ONNX=yes
    ;;
  pytorch-linux-focal-py3.9-clang10)
    ANACONDA_PYTHON_VERSION=3.9
    CLANG_VERSION=10
    VISION=yes
    CONDA_CMAKE=yes
    TRITON=yes
    ;;
  pytorch-linux-focal-py3.11-clang10)
    ANACONDA_PYTHON_VERSION=3.11
    CLANG_VERSION=10
    VISION=yes
    CONDA_CMAKE=yes
    TRITON=yes
    ;;
  pytorch-linux-focal-py3.9-gcc9)
    ANACONDA_PYTHON_VERSION=3.9
    GCC_VERSION=9
    VISION=yes
    CONDA_CMAKE=yes
    TRITON=yes
    ;;
  pytorch-linux-focal-rocm-n-1-py3)
    ANACONDA_PYTHON_VERSION=3.10
    GCC_VERSION=11
    VISION=yes
    ROCM_VERSION=6.2.4
    NINJA_VERSION=1.9.0
    CONDA_CMAKE=yes
    TRITON=yes
    KATEX=yes
    UCX_COMMIT=${_UCX_COMMIT}
    UCC_COMMIT=${_UCC_COMMIT}
    INDUCTOR_BENCHMARKS=yes
    ;;
  pytorch-linux-focal-rocm-n-py3)
    ANACONDA_PYTHON_VERSION=3.10
    GCC_VERSION=11
    VISION=yes
    ROCM_VERSION=6.3
    NINJA_VERSION=1.9.0
    CONDA_CMAKE=yes
    TRITON=yes
    KATEX=yes
    UCX_COMMIT=${_UCX_COMMIT}
    UCC_COMMIT=${_UCC_COMMIT}
    INDUCTOR_BENCHMARKS=yes
    ;;
  pytorch-linux-jammy-xpu-2024.0-py3)
    ANACONDA_PYTHON_VERSION=3.9
    GCC_VERSION=11
    VISION=yes
    XPU_VERSION=0.5
    NINJA_VERSION=1.9.0
    CONDA_CMAKE=yes
    TRITON=yes
    ;;
  pytorch-linux-jammy-xpu-2025.0-py3)
    ANACONDA_PYTHON_VERSION=3.9
    GCC_VERSION=11
    VISION=yes
    XPU_VERSION=2025.0
    NINJA_VERSION=1.9.0
    CONDA_CMAKE=yes
    TRITON=yes
    ;;
    pytorch-linux-jammy-py3.9-gcc11-inductor-benchmarks)
    ANACONDA_PYTHON_VERSION=3.9
    GCC_VERSION=11
    VISION=yes
    KATEX=yes
    CONDA_CMAKE=yes
    TRITON=yes
    DOCS=yes
    INDUCTOR_BENCHMARKS=yes
    ;;
  pytorch-linux-jammy-cuda11.8-cudnn9-py3.9-clang12)
    ANACONDA_PYTHON_VERSION=3.9
    CUDA_VERSION=11.8
    CUDNN_VERSION=9
    CLANG_VERSION=12
    VISION=yes
    TRITON=yes
    ;;
  pytorch-linux-jammy-py3-clang12-asan)
    ANACONDA_PYTHON_VERSION=3.9
    CLANG_VERSION=12
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
  pytorch-linux-jammy-py3-clang18-asan)
    ANACONDA_PYTHON_VERSION=3.10
    CLANG_VERSION=18
    CONDA_CMAKE=yes
    VISION=yes
    ;;
  pytorch-linux-jammy-py3.9-gcc11)
    ANACONDA_PYTHON_VERSION=3.9
    GCC_VERSION=11
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
  pytorch-linux-jammy-py3.12-halide)
    CUDA_VERSION=12.6
    ANACONDA_PYTHON_VERSION=3.12
    GCC_VERSION=11
    CONDA_CMAKE=yes
    HALIDE=yes
    TRITON=yes
    ;;
  pytorch-linux-jammy-py3.12-triton-cpu)
    CUDA_VERSION=12.6
    ANACONDA_PYTHON_VERSION=3.12
    GCC_VERSION=11
    CONDA_CMAKE=yes
    TRITON_CPU=yes
    ;;
  pytorch-linux-focal-linter)
    # TODO: Use 3.9 here because of this issue https://github.com/python/mypy/issues/13627.
    # We will need to update mypy version eventually, but that's for another day. The task
    # would be to upgrade mypy to 1.0.0 with Python 3.11
    PYTHON_VERSION=3.9
    PIP_CMAKE=yes
    ;;
  pytorch-linux-jammy-cuda11.8-cudnn9-py3.9-linter)
    PYTHON_VERSION=3.9
    CUDA_VERSION=11.8
    PIP_CMAKE=yes
    ;;
  pytorch-linux-jammy-aarch64-py3.10-gcc11)
    ANACONDA_PYTHON_VERSION=3.10
    GCC_VERSION=11
    ACL=yes
    VISION=yes
    CONDA_CMAKE=yes
    # snadampal: skipping llvm src build install because the current version
    # from pytorch/llvm:9.0.1 is x86 specific
    SKIP_LLVM_SRC_BUILD_INSTALL=yes
    ;;
  pytorch-linux-jammy-aarch64-py3.10-gcc11-inductor-benchmarks)
    ANACONDA_PYTHON_VERSION=3.10
    GCC_VERSION=11
    ACL=yes
    VISION=yes
    CONDA_CMAKE=yes
    # snadampal: skipping llvm src build install because the current version
    # from pytorch/llvm:9.0.1 is x86 specific
    SKIP_LLVM_SRC_BUILD_INSTALL=yes
    INDUCTOR_BENCHMARKS=yes
    ;;
  *)
    # Catch-all for builds that are not hardcoded.
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

no_cache_flag=""
progress_flag=""
# Do not use cache and progress=plain when in CI
if [[ -n "${CI:-}" ]]; then
  no_cache_flag="--no-cache"
  progress_flag="--progress=plain"
fi

# Build image
docker build \
       ${no_cache_flag} \
       ${progress_flag} \
       --build-arg "BUILD_ENVIRONMENT=${image}" \
       --build-arg "LLVMDEV=${LLVMDEV:-}" \
       --build-arg "VISION=${VISION:-}" \
       --build-arg "UBUNTU_VERSION=${UBUNTU_VERSION}" \
       --build-arg "CENTOS_VERSION=${CENTOS_VERSION}" \
       --build-arg "DEVTOOLSET_VERSION=${DEVTOOLSET_VERSION}" \
       --build-arg "GLIBC_VERSION=${GLIBC_VERSION}" \
       --build-arg "CLANG_VERSION=${CLANG_VERSION}" \
       --build-arg "ANACONDA_PYTHON_VERSION=${ANACONDA_PYTHON_VERSION}" \
       --build-arg "PYTHON_VERSION=${PYTHON_VERSION}" \
       --build-arg "GCC_VERSION=${GCC_VERSION}" \
       --build-arg "CUDA_VERSION=${CUDA_VERSION}" \
       --build-arg "CUDNN_VERSION=${CUDNN_VERSION}" \
       --build-arg "TENSORRT_VERSION=${TENSORRT_VERSION}" \
       --build-arg "GRADLE_VERSION=${GRADLE_VERSION}" \
       --build-arg "CMAKE_VERSION=${CMAKE_VERSION:-}" \
       --build-arg "NINJA_VERSION=${NINJA_VERSION:-}" \
       --build-arg "KATEX=${KATEX:-}" \
       --build-arg "ROCM_VERSION=${ROCM_VERSION:-}" \
       --build-arg "PYTORCH_ROCM_ARCH=${PYTORCH_ROCM_ARCH:-gfx90a;gfx942}" \
       --build-arg "IMAGE_NAME=${IMAGE_NAME}" \
       --build-arg "UCX_COMMIT=${UCX_COMMIT}" \
       --build-arg "UCC_COMMIT=${UCC_COMMIT}" \
       --build-arg "CONDA_CMAKE=${CONDA_CMAKE}" \
       --build-arg "PIP_CMAKE=${PIP_CMAKE}" \
       --build-arg "TRITON=${TRITON}" \
       --build-arg "TRITON_CPU=${TRITON_CPU}" \
       --build-arg "ONNX=${ONNX}" \
       --build-arg "DOCS=${DOCS}" \
       --build-arg "INDUCTOR_BENCHMARKS=${INDUCTOR_BENCHMARKS}" \
       --build-arg "EXECUTORCH=${EXECUTORCH}" \
       --build-arg "HALIDE=${HALIDE}" \
       --build-arg "XPU_VERSION=${XPU_VERSION}" \
       --build-arg "UNINSTALL_DILL=${UNINSTALL_DILL}" \
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
  docker run --rm "$tmp_tag" "$@"
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

HAS_TRITON=$(drun python -c "import triton" > /dev/null 2>&1 && echo "yes" || echo "no")
if [[ -n "$TRITON" || -n "$TRITON_CPU" ]]; then
  if [ "$HAS_TRITON" = "no" ]; then
    echo "expecting triton to be installed, but it is not"
    exit 1
  fi
elif [ "$HAS_TRITON" = "yes" ]; then
  echo "expecting triton to not be installed, but it is"
  exit 1
fi
