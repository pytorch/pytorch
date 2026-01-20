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

if [[ "$image" == *-jammy* ]]; then
  UBUNTU_VERSION=22.04
elif [[ "$image" == *-noble* ]]; then
  UBUNTU_VERSION=24.04
elif [[ "$image" == *ubuntu* ]]; then
  extract_version_from_image_name ubuntu UBUNTU_VERSION
fi

if [ -n "${UBUNTU_VERSION}" ]; then
  OS="ubuntu"
else
  echo "Unable to derive operating system base..."
  exit 1
fi

DOCKERFILE="${OS}/Dockerfile"
if [[ "$image" == *rocm* ]]; then
  DOCKERFILE="${OS}-rocm/Dockerfile"
elif [[ "$image" == *xpu* ]]; then
  DOCKERFILE="${OS}-xpu/Dockerfile"
elif [[ "$image" == *cuda*linter* ]]; then
  # Use a separate Dockerfile for linter to keep a small image size
  DOCKERFILE="linter-cuda/Dockerfile"
elif [[ "$image" == *linter* ]]; then
  # Use a separate Dockerfile for linter to keep a small image size
  DOCKERFILE="linter/Dockerfile"
elif [[ "$image" == *riscv* ]]; then
  # Use RISC-V specific Dockerfile
  DOCKERFILE="ubuntu-cross-riscv/Dockerfile"
fi

_UCX_COMMIT=7836b165abdbe468a2f607e7254011c07d788152
_UCC_COMMIT=430e241bf5d38cbc73fc7a6b89155397232e3f96
if [[ "$image" == *rocm* ]]; then
  _UCX_COMMIT=29831d319e6be55cb8c768ca61de335c934ca39e
  _UCC_COMMIT=9f4b242cbbd8b1462cbc732eb29316cdfa124b77
fi

tag=$(echo $image | awk -F':' '{print $2}')
# If no tag (no colon in image name), use the image name itself
if [[ -z "$tag" ]]; then
  tag="$image"
fi

# It's annoying to rename jobs every time you want to rewrite a
# configuration, so we hardcode everything here rather than do it
# from scratch
case "$tag" in
  pytorch-linux-jammy-cuda12.4-cudnn9-py3-gcc11)
    CUDA_VERSION=12.4
    ANACONDA_PYTHON_VERSION=3.10
    GCC_VERSION=11
    VISION=yes
    KATEX=yes
    UCX_COMMIT=${_UCX_COMMIT}
    UCC_COMMIT=${_UCC_COMMIT}
    TRITON=yes
    ;;
  pytorch-linux-jammy-cuda12.8-cudnn9-py3-gcc11)
    CUDA_VERSION=12.8.1
    ANACONDA_PYTHON_VERSION=3.10
    GCC_VERSION=11
    VISION=yes
    KATEX=yes
    UCX_COMMIT=${_UCX_COMMIT}
    UCC_COMMIT=${_UCC_COMMIT}
    TRITON=yes
    INSTALL_MINGW=yes
    ;;
  pytorch-linux-jammy-cuda13.0-cudnn9-py3-gcc11)
    CUDA_VERSION=13.0.2
    ANACONDA_PYTHON_VERSION=3.10
    GCC_VERSION=11
    VISION=yes
    KATEX=yes
    UCX_COMMIT=${_UCX_COMMIT}
    UCC_COMMIT=${_UCC_COMMIT}
    TRITON=yes
    ;;
  pytorch-linux-jammy-cuda12.8-cudnn9-py3-gcc11-inductor-benchmarks)
    CUDA_VERSION=12.8.1
    ANACONDA_PYTHON_VERSION=3.10
    GCC_VERSION=11
    VISION=yes
    KATEX=yes
    UCX_COMMIT=${_UCX_COMMIT}
    UCC_COMMIT=${_UCC_COMMIT}
    TRITON=yes
    INDUCTOR_BENCHMARKS=yes
    ;;
  pytorch-linux-jammy-cuda13.0-cudnn9-py3-gcc11-inductor-benchmarks)
    CUDA_VERSION=13.0.2
    ANACONDA_PYTHON_VERSION=3.10
    GCC_VERSION=11
    VISION=yes
    KATEX=yes
    UCX_COMMIT=${_UCX_COMMIT}
    UCC_COMMIT=${_UCC_COMMIT}
    TRITON=yes
    INDUCTOR_BENCHMARKS=yes
    ;;
  pytorch-linux-jammy-cuda12.9-cudnn9-py3.12-gcc11-vllm)
    CUDA_VERSION=12.9.1
    ANACONDA_PYTHON_VERSION=3.12
    GCC_VERSION=11
    VISION=yes
    KATEX=yes
    UCX_COMMIT=${_UCX_COMMIT}
    UCC_COMMIT=${_UCC_COMMIT}
    TRITON=yes
    ;;
  pytorch-linux-jammy-py3-clang12-onnx)
    ANACONDA_PYTHON_VERSION=3.10
    CLANG_VERSION=12
    VISION=yes
    ONNX=yes
    ;;
  pytorch-linux-jammy-py3.10-clang12)
    ANACONDA_PYTHON_VERSION=3.10
    CLANG_VERSION=12
    VISION=yes
    TRITON=yes
    ;;
  pytorch-linux-jammy-py3.11-clang12)
    ANACONDA_PYTHON_VERSION=3.11
    CLANG_VERSION=12
    VISION=no
    TRITON=no
    ;;
  pytorch-linux-jammy-py3.12-clang12)
    ANACONDA_PYTHON_VERSION=3.12
    CLANG_VERSION=12
    VISION=no
    TRITON=no
    ;;
  pytorch-linux-jammy-rocm-n-py3 | pytorch-linux-jammy-rocm-n-py3-benchmarks | pytorch-linux-noble-rocm-n-py3)
    if [[ $tag =~ "jammy" ]]; then
      ANACONDA_PYTHON_VERSION=3.10
    else
      ANACONDA_PYTHON_VERSION=3.12
    fi
    GCC_VERSION=11
    VISION=yes
    ROCM_VERSION=7.1
    NINJA_VERSION=1.9.0
    TRITON=yes
    KATEX=yes
    UCX_COMMIT=${_UCX_COMMIT}
    UCC_COMMIT=${_UCC_COMMIT}
    PYTORCH_ROCM_ARCH="gfx90a;gfx942;gfx950;gfx1100"
    if [[ $tag =~ "benchmarks" ]]; then
      INDUCTOR_BENCHMARKS=yes
    fi
    ;;
  pytorch-linux-noble-rocm-nightly-py3)
    ANACONDA_PYTHON_VERSION=3.12
    GCC_VERSION=11
    VISION=yes
    ROCM_VERSION=nightly
    NINJA_VERSION=1.9.0
    TRITON=yes
    KATEX=yes
    UCX_COMMIT=${_UCX_COMMIT}
    UCC_COMMIT=${_UCC_COMMIT}
    PYTORCH_ROCM_ARCH="gfx942"
    ;;
  pytorch-linux-jammy-xpu-n-1-py3)
    ANACONDA_PYTHON_VERSION=3.10
    GCC_VERSION=11
    VISION=yes
    XPU_VERSION=2025.2
    NINJA_VERSION=1.9.0
    TRITON=yes
    ;;
  pytorch-linux-noble-xpu-n-py3 | pytorch-linux-noble-xpu-n-py3-inductor-benchmarks)
    ANACONDA_PYTHON_VERSION=3.10
    GCC_VERSION=13
    VISION=yes
    XPU_VERSION=2025.3
    NINJA_VERSION=1.9.0
    TRITON=yes
    if [[ $tag =~ "benchmarks" ]]; then
      INDUCTOR_BENCHMARKS=yes
    fi
    ;;
  pytorch-linux-jammy-py3-gcc11-inductor-benchmarks)
    ANACONDA_PYTHON_VERSION=3.10
    GCC_VERSION=11
    VISION=yes
    KATEX=yes
    TRITON=yes
    DOCS=yes
    INDUCTOR_BENCHMARKS=yes
    ;;
  pytorch-linux-jammy-cuda12.8-cudnn9-py3.10-clang12)
    ANACONDA_PYTHON_VERSION=3.10
    CUDA_VERSION=12.8.1
    CLANG_VERSION=12
    VISION=yes
    TRITON=yes
    ;;
  pytorch-linux-jammy-py3-clang18-asan)
    ANACONDA_PYTHON_VERSION=3.10
    CLANG_VERSION=18
    VISION=yes
    ;;
  pytorch-linux-jammy-py3.10-gcc11)
    ANACONDA_PYTHON_VERSION=3.10
    GCC_VERSION=11
    VISION=yes
    KATEX=yes
    TRITON=yes
    DOCS=yes
    UNINSTALL_DILL=yes
    ;;
  pytorch-linux-jammy-py3-clang12-executorch)
    ANACONDA_PYTHON_VERSION=3.10
    CLANG_VERSION=12
    EXECUTORCH=yes
    ;;
  pytorch-linux-jammy-py3.12-halide)
    CUDA_VERSION=12.6
    ANACONDA_PYTHON_VERSION=3.12
    GCC_VERSION=11
    HALIDE=yes
    TRITON=yes
    ;;
  pytorch-linux-jammy-py3.12-pallas)
    ANACONDA_PYTHON_VERSION=3.12
    GCC_VERSION=11
    PALLAS=yes
    ;;
  pytorch-linux-jammy-cuda12.8-py3.12-pallas)
    CUDA_VERSION=12.8.1
    ANACONDA_PYTHON_VERSION=3.12
    GCC_VERSION=11
    PALLAS=yes
    TRITON=yes
    ;;
  pytorch-linux-jammy-tpu-py3.12-pallas)
    ANACONDA_PYTHON_VERSION=3.12
    GCC_VERSION=11
    PALLAS=yes
    TPU=yes
    ;;
  pytorch-linux-jammy-py3.12-triton-cpu)
    CUDA_VERSION=12.6
    ANACONDA_PYTHON_VERSION=3.12
    GCC_VERSION=11
    TRITON_CPU=yes
    ;;
  pytorch-linux-jammy-linter)
    PYTHON_VERSION=3.10
    ;;
  pytorch-linux-jammy-cuda12.8-cudnn9-py3.10-linter)
    PYTHON_VERSION=3.10
    CUDA_VERSION=12.8.1
    ;;
  pytorch-linux-jammy-aarch64-py3.10-gcc13)
    ANACONDA_PYTHON_VERSION=3.10
    GCC_VERSION=13
    ACL=yes
    VISION=yes
    OPENBLAS=yes
    # snadampal: skipping llvm src build install because the current version
    # from pytorch/llvm:9.0.1 is x86 specific
    SKIP_LLVM_SRC_BUILD_INSTALL=yes
    ;;
  pytorch-linux-jammy-aarch64-py3.10-clang21)
    ANACONDA_PYTHON_VERSION=3.10
    CLANG_VERSION=21
    ACL=yes
    VISION=yes
    OPENBLAS=yes
    # snadampal: skipping llvm src build install because the current version
    # from pytorch/llvm:9.0.1 is x86 specific
    SKIP_LLVM_SRC_BUILD_INSTALL=yes
    ;;
  pytorch-linux-jammy-aarch64-py3.10-gcc13-inductor-benchmarks)
    ANACONDA_PYTHON_VERSION=3.10
    GCC_VERSION=13
    ACL=yes
    VISION=yes
    OPENBLAS=yes
    # snadampal: skipping llvm src build install because the current version
    # from pytorch/llvm:9.0.1 is x86 specific
    SKIP_LLVM_SRC_BUILD_INSTALL=yes
    INDUCTOR_BENCHMARKS=yes
    ;;
  pytorch-linux-noble-riscv64-py3.12-gcc14)
    GCC_VERSION=14
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
    fi
    if [[ "$image" == *rocm* ]]; then
      if [[ -z "$ROCM_VERSION" ]]; then
        extract_version_from_image_name rocm ROCM_VERSION
      fi
      NINJA_VERSION=1.9.0
      TRITON=yes
      # To ensure that any ROCm config will build using conda cmake
      # and thus have LAPACK/MKL enabled
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
  ;;
esac

tmp_tag=$(basename "$(mktemp -u)" | tr '[:upper:]' '[:lower:]')

no_cache_flag=""
progress_flag=""
# Do not use cache and progress=plain when in CI
if [[ -n "${CI:-}" ]]; then
  no_cache_flag="--no-cache"
  progress_flag="--progress=plain"
fi

# Build image
docker buildx build \
       ${no_cache_flag} \
       ${progress_flag} \
       --build-arg "BUILD_ENVIRONMENT=${image}" \
       --build-arg "LLVMDEV=${LLVMDEV:-}" \
       --build-arg "VISION=${VISION:-}" \
       --build-arg "UBUNTU_VERSION=${UBUNTU_VERSION}" \
       --build-arg "DEVTOOLSET_VERSION=${DEVTOOLSET_VERSION}" \
       --build-arg "GLIBC_VERSION=${GLIBC_VERSION}" \
       --build-arg "CLANG_VERSION=${CLANG_VERSION}" \
       --build-arg "ANACONDA_PYTHON_VERSION=${ANACONDA_PYTHON_VERSION}" \
       --build-arg "PYTHON_VERSION=${PYTHON_VERSION}" \
       --build-arg "GCC_VERSION=${GCC_VERSION}" \
       --build-arg "CUDA_VERSION=${CUDA_VERSION}" \
       --build-arg "NINJA_VERSION=${NINJA_VERSION:-}" \
       --build-arg "KATEX=${KATEX:-}" \
       --build-arg "ROCM_VERSION=${ROCM_VERSION:-}" \
       --build-arg "PYTORCH_ROCM_ARCH=${PYTORCH_ROCM_ARCH}" \
       --build-arg "IMAGE_NAME=${IMAGE_NAME}" \
       --build-arg "UCX_COMMIT=${UCX_COMMIT}" \
       --build-arg "UCC_COMMIT=${UCC_COMMIT}" \
       --build-arg "TRITON=${TRITON}" \
       --build-arg "TRITON_CPU=${TRITON_CPU}" \
       --build-arg "ONNX=${ONNX}" \
       --build-arg "DOCS=${DOCS}" \
       --build-arg "INDUCTOR_BENCHMARKS=${INDUCTOR_BENCHMARKS}" \
       --build-arg "EXECUTORCH=${EXECUTORCH}" \
       --build-arg "HALIDE=${HALIDE}" \
       --build-arg "PALLAS=${PALLAS}" \
       --build-arg "TPU=${TPU}" \
       --build-arg "XPU_VERSION=${XPU_VERSION}" \
       --build-arg "UNINSTALL_DILL=${UNINSTALL_DILL}" \
       --build-arg "ACL=${ACL:-}" \
       --build-arg "OPENBLAS=${OPENBLAS:-}" \
       --build-arg "SKIP_SCCACHE_INSTALL=${SKIP_SCCACHE_INSTALL:-}" \
       --build-arg "SKIP_LLVM_SRC_BUILD_INSTALL=${SKIP_LLVM_SRC_BUILD_INSTALL:-}" \
       --build-arg "INSTALL_MINGW=${INSTALL_MINGW:-}" \
       -f $(dirname ${DOCKERFILE})/Dockerfile \
       --load \
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
  if [[ "$image" == *riscv* ]]; then
    # Check RISC-V cross-compilation toolchain version
    if !(drun riscv64-linux-gnu-gcc-${GCC_VERSION} --version 2>&1 | grep -q " $GCC_VERSION\\W"); then
      echo "RISC-V GCC_VERSION=$GCC_VERSION, but:"
      drun riscv64-linux-gnu-gcc-${GCC_VERSION} --version
      exit 1
    fi
  elif !(drun gcc --version 2>&1 | grep -q " $GCC_VERSION\\W"); then
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
