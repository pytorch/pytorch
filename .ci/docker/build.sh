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
fi

PY_HARDCODED_CONFIG_SCRIPT=$(python3 get_config.py --image "$image")

if [[ $? -eq 0 ]]; then
  eval "$PY_HARDCODED_CONFIG_SCRIPT"
else
  echo "[Fallback] Python script failed or no match — fallback to hardcoded shell case"
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
docker build \
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
       --build-arg "CUDNN_VERSION=${CUDNN_VERSION}" \
       --build-arg "TENSORRT_VERSION=${TENSORRT_VERSION}" \
       --build-arg "GRADLE_VERSION=${GRADLE_VERSION}" \
       --build-arg "NINJA_VERSION=${NINJA_VERSION:-}" \
       --build-arg "KATEX=${KATEX:-}" \
       --build-arg "ROCM_VERSION=${ROCM_VERSION:-}" \
       --build-arg "PYTORCH_ROCM_ARCH=${PYTORCH_ROCM_ARCH:-gfx90a;gfx942}" \
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
       --build-arg "XPU_VERSION=${XPU_VERSION}" \
       --build-arg "UNINSTALL_DILL=${UNINSTALL_DILL}" \
       --build-arg "ACL=${ACL:-}" \
       --build-arg "OPENBLAS=${OPENBLAS:-}" \
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

# Sanity check cmake version.  Executorch reinstalls cmake and I'm not sure if
# they support 4.0.0 yet, so exclude them from this check.
CMAKE_VERSION=$(drun cmake --version)
if [[ "$EXECUTORCH" != *yes* && "$CMAKE_VERSION" != *4.* ]]; then
  echo "CMake version is not 4.0.0:"
  drun cmake --version
  exit 1
fi
