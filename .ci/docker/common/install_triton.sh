#!/bin/bash

set -ex

mkdir -p /opt/triton
if [ -z "${TRITON}" ] && [ -z "${TRITON_CPU}" ]; then
  echo "TRITON and TRITON_CPU are not set. Exiting..."
  exit 0
fi

source "$(dirname "${BASH_SOURCE[0]}")/common_utils.sh"

get_pip_version() {
  env_run pip list | grep -w $* | head -n 1 | awk '{print $2}'
}

if [ -n "${XPU_VERSION}" ]; then
  TRITON_REPO="https://github.com/intel/intel-xpu-backend-for-triton"
  TRITON_TEXT_FILE="triton-xpu"
  # XPU believes new ninja is bad, see https://github.com/intel/intel-xpu-backend-for-triton/commit/fe21682167b831e48bba2544712012abe2f74bb1
  pip_install ninja==1.11.1.4
elif [ -n "${TRITON_CPU}" ]; then
  TRITON_REPO="https://github.com/triton-lang/triton-cpu"
  TRITON_TEXT_FILE="triton-cpu"
else
  TRITON_REPO="https://github.com/triton-lang/triton"
  TRITON_TEXT_FILE="triton"
fi

# The logic here is copied from .ci/pytorch/common_utils.sh
TRITON_PINNED_COMMIT=$(get_pinned_commit ${TRITON_TEXT_FILE})

# ptxas 13.4.46 is currently available only from NVIDIA's preview package
# repository, not the redistributable archive URL used by Triton. Seed the
# expected cache entry so this draft pin can exercise the preview compiler.
if [[ "${TRITON_PINNED_COMMIT}" == "ef4ab63bf41fc21e63bf3d77d11d9365837d0254" ]]; then
  case "$(uname -m)" in
    x86_64)
      package_arch="amd64"
      triton_arch="x86_64"
      package_sha256="4664ae5f28e4eaebf8fea98eca879299a71ee9e54943a5c5a30774f18b69b44e"
      ;;
    aarch64|arm64)
      package_arch="arm64"
      triton_arch="sbsa"
      package_sha256="88cfe8bee7b12d380a05286545462be1de9c6f303ee9bef2a045b3f06ad2fe4e"
      ;;
    *)
      echo "Unsupported architecture for ptxas 13.4.46: $(uname -m)"
      exit 1
      ;;
  esac

  package="cuda-nvcc-13-4_13.4.46-1_${package_arch}.deb"
  package_url="https://packages.nvidia.com/jammy/pool/${package_arch}/5B515474-7E78-11F1-8656-C51E4F4B317F/${package}"
  extract_dir=$(mktemp -d)
  curl --retry 3 -fsSL "${package_url}" -o "${extract_dir}/${package}"
  echo "${package_sha256}  ${extract_dir}/${package}" | sha256sum --check
  dpkg-deb --extract "${extract_dir}/${package}" "${extract_dir}/contents"

  cache_dir="/var/lib/jenkins/.triton/nvidia/nvcc-blackwell/cuda_nvcc-linux-${triton_arch}-13.4.46-archive/bin"
  install -D -m 755 "${extract_dir}/contents/usr/local/cuda-13.4/bin/ptxas" "${cache_dir}/ptxas"
  chown -R jenkins:jenkins /var/lib/jenkins/.triton
  rm -rf "${extract_dir}"
fi

if [ -n "${UBUNTU_VERSION}" ];then
    apt update
    apt-get install -y gpg-agent
fi

# Keep the current cmake and numpy version here, so we can reinstall them later
CMAKE_VERSION=$(get_pip_version cmake)
NUMPY_VERSION=$(get_pip_version numpy)

if [ -z "${MAX_JOBS}" ]; then
    export MAX_JOBS=$(nproc)
fi

# Git checkout triton
mkdir /var/lib/jenkins/triton
chown -R jenkins /var/lib/jenkins/triton
chgrp -R jenkins /var/lib/jenkins/triton
pushd /var/lib/jenkins/

as_jenkins git clone --recursive ${TRITON_REPO} triton
cd triton
as_jenkins git checkout ${TRITON_PINNED_COMMIT}
as_jenkins git submodule update --init --recursive

# Old versions of python have setup.py in ./python; newer versions have it in ./
if [ ! -f setup.py ]; then
  cd python
fi

pip_install pybind11==3.0.1

# TODO: remove patch setup.py once we have a proper fix for https://github.com/triton-lang/triton/issues/4527
as_jenkins sed -i -e 's/https:\/\/tritonlang.blob.core.windows.net\/llvm-builds/https:\/\/oaitriton.blob.core.windows.net\/public\/llvm-builds/g' setup.py

if [ -n "${UBUNTU_VERSION}" ] && [ -n "${GCC_VERSION}" ] && [[ "${GCC_VERSION}" == "7" ]]; then
  # Triton needs at least gcc-9 to build
  apt-get install -y g++-9

  CXX=g++-9 env_run python -m build --wheel --no-isolation
elif [ -n "${UBUNTU_VERSION}" ] && [ -n "${CLANG_VERSION}" ]; then
  # Triton needs <filesystem> which surprisingly is not available with clang-9 toolchain
  add-apt-repository -y ppa:ubuntu-toolchain-r/test
  apt-get install -y g++-9

  CXX=g++-9 env_run python -m build --wheel --no-isolation
else
  env_run python -m build --wheel --no-isolation
fi

# Copy the wheel to /opt for multi stage docker builds
cp dist/*.whl /opt/triton
# Install the wheel for docker builds that don't use multi stage
pip_install dist/*.whl

# TODO: This is to make sure that the same cmake and numpy version from install conda
# script is used. Without this step, the newer cmake version (3.25.2) downloaded by
# triton build step via pip will fail to detect conda MKL. Once that issue is fixed,
# this can be removed.
#
# The correct numpy version also needs to be set here because conda claims that it
# causes inconsistent environment.  Without this, conda will attempt to install the
# latest numpy version, which fails ASAN tests with the following import error: Numba
# needs NumPy 1.20 or less.
# Note that we install numpy with pip as conda might not have the version we want
if [ -n "${CMAKE_VERSION}" ]; then
  pip_install "cmake==${CMAKE_VERSION}"
fi
if [ -n "${NUMPY_VERSION}" ]; then
  pip_install "numpy==${NUMPY_VERSION}"
fi

# IMPORTANT: helion needs to be installed without dependencies.
# It depends on torch and triton. We don't want to install
# triton and torch from production on Docker CI images
if [[ "$ANACONDA_PYTHON_VERSION" != 3.9* ]]; then
  pip_install helion --no-deps
fi
