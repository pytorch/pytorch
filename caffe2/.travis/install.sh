#!/bin/bash
set -e

if [[ $BUILD_TARGET == 'android' ]]; then
#**********************************************#
# Android installation, both on OS X and Linux #
#**********************************************#

  if [[ $TRAVIS_OS_NAME == 'osx' ]]; then
    brew install automake libtool
    # Install android ndk
    wget https://dl.google.com/android/repository/android-ndk-r13b-darwin-x86_64.zip
    sudo mkdir -p /opt/android_ndk
    sudo chmod a+rwx /opt/android_ndk
    unzip -qo android-ndk-r13b-darwin-x86_64.zip -d /opt/android_ndk
  else
    sudo apt-get install autotools-dev autoconf
    # Install android ndk
    wget https://dl.google.com/android/repository/android-ndk-r13b-linux-x86_64.zip
    sudo mkdir -p /opt/android_ndk
    sudo chmod a+rwx /opt/android_ndk
    unzip -qo android-ndk-r13b-linux-x86_64.zip -d /opt/android_ndk
  fi

elif [[ $TRAVIS_OS_NAME == 'osx' ]]; then
#*******************#
# OS X installation #
#*******************#

  ########################
  # Install dependencies #
  ########################

  brew install glog automake protobuf leveldb lmdb ninja
  sudo pip install numpy
  # Dependencies needed for NNPACK: PeachPy and confu
  sudo pip install --upgrade git+https://github.com/Maratyszcza/PeachPy
  sudo pip install --upgrade git+https://github.com/Maratyszcza/confu

else
#********************#
# Linux installation #
#********************#

  ########################
  # Install dependencies #
  ########################
  sudo add-apt-repository -y ppa:ubuntu-toolchain-r/test
  sudo apt-get update
  sudo apt-get install libprotobuf-dev protobuf-compiler libatlas-base-dev libgoogle-glog-dev liblmdb-dev libleveldb-dev libsnappy-dev python-dev python-pip libiomp-dev libopencv-dev libpthread-stubs0-dev
  # Dependency needed for NNPACK: the most recent version of ninja build
  git clone https://github.com/ninja-build/ninja.git /tmp/ninja
  pushd /tmp/ninja
  git checkout release
  python configure.py --bootstrap
  mkdir -p $HOME/.local/bin
  install -m 755 /tmp/ninja/ninja $HOME/.local/bin/ninja
  popd
  export PATH=$HOME/.local/bin:$PATH
  # Dependencies needed for NNPACK: PeachPy and confu
  pip install --upgrade git+https://github.com/Maratyszcza/PeachPy
  pip install --upgrade git+https://github.com/Maratyszcza/confu
  pip install numpy

  #########################
  # Install MKL if needed #
  #########################


  if [[ $BLAS == 'MKL' ]]; then
    wget http://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS-2019.PUB
    sudo apt-key add GPG-PUB-KEY-INTEL-SW-PRODUCTS-2019.PUB
    sudo sh -c 'echo deb http://apt.repos.intel.com/mkl all main > /etc/apt/sources.list.d/intel-mkl.list'
    sudo apt-get update
    sudo apt-get install intel-mkl-64bit-2017.2-050
  fi

  ################
  # Install CUDA #
  ################

  source /etc/lsb-release

  REPO="ubuntu1404"
  if [ "${DISTRIB_RELEASE}" == "16.04" ]; then
      REPO="ubuntu1604"
  fi

  CUDA_REPO_PKG="cuda-repo-${REPO}_8.0.44-1_amd64.deb"
  CUDA_PKG_VERSION="8-0"
  CUDA_VERSION="8.0"

  wget "http://developer.download.nvidia.com/compute/cuda/repos/${REPO}/x86_64/${CUDA_REPO_PKG}"
  sudo dpkg -i "${CUDA_REPO_PKG}"
  rm -f "${CUDA_REPO_PKG}"
  sudo apt-get update
  sudo apt-get install -y --no-install-recommends \
      "cuda-core-${CUDA_PKG_VERSION}" \
      "cuda-cublas-dev-${CUDA_PKG_VERSION}" \
      "cuda-cudart-dev-${CUDA_PKG_VERSION}" \
      "cuda-curand-dev-${CUDA_PKG_VERSION}" \
      "cuda-driver-dev-${CUDA_PKG_VERSION}" \
      "cuda-nvrtc-dev-${CUDA_PKG_VERSION}"

  # manually create CUDA symlink
  sudo ln -sf /usr/local/cuda-$CUDA_VERSION /usr/local/cuda

  #################
  # Install cudnn #
  #################

  # Found here:
  # https://gitlab.com/nvidia/cuda/blob/ff2d7c34fe/8.0/devel/cudnn5/Dockerfile
  CUDNN_DOWNLOAD_SUM=c10719b36f2dd6e9ddc63e3189affaa1a94d7d027e63b71c3f64d449ab0645ce
  CUDNN_URL="http://developer.download.nvidia.com/compute/redist/cudnn/v5.1/cudnn-8.0-linux-x64-v5.1.tgz"
  curl -fsSL ${CUDNN_URL} -O
  echo "$CUDNN_DOWNLOAD_SUM  cudnn-8.0-linux-x64-v5.1.tgz" | sha256sum -c --strict -
  sudo tar -xzf cudnn-8.0-linux-x64-v5.1.tgz -C /usr/local
  rm cudnn-8.0-linux-x64-v5.1.tgz
  sudo ldconfig
fi
