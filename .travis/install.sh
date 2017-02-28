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

  brew install glog automake protobuf leveldb lmdb
  sudo pip install numpy

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
  pip install numpy


  #########################
  # Install MKL if needed #
  #########################


  if [[ $BLAS == 'MKL' ]]; then
    wget http://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS-2019.PUB
    sudo apt-key add GPG-PUB-KEY-INTEL-SW-PRODUCTS-2019.PUB
    sudo sh -c 'echo deb http://apt.repos.intel.com/mkl stable main > /etc/apt/sources.list.d/intel-mkl.list'
    sudo apt-get update
    sudo apt-get install intel-mkl
  fi

  ################
  # Install CUDA #
  ################

  CUDA_REPO_PKG="cuda-repo-ubuntu1404_8.0.44-1_amd64.deb"
  CUDA_PKG_VERSION="8-0"
  CUDA_VERSION="8.0"

  wget http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1404/x86_64/$CUDA_REPO_PKG
  sudo dpkg -i $CUDA_REPO_PKG
  rm $CUDA_REPO_PKG
  sudo apt-get update
  sudo apt-get install -y --no-install-recommends \
      cuda-core-$CUDA_PKG_VERSION \
      cuda-cublas-dev-$CUDA_PKG_VERSION \
      cuda-cudart-dev-$CUDA_PKG_VERSION \
      cuda-curand-dev-$CUDA_PKG_VERSION \
      cuda-driver-dev-$CUDA_PKG_VERSION \
      cuda-nvrtc-dev-$CUDA_PKG_VERSION

  # manually create CUDA symlink
  sudo ln -s /usr/local/cuda-$CUDA_VERSION /usr/local/cuda

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
