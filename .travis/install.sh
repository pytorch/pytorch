#!/bin/bash

if [[ $TRAVIS_OS_NAME == 'osx' ]]; then
#*******************#
# OS X installation #
#*******************#

  ########################
  # Install dependencies #
  ########################

  brew install homebrew/science/openblas
  brew install glog automake protobuf leveled lmdb

else
#********************#
# Linux installation #
#********************#

  ########################
  # Install dependencies #
  ########################

  sudo apt-get install libprotobuf-dev protobuf-compiler libatlas-base-dev libgoogle-glog-dev liblmdb-dev libleveldb-dev libsnappy-dev python-dev python-pip libiomp-dev libopencv-dev libpthread-stubs0-dev
  sudo pip install numpy

  ################
  # Install CUDA #
  ################

  wget http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1404/x86_64/cuda-repo-ubuntu1404_8.0.44-1_amd64.deb   
  sudo dpkg -i cuda-repo-ubuntu1404_8.0.44-1_amd64.deb
  sudo apt-get update
  sudo apt-get install cuda
  
  #################
  # Install cudnn #
  #################

  # Found here:
  # https://github.com/NVIDIA/nvidia-docker/blob/master/ubuntu-16.04/cuda/8.0/devel/cudnn5/Dockerfile#L11-L16
  CUDNN_DOWNLOAD_SUM=a87cb2df2e5e7cc0a05e266734e679ee1a2fadad6f06af82a76ed81a23b102c8
  CUDNN_URL="http://developer.download.nvidia.com/compute/redist/cudnn/v5.1/cudnn-8.0-linux-x64-v5.1.tgz"
  curl -fsSL ${CUDNN_URL} -O
  echo "$CUDNN_DOWNLOAD_SUM  cudnn-8.0-linux-x64-v5.1.tgz" | sha256sum -c --strict -
  sudo tar -xzf cudnn-8.0-linux-x64-v5.1.tgz -C /usr/local
  rm cudnn-8.0-linux-x64-v5.1.tgz
  sudo ldconfig

fi
