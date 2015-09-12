FROM ubuntu:12.04
MAINTAINER caffe-dev <caffe-dev@googlegroups.com>

# A docker container with CUDA and caffe2 installed.
# Note: this exists mostly as an illustrative example for existing ubuntu 12.04
# users. Now that 14.04 is out and 15.04 is on the way, you probably want to
# upgrade. Especially, if you are using docker, do NOT use this image. Use the
# 14.04 version instead.

################################################################################
# Step 1: set up build essentials and cuda on the ubuntu box.
################################################################################

# Install build essentialls.
RUN apt-get update && apt-get install -q -y \
  build-essential \
  python-software-properties \
  software-properties-common \
  wget

# We have to update gcc to support c++11 on ubuntu 12.04
RUN add-apt-repository ppa:ubuntu-toolchain-r/test

# Use gcc 4.8 as the default compiler.
RUN apt-get update && apt-get install -y gcc-4.8 g++-4.8 && \
  update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-4.8 20 && \
  update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-4.8 20 && \
  update-alternatives --config gcc && \
  update-alternatives --config g++

# Install cuda.
RUN cd /tmp && \
  wget http://developer.download.nvidia.com/compute/cuda/7_0/Prod/local_installers/cuda_7.0.28_linux.run && \
  chmod +x cuda_*_linux.run && ./cuda_*_linux.run -extract=`pwd` && \
  ./NVIDIA-Linux-x86_64-*.run -s --no-kernel-module && \
  ./cuda-linux64-rel-*.run -noprompt && \
  rm -rf *

# Ensure the CUDA libs and binaries are in the correct environment variables
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64
ENV PATH=$PATH:/usr/local/cuda/bin

################################################################################
# Step 2: set up caffe2 pre-requisites
################################################################################

RUN apt-get update && apt-get install -q -y \
  git \
  libeigen3-dev \
  libleveldb-dev \
  libopencv-dev \
  libprotobuf-dev \
  libsnappy-dev \
  protobuf-compiler \
  python-dev \
  python-pip \
  unzip

# cmake. Ubuntu 12.04 ships a too old cmake version, so we build from source.
# This is needed by the most recent version of gflags.
RUN cd /tmp && \
  wget http://www.cmake.org/files/v3.3/cmake-3.3.1.tar.gz && \
  tar xzf cmake-3.3.1.tar.gz && \
  cd cmake-3.3.1 && \
  ./bootstrap && make && make install && \
  cd / && rm /tmp/cmake-3.3.1.tar.gz && rm -rf /tmp/cmake-3.3.1

# glog
RUN cd /tmp && \
  wget https://google-glog.googlecode.com/files/glog-0.3.3.tar.gz && \
  tar xzf glog-0.3.3.tar.gz && \
  cd glog-0.3.3 && \
  ./configure && \
  make && make install && \
  cd / && rm -rf /tmp/glog-0.3.3 /tmp/glog-0.3.3.tar.gz

# gflags
RUN cd /tmp && \
  wget https://github.com/schuhschuh/gflags/archive/master.zip && \
  unzip master.zip && \
  cd gflags-master && \
  mkdir build && cd build && \
  export CXXFLAGS="-fPIC" && cmake .. && make VERBOSE=1 && \
  make install && \
  cd / && rm /tmp/master.zip && rm -rf /tmp/gflags-master

# lmdb. lmdb's Makefile has a potential error: if /usr/local/share/man/man1 does
# not exist as a folder yet, the installation script will copy lmdb's doc as a
# man1 *file*. We'll manually fix this by creating the man1 folder first.
RUN mkdir -p /usr/local/share/man/man1 && \
  cd /tmp && \
  git clone https://github.com/LMDB/lmdb && \
  cd lmdb/libraries/liblmdb && \
  make && make install && \
  cd / && rm -rf /tmp/lmdb

# Caffe2 works best with openmpi 1.8.5 or above (which has cuda support).
# If you do not need openmpi, skip this step.
RUN cd /tmp && \
  wget http://www.open-mpi.org/software/ompi/v1.8/downloads/openmpi-1.8.7.tar.gz && \
  tar xzvf openmpi-1.8.7.tar.gz && \
  cd /tmp/openmpi-1.8.7 && \
  ./configure --with-cuda --with-threads --enable-mpi-thread-multiple && \
  make && make install && \
  cd / && \
  rm -rf /tmp/openmpi-1.8.7 && \
  rm /tmp/openmpi-1.8.7.tar.gz

# Caffe2 requires zeromq 4.0 or above, manually install.
# If you do not need zeromq, skip this step.
RUN cd /tmp && \
  wget http://download.zeromq.org/zeromq-4.1.3.tar.gz && \
  tar xzvf zeromq-4.1.3.tar.gz && \
  cd /tmp/zeromq-4.1.3 && \
  ./configure --without-libsodium && \
  make && make install && \
  cd / && \
  rm -rf /tmp/zeromq-4.1.3 && \
  rm /tmp/zeromq-4.1.3.tar.gz

# pip self upgrade
RUN pip install --upgrade pip

# pip update distribute. 12.04's distribute version is too old for matplotlib.
RUN pip install --upgrade distribute

# Python dependencies
RUN pip install \
  matplotlib \
  numpy \
  protobuf

################################################################################
# Step 3: install optional dependencies ("good to have" features)
################################################################################

RUN apt-get install -q -y \
  gfortran \
  graphviz \
  libatlas-base-dev \
  vim

RUN pip install \
  flask \
  ipython \
  notebook \
  pydot \
  python-nvd3 \
  scipy \
  tornado

# This is intentional. scikit-image has to be after scipy.
RUN pip install \
  scikit-image

################################################################################
# Step 4: set up caffe2
################################################################################

# Get the repository, and build.
RUN cd /opt && \
  git clone https://github.com/Yangqing/caffe2.git && \
  cd /opt/caffe2 && \
  make

# Now, we know that some of the caffe tests will fail. How do we deal with
# those?
