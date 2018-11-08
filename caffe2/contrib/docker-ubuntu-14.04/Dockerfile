FROM ubuntu:14.04
MAINTAINER caffe-dev <caffe-dev@googlegroups.com>

# A docker container with CUDA and caffe2 installed.
# Note: this should install everything but cudnn, which requires you to have a
# manual registration and download from the NVidia website. After creating this
# docker image, the Caffe2 repository is located at /opt/caffe2. You can install
# cudnn manually and re-compile caffe2.

################################################################################
# Step 1: set up cuda on the ubuntu box.
################################################################################

RUN apt-get update && apt-get install -q -y \
  build-essential \
  wget

RUN cd /tmp && \
  wget http://developer.download.nvidia.com/compute/cuda/7_0/Prod/local_installers/cuda_7.0.28_linux.run && \
  chmod +x cuda_*_linux.run && ./cuda_*_linux.run -extract=`pwd` && \
  ./NVIDIA-Linux-x86_64-*.run -s --no-kernel-module && \
  ./cuda-linux64-rel-*.run -noprompt && \
  rm -rf *

# Ensure the CUDA libs and binaries are in the correct environment variables
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64
ENV PATH=$PATH:/usr/local/cuda/bin

# Run nvcc to make sure things are set correctly.
RUN nvcc --version

################################################################################
# Step 2: set up caffe2 pre-requisites
################################################################################

RUN apt-get update && apt-get install -q -y \
  git \
  libeigen3-dev \
  libgoogle-glog-dev \
  libleveldb-dev \
  liblmdb-dev \
  libopencv-dev \
  libprotobuf-dev \
  libsnappy-dev \
  zlib1g-dev \
  libbz2-dev \
  protobuf-compiler \
  python-dev \
  python-pip

RUN cd /tmp && \
  git clone https://github.com/facebook/rocksdb.git && \
  cd /tmp/rocksdb && \
  make && make install && \
  cd / && \
  rm -rf /tmp/rocksdb

# Caffe2 works best with openmpi 1.8.5 or above (which has cuda support).
# If you do not need openmpi, skip this step.
RUN cd /tmp && \
  wget http://www.open-mpi.org/software/ompi/v1.10/downloads/openmpi-1.10.0.tar.gz && \
  tar xzvf openmpi-1.10.0.tar.gz && \
  cd /tmp/openmpi-1.10.0 && \
  ./configure --with-cuda --with-threads && \
  make && make install && \
  cd / && \
  rm -rf /tmp/openmpi-1.10.0 && \
  rm /tmp/openmpi-1.10.0.tar.gz

# Caffe2 requires zeromq 4.0 or above, manually install.
# If you do not need zeromq, skip this step.
RUN apt-get install -q -y autoconf libtool
RUN mkdir /tmp/zeromq-build && \
  cd /tmp/zeromq-build && \
  wget https://github.com/zeromq/zeromq4-1/archive/v4.1.3.tar.gz && \
  tar xzvf v4.1.3.tar.gz --strip 1 && \
  ./autogen.sh && \
  ./configure --without-libsodium && \
  make && make install && \
  cd / && \
  rm -rf /tmp/zeromq-build

# pip self upgrade
RUN pip install --upgrade pip

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
