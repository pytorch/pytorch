FROM ubuntu:14.04
LABEL maintainer="aaronmarkham@fb.com"

# caffe2 install with cpu support

########## REQUIRED DEPENDENCIES ################
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    cmake \
    git \
    libgoogle-glog-dev \
    libprotobuf-dev \
    python-pip \
    protobuf-compiler \
    python-dev \
    && rm -rf /var/lib/apt/lists/*

# Don't use deb package because trusty's pip is too old for --no-cache-dir
RUN curl -O https://bootstrap.pypa.io/get-pip.py \
    && python get-pip.py \
    && rm get-pip.py
RUN pip install --no-cache-dir --upgrade pip setuptools wheel
RUN pip install --no-cache-dir future hypothesis numpy protobuf six

########## INSTALLATION STEPS ###################
RUN git clone --branch master --recursive https://github.com/caffe2/caffe2.git
RUN cd caffe2 && mkdir build && cd build \
    && cmake .. \
    -DUSE_CUDA=OFF \
    -DUSE_NNPACK=OFF \
    -DUSE_ROCKSDB=OFF \
    && make -j"$(nproc)" install \
    && ldconfig \
    && make clean \
    && cd .. \
    && rm -rf build

ENV PYTHONPATH /usr/local
