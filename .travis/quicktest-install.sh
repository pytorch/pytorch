#!/bin/bash
set -e
set -x

sudo apt-get update
sudo apt-get install -y --no-install-recommends \
    libatlas-base-dev \
    libgoogle-glog-dev \
    libiomp-dev \
    libleveldb-dev \
    liblmdb-dev \
    libopencv-dev \
    libprotobuf-dev \
    libpthread-stubs0-dev \
    libsnappy-dev \
    protobuf-compiler \
    python-dev \
    python-pip

# Can't use deb packages because of TravisCI's virtualenv
pip install numpy
