#!/bin/bash

set -ex

# Adapted from https://hub.docker.com/r/continuumio/anaconda/~/dockerfile/
export LANG=C.UTF-8
export LC_ALL=C.UTF-8

# Install needed packages
apt-get update --fix-missing
apt-get install -y wget

# Install anaconda
echo 'export PATH=/opt/conda/bin:$PATH' > /etc/profile.d/conda.sh
case "$ANACONDA_VERSION" in
  2*)
    wget --quiet https://repo.continuum.io/archive/Anaconda2-5.0.1-Linux-x86_64.sh -O ~/anaconda.sh
  ;;
  3*)
    wget --quiet https://repo.continuum.io/archive/Anaconda3-5.0.1-Linux-x86_64.sh -O ~/anaconda.sh
  ;;
  *)
    echo "Invalid ANACONDA_VERSION..."
    echo $ANACONDA_VERSION
    exit 1
  ;;
esac
/bin/bash ~/anaconda.sh -b -p /opt/conda
rm ~/anaconda.sh

export PATH="/opt/conda/bin:$PATH"
echo 'export PATH=/opt/conda/bin:$PATH' > ~/.bashrc

# This follows the instructions from
# https://caffe2.ai/docs/getting-started.html?platform=ubuntu&configuration=compile
# as closely as possible to install and build. Anaconda should already be
# installed.

# Required dependencies are already installed in install_base.sh

# Optional dependencies not yet installed by install_base.sh
apt-get install -y --no-install-recommends \
      libgflags-dev \
      libgtest-dev \
      libopenmpi-dev

# Optional dependencies installed by pip are not important here
