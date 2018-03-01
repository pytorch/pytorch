#!/bin/bash

set -ex

# Adapted from https://hub.docker.com/r/continuumio/anaconda/~/dockerfile/
export LANG=C.UTF-8
export LC_ALL=C.UTF-8

# Install needed packages
# This also needs build-essentials but that should be installed already
apt-get update --fix-missing
apt-get install -y wget

# Install anaconda
echo 'export PATH=/opt/conda/bin:$PATH' > /etc/profile.d/conda.sh
case "$ANACONDA_VERSION" in
  2*)
    wget https://repo.continuum.io/archive/Anaconda2-5.0.1-Linux-x86_64.sh -O ~/anaconda.sh
  ;;
  3*)
    wget https://repo.continuum.io/archive/Anaconda3-5.0.1-Linux-x86_64.sh -O ~/anaconda.sh
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
