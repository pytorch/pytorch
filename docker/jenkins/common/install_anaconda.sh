#!/bin/bash

set -ex

# Adapted from https://hub.docker.com/r/continuumio/anaconda/~/dockerfile/
export LANG=C.UTF-8
export LC_ALL=C.UTF-8

# Pick correct Anaconda package
CONDA_PKG_NAME="Anaconda${ANACONDA_VERSION}-5.0.1-Linux-x86_64.sh"
CONDA_PKG_URL="https://repo.continuum.io/archive/${CONDA_PKG_NAME}"

# Install anaconda
echo 'export PATH=/opt/conda/bin:$PATH' > /etc/profile.d/conda.sh
curl -LO "$CONDA_PKG_URL"
/bin/bash "$CONDA_PKG_NAME" -b -p /opt/conda
rm "$CONDA_PKG_NAME"

export PATH="/opt/conda/bin:$PATH"
echo 'export PATH=/opt/conda/bin:$PATH' > ~/.bashrc
