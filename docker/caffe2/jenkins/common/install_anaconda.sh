#!/bin/bash

set -ex

# Pick correct Anaconda package
CONDA_PKG_NAME="Anaconda${ANACONDA_VERSION}-5.0.1-Linux-x86_64.sh"
CONDA_PKG_URL="https://repo.continuum.io/archive/${CONDA_PKG_NAME}"

# Install anaconda
echo 'export PATH=/opt/conda/bin:$PATH' > /etc/profile.d/conda.sh
curl -LO "$CONDA_PKG_URL"
/bin/bash "$CONDA_PKG_NAME" -b -p /opt/conda
rm "$CONDA_PKG_NAME"

# Install packages needed for tests, but that aren't included in the base conda
# requirements to keep them slim
# pyyaml is needed to build Aten
/opt/conda/bin/conda install -y hypothesis tabulate pydot pyyaml mock
