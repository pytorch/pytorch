#!/bin/bash

set -ex

if [ -n "${UBUNTU_VERSION}" ]; then
  apt update
  apt-get install -y clang doxygen git graphviz nodejs npm libtinfo5
fi

# Do shallow clone of PyTorch so that we can init lintrunner in Docker build context
git clone https://github.com/pytorch/pytorch.git --depth 1
chown -R jenkins pytorch

pushd pytorch
# Install all linter dependencies
pip install -r requirements.txt
lintrunner init

# Cache .lintbin directory as part of the Docker image
cp -r .lintbin /tmp
popd

# Node dependencies required by toc linter job
npm install -g markdown-toc

# Cleaning up
rm -rf pytorch
