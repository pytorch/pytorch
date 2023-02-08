#!/bin/bash

set -ex

source "$(dirname "${BASH_SOURCE[0]}")/common_utils.sh"

if [ -n "${UBUNTU_VERSION}" ]; then
  apt update
  apt-get install -y clang doxygen graphviz nodejs npm libtinfo5
fi

# Install all linter dependencies
pip_install -r /opt/conda/requirements.txt
lintrunner init

# Node dependencies required by toc linter job
npm install -g markdown-toc
