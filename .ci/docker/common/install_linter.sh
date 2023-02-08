#!/bin/bash

set -ex

source "$(dirname "${BASH_SOURCE[0]}")/common_utils.sh"

if [ -n "${UBUNTU_VERSION}" ]; then
  apt-get install -y doxygen nodejs npm
fi

# Install all linter dependencies
pip_install -r requirements.txt
pip_install -r .github/requirements-gha-cache.txt

conda_run lintrunner init

# Node dependencies required by toc linter job
npm install -g markdown-toc
