#!/bin/bash

set -ex

source "$(dirname "${BASH_SOURCE[0]}")/common_utils.sh"

if [ -n "${UBUNTU_VERSION}" ]; then
  apt update
  apt-get install -y doxygen graphviz nodejs npm libtinfo5
fi

# Install all linter dependencies. The lintrunner init step would still need to be
# run later in the CI because the initialization requires some custom scripts from
# PyTorch. The good news is having these dependencies installed here would act as
# a cache, so lintrunner init wouldn't need to download them all again
pip_install -r /opt/conda/requirements.txt
pip_install -r /opt/conda/requirements-linter.txt

# Node dependencies required by toc linter job
npm install -g markdown-toc
