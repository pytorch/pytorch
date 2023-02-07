#!/bin/bash

set -ex

if [ -n "${UBUNTU_VERSION}" ]; then
  sudo apt-get install -y doxygen nodejs npm
fi

# Install all linter dependencies
python3 -mpip install -r requirements.txt --user
python3 -mpip install -r .github/requirements-gha-cache.txt --user
lintrunner init

# Node dependencies required by toc linter job
npm install -g markdown-toc
