#!/bin/bash

set -ex

if [ -n "${UBUNTU_VERSION}" ]; then
  sudo apt-get install -y doxygen
fi

# Install all linter dependencies
python3 -mpip install -r .github/requirements-gha-cache.txt --user
lintrunner init
