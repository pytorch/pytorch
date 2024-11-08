#!/bin/bash
set -eux -o pipefail

# This script is used to prepare the Docker container for aarch64_ci_wheel_build.py python script
# By creating symlinks from desired /opt/python to /usr/local/bin/

SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"
source $SCRIPTPATH/../manywheel/set_desired_python.sh

pip install -q numpy==2.1.2 pyyaml==6.0.2 scons==4.7.0 ninja==1.11.1 patchelf==0.17.2 pygit2==1.15.1

for tool in python python3 pip pip3 ninja scons patchelf; do
    ln -sf ${DESIRED_PYTHON_BIN_DIR}/${tool} /usr/local/bin;
done

python --version
