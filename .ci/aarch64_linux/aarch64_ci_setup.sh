#!/bin/bash
set -eux -o pipefail

# This script is used to prepare the Docker container for aarch64_ci_wheel_build.py python script
# as we need to install conda and setup the python version for the build.

CONDA_PYTHON_EXE=/opt/conda/bin/python
CONDA_EXE=/opt/conda/bin/conda
CONDA_ENV_NAME=aarch64_env
PATH=/opt/conda/bin:$PATH
LD_LIBRARY_PATH=/opt/conda/envs/${CONDA_ENV_NAME}/lib/:/opt/conda/lib:$LD_LIBRARY_PATH

###############################################################################
# Install conda
# disable SSL_verify due to getting "Could not find a suitable TLS CA certificate bundle, invalid path"
# when using Python version, less than the conda latest
###############################################################################
echo 'Installing conda-forge'
curl -L -o /mambaforge.sh https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-aarch64.sh
chmod +x /mambaforge.sh
/mambaforge.sh -b -p /opt/conda
rm /mambaforge.sh
source /opt/conda/etc/profile.d/conda.sh
conda config --set ssl_verify False
conda create -y -c conda-forge -n "${CONDA_ENV_NAME}" python=${DESIRED_PYTHON}
conda activate "${CONDA_ENV_NAME}"

if [[ "$DESIRED_PYTHON"  == "3.13" ]]; then
    pip install -q --pre numpy==2.1.2
    conda install -y -c conda-forge pyyaml==6.0.2 patchelf==0.17.2 pygit2==1.15.1 ninja==1.11.1 scons==4.7.0
else
    pip install -q --pre numpy==2.0.2
    conda install -y -c conda-forge pyyaml==6.0.1 patchelf==0.17.2 pygit2==1.13.2 ninja==1.11.1 scons==4.5.2
fi

python --version
conda --version
