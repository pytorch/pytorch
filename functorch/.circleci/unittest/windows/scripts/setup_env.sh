#!/usr/bin/env bash

# This script is for setting up environment in which unit test is ran.
# To speed up the CI time, the resulting environment is cached.
#
# Do not install PyTorch and torchvision here, otherwise they also get cached.

set -e

this_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
root_dir="$(git rev-parse --show-toplevel)"
conda_dir="${root_dir}/conda"
env_dir="${root_dir}/env"

cd "${root_dir}"

# 1. Install conda at ./conda
if [ ! -d "${conda_dir}" ]; then
    printf "* Installing conda\n"
    export tmp_conda="$(echo $conda_dir | tr '/' '\\')"
    export miniconda_exe="$(echo $root_dir | tr '/' '\\')\\miniconda.exe"
    curl --output miniconda.exe https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe -O
    "$this_dir/install_conda.bat"
    unset tmp_conda
    unset miniconda_exe
fi

eval "$(${conda_dir}/Scripts/conda.exe 'shell.bash' 'hook')"

# 2. Create test environment at ./env
if [ ! -d "${env_dir}" ]; then
    printf "* Creating a test environment\n"
    conda create --prefix "${env_dir}" -y python="$PYTHON_VERSION"
fi
conda activate "${env_dir}"

# 3. Install Conda dependencies
printf "* Installing dependencies (except PyTorch)\n"
conda env update --file "${this_dir}/environment.yml" --prune
