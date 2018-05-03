#!/bin/bash

COMPACT_JOB_NAME=pytorch-macos-10.13-py3-build-test
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

# Set up conda environment
curl https://repo.continuum.io/miniconda/Miniconda3-latest-MacOSX-x86_64.sh -o $PWD/miniconda3.sh
rm -rf $PWD/miniconda3
bash $PWD/miniconda3.sh -b -p $PWD/miniconda3
export PATH="$PWD/miniconda3/bin:$PATH"
source $PWD/miniconda3/bin/activate
conda install -y mkl mkl-include numpy pyyaml setuptools cmake cffi ninja

git submodule update --init --recursive
export CMAKE_PREFIX_PATH=$PWD/miniconda3/

# Build and test PyTorch
export MACOSX_DEPLOYMENT_TARGET=10.9
export CXX=clang++
export CC=clang
# If we run too many parallel jobs, we will OOM
export MAX_JOBS=2
python setup.py install
echo "Ninja version: $(ninja --version)"
python test/run_test.py --verbose

# C++ API

# NB: Install outside of source directory (at the same level as the root
# pytorch folder) so that it doesn't get cleaned away prior to docker push.
CPP_BUILD="$PWD/../cpp-build"
WERROR=1 VERBOSE=1 tools/cpp_build/build_all.sh "$CPP_BUILD"

python tools/download_mnist.py --quiet -d test/cpp/api/mnist
"$CPP_BUILD"/libtorch/bin/test_api
