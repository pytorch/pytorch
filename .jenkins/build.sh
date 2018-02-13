#!/bin/bash

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

# Required environment variables:
#   $JOB_NAME
#   $PYTHON_VERSION
#   $GCC_VERSION
#
# TODO: change this script to make use of $BUILD_ENVIRONMENT,
# which we can hard code into Docker images and then these scripts
# will work out of the box without having to set any env vars.

export PATH=/opt/conda/bin:$PATH

# Install sccache binaries
pushd /tmp
SCCACHE_BASE_URL="https://github.com/mozilla/sccache/releases/download/"
SCCACHE_VERSION="0.2.5"
SCCACHE_BASE="sccache-${SCCACHE_VERSION}-x86_64-unknown-linux-musl"
SCCACHE_FILE="$SCCACHE_BASE.tar.gz"
wget -q "$SCCACHE_BASE_URL/$SCCACHE_VERSION/$SCCACHE_FILE"
tar xzf $SCCACHE_FILE
sudo mv "$SCCACHE_BASE/sccache" /usr/local/bin
popd

if [[ "$JOB_NAME" != *cuda* ]]; then
   export PATH=/opt/python/${PYTHON_VERSION}/bin:$PATH
   export LD_LIBRARY_PATH=/opt/python/${PYTHON_VERSION}/lib:$LD_LIBRARY_PATH
   export CC="/usr/local/bin/sccache ccache /usr/bin/gcc-${GCC_VERSION}"
   export CXX="/usr/local/bin/sccache ccache /usr/bin/g++-${GCC_VERSION}"
else
   # CMake should use ccache symlink for nvcc
   export CUDA_NVCC_EXECUTABLE=/usr/local/bin/nvcc
   # The ccache wrapper should be able to find the real nvcc
   export PATH=/usr/local/cuda/bin:$PATH

   # Add CUDA stub/real path to loader path
   export LD_LIBRARY_PATH=/usr/local/cuda/lib64/stubs:$LD_LIBRARY_PATH
   export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

   # Build for Maxwell
   export TORCH_CUDA_ARCH_LIST="Maxwell"
   export TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
fi

echo "Python Version:"
which python
python --version

pip install -r requirements.txt || true

time python setup.py install

if [[ "$JOB_NAME" != *cuda* ]]; then
   echo "Testing ATen"
   time tools/run_aten_tests.sh
fi

# Test C FFI plugins
# cffi install doesn't work for Python 3.7
if [[ "$JOB_NAME" != *pynightly* ]]; then
   pip install cffi
   git clone https://github.com/pytorch/extension-ffi.git
   cd extension-ffi/script
   python build.py
fi
