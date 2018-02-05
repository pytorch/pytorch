#!/bin/bash

set -ex

# Required environment variables:
#   $JOB_NAME
#   $PYTHON_VERSION
#   $GCC_VERSION

export PATH=/opt/conda/bin:$PATH

if [[ "$JOB_NAME" == *cuda* ]]; then
   export LD_LIBRARY_PATH=/usr/local/cuda/lib64/stubs:$LD_LIBRARY_PATH
   export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
else
   export PATH=/opt/python/${PYTHON_VERSION}/bin:$PATH
   export LD_LIBRARY_PATH=/opt/python/${PYTHON_VERSION}/lib:$LD_LIBRARY_PATH

   # NB: setup.py chokes on a setting of CC='ccache gcc' (two words),
   # so we created a symlinked binary that we can pass as CC in one word
   mkdir ./ccache
   ln -sf "$(which ccache)" ./ccache/gcc-${GCC_VERSION}
   ln -sf "$(which ccache)" ./ccache/g++-${GCC_VERSION}
   export CC="$PWD/ccache/gcc-${GCC_VERSION}"
   export CXX="$PWD/ccache/g++-${GCC_VERSION}"
fi

echo "Installing torchvision at branch master"
rm -rf vision
git clone https://github.com/pytorch/vision --quiet
if [[ "$JOB_NAME" == *cuda* ]]; then
   conda install -y pillow
else
   pip install pillow
fi

echo "ENTERED_USER_LAND"

echo "Testing pytorch"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

# JIT C++ extensions require ninja.
git clone https://github.com/ninja-build/ninja --quiet
pushd ninja
python ./configure.py --bootstrap
export PATH="$PWD:$PATH"
popd

time test/run_test.sh

rm -rf ninja

pushd vision
time python setup.py install
popd

echo "EXITED_USER_LAND"
