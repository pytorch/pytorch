#!/bin/bash

set -x

rgpu getGpuInfo | jq .

CONDA_ENV_DIR="${CONDA_ENV_DIR:-/conda}"

if [ -z "$CONDA_ENV_FBPKG" ]; then
  # Copy the folder because the default dir is read-only
  cp -r ${CONDA_ENV_DIR} .
else
  mkdir conda
  fbpkg.fetch $CONDA_ENV_FBPKG -d conda
fi

# Prepare conda environment
pushd conda

echo "Unpacking conda env..."
time conda/bin/conda-unpack
echo $?

echo "Activating conda env..."
time source conda/bin/activate
echo $?

popd

# Install pre-downloaded pypi wheels
PYPI_DIR="${PYPI_DIR:-downloaded_pypi}"
echo "Pip installing deps from ${PYPI_DIR}..."
time pip install ${PYPI_DIR}/*whl

LIBCUDA=/usr/local/fbcode/platform010/lib/libcuda.so
LIBNVIDIA_ML=/usr/local/fbcode/platform010/lib/libnvidia-ml.so

# Add conda libs to LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$(pwd)/conda/conda/lib:${LD_LIBRARY_PATH}
# Add libcuda.so from /usr/local/fbcode
export LD_LIBRARY_PATH=${LIBCUDA}:$LD_LIBRARY_PATH
echo ${LD_LIBRARY_PATH}

export LD_PRELOAD="${LIBCUDA}:${LIBNVIDIA_ML}"
echo ${LD_PRELOAD}

# Executing tests
mkdir -p build/custom_test_artifacts
TEST_CONFIG="genai" BUILD_ENVIRONMENT="sandcastle" .ci/pytorch/test.sh
