#!/bin/bash
set -eux -o pipefail

source "/Users/distiller/project/env"
export "PATH=$workdir/miniconda/bin:$PATH"
pkg="$workdir/final_pkgs/$(ls $workdir/final_pkgs)"

# Don't test libtorch
# TODO we should test libtorch
if [[ "$PACKAGE_TYPE" == libtorch ]]; then
  exit 0
fi

# Create a new test env
# TODO cut all this out into a separate test job and have an entirely different
# miniconda
source deactivate || true
conda create -qyn test python="$DESIRED_PYTHON"
source activate test >/dev/null

# Install the package
if [[ "$PACKAGE_TYPE" == conda ]]; then
  conda install -y "$pkg" --offline
else
  pip install "$pkg" --no-index --no-dependencies -v
fi

# Test
pushd "$workdir/pytorch"
$workdir/builder/run_tests.sh "$PACKAGE_TYPE" "$DESIRED_PYTHON" "$DESIRED_CUDA"
