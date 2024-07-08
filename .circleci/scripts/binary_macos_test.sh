#!/bin/bash
set -eux -o pipefail

source "/Users/distiller/project/env"
export "PATH=$workdir/miniconda/bin:$PATH"
pkg="$workdir/final_pkgs/$(ls $workdir/final_pkgs)"

# Create a new test env
# TODO cut all this out into a separate test job and have an entirely different
# miniconda
if [[ "$PACKAGE_TYPE" != libtorch ]]; then
  source deactivate || true
  conda create -qyn test python="$DESIRED_PYTHON"
  source activate test >/dev/null
fi

# Install the package
if [[ "$PACKAGE_TYPE" == libtorch ]]; then
  pkg="$(ls $workdir/final_pkgs/*-latest.zip)"
  unzip "$pkg" -d /tmp
  cd /tmp/libtorch
elif [[ "$PACKAGE_TYPE" == conda ]]; then
  conda install -y "$pkg"
else
  pip install "$pkg" -v
fi

# Test
if [[ "$PACKAGE_TYPE" == libtorch ]]; then
  $workdir/builder/check_binary.sh
else
  pushd "$workdir/pytorch"
  $workdir/builder/run_tests.sh "$PACKAGE_TYPE" "$DESIRED_PYTHON" "$DESIRED_CUDA"
fi
