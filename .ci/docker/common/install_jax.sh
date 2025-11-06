#!/bin/bash

set -ex

# Get the pinned JAX version
JAX_VERSION=$(cat /ci_commit_pins/jax.txt)

function install_jax_12() {
  echo "Installing JAX ${JAX_VERSION} with CUDA 12 support"
  pip install --progress-bar off "jax[cuda12]==${JAX_VERSION}" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

  # Verify installation
  python -c "import jax; print('JAX version:', jax.__version__)"
  echo "JAX ${JAX_VERSION} installation completed successfully for CUDA 12"
}

function install_jax_13() {
  echo "Installing JAX ${JAX_VERSION} with CUDA 13 support"
  # JAX may not have separate cuda13 packages yet, use cuda12
  pip install --progress-bar off "jax[cuda12]==${JAX_VERSION}" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

  # Verify installation
  python -c "import jax; print('JAX version:', jax.__version__)"
  echo "JAX ${JAX_VERSION} installation completed successfully for CUDA 13"
}

# idiomatic parameter and option handling in sh
while test $# -gt 0
do
    case "$1" in
    12.4|12.6|12.6.*|12.8|12.8.*|12.9|12.9.*) install_jax_12;
        ;;
    13.0|13.0.*) install_jax_13;
        ;;
    *) echo "bad argument $1"; exit 1
        ;;
    esac
    shift
done
