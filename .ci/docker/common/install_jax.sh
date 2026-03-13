#!/bin/bash

set -ex

source "$(dirname "${BASH_SOURCE[0]}")/common_utils.sh"

# Get the pinned JAX version (same for all CUDA versions)
JAX_VERSION=$(get_pinned_commit /ci_commit_pins/jax)

function install_jax_cpu() {
  echo "Installing JAX ${JAX_VERSION} (CPU only)"
  pip_install "jax[cpu]==${JAX_VERSION}"

  # Verify installation
  python -c "import jax"  # check for errors
  echo "JAX ${JAX_VERSION} installation completed successfully (CPU only)"
}

function install_jax_12() {
  echo "Installing JAX ${JAX_VERSION} with CUDA 12 support"
  pip_install "jax[cuda12]==${JAX_VERSION}" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

  # Verify installation
  python -c "import jax"  # check for errors
  echo "JAX ${JAX_VERSION} installation completed successfully for CUDA 12"
}

function install_jax_13() {
  echo "Installing JAX ${JAX_VERSION} with CUDA 13 support"
  pip_install "jax[cuda13]==${JAX_VERSION}" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

  # Verify installation
  python -c "import jax"  # check for errors
  echo "JAX ${JAX_VERSION} installation completed successfully for CUDA 13"
}

function install_jax_tpu() {
  echo "Installing JAX ${JAX_VERSION} with TPU support"
  pip_install "jax[tpu]==${JAX_VERSION}" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html

  # Verify installation
  python -c "import jax"  # check for errors
  echo "JAX ${JAX_VERSION} installation completed successfully for TPU"
}

# idiomatic parameter and option handling in sh
while test $# -gt 0
do
    case "$1" in
    cpu) install_jax_cpu;
        ;;
    tpu) install_jax_tpu;
        ;;
    12.4|12.6|12.6.*|12.8|12.8.*|12.9|12.9.*) install_jax_12;
        ;;
    13.0|13.0.*) install_jax_13;
        ;;
    *) echo "bad argument $1"; exit 1
        ;;
    esac
    shift
done
