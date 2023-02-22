#!/bin/bash

set -ex

source "$(dirname "${BASH_SOURCE[0]}")/common_utils.sh"

as_jenkins_with_env() {
  # Adding this to the generic as_jenkins bash function breaks conda installation somehow.
  # We need -E to keep the CXX environment variable when building triton here
  $SUDO -E -H -u jenkins env -u SUDO_UID -u SUDO_GID -u SUDO_COMMAND -u SUDO_USER env "PATH=$PATH" "LD_LIBRARY_PATH=$LD_LIBRARY_PATH" $*
}

pip_install_with_env() {
  as_jenkins_with_env conda run -n py_$ANACONDA_PYTHON_VERSION pip install --progress-bar off $*
}

# The logic here is copied from .ci/pytorch/common_utils.sh
TRITON_PINNED_COMMIT=$(get_pinned_commit triton)

apt update
apt-get install -y gpg-agent

if [ -n "${GCC_VERSION}" ] && [[ "${GCC_VERSION}" == "7" ]]; then
  # Triton needs at least gcc-9 to build
  apt-get install -y g++-9

  CXX=g++-9 pip_install_with_env "git+https://github.com/openai/triton@${TRITON_PINNED_COMMIT}#subdirectory=python"
elif [ -n "${CLANG_VERSION}" ]; then
  # Triton needs <filesystem> which surprisingly is not available with clang-9 toolchain
  add-apt-repository -y ppa:ubuntu-toolchain-r/test
  apt-get install -y g++-9

  CXX=g++-9 pip_install_with_env "git+https://github.com/openai/triton@${TRITON_PINNED_COMMIT}#subdirectory=python"
else
  pip_install "git+https://github.com/openai/triton@${TRITON_PINNED_COMMIT}#subdirectory=python"
fi
