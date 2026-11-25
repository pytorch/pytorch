#!/bin/bash
# MoRI (https://github.com/ROCm/mori) backs the TokenSwitchMoRI tests on MI300
# and MI350. Single-node only: no NIC provider (USE_IONIC/USE_BNXT) is built.
# Its flydsl EP backend uses the flydsl that .ci/pytorch/test.sh installs for
# the ROCm distributed tests.

set -ex

source /etc/rocm_env.sh
source "$(dirname "${BASH_SOURCE[0]}")/common_utils.sh"

apt-get update
apt-get install -y --no-install-recommends libibverbs-dev libdrm-dev
apt-get clean
rm -rf /var/lib/apt/lists/*

mori_commit=$(get_pinned_commit rocm-mori)
tmp_dir=$(mktemp -d)
chown jenkins:jenkins "${tmp_dir}"
pushd "${tmp_dir}"
as_jenkins git clone https://github.com/ROCm/mori.git
cd mori
as_jenkins git checkout "${mori_commit}"
# The default build needs only these two of MoRI's submodules.
as_jenkins git submodule update --init --recursive 3rdparty/spdlog 3rdparty/msgpack-c
# Build requirements missing from requirements-ci.txt, pinned so they keep the
# image's packaging pin. Without Cython the build only warns and drops
# mori.cco.cco, which carries the flydsl EP backend.
pip_install "Cython==3.1.6" "setuptools-scm==8.3.1" "pybind11==2.13.6" "wheel==0.45.1"
# Kernels are JIT-compiled on first use, so the build needs no GPU or arch list.
MORI_SKIP_PRECOMPILE=1 BUILD_TORCH_SYMM=OFF pip_install --no-build-isolation .
popd
rm -rf "${tmp_dir}"
# tools/verify_install.py checks the same extension but imports torch, which
# the image does not have yet.
python -c "import mori.cco.cco"
