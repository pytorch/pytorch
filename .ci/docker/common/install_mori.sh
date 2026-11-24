#!/bin/bash
# MoRI (https://github.com/ROCm/mori) backs the TokenSwitchMoRI and FSDP
# all-gather layout tests on MI300 and MI350. Single-node only: no NIC
# provider (USE_IONIC/USE_BNXT) is built. Its flydsl EP backend uses the flydsl
# that .ci/pytorch/test.sh installs for the ROCm distributed tests.

set -ex

source /etc/rocm_env.sh
source "$(dirname "${BASH_SOURCE[0]}")/common_utils.sh"

apt-get update
apt-get install -y --no-install-recommends libibverbs-dev libdrm-dev
apt-get clean
rm -rf /var/lib/apt/lists/*

mori_commit=$(get_pinned_commit mori)
tmp_dir=$(mktemp -d)
chown jenkins:jenkins "${tmp_dir}"
pushd "${tmp_dir}"
as_jenkins git clone --recursive https://github.com/ROCm/mori.git
cd mori
as_jenkins git checkout "${mori_commit}"
as_jenkins git submodule update --init --recursive
# Kernels are JIT-compiled on first use, so the build needs no GPU or arch list.
pip_install -r requirements-build.txt
MORI_SKIP_PRECOMPILE=1 BUILD_TORCH_SYMM=OFF pip_install --no-build-isolation .
popd
rm -rf "${tmp_dir}"
# MoRI only warns when Cython is missing and then drops this extension, which
# carries the flydsl EP backend. tools/verify_install.py checks the same thing
# but imports torch, which the image does not have yet.
env_run python -c "import mori.cco.cco"
