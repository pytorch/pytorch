#!/bin/bash
# Script used only in CD pipeline

set -eux

ACL_VERSION=${ACL_VERSION:-"v52.6.0"}
ACL_INSTALL_DIR="/acl"

# Optional ccache support
USE_CCACHE=${USE_CCACHE:-0}
SCONS_CACHE_ARG=""
if [ "${USE_CCACHE}" != "0" ]; then
  if command -v ccache >/dev/null 2>&1; then
    echo "Using ccache for ACL build"
    SCONS_CACHE_ARG="compiler_cache=ccache"
  else
    echo "USE_CCACHE is set but 'ccache' is not available on PATH; proceeding without ccache"
  fi
fi

# Clone ACL
git clone https://github.com/ARM-software/ComputeLibrary.git -b "${ACL_VERSION}" --depth 1 --shallow-submodules

ACL_CHECKOUT_DIR="ComputeLibrary"
# Build with scons
pushd $ACL_CHECKOUT_DIR
scons -j8  Werror=0 debug=0 neon=1 opencl=0 embed_kernels=0 \
  os=linux arch=armv8a build=native multi_isa=1 \
  fixed_format_kernels=1 openmp=1 cppthreads=0 \
  ${SCONS_CACHE_ARG}
popd

# Install ACL
sudo mkdir -p ${ACL_INSTALL_DIR}
for d in arm_compute include utils support src build
do
  sudo cp -r ${ACL_CHECKOUT_DIR}/${d} ${ACL_INSTALL_DIR}/${d}
done

rm -rf $ACL_CHECKOUT_DIR
