#!/bin/bash
# Script used only in CD pipeline

set -eux

ACL_VERSION=${ACL_VERSION:-"451a2d05110892781715ebdbfb9b08d86733376d"}
ACL_INSTALL_DIR="/acl"
ACL_REPO="https://github.com/ARM-software/ComputeLibrary.git"
ACL_CHECKOUT_DIR="ComputeLibrary"

# Clone ACL. ACL_VERSION may be a tag, branch, or commit SHA.
git init --quiet "${ACL_CHECKOUT_DIR}"
git -C "${ACL_CHECKOUT_DIR}" remote add origin "${ACL_REPO}"
git -C "${ACL_CHECKOUT_DIR}" fetch --depth 1 origin "${ACL_VERSION}"
git -C "${ACL_CHECKOUT_DIR}" checkout --detach FETCH_HEAD

# Build with scons
pushd $ACL_CHECKOUT_DIR
scons -j8  Werror=0 debug=0 neon=1 opencl=0 embed_kernels=0 \
  os=linux arch=armv8a build=native multi_isa=1 \
  fixed_format_kernels=1 openmp=1 cppthreads=0
popd

# Install ACL
sudo mkdir -p ${ACL_INSTALL_DIR}
for d in arm_compute include utils support src build
do
  sudo cp -r ${ACL_CHECKOUT_DIR}/${d} ${ACL_INSTALL_DIR}/${d}
done

rm -rf $ACL_CHECKOUT_DIR
