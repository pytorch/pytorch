#!/bin/bash
# Script used only in CD pipeline
set -euo pipefail
set -x

# can be a named ref or SHA
ACL_VERSION=${ACL_VERSION:-"v52.6.0"}
NPROC=${NPROC:-$(( $(nproc) - 2 ))}

ACL_CHECKOUT_DIR="ComputeLibrary"
ACL_INSTALL_DIR="/acl"
ACL_REPO_URL="https://github.com/ARM-software/ComputeLibrary.git"

# Clone ACL
mkdir -p "$ACL_CHECKOUT_DIR"
(
  # shallow clone ACL_VERSION
  cd "$ACL_CHECKOUT_DIR"
  git init
  git remote add origin "$ACL_REPO_URL"
  git fetch --depth=1 --recurse-submodules=no origin "$ACL_VERSION"
  git checkout -f FETCH_HEAD

  # Build with scons
  scons -j${NPROC}  Werror=0 debug=0 neon=1 opencl=0 embed_kernels=0 \
    os=linux arch=armv8a build=native multi_isa=1 \
    fixed_format_kernels=1 openmp=1 cppthreads=0
)

# Install ACL
if [[ -d "${ACL_INSTALL_DIR}" ]]; then
  echo "Deleting existing install at ${ACL_INSTALL_DIR}"
  rm -rf "${ACL_INSTALL_DIR}"
fi
sudo mkdir -p ${ACL_INSTALL_DIR}
sudo cp -a "$ACL_CHECKOUT_DIR"/{arm_compute,include,utils,support,src,build} \
  "$ACL_INSTALL_DIR"/

# Clean up checkout
rm -rf $ACL_CHECKOUT_DIR