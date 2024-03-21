set -euo pipefail

readonly version=v23.08
readonly src_host=https://review.mlplatform.org/ml
readonly src_repo=ComputeLibrary

# Clone ACL
[[ ! -d ${src_repo} ]] && git clone ${src_host}/${src_repo}.git
cd ${src_repo}

git checkout $version

# Build with scons
scons -j8  Werror=0 debug=0 neon=1 opencl=0 embed_kernels=0 \
  os=linux arch=armv8a build=native multi_isa=1 \
  fixed_format_kernels=1 openmp=1 cppthreads=0
