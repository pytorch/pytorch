#!/bin/bash

set -ex

mkdir -p /opt/cache/bin
mkdir -p /opt/cache/lib
sed -e 's|PATH="\(.*\)"|PATH="/opt/cache/bin:\1"|g' -i /etc/environment
export PATH="/opt/cache/bin:$PATH"

# Setup compiler cache
curl --retry 3 https://s3.amazonaws.com/ossci-linux/sccache -o /opt/cache/bin/sccache
chmod a+x /opt/cache/bin/sccache

function write_sccache_stub() {
  printf "#!/bin/sh\nexec sccache $(which $1) \$*" > "/opt/cache/bin/$1"
  chmod a+x "/opt/cache/bin/$1"
}

write_sccache_stub cc
write_sccache_stub c++
write_sccache_stub gcc
write_sccache_stub g++
write_sccache_stub clang
write_sccache_stub clang++

if [ -n "$CUDA_VERSION" ]; then
  # TODO: This is a workaround for the fact that PyTorch's FindCUDA
  # implementation cannot find nvcc if it is setup this way, because it
  # appears to search for the nvcc in PATH, and use its path to infer
  # where CUDA is installed.  Instead, we install an nvcc symlink outside
  # of the PATH, and set CUDA_NVCC_EXECUTABLE so that we make use of it.

  printf "#!/bin/sh\nexec sccache $(which nvcc) \"\$@\"" > /opt/cache/lib/nvcc
  chmod a+x /opt/cache/lib/nvcc
fi
