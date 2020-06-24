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
# do not wrap clang for rocm images, see note below
if [[ "${BUILD_ENVIRONMENT}" != *-rocm* ]]; then
  write_sccache_stub clang
  write_sccache_stub clang++
fi

if [ -n "$CUDA_VERSION" ]; then
  # TODO: This is a workaround for the fact that PyTorch's FindCUDA
  # implementation cannot find nvcc if it is setup this way, because it
  # appears to search for the nvcc in PATH, and use its path to infer
  # where CUDA is installed.  Instead, we install an nvcc symlink outside
  # of the PATH, and set CUDA_NVCC_EXECUTABLE so that we make use of it.

  printf "#!/bin/sh\nexec sccache $(which nvcc) \"\$@\"" > /opt/cache/lib/nvcc
  chmod a+x /opt/cache/lib/nvcc
fi

# ROCm compiler is clang. However, it is commonly invoked via hipcc wrapper.
# hipcc will call either hcc or clang using an absolute path starting with /opt/rocm,
# causing the /opt/cache/bin to be skipped. We must create the sccache wrappers
# directly under /opt/rocm.
if [[ "${BUILD_ENVIRONMENT}" == *-rocm* ]]; then
  if [[ -e "/opt/rocm/hcc/bin/hcc" ]]; then
    HIPCOM_DEST_PATH="$(readlink -f /opt/rocm/hcc/bin/hcc )"
  else
    HIPCOM_DEST_PATH="$(readlink -f /opt/rocm/llvm/bin/clang )"
  fi
  HIPCOM_REAL_BINARY="$(dirname $HIPCOM_DEST_PATH)/hipcompiler_original"
  mv "$HIPCOM_DEST_PATH" "$HIPCOM_REAL_BINARY"

  # Create sccache wrapper.
  printf "#!/bin/sh\nexec sccache $HIPCOM_REAL_BINARY \$*" > "$HIPCOM_DEST_PATH"
  chmod a+x "$HIPCOM_DEST_PATH"
fi
