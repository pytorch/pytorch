#!/bin/bash

set -ex

mkdir -p /opt/cache/bin
mkdir -p /opt/cache/lib
sed -e 's|PATH="\(.*\)"|PATH="/opt/cache/bin:\1"|g' -i /etc/environment
export PATH="/opt/cache/bin:$PATH"

# Setup compiler cache
if [ -n "$ROCM_VERSION" ]; then
  curl --retry 3 http://repo.radeon.com/misc/.sccache_amd/sccache -o /opt/cache/bin/sccache
else
  curl --retry 3 https://s3.amazonaws.com/ossci-linux/sccache -o /opt/cache/bin/sccache
fi
chmod a+x /opt/cache/bin/sccache

function write_sccache_stub() {
  printf "#!/bin/sh\nexec sccache $(which $1) \$*" > "/opt/cache/bin/$1"
  chmod a+x "/opt/cache/bin/$1"
}

write_sccache_stub cc
write_sccache_stub c++
write_sccache_stub gcc
write_sccache_stub g++

# NOTE: See specific ROCM_VERSION case below.
if [ "x$ROCM_VERSION" = x ]; then
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

if [ -n "$ROCM_VERSION" ]; then
  # ROCm compiler is hcc or clang. However, it is commonly invoked via hipcc wrapper.
  # hipcc will call either hcc or clang using an absolute path starting with /opt/rocm,
  # causing the /opt/cache/bin to be skipped. We must create the sccache wrappers
  # directly under /opt/rocm while also preserving the original compiler names.
  # Note symlinks will chain as follows: [hcc or clang++] -> clang -> clang-??
  # Final link in symlink chain must point back to original directory.

  # Original compiler is moved one directory deeper. Wrapper replaces it.
  function write_sccache_stub_rocm() {
    OLDCOMP=$1
    COMPNAME=$(basename $OLDCOMP)
    TOPDIR=$(dirname $OLDCOMP)
    WRAPPED="$TOPDIR/original/$COMPNAME"
    mv "$OLDCOMP" "$WRAPPED"
    printf "#!/bin/sh\nexec sccache $WRAPPED \$*" > "$OLDCOMP"
    chmod a+x "$1"
  }

  if [[ -e "/opt/rocm/hcc/bin/hcc" ]]; then
    # ROCm 3.3 or earlier.
    mkdir /opt/rocm/hcc/bin/original
    write_sccache_stub_rocm /opt/rocm/hcc/bin/hcc
    write_sccache_stub_rocm /opt/rocm/hcc/bin/clang
    write_sccache_stub_rocm /opt/rocm/hcc/bin/clang++
    # Fix last link in symlink chain, clang points to versioned clang in prior dir
    pushd /opt/rocm/hcc/bin/original
    ln -s ../$(readlink clang)
    popd
  elif [[ -e "/opt/rocm/llvm/bin/clang" ]]; then
    # ROCm 3.5 and beyond.
    mkdir /opt/rocm/llvm/bin/original
    write_sccache_stub_rocm /opt/rocm/llvm/bin/clang
    write_sccache_stub_rocm /opt/rocm/llvm/bin/clang++
    # Fix last link in symlink chain, clang points to versioned clang in prior dir
    pushd /opt/rocm/llvm/bin/original
    ln -s ../$(readlink clang)
    popd
  else
    echo "Cannot find ROCm compiler."
    exit 1
  fi
fi
