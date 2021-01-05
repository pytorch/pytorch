#!/bin/bash

set -ex

# Install sccache from pre-compiled binary.
curl https://s3.amazonaws.com/ossci-linux/sccache -o /usr/local/bin/sccache
chmod a+x /usr/local/bin/sccache

# Setup SCCACHE
###############################################################################
SCCACHE="$(which sccache)"
if [ -z "${SCCACHE}" ]; then
  echo "Unable to find sccache..."
  exit 1
fi

if [[ "${BUILD_ENVIRONMENT}" == *-rocm* ]]; then
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
