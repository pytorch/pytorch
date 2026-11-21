#!/bin/bash

set -ex

install_ubuntu() {
  local VERSION=0.18.0
  local ARCH
  ARCH="$(uname -m)"
  if [[ "$ARCH" == "riscv64" ]]; then
    ARCH="riscv64gc"
  fi

  local FEATURES=(sccache)
  if [[ "$ARCH" == "x86_64" || "$ARCH" == "aarch64" ]]; then
    FEATURES+=(sccache-dist)
  fi

  echo "Downloading sccache ${VERSION} binaries from GitHub releases"
  for feature in "${FEATURES[@]}"; do
    curl --retry 3 -fsSL https://github.com/mozilla/sccache/releases/download/v${VERSION}/${feature}-v${VERSION}-${ARCH}-unknown-linux-musl.tar.gz | \
      tar -xz -C /opt/cache/bin --strip-components=1 ${feature}-v${VERSION}-${ARCH}-unknown-linux-musl/${feature}
    chmod a+x /opt/cache/bin/${feature}
  done

  echo "Downloading old sccache binary from S3 repo for PCH builds"
  curl --retry 3 https://s3.amazonaws.com/ossci-linux/sccache -o /opt/cache/bin/sccache-0.2.14a
  chmod 755 /opt/cache/bin/sccache-0.2.14a
}

install_binary() {
  echo "Downloading sccache binary from S3 repo"
  curl --retry 3 https://s3.amazonaws.com/ossci-linux/sccache -o /opt/cache/bin/sccache
}

mkdir -p /opt/cache/bin
mkdir -p /opt/cache/lib
sed -e 's|PATH="\(.*\)"|PATH="/opt/cache/bin:\1"|g' -i /etc/environment
export PATH="/opt/cache/bin:$PATH"

# Setup compiler cache
install_ubuntu

function write_sccache_stub() {
  # Unset LD_PRELOAD for ps because of asan + ps issues
  # https://gcc.gnu.org/bugzilla/show_bug.cgi?id=90589
  if [ $1 == "gcc" ]; then
    # Do not call sccache recursively when dumping preprocessor argument
    # For some reason it's very important for the first cached nvcc invocation
    cat >"/opt/cache/bin/$1" <<EOF
#!/bin/sh

# sccache does not support -E flag, so we need to call the original compiler directly in order to avoid calling this wrapper recursively
for arg in "\$@"; do
  if [ "\$arg" = "-E" ]; then
    exec $(which $1) "\$@"
  fi
done

if [ \$(env -u LD_PRELOAD ps -p \$PPID -o comm=) != sccache ]; then
  exec sccache $(which $1) "\$@"
else
  exec $(which $1) "\$@"
fi
EOF
  else
    cat >"/opt/cache/bin/$1" <<EOF
#!/bin/sh

if [ \$(env -u LD_PRELOAD ps -p \$PPID -o comm=) != sccache ]; then
  exec sccache $(which $1) "\$@"
else
  exec $(which $1) "\$@"
fi
EOF
  fi
  chmod a+x "/opt/cache/bin/$1"
}

# Skip all sccache wrapping for TheRock ROCm: sccache PATH wrappers
# intercept assembly (.s) compilation and fail because the assembler does not
# produce the .d dependency file that sccache expects.
if [ -z "$ROCM_VERSION" ]; then
  write_sccache_stub cc
  write_sccache_stub c++
  write_sccache_stub gcc
  write_sccache_stub g++
fi

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

  write_sccache_stub nvcc
  mv /opt/cache/bin/nvcc /opt/cache/lib/
fi

if [ -n "$ROCM_VERSION" ]; then
  echo "Deferring TheRock ROCm compiler launchers to the PyTorch build"
fi
