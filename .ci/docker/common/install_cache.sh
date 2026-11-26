#!/bin/bash

set -ex

# Build sccache from source at v0.16.0 with the nvcc 13.3 dryrun-parsing fix
# backported from mozilla/sccache#2722 (see
# patches/sccache-nvcc-13.3-dryrun-parsing.patch). The prebuilt release binary
# mis-parses nvcc 13.3+ --dryrun output (it skips the device compile, leaving
# "fatbinary: Could not open input file '*.cubin'"), so build from source until
# a fixed sccache release ships. Reverts #189365 (prebuilt-binary download).
build_sccache_from_source() {
  local VERSION=0.16.0
  echo "Building sccache ${VERSION} from source with the nvcc 13.3 dryrun fix"
  # The builder images pre-install a pinned rust at CARGO_HOME=/opt/rust with its
  # bin on PATH (see #186302), so build with that toolchain directly instead of
  # (re)installing rustup and sourcing its env.
  apt-get update && apt-get install -y --no-install-recommends pkg-config libssl-dev curl git ca-certificates
  git clone --depth 1 --branch "v${VERSION}" https://github.com/mozilla/sccache /tmp/sccache
  local patch=/opt/cache/patches/sccache-nvcc-13.3-dryrun-parsing.patch
  [ -f "$patch" ] || { echo "ERROR: $patch missing; the Dockerfile must 'COPY ./common/patches /opt/cache/patches'"; exit 1; }
  git -C /tmp/sccache apply "$patch"
  cargo build --manifest-path /tmp/sccache/Cargo.toml --release --features="dist-client dist-server"
  cp /tmp/sccache/target/release/sccache /opt/cache/bin
  cp /tmp/sccache/target/release/sccache-dist /opt/cache/bin
  chmod a+x /opt/cache/bin/sccache /opt/cache/bin/sccache-dist
  rm -rf /tmp/sccache
}

install_ubuntu() {
  # riscv64 (built under QEMU emulation) has no nvcc, so it is unaffected by the
  # --simt-only bug; it also has no sccache-dist, and a from-source build under
  # emulation would be impractically slow. Use the prebuilt binary there and
  # build from source everywhere else.
  if [[ "$(uname -m)" == "riscv64" ]]; then
    local VERSION=0.16.0
    # Rust's riscv64 arch is riscv64gc; sccache-dist is not available on riscv64.
    echo "Downloading prebuilt sccache ${VERSION} for riscv64"
    curl --retry 3 -fsSL https://github.com/mozilla/sccache/releases/download/v${VERSION}/sccache-v${VERSION}-riscv64gc-unknown-linux-musl.tar.gz | \
      tar -xz -C /opt/cache/bin --strip-components=1 sccache-v${VERSION}-riscv64gc-unknown-linux-musl/sccache
    chmod a+x /opt/cache/bin/sccache
  else
    build_sccache_from_source
  fi

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
