#!/bin/bash

set -ex

install_ubuntu() {
  echo "Installing pkg-config and libssl-dev"
  apt-get update && apt-get install -y pkg-config libssl-dev curl
  echo "Installing rust"
  curl https://sh.rustup.rs -sSf | sh -s -- -y
  echo "Checking out sccache repo"
  git clone https://github.com/mozilla/sccache -b v0.13.0
  cd sccache
  echo "Patch dist build on aarch64"
  sed -i '/all(target_os = "linux", target_arch = "x86_64"),/{ p; s/x86_64/aarch64/; }' src/bin/sccache-dist/main.rs
  echo "Building sccache"
  . "$HOME/.cargo/env" && cargo build --release --features="dist-client dist-server"
  cp target/release/sccache /opt/cache/bin
  cp target/release/sccache-dist /opt/cache/bin
  echo "Cleaning up"
  cd ..
  rm -rf sccache .cargo
  apt-get remove -y pkg-config libssl-dev
  apt-get autoclean && apt-get clean

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
chmod a+x /opt/cache/bin/sccache

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

# Skip all sccache wrapping for theRock nightly: sccache PATH wrappers
# intercept assembly (.s) compilation and fail because the assembler does not
# produce the .d dependency file that sccache expects.
if [ "$ROCM_VERSION" != "nightly" ]; then
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
  # Skip sccache wrapping for theRock nightly - sccache has issues parsing
  # theRock's complex include paths and causes hipconfig to fail
  if [ "$ROCM_VERSION" = "nightly" ]; then
    echo "Skipping sccache wrapping for theRock nightly ROCm"
  else
    source /etc/rocm_env.sh

    # ROCm compiler is hcc or clang. However, it is commonly invoked via hipcc wrapper.
    # hipcc will call either hcc or clang using an absolute path starting with $ROCM_PATH,
    # causing the /opt/cache/bin to be skipped. We must create the sccache wrappers
    # directly under $ROCM_PATH while also preserving the original compiler names.
    # Note symlinks will chain as follows: [hcc or clang++] -> clang -> clang-??
    # Final link in symlink chain must point back to original directory.

    # Original compiler is moved one directory deeper. Wrapper replaces it.
    function write_sccache_stub_rocm() {
      OLDCOMP=$1
      COMPNAME=$(basename $OLDCOMP)
      TOPDIR=$(dirname $OLDCOMP)
      WRAPPED="$TOPDIR/original/$COMPNAME"
      mv "$OLDCOMP" "$WRAPPED"
      printf "#!/bin/sh\nexec sccache $WRAPPED \"\$@\"" >"$OLDCOMP"
      chmod a+x "$OLDCOMP"
    }

    # ROCm 3.5 and beyond use llvm/bin/clang
    if [[ -e "${ROCM_PATH}/llvm/bin/clang" ]]; then
      mkdir ${ROCM_PATH}/llvm/bin/original
      write_sccache_stub_rocm ${ROCM_PATH}/llvm/bin/clang
      write_sccache_stub_rocm ${ROCM_PATH}/llvm/bin/clang++
      # Fix last link in symlink chain for traditional ROCm where clang -> clang-17
      pushd ${ROCM_PATH}/llvm/bin/original
      if [[ -L clang ]] && [[ "$(readlink clang)" == clang-* ]]; then
        ln -s ../$(readlink clang)
      fi
      popd
    else
      echo "Cannot find ROCm compiler."
      exit 1
    fi
  fi
fi
