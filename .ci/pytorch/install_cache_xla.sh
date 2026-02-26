#!/bin/bash

# Script for installing sccache on the xla build job, which uses xla's docker
# image, which has sccache installed but doesn't write the stubs.  This is
# mostly copied from .ci/docker/install_cache.sh.  Changes are: removing checks
# that will always return the same thing, ex checks for for rocm, CUDA, changing
# the path where sccache is installed, not changing /etc/environment, and not
# installing/downloading sccache as it is already in the docker image.

set -ex -o pipefail

mkdir -p /tmp/cache/bin
export PATH="/tmp/cache/bin:$PATH"

function write_sccache_stub() {
  # Unset LD_PRELOAD for ps because of asan + ps issues
  # https://gcc.gnu.org/bugzilla/show_bug.cgi?id=90589
  if [ "$1" == "gcc" ]; then
    # Do not call sccache recursively when dumping preprocessor argument
    # For some reason it's very important for the first cached nvcc invocation
    cat >"/tmp/cache/bin/$1" <<EOF
#!/bin/sh

# sccache does not support -E flag, so we need to call the original compiler directly in order to avoid calling this wrapper recursively
for arg in "\$@"; do
  if [ "\$arg" = "-E" ]; then
    exec $(which "$1") "\$@"
  fi
done

if [ \$(env -u LD_PRELOAD ps -p \$PPID -o comm=) != sccache ]; then
  exec sccache $(which "$1") "\$@"
else
  exec $(which "$1") "\$@"
fi
EOF
  else
    cat >"/tmp/cache/bin/$1" <<EOF
#!/bin/sh

if [ \$(env -u LD_PRELOAD ps -p \$PPID -o comm=) != sccache ]; then
  exec sccache $(which "$1") "\$@"
else
  exec $(which "$1") "\$@"
fi
EOF
  fi
  chmod a+x "/tmp/cache/bin/$1"
}

write_sccache_stub cc
write_sccache_stub c++
write_sccache_stub gcc
write_sccache_stub g++
write_sccache_stub clang
write_sccache_stub clang++
