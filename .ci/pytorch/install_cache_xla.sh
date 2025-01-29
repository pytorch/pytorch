#!/bin/bash

# Script for installing sccache on the xla build job, which uses xla's docker
# image and doesn't have sccache installed on it.  This is mostly copied from
# .ci/docker/install_cache.sh.  Changes are: removing checks that will always
# return the same thing, ex checks for for rocm, CUDA, and changing the path
# where sccache is installed, and not changing /etc/environment.

set -ex -o pipefail

install_binary() {
  echo "Downloading sccache binary from S3 repo"
  curl --retry 3 https://s3.amazonaws.com/ossci-linux/sccache -o /tmp/cache/bin/sccache
}

mkdir -p /tmp/cache/bin
mkdir -p /tmp/cache/lib
export PATH="/tmp/cache/bin:$PATH"

install_binary
chmod a+x /tmp/cache/bin/sccache

function write_sccache_stub() {
  # Unset LD_PRELOAD for ps because of asan + ps issues
  # https://gcc.gnu.org/bugzilla/show_bug.cgi?id=90589
  # shellcheck disable=SC2086
  # shellcheck disable=SC2059
  printf "#!/bin/sh\nif [ \$(env -u LD_PRELOAD ps -p \$PPID -o comm=) != sccache ]; then\n  exec sccache $(which $1) \"\$@\"\nelse\n  exec $(which $1) \"\$@\"\nfi" > "/tmp/cache/bin/$1"
  chmod a+x "/tmp/cache/bin/$1"
}

write_sccache_stub cc
write_sccache_stub c++
write_sccache_stub gcc
write_sccache_stub g++
write_sccache_stub clang
write_sccache_stub clang++
