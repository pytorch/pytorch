#!/bin/bash
# Script used only in CD pipeline

set -ex

# install dependencies
dnf -y install gmp-devel libmpc-devel texinfo flex bison

cd /usr/local/src
# fetch source for gcc 13
git clone --depth 1 --single-branch -b releases/gcc-13.3.0 https://github.com/gcc-mirror/gcc.git gcc-13.3.0

mkdir -p gcc-13.3.0/build-gomp
cd gcc-13.3.0/build-gomp

# configure gcc build
# I got these flags by:
# 1. downloading the source rpm for gcc-11 on AlmaLinux 8 container
#    dnf install -y dnf-plugins-core rpmdevtools
#   dnf download --source libgomp
# 2. extracting the gcc.spec from the source.
#    rpmdev-extract gcc-xx.src.rpm
# 3. extracting optflags and ld_flags from gcc.spec:
#    rpm --eval '%{optflags}'
#    rpm --eval '%{build_ldflags}'
#
# I had to remove the following flags because they didn't compile for this version of libgomp:
#   -Werror=format-security
#   -specs=/usr/lib/rpm/redhat/redhat-hardened-cc1
#   -specs=/usr/lib/rpm/redhat/redhat-annobin-cc1
#
# I added -march=armv8-a -mtune=generic to make them explicit. I don't think they're strictly needed.

OPT_FLAGS='-O2 -march=armv8-a -mtune=generic'\
' -fexceptions -g -grecord-gcc-switches -pipe -Wall'\
' -Wp,-D_FORTIFY_SOURCE=2 -Wp,-D_GLIBCXX_ASSERTIONS'\
' -fstack-protector-strong -fasynchronous-unwind-tables'\
' -fstack-clash-protection'

LDFLAGS='-Wl,-z,relro -Wl,--as-needed -Wl,-z,now'

CFLAGS="$OPT_FLAGS" \
CXXFLAGS="$OPT_FLAGS" \
LDFLAGS="$LDFLAGS" \
../configure \
  --prefix=/usr \
  --libdir=/usr/lib64 \
  --enable-languages=c,c++ \
  --disable-multilib \
  --disable-bootstrap \
  --enable-libgomp

# only build libgomp
make -j$(nproc) all-target-libgomp

make install-target-libgomp