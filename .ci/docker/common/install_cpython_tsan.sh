#!/bin/bash
# Build a TSan-instrumented CPython 3.14 free-threaded for use in CI testing.
# The resulting interpreter is installed to /opt/cpython-tsan/ and is used
# only at test time -- the build phase uses the normal conda Python.

set -ex

CPYTHON_VERSION=3.14.0
INSTALL_PREFIX=/opt/cpython-tsan

PYTHON_DOWNLOAD_URL="https://www.python.org/ftp/python/${CPYTHON_VERSION}/Python-${CPYTHON_VERSION}.tgz"

# Install build dependencies
apt-get update
apt-get install -y --no-install-recommends \
  build-essential libffi-dev libssl-dev zlib1g-dev libbz2-dev \
  libreadline-dev libsqlite3-dev libncurses5-dev libgdbm-dev libnss3-dev \
  liblzma-dev patchelf

pushd /tmp
wget -q "${PYTHON_DOWNLOAD_URL}" -O Python-${CPYTHON_VERSION}.tgz
tar xzf Python-${CPYTHON_VERSION}.tgz
cd Python-${CPYTHON_VERSION}

mkdir -p "${INSTALL_PREFIX}/lib"

CC=clang-18 CXX=clang++-18 \
  CFLAGS="-Wformat" \
  ./configure \
    --prefix="${INSTALL_PREFIX}" \
    --disable-gil \
    --with-thread-sanitizer \
    --enable-shared \
    --with-openssl=/opt/openssl \
    --with-openssl-rpath=auto

make -j"$(nproc)"
make install

# Fix rpath so the interpreter can find its own libpython
patchelf --set-rpath '${ORIGIN}/../lib' "${INSTALL_PREFIX}/bin/python3"

# Create convenience symlinks
ln -sf python3 "${INSTALL_PREFIX}/bin/python"
if [ -e "${INSTALL_PREFIX}/bin/pip3" ] && [ ! -e "${INSTALL_PREFIX}/bin/pip" ]; then
  ln -sf pip3 "${INSTALL_PREFIX}/bin/pip"
fi

# Install pip
"${INSTALL_PREFIX}/bin/python" -m ensurepip
"${INSTALL_PREFIX}/bin/python" -m pip install --upgrade pip setuptools wheel

popd

# Clean up build artifacts
rm -rf /tmp/Python-${CPYTHON_VERSION} /tmp/Python-${CPYTHON_VERSION}.tgz
apt-get clean
