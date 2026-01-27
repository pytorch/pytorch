#!/bin/bash

set -ex

# Pinned version of doxygen to install
# This is decoupled from the Ubuntu version to ensure consistent behavior
DOXYGEN_VERSION=${DOXYGEN_VERSION:-1.11.0}

install_doxygen_binary() {
  # Install doxygen from pre-built binary (similar to install_ninja.sh pattern)
  # Note: Requires GLIBCXX_3.4.30+ (available in Ubuntu 24.04+ / noble)
  pushd /tmp
  curl --retry 3 -fsSL "https://github.com/doxygen/doxygen/releases/download/Release_${DOXYGEN_VERSION//./_}/doxygen-${DOXYGEN_VERSION}.linux.bin.tar.gz" -o doxygen.tar.gz
  tar -xzf doxygen.tar.gz
  cp doxygen-${DOXYGEN_VERSION}/bin/doxygen /usr/local/bin/
  rm -rf doxygen-${DOXYGEN_VERSION} doxygen.tar.gz
  popd
}

install_doxygen_source() {
  # Build doxygen from source (fallback for older systems without GLIBCXX_3.4.30)
  apt-get install -y --no-install-recommends cmake flex bison
  pushd /tmp
  curl --retry 3 -fsSL "https://github.com/doxygen/doxygen/releases/download/Release_${DOXYGEN_VERSION//./_}/doxygen-${DOXYGEN_VERSION}.src.tar.gz" -o doxygen.tar.gz
  tar -xzf doxygen.tar.gz
  cd doxygen-${DOXYGEN_VERSION}
  cmake -G "Unix Makefiles" .
  make -j"$(nproc)"
  make install
  cd ..
  rm -rf doxygen-${DOXYGEN_VERSION} doxygen.tar.gz
  popd
}

install_doxygen() {
  # Try pre-built binary first, fall back to source build if it fails
  install_doxygen_binary
  if ! doxygen --version > /dev/null 2>&1; then
    echo "Pre-built doxygen binary failed (likely GLIBCXX issue), building from source..."
    rm -f /usr/local/bin/doxygen
    install_doxygen_source
  fi
  echo "Installed doxygen version: $(doxygen --version)"
}

if [ -n "$KATEX" ]; then
  apt-get update
  # Ignore error if gpg-agent doesn't exist (for Ubuntu 16.04)
  apt-get install -y gpg-agent || :

  curl --retry 3 -sL https://deb.nodesource.com/setup_16.x | sudo -E bash -
  sudo apt-get install -y nodejs

  curl --retry 3 -sS https://dl.yarnpkg.com/debian/pubkey.gpg | sudo apt-key add -
  echo "deb https://dl.yarnpkg.com/debian/ stable main" | sudo tee /etc/apt/sources.list.d/yarn.list

  apt-get update
  apt-get install -y --no-install-recommends yarn
  yarn global add katex --prefix /usr/local

  install_doxygen

  apt-get autoclean && apt-get clean
  rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

fi
