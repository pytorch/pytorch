#!/bin/bash
set -ex
wget https://www.openssl.org/source/openssl-1.1.1k.tar.gz
tar xf openssl-1.1.1k.tar.gz
(cd openssl-1.1.1k && ./config --prefix="$PYTHON_INSTALL_DIR" && make -j32 && make install)
CFLAGS=-fPIC CPPFLAGS=-fPIC ./configure --prefix "$PYTHON_INSTALL_DIR" --with-openssl="$PYTHON_INSTALL_DIR"
