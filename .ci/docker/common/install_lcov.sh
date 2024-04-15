#!/bin/bash

set -ex

git clone --branch v1.15 https://github.com/linux-test-project/lcov.git
pushd lcov
sudo make install   # will be installed in /usr/local/bin/lcov
popd
