#!/bin/sh

set -e

curl -L https://github.com/redis/hiredis/archive/v0.13.3.tar.gz | tar zx

# Compile and install to third-party/hiredis
(
    cd ./hiredis-0.13.3
    PREFIX=$PWD/../hiredis make all install
)
