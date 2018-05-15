#!/bin/bash

set -ex

mkdir -p build
cd build
cmake ../
make all test
