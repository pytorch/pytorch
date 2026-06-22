#!/bin/bash
set -xe
# Script used in Linux x86 and aarch64 CD pipeline

# Workaround for exposing statically linked libstdc++ CXX11 ABI symbols.
# see: https://github.com/pytorch/pytorch/issues/133437
LIBNONSHARED=$(gcc -print-file-name=libstdc++_nonshared.a)
nm -g $LIBNONSHARED | grep " T " | grep recursive_directory_iterator | cut -c 20-  > weaken-symbols.txt
objcopy --weaken-symbols weaken-symbols.txt $LIBNONSHARED $LIBNONSHARED
