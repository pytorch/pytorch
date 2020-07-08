#!/bin/bash
shopt -s extglob

ORIG_COMP=/opt/rocm/hcc/bin/clang-*_original
# note that the wrapping always names the compiler "clang-7.0_original"
if [ -e $ORIG_COMP ]; then
   WRAPPED=/opt/rocm/hcc/bin/clang-?([0-9])?([0-9])[0-9]
   sudo mv $ORIG_COMP $WRAPPED
fi
