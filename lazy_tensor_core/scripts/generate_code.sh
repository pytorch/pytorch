#!/bin/bash

CDIR="$(cd "$(dirname "$0")" ; pwd -P)"
XDIR="$CDIR/.."
PTDIR="$XDIR/.."
if [ -z "$PT_INC_DIR" ]; then
  PT_INC_DIR="$PTDIR/build/aten/src/ATen"
fi

set -e
pushd $PTDIR
python -m tools.codegen.gen_lazy_tensor \
  --output_dir="$XDIR/lazy_tensor_core/csrc/ts_backend" \
  --source_yaml="$XDIR/ts_native_functions.yaml"\
  --impl_path="$XDIR/lazy_tensor_core/csrc/ts_backend/aten_ltc_ts_type.cpp"\

popd
