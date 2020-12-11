#!/usr/bin/env bash

# This script can also be used to test whether your diff changes any codegen output.
#
# Run it before and after your change:
#   .jenkins/pytorch/codegen-test.sh <baseline_output_dir>
#   .jenkins/pytorch/codegen-test.sh <test_output_dir>
#
# Then run diff to compare the generated files:
#   diff -Naur <baseline_output_dir> <test_output_dir>

set -eu -o pipefail

if [ "$#" -eq 0 ]; then
  # shellcheck disable=SC2034
  COMPACT_JOB_NAME="${BUILD_ENVIRONMENT}"
  source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
  OUT="$(dirname "${BASH_SOURCE[0]}")/../../codegen_result"
else
  OUT=$1
fi

set -x

rm -rf "$OUT"

# aten codegen
python -m tools.codegen.gen \
  -d "$OUT"/torch/share/ATen

# torch codegen
python -m tools.setup_helpers.generate_code \
  --declarations-path "$OUT"/torch/share/ATen/Declarations.yaml \
  --install_dir "$OUT"

# pyi codegen
mkdir -p "$OUT"/pyi/torch/_C
mkdir -p "$OUT"/pyi/torch/nn
python -m tools.pyi.gen_pyi \
  --native-functions-path aten/src/ATen/native/native_functions.yaml \
  --deprecated-functions-path tools/autograd/deprecated.yaml \
  --out "$OUT"/pyi

# autograd codegen (called by torch codegen but can run independently)
python -m tools.autograd.gen_autograd \
  "$OUT"/torch/share/ATen/Declarations.yaml \
  aten/src/ATen/native/native_functions.yaml \
  "$OUT"/autograd \
  tools/autograd

# annotated_fn_args codegen (called by torch codegen but can run independently)
mkdir -p "$OUT"/annotated_fn_args
python -m tools.autograd.gen_annotated_fn_args \
  aten/src/ATen/native/native_functions.yaml \
  "$OUT"/annotated_fn_args \
  tools/autograd
