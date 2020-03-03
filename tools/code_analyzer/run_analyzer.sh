#!/bin/bash
##############################################################################
# Invoke code analyzer binary with pre-defined parameters for LibTorch.
# This script should be called via build.sh. Do NOT use it directly.
##############################################################################

set -exu

echo "Analyze: ${INPUT}"

"${ANALYZER_BIN}" \
  -op_schema_pattern="^(_aten|_prim|aten|quantized|profiler|_test)::[^ ]+" \
  -op_register_pattern="c10::RegisterOperators::(op|checkSchemaAndRegisterOp_)|c10::Module::(def|impl)" \
  -op_invoke_pattern="c10::Dispatcher::findSchema|callOp" \
  -format="${FORMAT}" \
  ${EXTRA_ANALYZER_FLAGS} \
  "${INPUT}" \
  > "${OUTPUT}"

echo "Result: ${OUTPUT}"
