#!/bin/bash
##############################################################################
# Invoke code analyzer binary with pre-defined parameters for LibTorch.
# This script should be called via build.sh. Do NOT use it directly.
##############################################################################

set -exu

echo "Analyze: ${INPUT}"

# NB: op_register_pattern actually contains "too" many entries.  We only
# need to regex for symbols which occur after inlining; and most of the
# public API for the registration API disappears after inlining (e.g.,
# only _def and _impl are retained).  But the inliner isn't guaranteed
# to operate, so for safety we match a more expansive set.
"${ANALYZER_BIN}" \
  -op_schema_pattern="^(_aten|_prim|aten|quantized|_quantized|prepacked|profiler|_test)::[a-zA-Z0-9_.]+(\(.*)?$" \
  -op_register_pattern="c10::RegisterOperators::(op|checkSchemaAndRegisterOp_)|c10::Module::(_?def|_?impl)|torch::Library::(_?def|_?impl)" \
  -op_invoke_pattern="c10::Dispatcher::findSchema" \
  -root_symbol_pattern="torch::jit::[^(]" \
  -torch_library_init_pattern="^.*TORCH_LIBRARY_init_([^(]+)(\(.*)?$" \
  -torch_library_init_pattern="^.*TORCH_LIBRARY_FRAGMENT_init_([_]*[^_]+)_[0-9]+(\(.*)?$" \
  -torch_library_init_pattern="^.*TORCH_LIBRARY_IMPL_init_([_]*[^_]+)_([^_]+)_[0-9]+(\(.*)?$" \
  ${EXTRA_ANALYZER_FLAGS} \
  "${INPUT}" \
  > "${OUTPUT}"

echo "Result: ${OUTPUT}"
