#!/bin/bash
# Run this script inside its folder to generate PyTorch ONNX Export Diagnostic rules
# for C++, Python and documentations.
# The rules are defined in torch/onnx/_internal/diagnostics/rules.yaml.

set -e -x
ROOT="${PWD}/../../"
pushd "$ROOT"
(
python -m tools.onnx.gen_diagnostics \
    torch/onnx/_internal/diagnostics/rules.yaml \
    torch/onnx/_internal/diagnostics \
    torch/csrc/onnx/diagnostics/generated \
    torch/docs/source
)
popd
