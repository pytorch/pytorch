#!/bin/bash
cd ../../
python -m tools.onnx.gen_diagnostics \
    torch/onnx/_internal/diagnostics/rules.yaml \
    torch/onnx/_internal/diagnostics \
    torch/csrc/onnx/diagnostics/generated \
    torch/docs/source
