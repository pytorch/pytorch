cd ../../
python -m tools.onnx.gen_diagnostic \
    torch/onnx/diagnostic/rules.yaml \
    torch/onnx/diagnostic/generated \
    torch/csrc/onnx/diagnostic/generated \
    torch/docs/source
