"""Constant values used in ONNX."""

onnx_default_opset = 13
onnx_main_opset = 16
onnx_stable_opsets = tuple(range(7, onnx_main_opset))
onnx_constant_folding_opsets = tuple(range(9, onnx_main_opset + 1))
