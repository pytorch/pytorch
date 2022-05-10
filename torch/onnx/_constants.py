"""Constant values used in ONNX."""

default_onnx_opset_version = 13
onnx_main_opset = 16
onnx_stable_opsets = tuple(range(7, onnx_main_opset))
export_onnx_opset_version = default_onnx_opset_version
constant_folding_opset_versions = tuple(range(9, onnx_main_opset + 1))
