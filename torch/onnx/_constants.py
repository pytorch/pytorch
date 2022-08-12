"""Constant values used in ONNX."""

ONNX_ARCHIVE_MODEL_PROTO_NAME = "__MODEL_PROTO"
onnx_default_opset = 13
onnx_main_opset = 17
onnx_stable_opsets = tuple(range(7, onnx_main_opset))
onnx_constant_folding_opsets = tuple(range(9, onnx_main_opset + 1))

PYTORCH_GITHUB_ISSUES_URL = "https://github.com/pytorch/pytorch/issues"
