import os


onnx_model_dir = os.path.join(os.path.dirname(
    os.path.realpath(__file__)), os.pardir, "repos", "onnx", "onnx",
    "backend", "test", "data")


pytorch_converted_dir = os.path.join(onnx_model_dir, "pytorch-converted")


pytorch_operator_dir = os.path.join(onnx_model_dir, "pytorch-operator")
