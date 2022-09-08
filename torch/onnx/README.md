# torch.onnx

Torch->ONNX converter / exporter.

[User-facing docs](https://pytorch.org/docs/master/onnx.html).

[Developer docs](https://github.com/pytorch/pytorch/wiki/PyTorch-ONNX-exporter).

## Symbolic functions Opsets

Opset 9 is the base version. It is selected as the base version because

1. It is the first opset version supported by PyTorch export.
2. Opset 9 is more robust than previous opset versions. Opset versions like 7/8 have limitations
    that certain basic operators cannot be expressed in ONNX. Instead of basing on these limitations,
    we chose to handle them as special cases separately.

Backward support for opset versions beyond opset 7 is not in our roadmap.

For opset versions other than 9, by default they will inherit the symbolic functions defined in
symbolic_opset9.py.

To extend support for updated operators in different opset versions on top of opset 9,
simply add the updated symbolic functions in the respective symbolic_opset{version}.py file.
Checkout topk in symbolic_opset10.py, and upsample_nearest2d in symbolic_opset8.py for example.
