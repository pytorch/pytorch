# torch.onnx

Torch->ONNX converter / exporter.

- User-facing docs: https://pytorch.org/docs/main/onnx.html
- Developer docs: https://github.com/pytorch/pytorch/wiki/PyTorch-ONNX-exporter

Native operator lowerings explicitly promote their participating operands with
`_torchlib.ops._promotion`. Promotion inspects logical source dtypes and tensor
ranks, including zero-dimensional tensors and wrapped Python scalars, without
executing tensors. Packed complex values retain their original PyTorch dtype.
Lowerings choose accumulation and result dtypes separately. The generic IR
builder only converts Python literals; it never silently promotes tensor inputs.
Custom translations remain responsible for their own explicit tensor casts.
The isolated legacy ONNX Script adapter preserves its historical ATen input
promotion and output casts; it does not change the native custom contract.

Export preparation does not rerun the FX graph for promotion or insert ATen cast
nodes. Native lowerings insert ONNX casts locally, preserving decomposed graph
metadata, including strides needed by subsequent views.
