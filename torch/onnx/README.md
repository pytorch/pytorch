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

The FX promotion pass remains active during migration to operator-owned
promotion; removing it is a separate dependent change.
