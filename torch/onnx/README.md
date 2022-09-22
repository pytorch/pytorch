# torch.onnx

Torch->ONNX converter / exporter.

- User-facing docs: https://pytorch.org/docs/master/onnx.html
- Developer docs: https://github.com/pytorch/pytorch/wiki/PyTorch-ONNX-exporter

> Read the following if you are contributing to `torch.onnx`

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

## Editing Symbolic Files

- Use the internal `registration.onnx_symbolic` decorator to register a new symbolic function. Search for `def reshape(g, self, shape):` to see an example.
- Parameter names must *exactly* match the names in
  aten/src/ATen/native/native_functions.yaml, because
  dispatch is done with keyword arguments.
- Looking for inplace ops? They're detected by
  `_jit_pass_onnx_remove_inplace_ops_for_onnx`, and
  transparently dispatched to their non inplace versions in
  "run_symbolic_function". See Note [Export inplace](#export-inplace)
- Required: Annotate new symbolic functions with type annotations and decorate
  with `@_beartype.beartype` to enable runtime type checking.
  `@_beartype.beartype` should typically be the closest to the function to
  ensure proper type checking.

### A note on Tensor types

In general, we should avoid depending on the type of Tensor Values contained
within the trace graph. However, this is sometimes unavoidable (due to ONNX
spec requirements, etc). The TensorType object has accessors for these properties that return the property if it is statically known and return nullopt otherwise.

In general, we should prefer to rely on the least specific information possible.
For example, not relying on tensor properties at all is better than relying
on the number of dimensions which is better than relying on
concrete shapes. Doing so will make the export symbolics
more robust to different graphs.

### Extra context for symbolic functions

The first argument of a symbolic function is always a `GraphContext` object.

`GraphContext` contains all methods defined in a `torch.Graph` object and context
for the symbolic function.

In general, symbolic functions only require inputs and attributes to
the original node. An example of a symbolic function needing context is
`prim::Loop`. It needs access to the sub-block of the original node.

### Export inplace

It would be better for us to export inplace annotations,
than to not export them, since it is useful information that can
help the target of an ONNX export export more efficiently. However,
ONNX doesn't currently formalize inplace. Fortunately, it's sound to drop
inplace annotations, but we are losing information this way.

### Pointwise by scalar

What happens if you add a tensor with a constant (e.g., x + 2)?  There are
some moving parts to implementing the ONNX translation in this case:

- By the time we get the scalar in a symbolic function here, it is no longer a
  Python long/float, but a PyTorch tensor with `numel == 1` (eventually, we want
  it to be a zero dim tensor but this change has not happened yet.) However, the
  type of this scalar is *exactly* what the user wrote in Python, which may not
  match the tensor it is being added to. PyTorch will do implicit conversions on
  scalars; however, ONNX will not, so we must do the conversion ourselves. This
  is what `symbolic_helper._if_scalar_type_as()` and
  `_jit_pass_onnx_scalar_type_analysis` does.

- Dispatch to these functions takes advantage an outrageous coincidence
    between the tensor and scalar name.  When we add two tensors together,
    you get the dispatch:

    add(*[self, other], **{"alpha": alpha})

    When you add a tensor and a scalar, you get the dispatch:

    add(*[self], **{"other": other, "alpha": alpha})

    By having the argument name line up with the name of the scalar attribute
    if it exists, we can write a single function for both overloads.
