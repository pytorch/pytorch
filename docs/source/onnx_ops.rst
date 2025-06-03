torch.onnx.ops
==============

.. automodule:: torch.onnx.ops

Symbolic Operators
------------------

Operators that can be used to create any ONNX ops in the FX graph symbolically.
These operators do not do actual computation. It's recommended that you used them
inside an ``if torch.onnx.is_in_onnx_export`` block.

.. autofunction:: torch.onnx.ops.symbolic

.. autofunction:: torch.onnx.ops.symbolic_multi_out


ONNX Operators
--------------

The following operators are implemented as native PyTorch ops and can be exported as
ONNX operators. They can be used natively in an ``nn.Module``.

For example,

.. autofunction:: torch.onnx.ops.rotary_embedding

ONNX to ATen Decomposition Table
--------------------------------

You can use :func:`torch.onnx.ops.aten_decompositions` to obtain a decomposition Table
to decompose ONNX operators defined above to ATen operators.
