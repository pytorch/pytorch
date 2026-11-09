# torch.onnx


## Overview

[Open Neural Network eXchange (ONNX)](https://onnx.ai/) is an open standard
format for representing machine learning models. The `torch.onnx` module captures the computation graph from a
native PyTorch {class}`torch.nn.Module` model and converts it into an
[ONNX graph](https://github.com/onnx/onnx/blob/main/docs/IR.md).

The exported model can be consumed by any of the many
[runtimes that support ONNX](https://onnx.ai/supported-tools.html#deployModel), including
Microsoft's [ONNX Runtime](https://www.onnxruntime.ai).

Next example shows how to export a simple model.

```python
import torch

class MyModel(torch.nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 128, 5)

    def forward(self, x):
        return torch.relu(self.conv1(x))

input_tensor = torch.rand((1, 1, 128, 128), dtype=torch.float32)

model = MyModel()

torch.onnx.export(
    model,                  # model to export
    (input_tensor,),        # inputs of the model,
    "my_model.onnx",        # filename of the ONNX model
    input_names=["input"],  # Rename inputs for the ONNX model
    dynamo=True             # True or False to select the exporter to use
)
```


## torch.export-based ONNX Exporter

*The torch.export-based ONNX exporter is the newest exporter for PyTorch 2.6 and newer*

{ref}`torch.export <torch.export>` engine is leveraged to produce a traced graph representing only the Tensor computation of the function in an
Ahead-of-Time (AOT) fashion. The resulting traced graph (1) produces normalized operators in the functional
ATen operator set (as well as any user-specified custom operators), (2) has eliminated all Python control
flow and data structures (with certain exceptions), and (3) records the set of shape constraints needed to
show that this normalization and control-flow elimination is sound for future inputs, before it is finally
translated into an ONNX graph.

{doc}`Learn more about the torch.export-based ONNX Exporter <onnx_export>`

## Frequently Asked Questions

Q: I have exported my LLM model, but its input size seems to be fixed?

  The tracer records the shapes of the example inputs. If the model should accept
  inputs of dynamic shapes, set ``dynamic_shapes`` when calling {func}`torch.onnx.export`.

Q: How to export models containing loops?

  See {ref}`torch.cond <cond>`.


## Contributing / Developing

The ONNX exporter is a community project and we welcome contributions. We follow the
[PyTorch guidelines for contributions](https://github.com/pytorch/pytorch/blob/main/CONTRIBUTING.md), but you might
also be interested in reading our [development wiki](https://github.com/pytorch/pytorch/wiki/PyTorch-ONNX-exporter).


## torch.onnx APIs

```{eval-rst}
.. automodule:: torch.onnx
```

### Functions

```{eval-rst}
.. autofunction:: export
    :noindex:
.. autofunction:: is_in_onnx_export
    :noindex:
.. autofunction:: register_custom_translation
    :noindex:
.. autofunction:: unregister_custom_translation
    :noindex:
.. autofunction:: get_translation_builder
    :noindex:
```

### Native custom translations

Global translations apply to subsequent `dynamo=True` exports. They build ONNX
IR directly, without ONNX Script evaluation. For example, to replace `aten.add`
with an equivalent explicit translation:

```python
import torch
import onnx_ir as ir

@torch.onnx.register_custom_translation(torch.ops.aten.add.Tensor)
def add(self: ir.Value, other: ir.Value, alpha: float = 1):
    builder = torch.onnx.get_translation_builder()
    if alpha != 1:
        (other,) = builder.call("Mul", other, alpha)
    return builder.call("Add", self, other)[0]

class Model(torch.nn.Module):
    def forward(self, x, y):
        return torch.add(x, y, alpha=2)

try:
    program = torch.onnx.export(Model(), (torch.ones(2), torch.ones(2)))
finally:
    torch.onnx.unregister_custom_translation(torch.ops.aten.add.Tensor)
```

Direct registration, `register_custom_translation(target, function)`, returns the
same callable as the decorator form. `opset_version` is a minimum; the highest
compatible variant is selected separately for real and `is_complex=True` inputs.
Duplicate keys raise an error unless `replace=True`; replacement and unregister
affect only that exact target/complex/minimum-opset variant. Missing unregister
keys raise `KeyError`.

Export-local `custom_translation_table` overrides globals, which override
built-ins. Existing ONNX Script functions in that table remain supported. Global
state is snapshotted before model capture: registrations changed by model code or
another thread affect only later exports. Registration never loads ONNX Script or
the built-in operator library.

The builder's `call` returns a tuple and converts Python constants according to
ONNX schemas, but never promotes or casts tensors. Use explicit `Cast`,
`Reshape`/`Concat`, sequence and control-flow nodes as necessary. Raw/custom
operators use `node(..., output_types=..., domain=..., version=...)`; local IR
functions use `call_function`. Optional `signature` metadata uses
`onnx_ir.schemas.OpSignature` with parameter names matching the Python callable;
native IR tensor constraints and shared homogeneous bindings are checked before
invocation. The active builder is
scoped to the current translation; accessing it outside that scope raises an
actionable error. `torch.onnx.ops.symbolic` is a separate FX-level API.

Arithmetic and elementwise built-ins are being migrated to direct ONNX IR
construction in PyTorch. Native implementations use explicit casts and represent
complex tensors with a trailing dimension of size two for real and imaginary
components. Real/complex and opset eligibility select an implementation; tensor
dtypes do not select overloads.

Until all built-ins are migrated, legacy translations still use their isolated
ONNX Script evaluator path. When a native value crosses into that path, an
`Identity` supplies the legacy Python-operator interface; normal optimization
can remove this compatibility node.

### Classes

```{eval-rst}
.. autoclass:: ONNXProgram
    :noindex:
.. autoclass:: OnnxExporterError
    :noindex:
```

```{eval-rst}
.. toctree::
    :hidden:

    onnx_export
    onnx_ops
    onnx_verification
    onnx_testing
```

### Deprecated APIs

```{eval-rst}
.. deprecated:: 2.6
    These functions are deprecated and will be removed in a future version.

.. autofunction:: register_custom_op_symbolic
.. autofunction:: unregister_custom_op_symbolic
.. autofunction:: select_model_mode_for_export
```

```{eval-rst}
.. py:module:: torch.onnx.errors
.. py:module:: torch.onnx.operators
.. py:module:: torch.onnx.symbolic_helper
.. py:module:: torch.onnx.symbolic_opset10
.. py:module:: torch.onnx.symbolic_opset11
.. py:module:: torch.onnx.symbolic_opset12
.. py:module:: torch.onnx.symbolic_opset13
.. py:module:: torch.onnx.symbolic_opset14
.. py:module:: torch.onnx.symbolic_opset15
.. py:module:: torch.onnx.symbolic_opset16
.. py:module:: torch.onnx.symbolic_opset17
.. py:module:: torch.onnx.symbolic_opset18
.. py:module:: torch.onnx.symbolic_opset19
.. py:module:: torch.onnx.symbolic_opset20
.. py:module:: torch.onnx.symbolic_opset7
.. py:module:: torch.onnx.symbolic_opset8
.. py:module:: torch.onnx.symbolic_opset9
.. py:module:: torch.onnx.utils
```
