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
.. autofunction:: enable_fake_mode
    :noindex:
```

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
```

### Deprecated APIs

```{eval-rst}
.. deprecated:: 2.6
    These functions are deprecated and will be removed in a future version.

.. autofunction:: register_custom_op_symbolic
.. autofunction:: unregister_custom_op_symbolic
.. autofunction:: select_model_mode_for_export
.. autoclass:: JitScalarType
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
