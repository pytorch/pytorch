torch.onnx
==========

Overview
--------

`Open Neural Network eXchange (ONNX) <https://onnx.ai/>`_ is an open standard
format for representing machine learning models. The ``torch.onnx`` module captures the computation graph from a
native PyTorch :class:`torch.nn.Module` model and converts it into an
`ONNX graph <https://github.com/onnx/onnx/blob/main/docs/IR.md>`_.

The exported model can be consumed by any of the many
`runtimes that support ONNX <https://onnx.ai/supported-tools.html#deployModel>`_, including
Microsoft's `ONNX Runtime <https://www.onnxruntime.ai>`_.

**There are two flavors of ONNX exporter API that you can use, as listed below:**

TorchDynamo-based ONNX Exporter
-------------------------------

*The TorchDynamo-based ONNX exporter is the newest (and Beta) exporter for PyTorch 2.0 and newer*

TorchDynamo engine is leveraged to hook into Python's frame evaluation API and dynamically rewrite its
bytecode into an FX Graph. The resulting FX Graph is then polished before it is finally translated into an
ONNX graph.

The main advantage of this approach is that the `FX graph <https://pytorch.org/docs/stable/fx.html>`_ is captured using
bytecode analysis that preserves the dynamic nature of the model instead of using traditional static tracing techniques.

:doc:`Learn more about the TorchDynamo-based ONNX Exporter <onnx_dynamo>`

TorchScript-based ONNX Exporter
-------------------------------

*The TorchScript-based ONNX exporter is available since PyTorch 1.2.0*

`TorchScript <https://pytorch.org/docs/stable/jit.html>`_ is leveraged to trace (through :func:`torch.jit.trace`)
the model and capture a static computation graph.

As a consequence, the resulting graph has a couple limitations:

* It does not record any control-flow, like if-statements or loops;
* Does not handle nuances between ``training`` and ``eval`` mode;
* Does not truly handle dynamic inputs

As an attempt to support the static tracing limitations, the exporter also supports TorchScript scripting
(through :func:`torch.jit.script`), which adds support for data-dependent control-flow, for example. However, TorchScript
itself is a subset of the Python language, so not all features in Python are supported, such as in-place operations.

:doc:`Learn more about the TorchScript-based ONNX Exporter <onnx_torchscript>`

Contributing / Developing
-------------------------

The ONNX exporter is a community project and we welcome contributions. We follow the
`PyTorch guidelines for contributions <https://github.com/pytorch/pytorch/blob/main/CONTRIBUTING.md>`_, but you might
also be interested in reading our `development wiki <https://github.com/pytorch/pytorch/wiki/PyTorch-ONNX-exporter>`_.

.. toctree::
    :hidden:

    onnx_dynamo
    onnx_dynamo_onnxruntime_backend
    onnx_torchscript
