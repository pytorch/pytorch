TorchDynamo-based ONNX Exporter
===============================

.. automodule:: torch.onnx
  :noindex:

.. contents:: :local:
    :depth: 3

.. warning::
  The ONNX exporter for TorchDynamo is a rapidly evolving beta technology.

Overview
--------

The ONNX exporter leverages TorchDynamo engine to hook into Python's frame evaluation API
and dynamically rewrite its bytecode into an FX Graph.
The resulting FX Graph is then polished before it is finally translated into an ONNX graph.

The main advantage of this approach is that the `FX graph <https://pytorch.org/docs/stable/fx.html>`_ is captured using
bytecode analysis that preserves the dynamic nature of the model instead of using traditional static tracing techniques.

In addition, during the export process, memory usage is significantly reduced compared to the TorchScript-enabled exporter.
See the :doc:`memory usage documentation <onnx_dynamo_memory_usage>` for more information.


Dependencies
------------

The ONNX exporter depends on extra Python packages:

  - `ONNX <https://onnx.ai>`_
  - `ONNX Script <https://onnxscript.ai>`_

They can be installed through `pip <https://pypi.org/project/pip/>`_:

.. code-block:: bash

  pip install --upgrade onnx onnxscript

`onnxruntime <https://onnxruntime.ai>`_ can then be used to execute the model
on a large variety of processors.

A simple example
----------------

See below a demonstration of exporter API in action with a simple Multilayer Perceptron (MLP) as example:

.. code-block:: python

  import torch
  import torch.nn as nn

  class MLPModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc0 = nn.Linear(8, 8, bias=True)
        self.fc1 = nn.Linear(8, 4, bias=True)
        self.fc2 = nn.Linear(4, 2, bias=True)
        self.fc3 = nn.Linear(2, 2, bias=True)

    def forward(self, tensor_x: torch.Tensor):
        tensor_x = self.fc0(tensor_x)
        tensor_x = torch.sigmoid(tensor_x)
        tensor_x = self.fc1(tensor_x)
        tensor_x = torch.sigmoid(tensor_x)
        tensor_x = self.fc2(tensor_x)
        tensor_x = torch.sigmoid(tensor_x)
        output = self.fc3(tensor_x)
        return output

  model = MLPModel()
  tensor_x = torch.rand((97, 8), dtype=torch.float32)
  onnx_program = torch.onnx.export(model, (tensor_x,), dynamo=True)

As the code above shows, all you need is to provide :func:`torch.onnx.export` with an instance of the model and its input.
The exporter will then return an instance of :class:`torch.onnx.ONNXProgram` that contains the exported ONNX graph along with extra information.

``onnx_program.optimize()`` can be called to optimize the ONNX graph with constant folding and elimination of redundant operators. The optimization is done in-place.

.. code-block:: python

  onnx_program.optimize()

The in-memory model available through ``onnx_program.model_proto`` is an ``onnx.ModelProto`` object in compliance with the `ONNX IR spec <https://github.com/onnx/onnx/blob/main/docs/IR.md>`_.
The ONNX model may then be serialized into a `Protobuf file <https://protobuf.dev/>`_ using the :meth:`torch.onnx.ONNXProgram.save` API.

.. code-block:: python

  onnx_program.save("mlp.onnx")

Two functions exist to export the model to ONNX based on TorchDynamo engine.
They slightly differ in the way they produce the :class:`torch.export.ExportedProgram`.
:func:`torch.onnx.dynamo_export` was introduced with PyTorch 2.1 and
:func:`torch.onnx.export` was extended with PyTorch 2.5 to easily switch
from TorchScript to TorchDynamo. To call the former function,
the last line of the previous example can be replaced by the following one.

.. note::
    :func:`torch.onnx.dynamo_export` will be deprecated in the future. Please use :func:`torch.onnx.export` with the parameter ``dynamo=True`` instead.

.. code-block:: python

  onnx_program = torch.onnx.dynamo_export(model, tensor_x)

Inspecting the ONNX model using GUI
-----------------------------------

You can view the exported model using `Netron <https://netron.app/>`__.

.. image:: _static/img/onnx/onnx_dynamo_mlp_model.png
    :width: 40%
    :alt: MLP model as viewed using Netron

When the conversion fails
-------------------------

Function :func:`torch.onnx.export` should called a second time with
parameter ``report=True``. A markdown report is generated to help the user
to resolve the issue.

.. toctree::
    :hidden:

    onnx_dynamo_memory_usage

API Reference
-------------

.. autofunction:: torch.onnx.dynamo_export

.. autoclass:: torch.onnx.ONNXProgram
    :members:

.. autoclass:: torch.onnx.ExportOptions
    :members:

.. autofunction:: torch.onnx.enable_fake_mode

.. autoclass:: torch.onnx.ONNXRuntimeOptions
    :members:

.. autoclass:: torch.onnx.OnnxExporterError
    :members:

.. autoclass:: torch.onnx.OnnxRegistry
    :members:

.. autoclass:: torch.onnx.DiagnosticOptions
    :members: