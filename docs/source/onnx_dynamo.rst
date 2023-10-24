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

The exporter is designed to be modular and extensible. It is composed of the following components:

  - **ONNX Exporter**: :class:`Exporter` main class that orchestrates the export process.
  - **ONNX Export Options**: :class:`ExportOptions` has a set of options that control the export process.
  - **ONNX Registry**: :class:`OnnxRegistry` is the registry of ONNX operators and functions.
  - **FX Graph Extractor**: :class:`FXGraphExtractor` extracts the FX graph from the PyTorch model.
  - **Fake Mode**: :class:`ONNXFakeContext` is a context manager that enables fake mode for large scale models.
  - **ONNX Export Output**: :class:`ExportOutput` is the output of the exporter that contains the exported ONNX graph and diagnostics.
  - **ONNX Export Output Serializer**: :class:`ExportOutputSerializer` serializes the exported model to a file.
  - **ONNX Diagnostic Options**: :class:`DiagnosticOptions` has a set of options that control the diagnostics emitted by the exporter.

Dependencies
------------

The ONNX exporter depends on extra Python packages:

  - `ONNX <https://onnx.ai>`_
  - `ONNX Script <https://onnxscript.ai>`_

They can be installed through `pip <https://pypi.org/project/pip/>`_:

.. code-block:: bash

  pip install --upgrade onnx onnxscript

A simple example
----------------

See below a demonstration of exporter API in action with a simple Multilayer Perceptron (MLP) as example:

.. code-block:: python

  import torch

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
  export_output = torch.onnx.dynamo_export(model, tensor_x)

As the code above shows, all you need is to provide :func:`torch.onnx.dynamo_export` with an instance of the model and its input.
The exporter will then return an instance of :class:`torch.onnx.ExportOutput` that contains the exported ONNX graph along with extra information.

The in-memory model available through ``export_output.model_proto`` is an ``onnx.ModelProto`` object in compliance with the `ONNX IR spec <https://github.com/onnx/onnx/blob/main/docs/IR.md>`_.
The ONNX model may then be serialized into a `Protobuf file <https://protobuf.dev/>`_ using the :meth:`torch.onnx.ExportOutput.save` API.

.. code-block:: python

  export_output.save("mlp.onnx")

Inspecting the ONNX model using GUI
-----------------------------------

You can view the exported model using `Netron <https://netron.app/>`__.

.. image:: _static/img/onnx/onnx_dynamo_mlp_model.png
    :width: 40%
    :alt: MLP model as viewed using Netron

Note that each layer is represented in a rectangular box with a *f* icon in the top right corner.

.. image:: _static/img/onnx/onnx_dynamo_mlp_model_function_highlight.png
    :width: 40%
    :alt: ONNX function highlighted on MLP model

By expanding it, the function body is shown.

.. image:: _static/img/onnx/onnx_dynamo_mlp_model_function_body.png
    :width: 50%
    :alt: ONNX function body

The function body is a sequence of ONNX operators or other functions.

Diagnosing issues with SARIF
----------------------------

ONNX diagnostics goes beyond regular logs through the adoption of
`Static Analysis Results Interchange Format (aka SARIF) <https://docs.oasis-open.org/sarif/sarif/v2.1.0/sarif-v2.1.0.html>`__
to help users debug and improve their model using a GUI, such as
Visual Studio Code's `SARIF Viewer <https://marketplace.visualstudio.com/items?itemName=MS-SarifVSCode.sarif-viewer>`_.

The main advantages are:

  - The diagnostics are emitted in machine parseable `Static Analysis Results Interchange Format (SARIF) <https://docs.oasis-open.org/sarif/sarif/v2.1.0/sarif-v2.1.0.html>`__.
  - A new clearer, structured way to add new and keep track of diagnostic rules.
  - Serve as foundation for more future improvements consuming the diagnostics.

.. toctree::
   :maxdepth: 1
   :caption: ONNX Diagnostic SARIF Rules
   :glob:

   generated/onnx_dynamo_diagnostics_rules/*

API Reference
-------------

.. autofunction:: torch.onnx.dynamo_export

.. autoclass:: torch.onnx.ExportOptions
    :members:

.. autofunction:: torch.onnx.enable_fake_mode

.. autoclass:: torch.onnx.ExportOutput
    :members:

.. autoclass:: torch.onnx.ExportOutputSerializer
    :members:

.. autoclass:: torch.onnx.OnnxExporterError
    :members:

.. autoclass:: torch.onnx.OnnxRegistry
    :members:

.. autoclass:: torch.onnx.DiagnosticOptions
    :members:
