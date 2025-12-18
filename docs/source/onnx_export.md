# torch.export-based ONNX Exporter

```{eval-rst}
.. automodule:: torch.onnx
  :noindex:
```

```{contents}
:local:
:depth: 1
```

## Overview

{ref}`torch.export <torch.export>` engine is leveraged to produce a traced graph representing only the Tensor computation of the function in an
Ahead-of-Time (AOT) fashion. The resulting traced graph (1) produces normalized operators in the functional
ATen operator set (as well as any user-specified custom operators), (2) has eliminated all Python control
flow and data structures (with certain exceptions), and (3) records the set of shape constraints needed to
show that this normalization and control-flow elimination is sound for future inputs, before it is finally
translated into an ONNX graph.

In addition, during the export process, memory usage is significantly reduced.

## Dependencies

The ONNX exporter depends on extra Python packages:

  - [ONNX](https://onnx.ai)
  - [ONNX Script](https://microsoft.github.io/onnxscript)

They can be installed through [pip](https://pypi.org/project/pip/):

```{code-block} bash

  pip install --upgrade onnx onnxscript
```

[onnxruntime](https://onnxruntime.ai) can then be used to execute the model
on a large variety of processors.

## A simple example

See below a demonstration of exporter API in action with a simple Multilayer Perceptron (MLP) as example:

```{code-block} python
import torch
import torch.nn as nn

class MLPModel(nn.Module):
  def __init__(self):
      super().__init__()
      self.fc0 = nn.Linear(8, 8, bias=True)
      self.fc1 = nn.Linear(8, 4, bias=True)
      self.fc2 = nn.Linear(4, 2, bias=True)
      self.fc3 = nn.Linear(2, 2, bias=True)
      self.fc_combined = nn.Linear(8 + 8 + 8, 8, bias=True)  # Combine all inputs

  def forward(self, tensor_x: torch.Tensor, input_dict: dict, input_list: list):
      """
      Forward method that requires all inputs:
      - tensor_x: A direct tensor input.
      - input_dict: A dictionary containing the tensor under the key 'tensor_x'.
      - input_list: A list where the first element is the tensor.
      """
      # Extract tensors from inputs
      dict_tensor = input_dict['tensor_x']
      list_tensor = input_list[0]

      # Combine all inputs into a single tensor
      combined_tensor = torch.cat([tensor_x, dict_tensor, list_tensor], dim=1)

      # Process the combined tensor through the layers
      combined_tensor = self.fc_combined(combined_tensor)
      combined_tensor = torch.sigmoid(combined_tensor)
      combined_tensor = self.fc0(combined_tensor)
      combined_tensor = torch.sigmoid(combined_tensor)
      combined_tensor = self.fc1(combined_tensor)
      combined_tensor = torch.sigmoid(combined_tensor)
      combined_tensor = self.fc2(combined_tensor)
      combined_tensor = torch.sigmoid(combined_tensor)
      output = self.fc3(combined_tensor)
      return output

model = MLPModel()

# Example inputs
tensor_input = torch.rand((97, 8), dtype=torch.float32)
dict_input = {'tensor_x': torch.rand((97, 8), dtype=torch.float32)}
list_input = [torch.rand((97, 8), dtype=torch.float32)]

# The input_names and output_names are used to identify the inputs and outputs of the ONNX model
input_names = ['tensor_input', 'tensor_x', 'list_input_index_0']
output_names = ['output']

# Exporting the model with all required inputs
onnx_program = torch.onnx.export(model,(tensor_input, dict_input, list_input), dynamic_shapes=({0: "batch_size"},{"tensor_x": {0: "batch_size"}},[{0: "batch_size"}]), input_names=input_names, output_names=output_names, dynamo=True,)

# Check the exported ONNX model is dynamic
assert onnx_program.model.graph.inputs[0].shape == ("batch_size", 8)
assert onnx_program.model.graph.inputs[1].shape == ("batch_size", 8)
assert onnx_program.model.graph.inputs[2].shape == ("batch_size", 8)
```

As the code above shows, all you need is to provide {func}`torch.onnx.export` with an instance of the model and its input.
The exporter will then return an instance of {class}`torch.onnx.ONNXProgram` that contains the exported ONNX graph along with extra information.

The in-memory model available through ``onnx_program.model_proto`` is an ``onnx.ModelProto`` object in compliance with the [ONNX IR spec](https://github.com/onnx/onnx/blob/main/docs/IR.md).
The ONNX model may then be serialized into a [Protobuf file](https://protobuf.dev/) using the {meth}`torch.onnx.ONNXProgram.save` API.

```{code-block} python
  onnx_program.save("mlp.onnx")
```

## Inspecting the ONNX model using GUI

You can view the exported model using [Netron](https://netron.app/).

```{image} _static/img/onnx/onnx_dynamo_mlp_model.png
:alt: MLP model as viewed using Netron
:width: 30%
:align: center
```

## When the conversion fails

Function {func}`torch.onnx.export` should be called a second time with
parameter ``report=True``. A markdown report is generated to help the user
to resolve the issue.

## Metadata

During ONNX export, each ONNX node is annotated with metadata that helps trace its origin and context from the original PyTorch model. This metadata is useful for debugging, model inspection, and understanding the mapping between PyTorch and ONNX graphs.

The following metadata fields are added to each ONNX node:

- **namespace**

  A string representing the hierarchical namespace of the node, consisting of a stack trace of modules/methods.

  *Example:*
  `__main__.SimpleAddModel/add: aten.add.Tensor`

- **pkg.torch.onnx.class_hierarchy**

  A list of class names representing the hierarchy of modules leading to this node.

  *Example:*
  `['__main__.SimpleAddModel', 'aten.add.Tensor']`

- **pkg.torch.onnx.fx_node**

  The string representation of the original FX node, including its name, number of consumers, the targeted torch op, arguments, and keyword arguments.

  *Example:*
  `%cat : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%tensor_x, %input_dict_tensor_x, %input_list_0], 1), kwargs = {})`

- **pkg.torch.onnx.name_scopes**

  A list of name scopes (methods) representing the path to this node in the PyTorch model.

  *Example:*
  `['', 'add']`

- **pkg.torch.onnx.stack_trace**

  The stack trace from the original code where this node was created, if available.

  *Example:*
  ```
  File "simpleadd.py", line 7, in forward
      return torch.add(x, y)
  ```

These metadata fields are stored in the metadata_props attribute of each ONNX node and can be inspected using Netron or programmatically.

The overall ONNX graph has the following `metadata_props`:

- **pkg.torch.export.ExportedProgram.graph_signature**

  This property contains a string representation of the graph_signature from the original PyTorch ExportedProgram. The graph signature describes the structure of the model's inputs and outputs and how they map to the ONNX graph. The inputs are defined as `InputSpec` objects, which include the kind of input (e.g., `InputKind.PARAMETER` for parameters, `InputKind.USER_INPUT` for user-defined inputs), the argument name, the target (which can be a specific node in the model), and whether the input is persistent. The outputs are defined as `OutputSpec` objects, which specify the kind of output (e.g., `OutputKind.USER_OUTPUT`) and the argument name.

  To read more about the graph signature, please see the {doc}`torch.export <user_guide/torch_compiler/export>` for more information.

- **pkg.torch.export.ExportedProgram.range_constraints**

  This property contains a string representation of any range constraints that were present in the original PyTorch ExportedProgram. Range constraints specify valid ranges for symbolic shapes or values in the model, which can be important for models that use dynamic shapes or symbolic dimensions.

  *Example:*
  `s0: VR[2, int_oo]`, which indicates that the size of the input tensor must be at least 2.

  To read more about range constraints, please see the {doc}`torch.export <user_guide/torch_compiler/export>` for more information.

Each input value in the ONNX graph may have the following metadata property:

- **pkg.torch.export.graph_signature.InputSpec.kind**

  The kind of input, as defined by PyTorch's InputKind enum.

  *Example values:*
  - "USER_INPUT": A user-provided input to the model.
  - "PARAMETER": A model parameter (e.g., weight).
  - "BUFFER": A model buffer (e.g., running mean in BatchNorm).
  - "CONSTANT_TENSOR": A constant tensor argument.
  - "CUSTOM_OBJ": A custom object input.
  - "TOKEN": A token input.

- **pkg.torch.export.graph_signature.InputSpec.persistent**

  Indicates whether the input is persistent (i.e., should be saved as part of the model's state).

  *Example values:*
  - "True"
  - "False"

Each output value in the ONNX graph may have the following metadata property:

- **pkg.torch.export.graph_signature.OutputSpec.kind**

  The kind of input, as defined by PyTorch's OutputKind enum.

  *Example values:*
  - "USER_OUTPUT": A user-visible output.
  - "LOSS_OUTPUT": A loss value output.
  - "BUFFER_MUTATION": Indicates a buffer was mutated.
  - "GRADIENT_TO_PARAMETER": Gradient output for a parameter.
  - "GRADIENT_TO_USER_INPUT": Gradient output for a user input.
  - "USER_INPUT_MUTATION": Indicates a user input was mutated.
  - "TOKEN": A token output.

Each initialized value, input, output has the following metadata:

- **pkg.torch.onnx.original_node_name**

  The original name of the node in the PyTorch FX graph that produced this value in the case where the value was renamed. This helps trace initializers back to their source in the original model.

  *Example:*
  `fc1.weight`

## API Reference

```{eval-rst}
.. autofunction:: torch.onnx.export
.. autoclass:: torch.onnx.ONNXProgram
    :members:
.. autoclass:: torch.onnx.ExportableModule
    :members:
.. autofunction:: torch.onnx.is_in_onnx_export
.. autoclass:: torch.onnx.OnnxExporterError
    :members:
```
