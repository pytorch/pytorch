(torch.export)=

# torch.export

## Overview

{func}`torch.export.export` takes a {class}`torch.nn.Module` and produces a traced graph
representing only the Tensor computation of the function in an Ahead-of-Time
(AOT) fashion, which can subsequently be executed with different outputs or
serialized.

```python
import torch
from torch.export import export

class Mod(torch.nn.Module):
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        a = torch.sin(x)
        b = torch.cos(y)
        return a + b

example_args = (torch.randn(10, 10), torch.randn(10, 10))

exported_program: torch.export.ExportedProgram = export(
    Mod(), args=example_args
)
print(exported_program)
```

```python
ExportedProgram:
    class GraphModule(torch.nn.Module):
        def forward(self, x: "f32[10, 10]", y: "f32[10, 10]"):
            # code: a = torch.sin(x)
            sin: "f32[10, 10]" = torch.ops.aten.sin.default(x)

            # code: b = torch.cos(y)
            cos: "f32[10, 10]" = torch.ops.aten.cos.default(y)

            # code: return a + b
            add: f32[10, 10] = torch.ops.aten.add.Tensor(sin, cos)
            return (add,)

    Graph signature:
        ExportGraphSignature(
            input_specs=[
                InputSpec(
                    kind=<InputKind.USER_INPUT: 1>,
                    arg=TensorArgument(name='x'),
                    target=None,
                    persistent=None
                ),
                InputSpec(
                    kind=<InputKind.USER_INPUT: 1>,
                    arg=TensorArgument(name='y'),
                    target=None,
                    persistent=None
                )
            ],
            output_specs=[
                OutputSpec(
                    kind=<OutputKind.USER_OUTPUT: 1>,
                    arg=TensorArgument(name='add'),
                    target=None
                )
            ]
        )
    Range constraints: {}
```

`torch.export` produces a clean intermediate representation (IR) with the
following invariants. More specifications about the IR can be found
{ref}`here <export.ir_spec>`.

- **Soundness**: It is guaranteed to be a sound representation of the original
  program, and maintains the same calling conventions of the original program.
- **Normalized**: There are no Python semantics within the graph. Submodules
  from the original programs are inlined to form one fully flattened
  computational graph.
- **Graph properties**: The graph is purely functional, meaning it does not
  contain operations with side effects such as mutations or aliasing. It does
  not mutate any intermediate values, parameters, or buffers.
- **Metadata**: The graph contains metadata captured during tracing, such as a
  stacktrace from user's code.

Under the hood, `torch.export` leverages the following latest technologies:

- **TorchDynamo (torch._dynamo)** is an internal API that uses a CPython feature
  called the Frame Evaluation API to safely trace PyTorch graphs. This
  provides a massively improved graph capturing experience, with much fewer
  rewrites needed in order to fully trace the PyTorch code.
- **AOT Autograd** provides a functionalized PyTorch graph and ensures the graph
  is decomposed/lowered to the ATen operator set.
- **Torch FX (torch.fx)** is the underlying representation of the graph,
  allowing flexible Python-based transformations.

### Existing frameworks

{func}`torch.compile` also utilizes the same PT2 stack as `torch.export`, but
is slightly different:

- **JIT vs. AOT**: {func}`torch.compile` is a JIT compiler whereas
  which is not intended to be used to produce compiled artifacts outside of
  deployment.
- **Partial vs. Full Graph Capture**: When {func}`torch.compile` runs into an
  untraceable part of a model, it will "graph break" and fall back to running
  the program in the eager Python runtime. In comparison, `torch.export` aims
  to get a full graph representation of a PyTorch model, so it will error out
  when something untraceable is reached. Since `torch.export` produces a full
  graph disjoint from any Python features or runtime, this graph can then be
  saved, loaded, and run in different environments and languages.
- **Usability tradeoff**: Since {func}`torch.compile` is able to fallback to the
  Python runtime whenever it reaches something untraceable, it is a lot more
  flexible. `torch.export` will instead require users to provide more
  information or rewrite their code to make it traceable.

Compared to {func}`torch.fx.symbolic_trace`, `torch.export` traces using
TorchDynamo which operates at the Python bytecode level, giving it the ability
to trace arbitrary Python constructs not limited by what Python operator
overloading supports. Additionally, `torch.export` keeps fine-grained track of
tensor metadata, so that conditionals on things like tensor shapes do not
fail tracing. In general, `torch.export` is expected to work on more user
programs, and produce lower-level graphs (at the `torch.ops.aten` operator
level). Note that users can still use {func}`torch.fx.symbolic_trace` as a
preprocessing step before `torch.export`.

Compared to {func}`torch.jit.script`, `torch.export` does not capture Python
control flow or data structures, unless using explicit {ref}`control flow operators <cond>`,
but it supports more Python language features due to its comprehensive coverage
over Python bytecodes. The resulting graphs are simpler and only have straight
line control flow, except for explicit control flow operators.

Compared to {func}`torch.jit.trace`, `torch.export` is sound:
it can trace code that performs integer computation on sizes and records
all of the side-conditions necessary to ensure that a particular
trace is valid for other inputs.

## Exporting a PyTorch Model

The main entrypoint is through {func}`torch.export.export`, which takes a
{class}`torch.nn.Module` and sample inputs, and
captures the computation graph into an {class}`torch.export.ExportedProgram`. An
example:

```python
import torch
from torch.export import export

# Simple module for demonstration
class M(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = torch.nn.Conv2d(
            in_channels=3, out_channels=16, kernel_size=3, padding=1
        )
        self.relu = torch.nn.ReLU()
        self.maxpool = torch.nn.MaxPool2d(kernel_size=3)

    def forward(self, x: torch.Tensor, *, constant=None) -> torch.Tensor:
        a = self.conv(x)
        a.add_(constant)
        return self.maxpool(self.relu(a))

example_args = (torch.randn(1, 3, 256, 256),)
example_kwargs = {"constant": torch.ones(1, 16, 256, 256)}

exported_program: torch.export.ExportedProgram = export(
    M(), args=example_args, kwargs=example_kwargs
)
print(exported_program)
```

```python
ExportedProgram:
    class GraphModule(torch.nn.Module):
    def forward(self, p_conv_weight: "f32[16, 3, 3, 3]", p_conv_bias: "f32[16]", x: "f32[1, 3, 256, 256]", constant: "f32[1, 16, 256, 256]"):
            # code: a = self.conv(x)
            conv2d: "f32[1, 16, 256, 256]" = torch.ops.aten.conv2d.default(x, p_conv_weight, p_conv_bias, [1, 1], [1, 1])

            # code: a.add_(constant)
            add_: "f32[1, 16, 256, 256]" = torch.ops.aten.add_.Tensor(conv2d, constant)

            # code: return self.maxpool(self.relu(a))
            relu: "f32[1, 16, 256, 256]" = torch.ops.aten.relu.default(add_)
            max_pool2d: "f32[1, 16, 85, 85]" = torch.ops.aten.max_pool2d.default(relu, [3, 3], [3, 3])
            return (max_pool2d,)

Graph signature:
    ExportGraphSignature(
        input_specs=[
            InputSpec(
                kind=<InputKind.PARAMETER: 2>,
                arg=TensorArgument(name='p_conv_weight'),
                target='conv.weight',
                persistent=None
            ),
            InputSpec(
                kind=<InputKind.PARAMETER: 2>,
                arg=TensorArgument(name='p_conv_bias'),
                target='conv.bias',
                persistent=None
            ),
            InputSpec(
                kind=<InputKind.USER_INPUT: 1>,
                arg=TensorArgument(name='x'),
                target=None,
                persistent=None
            ),
            InputSpec(
                kind=<InputKind.USER_INPUT: 1>,
                arg=TensorArgument(name='constant'),
                target=None,
                persistent=None
            )
        ],
        output_specs=[
            OutputSpec(
                kind=<OutputKind.USER_OUTPUT: 1>,
                arg=TensorArgument(name='max_pool2d'),
                target=None
            )
        ]
    )
Range constraints: {}
```

Inspecting the `ExportedProgram`, we can note the following:

- The {class}`torch.fx.Graph` contains the computation graph of the original
  program, along with records of the original code for easy debugging.
- The graph contains only `torch.ops.aten` operators found [here](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/native_functions.yaml)
  and custom operators.
- The parameters (weight and bias to conv) are lifted as inputs to the graph,
  resulting in no `get_attr` nodes in the graph, which previously existed in
  the result of {func}`torch.fx.symbolic_trace`.
- The {class}`torch.export.ExportGraphSignature` models the input and output
  signature, along with specifying which inputs are parameters.
- The resulting shape and dtype of tensors produced by each node in the graph is
  noted. For example, the `conv2d` node will result in a tensor of dtype
  `torch.float32` and shape (1, 16, 256, 256).

## Expressing Dynamism

By default `torch.export` will trace the program assuming all input shapes are
**static**, and specializing the exported program to those dimensions. However,
some dimensions, such as a batch dimension, can be dynamic and vary from run to
run. Such dimensions must be specified by using the
{func}`torch.export.Dim` API to create them and by passing them into
{func}`torch.export.export` through the `dynamic_shapes` argument.

An example:

```python
import torch
from torch.export import Dim, export

class M(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.branch1 = torch.nn.Sequential(
            torch.nn.Linear(64, 32), torch.nn.ReLU()
        )
        self.branch2 = torch.nn.Sequential(
            torch.nn.Linear(128, 64), torch.nn.ReLU()
        )
        self.buffer = torch.ones(32)

    def forward(self, x1, x2):
        out1 = self.branch1(x1)
        out2 = self.branch2(x2)
        return (out1 + self.buffer, out2)

example_args = (torch.randn(32, 64), torch.randn(32, 128))

# Create a dynamic batch size
batch = Dim("batch")
# Specify that the first dimension of each input is that batch size
dynamic_shapes = {"x1": {0: dim}, "x2": {0: batch}}

exported_program: torch.export.ExportedProgram = export(
    M(), args=example_args, dynamic_shapes=dynamic_shapes
)
print(exported_program)
```

```python
ExportedProgram:
class GraphModule(torch.nn.Module):
    def forward(self, p_branch1_0_weight: "f32[32, 64]", p_branch1_0_bias: "f32[32]", p_branch2_0_weight: "f32[64, 128]", p_branch2_0_bias: "f32[64]", c_buffer: "f32[32]", x1: "f32[s0, 64]", x2: "f32[s0, 128]"):

         # code: out1 = self.branch1(x1)
        linear: "f32[s0, 32]" = torch.ops.aten.linear.default(x1, p_branch1_0_weight, p_branch1_0_bias)
        relu: "f32[s0, 32]" = torch.ops.aten.relu.default(linear)

         # code: out2 = self.branch2(x2)
        linear_1: "f32[s0, 64]" = torch.ops.aten.linear.default(x2, p_branch2_0_weight, p_branch2_0_bias)
        relu_1: "f32[s0, 64]" = torch.ops.aten.relu.default(linear_1)

         # code: return (out1 + self.buffer, out2)
        add: "f32[s0, 32]" = torch.ops.aten.add.Tensor(relu, c_buffer)
        return (add, relu_1)

Range constraints: {s0: VR[0, int_oo]}
```

Some additional things to note:

- Through the {func}`torch.export.Dim` API and the `dynamic_shapes` argument, we specified the first
  dimension of each input to be dynamic. Looking at the inputs `x1` and
  `x2`, they have a symbolic shape of (s0, 64) and (s0, 128), instead of
  the (32, 64) and (32, 128) shaped tensors that we passed in as example inputs.
  `s0` is a symbol representing that this dimension can be a range
  of values.
- `exported_program.range_constraints` describes the ranges of each symbol
  appearing in the graph. In this case, we see that `s0` has the range
  [0, int_oo]. For technical reasons that are difficult to explain here, they are
  assumed to be not 0 or 1. This is not a bug, and does not necessarily mean
  that the exported program will not work for dimensions 0 or 1. See
  [The 0/1 Specialization Problem](https://docs.google.com/document/d/16VPOa3d-Liikf48teAOmxLc92rgvJdfosIy-yoT38Io/edit?fbclid=IwAR3HNwmmexcitV0pbZm_x1a4ykdXZ9th_eJWK-3hBtVgKnrkmemz6Pm5jRQ#heading=h.ez923tomjvyk)
  for an in-depth discussion of this topic.


In the example, we used `Dim("batch")` to create a dynamic dimension. This is
the most explicit way to specify dynamism. We can also use `Dim.DYNAMIC` and
`Dim.AUTO` to specify dynamism. We will go over both methods in the next section.

### Named Dims

For every dimension specified with `Dim("name")`, we will allocate a symbolic
shape. Specifying a `Dim` with the same name will result in the same symbol
to be generated. This allows users to specify what symbols are allocated for
each input dimension.

```python
batch = Dim("batch")
dynamic_shapes = {"x1": {0: dim}, "x2": {0: batch}}
```

For each `Dim`, we can specify minimum and maximum values. We also allow
specifying relations between `Dim`s in univariate linear expressions: `A * dim + B`.
This allows users to specify more complex constraints like integer divisibility
for dynamic dimensions. These features allow for users to place explicit
restrictions on the dynamic behavior of the `ExportedProgram` produced.

```python
dx = Dim("dx", min=4, max=256)
dh = Dim("dh", max=512)
dynamic_shapes = {
    "x": (dx, None),
    "y": (2 * dx, dh),
}
```

However, `ConstraintViolationErrors` will be raised if the while tracing, we emit guards
that conflict with the relations or static/dynamic specifications given. For
example, in the above specification, the following is asserted:

* `x.shape[0]` is to have range `[4, 256]`, and related to `y.shape[0]` by `y.shape[0] == 2 * x.shape[0]`.
* `x.shape[1]` is static.
* `y.shape[1]` has range `[0, 512]`, and is unrelated to any other dimension.

If any of these assertions are found to be incorrect while tracing (ex.
`x.shape[0]` is static, or `y.shape[1]` has a smaller range, or
`y.shape[0] != 2 * x.shape[0]`), then a `ConstraintViolationError` will be
raised, and the user will need to change their `dynamic_shapes` specification.

### Dim Hints

Instead of explicitly specifying dynamism using `Dim("name")`, we can let
`torch.export` infer the ranges and relationships of the dynamic values using
`Dim.DYNAMIC`. This is also a more convenient way to specify dynamism when you
don't know specifically *how* dynamic your dynamic values are.

```python
dynamic_shapes = {
    "x": (Dim.DYNAMIC, None),
    "y": (Dim.DYNAMIC, Dim.DYNAMIC),
}
```

We can also specify min/max values for `Dim.DYNAMIC`, which will serve as hints
to export. But if while tracing export found the range to be different, it will
automatically update the range without raising an error. We also cannot specify
relationships between dynamic values. Instead, this will be inferred by export,
and exposed to users through an inspection of assertions within the graph.  In
this method of specifying dynamism, `ConstraintViolationErrors` will **only** be
raised if the specified value is inferred to be **static**.

An even more convenient way to specify dynamism is to use `Dim.AUTO`, which will
behave like `Dim.DYNAMIC`, but will **not** raise an error if the dimension is
inferred to be static. This is useful for when you have no idea what the dynamic
values are, and want to export the program with a "best effort" dynamic approach.

### ShapesCollection

When specifying which inputs are dynamic via `dynamic_shapes`, we must specify
the dynamism of every input. For example, given the following inputs:

```python
args = {"x": tensor_x, "others": [tensor_y, tensor_z]}
```

we would need to specify the dynamism of `tensor_x`, `tensor_y`, and `tensor_z`
along with the dynamic shapes:

```python
# With named-Dims
dim = torch.export.Dim(...)
dynamic_shapes = {"x": {0: dim, 1: dim + 1}, "others": [{0: dim * 2}, None]}

torch.export(..., args, dynamic_shapes=dynamic_shapes)
```

However, this is particularly complicated as we need to specify the
`dynamic_shapes` specification in the same nested input structure as the input
arguments. Instead, an easier way to specify dynamic shapes is with the helper
utility {class}`torch.export.ShapesCollection`, where instead of specifying the
dynamism of every single input, we can just assign directly which input
dimensions are dynamic.

```python
dim = torch.export.Dim(...)
dynamic_shapes = torch.export.ShapesCollection()
dynamic_shapes[tensor_x] = (dim, dim + 1, 8)
dynamic_shapes[tensor_y] = {0: dim * 2}

torch.export(..., args, dynamic_shapes=dynamic_shapes)
```

### AdditionalInputs

In the case where you don't know how dynamic your inputs are, but you have an
ample set of testing or profiling data that can provide a fair sense of
representative inputs for a model, you can use
{class}`torch.export.AdditionalInputs` in place of `dynamic_shapes`. You can
specify all the possible inputs used to trace the program, and
`AdditionalInputs` will infer which inputs are dynamic based on which input
shapes are changing.

Example:

```python
args0, kwargs0 = ...  # example inputs for export

# other representative inputs that the exported program will run on
dynamic_shapes = torch.export.AdditionalInputs()
dynamic_shapes.add(args1, kwargs1)
...
dynamic_shapes.add(argsN, kwargsN)

torch.export(..., args0, kwargs0, dynamic_shapes=dynamic_shapes)
```

## Serialization

To save the `ExportedProgram`, users can use the {func}`torch.export.save` and
{func}`torch.export.load` APIs. The resulting file is a zipfile with a specific
structure. The details of the structure are defined in the
{ref}`PT2 Archive Spec <export.pt2_archive>`.

An example:

```python
import torch
import io

class MyModule(torch.nn.Module):
    def forward(self, x):
        return x + 10

exported_program = torch.export.export(MyModule(), torch.randn(5))

torch.export.save(exported_program, 'exported_program.pt2')
saved_exported_program = torch.export.load('exported_program.pt2')
```

(training-export)=

## Export IR, Decompositions

The graph produced by `torch.export` returns a graph containing only ATen
operators, which are the basic unit of computation in PyTorch. As there are over
3000 ATen operators, export provides a way to narrow down the operator set used
in the graph based on certain characteristics, creating different IRs.

By default, export produces the most generic IR which contains all ATen
operators, including both functional and non-functional operators. A functional
operator is one that does not contain any mutations or aliasing of the inputs.
This operator set also allows you to train in eager PyTorch Autograd.

However, if you want to use the IR for inference, or decrease the amount of
operators being used, you can lower the graph through the {func}`ExportedProgram.run_decompositions` API.

* By specifying an empty set to the `decomp_table` argument, we get rid of all
    non-functional operators, reducing the operator set to ~2000 operators. This
    is ideal for inference cases as there are no mutations or aliasing, making
    it easy to write optimization passes.
* By specifying None to `decomp_table` argument, we can reduce the operator set
    to just the {ref}`Core ATen Operator Set <torch.compiler_ir>`, which is a
    collection of only ~180 operators. This IR is optimal for backends who do
    not want to reimplement all ATen operators.

```python
class ConvBatchnorm(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 3, 1, 1)
        self.bn = torch.nn.BatchNorm2d(3)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return (x,)

mod = ConvBatchnorm()
inp = torch.randn(1, 1, 3, 3)

ep_for_training = torch.export.export(mod, (inp,))
ep_for_inference = ep_for_training.run_decompositions(decomp_table={})
```

A tutorial on how to use this API can be found
[here](https://docs.pytorch.org/tutorials/intermediate/torch_export_tutorial.html#ir-decompositions).

(limitations-of-torch-export)=

## Limitations of torch.export

As `torch.export` is a one-shot process for capturing a computation graph from
a PyTorch program, it might ultimately run into untraceable parts of programs as
it is nearly impossible to support tracing all PyTorch and Python features. In
the case of `torch.compile`, an unsupported operation will cause a "graph
break" and the unsupported operation will be run with default Python evaluation.
In contrast, `torch.export` will require users to provide additional
information or rewrite parts of their code to make it traceable.

{ref}`Draft-export <export.draft_export>` is a great resource for listing out
graphs breaks that will be encountered when tracing the program, along with
additional debug information to solve those errors.

{ref}`ExportDB <torch.export_db>` is also great resource for learning about the
kinds of programs that are supported and unsupported, along with ways to rewrite
programs to make them traceable.

### TorchDynamo unsupported

When using `torch.export` with `strict=True`, this will use TorchDynamo to
evaluate the program at the Python bytecode level to trace the program into a
graph. Compared to previous tracing frameworks, there will be significantly
fewer rewrites required to make a program traceable, but there will still be
some Python features that are unsupported. An option to get past dealing with
this graph breaks is by using
{ref}`non-strict export <non-strict-export>` through changing the `strict` flag
to `strict=False`.

(data-shape-dependent-control-flow)=

### Data/Shape-Dependent Control Flow

Graph breaks can also be encountered on data-dependent control flow (`if
x.shape[0] > 2`) when shapes are not being specialized, as a tracing compiler cannot
possibly deal with without generating code for a combinatorially exploding
number of paths. In such cases, users will need to rewrite their code using
special control flow operators. Currently, we support {ref}`torch.cond <cond>`
to express if-else like control flow (more coming soon!).

You can also refer to this
[tutorial](https://docs.pytorch.org/tutorials/intermediate/torch_export_tutorial.html#data-dependent-errors)
for more ways of addressing data-dependent errors.

### Missing Fake/Meta Kernels for Operators

When tracing, a FakeTensor kernel (aka meta kernel) is required for all
operators. This is used to reason about the input/output shapes for this
operator.

Please see this [tutorial](https://docs.pytorch.org/tutorials/advanced/custom_ops_landing_page.html)
for more details.

In the unfortunate case where your model uses an ATen operator that is does not
have a FakeTensor kernel implementation yet, please file an issue.

## Read More

```{toctree}
:caption: Additional Links for Export Users
:maxdepth: 1

export/programming_model
export/ir_spec
export/pt2_archive
export/draft_export
cond
generated/exportdb/index
torch.compiler_aot_inductor
torch.compiler_ir
```

```{toctree}
:caption: Deep Dive for PyTorch Developers
:maxdepth: 1

torch.compiler_dynamic_shapes
torch.compiler_fake_tensor
torch.compiler_transformations
```

## API Reference

```{eval-rst}
.. automodule:: torch.export

.. autofunction:: torch.export.export

.. autoclass:: torch.export.ExportedProgram
   :members:
   :exclude-members: __init__

.. automodule:: torch.export.dynamic_shapes
   :members: Dim, ShapesCollection, AdditionalInputs, refine_dynamic_shapes_from_suggested_fixes

.. autofunction:: torch.export.save

.. autofunction:: torch.export.load

.. autofunction:: torch.export.pt2_archive._package.package_pt2

.. autofunction:: torch.export.pt2_archive._package.load_pt2

.. autofunction:: torch.export.draft_export

.. automodule:: torch.export.unflatten
    :members:

.. autofunction:: torch.export.register_dataclass

.. automodule:: torch.export.decomp_utils
   :members:
   :ignore-module-all:
   :undoc-members:

.. automodule:: torch.export.experimental
   :members:
   :ignore-module-all:

.. automodule:: torch.export.passes
   :members:

.. automodule:: torch.export.pt2_archive
   :members:
   :ignore-module-all:

.. automodule:: torch.export.pt2_archive.constants
   :members:
   :ignore-module-all:

.. automodule:: torch.export.exported_program
   :members:
   :ignore-module-all:
   :exclude-members: ExportedProgram

.. automodule:: torch.export.custom_ops
   :members:
   :ignore-module-all:

.. automodule:: torch.export.custom_obj
   :members:
   :ignore-module-all:

.. automodule:: torch.export.graph_signature
   :members:
   :ignore-module-all:
   :undoc-members:
```
