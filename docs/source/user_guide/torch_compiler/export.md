---
file_format: mystnb
kernelspec:
  name: python3
mystnb:
  execution_timeout: 30
  execution_show_tb: True
  merge_streams: True
---

(torch.export)=

# torch.export

## Overview

{func}`torch.export.export` takes a {class}`torch.nn.Module` and produces a traced graph
representing only the Tensor computation of the function in an Ahead-of-Time
(AOT) fashion, which can subsequently be executed with different outputs or
serialized.

```{code-cell}
import torch
from torch.export import export, ExportedProgram

class Mod(torch.nn.Module):
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        a = torch.sin(x)
        b = torch.cos(y)
        return a + b

example_args = (torch.randn(10, 10), torch.randn(10, 10))

exported_program: ExportedProgram = export(Mod(), args=example_args)
print(exported_program)
```

`torch.export` produces a clean intermediate representation (IR) with the
following invariants. More specifications about the IR can be found
{ref}`here <export.ir_spec>`.

- **Soundness**: It is guaranteed to be a sound representation of the original
  program, and maintains the same calling conventions of the original program.
- **Normalized**: There are no Python semantics within the graph. Submodules
  from the original programs are inlined to form one fully flattened
  computational graph.
- **Graph properties**: By default, the graph may contain both functional and
  non-functional operators (including mutations). To obtain a purely functional
  graph, use `run_decompositions()` which removes mutations and aliasing.
- **Metadata**: The graph contains metadata captured during tracing, such as a
  stacktrace from user's code.

Under the hood, `torch.export` leverages the following latest technologies:

- **TorchDynamo (torch._dynamo)** is an internal API that uses a CPython feature
  called the Frame Evaluation API to safely trace PyTorch graphs. This
  provides a massively improved graph capturing experience, with much fewer
  rewrites needed in order to fully trace the PyTorch code.
- **AOT Autograd** ensures the graph is decomposed/lowered to the ATen operator
  set. When using `run_decompositions()`, it can also provide functionalization.
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

```{code-cell}
import torch
from torch.export import export, ExportedProgram

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

exported_program: ExportedProgram = export(
    M(), args=example_args, kwargs=example_kwargs
)
print(exported_program)

# To run the exported program, we can use the `module()` method
print(exported_program.module()(torch.randn(1, 3, 256, 256), constant=torch.ones(1, 16, 256, 256)))
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
**static**, and specializing the exported program to those dimensions. One
consequence of this is that at runtime, the program won’t work on inputs with
different shapes, even if they’re valid in eager mode.

An example:

```{code-cell}
import torch
import traceback as tb

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

ep = torch.export.export(M(), example_args)
print(ep)

example_args2 = (torch.randn(64, 64), torch.randn(64, 128))
try:
    ep.module()(*example_args2)  # fails
except Exception:
    tb.print_exc()
```


However, some dimensions, such as a batch dimension, can be dynamic and vary
from run to run. Such dimensions must be specified by using the
{func}`torch.export.Dim()` API to create them and by passing them into
{func}`torch.export.export()` through the `dynamic_shapes` argument.

```{code-cell}
import torch

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
batch = torch.export.Dim("batch")
# Specify that the first dimension of each input is that batch size
dynamic_shapes = {"x1": {0: batch}, "x2": {0: batch}}

ep = torch.export.export(
    M(), args=example_args, dynamic_shapes=dynamic_shapes
)
print(ep)

example_args2 = (torch.randn(64, 64), torch.randn(64, 128))
ep.module()(*example_args2)  # success
```

Some additional things to note:

- Through the {func}`torch.export.Dim` API and the `dynamic_shapes` argument, we specified the first
  dimension of each input to be dynamic. Looking at the inputs `x1` and
  `x2`, they have a symbolic shape of `(s0, 64)` and `(s0, 128)`, instead of
  the `(32, 64)` and `(32, 128)` shaped tensors that we passed in as example inputs.
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

```{code-cell}
import torch

class M(torch.nn.Module):
    def forward(self, inp):
        x = inp["x"] * 1
        y = inp["others"][0] * 2
        z = inp["others"][1] * 3
        return x, y, z

tensor_x = torch.randn(3, 4, 8)
tensor_y = torch.randn(6)
tensor_z = torch.randn(6)
args = {"x": tensor_x, "others": [tensor_y, tensor_z]}

dim = torch.export.Dim("dim")
sc = torch.export.ShapesCollection()
sc[tensor_x] = (dim, dim + 1, 8)
sc[tensor_y] = {0: dim * 2}

print(sc.dynamic_shapes(M(), (args,)))
ep = torch.export.export(M(), (args,), dynamic_shapes=sc)
print(ep)
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

```{code-cell}
import dataclasses
import torch
import torch.utils._pytree as pytree

@dataclasses.dataclass
class D:
    b: bool
    i: int
    f: float
    t: torch.Tensor

pytree.register_dataclass(D)

class M(torch.nn.Module):
    def forward(self, d: D):
        return d.i + d.f + d.t

input1 = (D(True, 3, 3.0, torch.ones(3)),)
input2 = (D(True, 4, 3.0, torch.ones(4)),)
ai = torch.export.AdditionalInputs()
ai.add(input1)
ai.add(input2)

print(ai.dynamic_shapes(M(), input1))
ep = torch.export.export(M(), input1, dynamic_shapes=ai)
print(ep)
```

## Serialization

To save the `ExportedProgram`, users can use the {func}`torch.export.save` and
{func}`torch.export.load` APIs. The resulting file is a zipfile with a specific
structure. The details of the structure are defined in the
{ref}`PT2 Archive Spec <export.pt2_archive>`.

An example:

```python
import torch

class MyModule(torch.nn.Module):
    def forward(self, x):
        return x + 10

exported_program = torch.export.export(MyModule(), (torch.randn(5),))

torch.export.save(exported_program, 'exported_program.pt2')
saved_exported_program = torch.export.load('exported_program.pt2')
```

(training-export)=

## Export IR: Training vs Inference

The graph produced by `torch.export` returns a graph containing only
[ATen operators](https://pytorch.org/cppdocs/#aten), which are the basic unit of
computation in PyTorch. Export provides different IR levels based on your use case:

| IR Type | How to Obtain | Properties | Operator Count | Use Case |
|---------|---------------|------------|----------------|----------|
| Training IR | `torch.export.export()` (default) | May contain mutations | ~3000 | Training with autograd |
| Inference IR | `ep.run_decompositions(decomp_table={})` | Purely functional | ~2000 | Inference deployment |
| Core ATen IR | `ep.run_decompositions(decomp_table=None)` | Purely functional, highly decomposed | ~180 | Minimal backend support |

### Training IR (Default)

By default, export produces a **Training IR** which contains all ATen
operators, including both functional and non-functional (mutating) operators.
A functional operator is one that does not contain any mutations or aliasing
of the inputs, while non-functional operators may modify their inputs in-place.
You can find a list of all ATen operators
[here](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/native_functions.yaml)
and you can inspect if an operator is functional by checking
`op._schema.is_mutable`.

This Training IR, which may contain mutations, is designed for training use
cases and can be used with eager PyTorch Autograd.

```{code-cell}
import torch

class M(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 3, 1, 1)
        self.bn = torch.nn.BatchNorm2d(3)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return (x,)

ep_for_training = torch.export.export(M(), (torch.randn(1, 1, 3, 3),))
print(ep_for_training.graph_module.print_readable(print_output=False))
```

### Inference IR (via run_decompositions)

To obtain an **Inference IR** suitable for deployment, use the
{func}`ExportedProgram.run_decompositions` API. This method automatically:
1. Functionalizes the graph (removes all mutations and converts them to functional equivalents)
2. Optionally decomposes ATen operators based on the provided decomposition table

This produces a purely functional graph ideal for inference scenarios.

By specifying an empty decomposition table (`decomp_table={}`), you get just
the functionalization without additional decompositions. This produces an
Inference IR with ~2000 functional operators (compared to 3000+ in Training IR).

```{code-cell}
import torch

class M(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 3, 1, 1)
        self.bn = torch.nn.BatchNorm2d(3)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return (x,)

ep_for_training = torch.export.export(M(), (torch.randn(1, 1, 3, 3),))
with torch.no_grad():
    ep_for_inference = ep_for_training.run_decompositions(decomp_table={})
print(ep_for_inference.graph_module.print_readable(print_output=False))
```

As we can see, the previously in-place operator,
`torch.ops.aten.add_.default` has now been replaced with
`torch.ops.aten.add.default`, a functional operator.

### Core ATen IR

We can further lower the Inference IR to the
`Core ATen Operator Set <https://docs.pytorch.org/docs/main/user_guide/torch_compiler/torch.compiler_ir.html#core-aten-ir>`__,
which contains only ~180 operators. This is achieved by passing `decomp_table=None`
(which uses the default decomposition table) to `run_decompositions()`. This IR
is optimal for backends who want to minimize the number of operators they need
to implement.

```{code-cell}
import torch

class M(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 3, 1, 1)
        self.bn = torch.nn.BatchNorm2d(3)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return (x,)

ep_for_training = torch.export.export(M(), (torch.randn(1, 1, 3, 3),))
with torch.no_grad():
    core_aten_ir = ep_for_training.run_decompositions(decomp_table=None)
print(core_aten_ir.graph_module.print_readable(print_output=False))
```

We now see that `torch.ops.aten.conv2d.default` has been decomposed
into `torch.ops.aten.convolution.default`. This is because `convolution`
is a more "core" operator, as operations like `conv1d` and `conv2d` can be
implemented using the same op.

We can also specify our own decomposition behaviors:

```{code-cell}
class M(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 3, 1, 1)
        self.bn = torch.nn.BatchNorm2d(3)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return (x,)

ep_for_training = torch.export.export(M(), (torch.randn(1, 1, 3, 3),))

my_decomp_table = torch.export.default_decompositions()

def my_awesome_custom_conv2d_function(x, weight, bias, stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=1):
    return 2 * torch.ops.aten.convolution(x, weight, bias, stride, padding, dilation, False, [0, 0], groups)

my_decomp_table[torch.ops.aten.conv2d.default] = my_awesome_custom_conv2d_function
my_ep = ep_for_training.run_decompositions(my_decomp_table)
print(my_ep.graph_module.print_readable(print_output=False))
```

Notice that instead of `torch.ops.aten.conv2d.default` being decomposed
into `torch.ops.aten.convolution.default`, it is now decomposed into
`torch.ops.aten.convolution.default` and `torch.ops.aten.mul.Tensor`,
which matches our custom decomposition rule.

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

export/api_reference
export/programming_model
export/ir_spec
export/pt2_archive
export/draft_export
export/joint_with_descriptors
../../cond
../../generated/exportdb/index
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
