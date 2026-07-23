(export.programming_model)=

# torch.export Programming Model

This document aims to explain the behaviors and capabilities of
{func}`torch.export.export`. It is intended to help build your intuition
for how {func}`torch.export.export` handles code.

## Basics of Tracing

{func}`torch.export.export` captures a graph representing your model by
tracing its execution on "example" inputs and recording the PyTorch operations
and conditions observed along the traced path. This graph can then be run
on different inputs as long as they satisfy the same conditions.

The basic output of {func}`torch.export.export` is a single graph of PyTorch
operations, with associated metadata. The exact format of this output is
covered in the {ref}`export IR spec <export.ir_spec>`.

(non-strict-export)=

### Strict vs. Non-Strict Tracing

{func}`torch.export.export` provides two modes of tracing.

In *non-strict mode*, we trace through the program using the normal Python
interpreter. Your code executes exactly as it would in eager mode; the only
difference is that all Tensors are replaced by
[fake Tensors](https://docs.pytorch.org/docs/main/user_guide/torch_compiler/torch.compiler_fake_tensor.html),
**which have shapes and other forms of metadata but no data**, wrapped in
[Proxy objects](https://pytorch.org/docs/main/fx.html) that record all
operations on them into a graph. We also capture
[conditions on Tensor shapes](https://docs.pytorch.org/docs/main/user_guide/torch_compiler/torch.compiler_dynamic_shapes.html#the-guard-model)
**that guard the correctness of the generated code**.

In *strict mode*, we first trace through the program using
{ref}`TorchDynamo <torch.compiler_dynamo_deepdive>`, a Python bytecode
analysis engine. TorchDynamo does not actually execute your Python code.
Instead, it symbolically analyzes it and builds a graph based on the results.
On the one hand, this analysis allows {func}`torch.export.export` to provide
additional guarantees on Python-level safety (beyond capturing conditions on
Tensor shapes, as in non-strict mode). On the other hand, not all Python
features are supported by this analysis.

Although currently the default mode of tracing is strict, **we strongly
recommend using non-strict**, which will soon become the default.
For most models, conditions on Tensor shapes are enough for soundness, and
the additional guarantees on Python-level safety have no impact; at the same
time, the possibility of hitting unsupported Python features in TorchDynamo
presents an unnecessary risk.

In the rest of this document we assume we are tracing in
[non-strict mode](https://pytorch.org/docs/main/export.html#non-strict-export);
in particular, we assume that **all Python features are supported**.

## Values: Static vs. Dynamic

A key concept in understanding the behavior of {func}`torch.export.export` is
the difference between *static* and *dynamic* values.

### Static Values

A *static* value is a value that is **fixed at export time and cannot change
between executions of the exported program**. When the value is encountered
during tracing, we treat it as a constant and hard-code it into the graph.

When an operation is performed (e.g. `x + y`) and all inputs are static,
the output of the operation is directly hard-coded into the graph and the
operation does not show up (i.e. it gets "constant-folded").

When a value has been hard-coded into the graph, we say that the graph has
been *specialized* to that value. For example:

```python
import torch

class MyMod(torch.nn.Module):
    def forward(self, x, y):
        z = y + 7
        return x + z

m = torch.export.export(MyMod(), (torch.randn(1), 3))
print(m.graph_module.code)

"""
def forward(self, arg0_1, arg1_1):
    add = torch.ops.aten.add.Tensor(arg0_1, 10);  arg0_1 = None
    return (add,)

"""
```

Here, we provide `3` as the traced value for `y`; it is treated as a static
value and added to `7`, burning in the static value `10` in the graph.

### Dynamic Values

A *dynamic* value is one that **can change from run to run**. It behaves just
like a "normal" function argument: you can pass different inputs and expect
your function to do the right thing.

### Which values are static vs. dynamic?

Whether a value is static or dynamic depends on its type:

- For Tensor:

  - Tensor *data* is treated as dynamic.

  - Tensor *shapes* can be treated by the system as static or dynamic.

    - By default, shapes of all input Tensors are considered static.
      The user can override this behavior for any input Tensor by specifying
      a [dynamic shape](https://pytorch.org/docs/main/export.html#expressing-dynamism)
      for it.
    - Tensors that are part of module state, i.e., parameters and buffers,
      always have static shapes.

  - Other forms of Tensor *metadata* (e.g. `device`, `dtype`) are static.

- Python *primitives* (`int`, `float`, `bool`, `str`, `None`) are static.

  - There are dynamic variants for some primitive types (`SymInt`,
    `SymFloat`, `SymBool`). Typically users do not have to deal with them.
  - Users can specify integer inputs as dynamic by specifying
    a [dynamic shape](https://pytorch.org/docs/main/export.html#expressing-dynamism)
    for it.

- For Python *standard containers* (`list`, `tuple`, `dict`, `namedtuple`):

  - The structure (i.e., length for `list` and `tuple` values, and key
    sequence for `dict` and `namedtuple` values) is static.
  - The contained elements have these rules applied to them recursively
    (basically the
    [PyTree](https://jax.readthedocs.io/en/latest/pytrees.html) scheme)
    with leaves that are either Tensor or primitive types.

- Other *classes* (including data classes) can be registered with PyTree
  (see below), and follow the same rules as the standard containers.

## Input types

Inputs will be treated as either static or dynamic, based on their type
(as explained above).

- A static input will get hard-coded into the graph, and passing a different
  value at run time will result in an error. Recall that these are mostly
  values of primitive types.
- A dynamic input behaves like a "normal" function input. Recall that these
  are mostly values of Tensor types.

By default, the types of inputs you can use for your program are:

- Tensor
- Python primitives (`int`, `float`, `bool`, `str`, `None`)
- Python standard containers (`list`, `tuple`, `dict`, `namedtuple`)

### Custom Input Types (PyTree)

In addition, you can also define your own (custom) class and use it as an
input type, but you will need to register such a class as a PyTree.

Here's an example of using an utility to register a dataclass that is used as
an input type.

```python
@dataclass
class Input:
    f: torch.Tensor
    p: torch.Tensor

import torch.utils._pytree as pytree
pytree.register_dataclass(Input)

class M(torch.nn.Module):
    def forward(self, x: Input):
        return x.f + 1

torch.export.export(M(), (Input(f=torch.ones(10, 4), p=torch.zeros(10, 4)),))
```

### Optional input types

For optional inputs to the program that are not passed in,
{func}`torch.export.export` will specialize to their default values. As a
result, the exported program will require users to explicitly pass in all
arguments, and will lose the defaulting behavior. For example:

```python
class M(torch.nn.Module):
    def forward(self, x, y=None):
        if y is not None:
            return y * x
        return x + x

# Optional input is passed in
ep = torch.export.export(M(), (torch.randn(3, 3), torch.randn(3, 3)))
print(ep)
"""
ExportedProgram:
    class GraphModule(torch.nn.Module):
        def forward(self, x: "f32[3, 3]", y: "f32[3, 3]"):
            # File: /data/users/angelayi/pytorch/moo.py:15 in forward, code: return y * x
            mul: "f32[3, 3]" = torch.ops.aten.mul.Tensor(y, x);  y = x = None
            return (mul,)
"""

# Optional input is not passed in
ep = torch.export.export(M(), (torch.randn(3, 3),))
print(ep)
"""
ExportedProgram:
    class GraphModule(torch.nn.Module):
        def forward(self, x: "f32[3, 3]", y):
            # File: /data/users/angelayi/pytorch/moo.py:16 in forward, code: return x + x
            add: "f32[3, 3]" = torch.ops.aten.add.Tensor(x, x);  x = None
            return (add,)
"""
```

## Control Flow: Static vs. Dynamic

Control flow is supported by {func}`torch.export.export`. The behavior of
control flow depends on whether the value you are branching on is static or
dynamic.

### Static Control Flow

**Python control flow over static values is supported transparently**. (Recall
that static values include static shapes, so control flow over static shapes
is also covered by this case.)

As mentioned above, we "burn in" static values, so the exported graph will
never see any control flow over static values.

In the case of an `if` statement, we will continue tracing the branch taken
at export time. In the case of a `for` or `while` statement, we will continue
tracing by unrolling the loop.

### Dynamic Control Flow: Shape-Dependent vs. Data-Dependent

When the value involved in a control flow is dynamic, it could depend on
dynamic shapes or dynamic data. Given that the compiler traces with
information on shapes rather than data, the implications on the programming
model are different in these cases.

#### Dynamic Shape-Dependent Control Flow

When the value involved in a control flow is a
[dynamic shape](https://docs.pytorch.org/docs/main/user_guide/torch_compiler/torch.compiler_dynamic_shapes.html),
in most cases **we will also know the concrete value of the dynamic shape
during tracing**: see the following section for more details on how the
compiler tracks this information.

In these cases we say that the control flow is shape-dependent. **We use the
concrete value of the dynamic shape to evaluate the condition** to either
`True` or `False` and continue tracing (as discussed above), additionally
emitting a guard corresponding to the condition just evaluated.

Otherwise the control flow is considered data-dependent. We cannot evaluate
the condition to either `True` or `False`, so cannot continue tracing and have to
raise an error at export time. See next section.

#### Dynamic Data-Dependent Control Flow

**Data-dependent control flow over dynamic values is supported, but you must
use one of PyTorch's explicit operators** to continue tracing. Using Python
control flow statements over dynamic values is not permitted, because the
compiler cannot evaluate the conditions necessary to continue tracing and
thus an error must be raised at export time.

We provide **operators to express general conditionals and loops over dynamic
values**, e.g., `torch.cond`, `torch.map`. Note that you only need to use these
if you truly want *data-dependent control flow*.

Here's an example of an `if` statement on a data-dependent condition,
`x.sum() > 0`, where `x` is an input Tensor, rewritten using `torch.cond`.
Instead of having to decide which branch to trace, now both branches are
traced.

```python
class M_old(torch.nn.Module):
    def forward(self, x):
        if x.sum() > 0:
            return x.sin()
        else:
            return x.cos()

class M_new(torch.nn.Module):
    def forward(self, x):
        return torch.cond(
            pred=x.sum() > 0,
            true_fn=lambda x: x.sin(),
            false_fn=lambda x: x.cos(),
            operands=(x,),
        )
```

A special case of data-dependent control flow is where it involves a
[data-dependent dynamic shape](https://docs.pytorch.org/docs/main/user_guide/torch_compiler/torch.compiler_dynamic_shapes.html#unbacked-symints):
typically, the shape of some intermediate Tensor that depends on input data
rather than on input shapes (thus not shape-dependent). Instead of using a
control flow operator, in this case you can provide an assertion that decides
whether the condition is `True` or `False`. Given such an assertion, we can
continue tracing, emitting a guard as above.

We provide **operators to express assertions on dynamic shapes**, e.g.,
`torch._check`. Note that you only need to use this when there is control
flow on data-dependent dynamic shapes.

Here's an example of an `if` statement on a condition involving a
data-dependent dynamic shape, `nz.shape[0] > 0`, where `nz` is the result of
calling {func}`torch.nonzero`, an operator whose output shape depends on input
data. Instead of rewriting it, you can add an assertion using `torch._check`
to effectively decide which branch to trace.

```python
class M_old(torch.nn.Module):
    def forward(self, x):
        nz = x.nonzero()
        if nz.shape[0] > 0:
            return x.sin()
        else:
            return x.cos()

class M_new(torch.nn.Module):
    def forward(self, x):
        nz = x.nonzero()
        torch._check(nz.shape[0] > 0)
        if nz.shape[0] > 0:
            return x.sin()
        else:
            return x.cos()
```

## Basics of Symbolic Shapes

During tracing, dynamic Tensor shapes and conditions over them are encoded as
"symbolic expressions." (In contrast, static Tensor shapes and conditions
over them are simply `int` and `bool` values.)

A *symbol* is like a variable; it describes a dynamic Tensor shape.

As tracing proceeds, shapes of intermediate Tensors may be described by more
general expressions, typically involving integer arithmetic operators. This
is because **for most PyTorch operators, shapes of output Tensors can be
described as functions of shapes of input Tensors**. For example, the shape of
the output of {func}`torch.cat` is the sum of the shapes of its inputs.

Moreover, as we encounter control flow in the program, we create boolean
expressions, typically involving relational operators, describing conditions
along the traced path. These **expressions are evaluated to decide which path
to trace through the program**, and recorded in a
[shape environment](https://docs.pytorch.org/docs/main/user_guide/torch_compiler/torch.compiler_dynamic_shapes.html#overall-architecture)
to guard the correctness of the traced path and to evaluate subsequently
created expressions.

We briefly introduce these subsystems next.

### Fake Implementations of PyTorch Operators

Recall that during tracing, we are executing the program with
[fake Tensors](https://docs.pytorch.org/docs/main/user_guide/torch_compiler/torch.compiler_fake_tensor.html),
which have no data. In general we cannot call the actual implementations of
PyTorch operators with fake Tensors. Thus each operator needs to have an
additional fake (a.k.a. "meta") implementation, which inputs and outputs fake
Tensors, that matches the behavior of the actual implementation in terms of
shapes and other forms of metadata carried by fake Tensors.

For example, note how the fake implementation of {func}`torch.index_select`
computes the shape of the output using the shape of the input (while ignoring
input data and returning empty output data).

```python
def meta_index_select(self, dim, index):
    result_size = list(self.size())
    if self.dim() > 0:
        result_size[dim] = index.numel()
    return self.new_empty(result_size)
```

#### Shape Propagation: Backed vs. Unbacked Dynamic Shapes

Shapes are propagated using fake implementations of PyTorch operators.

A key concept to understand the propagation of dynamic shapes in particular
is the difference between *backed* and *unbacked* dynamic shapes: we know the
concrete values of the former but not the latter.

Propagation of shapes, including tracking backed and unbacked dynamic shapes,
proceeds as follows:

- The shapes of Tensors representing inputs can be static or dynamic. When
  dynamic, they are described by symbols; moreover, **such symbols are backed
  since we also know their concrete values given the "real" example inputs
  provided by the user at export time**.

- The output shape of an operator is computed by its fake implementation, and
  is either static or dynamic. When dynamic, in general it is described by a
  symbolic expression. Moreover:

  - If the output shape depends only on input shapes, it is either static or
    backed dynamic whenever the input shapes are all static or backed dynamic.
  - On the other hand, **if the output shape depends on input data**, it is
    necessarily dynamic, and moreover, **because we cannot know its concrete
    value it is unbacked**.

### Control Flow: Guards and Assertions

When a condition on shapes is encountered, it either involves only static
shapes, in which case it is a `bool`, or it involves dynamic shapes, in which
case it is a symbolic boolean expression. For the latter:

- When the condition involves only backed dynamic shapes, we can use the
  concrete values of those dynamic shapes to evaluate the condition to `True`
  or `False`. We can then add a guard to the shape environment that states
  that the corresponding symbolic boolean expression is `True` or `False`,
  and continue tracing.
- Otherwise the condition involves unbacked dynamic shapes. In general we
  cannot evaluate such a condition without additional information; thus we
  cannot continue tracing, and we must raise an error at export time. The
  user is expected to use an explicit PyTorch operator for tracing to
  continue. This information is added as a guard in the shape environment,
  and can also possibly help evaluate other subsequently encountered
  conditions to `True` or `False`.

Once the model is exported, **any guards on backed dynamic shapes can be
understood as conditions on input dynamic shapes**. These are verified against
a dynamic shape specification that must have been provided to export,
describing conditions on dynamic shapes that not only example inputs but also
all future inputs are expected to satisfy for the generated code to be
correct. More precisely, the dynamic shape specification must logically imply
the generated guards, otherwise an error is raised at export time (along with
suggested fixes to the dynamic shape specification). On the other hand, when
there are no generated guards on backed dynamic shapes (in particular, when
all shapes are static) no dynamic shape specification needs to be provided to
export. In general, the dynamic shape specification is converted to runtime
assertions on the inputs of the generated code.

Finally, **any guards on unbacked dynamic shapes are converted to "inline"
runtime assertions**. These are added in the generated code at the locations
where those unbacked dynamic shapes were created: typically, right after
data-dependent operator calls.

## Allowed PyTorch operators

All PyTorch operators are permitted.

### Custom operators

In addition, you can define and use
[custom operators](https://pytorch.org/tutorials/advanced/python_custom_ops#python-custom-ops-tutorial).
Defining a custom operator includes defining a fake implementation for it,
just like any other PyTorch operator (see previous section).

Here's an example of a custom `sin` operator that wraps NumPy, and its
registered (trivial) fake implementation.

```python
@torch.library.custom_op("mylib::sin", mutates_args=())
def sin(x: Tensor) -> Tensor:
    x_np = x.numpy()
    y_np = np.sin(x_np)
    return torch.from_numpy(y_np)

@torch.library.register_fake("mylib::sin")
def _(x: Tensor) -> Tensor:
    return torch.empty_like(x)
```

**Sometimes your custom operator's fake implementation will involve
data-dependent shapes**. Here's how a fake implementation for a custom
`nonzero` might look like.

```python
...

@torch.library.register_fake("mylib::custom_nonzero")
def _(x):
    nnz = torch.library.get_ctx().new_dynamic_size()
    shape = [nnz, x.dim()]
    return x.new_empty(shape, dtype=torch.int64)
```

## Module State: Reads vs. Updates

Module states include parameters, buffers, and regular attributes.

- A regular attribute can be of any type.
- On the other hand, parameters and buffers are always Tensors.

Module states can be dynamic or static, based on their types as outlined
above. For example, `self.training` is a `bool`, which means it is static; on
the other hand, any parameter or buffer is dynamic.

The *shapes* of any Tensors contained in module states cannot be dynamic, i.e.,
those shapes are fixed at export time, and cannot change between executions
of the exported program.

### Access rules

**All module states must be initialized**. Accessing a module state that is
not already initialized causes an error to be raised at export time.

**Reading module states is always permitted**.

Updating module states is possible, but must follow the rules below:

- **A static regular attribute** (e.g., of primitive type) **can be updated**.
  Reads and updates can be freely interleaved, and as expected, any reads
  will always see the values of the latest updates. Because these attributes
  are static, we will also burn the values in, so the generated code will not
  have any instructions to actually "get" or "set" such attributes.
- **A dynamic regular attribute** (e.g., of Tensor type) **cannot be updated**.
  To do so, it must be registered as a buffer during module initialization.
- **A buffer can be updated**, where the updating can be in-place (e.g.,
  `self.buffer[:] = ...`) or not (e.g., `self.buffer = ...`).
- **A parameter cannot be updated**. Typically parameters are updated only
  during training, not during inference. We recommend exporting with
  {func}`torch.no_grad` to avoid parameter updates at export time.

### Effects of functionalization

Any dynamic module state that is read and/or updated is "lifted"
(respectively) as an input and/or output of the generated code.

The exported program stores, along with the generated code, the initial
values of parameters and buffers and the constant values of other Tensor
attributes.
