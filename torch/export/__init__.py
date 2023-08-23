from typing import Any, Callable, Dict, List, Optional, Tuple

import torch


__all__ = [
    "dynamic_dim",
    "export",
]


def dynamic_dim(t: torch.Tensor, index: int):
    """
    `dynamic_dim` constructs a `Constraint` object that describes the dynamism of
    a dimension `index` of tensor `t`. `Constraint` objects should be passed to
    `constraints` argument of `export()`.

    Specifically `dynamic_dim` can be used to express following types of dynamism.

    - Size of a dimension is dynamic and unbounded::

        t0 = torch.rand(2, 3)
        t1 = torch.rand(3, 4)

        # First dimension of t0 can be dynamic size rather than always being static size 2
        constraints = [dynamic_dim(t0, 0)]
        ep = export(fn, (t0, t1), constraints=constraints)

    - Size of a dimension is dynamic with a lower bound::

        t0 = torch.rand(10, 3)
        t1 = torch.rand(3, 4)

        # First dimension of t0 can be dynamic size with a lower bound of 5 (inclusive)
        # Second dimension of t1 can be dynamic size with a lower bound of 2 (exclusive)
        constraints = [
            dynamic_dim(t0, 0) >= 5,
            dynamic_dim(t1, 1) > 2,
        ]
        ep = export(fn, (t0, t1), constraints=constraints)

    - Size of a dimension is dynamic with an upper bound::

        t0 = torch.rand(10, 3)
        t1 = torch.rand(3, 4)

        # First dimension of t0 can be dynamic size with a upper bound of 16 (inclusive)
        # Second dimension of t1 can be dynamic size with a upper bound of 8 (exclusive)
        constraints = [
            dynamic_dim(t0, 0) <= 16,
            dynamic_dim(t1, 1) < 8,
        ]
        ep = export(fn, (t0, t1), constraints=constraints)

    - Size of a dimension is dynamic and it is always equal to size of another dynamic dimension::

        t0 = torch.rand(10, 3)
        t1 = torch.rand(3, 4)

        # Sizes of second dimension of t0 and first dimension are always equal
        constraints = [
            dynamic_dim(t0, 1) == dynamic_dim(t1, 0),
        ]
        ep = export(fn, (t0, t1), constraints=constraints)

    - Mix and match all types above as long as they do not express conflicting requirements

    Args:
        t (torch.Tensor): Example input tensor that have dynamic dimension size(s)
        index (int): Index of dynamic dimension

    Returns:
        A `Constraint` object that describes shape dynamism. It can be passed to `export()` so
        that `export()` does not assume static size of specified tensor, i.e. keeping it dynamic
        as a symbolic size rather than specializing according to size of example tracing input.

    """
    from torch._export import dynamic_dim

    return dynamic_dim(t, index)


def export(
    f: Callable,
    args: Tuple[Any],
    kwargs: Optional[Dict[str, Any]] = None,
    *,
    constraints: Optional[List["torch._dynamo.eval_frame.Constraint"]] = None,
) -> "torch._export.exported_program.ExportedProgram":  # type: ignore[name-defined]
    """
    :func:`export` takes an arbitrary Python callable (an nn.Module, a function or
    a method) and produces a traced graph representing only the Tensor
    computation of the function in an Ahead-of-Time (AOT) fashion, which can
    subsequently be executed with different outputs or serialized.  The traced
    graph (1) produces a normalized operator set consisting only of functional
    [core ATen operator set](LINK HERE) and user specified custom operators, (2) has
    eliminated all Python control flow and data structures (except for certain
    conditions), and (3) has the set of shape constraints needed to show that
    this normalization and control flow elimination is sound for a future
    input.

    Args:
        f: The callable to trace.

        args: Example positional inputs.

        kwargs: Optional example keyword inputs.

        constraints: An optional list of constraints on the dynamic arguments
         that specify their possible range of shapes. By default, shapes of
         input torch.Tensors are assumed to be static. If an input torch.Tensor
         is expected to have dynamic shapes, please use `torch.export.dynamic_dim()`
         to define `Constraint` objects that specify the dynamics and the possible
         range of shapes. See torch.export.dynamic_dim() docstring for examples on
         how to use it.

    Returns:
        An ExportedProgram containing the traced callable.

    **How it works**

    :func:`export` operates by using TorchDynamo to symbolically trace the
    bytecode of the passed in Python callable, inlining functions and control
    flow and recording the Tensor calls into an FX graph.  Export then
    applies further lowering passes to normalize these calls into a
    functionalized core ATen operator set.

    Compared to :func:`torch.fx.symbolic_trace`, :func:`export` operates at
    the bytecode level, giving it the ability to trace arbitrary Python constructs
    not limited by what Python operator overloading supports.  We also keep
    fine-grained track of Tensor metadata, so that conditionals on things like
    ``tensor.size(i)`` do not fail tracing, and then normalize the graph to ATen operators.
    In general, you should expect :func:`export` to work on more user programs, but to
    produce lower-level graphs.  You can still use :func:`torch.fx.symbolic_trace` as
    a pre-processing step before :func:`export`

    Compared to :func:`torch.jit.script`, :func:`export` does not capture Python
    control flow or data structures.  We support more Python language features than
    TorchScript (as it is easier to have comprehensive coverage over Python
    bytecodes).  The resulting graphs are simpler and only have straight line
    control flow (except for explicit control flow operators).

    Compared to :func:`torch.jit.trace`, :func:`export` is sound: it is able to
    trace code that performs integer computation on sizes and records all of the
    side-conditions necessary to show that a particular trace is valid for other inputs.

    **Acceptable input/output types**

    Acceptable types of inputs (for `args` and `kwargs`) and outputs include:

    - Primitive types, i.e. `torch.Tensor`, `int`, `float`, `bool` and `str`.
    - Dataclasses (must be registered with torch._export.utils.register_dataclass_as_pytree_node` first)
    - (Nested) Data structures comprising of `dict`, `list`, `tuple`, `namedtuple` and
      `OrderedDict` containing all above types.


    **What's specialized in the program?**

    1. Non-tensor inputs

    `export()` specializes the traced program based on the values of
    inputs that are not torch.Tensors, ie. `int`, `float`, `bool` and `str`.

    For example::

        from torch.export import export

        def fn(x: torch.Tensor, i: int):
            return x + i

        example_inputs = (torch.rand(2, 2), 1)  # i is set to 1 in example inputs
        ep = export(fn, example_inputs)

    would yield an `ExportedProgram` containing following graph::

        %arg0_1 : [num_users=1] = placeholder[target=arg0_1]
        %arg1_1 : [num_users=0] = placeholder[target=arg1_1]
        %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg0_1, 1), kwargs = {})
        return (add,)

    Notice that `%add` is computed by adding `%arg0_1` and `1`, which is a
    constant rather than `%arg1_1` because integers are specialized.

    2. Rank and static shapes (not values) of input Tensors

    Rank of a tensor is always specialized and treated as constant. Sizes of
    dimensions are also specialized as constant, i.e. static shapes unless
    specified as dynamic via `dynamic_dim` API, for example::

        from torch.export import export

        def fn(x):
            if x.shape[0] > 5:
                return x + 1
            else:
                return x

        example_inputs = (torch.rand(10, 2))
        ep = export(fn, example_inputs)

    Would produce an `ExportedProgram` containing following graph::

        %arg0_1 : [num_users=1] = placeholder[target=arg0_1]
        %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg0_1, 1), kwargs = {})
        return (add,)

    You can see that the conditional on `x.shape[0]>5` is removed because the
    example inputs has the static shape of `(10, 2)`. `torch.export()` specializes
    on the static shape, thus the `else` branch will never be reached, thus it
    does not show up in the exported program.

    Note:
    If you want to preserve dynamic branching behavior based on value or
    shape of torch.Tensor in the generated graph, you will need to use
    `torch.export.dynamic_dim` to make a dimension of input tensor to be dynamic
    and rewrite the source code using control flow operations like
    `torch.ops.higher_order.cond`.

    3. Control flow

    By default, control flow (like `if`) branching decisions are spcialized
    according to execution flow observed during tracing run. See following
    section on how to preserve dynamic control flow

    **How to express Dynamism**

    1. Shape Dynamism

    Because static shape use cases are more dominant, `export()` chooses to
    assume shapes are all static by default unless there are explicit user
    instructions that say otherwise. Specifically, users can use
    `torch.export.dynamic_dim` to give a hint to `export()` about dynamism
    and range of an input tensor dimension.

    2. Dynamic Control Flow

    To preserve dynamic branching behavior of control flows (like `if`), users
    need to rewrite source code of original program to use PyTorch's higher order
    operators (like `torch.ops.higher_order.cond`).



    [1, Inf]
    [2, Inf]

    f(x)
    x in [2, Inf]  ~>  FAILS (ConstraintViolation)
    then
    x in [1, Inf]  ~>  ALSO FAILS

    hint x == 2

    if x == 1:
        something()

    ~~ConstraintViolation: x was dynamic, but actually specialized~~

    * What is a value ranges on variables [-Inf, Inf] vs [0, Inf]
    * Size variables intuitively are [0, Inf], but we treat them as [2, Inf]
      for reasoning purposes
        if sizevar == 0:  # this is always False

      If you don't use constraint_as_size, and you pass a value to a
      function, you will get error like "unbacked symint could not prove i0 == 0"
    * Upshot:
        - if your thing is going to be used like a size (pass as a size arg to
          a factory, or view), use constrain_as_size
        - otherwise, it probably doesn't matter
        - if your size is intended to be dynamic, do NOT test if sizes are ==
          0, 1, these will SILENTLY report false and be bypassed
        - the more elaborate, obscure explanatioN


    **Soundness Guarantee**

    While tracing, `export()` takes note of shape-related assumptions
    made by the user program and the underlying PyTorch operator kernels.
    The output ExportedProgram is considered valid only when these
    assumptions hold true.

    There are 2 types of assumptions made during tracing

    - Shapes (not values) of input tensors.
    - Ranges (lower and upper bound) of values extracted from intermediate tensors via `.item()` or direct indexing.


    All assumptions must be validated at graph capture time for `export()`
    to succeed. Specifically:

    - Assumptions on static shapes of input tensors are automatically validated without additional effort.
    - Assumptions on dynamic shape of input tensors require explicit `Input Constraint`
      constructed with `torch.export.dynamic_dim` APIs
    - Assumptions on range of intermediate values require explicit `Inline Constraint`,
      constructed use `constrain_as_size` and `constraint_as_value` APIs.

    If any assumption can not be validated, a fatal error will be raised. When that happens,
    the error message will include suggested code needed to construct necessary
    constraints to validate the assumptions, for example `export()` would suggest
    following code for input constraints::

        def specify_constraints(x):
            return [
                # x:
                dynamic_dim(x, 0),
                dynamic_dim(x, 0) <= 5,
            ]

    This example means the program requires the dim 0 of input `x` to be less
    than or equal to 5 to be valid. You can inspect the constraints needed and
    then copy this exact function into your code to generated needed
    constraints to be passed into `constraints` argument.

    **ExportedProgram Invariants**

    The returned `ExportedProgram` maintains the following invariants:

    - It is guaranteed to be a sound representation of the original
      program.
    - It maintains the exact calling convention of the original program.
    - It contains a `state_dict` that stores the `torch.nn.Parameters`
      involved in computation of the original program.
    - It includes an fx.GraphModule that represents the computation of
      the original program. The GraphModule:

     - Contains only `placeholder`, `call_function`, `get_attr` and `return` nodes.
     - Inlines all submodules from the original programs.
     - Lifts all parameters and buffers of the original program as inputs to the graph.
     - Does not mutate intermediate values, parameters, or buffers.
     - Does not include operations with side effects.
     - Contains only a curated subset of ATen operations and registered
       custom operations (by default). See the list of Core ATen Ops
       here: https://pytorch.org/docs/stable/ir.html

    """

    from torch._export import export

    return export(f, args, kwargs, constraints)
