from typing import Any, Callable, Dict, List, Optional, Tuple

import torch


__all__ = [
    "constrain_as_size",
    "constrain_as_value",
    "dynamic_dim",
    "export",
]


def constrain_as_value(symbol, min: Optional[int] = None, max: Optional[int] = None):
    """
    Hint `export()` about the constraint of an intermediate scalar value so that subsequent
    branching behaviors that check on the range of aforementioned scalar value can be
    soundly traced.

    .. warning::
        (Note that if the intermediate scalar value will be used as a shape,
        call `constrain_as_size` API instead.)

    For example, following program can not be traced soundly wihout using
    `constrain_as_value` to give `export()` a hint about which branch to take::

        def fn(x):
            v = x.max().item()
            if v > 1024:
                return x
            else:
                return x * 2

    `export()` would give following error::

        torch._dynamo.exc.UserError: Consider annotating your code using
        torch.export.constrain_as_size() or torch.export().constrain_as_value() APIs.
        It appears that you're trying to get a value out of symbolic int/float whose value
        is data-dependent (and thus we do not know the true value.)  The expression we were
        trying to evaluate is f0 > 1024 (unhinted: f0 > 1024).

    Assuming the actual range of `v` can be between [10, 200], you can add a call to
    `constrain_as_value` in the source code like this::

        def fn(x):
            v = x.max().item()

            # Give export() a hint
            torch.export.constrain_as_value(v, min=10, max=200)

            if v > 1024:
                return x
            else:
                return x * 2

    With the additional hint, `export()` would be able to trace the program correctly by taking
    the `else` branch, resulting in following graph::

        graph():
            %arg0_1 := placeholder[target=arg0_1]

            # v = x.max().item()
            %max_1 := call_function[target=torch.ops.aten.max.default](args = (%arg0_1,))
            %_local_scalar_dense := call_function[target=torch.ops.aten._local_scalar_dense.default](args = (%max_1,))

            # Asserting 10 <= v <= 200
            %ge := call_function[target=operator.ge](args = (%_local_scalar_dense, 10))
            %scalar_tensor := call_function[target=torch.ops.aten.scalar_tensor.default](args = (%ge,))
            %_assert_async := call_function[target=torch.ops.aten._assert_async.msg](
                args = (%scalar_tensor, _local_scalar_dense is outside of inline constraint [10, 200].))
            %le := call_function[target=operator.le](args = (%_local_scalar_dense, 200))
            %scalar_tensor_1 := call_function[target=torch.ops.aten.scalar_tensor.default](args = (%le,))
            %_assert_async_1 := call_function[target=torch.ops.aten._assert_async.msg](
                args = (%scalar_tensor_1, _local_scalar_dense is outside of inline constraint [10, 200].))
            %sym_constrain_range := call_function[target=torch.ops.aten.sym_constrain_range.default](
                args = (%_local_scalar_dense,), kwargs = {min: 10, max: 200})

            # Always taking `else` branch to multiply elements `x` by 2 due to hints above
            %mul := call_function[target=torch.ops.aten.mul.Tensor](args = (%arg0_1, 2), kwargs = {})
            return (mul,)


    Args:
        symbol: Intermediate scalar value (int-only now) to apply range constraint on.
        min (Optional[int]): Minimum possible value of given symbol (inclusive)
        max (Optional[int]): Maximum possible value of given symbol (inclusive)

    Returns:
        None

    """
    from torch._export.constraints import constrain_as_value

    return constrain_as_value(symbol, min, max)


def constrain_as_size(symbol, min: Optional[int] = None, max: Optional[int] = None):
    """
    Hint `export()` about the constraint of an intermediate scalar value that
    represents shape of a tensor so that subsequent tensor constructors can be
    traced correctly because many operators need to make assumption about range
    of sizes.

    For example, following program can not be traced soundly wihout using
    `constrain_as_size` to give `export()` a hint about shape ranges::

        def fn(x):
            d = x.max().item()
            return torch.ones(v)

    `export()` would give following error::

        torch._dynamo.exc.Unsupported: guard on data-dependent symbolic int/float

    Assuming the actual range of `d` can be between [3, 10], you can add a call to
    `constrain_as_size` in the source code like this::

        def fn(x):
            d = x.max().item()
            torch.export.constrain_as_size(d, min=3, max=10)
            return torch.ones(d)

    With the additional hint, `export()` would be able to trace the program correctly by taking
    the `else` branch, resulting in following graph::

        graph():
            %arg0_1 := placeholder[target=arg0_1]

            # d = x.max().item()
            %max_1 := call_function[target=torch.ops.aten.max.default](args = (%arg0_1,))
            %_local_scalar_dense := call_function[target=torch.ops.aten._local_scalar_dense.default](args = (%max_1,))

            # Asserting 3 <= d <= 10
            %ge := call_function[target=operator.ge](args = (%_local_scalar_dense, 3))
            %scalar_tensor := call_function[target=torch.ops.aten.scalar_tensor.default](args = (%ge,))
            %_assert_async := call_function[target=torch.ops.aten._assert_async.msg](
                args = (%scalar_tensor, _local_scalar_dense is outside of inline constraint [3, 10].))
            %le := call_function[target=operator.le](args = (%_local_scalar_dense, 10))
            %scalar_tensor_1 := call_function[target=torch.ops.aten.scalar_tensor.default](args = (%le,))
            %_assert_async_1 := call_function[target=torch.ops.aten._assert_async.msg](
                args = (%scalar_tensor_1, _local_scalar_dense is outside of inline constraint [3, 10].))
            %sym_constrain_range_for_size := call_function[target=torch.ops.aten.sym_constrain_range_for_size.default](
                args = (%_local_scalar_dense,), kwargs = {min: 3, max: 10})

            # Constructing new tensor with d
            %full := call_function[target=torch.ops.aten.full.default](
                args = ([%_local_scalar_dense], 1),
                kwargs = {dtype: torch.float32, layout: torch.strided, device: cpu, pin_memory: False})

            ......


    .. warning::
        It is illegal to specify a range that contains 0 and 1. 0/1 values are always specialized
        and can not be part of dynamic range.

    Args:
        symbol: Intermediate scalar value (int-only now) to apply range constraint on.
        min (Optional[int]): Minimum possible value of given symbol (inclusive)
        max (Optional[int]): Maximum possible value of given symbol (inclusive)

    Returns:
        None

    """

    from torch._export.constraints import constrain_as_size

    return constrain_as_size(symbol, min, max)


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
    `export()` is a one-shot process for capturing a computation graph from
    a PyTorch program Ahead-of-Time (AOT).

    This function traces a callable (an nn.Module, a function or a method)
    containing PyTorch operations and produces an ExportedProgram. The
    ExportedProgram includes PyTorch operations that perform computations
    equivalent to those in the given nn.Module or callable.

    In specific terms, `export()` traces a function `f` by executing it
    with the provided `args` and `kwargs`. It records the PyTorch operations
    invoked during execution to produce the ExportedProgram.


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

    """

    from torch._export import export

    return export(f, args, kwargs, constraints)
