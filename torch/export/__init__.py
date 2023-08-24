from typing import Any, Callable, Dict, List, Optional, Tuple

import torch


__all__ = [
    "export",
]


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
    `torch._export.dynamic_dim` to make a dimension of input tensor to be dynamic
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
    `torch._export.dynamic_dim` to give a hint to `export()` about dynamism
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
      constructed with `torch._export.dynamic_dim` APIs
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
         is expected to have dynamic shapes, please use `torch._export.dynamic_dim()`
         to define `Constraint` objects that specify the dynamics and the possible
         range of shapes. See torch._export.dynamic_dim() docstring for examples on
         how to use it.

    Returns:
        An ExportedProgram containing the traced callable.

    """

    from torch._export import export

    return export(f, args, kwargs, constraints)
