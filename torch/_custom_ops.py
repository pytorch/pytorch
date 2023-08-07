import inspect

from torch._custom_op.impl import (
    _custom_op_with_schema,
    _find_custom_op,
    get_ctx,
    infer_schema,
    parse_qualname,
    validate_namespace,
)

__all__ = [
    "custom_op",
    "impl",
    "impl_abstract",
    "get_ctx",
    "impl_save_for_backward",
    "impl_backward",
]


def custom_op(qualname, func_or_schema=None):
    r"""Register a new custom operator

    In PyTorch, defining an op (short for "operator") is a two step-process:
    - we need to define the op (by providing an operator name and schema)
    - we need to implement behavior for how the operator interacts with
      various PyTorch subsystems, like CPU/CUDA Tensors, Autograd, etc.

    This entrypoint defines the custom operator (the first step)
    you must then perform the second step by calling various
    ``impl_*`` APIs.

    This API may be used as a decorator (see examples).

    For a detailed guide on custom ops, please see
    https://docs.google.com/document/d/1aGWtgxV3HppuxQAdddyPrs74_aEntpkYt9MalnCKnhk

    Arguments:
        qualname (str): Should be a string that looks like
            "namespace::operator_name". Operators in PyTorch need a namespace to
            avoid name collisions; a given operator may only be created once.
            If you are writing a Python library, we recommend the namespace to
            be the name of your top-level module.
        func_or_schema (Union[Callable, str]): Each PyTorch operator needs a
            schema that tells PyTorch the types of the inputs/outputs.
            If this is a Callable, we will automatically infer the schema from
            the type annotations on the function (see examples). Otherwise,
            if you don't want to use type annotations, you may provide us the
            schema string.

    Example::
        >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_CUDA)
        >>> import torch
        >>> import numpy as np
        >>> from torch import Tensor
        >>>
        >>> # Step 1: define the custom op.
        >>> # We need to provide the API a "prototype function"
        >>> # (a function that returns NotImplementedError), from which
        >>> # we will infer the types of the inputs and outputs.
        >>> @torch._custom_ops.custom_op("mylibrary::numpy_sin")
        >>> def numpy_sin(x: Tensor) -> Tensor:
        >>>     raise NotImplementedError()
        >>>
        >>> # The custom op is now accessible via the torch.ops module:
        >>> torch.ops.mylibrary.numpy_sin
        >>>
        >>> # Step 2: Register an implementation for various PyTorch subsystems
        >>>
        >>> # Register an implementation for CPU tensors
        >>> @torch._custom_ops.impl("mylibrary::numpy_sin", device_types="cpu")
        >>> def numpy_sin_impl_cpu(x):
        >>>     return torch.from_numpy(np.sin(x.numpy()))
        >>>
        >>> # Register an implementation for CUDA tensors
        >>> @torch._custom_ops.impl("mylibrary::numpy_sin", device_types="cuda")
        >>> def numpy_sin_impl_cuda(x):
        >>>     return torch.from_numpy(np.sin(x.cpu().numpy())).to(x.device)
        >>>
        >>> x = torch.randn(3)
        >>> torch.ops.mylibrary.numpy_sin(x)  # calls numpy_sin_impl_cpu
        >>>
        >>> x_cuda = x.cuda()
        >>> torch.ops.mylibrary.numpy_sin(x)  # calls numpy_sin_impl_cuda

    """
    ns, name = parse_qualname(qualname)
    validate_namespace(ns)

    def inner(func):
        if not inspect.isfunction(func):
            raise ValueError(
                f"custom_op(...)(func): Expected `func` to be a Python "
                f"function, got: {type(func)}"
            )

        if func.__name__ != name:
            raise ValueError(
                f"custom_op(qualname='{qualname}', ...)(func): expected `func` "
                f"to have name '{name}' but got '{func.__name__}'. "
                f"Please either change the name of `func` or the qualname that "
                f"is passed to `custom_op`"
            )

        schema = infer_schema(func)
        _custom_op_with_schema(qualname, schema)
        return func

    if func_or_schema is None:
        return inner
    if isinstance(func_or_schema, str):
        _custom_op_with_schema(qualname, func_or_schema)
    else:
        return inner(func_or_schema)


def impl(qualname, *, device_types=("cpu", "cuda"), func=None):
    r"""Register an implementation for a device type for this custom op.

    If the op is passed multiple Tensor inputs with different device
    types, it will dispatch to the registered implementation for the highest
    priority device type among those present.
    The supported device types, in order of priority, are {'cuda', 'cpu'}.

    This API may be used as a decorator (see examples).

    For a detailed guide on custom ops, please see
    https://docs.google.com/document/d/1aGWtgxV3HppuxQAdddyPrs74_aEntpkYt9MalnCKnhk

    Arguments:
        device_types (str or Iterable[str]): the device type(s) to register the function for.

    Example::
        >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_CUDA)
        >>> import torch
        >>> import numpy as np
        >>> from torch import Tensor
        >>>
        >>> # Step 1: define the custom op.
        >>> # We need to provide the API a "prototype function"
        >>> # (a function that returns NotImplementedError), from which
        >>> # we will infer the types of the inputs and outputs.
        >>> @torch._custom_ops.custom_op("mylibrary::numpy_sin")
        >>> def numpy_sin(x: Tensor) -> Tensor:
        >>>     raise NotImplementedError()
        >>>
        >>> # The custom op is now accessible via the torch.ops module:
        >>> torch.ops.mylibrary.numpy_sin
        >>>
        >>> # Step 2: Register an implementation for various PyTorch subsystems
        >>>
        >>> # Register an implementation for CPU tensors
        >>> @torch._custom_ops.impl("mylibrary::numpy_sin", device_types="cpu")
        >>> def numpy_sin_impl_cpu(x):
        >>>     return torch.from_numpy(np.sin(x.numpy()))
        >>>
        >>> # Register an implementation for CUDA tensors
        >>> @torch._custom_ops.impl("mylibrary::numpy_sin", device_types="cuda")
        >>> def numpy_sin_impl_cuda(x):
        >>>     return torch.from_numpy(np.sin(x.cpu().numpy())).to(x.device)
        >>>
        >>> x = torch.randn(3)
        >>> torch.ops.mylibrary.numpy_sin(x)  # calls numpy_sin_impl_cpu
        >>>
        >>> x_cuda = x.cuda()
        >>> torch.ops.mylibrary.numpy_sin(x)  # calls numpy_sin_impl_cuda

    """

    def inner(func):
        custom_op = _find_custom_op(qualname)
        custom_op.impl(device_types, _stacklevel=3)(func)
        return func

    if func is None:
        return inner
    return inner(func)


def impl_abstract(qualname, *, func=None):
    r"""Register an abstract implementation for this operator.

    An "abstract implementation" specifies the behavior of this operator on
    Tensors that carry no data. Given some input Tensors with certain properties
    (sizes/strides/storage_offset/device), it specifies what the properties of
    the output Tensors are.

    The abstract implementation has the same signature as the operator.
    It is run for both FakeTensors and meta tensors. To write an abstract
    implementation, assume that all Tensor inputs to the operator are
    regular CPU/CUDA/Meta tensors, but they do not have storage, and
    you are trying to return regular CPU/CUDA/Meta tensor(s) as output.
    The abstract implementation must consist of only PyTorch operations
    (and may not directly access the storage or data of any input or
    intermediate Tensors).

    This API may be used as a decorator (see examples).

    For a detailed guide on custom ops, please see
    https://docs.google.com/document/d/1aGWtgxV3HppuxQAdddyPrs74_aEntpkYt9MalnCKnhk

    Examples::
        >>> import numpy as np
        >>> from torch import Tensor
        >>>
        >>> # Example 1: an operator without data-dependent output shape
        >>> @torch._custom_ops.custom_op("mylibrary::custom_linear")
        >>> def custom_linear(x: Tensor, weight: Tensor, bias: Tensor) -> Tensor:
        >>>     raise NotImplementedError()
        >>>
        >>> @torch._custom_ops.impl_abstract("mylibrary::custom_linear"):
        >>> def custom_linear_abstract(x, weight):
        >>>     assert x.dim() == 2
        >>>     assert weight.dim() == 2
        >>>     assert bias.dim() == 1
        >>>     assert x.shape[1] == weight.shape[1]
        >>>     assert weight.shape[0] == bias.shape[0]
        >>>     assert x.device == weight.device
        >>>
        >>>     return (x @ weight.t()) + bias
        >>>
        >>> # Example 2: an operator with data-dependent output shape
        >>> @torch._custom_ops.custom_op('mylibrary::custom_nonzero')
        >>> def custom_nonzero(x: Tensor) -> Tensor:
        >>>     ...
        >>>
        >>> @torch._custom_ops.impl_abstract("mylibrary::custom_nonzero"):
        >>> def custom_nonzero_abstract(x):
        >>>     # Number of nonzero-elements is data-dependent.
        >>>     # Since we cannot peek at the data in an abstract impl,
        >>>     # we use the ctx object to construct a new symint that
        >>>     # represents the data-dependent size.
        >>>     ctx = torch._custom_ops.get_ctx()
        >>>     nnz = ctx.create_unbacked_symint()
        >>>     shape = [x.dim(), nnz]
        >>>     result = x.new_empty(shape, dtype=torch.long)
        >>>     return result
        >>>
        >>> @torch._custom_ops.impl("mylibrary::custom_nonzero")
        >>> def custom_nonzero_impl(x):
        >>>     x_np = to_numpy(x)
        >>>     res = np.stack(np.nonzero(x_np), axis=1)
        >>>     # unbacked symbolic ints in PyTorch must be >= 2, so we
        >>>     # constrain the range to at least 2
        >>>     if res.shape[0] <= 1:
        >>>         raise RuntimeError("not supported")
        >>>     return torch.tensor(res, device=x.device)

    """

    def inner(func):
        custom_op = _find_custom_op(qualname)
        custom_op.impl_abstract(_stacklevel=3)(func)
        return func

    if func is None:
        return inner
    return inner(func)


def impl_save_for_backward(qualname, *, func=None):
    r"""Register a function that tells us what to save for backward.

    Please see :func:`impl_backward` for more details.
    """

    def inner(func):
        custom_op = _find_custom_op(qualname)
        custom_op.impl_save_for_backward(_stacklevel=3)(func)
        return func

    if func is None:
        return inner
    return inner(func)


def impl_backward(qualname, output_differentiability=None, *, func=None):
    r"""Registers a backward formula for an operator.

    In order for an operator to work with autograd, you need to register
    a backward formula. There are two pieces to this:
    1. You must give us a function to specify what to save for backward.
       Call this the "save for backward" function.
    2. You must give us a function that computes gradients. Call this the
       "backward" function.

    Use `impl_save_for_backward` to define a "save for backward" function
    that specifies what gets saved for backward. The function should accept
    two arguments ``(inputs, output)`` and return the quantities to be saved
    for backward.

    During runtime, when you call the operator in a forwards pass, PyTorch
    will invoke the "save for backward" function with the inputs and output
    of the operator.

    Use `impl_backward` to define the "backward" function. The backward
    function must accept ``(ctx, saved, *grads)``:
    - ``ctx`` is a context object where we may provide information
    - ``saved`` is exactly what gets returned from the "save for backward"
      function
    - ``grads`` is one or more gradients. The number of gradients matches
      the number of outputs of the operator.

    The backward function must return a dict that maps the name of
    an input to the operator to its corresponding gradient. All inputs that
    were declared to be Tensors in the operator definition must be accounted
    for in the dict. The gradient may be a Tensor or None.

    For a detailed guide on custom ops, please see
    https://docs.google.com/document/d/1aGWtgxV3HppuxQAdddyPrs74_aEntpkYt9MalnCKnhk

    """

    def inner(func):
        custom_op = _find_custom_op(qualname)
        custom_op.impl_backward(output_differentiability, _stacklevel=3)(func)
        return func

    if func is None:
        return inner
    return inner(func)


def _destroy(qualname):
    """De-registers a custom op. For testing purposes only"""
    custom_op = _find_custom_op(qualname)
    custom_op._destroy()
