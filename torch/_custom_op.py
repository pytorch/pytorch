import inspect
import typing
import weakref

from torchgen.model import FunctionSchema, OperatorName, SchemaKind

import torch
import torch._C as _C
import torch.library as library
import torch.utils._pytree as pytree

"""
There are various APIs for defining custom-operator-like things in PyTorch:
- [user-facing] autograd.Function (Python)
- [user-facing] CustomOp (Python)
- [for power users] torch.library (Python)
- [for power users] TORCH_LIBRARY (C++)

This file contains the implementation for a Simple Custom Operator API (CustomOp).
Using CustomOp, you are able to define a custom operator and implement interactions
between the CustomOp and various PyTorch subsystems, including all the subsystems
that are necessary for a custom operator to work with torch.compile (i.e.,
autograd, meta, functionalization).

CustomOp is positioned as being safer and easier to use than
torch.library/TORCH_LIBRARY, which require deep understanding of PyTorch internals.
In additional, it supports torch.compile better than and is in general more
comprehensive than autograd.Function, which only supports implementing gradient
computation and vmap rules.
"""

__all__ = ["CustomOp"]


SUPPORTED_DEVICE_TYPES = ("cpu", "cuda")

DEVICE_TYPE_TO_KEY = {
    "cpu": "CPU",
    "cuda": "CUDA",
}

# We will not let users register CustomOps with anything that could look like
# PyTorch internals to avoid confusion.
RESERVED_NS = {
    "prim",
    "prims",
    "aten",
    "at",
    "torch",
    "pytorch",
}


class CustomOp:
    r"""Class for custom operators in PyTorch.

    Use the CustomOp API to create user-defined custom operators that behave
    just like regular PyTorch operators (e.g. torch.sin, torch.mm) when it
    comes to various PyTorch subsystems (like torch.compile).

    To construct a `CustomOp`, use :meth:`CustomOp.define`.
    """

    def __init__(self, lib, cpp_ns, operator_name, ophandle, *, _private_access=False):
        super(CustomOp, self).__init__()
        if not _private_access:
            raise RuntimeError(
                "The CustomOp constructor is private. Please use "
                "CustomOp.define to create a CustomOp object"
            )
        name = f"{cpp_ns}::{str(operator_name.name)}"
        self._lib: library.Library = lib
        self._ophandle: _C._DispatchOperatorHandle = ophandle
        # Has the name of the op, e.g. "foo". We cache here for convenience.
        self._opname: str = str(operator_name)
        # this is _opname but with namespace. e.g. "custom::foo"
        self._namespaced_opname: str = name

    def __repr__(self):
        raise RuntimeError("Please don't call this")

    @staticmethod
    def define(schema: str, *, ns: str) -> typing.Callable:
        r"""Creates a new CustomOp object.

        In PyTorch, defining an op (short for "operator") is a two step-process:
        - we need to define (create) the op
        - we need to implement behavior for how the operator interacts with
          various PyTorch subsystems, like CPU/CUDA Tensors, Autograd, etc.

        This entrypoint defines the CustomOp object (the first step);
        you must then perform the second step by calling various methods on
        the CustomOp object.

        This API is used as a decorator (see examples).

        Arguments:
            schema (str): The schema of the CustomOp.

        Example::
            >>> import numpy as np
            >>>
            >>> # Step 1: define the CustomOp.
            >>> # We need to provide the decorator a "prototype function"
            >>> # (a function with Python ellipses as the body).
            >>> @CustomOp.define('(Tensor x) -> Tensor')
            >>> def numpy_sin(x):
            >>>     ...
            >>>
            >>> # numpy_sin is now an instance of class CustomOp
            >>> print(type(numpy_sin))
            >>>
            >>> # Step 2: Register an implementation for various PyTorch subsystems
            >>>
            >>> # Register an implementation for CPU tensors
            >>> @numpy_sin.impl('cpu'):
            >>> def numpy_sin_impl_cpu(x):
            >>>     return torch.from_numpy(np.sin(x.numpy()))
            >>>
            >>> # Register an implementation for CUDA tensors
            >>> @numpy_sin.impl('cuda'):
            >>> def numpy_sin_impl_cuda(x):
            >>>     return torch.from_numpy(np.sin(x.cpu().numpy())).to(x.device)
            >>>
            >>> x = torch.randn(3)
            >>> numpy_sin(x)  # calls numpy_sin_impl_cpu
            >>>
            >>> x_cuda = x.cuda()
            >>> numpy_sin(x)  # calls numpy_sin_impl_cuda

        """

        def inner(func):
            if not inspect.isfunction(func):
                raise ValueError(
                    f"CustomOp.define(...)(func): Expected `func` to be a Python "
                    f"function, got: {type(func)}"
                )

            validate_namespace(ns)
            schema_str = f"{func.__name__}{schema}"
            function_schema = FunctionSchema.parse(schema_str)
            validate_schema(function_schema)

            lib = library.Library(ns, "FRAGMENT")
            lib.define(schema_str)
            ophandle = find_ophandle_or_throw(ns, function_schema.name)
            result = CustomOp(
                lib, ns, function_schema.name, ophandle, _private_access=True
            )

            result.__name__ = func.__name__
            result.__module__ = func.__module__
            result.__doc__ = func.__doc__

            # NYI: autograd not supported
            # In the near future we will either directly use the
            # autograd_not_implemented kernels or make those the default fallback
            # for the Autograd and ADInplaceOrView keys. Both of those are a bit tricky.
            library.impl(lib, result._opname, "Autograd")(
                get_autograd_not_implemented_kernel(weakref.proxy(result))
            )

            return result

        return inner

    def __call__(self, *args, **kwargs):
        # Bypass torch.ops.* and directly do OperatorHandle::callBoxed.
        # Using torch.ops.* is a bit of a pain (it can be slow and it has lifetime
        # issues from caching operators that make testing CustomOp difficult).
        result = _C._dispatch_call_boxed(self._ophandle, *args, **kwargs)
        return result

    def impl(
        self, device_types: typing.Union[str, typing.Iterable[str]]
    ) -> typing.Callable:
        r"""Register an implementation for a device type for this CustomOp object.

        If the CustomOp is passed multiple Tensor inputs with different device
        types, it will dispatch to the registered implementation for the highest
        priority device type among those present.
        The supported device types, in order of priority, are {'cuda', 'cpu'}.

        This API is used as a decorator (see examples).

        Arguments:
            device_types (str or Iterable[str]): the device type(s) to register the function for.

        Examples::
            >>> import numpy as np
            >>> numpy_sin = CustomOp.define('custom::numpy_sin')
            >>>
            >>> # Register an implementation for CPU and CUDA Tensors
            >>> @numpy_sin.impl('cpu'):
            >>> def numpy_sin_impl_cpu(x):
            >>>     return torch.from_numpy(np.sin(x.numpy()))
            >>>
            >>> # Register an implementation for CUDA Tensors
            >>> @numpy_sin.impl('cuda'):
            >>> def numpy_sin_impl_cuda(x):
            >>>     return torch.from_numpy(np.sin(x.cpu().numpy())).to(x.device)
            >>>
            >>> x = torch.randn(3)
            >>> numpy_sin(x)  # calls numpy_sin_impl_cpu
            >>>
            >>> x_cuda = x.cuda()
            >>> numpy_sin(x)  # calls numpy_sin_impl_cuda

        """
        if isinstance(device_types, str):
            device_types = [device_types]
        for device_type in device_types:
            validate_device_type(device_type)

        def inner(f):
            for device_type in set(device_types):
                dispatch_key = DEVICE_TYPE_TO_KEY[device_type]
                library.impl(self._lib, self._opname, dispatch_key)(f)
            return f

        return inner

    def impl_meta(self) -> typing.Callable:
        r"""Register a meta implementation for this CustomOp object.

        The meta implementation is a shape propagation rule that gets invoked
        for device='meta' Tensors and FakeTensors (Tensors that do not have storage).

        To register a data-dependent shape propagation rule, use the
        not-yet-implemented method to register a rule for FakeTensor.

        This API is used as a decorator (see examples).

        Examples::
            >>> import numpy as np
            >>> custom_sum = CustomOp.define('custom::sum(Tensor tensor, int dim)')
            >>>
            >>> @custom_sum.impl_meta():
            >>> def custom_sum(tensor, dim):
            >>>     output_shape = list(tensor.shape)
            >>>     del output_shape[dim]
            >>>     return tensor.new_empty(output_shape)
            >>>
            >>> x = torch.randn(2, 3, device='meta')
            >>> y = custom_sum(x, 1)
            >>> assert y.shape == (2,)

        """

        def inner(f):
            library.impl(self._lib, self._opname, "Meta")(f)
            return f

        return inner


def find_ophandle_or_throw(cpp_ns: str, operator_name: OperatorName):
    overload_name = (
        "" if operator_name.overload_name is None else operator_name.overload_name
    )
    return _C._dispatch_find_schema_or_throw(
        f"{cpp_ns}::{str(operator_name.name)}", overload_name
    )


def validate_namespace(ns: str) -> None:
    if ns not in RESERVED_NS:
        return
    raise ValueError(
        f"CustomOp.define(schema, ns={ns}): {ns} is a reserved namespace, "
        f"please choose something else. "
    )


def validate_schema(schema: FunctionSchema) -> None:
    # Coming in the future. Requires us to have correct logic for
    # the ADInplaceOrView key
    if schema.kind() != SchemaKind.functional:
        raise NotImplementedError(
            f"NYI: CustomOp.define does not support non-functional function schema. Got: {schema}"
        )

    rets = schema.returns
    is_non_mutating_view = len(rets) > 0 and any(
        r.annotation is not None and not r.annotation.is_write for r in rets
    )
    if is_non_mutating_view:
        raise NotImplementedError(
            f"NYI: CustomOp.define does not support view functions. Got: {schema}"
        )

    # Requires us to have handling for factory functions
    if not schema.arguments.has_tensor_arg():
        raise NotImplementedError(
            f"NYI: CustomOp.define does not support function schema with no Tensor inputs. Got: {schema}"
        )
    # Just seems weird so banning for now
    if not schema.returns:
        raise NotImplementedError(
            f"NYI: CustomOp.define does not support function schema with no outputs. Got: {schema}"
        )


def parse_namespace(namespaced_entity: str) -> typing.Tuple[str, str]:
    names = namespaced_entity.split("::", 1)
    if len(names) != 2:
        raise ValueError(f"Expected there to be a namespace in {namespaced_entity}.")
    return names[0], names[1]


def validate_device_type(device_type: str) -> None:
    if device_type not in SUPPORTED_DEVICE_TYPES:
        raise ValueError(
            f"CustomOp.impl(device_types=[{device_type}, ...]): we only support device_type "
            f"in {SUPPORTED_DEVICE_TYPES}."
        )


def get_autograd_not_implemented_kernel(custom_op) -> typing.Callable:
    def autograd_not_implemented(*args, **kwargs) -> None:
        if pytree.tree_any(
            lambda x: isinstance(x, torch.Tensor) and x.requires_grad, (args, kwargs)
        ):
            raise RuntimeError("Autograd has not been implemented for operator")
        # TODO(rzou): RAII guard should be contextmanager, or else bad things will happen
        guard = _C._AutoDispatchBelowAutograd()
        try:
            return custom_op(*args, **kwargs)
        finally:
            del guard

    return autograd_not_implemented
