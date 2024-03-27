import contextlib
import inspect
import threading
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple, Union

from .. import _C, _library, library, Tensor
from . import utils


device_types_t = Optional[Union[str, Sequence[str]]]


def custom_op(
    *,
    mutated_args: Sequence[str],
    types: device_types_t = None,
    qualname: Optional[str] = None,
) -> Callable:
    """Wraps a function into custom operator.

    Reasons why you may want to create a custom op include:
    - Wrapping a third-party library or custom kernel to work with PyTorch
      subsystems like Autograd.
    - Preventing torch.compile/export/FX tracing from peeking inside your function.

    This API is used as a decorator around a function (please see examples).
    The provided function must have type hints; these are needed to interface
    with PyTorch's various subsystems.

    Args:
        mutated_args (Sequence[str]): The names of args that the function mutates.
            This MUST be accurate, otherwise, the behavior is undefined.
        types (None | str | Sequence[str]): The device type(s) the function
            is valid for. If no device type is provided, then the function
            is used as the default implementation for all device types.
            Examples: "cpu", "cuda".
        qualname (None | str): An optional name for the operator. If provided,
            must look like "{namespace}::{name}", e.g. "mylib::my_linear"

    Examples::
        >>> import torch
        >>> from torch import Tensor
        >>> from torch.library import custom_op
        >>> import numpy as np
        >>>
        >>> @custom_op(mutated_args=())
        >>> def numpy_sin(x: Tensor) -> Tensor:
        >>>     x_np = x.cpu().numpy()
        >>>     y_np = np.sin(x_np)
        >>>     return torch.from_numpy(y_np).to(device=x.device)
        >>>
        >>> x = torch.randn(3)
        >>> y = numpy_sin(x)
        >>> assert torch.allclose(y, x.sin())
        >>>
        >>> # Example of a custom op that only works for one device type.
        >>> @custom_op(mutated_args=(), types="cpu")
        >>> def numpy_sin_cpu(x: Tensor) -> Tensor:
        >>>     x_np = x.numpy()
        >>>     y_np = np.sin(x_np)
        >>>     return torch.from_numpy(y_np)
        >>>
        >>> x = torch.randn(3)
        >>> y = numpy_sin_cpu(x)
        >>> assert torch.allclose(y, x.sin())

    """
    assert len(mutated_args) == 0, "NYI"
    assert qualname is None, "NYI"

    def inner(fn):
        return CustomOpDef._from_fn(fn, types, mutated_args)

    return inner


class CustomOpDef:
    def __init__(self, namespace: str, name: str, schema: str, fn: Callable) -> None:
        # Fields used to interface with the PyTorch dispatcher
        self._namespace = namespace
        self._name = name
        self._qualname = f"{self._namespace}::{self._name}"
        self._schema = schema

        self._init_fn = fn

        self._backend_fns: Dict[Union[str, None], Callable] = {}
        self._abstract_fn: Optional[Callable] = None

        self._lib = get_library_allowing_overwrite(self._namespace, self._name)
        self._build()

    @classmethod
    def _from_fn(
        cls,
        fn: Callable,
        types: device_types_t,
        mutated_args: Sequence[str],
    ) -> "CustomOpDef":
        import torch

        schema = torch._custom_op.impl.infer_schema(fn, mutated_args)
        namespace = reserved_namespace()
        name = utils.mangle(utils.unique_name(fn))
        result = cls(namespace, name, schema, fn)
        result.register_impl(types)(fn)
        return result

    def __repr__(self) -> str:
        return f"<CustomOpDef({self._init_fn})>"

    def register_impl(
        self, types: device_types_t, fn: Optional[Callable] = None
    ) -> Callable:
        """Register an implementation for a device type for this operator.

        Some valid types are: "cpu", "cuda", "xla", "mps", "ipu", "xpu".
        This API may be used as a decorator.

        Args:
            types (str | Sequence[str]): The device types to register an impl to.

        Examples::
            >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_CUDA)
            >>> import torch
            >>> from torch import Tensor
            >>> from torch.library import custom_op
            >>> import numpy as np
            >>>
            >>> # Example of split cpu and cuda definitions
            >>> @custom_op(mutated_args=(), types="cpu")
            >>> def numpy_sin(x: Tensor) -> Tensor:
            >>>     x_np = x.numpy()
            >>>     y_np = np.sin(x_np)
            >>>     return torch.from_numpy(y_np)
            >>>
            >>> # Add implementations for the cuda device
            >>> @numpy_sin.register_impl("cuda")
            >>> def _(x):
            >>>     x_np = x.cpu().numpy()
            >>>     y_np = np.sin(x_np)
            >>>     return torch.from_numpy(y_np).to(device=x.device)
            >>>
            >>> x_cpu = torch.randn(3)
            >>> x_cuda = x_cpu.cuda()
            >>> assert torch.allclose(numpy_sin(x_cpu), x_cpu.sin())
            >>> assert torch.allclose(numpy_sin(x_cuda), x_cuda.sin())

        """

        def inner(fn):
            if types is None or isinstance(types, str):
                dtypes: List[Union[str, None]] = [types]
            else:
                dtypes = list(types)
            for device_type in dtypes:
                if device_type not in self._backend_fns:

                    def backend_impl(*args, **kwargs):
                        # Checks the assumption that outputs cannot alias
                        # inputs or other outputs.
                        storages = set()
                        for tensor in iter_tensors(args, kwargs):
                            storages.add(id(tensor.untyped_storage()))

                        result = self._backend_fns[device_type](*args, **kwargs)

                        tuple_result = result
                        if isinstance(result, (Tensor, list)):
                            tuple_result = (result,)
                        for tensor in iter_tensors(tuple_result, {}):
                            key = id(tensor.untyped_storage())
                            if id(tensor.untyped_storage()) in storages:
                                fn = self._backend_fns[device_type]
                                module = inspect.getmodule(fn)
                                raise RuntimeError(
                                    f"Tensors returned from custom ops (1) must not "
                                    f"be inputs to the custom op and (2) may not alias "
                                    f"any inputs or other returns. Please clone the "
                                    f"the offending output tensors (e.g. output.clone()) "
                                    f"or refactor your code. "
                                    f"Offending op: {self._name} (with implementation in {module})"
                                )
                            storages.add(key)
                        return result

                    if device_type is None:
                        self._lib.impl(
                            self._name, backend_impl, "CompositeExplicitAutograd"
                        )
                    else:
                        self._lib.impl(
                            self._name,
                            backend_impl,
                            _C._dispatch_key_for_device(device_type),
                        )
                self._backend_fns[device_type] = fn
            return fn

        if fn is None:
            return inner
        return inner(fn)

    def register_fake(self, fn: Callable) -> Callable:
        r"""Register a FakeTensor implementation for this custom op.

        This is necessary to get the operator to work efficiently with torch.compile.

        The Fake impl (sometimes also known as a meta kernel or abstract impl)
        specifies the behavior of this operator on Tensors that carry no data.
        Given some input Tensors with certain properties
        (sizes/strides/storage_offset/device), it specifies what the properties of
        the output Tensors are.

        Please see :func:`torch.library.impl_abstract` for more details.

        Examples:
            >>> import torch
            >>> import numpy as np
            >>> from torch import Tensor
            >>>
            >>> # Example 1: an operator without data-dependent output shape
            >>> @torch.library.custom_op(mutated_args=())
            >>> def custom_linear(x: Tensor, weight: Tensor, bias: Tensor) -> Tensor:
            >>>     return (x @ weight.t()) + bias
            >>>
            >>> @custom_linear.register_fake
            >>> def _(x, weight, bias):
            >>>     assert x.dim() == 2
            >>>     assert weight.dim() == 2
            >>>     assert bias.dim() == 1
            >>>     assert x.shape[1] == weight.shape[1]
            >>>     assert weight.shape[0] == bias.shape[0]
            >>>     assert x.device == weight.device
            >>>     return x.new_empty(x.size(0), weight.size(0))
            >>>
            >>> x = torch.randn(2, 2)
            >>> weight = torch.randn(2, 2)
            >>> bias = torch.randn(2)
            >>> # xdoctest: +SKIP("Requires Python <= 3.11")
            >>> out = torch.compile(custom_linear, fullgraph=True)(x, weight, bias)
            >>> assert torch.allclose(out, torch.nn.functional.linear(x, weight, bias))
            >>>
            >>> # Example 2: an operator with data-dependent output shape
            >>> @torch.library.custom_op(mutated_args=())
            >>> def custom_nonzero(x: Tensor) -> Tensor:
            >>>     x_np = x.cpu().numpy()
            >>>     res = np.stack(np.nonzero(x_np), axis=1)
            >>>     return torch.tensor(res, device=x.device)
            >>>
            >>> @custom_nonzero.register_fake
            >>> def _(x):
            >>>     # Number of nonzero-elements is data-dependent.
            >>>     # Since we cannot peek at the data in an abstract impl,
            >>>     # we use the ctx object to construct a new symint that
            >>>     # represents the data-dependent size.
            >>>     ctx = torch.library.get_ctx()
            >>>     nnz = ctx.new_dynamic_size()
            >>>     shape = [nnz, x.dim()]
            >>>     result = x.new_empty(shape, dtype=torch.int64)
            >>>     return result
            >>>
            >>> x = torch.tensor([0, 1, 2, 0, 0, 1])
            >>> # xdoctest: +SKIP("Requires Python <= 3.11")
            >>> out = torch.compile(custom_nonzero)(x)
            >>> assert torch.allclose(out, x.nonzero())

        """
        self._abstract_fn = fn
        return fn

    def _build(self) -> None:
        lib = self._lib
        lib.define(f"{self._name}{self._schema}")
        self._opoverload = _library.utils.lookup_op(self._qualname)

        def fake_impl(*args, **kwargs):
            if self._abstract_fn is None:
                raise RuntimeError(
                    f"There was no fake impl registered for {self}. "
                    f"This is necessary for torch.compile/export/fx tracing to work. "
                    f"Please use `{self._init_fn.__name__}.register_fake` to add an "
                    f"fake impl."
                )
            return self._abstract_fn(*args, **kwargs)

        # TODO(rzou): I'm not sure why this needs to create a new library when
        # we pass one in?
        with allow_reserved_namespace_access():
            library.impl_abstract(self._qualname, lib=lib)(fake_impl)

    def __call__(self, *args, **kwargs):
        return self._opoverload(*args, **kwargs)


OPDEF_TO_LIB: Dict[str, "library.Library"] = {}


def get_library_allowing_overwrite(namespace: str, name: str) -> "library.Library":
    qualname = f"{namespace}::{name}"

    if qualname in OPDEF_TO_LIB:
        OPDEF_TO_LIB[qualname]._destroy()
        del OPDEF_TO_LIB[qualname]

    with allow_reserved_namespace_access():
        lib = library.Library(namespace, "FRAGMENT")
    OPDEF_TO_LIB[qualname] = lib
    return lib


def iter_tensors(
    args: Tuple[Any], kwargs: Dict[str, Any], allowed_nesting: int = 1
) -> Iterator[Tensor]:
    def check(arg):
        if isinstance(arg, Tensor):
            yield arg
        elif allowed_nesting > 0 and isinstance(arg, (tuple, list)):
            yield from iter_tensors(tuple(arg), {}, allowed_nesting - 1)

    for arg in args:
        yield from check(arg)
    for kwarg in kwargs.values():
        yield from check(kwarg)


# NOTE: [custom_op's automatic naming]
# If the user does not provide a manual qualname (e.g. mylib::linear) for the
# custom op, we will autogenerate one from the function passed to
# custom_op.
#
# The autogenerated namespace is {reserved_namespace()}; the autogenerated
# name is mangle(unique_name(fn)).
#
# The user should not depend on the details of the automatic naming (it is
# not robust to moving code around, among other things). To prevent this,
# we try our best to error out if we think the user is depending on it:
# - we disallow the user from creating a new Library object using the namespace
# - TODO(rzou): we disallow torch.export with custom ops with automatic names.
#
# Some of our custom ops infra need to access these though, so we provide
# some sidechannels (allow_reserved_namespace_access) to do so.

tls = threading.local()
tls.can_access_reserved_namespace = False


def reserved_namespace() -> str:
    return "DONT_USE_THIS_GIVE_EXPLICIT_NAMESPACE_IF_NEEDED"


def can_access_reserved_namespace() -> bool:
    return tls.can_access_reserved_namespace


@contextlib.contextmanager
def allow_reserved_namespace_access(allowed: bool = True) -> Iterator[None]:
    prev = tls.can_access_reserved_namespace
    try:
        tls.can_access_reserved_namespace = allowed
        yield
    finally:
        tls.can_access_reserved_namespace = prev
