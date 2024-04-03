import inspect
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple, Union

from torch.utils._exposed_in import exposed_in

from .. import _C, _library, library, Tensor


device_types_t = Optional[Union[str, Sequence[str]]]


@exposed_in("torch.library")
def custom_op(
    name: str,
    /,
    *,
    mutated_args: Sequence[str],
    device_types: device_types_t = None,
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
        name (str): A name for the custom op that looks like "{namespace}::{name}",
            e.g. "mylib::my_linear". The name is used as a stable identifier for
            if you wish to serialize the custom op, e.g., via torch.save/torch.export.
            To avoid name collisions, please use your project name as the namespace.
        mutated_args (Sequence[str]): The names of args that the function mutates.
            This MUST be accurate, otherwise, the behavior is undefined.
        device_types (None | str | Sequence[str]): The device type(s) the function
            is valid for. If no device type is provided, then the function
            is used as the default implementation for all device types.
            Examples: "cpu", "cuda".

    Examples::
        >>> import torch
        >>> from torch import Tensor
        >>> from torch.library import custom_op
        >>> import numpy as np
        >>>
        >>> @custom_op("mylib::numpy_sin", mutated_args=())
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
        >>> @custom_op("mylib::numpy_sin_cpu", mutated_args=(), device_types="cpu")
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

    def inner(fn):
        import torch

        schema = torch._custom_op.impl.infer_schema(fn, mutated_args)
        namespace, opname = name.split("::")
        result = CustomOpDef(namespace, opname, schema, fn)
        result.register_impl(device_types)(fn)
        return result

    return inner


class CustomOpDef:
    """CustomOpDef is a wrapper around a function that turns it into a custom op.

    It has various methods for registering additional behavior for this
    custom op.

    You should not instantiate CustomOpDef directly; instead, use the
    :func:`torch.library.custom_op` API.
    """

    def __init__(self, namespace: str, name: str, schema: str, fn: Callable) -> None:
        # Fields used to interface with the PyTorch dispatcher
        self._namespace = namespace
        self._name = name
        self._schema = schema

        self._init_fn = fn

        self._backend_fns: Dict[Union[str, None], Callable] = {}
        self._abstract_fn: Optional[Callable] = None

        self._lib = get_library_allowing_overwrite(self._namespace, self._name)
        self._register_to_dispatcher()

    @property
    def _qualname(self) -> str:
        return f"{self._namespace}::{self._name}"

    def __repr__(self) -> str:
        return f"<CustomOpDef({self._qualname})>"

    def register_impl(
        self, device_types: device_types_t, fn: Optional[Callable] = None
    ) -> Callable:
        """Register an implementation for a device type for this operator.

        Some valid device_types are: "cpu", "cuda", "xla", "mps", "ipu", "xpu".
        This API may be used as a decorator.

        Args:
            fn (Callable): The function to register as the implementation for
                the given device types.
            device_types (str | Sequence[str]): The device device_types to register an impl to.

        Examples::
            >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_CUDA)
            >>> import torch
            >>> from torch import Tensor
            >>> from torch.library import custom_op
            >>> import numpy as np
            >>>
            >>> # Example of split cpu and cuda definitions
            >>> @custom_op("mylib::numpy_sin", mutated_args=(), device_types="cpu")
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
            if device_types is None or isinstance(device_types, str):
                dtypes: List[Union[str, None]] = [device_types]
            else:
                dtypes = list(device_types)
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
                        if not isinstance(result, tuple):
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

        # See NOTE: [Supporting decorator and non-decorator usage]
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

        Args:
            fn (Callable): The function to register as the FakeTensor
                implementation.

        Examples:
            >>> import torch
            >>> import numpy as np
            >>> from torch import Tensor
            >>>
            >>> # Example 1: an operator without data-dependent output shape
            >>> @torch.library.custom_op("mylib::linear", mutated_args=())
            >>> def linear(x: Tensor, weight: Tensor, bias: Tensor) -> Tensor:
            >>>     return (x @ weight.t()) + bias
            >>>
            >>> @linear.register_fake
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
            >>> out = torch.compile(linear, fullgraph=True)(x, weight, bias)
            >>> # xdoctest: +SKIP("Requires Python <= 3.11")
            >>> assert torch.allclose(out, torch.nn.functional.linear(x, weight, bias))
            >>>
            >>> # Example 2: an operator with data-dependent output shape
            >>> @torch.library.custom_op("mylib::nonzero", mutated_args=())
            >>> def nonzero(x: Tensor) -> Tensor:
            >>>     x_np = x.cpu().numpy()
            >>>     res = np.stack(np.nonzero(x_np), axis=1)
            >>>     return torch.tensor(res, device=x.device)
            >>>
            >>> @nonzero.register_fake
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
            >>> out = torch.compile(nonzero, fullgraph=True)(x)
            >>> # xdoctest: +SKIP("Requires Python <= 3.11")
            >>> assert torch.allclose(out, x.nonzero())

        """
        self._abstract_fn = fn
        return fn

    def _register_to_dispatcher(self) -> None:
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

        library.impl_abstract(self._qualname, lib=lib)(fake_impl)

    def __call__(self, *args, **kwargs):
        return self._opoverload(*args, **kwargs)


# NOTE: [Supporting decorator and non-decorator usage]
#
# Some APIs may be both used as a decorator and not as a decorator.
# For example:
#
# >>> def fn(x):
# >>>     return x.sin()
# >>>
# >>> # Usage 1: not as a decorator
# >>> numpy_sin.register_impl("cuda", fn)
# >>>
# >>> # Usage 2: as a decorator
# >>> @numpy_sin.register_impl("cuda")
# >>> def fn2(x):
# >>>     return x.sin
#
# The way we support this is that `register_impl` accepts an optional `fn`.
# If `fn` is provided (Usage 1), then we know that the user is using it not
# as a decorator.
# If `fn` is not provided (Usage 2), then `register_impl` needs to return a
# decorator.


OPDEF_TO_LIB: Dict[str, "library.Library"] = {}


def get_library_allowing_overwrite(namespace: str, name: str) -> "library.Library":
    qualname = f"{namespace}::{name}"

    if qualname in OPDEF_TO_LIB:
        OPDEF_TO_LIB[qualname]._destroy()
        del OPDEF_TO_LIB[qualname]

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
