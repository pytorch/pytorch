import inspect
import sys
from typing import Callable, Dict, List, Optional, Sequence, Union

from .. import _C, _library, library, Tensor


device_types_t = Optional[Union[str, Sequence[str]]]


def opaque_op(*, mutated_args: Sequence[str], types: device_types_t = None):
    """Wraps a function into an opaque custom operator.

    Treats the function as a black-box (that is, PyTorch will never peek into
    the function). The function must have type hints; these are needed
    to interface with PyTorch's various subsystems.

    This API is used as a decorator (please see examples).

    Args:
        mutated_args (Sequence[str]): The names of args that the function mutates.
            This MUST be accurate, otherwise, the behavior is undefined.
        types (None | str | Sequence[str]): The device type(s) the function
            is valid for. If no device type is provided, then the function
            is used as the default implementation for all device types.

    Examples::
        >>> import torch
        >>> from torch import Tensor
        >>> from torch.library import opaque_op
        >>> import numpy as np
        >>>
        >>> @opaque_op(mutated_args=())
        >>> def numpy_sin(x: Tensor) -> Tensor:
        >>>     x_np = x.cpu().numpy()
        >>>     y_np = np.sin(x_np)
        >>>     return torch.from_numpy(y_np).to(device=x.device)
        >>>
        >>> x = torch.randn(3)
        >>> y = numpy_sin(x)
        >>> assert torch.allclose(y, x.sin())
        >>>
        >>> # Example of a opaque op that only works for one device type.
        >>> @opaque_op(mutated_args=(), types="cpu")
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
    mod = get_caller_module(stacklevel=2)
    if mod is None:
        namespace = "module_not_found"  # can happen in colab
    else:
        namespace = mangle_module(mod.__name__)

    def inner(fn):
        return OpaqueOpDef._from_fn(fn, types, mutated_args, namespace)

    return inner


class OpaqueOpDef:
    def __init__(self, namespace: str, name: str, schema: str):
        self._namespace = namespace
        self._name = name
        self._qualname = f"{self._namespace}::{self._name}"
        self._schema = schema

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
        namespace: str,
    ):
        import torch

        schema = torch._custom_op.impl.infer_schema(fn, mutated_args)
        name = fn.__name__
        result = cls(namespace, name, schema)
        result.impl(types)(fn)
        return result

    def impl(self, types: device_types_t, fn: Optional[Callable] = None):
        """Register an implementation for a device type for this operator.

        Some valid types are: "cpu", "cuda", "xla", "mps", "ipu", "xpu".
        This API may be used as a decorator.

        Args:
            types (str | Sequence[str]): The device types to register an impl to.

        Examples::
            >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_CUDA)
            >>> import torch
            >>> from torch import Tensor
            >>> from torch.library import opaque_op
            >>> import numpy as np
            >>>
            >>> # Example of split cpu and cuda definitions
            >>> @opaque_op(mutated_args=(), types="cpu")
            >>> def numpy_sin(x: Tensor) -> Tensor:
            >>>     x_np = x.numpy()
            >>>     y_np = np.sin(x_np)
            >>>     return torch.from_numpy(y_np)
            >>>
            >>> # Add implementations for the cuda device
            >>> @numpy_sin.impl("cuda")
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
                        storages = set({})
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

    def impl_abstract(self, fn: Callable) -> Callable:
        r"""Register an abstract implementation for this operator.

        This is necessary to get the operator to work efficiently with torch.compile.

        An "abstract implementation" specifies the behavior of this operator on
        Tensors that carry no data. Given some input Tensors with certain properties
        (sizes/strides/storage_offset/device), it specifies what the properties of
        the output Tensors are.

        Please see :func:`torch.library.impl_abstract` for more details.

        Examples:
            >>> import torch
            >>> import numpy as np
            >>> from torch import Tensor
            >>>
            >>> # Example 1: an operator without data-dependent output shape
            >>> @torch.library.opaque_op(mutated_args=())
            >>> def custom_linear(x: Tensor, weight: Tensor, bias: Tensor) -> Tensor:
            >>>     return (x @ weight.t()) + bias
            >>>
            >>> @custom_linear.impl_abstract
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
            >>> @torch.library.opaque_op(mutated_args=())
            >>> def custom_nonzero(x: Tensor) -> Tensor:
            >>>     x_np = x.cpu().numpy()
            >>>     res = np.stack(np.nonzero(x_np), axis=1)
            >>>     return torch.tensor(res, device=x.device)
            >>>
            >>> @custom_nonzero.impl_abstract
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

    def _build(self):
        lib = self._lib
        lib.define(f"{self._name}{self._schema}")
        self._opoverload = _library.utils.lookup_op(self._qualname)

        def abstract_impl(*args, **kwargs):
            if self._abstract_fn is None:
                import torch

                raise torch._subclasses.fake_tensor.UnsupportedOperatorException(
                    self._opoverload
                )
            return self._abstract_fn(*args, **kwargs)

        library.impl_abstract(self._qualname, lib=lib)(abstract_impl)

    def __call__(self, *args, **kwargs):
        return self._opoverload(*args, **kwargs)


OPDEF_TO_LIB: Dict[str, "library.Library"] = {}


def get_library_allowing_overwrite(namespace: str, name: str) -> "library.Library":
    qualname = f"{namespace}::{name}"

    if qualname in OPDEF_TO_LIB:
        OPDEF_TO_LIB[qualname]._destroy()
        del OPDEF_TO_LIB[qualname]

    lib = library.Library(namespace, "FRAGMENT")
    OPDEF_TO_LIB[qualname] = lib
    return lib


def mangle_module(module):
    """Mangles the module name.

    The scheme is replacing dots with some number of underscores
    (specified as mangledN where N is the number of underscores).

    Examples:
    foo.bar.baz -> mangled1_foo_bar_baz
    foo_bar.baz -> mangled2__foo_bar__baz
    foo.__baz__ -> mangled3___foo_____baz__

    Don't parse the mangled string directly; use mangle_module and demangle_module
    """
    sep = unique_underscoring(module)
    prefix = f"mangled{len(sep)}"
    splits = module.split(".")
    return sep.join([prefix, *splits])


def unique_underscoring(s: str):
    i = 1
    while True:
        result = "_" * i
        if result not in s:
            return result
        i += 1


def get_caller_module(stacklevel=1):
    """Returns the fully qualified name of the module that called this function."""
    frame = sys._getframe(stacklevel)
    mod = inspect.getmodule(frame)
    return mod


def iter_tensors(args, kwargs, allowed_nesting=1):
    def check(arg):
        if isinstance(arg, Tensor):
            yield arg
        elif allowed_nesting > 0 and isinstance(arg, (tuple, list)):
            yield from iter_tensors(arg, {}, allowed_nesting - 1)

    for arg in args:
        yield from check(arg)
    for kwarg in kwargs.values():
        yield from check(kwarg)
