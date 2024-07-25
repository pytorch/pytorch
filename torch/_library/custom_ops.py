# mypy: allow-untyped-defs
import inspect
import logging
import weakref
from contextlib import contextmanager
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    Literal,
    Optional,
    overload,
    Sequence,
    Set,
    Tuple,
    Union,
)

from torch.utils._exposed_in import exposed_in

from .. import _C, _library, _ops, autograd, library, Tensor
from . import utils


device_types_t = Optional[Union[str, Sequence[str]]]
log = logging.getLogger(__name__)


@overload
def custom_op(
    name: str,
    fn: Literal[None] = None,
    /,
    *,
    mutates_args: Union[str, Iterable[str]],
    device_types: device_types_t = None,
    schema: Optional[str] = None,
) -> Callable[[Callable[..., object]], "CustomOpDef"]:
    ...


@overload
def custom_op(
    name: str,
    fn: Callable[..., object],
    /,
    *,
    mutates_args: Union[str, Iterable[str]],
    device_types: device_types_t = None,
    schema: Optional[str] = None,
) -> "CustomOpDef":
    ...


@exposed_in("torch.library")
def custom_op(
    name: str,
    fn: Optional[Callable] = None,
    /,
    *,
    mutates_args: Union[str, Iterable[str]],
    device_types: device_types_t = None,
    schema: Optional[str] = None,
) -> Union[Callable[[Callable[..., object]], "CustomOpDef"], "CustomOpDef"]:
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
            e.g. "mylib::my_linear". The name is used as the op's stable identifier
            in PyTorch subsystems (e.g. torch.export, FX graphs).
            To avoid name collisions, please use your project name as the namespace;
            e.g. all custom ops in pytorch/fbgemm use "fbgemm" as the namespace.
        mutates_args (Iterable[str] or "unknown"): The names of args that the function mutates.
            This MUST be accurate, otherwise, the behavior is undefined. If "unknown",
            it pessimistically assumes that all inputs to the operator are being mutated.
        device_types (None | str | Sequence[str]): The device type(s) the function
            is valid for. If no device type is provided, then the function
            is used as the default implementation for all device types.
            Examples: "cpu", "cuda".
            When registering a device-specific implementation for an operator that accepts no Tensors,
            we require the operator to have a "device: torch.device argument".
        schema (None | str): A schema string for the operator. If None
            (recommended) we'll infer a schema for the operator from its type
            annotations. We recommend letting us infer a schema unless you
            have a specific reason not to.
            Example: "(Tensor x, int y) -> (Tensor, Tensor)".

    .. note::
        We recommend not passing in a ``schema`` arg and instead letting us infer
        it from the type annotations. It is error-prone to write your own schema.
        You may wish to provide your own schema if our interpretation of
        the type annotation is not what you want.
        For more info on how to write a schema string, see
        `here <https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/README.md#func>`_

    Examples::
        >>> import torch
        >>> from torch import Tensor
        >>> from torch.library import custom_op
        >>> import numpy as np
        >>>
        >>> @custom_op("mylib::numpy_sin", mutates_args=())
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
        >>> @custom_op("mylib::numpy_sin_cpu", mutates_args=(), device_types="cpu")
        >>> def numpy_sin_cpu(x: Tensor) -> Tensor:
        >>>     x_np = x.numpy()
        >>>     y_np = np.sin(x_np)
        >>>     return torch.from_numpy(y_np)
        >>>
        >>> x = torch.randn(3)
        >>> y = numpy_sin_cpu(x)
        >>> assert torch.allclose(y, x.sin())
        >>>
        >>> # Example of a custom op that mutates an input
        >>> @custom_op("mylib::numpy_sin_inplace", mutates_args={"x"}, device_types="cpu")
        >>> def numpy_sin_inplace(x: Tensor) -> None:
        >>>     x_np = x.numpy()
        >>>     np.sin(x_np, out=x_np)
        >>>
        >>> x = torch.randn(3)
        >>> expected = x.sin()
        >>> numpy_sin_inplace(x)
        >>> assert torch.allclose(x, expected)
        >>>
        >>> # Example of a factory function
        >>> @torch.library.custom_op("mylib::bar", mutates_args={}, device_types="cpu")
        >>> def bar(device: torch.device) -> Tensor:
        >>>     return torch.ones(3)
        >>>
        >>> bar("cpu")

    """

    def inner(fn: Callable[..., object]) -> CustomOpDef:
        import torch

        if schema is None:
            import torch._custom_op.impl

            schema_str = torch._custom_op.impl.infer_schema(fn, mutates_args)
        else:
            schema_str = schema

        namespace, opname = name.split("::")
        result = CustomOpDef(namespace, opname, schema_str, fn)
        if schema is not None:
            # Check that schema's alias annotations match those of `mutates_args`.
            expected = set()
            for arg in result._opoverload._schema.arguments:
                if arg.alias_info is not None and arg.alias_info.is_write:
                    expected.add(arg.name)
            if expected != set(mutates_args):
                raise ValueError(
                    f"Attempted to create a custom op with `mutates_args={mutates_args}` "
                    f"and `schema={schema}. The schema suggests that the op mutates {expected}"
                    f"which is different from what was provided to us in `mutates_args`. "
                    f"Please make these consistent."
                )
        result.register_kernel(device_types)(fn)
        return result

    if fn is None:
        return inner
    return inner(fn)


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
        self._setup_context_fn: Optional[Callable] = None
        self._backward_fn: Optional[Callable] = None
        self._torch_dispatch_fns: Dict[type, Callable] = {}
        self._vmap_fn: Optional[Callable] = None

        self._lib = get_library_allowing_overwrite(self._namespace, self._name)
        self._register_to_dispatcher()
        self._disabled_kernel: Set = set()
        OPDEFS[self._qualname] = self

    @property
    def _qualname(self) -> str:
        return f"{self._namespace}::{self._name}"

    def __repr__(self) -> str:
        return f"<CustomOpDef({self._qualname})>"

    @contextmanager
    def set_kernel_enabled(self, device_type: str, enabled: bool = True):
        """
        Disable or re-enable an already registered kernel for this custom operator.

        If the kernel is already disabled/enabled, this is a no-op.

        Note:
            If a kernel is first disabled and then registered, it is disabled until enabled again.

        Args:
            device_type (str): The device type to disable/enable the kernel for.
            disable (bool): Whether to disable or enable the kernel.

        Example:
            >>> inp = torch.randn(1)
            >>>
            >>> # define custom op `f`.
            >>> @custom_op("mylib::f", mutates_args=())
            >>> def f(x: Tensor) -> Tensor:
            >>>     return torch.zeros(1)
            >>>
            >>> print(f(inp))  # tensor([0.]), default kernel
            >>>
            >>> @f.register_kernel("cpu")
            >>> def _(x):
            >>>     return torch.ones(1)
            >>>
            >>> print(f(inp))  # tensor([1.]), CPU kernel
            >>>
            >>> # temporarily disable the CPU kernel
            >>> with f.set_kernel_enabled("cpu", enabled = False):
            >>>     print(f(inp))  # tensor([0.]) with CPU kernel disabled

        """
        action = "enable" if enabled else "disable"
        originally_disabled = device_type in self._disabled_kernel
        if device_type not in self._backend_fns:
            log.warning(
                "Attempted to %s kernel for %s but no kernel was registered for this device type.",
                action,
                device_type,
            )

        if not enabled:
            if originally_disabled:
                log.warning(
                    "Attempted to disable kernel for %s but it was already disabled.",
                    device_type,
                )
            else:
                self._disabled_kernel.add(device_type)
        else:  # enable the kernel
            if not originally_disabled:
                log.warning(
                    "Attempted to enable kernel for  %s but it was already enabled.",
                    device_type,
                )
            else:
                self._disabled_kernel.remove(device_type)

        try:
            yield
        finally:
            # restore original state
            if originally_disabled:
                self._disabled_kernel.add(device_type)
            else:
                self._disabled_kernel.discard(device_type)

    def register_kernel(
        self, device_types: device_types_t, fn: Optional[Callable] = None, /
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
            >>> # Create a custom op that works on cpu
            >>> @custom_op("mylib::numpy_sin", mutates_args=(), device_types="cpu")
            >>> def numpy_sin(x: Tensor) -> Tensor:
            >>>     x_np = x.numpy()
            >>>     y_np = np.sin(x_np)
            >>>     return torch.from_numpy(y_np)
            >>>
            >>> # Add implementations for the cuda device
            >>> @numpy_sin.register_kernel("cuda")
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
                        storages = {
                            id(tensor.untyped_storage())
                            for tensor in iter_tensors(args, kwargs)
                        }

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

                # Wrap function to choose between the default implementation or the device-specific
                # implementation depending on if the kernel is disabled.
                def wrapped_fn(*args, **kwargs):
                    if device_type in self._disabled_kernel:
                        return self._init_fn(*args, **kwargs)
                    else:
                        return fn(*args, **kwargs)

                self._backend_fns[device_type] = wrapped_fn
            return fn

        from torch._library.utils import get_device_arg_index, has_tensor_arg

        if device_types is not None and not has_tensor_arg(self._opoverload._schema):
            device_arg_index = get_device_arg_index(self._opoverload._schema)
            if device_arg_index is None:
                raise ValueError(
                    "Functions without tensor inputs are required to have a `device: torch.device` argument"
                )
            self._register_backend_select_dispatcher(device_arg_index)

        # See NOTE: [Supporting decorator and non-decorator usage]
        if fn is None:
            return inner
        return inner(fn)

    def register_fake(self, fn: Callable, /) -> Callable:
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
            >>> @torch.library.custom_op("mylib::linear", mutates_args=())
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
            >>> @torch.library.custom_op("mylib::nonzero", mutates_args=())
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

    def register_torch_dispatch(
        self, torch_dispatch_class: Any, fn: Optional[Callable] = None, /
    ) -> Callable:
        r"""Registers a torch_dispatch rule for the given operator and ``torch_dispatch_class``.

        This allows for open registration to specify the behavior between the operator
        and the ``torch_dispatch_class`` without needing to modify the ``torch_dispatch_class``
        or the operator directly.

        Please see :func:`torch.library.register_torch_dispatch` for examples and more details.
        """

        def register(fn):
            if torch_dispatch_class not in self._torch_dispatch_fns:

                def inner(*args, **kwargs):
                    return self._torch_dispatch_fns[torch_dispatch_class](
                        *args, **kwargs
                    )

                self._lib._register_torch_dispatch_rule(
                    self._name, torch_dispatch_class, inner
                )
            self._torch_dispatch_fns[torch_dispatch_class] = fn
            return fn

        if fn is None:
            return register
        else:
            return register(fn)

    def register_autograd(
        self,
        backward: Callable,
        /,
        *,
        setup_context: Optional[Callable] = None,
    ) -> None:
        r"""Register a backward formula for this custom op.

        In order for an operator to work with autograd, you need to register
        a backward formula:
        1. You must tell us how to compute gradients during the backward pass
        by providing us a "backward" function.
        2. If you need any values from the forward to compute gradients, you can
        use `setup_context` to save values for backward.

        ``backward_fn`` runs during the backward pass. It accepts ``(ctx, *grads)``:
        - ``grads`` is one or more gradients. The number of gradients matches
        the number of outputs of the operator.
        The ``ctx`` object is `the same ctx object <context_method_mixins>`_ used by
        :class:`torch.autograd.Function`. The semantics of ``backward_fn`` are the
        same as :meth:`torch.autograd.Function.backward`.

        ``setup_context(ctx, inputs, output)`` runs during the forward pass.
        Please save quantities needed for backward onto the ``ctx`` object via
        either :meth:`torch.autograd.function.FunctionCtx.save_for_backward`
        or assigning them as attributes of ``ctx``. If your custom op has
        kwarg-only arguments, we expect the signature of ``setup_context``
        to be ``setup_context(ctx, inputs, keyword_only_inputs, output)``.

        Both ``setup_context_fn`` and ``backward_fn`` must be traceable. That is,
        they may not directly access :meth:`torch.Tensor.data_ptr` and they must
        not depend on or mutate global state. If you need a non-traceable backward,
        you can make it a separate custom_op that you call inside ``backward_fn``.

        Examples:
            >>> import torch
            >>> import numpy as np
            >>> from torch import Tensor
            >>>
            >>> @torch.library.custom_op("mylib::numpy_sin", mutates_args=())
            >>> def numpy_sin(x: Tensor) -> Tensor:
            >>>     x_np = x.cpu().numpy()
            >>>     y_np = np.sin(x_np)
            >>>     return torch.from_numpy(y_np).to(device=x.device)
            >>>
            >>> def setup_context(ctx, inputs, output) -> Tensor:
            >>>     x, = inputs
            >>>     ctx.save_for_backward(x)
            >>>
            >>> def backward(ctx, grad):
            >>>     x, = ctx.saved_tensors
            >>>     return grad * x.cos()
            >>>
            >>> numpy_sin.register_autograd(backward, setup_context=setup_context)
            >>>
            >>> x = torch.randn(3, requires_grad=True)
            >>> y = numpy_sin(x)
            >>> grad_x, = torch.autograd.grad(y, x, torch.ones_like(y))
            >>> assert torch.allclose(grad_x, x.cos())
            >>>
            >>> # Example with a keyword-only arg
            >>> @torch.library.custom_op("mylib::numpy_mul", mutates_args=())
            >>> def numpy_mul(x: Tensor, *, val: float) -> Tensor:
            >>>     x_np = x.cpu().numpy()
            >>>     y_np = x_np * val
            >>>     return torch.from_numpy(y_np).to(device=x.device)
            >>>
            >>> def setup_context(ctx, inputs, keyword_only_inputs, output) -> Tensor:
            >>>     ctx.val = keyword_only_inputs["val"]
            >>>
            >>> def backward(ctx, grad):
            >>>     return grad * ctx.val
            >>>
            >>> numpy_mul.register_autograd(backward, setup_context=setup_context)
            >>>
            >>> x = torch.randn(3, requires_grad=True)
            >>> y = numpy_mul(x, val=3.14)
            >>> grad_x, = torch.autograd.grad(y, x, torch.ones_like(y))
            >>> assert torch.allclose(grad_x, torch.full_like(x, 3.14))

        """
        schema = self._opoverload._schema
        if not _library.utils.is_functional_schema(schema):
            raise RuntimeError(
                f"Cannot register autograd formula for non-functional operator "
                f"{self} with schema {schema}. Please create "
                f"a functional operator and register an autograd formula for that."
            )

        self._backward_fn = backward
        self._setup_context_fn = setup_context

    def _register_to_dispatcher(self) -> None:
        lib = self._lib
        schema_str = self._name + self._schema
        cpp_schema = _C.parse_schema(schema_str)
        if utils.has_kwarg_only_tensors(cpp_schema):
            # If you want to support this, the progression is:
            # - supporting kwarg-only Tensors that are non-differentiable
            # - supporting kwarg-only Tensors (regardless of differentiability)
            raise NotImplementedError(
                f"custom_op with kwarg-only Tensor args. Please make your "
                f"tensors not kwarg-only. Got: {schema_str}"
            )

        lib.define(
            schema_str,
            tags=[_C.Tag.pt2_compliant_tag, _C.Tag.needs_fixed_stride_order],
        )
        self._opoverload = _library.utils.lookup_op(self._qualname)

        def fake_impl(*args, **kwargs):
            if self._abstract_fn is None:
                if _library.utils.can_generate_trivial_fake_impl(self._opoverload):
                    return None
                raise RuntimeError(
                    f"There was no fake impl registered for {self}. "
                    f"This is necessary for torch.compile/export/fx tracing to work. "
                    f"Please use `{self._init_fn.__name__}.register_fake` to add an "
                    f"fake impl."
                )
            return self._abstract_fn(*args, **kwargs)

        lib._register_fake(self._name, fake_impl, _stacklevel=4)

        autograd_impl = _library.autograd.make_autograd_impl(self._opoverload, self)
        lib.impl(self._name, autograd_impl, "Autograd", with_keyset=True)

        schema = self._opoverload._schema
        if schema.is_mutable:

            def adinplaceorview_impl(keyset, *args, **kwargs):
                for arg, val in _library.utils.zip_schema(schema, args, kwargs):
                    if not arg.alias_info:
                        continue
                    if not arg.alias_info.is_write:
                        continue
                    if isinstance(val, Tensor):
                        autograd.graph.increment_version(val)
                    elif isinstance(val, (tuple, list)):
                        for v in val:
                            if isinstance(v, Tensor):
                                autograd.graph.increment_version(v)
                with _C._AutoDispatchBelowADInplaceOrView():
                    return self._opoverload.redispatch(
                        keyset & _C._after_ADInplaceOrView_keyset, *args, **kwargs
                    )

            lib.impl(
                self._name,
                adinplaceorview_impl,
                "ADInplaceOrView",
                with_keyset=True,
            )

    def _register_backend_select_dispatcher(self, device_arg_index: int):
        """
        Switch on the device argument to select the correct backend to dispatch to.
        """

        def backend_select(keyset, *args, **kwargs):
            device = args[device_arg_index].type
            if device not in self._backend_fns:
                raise RuntimeError(
                    f"{self._name} does not have a kernel registered for {device}. "
                    "Please use register_kernel to do so."
                )
            dispatch_key = _C._dispatch_key_for_device(device)
            dispatch_key = getattr(_C.DispatchKey, dispatch_key)
            return self._opoverload.redispatch(
                _C.DispatchKeySet(dispatch_key), *args, **kwargs
            )

        self._lib.impl(self._name, backend_select, "BackendSelect", with_keyset=True)

    def __call__(self, *args, **kwargs):
        return self._opoverload(*args, **kwargs)

    def register_vmap(
        self,
        func: Optional[Callable] = None,
    ):
        r"""Register a vmap implementation to support :func:`torch.vmap` for this custom op.

        This API may be used as a decorator.

        In order for an operator to work with :func:`torch.vmap`, you may need to register a
        vmap implementation in the following signature:

            ``vmap_func(info, in_dims: Tuple[Optional[int]], *args, **kwargs)``,

        where ``*args`` and ``**kwargs`` are the arguments and kwargs for ``op``.

        It specifies how do we compute the batched version of ``op`` given inputs with an additional
        dimension (specified by ``in_dims``).

        For each arg in ``args``, ``in_dims`` has a corresponding ``Optional[int]``. It is ``None``
        if the arg is not a Tensor or if the arg is not being vmapped over, otherwise, it is an integer
        specifying what dimension of the Tensor is being vmapped over.

        ``info`` is a collection of additional metadata that may be helpful:
        ``info.batch_size`` specifies the size of the dimension being vmapped over, while
        ``info.randomness`` is the ``randomness`` option that was passed to :func:`torch.vmap`.

        The return of the function ``func`` is a tuple of ``(output, out_dims)``. Similar to ``in_dims``,
        ``out_dims`` should be of the same structure as ``output`` and contain one ``out_dim``
        per output that specifies if the output has the vmapped dimension and what index it is in.

        Examples:
            >>> import torch
            >>> import numpy as np
            >>> from torch import Tensor
            >>> from typing import Tuple
            >>>
            >>> def to_numpy(tensor):
            >>>     return tensor.cpu().numpy()
            >>>
            >>> lib = torch.library.Library("mylib", "FRAGMENT")
            >>> @torch.library.custom_op("mylib::numpy_cube", mutates_args=())
            >>> def numpy_cube(x: Tensor) -> Tuple[Tensor, Tensor]:
            >>>     x_np = to_numpy(x)
            >>>     dx = torch.tensor(3 * x_np ** 2, device=x.device)
            >>>     return torch.tensor(x_np ** 3, device=x.device), dx
            >>>
            >>> def numpy_cube_vmap(info, in_dims, x):
            >>>     result = numpy_cube(x)
            >>>     return result, (in_dims[0], in_dims[0])
            >>>
            >>> numpy_cube.register_vmap(numpy_cube_vmap)
            >>>
            >>> x = torch.randn(3)
            >>> torch.vmap(numpy_cube)(x)
            >>>
            >>> @torch.library.custom_op("mylib::numpy_mul", mutates_args=())
            >>> def numpy_mul(x: Tensor, y: Tensor) -> Tensor:
            >>>     return torch.tensor(to_numpy(x) * to_numpy(y), device=x.device)
            >>>
            >>> @numpy_mul.register_vmap
            >>> def numpy_mul_vmap(info, in_dims, x, y):
            >>>     x_bdim, y_bdim = in_dims
            >>>     x = x.movedim(x_bdim, -1) if x_bdim is not None else x.unsqueeze(-1)
            >>>     y = y.movedim(y_bdim, -1) if y_bdim is not None else y.unsqueeze(-1)
            >>>     result = x * y
            >>>     result = result.movedim(-1, 0)
            >>>     return result, 0
            >>>
            >>>
            >>> x = torch.randn(3)
            >>> y = torch.randn(3)
            >>> torch.vmap(numpy_mul)(x, y)
        """
        from torch._functorch.autograd_function import custom_function_call_vmap_helper
        from torch._functorch.pyfunctorch import retrieve_current_functorch_interpreter

        def register(func):
            need_register = self._vmap_fn is None
            self._vmap_fn = func

            if need_register:

                def wrapped_func(keyset, *args, **kwargs):
                    interpreter = retrieve_current_functorch_interpreter()
                    return custom_function_call_vmap_helper(
                        interpreter, self._vmap_fn, self._opoverload, *args, **kwargs
                    )

                self._lib.impl(
                    self._name, wrapped_func, "FuncTorchBatched", with_keyset=True
                )

        if func is None:
            return register
        else:
            return register(func)


# NOTE: [Supporting decorator and non-decorator usage]
#
# Some APIs may be both used as a decorator and not as a decorator.
# For example:
#
# >>> def fn(x):
# >>>     return x.sin()
# >>>
# >>> # Usage 1: not as a decorator
# >>> numpy_sin.register_kernel("cuda", fn)
# >>>
# >>> # Usage 2: as a decorator
# >>> @numpy_sin.register_kernel("cuda")
# >>> def fn2(x):
# >>>     return x.sin
#
# The way we support this is that `register_kernel` accepts an optional `fn`.
# If `fn` is provided (Usage 1), then we know that the user is using it not
# as a decorator.
# If `fn` is not provided (Usage 2), then `register_kernel` needs to return a
# decorator.


OPDEF_TO_LIB: Dict[str, "library.Library"] = {}
OPDEFS: weakref.WeakValueDictionary = weakref.WeakValueDictionary()


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


def _maybe_get_opdef(
    op: Union[CustomOpDef, _ops.OpOverload, str]
) -> Optional[CustomOpDef]:
    if isinstance(op, CustomOpDef):
        return op
    if isinstance(op, _ops.OpOverload):
        op = op._name
    assert isinstance(op, str)
    if op in OPDEFS:
        return OPDEFS[op]
    return None
