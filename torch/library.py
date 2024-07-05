# mypy: allow-untyped-defs
import contextlib
import functools
import inspect
import re
import sys
import traceback
import weakref
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple, Union
from typing_extensions import deprecated

import torch
import torch._library as _library
from torch._library.custom_ops import (
    _maybe_get_opdef,
    custom_op,
    CustomOpDef,
    device_types_t,
)
from torch._library.infer_schema import infer_schema  # noqa: F401
from torch._ops import OpOverload


__all__ = [
    "Library",
    "impl",
    "define",
    "fallthrough_kernel",
    "impl_abstract",
    "register_fake",
    "get_ctx",
    "custom_op",
]

# Set containing the combination of (namespace, operator, DispatchKey) for which a new kernel has been registered
# The keys in the set are of the form `namespace + "/" + op_name + "/" + dispatch_key`.
# This set is maintained to ensure that two libraries don't try to override the exact same functionality to avoid
# libraries calling into kernels not intended to be called.
_impls: Set[str] = set()
_defs: Set[str] = set()

# prim is reserved by TorchScript interpreter
_reserved_namespaces = ["prim"]


def fallthrough_kernel():
    """
    A dummy function to pass to ``Library.impl`` in order to register a fallthrough.
    """
    raise NotImplementedError("fallthrough_kernel() should never be called.")


class Library:
    """
    A class to create libraries that can be used to register new operators or
    override operators in existing libraries from Python.
    A user can optionally pass in a dispatch keyname if they only want to register
    kernels corresponding to only one specific dispatch key.

    To create a library to override operators in an existing library (with name ns), set the kind to "IMPL".
    To create a new library (with name ns) to register new operators, set the kind to "DEF".
    To create a fragment of a possibly existing library to register operators (and bypass
    the limitation that there is only one library for a given namespace), set the kind to
    "FRAGMENT".

    Args:
        ns: library name
        kind: "DEF", "IMPL" (default: "IMPL"), "FRAGMENT"
        dispatch_key: PyTorch dispatch key (default: "")
    """

    def __init__(self, ns, kind, dispatch_key=""):
        if kind not in ("IMPL", "DEF", "FRAGMENT"):
            raise ValueError("Unsupported kind: ", kind)

        if ns in _reserved_namespaces and (kind == "DEF" or kind == "FRAGMENT"):
            raise ValueError(
                ns,
                " is a reserved namespace. Please try creating a library with another name.",
            )

        frame = traceback.extract_stack(limit=3)[0]
        filename, lineno = frame.filename, frame.lineno
        self.m: Optional[Any] = torch._C._dispatch_library(
            kind, ns, dispatch_key, filename, lineno
        )
        self.ns = ns
        self._op_defs: Set[str] = set()
        self._op_impls: Set[str] = set()
        self._registration_handles: List[torch._library.utils.RegistrationHandle] = []
        self.kind = kind
        self.dispatch_key = dispatch_key
        # Use a finalizer to setup the "destructor" instead of __del__.
        # Python __del__ can lead to weird things (globals and locals may already
        # be gone when __del__ actually gets called!). finalizers help the
        # situation because it lets us capture references and keeps them alive
        weakref.finalize(
            self,
            _del_library,
            _impls,
            self._op_impls,
            _defs,
            self._op_defs,
            self._registration_handles,
        )

    def __repr__(self):
        return f"Library(kind={self.kind}, ns={self.ns}, dispatch_key={self.dispatch_key})>"

    def define(self, schema, alias_analysis="", *, tags=()):
        r"""Defines a new operator and its semantics in the ns namespace.

        Args:
            schema: function schema to define a new operator.
            alias_analysis (optional): Indicates if the aliasing properties of the operator arguments can be
                                       inferred from the schema (default behavior) or not ("CONSERVATIVE").
            tags (Tag | Sequence[Tag]): one or more torch.Tag to apply to this
                                       operator. Tagging an operator changes the operator's behavior
                                       under various PyTorch subsystems; please read the docs for the
                                       torch.Tag carefully before applying it.

        Returns:
            name of the operator as inferred from the schema.

        Example::
            >>> my_lib = Library("mylib", "DEF")
            >>> my_lib.define("sum(Tensor self) -> Tensor")
        """
        # This is added because we also want to disallow PURE_FUNCTION alias analysis which is a valid
        # AliasAnalysis type in C++
        if alias_analysis not in ["", "FROM_SCHEMA", "CONSERVATIVE"]:
            raise RuntimeError(f"Invalid alias_analysis type {alias_analysis}")
        assert self.m is not None
        if isinstance(tags, torch.Tag):
            tags = (tags,)

        name = schema.split("(")[0]
        packet_name = name.split(".")[0] if "." in name else name
        has_preexisting_packet = hasattr(torch.ops, self.ns) and hasattr(
            getattr(torch.ops, self.ns), packet_name
        )

        result = self.m.define(schema, alias_analysis, tuple(tags))
        name = schema.split("(")[0]
        qualname = self.ns + "::" + name

        # If the OpOverloadPacket exists already, then this means we're adding a
        # new OpOverload for it. Refresh the packet to include the new OpOverload.
        if has_preexisting_packet:
            ns = getattr(torch.ops, self.ns)
            packet = getattr(ns, packet_name)
            torch._ops._refresh_packet(packet)

        self._op_defs.add(qualname)
        _defs.add(qualname)
        return result

    def _register_fake(self, op_name, fn, _stacklevel=1):
        r"""Registers the fake impl for an operator defined in the library."""
        source = torch._library.utils.get_source(_stacklevel + 1)
        frame = sys._getframe(_stacklevel)
        caller_module = inspect.getmodule(frame)
        # Can be none if you call register_fake from somewhere there isn't a module
        # (e.g. __main__)
        caller_module_name = None if caller_module is None else caller_module.__name__

        # TODO(rzou): We're gonna need to stage this change with torchvision,
        # since torchvision is github first.
        if caller_module_name is not None and caller_module_name.startswith(
            "torchvision."
        ):
            caller_module_name = None

        qualname = f"{self.ns}::{op_name}"
        entry = torch._library.simple_registry.singleton.find(qualname)
        if caller_module_name is not None:
            func_to_register = _check_pystubs_once(fn, qualname, caller_module_name)
        else:
            func_to_register = fn

        handle = entry.fake_impl.register(func_to_register, source)
        self._registration_handles.append(handle)

    def _impl_with_aoti_compile(self, op_name, dispatch_key=""):
        r"""Register the operator to use the AOTI-compiled implementation.

        Args:
            op_name: operator name (along with the overload) or OpOverload object.
            dispatch_key: dispatch key that the input function should be registered for. By default, it uses
                          the dispatch key that the library was created with.

        Example::
            >>> my_lib = Library("aten", "IMPL")
            >>> my_lib._impl_with_aoti_compile("div.Tensor", "CPU")
        """
        if dispatch_key == "":
            dispatch_key = self.dispatch_key
        assert torch.DispatchKeySet(dispatch_key).has(torch._C.DispatchKey.Dense)

        if isinstance(op_name, str):
            name = op_name
        elif isinstance(op_name, OpOverload):
            name = op_name._schema.name
            overload_name = op_name._schema.overload_name
            if overload_name != "":
                name = name + "." + overload_name
        else:
            raise RuntimeError(
                "_impl_with_aoti_compile should be passed either a name or an OpOverload object "
                "as the first argument"
            )

        key = self.ns + "/" + name.split("::")[-1] + "/" + dispatch_key
        if key in _impls:
            # TODO: in future, add more info about where the existing function is registered (this info is
            # today already returned by the C++ warning when _impl_with_aoti_compile is called but we error out before that)
            raise RuntimeError(
                "This is not allowed since there's already a kernel registered from python overriding {}"
                "'s behavior for {} dispatch key and {} namespace.".format(
                    name.split("::")[-1], dispatch_key, self.ns
                )
            )

        assert self.m is not None
        impl_fn: Callable = self.m.impl_with_aoti_compile
        impl_fn(self.ns, name.split("::")[-1], dispatch_key)

        _impls.add(key)
        self._op_impls.add(key)

    def impl(self, op_name, fn, dispatch_key="", *, with_keyset=False):
        r"""Registers the function implementation for an operator defined in the library.

        Args:
            op_name: operator name (along with the overload) or OpOverload object.
            fn: function that's the operator implementation for the input dispatch key or :func:`~fallthrough_kernel`
                to register a fallthrough.
            dispatch_key: dispatch key that the input function should be registered for. By default, it uses
                          the dispatch key that the library was created with.

        Example::
            >>> my_lib = Library("aten", "IMPL")
            >>> def div_cpu(self, other):
            >>>     return self * (1 / other)
            >>> my_lib.impl("div.Tensor", div_cpu, "CPU")
        """
        if not callable(fn):
            raise TypeError(
                f"Input function is required to be a callable but found type {type(fn)}"
            )
        if dispatch_key == "":
            dispatch_key = self.dispatch_key

        if isinstance(op_name, str):
            name = op_name
        elif isinstance(op_name, OpOverload):
            name = op_name._schema.name
            overload_name = op_name._schema.overload_name
            if overload_name != "":
                name = name + "." + overload_name
        else:
            raise RuntimeError(
                "impl should be passed either a name or an OpOverload object as the first argument"
            )

        key = self.ns + "/" + name.split("::")[-1] + "/" + dispatch_key
        if key in _impls:
            # TODO: in future, add more info about where the existing function is registered (this info is
            # today already returned by the C++ warning when impl is called but we error out before that)
            raise RuntimeError(
                "This is not allowed since there's already a kernel registered from python overriding {}"
                "'s behavior for {} dispatch key and {} namespace.".format(
                    name.split("::")[-1], dispatch_key, self.ns
                )
            )

        if dispatch_key == "Meta":
            dispatcher_op_name = name
            if "::" not in dispatcher_op_name:
                dispatcher_op_name = f"{self.ns}::{dispatcher_op_name}"

            # Internally, we shouldn't be registering meta kernels for any operators that
            # have CompositeImplicitAutograd kernels.
            # Instead, we should be letting those decompositions run, and writing meta kernels
            # only for the base operators.
            if torch._C._dispatch_has_kernel_for_dispatch_key(
                dispatcher_op_name, "CompositeImplicitAutograd"
            ):
                raise RuntimeError(
                    f"We should not register a meta kernel directly to the operator '{name}',"
                    " because it has a CompositeImplicitAutograd kernel in core."
                    " Instead we should let the operator decompose, and ensure that we have meta kernels"
                    " for the base ops that it decomposes into."
                )

        assert self.m is not None
        self.m.impl(
            name,
            dispatch_key if dispatch_key != "" else "CompositeImplicitAutograd",
            fn,
            with_keyset,
        )

        _impls.add(key)
        self._op_impls.add(key)

    def _destroy(self):
        if self.m is not None:
            self.m.reset()
        self.m = None
        for handle in self._registration_handles:
            handle.destroy()
        self._registration_handles.clear()
        global _impls
        _impls -= self._op_impls
        for name in self._op_defs:
            # Delete the cached torch.ops.ns.foo if it was registered.
            # Otherwise, accessing it leads to a segfault.
            # It's possible that we only registered an overload in this Library
            # and another library owns an alive overload.
            # That's OK - the next time torch.ops.ns.foo gets called, it'll be
            # recomputed to point at the right collection of overloads.
            ns, name_with_overload = name.split("::")
            name = name_with_overload.split(".")[0]
            if not hasattr(torch.ops, ns):
                continue
            namespace = getattr(torch.ops, ns)
            if not hasattr(namespace, name):
                continue
            delattr(namespace, name)


def _del_library(
    captured_impls,
    op_impls,
    captured_defs,
    op_defs,
    registration_handles,
):
    captured_impls -= op_impls
    captured_defs -= op_defs
    for handle in registration_handles:
        handle.destroy()


@contextlib.contextmanager
def _scoped_library(*args, **kwargs):
    try:
        lib = Library(*args, **kwargs)
        yield lib
    finally:
        lib._destroy()


_keep_alive: List[Library] = []


NAMELESS_SCHEMA = re.compile(r"\(.*\) -> .*")


@functools.singledispatch
def define(qualname, schema, *, lib=None, tags=()):
    r"""Defines a new operator.

    In PyTorch, defining an op (short for "operator") is a two step-process:
    - we need to define the op (by providing an operator name and schema)
    - we need to implement behavior for how the operator interacts with
    various PyTorch subsystems, like CPU/CUDA Tensors, Autograd, etc.

    This entrypoint defines the custom operator (the first step)
    you must then perform the second step by calling various
    ``impl_*`` APIs, like :func:`torch.library.impl` or
    :func:`torch.library.register_fake`.

    Args:
        qualname (str): The qualified name for the operator. Should be
            a string that looks like "namespace::name", e.g. "aten::sin".
            Operators in PyTorch need a namespace to
            avoid name collisions; a given operator may only be created once.
            If you are writing a Python library, we recommend the namespace to
            be the name of your top-level module.
        schema (str): The schema of the operator. E.g. "(Tensor x) -> Tensor"
            for an op that accepts one Tensor and returns one Tensor. It does
            not contain the operator name (that is passed in ``qualname``).
        lib (Optional[Library]): If provided, the lifetime of this operator
            will be tied to the lifetime of the Library object.
        tags (Tag | Sequence[Tag]): one or more torch.Tag to apply to this
            operator. Tagging an operator changes the operator's behavior
            under various PyTorch subsystems; please read the docs for the
            torch.Tag carefully before applying it.

    Example::
        >>> import torch
        >>> import numpy as np
        >>>
        >>> # Define the operator
        >>> torch.library.define("mylib::sin", "(Tensor x) -> Tensor")
        >>>
        >>> # Add implementations for the operator
        >>> @torch.library.impl("mylib::sin", "cpu")
        >>> def f(x):
        >>>     return torch.from_numpy(np.sin(x.numpy()))
        >>>
        >>> # Call the new operator from torch.ops.
        >>> x = torch.randn(3)
        >>> y = torch.ops.mylib.sin(x)
        >>> assert torch.allclose(y, x.sin())

    """
    if not isinstance(qualname, str):
        raise ValueError(
            f"define(qualname, schema): expected qualname "
            f"to be instance of str, got {type(qualname)}"
        )
    namespace, name = torch._library.utils.parse_namespace(qualname)
    if lib is None:
        lib = Library(namespace, "FRAGMENT")
        _keep_alive.append(lib)
    if not NAMELESS_SCHEMA.fullmatch(schema):
        raise ValueError(
            f"define(qualname, schema, ...): expected schema "
            f'to look like e.g. "(Tensor x) -> Tensor" but '
            f'got "{schema}"'
        )
    lib.define(name + schema, alias_analysis="", tags=tags)


@define.register
def _(lib: Library, schema, alias_analysis=""):
    """The old torch.library.define.
    We're keeping this around for BC reasons
    """

    def wrap(f):
        name = lib.define(schema, alias_analysis)
        lib.impl(name, f)
        return f

    return wrap


@functools.singledispatch
def impl(qualname, types, func=None, *, lib=None):
    """Register an implementation for a device type for this operator.

    You may pass "default" for ``types`` to register this implementation as the
    default implementation for ALL device types.
    Please only use this if the implementation truly supports all device types;
    for example, this is true if it is a composition of built-in PyTorch operators.

    Some valid types are: "cpu", "cuda", "xla", "mps", "ipu", "xpu".

    Args:
        qualname (str): Should be a string that looks like "namespace::operator_name".
        types (str | Sequence[str]): The device types to register an impl to.
        lib (Optional[Library]): If provided, the lifetime of this registration
            will be tied to the lifetime of the Library object.

    Examples:
        >>> import torch
        >>> import numpy as np
        >>>
        >>> # Define the operator
        >>> torch.library.define("mylib::mysin", "(Tensor x) -> Tensor")
        >>>
        >>> # Add implementations for the cpu device
        >>> @torch.library.impl("mylib::mysin", "cpu")
        >>> def f(x):
        >>>     return torch.from_numpy(np.sin(x.numpy()))
        >>>
        >>> x = torch.randn(3)
        >>> y = torch.ops.mylib.mysin(x)
        >>> assert torch.allclose(y, x.sin())
    """
    if isinstance(types, str):
        types = (types,)
    keys = set({})
    for typ in types:
        is_dispatch_key = torch._C._parse_dispatch_key(typ)
        if is_dispatch_key:
            # We also support passing a DispatchKey to impl. Please prefer using
            # the higher-level torch.library APIs and only pass DispatchKey to
            # torch.library.impl with caution (or even better, don't use this
            # option and file an issue on GitHub for what you need).
            # We don't advertise this to users because
            # it is very easy to shoot yourself in the foot.
            keys.add(typ)
        else:
            keys.add(_device_type_to_key(typ))

    def register(func):
        namespace, _ = torch._library.utils.parse_namespace(qualname)
        if lib is None:
            use_lib = Library(namespace, "FRAGMENT")
            _keep_alive.append(use_lib)
        else:
            use_lib = lib
        for key in keys:
            use_lib.impl(qualname, func, key)

    if func is None:
        return register
    else:
        register(func)


def _device_type_to_key(device_type: str) -> str:
    if device_type == "default":
        # This is technically not correct, because although all device_type
        # DispatchKeys are included in CompositeExplicitAutograd,
        # not everything in CompositeExplicitAutograd is associated with a
        # device_type. I don't really care that much about the difference.
        return "CompositeExplicitAutograd"
    return torch._C._dispatch_key_for_device(device_type)


@impl.register
def _(lib: Library, name, dispatch_key=""):
    """Legacy torch.library.impl API. Kept around for BC"""

    def wrap(f):
        lib.impl(name, f, dispatch_key)
        return f

    return wrap


@deprecated(
    "`torch.library.impl_abstract` was renamed to `torch.library.register_fake`. Please use that "
    "instead; we will remove `torch.library.impl_abstract` in a future version of PyTorch.",
    category=FutureWarning,
)
def impl_abstract(qualname, func=None, *, lib=None, _stacklevel=1):
    r"""This API was renamed to :func:`torch.library.register_fake` in PyTorch 2.4.
    Please use that instead.
    """
    if func is not None:
        _stacklevel = _stacklevel + 1
    return register_fake(qualname, func, lib=lib, _stacklevel=_stacklevel)


_op_identifier = Union[
    str, "torch._ops.OpOverload", "torch._library.custom_ops.CustomOpDef"
]


def register_kernel(
    op: _op_identifier,
    device_types: device_types_t,
    func: Optional[Callable] = None,
    /,
    *,
    lib: Optional[Library] = None,
):
    """Register an implementation for a device type for this operator.

    Some valid device_types are: "cpu", "cuda", "xla", "mps", "ipu", "xpu".
    This API may be used as a decorator.

    Args:
        fn (Callable): The function to register as the implementation for
            the given device types.
        device_types (None | str | Sequence[str]): The device_types to register an impl to.
            If None, we will register to all device types -- please only use
            this option if your implementation is truly device-type-agnostic.

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
        >>> @torch.library.register_kernel("mylib::numpy_sin", "cuda")
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

    if not isinstance(
        op, (str, torch._ops.OpOverload, torch._library.custom_ops.CustomOpDef)
    ):
        raise ValueError("register_kernel(op): got unexpected type for op: {type(op)}")
    if isinstance(op, torch._ops.OpOverload):
        op = op._name
    opdef = _maybe_get_opdef(op)
    if opdef is not None:
        return opdef.register_kernel(device_types, func)
    assert isinstance(op, str)
    if device_types is None:
        device_types = "CompositeExplicitAutograd"
    return impl(op, device_types, func, lib=lib)


def register_fake(
    op: _op_identifier,
    func: Optional[Callable] = None,
    /,
    *,
    lib: Optional[Library] = None,
    _stacklevel: int = 1,
):
    r"""Register a FakeTensor implementation ("fake impl") for this operator.

    Also sometimes known as a "meta kernel", "abstract impl".

    An "FakeTensor implementation" specifies the behavior of this operator on
    Tensors that carry no data ("FakeTensor"). Given some input Tensors with
    certain properties (sizes/strides/storage_offset/device), it specifies
    what the properties of the output Tensors are.

    The FakeTensor implementation has the same signature as the operator.
    It is run for both FakeTensors and meta tensors. To write a FakeTensor
    implementation, assume that all Tensor inputs to the operator are
    regular CPU/CUDA/Meta tensors, but they do not have storage, and
    you are trying to return regular CPU/CUDA/Meta tensor(s) as output.
    The FakeTensor implementation must consist of only PyTorch operations
    (and may not directly access the storage or data of any input or
    intermediate Tensors).

    This API may be used as a decorator (see examples).

    For a detailed guide on custom ops, please see
    https://pytorch.org/tutorials/advanced/custom_ops_landing_page.html

    Examples:
        >>> import torch
        >>> import numpy as np
        >>> from torch import Tensor
        >>>
        >>> # Example 1: an operator without data-dependent output shape
        >>> @torch.library.custom_op("mylib::custom_linear", mutates_args=())
        >>> def custom_linear(x: Tensor, weight: Tensor, bias: Tensor) -> Tensor:
        >>>     raise NotImplementedError("Implementation goes here")
        >>>
        >>> @torch.library.register_fake("mylib::custom_linear")
        >>> def _(x, weight, bias):
        >>>     assert x.dim() == 2
        >>>     assert weight.dim() == 2
        >>>     assert bias.dim() == 1
        >>>     assert x.shape[1] == weight.shape[1]
        >>>     assert weight.shape[0] == bias.shape[0]
        >>>     assert x.device == weight.device
        >>>
        >>>     return (x @ weight.t()) + bias
        >>>
        >>> with torch._subclasses.fake_tensor.FakeTensorMode():
        >>>     x = torch.randn(2, 3)
        >>>     w = torch.randn(3, 3)
        >>>     b = torch.randn(3)
        >>>     y = torch.ops.mylib.custom_linear(x, w, b)
        >>>
        >>> assert y.shape == (2, 3)
        >>>
        >>> # Example 2: an operator with data-dependent output shape
        >>> @torch.library.custom_op("mylib::custom_nonzero", mutates_args=())
        >>> def custom_nonzero(x: Tensor) -> Tensor:
        >>>     x_np = x.numpy(force=True)
        >>>     res = np.stack(np.nonzero(x_np), axis=1)
        >>>     return torch.tensor(res, device=x.device)
        >>>
        >>> @torch.library.register_fake("mylib::custom_nonzero")
        >>> def _(x):
        >>>     # Number of nonzero-elements is data-dependent.
        >>>     # Since we cannot peek at the data in an fake impl,
        >>>     # we use the ctx object to construct a new symint that
        >>>     # represents the data-dependent size.
        >>>     ctx = torch.library.get_ctx()
        >>>     nnz = ctx.new_dynamic_size()
        >>>     shape = [nnz, x.dim()]
        >>>     result = x.new_empty(shape, dtype=torch.int64)
        >>>     return result
        >>>
        >>> from torch.fx.experimental.proxy_tensor import make_fx
        >>>
        >>> x = torch.tensor([0, 1, 2, 3, 4, 0])
        >>> trace = make_fx(torch.ops.mylib.custom_nonzero, tracing_mode="symbolic")(x)
        >>> trace.print_readable()
        >>>
        >>> assert torch.allclose(trace(x), torch.ops.mylib.custom_nonzero(x))

    """
    if not isinstance(
        op, (str, torch._ops.OpOverload, torch._library.custom_ops.CustomOpDef)
    ):
        raise ValueError("register_fake(op): got unexpected type for op: {type(op)}")
    if isinstance(op, torch._ops.OpOverload):
        op = op._name
    opdef = _maybe_get_opdef(op)
    if opdef is not None:
        if func is None:
            return opdef.register_fake
        else:
            return opdef.register_fake(func)
    assert isinstance(op, str)

    stacklevel = _stacklevel

    def register(func):
        namespace, op_name = torch._library.utils.parse_namespace(op)
        if lib is None:
            use_lib = Library(namespace, "FRAGMENT")
            _keep_alive.append(use_lib)
        else:
            use_lib = lib
        use_lib._register_fake(op_name, func, _stacklevel=stacklevel + 1)
        return func

    if func is None:
        return register
    else:
        stacklevel += 1
        return register(func)


def register_autograd(
    op: _op_identifier,
    backward: Callable,
    /,
    *,
    setup_context: Optional[Callable] = None,
    lib=None,
) -> None:
    r"""Register a backward formula for this custom op.

    In order for an operator to work with autograd, you need to register
    a backward formula:
    1. You must tell us how to compute gradients during the backward pass
    by providing us a "backward" function.
    2. If you need any values from the forward to compute gradients, you can
    use `setup_context` to save values for backward.

    ``backward`` runs during the backward pass. It accepts ``(ctx, *grads)``:
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
        >>> torch.library.register_autograd("mylib::numpy_sin", backward, setup_context=setup_context)
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
        >>> torch.library.register_autograd("mylib::numpy_mul", backward, setup_context=setup_context)
        >>>
        >>> x = torch.randn(3, requires_grad=True)
        >>> y = numpy_mul(x, val=3.14)
        >>> grad_x, = torch.autograd.grad(y, x, torch.ones_like(y))
        >>> assert torch.allclose(grad_x, torch.full_like(x, 3.14))

    """
    if not isinstance(
        op, (str, torch._ops.OpOverload, torch._library.custom_ops.CustomOpDef)
    ):
        raise ValueError(
            f"register_autograd(op): got unexpected type for op: {type(op)}"
        )
    if isinstance(op, torch._ops.OpOverload):
        op = op._name
    opdef = _maybe_get_opdef(op)
    if opdef is not None:
        opdef.register_autograd(backward, setup_context=setup_context)
        return

    assert isinstance(op, str)
    qualname = op
    op = torch._library.utils.lookup_op(qualname)
    schema = op._schema
    if not _library.utils.is_functional_schema(schema):
        raise RuntimeError(
            f"Cannot register autograd formula for non-functional operator "
            f"{op} with schema {schema}. Please create "
            f"a functional operator and register an autograd formula for that."
        )
    if _library.utils.has_kwarg_only_tensors(schema):
        raise NotImplementedError(
            f"register_autograd with kwarg-only Tensor args. In the original "
            f"definition of the op, please make your tensors not kwarg-only. "
            f"Got: {schema}"
        )

    info = _library.autograd.Info(backward, setup_context)
    autograd_kernel = _library.autograd.make_autograd_impl(op, info)
    namespace, opname = torch._library.utils.parse_namespace(qualname)
    if lib is None:
        lib = Library(namespace, "FRAGMENT")
        _keep_alive.append(lib)
    lib.impl(opname, autograd_kernel, "Autograd", with_keyset=True)


# If the op was defined in C++, then we want to make sure there was an
# m.set_python_module(module, ...) call and that the module is the
# same as the module that called torch.library.register_fake.
def _check_pystubs_once(func, qualname, actual_module_name):
    checked = False

    def inner(*args, **kwargs):
        nonlocal checked
        if checked:
            return func(*args, **kwargs)

        op = torch._library.utils.lookup_op(qualname)
        if op._defined_in_python:
            checked = True
            return func(*args, **kwargs)

        maybe_pystub = torch._C._dispatch_pystub(
            op._schema.name, op._schema.overload_name
        )
        if maybe_pystub is None:
            if torch._library.utils.requires_set_python_module():
                namespace = op.namespace
                cpp_filename = op._handle.debug()
                raise RuntimeError(
                    f"Operator '{qualname}' was defined in C++ and has a Python "
                    f"fake impl. In this situation, we require there to also be a "
                    f'companion C++ `m.set_python_module("{actual_module_name}")` '
                    f"call, but we could not find one. Please add that to "
                    f"to the top of the C++ TORCH_LIBRARY({namespace}, ...) block the "
                    f"operator was registered in ({cpp_filename})"
                )
        else:
            pystub_module = maybe_pystub[0]
            if actual_module_name != pystub_module:
                cpp_filename = op._handle.debug()
                raise RuntimeError(
                    f"Operator '{qualname}' specified that its python fake impl "
                    f"is in the Python module '{pystub_module}' but it was actually found "
                    f"in '{actual_module_name}'. Please either move the fake impl "
                    f"or correct the m.set_python_module call ({cpp_filename})"
                )
        checked = True
        return func(*args, **kwargs)

    return inner


# NOTE [ctx inside the fake implementation]
# If a user has an operator with data-dependent output shape, then when writing
# a fake implementation they must query the current ctx and use methods on the
# ctx to construct a new unbacked symint.
#
# This is done via us setting the global_ctx_getter function every time a fake
# implementation is invoked.
def get_ctx() -> "torch._library.fake_impl.FakeImplCtx":
    """get_ctx() returns the current AbstractImplCtx object.

    Calling ``get_ctx()`` is only valid inside of an fake impl
    (see :func:`torch.library.register_fake` for more usage details.
    """
    return torch._library.fake_impl.global_ctx_getter()


_OPCHECK_DEFAULT_UTILS = (
    "test_schema",
    "test_autograd_registration",
    "test_faketensor",
    "test_aot_dispatch_dynamic",
)


def opcheck(
    op: Union[torch._ops.OpOverload, torch._ops.OpOverloadPacket, CustomOpDef],
    args: Tuple[Any, ...],
    kwargs: Optional[Dict[str, Any]] = None,
    *,
    test_utils: Union[str, Sequence[str]] = _OPCHECK_DEFAULT_UTILS,
    raise_exception: bool = True,
) -> Dict[str, str]:
    """Given an operator and some sample arguments, tests if the operator is
    registered correctly.

    That is, when you use the torch.library/TORCH_LIBRARY APIs to create a
    custom op, you specified metadata (e.g. mutability info) about the custom op
    and these APIs require that the functions you pass them satisfy certain
    properties (e.g. no data pointer access in the fake/meta/abstract kernel)
    ``opcheck`` tests these metadata and properties.

    Concretely, we test the following:
    - test_schema: if the operator's schema is correct.
    - test_autograd_registration: if autograd was registered correctly.
    - test_faketensor: If the operator has a FakeTensor kernel
    (and if it is correct). The FakeTensor kernel is necessary (
    but not sufficient) for the operator to work with PyTorch compilation
    APIs (torch.compile/export/FX).
    - test_aot_dispatch_dynamic: If the operator has correct behavior
    with PyTorch compilation APIs (torch.compile/export/FX).
    This checks that the outputs (and gradients, if applicable) are the
    same under eager-mode PyTorch and torch.compile.
    This test is a superset of ``test_faketensor``.

    For best results, please call ``opcheck`` multiple times with a
    representative set of inputs. If your operator supports
    autograd, please use ``opcheck`` with inputs with ``requires_grad = True``;
    if your operator supports multiple devices (e.g. CPU and CUDA), please
    use ``opcheck`` with inputs on all supported devices.

    Args:
        op: The operator. Must either be a function decorated with
            :func:`torch.library.custom_op` or an OpOverload/OpOverloadPacket
            found in torch.ops.* (e.g. torch.ops.aten.sin, torch.ops.mylib.foo)
        args: The args to the operator
        kwargs: The kwargs to the operator
        test_utils: Tests that we should run. Default: all of them.
            Example: ("test_schema", "test_faketensor")
        raise_exception: If we should raise an exception on the first
            error. If False, we will return a dict with information
            on if each test passed or not.

    .. warning::

        opcheck and :func:`torch.autograd.gradcheck` test different things;
        opcheck tests if your usage of torch.library APIs is correct while
        :func:`torch.autograd.gradcheck` tests if your autograd formula is
        mathematically correct. Use both to test custom ops that support
        gradient computation.

    Example:

        >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_CUDA)
        >>> @torch.library.custom_op("mylib::numpy_mul", mutates_args=())
        >>> def numpy_add(x: Tensor, y: float) -> Tensor:
        >>>     x_np = x.numpy(force=True)
        >>>     z_np = x_np + y
        >>>     return torch.from_numpy(z_np).to(x.device)
        >>>
        >>> @numpy_sin.register_fake
        >>> def _(x, y):
        >>>     return torch.empty_like(x)
        >>>
        >>> def setup_context(ctx, inputs, output):
        >>>     y, = inputs
        >>>     ctx.y = y
        >>>
        >>> def backward(ctx, grad):
        >>>     return grad * ctx.y, None
        >>>
        >>> numpy_sin.register_autograd(backward, setup_context=setup_context)
        >>>
        >>> sample_inputs = [
        >>>     (torch.randn(3), 3.14),
        >>>     (torch.randn(2, 3, device='cuda'), 2.718),
        >>>     (torch.randn(1, 10, requires_grad=True), 1.234),
        >>>     (torch.randn(64, 64, device='cuda', requires_grad=True), 90.18),
        >>> ]
        >>>
        >>> for args in sample_inputs:
        >>>     torch.library.opcheck(foo, args)

    """
    import torch.testing._internal.optests as optests

    return optests.opcheck(
        op, args, kwargs, test_utils=test_utils, raise_exception=raise_exception
    )
