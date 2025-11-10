# mypy: allow-untyped-defs
import contextlib
import functools
import inspect
import re
import sys
import traceback
import weakref
from collections.abc import Callable, Sequence
from typing import Any, Optional, overload, TYPE_CHECKING, TypeVar, Union
from typing_extensions import deprecated, ParamSpec

import torch
import torch._library as _library
from torch._library.custom_ops import (
    _cast,
    _maybe_get_opdef,
    custom_op,
    CustomOpDef,
    device_types_t,
)
from torch._library.infer_schema import infer_schema  # noqa: F401
from torch._library.triton import triton_op, wrap_triton
from torch._ops import OpOverload
from torch.types import _dtype


__all__ = [
    "Library",
    "impl",
    "define",
    "fallthrough_kernel",
    "impl_abstract",
    "register_autocast",
    "register_fake",
    "register_torch_dispatch",
    "register_vmap",
    "get_ctx",
    "get_kernel",
    "custom_op",
    "triton_op",
    "wrap_triton",
    "infer_schema",
]

_T = TypeVar("_T")
_P = ParamSpec("_P")

# Set containing the combination of (namespace, operator, DispatchKey) for which a new kernel has been registered
# The keys in the set are of the form `namespace + "/" + op_name + "/" + dispatch_key`.
# This set is maintained to ensure that two libraries don't try to override the exact same functionality to avoid
# libraries calling into kernels not intended to be called.
_impls: set[str] = set()
_defs: set[str] = set()

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
        kind: "DEF", "IMPL", "FRAGMENT"
        dispatch_key: PyTorch dispatch key (default: "")
    """

    def __init__(self, ns, kind, dispatch_key=""):
        from torch.fx.operator_schemas import _SCHEMA_TO_SIGNATURE_CACHE

        if kind not in ("IMPL", "DEF", "FRAGMENT"):
            raise ValueError("Unsupported kind: ", kind)

        if ns in _reserved_namespaces and (kind == "DEF" or kind == "FRAGMENT"):
            raise ValueError(
                ns,
                " is a reserved namespace. Please try creating a library with another name.",
            )

        frame = traceback.extract_stack(limit=2)[0]
        filename, lineno = frame.filename, frame.lineno
        self.m: Optional[Any] = torch._C._dispatch_library(
            kind, ns, dispatch_key, filename, lineno
        )
        self.ns = ns
        self._op_defs: set[str] = set()
        self._op_impls: set[str] = set()
        self._registration_handles: list[torch._library.utils.RegistrationHandle] = []
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
            self.m,
            _SCHEMA_TO_SIGNATURE_CACHE,
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

    def _register_fake(self, op_name, fn, _stacklevel=1, *, allow_override=False):
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

        handle = entry.fake_impl.register(
            func_to_register, source, lib=self, allow_override=allow_override
        )
        self._registration_handles.append(handle)

    def _register_torch_dispatch_rule(self, op_name, torch_dispatch_class, fn):
        r"""Registers a torch_dispatch rule for the given operator and torch_dispatch_class.

        This allows for open registration to specify the behavior between the operator
        and the torch_dispatch_class without needing to modify the torch_dispatch_class
        or the operator directly.

        The torch_dispatch_class is either a Tensor subclass with `__torch_dispatch__` or a
        TorchDispatchMode.

        If it is a Tensor subclass, we expect fn to have the following signature:
        (cls, func: OpOverload, types: Tuple[type, ...], args, kwargs) -> Any

        If it is a TorchDispatchMode, we expect fn to have the following signature:
        (mode, func: OpOverload, types: Tuple[type, ...], args, kwargs) -> Any
        """

        qualname = f"{self.ns}::{op_name}"
        entry = torch._library.simple_registry.singleton.find(qualname)
        handle = entry.torch_dispatch_rules.register(torch_dispatch_class, fn)
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
        # pyrefly: ignore [bad-argument-type]
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

    def impl(
        self, op_name, fn, dispatch_key="", *, with_keyset=False, allow_override=False
    ):
        r"""Registers the function implementation for an operator defined in the library.

        Args:
            op_name: operator name (along with the overload) or OpOverload object.
            fn: function that's the operator implementation for the input dispatch key or :func:`~fallthrough_kernel`
                to register a fallthrough.
            dispatch_key: dispatch key that the input function should be registered for. By default, it uses
                          the dispatch key that the library was created with.
            with_keyset: flag controlling if the current dispatcher call keyset should be passed as the first argument
                         to :attr:`fn` when calling. This should be used to create the appropriate keyset for redispatch calls.
            allow_override: Flag controlling if we want to override an
                         existing registered kernel implementation. This is by
                         default off, and will error you're trying to register a
                         kernel to a dispatch key with a kernel already
                         registered.

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
        if (not allow_override) and key in _impls:
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

    def fallback(self, fn, dispatch_key="", *, with_keyset=False):
        r"""Registers the function implementation as the fallback for the given key.

        This function only works for a library with global namespace ("_").

        Args:
            fn: function used as fallback for the given dispatch key or :func:`~fallthrough_kernel`
                to register a fallthrough.
            dispatch_key: dispatch key that the input function should be registered for. By default, it uses
                          the dispatch key that the library was created with.
            with_keyset: flag controlling if the current dispatcher call keyset should be passed as the first argument
                         to :attr:`fn` when calling. This should be used to create the appropriate keyset for redispatch calls.

        Example::

            >>> my_lib = Library("_", "IMPL")
            >>> def fallback_kernel(op, *args, **kwargs):
            >>>     # Handle all autocast ops generically
            >>>     # ...
            >>> my_lib.fallback(fallback_kernel, "Autocast")
        """

        if dispatch_key == "":
            dispatch_key = self.dispatch_key

        if self.ns != "_":
            raise RuntimeError(
                f"""Fallback can only be registered using library fragment on the global namespace "_" but it is {self.ns}"""
            )

        assert dispatch_key != ""
        assert self.m is not None

        self.m.fallback(dispatch_key, fn, with_keyset)

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
            namespace._dir.remove(name)


def _del_library(
    captured_impls,
    op_impls,
    captured_defs,
    op_defs,
    registration_handles,
    m,
    schema_to_signature_cache,
):
    for op_def in op_defs:
        name = op_def
        overload_name = ""
        if "." in op_def:
            name, overload_name = op_def.split(".")
        if (
            name,
            overload_name,
        ) in schema_to_signature_cache:
            del schema_to_signature_cache[(name, overload_name)]

    captured_impls -= op_impls
    captured_defs -= op_defs
    for handle in registration_handles:
        handle.destroy()

    if m is not None:
        m.reset()


@contextlib.contextmanager
def _scoped_library(*args, **kwargs):
    try:
        lib = Library(*args, **kwargs)
        yield lib
    finally:
        lib._destroy()


_keep_alive: list[Library] = []


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


@overload
def impl(
    qualname: str,
    types: Union[str, Sequence[str]],
    func: None = None,
    *,
    lib: Optional[Library] = None,
) -> Callable[[Callable[..., object]], None]: ...


@overload
def impl(
    qualname: str,
    types: Union[str, Sequence[str]],
    func: Callable[..., object],
    *,
    lib: Optional[Library] = None,
) -> None: ...


# Deprecated BC API
@overload
def impl(
    lib: Library,
    name: str,
    dispatch_key: str = "",
) -> Callable[[Callable[_P, _T]], Callable[_P, _T]]: ...


@functools.singledispatch
def impl(
    qualname: str,
    types: Union[str, Sequence[str]],
    func: Optional[Callable[_P, _T]] = None,
    *,
    lib: Optional[Library] = None,
) -> object:
    """Register an implementation for a device type for this operator.

    You may pass "default" for ``types`` to register this implementation as the
    default implementation for ALL device types.
    Please only use this if the implementation truly supports all device types;
    for example, this is true if it is a composition of built-in PyTorch operators.

    This API may be used as a decorator. You can use nested decorators
    with this API provided they return a function and are placed inside
    this API (see Example 2).

    Some valid types are: "cpu", "cuda", "xla", "mps", "ipu", "xpu".

    Args:
        qualname (str): Should be a string that looks like "namespace::operator_name".
        types (str | Sequence[str]): The device types to register an impl to.
        lib (Optional[Library]): If provided, the lifetime of this registration
            will be tied to the lifetime of the Library object.

    Examples:
        >>> import torch
        >>> import numpy as np
        >>> # Example 1: Register function.
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
        >>>
        >>> # Example 2: Register function with decorator.
        >>> def custom_decorator(func):
        >>>     def wrapper(*args, **kwargs):
        >>>         return func(*args, **kwargs) + 1
        >>>     return wrapper
        >>>
        >>> # Define the operator
        >>> torch.library.define("mylib::sin_plus_one", "(Tensor x) -> Tensor")
        >>>
        >>> # Add implementations for the operator
        >>> @torch.library.impl("mylib::sin_plus_one", "cpu")
        >>> @custom_decorator
        >>> def f(x):
        >>>     return torch.from_numpy(np.sin(x.numpy()))
        >>>
        >>> # Call the new operator from torch.ops.
        >>> x = torch.randn(3)
        >>>
        >>> y1 = torch.ops.mylib.sin_plus_one(x)
        >>> y2 = torch.sin(x) + 1
        >>> assert torch.allclose(y1, y2)
    """

    return _impl(qualname, types, func, lib=lib, disable_dynamo=False)


if not TYPE_CHECKING:

    @impl.register
    def _(
        lib: Library, name: str, dispatch_key: str = ""
    ) -> Callable[[Callable[_P, _T]], Callable[_P, _T]]:
        """Legacy torch.library.impl API. Kept around for BC"""

        def wrap(f: Callable[_P, _T]) -> Callable[_P, _T]:
            lib.impl(name, f, dispatch_key)
            return f

        return wrap


@overload
def _impl(
    qualname: str,
    types: Union[str, Sequence[str]],
    func: None = None,
    *,
    lib: Optional[Library] = None,
    disable_dynamo: bool = False,
) -> Callable[[Callable[..., object]], None]: ...


@overload
def _impl(
    qualname: str,
    types: Union[str, Sequence[str]],
    func: Callable[..., object],
    *,
    lib: Optional[Library] = None,
    disable_dynamo: bool = False,
) -> None: ...


def _impl(
    qualname: str,
    types: Union[str, Sequence[str]],
    func: Optional[Callable[..., object]] = None,
    *,
    lib: Optional[Library] = None,
    disable_dynamo: bool = False,
) -> Optional[Callable[[Callable[..., object]], None]]:
    # See impl()
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

    def register_(func: Callable[..., object]) -> None:
        namespace, _ = torch._library.utils.parse_namespace(qualname)

        if lib is None:
            use_lib = Library(namespace, "FRAGMENT")
            _keep_alive.append(use_lib)
        else:
            use_lib = lib
        if disable_dynamo:

            @torch._disable_dynamo
            def func_no_dynamo(*args, **kwargs):
                return func(*args, **kwargs)

            for key in keys:
                use_lib.impl(qualname, func_no_dynamo, key)
        else:
            for key in keys:
                use_lib.impl(qualname, func, key)

    if func is None:
        return register_
    else:
        register_(func)
        return None


def _device_type_to_key(device_type: str) -> str:
    if device_type == "default":
        # This is technically not correct, because although all device_type
        # DispatchKeys are included in CompositeExplicitAutograd,
        # not everything in CompositeExplicitAutograd is associated with a
        # device_type. I don't really care that much about the difference.
        return "CompositeExplicitAutograd"
    return torch._C._dispatch_key_for_device(device_type)


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
        op (str | OpOverload): The operator to register an impl to.
        device_types (None | str | Sequence[str]): The device_types to register an impl to.
            If None, we will register to all device types -- please only use
            this option if your implementation is truly device-type-agnostic.
        func (Callable): The function to register as the implementation for
            the given device types.
        lib (Optional[Library]): If provided, the lifetime of this registration

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
        raise ValueError(
            f"register_kernel({op}): got unexpected type for op: {type(op)}"
        )
    if isinstance(op, torch._ops.OpOverload):
        op = op._name
    opdef = _maybe_get_opdef(op)
    if opdef is not None:
        return opdef.register_kernel(device_types, func)
    assert isinstance(op, str)
    if device_types is None:
        device_types = "CompositeExplicitAutograd"

    return _impl(op, device_types, func, lib=lib, disable_dynamo=True)


def register_autocast(
    op: _op_identifier,
    device_type: str,
    cast_inputs: _dtype,
    /,
    *,
    lib: Optional[Library] = None,
):
    r"""Register an autocast dispatch rule for this custom op.

    Valid `device_type` include: "cpu" and "cuda".

    Args:
        op (str | OpOverload): The operator to register an autocast dispatch rule to.
        device_type(str):  Device type to use. 'cuda' or 'cpu'.
            The type is the same as the `type` attribute of a :class:`torch.device`.
            Thus, you may obtain the device type of a tensor using `Tensor.device.type`.
        cast_inputs (:class:`torch.dtype`): When custom op runs in an autocast-enabled region,
            casts incoming floating-point Tensors to the target dtype (non-floating-point Tensors
            are not affected), then executes custom op with autocast disabled.
        lib (Optional[Library]): If provided, the lifetime of this registration

    Examples::
        >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_CUDA)
        >>> import torch
        >>> from torch import Tensor
        >>> from torch.library import custom_op
        >>>
        >>> # Create a custom op that works on cuda
        >>> @torch.library.custom_op("mylib::my_sin", mutates_args=())
        >>> def my_sin(x: Tensor) -> Tensor:
        >>>     return torch.sin(x)
        >>>
        >>> # Register autocast dispatch rule for the cuda device
        >>> torch.library.register_autocast("mylib::my_sin", "cuda", torch.float16)
        >>>
        >>> x = torch.randn(3, dtype=torch.float32, device="cuda")
        >>> with torch.autocast("cuda", dtype=torch.float16):
        >>>     y = torch.ops.mylib.my_sin(x)
        >>> assert y.dtype == torch.float16

    """
    if not isinstance(
        op, (str, torch._ops.OpOverload, torch._library.custom_ops.CustomOpDef)
    ):
        raise ValueError(
            f"register_autocast({op}): got unexpected type for op: {type(op)}"
        )
    if device_type not in ["cpu", "cuda"]:
        raise ValueError(f"Unknown device type: {device_type}")

    if isinstance(op, torch._ops.OpOverload):
        op = op._name
    opdef = _maybe_get_opdef(op)
    if opdef is not None:
        return opdef.register_autocast(device_type, cast_inputs)

    assert isinstance(op, str)
    qualname = op
    _op = torch._library.utils.lookup_op(qualname)

    namespace, opname = torch._library.utils.parse_namespace(qualname)
    if lib is None:
        lib = Library(namespace, "FRAGMENT")
        _keep_alive.append(lib)

    def _maybe_override_py_impl(op: torch._ops.OpOverload, dispatch_key):
        def inner(kernel):
            if op.has_kernel_for_dispatch_key(dispatch_key):
                op.py_kernels.pop(dispatch_key)
            return op.py_impl(dispatch_key)(kernel)

        return inner

    @_maybe_override_py_impl(_op, torch._C.DispatchKey.AutocastCPU)
    @_maybe_override_py_impl(_op, torch._C.DispatchKey.AutocastCUDA)
    def _autocast_py_impl(*args, **kwargs):
        assert len(kwargs) == 0, "Custom ops do not support kwargs yet."
        autocast_keyset = torch._C.DispatchKeySet(
            torch._C.DispatchKey.AutocastCPU
        ) | torch._C.DispatchKeySet(torch._C.DispatchKey.AutocastCUDA)
        with torch._C._ExcludeDispatchKeyGuard(autocast_keyset):
            return _op(*_cast(args, device_type, cast_inputs))

    def kernel(_, *args, **kwargs):
        assert len(kwargs) == 0, "Custom ops do not support kwargs yet."
        return _autocast_py_impl(*args, **kwargs)

    if device_type == "cuda":
        return lib.impl(opname, kernel, "AutocastCUDA", with_keyset=True)
    else:
        # device_type is "cpu"
        return lib.impl(opname, kernel, "AutocastCPU", with_keyset=True)


def register_fake(
    op: _op_identifier,
    func: Optional[Callable] = None,
    /,
    *,
    lib: Optional[Library] = None,
    _stacklevel: int = 1,
    allow_override: bool = False,
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

    Args:
        op_name: Operator name (along with the overload) or OpOverload object.
        func: Fake tensor implementation.
        lib (Optional[Library]): Library to register the fake tensor to.
        allow_override: Flag controlling if we want to override an
                        existing registered fake impl. This is by default off,
                        and will error you're trying to register a fake impl to
                        an operator that already has a fake impl. This also only
                        applies if the custom operator was not created via
                        torch.library.custom_op, as overriding and existing fake
                        impl is already allowed.

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
        >>> # Number of nonzero-elements is data-dependent.
        >>> # Since we cannot peek at the data in an fake impl,
        >>> # we use the ctx object to construct a new symint that
        >>> # represents the data-dependent size.
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
        raise ValueError(f"register_fake({op}): got unexpected type for op: {type(op)}")
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
        use_lib._register_fake(
            op_name, func, _stacklevel=stacklevel + 1, allow_override=allow_override
        )
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

    If you need different autograd behavior on different devices, then we
    recommend creating two different custom operators, one for each device
    that needs different behavior, and switching between them at runtime.

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
        >>> torch.library.register_autograd(
        ...     "mylib::numpy_sin", backward, setup_context=setup_context
        ... )
        >>>
        >>> x = torch.randn(3, requires_grad=True)
        >>> y = numpy_sin(x)
        >>> (grad_x,) = torch.autograd.grad(y, x, torch.ones_like(y))
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
        >>> torch.library.register_autograd(
        ...     "mylib::numpy_mul", backward, setup_context=setup_context
        ... )
        >>>
        >>> x = torch.randn(3, requires_grad=True)
        >>> y = numpy_mul(x, val=3.14)
        >>> (grad_x,) = torch.autograd.grad(y, x, torch.ones_like(y))
        >>> assert torch.allclose(grad_x, torch.full_like(x, 3.14))

    """
    if not isinstance(
        op, (str, torch._ops.OpOverload, torch._library.custom_ops.CustomOpDef)
    ):
        raise ValueError(
            f"register_autograd({op}): got unexpected type for op: {type(op)}"
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


def register_torch_dispatch(
    op: _op_identifier,
    torch_dispatch_class: Any,
    func: Optional[Callable] = None,
    /,
    *,
    lib: Optional[Library] = None,
):
    r"""Registers a torch_dispatch rule for the given operator and ``torch_dispatch_class``.

    This allows for open registration to specify the behavior between the operator
    and the ``torch_dispatch_class`` without needing to modify the ``torch_dispatch_class``
    or the operator directly.

    The ``torch_dispatch_class`` is either a Tensor subclass with ``__torch_dispatch__`` or a
    TorchDispatchMode.

    If it is a Tensor subclass, we expect ``func`` to have the following signature:
    ``(cls, func: OpOverload, types: Tuple[type, ...], args, kwargs) -> Any``

    If it is a TorchDispatchMode, we expect ``func`` to have the following signature:
    ``(mode, func: OpOverload, types: Tuple[type, ...], args, kwargs) -> Any``

    ``args`` and ``kwargs`` will have been normalized the same way they are
    in ``__torch_dispatch__`` (see :ref:`torch-dispatch-calling-convention`).

    Examples:

        >>> import torch
        >>>
        >>> @torch.library.custom_op("mylib::foo", mutates_args={})
        >>> def foo(x: torch.Tensor) -> torch.Tensor:
        >>>     return x.clone()
        >>>
        >>> class MyMode(torch.utils._python_dispatch.TorchDispatchMode):
        >>>     def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        >>>         return func(*args, **kwargs)
        >>>
        >>> @torch.library.register_torch_dispatch("mylib::foo", MyMode)
        >>> def _(mode, func, types, args, kwargs):
        >>>     x, = args
        >>>     return x + 1
        >>>
        >>> x = torch.randn(3)
        >>> y = foo(x)
        >>> assert torch.allclose(y, x)
        >>>
        >>> with MyMode():
        >>>     y = foo(x)
        >>> assert torch.allclose(y, x + 1)

    """
    if not isinstance(
        op, (str, torch._ops.OpOverload, torch._library.custom_ops.CustomOpDef)
    ):
        raise ValueError(
            f"register_torch_dispatch({op}): got unexpected type for op: {type(op)}"
        )
    if isinstance(op, torch._ops.OpOverload):
        op = op._name
    opdef = _maybe_get_opdef(op)
    if opdef is not None:
        return opdef.register_torch_dispatch(torch_dispatch_class, func)
    assert isinstance(op, str)

    def register(func):
        namespace, op_name = torch._library.utils.parse_namespace(op)
        if lib is None:
            use_lib = Library(namespace, "FRAGMENT")
            _keep_alive.append(use_lib)
        else:
            use_lib = lib
        use_lib._register_torch_dispatch_rule(op_name, torch_dispatch_class, func)
        return func

    if func is None:
        return register
    else:
        return register(func)


def register_vmap(
    op: _op_identifier,
    func: Optional[Callable] = None,
    /,
    *,
    lib=None,
):
    r"""Register a vmap implementation to support :func:`torch.vmap` for this custom op.

    This API may be used as a decorator (see examples).

    In order for an operator to work with :func:`torch.vmap`, you may need to register a
    vmap implementation in the following signature:

        ``vmap_func(info, in_dims: Tuple[Optional[int]], *args, **kwargs)``,

    where ``*args`` and ``**kwargs`` are the arguments and kwargs for ``op``.
    We do not support kwarg-only Tensor args.

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
        >>> torch.library.register_vmap(numpy_cube, numpy_cube_vmap)
        >>>
        >>> x = torch.randn(3)
        >>> torch.vmap(numpy_cube)(x)
        >>>
        >>> @torch.library.custom_op("mylib::numpy_mul", mutates_args=())
        >>> def numpy_mul(x: Tensor, y: Tensor) -> Tensor:
        >>>     return torch.tensor(to_numpy(x) * to_numpy(y), device=x.device)
        >>>
        >>> @torch.library.register_vmap("mylib::numpy_mul")
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

    .. note::
        The vmap function should aim to preserve the semantics of the entire custom operator.
        That is, ``grad(vmap(op))`` should be replaceable with a ``grad(map(op))``.

        If your custom operator has any custom behavior in the backward pass, please
        keep this in mind.

    """
    if not isinstance(
        op, (str, torch._ops.OpOverload, torch._library.custom_ops.CustomOpDef)
    ):
        raise ValueError(f"register_vmap({op}): got unexpected type for op: {type(op)}")
    if isinstance(op, torch._ops.OpOverload):
        op = op._name
    opdef = _maybe_get_opdef(op)
    if opdef is not None:
        return opdef.register_vmap(func)
    assert isinstance(op, str)
    qualname = op
    op = torch._library.utils.lookup_op(qualname)
    schema = op._schema
    if _library.utils.has_kwarg_only_tensors(schema):
        raise NotImplementedError(
            f"register_vmap with kwarg-only Tensor args. In the original "
            f"definition of the op, please make your tensors not kwarg-only. "
            f"Got: {schema}"
        )

    def register(func):
        nonlocal op, lib

        namespace, opname = torch._library.utils.parse_namespace(qualname)
        if lib is None:
            lib = Library(namespace, "FRAGMENT")
            _keep_alive.append(lib)

        from torch._functorch.autograd_function import custom_function_call_vmap_helper
        from torch._functorch.pyfunctorch import retrieve_current_functorch_interpreter

        def wrapped_func(keyset, *args, **kwargs):
            interpreter = retrieve_current_functorch_interpreter()
            return custom_function_call_vmap_helper(
                interpreter, func, op, *args, **kwargs
            )

        lib.impl(opname, wrapped_func, "FuncTorchBatched", with_keyset=True)

    if func is None:
        return register
    else:
        return register(func)


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


def get_kernel(
    op: _op_identifier, dispatch_key: Union[str, torch.DispatchKey]
) -> torch._C._SafeKernelFunction:
    """Returns the computed kernel for a given operator and dispatch key.

    This function retrieves the kernel that would be executed for a given
    operator and dispatch key combination. The returned SafeKernelFunction
    can be used to call the kernel in a boxed fashion. The intended use
    case for this function is to retrieve the original kernel for a given
    dispatch key and then register another kernel to the same dispatch key
    that calls into the original kernel for certain cases.

    Args:
        op: Operator name (along with the overload) or OpOverload object
            Can be a string (e.g., "aten::add.Tensor"), an OpOverload, or a CustomOpDef.
        dispatch_key (str | torch.DispatchKey): The dispatch key to get the kernel for.
            Can be a string (e.g., "CPU", "CUDA") or a DispatchKey enum value.

    Returns:
        torch._C._SafeKernelFunction: A safe kernel function that can be used to
            call the kernel.

    Raises:
        RuntimeError: If the operator does not exist.

    Example:
        >>> # Get the CPU kernel for torch.add
        >>> kernel = torch.library.get_kernel("aten::add.Tensor", "CPU")
        >>>
        >>> # You can also use DispatchKey enum
        >>> kernel = torch.library.get_kernel("aten::add.Tensor", torch.DispatchKey.CPU)
        >>>
        >>> # Or use an OpOverload directly
        >>> kernel = torch.library.get_kernel(torch.ops.aten.add.Tensor, "CPU")
        >>>
        >>> # Example: Using get_kernel in a custom op with conditional dispatch
        >>> # Get the original kernel for torch.sin
        >>> original_sin_kernel = torch.library.get_kernel("aten::sin", "CPU")
        >>>
        >>> # If input has negative values, use original sin, otherwise return zeros
        >>> def conditional_sin_impl(dispatch_keys, x):
        >>>     if (x < 0).any():
        >>>         return original_sin_kernel.call_boxed(dispatch_keys, x)
        >>>     else:
        >>>         return torch.zeros_like(x)
        >>>
        >>> lib = torch.library.Library("aten", "IMPL")
        >>> # with_keyset=True so the first argument to the impl is the current DispatchKeySet
        >>> which needs to be the first argument to ``kernel.call_boxed``
        >>> lib.impl("sin", conditional_sin_impl, "CPU", with_keyset=True)
        >>>
        >>> # Test the conditional behavior
        >>> x_positive = torch.tensor([1.0, 2.0])
        >>> x_mixed = torch.tensor([-1.0, 2.0])
        >>> torch.sin(x_positive)
        tensor([0., 0.])
        >>> torch.sin(x_mixed)
        tensor([-0.8415, 0.9093])
    """
    if not isinstance(op, (str, torch._ops.OpOverload)):
        raise ValueError(f"get_kernel({op}): got unexpected type for op: {type(op)}")

    if isinstance(op, torch._ops.OpOverload):
        op = op._name

    if isinstance(dispatch_key, str):
        try:
            dispatch_key = torch._C.DispatchKey.__members__[dispatch_key]
        except KeyError:
            raise ValueError(f"Invalid dispatch key: {dispatch_key}") from None

    return torch._C._dispatch_get_computed_kernel_for_dispatch_key(op, dispatch_key)


_OPCHECK_DEFAULT_UTILS = (
    "test_schema",
    "test_autograd_registration",
    "test_faketensor",
    "test_aot_dispatch_dynamic",
)


def opcheck(
    op: Union[torch._ops.OpOverload, torch._ops.OpOverloadPacket, CustomOpDef],
    args: tuple[Any, ...],
    kwargs: Optional[dict[str, Any]] = None,
    *,
    test_utils: Union[str, Sequence[str]] = _OPCHECK_DEFAULT_UTILS,
    raise_exception: bool = True,
    atol=None,
    rtol=None,
) -> dict[str, str]:
    """Given an operator and some sample arguments, tests if the operator is
    registered correctly.

    That is, when you use the torch.library/TORCH_LIBRARY APIs to create a
    custom op, you specified metadata (e.g. mutability info) about the custom op
    and these APIs require that the functions you pass them satisfy certain
    properties (e.g. no data pointer access in the fake/meta/abstract kernel)
    ``opcheck`` tests these metadata and properties.

    Concretely, we test the following:

    - test_schema: If the schema matches the implementation of
      the operator. For example: if the schema specifies a Tensor is mutated,
      then we check the implementation mutates the Tensor. If the schema
      specifies that we return a new Tensor, then we check that the
      implementation returns a new Tensor (instead of an existing one or
      a view of an existing one).
    - test_autograd_registration: If the operator supports training
      (autograd): we check that its autograd formula is registered via
      torch.library.register_autograd or a manual registration to one
      or more DispatchKey::Autograd keys. Any other DispatchKey-based
      registrations may lead to undefined behavior.
    - test_faketensor: If the operator has a FakeTensor kernel
      (and if it is correct). The FakeTensor kernel is necessary (
      but not sufficient) for the operator to work with PyTorch compilation
      APIs (torch.compile/export/FX). We check that a FakeTensor kernel
      (also sometimes known as a meta kernel) was registered for the
      operator and that it is correct. This test takes the result of
      running the operator on real tensors and the result of running
      the operator on FakeTensors and checks that they have the same
      Tensor metadata (sizes/strides/dtype/device/etc).
    - test_aot_dispatch_dynamic: If the operator has correct behavior
      with PyTorch compilation APIs (torch.compile/export/FX).
      This checks that the outputs (and gradients, if applicable) are the
      same under eager-mode PyTorch and torch.compile.
      This test is a superset of ``test_faketensor`` and is an e2e test;
      other things it tests are that the operator supports
      functionalization and that the backward pass (if it exists) also
      supports FakeTensor and functionalization.

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
        rtol (Optional[float]): Relative tolerance for floating point comparisons.
            If specified ``atol`` must also be specified.
            If omitted, default values based on the ``dtype`` are selected
            (see the table in :func:`torch.testing.assert_close`).
        atol (Optional[float]): Absolute tolerance for floating point comparisons.
            If specified ``rtol`` must also be specified.
            If omitted, default values based on the ``dtype`` are selected
            (see the table in :func:`torch.testing.assert_close`).

    .. warning::

        opcheck and :func:`torch.autograd.gradcheck` test different things;
        opcheck tests if your usage of torch.library APIs is correct while
        :func:`torch.autograd.gradcheck` tests if your autograd formula is
        mathematically correct. Use both to test custom ops that support
        gradient computation.

    Example:

        >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_CUDA)
        >>> @torch.library.custom_op("mylib::numpy_mul", mutates_args=())
        >>> def numpy_mul(x: Tensor, y: float) -> Tensor:
        >>>     x_np = x.numpy(force=True)
        >>>     z_np = x_np * y
        >>>     return torch.from_numpy(z_np).to(x.device)
        >>>
        >>> @numpy_mul.register_fake
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
        >>> numpy_mul.register_autograd(backward, setup_context=setup_context)
        >>>
        >>> sample_inputs = [
        >>>     (torch.randn(3), 3.14),
        >>>     (torch.randn(2, 3, device='cuda'), 2.718),
        >>>     (torch.randn(1, 10, requires_grad=True), 1.234),
        >>>     (torch.randn(64, 64, device='cuda', requires_grad=True), 90.18),
        >>> ]
        >>>
        >>> for args in sample_inputs:
        >>>     torch.library.opcheck(numpy_mul, args)

    """
    import torch.testing._internal.optests as optests

    return optests.opcheck(
        op,
        args,
        kwargs,
        test_utils=test_utils,
        raise_exception=raise_exception,
        rtol=rtol,
        atol=atol,
    )
