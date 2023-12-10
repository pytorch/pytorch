from ._ops import OpOverload
from typing import Any, Optional, Set, List
import traceback
import torch
import weakref
import functools
import inspect
import re
import sys

__all__ = [
    'Library',
    'impl',
    'define',
    'fallthrough_kernel',
    'impl_abstract',
    'get_ctx',
]

# Set containing the combination of (namespace, operator, DispatchKey) for which a new kernel has been registered
# The keys in the set are of the form `namespace + "/" + op_name + "/" + dispatch_key`.
# This set is maintained to ensure that two libraries don't try to override the exact same functionality to avoid
# libraries calling into kernels not intended to be called.
_impls: Set[str] = set()
_defs: Set[str] = set()

# prim is reserved by TorchScript interpreter
_reserved_namespaces = ['prim']

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
        if kind not in ('IMPL', 'DEF', 'FRAGMENT'):
            raise ValueError("Unsupported kind: ", kind)

        if ns in _reserved_namespaces and (kind == "DEF" or kind == 'FRAGMENT'):
            raise ValueError(ns, " is a reserved namespace. Please try creating a library with another name.")

        frame = traceback.extract_stack(limit=3)[0]
        filename, lineno = frame.filename, frame.lineno
        self.m: Optional[Any] = torch._C._dispatch_library(kind, ns, dispatch_key, filename, lineno)
        self.ns = ns
        self._op_defs: Set[str] = set()
        self._op_impls: Set[str] = set()
        self._registration_handles: List["torch._library.utils.RegistrationHandle"] = []
        self.kind = kind
        self.dispatch_key = dispatch_key
        # Use a finalizer to setup the "destructor" instead of __del__.
        # Python __del__ can lead to weird things (globals and locals may already
        # be gone when __del__ actually gets called!). finalizers help the
        # situation because it lets us capture references and keeps them alive
        weakref.finalize(self, _del_library, _impls, self._op_impls, _defs, self._op_defs, self._registration_handles)

    def __repr__(self):
        return f"Library(kind={self.kind}, ns={self.ns}, dispatch_key={self.dispatch_key})>"

    def define(self, schema, alias_analysis="", *, tags=()):
        r'''Defines a new operator and its semantics in the ns namespace.

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
            >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_LIBRARY)
            >>> my_lib = Library("foo", "DEF")
            >>> my_lib.define("sum(Tensor self) -> Tensor")
        '''
        # This is added because we also want to disallow PURE_FUNCTION alias analysis which is a valid
        # AliasAnalysis type in C++
        if alias_analysis not in ["", "FROM_SCHEMA", "CONSERVATIVE"]:
            raise RuntimeError(f"Invalid alias_analysis type {alias_analysis}")
        assert self.m is not None
        if isinstance(tags, torch.Tag):
            tags = (tags,)
        result = self.m.define(schema, alias_analysis, tuple(tags))
        qualname = self.ns + "::" + schema.split("(")[0]
        self._op_defs.add(qualname)
        _defs.add(qualname)
        return result

    def impl(self, op_name, fn, dispatch_key=''):
        r'''Registers the function implementation for an operator defined in the library.

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
        '''
        if not callable(fn):
            raise TypeError(f"Input function is required to be a callable but found type {type(fn)}")
        if dispatch_key == '':
            dispatch_key = self.dispatch_key

        if isinstance(op_name, str):
            name = op_name
        elif isinstance(op_name, OpOverload):
            name = op_name._schema.name
            overload_name = op_name._schema.overload_name
            if overload_name != '':
                name = name + '.' + overload_name
        else:
            raise RuntimeError("impl should be passed either a name or an OpOverload object as the first argument")

        key = self.ns + "/" + name.split("::")[-1] + "/" + dispatch_key
        if key in _impls:
            # TODO: in future, add more info about where the existing function is registered (this info is
            # today already returned by the C++ warning when impl is called but we error out before that)
            raise RuntimeError("This is not allowed since there's already a kernel registered from python overriding {}"
                               "'s behavior for {} dispatch key and {} namespace.".
                               format(name.split("::")[-1], dispatch_key, self.ns))

        if dispatch_key == "Meta":
            dispatcher_op_name = name
            if '::' not in dispatcher_op_name:
                dispatcher_op_name = f'{self.ns}::{dispatcher_op_name}'

            # Internally, we shouldn't be registering meta kernels for any operators that
            # have CompositeImplicitAutograd kernels.
            # Instead, we should be letting those decompositions run, and writing meta kernels
            # only for the base operators.
            if torch._C._dispatch_has_kernel_for_dispatch_key(dispatcher_op_name, "CompositeImplicitAutograd"):
                raise RuntimeError(
                    f"We should not register a meta kernel directly to the operator '{name}',"
                    " because it has a CompositeImplicitAutograd kernel in core."
                    " Instead we should let the operator decompose, and ensure that we have meta kernels"
                    " for the base ops that it decomposes into.")

        assert self.m is not None
        self.m.impl(name, dispatch_key if dispatch_key != "" else "CompositeImplicitAutograd", fn)

        _impls.add(key)
        self._op_impls.add(key)

    def _destroy(self):
        self.m = None
        for handle in self._registration_handles:
            handle.destroy()
        self._registration_handles.clear()


def _del_library(captured_impls, op_impls, captured_defs, op_defs, registration_handles):
    captured_impls -= op_impls
    captured_defs -= op_defs
    for handle in registration_handles:
        handle.destroy()


_keep_alive = []


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
    :func:`torch.library.impl_abstract`.

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
        >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_LIBRARY)
        >>> import torch
        >>> import numpy as np
        >>>
        >>> # Define the operator
        >>> torch.library.define("mylib::sin", "(Tensor x) -> Tensor")
        >>>
        >>> # Add implementations for the operator
        >>> @torch.library.impl("mylibrary::sin", "cpu")
        >>> def f(x):
        >>>     return torch.from_numpy(np.sin(x.numpy()))
        >>>
        >>> # Call the new operator from torch.ops.
        >>> x = torch.randn(3)
        >>> y = torch.ops.mylib.sin(x)
        >>> assert torch.allclose(y, x)

    """
    if not isinstance(qualname, str):
        raise ValueError(
            f"define(qualname, schema): expected qualname "
            f"to be instance of str, got {type(qualname)}")
    namespace, name = torch._library.utils.parse_namespace(qualname)
    if lib is None:
        lib = Library(namespace, "FRAGMENT")
        _keep_alive.append(lib)
    if not NAMELESS_SCHEMA.fullmatch(schema):
        raise ValueError(
            f"define(qualname, schema, ...): expected schema "
            f"to look like e.g. \"(Tensor x) -> Tensor\" but "
            f"got \"{schema}\"")
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
        >>> torch.library.define("mylibrary::sin", "(Tensor x) -> Tensor")
        >>>
        >>> # Add implementations for the cpu device
        >>> @torch.library.impl("mylibrary::sin", "cpu")
        >>> def f(x):
        >>>     return torch.from_numpy(np.sin(x.numpy()))
        >>>
        >>> x = torch.randn(3)
        >>> y = torch.ops.mylibrary.sin(x)
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



def impl_abstract(qualname, func=None, *, lib=None, _stacklevel=1):
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
    https://docs.google.com/document/d/1W--T6wz8IY8fOI0Vm8BF44PdBgs283QvpelJZWieQWQ/edit

    Examples:
        >>> import torch
        >>> import numpy as np
        >>> from torch import Tensor
        >>>
        >>> # Example 1: an operator without data-dependent output shape
        >>> torch.library.define(
        >>>     "mylib::custom_linear",
        >>>     "(Tensor x, Tensor weight, Tensor bias) -> Tensor")
        >>>
        >>> @torch.library.impl_abstract("mylib::custom_linear")
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
        >>> torch.library.define("mylib::custom_nonzero", "(Tensor x) -> Tensor")
        >>>
        >>> @torch.library.impl_abstract("mylib::custom_nonzero")
        >>> def custom_nonzero_abstract(x):
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
        >>> @torch.library.impl("mylib::custom_nonzero", "cpu")
        >>> def custom_nonzero_cpu(x):
        >>>     x_np = x.numpy()
        >>>     res = np.stack(np.nonzero(x_np), axis=1)
        >>>     return torch.tensor(res, device=x.device)

    """
    source = torch._library.utils.get_source(_stacklevel + 1)
    frame = sys._getframe(_stacklevel)
    caller_module = inspect.getmodule(frame)
    # Can be none if you call impl_abstract from somewhere there isn't a module
    # (e.g. __main__)
    caller_module_name = None if caller_module is None else caller_module.__name__

    # TODO(rzou): We're gonna need to stage this change with torchvision,
    # since torchvision is github first.
    if caller_module_name is not None and caller_module_name.startswith("torchvision."):
        caller_module_name = None

    def inner(func):
        entry = torch._library.simple_registry.singleton.find(qualname)
        if caller_module_name is not None:
            func_to_register = _check_pystubs_once(func, qualname, caller_module_name)
        else:
            func_to_register = func

        handle = entry.abstract_impl.register(func_to_register, source)
        if lib is not None:
            lib._registration_handles.append(handle)
        return func

    if func is None:
        return inner
    return inner(func)


# If the op was defined in C++, then we want to make sure there was an
# m.impl_abstract_pystub(module, ...) call and that the module is the
# same as the module that called torch.library.impl_abstract.
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
            op._schema.name,
            op._schema.overload_name)
        if not maybe_pystub:
            raise RuntimeError(
                f"Operator '{qualname}' was defined in C++ and has a Python "
                f"abstract impl. In this situation, it is required to have a "
                f"C++ `m.impl_abstract_pystub` call, but we could not find one."
                f"Please add a call to `m.impl_abstract_pystub(\"{actual_module_name}\");` "
                f"to the C++ TORCH_LIBRARY block the operator was "
                f"defined in.")
        pystub_module = maybe_pystub[0]
        if actual_module_name != pystub_module:
            raise RuntimeError(
                f"Operator '{qualname}' specified that its python abstract impl "
                f"is in the Python module '{pystub_module}' but it was actually found "
                f"in '{actual_module_name}'. Please either move the abstract impl "
                f"or correct the m.impl_abstract_pystub call.")
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
def get_ctx() -> "torch._library.abstract_impl.AbstractImplCtx":
    """get_ctx() returns the current AbstractImplCtx object.

    Calling ``get_ctx()`` is only valid inside of an abstract impl
    (see :func:`torch.library.impl_abstract` for more usage details.
    """
    return torch._library.abstract_impl.global_ctx_getter()
