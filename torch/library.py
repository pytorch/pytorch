from ._ops import OpOverload
from typing import Callable, Dict, Set
import traceback
import torch
import dataclasses
import inspect

__all__ = ['Library', 'impl', 'define', 'impl_fake']

# Set containing the combination of (namespace, operator, DispatchKey) for which a new kernel has been registered
# The keys in the set are of the form `namespace + "/" + op_name + "/" + dispatch_key`.
# This set is maintained to ensure that two libraries don't try to override the exact same functionality to avoid
# libraries calling into kernels not intended to be called.
_impls: Set[str] = set()

# prim is reserved by TorchScript interpreter
_reserved_namespaces = ['prim']


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
        self.m = torch._C._dispatch_library(kind, ns, dispatch_key, filename, lineno)
        self.ns = ns
        self._op_impls = set()
        self.kind = kind
        self.dispatch_key = dispatch_key

    def __repr__(self):
        return "Library(kind={}, ns={}, dispatch_key={})>".format(self.kind, self.ns, self.dispatch_key)

    def define(self, schema, alias_analysis=""):
        r'''Defines a new operator and its semantics in the ns namespace.

        Args:
            schema: function schema to define a new operator.
            alias_analysis (optional): Indicates if the aliasing properties of the operator arguments can be
                                       inferred from the schema (default behavior) or not ("CONSERVATIVE").
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
            raise RuntimeError("Invalid alias_analysis type {}".format(alias_analysis))
        return self.m.define(schema, alias_analysis)

    def impl(self, op_name, fn, dispatch_key=''):
        r'''Registers the function implementation for an operator defined in the library.

        Args:
            op_name: operator name (along with the overload) or OpOverload object.
            fn: function that's the operator implementation for the input dispatch key.
            dispatch_key: dispatch key that the input function should be registered for. By default, it uses
                          the dispatch key that the library was created with.

        Example::
            >>> my_lib = Library("aten", "IMPL")
            >>> def div_cpu(self, other):
            >>>     return self * (1 / other)
            >>> my_lib.impl("div.Tensor", div_cpu, "CPU")
        '''
        if not callable(fn):
            raise TypeError("Input function is required to be a callable but found type {}".format(type(fn)))
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

        self.m.impl(name, dispatch_key if dispatch_key != "" else "CompositeImplicitAutograd", fn)

        _impls.add(key)
        self._op_impls.add(key)

    def impl_fake(self, op_name, fn, *, _stacklevel=2):
        r'''Registers a FakeTensor implementation for an operator defined in the library.

        The FakeTensor implementation is a shape propagation rule that gets invoked
        for FakeTensors (Tensors that do not have storage).

        All Tensor inputs to the FakeTensor implementation are FakeTensors
        and all Tensor returns must be FakeTensors.

        If the signature of your operator is (*args, **kwargs), then the
        signature of the FakeTensor implementation must be (ctx, *args, **kwargs).
        ctx is a context object that has helper functions for writing
        FakeTensor implementations.

        Args:
            op_name: operator name (along with the overload) or OpOverload object.
            fn: the FakeTensor implementation

        Example::
            >>> my_lib = Library("aten", "IMPL")
            >>>
            >>> def div_fake(ctx, self, other):
            >>>     return self * other.reciprocal()
            >>>
            >>> my_lib.impl_fake("div.Tensor", div_fake)

        '''
        qualname = f'{self.ns}::{op_name}'
        frame = inspect.stack()[_stacklevel]
        location = f'{frame.filename}:{frame.lineno}'
        if qualname in _fake_tensor_registry:
            location = _fake_tensor_registry[qualname].location
            raise RuntimeError(
                f"Attempting to register a FakeTensor rule for operator {qualname} "
                f"that already has a FakeTensor rule registered from Python at "
                f"{location}. This is not supported.")
        _fake_tensor_registry[qualname] = _FuncAndLocation(fn, location)

    def __del__(self):
        # _op_impls might not have been initialized if an error was thrown in __init__
        _op_impls_ = getattr(self, '_op_impls', None)
        if _op_impls_:
            for key in self._op_impls:
                _impls.remove(key)
            del self.m


@dataclasses.dataclass
class _FuncAndLocation:
    func: Callable
    location: str


_fake_tensor_registry: Dict["qualname", _FuncAndLocation] = {}

# decorator to register python functions for library ops
# Note: this decorator API should remain consistent with `Library.impl` API
def impl(lib, name, dispatch_key=""):
    def wrap(f):
        lib.impl(name, f, dispatch_key)
        return f
    return wrap


def impl_fake(lib, op_name, *, _stacklevel=3):
    r"""Register a FakeTensor implementation.

    The FakeTensor implementation is a shape propagation rule that gets invoked
    for FakeTensors (Tensors that do not have storage).

    All Tensor inputs to the FakeTensor implementation are FakeTensors
    and all Tensor returns must be FakeTensors.

    If the signature of your operator is (*args, **kwargs), then the
    signature of the FakeTensor implementation must be (ctx, *args, **kwargs).
    ctx is a context object that has helper functions for writing
    FakeTensor implementations.

    This API is used as a decorator (see examples).

    Args:
        lib: a torch.Library object
        op_name: operator name (along with the overload) or OpOverload object.

    Examples::
        >>> import numpy as np
        >>>
        >>> lib = Library('custom', 'FRAGMENT')
        >>>
        >>> # Example 1: an operator without data-dependent output shape
        >>> lib.define('linear(Tensor x, Tensor weight, Tensor bias) -> Tensor')
        >>>
        >>> @impl_fake(lib, 'linear'):
        >>> def custom_linear_fake(x, weight):
        >>>     assert x.dim() == 2
        >>>     assert weight.dim() == 2
        >>>     assert bias.dim() == 1
        >>>     assert x.shape[1] == weight.shape[1]
        >>>     assert weight.shape[0] == bias.shape[0]
        >>>
        >>>     return (x @ weight.t()) + bias
        >>>
        >>> # Example 2: an operator with data-dependent output shape
        >>> lib.define('nonzero(Tensor x) -> Tensor')
        >>>
        >>> @impl_fake(lib, 'nonzero')
        >>> def custom_nonzero(x):
        >>>     x_np = to_numpy(x)
        >>>     res = np.stack(np.nonzero(x_np), axis=1)
        >>>     # symbolic ints in PyTorch must be >= 2, so we constrain the
        >>>     # range to at least 2.
        >>>     if res.shape[0] <= 1:
        >>>         raise RuntimeError("not supported")
        >>>     return torch.tensor(res, device=x.device)

    """
    def wrap(fn):
        lib.impl_fake(op_name, fn, _stacklevel=_stacklevel)
        return fn
    return wrap


def define(lib, schema, alias_analysis=""):
    def wrap(f):
        name = lib.define(schema, alias_analysis)
        lib.impl(name, f)
        return f
    return wrap
