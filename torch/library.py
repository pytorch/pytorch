from ._ops import OpOverload
from typing import Set
import traceback
import torch

__all__ = ['Library']

# Set containing the combination of (namespace, operator, DispatchKey) for which a new kernel has been registered
# The keys in the set are of the form `namespace + "/" + op_name + "/" + dispatch_key`.
# This set is maintained to ensure that two libraries don't try to override the exact same functionality to avoid
# libraries calling into kernels not intended to be called.
_impls: Set[str] = set()

class Library:
    """
    A class to create libraries that can be used to register new operators or
    override operators in existing libraries from Python.
    A user can optionally pass in a dispatch keyname if they only want to register
    kernels corresponding to only one specific dispatch key.

    Args:
        ns: library name
        kind: "DEF", "IMPL" (default: "IMPL")
        dispatch_key: PyTorch dispatch key (default: "")
    """
    def __init__(self, ns, kind, dispatch_key=""):
        if kind != "IMPL" and kind != "DEF":
            raise ValueError("Unsupported kind: ", kind)
        frame = traceback.extract_stack(limit=3)[0]
        filename, lineno = frame.filename, frame.lineno
        self.m = torch._C._dispatch_library(kind, ns, dispatch_key, filename, lineno)
        self.ns = ns
        self._op_impls = set()
        self.kind = kind
        self.dispatch_key = dispatch_key

    def __repr__(self):
        return "Library(kind={}, ns={}, dispatch_key={})>".format(self.kind, self.ns, self.dispatch_key)

    def impl(self, op_name, fn, dispatch_key=''):
        if dispatch_key == '':
            if self.dispatch_key == '':
                raise RuntimeError("Please specify the dispatch key that you want to register the kernel for.")
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

        self.m.impl(name, dispatch_key, fn)
        _impls.add(key)
        self._op_impls.add(key)

    def define(self, schema):
        self.m.define(schema)

    def __del__(self):
        for key in self._op_impls:
            _impls.remove(key)
        del self.m
