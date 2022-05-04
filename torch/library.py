from ._ops import OpOverload
from typing import Set
import traceback
import torch

__all__ = ['extend_library', 'create_library']

# User created libraries to extend existing libraries
# Each user created library is added here to ensure that it's not automatically removed outside the
# scope of the function it was created in.
_impls_for_existing_libraries = {}

# User created custom libraries
_libraries = {}

# Set containing the combination of (namespace, operator, DispatchKey) for which a new kernel has been registered
# The keys in the set are of the form `namespace + "/" + op_name + "/" + dispatch_key`.
# This set is maintained to ensure that two libraries don't try to override the exact same functionality to avoid
# libraries calling into kernels not intended to be called.
_impls: Set[str] = set()

class _Library:
    # kind can be DEF, IMPL
    def __init__(self, kind, ns, dispatch_key=""):
        frame = traceback.extract_stack(limit=3)[0]
        filename, lineno = frame.filename, frame.lineno
        self.m = torch._C._dispatch_library(kind, ns, dispatch_key, filename, lineno)
        self.ns = ns
        self._op_impls = set()
        self.kind = kind
        self.dispatch_key = dispatch_key
        if kind == "IMPL":
            _impls_for_existing_libraries[id(self)] = self
        elif kind == "DEF":
            getattr(torch.ops, ns)
            _libraries[id(self)] = self
        else:
            raise ValueError("Unsupported kind: ", kind)

    def __repr__(self):
        return "<Library(kind='{}', ns='{}', dispatch_key='{}')>".format(self.kind, self.ns, self.dispatch_key)

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

    # Libraries can be removed at any point by explicitly calling .remove()
    def remove(self):
        for key in self._op_impls:
            _impls.remove(key)
        if self.kind == "DEF":
            del _libraries[id(self)]
            torch.ops.__dict__.pop(self.ns)
        else:
            del _impls_for_existing_libraries[id(self)]
        del self.m

# Every user can create their own IMPL to extend existing C++ libraries
# We don't guarantee the user that another library that they imported is not overriding aten
# However two libraries are not allowed to override the same operator in the same namespace for the same dispatch key.
def extend_library(ns, dispatch_key=""):
    """Creates a library IMPL object that can be used to override kernels for a given library name.
       Optionally a user can pass in a dispatch keyname if they only want to override kernels corresponding
       to one specific dispatch key.

    Args:
        ns: library name
        dispatch_key: PyTorch dispatch key (default: "")

    Returns:
        None
    """
    # TODO: check if there's an existing library with name ns
    return _Library("IMPL", ns, dispatch_key)

def create_library(ns):
    return _Library("DEF", ns, "")
