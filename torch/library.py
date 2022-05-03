from ._ops import OpOverload
from typing import Set
import traceback
import torch._C as C
import warnings
__all__ = ['extend_library']

# User created libraries to extend existing libraries
# Each user created library is added here to ensure that it's not automatically removed outside the
# scope of the function it was created in.
impls_for_existing_libraries = {}

# Set containing the combination of (namespace, operator, DispatchKey) for which a new kernel has been registered
# The keys in the set are of the form `namespace + "/" + op_name + "/" + dispatch_key`.
# This set is maintained to ensure that two libraries don't try to override the exact same functionality to avoid
# libraries calling into kernels not intended to be called.
_impls: Set[str] = set()

class Library:
    # kind can be DEF, IMPL
    def __init__(self, kind, ns, dispatch_key="", message=""):
        frame = traceback.extract_stack()[0]
        filename, lineno = frame.filename, frame.lineno
        self.m = C._dispatch_library(kind, ns, dispatch_key, filename, lineno) # type: ignore[attr-defined]
        self.ns = ns
        self._op_impls = set()
        self.kind = kind
        self.dispatch_key = dispatch_key
        if kind == "IMPL":
            impls_for_existing_libraries[id(self)] = self
        else:
            raise ValueError("Unsupported kind: ", kind)

    def __repr__(self):
        return "<Library(kind='{}', ns='{}', dispatch_key='{}')>".format(self.kind, self.ns, self.dispatch_key)

    def impl(self, op_name, fn, dispatch_key=''):
        if dispatch_key == '':
            assert self.dispatch_key != '', "Please specify the dispatch key that you want to register the kernel for."
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
            raise RuntimeError("This is not allowed since there's already a kernel overriding {}"
                               "'s behavior for {} dispatch key and {} namespace.".
                               format(name.split("::")[-1], dispatch_key, self.ns))

        # ignore the warning when overriding an existing kernel for an op
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.m.impl(name, dispatch_key, fn)
        _impls.add(key)
        self._op_impls.add(key)

    # Libraries can be removed at any point by explicitly calling .remove()
    def remove(self):
        for key in self._op_impls:
            _impls.remove(key)
        del impls_for_existing_libraries[id(self)]
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
    return Library("IMPL", ns, dispatch_key)
