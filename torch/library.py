from matplotlib.style import library
import torch
from ._ops import OpOverload
from typing import Set
import traceback
import torch._C as C

__all__ = ['extend_library', 'create_library']

# User created libraries to extend existing libraries
# Each user created library is added here to ensure that it's not automatically removed outside the
# scope of the function it was created in.
impls_for_existing_libraries = {}

# User created custom libraries
libraries = {}

# Set containing the combination of (namespace, operator, DispatchKey) for which a new kernel has been registered
# The keys in the set are of the form `namespace + "/" + op_name + "/" + dispatch_key`.
# This set is maintained to ensure that two libraries don't try to override the exact same functionality to avoid
# libraries calling into kernels not intended to be called.
impls: Set[str] = set()

class Library:
    # kind can be DEF, FRAGMENT (in C++ IMPL)
    def __init__(self, kind, ns, dispatch_key="", message=""):
        self.m = C._dispatch_library(kind, ns, dispatch_key, message)
        self.ns = ns
        self.op_impls = set()
        if kind == "IMPL":
            impls_for_existing_libraries[id(self)] = self
        elif kind == "DEF":
            libraries[id(self)] = self
        else:
            raise ValueError("Unsupported kind: ", kind)

    def impl(self, op_name, dispatch_key, fn):
        if isinstance(op_name, str):
            name = op_name
        elif isinstance(op_name, OpOverload):
            name = op_name._schema.name
        else:
            raise RuntimeError("impl should be passed either a name or an OpOverload object as the first argument")

        key = self.ns + "/" + name + "/" + dispatch_key
        if key in impls:
            raise RuntimeError("This is not allowed since there's already a kernel overriding "
                               + str(getattr(getattr(torch.ops, self.ns), name)) + "'s behavior for " + dispatch_key + " dispatch key.")
        self.m.impl(name, dispatch_key, fn)
        impls.add(key)
        self.op_impls.add(key)
        print("name: ", name, " key: ", key)

    # Libraries can be removed at any point by explicitly calling .remove()
    def remove(self):
        for key in self.op_impls:
            impls.remove(key)
        if self.kind == "DEF":
            del libraries[self.ns]
        else:
            del impls_for_existing_libraries[id(self)]
        del self.m

# Every user can create their own library to extend existing C++ libraries
# We don't guarantee the user that another library that they imported is not overriding aten
# However two libraries are not allowed to override the same operator in the same namespace for the same dispatch key
def extend_library(ns, dispatch_key=""):
    '''
    something
    '''
    message  = "Library IMPL created at : \n"
    message += ''.join(traceback.format_stack())
    return Library("IMPL", ns, dispatch_key, message=message)

def create_library(ns):
    return Library("DEF", ns, '', message='')
