import torch
from ._ops import OpOverload
from typing import Set
import warnings

__all__ = ['extend_library', 'create_library']

# User created fragment libraries to extend existing libraries
fragments_for_existing_libraries = {}

# Set containing the combination of (namespace, operator, DispatchKey) for which a new kernel has been registered
# The keys in the set are of the form `namespace + "/" + op_name + "/" + dispatch_key`.
# This set is maintained to ensure that two libraries don't try to override the exact same functionality to avoid
# libraries calling into kernels not intended to be called.
impls: Set[str] = set()

class Library:
    # kind can be DEF, FRAGMENT (in C++ IMPL)
    def __init__(self, kind, ns, dispatch_key=""):
        # also prevent users from creating libraries with existing C++ libraries, e.g., aten
        # if ns in existing_libraries and kind == "DEF":
        #     raise ValueError("A library with name '" + ns + "' already exists. "
        #                      "It's not allowed to have more than one library with the same name")
        # elif kind == "FRAGMENT":
        #     self.m = torch._C._dispatch_library(kind, ns, dispatch_key)
        #     self.ns = ns
        #     fragments_for_existing_libraries[id(self)] = self
        #     self.op_impls = set()
        # else:
        #     raise ValueError("Unsupported kind: ", kind)
        self.m = torch._C._dispatch_library(kind, ns, dispatch_key)
        self.ns = ns
        fragments_for_existing_libraries[id(self)] = self
        self.op_impls = set()

    def impl(self, op_name, dispatch_key, fn):
        if isinstance(op_name, str):
            name = self.ns + '.' + op_name
        elif isinstance(op_name, OpOverload):
            name = op_name.__str__() if op_name.overloadname != 'default' else op_name.overloadpacket.__str__()
        else:
            raise RuntimeError("impl should be passed either a name or an OpOverload object as the first argument")

        key = self.ns + "/" + name + "/" + dispatch_key
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            # try:
            self.m.impl(name, dispatch_key, fn)
        #     except RuntimeWarning:
        #         if key in impls:
        # if key in impls:
            # raise RuntimeError("This is not allowed since there's already a kernel overriding "
            #                    + str(getattr(getattr(torch.ops, self.ns), name)) + "'s behavior for " + dispatch_key + " dispatch key.")
        # else:
        impls.add(key)
        self.op_impls.add(key)


    def remove(self):
        for key in self.op_impls:
            impls.remove(key)
        # if self.kind == "DEF":
        #     del libraries[self.ns]
        # else:
        #     # kind = "FRAGMENT"
        #     del fragments_for_existing_libraries[id(self)]
        del self.m

# Every user can create their own fragment to extend existing C++ libraries
# We don't guarantee the user that another library that they imported is not overriding aten
# However two libraries are not allowed to override the same operator in the same namespace for the same dispatch key
def extend_library(ns, dispatch_key=""):
    return Library("IMPL", ns, dispatch_key)
    # if ns in existing_libraries:
    #     return Library("FRAGMENT", ns, dispatch_key)
    # else:
    #     raise ValueError("A library with name " + ns + " does not exist.")

def create_library(ns):
    return Library("DEF", ns)
    # if not ns in libraries:
    #     return Library("DEF", ns)
    # else:
    #     raise ValueError("A library with name " + ns + " already exists.")