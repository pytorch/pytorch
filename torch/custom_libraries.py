import torch

existing_libraries = {'aten', 'quantized'}

# User created fragment libraries to extend existing libraries
fragments_for_existing_libraries = {}

# User defined libraries
libraries = {}

# Set containing the combination of (namespace, operator, DispatchKey) for which a new kernel has been registered
# The keys in the set are of the form `namespace + "/" + op_name + "/" + dispatch_key`.
# This set is maintained to ensure that two libraries don't try to override the exact same functionality to avoid
# libraries calling into kernels not intended to be called.
impls = set()

class _Library:
    # kind can only be FRAGMENT for now.
    # TODO: add support for DEF
    def __init__(self, kind, ns, dispatch_key=""):
       # also prevent users from creating libraries with existing C++ libraries, e.g., aten
       if kind == "DEF":
          if ns in existing_libraries or ns in libraries:
            raise ValueError("A library with name " + ns + " already exists. " \
                                "It's not allowed to have more than one library with the same name")
          else:
            libraries[ns] = self
       elif kind == "FRAGMENT":
          fragments_for_existing_libraries[id(self)] = self
       else:
            raise ValueError("Unsupported kind: ", kind)

       self.m = torch._C._dispatch_library(kind, ns, dispatch_key)
       self.ns = ns
       self.op_impls = set()
       self.kind = kind

    def impl(self, name, dispatch_key, fn):
        key = self.ns + "/" + name + "/" + dispatch_key
        if key in impls:
            raise RuntimeError("This is not allowed since there's already a kernel overriding " \
                            + str(getattr(getattr(torch.ops, self.ns), name)) + "'s behavior for " + dispatch_key + " dispatch key.")
        else:
            impls.add(key)
            self.op_impls.add(key)
            self.m.impl(name, dispatch_key, fn)

    def define(self, schema):
        self.m.define(schema)

    def remove(self):
        for key in self.op_impls:
            impls.remove(key)
        if self.kind == "DEF":
            del libraries[self.ns]
        else:
            # kind = "FRAGMENT"
            del fragments_for_existing_libraries[id(self)]
        del self.m

# Every user can create their own fragment to extend existing C++ libraries
# We don't guarantee the user that another library that they imported is not overriding aten
# However two libraries are not allowed to override the same operator in the same namespace for the same dispatch key
def extend_library(ns):
    if ns in existing_libraries or ns in libraries:
        return _Library("FRAGMENT", ns)
    else:
        raise ValueError("A library with name " + ns + " does not exist.")

def create_library(ns):
    if not (ns in existing_libraries or ns in libraries):
        return _Library("DEF", ns)
    else:
        raise ValueError("A library with name " + ns + " already exists.")
