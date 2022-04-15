import torch

libraries = {}

class Library:
    # kind can be DEF, IMPL, FRAGMENT
    def __init__(self, kind, ns, dispatch_key=""):
       # also prevent users from creating libraries with existing C++ libraries, e.g., aten
       if ns in libraries:
          raise ValueError("A library with name ", ns, " already exists. " \
                            "It's not allowed to have more than one library with the same name")
       else:
          self.m = torch._C._dispatch_library(kind, ns, dispatch_key)
          self.name = ns
          libraries[ns] = self

    def impl(self, name, dispatch_key, fn):
        self.m.impl(name, dispatch_key, fn)
        return

    def __del__(self):
        print("Entering destructor")
        del self.m

def get_library(ns):
    print("Entering")
    if ns not in libraries:
        libraries[ns] = Library("FRAGMENT", ns)
    return libraries[ns]

def remove_library(ns):
    if ns in libraries:
        del libraries[ns]
    return