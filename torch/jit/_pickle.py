# These functions are referenced from the pickle archives produced by
# ScriptModule.save()
import sys

def build_intlist(data):
    return data


def build_tensorlist(data):
    return data


def build_doublelist(data):
    return data


def build_boollist(data):
    return data


def build_class(qualname, state):
    path = qualname.split(".")
    module_name = ".".join(path[:-1]).replace('__torch__', '__main__')
    class_name = path[-1]
    m = getattr(sys.modules[module_name], class_name)
    obj = m.__new__(m)
    if hasattr(obj, '__setstate__'):
        obj.__setstate__(state)
    return obj


def build_tensor_from_id(data):
    if isinstance(data, int):
        # just the id, can't really do anything
        return data
