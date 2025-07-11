from ..decorators import substitute_in_graph

import pickle


__all__ = [
    "loads",
    "dumps",
]


@substitute_in_graph(pickle.dumps, can_constant_fold_through=True)
def dumps(obj, protocol=None, *, fix_imports=True, buffer_callback=None):
    return pickle.dumps(obj, protocol=protocol, fix_imports=fix_imports, buffer_callback=buffer_callback)


@substitute_in_graph(pickle.loads, can_constant_fold_through=True)
def loads(data, /, *, fix_imports=True, encoding="ASCII", errors="strict",
           buffers=()):
    return pickle.loads(data, fix_imports=fix_imports, encoding=encoding, errors=errors, buffers=buffers)
