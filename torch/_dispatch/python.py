import torch._C
from contextlib import contextmanager
import torch._ops

from typing import Dict, Callable

__all__ = ['enable_python_dispatcher', 'no_python_dispatcher', 'patch_py_impls']

@contextmanager
def no_python_dispatcher():
    g = torch._C._DisablePythonDispatcher()
    try:
        yield
    finally:
        del g

@contextmanager
def enable_python_dispatcher():
    g = torch._C._EnablePythonDispatcher()
    try:
        yield
    finally:
        del g

@contextmanager
def patch_py_impls(all_patches: Dict[torch._ops.OpOverload, Dict[torch._C.DispatchKey, Callable]]):
    """
    Temporarily patch the dispatcher registrations in the Python Dispatcher,
    undoing them when you exit the context manager.  This is useful for
    temporarily adding pre-autograd decompositions, among other things.
    """
    saved_tables = {}
    for op, patches in all_patches.items():
        # TODO: Make this public API on OpOverload instead
        # of groveling the attribute directly
        saved_tables[op] = op.py_kernels.copy()
        for k, fn in patches.items():
            op.py_impl(k)(fn)
    try:
        yield
    finally:
        for op in all_patches:
            # TODO: Make this OpOverload API
            op.py_kernels.clear()
            op.py_kernels.update(saved_tables[op])
            op._dispatch_cache.clear()
