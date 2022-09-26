import torch._C
from contextlib import contextmanager

__all__ = ['enable_python_dispatcher', 'no_python_dispatcher']

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
