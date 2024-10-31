r"""
Decorator used in test_decorator.py. We define it in a
separate file on purpose to test that the names in different modules
are resolved correctly.
"""

import functools


def my_decorator(func):
    """Dummy decorator that removes itself when torchscripting"""

    @functools.wraps(func)
    def wrapped_func(*args, **kwargs):
        return func(*args, **kwargs)

    # torch.jit.script() uses __prepare_scriptable__ to remove the decorator
    wrapped_func.__prepare_scriptable__ = lambda: func

    return wrapped_func
