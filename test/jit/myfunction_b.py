r"""
Helper function used in test_decorator.py. We define it in a
separate file on purpose to test that the names in different modules
are resolved correctly.
"""

from jit.mydecorator import my_decorator


@my_decorator
def my_function_b(x: float) -> float:
    return my_function_c(x) + 2


def my_function_c(x: float) -> float:
    return x + 3
