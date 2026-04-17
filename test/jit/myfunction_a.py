"""
Helper function used in test_decorator.py. We define it in a
separate file on purpose to test that the names in different modules
are resolved correctly.
"""

from jit.mydecorator import my_decorator
from jit.myfunction_b import my_function_b


@my_decorator
def my_function_a(x: float) -> float:
    return my_function_b(x) + 1
