from sympy.core import Symbol, Integer

x = Symbol('x')
i3 = Integer(3)


def timeit_x_is_integer():
    x.is_integer


def timeit_Integer_is_irrational():
    i3.is_irrational
