from sympy.core.symbol import Symbol
from sympy.functions.elementary.trigonometric import sin
from sympy.integrals.integrals import integrate

x = Symbol('x')


def bench_integrate_sin():
    integrate(sin(x), x)


def bench_integrate_x1sin():
    integrate(x**1*sin(x), x)


def bench_integrate_x2sin():
    integrate(x**2*sin(x), x)


def bench_integrate_x3sin():
    integrate(x**3*sin(x), x)
