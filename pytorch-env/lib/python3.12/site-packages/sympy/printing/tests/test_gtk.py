from sympy.functions.elementary.trigonometric import sin
from sympy.printing.gtk import print_gtk
from sympy.testing.pytest import XFAIL, raises

# this test fails if python-lxml isn't installed. We don't want to depend on
# anything with SymPy


@XFAIL
def test_1():
    from sympy.abc import x
    print_gtk(x**2, start_viewer=False)
    print_gtk(x**2 + sin(x)/4, start_viewer=False)


def test_settings():
    from sympy.abc import x
    raises(TypeError, lambda: print_gtk(x, method="garbage"))
