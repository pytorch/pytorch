# This test file tests the SymPy function interface, that people use to create
# their own new functions. It should be as easy as possible.
from sympy.core.function import Function
from sympy.core.sympify import sympify
from sympy.functions.elementary.hyperbolic import tanh
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.series.limits import limit
from sympy.abc import x


def test_function_series1():
    """Create our new "sin" function."""

    class my_function(Function):

        def fdiff(self, argindex=1):
            return cos(self.args[0])

        @classmethod
        def eval(cls, arg):
            arg = sympify(arg)
            if arg == 0:
                return sympify(0)

    #Test that the taylor series is correct
    assert my_function(x).series(x, 0, 10) == sin(x).series(x, 0, 10)
    assert limit(my_function(x)/x, x, 0) == 1


def test_function_series2():
    """Create our new "cos" function."""

    class my_function2(Function):

        def fdiff(self, argindex=1):
            return -sin(self.args[0])

        @classmethod
        def eval(cls, arg):
            arg = sympify(arg)
            if arg == 0:
                return sympify(1)

    #Test that the taylor series is correct
    assert my_function2(x).series(x, 0, 10) == cos(x).series(x, 0, 10)


def test_function_series3():
    """
    Test our easy "tanh" function.

    This test tests two things:
      * that the Function interface works as expected and it's easy to use
      * that the general algorithm for the series expansion works even when the
        derivative is defined recursively in terms of the original function,
        since tanh(x).diff(x) == 1-tanh(x)**2
    """

    class mytanh(Function):

        def fdiff(self, argindex=1):
            return 1 - mytanh(self.args[0])**2

        @classmethod
        def eval(cls, arg):
            arg = sympify(arg)
            if arg == 0:
                return sympify(0)

    e = tanh(x)
    f = mytanh(x)
    assert e.series(x, 0, 6) == f.series(x, 0, 6)
