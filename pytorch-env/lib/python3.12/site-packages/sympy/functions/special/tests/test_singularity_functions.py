from sympy.core.function import (Derivative, diff)
from sympy.core.numbers import (Float, I, nan, oo, pi)
from sympy.core.relational import Eq
from sympy.core.symbol import (Symbol, symbols)
from sympy.functions.elementary.piecewise import Piecewise
from sympy.functions.special.delta_functions import (DiracDelta, Heaviside)
from sympy.functions.special.singularity_functions import SingularityFunction
from sympy.series.order import O


from sympy.core.expr import unchanged
from sympy.core.function import ArgumentIndexError
from sympy.testing.pytest import raises

x, y, a, n = symbols('x y a n')


def test_fdiff():
    assert SingularityFunction(x, 4, 5).fdiff() == 5*SingularityFunction(x, 4, 4)
    assert SingularityFunction(x, 4, -1).fdiff() == SingularityFunction(x, 4, -2)
    assert SingularityFunction(x, 4, -2).fdiff() == SingularityFunction(x, 4, -3)
    assert SingularityFunction(x, 4, -3).fdiff() == SingularityFunction(x, 4, -4)
    assert SingularityFunction(x, 4, 0).fdiff() == SingularityFunction(x, 4, -1)

    assert SingularityFunction(y, 6, 2).diff(y) == 2*SingularityFunction(y, 6, 1)
    assert SingularityFunction(y, -4, -1).diff(y) == SingularityFunction(y, -4, -2)
    assert SingularityFunction(y, 4, 0).diff(y) == SingularityFunction(y, 4, -1)
    assert SingularityFunction(y, 4, 0).diff(y, 2) == SingularityFunction(y, 4, -2)

    n = Symbol('n', positive=True)
    assert SingularityFunction(x, a, n).fdiff() == n*SingularityFunction(x, a, n - 1)
    assert SingularityFunction(y, a, n).diff(y) == n*SingularityFunction(y, a, n - 1)

    expr_in = 4*SingularityFunction(x, a, n) + 3*SingularityFunction(x, a, -1) + -10*SingularityFunction(x, a, 0)
    expr_out = n*4*SingularityFunction(x, a, n - 1) + 3*SingularityFunction(x, a, -2) - 10*SingularityFunction(x, a, -1)
    assert diff(expr_in, x) == expr_out

    assert SingularityFunction(x, -10, 5).diff(evaluate=False) == (
        Derivative(SingularityFunction(x, -10, 5), x))

    raises(ArgumentIndexError, lambda: SingularityFunction(x, 4, 5).fdiff(2))


def test_eval():
    assert SingularityFunction(x, a, n).func == SingularityFunction
    assert unchanged(SingularityFunction, x, 5, n)
    assert SingularityFunction(5, 3, 2) == 4
    assert SingularityFunction(3, 5, 1) == 0
    assert SingularityFunction(3, 3, 0) == 1
    assert SingularityFunction(3, 3, 1) == 0
    assert SingularityFunction(Symbol('z', zero=True), 0, 1) == 0  # like sin(z) == 0
    assert SingularityFunction(4, 4, -1) is oo
    assert SingularityFunction(4, 2, -1) == 0
    assert SingularityFunction(4, 7, -1) == 0
    assert SingularityFunction(5, 6, -2) == 0
    assert SingularityFunction(4, 2, -2) == 0
    assert SingularityFunction(4, 4, -2) is oo
    assert SingularityFunction(4, 2, -3) == 0
    assert SingularityFunction(8, 8, -3) is oo
    assert SingularityFunction(4, 2, -4) == 0
    assert SingularityFunction(8, 8, -4) is oo
    assert (SingularityFunction(6.1, 4, 5)).evalf(5) == Float('40.841', '5')
    assert SingularityFunction(6.1, pi, 2) == (-pi + 6.1)**2
    assert SingularityFunction(x, a, nan) is nan
    assert SingularityFunction(x, nan, 1) is nan
    assert SingularityFunction(nan, a, n) is nan

    raises(ValueError, lambda: SingularityFunction(x, a, I))
    raises(ValueError, lambda: SingularityFunction(2*I, I, n))
    raises(ValueError, lambda: SingularityFunction(x, a, -5))


def test_leading_term():
    l = Symbol('l', positive=True)
    assert SingularityFunction(x, 3, 2).as_leading_term(x) == 0
    assert SingularityFunction(x, -2, 1).as_leading_term(x) == 2
    assert SingularityFunction(x, 0, 0).as_leading_term(x) == 1
    assert SingularityFunction(x, 0, 0).as_leading_term(x, cdir=-1) == 0
    assert SingularityFunction(x, 0, -1).as_leading_term(x) == 0
    assert SingularityFunction(x, 0, -2).as_leading_term(x) == 0
    assert SingularityFunction(x, 0, -3).as_leading_term(x) == 0
    assert SingularityFunction(x, 0, -4).as_leading_term(x) == 0
    assert (SingularityFunction(x + l, 0, 1)/2\
        - SingularityFunction(x + l, l/2, 1)\
        + SingularityFunction(x + l, l, 1)/2).as_leading_term(x) == -x/2


def test_series():
    l = Symbol('l', positive=True)
    assert SingularityFunction(x, -3, 2).series(x) == x**2 + 6*x + 9
    assert SingularityFunction(x, -2, 1).series(x) == x + 2
    assert SingularityFunction(x, 0, 0).series(x) == 1
    assert SingularityFunction(x, 0, 0).series(x, dir='-') == 0
    assert SingularityFunction(x, 0, -1).series(x) == 0
    assert SingularityFunction(x, 0, -2).series(x) == 0
    assert SingularityFunction(x, 0, -3).series(x) == 0
    assert SingularityFunction(x, 0, -4).series(x) == 0
    assert (SingularityFunction(x + l, 0, 1)/2\
        - SingularityFunction(x + l, l/2, 1)\
        + SingularityFunction(x + l, l, 1)/2).nseries(x) == -x/2 + O(x**6)


def test_rewrite():
    assert SingularityFunction(x, 4, 5).rewrite(Piecewise) == (
        Piecewise(((x - 4)**5, x - 4 >= 0), (0, True)))
    assert SingularityFunction(x, -10, 0).rewrite(Piecewise) == (
        Piecewise((1, x + 10 >= 0), (0, True)))
    assert SingularityFunction(x, 2, -1).rewrite(Piecewise) == (
        Piecewise((oo, Eq(x - 2, 0)), (0, True)))
    assert SingularityFunction(x, 0, -2).rewrite(Piecewise) == (
        Piecewise((oo, Eq(x, 0)), (0, True)))

    n = Symbol('n', nonnegative=True)
    p = SingularityFunction(x, a, n).rewrite(Piecewise)
    assert p == (
        Piecewise(((x - a)**n, x - a >= 0), (0, True)))
    assert p.subs(x, a).subs(n, 0) == 1

    expr_in = SingularityFunction(x, 4, 5) + SingularityFunction(x, -3, -1) - SingularityFunction(x, 0, -2)
    expr_out = (x - 4)**5*Heaviside(x - 4, 1) + DiracDelta(x + 3) - DiracDelta(x, 1)
    assert expr_in.rewrite(Heaviside) == expr_out
    assert expr_in.rewrite(DiracDelta) == expr_out
    assert expr_in.rewrite('HeavisideDiracDelta') == expr_out

    expr_in = SingularityFunction(x, a, n) + SingularityFunction(x, a, -1) - SingularityFunction(x, a, -2)
    expr_out = (x - a)**n*Heaviside(x - a, 1) + DiracDelta(x - a) + DiracDelta(a - x, 1)
    assert expr_in.rewrite(Heaviside) == expr_out
    assert expr_in.rewrite(DiracDelta) == expr_out
    assert expr_in.rewrite('HeavisideDiracDelta') == expr_out
