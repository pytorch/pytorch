from sympy.core.numbers import (I, nan, oo, pi)
from sympy.core.relational import (Eq, Ne)
from sympy.core.singleton import S
from sympy.core.symbol import (Symbol, symbols)
from sympy.functions.elementary.complexes import (adjoint, conjugate, sign, transpose)
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.piecewise import Piecewise
from sympy.functions.special.delta_functions import (DiracDelta, Heaviside)
from sympy.functions.special.singularity_functions import SingularityFunction
from sympy.simplify.simplify import signsimp


from sympy.testing.pytest import raises

from sympy.core.expr import unchanged

from sympy.core.function import ArgumentIndexError


x, y = symbols('x y')
i = symbols('t', nonzero=True)
j = symbols('j', positive=True)
k = symbols('k', negative=True)

def test_DiracDelta():
    assert DiracDelta(1) == 0
    assert DiracDelta(5.1) == 0
    assert DiracDelta(-pi) == 0
    assert DiracDelta(5, 7) == 0
    assert DiracDelta(x, 0) == DiracDelta(x)
    assert DiracDelta(i) == 0
    assert DiracDelta(j) == 0
    assert DiracDelta(k) == 0
    assert DiracDelta(nan) is nan
    assert DiracDelta(0).func is DiracDelta
    assert DiracDelta(x).func is DiracDelta
    # FIXME: this is generally undefined @ x=0
    #         But then limit(Delta(c)*Heaviside(x),x,-oo)
    #         need's to be implemented.
    # assert 0*DiracDelta(x) == 0

    assert adjoint(DiracDelta(x)) == DiracDelta(x)
    assert adjoint(DiracDelta(x - y)) == DiracDelta(x - y)
    assert conjugate(DiracDelta(x)) == DiracDelta(x)
    assert conjugate(DiracDelta(x - y)) == DiracDelta(x - y)
    assert transpose(DiracDelta(x)) == DiracDelta(x)
    assert transpose(DiracDelta(x - y)) == DiracDelta(x - y)

    assert DiracDelta(x).diff(x) == DiracDelta(x, 1)
    assert DiracDelta(x, 1).diff(x) == DiracDelta(x, 2)

    assert DiracDelta(x).is_simple(x) is True
    assert DiracDelta(3*x).is_simple(x) is True
    assert DiracDelta(x**2).is_simple(x) is False
    assert DiracDelta(sqrt(x)).is_simple(x) is False
    assert DiracDelta(x).is_simple(y) is False

    assert DiracDelta(x*y).expand(diracdelta=True, wrt=x) == DiracDelta(x)/abs(y)
    assert DiracDelta(x*y).expand(diracdelta=True, wrt=y) == DiracDelta(y)/abs(x)
    assert DiracDelta(x**2*y).expand(diracdelta=True, wrt=x) == DiracDelta(x**2*y)
    assert DiracDelta(y).expand(diracdelta=True, wrt=x) == DiracDelta(y)
    assert DiracDelta((x - 1)*(x - 2)*(x - 3)).expand(diracdelta=True, wrt=x) == (
        DiracDelta(x - 3)/2 + DiracDelta(x - 2) + DiracDelta(x - 1)/2)

    assert DiracDelta(2*x) != DiracDelta(x)  # scaling property
    assert DiracDelta(x) == DiracDelta(-x)  # even function
    assert DiracDelta(-x, 2) == DiracDelta(x, 2)
    assert DiracDelta(-x, 1) == -DiracDelta(x, 1)  # odd deriv is odd
    assert DiracDelta(-oo*x) == DiracDelta(oo*x)
    assert DiracDelta(x - y) != DiracDelta(y - x)
    assert signsimp(DiracDelta(x - y) - DiracDelta(y - x)) == 0

    assert DiracDelta(x*y).expand(diracdelta=True, wrt=x) == DiracDelta(x)/abs(y)
    assert DiracDelta(x*y).expand(diracdelta=True, wrt=y) == DiracDelta(y)/abs(x)
    assert DiracDelta(x**2*y).expand(diracdelta=True, wrt=x) == DiracDelta(x**2*y)
    assert DiracDelta(y).expand(diracdelta=True, wrt=x) == DiracDelta(y)
    assert DiracDelta((x - 1)*(x - 2)*(x - 3)).expand(diracdelta=True) == (
            DiracDelta(x - 3)/2 + DiracDelta(x - 2) + DiracDelta(x - 1)/2)

    raises(ArgumentIndexError, lambda: DiracDelta(x).fdiff(2))
    raises(ValueError, lambda: DiracDelta(x, -1))
    raises(ValueError, lambda: DiracDelta(I))
    raises(ValueError, lambda: DiracDelta(2 + 3*I))


def test_heaviside():
    assert Heaviside(-5) == 0
    assert Heaviside(1) == 1
    assert Heaviside(0) == S.Half

    assert Heaviside(0, x) == x
    assert unchanged(Heaviside,x, nan)
    assert Heaviside(0, nan) == nan

    h0  = Heaviside(x, 0)
    h12 = Heaviside(x, S.Half)
    h1  = Heaviside(x, 1)

    assert h0.args == h0.pargs == (x, 0)
    assert h1.args == h1.pargs == (x, 1)
    assert h12.args == (x, S.Half)
    assert h12.pargs == (x,) # default 1/2 suppressed

    assert adjoint(Heaviside(x)) == Heaviside(x)
    assert adjoint(Heaviside(x - y)) == Heaviside(x - y)
    assert conjugate(Heaviside(x)) == Heaviside(x)
    assert conjugate(Heaviside(x - y)) == Heaviside(x - y)
    assert transpose(Heaviside(x)) == Heaviside(x)
    assert transpose(Heaviside(x - y)) == Heaviside(x - y)

    assert Heaviside(x).diff(x) == DiracDelta(x)
    assert Heaviside(x + I).is_Function is True
    assert Heaviside(I*x).is_Function is True

    raises(ArgumentIndexError, lambda: Heaviside(x).fdiff(2))
    raises(ValueError, lambda: Heaviside(I))
    raises(ValueError, lambda: Heaviside(2 + 3*I))


def test_rewrite():
    x, y = Symbol('x', real=True), Symbol('y')
    assert Heaviside(x).rewrite(Piecewise) == (
        Piecewise((0, x < 0), (Heaviside(0), Eq(x, 0)), (1, True)))
    assert Heaviside(y).rewrite(Piecewise) == (
        Piecewise((0, y < 0), (Heaviside(0), Eq(y, 0)), (1, True)))
    assert Heaviside(x, y).rewrite(Piecewise) == (
        Piecewise((0, x < 0), (y, Eq(x, 0)), (1, True)))
    assert Heaviside(x, 0).rewrite(Piecewise) == (
        Piecewise((0, x <= 0), (1, True)))
    assert Heaviside(x, 1).rewrite(Piecewise) == (
        Piecewise((0, x < 0), (1, True)))
    assert Heaviside(x, nan).rewrite(Piecewise) == (
        Piecewise((0, x < 0), (nan, Eq(x, 0)), (1, True)))

    assert Heaviside(x).rewrite(sign) == \
        Heaviside(x, H0=Heaviside(0)).rewrite(sign) == \
        Piecewise(
            (sign(x)/2 + S(1)/2, Eq(Heaviside(0), S(1)/2)),
            (Piecewise(
                (sign(x)/2 + S(1)/2, Ne(x, 0)), (Heaviside(0), True)), True)
        )

    assert Heaviside(y).rewrite(sign) == Heaviside(y)
    assert Heaviside(x, S.Half).rewrite(sign) == (sign(x)+1)/2
    assert Heaviside(x, y).rewrite(sign) == \
        Piecewise(
            (sign(x)/2 + S(1)/2, Eq(y, S(1)/2)),
            (Piecewise(
                (sign(x)/2 + S(1)/2, Ne(x, 0)), (y, True)), True)
        )

    assert DiracDelta(y).rewrite(Piecewise) == Piecewise((DiracDelta(0), Eq(y, 0)), (0, True))
    assert DiracDelta(y, 1).rewrite(Piecewise) == DiracDelta(y, 1)
    assert DiracDelta(x - 5).rewrite(Piecewise) == (
        Piecewise((DiracDelta(0), Eq(x - 5, 0)), (0, True)))

    assert (x*DiracDelta(x - 10)).rewrite(SingularityFunction) == x*SingularityFunction(x, 10, -1)
    assert 5*x*y*DiracDelta(y, 1).rewrite(SingularityFunction) == 5*x*y*SingularityFunction(y, 0, -2)
    assert DiracDelta(0).rewrite(SingularityFunction) == SingularityFunction(0, 0, -1)
    assert DiracDelta(0, 1).rewrite(SingularityFunction) == SingularityFunction(0, 0, -2)

    assert Heaviside(x).rewrite(SingularityFunction) == SingularityFunction(x, 0, 0)
    assert 5*x*y*Heaviside(y + 1).rewrite(SingularityFunction) == 5*x*y*SingularityFunction(y, -1, 0)
    assert ((x - 3)**3*Heaviside(x - 3)).rewrite(SingularityFunction) == (x - 3)**3*SingularityFunction(x, 3, 0)
    assert Heaviside(0).rewrite(SingularityFunction) == S.Half
