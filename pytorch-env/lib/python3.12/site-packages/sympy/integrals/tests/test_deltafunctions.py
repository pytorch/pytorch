from sympy.core.function import Function
from sympy.core.numbers import (Rational, pi)
from sympy.core.singleton import S
from sympy.core.symbol import symbols
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.functions.special.delta_functions import (DiracDelta, Heaviside)
from sympy.integrals.deltafunctions import change_mul, deltaintegrate

f = Function("f")
x_1, x_2, x, y, z = symbols("x_1 x_2 x y z")


def test_change_mul():
    assert change_mul(x, x) == (None, None)
    assert change_mul(x*y, x) == (None, None)
    assert change_mul(x*y*DiracDelta(x), x) == (DiracDelta(x), x*y)
    assert change_mul(x*y*DiracDelta(x)*DiracDelta(y), x) == \
        (DiracDelta(x), x*y*DiracDelta(y))
    assert change_mul(DiracDelta(x)**2, x) == \
        (DiracDelta(x), DiracDelta(x))
    assert change_mul(y*DiracDelta(x)**2, x) == \
        (DiracDelta(x), y*DiracDelta(x))


def test_deltaintegrate():
    assert deltaintegrate(x, x) is None
    assert deltaintegrate(x + DiracDelta(x), x) is None
    assert deltaintegrate(DiracDelta(x, 0), x) == Heaviside(x)
    for n in range(10):
        assert deltaintegrate(DiracDelta(x, n + 1), x) == DiracDelta(x, n)
    assert deltaintegrate(DiracDelta(x), x) == Heaviside(x)
    assert deltaintegrate(DiracDelta(-x), x) == Heaviside(x)
    assert deltaintegrate(DiracDelta(x - y), x) == Heaviside(x - y)
    assert deltaintegrate(DiracDelta(y - x), x) == Heaviside(x - y)

    assert deltaintegrate(x*DiracDelta(x), x) == 0
    assert deltaintegrate((x - y)*DiracDelta(x - y), x) == 0

    assert deltaintegrate(DiracDelta(x)**2, x) == DiracDelta(0)*Heaviside(x)
    assert deltaintegrate(y*DiracDelta(x)**2, x) == \
        y*DiracDelta(0)*Heaviside(x)
    assert deltaintegrate(DiracDelta(x, 1), x) == DiracDelta(x, 0)
    assert deltaintegrate(y*DiracDelta(x, 1), x) == y*DiracDelta(x, 0)
    assert deltaintegrate(DiracDelta(x, 1)**2, x) == -DiracDelta(0, 2)*Heaviside(x)
    assert deltaintegrate(y*DiracDelta(x, 1)**2, x) == -y*DiracDelta(0, 2)*Heaviside(x)


    assert deltaintegrate(DiracDelta(x) * f(x), x) == f(0) * Heaviside(x)
    assert deltaintegrate(DiracDelta(-x) * f(x), x) == f(0) * Heaviside(x)
    assert deltaintegrate(DiracDelta(x - 1) * f(x), x) == f(1) * Heaviside(x - 1)
    assert deltaintegrate(DiracDelta(1 - x) * f(x), x) == f(1) * Heaviside(x - 1)
    assert deltaintegrate(DiracDelta(x**2 + x - 2), x) == \
        Heaviside(x - 1)/3 + Heaviside(x + 2)/3

    p = cos(x)*(DiracDelta(x) + DiracDelta(x**2 - 1))*sin(x)*(x - pi)
    assert deltaintegrate(p, x) - (-pi*(cos(1)*Heaviside(-1 + x)*sin(1)/2 - \
        cos(1)*Heaviside(1 + x)*sin(1)/2) + \
        cos(1)*Heaviside(1 + x)*sin(1)/2 + \
        cos(1)*Heaviside(-1 + x)*sin(1)/2) == 0

    p = x_2*DiracDelta(x - x_2)*DiracDelta(x_2 - x_1)
    assert deltaintegrate(p, x_2) == x*DiracDelta(x - x_1)*Heaviside(x_2 - x)

    p = x*y**2*z*DiracDelta(y - x)*DiracDelta(y - z)*DiracDelta(x - z)
    assert deltaintegrate(p, y) == x**3*z*DiracDelta(x - z)**2*Heaviside(y - x)
    assert deltaintegrate((x + 1)*DiracDelta(2*x), x) == S.Half * Heaviside(x)
    assert deltaintegrate((x + 1)*DiracDelta(x*Rational(2, 3) + Rational(4, 9)), x) == \
        S.Half * Heaviside(x + Rational(2, 3))

    a, b, c = symbols('a b c', commutative=False)
    assert deltaintegrate(DiracDelta(x - y)*f(x - b)*f(x - a), x) == \
        f(y - b)*f(y - a)*Heaviside(x - y)

    p = f(x - a)*DiracDelta(x - y)*f(x - c)*f(x - b)
    assert deltaintegrate(p, x) == f(y - a)*f(y - c)*f(y - b)*Heaviside(x - y)

    p = DiracDelta(x - z)*f(x - b)*f(x - a)*DiracDelta(x - y)
    assert deltaintegrate(p, x) == DiracDelta(y - z)*f(y - b)*f(y - a) * \
        Heaviside(x - y)
