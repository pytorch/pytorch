from sympy.core.add import Add
from sympy.core.numbers import (Rational, oo, pi)
from sympy.core.singleton import S
from sympy.core.symbol import symbols
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.piecewise import Piecewise
from sympy.functions.elementary.trigonometric import (cos, sin, sinc, tan)
from sympy.series.fourier import fourier_series
from sympy.series.fourier import FourierSeries
from sympy.testing.pytest import raises
from functools import lru_cache

x, y, z = symbols('x y z')

# Don't declare these during import because they are slow
@lru_cache()
def _get_examples():
    fo = fourier_series(x, (x, -pi, pi))
    fe = fourier_series(x**2, (-pi, pi))
    fp = fourier_series(Piecewise((0, x < 0), (pi, True)), (x, -pi, pi))
    return fo, fe, fp


def test_FourierSeries():
    fo, fe, fp = _get_examples()

    assert fourier_series(1, (-pi, pi)) == 1
    assert (Piecewise((0, x < 0), (pi, True)).
            fourier_series((x, -pi, pi)).truncate()) == fp.truncate()
    assert isinstance(fo, FourierSeries)
    assert fo.function == x
    assert fo.x == x
    assert fo.period == (-pi, pi)

    assert fo.term(3) == 2*sin(3*x) / 3
    assert fe.term(3) == -4*cos(3*x) / 9
    assert fp.term(3) == 2*sin(3*x) / 3

    assert fo.as_leading_term(x) == 2*sin(x)
    assert fe.as_leading_term(x) == pi**2 / 3
    assert fp.as_leading_term(x) == pi / 2

    assert fo.truncate() == 2*sin(x) - sin(2*x) + (2*sin(3*x) / 3)
    assert fe.truncate() == -4*cos(x) + cos(2*x) + pi**2 / 3
    assert fp.truncate() == 2*sin(x) + (2*sin(3*x) / 3) + pi / 2

    fot = fo.truncate(n=None)
    s = [0, 2*sin(x), -sin(2*x)]
    for i, t in enumerate(fot):
        if i == 3:
            break
        assert s[i] == t

    def _check_iter(f, i):
        for ind, t in enumerate(f):
            assert t == f[ind]  # noqa: PLR1736
            if ind == i:
                break

    _check_iter(fo, 3)
    _check_iter(fe, 3)
    _check_iter(fp, 3)

    assert fo.subs(x, x**2) == fo

    raises(ValueError, lambda: fourier_series(x, (0, 1, 2)))
    raises(ValueError, lambda: fourier_series(x, (x, 0, oo)))
    raises(ValueError, lambda: fourier_series(x*y, (0, oo)))


def test_FourierSeries_2():
    p = Piecewise((0, x < 0), (x, True))
    f = fourier_series(p, (x, -2, 2))

    assert f.term(3) == (2*sin(3*pi*x / 2) / (3*pi) -
                         4*cos(3*pi*x / 2) / (9*pi**2))
    assert f.truncate() == (2*sin(pi*x / 2) / pi - sin(pi*x) / pi -
                            4*cos(pi*x / 2) / pi**2 + S.Half)


def test_square_wave():
    """Test if fourier_series approximates discontinuous function correctly."""
    square_wave = Piecewise((1, x < pi), (-1, True))
    s = fourier_series(square_wave, (x, 0, 2*pi))

    assert s.truncate(3) == 4 / pi * sin(x) + 4 / (3 * pi) * sin(3 * x) + \
        4 / (5 * pi) * sin(5 * x)
    assert s.sigma_approximation(4) == 4 / pi * sin(x) * sinc(pi / 4) + \
        4 / (3 * pi) * sin(3 * x) * sinc(3 * pi / 4)


def test_sawtooth_wave():
    s = fourier_series(x, (x, 0, pi))
    assert s.truncate(4) == \
        pi/2 - sin(2*x) - sin(4*x)/2 - sin(6*x)/3
    s = fourier_series(x, (x, 0, 1))
    assert s.truncate(4) == \
        S.Half - sin(2*pi*x)/pi - sin(4*pi*x)/(2*pi) - sin(6*pi*x)/(3*pi)


def test_FourierSeries__operations():
    fo, fe, fp = _get_examples()

    fes = fe.scale(-1).shift(pi**2)
    assert fes.truncate() == 4*cos(x) - cos(2*x) + 2*pi**2 / 3

    assert fp.shift(-pi/2).truncate() == (2*sin(x) + (2*sin(3*x) / 3) +
                                          (2*sin(5*x) / 5))

    fos = fo.scale(3)
    assert fos.truncate() == 6*sin(x) - 3*sin(2*x) + 2*sin(3*x)

    fx = fe.scalex(2).shiftx(1)
    assert fx.truncate() == -4*cos(2*x + 2) + cos(4*x + 4) + pi**2 / 3

    fl = fe.scalex(3).shift(-pi).scalex(2).shiftx(1).scale(4)
    assert fl.truncate() == (-16*cos(6*x + 6) + 4*cos(12*x + 12) -
                             4*pi + 4*pi**2 / 3)

    raises(ValueError, lambda: fo.shift(x))
    raises(ValueError, lambda: fo.shiftx(sin(x)))
    raises(ValueError, lambda: fo.scale(x*y))
    raises(ValueError, lambda: fo.scalex(x**2))


def test_FourierSeries__neg():
    fo, fe, fp = _get_examples()

    assert (-fo).truncate() == -2*sin(x) + sin(2*x) - (2*sin(3*x) / 3)
    assert (-fe).truncate() == +4*cos(x) - cos(2*x) - pi**2 / 3


def test_FourierSeries__add__sub():
    fo, fe, fp = _get_examples()

    assert fo + fo == fo.scale(2)
    assert fo - fo == 0
    assert -fe - fe == fe.scale(-2)

    assert (fo + fe).truncate() == 2*sin(x) - sin(2*x) - 4*cos(x) + cos(2*x) \
        + pi**2 / 3
    assert (fo - fe).truncate() == 2*sin(x) - sin(2*x) + 4*cos(x) - cos(2*x) \
        - pi**2 / 3

    assert isinstance(fo + 1, Add)

    raises(ValueError, lambda: fo + fourier_series(x, (x, 0, 2)))


def test_FourierSeries_finite():

    assert fourier_series(sin(x)).truncate(1) == sin(x)
    # assert type(fourier_series(sin(x)*log(x))).truncate() == FourierSeries
    # assert type(fourier_series(sin(x**2+6))).truncate() == FourierSeries
    assert fourier_series(sin(x)*log(y)*exp(z),(x,pi,-pi)).truncate() == sin(x)*log(y)*exp(z)
    assert fourier_series(sin(x)**6).truncate(oo) == -15*cos(2*x)/32 + 3*cos(4*x)/16 - cos(6*x)/32 \
           + Rational(5, 16)
    assert fourier_series(sin(x) ** 6).truncate() == -15 * cos(2 * x) / 32 + 3 * cos(4 * x) / 16 \
           + Rational(5, 16)
    assert fourier_series(sin(4*x+3) + cos(3*x+4)).truncate(oo) ==  -sin(4)*sin(3*x) + sin(4*x)*cos(3) \
           + cos(4)*cos(3*x) + sin(3)*cos(4*x)
    assert fourier_series(sin(x)+cos(x)*tan(x)).truncate(oo) == 2*sin(x)
    assert fourier_series(cos(pi*x), (x, -1, 1)).truncate(oo) == cos(pi*x)
    assert fourier_series(cos(3*pi*x + 4) - sin(4*pi*x)*log(pi*y), (x, -1, 1)).truncate(oo) == -log(pi*y)*sin(4*pi*x)\
           - sin(4)*sin(3*pi*x) + cos(4)*cos(3*pi*x)
