from sympy.functions.elementary.miscellaneous import sqrt
from sympy.core import S, Symbol, symbols, I, Rational
from sympy.discrete import (fft, ifft, ntt, intt, fwht, ifwht,
    mobius_transform, inverse_mobius_transform)
from sympy.testing.pytest import raises


def test_fft_ifft():
    assert all(tf(ls) == ls for tf in (fft, ifft)
                            for ls in ([], [Rational(5, 3)]))

    ls = list(range(6))
    fls = [15, -7*sqrt(2)/2 - 4 - sqrt(2)*I/2 + 2*I, 2 + 3*I,
             -4 + 7*sqrt(2)/2 - 2*I - sqrt(2)*I/2, -3,
             -4 + 7*sqrt(2)/2 + sqrt(2)*I/2 + 2*I,
              2 - 3*I, -7*sqrt(2)/2 - 4 - 2*I + sqrt(2)*I/2]

    assert fft(ls) == fls
    assert ifft(fls) == ls + [S.Zero]*2

    ls = [1 + 2*I, 3 + 4*I, 5 + 6*I]
    ifls = [Rational(9, 4) + 3*I, I*Rational(-7, 4), Rational(3, 4) + I, -2 - I/4]

    assert ifft(ls) == ifls
    assert fft(ifls) == ls + [S.Zero]

    x = Symbol('x', real=True)
    raises(TypeError, lambda: fft(x))
    raises(ValueError, lambda: ifft([x, 2*x, 3*x**2, 4*x**3]))


def test_ntt_intt():
    # prime moduli of the form (m*2**k + 1), sequence length
    # should be a divisor of 2**k
    p = 7*17*2**23 + 1
    q = 2*500000003 + 1 # only for sequences of length 1 or 2
    r = 2*3*5*7 # composite modulus

    assert all(tf(ls, p) == ls for tf in (ntt, intt)
                                for ls in ([], [5]))

    ls = list(range(6))
    nls = [15, 801133602, 738493201, 334102277, 998244350, 849020224,
            259751156, 12232587]

    assert ntt(ls, p) == nls
    assert intt(nls, p) == ls + [0]*2

    ls = [1 + 2*I, 3 + 4*I, 5 + 6*I]
    x = Symbol('x', integer=True)

    raises(TypeError, lambda: ntt(x, p))
    raises(ValueError, lambda: intt([x, 2*x, 3*x**2, 4*x**3], p))
    raises(ValueError, lambda: intt(ls, p))
    raises(ValueError, lambda: ntt([1.2, 2.1, 3.5], p))
    raises(ValueError, lambda: ntt([3, 5, 6], q))
    raises(ValueError, lambda: ntt([4, 5, 7], r))
    raises(ValueError, lambda: ntt([1.0, 2.0, 3.0], p))


def test_fwht_ifwht():
    assert all(tf(ls) == ls for tf in (fwht, ifwht) \
                        for ls in ([], [Rational(7, 4)]))

    ls = [213, 321, 43235, 5325, 312, 53]
    fls = [49459, 38061, -47661, -37759, 48729, 37543, -48391, -38277]

    assert fwht(ls) == fls
    assert ifwht(fls) == ls + [S.Zero]*2

    ls = [S.Half + 2*I, Rational(3, 7) + 4*I, Rational(5, 6) + 6*I, Rational(7, 3), Rational(9, 4)]
    ifls = [Rational(533, 672) + I*3/2, Rational(23, 224) + I/2, Rational(1, 672), Rational(107, 224) - I,
        Rational(155, 672) + I*3/2, Rational(-103, 224) + I/2, Rational(-377, 672), Rational(-19, 224) - I]

    assert ifwht(ls) == ifls
    assert fwht(ifls) == ls + [S.Zero]*3

    x, y = symbols('x y')

    raises(TypeError, lambda: fwht(x))

    ls = [x, 2*x, 3*x**2, 4*x**3]
    ifls = [x**3 + 3*x**2/4 + x*Rational(3, 4),
        -x**3 + 3*x**2/4 - x/4,
        -x**3 - 3*x**2/4 + x*Rational(3, 4),
        x**3 - 3*x**2/4 - x/4]

    assert ifwht(ls) == ifls
    assert fwht(ifls) == ls

    ls = [x, y, x**2, y**2, x*y]
    fls = [x**2 + x*y + x + y**2 + y,
        x**2 + x*y + x - y**2 - y,
        -x**2 + x*y + x - y**2 + y,
        -x**2 + x*y + x + y**2 - y,
        x**2 - x*y + x + y**2 + y,
        x**2 - x*y + x - y**2 - y,
        -x**2 - x*y + x - y**2 + y,
        -x**2 - x*y + x + y**2 - y]

    assert fwht(ls) == fls
    assert ifwht(fls) == ls + [S.Zero]*3

    ls = list(range(6))

    assert fwht(ls) == [x*8 for x in ifwht(ls)]


def test_mobius_transform():
    assert all(tf(ls, subset=subset) == ls
                for ls in ([], [Rational(7, 4)]) for subset in (True, False)
                for tf in (mobius_transform, inverse_mobius_transform))

    w, x, y, z = symbols('w x y z')

    assert mobius_transform([x, y]) == [x, x + y]
    assert inverse_mobius_transform([x, x + y]) == [x, y]
    assert mobius_transform([x, y], subset=False) == [x + y, y]
    assert inverse_mobius_transform([x + y, y], subset=False) == [x, y]

    assert mobius_transform([w, x, y, z]) == [w, w + x, w + y, w + x + y + z]
    assert inverse_mobius_transform([w, w + x, w + y, w + x + y + z]) == \
            [w, x, y, z]
    assert mobius_transform([w, x, y, z], subset=False) == \
            [w + x + y + z, x + z, y + z, z]
    assert inverse_mobius_transform([w + x + y + z, x + z, y + z, z], subset=False) == \
            [w, x, y, z]

    ls = [Rational(2, 3), Rational(6, 7), Rational(5, 8), 9, Rational(5, 3) + 7*I]
    mls = [Rational(2, 3), Rational(32, 21), Rational(31, 24), Rational(1873, 168),
            Rational(7, 3) + 7*I, Rational(67, 21) + 7*I, Rational(71, 24) + 7*I,
            Rational(2153, 168) + 7*I]

    assert mobius_transform(ls) == mls
    assert inverse_mobius_transform(mls) == ls + [S.Zero]*3

    mls = [Rational(2153, 168) + 7*I, Rational(69, 7), Rational(77, 8), 9, Rational(5, 3) + 7*I, 0, 0, 0]

    assert mobius_transform(ls, subset=False) == mls
    assert inverse_mobius_transform(mls, subset=False) == ls + [S.Zero]*3

    ls = ls[:-1]
    mls = [Rational(2, 3), Rational(32, 21), Rational(31, 24), Rational(1873, 168)]

    assert mobius_transform(ls) == mls
    assert inverse_mobius_transform(mls) == ls

    mls = [Rational(1873, 168), Rational(69, 7), Rational(77, 8), 9]

    assert mobius_transform(ls, subset=False) == mls
    assert inverse_mobius_transform(mls, subset=False) == ls

    raises(TypeError, lambda: mobius_transform(x, subset=True))
    raises(TypeError, lambda: inverse_mobius_transform(y, subset=False))
