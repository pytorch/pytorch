"""Tests for tools for constructing domains for expressions. """

from sympy.testing.pytest import tooslow

from sympy.polys.constructor import construct_domain
from sympy.polys.domains import ZZ, QQ, ZZ_I, QQ_I, RR, CC, EX
from sympy.polys.domains.realfield import RealField
from sympy.polys.domains.complexfield import ComplexField

from sympy.core import (Catalan, GoldenRatio)
from sympy.core.numbers import (E, Float, I, Rational, pi)
from sympy.core.singleton import S
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import sin
from sympy import rootof

from sympy.abc import x, y


def test_construct_domain():

    assert construct_domain([1, 2, 3]) == (ZZ, [ZZ(1), ZZ(2), ZZ(3)])
    assert construct_domain([1, 2, 3], field=True) == (QQ, [QQ(1), QQ(2), QQ(3)])

    assert construct_domain([S.One, S(2), S(3)]) == (ZZ, [ZZ(1), ZZ(2), ZZ(3)])
    assert construct_domain([S.One, S(2), S(3)], field=True) == (QQ, [QQ(1), QQ(2), QQ(3)])

    assert construct_domain([S.Half, S(2)]) == (QQ, [QQ(1, 2), QQ(2)])
    result = construct_domain([3.14, 1, S.Half])
    assert isinstance(result[0], RealField)
    assert result[1] == [RR(3.14), RR(1.0), RR(0.5)]

    result = construct_domain([3.14, I, S.Half])
    assert isinstance(result[0], ComplexField)
    assert result[1] == [CC(3.14), CC(1.0j), CC(0.5)]

    assert construct_domain([1.0+I]) == (CC, [CC(1.0, 1.0)])
    assert construct_domain([2.0+3.0*I]) == (CC, [CC(2.0, 3.0)])

    assert construct_domain([1, I]) == (ZZ_I, [ZZ_I(1, 0), ZZ_I(0, 1)])
    assert construct_domain([1, I/2]) == (QQ_I, [QQ_I(1, 0), QQ_I(0, S.Half)])

    assert construct_domain([3.14, sqrt(2)], extension=None) == (EX, [EX(3.14), EX(sqrt(2))])
    assert construct_domain([3.14, sqrt(2)], extension=True) == (EX, [EX(3.14), EX(sqrt(2))])

    assert construct_domain([1, sqrt(2)], extension=None) == (EX, [EX(1), EX(sqrt(2))])

    assert construct_domain([x, sqrt(x)]) == (EX, [EX(x), EX(sqrt(x))])
    assert construct_domain([x, sqrt(x), sqrt(y)]) == (EX, [EX(x), EX(sqrt(x)), EX(sqrt(y))])

    alg = QQ.algebraic_field(sqrt(2))

    assert construct_domain([7, S.Half, sqrt(2)], extension=True) == \
        (alg, [alg.convert(7), alg.convert(S.Half), alg.convert(sqrt(2))])

    alg = QQ.algebraic_field(sqrt(2) + sqrt(3))

    assert construct_domain([7, sqrt(2), sqrt(3)], extension=True) == \
        (alg, [alg.convert(7), alg.convert(sqrt(2)), alg.convert(sqrt(3))])

    dom = ZZ[x]

    assert construct_domain([2*x, 3]) == \
        (dom, [dom.convert(2*x), dom.convert(3)])

    dom = ZZ[x, y]

    assert construct_domain([2*x, 3*y]) == \
        (dom, [dom.convert(2*x), dom.convert(3*y)])

    dom = QQ[x]

    assert construct_domain([x/2, 3]) == \
        (dom, [dom.convert(x/2), dom.convert(3)])

    dom = QQ[x, y]

    assert construct_domain([x/2, 3*y]) == \
        (dom, [dom.convert(x/2), dom.convert(3*y)])

    dom = ZZ_I[x]

    assert construct_domain([2*x, I]) == \
        (dom, [dom.convert(2*x), dom.convert(I)])

    dom = ZZ_I[x, y]

    assert construct_domain([2*x, I*y]) == \
        (dom, [dom.convert(2*x), dom.convert(I*y)])

    dom = QQ_I[x]

    assert construct_domain([x/2, I]) == \
        (dom, [dom.convert(x/2), dom.convert(I)])

    dom = QQ_I[x, y]

    assert construct_domain([x/2, I*y]) == \
        (dom, [dom.convert(x/2), dom.convert(I*y)])

    dom = RR[x]

    assert construct_domain([x/2, 3.5]) == \
        (dom, [dom.convert(x/2), dom.convert(3.5)])

    dom = RR[x, y]

    assert construct_domain([x/2, 3.5*y]) == \
        (dom, [dom.convert(x/2), dom.convert(3.5*y)])

    dom = CC[x]

    assert construct_domain([I*x/2, 3.5]) == \
        (dom, [dom.convert(I*x/2), dom.convert(3.5)])

    dom = CC[x, y]

    assert construct_domain([I*x/2, 3.5*y]) == \
        (dom, [dom.convert(I*x/2), dom.convert(3.5*y)])

    dom = CC[x]

    assert construct_domain([x/2, I*3.5]) == \
        (dom, [dom.convert(x/2), dom.convert(I*3.5)])

    dom = CC[x, y]

    assert construct_domain([x/2, I*3.5*y]) == \
        (dom, [dom.convert(x/2), dom.convert(I*3.5*y)])

    dom = ZZ.frac_field(x)

    assert construct_domain([2/x, 3]) == \
        (dom, [dom.convert(2/x), dom.convert(3)])

    dom = ZZ.frac_field(x, y)

    assert construct_domain([2/x, 3*y]) == \
        (dom, [dom.convert(2/x), dom.convert(3*y)])

    dom = RR.frac_field(x)

    assert construct_domain([2/x, 3.5]) == \
        (dom, [dom.convert(2/x), dom.convert(3.5)])

    dom = RR.frac_field(x, y)

    assert construct_domain([2/x, 3.5*y]) == \
        (dom, [dom.convert(2/x), dom.convert(3.5*y)])

    dom = RealField(prec=336)[x]

    assert construct_domain([pi.evalf(100)*x]) == \
        (dom, [dom.convert(pi.evalf(100)*x)])

    assert construct_domain(2) == (ZZ, ZZ(2))
    assert construct_domain(S(2)/3) == (QQ, QQ(2, 3))
    assert construct_domain(Rational(2, 3)) == (QQ, QQ(2, 3))

    assert construct_domain({}) == (ZZ, {})


def test_complex_exponential():
    w = exp(-I*2*pi/3, evaluate=False)
    alg = QQ.algebraic_field(w)
    assert construct_domain([w**2, w, 1], extension=True) == (
        alg,
        [alg.convert(w**2),
         alg.convert(w),
         alg.convert(1)]
    )


def test_rootof():
    r1 = rootof(x**3 + x + 1, 0)
    r2 = rootof(x**3 + x + 1, 1)
    K1 = QQ.algebraic_field(r1)
    K2 = QQ.algebraic_field(r2)
    assert construct_domain([r1]) == (EX, [EX(r1)])
    assert construct_domain([r2]) == (EX, [EX(r2)])
    assert construct_domain([r1, r2]) == (EX, [EX(r1), EX(r2)])

    assert construct_domain([r1], extension=True) == (
            K1, [K1.from_sympy(r1)])
    assert construct_domain([r2], extension=True) == (
            K2, [K2.from_sympy(r2)])


@tooslow
def test_rootof_primitive_element():
    r1 = rootof(x**3 + x + 1, 0)
    r2 = rootof(x**3 + x + 1, 1)
    K12 = QQ.algebraic_field(r1 + r2)
    assert construct_domain([r1, r2], extension=True) == (
            K12, [K12.from_sympy(r1), K12.from_sympy(r2)])


def test_composite_option():
    assert construct_domain({(1,): sin(y)}, composite=False) == \
        (EX, {(1,): EX(sin(y))})

    assert construct_domain({(1,): y}, composite=False) == \
        (EX, {(1,): EX(y)})

    assert construct_domain({(1, 1): 1}, composite=False) == \
        (ZZ, {(1, 1): 1})

    assert construct_domain({(1, 0): y}, composite=False) == \
        (EX, {(1, 0): EX(y)})


def test_precision():
    f1 = Float("1.01")
    f2 = Float("1.0000000000000000000001")
    for u in [1, 1e-2, 1e-6, 1e-13, 1e-14, 1e-16, 1e-20, 1e-100, 1e-300,
            f1, f2]:
        result = construct_domain([u])
        v = float(result[1][0])
        assert abs(u - v) / u < 1e-14  # Test relative accuracy

    result = construct_domain([f1])
    y = result[1][0]
    assert y-1 > 1e-50

    result = construct_domain([f2])
    y = result[1][0]
    assert y-1 > 1e-50


def test_issue_11538():
    for n in [E, pi, Catalan]:
        assert construct_domain(n)[0] == ZZ[n]
        assert construct_domain(x + n)[0] == ZZ[x, n]
    assert construct_domain(GoldenRatio)[0] == EX
    assert construct_domain(x + GoldenRatio)[0] == EX
