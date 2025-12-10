"""Tests for user-friendly public interface to polynomial functions. """

import pickle

from sympy.polys.polytools import (
    Poly, PurePoly, poly,
    parallel_poly_from_expr,
    degree, degree_list,
    total_degree,
    LC, LM, LT,
    pdiv, prem, pquo, pexquo,
    div, rem, quo, exquo,
    half_gcdex, gcdex, invert,
    subresultants,
    resultant, discriminant,
    terms_gcd, cofactors,
    gcd, gcd_list,
    lcm, lcm_list,
    trunc,
    monic, content, primitive,
    compose, decompose,
    sturm,
    gff_list, gff,
    sqf_norm, sqf_part, sqf_list, sqf,
    factor_list, factor,
    intervals, refine_root, count_roots,
    all_roots, real_roots, nroots, ground_roots,
    nth_power_roots_poly,
    cancel, reduced, groebner,
    GroebnerBasis, is_zero_dimensional,
    _torational_factor_list,
    to_rational_coeffs)

from sympy.polys.polyerrors import (
    MultivariatePolynomialError,
    ExactQuotientFailed,
    PolificationFailed,
    ComputationFailed,
    UnificationFailed,
    RefinementFailed,
    GeneratorsNeeded,
    GeneratorsError,
    PolynomialError,
    CoercionFailed,
    DomainError,
    OptionError,
    FlagError)

from sympy.polys.polyclasses import DMP

from sympy.polys.fields import field
from sympy.polys.domains import FF, ZZ, QQ, ZZ_I, QQ_I, RR, EX
from sympy.polys.domains.realfield import RealField
from sympy.polys.domains.complexfield import ComplexField
from sympy.polys.orderings import lex, grlex, grevlex

from sympy.combinatorics.galois import S4TransitiveSubgroups
from sympy.core.add import Add
from sympy.core.basic import _aresame
from sympy.core.containers import Tuple
from sympy.core.expr import Expr
from sympy.core.function import (Derivative, diff, expand)
from sympy.core.mul import _keep_coeff, Mul
from sympy.core.numbers import (Float, I, Integer, Rational, oo, pi)
from sympy.core.power import Pow
from sympy.core.relational import Eq
from sympy.core.singleton import S
from sympy.core.symbol import Symbol
from sympy.functions.elementary.complexes import (im, re)
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.hyperbolic import tanh
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.piecewise import Piecewise
from sympy.functions.elementary.trigonometric import sin
from sympy.matrices.dense import Matrix
from sympy.matrices.expressions.matexpr import MatrixSymbol
from sympy.polys.rootoftools import rootof
from sympy.simplify.simplify import signsimp
from sympy.utilities.iterables import iterable
from sympy.utilities.exceptions import SymPyDeprecationWarning

from sympy.testing.pytest import (
    raises, warns_deprecated_sympy, warns, tooslow, XFAIL
)

from sympy.abc import a, b, c, d, p, q, t, w, x, y, z


def _epsilon_eq(a, b):
    for u, v in zip(a, b):
        if abs(u - v) > 1e-10:
            return False
    return True


def _strict_eq(a, b):
    if type(a) == type(b):
        if iterable(a):
            if len(a) == len(b):
                return all(_strict_eq(c, d) for c, d in zip(a, b))
            else:
                return False
        else:
            return isinstance(a, Poly) and a.eq(b, strict=True)
    else:
        return False


def test_Poly_mixed_operations():
    p = Poly(x, x)
    with warns_deprecated_sympy():
        p * exp(x)
    with warns_deprecated_sympy():
        p + exp(x)
    with warns_deprecated_sympy():
        p - exp(x)


def test_Poly_from_dict():
    K = FF(3)

    assert Poly.from_dict(
        {0: 1, 1: 2}, gens=x, domain=K).rep == DMP([K(2), K(1)], K)
    assert Poly.from_dict(
        {0: 1, 1: 5}, gens=x, domain=K).rep == DMP([K(2), K(1)], K)

    assert Poly.from_dict(
        {(0,): 1, (1,): 2}, gens=x, domain=K).rep == DMP([K(2), K(1)], K)
    assert Poly.from_dict(
        {(0,): 1, (1,): 5}, gens=x, domain=K).rep == DMP([K(2), K(1)], K)

    assert Poly.from_dict({(0, 0): 1, (1, 1): 2}, gens=(
        x, y), domain=K).rep == DMP([[K(2), K(0)], [K(1)]], K)

    assert Poly.from_dict({0: 1, 1: 2}, gens=x).rep == DMP([ZZ(2), ZZ(1)], ZZ)
    assert Poly.from_dict(
        {0: 1, 1: 2}, gens=x, field=True).rep == DMP([QQ(2), QQ(1)], QQ)

    assert Poly.from_dict(
        {0: 1, 1: 2}, gens=x, domain=ZZ).rep == DMP([ZZ(2), ZZ(1)], ZZ)
    assert Poly.from_dict(
        {0: 1, 1: 2}, gens=x, domain=QQ).rep == DMP([QQ(2), QQ(1)], QQ)

    assert Poly.from_dict(
        {(0,): 1, (1,): 2}, gens=x).rep == DMP([ZZ(2), ZZ(1)], ZZ)
    assert Poly.from_dict(
        {(0,): 1, (1,): 2}, gens=x, field=True).rep == DMP([QQ(2), QQ(1)], QQ)

    assert Poly.from_dict(
        {(0,): 1, (1,): 2}, gens=x, domain=ZZ).rep == DMP([ZZ(2), ZZ(1)], ZZ)
    assert Poly.from_dict(
        {(0,): 1, (1,): 2}, gens=x, domain=QQ).rep == DMP([QQ(2), QQ(1)], QQ)

    assert Poly.from_dict({(1,): sin(y)}, gens=x, composite=False) == \
        Poly(sin(y)*x, x, domain='EX')
    assert Poly.from_dict({(1,): y}, gens=x, composite=False) == \
        Poly(y*x, x, domain='EX')
    assert Poly.from_dict({(1, 1): 1}, gens=(x, y), composite=False) == \
        Poly(x*y, x, y, domain='ZZ')
    assert Poly.from_dict({(1, 0): y}, gens=(x, z), composite=False) == \
        Poly(y*x, x, z, domain='EX')


def test_Poly_from_list():
    K = FF(3)

    assert Poly.from_list([2, 1], gens=x, domain=K).rep == DMP([K(2), K(1)], K)
    assert Poly.from_list([5, 1], gens=x, domain=K).rep == DMP([K(2), K(1)], K)

    assert Poly.from_list([2, 1], gens=x).rep == DMP([ZZ(2), ZZ(1)], ZZ)
    assert Poly.from_list([2, 1], gens=x, field=True).rep == DMP([QQ(2), QQ(1)], QQ)

    assert Poly.from_list([2, 1], gens=x, domain=ZZ).rep == DMP([ZZ(2), ZZ(1)], ZZ)
    assert Poly.from_list([2, 1], gens=x, domain=QQ).rep == DMP([QQ(2), QQ(1)], QQ)

    assert Poly.from_list([0, 1.0], gens=x).rep == DMP([RR(1.0)], RR)
    assert Poly.from_list([1.0, 0], gens=x).rep == DMP([RR(1.0), RR(0.0)], RR)

    raises(MultivariatePolynomialError, lambda: Poly.from_list([[]], gens=(x, y)))


def test_Poly_from_poly():
    f = Poly(x + 7, x, domain=ZZ)
    g = Poly(x + 2, x, modulus=3)
    h = Poly(x + y, x, y, domain=ZZ)

    K = FF(3)

    assert Poly.from_poly(f) == f
    assert Poly.from_poly(f, domain=K).rep == DMP([K(1), K(1)], K)
    assert Poly.from_poly(f, domain=ZZ).rep == DMP([ZZ(1), ZZ(7)], ZZ)
    assert Poly.from_poly(f, domain=QQ).rep == DMP([QQ(1), QQ(7)], QQ)

    assert Poly.from_poly(f, gens=x) == f
    assert Poly.from_poly(f, gens=x, domain=K).rep == DMP([K(1), K(1)], K)
    assert Poly.from_poly(f, gens=x, domain=ZZ).rep == DMP([ZZ(1), ZZ(7)], ZZ)
    assert Poly.from_poly(f, gens=x, domain=QQ).rep == DMP([QQ(1), QQ(7)], QQ)

    assert Poly.from_poly(f, gens=y) == Poly(x + 7, y, domain='ZZ[x]')
    raises(CoercionFailed, lambda: Poly.from_poly(f, gens=y, domain=K))
    raises(CoercionFailed, lambda: Poly.from_poly(f, gens=y, domain=ZZ))
    raises(CoercionFailed, lambda: Poly.from_poly(f, gens=y, domain=QQ))

    assert Poly.from_poly(f, gens=(x, y)) == Poly(x + 7, x, y, domain='ZZ')
    assert Poly.from_poly(
        f, gens=(x, y), domain=ZZ) == Poly(x + 7, x, y, domain='ZZ')
    assert Poly.from_poly(
        f, gens=(x, y), domain=QQ) == Poly(x + 7, x, y, domain='QQ')
    assert Poly.from_poly(
        f, gens=(x, y), modulus=3) == Poly(x + 7, x, y, domain='FF(3)')

    K = FF(2)

    assert Poly.from_poly(g) == g
    assert Poly.from_poly(g, domain=ZZ).rep == DMP([ZZ(1), ZZ(-1)], ZZ)
    raises(CoercionFailed, lambda: Poly.from_poly(g, domain=QQ))
    assert Poly.from_poly(g, domain=K).rep == DMP([K(1), K(0)], K)

    assert Poly.from_poly(g, gens=x) == g
    assert Poly.from_poly(g, gens=x, domain=ZZ).rep == DMP([ZZ(1), ZZ(-1)], ZZ)
    raises(CoercionFailed, lambda: Poly.from_poly(g, gens=x, domain=QQ))
    assert Poly.from_poly(g, gens=x, domain=K).rep == DMP([K(1), K(0)], K)

    K = FF(3)

    assert Poly.from_poly(h) == h
    assert Poly.from_poly(
        h, domain=ZZ).rep == DMP([[ZZ(1)], [ZZ(1), ZZ(0)]], ZZ)
    assert Poly.from_poly(
        h, domain=QQ).rep == DMP([[QQ(1)], [QQ(1), QQ(0)]], QQ)
    assert Poly.from_poly(h, domain=K).rep == DMP([[K(1)], [K(1), K(0)]], K)

    assert Poly.from_poly(h, gens=x) == Poly(x + y, x, domain=ZZ[y])
    raises(CoercionFailed, lambda: Poly.from_poly(h, gens=x, domain=ZZ))
    assert Poly.from_poly(
        h, gens=x, domain=ZZ[y]) == Poly(x + y, x, domain=ZZ[y])
    raises(CoercionFailed, lambda: Poly.from_poly(h, gens=x, domain=QQ))
    assert Poly.from_poly(
        h, gens=x, domain=QQ[y]) == Poly(x + y, x, domain=QQ[y])
    raises(CoercionFailed, lambda: Poly.from_poly(h, gens=x, modulus=3))

    assert Poly.from_poly(h, gens=y) == Poly(x + y, y, domain=ZZ[x])
    raises(CoercionFailed, lambda: Poly.from_poly(h, gens=y, domain=ZZ))
    assert Poly.from_poly(
        h, gens=y, domain=ZZ[x]) == Poly(x + y, y, domain=ZZ[x])
    raises(CoercionFailed, lambda: Poly.from_poly(h, gens=y, domain=QQ))
    assert Poly.from_poly(
        h, gens=y, domain=QQ[x]) == Poly(x + y, y, domain=QQ[x])
    raises(CoercionFailed, lambda: Poly.from_poly(h, gens=y, modulus=3))

    assert Poly.from_poly(h, gens=(x, y)) == h
    assert Poly.from_poly(
        h, gens=(x, y), domain=ZZ).rep == DMP([[ZZ(1)], [ZZ(1), ZZ(0)]], ZZ)
    assert Poly.from_poly(
        h, gens=(x, y), domain=QQ).rep == DMP([[QQ(1)], [QQ(1), QQ(0)]], QQ)
    assert Poly.from_poly(
        h, gens=(x, y), domain=K).rep == DMP([[K(1)], [K(1), K(0)]], K)

    assert Poly.from_poly(
        h, gens=(y, x)).rep == DMP([[ZZ(1)], [ZZ(1), ZZ(0)]], ZZ)
    assert Poly.from_poly(
        h, gens=(y, x), domain=ZZ).rep == DMP([[ZZ(1)], [ZZ(1), ZZ(0)]], ZZ)
    assert Poly.from_poly(
        h, gens=(y, x), domain=QQ).rep == DMP([[QQ(1)], [QQ(1), QQ(0)]], QQ)
    assert Poly.from_poly(
        h, gens=(y, x), domain=K).rep == DMP([[K(1)], [K(1), K(0)]], K)

    assert Poly.from_poly(
        h, gens=(x, y), field=True).rep == DMP([[QQ(1)], [QQ(1), QQ(0)]], QQ)
    assert Poly.from_poly(
        h, gens=(x, y), field=True).rep == DMP([[QQ(1)], [QQ(1), QQ(0)]], QQ)


def test_Poly_from_expr():
    raises(GeneratorsNeeded, lambda: Poly.from_expr(S.Zero))
    raises(GeneratorsNeeded, lambda: Poly.from_expr(S(7)))

    F3 = FF(3)

    assert Poly.from_expr(x + 5, domain=F3).rep == DMP([F3(1), F3(2)], F3)
    assert Poly.from_expr(y + 5, domain=F3).rep == DMP([F3(1), F3(2)], F3)

    assert Poly.from_expr(x + 5, x, domain=F3).rep == DMP([F3(1), F3(2)], F3)
    assert Poly.from_expr(y + 5, y, domain=F3).rep == DMP([F3(1), F3(2)], F3)

    assert Poly.from_expr(x + y, domain=F3).rep == DMP([[F3(1)], [F3(1), F3(0)]], F3)
    assert Poly.from_expr(x + y, x, y, domain=F3).rep == DMP([[F3(1)], [F3(1), F3(0)]], F3)

    assert Poly.from_expr(x + 5).rep == DMP([ZZ(1), ZZ(5)], ZZ)
    assert Poly.from_expr(y + 5).rep == DMP([ZZ(1), ZZ(5)], ZZ)

    assert Poly.from_expr(x + 5, x).rep == DMP([ZZ(1), ZZ(5)], ZZ)
    assert Poly.from_expr(y + 5, y).rep == DMP([ZZ(1), ZZ(5)], ZZ)

    assert Poly.from_expr(x + 5, domain=ZZ).rep == DMP([ZZ(1), ZZ(5)], ZZ)
    assert Poly.from_expr(y + 5, domain=ZZ).rep == DMP([ZZ(1), ZZ(5)], ZZ)

    assert Poly.from_expr(x + 5, x, domain=ZZ).rep == DMP([ZZ(1), ZZ(5)], ZZ)
    assert Poly.from_expr(y + 5, y, domain=ZZ).rep == DMP([ZZ(1), ZZ(5)], ZZ)

    assert Poly.from_expr(x + 5, x, y, domain=ZZ).rep == DMP([[ZZ(1)], [ZZ(5)]], ZZ)
    assert Poly.from_expr(y + 5, x, y, domain=ZZ).rep == DMP([[ZZ(1), ZZ(5)]], ZZ)


def test_Poly_rootof_extension():
    r1 = rootof(x**3 + x + 3, 0)
    r2 = rootof(x**3 + x + 3, 1)
    K1 = QQ.algebraic_field(r1)
    K2 = QQ.algebraic_field(r2)
    assert Poly(r1, y) == Poly(r1, y, domain=EX)
    assert Poly(r2, y) == Poly(r2, y, domain=EX)
    assert Poly(r1, y, extension=True) == Poly(r1, y, domain=K1)
    assert Poly(r2, y, extension=True) == Poly(r2, y, domain=K2)


@tooslow
def test_Poly_rootof_extension_primitive_element():
    r1 = rootof(x**3 + x + 3, 0)
    r2 = rootof(x**3 + x + 3, 1)
    K12 = QQ.algebraic_field(r1 + r2)
    assert Poly(r1*y + r2, y, extension=True) == Poly(r1*y + r2, y, domain=K12)


@XFAIL
def test_Poly_rootof_same_symbol_issue_26808():
    # XXX: This fails because r1 contains x.
    r1 = rootof(x**3 + x + 3, 0)
    K1 = QQ.algebraic_field(r1)
    assert Poly(r1, x) == Poly(r1, x, domain=EX)
    assert Poly(r1, x, extension=True) == Poly(r1, x, domain=K1)


def test_Poly_rootof_extension_to_sympy():
    # Verify that when primitive elements and RootOf are used, the expression
    # is not exploded on the way back to sympy.
    r1 = rootof(y**3 + y**2 - 1, 0)
    r2 = rootof(z**5 + z**2 - 1, 0)
    p = -x**5 + x**2 + x*r1 - r2 + 3*r1**2
    assert p.as_poly(x, extension=True).as_expr() == p


def test_poly_from_domain_element():
    dom = ZZ[x]
    assert Poly(dom(x+1), y, domain=dom).rep == DMP([dom(x+1)], dom)
    dom = dom.get_field()
    assert Poly(dom(x+1), y, domain=dom).rep == DMP([dom(x+1)], dom)

    dom = QQ[x]
    assert Poly(dom(x+1), y, domain=dom).rep == DMP([dom(x+1)], dom)
    dom = dom.get_field()
    assert Poly(dom(x+1), y, domain=dom).rep == DMP([dom(x+1)], dom)

    dom = ZZ.old_poly_ring(x)
    assert Poly(dom([ZZ(1), ZZ(1)]), y, domain=dom).rep == DMP([dom([ZZ(1), ZZ(1)])], dom)
    dom = dom.get_field()
    assert Poly(dom([ZZ(1), ZZ(1)]), y, domain=dom).rep == DMP([dom([ZZ(1), ZZ(1)])], dom)

    dom = QQ.old_poly_ring(x)
    assert Poly(dom([QQ(1), QQ(1)]), y, domain=dom).rep == DMP([dom([QQ(1), QQ(1)])], dom)
    dom = dom.get_field()
    assert Poly(dom([QQ(1), QQ(1)]), y, domain=dom).rep == DMP([dom([QQ(1), QQ(1)])], dom)

    dom = QQ.algebraic_field(I)
    assert Poly(dom([1, 1]), x, domain=dom).rep == DMP([dom([1, 1])], dom)


def test_Poly__new__():
    raises(GeneratorsError, lambda: Poly(x + 1, x, x))

    raises(GeneratorsError, lambda: Poly(x + y, x, y, domain=ZZ[x]))
    raises(GeneratorsError, lambda: Poly(x + y, x, y, domain=ZZ[y]))

    raises(OptionError, lambda: Poly(x, x, symmetric=True))
    raises(OptionError, lambda: Poly(x + 2, x, modulus=3, domain=QQ))

    raises(OptionError, lambda: Poly(x + 2, x, domain=ZZ, gaussian=True))
    raises(OptionError, lambda: Poly(x + 2, x, modulus=3, gaussian=True))

    raises(OptionError, lambda: Poly(x + 2, x, domain=ZZ, extension=[sqrt(3)]))
    raises(OptionError, lambda: Poly(x + 2, x, modulus=3, extension=[sqrt(3)]))

    raises(OptionError, lambda: Poly(x + 2, x, domain=ZZ, extension=True))
    raises(OptionError, lambda: Poly(x + 2, x, modulus=3, extension=True))

    raises(OptionError, lambda: Poly(x + 2, x, domain=ZZ, greedy=True))
    raises(OptionError, lambda: Poly(x + 2, x, domain=QQ, field=True))

    raises(OptionError, lambda: Poly(x + 2, x, domain=ZZ, greedy=False))
    raises(OptionError, lambda: Poly(x + 2, x, domain=QQ, field=False))

    raises(NotImplementedError, lambda: Poly(x + 1, x, modulus=3, order='grlex'))
    raises(NotImplementedError, lambda: Poly(x + 1, x, order='grlex'))

    raises(GeneratorsNeeded, lambda: Poly({1: 2, 0: 1}))
    raises(GeneratorsNeeded, lambda: Poly([2, 1]))
    raises(GeneratorsNeeded, lambda: Poly((2, 1)))

    raises(GeneratorsNeeded, lambda: Poly(1))

    assert Poly('x-x') == Poly(0, x)

    f = a*x**2 + b*x + c

    assert Poly({2: a, 1: b, 0: c}, x) == f
    assert Poly(iter([a, b, c]), x) == f
    assert Poly([a, b, c], x) == f
    assert Poly((a, b, c), x) == f

    f = Poly({}, x, y, z)

    assert f.gens == (x, y, z) and f.as_expr() == 0

    assert Poly(Poly(a*x + b*y, x, y), x) == Poly(a*x + b*y, x)

    assert Poly(3*x**2 + 2*x + 1, domain='ZZ').all_coeffs() == [3, 2, 1]
    assert Poly(3*x**2 + 2*x + 1, domain='QQ').all_coeffs() == [3, 2, 1]
    assert Poly(3*x**2 + 2*x + 1, domain='RR').all_coeffs() == [3.0, 2.0, 1.0]

    raises(CoercionFailed, lambda: Poly(3*x**2/5 + x*Rational(2, 5) + 1, domain='ZZ'))
    assert Poly(
        3*x**2/5 + x*Rational(2, 5) + 1, domain='QQ').all_coeffs() == [Rational(3, 5), Rational(2, 5), 1]
    assert _epsilon_eq(
        Poly(3*x**2/5 + x*Rational(2, 5) + 1, domain='RR').all_coeffs(), [0.6, 0.4, 1.0])

    assert Poly(3.0*x**2 + 2.0*x + 1, domain='ZZ').all_coeffs() == [3, 2, 1]
    assert Poly(3.0*x**2 + 2.0*x + 1, domain='QQ').all_coeffs() == [3, 2, 1]
    assert Poly(
        3.0*x**2 + 2.0*x + 1, domain='RR').all_coeffs() == [3.0, 2.0, 1.0]

    raises(CoercionFailed, lambda: Poly(3.1*x**2 + 2.1*x + 1, domain='ZZ'))
    assert Poly(3.1*x**2 + 2.1*x + 1, domain='QQ').all_coeffs() == [Rational(31, 10), Rational(21, 10), 1]
    assert Poly(3.1*x**2 + 2.1*x + 1, domain='RR').all_coeffs() == [3.1, 2.1, 1.0]

    assert Poly({(2, 1): 1, (1, 2): 2, (1, 1): 3}, x, y) == \
        Poly(x**2*y + 2*x*y**2 + 3*x*y, x, y)

    assert Poly(x**2 + 1, extension=I).get_domain() == QQ.algebraic_field(I)

    f = 3*x**5 - x**4 + x**3 - x** 2 + 65538

    assert Poly(f, x, modulus=65537, symmetric=True) == \
        Poly(3*x**5 - x**4 + x**3 - x** 2 + 1, x, modulus=65537,
             symmetric=True)
    assert Poly(f, x, modulus=65537, symmetric=False) == \
        Poly(3*x**5 + 65536*x**4 + x**3 + 65536*x** 2 + 1, x,
             modulus=65537, symmetric=False)

    N = 10**100
    assert Poly(-1, x, modulus=N, symmetric=False).as_expr() == N - 1

    assert isinstance(Poly(x**2 + x + 1.0).get_domain(), RealField)
    assert isinstance(Poly(x**2 + x + I + 1.0).get_domain(), ComplexField)


def test_Poly__args():
    assert Poly(x**2 + 1).args == (x**2 + 1, x)


def test_Poly__gens():
    assert Poly((x - p)*(x - q), x).gens == (x,)
    assert Poly((x - p)*(x - q), p).gens == (p,)
    assert Poly((x - p)*(x - q), q).gens == (q,)

    assert Poly((x - p)*(x - q), x, p).gens == (x, p)
    assert Poly((x - p)*(x - q), x, q).gens == (x, q)

    assert Poly((x - p)*(x - q), x, p, q).gens == (x, p, q)
    assert Poly((x - p)*(x - q), p, x, q).gens == (p, x, q)
    assert Poly((x - p)*(x - q), p, q, x).gens == (p, q, x)

    assert Poly((x - p)*(x - q)).gens == (x, p, q)

    assert Poly((x - p)*(x - q), sort='x > p > q').gens == (x, p, q)
    assert Poly((x - p)*(x - q), sort='p > x > q').gens == (p, x, q)
    assert Poly((x - p)*(x - q), sort='p > q > x').gens == (p, q, x)

    assert Poly((x - p)*(x - q), x, p, q, sort='p > q > x').gens == (x, p, q)

    assert Poly((x - p)*(x - q), wrt='x').gens == (x, p, q)
    assert Poly((x - p)*(x - q), wrt='p').gens == (p, x, q)
    assert Poly((x - p)*(x - q), wrt='q').gens == (q, x, p)

    assert Poly((x - p)*(x - q), wrt=x).gens == (x, p, q)
    assert Poly((x - p)*(x - q), wrt=p).gens == (p, x, q)
    assert Poly((x - p)*(x - q), wrt=q).gens == (q, x, p)

    assert Poly((x - p)*(x - q), x, p, q, wrt='p').gens == (x, p, q)

    assert Poly((x - p)*(x - q), wrt='p', sort='q > x').gens == (p, q, x)
    assert Poly((x - p)*(x - q), wrt='q', sort='p > x').gens == (q, p, x)


def test_Poly_zero():
    assert Poly(x).zero == Poly(0, x, domain=ZZ)
    assert Poly(x/2).zero == Poly(0, x, domain=QQ)


def test_Poly_one():
    assert Poly(x).one == Poly(1, x, domain=ZZ)
    assert Poly(x/2).one == Poly(1, x, domain=QQ)


def test_Poly__unify():
    raises(UnificationFailed, lambda: Poly(x)._unify(y))

    F3 = FF(3)

    assert Poly(x, x, modulus=3)._unify(Poly(y, y, modulus=3))[2:] == (
        DMP([[F3(1)], []], F3), DMP([[F3(1), F3(0)]], F3))
    raises(UnificationFailed, lambda: Poly(x, x, modulus=3)._unify(Poly(y, y, modulus=5)))

    raises(UnificationFailed, lambda: Poly(y, x, y)._unify(Poly(x, x, modulus=3)))
    raises(UnificationFailed, lambda: Poly(x, x, modulus=3)._unify(Poly(y, x, y)))

    assert Poly(x + 1, x)._unify(Poly(x + 2, x))[2:] ==\
        (DMP([ZZ(1), ZZ(1)], ZZ), DMP([ZZ(1), ZZ(2)], ZZ))
    assert Poly(x + 1, x, domain='QQ')._unify(Poly(x + 2, x))[2:] ==\
        (DMP([QQ(1), QQ(1)], QQ), DMP([QQ(1), QQ(2)], QQ))
    assert Poly(x + 1, x)._unify(Poly(x + 2, x, domain='QQ'))[2:] ==\
        (DMP([QQ(1), QQ(1)], QQ), DMP([QQ(1), QQ(2)], QQ))

    assert Poly(x + 1, x)._unify(Poly(x + 2, x, y))[2:] ==\
        (DMP([[ZZ(1)], [ZZ(1)]], ZZ), DMP([[ZZ(1)], [ZZ(2)]], ZZ))
    assert Poly(x + 1, x, domain='QQ')._unify(Poly(x + 2, x, y))[2:] ==\
        (DMP([[QQ(1)], [QQ(1)]], QQ), DMP([[QQ(1)], [QQ(2)]], QQ))
    assert Poly(x + 1, x)._unify(Poly(x + 2, x, y, domain='QQ'))[2:] ==\
        (DMP([[QQ(1)], [QQ(1)]], QQ), DMP([[QQ(1)], [QQ(2)]], QQ))

    assert Poly(x + 1, x, y)._unify(Poly(x + 2, x))[2:] ==\
        (DMP([[ZZ(1)], [ZZ(1)]], ZZ), DMP([[ZZ(1)], [ZZ(2)]], ZZ))
    assert Poly(x + 1, x, y, domain='QQ')._unify(Poly(x + 2, x))[2:] ==\
        (DMP([[QQ(1)], [QQ(1)]], QQ), DMP([[QQ(1)], [QQ(2)]], QQ))
    assert Poly(x + 1, x, y)._unify(Poly(x + 2, x, domain='QQ'))[2:] ==\
        (DMP([[QQ(1)], [QQ(1)]], QQ), DMP([[QQ(1)], [QQ(2)]], QQ))

    assert Poly(x + 1, x, y)._unify(Poly(x + 2, x, y))[2:] ==\
        (DMP([[ZZ(1)], [ZZ(1)]], ZZ), DMP([[ZZ(1)], [ZZ(2)]], ZZ))
    assert Poly(x + 1, x, y, domain='QQ')._unify(Poly(x + 2, x, y))[2:] ==\
        (DMP([[QQ(1)], [QQ(1)]], QQ), DMP([[QQ(1)], [QQ(2)]], QQ))
    assert Poly(x + 1, x, y)._unify(Poly(x + 2, x, y, domain='QQ'))[2:] ==\
        (DMP([[QQ(1)], [QQ(1)]], QQ), DMP([[QQ(1)], [QQ(2)]], QQ))

    assert Poly(x + 1, x)._unify(Poly(x + 2, y, x))[2:] ==\
        (DMP([[ZZ(1), ZZ(1)]], ZZ), DMP([[ZZ(1), ZZ(2)]], ZZ))
    assert Poly(x + 1, x, domain='QQ')._unify(Poly(x + 2, y, x))[2:] ==\
        (DMP([[QQ(1), QQ(1)]], QQ), DMP([[QQ(1), QQ(2)]], QQ))
    assert Poly(x + 1, x)._unify(Poly(x + 2, y, x, domain='QQ'))[2:] ==\
        (DMP([[QQ(1), QQ(1)]], QQ), DMP([[QQ(1), QQ(2)]], QQ))

    assert Poly(x + 1, y, x)._unify(Poly(x + 2, x))[2:] ==\
        (DMP([[ZZ(1), ZZ(1)]], ZZ), DMP([[ZZ(1), ZZ(2)]], ZZ))
    assert Poly(x + 1, y, x, domain='QQ')._unify(Poly(x + 2, x))[2:] ==\
        (DMP([[QQ(1), QQ(1)]], QQ), DMP([[QQ(1), QQ(2)]], QQ))
    assert Poly(x + 1, y, x)._unify(Poly(x + 2, x, domain='QQ'))[2:] ==\
        (DMP([[QQ(1), QQ(1)]], QQ), DMP([[QQ(1), QQ(2)]], QQ))

    assert Poly(x + 1, x, y)._unify(Poly(x + 2, y, x))[2:] ==\
        (DMP([[ZZ(1)], [ZZ(1)]], ZZ), DMP([[ZZ(1)], [ZZ(2)]], ZZ))
    assert Poly(x + 1, x, y, domain='QQ')._unify(Poly(x + 2, y, x))[2:] ==\
        (DMP([[QQ(1)], [QQ(1)]], QQ), DMP([[QQ(1)], [QQ(2)]], QQ))
    assert Poly(x + 1, x, y)._unify(Poly(x + 2, y, x, domain='QQ'))[2:] ==\
        (DMP([[QQ(1)], [QQ(1)]], QQ), DMP([[QQ(1)], [QQ(2)]], QQ))

    assert Poly(x + 1, y, x)._unify(Poly(x + 2, x, y))[2:] ==\
        (DMP([[ZZ(1), ZZ(1)]], ZZ), DMP([[ZZ(1), ZZ(2)]], ZZ))
    assert Poly(x + 1, y, x, domain='QQ')._unify(Poly(x + 2, x, y))[2:] ==\
        (DMP([[QQ(1), QQ(1)]], QQ), DMP([[QQ(1), QQ(2)]], QQ))
    assert Poly(x + 1, y, x)._unify(Poly(x + 2, x, y, domain='QQ'))[2:] ==\
        (DMP([[QQ(1), QQ(1)]], QQ), DMP([[QQ(1), QQ(2)]], QQ))

    assert Poly(x**2 + I, x, domain=ZZ_I).unify(Poly(x**2 + sqrt(2), x, extension=True)) == \
            (Poly(x**2 + I, x, domain='QQ<sqrt(2) + I>'), Poly(x**2 + sqrt(2), x, domain='QQ<sqrt(2) + I>'))

    F, A, B = field("a,b", ZZ)

    assert Poly(a*x, x, domain='ZZ[a]')._unify(Poly(a*b*x, x, domain='ZZ(a,b)'))[2:] == \
        (DMP([A, F(0)], F.to_domain()), DMP([A*B, F(0)], F.to_domain()))

    assert Poly(a*x, x, domain='ZZ(a)')._unify(Poly(a*b*x, x, domain='ZZ(a,b)'))[2:] == \
        (DMP([A, F(0)], F.to_domain()), DMP([A*B, F(0)], F.to_domain()))

    raises(CoercionFailed, lambda: Poly(Poly(x**2 + x**2*z, y, field=True), domain='ZZ(x)'))

    f = Poly(t**2 + t/3 + x, t, domain='QQ(x)')
    g = Poly(t**2 + t/3 + x, t, domain='QQ[x]')

    assert f._unify(g)[2:] == (f.rep, f.rep)


def test_Poly_free_symbols():
    assert Poly(x**2 + 1).free_symbols == {x}
    assert Poly(x**2 + y*z).free_symbols == {x, y, z}
    assert Poly(x**2 + y*z, x).free_symbols == {x, y, z}
    assert Poly(x**2 + sin(y*z)).free_symbols == {x, y, z}
    assert Poly(x**2 + sin(y*z), x).free_symbols == {x, y, z}
    assert Poly(x**2 + sin(y*z), x, domain=EX).free_symbols == {x, y, z}
    assert Poly(1 + x + x**2, x, y, z).free_symbols == {x}
    assert Poly(x + sin(y), z).free_symbols == {x, y}


def test_PurePoly_free_symbols():
    assert PurePoly(x**2 + 1).free_symbols == set()
    assert PurePoly(x**2 + y*z).free_symbols == set()
    assert PurePoly(x**2 + y*z, x).free_symbols == {y, z}
    assert PurePoly(x**2 + sin(y*z)).free_symbols == set()
    assert PurePoly(x**2 + sin(y*z), x).free_symbols == {y, z}
    assert PurePoly(x**2 + sin(y*z), x, domain=EX).free_symbols == {y, z}


def test_Poly__eq__():
    assert (Poly(x, x) == Poly(x, x)) is True
    assert (Poly(x, x, domain=QQ) == Poly(x, x)) is False
    assert (Poly(x, x) == Poly(x, x, domain=QQ)) is False

    assert (Poly(x, x, domain=ZZ[a]) == Poly(x, x)) is False
    assert (Poly(x, x) == Poly(x, x, domain=ZZ[a])) is False

    assert (Poly(x*y, x, y) == Poly(x, x)) is False

    assert (Poly(x, x, y) == Poly(x, x)) is False
    assert (Poly(x, x) == Poly(x, x, y)) is False

    assert (Poly(x**2 + 1, x) == Poly(y**2 + 1, y)) is False
    assert (Poly(y**2 + 1, y) == Poly(x**2 + 1, x)) is False

    f = Poly(x, x, domain=ZZ)
    g = Poly(x, x, domain=QQ)

    assert f.eq(g) is False
    assert f.ne(g) is True

    assert f.eq(g, strict=True) is False
    assert f.ne(g, strict=True) is True

    t0 = Symbol('t0')

    f =  Poly((t0/2 + x**2)*t**2 - x**2*t, t, domain='QQ[x,t0]')
    g =  Poly((t0/2 + x**2)*t**2 - x**2*t, t, domain='ZZ(x,t0)')

    assert (f == g) is False


def test_PurePoly__eq__():
    assert (PurePoly(x, x) == PurePoly(x, x)) is True
    assert (PurePoly(x, x, domain=QQ) == PurePoly(x, x)) is True
    assert (PurePoly(x, x) == PurePoly(x, x, domain=QQ)) is True

    assert (PurePoly(x, x, domain=ZZ[a]) == PurePoly(x, x)) is True
    assert (PurePoly(x, x) == PurePoly(x, x, domain=ZZ[a])) is True

    assert (PurePoly(x*y, x, y) == PurePoly(x, x)) is False

    assert (PurePoly(x, x, y) == PurePoly(x, x)) is False
    assert (PurePoly(x, x) == PurePoly(x, x, y)) is False

    assert (PurePoly(x**2 + 1, x) == PurePoly(y**2 + 1, y)) is True
    assert (PurePoly(y**2 + 1, y) == PurePoly(x**2 + 1, x)) is True

    f = PurePoly(x, x, domain=ZZ)
    g = PurePoly(x, x, domain=QQ)

    assert f.eq(g) is True
    assert f.ne(g) is False

    assert f.eq(g, strict=True) is False
    assert f.ne(g, strict=True) is True

    f = PurePoly(x, x, domain=ZZ)
    g = PurePoly(y, y, domain=QQ)

    assert f.eq(g) is True
    assert f.ne(g) is False

    assert f.eq(g, strict=True) is False
    assert f.ne(g, strict=True) is True


def test_PurePoly_Poly():
    assert isinstance(PurePoly(Poly(x**2 + 1)), PurePoly) is True
    assert isinstance(Poly(PurePoly(x**2 + 1)), Poly) is True


def test_Poly_get_domain():
    assert Poly(2*x).get_domain() == ZZ

    assert Poly(2*x, domain='ZZ').get_domain() == ZZ
    assert Poly(2*x, domain='QQ').get_domain() == QQ

    assert Poly(x/2).get_domain() == QQ

    raises(CoercionFailed, lambda: Poly(x/2, domain='ZZ'))
    assert Poly(x/2, domain='QQ').get_domain() == QQ

    assert isinstance(Poly(0.2*x).get_domain(), RealField)


def test_Poly_set_domain():
    assert Poly(2*x + 1).set_domain(ZZ) == Poly(2*x + 1)
    assert Poly(2*x + 1).set_domain('ZZ') == Poly(2*x + 1)

    assert Poly(2*x + 1).set_domain(QQ) == Poly(2*x + 1, domain='QQ')
    assert Poly(2*x + 1).set_domain('QQ') == Poly(2*x + 1, domain='QQ')

    assert Poly(Rational(2, 10)*x + Rational(1, 10)).set_domain('RR') == Poly(0.2*x + 0.1)
    assert Poly(0.2*x + 0.1).set_domain('QQ') == Poly(Rational(2, 10)*x + Rational(1, 10))

    raises(CoercionFailed, lambda: Poly(x/2 + 1).set_domain(ZZ))
    raises(CoercionFailed, lambda: Poly(x + 1, modulus=2).set_domain(QQ))

    raises(GeneratorsError, lambda: Poly(x*y, x, y).set_domain(ZZ[y]))


def test_Poly_get_modulus():
    assert Poly(x**2 + 1, modulus=2).get_modulus() == 2
    raises(PolynomialError, lambda: Poly(x**2 + 1).get_modulus())


def test_Poly_set_modulus():
    assert Poly(
        x**2 + 1, modulus=2).set_modulus(7) == Poly(x**2 + 1, modulus=7)
    assert Poly(
        x**2 + 5, modulus=7).set_modulus(2) == Poly(x**2 + 1, modulus=2)

    assert Poly(x**2 + 1).set_modulus(2) == Poly(x**2 + 1, modulus=2)

    raises(CoercionFailed, lambda: Poly(x/2 + 1).set_modulus(2))


def test_Poly_add_ground():
    assert Poly(x + 1).add_ground(2) == Poly(x + 3)


def test_Poly_sub_ground():
    assert Poly(x + 1).sub_ground(2) == Poly(x - 1)


def test_Poly_mul_ground():
    assert Poly(x + 1).mul_ground(2) == Poly(2*x + 2)


def test_Poly_quo_ground():
    assert Poly(2*x + 4).quo_ground(2) == Poly(x + 2)
    assert Poly(2*x + 3).quo_ground(2) == Poly(x + 1)


def test_Poly_exquo_ground():
    assert Poly(2*x + 4).exquo_ground(2) == Poly(x + 2)
    raises(ExactQuotientFailed, lambda: Poly(2*x + 3).exquo_ground(2))


def test_Poly_abs():
    assert Poly(-x + 1, x).abs() == abs(Poly(-x + 1, x)) == Poly(x + 1, x)


def test_Poly_neg():
    assert Poly(-x + 1, x).neg() == -Poly(-x + 1, x) == Poly(x - 1, x)


def test_Poly_add():
    assert Poly(0, x).add(Poly(0, x)) == Poly(0, x)
    assert Poly(0, x) + Poly(0, x) == Poly(0, x)

    assert Poly(1, x).add(Poly(0, x)) == Poly(1, x)
    assert Poly(1, x, y) + Poly(0, x) == Poly(1, x, y)
    assert Poly(0, x).add(Poly(1, x, y)) == Poly(1, x, y)
    assert Poly(0, x, y) + Poly(1, x, y) == Poly(1, x, y)

    assert Poly(1, x) + x == Poly(x + 1, x)
    with warns_deprecated_sympy():
        Poly(1, x) + sin(x)

    assert Poly(x, x) + 1 == Poly(x + 1, x)
    assert 1 + Poly(x, x) == Poly(x + 1, x)


def test_Poly_sub():
    assert Poly(0, x).sub(Poly(0, x)) == Poly(0, x)
    assert Poly(0, x) - Poly(0, x) == Poly(0, x)

    assert Poly(1, x).sub(Poly(0, x)) == Poly(1, x)
    assert Poly(1, x, y) - Poly(0, x) == Poly(1, x, y)
    assert Poly(0, x).sub(Poly(1, x, y)) == Poly(-1, x, y)
    assert Poly(0, x, y) - Poly(1, x, y) == Poly(-1, x, y)

    assert Poly(1, x) - x == Poly(1 - x, x)
    with warns_deprecated_sympy():
        Poly(1, x) - sin(x)

    assert Poly(x, x) - 1 == Poly(x - 1, x)
    assert 1 - Poly(x, x) == Poly(1 - x, x)


def test_Poly_mul():
    assert Poly(0, x).mul(Poly(0, x)) == Poly(0, x)
    assert Poly(0, x) * Poly(0, x) == Poly(0, x)

    assert Poly(2, x).mul(Poly(4, x)) == Poly(8, x)
    assert Poly(2, x, y) * Poly(4, x) == Poly(8, x, y)
    assert Poly(4, x).mul(Poly(2, x, y)) == Poly(8, x, y)
    assert Poly(4, x, y) * Poly(2, x, y) == Poly(8, x, y)

    assert Poly(1, x) * x == Poly(x, x)
    with warns_deprecated_sympy():
        Poly(1, x) * sin(x)

    assert Poly(x, x) * 2 == Poly(2*x, x)
    assert 2 * Poly(x, x) == Poly(2*x, x)

def test_issue_13079():
    assert Poly(x)*x == Poly(x**2, x, domain='ZZ')
    assert x*Poly(x) == Poly(x**2, x, domain='ZZ')
    assert -2*Poly(x) == Poly(-2*x, x, domain='ZZ')
    assert S(-2)*Poly(x) == Poly(-2*x, x, domain='ZZ')
    assert Poly(x)*S(-2) == Poly(-2*x, x, domain='ZZ')

def test_Poly_sqr():
    assert Poly(x*y, x, y).sqr() == Poly(x**2*y**2, x, y)


def test_Poly_pow():
    assert Poly(x, x).pow(10) == Poly(x**10, x)
    assert Poly(x, x).pow(Integer(10)) == Poly(x**10, x)

    assert Poly(2*y, x, y).pow(4) == Poly(16*y**4, x, y)
    assert Poly(2*y, x, y).pow(Integer(4)) == Poly(16*y**4, x, y)

    assert Poly(7*x*y, x, y)**3 == Poly(343*x**3*y**3, x, y)

    raises(TypeError, lambda: Poly(x*y + 1, x, y)**(-1))
    raises(TypeError, lambda: Poly(x*y + 1, x, y)**x)


def test_Poly_divmod():
    f, g = Poly(x**2), Poly(x)
    q, r = g, Poly(0, x)

    assert divmod(f, g) == (q, r)
    assert f // g == q
    assert f % g == r

    assert divmod(f, x) == (q, r)
    assert f // x == q
    assert f % x == r

    q, r = Poly(0, x), Poly(2, x)

    assert divmod(2, g) == (q, r)
    assert 2 // g == q
    assert 2 % g == r

    assert Poly(x)/Poly(x) == 1
    assert Poly(x**2)/Poly(x) == x
    assert Poly(x)/Poly(x**2) == 1/x


def test_Poly_eq_ne():
    assert (Poly(x + y, x, y) == Poly(x + y, x, y)) is True
    assert (Poly(x + y, x) == Poly(x + y, x, y)) is False
    assert (Poly(x + y, x, y) == Poly(x + y, x)) is False
    assert (Poly(x + y, x) == Poly(x + y, x)) is True
    assert (Poly(x + y, y) == Poly(x + y, y)) is True

    assert (Poly(x + y, x, y) == x + y) is True
    assert (Poly(x + y, x) == x + y) is True
    assert (Poly(x + y, x, y) == x + y) is True
    assert (Poly(x + y, x) == x + y) is True
    assert (Poly(x + y, y) == x + y) is True

    assert (Poly(x + y, x, y) != Poly(x + y, x, y)) is False
    assert (Poly(x + y, x) != Poly(x + y, x, y)) is True
    assert (Poly(x + y, x, y) != Poly(x + y, x)) is True
    assert (Poly(x + y, x) != Poly(x + y, x)) is False
    assert (Poly(x + y, y) != Poly(x + y, y)) is False

    assert (Poly(x + y, x, y) != x + y) is False
    assert (Poly(x + y, x) != x + y) is False
    assert (Poly(x + y, x, y) != x + y) is False
    assert (Poly(x + y, x) != x + y) is False
    assert (Poly(x + y, y) != x + y) is False

    assert (Poly(x, x) == sin(x)) is False
    assert (Poly(x, x) != sin(x)) is True


def test_Poly_nonzero():
    assert not bool(Poly(0, x)) is True
    assert not bool(Poly(1, x)) is False


def test_Poly_properties():
    assert Poly(0, x).is_zero is True
    assert Poly(1, x).is_zero is False

    assert Poly(1, x).is_one is True
    assert Poly(2, x).is_one is False

    assert Poly(x - 1, x).is_sqf is True
    assert Poly((x - 1)**2, x).is_sqf is False

    assert Poly(x - 1, x).is_monic is True
    assert Poly(2*x - 1, x).is_monic is False

    assert Poly(3*x + 2, x).is_primitive is True
    assert Poly(4*x + 2, x).is_primitive is False

    assert Poly(1, x).is_ground is True
    assert Poly(x, x).is_ground is False

    assert Poly(x + y + z + 1).is_linear is True
    assert Poly(x*y*z + 1).is_linear is False

    assert Poly(x*y + z + 1).is_quadratic is True
    assert Poly(x*y*z + 1).is_quadratic is False

    assert Poly(x*y).is_monomial is True
    assert Poly(x*y + 1).is_monomial is False

    assert Poly(x**2 + x*y).is_homogeneous is True
    assert Poly(x**3 + x*y).is_homogeneous is False

    assert Poly(x).is_univariate is True
    assert Poly(x*y).is_univariate is False

    assert Poly(x*y).is_multivariate is True
    assert Poly(x).is_multivariate is False

    assert Poly(
        x**16 + x**14 - x**10 + x**8 - x**6 + x**2 + 1).is_cyclotomic is False
    assert Poly(
        x**16 + x**14 - x**10 - x**8 - x**6 + x**2 + 1).is_cyclotomic is True


def test_Poly_is_irreducible():
    assert Poly(x**2 + x + 1).is_irreducible is True
    assert Poly(x**2 + 2*x + 1).is_irreducible is False

    assert Poly(7*x + 3, modulus=11).is_irreducible is True
    assert Poly(7*x**2 + 3*x + 1, modulus=11).is_irreducible is False


def test_Poly_subs():
    assert Poly(x + 1).subs(x, 0) == 1

    assert Poly(x + 1).subs(x, x) == Poly(x + 1)
    assert Poly(x + 1).subs(x, y) == Poly(y + 1)

    assert Poly(x*y, x).subs(y, x) == x**2
    assert Poly(x*y, x).subs(x, y) == y**2


def test_Poly_replace():
    assert Poly(x + 1).replace(x) == Poly(x + 1)
    assert Poly(x + 1).replace(y) == Poly(y + 1)

    raises(PolynomialError, lambda: Poly(x + y).replace(z))

    assert Poly(x + 1).replace(x, x) == Poly(x + 1)
    assert Poly(x + 1).replace(x, y) == Poly(y + 1)

    assert Poly(x + y).replace(x, x) == Poly(x + y)
    assert Poly(x + y).replace(x, z) == Poly(z + y, z, y)

    assert Poly(x + y).replace(y, y) == Poly(x + y)
    assert Poly(x + y).replace(y, z) == Poly(x + z, x, z)
    assert Poly(x + y).replace(z, t) == Poly(x + y)

    raises(PolynomialError, lambda: Poly(x + y).replace(x, y))

    assert Poly(x + y, x).replace(x, z) == Poly(z + y, z)
    assert Poly(x + y, y).replace(y, z) == Poly(x + z, z)

    raises(PolynomialError, lambda: Poly(x + y, x).replace(x, y))
    raises(PolynomialError, lambda: Poly(x + y, y).replace(y, x))


def test_Poly_reorder():
    raises(PolynomialError, lambda: Poly(x + y).reorder(x, z))

    assert Poly(x + y, x, y).reorder(x, y) == Poly(x + y, x, y)
    assert Poly(x + y, x, y).reorder(y, x) == Poly(x + y, y, x)

    assert Poly(x + y, y, x).reorder(x, y) == Poly(x + y, x, y)
    assert Poly(x + y, y, x).reorder(y, x) == Poly(x + y, y, x)

    assert Poly(x + y, x, y).reorder(wrt=x) == Poly(x + y, x, y)
    assert Poly(x + y, x, y).reorder(wrt=y) == Poly(x + y, y, x)


def test_Poly_ltrim():
    f = Poly(y**2 + y*z**2, x, y, z).ltrim(y)
    assert f.as_expr() == y**2 + y*z**2 and f.gens == (y, z)
    assert Poly(x*y - x, z, x, y).ltrim(1) == Poly(x*y - x, x, y)

    raises(PolynomialError, lambda: Poly(x*y**2 + y**2, x, y).ltrim(y))
    raises(PolynomialError, lambda: Poly(x*y - x, x, y).ltrim(-1))

def test_Poly_has_only_gens():
    assert Poly(x*y + 1, x, y, z).has_only_gens(x, y) is True
    assert Poly(x*y + z, x, y, z).has_only_gens(x, y) is False

    raises(GeneratorsError, lambda: Poly(x*y**2 + y**2, x, y).has_only_gens(t))


def test_Poly_to_ring():
    assert Poly(2*x + 1, domain='ZZ').to_ring() == Poly(2*x + 1, domain='ZZ')
    assert Poly(2*x + 1, domain='QQ').to_ring() == Poly(2*x + 1, domain='ZZ')

    raises(CoercionFailed, lambda: Poly(x/2 + 1).to_ring())
    raises(DomainError, lambda: Poly(2*x + 1, modulus=3).to_ring())


def test_Poly_to_field():
    assert Poly(2*x + 1, domain='ZZ').to_field() == Poly(2*x + 1, domain='QQ')
    assert Poly(2*x + 1, domain='QQ').to_field() == Poly(2*x + 1, domain='QQ')

    assert Poly(x/2 + 1, domain='QQ').to_field() == Poly(x/2 + 1, domain='QQ')
    assert Poly(2*x + 1, modulus=3).to_field() == Poly(2*x + 1, modulus=3)

    assert Poly(2.0*x + 1.0).to_field() == Poly(2.0*x + 1.0)


def test_Poly_to_exact():
    assert Poly(2*x).to_exact() == Poly(2*x)
    assert Poly(x/2).to_exact() == Poly(x/2)

    assert Poly(0.1*x).to_exact() == Poly(x/10)


def test_Poly_retract():
    f = Poly(x**2 + 1, x, domain=QQ[y])

    assert f.retract() == Poly(x**2 + 1, x, domain='ZZ')
    assert f.retract(field=True) == Poly(x**2 + 1, x, domain='QQ')

    assert Poly(0, x, y).retract() == Poly(0, x, y)


def test_Poly_slice():
    f = Poly(x**3 + 2*x**2 + 3*x + 4)

    assert f.slice(0, 0) == Poly(0, x)
    assert f.slice(0, 1) == Poly(4, x)
    assert f.slice(0, 2) == Poly(3*x + 4, x)
    assert f.slice(0, 3) == Poly(2*x**2 + 3*x + 4, x)
    assert f.slice(0, 4) == Poly(x**3 + 2*x**2 + 3*x + 4, x)

    assert f.slice(x, 0, 0) == Poly(0, x)
    assert f.slice(x, 0, 1) == Poly(4, x)
    assert f.slice(x, 0, 2) == Poly(3*x + 4, x)
    assert f.slice(x, 0, 3) == Poly(2*x**2 + 3*x + 4, x)
    assert f.slice(x, 0, 4) == Poly(x**3 + 2*x**2 + 3*x + 4, x)

    g = Poly(x**3 + 1)

    assert g.slice(0, 3) == Poly(1, x)


def test_Poly_coeffs():
    assert Poly(0, x).coeffs() == [0]
    assert Poly(1, x).coeffs() == [1]

    assert Poly(2*x + 1, x).coeffs() == [2, 1]

    assert Poly(7*x**2 + 2*x + 1, x).coeffs() == [7, 2, 1]
    assert Poly(7*x**4 + 2*x + 1, x).coeffs() == [7, 2, 1]

    assert Poly(x*y**7 + 2*x**2*y**3).coeffs('lex') == [2, 1]
    assert Poly(x*y**7 + 2*x**2*y**3).coeffs('grlex') == [1, 2]


def test_Poly_monoms():
    assert Poly(0, x).monoms() == [(0,)]
    assert Poly(1, x).monoms() == [(0,)]

    assert Poly(2*x + 1, x).monoms() == [(1,), (0,)]

    assert Poly(7*x**2 + 2*x + 1, x).monoms() == [(2,), (1,), (0,)]
    assert Poly(7*x**4 + 2*x + 1, x).monoms() == [(4,), (1,), (0,)]

    assert Poly(x*y**7 + 2*x**2*y**3).monoms('lex') == [(2, 3), (1, 7)]
    assert Poly(x*y**7 + 2*x**2*y**3).monoms('grlex') == [(1, 7), (2, 3)]


def test_Poly_terms():
    assert Poly(0, x).terms() == [((0,), 0)]
    assert Poly(1, x).terms() == [((0,), 1)]

    assert Poly(2*x + 1, x).terms() == [((1,), 2), ((0,), 1)]

    assert Poly(7*x**2 + 2*x + 1, x).terms() == [((2,), 7), ((1,), 2), ((0,), 1)]
    assert Poly(7*x**4 + 2*x + 1, x).terms() == [((4,), 7), ((1,), 2), ((0,), 1)]

    assert Poly(
        x*y**7 + 2*x**2*y**3).terms('lex') == [((2, 3), 2), ((1, 7), 1)]
    assert Poly(
        x*y**7 + 2*x**2*y**3).terms('grlex') == [((1, 7), 1), ((2, 3), 2)]


def test_Poly_all_coeffs():
    assert Poly(0, x).all_coeffs() == [0]
    assert Poly(1, x).all_coeffs() == [1]

    assert Poly(2*x + 1, x).all_coeffs() == [2, 1]

    assert Poly(7*x**2 + 2*x + 1, x).all_coeffs() == [7, 2, 1]
    assert Poly(7*x**4 + 2*x + 1, x).all_coeffs() == [7, 0, 0, 2, 1]


def test_Poly_all_monoms():
    assert Poly(0, x).all_monoms() == [(0,)]
    assert Poly(1, x).all_monoms() == [(0,)]

    assert Poly(2*x + 1, x).all_monoms() == [(1,), (0,)]

    assert Poly(7*x**2 + 2*x + 1, x).all_monoms() == [(2,), (1,), (0,)]
    assert Poly(7*x**4 + 2*x + 1, x).all_monoms() == [(4,), (3,), (2,), (1,), (0,)]


def test_Poly_all_terms():
    assert Poly(0, x).all_terms() == [((0,), 0)]
    assert Poly(1, x).all_terms() == [((0,), 1)]

    assert Poly(2*x + 1, x).all_terms() == [((1,), 2), ((0,), 1)]

    assert Poly(7*x**2 + 2*x + 1, x).all_terms() == \
        [((2,), 7), ((1,), 2), ((0,), 1)]
    assert Poly(7*x**4 + 2*x + 1, x).all_terms() == \
        [((4,), 7), ((3,), 0), ((2,), 0), ((1,), 2), ((0,), 1)]


def test_Poly_termwise():
    f = Poly(x**2 + 20*x + 400)
    g = Poly(x**2 + 2*x + 4)

    def func(monom, coeff):
        (k,) = monom
        return coeff//10**(2 - k)

    assert f.termwise(func) == g

    def func(monom, coeff):
        (k,) = monom
        return (k,), coeff//10**(2 - k)

    assert f.termwise(func) == g


def test_Poly_length():
    assert Poly(0, x).length() == 0
    assert Poly(1, x).length() == 1
    assert Poly(x, x).length() == 1

    assert Poly(x + 1, x).length() == 2
    assert Poly(x**2 + 1, x).length() == 2
    assert Poly(x**2 + x + 1, x).length() == 3


def test_Poly_as_dict():
    assert Poly(0, x).as_dict() == {}
    assert Poly(0, x, y, z).as_dict() == {}

    assert Poly(1, x).as_dict() == {(0,): 1}
    assert Poly(1, x, y, z).as_dict() == {(0, 0, 0): 1}

    assert Poly(x**2 + 3, x).as_dict() == {(2,): 1, (0,): 3}
    assert Poly(x**2 + 3, x, y, z).as_dict() == {(2, 0, 0): 1, (0, 0, 0): 3}

    assert Poly(3*x**2*y*z**3 + 4*x*y + 5*x*z).as_dict() == {(2, 1, 3): 3,
                (1, 1, 0): 4, (1, 0, 1): 5}


def test_Poly_as_expr():
    assert Poly(0, x).as_expr() == 0
    assert Poly(0, x, y, z).as_expr() == 0

    assert Poly(1, x).as_expr() == 1
    assert Poly(1, x, y, z).as_expr() == 1

    assert Poly(x**2 + 3, x).as_expr() == x**2 + 3
    assert Poly(x**2 + 3, x, y, z).as_expr() == x**2 + 3

    assert Poly(
        3*x**2*y*z**3 + 4*x*y + 5*x*z).as_expr() == 3*x**2*y*z**3 + 4*x*y + 5*x*z

    f = Poly(x**2 + 2*x*y**2 - y, x, y)

    assert f.as_expr() == -y + x**2 + 2*x*y**2

    assert f.as_expr({x: 5}) == 25 - y + 10*y**2
    assert f.as_expr({y: 6}) == -6 + 72*x + x**2

    assert f.as_expr({x: 5, y: 6}) == 379
    assert f.as_expr(5, 6) == 379

    raises(GeneratorsError, lambda: f.as_expr({z: 7}))


def test_Poly_lift():
    p = Poly(x**4 - I*x + 17*I, x, gaussian=True)
    assert p.lift() == Poly(x**8 + x**2 - 34*x + 289, x, domain='QQ')


def test_Poly_lift_multiple():

    r1 = rootof(y**3 + y**2 - 1, 0)
    r2 = rootof(z**5 + z**2 - 1, 0)
    p = Poly(r1*x + 3*r1**2 - r2 + x**2 - x**5, x, extension=True)

    assert p.lift() == Poly(
        -x**75 + 15*x**72 - 5*x**71 + 15*x**70 - 105*x**69 + 70*x**68 -
        220*x**67 + 560*x**66 - 635*x**65 + 1495*x**64 - 2735*x**63 +
        4415*x**62 - 7410*x**61 + 12741*x**60 - 22090*x**59 + 32125*x**58 -
        56281*x**57 + 88157*x**56 - 126842*x**55 + 214223*x**54 - 311802*x**53
        + 462667*x**52 - 700883*x**51 + 1006278*x**50 - 1480950*x**49 +
        2078055*x**48 - 3004675*x**47 + 4140410*x**46 - 5664222*x**45 +
        8029445*x**44 - 10528785*x**43 + 14309614*x**42 - 19032988*x**41 +
        24570573*x**40 - 32530459*x**39 + 41239581*x**38 - 52968051*x**37 +
        65891606*x**36 - 81997276*x**35 + 102530732*x**34 - 122009994*x**33 +
        150227996*x**32 - 176452478*x**31 + 206393768*x**30 - 245291426*x**29 +
        276598718*x**28 - 320005297*x**27 + 353649032*x**26
        - 393246309*x**25 + 434566186*x**24 - 460608964*x**23 + 508052079*x**22
        - 513976618*x**21 + 539374498*x**20 - 557851717*x**19 + 540788016*x**18
        - 564949060*x**17 + 520866566*x**16
        - 507861375*x**15 + 474999819*x**14 - 423619160*x**13 + 414540540*x**12
        - 322522367*x**11 + 311586511*x**10 - 238812299*x**9 + 184482053*x**8
        - 189265274*x**7 + 93619528*x**6 - 106852385*x**5 + 57294385*x**4 -
        26486666*x**3 + 42614683*x**2 - 1511583*x + 15975845, x, domain='QQ'
    )


def test_Poly_deflate():
    assert Poly(0, x).deflate() == ((1,), Poly(0, x))
    assert Poly(1, x).deflate() == ((1,), Poly(1, x))
    assert Poly(x, x).deflate() == ((1,), Poly(x, x))

    assert Poly(x**2, x).deflate() == ((2,), Poly(x, x))
    assert Poly(x**17, x).deflate() == ((17,), Poly(x, x))

    assert Poly(
        x**2*y*z**11 + x**4*z**11).deflate() == ((2, 1, 11), Poly(x*y*z + x**2*z))


def test_Poly_inject():
    f = Poly(x**2*y + x*y**3 + x*y + 1, x)

    assert f.inject() == Poly(x**2*y + x*y**3 + x*y + 1, x, y)
    assert f.inject(front=True) == Poly(y**3*x + y*x**2 + y*x + 1, y, x)


def test_Poly_eject():
    f = Poly(x**2*y + x*y**3 + x*y + 1, x, y)

    assert f.eject(x) == Poly(x*y**3 + (x**2 + x)*y + 1, y, domain='ZZ[x]')
    assert f.eject(y) == Poly(y*x**2 + (y**3 + y)*x + 1, x, domain='ZZ[y]')

    ex = x + y + z + t + w
    g = Poly(ex, x, y, z, t, w)

    assert g.eject(x) == Poly(ex, y, z, t, w, domain='ZZ[x]')
    assert g.eject(x, y) == Poly(ex, z, t, w, domain='ZZ[x, y]')
    assert g.eject(x, y, z) == Poly(ex, t, w, domain='ZZ[x, y, z]')
    assert g.eject(w) == Poly(ex, x, y, z, t, domain='ZZ[w]')
    assert g.eject(t, w) == Poly(ex, x, y, z, domain='ZZ[t, w]')
    assert g.eject(z, t, w) == Poly(ex, x, y, domain='ZZ[z, t, w]')

    raises(DomainError, lambda: Poly(x*y, x, y, domain=ZZ[z]).eject(y))
    raises(NotImplementedError, lambda: Poly(x*y, x, y, z).eject(y))


def test_Poly_exclude():
    assert Poly(x, x, y).exclude() == Poly(x, x)
    assert Poly(x*y, x, y).exclude() == Poly(x*y, x, y)
    assert Poly(1, x, y).exclude() == Poly(1, x, y)


def test_Poly__gen_to_level():
    assert Poly(1, x, y)._gen_to_level(-2) == 0
    assert Poly(1, x, y)._gen_to_level(-1) == 1
    assert Poly(1, x, y)._gen_to_level( 0) == 0
    assert Poly(1, x, y)._gen_to_level( 1) == 1

    raises(PolynomialError, lambda: Poly(1, x, y)._gen_to_level(-3))
    raises(PolynomialError, lambda: Poly(1, x, y)._gen_to_level( 2))

    assert Poly(1, x, y)._gen_to_level(x) == 0
    assert Poly(1, x, y)._gen_to_level(y) == 1

    assert Poly(1, x, y)._gen_to_level('x') == 0
    assert Poly(1, x, y)._gen_to_level('y') == 1

    raises(PolynomialError, lambda: Poly(1, x, y)._gen_to_level(z))
    raises(PolynomialError, lambda: Poly(1, x, y)._gen_to_level('z'))


def test_Poly_degree():
    assert Poly(0, x).degree() is -oo
    assert Poly(1, x).degree() == 0
    assert Poly(x, x).degree() == 1

    assert Poly(0, x).degree(gen=0) is -oo
    assert Poly(1, x).degree(gen=0) == 0
    assert Poly(x, x).degree(gen=0) == 1

    assert Poly(0, x).degree(gen=x) is -oo
    assert Poly(1, x).degree(gen=x) == 0
    assert Poly(x, x).degree(gen=x) == 1

    assert Poly(0, x).degree(gen='x') is -oo
    assert Poly(1, x).degree(gen='x') == 0
    assert Poly(x, x).degree(gen='x') == 1

    raises(PolynomialError, lambda: Poly(1, x).degree(gen=1))
    raises(PolynomialError, lambda: Poly(1, x).degree(gen=y))
    raises(PolynomialError, lambda: Poly(1, x).degree(gen='y'))

    assert Poly(1, x, y).degree() == 0
    assert Poly(2*y, x, y).degree() == 0
    assert Poly(x*y, x, y).degree() == 1

    assert Poly(1, x, y).degree(gen=x) == 0
    assert Poly(2*y, x, y).degree(gen=x) == 0
    assert Poly(x*y, x, y).degree(gen=x) == 1

    assert Poly(1, x, y).degree(gen=y) == 0
    assert Poly(2*y, x, y).degree(gen=y) == 1
    assert Poly(x*y, x, y).degree(gen=y) == 1

    assert degree(0, x) is -oo
    assert degree(1, x) == 0
    assert degree(x, x) == 1

    assert degree(x*y**2, x) == 1
    assert degree(x*y**2, y) == 2
    assert degree(x*y**2, z) == 0

    assert degree(pi) == 1

    raises(TypeError, lambda: degree(y**2 + x**3))
    raises(TypeError, lambda: degree(y**2 + x**3, 1))
    raises(PolynomialError, lambda: degree(x, 1.1))
    raises(PolynomialError, lambda: degree(x**2/(x**3 + 1), x))

    assert degree(Poly(0,x),z) is -oo
    assert degree(Poly(1,x),z) == 0
    assert degree(Poly(x**2+y**3,y)) == 3
    assert degree(Poly(y**2 + x**3, y, x), 1) == 3
    assert degree(Poly(y**2 + x**3, x), z) == 0
    assert degree(Poly(y**2 + x**3 + z**4, x), z) == 4

def test_Poly_degree_list():
    assert Poly(0, x).degree_list() == (-oo,)
    assert Poly(0, x, y).degree_list() == (-oo, -oo)
    assert Poly(0, x, y, z).degree_list() == (-oo, -oo, -oo)

    assert Poly(1, x).degree_list() == (0,)
    assert Poly(1, x, y).degree_list() == (0, 0)
    assert Poly(1, x, y, z).degree_list() == (0, 0, 0)

    assert Poly(x**2*y + x**3*z**2 + 1).degree_list() == (3, 1, 2)

    assert degree_list(1, x) == (0,)
    assert degree_list(x, x) == (1,)

    assert degree_list(x*y**2) == (1, 2)

    raises(ComputationFailed, lambda: degree_list(1))


def test_Poly_total_degree():
    assert Poly(x**2*y + x**3*z**2 + 1).total_degree() == 5
    assert Poly(x**2 + z**3).total_degree() == 3
    assert Poly(x*y*z + z**4).total_degree() == 4
    assert Poly(x**3 + x + 1).total_degree() == 3

    assert total_degree(x*y + z**3) == 3
    assert total_degree(x*y + z**3, x, y) == 2
    assert total_degree(1) == 0
    assert total_degree(Poly(y**2 + x**3 + z**4)) == 4
    assert total_degree(Poly(y**2 + x**3 + z**4, x)) == 3
    assert total_degree(Poly(y**2 + x**3 + z**4, x), z) == 4
    assert total_degree(Poly(x**9 + x*z*y + x**3*z**2 + z**7,x), z) == 7

def test_Poly_homogenize():
    assert Poly(x**2+y).homogenize(z) == Poly(x**2+y*z)
    assert Poly(x+y).homogenize(z) == Poly(x+y, x, y, z)
    assert Poly(x+y**2).homogenize(y) == Poly(x*y+y**2)


def test_Poly_homogeneous_order():
    assert Poly(0, x, y).homogeneous_order() is -oo
    assert Poly(1, x, y).homogeneous_order() == 0
    assert Poly(x, x, y).homogeneous_order() == 1
    assert Poly(x*y, x, y).homogeneous_order() == 2

    assert Poly(x + 1, x, y).homogeneous_order() is None
    assert Poly(x*y + x, x, y).homogeneous_order() is None

    assert Poly(x**5 + 2*x**3*y**2 + 9*x*y**4).homogeneous_order() == 5
    assert Poly(x**5 + 2*x**3*y**3 + 9*x*y**4).homogeneous_order() is None


def test_Poly_LC():
    assert Poly(0, x).LC() == 0
    assert Poly(1, x).LC() == 1
    assert Poly(2*x**2 + x, x).LC() == 2

    assert Poly(x*y**7 + 2*x**2*y**3).LC('lex') == 2
    assert Poly(x*y**7 + 2*x**2*y**3).LC('grlex') == 1

    assert LC(x*y**7 + 2*x**2*y**3, order='lex') == 2
    assert LC(x*y**7 + 2*x**2*y**3, order='grlex') == 1


def test_Poly_TC():
    assert Poly(0, x).TC() == 0
    assert Poly(1, x).TC() == 1
    assert Poly(2*x**2 + x, x).TC() == 0


def test_Poly_EC():
    assert Poly(0, x).EC() == 0
    assert Poly(1, x).EC() == 1
    assert Poly(2*x**2 + x, x).EC() == 1

    assert Poly(x*y**7 + 2*x**2*y**3).EC('lex') == 1
    assert Poly(x*y**7 + 2*x**2*y**3).EC('grlex') == 2


def test_Poly_coeff():
    assert Poly(0, x).coeff_monomial(1) == 0
    assert Poly(0, x).coeff_monomial(x) == 0

    assert Poly(1, x).coeff_monomial(1) == 1
    assert Poly(1, x).coeff_monomial(x) == 0

    assert Poly(x**8, x).coeff_monomial(1) == 0
    assert Poly(x**8, x).coeff_monomial(x**7) == 0
    assert Poly(x**8, x).coeff_monomial(x**8) == 1
    assert Poly(x**8, x).coeff_monomial(x**9) == 0

    assert Poly(3*x*y**2 + 1, x, y).coeff_monomial(1) == 1
    assert Poly(3*x*y**2 + 1, x, y).coeff_monomial(x*y**2) == 3

    p = Poly(24*x*y*exp(8) + 23*x, x, y)

    assert p.coeff_monomial(x) == 23
    assert p.coeff_monomial(y) == 0
    assert p.coeff_monomial(x*y) == 24*exp(8)

    assert p.as_expr().coeff(x) == 24*y*exp(8) + 23
    raises(NotImplementedError, lambda: p.coeff(x))

    raises(ValueError, lambda: Poly(x + 1).coeff_monomial(0))
    raises(ValueError, lambda: Poly(x + 1).coeff_monomial(3*x))
    raises(ValueError, lambda: Poly(x + 1).coeff_monomial(3*x*y))


def test_Poly_nth():
    assert Poly(0, x).nth(0) == 0
    assert Poly(0, x).nth(1) == 0

    assert Poly(1, x).nth(0) == 1
    assert Poly(1, x).nth(1) == 0

    assert Poly(x**8, x).nth(0) == 0
    assert Poly(x**8, x).nth(7) == 0
    assert Poly(x**8, x).nth(8) == 1
    assert Poly(x**8, x).nth(9) == 0

    assert Poly(3*x*y**2 + 1, x, y).nth(0, 0) == 1
    assert Poly(3*x*y**2 + 1, x, y).nth(1, 2) == 3

    raises(ValueError, lambda: Poly(x*y + 1, x, y).nth(1))


def test_Poly_LM():
    assert Poly(0, x).LM() == (0,)
    assert Poly(1, x).LM() == (0,)
    assert Poly(2*x**2 + x, x).LM() == (2,)

    assert Poly(x*y**7 + 2*x**2*y**3).LM('lex') == (2, 3)
    assert Poly(x*y**7 + 2*x**2*y**3).LM('grlex') == (1, 7)

    assert LM(x*y**7 + 2*x**2*y**3, order='lex') == x**2*y**3
    assert LM(x*y**7 + 2*x**2*y**3, order='grlex') == x*y**7


def test_Poly_LM_custom_order():
    f = Poly(x**2*y**3*z + x**2*y*z**3 + x*y*z + 1)
    rev_lex = lambda monom: tuple(reversed(monom))

    assert f.LM(order='lex') == (2, 3, 1)
    assert f.LM(order=rev_lex) == (2, 1, 3)


def test_Poly_EM():
    assert Poly(0, x).EM() == (0,)
    assert Poly(1, x).EM() == (0,)
    assert Poly(2*x**2 + x, x).EM() == (1,)

    assert Poly(x*y**7 + 2*x**2*y**3).EM('lex') == (1, 7)
    assert Poly(x*y**7 + 2*x**2*y**3).EM('grlex') == (2, 3)


def test_Poly_LT():
    assert Poly(0, x).LT() == ((0,), 0)
    assert Poly(1, x).LT() == ((0,), 1)
    assert Poly(2*x**2 + x, x).LT() == ((2,), 2)

    assert Poly(x*y**7 + 2*x**2*y**3).LT('lex') == ((2, 3), 2)
    assert Poly(x*y**7 + 2*x**2*y**3).LT('grlex') == ((1, 7), 1)

    assert LT(x*y**7 + 2*x**2*y**3, order='lex') == 2*x**2*y**3
    assert LT(x*y**7 + 2*x**2*y**3, order='grlex') == x*y**7


def test_Poly_ET():
    assert Poly(0, x).ET() == ((0,), 0)
    assert Poly(1, x).ET() == ((0,), 1)
    assert Poly(2*x**2 + x, x).ET() == ((1,), 1)

    assert Poly(x*y**7 + 2*x**2*y**3).ET('lex') == ((1, 7), 1)
    assert Poly(x*y**7 + 2*x**2*y**3).ET('grlex') == ((2, 3), 2)


def test_Poly_max_norm():
    assert Poly(-1, x).max_norm() == 1
    assert Poly( 0, x).max_norm() == 0
    assert Poly( 1, x).max_norm() == 1


def test_Poly_l1_norm():
    assert Poly(-1, x).l1_norm() == 1
    assert Poly( 0, x).l1_norm() == 0
    assert Poly( 1, x).l1_norm() == 1


def test_Poly_clear_denoms():
    coeff, poly = Poly(x + 2, x).clear_denoms()
    assert coeff == 1 and poly == Poly(
        x + 2, x, domain='ZZ') and poly.get_domain() == ZZ

    coeff, poly = Poly(x/2 + 1, x).clear_denoms()
    assert coeff == 2 and poly == Poly(
        x + 2, x, domain='QQ') and poly.get_domain() == QQ

    coeff, poly = Poly(2*x**2 + 3, modulus=5).clear_denoms()
    assert coeff == 1 and poly == Poly(
        2*x**2 + 3, x, modulus=5) and poly.get_domain() == FF(5)

    coeff, poly = Poly(x/2 + 1, x).clear_denoms(convert=True)
    assert coeff == 2 and poly == Poly(
        x + 2, x, domain='ZZ') and poly.get_domain() == ZZ

    coeff, poly = Poly(x/y + 1, x).clear_denoms(convert=True)
    assert coeff == y and poly == Poly(
        x + y, x, domain='ZZ[y]') and poly.get_domain() == ZZ[y]

    coeff, poly = Poly(x/3 + sqrt(2), x, domain='EX').clear_denoms()
    assert coeff == 3 and poly == Poly(
        x + 3*sqrt(2), x, domain='EX') and poly.get_domain() == EX

    coeff, poly = Poly(
        x/3 + sqrt(2), x, domain='EX').clear_denoms(convert=True)
    assert coeff == 3 and poly == Poly(
        x + 3*sqrt(2), x, domain='EX') and poly.get_domain() == EX


def test_Poly_rat_clear_denoms():
    f = Poly(x**2/y + 1, x)
    g = Poly(x**3 + y, x)

    assert f.rat_clear_denoms(g) == \
        (Poly(x**2 + y, x), Poly(y*x**3 + y**2, x))

    f = f.set_domain(EX)
    g = g.set_domain(EX)

    assert f.rat_clear_denoms(g) == (f, g)


def test_issue_20427():
    f = Poly(-117968192370600*18**(S(1)/3)/(217603955769048*(24201 +
        253*sqrt(9165))**(S(1)/3) + 2273005839412*sqrt(9165)*(24201 +
        253*sqrt(9165))**(S(1)/3)) - 15720318185*2**(S(2)/3)*3**(S(1)/3)*(24201
        + 253*sqrt(9165))**(S(2)/3)/(217603955769048*(24201 + 253*sqrt(9165))**
        (S(1)/3) + 2273005839412*sqrt(9165)*(24201 + 253*sqrt(9165))**(S(1)/3))
        + 15720318185*12**(S(1)/3)*(24201 + 253*sqrt(9165))**(S(2)/3)/(
        217603955769048*(24201 + 253*sqrt(9165))**(S(1)/3) + 2273005839412*
        sqrt(9165)*(24201 + 253*sqrt(9165))**(S(1)/3)) + 117968192370600*2**(
        S(1)/3)*3**(S(2)/3)/(217603955769048*(24201 + 253*sqrt(9165))**(S(1)/3)
        + 2273005839412*sqrt(9165)*(24201 + 253*sqrt(9165))**(S(1)/3)), x)
    assert f == Poly(0, x, domain='EX')


def test_Poly_integrate():
    assert Poly(x + 1).integrate() == Poly(x**2/2 + x)
    assert Poly(x + 1).integrate(x) == Poly(x**2/2 + x)
    assert Poly(x + 1).integrate((x, 1)) == Poly(x**2/2 + x)

    assert Poly(x*y + 1).integrate(x) == Poly(x**2*y/2 + x)
    assert Poly(x*y + 1).integrate(y) == Poly(x*y**2/2 + y)

    assert Poly(x*y + 1).integrate(x, x) == Poly(x**3*y/6 + x**2/2)
    assert Poly(x*y + 1).integrate(y, y) == Poly(x*y**3/6 + y**2/2)

    assert Poly(x*y + 1).integrate((x, 2)) == Poly(x**3*y/6 + x**2/2)
    assert Poly(x*y + 1).integrate((y, 2)) == Poly(x*y**3/6 + y**2/2)

    assert Poly(x*y + 1).integrate(x, y) == Poly(x**2*y**2/4 + x*y)
    assert Poly(x*y + 1).integrate(y, x) == Poly(x**2*y**2/4 + x*y)


def test_Poly_diff():
    assert Poly(x**2 + x).diff() == Poly(2*x + 1)
    assert Poly(x**2 + x).diff(x) == Poly(2*x + 1)
    assert Poly(x**2 + x).diff((x, 1)) == Poly(2*x + 1)

    assert Poly(x**2*y**2 + x*y).diff(x) == Poly(2*x*y**2 + y)
    assert Poly(x**2*y**2 + x*y).diff(y) == Poly(2*x**2*y + x)

    assert Poly(x**2*y**2 + x*y).diff(x, x) == Poly(2*y**2, x, y)
    assert Poly(x**2*y**2 + x*y).diff(y, y) == Poly(2*x**2, x, y)

    assert Poly(x**2*y**2 + x*y).diff((x, 2)) == Poly(2*y**2, x, y)
    assert Poly(x**2*y**2 + x*y).diff((y, 2)) == Poly(2*x**2, x, y)

    assert Poly(x**2*y**2 + x*y).diff(x, y) == Poly(4*x*y + 1)
    assert Poly(x**2*y**2 + x*y).diff(y, x) == Poly(4*x*y + 1)


def test_issue_9585():
    assert diff(Poly(x**2 + x)) == Poly(2*x + 1)
    assert diff(Poly(x**2 + x), x, evaluate=False) == \
        Derivative(Poly(x**2 + x), x)
    assert Derivative(Poly(x**2 + x), x).doit() == Poly(2*x + 1)


def test_Poly_eval():
    assert Poly(0, x).eval(7) == 0
    assert Poly(1, x).eval(7) == 1
    assert Poly(x, x).eval(7) == 7

    assert Poly(0, x).eval(0, 7) == 0
    assert Poly(1, x).eval(0, 7) == 1
    assert Poly(x, x).eval(0, 7) == 7

    assert Poly(0, x).eval(x, 7) == 0
    assert Poly(1, x).eval(x, 7) == 1
    assert Poly(x, x).eval(x, 7) == 7

    assert Poly(0, x).eval('x', 7) == 0
    assert Poly(1, x).eval('x', 7) == 1
    assert Poly(x, x).eval('x', 7) == 7

    raises(PolynomialError, lambda: Poly(1, x).eval(1, 7))
    raises(PolynomialError, lambda: Poly(1, x).eval(y, 7))
    raises(PolynomialError, lambda: Poly(1, x).eval('y', 7))

    assert Poly(123, x, y).eval(7) == Poly(123, y)
    assert Poly(2*y, x, y).eval(7) == Poly(2*y, y)
    assert Poly(x*y, x, y).eval(7) == Poly(7*y, y)

    assert Poly(123, x, y).eval(x, 7) == Poly(123, y)
    assert Poly(2*y, x, y).eval(x, 7) == Poly(2*y, y)
    assert Poly(x*y, x, y).eval(x, 7) == Poly(7*y, y)

    assert Poly(123, x, y).eval(y, 7) == Poly(123, x)
    assert Poly(2*y, x, y).eval(y, 7) == Poly(14, x)
    assert Poly(x*y, x, y).eval(y, 7) == Poly(7*x, x)

    assert Poly(x*y + y, x, y).eval({x: 7}) == Poly(8*y, y)
    assert Poly(x*y + y, x, y).eval({y: 7}) == Poly(7*x + 7, x)

    assert Poly(x*y + y, x, y).eval({x: 6, y: 7}) == 49
    assert Poly(x*y + y, x, y).eval({x: 7, y: 6}) == 48

    assert Poly(x*y + y, x, y).eval((6, 7)) == 49
    assert Poly(x*y + y, x, y).eval([6, 7]) == 49

    assert Poly(x + 1, domain='ZZ').eval(S.Half) == Rational(3, 2)
    assert Poly(x + 1, domain='ZZ').eval(sqrt(2)) == sqrt(2) + 1

    raises(ValueError, lambda: Poly(x*y + y, x, y).eval((6, 7, 8)))
    raises(DomainError, lambda: Poly(x + 1, domain='ZZ').eval(S.Half, auto=False))

    # issue 6344
    alpha = Symbol('alpha')
    result = (2*alpha*z - 2*alpha + z**2 + 3)/(z**2 - 2*z + 1)

    f = Poly(x**2 + (alpha - 1)*x - alpha + 1, x, domain='ZZ[alpha]')
    assert f.eval((z + 1)/(z - 1)) == result

    g = Poly(x**2 + (alpha - 1)*x - alpha + 1, x, y, domain='ZZ[alpha]')
    assert g.eval((z + 1)/(z - 1)) == Poly(result, y, domain='ZZ(alpha,z)')

def test_Poly___call__():
    f = Poly(2*x*y + 3*x + y + 2*z)

    assert f(2) == Poly(5*y + 2*z + 6)
    assert f(2, 5) == Poly(2*z + 31)
    assert f(2, 5, 7) == 45


def test_parallel_poly_from_expr():
    assert parallel_poly_from_expr(
        [x - 1, x**2 - 1], x)[0] == [Poly(x - 1, x), Poly(x**2 - 1, x)]
    assert parallel_poly_from_expr(
        [Poly(x - 1, x), x**2 - 1], x)[0] == [Poly(x - 1, x), Poly(x**2 - 1, x)]
    assert parallel_poly_from_expr(
        [x - 1, Poly(x**2 - 1, x)], x)[0] == [Poly(x - 1, x), Poly(x**2 - 1, x)]
    assert parallel_poly_from_expr([Poly(
        x - 1, x), Poly(x**2 - 1, x)], x)[0] == [Poly(x - 1, x), Poly(x**2 - 1, x)]

    assert parallel_poly_from_expr(
        [x - 1, x**2 - 1], x, y)[0] == [Poly(x - 1, x, y), Poly(x**2 - 1, x, y)]
    assert parallel_poly_from_expr([Poly(
        x - 1, x), x**2 - 1], x, y)[0] == [Poly(x - 1, x, y), Poly(x**2 - 1, x, y)]
    assert parallel_poly_from_expr([x - 1, Poly(
        x**2 - 1, x)], x, y)[0] == [Poly(x - 1, x, y), Poly(x**2 - 1, x, y)]
    assert parallel_poly_from_expr([Poly(x - 1, x), Poly(
        x**2 - 1, x)], x, y)[0] == [Poly(x - 1, x, y), Poly(x**2 - 1, x, y)]

    assert parallel_poly_from_expr(
        [x - 1, x**2 - 1])[0] == [Poly(x - 1, x), Poly(x**2 - 1, x)]
    assert parallel_poly_from_expr(
        [Poly(x - 1, x), x**2 - 1])[0] == [Poly(x - 1, x), Poly(x**2 - 1, x)]
    assert parallel_poly_from_expr(
        [x - 1, Poly(x**2 - 1, x)])[0] == [Poly(x - 1, x), Poly(x**2 - 1, x)]
    assert parallel_poly_from_expr(
        [Poly(x - 1, x), Poly(x**2 - 1, x)])[0] == [Poly(x - 1, x), Poly(x**2 - 1, x)]

    assert parallel_poly_from_expr(
        [1, x**2 - 1])[0] == [Poly(1, x), Poly(x**2 - 1, x)]
    assert parallel_poly_from_expr(
        [1, x**2 - 1])[0] == [Poly(1, x), Poly(x**2 - 1, x)]
    assert parallel_poly_from_expr(
        [1, Poly(x**2 - 1, x)])[0] == [Poly(1, x), Poly(x**2 - 1, x)]
    assert parallel_poly_from_expr(
        [1, Poly(x**2 - 1, x)])[0] == [Poly(1, x), Poly(x**2 - 1, x)]

    assert parallel_poly_from_expr(
        [x**2 - 1, 1])[0] == [Poly(x**2 - 1, x), Poly(1, x)]
    assert parallel_poly_from_expr(
        [x**2 - 1, 1])[0] == [Poly(x**2 - 1, x), Poly(1, x)]
    assert parallel_poly_from_expr(
        [Poly(x**2 - 1, x), 1])[0] == [Poly(x**2 - 1, x), Poly(1, x)]
    assert parallel_poly_from_expr(
        [Poly(x**2 - 1, x), 1])[0] == [Poly(x**2 - 1, x), Poly(1, x)]

    assert parallel_poly_from_expr([Poly(x, x, y), Poly(y, x, y)], x, y, order='lex')[0] == \
        [Poly(x, x, y, domain='ZZ'), Poly(y, x, y, domain='ZZ')]

    raises(PolificationFailed, lambda: parallel_poly_from_expr([0, 1]))


def test_pdiv():
    f, g = x**2 - y**2, x - y
    q, r = x + y, 0

    F, G, Q, R = [ Poly(h, x, y) for h in (f, g, q, r) ]

    assert F.pdiv(G) == (Q, R)
    assert F.prem(G) == R
    assert F.pquo(G) == Q
    assert F.pexquo(G) == Q

    assert pdiv(f, g) == (q, r)
    assert prem(f, g) == r
    assert pquo(f, g) == q
    assert pexquo(f, g) == q

    assert pdiv(f, g, x, y) == (q, r)
    assert prem(f, g, x, y) == r
    assert pquo(f, g, x, y) == q
    assert pexquo(f, g, x, y) == q

    assert pdiv(f, g, (x, y)) == (q, r)
    assert prem(f, g, (x, y)) == r
    assert pquo(f, g, (x, y)) == q
    assert pexquo(f, g, (x, y)) == q

    assert pdiv(F, G) == (Q, R)
    assert prem(F, G) == R
    assert pquo(F, G) == Q
    assert pexquo(F, G) == Q

    assert pdiv(f, g, polys=True) == (Q, R)
    assert prem(f, g, polys=True) == R
    assert pquo(f, g, polys=True) == Q
    assert pexquo(f, g, polys=True) == Q

    assert pdiv(F, G, polys=False) == (q, r)
    assert prem(F, G, polys=False) == r
    assert pquo(F, G, polys=False) == q
    assert pexquo(F, G, polys=False) == q

    raises(ComputationFailed, lambda: pdiv(4, 2))
    raises(ComputationFailed, lambda: prem(4, 2))
    raises(ComputationFailed, lambda: pquo(4, 2))
    raises(ComputationFailed, lambda: pexquo(4, 2))


def test_div():
    f, g = x**2 - y**2, x - y
    q, r = x + y, 0

    F, G, Q, R = [ Poly(h, x, y) for h in (f, g, q, r) ]

    assert F.div(G) == (Q, R)
    assert F.rem(G) == R
    assert F.quo(G) == Q
    assert F.exquo(G) == Q

    assert div(f, g) == (q, r)
    assert rem(f, g) == r
    assert quo(f, g) == q
    assert exquo(f, g) == q

    assert div(f, g, x, y) == (q, r)
    assert rem(f, g, x, y) == r
    assert quo(f, g, x, y) == q
    assert exquo(f, g, x, y) == q

    assert div(f, g, (x, y)) == (q, r)
    assert rem(f, g, (x, y)) == r
    assert quo(f, g, (x, y)) == q
    assert exquo(f, g, (x, y)) == q

    assert div(F, G) == (Q, R)
    assert rem(F, G) == R
    assert quo(F, G) == Q
    assert exquo(F, G) == Q

    assert div(f, g, polys=True) == (Q, R)
    assert rem(f, g, polys=True) == R
    assert quo(f, g, polys=True) == Q
    assert exquo(f, g, polys=True) == Q

    assert div(F, G, polys=False) == (q, r)
    assert rem(F, G, polys=False) == r
    assert quo(F, G, polys=False) == q
    assert exquo(F, G, polys=False) == q

    raises(ComputationFailed, lambda: div(4, 2))
    raises(ComputationFailed, lambda: rem(4, 2))
    raises(ComputationFailed, lambda: quo(4, 2))
    raises(ComputationFailed, lambda: exquo(4, 2))

    f, g = x**2 + 1, 2*x - 4

    qz, rz = 0, x**2 + 1
    qq, rq = x/2 + 1, 5

    assert div(f, g) == (qq, rq)
    assert div(f, g, auto=True) == (qq, rq)
    assert div(f, g, auto=False) == (qz, rz)
    assert div(f, g, domain=ZZ) == (qz, rz)
    assert div(f, g, domain=QQ) == (qq, rq)
    assert div(f, g, domain=ZZ, auto=True) == (qq, rq)
    assert div(f, g, domain=ZZ, auto=False) == (qz, rz)
    assert div(f, g, domain=QQ, auto=True) == (qq, rq)
    assert div(f, g, domain=QQ, auto=False) == (qq, rq)

    assert rem(f, g) == rq
    assert rem(f, g, auto=True) == rq
    assert rem(f, g, auto=False) == rz
    assert rem(f, g, domain=ZZ) == rz
    assert rem(f, g, domain=QQ) == rq
    assert rem(f, g, domain=ZZ, auto=True) == rq
    assert rem(f, g, domain=ZZ, auto=False) == rz
    assert rem(f, g, domain=QQ, auto=True) == rq
    assert rem(f, g, domain=QQ, auto=False) == rq

    assert quo(f, g) == qq
    assert quo(f, g, auto=True) == qq
    assert quo(f, g, auto=False) == qz
    assert quo(f, g, domain=ZZ) == qz
    assert quo(f, g, domain=QQ) == qq
    assert quo(f, g, domain=ZZ, auto=True) == qq
    assert quo(f, g, domain=ZZ, auto=False) == qz
    assert quo(f, g, domain=QQ, auto=True) == qq
    assert quo(f, g, domain=QQ, auto=False) == qq

    f, g, q = x**2, 2*x, x/2

    assert exquo(f, g) == q
    assert exquo(f, g, auto=True) == q
    raises(ExactQuotientFailed, lambda: exquo(f, g, auto=False))
    raises(ExactQuotientFailed, lambda: exquo(f, g, domain=ZZ))
    assert exquo(f, g, domain=QQ) == q
    assert exquo(f, g, domain=ZZ, auto=True) == q
    raises(ExactQuotientFailed, lambda: exquo(f, g, domain=ZZ, auto=False))
    assert exquo(f, g, domain=QQ, auto=True) == q
    assert exquo(f, g, domain=QQ, auto=False) == q

    f, g = Poly(x**2), Poly(x)

    q, r = f.div(g)
    assert q.get_domain().is_ZZ and r.get_domain().is_ZZ
    r = f.rem(g)
    assert r.get_domain().is_ZZ
    q = f.quo(g)
    assert q.get_domain().is_ZZ
    q = f.exquo(g)
    assert q.get_domain().is_ZZ

    f, g = Poly(x+y, x), Poly(2*x+y, x)
    q, r = f.div(g)
    assert q.get_domain().is_Frac and r.get_domain().is_Frac

    # https://github.com/sympy/sympy/issues/19579
    p = Poly(2+3*I, x, domain=ZZ_I)
    q = Poly(1-I, x, domain=ZZ_I)
    assert p.div(q, auto=False) == \
        (Poly(0, x, domain='ZZ_I'), Poly(2 + 3*I, x, domain='ZZ_I'))
    assert p.div(q, auto=True) == \
        (Poly(-S(1)/2 + 5*I/2, x, domain='QQ_I'), Poly(0, x, domain='QQ_I'))

    f = 5*x**2 + 10*x + 3
    g = 2*x + 2
    assert div(f, g, domain=ZZ) == (0, f)


def test_issue_7864():
    q, r = div(a, .408248290463863*a)
    assert abs(q - 2.44948974278318) < 1e-14
    assert r == 0


def test_gcdex():
    f, g = 2*x, x**2 - 16
    s, t, h = x/32, Rational(-1, 16), 1

    F, G, S, T, H = [ Poly(u, x, domain='QQ') for u in (f, g, s, t, h) ]

    assert F.half_gcdex(G) == (S, H)
    assert F.gcdex(G) == (S, T, H)
    assert F.invert(G) == S

    assert half_gcdex(f, g) == (s, h)
    assert gcdex(f, g) == (s, t, h)
    assert invert(f, g) == s

    assert half_gcdex(f, g, x) == (s, h)
    assert gcdex(f, g, x) == (s, t, h)
    assert invert(f, g, x) == s

    assert half_gcdex(f, g, (x,)) == (s, h)
    assert gcdex(f, g, (x,)) == (s, t, h)
    assert invert(f, g, (x,)) == s

    assert half_gcdex(F, G) == (S, H)
    assert gcdex(F, G) == (S, T, H)
    assert invert(F, G) == S

    assert half_gcdex(f, g, polys=True) == (S, H)
    assert gcdex(f, g, polys=True) == (S, T, H)
    assert invert(f, g, polys=True) == S

    assert half_gcdex(F, G, polys=False) == (s, h)
    assert gcdex(F, G, polys=False) == (s, t, h)
    assert invert(F, G, polys=False) == s

    assert half_gcdex(100, 2004) == (-20, 4)
    assert gcdex(100, 2004) == (-20, 1, 4)
    assert invert(3, 7) == 5

    raises(DomainError, lambda: half_gcdex(x + 1, 2*x + 1, auto=False))
    raises(DomainError, lambda: gcdex(x + 1, 2*x + 1, auto=False))
    raises(DomainError, lambda: invert(x + 1, 2*x + 1, auto=False))


def test_revert():
    f = Poly(1 - x**2/2 + x**4/24 - x**6/720)
    g = Poly(61*x**6/720 + 5*x**4/24 + x**2/2 + 1)

    assert f.revert(8) == g


def test_subresultants():
    f, g, h = x**2 - 2*x + 1, x**2 - 1, 2*x - 2
    F, G, H = Poly(f), Poly(g), Poly(h)

    assert F.subresultants(G) == [F, G, H]
    assert subresultants(f, g) == [f, g, h]
    assert subresultants(f, g, x) == [f, g, h]
    assert subresultants(f, g, (x,)) == [f, g, h]
    assert subresultants(F, G) == [F, G, H]
    assert subresultants(f, g, polys=True) == [F, G, H]
    assert subresultants(F, G, polys=False) == [f, g, h]

    raises(ComputationFailed, lambda: subresultants(4, 2))


def test_resultant():
    f, g, h = x**2 - 2*x + 1, x**2 - 1, 0
    F, G = Poly(f), Poly(g)

    assert F.resultant(G) == h
    assert resultant(f, g) == h
    assert resultant(f, g, x) == h
    assert resultant(f, g, (x,)) == h
    assert resultant(F, G) == h
    assert resultant(f, g, polys=True) == h
    assert resultant(F, G, polys=False) == h
    assert resultant(f, g, includePRS=True) == (h, [f, g, 2*x - 2])

    f, g, h = x - a, x - b, a - b
    F, G, H = Poly(f), Poly(g), Poly(h)

    assert F.resultant(G) == H
    assert resultant(f, g) == h
    assert resultant(f, g, x) == h
    assert resultant(f, g, (x,)) == h
    assert resultant(F, G) == H
    assert resultant(f, g, polys=True) == H
    assert resultant(F, G, polys=False) == h

    raises(ComputationFailed, lambda: resultant(4, 2))


def test_discriminant():
    f, g = x**3 + 3*x**2 + 9*x - 13, -11664
    F = Poly(f)

    assert F.discriminant() == g
    assert discriminant(f) == g
    assert discriminant(f, x) == g
    assert discriminant(f, (x,)) == g
    assert discriminant(F) == g
    assert discriminant(f, polys=True) == g
    assert discriminant(F, polys=False) == g

    f, g = a*x**2 + b*x + c, b**2 - 4*a*c
    F, G = Poly(f), Poly(g)

    assert F.discriminant() == G
    assert discriminant(f) == g
    assert discriminant(f, x, a, b, c) == g
    assert discriminant(f, (x, a, b, c)) == g
    assert discriminant(F) == G
    assert discriminant(f, polys=True) == G
    assert discriminant(F, polys=False) == g

    raises(ComputationFailed, lambda: discriminant(4))


def test_dispersion():
    # We test only the API here. For more mathematical
    # tests see the dedicated test file.
    fp = poly((x + 1)*(x + 2), x)
    assert sorted(fp.dispersionset()) == [0, 1]
    assert fp.dispersion() == 1

    fp = poly(x**4 - 3*x**2 + 1, x)
    gp = fp.shift(-3)
    assert sorted(fp.dispersionset(gp)) == [2, 3, 4]
    assert fp.dispersion(gp) == 4


def test_gcd_list():
    F = [x**3 - 1, x**2 - 1, x**2 - 3*x + 2]

    assert gcd_list(F) == x - 1
    assert gcd_list(F, polys=True) == Poly(x - 1)

    assert gcd_list([]) == 0
    assert gcd_list([1, 2]) == 1
    assert gcd_list([4, 6, 8]) == 2

    assert gcd_list([x*(y + 42) - x*y - x*42]) == 0

    gcd = gcd_list([], x)
    assert gcd.is_Number and gcd is S.Zero

    gcd = gcd_list([], x, polys=True)
    assert gcd.is_Poly and gcd.is_zero

    a = sqrt(2)
    assert gcd_list([a, -a]) == gcd_list([-a, a]) == a

    raises(ComputationFailed, lambda: gcd_list([], polys=True))


def test_lcm_list():
    F = [x**3 - 1, x**2 - 1, x**2 - 3*x + 2]

    assert lcm_list(F) == x**5 - x**4 - 2*x**3 - x**2 + x + 2
    assert lcm_list(F, polys=True) == Poly(x**5 - x**4 - 2*x**3 - x**2 + x + 2)

    assert lcm_list([]) == 1
    assert lcm_list([1, 2]) == 2
    assert lcm_list([4, 6, 8]) == 24

    assert lcm_list([x*(y + 42) - x*y - x*42]) == 0

    lcm = lcm_list([], x)
    assert lcm.is_Number and lcm is S.One

    lcm = lcm_list([], x, polys=True)
    assert lcm.is_Poly and lcm.is_one

    raises(ComputationFailed, lambda: lcm_list([], polys=True))


def test_gcd():
    f, g = x**3 - 1, x**2 - 1
    s, t = x**2 + x + 1, x + 1
    h, r = x - 1, x**4 + x**3 - x - 1

    F, G, S, T, H, R = [ Poly(u) for u in (f, g, s, t, h, r) ]

    assert F.cofactors(G) == (H, S, T)
    assert F.gcd(G) == H
    assert F.lcm(G) == R

    assert cofactors(f, g) == (h, s, t)
    assert gcd(f, g) == h
    assert lcm(f, g) == r

    assert cofactors(f, g, x) == (h, s, t)
    assert gcd(f, g, x) == h
    assert lcm(f, g, x) == r

    assert cofactors(f, g, (x,)) == (h, s, t)
    assert gcd(f, g, (x,)) == h
    assert lcm(f, g, (x,)) == r

    assert cofactors(F, G) == (H, S, T)
    assert gcd(F, G) == H
    assert lcm(F, G) == R

    assert cofactors(f, g, polys=True) == (H, S, T)
    assert gcd(f, g, polys=True) == H
    assert lcm(f, g, polys=True) == R

    assert cofactors(F, G, polys=False) == (h, s, t)
    assert gcd(F, G, polys=False) == h
    assert lcm(F, G, polys=False) == r

    f, g = 1.0*x**2 - 1.0, 1.0*x - 1.0
    h, s, t = g, 1.0*x + 1.0, 1.0

    assert cofactors(f, g) == (h, s, t)
    assert gcd(f, g) == h
    assert lcm(f, g) == f

    f, g = 1.0*x**2 - 1.0, 1.0*x - 1.0
    h, s, t = g, 1.0*x + 1.0, 1.0

    assert cofactors(f, g) == (h, s, t)
    assert gcd(f, g) == h
    assert lcm(f, g) == f

    assert cofactors(8, 6) == (2, 4, 3)
    assert gcd(8, 6) == 2
    assert lcm(8, 6) == 24

    f, g = x**2 - 3*x - 4, x**3 - 4*x**2 + x - 4
    l = x**4 - 3*x**3 - 3*x**2 - 3*x - 4
    h, s, t = x - 4, x + 1, x**2 + 1

    assert cofactors(f, g, modulus=11) == (h, s, t)
    assert gcd(f, g, modulus=11) == h
    assert lcm(f, g, modulus=11) == l

    f, g = x**2 + 8*x + 7, x**3 + 7*x**2 + x + 7
    l = x**4 + 8*x**3 + 8*x**2 + 8*x + 7
    h, s, t = x + 7, x + 1, x**2 + 1

    assert cofactors(f, g, modulus=11, symmetric=False) == (h, s, t)
    assert gcd(f, g, modulus=11, symmetric=False) == h
    assert lcm(f, g, modulus=11, symmetric=False) == l

    a, b = sqrt(2), -sqrt(2)
    assert gcd(a, b) == gcd(b, a) == sqrt(2)

    a, b = sqrt(-2), -sqrt(-2)
    assert gcd(a, b) == gcd(b, a) == sqrt(2)

    assert gcd(Poly(x - 2, x), Poly(I*x, x)) == Poly(1, x, domain=ZZ_I)

    raises(TypeError, lambda: gcd(x))
    raises(TypeError, lambda: lcm(x))

    f = Poly(-1, x)
    g = Poly(1, x)
    assert lcm(f, g) == Poly(1, x)

    f = Poly(0, x)
    g = Poly([1, 1], x)
    for i in (f, g):
        assert lcm(i, 0) == 0
        assert lcm(0, i) == 0
        assert lcm(i, f) == 0
        assert lcm(f, i) == 0

    f = 4*x**2 + x + 2
    pfz = Poly(f, domain=ZZ)
    pfq = Poly(f, domain=QQ)

    assert pfz.gcd(pfz) == pfz
    assert pfz.lcm(pfz) == pfz
    assert pfq.gcd(pfq) == pfq.monic()
    assert pfq.lcm(pfq) == pfq.monic()
    assert gcd(f, f) == f
    assert lcm(f, f) == f
    assert gcd(f, f, domain=QQ) == monic(f)
    assert lcm(f, f, domain=QQ) == monic(f)


def test_gcd_numbers_vs_polys():
    assert isinstance(gcd(3, 9), Integer)
    assert isinstance(gcd(3*x, 9), Integer)

    assert gcd(3, 9) == 3
    assert gcd(3*x, 9) == 3

    assert isinstance(gcd(Rational(3, 2), Rational(9, 4)), Rational)
    assert isinstance(gcd(Rational(3, 2)*x, Rational(9, 4)), Rational)

    assert gcd(Rational(3, 2), Rational(9, 4)) == Rational(3, 4)
    assert gcd(Rational(3, 2)*x, Rational(9, 4)) == 1

    assert isinstance(gcd(3.0, 9.0), Float)
    assert isinstance(gcd(3.0*x, 9.0), Float)

    assert gcd(3.0, 9.0) == 1.0
    assert gcd(3.0*x, 9.0) == 1.0

    # partial fix of 20597
    assert gcd(Mul(2, 3, evaluate=False), 2) == 2


def test_terms_gcd():
    assert terms_gcd(1) == 1
    assert terms_gcd(1, x) == 1

    assert terms_gcd(x - 1) == x - 1
    assert terms_gcd(-x - 1) == -x - 1

    assert terms_gcd(2*x + 3) == 2*x + 3
    assert terms_gcd(6*x + 4) == Mul(2, 3*x + 2, evaluate=False)

    assert terms_gcd(x**3*y + x*y**3) == x*y*(x**2 + y**2)
    assert terms_gcd(2*x**3*y + 2*x*y**3) == 2*x*y*(x**2 + y**2)
    assert terms_gcd(x**3*y/2 + x*y**3/2) == x*y/2*(x**2 + y**2)

    assert terms_gcd(x**3*y + 2*x*y**3) == x*y*(x**2 + 2*y**2)
    assert terms_gcd(2*x**3*y + 4*x*y**3) == 2*x*y*(x**2 + 2*y**2)
    assert terms_gcd(2*x**3*y/3 + 4*x*y**3/5) == x*y*Rational(2, 15)*(5*x**2 + 6*y**2)

    assert terms_gcd(2.0*x**3*y + 4.1*x*y**3) == x*y*(2.0*x**2 + 4.1*y**2)
    assert _aresame(terms_gcd(2.0*x + 3), 2.0*x + 3)

    assert terms_gcd((3 + 3*x)*(x + x*y), expand=False) == \
        (3*x + 3)*(x*y + x)
    assert terms_gcd((3 + 3*x)*(x + x*sin(3 + 3*y)), expand=False, deep=True) == \
        3*x*(x + 1)*(sin(Mul(3, y + 1, evaluate=False)) + 1)
    assert terms_gcd(sin(x + x*y), deep=True) == \
        sin(x*(y + 1))

    eq = Eq(2*x, 2*y + 2*z*y)
    assert terms_gcd(eq) == Eq(2*x, 2*y*(z + 1))
    assert terms_gcd(eq, deep=True) == Eq(2*x, 2*y*(z + 1))

    raises(TypeError, lambda: terms_gcd(x < 2))


def test_trunc():
    f, g = x**5 + 2*x**4 + 3*x**3 + 4*x**2 + 5*x + 6, x**5 - x**4 + x**2 - x
    F, G = Poly(f), Poly(g)

    assert F.trunc(3) == G
    assert trunc(f, 3) == g
    assert trunc(f, 3, x) == g
    assert trunc(f, 3, (x,)) == g
    assert trunc(F, 3) == G
    assert trunc(f, 3, polys=True) == G
    assert trunc(F, 3, polys=False) == g

    f, g = 6*x**5 + 5*x**4 + 4*x**3 + 3*x**2 + 2*x + 1, -x**4 + x**3 - x + 1
    F, G = Poly(f), Poly(g)

    assert F.trunc(3) == G
    assert trunc(f, 3) == g
    assert trunc(f, 3, x) == g
    assert trunc(f, 3, (x,)) == g
    assert trunc(F, 3) == G
    assert trunc(f, 3, polys=True) == G
    assert trunc(F, 3, polys=False) == g

    f = Poly(x**2 + 2*x + 3, modulus=5)

    assert f.trunc(2) == Poly(x**2 + 1, modulus=5)


def test_monic():
    f, g = 2*x - 1, x - S.Half
    F, G = Poly(f, domain='QQ'), Poly(g)

    assert F.monic() == G
    assert monic(f) == g
    assert monic(f, x) == g
    assert monic(f, (x,)) == g
    assert monic(F) == G
    assert monic(f, polys=True) == G
    assert monic(F, polys=False) == g

    raises(ComputationFailed, lambda: monic(4))

    assert monic(2*x**2 + 6*x + 4, auto=False) == x**2 + 3*x + 2
    raises(ExactQuotientFailed, lambda: monic(2*x + 6*x + 1, auto=False))

    assert monic(2.0*x**2 + 6.0*x + 4.0) == 1.0*x**2 + 3.0*x + 2.0
    assert monic(2*x**2 + 3*x + 4, modulus=5) == x**2 - x + 2


def test_content():
    f, F = 4*x + 2, Poly(4*x + 2)

    assert F.content() == 2
    assert content(f) == 2

    raises(ComputationFailed, lambda: content(4))

    f = Poly(2*x, modulus=3)

    assert f.content() == 1


def test_primitive():
    f, g = 4*x + 2, 2*x + 1
    F, G = Poly(f), Poly(g)

    assert F.primitive() == (2, G)
    assert primitive(f) == (2, g)
    assert primitive(f, x) == (2, g)
    assert primitive(f, (x,)) == (2, g)
    assert primitive(F) == (2, G)
    assert primitive(f, polys=True) == (2, G)
    assert primitive(F, polys=False) == (2, g)

    raises(ComputationFailed, lambda: primitive(4))

    f = Poly(2*x, modulus=3)
    g = Poly(2.0*x, domain=RR)

    assert f.primitive() == (1, f)
    assert g.primitive() == (1.0, g)

    assert primitive(S('-3*x/4 + y + 11/8')) == \
        S('(1/8, -6*x + 8*y + 11)')


def test_compose():
    f = x**12 + 20*x**10 + 150*x**8 + 500*x**6 + 625*x**4 - 2*x**3 - 10*x + 9
    g = x**4 - 2*x + 9
    h = x**3 + 5*x

    F, G, H = map(Poly, (f, g, h))

    assert G.compose(H) == F
    assert compose(g, h) == f
    assert compose(g, h, x) == f
    assert compose(g, h, (x,)) == f
    assert compose(G, H) == F
    assert compose(g, h, polys=True) == F
    assert compose(G, H, polys=False) == f

    assert F.decompose() == [G, H]
    assert decompose(f) == [g, h]
    assert decompose(f, x) == [g, h]
    assert decompose(f, (x,)) == [g, h]
    assert decompose(F) == [G, H]
    assert decompose(f, polys=True) == [G, H]
    assert decompose(F, polys=False) == [g, h]

    raises(ComputationFailed, lambda: compose(4, 2))
    raises(ComputationFailed, lambda: decompose(4))

    assert compose(x**2 - y**2, x - y, x, y) == x**2 - 2*x*y
    assert compose(x**2 - y**2, x - y, y, x) == -y**2 + 2*x*y


def test_shift():
    assert Poly(x**2 - 2*x + 1, x).shift(2) == Poly(x**2 + 2*x + 1, x)


def test_shift_list():
    assert Poly(x*y, [x,y]).shift_list([1,2]) == Poly((x+1)*(y+2), [x,y])


def test_transform():
    # Also test that 3-way unification is done correctly
    assert Poly(x**2 - 2*x + 1, x).transform(Poly(x + 1), Poly(x - 1)) == \
        Poly(4, x) == \
        cancel((x - 1)**2*(x**2 - 2*x + 1).subs(x, (x + 1)/(x - 1)))

    assert Poly(x**2 - x/2 + 1, x).transform(Poly(x + 1), Poly(x - 1)) == \
        Poly(3*x**2/2 + Rational(5, 2), x) == \
        cancel((x - 1)**2*(x**2 - x/2 + 1).subs(x, (x + 1)/(x - 1)))

    assert Poly(x**2 - 2*x + 1, x).transform(Poly(x + S.Half), Poly(x - 1)) == \
        Poly(Rational(9, 4), x) == \
        cancel((x - 1)**2*(x**2 - 2*x + 1).subs(x, (x + S.Half)/(x - 1)))

    assert Poly(x**2 - 2*x + 1, x).transform(Poly(x + 1), Poly(x - S.Half)) == \
        Poly(Rational(9, 4), x) == \
        cancel((x - S.Half)**2*(x**2 - 2*x + 1).subs(x, (x + 1)/(x - S.Half)))

    # Unify ZZ, QQ, and RR
    assert Poly(x**2 - 2*x + 1, x).transform(Poly(x + 1.0), Poly(x - S.Half)) == \
        Poly(Rational(9, 4), x, domain='RR') == \
        cancel((x - S.Half)**2*(x**2 - 2*x + 1).subs(x, (x + 1.0)/(x - S.Half)))

    raises(ValueError, lambda: Poly(x*y).transform(Poly(x + 1), Poly(x - 1)))
    raises(ValueError, lambda: Poly(x).transform(Poly(y + 1), Poly(x - 1)))
    raises(ValueError, lambda: Poly(x).transform(Poly(x + 1), Poly(y - 1)))
    raises(ValueError, lambda: Poly(x).transform(Poly(x*y + 1), Poly(x - 1)))
    raises(ValueError, lambda: Poly(x).transform(Poly(x + 1), Poly(x*y - 1)))


def test_sturm():
    f, F = x, Poly(x, domain='QQ')
    g, G = 1, Poly(1, x, domain='QQ')

    assert F.sturm() == [F, G]
    assert sturm(f) == [f, g]
    assert sturm(f, x) == [f, g]
    assert sturm(f, (x,)) == [f, g]
    assert sturm(F) == [F, G]
    assert sturm(f, polys=True) == [F, G]
    assert sturm(F, polys=False) == [f, g]

    raises(ComputationFailed, lambda: sturm(4))
    raises(DomainError, lambda: sturm(f, auto=False))

    f = Poly(S(1024)/(15625*pi**8)*x**5
           - S(4096)/(625*pi**8)*x**4
           + S(32)/(15625*pi**4)*x**3
           - S(128)/(625*pi**4)*x**2
           + Rational(1, 62500)*x
           - Rational(1, 625), x, domain='ZZ(pi)')

    assert sturm(f) == \
        [Poly(x**3 - 100*x**2 + pi**4/64*x - 25*pi**4/16, x, domain='ZZ(pi)'),
         Poly(3*x**2 - 200*x + pi**4/64, x, domain='ZZ(pi)'),
         Poly((Rational(20000, 9) - pi**4/96)*x + 25*pi**4/18, x, domain='ZZ(pi)'),
         Poly((-3686400000000*pi**4 - 11520000*pi**8 - 9*pi**12)/(26214400000000 - 245760000*pi**4 + 576*pi**8), x, domain='ZZ(pi)')]


def test_gff():
    f = x**5 + 2*x**4 - x**3 - 2*x**2

    assert Poly(f).gff_list() == [(Poly(x), 1), (Poly(x + 2), 4)]
    assert gff_list(f) == [(x, 1), (x + 2, 4)]

    raises(NotImplementedError, lambda: gff(f))

    f = x*(x - 1)**3*(x - 2)**2*(x - 4)**2*(x - 5)

    assert Poly(f).gff_list() == [(
        Poly(x**2 - 5*x + 4), 1), (Poly(x**2 - 5*x + 4), 2), (Poly(x), 3)]
    assert gff_list(f) == [(x**2 - 5*x + 4, 1), (x**2 - 5*x + 4, 2), (x, 3)]

    raises(NotImplementedError, lambda: gff(f))


def test_norm():
    a, b = sqrt(2), sqrt(3)
    f = Poly(a*x + b*y, x, y, extension=(a, b))
    assert f.norm() == Poly(4*x**4 - 12*x**2*y**2 + 9*y**4, x, y, domain='QQ')


def test_sqf_norm():
    assert sqf_norm(x**2 - 2, extension=sqrt(3)) == \
        ([1], x**2 - 2*sqrt(3)*x + 1, x**4 - 10*x**2 + 1)
    assert sqf_norm(x**2 - 3, extension=sqrt(2)) == \
        ([1], x**2 - 2*sqrt(2)*x - 1, x**4 - 10*x**2 + 1)

    assert Poly(x**2 - 2, extension=sqrt(3)).sqf_norm() == \
        ([1], Poly(x**2 - 2*sqrt(3)*x + 1, x, extension=sqrt(3)),
              Poly(x**4 - 10*x**2 + 1, x, domain='QQ'))

    assert Poly(x**2 - 3, extension=sqrt(2)).sqf_norm() == \
        ([1], Poly(x**2 - 2*sqrt(2)*x - 1, x, extension=sqrt(2)),
              Poly(x**4 - 10*x**2 + 1, x, domain='QQ'))


def test_sqf():
    f = x**5 - x**3 - x**2 + 1
    g = x**3 + 2*x**2 + 2*x + 1
    h = x - 1

    p = x**4 + x**3 - x - 1

    F, G, H, P = map(Poly, (f, g, h, p))

    assert F.sqf_part() == P
    assert sqf_part(f) == p
    assert sqf_part(f, x) == p
    assert sqf_part(f, (x,)) == p
    assert sqf_part(F) == P
    assert sqf_part(f, polys=True) == P
    assert sqf_part(F, polys=False) == p

    assert F.sqf_list() == (1, [(G, 1), (H, 2)])
    assert sqf_list(f) == (1, [(g, 1), (h, 2)])
    assert sqf_list(f, x) == (1, [(g, 1), (h, 2)])
    assert sqf_list(f, (x,)) == (1, [(g, 1), (h, 2)])
    assert sqf_list(F) == (1, [(G, 1), (H, 2)])
    assert sqf_list(f, polys=True) == (1, [(G, 1), (H, 2)])
    assert sqf_list(F, polys=False) == (1, [(g, 1), (h, 2)])

    assert F.sqf_list_include() == [(G, 1), (H, 2)]

    raises(ComputationFailed, lambda: sqf_part(4))

    assert sqf(1) == 1
    assert sqf_list(1) == (1, [])

    assert sqf((2*x**2 + 2)**7) == 128*(x**2 + 1)**7

    assert sqf(f) == g*h**2
    assert sqf(f, x) == g*h**2
    assert sqf(f, (x,)) == g*h**2

    d = x**2 + y**2

    assert sqf(f/d) == (g*h**2)/d
    assert sqf(f/d, x) == (g*h**2)/d
    assert sqf(f/d, (x,)) == (g*h**2)/d

    assert sqf(x - 1) == x - 1
    assert sqf(-x - 1) == -x - 1

    assert sqf(x - 1) == x - 1
    assert sqf(6*x - 10) == Mul(2, 3*x - 5, evaluate=False)

    assert sqf((6*x - 10)/(3*x - 6)) == Rational(2, 3)*((3*x - 5)/(x - 2))
    assert sqf(Poly(x**2 - 2*x + 1)) == (x - 1)**2

    f = 3 + x - x*(1 + x) + x**2

    assert sqf(f) == 3

    f = (x**2 + 2*x + 1)**20000000000

    assert sqf(f) == (x + 1)**40000000000
    assert sqf_list(f) == (1, [(x + 1, 40000000000)])

    # https://github.com/sympy/sympy/issues/26497
    assert sqf(expand(((y - 2)**2 * (y + 2) * (x + 1)))) == \
        (y - 2)**2 * expand((y + 2) * (x + 1))
    assert sqf(expand(((y - 2)**2 * (y + 2) * (z + 1)))) == \
        (y - 2)**2 * expand((y + 2) * (z + 1))
    assert sqf(expand(((y - I)**2 * (y + I) * (x + 1)))) == \
        (y - I)**2 * expand((y + I) * (x + 1))
    assert sqf(expand(((y - I)**2 * (y + I) * (z + 1)))) == \
        (y - I)**2 * expand((y + I) * (z + 1))

    # Check that factors are combined and sorted.
    p = (x - 2)**2*(x - 1)*(x + y)**2*(y - 2)**2*(y - 1)
    assert Poly(p).sqf_list() == (1, [
        (Poly(x*y - x - y + 1), 1),
        (Poly(x**2*y - 2*x**2 + x*y**2 - 4*x*y + 4*x - 2*y**2 + 4*y), 2)
    ])


def test_factor():
    f = x**5 - x**3 - x**2 + 1

    u = x + 1
    v = x - 1
    w = x**2 + x + 1

    F, U, V, W = map(Poly, (f, u, v, w))

    assert F.factor_list() == (1, [(U, 1), (V, 2), (W, 1)])
    assert factor_list(f) == (1, [(u, 1), (v, 2), (w, 1)])
    assert factor_list(f, x) == (1, [(u, 1), (v, 2), (w, 1)])
    assert factor_list(f, (x,)) == (1, [(u, 1), (v, 2), (w, 1)])
    assert factor_list(F) == (1, [(U, 1), (V, 2), (W, 1)])
    assert factor_list(f, polys=True) == (1, [(U, 1), (V, 2), (W, 1)])
    assert factor_list(F, polys=False) == (1, [(u, 1), (v, 2), (w, 1)])

    assert F.factor_list_include() == [(U, 1), (V, 2), (W, 1)]

    assert factor_list(1) == (1, [])
    assert factor_list(6) == (6, [])
    assert factor_list(sqrt(3), x) == (sqrt(3), [])
    assert factor_list((-1)**x, x) == (1, [(-1, x)])
    assert factor_list((2*x)**y, x) == (1, [(2, y), (x, y)])
    assert factor_list(sqrt(x*y), x) == (1, [(x*y, S.Half)])

    assert factor(6) == 6 and factor(6).is_Integer

    assert factor_list(3*x) == (3, [(x, 1)])
    assert factor_list(3*x**2) == (3, [(x, 2)])

    assert factor(3*x) == 3*x
    assert factor(3*x**2) == 3*x**2

    assert factor((2*x**2 + 2)**7) == 128*(x**2 + 1)**7

    assert factor(f) == u*v**2*w
    assert factor(f, x) == u*v**2*w
    assert factor(f, (x,)) == u*v**2*w

    g, p, q, r = x**2 - y**2, x - y, x + y, x**2 + 1

    assert factor(f/g) == (u*v**2*w)/(p*q)
    assert factor(f/g, x) == (u*v**2*w)/(p*q)
    assert factor(f/g, (x,)) == (u*v**2*w)/(p*q)

    p = Symbol('p', positive=True)
    i = Symbol('i', integer=True)
    r = Symbol('r', real=True)

    assert factor(sqrt(x*y)).is_Pow is True

    assert factor(sqrt(3*x**2 - 3)) == sqrt(3)*sqrt((x - 1)*(x + 1))
    assert factor(sqrt(3*x**2 + 3)) == sqrt(3)*sqrt(x**2 + 1)

    assert factor((y*x**2 - y)**i) == y**i*(x - 1)**i*(x + 1)**i
    assert factor((y*x**2 + y)**i) == y**i*(x**2 + 1)**i

    assert factor((y*x**2 - y)**t) == (y*(x - 1)*(x + 1))**t
    assert factor((y*x**2 + y)**t) == (y*(x**2 + 1))**t

    f = sqrt(expand((r**2 + 1)*(p + 1)*(p - 1)*(p - 2)**3))
    g = sqrt((p - 2)**3*(p - 1))*sqrt(p + 1)*sqrt(r**2 + 1)

    assert factor(f) == g
    assert factor(g) == g

    g = (x - 1)**5*(r**2 + 1)
    f = sqrt(expand(g))

    assert factor(f) == sqrt(g)

    f = Poly(sin(1)*x + 1, x, domain=EX)

    assert f.factor_list() == (1, [(f, 1)])

    f = x**4 + 1

    assert factor(f) == f
    assert factor(f, extension=I) == (x**2 - I)*(x**2 + I)
    assert factor(f, gaussian=True) == (x**2 - I)*(x**2 + I)
    assert factor(
        f, extension=sqrt(2)) == (x**2 + sqrt(2)*x + 1)*(x**2 - sqrt(2)*x + 1)

    assert factor(x**2 + 4*I*x - 4) == (x + 2*I)**2

    f = x**2 + 2*I*x - 4

    assert factor(f) == f

    f = 8192*x**2 + x*(22656 + 175232*I) - 921416 + 242313*I
    f_zzi = I*(x*(64 - 64*I) + 773 + 596*I)**2
    f_qqi = 8192*(x + S(177)/128 + 1369*I/128)**2

    assert factor(f) == f_zzi
    assert factor(f, domain=ZZ_I) == f_zzi
    assert factor(f, domain=QQ_I) == f_qqi

    f = x**2 + 2*sqrt(2)*x + 2

    assert factor(f, extension=sqrt(2)) == (x + sqrt(2))**2
    assert factor(f**3, extension=sqrt(2)) == (x + sqrt(2))**6

    assert factor(x**2 - 2*y**2, extension=sqrt(2)) == \
        (x + sqrt(2)*y)*(x - sqrt(2)*y)
    assert factor(2*x**2 - 4*y**2, extension=sqrt(2)) == \
        2*((x + sqrt(2)*y)*(x - sqrt(2)*y))

    assert factor(x - 1) == x - 1
    assert factor(-x - 1) == -x - 1

    assert factor(x - 1) == x - 1

    assert factor(6*x - 10) == Mul(2, 3*x - 5, evaluate=False)

    assert factor(x**11 + x + 1, modulus=65537, symmetric=True) == \
        (x**2 + x + 1)*(x**9 - x**8 + x**6 - x**5 + x**3 - x** 2 + 1)
    assert factor(x**11 + x + 1, modulus=65537, symmetric=False) == \
        (x**2 + x + 1)*(x**9 + 65536*x**8 + x**6 + 65536*x**5 +
         x**3 + 65536*x** 2 + 1)

    f = x/pi + x*sin(x)/pi
    g = y/(pi**2 + 2*pi + 1) + y*sin(x)/(pi**2 + 2*pi + 1)

    assert factor(f) == x*(sin(x) + 1)/pi
    assert factor(g) == y*(sin(x) + 1)/(pi + 1)**2

    assert factor(Eq(
        x**2 + 2*x + 1, x**3 + 1)) == Eq((x + 1)**2, (x + 1)*(x**2 - x + 1))

    f = (x**2 - 1)/(x**2 + 4*x + 4)

    assert factor(f) == (x + 1)*(x - 1)/(x + 2)**2
    assert factor(f, x) == (x + 1)*(x - 1)/(x + 2)**2

    f = 3 + x - x*(1 + x) + x**2

    assert factor(f) == 3
    assert factor(f, x) == 3

    assert factor(1/(x**2 + 2*x + 1/x) - 1) == -((1 - x + 2*x**2 +
                  x**3)/(1 + 2*x**2 + x**3))

    assert factor(f, expand=False) == f
    raises(PolynomialError, lambda: factor(f, x, expand=False))

    raises(FlagError, lambda: factor(x**2 - 1, polys=True))

    assert factor([x, Eq(x**2 - y**2, Tuple(x**2 - z**2, 1/x + 1/y))]) == \
        [x, Eq((x - y)*(x + y), Tuple((x - z)*(x + z), (x + y)/x/y))]

    assert not isinstance(
        Poly(x**3 + x + 1).factor_list()[1][0][0], PurePoly) is True
    assert isinstance(
        PurePoly(x**3 + x + 1).factor_list()[1][0][0], PurePoly) is True

    assert factor(sqrt(-x)) == sqrt(-x)

    # issue 5917
    e = (-2*x*(-x + 1)*(x - 1)*(-x*(-x + 1)*(x - 1) - x*(x - 1)**2)*(x**2*(x -
    1) - x*(x - 1) - x) - (-2*x**2*(x - 1)**2 - x*(-x + 1)*(-x*(-x + 1) +
    x*(x - 1)))*(x**2*(x - 1)**4 - x*(-x*(-x + 1)*(x - 1) - x*(x - 1)**2)))
    assert factor(e) == 0

    # deep option
    assert factor(sin(x**2 + x) + x, deep=True) == sin(x*(x + 1)) + x
    assert factor(sin(x**2 + x)*x, deep=True) == sin(x*(x + 1))*x

    assert factor(sqrt(x**2)) == sqrt(x**2)

    # issue 13149
    assert factor(expand((0.5*x+1)*(0.5*y+1))) == Mul(1.0, 0.5*x + 1.0,
        0.5*y + 1.0, evaluate = False)
    assert factor(expand((0.5*x+0.5)**2)) == 0.25*(1.0*x + 1.0)**2

    eq = x**2*y**2 + 11*x**2*y + 30*x**2 + 7*x*y**2 + 77*x*y + 210*x + 12*y**2 + 132*y + 360
    assert factor(eq, x) == (x + 3)*(x + 4)*(y**2 + 11*y + 30)
    assert factor(eq, x, deep=True) == (x + 3)*(x + 4)*(y**2 + 11*y + 30)
    assert factor(eq, y, deep=True) == (y + 5)*(y + 6)*(x**2 + 7*x + 12)

    # fraction option
    f = 5*x + 3*exp(2 - 7*x)
    assert factor(f, deep=True) == factor(f, deep=True, fraction=True)
    assert factor(f, deep=True, fraction=False) == 5*x + 3*exp(2)*exp(-7*x)

    assert factor_list(x**3 - x*y**2, t, w, x) == (
        1, [(x, 1), (x - y, 1), (x + y, 1)])
    assert factor_list((x+1)*(x**6-1)) == (
        1, [(x - 1, 1), (x + 1, 2), (x**2 - x + 1, 1), (x**2 + x + 1, 1)])

    # https://github.com/sympy/sympy/issues/24952
    s2, s2p, s2n = sqrt(2), 1 + sqrt(2), 1 - sqrt(2)
    pip, pin = 1 + pi, 1 - pi
    assert factor_list(s2p*s2n) == (-1, [(-s2n, 1), (s2p, 1)])
    assert factor_list(pip*pin) == (-1, [(-pin, 1), (pip, 1)])
    # Not sure about this one. Maybe coeff should be 1 or -1?
    assert factor_list(s2*s2n) == (-s2, [(-s2n, 1)])
    assert factor_list(pi*pin) == (-1, [(-pin, 1), (pi, 1)])
    assert factor_list(s2p*s2n, x) == (s2p*s2n, [])
    assert factor_list(pip*pin, x) == (pip*pin, [])
    assert factor_list(s2*s2n, x) == (s2*s2n, [])
    assert factor_list(pi*pin, x) == (pi*pin, [])
    assert factor_list((x - sqrt(2)*pi)*(x + sqrt(2)*pi), x) == (
        1, [(x - sqrt(2)*pi, 1), (x + sqrt(2)*pi, 1)])

    # https://github.com/sympy/sympy/issues/26497
    p = ((y - I)**2 * (y + I) * (x + 1))
    assert factor(expand(p)) == p

    p = ((x - I)**2 * (x + I) * (y + 1))
    assert factor(expand(p)) == p

    p = (y + 1)**2*(y + sqrt(2))**2*(x**2 + x + 2 + 3*sqrt(2))**2
    assert factor(expand(p), extension=True) == p

    e = (
        -x**2*y**4/(y**2 + 1) + 2*I*x**2*y**3/(y**2 + 1) + 2*I*x**2*y/(y**2 + 1) +
        x**2/(y**2 + 1) - 2*x*y**4/(y**2 + 1) + 4*I*x*y**3/(y**2 + 1) +
        4*I*x*y/(y**2 + 1) + 2*x/(y**2 + 1) - y**4 - y**4/(y**2 + 1) + 2*I*y**3
        + 2*I*y**3/(y**2 + 1) + 2*I*y + 2*I*y/(y**2 + 1) + 1 + 1/(y**2 + 1)
    )
    assert factor(e) == -(y - I)**3*(y + I)*(x**2 + 2*x + y**2 + 2)/(y**2 + 1)

    # issue 27506
    e = (I*t*x*y - 3*I*t - I*x*y*z - 6*x*y + 3*I*z + 18)
    assert factor(e) == -I*(x*y - 3)*(-t + z - 6*I)

    e = (8*x**2*z**2 - 32*x**2*z*t + 24*x**2*t**2 - 4*I*x*y*z**2 + 16*I*x*y*z*t -
         12*I*x*y*t**2 + z**4 - 8*z**3*t + 22*z**2*t**2 - 24*z*t**3 + 9*t**4)
    assert factor(e) == (-3*t + z)*(-t + z)*(3*t**2 - 4*t*z + 8*x**2 - 4*I*x*y + z**2)


def test_factor_large():
    f = (x**2 + 4*x + 4)**10000000*(x**2 + 1)*(x**2 + 2*x + 1)**1234567
    g = ((x**2 + 2*x + 1)**3000*y**2 + (x**2 + 2*x + 1)**3000*2*y + (
        x**2 + 2*x + 1)**3000)

    assert factor(f) == (x + 2)**20000000*(x**2 + 1)*(x + 1)**2469134
    assert factor(g) == (x + 1)**6000*(y + 1)**2

    assert factor_list(
        f) == (1, [(x + 1, 2469134), (x + 2, 20000000), (x**2 + 1, 1)])
    assert factor_list(g) == (1, [(y + 1, 2), (x + 1, 6000)])

    f = (x**2 - y**2)**200000*(x**7 + 1)
    g = (x**2 + y**2)**200000*(x**7 + 1)

    assert factor(f) == \
        (x + 1)*(x - y)**200000*(x + y)**200000*(x**6 - x**5 +
         x**4 - x**3 + x**2 - x + 1)
    assert factor(g, gaussian=True) == \
        (x + 1)*(x - I*y)**200000*(x + I*y)**200000*(x**6 - x**5 +
         x**4 - x**3 + x**2 - x + 1)

    assert factor_list(f) == \
        (1, [(x + 1, 1), (x - y, 200000), (x + y, 200000), (x**6 -
         x**5 + x**4 - x**3 + x**2 - x + 1, 1)])
    assert factor_list(g, gaussian=True) == \
        (1, [(x + 1, 1), (x - I*y, 200000), (x + I*y, 200000), (
            x**6 - x**5 + x**4 - x**3 + x**2 - x + 1, 1)])


def test_factor_noeval():
    assert factor(6*x - 10) == Mul(2, 3*x - 5, evaluate=False)
    assert factor((6*x - 10)/(3*x - 6)) == Mul(Rational(2, 3), 3*x - 5, 1/(x - 2))


def test_intervals():
    assert intervals(0) == []
    assert intervals(1) == []

    assert intervals(x, sqf=True) == [(0, 0)]
    assert intervals(x) == [((0, 0), 1)]

    assert intervals(x**128) == [((0, 0), 128)]
    assert intervals([x**2, x**4]) == [((0, 0), {0: 2, 1: 4})]

    f = Poly((x*Rational(2, 5) - Rational(17, 3))*(4*x + Rational(1, 257)))

    assert f.intervals(sqf=True) == [(-1, 0), (14, 15)]
    assert f.intervals() == [((-1, 0), 1), ((14, 15), 1)]

    assert f.intervals(fast=True, sqf=True) == [(-1, 0), (14, 15)]
    assert f.intervals(fast=True) == [((-1, 0), 1), ((14, 15), 1)]

    assert f.intervals(eps=Rational(1, 10)) == f.intervals(eps=0.1) == \
        [((Rational(-1, 258), 0), 1), ((Rational(85, 6), Rational(85, 6)), 1)]
    assert f.intervals(eps=Rational(1, 100)) == f.intervals(eps=0.01) == \
        [((Rational(-1, 258), 0), 1), ((Rational(85, 6), Rational(85, 6)), 1)]
    assert f.intervals(eps=Rational(1, 1000)) == f.intervals(eps=0.001) == \
        [((Rational(-1, 1002), 0), 1), ((Rational(85, 6), Rational(85, 6)), 1)]
    assert f.intervals(eps=Rational(1, 10000)) == f.intervals(eps=0.0001) == \
        [((Rational(-1, 1028), Rational(-1, 1028)), 1), ((Rational(85, 6), Rational(85, 6)), 1)]

    f = (x*Rational(2, 5) - Rational(17, 3))*(4*x + Rational(1, 257))

    assert intervals(f, sqf=True) == [(-1, 0), (14, 15)]
    assert intervals(f) == [((-1, 0), 1), ((14, 15), 1)]

    assert intervals(f, eps=Rational(1, 10)) == intervals(f, eps=0.1) == \
        [((Rational(-1, 258), 0), 1), ((Rational(85, 6), Rational(85, 6)), 1)]
    assert intervals(f, eps=Rational(1, 100)) == intervals(f, eps=0.01) == \
        [((Rational(-1, 258), 0), 1), ((Rational(85, 6), Rational(85, 6)), 1)]
    assert intervals(f, eps=Rational(1, 1000)) == intervals(f, eps=0.001) == \
        [((Rational(-1, 1002), 0), 1), ((Rational(85, 6), Rational(85, 6)), 1)]
    assert intervals(f, eps=Rational(1, 10000)) == intervals(f, eps=0.0001) == \
        [((Rational(-1, 1028), Rational(-1, 1028)), 1), ((Rational(85, 6), Rational(85, 6)), 1)]

    f = Poly((x**2 - 2)*(x**2 - 3)**7*(x + 1)*(7*x + 3)**3)

    assert f.intervals() == \
        [((-2, Rational(-3, 2)), 7), ((Rational(-3, 2), -1), 1),
         ((-1, -1), 1), ((-1, 0), 3),
         ((1, Rational(3, 2)), 1), ((Rational(3, 2), 2), 7)]

    assert intervals([x**5 - 200, x**5 - 201]) == \
        [((Rational(75, 26), Rational(101, 35)), {0: 1}), ((Rational(309, 107), Rational(26, 9)), {1: 1})]

    assert intervals([x**5 - 200, x**5 - 201], fast=True) == \
        [((Rational(75, 26), Rational(101, 35)), {0: 1}), ((Rational(309, 107), Rational(26, 9)), {1: 1})]

    assert intervals([x**2 - 200, x**2 - 201]) == \
        [((Rational(-71, 5), Rational(-85, 6)), {1: 1}), ((Rational(-85, 6), -14), {0: 1}),
         ((14, Rational(85, 6)), {0: 1}), ((Rational(85, 6), Rational(71, 5)), {1: 1})]

    assert intervals([x + 1, x + 2, x - 1, x + 1, 1, x - 1, x - 1, (x - 2)**2]) == \
        [((-2, -2), {1: 1}), ((-1, -1), {0: 1, 3: 1}), ((1, 1), {2:
          1, 5: 1, 6: 1}), ((2, 2), {7: 2})]

    f, g, h = x**2 - 2, x**4 - 4*x**2 + 4, x - 1

    assert intervals(f, inf=Rational(7, 4), sqf=True) == []
    assert intervals(f, inf=Rational(7, 5), sqf=True) == [(Rational(7, 5), Rational(3, 2))]
    assert intervals(f, sup=Rational(7, 4), sqf=True) == [(-2, -1), (1, Rational(3, 2))]
    assert intervals(f, sup=Rational(7, 5), sqf=True) == [(-2, -1)]

    assert intervals(g, inf=Rational(7, 4)) == []
    assert intervals(g, inf=Rational(7, 5)) == [((Rational(7, 5), Rational(3, 2)), 2)]
    assert intervals(g, sup=Rational(7, 4)) == [((-2, -1), 2), ((1, Rational(3, 2)), 2)]
    assert intervals(g, sup=Rational(7, 5)) == [((-2, -1), 2)]

    assert intervals([g, h], inf=Rational(7, 4)) == []
    assert intervals([g, h], inf=Rational(7, 5)) == [((Rational(7, 5), Rational(3, 2)), {0: 2})]
    assert intervals([g, h], sup=S(
        7)/4) == [((-2, -1), {0: 2}), ((1, 1), {1: 1}), ((1, Rational(3, 2)), {0: 2})]
    assert intervals(
        [g, h], sup=Rational(7, 5)) == [((-2, -1), {0: 2}), ((1, 1), {1: 1})]

    assert intervals([x + 2, x**2 - 2]) == \
        [((-2, -2), {0: 1}), ((-2, -1), {1: 1}), ((1, 2), {1: 1})]
    assert intervals([x + 2, x**2 - 2], strict=True) == \
        [((-2, -2), {0: 1}), ((Rational(-3, 2), -1), {1: 1}), ((1, 2), {1: 1})]

    f = 7*z**4 - 19*z**3 + 20*z**2 + 17*z + 20

    assert intervals(f) == []

    real_part, complex_part = intervals(f, all=True, sqf=True)

    assert real_part == []
    assert all(re(a) < re(r) < re(b) and im(
        a) < im(r) < im(b) for (a, b), r in zip(complex_part, nroots(f)))

    assert complex_part == [(Rational(-40, 7) - I*40/7, 0),
                            (Rational(-40, 7), I*40/7),
                            (I*Rational(-40, 7), Rational(40, 7)),
                            (0, Rational(40, 7) + I*40/7)]

    real_part, complex_part = intervals(f, all=True, sqf=True, eps=Rational(1, 10))

    assert real_part == []
    assert all(re(a) < re(r) < re(b) and im(
        a) < im(r) < im(b) for (a, b), r in zip(complex_part, nroots(f)))

    raises(ValueError, lambda: intervals(x**2 - 2, eps=10**-100000))
    raises(ValueError, lambda: Poly(x**2 - 2).intervals(eps=10**-100000))
    raises(
        ValueError, lambda: intervals([x**2 - 2, x**2 - 3], eps=10**-100000))


def test_refine_root():
    f = Poly(x**2 - 2)

    assert f.refine_root(1, 2, steps=0) == (1, 2)
    assert f.refine_root(-2, -1, steps=0) == (-2, -1)

    assert f.refine_root(1, 2, steps=None) == (1, Rational(3, 2))
    assert f.refine_root(-2, -1, steps=None) == (Rational(-3, 2), -1)

    assert f.refine_root(1, 2, steps=1) == (1, Rational(3, 2))
    assert f.refine_root(-2, -1, steps=1) == (Rational(-3, 2), -1)

    assert f.refine_root(1, 2, steps=1, fast=True) == (1, Rational(3, 2))
    assert f.refine_root(-2, -1, steps=1, fast=True) == (Rational(-3, 2), -1)

    assert f.refine_root(1, 2, eps=Rational(1, 100)) == (Rational(24, 17), Rational(17, 12))
    assert f.refine_root(1, 2, eps=1e-2) == (Rational(24, 17), Rational(17, 12))

    raises(PolynomialError, lambda: (f**2).refine_root(1, 2, check_sqf=True))

    raises(RefinementFailed, lambda: (f**2).refine_root(1, 2))
    raises(RefinementFailed, lambda: (f**2).refine_root(2, 3))

    f = x**2 - 2

    assert refine_root(f, 1, 2, steps=1) == (1, Rational(3, 2))
    assert refine_root(f, -2, -1, steps=1) == (Rational(-3, 2), -1)

    assert refine_root(f, 1, 2, steps=1, fast=True) == (1, Rational(3, 2))
    assert refine_root(f, -2, -1, steps=1, fast=True) == (Rational(-3, 2), -1)

    assert refine_root(f, 1, 2, eps=Rational(1, 100)) == (Rational(24, 17), Rational(17, 12))
    assert refine_root(f, 1, 2, eps=1e-2) == (Rational(24, 17), Rational(17, 12))

    raises(PolynomialError, lambda: refine_root(1, 7, 8, eps=Rational(1, 100)))

    raises(ValueError, lambda: Poly(f).refine_root(1, 2, eps=10**-100000))
    raises(ValueError, lambda: refine_root(f, 1, 2, eps=10**-100000))


def test_count_roots():
    assert count_roots(x**2 - 2) == 2

    assert count_roots(x**2 - 2, inf=-oo) == 2
    assert count_roots(x**2 - 2, sup=+oo) == 2
    assert count_roots(x**2 - 2, inf=-oo, sup=+oo) == 2

    assert count_roots(x**2 - 2, inf=-2) == 2
    assert count_roots(x**2 - 2, inf=-1) == 1

    assert count_roots(x**2 - 2, sup=1) == 1
    assert count_roots(x**2 - 2, sup=2) == 2

    assert count_roots(x**2 - 2, inf=-1, sup=1) == 0
    assert count_roots(x**2 - 2, inf=-2, sup=2) == 2

    assert count_roots(x**2 - 2, inf=-1, sup=1) == 0
    assert count_roots(x**2 - 2, inf=-2, sup=2) == 2

    assert count_roots(x**2 + 2) == 0
    assert count_roots(x**2 + 2, inf=-2*I) == 2
    assert count_roots(x**2 + 2, sup=+2*I) == 2
    assert count_roots(x**2 + 2, inf=-2*I, sup=+2*I) == 2

    assert count_roots(x**2 + 2, inf=0) == 0
    assert count_roots(x**2 + 2, sup=0) == 0

    assert count_roots(x**2 + 2, inf=-I) == 1
    assert count_roots(x**2 + 2, sup=+I) == 1

    assert count_roots(x**2 + 2, inf=+I/2, sup=+I) == 0
    assert count_roots(x**2 + 2, inf=-I, sup=-I/2) == 0

    raises(PolynomialError, lambda: count_roots(1))


def test_count_roots_extension():

    p1 = Poly(sqrt(2)*x**2 - 2, x, extension=True)
    assert p1.count_roots() == 2
    assert p1.count_roots(inf=0) == 1
    assert p1.count_roots(sup=0) == 1

    p2 = Poly(x**2 + sqrt(2), x, extension=True)
    assert p2.count_roots() == 0

    p3 = Poly(x**2 + 2*sqrt(2)*x + 1, x, extension=True)
    assert p3.count_roots() == 2
    assert p3.count_roots(inf=-10, sup=10) == 2
    assert p3.count_roots(inf=-10, sup=0) == 2
    assert p3.count_roots(inf=-10, sup=-3) == 0
    assert p3.count_roots(inf=-3, sup=-2) == 1
    assert p3.count_roots(inf=-1, sup=0) == 1


def test_Poly_root():
    f = Poly(2*x**3 - 7*x**2 + 4*x + 4)

    assert f.root(0) == Rational(-1, 2)
    assert f.root(1) == 2
    assert f.root(2) == 2
    raises(IndexError, lambda: f.root(3))

    assert Poly(x**5 + x + 1).root(0) == rootof(x**3 - x**2 + 1, 0)


def test_real_roots():

    assert real_roots(x) == [0]
    assert real_roots(x, multiple=False) == [(0, 1)]

    assert real_roots(x**3) == [0, 0, 0]
    assert real_roots(x**3, multiple=False) == [(0, 3)]

    assert real_roots(x*(x**3 + x + 3)) == [rootof(x**3 + x + 3, 0), 0]
    assert real_roots(x*(x**3 + x + 3), multiple=False) == [(rootof(
        x**3 + x + 3, 0), 1), (0, 1)]

    assert real_roots(
        x**3*(x**3 + x + 3)) == [rootof(x**3 + x + 3, 0), 0, 0, 0]
    assert real_roots(x**3*(x**3 + x + 3), multiple=False) == [(rootof(
        x**3 + x + 3, 0), 1), (0, 3)]

    assert real_roots(x**2 - 2, radicals=False) == [
            rootof(x**2 - 2, 0, radicals=False),
            rootof(x**2 - 2, 1, radicals=False),
        ]

    f = 2*x**3 - 7*x**2 + 4*x + 4
    g = x**3 + x + 1

    assert Poly(f).real_roots() == [Rational(-1, 2), 2, 2]
    assert Poly(g).real_roots() == [rootof(g, 0)]

    # testing extension
    f = x**2 - sqrt(2)
    roots = [-2**(S(1)/4), 2**(S(1)/4)]
    raises(NotImplementedError, lambda: real_roots(f))
    raises(NotImplementedError, lambda: real_roots(Poly(f, x)))
    assert real_roots(f, extension=True) == roots
    assert real_roots(Poly(f, extension=True)) == roots
    assert real_roots(Poly(f), extension=True) == roots


def test_all_roots():

    f = 2*x**3 - 7*x**2 + 4*x + 4
    froots = [Rational(-1, 2), 2, 2]
    assert all_roots(f) == Poly(f).all_roots() == froots

    g = x**3 + x + 1
    groots = [rootof(g, 0), rootof(g, 1), rootof(g, 2)]
    assert all_roots(g) == Poly(g).all_roots() == groots

    assert all_roots(x**2 - 2) == [-sqrt(2), sqrt(2)]
    assert all_roots(x**2 - 2, multiple=False) == [(-sqrt(2), 1), (sqrt(2), 1)]
    assert all_roots(x**2 - 2, radicals=False) == [
        rootof(x**2 - 2, 0, radicals=False),
        rootof(x**2 - 2, 1, radicals=False),
    ]

    p = x**5 - x - 1
    assert all_roots(p) == [
        rootof(p, 0), rootof(p, 1), rootof(p, 2), rootof(p, 3), rootof(p, 4)
    ]

    # testing extension
    f = x**2 + sqrt(2)
    roots = [-2**(S(1)/4)*I, 2**(S(1)/4)*I]
    raises(NotImplementedError, lambda: all_roots(f))
    raises(NotImplementedError, lambda : all_roots(Poly(f, x)))
    assert all_roots(f, extension=True) == roots
    assert all_roots(Poly(f, extension=True)) == roots
    assert all_roots(Poly(f), extension=True) == roots


def test_nroots():
    assert Poly(0, x).nroots() == []
    assert Poly(1, x).nroots() == []

    assert Poly(x**2 - 1, x).nroots() == [-1.0, 1.0]
    assert Poly(x**2 + 1, x).nroots() == [-1.0*I, 1.0*I]

    roots = Poly(x**2 - 1, x).nroots()
    assert roots == [-1.0, 1.0]

    roots = Poly(x**2 + 1, x).nroots()
    assert roots == [-1.0*I, 1.0*I]

    roots = Poly(x**2/3 - Rational(1, 3), x).nroots()
    assert roots == [-1.0, 1.0]

    roots = Poly(x**2/3 + Rational(1, 3), x).nroots()
    assert roots == [-1.0*I, 1.0*I]

    assert Poly(x**2 + 2*I, x).nroots() == [-1.0 + 1.0*I, 1.0 - 1.0*I]
    assert Poly(
        x**2 + 2*I, x, extension=I).nroots() == [-1.0 + 1.0*I, 1.0 - 1.0*I]

    assert Poly(0.2*x + 0.1).nroots() == [-0.5]

    roots = nroots(x**5 + x + 1, n=5)
    eps = Float("1e-5")

    assert re(roots[0]).epsilon_eq(-0.75487, eps) is S.true
    assert im(roots[0]) == 0
    assert re(roots[1]) == Float(-0.5, 5)
    assert im(roots[1]).epsilon_eq(-0.86602, eps) is S.true
    assert re(roots[2]) == Float(-0.5, 5)
    assert im(roots[2]).epsilon_eq(+0.86602, eps) is S.true
    assert re(roots[3]).epsilon_eq(+0.87743, eps) is S.true
    assert im(roots[3]).epsilon_eq(-0.74486, eps) is S.true
    assert re(roots[4]).epsilon_eq(+0.87743, eps) is S.true
    assert im(roots[4]).epsilon_eq(+0.74486, eps) is S.true

    eps = Float("1e-6")

    assert re(roots[0]).epsilon_eq(-0.75487, eps) is S.false
    assert im(roots[0]) == 0
    assert re(roots[1]) == Float(-0.5, 5)
    assert im(roots[1]).epsilon_eq(-0.86602, eps) is S.false
    assert re(roots[2]) == Float(-0.5, 5)
    assert im(roots[2]).epsilon_eq(+0.86602, eps) is S.false
    assert re(roots[3]).epsilon_eq(+0.87743, eps) is S.false
    assert im(roots[3]).epsilon_eq(-0.74486, eps) is S.false
    assert re(roots[4]).epsilon_eq(+0.87743, eps) is S.false
    assert im(roots[4]).epsilon_eq(+0.74486, eps) is S.false

    raises(DomainError, lambda: Poly(x + y, x).nroots())
    raises(MultivariatePolynomialError, lambda: Poly(x + y).nroots())

    assert nroots(x**2 - 1) == [-1.0, 1.0]

    roots = nroots(x**2 - 1)
    assert roots == [-1.0, 1.0]

    assert nroots(x + I) == [-1.0*I]
    assert nroots(x + 2*I) == [-2.0*I]

    raises(PolynomialError, lambda: nroots(0))

    # issue 8296
    f = Poly(x**4 - 1)
    assert f.nroots(2) == [w.n(2) for w in f.all_roots()]

    assert str(Poly(x**16 + 32*x**14 + 508*x**12 + 5440*x**10 +
        39510*x**8 + 204320*x**6 + 755548*x**4 + 1434496*x**2 +
        877969).nroots(2)) == ('[-1.7 - 1.9*I, -1.7 + 1.9*I, -1.7 '
        '- 2.5*I, -1.7 + 2.5*I, -1.0*I, 1.0*I, -1.7*I, 1.7*I, -2.8*I, '
        '2.8*I, -3.4*I, 3.4*I, 1.7 - 1.9*I, 1.7 + 1.9*I, 1.7 - 2.5*I, '
        '1.7 + 2.5*I]')
    assert str(Poly(1e-15*x**2 -1).nroots()) == ('[-31622776.6016838, 31622776.6016838]')

    # https://github.com/sympy/sympy/issues/23861

    i = Float('3.000000000000000000000000000000000000000000000000001')
    [r] = nroots(x + I*i, n=300)
    assert abs(r + I*i) < 1e-300


def test_ground_roots():
    f = x**6 - 4*x**4 + 4*x**3 - x**2

    assert Poly(f).ground_roots() == {S.One: 2, S.Zero: 2}
    assert ground_roots(f) == {S.One: 2, S.Zero: 2}


def test_nth_power_roots_poly():
    f = x**4 - x**2 + 1

    f_2 = (x**2 - x + 1)**2
    f_3 = (x**2 + 1)**2
    f_4 = (x**2 + x + 1)**2
    f_12 = (x - 1)**4

    assert nth_power_roots_poly(f, 1) == f

    raises(ValueError, lambda: nth_power_roots_poly(f, 0))
    raises(ValueError, lambda: nth_power_roots_poly(f, x))

    assert factor(nth_power_roots_poly(f, 2)) == f_2
    assert factor(nth_power_roots_poly(f, 3)) == f_3
    assert factor(nth_power_roots_poly(f, 4)) == f_4
    assert factor(nth_power_roots_poly(f, 12)) == f_12

    raises(MultivariatePolynomialError, lambda: nth_power_roots_poly(
        x + y, 2, x, y))

def test_which_real_roots():
    f = Poly(x**4 - 1)

    assert f.which_real_roots([1, -1]) == [1, -1]
    assert f.which_real_roots([1, -1, 2, 4]) == [1, -1]
    assert f.which_real_roots([1, -1, -1, 1, 2, 5]) == [1, -1]
    assert f.which_real_roots([10, 8, 7, -1, 1]) == [-1, 1]

    # no real roots
    # (technically its still a superset)
    f = Poly(x**2 + 1)
    assert f.which_real_roots([5, 10]) == []

    # not square free
    f = Poly((x-1)**2)
    assert f.which_real_roots([1, 1, -1, 2]) == [1]

    # candidates not superset
    f = Poly(x**2 - 1)
    assert f.which_real_roots([0, 2]) == [0, 2]

def test_which_all_roots():
    f = Poly(x**4 - 1)

    assert f.which_all_roots([1, -1, I, -I]) == [1, -1, I, -I]
    assert f.which_all_roots([I, I, -I, 1, -1]) == [I, -I, 1, -1]

    f = Poly(x**2 + 1)
    assert f.which_all_roots([I, -I, I/2]) == [I, -I]

    # not square free
    f = Poly((x-I)**2)
    assert f.which_all_roots([I, I, 1, -1, 0]) == [I]

    # candidates not superset
    f = Poly(x**2 + 1)
    assert f.which_all_roots([I/2, -I/2]) == [I/2, -I/2]

def test_same_root():
    f = Poly(x**4 + x**3 + x**2 + x + 1)
    eq = f.same_root
    r0 = exp(2 * I * pi / 5)
    assert [i for i, r in enumerate(f.all_roots()) if eq(r, r0)] == [3]

    raises(PolynomialError,
           lambda: Poly(x + 1, domain=QQ).same_root(0, 0))
    raises(DomainError,
           lambda: Poly(x**2 + 1, domain=FF(7)).same_root(0, 0))
    raises(DomainError,
           lambda: Poly(x ** 2 + 1, domain=ZZ_I).same_root(0, 0))
    raises(DomainError,
           lambda: Poly(y * x**2 + 1, domain=ZZ[y]).same_root(0, 0))
    raises(MultivariatePolynomialError,
           lambda: Poly(x * y + 1, domain=ZZ).same_root(0, 0))


def test_torational_factor_list():
    p = expand(((x**2-1)*(x-2)).subs({x:x*(1 + sqrt(2))}))
    assert _torational_factor_list(p, x) == (-2, [
        (-x*(1 + sqrt(2))/2 + 1, 1),
        (-x*(1 + sqrt(2)) - 1, 1),
        (-x*(1 + sqrt(2)) + 1, 1)])


    p = expand(((x**2-1)*(x-2)).subs({x:x*(1 + 2**Rational(1, 4))}))
    assert _torational_factor_list(p, x) is None


def test_cancel():
    assert cancel(0) == 0
    assert cancel(7) == 7
    assert cancel(x) == x

    assert cancel(oo) is oo

    raises(ValueError, lambda: cancel((1, 2, 3)))

    # test first tuple returnr
    assert (t:=cancel((2, 3))) == (1, 2, 3)
    assert isinstance(t, tuple)

    # tests 2nd tuple return
    assert (t:=cancel((1, 0), x)) == (1, 1, 0)
    assert isinstance(t, tuple)
    assert cancel((0, 1), x) == (1, 0, 1)

    f, g, p, q = 4*x**2 - 4, 2*x - 2, 2*x + 2, 1
    F, G, P, Q = [ Poly(u, x) for u in (f, g, p, q) ]

    assert F.cancel(G) == (1, P, Q)
    assert cancel((f, g)) == (1, p, q)
    assert cancel((f, g), x) == (1, p, q)
    assert cancel((f, g), (x,)) == (1, p, q)
    # tests 3rd tuple return
    assert (t:=cancel((F, G))) == (1, P, Q)
    assert isinstance(t, tuple)
    assert cancel((f, g), polys=True) == (1, P, Q)
    assert cancel((F, G), polys=False) == (1, p, q)

    f = (x**2 - 2)/(x + sqrt(2))

    assert cancel(f) == f
    assert cancel(f, greedy=False) == x - sqrt(2)

    f = (x**2 - 2)/(x - sqrt(2))

    assert cancel(f) == f
    assert cancel(f, greedy=False) == x + sqrt(2)

    assert cancel((x**2/4 - 1, x/2 - 1)) == (1, x + 2, 2)
    # assert cancel((x**2/4 - 1, x/2 - 1)) == (S.Half, x + 2, 1)

    assert cancel((x**2 - y)/(x - y)) == 1/(x - y)*(x**2 - y)

    assert cancel((x**2 - y**2)/(x - y), x) == x + y
    assert cancel((x**2 - y**2)/(x - y), y) == x + y
    assert cancel((x**2 - y**2)/(x - y)) == x + y

    assert cancel((x**3 - 1)/(x**2 - 1)) == (x**2 + x + 1)/(x + 1)
    assert cancel((x**3/2 - S.Half)/(x**2 - 1)) == (x**2 + x + 1)/(2*x + 2)

    assert cancel((exp(2*x) + 2*exp(x) + 1)/(exp(x) + 1)) == exp(x) + 1

    f = Poly(x**2 - a**2, x)
    g = Poly(x - a, x)

    F = Poly(x + a, x, domain='ZZ[a]')
    G = Poly(1, x, domain='ZZ[a]')

    assert cancel((f, g)) == (1, F, G)

    f = x**3 + (sqrt(2) - 2)*x**2 - (2*sqrt(2) + 3)*x - 3*sqrt(2)
    g = x**2 - 2

    assert cancel((f, g), extension=True) == (1, x**2 - 2*x - 3, x - sqrt(2))

    f = Poly(-2*x + 3, x)
    g = Poly(-x**9 + x**8 + x**6 - x**5 + 2*x**2 - 3*x + 1, x)

    assert cancel((f, g)) == (1, -f, -g)

    f = Poly(x/3 + 1, x)
    g = Poly(x/7 + 1, x)

    assert f.cancel(g) == (S(7)/3,
        Poly(x + 3, x, domain=QQ),
        Poly(x + 7, x, domain=QQ))
    assert f.cancel(g, include=True) == (
        Poly(7*x + 21, x, domain=QQ),
        Poly(3*x + 21, x, domain=QQ))

    pairs = [
        (1 + x, 1 + x, 1, 1, 1),
        (1 + x, 1 - x, -1, -1-x, -1+x),
        (1 - x, 1 + x, -1, 1-x, 1+x),
        (1 - x, 1 - x, 1, 1, 1),
    ]
    for f, g, coeff, p, q in pairs:
        assert cancel((f, g)) == (1, p, q)
        pf = Poly(f, x)
        pg = Poly(g, x)
        pp = Poly(p, x)
        pq = Poly(q, x)
        assert pf.cancel(pg) == (coeff, coeff*pp, pq)
        assert pf.rep.cancel(pg.rep) == (pp.rep, pq.rep)
        assert pf.rep.cancel(pg.rep, include=True) == (pp.rep, pq.rep)

    f = Poly(y, y, domain='ZZ(x)')
    g = Poly(1, y, domain='ZZ[x]')

    assert f.cancel(
        g) == (1, Poly(y, y, domain='ZZ(x)'), Poly(1, y, domain='ZZ(x)'))
    assert f.cancel(g, include=True) == (
        Poly(y, y, domain='ZZ(x)'), Poly(1, y, domain='ZZ(x)'))

    f = Poly(5*x*y + x, y, domain='ZZ(x)')
    g = Poly(2*x**2*y, y, domain='ZZ(x)')

    assert f.cancel(g, include=True) == (
        Poly(5*y + 1, y, domain='ZZ(x)'), Poly(2*x*y, y, domain='ZZ(x)'))

    f = -(-2*x - 4*y + 0.005*(z - y)**2)/((z - y)*(-z + y + 2))
    assert cancel(f).is_Mul == True

    P = tanh(x - 3.0)
    Q = tanh(x + 3.0)
    f = ((-2*P**2 + 2)*(-P**2 + 1)*Q**2/2 + (-2*P**2 + 2)*(-2*Q**2 + 2)*P*Q - (-2*P**2 + 2)*P**2*Q**2 + (-2*Q**2 + 2)*(-Q**2 + 1)*P**2/2 - (-2*Q**2 + 2)*P**2*Q**2)/(2*sqrt(P**2*Q**2 + 0.0001)) \
      + (-(-2*P**2 + 2)*P*Q**2/2 - (-2*Q**2 + 2)*P**2*Q/2)*((-2*P**2 + 2)*P*Q**2/2 + (-2*Q**2 + 2)*P**2*Q/2)/(2*(P**2*Q**2 + 0.0001)**Rational(3, 2))
    assert cancel(f).is_Mul == True

    # issue 7022
    A = Symbol('A', commutative=False)
    p1 = Piecewise((A*(x**2 - 1)/(x + 1), x > 1), ((x + 2)/(x**2 + 2*x), True))
    p2 = Piecewise((A*(x - 1), x > 1), (1/x, True))
    assert cancel(p1) == p2
    assert cancel(2*p1) == 2*p2
    assert cancel(1 + p1) == 1 + p2
    assert cancel((x**2 - 1)/(x + 1)*p1) == (x - 1)*p2
    assert cancel((x**2 - 1)/(x + 1) + p1) == (x - 1) + p2
    p3 = Piecewise(((x**2 - 1)/(x + 1), x > 1), ((x + 2)/(x**2 + 2*x), True))
    p4 = Piecewise(((x - 1), x > 1), (1/x, True))
    assert cancel(p3) == p4
    assert cancel(2*p3) == 2*p4
    assert cancel(1 + p3) == 1 + p4
    assert cancel((x**2 - 1)/(x + 1)*p3) == (x - 1)*p4
    assert cancel((x**2 - 1)/(x + 1) + p3) == (x - 1) + p4

    # issue 4077
    q = S('''(2*1*(x - 1/x)/(x*(2*x - (-x + 1/x)/(x**2*(x - 1/x)**2) - 1/(x**2*(x -
        1/x)) - 2/x)) - 2*1*((x - 1/x)/((x*(x - 1/x)**2)) - 1/(x*(x -
        1/x)))*((-x + 1/x)*((x - 1/x)/((x*(x - 1/x)**2)) - 1/(x*(x -
        1/x)))/(2*x - (-x + 1/x)/(x**2*(x - 1/x)**2) - 1/(x**2*(x - 1/x)) -
        2/x) + 1)*((x - 1/x)/((x - 1/x)**2) - ((x - 1/x)/((x*(x - 1/x)**2)) -
        1/(x*(x - 1/x)))**2/(2*x - (-x + 1/x)/(x**2*(x - 1/x)**2) - 1/(x**2*(x
        - 1/x)) - 2/x) - 1/(x - 1/x))*(2*x - (-x + 1/x)/(x**2*(x - 1/x)**2) -
        1/(x**2*(x - 1/x)) - 2/x)/x - 1/x)*(((-x + 1/x)/((x*(x - 1/x)**2)) +
        1/(x*(x - 1/x)))*((-(x - 1/x)/(x*(x - 1/x)) - 1/x)*((x - 1/x)/((x*(x -
        1/x)**2)) - 1/(x*(x - 1/x)))/(2*x - (-x + 1/x)/(x**2*(x - 1/x)**2) -
        1/(x**2*(x - 1/x)) - 2/x) - 1 + (x - 1/x)/(x - 1/x))/((x*((x -
        1/x)/((x - 1/x)**2) - ((x - 1/x)/((x*(x - 1/x)**2)) - 1/(x*(x -
        1/x)))**2/(2*x - (-x + 1/x)/(x**2*(x - 1/x)**2) - 1/(x**2*(x - 1/x)) -
        2/x) - 1/(x - 1/x))*(2*x - (-x + 1/x)/(x**2*(x - 1/x)**2) - 1/(x**2*(x
        - 1/x)) - 2/x))) + ((x - 1/x)/((x*(x - 1/x))) + 1/x)/((x*(2*x - (-x +
        1/x)/(x**2*(x - 1/x)**2) - 1/(x**2*(x - 1/x)) - 2/x))) + 1/x)/(2*x +
        2*((x - 1/x)/((x*(x - 1/x)**2)) - 1/(x*(x - 1/x)))*((-(x - 1/x)/(x*(x
        - 1/x)) - 1/x)*((x - 1/x)/((x*(x - 1/x)**2)) - 1/(x*(x - 1/x)))/(2*x -
        (-x + 1/x)/(x**2*(x - 1/x)**2) - 1/(x**2*(x - 1/x)) - 2/x) - 1 + (x -
        1/x)/(x - 1/x))/((x*((x - 1/x)/((x - 1/x)**2) - ((x - 1/x)/((x*(x -
        1/x)**2)) - 1/(x*(x - 1/x)))**2/(2*x - (-x + 1/x)/(x**2*(x - 1/x)**2)
        - 1/(x**2*(x - 1/x)) - 2/x) - 1/(x - 1/x))*(2*x - (-x + 1/x)/(x**2*(x
        - 1/x)**2) - 1/(x**2*(x - 1/x)) - 2/x))) - 2*((x - 1/x)/((x*(x -
        1/x))) + 1/x)/(x*(2*x - (-x + 1/x)/(x**2*(x - 1/x)**2) - 1/(x**2*(x -
        1/x)) - 2/x)) - 2/x) - ((x - 1/x)/((x*(x - 1/x)**2)) - 1/(x*(x -
        1/x)))*((-x + 1/x)*((x - 1/x)/((x*(x - 1/x)**2)) - 1/(x*(x -
        1/x)))/(2*x - (-x + 1/x)/(x**2*(x - 1/x)**2) - 1/(x**2*(x - 1/x)) -
        2/x) + 1)/(x*((x - 1/x)/((x - 1/x)**2) - ((x - 1/x)/((x*(x - 1/x)**2))
        - 1/(x*(x - 1/x)))**2/(2*x - (-x + 1/x)/(x**2*(x - 1/x)**2) -
        1/(x**2*(x - 1/x)) - 2/x) - 1/(x - 1/x))*(2*x - (-x + 1/x)/(x**2*(x -
        1/x)**2) - 1/(x**2*(x - 1/x)) - 2/x)) + (x - 1/x)/((x*(2*x - (-x +
        1/x)/(x**2*(x - 1/x)**2) - 1/(x**2*(x - 1/x)) - 2/x))) - 1/x''',
        evaluate=False)
    assert cancel(q, _signsimp=False) is S.NaN
    assert q.subs(x, 2) is S.NaN
    assert signsimp(q) is S.NaN

    # issue 9363
    M = MatrixSymbol('M', 5, 5)
    assert cancel(M[0,0] + 7) == M[0,0] + 7
    expr = sin(M[1, 4] + M[2, 1] * 5 * M[4, 0]) - 5 * M[1, 2] / z
    assert cancel(expr) == (z*sin(M[1, 4] + M[2, 1] * 5 * M[4, 0]) - 5 * M[1, 2]) / z

    assert cancel((x**2 + 1)/(x - I)) == x + I


def test_cancel_modulus():
    assert cancel((x**2 - 1)/(x + 1), modulus=2) == x + 1
    assert Poly(x**2 - 1, modulus=2).cancel(Poly(x + 1, modulus=2)) ==\
            (1, Poly(x + 1, modulus=2), Poly(1, x, modulus=2))


def test_make_monic_over_integers_by_scaling_roots():
    f = Poly(x**2 + 3*x + 4, x, domain='ZZ')
    g, c = f.make_monic_over_integers_by_scaling_roots()
    assert g == f
    assert c == ZZ.one

    f = Poly(x**2 + 3*x + 4, x, domain='QQ')
    g, c = f.make_monic_over_integers_by_scaling_roots()
    assert g == f.to_ring()
    assert c == ZZ.one

    f = Poly(x**2/2 + S(1)/4 * x + S(1)/8, x, domain='QQ')
    g, c = f.make_monic_over_integers_by_scaling_roots()
    assert g == Poly(x**2 + 2*x + 4, x, domain='ZZ')
    assert c == 4

    f = Poly(x**3/2 + S(1)/4 * x + S(1)/8, x, domain='QQ')
    g, c = f.make_monic_over_integers_by_scaling_roots()
    assert g == Poly(x**3 + 8*x + 16, x, domain='ZZ')
    assert c == 4

    f = Poly(x*y, x, y)
    raises(ValueError, lambda: f.make_monic_over_integers_by_scaling_roots())

    f = Poly(x, domain='RR')
    raises(ValueError, lambda: f.make_monic_over_integers_by_scaling_roots())


def test_galois_group():
    f = Poly(x ** 4 - 2)
    G, alt = f.galois_group(by_name=True)
    assert G == S4TransitiveSubgroups.D4
    assert alt is False


def test_reduced():
    f = 2*x**4 + y**2 - x**2 + y**3
    G = [x**3 - x, y**3 - y]

    Q = [2*x, 1]
    r = x**2 + y**2 + y

    assert reduced(f, G) == (Q, r)
    assert reduced(f, G, x, y) == (Q, r)

    H = groebner(G)

    assert H.reduce(f) == (Q, r)

    Q = [Poly(2*x, x, y), Poly(1, x, y)]
    r = Poly(x**2 + y**2 + y, x, y)

    assert _strict_eq(reduced(f, G, polys=True), (Q, r))
    assert _strict_eq(reduced(f, G, x, y, polys=True), (Q, r))

    H = groebner(G, polys=True)

    assert _strict_eq(H.reduce(f), (Q, r))

    f = 2*x**3 + y**3 + 3*y
    G = groebner([x**2 + y**2 - 1, x*y - 2])

    Q = [x**2 - x*y**3/2 + x*y/2 + y**6/4 - y**4/2 + y**2/4, -y**5/4 + y**3/2 + y*Rational(3, 4)]
    r = 0

    assert reduced(f, G) == (Q, r)
    assert G.reduce(f) == (Q, r)

    assert reduced(f, G, auto=False)[1] != 0
    assert G.reduce(f, auto=False)[1] != 0

    assert G.contains(f) is True
    assert G.contains(f + 1) is False

    assert reduced(1, [1], x) == ([1], 0)
    raises(ComputationFailed, lambda: reduced(1, [1]))

    f_poly = Poly(2*x**3 + y**3 + 3*y)
    G_poly = groebner([Poly(x**2 + y**2 - 1), Poly(x*y - 2)])

    Q_poly = [Poly(x**2 - 1/2*x*y**3 + 1/2*x*y + 1/4*y**6 - 1/2*y**4 + 1/4*y**2, x, y, domain='QQ'),
              Poly(-1/4*y**5 + 1/2*y**3 + 3/4*y, x, y, domain='QQ')]
    r_poly = Poly(0, x, y, domain='QQ')

    assert G_poly.reduce(f_poly) == (Q_poly, r_poly)

    Q, r = G_poly.reduce(f)
    assert all(isinstance(q, Poly) for q in Q)
    assert isinstance(r, Poly)

    f_wrong_gens = Poly(2*x**3 + y**3 + 3*y, x, y, z)
    raises(ValueError, lambda: G_poly.reduce(f_wrong_gens))

    zero_poly = Poly(0, x, y)
    Q, r = G_poly.reduce(zero_poly)
    assert all(q.is_zero for q in Q)
    assert r.is_zero

    const_poly = Poly(1, x, y)
    Q, r = G_poly.reduce(const_poly)
    assert isinstance(r, Poly)
    assert r.as_expr() == 1
    assert all(q.is_zero for q in Q)


def test_groebner():
    assert groebner([], x, y, z) == []

    assert groebner([x**2 + 1, y**4*x + x**3], x, y, order='lex') == [1 + x**2, -1 + y**4]
    assert groebner([x**2 + 1, y**4*x + x**3, x*y*z**3], x, y, z, order='grevlex') == [-1 + y**4, z**3, 1 + x**2]

    assert groebner([x**2 + 1, y**4*x + x**3], x, y, order='lex', polys=True) == \
        [Poly(1 + x**2, x, y), Poly(-1 + y**4, x, y)]
    assert groebner([x**2 + 1, y**4*x + x**3, x*y*z**3], x, y, z, order='grevlex', polys=True) == \
        [Poly(-1 + y**4, x, y, z), Poly(z**3, x, y, z), Poly(1 + x**2, x, y, z)]

    assert groebner([x**3 - 1, x**2 - 1]) == [x - 1]
    assert groebner([Eq(x**3, 1), Eq(x**2, 1)]) == [x - 1]

    F = [3*x**2 + y*z - 5*x - 1, 2*x + 3*x*y + y**2, x - 3*y + x*z - 2*z**2]
    f = z**9 - x**2*y**3 - 3*x*y**2*z + 11*y*z**2 + x**2*z**2 - 5

    G = groebner(F, x, y, z, modulus=7, symmetric=False)

    assert G == [1 + x + y + 3*z + 2*z**2 + 2*z**3 + 6*z**4 + z**5,
                 1 + 3*y + y**2 + 6*z**2 + 3*z**3 + 3*z**4 + 3*z**5 + 4*z**6,
                 1 + 4*y + 4*z + y*z + 4*z**3 + z**4 + z**6,
                 6 + 6*z + z**2 + 4*z**3 + 3*z**4 + 6*z**5 + 3*z**6 + z**7]

    Q, r = reduced(f, G, x, y, z, modulus=7, symmetric=False, polys=True)

    assert sum([ q*g for q, g in zip(Q, G.polys)], r) == Poly(f, modulus=7)

    F = [x*y - 2*y, 2*y**2 - x**2]

    assert groebner(F, x, y, order='grevlex') == \
        [y**3 - 2*y, x**2 - 2*y**2, x*y - 2*y]
    assert groebner(F, y, x, order='grevlex') == \
        [x**3 - 2*x**2, -x**2 + 2*y**2, x*y - 2*y]
    assert groebner(F, order='grevlex', field=True) == \
        [y**3 - 2*y, x**2 - 2*y**2, x*y - 2*y]

    assert groebner([1], x) == [1]

    assert groebner([x**2 + 2.0*y], x, y) == [1.0*x**2 + 2.0*y]
    raises(ComputationFailed, lambda: groebner([1]))

    assert groebner([x**2 - 1, x**3 + 1], method='buchberger') == [x + 1]
    assert groebner([x**2 - 1, x**3 + 1], method='f5b') == [x + 1]

    raises(ValueError, lambda: groebner([x, y], method='unknown'))


def test_fglm():
    F = [a + b + c + d, a*b + a*d + b*c + b*d, a*b*c + a*b*d + a*c*d + b*c*d, a*b*c*d - 1]
    G = groebner(F, a, b, c, d, order=grlex)

    B = [
        4*a + 3*d**9 - 4*d**5 - 3*d,
        4*b + 4*c - 3*d**9 + 4*d**5 + 7*d,
        4*c**2 + 3*d**10 - 4*d**6 - 3*d**2,
        4*c*d**4 + 4*c - d**9 + 4*d**5 + 5*d,
        d**12 - d**8 - d**4 + 1,
    ]

    assert groebner(F, a, b, c, d, order=lex) == B
    assert G.fglm(lex) == B

    F = [9*x**8 + 36*x**7 - 32*x**6 - 252*x**5 - 78*x**4 + 468*x**3 + 288*x**2 - 108*x + 9,
        -72*t*x**7 - 252*t*x**6 + 192*t*x**5 + 1260*t*x**4 + 312*t*x**3 - 404*t*x**2 - 576*t*x + \
        108*t - 72*x**7 - 256*x**6 + 192*x**5 + 1280*x**4 + 312*x**3 - 576*x + 96]
    G = groebner(F, t, x, order=grlex)

    B = [
        203577793572507451707*t + 627982239411707112*x**7 - 666924143779443762*x**6 - \
        10874593056632447619*x**5 + 5119998792707079562*x**4 + 72917161949456066376*x**3 + \
        20362663855832380362*x**2 - 142079311455258371571*x + 183756699868981873194,
        9*x**8 + 36*x**7 - 32*x**6 - 252*x**5 - 78*x**4 + 468*x**3 + 288*x**2 - 108*x + 9,
    ]

    assert groebner(F, t, x, order=lex) == B
    assert G.fglm(lex) == B

    F = [x**2 - x - 3*y + 1, -2*x + y**2 + y - 1]
    G = groebner(F, x, y, order=lex)

    B = [
        x**2 - x - 3*y + 1,
        y**2 - 2*x + y - 1,
    ]

    assert groebner(F, x, y, order=grlex) == B
    assert G.fglm(grlex) == B


def test_is_zero_dimensional():
    assert is_zero_dimensional([x, y], x, y) is True
    assert is_zero_dimensional([x**3 + y**2], x, y) is False

    assert is_zero_dimensional([x, y, z], x, y, z) is True
    assert is_zero_dimensional([x, y, z], x, y, z, t) is False

    F = [x*y - z, y*z - x, x*y - y]
    assert is_zero_dimensional(F, x, y, z) is True

    F = [x**2 - 2*x*z + 5, x*y**2 + y*z**3, 3*y**2 - 8*z**2]
    assert is_zero_dimensional(F, x, y, z) is True


def test_GroebnerBasis():
    F = [x*y - 2*y, 2*y**2 - x**2]

    G = groebner(F, x, y, order='grevlex')
    H = [y**3 - 2*y, x**2 - 2*y**2, x*y - 2*y]
    P = [ Poly(h, x, y) for h in H ]

    assert groebner(F + [0], x, y, order='grevlex') == G
    assert isinstance(G, GroebnerBasis) is True

    assert len(G) == 3

    assert G[0] == H[0] and not G[0].is_Poly
    assert G[1] == H[1] and not G[1].is_Poly
    assert G[2] == H[2] and not G[2].is_Poly

    assert G[1:] == H[1:] and not any(g.is_Poly for g in G[1:])
    assert G[:2] == H[:2] and not any(g.is_Poly for g in G[1:])

    assert G.exprs == H
    assert G.polys == P
    assert G.gens == (x, y)
    assert G.domain == ZZ
    assert G.order == grevlex

    assert G == H
    assert G == tuple(H)
    assert G == P
    assert G == tuple(P)

    assert G != []

    G = groebner(F, x, y, order='grevlex', polys=True)

    assert G[0] == P[0] and G[0].is_Poly
    assert G[1] == P[1] and G[1].is_Poly
    assert G[2] == P[2] and G[2].is_Poly

    assert G[1:] == P[1:] and all(g.is_Poly for g in G[1:])
    assert G[:2] == P[:2] and all(g.is_Poly for g in G[1:])


def test_poly():
    assert poly(x) == Poly(x, x)
    assert poly(y) == Poly(y, y)

    assert poly(x + y) == Poly(x + y, x, y)
    assert poly(x + sin(x)) == Poly(x + sin(x), x, sin(x))

    assert poly(x + y, wrt=y) == Poly(x + y, y, x)
    assert poly(x + sin(x), wrt=sin(x)) == Poly(x + sin(x), sin(x), x)

    assert poly(x*y + 2*x*z**2 + 17) == Poly(x*y + 2*x*z**2 + 17, x, y, z)

    assert poly(2*(y + z)**2 - 1) == Poly(2*y**2 + 4*y*z + 2*z**2 - 1, y, z)
    assert poly(
        x*(y + z)**2 - 1) == Poly(x*y**2 + 2*x*y*z + x*z**2 - 1, x, y, z)
    assert poly(2*x*(
        y + z)**2 - 1) == Poly(2*x*y**2 + 4*x*y*z + 2*x*z**2 - 1, x, y, z)

    assert poly(2*(
        y + z)**2 - x - 1) == Poly(2*y**2 + 4*y*z + 2*z**2 - x - 1, x, y, z)
    assert poly(x*(
        y + z)**2 - x - 1) == Poly(x*y**2 + 2*x*y*z + x*z**2 - x - 1, x, y, z)
    assert poly(2*x*(y + z)**2 - x - 1) == Poly(2*x*y**2 + 4*x*y*z + 2*
                x*z**2 - x - 1, x, y, z)

    assert poly(x*y + (x + y)**2 + (x + z)**2) == \
        Poly(2*x*z + 3*x*y + y**2 + z**2 + 2*x**2, x, y, z)
    assert poly(x*y*(x + y)*(x + z)**2) == \
        Poly(x**3*y**2 + x*y**2*z**2 + y*x**2*z**2 + 2*z*x**2*
             y**2 + 2*y*z*x**3 + y*x**4, x, y, z)

    assert poly(Poly(x + y + z, y, x, z)) == Poly(x + y + z, y, x, z)

    assert poly((x + y)**2, x) == Poly(x**2 + 2*x*y + y**2, x, domain=ZZ[y])
    assert poly((x + y)**2, y) == Poly(x**2 + 2*x*y + y**2, y, domain=ZZ[x])

    assert poly(1, x) == Poly(1, x)
    raises(GeneratorsNeeded, lambda: poly(1))

    # issue 6184
    assert poly(x + y, x, y) == Poly(x + y, x, y)
    assert poly(x + y, y, x) == Poly(x + y, y, x)

    # https://github.com/sympy/sympy/issues/19755
    expr1 = x + (2*x + 3)**2/5 + S(6)/5
    assert poly(expr1).as_expr() == expr1.expand()
    expr2 = y*(y+1) + S(1)/3
    assert poly(expr2).as_expr() == expr2.expand()


def test_keep_coeff():
    u = Mul(2, x + 1, evaluate=False)
    assert _keep_coeff(S.One, x) == x
    assert _keep_coeff(S.NegativeOne, x) == -x
    assert _keep_coeff(S(1.0), x) == 1.0*x
    assert _keep_coeff(S(-1.0), x) == -1.0*x
    assert _keep_coeff(S.One, 2*x) == 2*x
    assert _keep_coeff(S(2), x/2) == x
    assert _keep_coeff(S(2), sin(x)) == 2*sin(x)
    assert _keep_coeff(S(2), x + 1) == u
    assert _keep_coeff(x, 1/x) == 1
    assert _keep_coeff(x + 1, S(2)) == u
    assert _keep_coeff(S.Half, S.One) == S.Half
    p = Pow(2, 3, evaluate=False)
    assert _keep_coeff(S(-1), p) == Mul(-1, p, evaluate=False)
    a = Add(2, p, evaluate=False)
    assert _keep_coeff(S.Half, a, clear=True
        ) == Mul(S.Half, a, evaluate=False)
    assert _keep_coeff(S.Half, a, clear=False
        ) == Add(1, Mul(S.Half, p, evaluate=False), evaluate=False)


def test_poly_matching_consistency():
    # Test for this issue:
    # https://github.com/sympy/sympy/issues/5514
    assert I * Poly(x, x) == Poly(I*x, x)
    assert Poly(x, x) * I == Poly(I*x, x)


def test_issue_5786():
    assert expand(factor(expand(
        (x - I*y)*(z - I*t)), extension=[I])) == -I*t*x - t*y + x*z - I*y*z


def test_noncommutative():
    class foo(Expr):
        is_commutative=False
    e = x/(x + x*y)
    c = 1/( 1 + y)
    assert cancel(foo(e)) == foo(c)
    assert cancel(e + foo(e)) == c + foo(c)
    assert cancel(e*foo(c)) == c*foo(c)


def test_to_rational_coeffs():
    assert to_rational_coeffs(
        Poly(x**3 + y*x**2 + sqrt(y), x, domain='EX')) is None
    # issue 21268
    assert to_rational_coeffs(
        Poly(y**3 + sqrt(2)*y**2*sin(x) + 1, y)) is None

    assert to_rational_coeffs(Poly(x, y)) is None
    assert to_rational_coeffs(Poly(sqrt(2)*y)) is None


def test_factor_terms():
    # issue 7067
    assert factor_list(x*(x + y)) == (1, [(x, 1), (x + y, 1)])
    assert sqf_list(x*(x + y)) == (1, [(x**2 + x*y, 1)])


def test_as_list():
    # issue 14496
    assert Poly(x**3 + 2, x, domain='ZZ').as_list() == [1, 0, 0, 2]
    assert Poly(x**2 + y + 1, x, y, domain='ZZ').as_list() == [[1], [], [1, 1]]
    assert Poly(x**2 + y + 1, x, y, z, domain='ZZ').as_list() == \
                                                    [[[1]], [[]], [[1], [1]]]


def test_issue_11198():
    assert factor_list(sqrt(2)*x) == (sqrt(2), [(x, 1)])
    assert factor_list(sqrt(2)*sin(x), sin(x)) == (sqrt(2), [(sin(x), 1)])


def test_Poly_precision():
    # Make sure Poly doesn't lose precision
    p = Poly(pi.evalf(100)*x)
    assert p.as_expr() == pi.evalf(100)*x


def test_issue_12400():
    # Correction of check for negative exponents
    assert poly(1/(1+sqrt(2)), x) == \
            Poly(1/(1+sqrt(2)), x, domain='EX')

def test_issue_14364():
    assert gcd(S(6)*(1 + sqrt(3))/5, S(3)*(1 + sqrt(3))/10) == Rational(3, 10) * (1 + sqrt(3))
    assert gcd(sqrt(5)*Rational(4, 7), sqrt(5)*Rational(2, 3)) == sqrt(5)*Rational(2, 21)

    assert lcm(Rational(2, 3)*sqrt(3), Rational(5, 6)*sqrt(3)) == S(10)*sqrt(3)/3
    assert lcm(3*sqrt(3), 4/sqrt(3)) == 12*sqrt(3)
    assert lcm(S(5)*(1 + 2**Rational(1, 3))/6, S(3)*(1 + 2**Rational(1, 3))/8) == Rational(15, 2) * (1 + 2**Rational(1, 3))

    assert gcd(Rational(2, 3)*sqrt(3), Rational(5, 6)/sqrt(3)) == sqrt(3)/18
    assert gcd(S(4)*sqrt(13)/7, S(3)*sqrt(13)/14) == sqrt(13)/14

    # gcd_list and lcm_list
    assert gcd([S(2)*sqrt(47)/7, S(6)*sqrt(47)/5, S(8)*sqrt(47)/5]) == sqrt(47)*Rational(2, 35)
    assert gcd([S(6)*(1 + sqrt(7))/5, S(2)*(1 + sqrt(7))/7, S(4)*(1 + sqrt(7))/13]) ==  (1 + sqrt(7))*Rational(2, 455)
    assert lcm((Rational(7, 2)/sqrt(15), Rational(5, 6)/sqrt(15), Rational(5, 8)/sqrt(15))) == Rational(35, 2)/sqrt(15)
    assert lcm([S(5)*(2 + 2**Rational(5, 7))/6, S(7)*(2 + 2**Rational(5, 7))/2, S(13)*(2 + 2**Rational(5, 7))/4]) == Rational(455, 2) * (2 + 2**Rational(5, 7))


def test_issue_15669():
    x = Symbol("x", positive=True)
    expr = (16*x**3/(-x**2 + sqrt(8*x**2 + (x**2 - 2)**2) + 2)**2 -
        2*2**Rational(4, 5)*x*(-x**2 + sqrt(8*x**2 + (x**2 - 2)**2) + 2)**Rational(3, 5) + 10*x)
    assert factor(expr, deep=True) == x*(x**2 + 2)


def test_issue_17988():
    x = Symbol('x')
    p = poly(x - 1)
    with warns_deprecated_sympy():
        M = Matrix([[poly(x + 1), poly(x + 1)]])
    with warns(SymPyDeprecationWarning, test_stacklevel=False):
        assert p * M == M * p == Matrix([[poly(x**2 - 1), poly(x**2 - 1)]])


def test_issue_18205():
    assert cancel((2 + I)*(3 - I)) == 7 + I
    assert cancel((2 + I)*(2 - I)) == 5


def test_issue_8695():
    p = (x**2 + 1) * (x - 1)**2 * (x - 2)**3 * (x - 3)**3
    result = (1, [(x**2 + 1, 1), (x - 1, 2), (x**2 - 5*x + 6, 3)])
    assert sqf_list(p) == result


def test_issue_19113():
    eq = sin(x)**3 - sin(x) + 1
    raises(PolynomialError, lambda: refine_root(eq, 1, 2, 1e-2))
    raises(PolynomialError, lambda: count_roots(eq, -1, 1))
    raises(PolynomialError, lambda: real_roots(eq))
    raises(PolynomialError, lambda: nroots(eq))
    raises(PolynomialError, lambda: ground_roots(eq))
    raises(PolynomialError, lambda: nth_power_roots_poly(eq, 2))


def test_issue_19360():
    f = 2*x**2 - 2*sqrt(2)*x*y + y**2
    assert factor(f, extension=sqrt(2)) == 2*(x - (sqrt(2)*y/2))**2

    f = -I*t*x - t*y + x*z - I*y*z
    assert factor(f, extension=I) == (x - I*y)*(-I*t + z)


def test_poly_copy_equals_original():
    poly = Poly(x + y, x, y, z)
    copy = poly.copy()
    assert poly == copy, (
        "Copied polynomial not equal to original.")
    assert poly.gens == copy.gens, (
        "Copied polynomial has different generators than original.")


def test_deserialized_poly_equals_original():
    poly = Poly(x + y, x, y, z)
    deserialized = pickle.loads(pickle.dumps(poly))
    assert poly == deserialized, (
        "Deserialized polynomial not equal to original.")
    assert poly.gens == deserialized.gens, (
        "Deserialized polynomial has different generators than original.")


def test_issue_20389():
    result = degree(x * (x + 1) - x ** 2 - x, x)
    assert result == -oo


def test_issue_20985():
    from sympy.core.symbol import symbols
    w, R = symbols('w R')
    poly = Poly(1.0 + I*w/R, w, 1/R)
    assert poly.degree() == S(1)
