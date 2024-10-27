"""Tests for useful utilities for higher level polynomial classes. """

from sympy.core.mul import Mul
from sympy.core.numbers import (Integer, pi)
from sympy.core.relational import Eq
from sympy.core.singleton import S
from sympy.core.symbol import (Symbol, symbols)
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.integrals.integrals import Integral
from sympy.testing.pytest import raises

from sympy.polys.polyutils import (
    _nsort,
    _sort_gens,
    _unify_gens,
    _analyze_gens,
    _sort_factors,
    parallel_dict_from_expr,
    dict_from_expr,
)

from sympy.polys.polyerrors import PolynomialError

from sympy.polys.domains import ZZ

x, y, z, p, q, r, s, t, u, v, w = symbols('x,y,z,p,q,r,s,t,u,v,w')
A, B = symbols('A,B', commutative=False)


def test__nsort():
    # issue 6137
    r = S('''[3/2 + sqrt(-14/3 - 2*(-415/216 + 13*I/12)**(1/3) - 4/sqrt(-7/3 +
    61/(18*(-415/216 + 13*I/12)**(1/3)) + 2*(-415/216 + 13*I/12)**(1/3)) -
    61/(18*(-415/216 + 13*I/12)**(1/3)))/2 - sqrt(-7/3 + 61/(18*(-415/216
    + 13*I/12)**(1/3)) + 2*(-415/216 + 13*I/12)**(1/3))/2, 3/2 - sqrt(-7/3
    + 61/(18*(-415/216 + 13*I/12)**(1/3)) + 2*(-415/216 +
    13*I/12)**(1/3))/2 - sqrt(-14/3 - 2*(-415/216 + 13*I/12)**(1/3) -
    4/sqrt(-7/3 + 61/(18*(-415/216 + 13*I/12)**(1/3)) + 2*(-415/216 +
    13*I/12)**(1/3)) - 61/(18*(-415/216 + 13*I/12)**(1/3)))/2, 3/2 +
    sqrt(-14/3 - 2*(-415/216 + 13*I/12)**(1/3) + 4/sqrt(-7/3 +
    61/(18*(-415/216 + 13*I/12)**(1/3)) + 2*(-415/216 + 13*I/12)**(1/3)) -
    61/(18*(-415/216 + 13*I/12)**(1/3)))/2 + sqrt(-7/3 + 61/(18*(-415/216
    + 13*I/12)**(1/3)) + 2*(-415/216 + 13*I/12)**(1/3))/2, 3/2 + sqrt(-7/3
    + 61/(18*(-415/216 + 13*I/12)**(1/3)) + 2*(-415/216 +
    13*I/12)**(1/3))/2 - sqrt(-14/3 - 2*(-415/216 + 13*I/12)**(1/3) +
    4/sqrt(-7/3 + 61/(18*(-415/216 + 13*I/12)**(1/3)) + 2*(-415/216 +
    13*I/12)**(1/3)) - 61/(18*(-415/216 + 13*I/12)**(1/3)))/2]''')
    ans = [r[1], r[0], r[-1], r[-2]]
    assert _nsort(r) == ans
    assert len(_nsort(r, separated=True)[0]) == 0
    b, c, a = exp(-1000), exp(-999), exp(-1001)
    assert _nsort((b, c, a)) == [a, b, c]
    # issue 12560
    a = cos(1)**2 + sin(1)**2 - 1
    assert _nsort([a]) == [a]


def test__sort_gens():
    assert _sort_gens([]) == ()

    assert _sort_gens([x]) == (x,)
    assert _sort_gens([p]) == (p,)
    assert _sort_gens([q]) == (q,)

    assert _sort_gens([x, p]) == (x, p)
    assert _sort_gens([p, x]) == (x, p)
    assert _sort_gens([q, p]) == (p, q)

    assert _sort_gens([q, p, x]) == (x, p, q)

    assert _sort_gens([x, p, q], wrt=x) == (x, p, q)
    assert _sort_gens([x, p, q], wrt=p) == (p, x, q)
    assert _sort_gens([x, p, q], wrt=q) == (q, x, p)

    assert _sort_gens([x, p, q], wrt='x') == (x, p, q)
    assert _sort_gens([x, p, q], wrt='p') == (p, x, q)
    assert _sort_gens([x, p, q], wrt='q') == (q, x, p)

    assert _sort_gens([x, p, q], wrt='x,q') == (x, q, p)
    assert _sort_gens([x, p, q], wrt='q,x') == (q, x, p)
    assert _sort_gens([x, p, q], wrt='p,q') == (p, q, x)
    assert _sort_gens([x, p, q], wrt='q,p') == (q, p, x)

    assert _sort_gens([x, p, q], wrt='x, q') == (x, q, p)
    assert _sort_gens([x, p, q], wrt='q, x') == (q, x, p)
    assert _sort_gens([x, p, q], wrt='p, q') == (p, q, x)
    assert _sort_gens([x, p, q], wrt='q, p') == (q, p, x)

    assert _sort_gens([x, p, q], wrt=[x, 'q']) == (x, q, p)
    assert _sort_gens([x, p, q], wrt=[q, 'x']) == (q, x, p)
    assert _sort_gens([x, p, q], wrt=[p, 'q']) == (p, q, x)
    assert _sort_gens([x, p, q], wrt=[q, 'p']) == (q, p, x)

    assert _sort_gens([x, p, q], wrt=['x', 'q']) == (x, q, p)
    assert _sort_gens([x, p, q], wrt=['q', 'x']) == (q, x, p)
    assert _sort_gens([x, p, q], wrt=['p', 'q']) == (p, q, x)
    assert _sort_gens([x, p, q], wrt=['q', 'p']) == (q, p, x)

    assert _sort_gens([x, p, q], sort='x > p > q') == (x, p, q)
    assert _sort_gens([x, p, q], sort='p > x > q') == (p, x, q)
    assert _sort_gens([x, p, q], sort='p > q > x') == (p, q, x)

    assert _sort_gens([x, p, q], wrt='x', sort='q > p') == (x, q, p)
    assert _sort_gens([x, p, q], wrt='p', sort='q > x') == (p, q, x)
    assert _sort_gens([x, p, q], wrt='q', sort='p > x') == (q, p, x)

    # https://github.com/sympy/sympy/issues/19353
    n1 = Symbol('\n1')
    assert _sort_gens([n1]) == (n1,)
    assert _sort_gens([x, n1]) == (x, n1)

    X = symbols('x0,x1,x2,x10,x11,x12,x20,x21,x22')

    assert _sort_gens(X) == X


def test__unify_gens():
    assert _unify_gens([], []) == ()

    assert _unify_gens([x], [x]) == (x,)
    assert _unify_gens([y], [y]) == (y,)

    assert _unify_gens([x, y], [x]) == (x, y)
    assert _unify_gens([x], [x, y]) == (x, y)

    assert _unify_gens([x, y], [x, y]) == (x, y)
    assert _unify_gens([y, x], [y, x]) == (y, x)

    assert _unify_gens([x], [y]) == (x, y)
    assert _unify_gens([y], [x]) == (y, x)

    assert _unify_gens([x], [y, x]) == (y, x)
    assert _unify_gens([y, x], [x]) == (y, x)

    assert _unify_gens([x, y, z], [x, y, z]) == (x, y, z)
    assert _unify_gens([z, y, x], [x, y, z]) == (z, y, x)
    assert _unify_gens([x, y, z], [z, y, x]) == (x, y, z)
    assert _unify_gens([z, y, x], [z, y, x]) == (z, y, x)

    assert _unify_gens([x, y, z], [t, x, p, q, z]) == (t, x, y, p, q, z)


def test__analyze_gens():
    assert _analyze_gens((x, y, z)) == (x, y, z)
    assert _analyze_gens([x, y, z]) == (x, y, z)

    assert _analyze_gens(([x, y, z],)) == (x, y, z)
    assert _analyze_gens(((x, y, z),)) == (x, y, z)


def test__sort_factors():
    assert _sort_factors([], multiple=True) == []
    assert _sort_factors([], multiple=False) == []

    F = [[1, 2, 3], [1, 2], [1]]
    G = [[1], [1, 2], [1, 2, 3]]

    assert _sort_factors(F, multiple=False) == G

    F = [[1, 2], [1, 2, 3], [1, 2], [1]]
    G = [[1], [1, 2], [1, 2], [1, 2, 3]]

    assert _sort_factors(F, multiple=False) == G

    F = [[2, 2], [1, 2, 3], [1, 2], [1]]
    G = [[1], [1, 2], [2, 2], [1, 2, 3]]

    assert _sort_factors(F, multiple=False) == G

    F = [([1, 2, 3], 1), ([1, 2], 1), ([1], 1)]
    G = [([1], 1), ([1, 2], 1), ([1, 2, 3], 1)]

    assert _sort_factors(F, multiple=True) == G

    F = [([1, 2], 1), ([1, 2, 3], 1), ([1, 2], 1), ([1], 1)]
    G = [([1], 1), ([1, 2], 1), ([1, 2], 1), ([1, 2, 3], 1)]

    assert _sort_factors(F, multiple=True) == G

    F = [([2, 2], 1), ([1, 2, 3], 1), ([1, 2], 1), ([1], 1)]
    G = [([1], 1), ([1, 2], 1), ([2, 2], 1), ([1, 2, 3], 1)]

    assert _sort_factors(F, multiple=True) == G

    F = [([2, 2], 1), ([1, 2, 3], 1), ([1, 2], 2), ([1], 1)]
    G = [([1], 1), ([2, 2], 1), ([1, 2], 2), ([1, 2, 3], 1)]

    assert _sort_factors(F, multiple=True) == G


def test__dict_from_expr_if_gens():
    assert dict_from_expr(
        Integer(17), gens=(x,)) == ({(0,): Integer(17)}, (x,))
    assert dict_from_expr(
        Integer(17), gens=(x, y)) == ({(0, 0): Integer(17)}, (x, y))
    assert dict_from_expr(
        Integer(17), gens=(x, y, z)) == ({(0, 0, 0): Integer(17)}, (x, y, z))

    assert dict_from_expr(
        Integer(-17), gens=(x,)) == ({(0,): Integer(-17)}, (x,))
    assert dict_from_expr(
        Integer(-17), gens=(x, y)) == ({(0, 0): Integer(-17)}, (x, y))
    assert dict_from_expr(Integer(
        -17), gens=(x, y, z)) == ({(0, 0, 0): Integer(-17)}, (x, y, z))

    assert dict_from_expr(
        Integer(17)*x, gens=(x,)) == ({(1,): Integer(17)}, (x,))
    assert dict_from_expr(
        Integer(17)*x, gens=(x, y)) == ({(1, 0): Integer(17)}, (x, y))
    assert dict_from_expr(Integer(
        17)*x, gens=(x, y, z)) == ({(1, 0, 0): Integer(17)}, (x, y, z))

    assert dict_from_expr(
        Integer(17)*x**7, gens=(x,)) == ({(7,): Integer(17)}, (x,))
    assert dict_from_expr(
        Integer(17)*x**7*y, gens=(x, y)) == ({(7, 1): Integer(17)}, (x, y))
    assert dict_from_expr(Integer(17)*x**7*y*z**12, gens=(
        x, y, z)) == ({(7, 1, 12): Integer(17)}, (x, y, z))

    assert dict_from_expr(x + 2*y + 3*z, gens=(x,)) == \
        ({(1,): Integer(1), (0,): 2*y + 3*z}, (x,))
    assert dict_from_expr(x + 2*y + 3*z, gens=(x, y)) == \
        ({(1, 0): Integer(1), (0, 1): Integer(2), (0, 0): 3*z}, (x, y))
    assert dict_from_expr(x + 2*y + 3*z, gens=(x, y, z)) == \
        ({(1, 0, 0): Integer(
            1), (0, 1, 0): Integer(2), (0, 0, 1): Integer(3)}, (x, y, z))

    assert dict_from_expr(x*y + 2*x*z + 3*y*z, gens=(x,)) == \
        ({(1,): y + 2*z, (0,): 3*y*z}, (x,))
    assert dict_from_expr(x*y + 2*x*z + 3*y*z, gens=(x, y)) == \
        ({(1, 1): Integer(1), (1, 0): 2*z, (0, 1): 3*z}, (x, y))
    assert dict_from_expr(x*y + 2*x*z + 3*y*z, gens=(x, y, z)) == \
        ({(1, 1, 0): Integer(
            1), (1, 0, 1): Integer(2), (0, 1, 1): Integer(3)}, (x, y, z))

    assert dict_from_expr(2**y*x, gens=(x,)) == ({(1,): 2**y}, (x,))
    assert dict_from_expr(Integral(x, (x, 1, 2)) + x) == (
        {(0, 1): 1, (1, 0): 1}, (x, Integral(x, (x, 1, 2))))
    raises(PolynomialError, lambda: dict_from_expr(2**y*x, gens=(x, y)))


def test__dict_from_expr_no_gens():
    assert dict_from_expr(Integer(17)) == ({(): Integer(17)}, ())

    assert dict_from_expr(x) == ({(1,): Integer(1)}, (x,))
    assert dict_from_expr(y) == ({(1,): Integer(1)}, (y,))

    assert dict_from_expr(x*y) == ({(1, 1): Integer(1)}, (x, y))
    assert dict_from_expr(
        x + y) == ({(1, 0): Integer(1), (0, 1): Integer(1)}, (x, y))

    assert dict_from_expr(sqrt(2)) == ({(1,): Integer(1)}, (sqrt(2),))
    assert dict_from_expr(sqrt(2), greedy=False) == ({(): sqrt(2)}, ())

    assert dict_from_expr(x*y, domain=ZZ[x]) == ({(1,): x}, (y,))
    assert dict_from_expr(x*y, domain=ZZ[y]) == ({(1,): y}, (x,))

    assert dict_from_expr(3*sqrt(
        2)*pi*x*y, extension=None) == ({(1, 1, 1, 1): 3}, (x, y, pi, sqrt(2)))
    assert dict_from_expr(3*sqrt(
        2)*pi*x*y, extension=True) == ({(1, 1, 1): 3*sqrt(2)}, (x, y, pi))

    assert dict_from_expr(3*sqrt(
        2)*pi*x*y, extension=True) == ({(1, 1, 1): 3*sqrt(2)}, (x, y, pi))

    f = cos(x)*sin(x) + cos(x)*sin(y) + cos(y)*sin(x) + cos(y)*sin(y)

    assert dict_from_expr(f) == ({(0, 1, 0, 1): 1, (0, 1, 1, 0): 1,
        (1, 0, 0, 1): 1, (1, 0, 1, 0): 1}, (cos(x), cos(y), sin(x), sin(y)))


def test__parallel_dict_from_expr_if_gens():
    assert parallel_dict_from_expr([x + 2*y + 3*z, Integer(7)], gens=(x,)) == \
        ([{(1,): Integer(1), (0,): 2*y + 3*z}, {(0,): Integer(7)}], (x,))


def test__parallel_dict_from_expr_no_gens():
    assert parallel_dict_from_expr([x*y, Integer(3)]) == \
        ([{(1, 1): Integer(1)}, {(0, 0): Integer(3)}], (x, y))
    assert parallel_dict_from_expr([x*y, 2*z, Integer(3)]) == \
        ([{(1, 1, 0): Integer(
            1)}, {(0, 0, 1): Integer(2)}, {(0, 0, 0): Integer(3)}], (x, y, z))
    assert parallel_dict_from_expr((Mul(x, x**2, evaluate=False),)) == \
        ([{(3,): 1}], (x,))


def test_parallel_dict_from_expr():
    assert parallel_dict_from_expr([Eq(x, 1), Eq(
        x**2, 2)]) == ([{(0,): -Integer(1), (1,): Integer(1)},
                        {(0,): -Integer(2), (2,): Integer(1)}], (x,))
    raises(PolynomialError, lambda: parallel_dict_from_expr([A*B - B*A]))


def test_dict_from_expr():
    assert dict_from_expr(Eq(x, 1)) == \
        ({(0,): -Integer(1), (1,): Integer(1)}, (x,))
    raises(PolynomialError, lambda: dict_from_expr(A*B - B*A))
    raises(PolynomialError, lambda: dict_from_expr(S.true))
