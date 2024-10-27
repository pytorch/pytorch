from sympy.core.numbers import (Float, pi)
from sympy.core.symbol import symbols
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.matrices.immutable import ImmutableDenseMatrix as Matrix
from sympy.physics.vector import ReferenceFrame, dynamicsymbols, outer
from sympy.physics.vector.dyadic import _check_dyadic
from sympy.testing.pytest import raises

A = ReferenceFrame('A')


def test_dyadic():
    d1 = A.x | A.x
    d2 = A.y | A.y
    d3 = A.x | A.y
    assert d1 * 0 == 0
    assert d1 != 0
    assert d1 * 2 == 2 * A.x | A.x
    assert d1 / 2. == 0.5 * d1
    assert d1 & (0 * d1) == 0
    assert d1 & d2 == 0
    assert d1 & A.x == A.x
    assert d1 ^ A.x == 0
    assert d1 ^ A.y == A.x | A.z
    assert d1 ^ A.z == - A.x | A.y
    assert d2 ^ A.x == - A.y | A.z
    assert A.x ^ d1 == 0
    assert A.y ^ d1 == - A.z | A.x
    assert A.z ^ d1 == A.y | A.x
    assert A.x & d1 == A.x
    assert A.y & d1 == 0
    assert A.y & d2 == A.y
    assert d1 & d3 == A.x | A.y
    assert d3 & d1 == 0
    assert d1.dt(A) == 0
    q = dynamicsymbols('q')
    qd = dynamicsymbols('q', 1)
    B = A.orientnew('B', 'Axis', [q, A.z])
    assert d1.express(B) == d1.express(B, B)
    assert d1.express(B) == ((cos(q)**2) * (B.x | B.x) + (-sin(q) * cos(q)) *
            (B.x | B.y) + (-sin(q) * cos(q)) * (B.y | B.x) + (sin(q)**2) *
            (B.y | B.y))
    assert d1.express(B, A) == (cos(q)) * (B.x | A.x) + (-sin(q)) * (B.y | A.x)
    assert d1.express(A, B) == (cos(q)) * (A.x | B.x) + (-sin(q)) * (A.x | B.y)
    assert d1.dt(B) == (-qd) * (A.y | A.x) + (-qd) * (A.x | A.y)

    assert d1.to_matrix(A) == Matrix([[1, 0, 0], [0, 0, 0], [0, 0, 0]])
    assert d1.to_matrix(A, B) == Matrix([[cos(q), -sin(q), 0],
                                         [0, 0, 0],
                                         [0, 0, 0]])
    assert d3.to_matrix(A) == Matrix([[0, 1, 0], [0, 0, 0], [0, 0, 0]])
    a, b, c, d, e, f = symbols('a, b, c, d, e, f')
    v1 = a * A.x + b * A.y + c * A.z
    v2 = d * A.x + e * A.y + f * A.z
    d4 = v1.outer(v2)
    assert d4.to_matrix(A) == Matrix([[a * d, a * e, a * f],
                                      [b * d, b * e, b * f],
                                      [c * d, c * e, c * f]])
    d5 = v1.outer(v1)
    C = A.orientnew('C', 'Axis', [q, A.x])
    for expected, actual in zip(C.dcm(A) * d5.to_matrix(A) * C.dcm(A).T,
                                d5.to_matrix(C)):
        assert (expected - actual).simplify() == 0

    raises(TypeError, lambda: d1.applyfunc(0))


def test_dyadic_simplify():
    x, y, z, k, n, m, w, f, s, A = symbols('x, y, z, k, n, m, w, f, s, A')
    N = ReferenceFrame('N')

    dy = N.x | N.x
    test1 = (1 / x + 1 / y) * dy
    assert (N.x & test1 & N.x) != (x + y) / (x * y)
    test1 = test1.simplify()
    assert (N.x & test1 & N.x) == (x + y) / (x * y)

    test2 = (A**2 * s**4 / (4 * pi * k * m**3)) * dy
    test2 = test2.simplify()
    assert (N.x & test2 & N.x) == (A**2 * s**4 / (4 * pi * k * m**3))

    test3 = ((4 + 4 * x - 2 * (2 + 2 * x)) / (2 + 2 * x)) * dy
    test3 = test3.simplify()
    assert (N.x & test3 & N.x) == 0

    test4 = ((-4 * x * y**2 - 2 * y**3 - 2 * x**2 * y) / (x + y)**2) * dy
    test4 = test4.simplify()
    assert (N.x & test4 & N.x) == -2 * y


def test_dyadic_subs():
    N = ReferenceFrame('N')
    s = symbols('s')
    a = s*(N.x | N.x)
    assert a.subs({s: 2}) == 2*(N.x | N.x)


def test_check_dyadic():
    raises(TypeError, lambda: _check_dyadic(0))


def test_dyadic_evalf():
    N = ReferenceFrame('N')
    a = pi * (N.x | N.x)
    assert a.evalf(3) == Float('3.1416', 3) * (N.x | N.x)
    s = symbols('s')
    a = 5 * s * pi* (N.x | N.x)
    assert a.evalf(2) == Float('5', 2) * Float('3.1416', 2) * s * (N.x | N.x)
    assert a.evalf(9, subs={s: 5.124}) == Float('80.48760378', 9) * (N.x | N.x)


def test_dyadic_xreplace():
    x, y, z = symbols('x y z')
    N = ReferenceFrame('N')
    D = outer(N.x, N.x)
    v = x*y * D
    assert v.xreplace({x : cos(x)}) == cos(x)*y * D
    assert v.xreplace({x*y : pi}) == pi * D
    v = (x*y)**z * D
    assert v.xreplace({(x*y)**z : 1}) == D
    assert v.xreplace({x:1, z:0}) == D
    raises(TypeError, lambda: v.xreplace())
    raises(TypeError, lambda: v.xreplace([x, y]))
