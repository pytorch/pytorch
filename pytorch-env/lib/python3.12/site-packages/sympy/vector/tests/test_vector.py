from sympy.core import Rational, S
from sympy.simplify import simplify, trigsimp
from sympy.core.function import (Derivative, Function, diff)
from sympy.core.numbers import pi
from sympy.core.symbol import symbols
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.integrals.integrals import Integral
from sympy.matrices.immutable import ImmutableDenseMatrix as Matrix
from sympy.vector.vector import Vector, BaseVector, VectorAdd, \
     VectorMul, VectorZero
from sympy.vector.coordsysrect import CoordSys3D
from sympy.vector.vector import Cross, Dot, cross
from sympy.testing.pytest import raises

C = CoordSys3D('C')

i, j, k = C.base_vectors()
a, b, c = symbols('a b c')


def test_cross():
    v1 = C.x * i + C.z * C.z * j
    v2 = C.x * i + C.y * j + C.z * k
    assert Cross(v1, v2) == Cross(C.x*C.i + C.z**2*C.j, C.x*C.i + C.y*C.j + C.z*C.k)
    assert Cross(v1, v2).doit() == C.z**3*C.i + (-C.x*C.z)*C.j + (C.x*C.y - C.x*C.z**2)*C.k
    assert cross(v1, v2) == C.z**3*C.i + (-C.x*C.z)*C.j + (C.x*C.y - C.x*C.z**2)*C.k
    assert Cross(v1, v2) == -Cross(v2, v1)
    assert Cross(v1, v2) + Cross(v2, v1) == Vector.zero


def test_dot():
    v1 = C.x * i + C.z * C.z * j
    v2 = C.x * i + C.y * j + C.z * k
    assert Dot(v1, v2) == Dot(C.x*C.i + C.z**2*C.j, C.x*C.i + C.y*C.j + C.z*C.k)
    assert Dot(v1, v2).doit() == C.x**2 + C.y*C.z**2
    assert Dot(v1, v2).doit() == C.x**2 + C.y*C.z**2
    assert Dot(v1, v2) == Dot(v2, v1)


def test_vector_sympy():
    """
    Test whether the Vector framework confirms to the hashing
    and equality testing properties of SymPy.
    """
    v1 = 3*j
    assert v1 == j*3
    assert v1.components == {j: 3}
    v2 = 3*i + 4*j + 5*k
    v3 = 2*i + 4*j + i + 4*k + k
    assert v3 == v2
    assert v3.__hash__() == v2.__hash__()


def test_vector():
    assert isinstance(i, BaseVector)
    assert i != j
    assert j != k
    assert k != i
    assert i - i == Vector.zero
    assert i + Vector.zero == i
    assert i - Vector.zero == i
    assert Vector.zero != 0
    assert -Vector.zero == Vector.zero

    v1 = a*i + b*j + c*k
    v2 = a**2*i + b**2*j + c**2*k
    v3 = v1 + v2
    v4 = 2 * v1
    v5 = a * i

    assert isinstance(v1, VectorAdd)
    assert v1 - v1 == Vector.zero
    assert v1 + Vector.zero == v1
    assert v1.dot(i) == a
    assert v1.dot(j) == b
    assert v1.dot(k) == c
    assert i.dot(v2) == a**2
    assert j.dot(v2) == b**2
    assert k.dot(v2) == c**2
    assert v3.dot(i) == a**2 + a
    assert v3.dot(j) == b**2 + b
    assert v3.dot(k) == c**2 + c

    assert v1 + v2 == v2 + v1
    assert v1 - v2 == -1 * (v2 - v1)
    assert a * v1 == v1 * a

    assert isinstance(v5, VectorMul)
    assert v5.base_vector == i
    assert v5.measure_number == a
    assert isinstance(v4, Vector)
    assert isinstance(v4, VectorAdd)
    assert isinstance(v4, Vector)
    assert isinstance(Vector.zero, VectorZero)
    assert isinstance(Vector.zero, Vector)
    assert isinstance(v1 * 0, VectorZero)

    assert v1.to_matrix(C) == Matrix([[a], [b], [c]])

    assert i.components == {i: 1}
    assert v5.components == {i: a}
    assert v1.components == {i: a, j: b, k: c}

    assert VectorAdd(v1, Vector.zero) == v1
    assert VectorMul(a, v1) == v1*a
    assert VectorMul(1, i) == i
    assert VectorAdd(v1, Vector.zero) == v1
    assert VectorMul(0, Vector.zero) == Vector.zero
    raises(TypeError, lambda: v1.outer(1))
    raises(TypeError, lambda: v1.dot(1))


def test_vector_magnitude_normalize():
    assert Vector.zero.magnitude() == 0
    assert Vector.zero.normalize() == Vector.zero

    assert i.magnitude() == 1
    assert j.magnitude() == 1
    assert k.magnitude() == 1
    assert i.normalize() == i
    assert j.normalize() == j
    assert k.normalize() == k

    v1 = a * i
    assert v1.normalize() == (a/sqrt(a**2))*i
    assert v1.magnitude() == sqrt(a**2)

    v2 = a*i + b*j + c*k
    assert v2.magnitude() == sqrt(a**2 + b**2 + c**2)
    assert v2.normalize() == v2 / v2.magnitude()

    v3 = i + j
    assert v3.normalize() == (sqrt(2)/2)*C.i + (sqrt(2)/2)*C.j


def test_vector_simplify():
    A, s, k, m = symbols('A, s, k, m')

    test1 = (1 / a + 1 / b) * i
    assert (test1 & i) != (a + b) / (a * b)
    test1 = simplify(test1)
    assert (test1 & i) == (a + b) / (a * b)
    assert test1.simplify() == simplify(test1)

    test2 = (A**2 * s**4 / (4 * pi * k * m**3)) * i
    test2 = simplify(test2)
    assert (test2 & i) == (A**2 * s**4 / (4 * pi * k * m**3))

    test3 = ((4 + 4 * a - 2 * (2 + 2 * a)) / (2 + 2 * a)) * i
    test3 = simplify(test3)
    assert (test3 & i) == 0

    test4 = ((-4 * a * b**2 - 2 * b**3 - 2 * a**2 * b) / (a + b)**2) * i
    test4 = simplify(test4)
    assert (test4 & i) == -2 * b

    v = (sin(a)+cos(a))**2*i - j
    assert trigsimp(v) == (2*sin(a + pi/4)**2)*i + (-1)*j
    assert trigsimp(v) == v.trigsimp()

    assert simplify(Vector.zero) == Vector.zero


def test_vector_dot():
    assert i.dot(Vector.zero) == 0
    assert Vector.zero.dot(i) == 0
    assert i & Vector.zero == 0

    assert i.dot(i) == 1
    assert i.dot(j) == 0
    assert i.dot(k) == 0
    assert i & i == 1
    assert i & j == 0
    assert i & k == 0

    assert j.dot(i) == 0
    assert j.dot(j) == 1
    assert j.dot(k) == 0
    assert j & i == 0
    assert j & j == 1
    assert j & k == 0

    assert k.dot(i) == 0
    assert k.dot(j) == 0
    assert k.dot(k) == 1
    assert k & i == 0
    assert k & j == 0
    assert k & k == 1

    raises(TypeError, lambda: k.dot(1))


def test_vector_cross():
    assert i.cross(Vector.zero) == Vector.zero
    assert Vector.zero.cross(i) == Vector.zero

    assert i.cross(i) == Vector.zero
    assert i.cross(j) == k
    assert i.cross(k) == -j
    assert i ^ i == Vector.zero
    assert i ^ j == k
    assert i ^ k == -j

    assert j.cross(i) == -k
    assert j.cross(j) == Vector.zero
    assert j.cross(k) == i
    assert j ^ i == -k
    assert j ^ j == Vector.zero
    assert j ^ k == i

    assert k.cross(i) == j
    assert k.cross(j) == -i
    assert k.cross(k) == Vector.zero
    assert k ^ i == j
    assert k ^ j == -i
    assert k ^ k == Vector.zero

    assert k.cross(1) == Cross(k, 1)


def test_projection():
    v1 = i + j + k
    v2 = 3*i + 4*j
    v3 = 0*i + 0*j
    assert v1.projection(v1) == i + j + k
    assert v1.projection(v2) == Rational(7, 3)*C.i + Rational(7, 3)*C.j + Rational(7, 3)*C.k
    assert v1.projection(v1, scalar=True) == S.One
    assert v1.projection(v2, scalar=True) == Rational(7, 3)
    assert v3.projection(v1) == Vector.zero
    assert v3.projection(v1, scalar=True) == S.Zero


def test_vector_diff_integrate():
    f = Function('f')
    v = f(a)*C.i + a**2*C.j - C.k
    assert Derivative(v, a) == Derivative((f(a))*C.i +
                                          a**2*C.j + (-1)*C.k, a)
    assert (diff(v, a) == v.diff(a) == Derivative(v, a).doit() ==
            (Derivative(f(a), a))*C.i + 2*a*C.j)
    assert (Integral(v, a) == (Integral(f(a), a))*C.i +
            (Integral(a**2, a))*C.j + (Integral(-1, a))*C.k)


def test_vector_args():
    raises(ValueError, lambda: BaseVector(3, C))
    raises(TypeError, lambda: BaseVector(0, Vector.zero))


def test_srepr():
    from sympy.printing.repr import srepr
    res = "CoordSys3D(Str('C'), Tuple(ImmutableDenseMatrix([[Integer(1), "\
            "Integer(0), Integer(0)], [Integer(0), Integer(1), Integer(0)], "\
            "[Integer(0), Integer(0), Integer(1)]]), VectorZero())).i"
    assert srepr(C.i) == res


def test_scalar():
    from sympy.vector import CoordSys3D
    C = CoordSys3D('C')
    v1 = 3*C.i + 4*C.j + 5*C.k
    v2 = 3*C.i - 4*C.j + 5*C.k
    assert v1.is_Vector is True
    assert v1.is_scalar is False
    assert (v1.dot(v2)).is_scalar is True
    assert (v1.cross(v2)).is_scalar is False
