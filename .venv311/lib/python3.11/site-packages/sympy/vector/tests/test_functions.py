from sympy.vector.vector import Vector
from sympy.vector.coordsysrect import CoordSys3D
from sympy.vector.functions import express, matrix_to_vector, orthogonalize
from sympy.core.numbers import Rational
from sympy.core.singleton import S
from sympy.core.symbol import symbols
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.matrices.immutable import ImmutableDenseMatrix as Matrix
from sympy.testing.pytest import raises

N = CoordSys3D('N')
q1, q2, q3, q4, q5 = symbols('q1 q2 q3 q4 q5')
A = N.orient_new_axis('A', q1, N.k)  # type: ignore
B = A.orient_new_axis('B', q2, A.i)
C = B.orient_new_axis('C', q3, B.j)


def test_express():
    assert express(Vector.zero, N) == Vector.zero
    assert express(S.Zero, N) is S.Zero
    assert express(A.i, C) == cos(q3)*C.i + sin(q3)*C.k
    assert express(A.j, C) == sin(q2)*sin(q3)*C.i + cos(q2)*C.j - \
        sin(q2)*cos(q3)*C.k
    assert express(A.k, C) == -sin(q3)*cos(q2)*C.i + sin(q2)*C.j + \
        cos(q2)*cos(q3)*C.k
    assert express(A.i, N) == cos(q1)*N.i + sin(q1)*N.j
    assert express(A.j, N) == -sin(q1)*N.i + cos(q1)*N.j
    assert express(A.k, N) == N.k
    assert express(A.i, A) == A.i
    assert express(A.j, A) == A.j
    assert express(A.k, A) == A.k
    assert express(A.i, B) == B.i
    assert express(A.j, B) == cos(q2)*B.j - sin(q2)*B.k
    assert express(A.k, B) == sin(q2)*B.j + cos(q2)*B.k
    assert express(A.i, C) == cos(q3)*C.i + sin(q3)*C.k
    assert express(A.j, C) == sin(q2)*sin(q3)*C.i + cos(q2)*C.j - \
        sin(q2)*cos(q3)*C.k
    assert express(A.k, C) == -sin(q3)*cos(q2)*C.i + sin(q2)*C.j + \
        cos(q2)*cos(q3)*C.k
    # Check to make sure UnitVectors get converted properly
    assert express(N.i, N) == N.i
    assert express(N.j, N) == N.j
    assert express(N.k, N) == N.k
    assert express(N.i, A) == (cos(q1)*A.i - sin(q1)*A.j)
    assert express(N.j, A) == (sin(q1)*A.i + cos(q1)*A.j)
    assert express(N.k, A) == A.k
    assert express(N.i, B) == (cos(q1)*B.i - sin(q1)*cos(q2)*B.j +
            sin(q1)*sin(q2)*B.k)
    assert express(N.j, B) == (sin(q1)*B.i + cos(q1)*cos(q2)*B.j -
            sin(q2)*cos(q1)*B.k)
    assert express(N.k, B) == (sin(q2)*B.j + cos(q2)*B.k)
    assert express(N.i, C) == (
        (cos(q1)*cos(q3) - sin(q1)*sin(q2)*sin(q3))*C.i -
        sin(q1)*cos(q2)*C.j +
        (sin(q3)*cos(q1) + sin(q1)*sin(q2)*cos(q3))*C.k)
    assert express(N.j, C) == (
        (sin(q1)*cos(q3) + sin(q2)*sin(q3)*cos(q1))*C.i +
        cos(q1)*cos(q2)*C.j +
        (sin(q1)*sin(q3) - sin(q2)*cos(q1)*cos(q3))*C.k)
    assert express(N.k, C) == (-sin(q3)*cos(q2)*C.i + sin(q2)*C.j +
            cos(q2)*cos(q3)*C.k)

    assert express(A.i, N) == (cos(q1)*N.i + sin(q1)*N.j)
    assert express(A.j, N) == (-sin(q1)*N.i + cos(q1)*N.j)
    assert express(A.k, N) == N.k
    assert express(A.i, A) == A.i
    assert express(A.j, A) == A.j
    assert express(A.k, A) == A.k
    assert express(A.i, B) == B.i
    assert express(A.j, B) == (cos(q2)*B.j - sin(q2)*B.k)
    assert express(A.k, B) == (sin(q2)*B.j + cos(q2)*B.k)
    assert express(A.i, C) == (cos(q3)*C.i + sin(q3)*C.k)
    assert express(A.j, C) == (sin(q2)*sin(q3)*C.i + cos(q2)*C.j -
            sin(q2)*cos(q3)*C.k)
    assert express(A.k, C) == (-sin(q3)*cos(q2)*C.i + sin(q2)*C.j +
            cos(q2)*cos(q3)*C.k)

    assert express(B.i, N) == (cos(q1)*N.i + sin(q1)*N.j)
    assert express(B.j, N) == (-sin(q1)*cos(q2)*N.i +
            cos(q1)*cos(q2)*N.j + sin(q2)*N.k)
    assert express(B.k, N) == (sin(q1)*sin(q2)*N.i -
            sin(q2)*cos(q1)*N.j + cos(q2)*N.k)
    assert express(B.i, A) == A.i
    assert express(B.j, A) == (cos(q2)*A.j + sin(q2)*A.k)
    assert express(B.k, A) == (-sin(q2)*A.j + cos(q2)*A.k)
    assert express(B.i, B) == B.i
    assert express(B.j, B) == B.j
    assert express(B.k, B) == B.k
    assert express(B.i, C) == (cos(q3)*C.i + sin(q3)*C.k)
    assert express(B.j, C) == C.j
    assert express(B.k, C) == (-sin(q3)*C.i + cos(q3)*C.k)

    assert express(C.i, N) == (
        (cos(q1)*cos(q3) - sin(q1)*sin(q2)*sin(q3))*N.i +
        (sin(q1)*cos(q3) + sin(q2)*sin(q3)*cos(q1))*N.j -
        sin(q3)*cos(q2)*N.k)
    assert express(C.j, N) == (
        -sin(q1)*cos(q2)*N.i + cos(q1)*cos(q2)*N.j + sin(q2)*N.k)
    assert express(C.k, N) == (
        (sin(q3)*cos(q1) + sin(q1)*sin(q2)*cos(q3))*N.i +
        (sin(q1)*sin(q3) - sin(q2)*cos(q1)*cos(q3))*N.j +
        cos(q2)*cos(q3)*N.k)
    assert express(C.i, A) == (cos(q3)*A.i + sin(q2)*sin(q3)*A.j -
            sin(q3)*cos(q2)*A.k)
    assert express(C.j, A) == (cos(q2)*A.j + sin(q2)*A.k)
    assert express(C.k, A) == (sin(q3)*A.i - sin(q2)*cos(q3)*A.j +
            cos(q2)*cos(q3)*A.k)
    assert express(C.i, B) == (cos(q3)*B.i - sin(q3)*B.k)
    assert express(C.j, B) == B.j
    assert express(C.k, B) == (sin(q3)*B.i + cos(q3)*B.k)
    assert express(C.i, C) == C.i
    assert express(C.j, C) == C.j
    assert express(C.k, C) == C.k == (C.k)

    #  Check to make sure Vectors get converted back to UnitVectors
    assert N.i == express((cos(q1)*A.i - sin(q1)*A.j), N).simplify()
    assert N.j == express((sin(q1)*A.i + cos(q1)*A.j), N).simplify()
    assert N.i == express((cos(q1)*B.i - sin(q1)*cos(q2)*B.j +
            sin(q1)*sin(q2)*B.k), N).simplify()
    assert N.j == express((sin(q1)*B.i + cos(q1)*cos(q2)*B.j -
        sin(q2)*cos(q1)*B.k), N).simplify()
    assert N.k == express((sin(q2)*B.j + cos(q2)*B.k), N).simplify()


    assert A.i == express((cos(q1)*N.i + sin(q1)*N.j), A).simplify()
    assert A.j == express((-sin(q1)*N.i + cos(q1)*N.j), A).simplify()

    assert A.j == express((cos(q2)*B.j - sin(q2)*B.k), A).simplify()
    assert A.k == express((sin(q2)*B.j + cos(q2)*B.k), A).simplify()

    assert A.i == express((cos(q3)*C.i + sin(q3)*C.k), A).simplify()
    assert A.j == express((sin(q2)*sin(q3)*C.i + cos(q2)*C.j -
            sin(q2)*cos(q3)*C.k), A).simplify()

    assert A.k == express((-sin(q3)*cos(q2)*C.i + sin(q2)*C.j +
            cos(q2)*cos(q3)*C.k), A).simplify()
    assert B.i == express((cos(q1)*N.i + sin(q1)*N.j), B).simplify()
    assert B.j == express((-sin(q1)*cos(q2)*N.i +
            cos(q1)*cos(q2)*N.j + sin(q2)*N.k), B).simplify()

    assert B.k == express((sin(q1)*sin(q2)*N.i -
            sin(q2)*cos(q1)*N.j + cos(q2)*N.k), B).simplify()

    assert B.j == express((cos(q2)*A.j + sin(q2)*A.k), B).simplify()
    assert B.k == express((-sin(q2)*A.j + cos(q2)*A.k), B).simplify()
    assert B.i == express((cos(q3)*C.i + sin(q3)*C.k), B).simplify()
    assert B.k == express((-sin(q3)*C.i + cos(q3)*C.k), B).simplify()
    assert C.i == express((cos(q3)*A.i + sin(q2)*sin(q3)*A.j -
            sin(q3)*cos(q2)*A.k), C).simplify()
    assert C.j == express((cos(q2)*A.j + sin(q2)*A.k), C).simplify()
    assert C.k == express((sin(q3)*A.i - sin(q2)*cos(q3)*A.j +
            cos(q2)*cos(q3)*A.k), C).simplify()
    assert C.i == express((cos(q3)*B.i - sin(q3)*B.k), C).simplify()
    assert C.k == express((sin(q3)*B.i + cos(q3)*B.k), C).simplify()


def test_matrix_to_vector():
    m = Matrix([[1], [2], [3]])
    assert matrix_to_vector(m, C) == C.i + 2*C.j + 3*C.k
    m = Matrix([[0], [0], [0]])
    assert matrix_to_vector(m, N) == matrix_to_vector(m, C) == \
           Vector.zero
    m = Matrix([[q1], [q2], [q3]])
    assert matrix_to_vector(m, N) == q1*N.i + q2*N.j + q3*N.k


def test_orthogonalize():
    C = CoordSys3D('C')
    a, b = symbols('a b', integer=True)
    i, j, k = C.base_vectors()
    v1 = i + 2*j
    v2 = 2*i + 3*j
    v3 = 3*i + 5*j
    v4 = 3*i + j
    v5 = 2*i + 2*j
    v6 = a*i + b*j
    v7 = 4*a*i + 4*b*j
    assert orthogonalize(v1, v2) == [C.i + 2*C.j, C.i*Rational(2, 5) + -C.j/5]
    # from wikipedia
    assert orthogonalize(v4, v5, orthonormal=True) == \
        [(3*sqrt(10))*C.i/10 + (sqrt(10))*C.j/10, (-sqrt(10))*C.i/10 + (3*sqrt(10))*C.j/10]
    raises(ValueError, lambda: orthogonalize(v1, v2, v3))
    raises(ValueError, lambda: orthogonalize(v6, v7))
