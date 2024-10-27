from sympy.core.numbers import pi
from sympy.core.singleton import S
from sympy.core.symbol import symbols
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.integrals.integrals import Integral
from sympy.physics.vector import Dyadic, Point, ReferenceFrame, Vector
from sympy.physics.vector.functions import (cross, dot, express,
                                            time_derivative,
                                            kinematic_equations, outer,
                                            partial_velocity,
                                            get_motion_params, dynamicsymbols)
from sympy.simplify import trigsimp
from sympy.testing.pytest import raises

q1, q2, q3, q4, q5 = symbols('q1 q2 q3 q4 q5')
N = ReferenceFrame('N')
A = N.orientnew('A', 'Axis', [q1, N.z])
B = A.orientnew('B', 'Axis', [q2, A.x])
C = B.orientnew('C', 'Axis', [q3, B.y])


def test_dot():
    assert dot(A.x, A.x) == 1
    assert dot(A.x, A.y) == 0
    assert dot(A.x, A.z) == 0

    assert dot(A.y, A.x) == 0
    assert dot(A.y, A.y) == 1
    assert dot(A.y, A.z) == 0

    assert dot(A.z, A.x) == 0
    assert dot(A.z, A.y) == 0
    assert dot(A.z, A.z) == 1


def test_dot_different_frames():
    assert dot(N.x, A.x) == cos(q1)
    assert dot(N.x, A.y) == -sin(q1)
    assert dot(N.x, A.z) == 0
    assert dot(N.y, A.x) == sin(q1)
    assert dot(N.y, A.y) == cos(q1)
    assert dot(N.y, A.z) == 0
    assert dot(N.z, A.x) == 0
    assert dot(N.z, A.y) == 0
    assert dot(N.z, A.z) == 1

    assert trigsimp(dot(N.x, A.x + A.y)) == sqrt(2)*cos(q1 + pi/4)
    assert trigsimp(dot(N.x, A.x + A.y)) == trigsimp(dot(A.x + A.y, N.x))

    assert dot(A.x, C.x) == cos(q3)
    assert dot(A.x, C.y) == 0
    assert dot(A.x, C.z) == sin(q3)
    assert dot(A.y, C.x) == sin(q2)*sin(q3)
    assert dot(A.y, C.y) == cos(q2)
    assert dot(A.y, C.z) == -sin(q2)*cos(q3)
    assert dot(A.z, C.x) == -cos(q2)*sin(q3)
    assert dot(A.z, C.y) == sin(q2)
    assert dot(A.z, C.z) == cos(q2)*cos(q3)


def test_cross():
    assert cross(A.x, A.x) == 0
    assert cross(A.x, A.y) == A.z
    assert cross(A.x, A.z) == -A.y

    assert cross(A.y, A.x) == -A.z
    assert cross(A.y, A.y) == 0
    assert cross(A.y, A.z) == A.x

    assert cross(A.z, A.x) == A.y
    assert cross(A.z, A.y) == -A.x
    assert cross(A.z, A.z) == 0


def test_cross_different_frames():
    assert cross(N.x, A.x) == sin(q1)*A.z
    assert cross(N.x, A.y) == cos(q1)*A.z
    assert cross(N.x, A.z) == -sin(q1)*A.x - cos(q1)*A.y
    assert cross(N.y, A.x) == -cos(q1)*A.z
    assert cross(N.y, A.y) == sin(q1)*A.z
    assert cross(N.y, A.z) == cos(q1)*A.x - sin(q1)*A.y
    assert cross(N.z, A.x) == A.y
    assert cross(N.z, A.y) == -A.x
    assert cross(N.z, A.z) == 0

    assert cross(N.x, A.x) == sin(q1)*A.z
    assert cross(N.x, A.y) == cos(q1)*A.z
    assert cross(N.x, A.x + A.y) == sin(q1)*A.z + cos(q1)*A.z
    assert cross(A.x + A.y, N.x) == -sin(q1)*A.z - cos(q1)*A.z

    assert cross(A.x, C.x) == sin(q3)*C.y
    assert cross(A.x, C.y) == -sin(q3)*C.x + cos(q3)*C.z
    assert cross(A.x, C.z) == -cos(q3)*C.y
    assert cross(C.x, A.x) == -sin(q3)*C.y
    assert cross(C.y, A.x).express(C).simplify() == sin(q3)*C.x - cos(q3)*C.z
    assert cross(C.z, A.x) == cos(q3)*C.y

def test_operator_match():
    """Test that the output of dot, cross, outer functions match
    operator behavior.
    """
    A = ReferenceFrame('A')
    v = A.x + A.y
    d = v | v
    zerov = Vector(0)
    zerod = Dyadic(0)

    # dot products
    assert d & d == dot(d, d)
    assert d & zerod == dot(d, zerod)
    assert zerod & d == dot(zerod, d)
    assert d & v == dot(d, v)
    assert v & d == dot(v, d)
    assert d & zerov == dot(d, zerov)
    assert zerov & d == dot(zerov, d)
    raises(TypeError, lambda: dot(d, S.Zero))
    raises(TypeError, lambda: dot(S.Zero, d))
    raises(TypeError, lambda: dot(d, 0))
    raises(TypeError, lambda: dot(0, d))
    assert v & v == dot(v, v)
    assert v & zerov == dot(v, zerov)
    assert zerov & v == dot(zerov, v)
    raises(TypeError, lambda: dot(v, S.Zero))
    raises(TypeError, lambda: dot(S.Zero, v))
    raises(TypeError, lambda: dot(v, 0))
    raises(TypeError, lambda: dot(0, v))

    # cross products
    raises(TypeError, lambda: cross(d, d))
    raises(TypeError, lambda: cross(d, zerod))
    raises(TypeError, lambda: cross(zerod, d))
    assert d ^ v == cross(d, v)
    assert v ^ d == cross(v, d)
    assert d ^ zerov == cross(d, zerov)
    assert zerov ^ d == cross(zerov, d)
    assert zerov ^ d == cross(zerov, d)
    raises(TypeError, lambda: cross(d, S.Zero))
    raises(TypeError, lambda: cross(S.Zero, d))
    raises(TypeError, lambda: cross(d, 0))
    raises(TypeError, lambda: cross(0, d))
    assert v ^ v == cross(v, v)
    assert v ^ zerov == cross(v, zerov)
    assert zerov ^ v == cross(zerov, v)
    raises(TypeError, lambda: cross(v, S.Zero))
    raises(TypeError, lambda: cross(S.Zero, v))
    raises(TypeError, lambda: cross(v, 0))
    raises(TypeError, lambda: cross(0, v))

    # outer products
    raises(TypeError, lambda: outer(d, d))
    raises(TypeError, lambda: outer(d, zerod))
    raises(TypeError, lambda: outer(zerod, d))
    raises(TypeError, lambda: outer(d, v))
    raises(TypeError, lambda: outer(v, d))
    raises(TypeError, lambda: outer(d, zerov))
    raises(TypeError, lambda: outer(zerov, d))
    raises(TypeError, lambda: outer(zerov, d))
    raises(TypeError, lambda: outer(d, S.Zero))
    raises(TypeError, lambda: outer(S.Zero, d))
    raises(TypeError, lambda: outer(d, 0))
    raises(TypeError, lambda: outer(0, d))
    assert v | v == outer(v, v)
    assert v | zerov == outer(v, zerov)
    assert zerov | v == outer(zerov, v)
    raises(TypeError, lambda: outer(v, S.Zero))
    raises(TypeError, lambda: outer(S.Zero, v))
    raises(TypeError, lambda: outer(v, 0))
    raises(TypeError, lambda: outer(0, v))


def test_express():
    assert express(Vector(0), N) == Vector(0)
    assert express(S.Zero, N) is S.Zero
    assert express(A.x, C) == cos(q3)*C.x + sin(q3)*C.z
    assert express(A.y, C) == sin(q2)*sin(q3)*C.x + cos(q2)*C.y - \
        sin(q2)*cos(q3)*C.z
    assert express(A.z, C) == -sin(q3)*cos(q2)*C.x + sin(q2)*C.y + \
        cos(q2)*cos(q3)*C.z
    assert express(A.x, N) == cos(q1)*N.x + sin(q1)*N.y
    assert express(A.y, N) == -sin(q1)*N.x + cos(q1)*N.y
    assert express(A.z, N) == N.z
    assert express(A.x, A) == A.x
    assert express(A.y, A) == A.y
    assert express(A.z, A) == A.z
    assert express(A.x, B) == B.x
    assert express(A.y, B) == cos(q2)*B.y - sin(q2)*B.z
    assert express(A.z, B) == sin(q2)*B.y + cos(q2)*B.z
    assert express(A.x, C) == cos(q3)*C.x + sin(q3)*C.z
    assert express(A.y, C) == sin(q2)*sin(q3)*C.x + cos(q2)*C.y - \
        sin(q2)*cos(q3)*C.z
    assert express(A.z, C) == -sin(q3)*cos(q2)*C.x + sin(q2)*C.y + \
        cos(q2)*cos(q3)*C.z
    # Check to make sure UnitVectors get converted properly
    assert express(N.x, N) == N.x
    assert express(N.y, N) == N.y
    assert express(N.z, N) == N.z
    assert express(N.x, A) == (cos(q1)*A.x - sin(q1)*A.y)
    assert express(N.y, A) == (sin(q1)*A.x + cos(q1)*A.y)
    assert express(N.z, A) == A.z
    assert express(N.x, B) == (cos(q1)*B.x - sin(q1)*cos(q2)*B.y +
            sin(q1)*sin(q2)*B.z)
    assert express(N.y, B) == (sin(q1)*B.x + cos(q1)*cos(q2)*B.y -
            sin(q2)*cos(q1)*B.z)
    assert express(N.z, B) == (sin(q2)*B.y + cos(q2)*B.z)
    assert express(N.x, C) == (
        (cos(q1)*cos(q3) - sin(q1)*sin(q2)*sin(q3))*C.x -
        sin(q1)*cos(q2)*C.y +
        (sin(q3)*cos(q1) + sin(q1)*sin(q2)*cos(q3))*C.z)
    assert express(N.y, C) == (
        (sin(q1)*cos(q3) + sin(q2)*sin(q3)*cos(q1))*C.x +
        cos(q1)*cos(q2)*C.y +
        (sin(q1)*sin(q3) - sin(q2)*cos(q1)*cos(q3))*C.z)
    assert express(N.z, C) == (-sin(q3)*cos(q2)*C.x + sin(q2)*C.y +
            cos(q2)*cos(q3)*C.z)

    assert express(A.x, N) == (cos(q1)*N.x + sin(q1)*N.y)
    assert express(A.y, N) == (-sin(q1)*N.x + cos(q1)*N.y)
    assert express(A.z, N) == N.z
    assert express(A.x, A) == A.x
    assert express(A.y, A) == A.y
    assert express(A.z, A) == A.z
    assert express(A.x, B) == B.x
    assert express(A.y, B) == (cos(q2)*B.y - sin(q2)*B.z)
    assert express(A.z, B) == (sin(q2)*B.y + cos(q2)*B.z)
    assert express(A.x, C) == (cos(q3)*C.x + sin(q3)*C.z)
    assert express(A.y, C) == (sin(q2)*sin(q3)*C.x + cos(q2)*C.y -
            sin(q2)*cos(q3)*C.z)
    assert express(A.z, C) == (-sin(q3)*cos(q2)*C.x + sin(q2)*C.y +
            cos(q2)*cos(q3)*C.z)

    assert express(B.x, N) == (cos(q1)*N.x + sin(q1)*N.y)
    assert express(B.y, N) == (-sin(q1)*cos(q2)*N.x +
            cos(q1)*cos(q2)*N.y + sin(q2)*N.z)
    assert express(B.z, N) == (sin(q1)*sin(q2)*N.x -
            sin(q2)*cos(q1)*N.y + cos(q2)*N.z)
    assert express(B.x, A) == A.x
    assert express(B.y, A) == (cos(q2)*A.y + sin(q2)*A.z)
    assert express(B.z, A) == (-sin(q2)*A.y + cos(q2)*A.z)
    assert express(B.x, B) == B.x
    assert express(B.y, B) == B.y
    assert express(B.z, B) == B.z
    assert express(B.x, C) == (cos(q3)*C.x + sin(q3)*C.z)
    assert express(B.y, C) == C.y
    assert express(B.z, C) == (-sin(q3)*C.x + cos(q3)*C.z)

    assert express(C.x, N) == (
        (cos(q1)*cos(q3) - sin(q1)*sin(q2)*sin(q3))*N.x +
        (sin(q1)*cos(q3) + sin(q2)*sin(q3)*cos(q1))*N.y -
        sin(q3)*cos(q2)*N.z)
    assert express(C.y, N) == (
        -sin(q1)*cos(q2)*N.x + cos(q1)*cos(q2)*N.y + sin(q2)*N.z)
    assert express(C.z, N) == (
        (sin(q3)*cos(q1) + sin(q1)*sin(q2)*cos(q3))*N.x +
        (sin(q1)*sin(q3) - sin(q2)*cos(q1)*cos(q3))*N.y +
        cos(q2)*cos(q3)*N.z)
    assert express(C.x, A) == (cos(q3)*A.x + sin(q2)*sin(q3)*A.y -
            sin(q3)*cos(q2)*A.z)
    assert express(C.y, A) == (cos(q2)*A.y + sin(q2)*A.z)
    assert express(C.z, A) == (sin(q3)*A.x - sin(q2)*cos(q3)*A.y +
            cos(q2)*cos(q3)*A.z)
    assert express(C.x, B) == (cos(q3)*B.x - sin(q3)*B.z)
    assert express(C.y, B) == B.y
    assert express(C.z, B) == (sin(q3)*B.x + cos(q3)*B.z)
    assert express(C.x, C) == C.x
    assert express(C.y, C) == C.y
    assert express(C.z, C) == C.z == (C.z)

    #  Check to make sure Vectors get converted back to UnitVectors
    assert N.x == express((cos(q1)*A.x - sin(q1)*A.y), N).simplify()
    assert N.y == express((sin(q1)*A.x + cos(q1)*A.y), N).simplify()
    assert N.x == express((cos(q1)*B.x - sin(q1)*cos(q2)*B.y +
            sin(q1)*sin(q2)*B.z), N).simplify()
    assert N.y == express((sin(q1)*B.x + cos(q1)*cos(q2)*B.y -
        sin(q2)*cos(q1)*B.z), N).simplify()
    assert N.z == express((sin(q2)*B.y + cos(q2)*B.z), N).simplify()

    """
    These don't really test our code, they instead test the auto simplification
    (or lack thereof) of SymPy.
    assert N.x == express((
            (cos(q1)*cos(q3)-sin(q1)*sin(q2)*sin(q3))*C.x -
            sin(q1)*cos(q2)*C.y +
            (sin(q3)*cos(q1)+sin(q1)*sin(q2)*cos(q3))*C.z), N)
    assert N.y == express((
            (sin(q1)*cos(q3) + sin(q2)*sin(q3)*cos(q1))*C.x +
            cos(q1)*cos(q2)*C.y +
            (sin(q1)*sin(q3) - sin(q2)*cos(q1)*cos(q3))*C.z), N)
    assert N.z == express((-sin(q3)*cos(q2)*C.x + sin(q2)*C.y +
            cos(q2)*cos(q3)*C.z), N)
    """

    assert A.x == express((cos(q1)*N.x + sin(q1)*N.y), A).simplify()
    assert A.y == express((-sin(q1)*N.x + cos(q1)*N.y), A).simplify()

    assert A.y == express((cos(q2)*B.y - sin(q2)*B.z), A).simplify()
    assert A.z == express((sin(q2)*B.y + cos(q2)*B.z), A).simplify()

    assert A.x == express((cos(q3)*C.x + sin(q3)*C.z), A).simplify()

    # Tripsimp messes up here too.
    #print express((sin(q2)*sin(q3)*C.x + cos(q2)*C.y -
    #        sin(q2)*cos(q3)*C.z), A)
    assert A.y == express((sin(q2)*sin(q3)*C.x + cos(q2)*C.y -
            sin(q2)*cos(q3)*C.z), A).simplify()

    assert A.z == express((-sin(q3)*cos(q2)*C.x + sin(q2)*C.y +
            cos(q2)*cos(q3)*C.z), A).simplify()
    assert B.x == express((cos(q1)*N.x + sin(q1)*N.y), B).simplify()
    assert B.y == express((-sin(q1)*cos(q2)*N.x +
            cos(q1)*cos(q2)*N.y + sin(q2)*N.z), B).simplify()

    assert B.z == express((sin(q1)*sin(q2)*N.x -
            sin(q2)*cos(q1)*N.y + cos(q2)*N.z), B).simplify()

    assert B.y == express((cos(q2)*A.y + sin(q2)*A.z), B).simplify()
    assert B.z == express((-sin(q2)*A.y + cos(q2)*A.z), B).simplify()
    assert B.x == express((cos(q3)*C.x + sin(q3)*C.z), B).simplify()
    assert B.z == express((-sin(q3)*C.x + cos(q3)*C.z), B).simplify()

    """
    assert C.x == express((
            (cos(q1)*cos(q3)-sin(q1)*sin(q2)*sin(q3))*N.x +
            (sin(q1)*cos(q3)+sin(q2)*sin(q3)*cos(q1))*N.y -
                sin(q3)*cos(q2)*N.z), C)
    assert C.y == express((
            -sin(q1)*cos(q2)*N.x + cos(q1)*cos(q2)*N.y + sin(q2)*N.z), C)
    assert C.z == express((
            (sin(q3)*cos(q1)+sin(q1)*sin(q2)*cos(q3))*N.x +
            (sin(q1)*sin(q3)-sin(q2)*cos(q1)*cos(q3))*N.y +
            cos(q2)*cos(q3)*N.z), C)
    """
    assert C.x == express((cos(q3)*A.x + sin(q2)*sin(q3)*A.y -
            sin(q3)*cos(q2)*A.z), C).simplify()
    assert C.y == express((cos(q2)*A.y + sin(q2)*A.z), C).simplify()
    assert C.z == express((sin(q3)*A.x - sin(q2)*cos(q3)*A.y +
            cos(q2)*cos(q3)*A.z), C).simplify()
    assert C.x == express((cos(q3)*B.x - sin(q3)*B.z), C).simplify()
    assert C.z == express((sin(q3)*B.x + cos(q3)*B.z), C).simplify()


def test_time_derivative():
    #The use of time_derivative for calculations pertaining to scalar
    #fields has been tested in test_coordinate_vars in test_essential.py
    A = ReferenceFrame('A')
    q = dynamicsymbols('q')
    qd = dynamicsymbols('q', 1)
    B = A.orientnew('B', 'Axis', [q, A.z])
    d = A.x | A.x
    assert time_derivative(d, B) == (-qd) * (A.y | A.x) + \
           (-qd) * (A.x | A.y)
    d1 = A.x | B.y
    assert time_derivative(d1, A) == - qd*(A.x|B.x)
    assert time_derivative(d1, B) == - qd*(A.y|B.y)
    d2 = A.x | B.x
    assert time_derivative(d2, A) == qd*(A.x|B.y)
    assert time_derivative(d2, B) == - qd*(A.y|B.x)
    d3 = A.x | B.z
    assert time_derivative(d3, A) == 0
    assert time_derivative(d3, B) == - qd*(A.y|B.z)
    q1, q2, q3, q4 = dynamicsymbols('q1 q2 q3 q4')
    q1d, q2d, q3d, q4d = dynamicsymbols('q1 q2 q3 q4', 1)
    q1dd, q2dd, q3dd, q4dd = dynamicsymbols('q1 q2 q3 q4', 2)
    C = B.orientnew('C', 'Axis', [q4, B.x])
    v1 = q1 * A.z
    v2 = q2*A.x + q3*B.y
    v3 = q1*A.x + q2*A.y + q3*A.z
    assert time_derivative(B.x, C) == 0
    assert time_derivative(B.y, C) == - q4d*B.z
    assert time_derivative(B.z, C) == q4d*B.y
    assert time_derivative(v1, B) == q1d*A.z
    assert time_derivative(v1, C) == - q1*sin(q)*q4d*A.x + \
           q1*cos(q)*q4d*A.y + q1d*A.z
    assert time_derivative(v2, A) == q2d*A.x - q3*qd*B.x + q3d*B.y
    assert time_derivative(v2, C) == q2d*A.x - q2*qd*A.y + \
           q2*sin(q)*q4d*A.z + q3d*B.y - q3*q4d*B.z
    assert time_derivative(v3, B) == (q2*qd + q1d)*A.x + \
           (-q1*qd + q2d)*A.y + q3d*A.z
    assert time_derivative(d, C) == - qd*(A.y|A.x) + \
           sin(q)*q4d*(A.z|A.x) - qd*(A.x|A.y) + sin(q)*q4d*(A.x|A.z)
    raises(ValueError, lambda: time_derivative(B.x, C, order=0.5))
    raises(ValueError, lambda: time_derivative(B.x, C, order=-1))


def test_get_motion_methods():
    #Initialization
    t = dynamicsymbols._t
    s1, s2, s3 = symbols('s1 s2 s3')
    S1, S2, S3 = symbols('S1 S2 S3')
    S4, S5, S6 = symbols('S4 S5 S6')
    t1, t2 = symbols('t1 t2')
    a, b, c = dynamicsymbols('a b c')
    ad, bd, cd = dynamicsymbols('a b c', 1)
    a2d, b2d, c2d = dynamicsymbols('a b c', 2)
    v0 = S1*N.x + S2*N.y + S3*N.z
    v01 = S4*N.x + S5*N.y + S6*N.z
    v1 = s1*N.x + s2*N.y + s3*N.z
    v2 = a*N.x + b*N.y + c*N.z
    v2d = ad*N.x + bd*N.y + cd*N.z
    v2dd = a2d*N.x + b2d*N.y + c2d*N.z
    #Test position parameter
    assert get_motion_params(frame = N) == (0, 0, 0)
    assert get_motion_params(N, position=v1) == (0, 0, v1)
    assert get_motion_params(N, position=v2) == (v2dd, v2d, v2)
    #Test velocity parameter
    assert get_motion_params(N, velocity=v1) == (0, v1, v1 * t)
    assert get_motion_params(N, velocity=v1, position=v0, timevalue1=t1) == \
           (0, v1, v0 + v1*(t - t1))
    answer = get_motion_params(N, velocity=v1, position=v2, timevalue1=t1)
    answer_expected = (0, v1, v1*t - v1*t1 + v2.subs(t, t1))
    assert answer == answer_expected

    answer = get_motion_params(N, velocity=v2, position=v0, timevalue1=t1)
    integral_vector = Integral(a, (t, t1, t))*N.x + Integral(b, (t, t1, t))*N.y \
            + Integral(c, (t, t1, t))*N.z
    answer_expected = (v2d, v2, v0 + integral_vector)
    assert answer == answer_expected

    #Test acceleration parameter
    assert get_motion_params(N, acceleration=v1) == \
           (v1, v1 * t, v1 * t**2/2)
    assert get_motion_params(N, acceleration=v1, velocity=v0,
                          position=v2, timevalue1=t1, timevalue2=t2) == \
           (v1, (v0 + v1*t - v1*t2),
            -v0*t1 + v1*t**2/2 + v1*t2*t1 - \
            v1*t1**2/2 + t*(v0 - v1*t2) + \
            v2.subs(t, t1))
    assert get_motion_params(N, acceleration=v1, velocity=v0,
                             position=v01, timevalue1=t1, timevalue2=t2) == \
           (v1, v0 + v1*t - v1*t2,
            -v0*t1 + v01 + v1*t**2/2 + \
            v1*t2*t1 - v1*t1**2/2 + \
            t*(v0 - v1*t2))
    answer = get_motion_params(N, acceleration=a*N.x, velocity=S1*N.x,
                          position=S2*N.x, timevalue1=t1, timevalue2=t2)
    i1 = Integral(a, (t, t2, t))
    answer_expected = (a*N.x, (S1 + i1)*N.x, \
        (S2 + Integral(S1 + i1, (t, t1, t)))*N.x)
    assert answer == answer_expected


def test_kin_eqs():
    q0, q1, q2, q3 = dynamicsymbols('q0 q1 q2 q3')
    q0d, q1d, q2d, q3d = dynamicsymbols('q0 q1 q2 q3', 1)
    u1, u2, u3 = dynamicsymbols('u1 u2 u3')
    ke = kinematic_equations([u1,u2,u3], [q1,q2,q3], 'body', 313)
    assert ke == kinematic_equations([u1,u2,u3], [q1,q2,q3], 'body', '313')
    kds = kinematic_equations([u1, u2, u3], [q0, q1, q2, q3], 'quaternion')
    assert kds == [-0.5 * q0 * u1 - 0.5 * q2 * u3 + 0.5 * q3 * u2 + q1d,
            -0.5 * q0 * u2 + 0.5 * q1 * u3 - 0.5 * q3 * u1 + q2d,
            -0.5 * q0 * u3 - 0.5 * q1 * u2 + 0.5 * q2 * u1 + q3d,
            0.5 * q1 * u1 + 0.5 * q2 * u2 + 0.5 * q3 * u3 + q0d]
    raises(ValueError, lambda: kinematic_equations([u1, u2, u3], [q0, q1, q2], 'quaternion'))
    raises(ValueError, lambda: kinematic_equations([u1, u2, u3], [q0, q1, q2, q3], 'quaternion', '123'))
    raises(ValueError, lambda: kinematic_equations([u1, u2, u3], [q0, q1, q2, q3], 'foo'))
    raises(TypeError, lambda: kinematic_equations(u1, [q0, q1, q2, q3], 'quaternion'))
    raises(TypeError, lambda: kinematic_equations([u1], [q0, q1, q2, q3], 'quaternion'))
    raises(TypeError, lambda: kinematic_equations([u1, u2, u3], q0, 'quaternion'))
    raises(ValueError, lambda: kinematic_equations([u1, u2, u3], [q0, q1, q2, q3], 'body'))
    raises(ValueError, lambda: kinematic_equations([u1, u2, u3], [q0, q1, q2, q3], 'space'))
    raises(ValueError, lambda: kinematic_equations([u1, u2, u3], [q0, q1, q2], 'body', '222'))
    assert kinematic_equations([0, 0, 0], [q0, q1, q2], 'space') == [S.Zero, S.Zero, S.Zero]


def test_partial_velocity():
    q1, q2, q3, u1, u2, u3 = dynamicsymbols('q1 q2 q3 u1 u2 u3')
    u4, u5 = dynamicsymbols('u4, u5')
    r = symbols('r')

    N = ReferenceFrame('N')
    Y = N.orientnew('Y', 'Axis', [q1, N.z])
    L = Y.orientnew('L', 'Axis', [q2, Y.x])
    R = L.orientnew('R', 'Axis', [q3, L.y])
    R.set_ang_vel(N, u1 * L.x + u2 * L.y + u3 * L.z)

    C = Point('C')
    C.set_vel(N, u4 * L.x + u5 * (Y.z ^ L.x))
    Dmc = C.locatenew('Dmc', r * L.z)
    Dmc.v2pt_theory(C, N, R)

    vel_list = [Dmc.vel(N), C.vel(N), R.ang_vel_in(N)]
    u_list = [u1, u2, u3, u4, u5]
    assert (partial_velocity(vel_list, u_list, N) ==
            [[- r*L.y, r*L.x, 0, L.x, cos(q2)*L.y - sin(q2)*L.z],
            [0, 0, 0, L.x, cos(q2)*L.y - sin(q2)*L.z],
            [L.x, L.y, L.z, 0, 0]])

    # Make sure that partial velocities can be computed regardless if the
    # orientation between frames is defined or not.
    A = ReferenceFrame('A')
    B = ReferenceFrame('B')
    v = u4 * A.x + u5 * B.y
    assert partial_velocity((v, ), (u4, u5), A) == [[A.x, B.y]]

    raises(TypeError, lambda: partial_velocity(Dmc.vel(N), u_list, N))
    raises(TypeError, lambda: partial_velocity(vel_list, u1, N))

def test_dynamicsymbols():
    #Tests to check the assumptions applied to dynamicsymbols
    f1 = dynamicsymbols('f1')
    f2 = dynamicsymbols('f2', real=True)
    f3 = dynamicsymbols('f3', positive=True)
    f4, f5 = dynamicsymbols('f4,f5', commutative=False)
    f6 = dynamicsymbols('f6', integer=True)
    assert f1.is_real is None
    assert f2.is_real
    assert f3.is_positive
    assert f4*f5 != f5*f4
    assert f6.is_integer
