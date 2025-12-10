from sympy.testing.pytest import slow
from sympy.core.function import diff
from sympy.core.function import expand
from sympy.core.numbers import (E, I, Rational, pi)
from sympy.core.singleton import S
from sympy.core.symbol import (Symbol, symbols)
from sympy.functions.elementary.complexes import (Abs, conjugate, im, re, sign)
from sympy.functions.elementary.exponential import log
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (acos, asin, cos, sin, atan2, atan)
from sympy.integrals.integrals import integrate
from sympy.matrices.dense import Matrix
from sympy.simplify import simplify
from sympy.simplify.trigsimp import trigsimp
from sympy.algebras.quaternion import Quaternion
from sympy.testing.pytest import raises
import math
from itertools import permutations, product

w, x, y, z = symbols('w:z')
phi = symbols('phi')

def test_quaternion_construction():
    q = Quaternion(w, x, y, z)
    assert q + q == Quaternion(2*w, 2*x, 2*y, 2*z)

    q2 = Quaternion.from_axis_angle((sqrt(3)/3, sqrt(3)/3, sqrt(3)/3),
                                    pi*Rational(2, 3))
    assert q2 == Quaternion(S.Half, S.Half,
                            S.Half, S.Half)

    M = Matrix([[cos(phi), -sin(phi), 0], [sin(phi), cos(phi), 0], [0, 0, 1]])
    q3 = trigsimp(Quaternion.from_rotation_matrix(M))
    assert q3 == Quaternion(
        sqrt(2)*sqrt(cos(phi) + 1)/2, 0, 0, sqrt(2 - 2*cos(phi))*sign(sin(phi))/2)

    nc = Symbol('nc', commutative=False)
    raises(ValueError, lambda: Quaternion(w, x, nc, z))


def test_quaternion_construction_norm():
    q1 = Quaternion(*symbols('a:d'))

    q2 = Quaternion(w, x, y, z)
    assert expand((q1*q2).norm()**2 - (q1.norm()**2 * q2.norm()**2)) == 0

    q3 = Quaternion(w, x, y, z, norm=1)
    assert (q1 * q3).norm() == q1.norm()


def test_issue_25254():
    # calculating the inverse cached the norm which caused problems
    # when multiplying
    p = Quaternion(1, 0, 0, 0)
    q = Quaternion.from_axis_angle((1, 1, 1), 3 * math.pi/4)
    qi = q.inverse()  # this operation cached the norm
    test = q * p * qi
    assert ((test - p).norm() < 1E-10)


def test_to_and_from_Matrix():
    q = Quaternion(w, x, y, z)
    q_full = Quaternion.from_Matrix(q.to_Matrix())
    q_vect = Quaternion.from_Matrix(q.to_Matrix(True))
    assert (q - q_full).is_zero_quaternion()
    assert (q.vector_part() - q_vect).is_zero_quaternion()


def test_product_matrices():
    q1 = Quaternion(w, x, y, z)
    q2 = Quaternion(*(symbols("a:d")))
    assert (q1 * q2).to_Matrix() == q1.product_matrix_left * q2.to_Matrix()
    assert (q1 * q2).to_Matrix() == q2.product_matrix_right * q1.to_Matrix()

    R1 = (q1.product_matrix_left * q1.product_matrix_right.T)[1:, 1:]
    R2 = simplify(q1.to_rotation_matrix()*q1.norm()**2)
    assert R1 == R2


def test_quaternion_axis_angle():

    test_data = [ # axis, angle, expected_quaternion
        ((1, 0, 0), 0, (1, 0, 0, 0)),
        ((1, 0, 0), pi/2, (sqrt(2)/2, sqrt(2)/2, 0, 0)),
        ((0, 1, 0), pi/2, (sqrt(2)/2, 0, sqrt(2)/2, 0)),
        ((0, 0, 1), pi/2, (sqrt(2)/2, 0, 0, sqrt(2)/2)),
        ((1, 0, 0), pi, (0, 1, 0, 0)),
        ((0, 1, 0), pi, (0, 0, 1, 0)),
        ((0, 0, 1), pi, (0, 0, 0, 1)),
        ((1, 1, 1), pi, (0, 1/sqrt(3),1/sqrt(3),1/sqrt(3))),
        ((sqrt(3)/3, sqrt(3)/3, sqrt(3)/3), pi*2/3, (S.Half, S.Half, S.Half, S.Half))
    ]

    for axis, angle, expected in test_data:
        assert Quaternion.from_axis_angle(axis, angle) == Quaternion(*expected)


def test_quaternion_axis_angle_simplification():
    result = Quaternion.from_axis_angle((1, 2, 3), asin(4))
    assert result.a == cos(asin(4)/2)
    assert result.b == sqrt(14)*sin(asin(4)/2)/14
    assert result.c == sqrt(14)*sin(asin(4)/2)/7
    assert result.d == 3*sqrt(14)*sin(asin(4)/2)/14

def test_quaternion_complex_real_addition():
    a = symbols("a", complex=True)
    b = symbols("b", real=True)
    # This symbol is not complex:
    c = symbols("c", commutative=False)

    q = Quaternion(w, x, y, z)
    assert a + q == Quaternion(w + re(a), x + im(a), y, z)
    assert 1 + q == Quaternion(1 + w, x, y, z)
    assert I + q == Quaternion(w, 1 + x, y, z)
    assert b + q == Quaternion(w + b, x, y, z)
    raises(ValueError, lambda: c + q)
    raises(ValueError, lambda: q * c)
    raises(ValueError, lambda: c * q)

    assert -q == Quaternion(-w, -x, -y, -z)

    q1 = Quaternion(3 + 4*I, 2 + 5*I, 0, 7 + 8*I, real_field = False)
    q2 = Quaternion(1, 4, 7, 8)

    assert q1 + (2 + 3*I) == Quaternion(5 + 7*I, 2 + 5*I, 0, 7 + 8*I)
    assert q2 + (2 + 3*I) == Quaternion(3, 7, 7, 8)
    assert q1 * (2 + 3*I) == \
    Quaternion((2 + 3*I)*(3 + 4*I), (2 + 3*I)*(2 + 5*I), 0, (2 + 3*I)*(7 + 8*I))
    assert q2 * (2 + 3*I) == Quaternion(-10, 11, 38, -5)

    q1 = Quaternion(1, 2, 3, 4)
    q0 = Quaternion(0, 0, 0, 0)
    assert q1 + q0 == q1
    assert q1 - q0 == q1
    assert q1 - q1 == q0


def test_quaternion_subs():
    q = Quaternion.from_axis_angle((0, 0, 1), phi)
    assert q.subs(phi, 0) == Quaternion(1, 0, 0, 0)


def test_quaternion_evalf():
    assert (Quaternion(sqrt(2), 0, 0, sqrt(3)).evalf() ==
            Quaternion(sqrt(2).evalf(), 0, 0, sqrt(3).evalf()))
    assert (Quaternion(1/sqrt(2), 0, 0, 1/sqrt(2)).evalf() ==
            Quaternion((1/sqrt(2)).evalf(), 0, 0, (1/sqrt(2)).evalf()))


def test_quaternion_functions():
    q = Quaternion(w, x, y, z)
    q1 = Quaternion(1, 2, 3, 4)
    q0 = Quaternion(0, 0, 0, 0)

    assert conjugate(q) == Quaternion(w, -x, -y, -z)
    assert q.norm() == sqrt(w**2 + x**2 + y**2 + z**2)
    assert q.normalize() == Quaternion(w, x, y, z) / sqrt(w**2 + x**2 + y**2 + z**2)
    assert q.inverse() == Quaternion(w, -x, -y, -z) / (w**2 + x**2 + y**2 + z**2)
    assert q.inverse() == q.pow(-1)
    raises(ValueError, lambda: q0.inverse())
    assert q.pow(2) == Quaternion(w**2 - x**2 - y**2 - z**2, 2*w*x, 2*w*y, 2*w*z)
    assert q**(2) == Quaternion(w**2 - x**2 - y**2 - z**2, 2*w*x, 2*w*y, 2*w*z)
    assert q1.pow(-2) == Quaternion(
        Rational(-7, 225), Rational(-1, 225), Rational(-1, 150), Rational(-2, 225))
    assert q1**(-2) == Quaternion(
        Rational(-7, 225), Rational(-1, 225), Rational(-1, 150), Rational(-2, 225))
    assert q1.pow(-0.5) == NotImplemented
    raises(TypeError, lambda: q1**(-0.5))

    assert q1.exp() == \
    Quaternion(E * cos(sqrt(29)),
               2 * sqrt(29) * E * sin(sqrt(29)) / 29,
               3 * sqrt(29) * E * sin(sqrt(29)) / 29,
               4 * sqrt(29) * E * sin(sqrt(29)) / 29)
    assert q1.log() == \
    Quaternion(log(sqrt(30)),
               2 * sqrt(29) * acos(sqrt(30)/30) / 29,
               3 * sqrt(29) * acos(sqrt(30)/30) / 29,
               4 * sqrt(29) * acos(sqrt(30)/30) / 29)

    assert q1.pow_cos_sin(2) == \
    Quaternion(30 * cos(2 * acos(sqrt(30)/30)),
               60 * sqrt(29) * sin(2 * acos(sqrt(30)/30)) / 29,
               90 * sqrt(29) * sin(2 * acos(sqrt(30)/30)) / 29,
               120 * sqrt(29) * sin(2 * acos(sqrt(30)/30)) / 29)

    assert diff(Quaternion(x, x, x, x), x) == Quaternion(1, 1, 1, 1)

    assert integrate(Quaternion(x, x, x, x), x) == \
    Quaternion(x**2 / 2, x**2 / 2, x**2 / 2, x**2 / 2)

    assert Quaternion(1, x, x**2, x**3).integrate(x) == \
    Quaternion(x, x**2/2, x**3/3, x**4/4)

    assert Quaternion(sin(x), cos(x), sin(2*x), cos(2*x)).integrate(x) == \
    Quaternion(-cos(x), sin(x), -cos(2*x)/2, sin(2*x)/2)

    assert Quaternion(x**2, y**2, z**2, x*y*z).integrate(x, y) == \
    Quaternion(x**3*y/3, x*y**3/3, x*y*z**2, x**2*y**2*z/4)

    assert Quaternion.rotate_point((1, 1, 1), q1) == (S.One / 5, 1, S(7) / 5)
    n = Symbol('n')
    raises(TypeError, lambda: q1**n)
    n = Symbol('n', integer=True)
    raises(TypeError, lambda: q1**n)

    assert Quaternion(22, 23, 55, 8).scalar_part() == 22
    assert Quaternion(w, x, y, z).scalar_part() == w

    assert Quaternion(22, 23, 55, 8).vector_part() == Quaternion(0, 23, 55, 8)
    assert Quaternion(w, x, y, z).vector_part() == Quaternion(0, x, y, z)

    assert q1.axis() == Quaternion(0, 2*sqrt(29)/29, 3*sqrt(29)/29, 4*sqrt(29)/29)
    assert q1.axis().pow(2) == Quaternion(-1, 0, 0, 0)
    assert q0.axis().scalar_part() == 0
    assert (q.axis() == Quaternion(0,
                                   x/sqrt(x**2 + y**2 + z**2),
                                   y/sqrt(x**2 + y**2 + z**2),
                                   z/sqrt(x**2 + y**2 + z**2)))

    assert q0.is_pure() is True
    assert q1.is_pure() is False
    assert Quaternion(0, 0, 0, 3).is_pure() is True
    assert Quaternion(0, 2, 10, 3).is_pure() is True
    assert Quaternion(w, 2, 10, 3).is_pure() is None

    assert q1.angle() == 2*atan(sqrt(29))
    assert q.angle() == 2*atan2(sqrt(x**2 + y**2 + z**2), w)

    assert Quaternion.arc_coplanar(q1, Quaternion(2, 4, 6, 8)) is True
    assert Quaternion.arc_coplanar(q1, Quaternion(1, -2, -3, -4)) is True
    assert Quaternion.arc_coplanar(q1, Quaternion(1, 8, 12, 16)) is True
    assert Quaternion.arc_coplanar(q1, Quaternion(1, 2, 3, 4)) is True
    assert Quaternion.arc_coplanar(q1, Quaternion(w, 4, 6, 8)) is True
    assert Quaternion.arc_coplanar(q1, Quaternion(2, 7, 4, 1)) is False
    assert Quaternion.arc_coplanar(q1, Quaternion(w, x, y, z)) is None
    raises(ValueError, lambda: Quaternion.arc_coplanar(q1, q0))

    assert Quaternion.vector_coplanar(
        Quaternion(0, 8, 12, 16),
        Quaternion(0, 4, 6, 8),
        Quaternion(0, 2, 3, 4)) is True
    assert Quaternion.vector_coplanar(
        Quaternion(0, 0, 0, 0), Quaternion(0, 4, 6, 8), Quaternion(0, 2, 3, 4)) is True
    assert Quaternion.vector_coplanar(
        Quaternion(0, 8, 2, 6), Quaternion(0, 1, 6, 6), Quaternion(0, 0, 3, 4)) is False
    assert Quaternion.vector_coplanar(
        Quaternion(0, 1, 3, 4),
        Quaternion(0, 4, w, 6),
        Quaternion(0, 6, 8, 1)) is None
    raises(ValueError, lambda:
        Quaternion.vector_coplanar(q0, Quaternion(0, 4, 6, 8), q1))

    assert Quaternion(0, 1, 2, 3).parallel(Quaternion(0, 2, 4, 6)) is True
    assert Quaternion(0, 1, 2, 3).parallel(Quaternion(0, 2, 2, 6)) is False
    assert Quaternion(0, 1, 2, 3).parallel(Quaternion(w, x, y, 6)) is None
    raises(ValueError, lambda: q0.parallel(q1))

    assert Quaternion(0, 1, 2, 3).orthogonal(Quaternion(0, -2, 1, 0)) is True
    assert Quaternion(0, 2, 4, 7).orthogonal(Quaternion(0, 2, 2, 6)) is False
    assert Quaternion(0, 2, 4, 7).orthogonal(Quaternion(w, x, y, 6)) is None
    raises(ValueError, lambda: q0.orthogonal(q1))

    assert q1.index_vector() == Quaternion(
        0, 2*sqrt(870)/29,
        3*sqrt(870)/29,
        4*sqrt(870)/29)
    assert Quaternion(0, 3, 9, 4).index_vector() == Quaternion(0, 3, 9, 4)

    assert Quaternion(4, 3, 9, 4).mensor() == log(sqrt(122))
    assert Quaternion(3, 3, 0, 2).mensor() == log(sqrt(22))

    assert q0.is_zero_quaternion() is True
    assert q1.is_zero_quaternion() is False
    assert Quaternion(w, 0, 0, 0).is_zero_quaternion() is None

def test_quaternion_conversions():
    q1 = Quaternion(1, 2, 3, 4)

    assert q1.to_axis_angle() == ((2 * sqrt(29)/29,
                                   3 * sqrt(29)/29,
                                   4 * sqrt(29)/29),
                                   2 * acos(sqrt(30)/30))

    assert (q1.to_rotation_matrix() ==
            Matrix([[Rational(-2, 3), Rational(2, 15), Rational(11, 15)],
                    [Rational(2, 3), Rational(-1, 3), Rational(2, 3)],
                    [Rational(1, 3), Rational(14, 15), Rational(2, 15)]]))

    assert (q1.to_rotation_matrix((1, 1, 1)) ==
            Matrix([
                [Rational(-2, 3), Rational(2, 15), Rational(11, 15), Rational(4, 5)],
                [Rational(2, 3), Rational(-1, 3), Rational(2, 3), S.Zero],
                [Rational(1, 3), Rational(14, 15), Rational(2, 15), Rational(-2, 5)],
                [S.Zero, S.Zero, S.Zero, S.One]]))

    theta = symbols("theta", real=True)
    q2 = Quaternion(cos(theta/2), 0, 0, sin(theta/2))

    assert trigsimp(q2.to_rotation_matrix()) == Matrix([
                                               [cos(theta), -sin(theta), 0],
                                               [sin(theta),  cos(theta), 0],
                                               [0,           0,          1]])

    assert q2.to_axis_angle() == ((0, 0, sin(theta/2)/Abs(sin(theta/2))),
                                   2*acos(cos(theta/2)))

    assert trigsimp(q2.to_rotation_matrix((1, 1, 1))) == Matrix([
               [cos(theta), -sin(theta), 0, sin(theta) - cos(theta) + 1],
               [sin(theta),  cos(theta), 0, -sin(theta) - cos(theta) + 1],
               [0,           0,          1,  0],
               [0,           0,          0,  1]])


def test_rotation_matrix_homogeneous():
    q = Quaternion(w, x, y, z)
    R1 = q.to_rotation_matrix(homogeneous=True) * q.norm()**2
    R2 = simplify(q.to_rotation_matrix(homogeneous=False) * q.norm()**2)
    assert R1 == R2


def test_quaternion_rotation_iss1593():
    """
    There was a sign mistake in the definition,
    of the rotation matrix. This tests that particular sign mistake.
    See issue 1593 for reference.
    See wikipedia
    https://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation#Quaternion-derived_rotation_matrix
    for the correct definition
    """
    q = Quaternion(cos(phi/2), sin(phi/2), 0, 0)
    assert(trigsimp(q.to_rotation_matrix()) == Matrix([
                [1,        0,         0],
                [0, cos(phi), -sin(phi)],
                [0, sin(phi),  cos(phi)]]))


def test_quaternion_multiplication():
    q1 = Quaternion(3 + 4*I, 2 + 5*I, 0, 7 + 8*I, real_field = False)
    q2 = Quaternion(1, 2, 3, 5)
    q3 = Quaternion(1, 1, 1, y)

    assert Quaternion._generic_mul(S(4), S.One) == 4
    assert (Quaternion._generic_mul(S(4), q1) ==
            Quaternion(12 + 16*I, 8 + 20*I, 0, 28 + 32*I))
    assert q2.mul(2) == Quaternion(2, 4, 6, 10)
    assert q2.mul(q3) == Quaternion(-5*y - 4, 3*y - 2, 9 - 2*y, y + 4)
    assert q2.mul(q3) == q2*q3

    z = symbols('z', complex=True)
    z_quat = Quaternion(re(z), im(z), 0, 0)
    q = Quaternion(*symbols('q:4', real=True))

    assert z * q == z_quat * q
    assert q * z == q * z_quat


def test_issue_16318():
    #for rtruediv
    q0 = Quaternion(0, 0, 0, 0)
    raises(ValueError, lambda: 1/q0)
    #for rotate_point
    q = Quaternion(1, 2, 3, 4)
    (axis, angle) = q.to_axis_angle()
    assert Quaternion.rotate_point((1, 1, 1), (axis, angle)) == (S.One / 5, 1, S(7) / 5)
    #test for to_axis_angle
    q = Quaternion(-1, 1, 1, 1)
    axis = (-sqrt(3)/3, -sqrt(3)/3, -sqrt(3)/3)
    angle = 2*pi/3
    assert (axis, angle) == q.to_axis_angle()


@slow
def test_to_euler():
    q = Quaternion(w, x, y, z)
    q_normalized = q.normalize()

    seqs = ['zxy', 'zyx', 'zyz', 'zxz']
    seqs += [seq.upper() for seq in seqs]

    for seq in seqs:
        euler_from_q = q.to_euler(seq)
        q_back = simplify(Quaternion.from_euler(euler_from_q, seq))
        assert q_back == q_normalized


def test_to_euler_iss24504():
    """
    There was a mistake in the degenerate case testing
    See issue 24504 for reference.
    """
    q = Quaternion.from_euler((phi, 0, 0), 'zyz')
    assert trigsimp(q.to_euler('zyz'), inverse=True) == (phi, 0, 0)


def test_to_euler_numerical_singilarities():

    def test_one_case(angles, seq):
        q = Quaternion.from_euler(angles, seq)
        assert q.to_euler(seq) == angles

    # symmetric
    test_one_case((pi/2,  0, 0), 'zyz')
    test_one_case((pi/2,  0, 0), 'ZYZ')
    test_one_case((pi/2,  pi, 0), 'zyz')
    test_one_case((pi/2,  pi, 0), 'ZYZ')

    # asymmetric
    test_one_case((pi/2,  pi/2, 0), 'zyx')
    test_one_case((pi/2,  -pi/2, 0), 'zyx')
    test_one_case((pi/2,  pi/2, 0), 'ZYX')
    test_one_case((pi/2,  -pi/2, 0), 'ZYX')


@slow
def test_to_euler_options():
    def test_one_case(q):
        angles1 = Matrix(q.to_euler(seq, True, True))
        angles2 = Matrix(q.to_euler(seq, False, False))
        angle_errors = simplify(angles1-angles2).evalf()
        for angle_error in angle_errors:
            # forcing angles to set {-pi, pi}
            angle_error = (angle_error + pi) % (2 * pi) - pi
            assert angle_error < 10e-7

    for xyz in ('xyz', 'XYZ'):
        for seq_tuple in permutations(xyz):
            for symmetric in (True, False):
                if symmetric:
                    seq = ''.join([seq_tuple[0], seq_tuple[1], seq_tuple[0]])
                else:
                    seq = ''.join(seq_tuple)

                for elements in product([-1, 0, 1], repeat=4):
                    q = Quaternion(*elements)
                    if not q.is_zero_quaternion():
                        test_one_case(q)
