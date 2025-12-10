from sympy.core.numbers import comp, Rational
from sympy.physics.optics.utils import (refraction_angle, fresnel_coefficients,
        deviation, brewster_angle, critical_angle, lens_makers_formula,
        mirror_formula, lens_formula, hyperfocal_distance,
        transverse_magnification)
from sympy.physics.optics.medium import Medium
from sympy.physics.units import e0

from sympy.core.numbers import oo
from sympy.core.symbol import symbols
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.matrices.dense import Matrix
from sympy.geometry.point import Point3D
from sympy.geometry.line import Ray3D
from sympy.geometry.plane import Plane

from sympy.testing.pytest import raises


ae = lambda a, b, n: comp(a, b, 10**-n)


def test_refraction_angle():
    n1, n2 = symbols('n1, n2')
    m1 = Medium('m1')
    m2 = Medium('m2')
    r1 = Ray3D(Point3D(-1, -1, 1), Point3D(0, 0, 0))
    i = Matrix([1, 1, 1])
    n = Matrix([0, 0, 1])
    normal_ray = Ray3D(Point3D(0, 0, 0), Point3D(0, 0, 1))
    P = Plane(Point3D(0, 0, 0), normal_vector=[0, 0, 1])
    assert refraction_angle(r1, 1, 1, n) == Matrix([
                                            [ 1],
                                            [ 1],
                                            [-1]])
    assert refraction_angle([1, 1, 1], 1, 1, n) == Matrix([
                                            [ 1],
                                            [ 1],
                                            [-1]])
    assert refraction_angle((1, 1, 1), 1, 1, n) == Matrix([
                                            [ 1],
                                            [ 1],
                                            [-1]])
    assert refraction_angle(i, 1, 1, [0, 0, 1]) == Matrix([
                                            [ 1],
                                            [ 1],
                                            [-1]])
    assert refraction_angle(i, 1, 1, (0, 0, 1)) == Matrix([
                                            [ 1],
                                            [ 1],
                                            [-1]])
    assert refraction_angle(i, 1, 1, normal_ray) == Matrix([
                                            [ 1],
                                            [ 1],
                                            [-1]])
    assert refraction_angle(i, 1, 1, plane=P) == Matrix([
                                            [ 1],
                                            [ 1],
                                            [-1]])
    assert refraction_angle(r1, 1, 1, plane=P) == \
        Ray3D(Point3D(0, 0, 0), Point3D(1, 1, -1))
    assert refraction_angle(r1, m1, 1.33, plane=P) == \
        Ray3D(Point3D(0, 0, 0), Point3D(Rational(100, 133), Rational(100, 133), -789378201649271*sqrt(3)/1000000000000000))
    assert refraction_angle(r1, 1, m2, plane=P) == \
        Ray3D(Point3D(0, 0, 0), Point3D(1, 1, -1))
    assert refraction_angle(r1, n1, n2, plane=P) == \
        Ray3D(Point3D(0, 0, 0), Point3D(n1/n2, n1/n2, -sqrt(3)*sqrt(-2*n1**2/(3*n2**2) + 1)))
    assert refraction_angle(r1, 1.33, 1, plane=P) == 0  # TIR
    assert refraction_angle(r1, 1, 1, normal_ray) == \
        Ray3D(Point3D(0, 0, 0), direction_ratio=[1, 1, -1])
    assert ae(refraction_angle(0.5, 1, 2), 0.24207, 5)
    assert ae(refraction_angle(0.5, 2, 1), 1.28293, 5)
    raises(ValueError, lambda: refraction_angle(r1, m1, m2, normal_ray, P))
    raises(TypeError, lambda: refraction_angle(m1, m1, m2)) # can add other values for arg[0]
    raises(TypeError, lambda: refraction_angle(r1, m1, m2, None, i))
    raises(TypeError, lambda: refraction_angle(r1, m1, m2, m2))


def test_fresnel_coefficients():
    assert all(ae(i, j, 5) for i, j in zip(
        fresnel_coefficients(0.5, 1, 1.33),
        [0.11163, -0.17138, 0.83581, 0.82862]))
    assert all(ae(i, j, 5) for i, j in zip(
        fresnel_coefficients(0.5, 1.33, 1),
        [-0.07726, 0.20482, 1.22724, 1.20482]))
    m1 = Medium('m1')
    m2 = Medium('m2', n=2)
    assert all(ae(i, j, 5) for i, j in zip(
        fresnel_coefficients(0.3, m1, m2),
        [0.31784, -0.34865, 0.65892, 0.65135]))
    ans = [[-0.23563, -0.97184], [0.81648, -0.57738]]
    got = fresnel_coefficients(0.6, m2, m1)
    for i, j in zip(got, ans):
        for a, b in zip(i.as_real_imag(), j):
            assert ae(a, b, 5)


def test_deviation():
    n1, n2 = symbols('n1, n2')
    r1 = Ray3D(Point3D(-1, -1, 1), Point3D(0, 0, 0))
    n = Matrix([0, 0, 1])
    i = Matrix([-1, -1, -1])
    normal_ray = Ray3D(Point3D(0, 0, 0), Point3D(0, 0, 1))
    P = Plane(Point3D(0, 0, 0), normal_vector=[0, 0, 1])
    assert deviation(r1, 1, 1, normal=n) == 0
    assert deviation(r1, 1, 1, plane=P) == 0
    assert deviation(r1, 1, 1.1, plane=P).evalf(3) + 0.119 < 1e-3
    assert deviation(i, 1, 1.1, normal=normal_ray).evalf(3) + 0.119 < 1e-3
    assert deviation(r1, 1.33, 1, plane=P) is None  # TIR
    assert deviation(r1, 1, 1, normal=[0, 0, 1]) == 0
    assert deviation([-1, -1, -1], 1, 1, normal=[0, 0, 1]) == 0
    assert ae(deviation(0.5, 1, 2), -0.25793, 5)
    assert ae(deviation(0.5, 2, 1), 0.78293, 5)


def test_brewster_angle():
    m1 = Medium('m1', n=1)
    m2 = Medium('m2', n=1.33)
    assert ae(brewster_angle(m1, m2), 0.93, 2)
    m1 = Medium('m1', permittivity=e0, n=1)
    m2 = Medium('m2', permittivity=e0, n=1.33)
    assert ae(brewster_angle(m1, m2), 0.93, 2)
    assert ae(brewster_angle(1, 1.33), 0.93, 2)


def test_critical_angle():
    m1 = Medium('m1', n=1)
    m2 = Medium('m2', n=1.33)
    assert ae(critical_angle(m2, m1), 0.85, 2)


def test_lens_makers_formula():
    n1, n2 = symbols('n1, n2')
    m1 = Medium('m1', permittivity=e0, n=1)
    m2 = Medium('m2', permittivity=e0, n=1.33)
    assert lens_makers_formula(n1, n2, 10, -10) == 5.0*n2/(n1 - n2)
    assert ae(lens_makers_formula(m1, m2, 10, -10), -20.15, 2)
    assert ae(lens_makers_formula(1.33, 1, 10, -10),  15.15, 2)


def test_mirror_formula():
    u, v, f = symbols('u, v, f')
    assert mirror_formula(focal_length=f, u=u) == f*u/(-f + u)
    assert mirror_formula(focal_length=f, v=v) == f*v/(-f + v)
    assert mirror_formula(u=u, v=v) == u*v/(u + v)
    assert mirror_formula(u=oo, v=v) == v
    assert mirror_formula(u=oo, v=oo) is oo
    assert mirror_formula(focal_length=oo, u=u) == -u
    assert mirror_formula(u=u, v=oo) == u
    assert mirror_formula(focal_length=oo, v=oo) is oo
    assert mirror_formula(focal_length=f, v=oo) == f
    assert mirror_formula(focal_length=oo, v=v) == -v
    assert mirror_formula(focal_length=oo, u=oo) is oo
    assert mirror_formula(focal_length=f, u=oo) == f
    assert mirror_formula(focal_length=oo, u=u) == -u
    raises(ValueError, lambda: mirror_formula(focal_length=f, u=u, v=v))


def test_lens_formula():
    u, v, f = symbols('u, v, f')
    assert lens_formula(focal_length=f, u=u) == f*u/(f + u)
    assert lens_formula(focal_length=f, v=v) == f*v/(f - v)
    assert lens_formula(u=u, v=v) == u*v/(u - v)
    assert lens_formula(u=oo, v=v) == v
    assert lens_formula(u=oo, v=oo) is oo
    assert lens_formula(focal_length=oo, u=u) == u
    assert lens_formula(u=u, v=oo) == -u
    assert lens_formula(focal_length=oo, v=oo) is -oo
    assert lens_formula(focal_length=oo, v=v) == v
    assert lens_formula(focal_length=f, v=oo) == -f
    assert lens_formula(focal_length=oo, u=oo) is oo
    assert lens_formula(focal_length=oo, u=u) == u
    assert lens_formula(focal_length=f, u=oo) == f
    raises(ValueError, lambda: lens_formula(focal_length=f, u=u, v=v))


def test_hyperfocal_distance():
    f, N, c = symbols('f, N, c')
    assert hyperfocal_distance(f=f, N=N, c=c) == f**2/(N*c)
    assert ae(hyperfocal_distance(f=0.5, N=8, c=0.0033), 9.47, 2)


def test_transverse_magnification():
    si, so = symbols('si, so')
    assert transverse_magnification(si, so) == -si/so
    assert transverse_magnification(30, 15) == -2


def test_lens_makers_formula_thick_lens():
    n1, n2 = symbols('n1, n2')
    m1 = Medium('m1', permittivity=e0, n=1)
    m2 = Medium('m2', permittivity=e0, n=1.33)
    assert ae(lens_makers_formula(m1, m2, 10, -10, d=1), -19.82, 2)
    assert lens_makers_formula(n1, n2, 1, -1, d=0.1) == n2/((2.0 - (0.1*n1 - 0.1*n2)/n1)*(n1 - n2))


def test_lens_makers_formula_plano_lens():
    n1, n2 = symbols('n1, n2')
    m1 = Medium('m1', permittivity=e0, n=1)
    m2 = Medium('m2', permittivity=e0, n=1.33)
    assert ae(lens_makers_formula(m1, m2, 10, oo), -40.30, 2)
    assert lens_makers_formula(n1, n2, 10, oo) == 10.0*n2/(n1 - n2)
