from sympy import symbols
from sympy.physics.mechanics import Point, Particle, ReferenceFrame, inertia
from sympy.physics.mechanics.body_base import BodyBase
from sympy.testing.pytest import raises, warns_deprecated_sympy


def test_particle_default():
    # Test default
    p = Particle('P')
    assert p.name == 'P'
    assert p.mass == symbols('P_mass')
    assert p.masscenter.name == 'P_masscenter'
    assert p.potential_energy == 0
    assert p.__str__() == 'P'
    assert p.__repr__() == ("Particle('P', masscenter=P_masscenter, "
                            "mass=P_mass)")
    raises(AttributeError, lambda: p.frame)


def test_particle():
    # Test initializing with parameters
    m, m2, v1, v2, v3, r, g, h = symbols('m m2 v1 v2 v3 r g h')
    P = Point('P')
    P2 = Point('P2')
    p = Particle('pa', P, m)
    assert isinstance(p, BodyBase)
    assert p.mass == m
    assert p.point == P
    # Test the mass setter
    p.mass = m2
    assert p.mass == m2
    # Test the point setter
    p.point = P2
    assert p.point == P2
    # Test the linear momentum function
    N = ReferenceFrame('N')
    O = Point('O')
    P2.set_pos(O, r * N.y)
    P2.set_vel(N, v1 * N.x)
    raises(TypeError, lambda: Particle(P, P, m))
    raises(TypeError, lambda: Particle('pa', m, m))
    assert p.linear_momentum(N) == m2 * v1 * N.x
    assert p.angular_momentum(O, N) == -m2 * r * v1 * N.z
    P2.set_vel(N, v2 * N.y)
    assert p.linear_momentum(N) == m2 * v2 * N.y
    assert p.angular_momentum(O, N) == 0
    P2.set_vel(N, v3 * N.z)
    assert p.linear_momentum(N) == m2 * v3 * N.z
    assert p.angular_momentum(O, N) == m2 * r * v3 * N.x
    P2.set_vel(N, v1 * N.x + v2 * N.y + v3 * N.z)
    assert p.linear_momentum(N) == m2 * (v1 * N.x + v2 * N.y + v3 * N.z)
    assert p.angular_momentum(O, N) == m2 * r * (v3 * N.x - v1 * N.z)
    p.potential_energy = m * g * h
    assert p.potential_energy == m * g * h
    # TODO make the result not be system-dependent
    assert p.kinetic_energy(
        N) in [m2 * (v1 ** 2 + v2 ** 2 + v3 ** 2) / 2,
               m2 * v1 ** 2 / 2 + m2 * v2 ** 2 / 2 + m2 * v3 ** 2 / 2]


def test_parallel_axis():
    N = ReferenceFrame('N')
    m, a, b = symbols('m, a, b')
    o = Point('o')
    p = o.locatenew('p', a * N.x + b * N.y)
    P = Particle('P', o, m)
    Ip = P.parallel_axis(p, N)
    Ip_expected = inertia(N, m * b ** 2, m * a ** 2, m * (a ** 2 + b ** 2),
                          ixy=-m * a * b)
    assert Ip == Ip_expected


def test_deprecated_set_potential_energy():
    m, g, h = symbols('m g h')
    P = Point('P')
    p = Particle('pa', P, m)
    with warns_deprecated_sympy():
        p.set_potential_energy(m * g * h)
