from sympy import symbols
from sympy.testing.pytest import raises
from sympy.physics.mechanics import (inertia, inertia_of_point_mass,
                                     Inertia, ReferenceFrame, Point)


def test_inertia_dyadic():
    N = ReferenceFrame('N')
    ixx, iyy, izz = symbols('ixx iyy izz')
    ixy, iyz, izx = symbols('ixy iyz izx')
    assert inertia(N, ixx, iyy, izz) == (ixx * (N.x | N.x) + iyy *
            (N.y | N.y) + izz * (N.z | N.z))
    assert inertia(N, 0, 0, 0) == 0 * (N.x | N.x)
    raises(TypeError, lambda: inertia(0, 0, 0, 0))
    assert inertia(N, ixx, iyy, izz, ixy, iyz, izx) == (ixx * (N.x | N.x) +
            ixy * (N.x | N.y) + izx * (N.x | N.z) + ixy * (N.y | N.x) + iyy *
        (N.y | N.y) + iyz * (N.y | N.z) + izx * (N.z | N.x) + iyz * (N.z |
            N.y) + izz * (N.z | N.z))


def test_inertia_of_point_mass():
    r, s, t, m = symbols('r s t m')
    N = ReferenceFrame('N')

    px = r * N.x
    I = inertia_of_point_mass(m, px, N)
    assert I == m * r**2 * (N.y | N.y) + m * r**2 * (N.z | N.z)

    py = s * N.y
    I = inertia_of_point_mass(m, py, N)
    assert I == m * s**2 * (N.x | N.x) + m * s**2 * (N.z | N.z)

    pz = t * N.z
    I = inertia_of_point_mass(m, pz, N)
    assert I == m * t**2 * (N.x | N.x) + m * t**2 * (N.y | N.y)

    p = px + py + pz
    I = inertia_of_point_mass(m, p, N)
    assert I == (m * (s**2 + t**2) * (N.x | N.x) -
                 m * r * s * (N.x | N.y) -
                 m * r * t * (N.x | N.z) -
                 m * r * s * (N.y | N.x) +
                 m * (r**2 + t**2) * (N.y | N.y) -
                 m * s * t * (N.y | N.z) -
                 m * r * t * (N.z | N.x) -
                 m * s * t * (N.z | N.y) +
                 m * (r**2 + s**2) * (N.z | N.z))


def test_inertia_object():
    N = ReferenceFrame('N')
    O = Point('O')
    ixx, iyy, izz = symbols('ixx iyy izz')
    I_dyadic = ixx * (N.x | N.x) + iyy * (N.y | N.y) + izz * (N.z | N.z)
    I = Inertia(inertia(N, ixx, iyy, izz), O)
    assert isinstance(I, tuple)
    assert I.__repr__() == ('Inertia(dyadic=ixx*(N.x|N.x) + iyy*(N.y|N.y) + '
                            'izz*(N.z|N.z), point=O)')
    assert I.dyadic == I_dyadic
    assert I.point == O
    assert I[0] == I_dyadic
    assert I[1] == O
    assert I == (I_dyadic, O)  # Test tuple equal
    raises(TypeError, lambda: I != (O, I_dyadic))  # Incorrect tuple order
    assert I == Inertia(O, I_dyadic)  # Parse changed argument order
    assert I == Inertia.from_inertia_scalars(O, N, ixx, iyy, izz)
    # Test invalid tuple operations
    raises(TypeError, lambda: I + (1, 2))
    raises(TypeError, lambda: (1, 2) + I)
    raises(TypeError, lambda: I * 2)
    raises(TypeError, lambda: 2 * I)
