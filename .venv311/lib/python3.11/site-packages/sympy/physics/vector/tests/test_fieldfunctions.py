from sympy.core.singleton import S
from sympy.core.symbol import Symbol
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.physics.vector import ReferenceFrame, Vector, Point, \
     dynamicsymbols
from sympy.physics.vector.fieldfunctions import divergence, \
     gradient, curl, is_conservative, is_solenoidal, \
     scalar_potential, scalar_potential_difference
from sympy.testing.pytest import raises

R = ReferenceFrame('R')
q = dynamicsymbols('q')
P = R.orientnew('P', 'Axis', [q, R.z])


def test_curl():
    assert curl(Vector(0), R) == Vector(0)
    assert curl(R.x, R) == Vector(0)
    assert curl(2*R[1]**2*R.y, R) == Vector(0)
    assert curl(R[0]*R[1]*R.z, R) == R[0]*R.x - R[1]*R.y
    assert curl(R[0]*R[1]*R[2] * (R.x+R.y+R.z), R) == \
           (-R[0]*R[1] + R[0]*R[2])*R.x + (R[0]*R[1] - R[1]*R[2])*R.y + \
           (-R[0]*R[2] + R[1]*R[2])*R.z
    assert curl(2*R[0]**2*R.y, R) == 4*R[0]*R.z
    assert curl(P[0]**2*R.x + P.y, R) == \
           - 2*(R[0]*cos(q) + R[1]*sin(q))*sin(q)*R.z
    assert curl(P[0]*R.y, P) == cos(q)*P.z


def test_divergence():
    assert divergence(Vector(0), R) is S.Zero
    assert divergence(R.x, R) is S.Zero
    assert divergence(R[0]**2*R.x, R) == 2*R[0]
    assert divergence(R[0]*R[1]*R[2] * (R.x+R.y+R.z), R) == \
           R[0]*R[1] + R[0]*R[2] + R[1]*R[2]
    assert divergence((1/(R[0]*R[1]*R[2])) * (R.x+R.y+R.z), R) == \
           -1/(R[0]*R[1]*R[2]**2) - 1/(R[0]*R[1]**2*R[2]) - \
           1/(R[0]**2*R[1]*R[2])
    v = P[0]*P.x + P[1]*P.y + P[2]*P.z
    assert divergence(v, P) == 3
    assert divergence(v, R).simplify() == 3
    assert divergence(P[0]*R.x + R[0]*P.x, R) == 2*cos(q)


def test_gradient():
    a = Symbol('a')
    assert gradient(0, R) == Vector(0)
    assert gradient(R[0], R) == R.x
    assert gradient(R[0]*R[1]*R[2], R) == \
           R[1]*R[2]*R.x + R[0]*R[2]*R.y + R[0]*R[1]*R.z
    assert gradient(2*R[0]**2, R) == 4*R[0]*R.x
    assert gradient(a*sin(R[1])/R[0], R) == \
           - a*sin(R[1])/R[0]**2*R.x + a*cos(R[1])/R[0]*R.y
    assert gradient(P[0]*P[1], R) == \
           ((-R[0]*sin(q) + R[1]*cos(q))*cos(q) - (R[0]*cos(q) + R[1]*sin(q))*sin(q))*R.x + \
           ((-R[0]*sin(q) + R[1]*cos(q))*sin(q) + (R[0]*cos(q) + R[1]*sin(q))*cos(q))*R.y
    assert gradient(P[0]*R[2], P) == P[2]*P.x + P[0]*P.z


scalar_field = 2*R[0]**2*R[1]*R[2]
grad_field = gradient(scalar_field, R)
vector_field = R[1]**2*R.x + 3*R[0]*R.y + 5*R[1]*R[2]*R.z
curl_field = curl(vector_field, R)


def test_conservative():
    assert is_conservative(0) is True
    assert is_conservative(R.x) is True
    assert is_conservative(2 * R.x + 3 * R.y + 4 * R.z) is True
    assert is_conservative(R[1]*R[2]*R.x + R[0]*R[2]*R.y + R[0]*R[1]*R.z) is \
           True
    assert is_conservative(R[0] * R.y) is False
    assert is_conservative(grad_field) is True
    assert is_conservative(curl_field) is False
    assert is_conservative(4*R[0]*R[1]*R[2]*R.x + 2*R[0]**2*R[2]*R.y) is \
                           False
    assert is_conservative(R[2]*P.x + P[0]*R.z) is True


def test_solenoidal():
    assert is_solenoidal(0) is True
    assert is_solenoidal(R.x) is True
    assert is_solenoidal(2 * R.x + 3 * R.y + 4 * R.z) is True
    assert is_solenoidal(R[1]*R[2]*R.x + R[0]*R[2]*R.y + R[0]*R[1]*R.z) is \
           True
    assert is_solenoidal(R[1] * R.y) is False
    assert is_solenoidal(grad_field) is False
    assert is_solenoidal(curl_field) is True
    assert is_solenoidal((-2*R[1] + 3)*R.z) is True
    assert is_solenoidal(cos(q)*R.x + sin(q)*R.y + cos(q)*P.z) is True
    assert is_solenoidal(R[2]*P.x + P[0]*R.z) is True


def test_scalar_potential():
    assert scalar_potential(0, R) == 0
    assert scalar_potential(R.x, R) == R[0]
    assert scalar_potential(R.y, R) == R[1]
    assert scalar_potential(R.z, R) == R[2]
    assert scalar_potential(R[1]*R[2]*R.x + R[0]*R[2]*R.y + \
                            R[0]*R[1]*R.z, R) == R[0]*R[1]*R[2]
    assert scalar_potential(grad_field, R) == scalar_field
    assert scalar_potential(R[2]*P.x + P[0]*R.z, R) == \
           R[0]*R[2]*cos(q) + R[1]*R[2]*sin(q)
    assert scalar_potential(R[2]*P.x + P[0]*R.z, P) == P[0]*P[2]
    raises(ValueError, lambda: scalar_potential(R[0] * R.y, R))


def test_scalar_potential_difference():
    origin = Point('O')
    point1 = origin.locatenew('P1', 1*R.x + 2*R.y + 3*R.z)
    point2 = origin.locatenew('P2', 4*R.x + 5*R.y + 6*R.z)
    genericpointR = origin.locatenew('RP', R[0]*R.x + R[1]*R.y + R[2]*R.z)
    genericpointP = origin.locatenew('PP', P[0]*P.x + P[1]*P.y + P[2]*P.z)
    assert scalar_potential_difference(S.Zero, R, point1, point2, \
                                       origin) == 0
    assert scalar_potential_difference(scalar_field, R, origin, \
                                       genericpointR, origin) == \
                                       scalar_field
    assert scalar_potential_difference(grad_field, R, origin, \
                                       genericpointR, origin) == \
                                       scalar_field
    assert scalar_potential_difference(grad_field, R, point1, point2,
                                       origin) == 948
    assert scalar_potential_difference(R[1]*R[2]*R.x + R[0]*R[2]*R.y + \
                                       R[0]*R[1]*R.z, R, point1,
                                       genericpointR, origin) == \
                                       R[0]*R[1]*R[2] - 6
    potential_diff_P = 2*P[2]*(P[0]*sin(q) + P[1]*cos(q))*\
                       (P[0]*cos(q) - P[1]*sin(q))**2
    assert scalar_potential_difference(grad_field, P, origin, \
                                       genericpointP, \
                                       origin).simplify() == \
                                       potential_diff_P
