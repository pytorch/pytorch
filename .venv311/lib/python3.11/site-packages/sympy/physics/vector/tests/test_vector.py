from sympy.core.numbers import (Float, pi)
from sympy.core.symbol import symbols
from sympy.core.sorting import ordered
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.matrices.immutable import ImmutableDenseMatrix as Matrix
from sympy.physics.vector import ReferenceFrame, Vector, dynamicsymbols, dot
from sympy.physics.vector.vector import VectorTypeError
from sympy.abc import x, y, z
from sympy.testing.pytest import raises

A = ReferenceFrame('A')


def test_free_dynamicsymbols():
    A, B, C, D = symbols('A, B, C, D', cls=ReferenceFrame)
    a, b, c, d, e, f = dynamicsymbols('a, b, c, d, e, f')
    B.orient_axis(A, a, A.x)
    C.orient_axis(B, b, B.y)
    D.orient_axis(C, c, C.x)

    v = d*D.x + e*D.y + f*D.z

    assert set(ordered(v.free_dynamicsymbols(A))) == {a, b, c, d, e, f}
    assert set(ordered(v.free_dynamicsymbols(B))) == {b, c, d, e, f}
    assert set(ordered(v.free_dynamicsymbols(C))) == {c, d, e, f}
    assert set(ordered(v.free_dynamicsymbols(D))) == {d, e, f}


def test_Vector():
    assert A.x != A.y
    assert A.y != A.z
    assert A.z != A.x

    assert A.x + 0 == A.x

    v1 = x*A.x + y*A.y + z*A.z
    v2 = x**2*A.x + y**2*A.y + z**2*A.z
    v3 = v1 + v2
    v4 = v1 - v2

    assert isinstance(v1, Vector)
    assert dot(v1, A.x) == x
    assert dot(v1, A.y) == y
    assert dot(v1, A.z) == z

    assert isinstance(v2, Vector)
    assert dot(v2, A.x) == x**2
    assert dot(v2, A.y) == y**2
    assert dot(v2, A.z) == z**2

    assert isinstance(v3, Vector)
    # We probably shouldn't be using simplify in dot...
    assert dot(v3, A.x) == x**2 + x
    assert dot(v3, A.y) == y**2 + y
    assert dot(v3, A.z) == z**2 + z

    assert isinstance(v4, Vector)
    # We probably shouldn't be using simplify in dot...
    assert dot(v4, A.x) == x - x**2
    assert dot(v4, A.y) == y - y**2
    assert dot(v4, A.z) == z - z**2

    assert v1.to_matrix(A) == Matrix([[x], [y], [z]])
    q = symbols('q')
    B = A.orientnew('B', 'Axis', (q, A.x))
    assert v1.to_matrix(B) == Matrix([[x],
                                      [ y * cos(q) + z * sin(q)],
                                      [-y * sin(q) + z * cos(q)]])

    #Test the separate method
    B = ReferenceFrame('B')
    v5 = x*A.x + y*A.y + z*B.z
    assert Vector(0).separate() == {}
    assert v1.separate() == {A: v1}
    assert v5.separate() == {A: x*A.x + y*A.y, B: z*B.z}

    #Test the free_symbols property
    v6 = x*A.x + y*A.y + z*A.z
    assert v6.free_symbols(A) == {x,y,z}

    raises(TypeError, lambda: v3.applyfunc(v1))


def test_Vector_diffs():
    q1, q2, q3, q4 = dynamicsymbols('q1 q2 q3 q4')
    q1d, q2d, q3d, q4d = dynamicsymbols('q1 q2 q3 q4', 1)
    q1dd, q2dd, q3dd, q4dd = dynamicsymbols('q1 q2 q3 q4', 2)
    N = ReferenceFrame('N')
    A = N.orientnew('A', 'Axis', [q3, N.z])
    B = A.orientnew('B', 'Axis', [q2, A.x])
    v1 = q2 * A.x + q3 * N.y
    v2 = q3 * B.x + v1
    v3 = v1.dt(B)
    v4 = v2.dt(B)
    v5 = q1*A.x + q2*A.y + q3*A.z

    assert v1.dt(N) == q2d * A.x + q2 * q3d * A.y + q3d * N.y
    assert v1.dt(A) == q2d * A.x + q3 * q3d * N.x + q3d * N.y
    assert v1.dt(B) == (q2d * A.x + q3 * q3d * N.x + q3d *
                        N.y - q3 * cos(q3) * q2d * N.z)
    assert v2.dt(N) == (q2d * A.x + (q2 + q3) * q3d * A.y + q3d * B.x + q3d *
                        N.y)
    assert v2.dt(A) == q2d * A.x + q3d * B.x + q3 * q3d * N.x + q3d * N.y
    assert v2.dt(B) == (q2d * A.x + q3d * B.x + q3 * q3d * N.x + q3d * N.y -
                        q3 * cos(q3) * q2d * N.z)
    assert v3.dt(N) == (q2dd * A.x + q2d * q3d * A.y + (q3d**2 + q3 * q3dd) *
                        N.x + q3dd * N.y + (q3 * sin(q3) * q2d * q3d -
                        cos(q3) * q2d * q3d - q3 * cos(q3) * q2dd) * N.z)
    assert v3.dt(A) == (q2dd * A.x + (2 * q3d**2 + q3 * q3dd) * N.x + (q3dd -
                        q3 * q3d**2) * N.y + (q3 * sin(q3) * q2d * q3d -
                        cos(q3) * q2d * q3d - q3 * cos(q3) * q2dd) * N.z)
    assert (v3.dt(B) - (q2dd*A.x - q3*cos(q3)*q2d**2*A.y + (2*q3d**2 +
        q3*q3dd)*N.x + (q3dd - q3*q3d**2)*N.y + (2*q3*sin(q3)*q2d*q3d -
        2*cos(q3)*q2d*q3d - q3*cos(q3)*q2dd)*N.z)).express(B).simplify() == 0
    assert v4.dt(N) == (q2dd * A.x + q3d * (q2d + q3d) * A.y + q3dd * B.x +
                        (q3d**2 + q3 * q3dd) * N.x + q3dd * N.y + (q3 *
                        sin(q3) * q2d * q3d - cos(q3) * q2d * q3d - q3 *
                        cos(q3) * q2dd) * N.z)
    assert v4.dt(A) == (q2dd * A.x + q3dd * B.x + (2 * q3d**2 + q3 * q3dd) *
                        N.x + (q3dd - q3 * q3d**2) * N.y + (q3 * sin(q3) *
                        q2d * q3d - cos(q3) * q2d * q3d - q3 * cos(q3) *
                        q2dd) * N.z)
    assert (v4.dt(B) - (q2dd*A.x - q3*cos(q3)*q2d**2*A.y + q3dd*B.x +
                        (2*q3d**2 + q3*q3dd)*N.x + (q3dd - q3*q3d**2)*N.y +
                        (2*q3*sin(q3)*q2d*q3d - 2*cos(q3)*q2d*q3d -
                         q3*cos(q3)*q2dd)*N.z)).express(B).simplify() == 0
    assert v5.dt(B) == q1d*A.x + (q3*q2d + q2d)*A.y + (-q2*q2d + q3d)*A.z
    assert v5.dt(A) == q1d*A.x + q2d*A.y + q3d*A.z
    assert v5.dt(N) == (-q2*q3d + q1d)*A.x + (q1*q3d + q2d)*A.y + q3d*A.z
    assert v3.diff(q1d, N) == 0
    assert v3.diff(q2d, N) == A.x - q3 * cos(q3) * N.z
    assert v3.diff(q3d, N) == q3 * N.x + N.y
    assert v3.diff(q1d, A) == 0
    assert v3.diff(q2d, A) == A.x - q3 * cos(q3) * N.z
    assert v3.diff(q3d, A) == q3 * N.x + N.y
    assert v3.diff(q1d, B) == 0
    assert v3.diff(q2d, B) == A.x - q3 * cos(q3) * N.z
    assert v3.diff(q3d, B) == q3 * N.x + N.y
    assert v4.diff(q1d, N) == 0
    assert v4.diff(q2d, N) == A.x - q3 * cos(q3) * N.z
    assert v4.diff(q3d, N) == B.x + q3 * N.x + N.y
    assert v4.diff(q1d, A) == 0
    assert v4.diff(q2d, A) == A.x - q3 * cos(q3) * N.z
    assert v4.diff(q3d, A) == B.x + q3 * N.x + N.y
    assert v4.diff(q1d, B) == 0
    assert v4.diff(q2d, B) == A.x - q3 * cos(q3) * N.z
    assert v4.diff(q3d, B) == B.x + q3 * N.x + N.y

    # diff() should only express vector components in the derivative frame if
    # the orientation of the component's frame depends on the variable
    v6 = q2**2*N.y + q2**2*A.y + q2**2*B.y
    # already expressed in N
    n_measy = 2*q2
    # A_C_N does not depend on q2, so don't express in N
    a_measy = 2*q2
    # B_C_N depends on q2, so express in N
    b_measx = (q2**2*B.y).dot(N.x).diff(q2)
    b_measy = (q2**2*B.y).dot(N.y).diff(q2)
    b_measz = (q2**2*B.y).dot(N.z).diff(q2)
    n_comp, a_comp = v6.diff(q2, N).args
    assert len(v6.diff(q2, N).args) == 2  # only N and A parts
    assert n_comp[1] == N
    assert a_comp[1] == A
    assert n_comp[0] == Matrix([b_measx, b_measy + n_measy, b_measz])
    assert a_comp[0] == Matrix([0, a_measy, 0])


def test_vector_var_in_dcm():

    N = ReferenceFrame('N')
    A = ReferenceFrame('A')
    B = ReferenceFrame('B')
    u1, u2, u3, u4 = dynamicsymbols('u1 u2 u3 u4')

    v = u1 * u2 * A.x + u3 * N.y + u4**2 * N.z

    assert v.diff(u1, N, var_in_dcm=False) == u2 * A.x
    assert v.diff(u1, A, var_in_dcm=False) == u2 * A.x
    assert v.diff(u3, N, var_in_dcm=False) == N.y
    assert v.diff(u3, A, var_in_dcm=False) == N.y
    assert v.diff(u3, B, var_in_dcm=False) == N.y
    assert v.diff(u4, N, var_in_dcm=False) == 2 * u4 * N.z

    raises(ValueError, lambda: v.diff(u1, N))


def test_vector_simplify():
    x, y, z, k, n, m, w, f, s, A = symbols('x, y, z, k, n, m, w, f, s, A')
    N = ReferenceFrame('N')

    test1 = (1 / x + 1 / y) * N.x
    assert (test1 & N.x) != (x + y) / (x * y)
    test1 = test1.simplify()
    assert (test1 & N.x) == (x + y) / (x * y)

    test2 = (A**2 * s**4 / (4 * pi * k * m**3)) * N.x
    test2 = test2.simplify()
    assert (test2 & N.x) == (A**2 * s**4 / (4 * pi * k * m**3))

    test3 = ((4 + 4 * x - 2 * (2 + 2 * x)) / (2 + 2 * x)) * N.x
    test3 = test3.simplify()
    assert (test3 & N.x) == 0

    test4 = ((-4 * x * y**2 - 2 * y**3 - 2 * x**2 * y) / (x + y)**2) * N.x
    test4 = test4.simplify()
    assert (test4 & N.x) == -2 * y


def test_vector_evalf():
    a, b = symbols('a b')
    v = pi * A.x
    assert v.evalf(2) == Float('3.1416', 2) * A.x
    v = pi * A.x + 5 * a * A.y - b * A.z
    assert v.evalf(3) == Float('3.1416', 3) * A.x + Float('5', 3) * a * A.y - b * A.z
    assert v.evalf(5, subs={a: 1.234, b:5.8973}) == Float('3.1415926536', 5) * A.x + Float('6.17', 5) * A.y - Float('5.8973', 5) * A.z


def test_vector_angle():
    A = ReferenceFrame('A')
    v1 = A.x + A.y
    v2 = A.z
    assert v1.angle_between(v2) == pi/2
    B = ReferenceFrame('B')
    B.orient_axis(A, A.x, pi)
    v3 = A.x
    v4 = B.x
    assert v3.angle_between(v4) == 0


def test_vector_xreplace():
    x, y, z = symbols('x y z')
    v = x**2 * A.x + x*y * A.y + x*y*z * A.z
    assert v.xreplace({x : cos(x)}) == cos(x)**2 * A.x + y*cos(x) * A.y + y*z*cos(x) * A.z
    assert v.xreplace({x*y : pi}) == x**2 * A.x + pi * A.y + x*y*z * A.z
    assert v.xreplace({x*y*z : 1}) == x**2*A.x + x*y*A.y + A.z
    assert v.xreplace({x:1, z:0}) == A.x + y * A.y
    raises(TypeError, lambda: v.xreplace())
    raises(TypeError, lambda: v.xreplace([x, y]))

def test_issue_23366():
    u1 = dynamicsymbols('u1')
    N = ReferenceFrame('N')
    N_v_A = u1*N.x
    raises(VectorTypeError, lambda: N_v_A.diff(N, u1))


def test_vector_outer():
    a, b, c, d, e, f = symbols('a, b, c, d, e, f')
    N = ReferenceFrame('N')
    v1 = a*N.x + b*N.y + c*N.z
    v2 = d*N.x + e*N.y + f*N.z
    v1v2 = Matrix([[a*d, a*e, a*f],
                   [b*d, b*e, b*f],
                   [c*d, c*e, c*f]])
    assert v1.outer(v2).to_matrix(N) == v1v2
    assert (v1 | v2).to_matrix(N) == v1v2
    v2v1 = Matrix([[d*a, d*b, d*c],
                   [e*a, e*b, e*c],
                   [f*a, f*b, f*c]])
    assert v2.outer(v1).to_matrix(N) == v2v1
    assert (v2 | v1).to_matrix(N) == v2v1


def test_overloaded_operators():
    a, b, c, d, e, f = symbols('a, b, c, d, e, f')
    N = ReferenceFrame('N')
    v1 = a*N.x + b*N.y + c*N.z
    v2 = d*N.x + e*N.y + f*N.z

    assert v1 + v2 == v2 + v1
    assert v1 - v2 == -v2 + v1
    assert v1 & v2 == v2 & v1
    assert v1 ^ v2 == v1.cross(v2)
    assert v2 ^ v1 == v2.cross(v1)
