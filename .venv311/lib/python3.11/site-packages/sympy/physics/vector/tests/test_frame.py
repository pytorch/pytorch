from sympy.core.numbers import pi
from sympy.core.symbol import symbols
from sympy.simplify import trigsimp
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.matrices.dense import (eye, zeros)
from sympy.matrices.immutable import ImmutableDenseMatrix as Matrix
from sympy.simplify.simplify import simplify
from sympy.physics.vector import (ReferenceFrame, Vector, CoordinateSym,
                                  dynamicsymbols, time_derivative, express,
                                  dot)
from sympy.physics.vector.frame import _check_frame
from sympy.physics.vector.vector import VectorTypeError
from sympy.testing.pytest import raises
import warnings
import pickle


def test_dict_list():

    A = ReferenceFrame('A')
    B = ReferenceFrame('B')
    C = ReferenceFrame('C')
    D = ReferenceFrame('D')
    E = ReferenceFrame('E')
    F = ReferenceFrame('F')

    B.orient_axis(A, A.x, 1.0)
    C.orient_axis(B, B.x, 1.0)
    D.orient_axis(C, C.x, 1.0)

    assert D._dict_list(A, 0) == [D, C, B, A]

    E.orient_axis(D, D.x, 1.0)

    assert C._dict_list(A, 0) == [C, B, A]
    assert C._dict_list(E, 0) == [C, D, E]

    # only 0, 1, 2 permitted for second argument
    raises(ValueError, lambda: C._dict_list(E, 5))
    # no connecting path
    raises(ValueError, lambda: F._dict_list(A, 0))


def test_coordinate_vars():
    """Tests the coordinate variables functionality"""
    A = ReferenceFrame('A')
    assert CoordinateSym('Ax', A, 0) == A[0]
    assert CoordinateSym('Ax', A, 1) == A[1]
    assert CoordinateSym('Ax', A, 2) == A[2]
    raises(ValueError, lambda: CoordinateSym('Ax', A, 3))
    q = dynamicsymbols('q')
    qd = dynamicsymbols('q', 1)
    assert isinstance(A[0], CoordinateSym) and \
           isinstance(A[0], CoordinateSym) and \
           isinstance(A[0], CoordinateSym)
    assert A.variable_map(A) == {A[0]:A[0], A[1]:A[1], A[2]:A[2]}
    assert A[0].frame == A
    B = A.orientnew('B', 'Axis', [q, A.z])
    assert B.variable_map(A) == {B[2]: A[2], B[1]: -A[0]*sin(q) + A[1]*cos(q),
                                 B[0]: A[0]*cos(q) + A[1]*sin(q)}
    assert A.variable_map(B) == {A[0]: B[0]*cos(q) - B[1]*sin(q),
                                 A[1]: B[0]*sin(q) + B[1]*cos(q), A[2]: B[2]}
    assert time_derivative(B[0], A) == -A[0]*sin(q)*qd + A[1]*cos(q)*qd
    assert time_derivative(B[1], A) == -A[0]*cos(q)*qd - A[1]*sin(q)*qd
    assert time_derivative(B[2], A) == 0
    assert express(B[0], A, variables=True) == A[0]*cos(q) + A[1]*sin(q)
    assert express(B[1], A, variables=True) == -A[0]*sin(q) + A[1]*cos(q)
    assert express(B[2], A, variables=True) == A[2]
    assert time_derivative(A[0]*A.x + A[1]*A.y + A[2]*A.z, B) == A[1]*qd*A.x - A[0]*qd*A.y
    assert time_derivative(B[0]*B.x + B[1]*B.y + B[2]*B.z, A) == - B[1]*qd*B.x + B[0]*qd*B.y
    assert express(B[0]*B[1]*B[2], A, variables=True) == \
           A[2]*(-A[0]*sin(q) + A[1]*cos(q))*(A[0]*cos(q) + A[1]*sin(q))
    assert (time_derivative(B[0]*B[1]*B[2], A) -
            (A[2]*(-A[0]**2*cos(2*q) -
             2*A[0]*A[1]*sin(2*q) +
             A[1]**2*cos(2*q))*qd)).trigsimp() == 0
    assert express(B[0]*B.x + B[1]*B.y + B[2]*B.z, A) == \
           (B[0]*cos(q) - B[1]*sin(q))*A.x + (B[0]*sin(q) + \
           B[1]*cos(q))*A.y + B[2]*A.z
    assert express(B[0]*B.x + B[1]*B.y + B[2]*B.z, A,
                   variables=True).simplify() == A[0]*A.x + A[1]*A.y + A[2]*A.z
    assert express(A[0]*A.x + A[1]*A.y + A[2]*A.z, B) == \
           (A[0]*cos(q) + A[1]*sin(q))*B.x + \
           (-A[0]*sin(q) + A[1]*cos(q))*B.y + A[2]*B.z
    assert express(A[0]*A.x + A[1]*A.y + A[2]*A.z, B,
                   variables=True).simplify() == B[0]*B.x + B[1]*B.y + B[2]*B.z
    N = B.orientnew('N', 'Axis', [-q, B.z])
    assert ({k: v.simplify() for k, v in N.variable_map(A).items()} ==
            {N[0]: A[0], N[2]: A[2], N[1]: A[1]})
    C = A.orientnew('C', 'Axis', [q, A.x + A.y + A.z])
    mapping = A.variable_map(C)
    assert trigsimp(mapping[A[0]]) == (2*C[0]*cos(q)/3 + C[0]/3 -
                                       2*C[1]*sin(q + pi/6)/3 +
                                       C[1]/3 - 2*C[2]*cos(q + pi/3)/3 +
                                       C[2]/3)
    assert trigsimp(mapping[A[1]]) == -2*C[0]*cos(q + pi/3)/3 + \
           C[0]/3 + 2*C[1]*cos(q)/3 + C[1]/3 - 2*C[2]*sin(q + pi/6)/3 + C[2]/3
    assert trigsimp(mapping[A[2]]) == -2*C[0]*sin(q + pi/6)/3 + C[0]/3 - \
           2*C[1]*cos(q + pi/3)/3 + C[1]/3 + 2*C[2]*cos(q)/3 + C[2]/3


def test_ang_vel():
    q1, q2, q3, q4 = dynamicsymbols('q1 q2 q3 q4')
    q1d, q2d, q3d, q4d = dynamicsymbols('q1 q2 q3 q4', 1)
    N = ReferenceFrame('N')
    A = N.orientnew('A', 'Axis', [q1, N.z])
    B = A.orientnew('B', 'Axis', [q2, A.x])
    C = B.orientnew('C', 'Axis', [q3, B.y])
    D = N.orientnew('D', 'Axis', [q4, N.y])
    u1, u2, u3 = dynamicsymbols('u1 u2 u3')
    assert A.ang_vel_in(N) == (q1d)*A.z
    assert B.ang_vel_in(N) == (q2d)*B.x + (q1d)*A.z
    assert C.ang_vel_in(N) == (q3d)*C.y + (q2d)*B.x + (q1d)*A.z

    A2 = N.orientnew('A2', 'Axis', [q4, N.y])
    assert N.ang_vel_in(N) == 0
    assert N.ang_vel_in(A) == -q1d*N.z
    assert N.ang_vel_in(B) == -q1d*A.z - q2d*B.x
    assert N.ang_vel_in(C) == -q1d*A.z - q2d*B.x - q3d*B.y
    assert N.ang_vel_in(A2) == -q4d*N.y

    assert A.ang_vel_in(N) == q1d*N.z
    assert A.ang_vel_in(A) == 0
    assert A.ang_vel_in(B) == - q2d*B.x
    assert A.ang_vel_in(C) == - q2d*B.x - q3d*B.y
    assert A.ang_vel_in(A2) == q1d*N.z - q4d*N.y

    assert B.ang_vel_in(N) == q1d*A.z + q2d*A.x
    assert B.ang_vel_in(A) == q2d*A.x
    assert B.ang_vel_in(B) == 0
    assert B.ang_vel_in(C) == -q3d*B.y
    assert B.ang_vel_in(A2) == q1d*A.z + q2d*A.x - q4d*N.y

    assert C.ang_vel_in(N) == q1d*A.z + q2d*A.x + q3d*B.y
    assert C.ang_vel_in(A) == q2d*A.x + q3d*C.y
    assert C.ang_vel_in(B) == q3d*B.y
    assert C.ang_vel_in(C) == 0
    assert C.ang_vel_in(A2) == q1d*A.z + q2d*A.x + q3d*B.y - q4d*N.y

    assert A2.ang_vel_in(N) == q4d*A2.y
    assert A2.ang_vel_in(A) == q4d*A2.y - q1d*N.z
    assert A2.ang_vel_in(B) == q4d*N.y - q1d*A.z - q2d*A.x
    assert A2.ang_vel_in(C) == q4d*N.y - q1d*A.z - q2d*A.x - q3d*B.y
    assert A2.ang_vel_in(A2) == 0

    C.set_ang_vel(N, u1*C.x + u2*C.y + u3*C.z)
    assert C.ang_vel_in(N) == (u1)*C.x + (u2)*C.y + (u3)*C.z
    assert N.ang_vel_in(C) == (-u1)*C.x + (-u2)*C.y + (-u3)*C.z
    assert C.ang_vel_in(D) == (u1)*C.x + (u2)*C.y + (u3)*C.z + (-q4d)*D.y
    assert D.ang_vel_in(C) == (-u1)*C.x + (-u2)*C.y + (-u3)*C.z + (q4d)*D.y

    q0 = dynamicsymbols('q0')
    q0d = dynamicsymbols('q0', 1)
    E = N.orientnew('E', 'Quaternion', (q0, q1, q2, q3))
    assert E.ang_vel_in(N) == (
        2 * (q1d * q0 + q2d * q3 - q3d * q2 - q0d * q1) * E.x +
        2 * (q2d * q0 + q3d * q1 - q1d * q3 - q0d * q2) * E.y +
        2 * (q3d * q0 + q1d * q2 - q2d * q1 - q0d * q3) * E.z)

    F = N.orientnew('F', 'Body', (q1, q2, q3), 313)
    assert F.ang_vel_in(N) == ((sin(q2)*sin(q3)*q1d + cos(q3)*q2d)*F.x +
        (sin(q2)*cos(q3)*q1d - sin(q3)*q2d)*F.y + (cos(q2)*q1d + q3d)*F.z)
    G = N.orientnew('G', 'Axis', (q1, N.x + N.y))
    assert G.ang_vel_in(N) == q1d * (N.x + N.y).normalize()
    assert N.ang_vel_in(G) == -q1d * (N.x + N.y).normalize()


def test_dcm():
    q1, q2, q3, q4 = dynamicsymbols('q1 q2 q3 q4')
    N = ReferenceFrame('N')
    A = N.orientnew('A', 'Axis', [q1, N.z])
    B = A.orientnew('B', 'Axis', [q2, A.x])
    C = B.orientnew('C', 'Axis', [q3, B.y])
    D = N.orientnew('D', 'Axis', [q4, N.y])
    E = N.orientnew('E', 'Space', [q1, q2, q3], '123')
    assert N.dcm(C) == Matrix([
        [- sin(q1) * sin(q2) * sin(q3) + cos(q1) * cos(q3), - sin(q1) *
        cos(q2), sin(q1) * sin(q2) * cos(q3) + sin(q3) * cos(q1)], [sin(q1) *
        cos(q3) + sin(q2) * sin(q3) * cos(q1), cos(q1) * cos(q2), sin(q1) *
            sin(q3) - sin(q2) * cos(q1) * cos(q3)], [- sin(q3) * cos(q2), sin(q2),
        cos(q2) * cos(q3)]])
    # This is a little touchy.  Is it ok to use simplify in assert?
    test_mat = D.dcm(C) - Matrix(
        [[cos(q1) * cos(q3) * cos(q4) - sin(q3) * (- sin(q4) * cos(q2) +
        sin(q1) * sin(q2) * cos(q4)), - sin(q2) * sin(q4) - sin(q1) *
            cos(q2) * cos(q4), sin(q3) * cos(q1) * cos(q4) + cos(q3) * (- sin(q4) *
        cos(q2) + sin(q1) * sin(q2) * cos(q4))], [sin(q1) * cos(q3) +
        sin(q2) * sin(q3) * cos(q1), cos(q1) * cos(q2), sin(q1) * sin(q3) -
            sin(q2) * cos(q1) * cos(q3)], [sin(q4) * cos(q1) * cos(q3) -
        sin(q3) * (cos(q2) * cos(q4) + sin(q1) * sin(q2) * sin(q4)), sin(q2) *
                cos(q4) - sin(q1) * sin(q4) * cos(q2), sin(q3) * sin(q4) * cos(q1) +
                cos(q3) * (cos(q2) * cos(q4) + sin(q1) * sin(q2) * sin(q4))]])
    assert test_mat.expand() == zeros(3, 3)
    assert E.dcm(N) == Matrix(
        [[cos(q2)*cos(q3), sin(q3)*cos(q2), -sin(q2)],
        [sin(q1)*sin(q2)*cos(q3) - sin(q3)*cos(q1), sin(q1)*sin(q2)*sin(q3) +
        cos(q1)*cos(q3), sin(q1)*cos(q2)], [sin(q1)*sin(q3) +
        sin(q2)*cos(q1)*cos(q3), - sin(q1)*cos(q3) + sin(q2)*sin(q3)*cos(q1),
         cos(q1)*cos(q2)]])

def test_w_diff_dcm1():
    # Ref:
    # Dynamics Theory and Applications, Kane 1985
    # Sec. 2.1 ANGULAR VELOCITY
    A = ReferenceFrame('A')
    B = ReferenceFrame('B')

    c11, c12, c13 = dynamicsymbols('C11 C12 C13')
    c21, c22, c23 = dynamicsymbols('C21 C22 C23')
    c31, c32, c33 = dynamicsymbols('C31 C32 C33')

    c11d, c12d, c13d = dynamicsymbols('C11 C12 C13', level=1)
    c21d, c22d, c23d = dynamicsymbols('C21 C22 C23', level=1)
    c31d, c32d, c33d = dynamicsymbols('C31 C32 C33', level=1)

    DCM = Matrix([
        [c11, c12, c13],
        [c21, c22, c23],
        [c31, c32, c33]
    ])

    B.orient(A, 'DCM', DCM)
    b1a = (B.x).express(A)
    b2a = (B.y).express(A)
    b3a = (B.z).express(A)

    # Equation (2.1.1)
    B.set_ang_vel(A, B.x*(dot((b3a).dt(A), B.y))
                   + B.y*(dot((b1a).dt(A), B.z))
                   + B.z*(dot((b2a).dt(A), B.x)))

    # Equation (2.1.21)
    expr = (  (c12*c13d + c22*c23d + c32*c33d)*B.x
            + (c13*c11d + c23*c21d + c33*c31d)*B.y
            + (c11*c12d + c21*c22d + c31*c32d)*B.z)
    assert B.ang_vel_in(A) - expr == 0

def test_w_diff_dcm2():
    q1, q2, q3 = dynamicsymbols('q1:4')
    N = ReferenceFrame('N')
    A = N.orientnew('A', 'axis', [q1, N.x])
    B = A.orientnew('B', 'axis', [q2, A.y])
    C = B.orientnew('C', 'axis', [q3, B.z])

    DCM = C.dcm(N).T
    D = N.orientnew('D', 'DCM', DCM)

    # Frames D and C are the same ReferenceFrame,
    # since they have equal DCM respect to frame N.
    # Therefore, D and C should have same angle velocity in N.
    assert D.dcm(N) == C.dcm(N) == Matrix([
        [cos(q2)*cos(q3), sin(q1)*sin(q2)*cos(q3) +
        sin(q3)*cos(q1), sin(q1)*sin(q3) -
        sin(q2)*cos(q1)*cos(q3)], [-sin(q3)*cos(q2),
        -sin(q1)*sin(q2)*sin(q3) + cos(q1)*cos(q3),
        sin(q1)*cos(q3) + sin(q2)*sin(q3)*cos(q1)],
        [sin(q2), -sin(q1)*cos(q2), cos(q1)*cos(q2)]])
    assert (D.ang_vel_in(N) - C.ang_vel_in(N)).express(N).simplify() == 0

def test_orientnew_respects_parent_class():
    class MyReferenceFrame(ReferenceFrame):
        pass
    B = MyReferenceFrame('B')
    C = B.orientnew('C', 'Axis', [0, B.x])
    assert isinstance(C, MyReferenceFrame)


def test_orientnew_respects_input_indices():
    N = ReferenceFrame('N')
    q1 = dynamicsymbols('q1')
    A = N.orientnew('a', 'Axis', [q1, N.z])
    #modify default indices:
    minds = [x+'1' for x in N.indices]
    B = N.orientnew('b', 'Axis', [q1, N.z], indices=minds)

    assert N.indices == A.indices
    assert B.indices == minds

def test_orientnew_respects_input_latexs():
    N = ReferenceFrame('N')
    q1 = dynamicsymbols('q1')
    A = N.orientnew('a', 'Axis', [q1, N.z])

    #build default and alternate latex_vecs:
    def_latex_vecs = [(r"\mathbf{\hat{%s}_%s}" % (A.name.lower(),
                      A.indices[0])), (r"\mathbf{\hat{%s}_%s}" %
                      (A.name.lower(), A.indices[1])),
                      (r"\mathbf{\hat{%s}_%s}" % (A.name.lower(),
                      A.indices[2]))]

    name = 'b'
    indices = [x+'1' for x in N.indices]
    new_latex_vecs = [(r"\mathbf{\hat{%s}_{%s}}" % (name.lower(),
                      indices[0])), (r"\mathbf{\hat{%s}_{%s}}" %
                      (name.lower(), indices[1])),
                      (r"\mathbf{\hat{%s}_{%s}}" % (name.lower(),
                      indices[2]))]

    B = N.orientnew(name, 'Axis', [q1, N.z], latexs=new_latex_vecs)

    assert A.latex_vecs == def_latex_vecs
    assert B.latex_vecs == new_latex_vecs
    assert B.indices != indices

def test_orientnew_respects_input_variables():
    N = ReferenceFrame('N')
    q1 = dynamicsymbols('q1')
    A = N.orientnew('a', 'Axis', [q1, N.z])

    #build non-standard variable names
    name = 'b'
    new_variables = ['notb_'+x+'1' for x in N.indices]
    B = N.orientnew(name, 'Axis', [q1, N.z], variables=new_variables)

    for j,var in enumerate(A.varlist):
        assert var.name == A.name + '_' + A.indices[j]

    for j,var in enumerate(B.varlist):
        assert var.name == new_variables[j]

def test_issue_10348():
    u = dynamicsymbols('u:3')
    I = ReferenceFrame('I')
    I.orientnew('A', 'space', u, 'XYZ')


def test_issue_11503():
    A = ReferenceFrame("A")
    A.orientnew("B", "Axis", [35, A.y])
    C = ReferenceFrame("C")
    A.orient(C, "Axis", [70, C.z])


def test_partial_velocity():

    N = ReferenceFrame('N')
    A = ReferenceFrame('A')

    u1, u2 = dynamicsymbols('u1, u2')

    A.set_ang_vel(N, u1 * A.x + u2 * N.y)

    assert N.partial_velocity(A, u1) == -A.x
    assert N.partial_velocity(A, u1, u2) == (-A.x, -N.y)

    assert A.partial_velocity(N, u1) == A.x
    assert A.partial_velocity(N, u1, u2) == (A.x, N.y)

    assert N.partial_velocity(N, u1) == 0
    assert A.partial_velocity(A, u1) == 0


def test_issue_11498():
    A = ReferenceFrame('A')
    B = ReferenceFrame('B')

    # Identity transformation
    A.orient(B, 'DCM', eye(3))
    assert A.dcm(B) == Matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    assert B.dcm(A) == Matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

    # x -> y
    # y -> -z
    # z -> -x
    A.orient(B, 'DCM', Matrix([[0, 1, 0], [0, 0, -1], [-1, 0, 0]]))
    assert B.dcm(A) == Matrix([[0, 1, 0], [0, 0, -1], [-1, 0, 0]])
    assert A.dcm(B) == Matrix([[0, 0, -1], [1, 0, 0], [0, -1, 0]])
    assert B.dcm(A).T == A.dcm(B)


def test_reference_frame():
    raises(TypeError, lambda: ReferenceFrame(0))
    raises(TypeError, lambda: ReferenceFrame('N', 0))
    raises(ValueError, lambda: ReferenceFrame('N', [0, 1]))
    raises(TypeError, lambda: ReferenceFrame('N', [0, 1, 2]))
    raises(TypeError, lambda: ReferenceFrame('N', ['a', 'b', 'c'], 0))
    raises(ValueError, lambda: ReferenceFrame('N', ['a', 'b', 'c'], [0, 1]))
    raises(TypeError, lambda: ReferenceFrame('N', ['a', 'b', 'c'], [0, 1, 2]))
    raises(TypeError, lambda: ReferenceFrame('N', ['a', 'b', 'c'],
                                                 ['a', 'b', 'c'], 0))
    raises(ValueError, lambda: ReferenceFrame('N', ['a', 'b', 'c'],
                                              ['a', 'b', 'c'], [0, 1]))
    raises(TypeError, lambda: ReferenceFrame('N', ['a', 'b', 'c'],
                                             ['a', 'b', 'c'], [0, 1, 2]))
    N = ReferenceFrame('N')
    assert N[0] == CoordinateSym('N_x', N, 0)
    assert N[1] == CoordinateSym('N_y', N, 1)
    assert N[2] == CoordinateSym('N_z', N, 2)
    raises(ValueError, lambda: N[3])
    N = ReferenceFrame('N', ['a', 'b', 'c'])
    assert N['a'] == N.x
    assert N['b'] == N.y
    assert N['c'] == N.z
    raises(ValueError, lambda: N['d'])
    assert str(N) == 'N'

    A = ReferenceFrame('A')
    B = ReferenceFrame('B')
    q0, q1, q2, q3 = symbols('q0 q1 q2 q3')
    raises(TypeError, lambda: A.orient(B, 'DCM', 0))
    raises(TypeError, lambda: B.orient(N, 'Space', [q1, q2, q3], '222'))
    raises(TypeError, lambda: B.orient(N, 'Axis', [q1, N.x + 2 * N.y], '222'))
    raises(TypeError, lambda: B.orient(N, 'Axis', q1))
    raises(IndexError, lambda: B.orient(N, 'Axis', [q1]))
    raises(TypeError, lambda: B.orient(N, 'Quaternion', [q0, q1, q2, q3], '222'))
    raises(TypeError, lambda: B.orient(N, 'Quaternion', q0))
    raises(TypeError, lambda: B.orient(N, 'Quaternion', [q0, q1, q2]))
    raises(NotImplementedError, lambda: B.orient(N, 'Foo', [q0, q1, q2]))
    raises(TypeError, lambda: B.orient(N, 'Body', [q1, q2], '232'))
    raises(TypeError, lambda: B.orient(N, 'Space', [q1, q2], '232'))

    N.set_ang_acc(B, 0)
    assert N.ang_acc_in(B) == Vector(0)
    N.set_ang_vel(B, 0)
    assert N.ang_vel_in(B) == Vector(0)


def test_check_frame():
    raises(VectorTypeError, lambda: _check_frame(0))


def test_dcm_diff_16824():
    # NOTE : This is a regression test for the bug introduced in PR 14758,
    # identified in 16824, and solved by PR 16828.

    # This is the solution to Problem 2.2 on page 264 in Kane & Lenvinson's
    # 1985 book.

    q1, q2, q3 = dynamicsymbols('q1:4')

    s1 = sin(q1)
    c1 = cos(q1)
    s2 = sin(q2)
    c2 = cos(q2)
    s3 = sin(q3)
    c3 = cos(q3)

    dcm = Matrix([[c2*c3, s1*s2*c3 - s3*c1, c1*s2*c3 + s3*s1],
                  [c2*s3, s1*s2*s3 + c3*c1, c1*s2*s3 - c3*s1],
                  [-s2,   s1*c2,            c1*c2]])

    A = ReferenceFrame('A')
    B = ReferenceFrame('B')
    B.orient(A, 'DCM', dcm)

    AwB = B.ang_vel_in(A)

    alpha2 = s3*c2*q1.diff() + c3*q2.diff()
    beta2 = s1*c2*q3.diff() + c1*q2.diff()

    assert simplify(AwB.dot(A.y) - alpha2) == 0
    assert simplify(AwB.dot(B.y) - beta2) == 0

def test_orient_explicit():
    cxx, cyy, czz = dynamicsymbols('c_{xx}, c_{yy}, c_{zz}')
    cxy, cxz, cyx = dynamicsymbols('c_{xy}, c_{xz}, c_{yx}')
    cyz, czx, czy = dynamicsymbols('c_{yz}, c_{zx}, c_{zy}')
    dcxx, dcyy, dczz = dynamicsymbols('c_{xx}, c_{yy}, c_{zz}', 1)
    dcxy, dcxz, dcyx = dynamicsymbols('c_{xy}, c_{xz}, c_{yx}', 1)
    dcyz, dczx, dczy = dynamicsymbols('c_{yz}, c_{zx}, c_{zy}', 1)
    A = ReferenceFrame('A')
    B = ReferenceFrame('B')
    B_C_A = Matrix([[cxx, cxy, cxz],
                    [cyx, cyy, cyz],
                    [czx, czy, czz]])
    B_w_A = ((cyx*dczx + cyy*dczy + cyz*dczz)*B.x +
            (czx*dcxx + czy*dcxy + czz*dcxz)*B.y +
            (cxx*dcyx + cxy*dcyy + cxz*dcyz)*B.z)
    A.orient_explicit(B, B_C_A)
    assert B.dcm(A) == B_C_A
    assert A.ang_vel_in(B) == B_w_A
    assert B.ang_vel_in(A) == -B_w_A

def test_orient_dcm():
    cxx, cyy, czz = dynamicsymbols('c_{xx}, c_{yy}, c_{zz}')
    cxy, cxz, cyx = dynamicsymbols('c_{xy}, c_{xz}, c_{yx}')
    cyz, czx, czy = dynamicsymbols('c_{yz}, c_{zx}, c_{zy}')
    B_C_A = Matrix([[cxx, cxy, cxz],
                    [cyx, cyy, cyz],
                    [czx, czy, czz]])
    A = ReferenceFrame('A')
    B = ReferenceFrame('B')
    B.orient_dcm(A, B_C_A)
    assert B.dcm(A) == Matrix([[cxx, cxy, cxz],
                               [cyx, cyy, cyz],
                               [czx, czy, czz]])

def test_orient_axis():
    A = ReferenceFrame('A')
    B = ReferenceFrame('B')
    A.orient_axis(B,-B.x, 1)
    A1 = A.dcm(B)
    A.orient_axis(B, B.x, -1)
    A2 = A.dcm(B)
    A.orient_axis(B, 1, -B.x)
    A3 = A.dcm(B)
    assert A1 == A2
    assert A2 == A3
    raises(TypeError, lambda: A.orient_axis(B, 1, 1))

def test_orient_body():
    A = ReferenceFrame('A')
    B = ReferenceFrame('B')
    B.orient_body_fixed(A, (1,1,0), 'XYX')
    assert B.dcm(A) == Matrix([[cos(1), sin(1)**2, -sin(1)*cos(1)], [0, cos(1), sin(1)], [sin(1), -sin(1)*cos(1), cos(1)**2]])


def test_orient_body_advanced():
    q1, q2, q3 = dynamicsymbols('q1:4')
    c1, c2, c3 = symbols('c1:4')
    u1, u2, u3 = dynamicsymbols('q1:4', 1)

    # Test with everything as dynamicsymbols
    A, B = ReferenceFrame('A'), ReferenceFrame('B')
    B.orient_body_fixed(A, (q1, q2, q3), 'zxy')
    assert A.dcm(B) == Matrix([
        [-sin(q1) * sin(q2) * sin(q3) + cos(q1) * cos(q3), -sin(q1) * cos(q2),
         sin(q1) * sin(q2) * cos(q3) + sin(q3) * cos(q1)],
        [sin(q1) * cos(q3) + sin(q2) * sin(q3) * cos(q1), cos(q1) * cos(q2),
         sin(q1) * sin(q3) - sin(q2) * cos(q1) * cos(q3)],
        [-sin(q3) * cos(q2), sin(q2), cos(q2) * cos(q3)]])
    assert B.ang_vel_in(A).to_matrix(B) == Matrix([
        [-sin(q3) * cos(q2) * u1 + cos(q3) * u2],
        [sin(q2) * u1 + u3],
        [sin(q3) * u2 + cos(q2) * cos(q3) * u1]])

    # Test with constant symbol
    A, B = ReferenceFrame('A'), ReferenceFrame('B')
    B.orient_body_fixed(A, (q1, c2, q3), 131)
    assert A.dcm(B) == Matrix([
        [cos(c2), -sin(c2) * cos(q3), sin(c2) * sin(q3)],
        [sin(c2) * cos(q1), -sin(q1) * sin(q3) + cos(c2) * cos(q1) * cos(q3),
         -sin(q1) * cos(q3) - sin(q3) * cos(c2) * cos(q1)],
        [sin(c2) * sin(q1), sin(q1) * cos(c2) * cos(q3) + sin(q3) * cos(q1),
         -sin(q1) * sin(q3) * cos(c2) + cos(q1) * cos(q3)]])
    assert B.ang_vel_in(A).to_matrix(B) == Matrix([
        [cos(c2) * u1 + u3],
        [-sin(c2) * cos(q3) * u1],
        [sin(c2) * sin(q3) * u1]])

    # Test all symbols not time dependent
    A, B = ReferenceFrame('A'), ReferenceFrame('B')
    B.orient_body_fixed(A, (c1, c2, c3), 123)
    assert B.ang_vel_in(A) == Vector(0)


def test_orient_space_advanced():
    # space fixed is in the end like body fixed only in opposite order
    q1, q2, q3 = dynamicsymbols('q1:4')
    c1, c2, c3 = symbols('c1:4')
    u1, u2, u3 = dynamicsymbols('q1:4', 1)

    # Test with everything as dynamicsymbols
    A, B = ReferenceFrame('A'), ReferenceFrame('B')
    B.orient_space_fixed(A, (q3, q2, q1), 'yxz')
    assert A.dcm(B) == Matrix([
        [-sin(q1) * sin(q2) * sin(q3) + cos(q1) * cos(q3), -sin(q1) * cos(q2),
         sin(q1) * sin(q2) * cos(q3) + sin(q3) * cos(q1)],
        [sin(q1) * cos(q3) + sin(q2) * sin(q3) * cos(q1), cos(q1) * cos(q2),
         sin(q1) * sin(q3) - sin(q2) * cos(q1) * cos(q3)],
        [-sin(q3) * cos(q2), sin(q2), cos(q2) * cos(q3)]])
    assert B.ang_vel_in(A).to_matrix(B) == Matrix([
        [-sin(q3) * cos(q2) * u1 + cos(q3) * u2],
        [sin(q2) * u1 + u3],
        [sin(q3) * u2 + cos(q2) * cos(q3) * u1]])

    # Test with constant symbol
    A, B = ReferenceFrame('A'), ReferenceFrame('B')
    B.orient_space_fixed(A, (q3, c2, q1), 131)
    assert A.dcm(B) == Matrix([
        [cos(c2), -sin(c2) * cos(q3), sin(c2) * sin(q3)],
        [sin(c2) * cos(q1), -sin(q1) * sin(q3) + cos(c2) * cos(q1) * cos(q3),
         -sin(q1) * cos(q3) - sin(q3) * cos(c2) * cos(q1)],
        [sin(c2) * sin(q1), sin(q1) * cos(c2) * cos(q3) + sin(q3) * cos(q1),
         -sin(q1) * sin(q3) * cos(c2) + cos(q1) * cos(q3)]])
    assert B.ang_vel_in(A).to_matrix(B) == Matrix([
        [cos(c2) * u1 + u3],
        [-sin(c2) * cos(q3) * u1],
        [sin(c2) * sin(q3) * u1]])

    # Test all symbols not time dependent
    A, B = ReferenceFrame('A'), ReferenceFrame('B')
    B.orient_space_fixed(A, (c1, c2, c3), 123)
    assert B.ang_vel_in(A) == Vector(0)


def test_orient_body_simple_ang_vel():
    """This test ensures that the simplest form of that linear system solution
    is returned, thus the == for the expression comparison."""

    psi, theta, phi = dynamicsymbols('psi, theta, varphi')
    t = dynamicsymbols._t
    A = ReferenceFrame('A')
    B = ReferenceFrame('B')
    B.orient_body_fixed(A, (psi, theta, phi), 'ZXZ')
    A_w_B = B.ang_vel_in(A)
    assert A_w_B.args[0][1] == B
    assert A_w_B.args[0][0][0] == (sin(theta)*sin(phi)*psi.diff(t) +
                                   cos(phi)*theta.diff(t))
    assert A_w_B.args[0][0][1] == (sin(theta)*cos(phi)*psi.diff(t) -
                                   sin(phi)*theta.diff(t))
    assert A_w_B.args[0][0][2] == cos(theta)*psi.diff(t) + phi.diff(t)


def test_orient_space():
    A = ReferenceFrame('A')
    B = ReferenceFrame('B')
    B.orient_space_fixed(A, (0,0,0), '123')
    assert B.dcm(A) == Matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

def test_orient_quaternion():
    A = ReferenceFrame('A')
    B = ReferenceFrame('B')
    B.orient_quaternion(A, (0,0,0,0))
    assert B.dcm(A) == Matrix([[0, 0, 0], [0, 0, 0], [0, 0, 0]])

def test_looped_frame_warning():
    A = ReferenceFrame('A')
    B = ReferenceFrame('B')
    C = ReferenceFrame('C')

    a, b, c = symbols('a b c')
    B.orient_axis(A, A.x, a)
    C.orient_axis(B, B.x, b)

    with warnings.catch_warnings(record = True) as w:
        warnings.simplefilter("always")
        A.orient_axis(C, C.x, c)
        assert issubclass(w[-1].category, UserWarning)
        assert 'Loops are defined among the orientation of frames. ' + \
            'This is likely not desired and may cause errors in your calculations.' in str(w[-1].message)

def test_frame_dict():
    A = ReferenceFrame('A')
    B = ReferenceFrame('B')
    C = ReferenceFrame('C')

    a, b, c = symbols('a b c')

    B.orient_axis(A, A.x, a)
    assert A._dcm_dict == {B: Matrix([[1, 0, 0],[0, cos(a), -sin(a)],[0, sin(a),  cos(a)]])}
    assert B._dcm_dict == {A: Matrix([[1, 0, 0],[0,  cos(a), sin(a)],[0, -sin(a), cos(a)]])}
    assert C._dcm_dict == {}

    B.orient_axis(C, C.x, b)
    # Previous relation is not wiped
    assert A._dcm_dict == {B: Matrix([[1, 0, 0],[0, cos(a), -sin(a)],[0, sin(a),  cos(a)]])}
    assert B._dcm_dict == {A: Matrix([[1, 0, 0],[0,  cos(a), sin(a)],[0, -sin(a), cos(a)]]), \
        C: Matrix([[1, 0, 0],[0,  cos(b), sin(b)],[0, -sin(b), cos(b)]])}
    assert C._dcm_dict == {B: Matrix([[1, 0, 0],[0, cos(b), -sin(b)],[0, sin(b),  cos(b)]])}

    A.orient_axis(B, B.x, c)
    # Previous relation is updated
    assert B._dcm_dict == {C: Matrix([[1, 0, 0],[0,  cos(b), sin(b)],[0, -sin(b), cos(b)]]),\
        A: Matrix([[1, 0, 0],[0, cos(c), -sin(c)],[0, sin(c),  cos(c)]])}
    assert A._dcm_dict == {B: Matrix([[1, 0, 0],[0,  cos(c), sin(c)],[0, -sin(c), cos(c)]])}
    assert C._dcm_dict == {B: Matrix([[1, 0, 0],[0, cos(b), -sin(b)],[0, sin(b),  cos(b)]])}

def test_dcm_cache_dict():
    A = ReferenceFrame('A')
    B = ReferenceFrame('B')
    C = ReferenceFrame('C')
    D = ReferenceFrame('D')

    a, b, c = symbols('a b c')

    B.orient_axis(A, A.x, a)
    C.orient_axis(B, B.x, b)
    D.orient_axis(C, C.x, c)

    assert D._dcm_dict == {C: Matrix([[1, 0, 0],[0,  cos(c), sin(c)],[0, -sin(c), cos(c)]])}
    assert C._dcm_dict == {B: Matrix([[1, 0, 0],[0,  cos(b), sin(b)],[0, -sin(b), cos(b)]]), \
        D: Matrix([[1, 0, 0],[0, cos(c), -sin(c)],[0, sin(c),  cos(c)]])}
    assert B._dcm_dict == {A: Matrix([[1, 0, 0],[0,  cos(a), sin(a)],[0, -sin(a), cos(a)]]), \
        C: Matrix([[1, 0, 0],[0, cos(b), -sin(b)],[0, sin(b),  cos(b)]])}
    assert A._dcm_dict == {B: Matrix([[1, 0, 0],[0, cos(a), -sin(a)],[0, sin(a),  cos(a)]])}

    assert D._dcm_dict == D._dcm_cache

    D.dcm(A) # Check calculated dcm relation is stored in _dcm_cache and not in _dcm_dict
    assert list(A._dcm_cache.keys()) == [A, B, D]
    assert list(D._dcm_cache.keys()) == [C, A]
    assert list(A._dcm_dict.keys()) == [B]
    assert list(D._dcm_dict.keys()) == [C]
    assert A._dcm_dict != A._dcm_cache

    A.orient_axis(B, B.x, b) # _dcm_cache of A is wiped out and new relation is stored.
    assert A._dcm_dict == {B: Matrix([[1, 0, 0],[0,  cos(b), sin(b)],[0, -sin(b), cos(b)]])}
    assert A._dcm_dict == A._dcm_cache
    assert B._dcm_dict == {C: Matrix([[1, 0, 0],[0, cos(b), -sin(b)],[0, sin(b),  cos(b)]]), \
        A: Matrix([[1, 0, 0],[0, cos(b), -sin(b)],[0, sin(b),  cos(b)]])}

def test_xx_dyad():
    N = ReferenceFrame('N')
    F = ReferenceFrame('F', indices=['1', '2', '3'])
    assert N.xx == Vector.outer(N.x, N.x)
    assert F.xx == Vector.outer(F.x, F.x)

def test_xy_dyad():
    N = ReferenceFrame('N')
    F = ReferenceFrame('F', indices=['1', '2', '3'])
    assert N.xy == Vector.outer(N.x, N.y)
    assert F.xy == Vector.outer(F.x, F.y)

def test_xz_dyad():
    N = ReferenceFrame('N')
    F = ReferenceFrame('F', indices=['1', '2', '3'])
    assert N.xz == Vector.outer(N.x, N.z)
    assert F.xz == Vector.outer(F.x, F.z)

def test_yx_dyad():
    N = ReferenceFrame('N')
    F = ReferenceFrame('F', indices=['1', '2', '3'])
    assert N.yx == Vector.outer(N.y, N.x)
    assert F.yx == Vector.outer(F.y, F.x)

def test_yy_dyad():
    N = ReferenceFrame('N')
    F = ReferenceFrame('F', indices=['1', '2', '3'])
    assert N.yy == Vector.outer(N.y, N.y)
    assert F.yy == Vector.outer(F.y, F.y)

def test_yz_dyad():
    N = ReferenceFrame('N')
    F = ReferenceFrame('F', indices=['1', '2', '3'])
    assert N.yz == Vector.outer(N.y, N.z)
    assert F.yz == Vector.outer(F.y, F.z)

def test_zx_dyad():
    N = ReferenceFrame('N')
    F = ReferenceFrame('F', indices=['1', '2', '3'])
    assert N.zx == Vector.outer(N.z, N.x)
    assert F.zx == Vector.outer(F.z, F.x)

def test_zy_dyad():
    N = ReferenceFrame('N')
    F = ReferenceFrame('F', indices=['1', '2', '3'])
    assert N.zy == Vector.outer(N.z, N.y)
    assert F.zy == Vector.outer(F.z, F.y)

def test_zz_dyad():
    N = ReferenceFrame('N')
    F = ReferenceFrame('F', indices=['1', '2', '3'])
    assert N.zz == Vector.outer(N.z, N.z)
    assert F.zz == Vector.outer(F.z, F.z)

def test_unit_dyadic():
    N = ReferenceFrame('N')
    F = ReferenceFrame('F', indices=['1', '2', '3'])
    assert N.u == N.xx + N.yy + N.zz
    assert F.u == F.xx + F.yy + F.zz


def test_pickle_frame():
    N = ReferenceFrame('N')
    A = ReferenceFrame('A')
    A.orient_axis(N, N.x, 1)
    A_C_N = A.dcm(N)
    N1 = pickle.loads(pickle.dumps(N))
    A1 = tuple(N1._dcm_dict.keys())[0]
    assert A1.dcm(N1) == A_C_N
