import pytest
from sympy.core.function import expand_mul
from sympy.core.numbers import (I, Rational)
from sympy.core.singleton import S
from sympy.core.symbol import (Symbol, symbols)
from sympy.core.sympify import sympify
from sympy.simplify.simplify import simplify
from sympy.matrices.exceptions import (ShapeError, NonSquareMatrixError)
from sympy.matrices import (
    ImmutableMatrix, Matrix, eye, ones, ImmutableDenseMatrix, dotprodsimp)
from sympy.matrices.determinant import _det_laplace
from sympy.testing.pytest import raises
from sympy.matrices.exceptions import NonInvertibleMatrixError
from sympy.polys.matrices.exceptions import DMShapeError
from sympy.solvers.solveset import linsolve
from sympy.abc import x, y

def test_issue_17247_expression_blowup_29():
    M = Matrix(S('''[
        [             -3/4,       45/32 - 37*I/16,                   0,                     0],
        [-149/64 + 49*I/32, -177/128 - 1369*I/128,                   0, -2063/256 + 541*I/128],
        [                0,         9/4 + 55*I/16, 2473/256 + 137*I/64,                     0],
        [                0,                     0,                   0, -177/128 - 1369*I/128]]'''))
    with dotprodsimp(True):
        assert M.gauss_jordan_solve(ones(4, 1)) == (Matrix(S('''[
            [                          -32549314808672/3306971225785 - 17397006745216*I/3306971225785],
            [                               67439348256/3306971225785 - 9167503335872*I/3306971225785],
            [-15091965363354518272/21217636514687010905 + 16890163109293858304*I/21217636514687010905],
            [                                                          -11328/952745 + 87616*I/952745]]''')), Matrix(0, 1, []))

def test_issue_17247_expression_blowup_30():
    M = Matrix(S('''[
        [             -3/4,       45/32 - 37*I/16,                   0,                     0],
        [-149/64 + 49*I/32, -177/128 - 1369*I/128,                   0, -2063/256 + 541*I/128],
        [                0,         9/4 + 55*I/16, 2473/256 + 137*I/64,                     0],
        [                0,                     0,                   0, -177/128 - 1369*I/128]]'''))
    with dotprodsimp(True):
        assert M.cholesky_solve(ones(4, 1)) == Matrix(S('''[
            [                          -32549314808672/3306971225785 - 17397006745216*I/3306971225785],
            [                               67439348256/3306971225785 - 9167503335872*I/3306971225785],
            [-15091965363354518272/21217636514687010905 + 16890163109293858304*I/21217636514687010905],
            [                                                          -11328/952745 + 87616*I/952745]]'''))

# @XFAIL # This calculation hangs with dotprodsimp.
# def test_issue_17247_expression_blowup_31():
#     M = Matrix([
#         [x + 1, 1 - x,     0,     0],
#         [1 - x, x + 1,     0, x + 1],
#         [    0, 1 - x, x + 1,     0],
#         [    0,     0,     0, x + 1]])
#     with dotprodsimp(True):
#         assert M.LDLsolve(ones(4, 1)) == Matrix([
#             [(x + 1)/(4*x)],
#             [(x - 1)/(4*x)],
#             [(x + 1)/(4*x)],
#             [    1/(x + 1)]])


def test_LUsolve_iszerofunc():
    # taken from https://github.com/sympy/sympy/issues/24679

    M = Matrix([[(x + 1)**2 - (x**2 + 2*x + 1), x], [x, 0]])
    b = Matrix([1, 1])
    is_zero_func = lambda e: False if e._random() else True

    x_exp = Matrix([1/x, (1-(-x**2 - 2*x + (x+1)**2 - 1)/x)/x])

    assert (x_exp - M.LUsolve(b, iszerofunc=is_zero_func)) == Matrix([0, 0])


def test_issue_17247_expression_blowup_32():
    M = Matrix([
        [x + 1, 1 - x,     0,     0],
        [1 - x, x + 1,     0, x + 1],
        [    0, 1 - x, x + 1,     0],
        [    0,     0,     0, x + 1]])
    with dotprodsimp(True):
        assert M.LUsolve(ones(4, 1)) == Matrix([
            [(x + 1)/(4*x)],
            [(x - 1)/(4*x)],
            [(x + 1)/(4*x)],
            [    1/(x + 1)]])

def test_LUsolve():
    A = Matrix([[2, 3, 5],
                [3, 6, 2],
                [8, 3, 6]])
    x = Matrix(3, 1, [3, 7, 5])
    b = A*x
    soln = A.LUsolve(b)
    assert soln == x
    A = Matrix([[0, -1, 2],
                [5, 10, 7],
                [8,  3, 4]])
    x = Matrix(3, 1, [-1, 2, 5])
    b = A*x
    soln = A.LUsolve(b)
    assert soln == x
    A = Matrix([[2, 1], [1, 0], [1, 0]])   # issue 14548
    b = Matrix([3, 1, 1])
    assert A.LUsolve(b) == Matrix([1, 1])
    b = Matrix([3, 1, 2])                  # inconsistent
    raises(ValueError, lambda: A.LUsolve(b))
    A = Matrix([[0, -1, 2],
                [5, 10, 7],
                [8,  3, 4],
                [2, 3, 5],
                [3, 6, 2],
                [8, 3, 6]])
    x = Matrix([2, 1, -4])
    b = A*x
    soln = A.LUsolve(b)
    assert soln == x
    A = Matrix([[0, -1, 2], [5, 10, 7]])  # underdetermined
    x = Matrix([-1, 2, 0])
    b = A*x
    raises(NotImplementedError, lambda: A.LUsolve(b))

    A = Matrix(4, 4, lambda i, j: 1/(i+j+1) if i != 3 else 0)
    b = Matrix.zeros(4, 1)
    raises(NonInvertibleMatrixError, lambda: A.LUsolve(b))


def test_QRsolve():
    A = Matrix([[2, 3, 5],
                [3, 6, 2],
                [8, 3, 6]])
    x = Matrix(3, 1, [3, 7, 5])
    b = A*x
    soln = A.QRsolve(b)
    assert soln == x
    x = Matrix([[1, 2], [3, 4], [5, 6]])
    b = A*x
    soln = A.QRsolve(b)
    assert soln == x

    A = Matrix([[0, -1, 2],
                [5, 10, 7],
                [8,  3, 4]])
    x = Matrix(3, 1, [-1, 2, 5])
    b = A*x
    soln = A.QRsolve(b)
    assert soln == x
    x = Matrix([[7, 8], [9, 10], [11, 12]])
    b = A*x
    soln = A.QRsolve(b)
    assert soln == x

def test_errors():
    raises(ShapeError, lambda: Matrix([1]).LUsolve(Matrix([[1, 2], [3, 4]])))

def test_cholesky_solve():
    A = Matrix([[2, 3, 5],
                [3, 6, 2],
                [8, 3, 6]])
    x = Matrix(3, 1, [3, 7, 5])
    b = A*x
    soln = A.cholesky_solve(b)
    assert soln == x
    A = Matrix([[0, -1, 2],
                [5, 10, 7],
                [8,  3, 4]])
    x = Matrix(3, 1, [-1, 2, 5])
    b = A*x
    soln = A.cholesky_solve(b)
    assert soln == x
    A = Matrix(((1, 5), (5, 1)))
    x = Matrix((4, -3))
    b = A*x
    soln = A.cholesky_solve(b)
    assert soln == x
    A = Matrix(((9, 3*I), (-3*I, 5)))
    x = Matrix((-2, 1))
    b = A*x
    soln = A.cholesky_solve(b)
    assert expand_mul(soln) == x
    A = Matrix(((9*I, 3), (-3 + I, 5)))
    x = Matrix((2 + 3*I, -1))
    b = A*x
    soln = A.cholesky_solve(b)
    assert expand_mul(soln) == x
    a00, a01, a11, b0, b1 = symbols('a00, a01, a11, b0, b1')
    A = Matrix(((a00, a01), (a01, a11)))
    b = Matrix((b0, b1))
    x = A.cholesky_solve(b)
    assert simplify(A*x) == b


def test_LDLsolve():
    A = Matrix([[2, 3, 5],
                [3, 6, 2],
                [8, 3, 6]])
    x = Matrix(3, 1, [3, 7, 5])
    b = A*x
    soln = A.LDLsolve(b)
    assert soln == x

    A = Matrix([[0, -1, 2],
                [5, 10, 7],
                [8,  3, 4]])
    x = Matrix(3, 1, [-1, 2, 5])
    b = A*x
    soln = A.LDLsolve(b)
    assert soln == x

    A = Matrix(((9, 3*I), (-3*I, 5)))
    x = Matrix((-2, 1))
    b = A*x
    soln = A.LDLsolve(b)
    assert expand_mul(soln) == x

    A = Matrix(((9*I, 3), (-3 + I, 5)))
    x = Matrix((2 + 3*I, -1))
    b = A*x
    soln = A.LDLsolve(b)
    assert expand_mul(soln) == x

    A = Matrix(((9, 3), (3, 9)))
    x = Matrix((1, 1))
    b = A * x
    soln = A.LDLsolve(b)
    assert expand_mul(soln) == x

    A = Matrix([[-5, -3, -4], [-3, -7, 7]])
    x = Matrix([[8], [7], [-2]])
    b = A * x
    raises(NotImplementedError, lambda: A.LDLsolve(b))


def test_lower_triangular_solve():

    raises(NonSquareMatrixError,
        lambda: Matrix([1, 0]).lower_triangular_solve(Matrix([0, 1])))
    raises(ShapeError,
        lambda: Matrix([[1, 0], [0, 1]]).lower_triangular_solve(Matrix([1])))
    raises(ValueError,
        lambda: Matrix([[2, 1], [1, 2]]).lower_triangular_solve(
            Matrix([[1, 0], [0, 1]])))

    A = Matrix([[1, 0], [0, 1]])
    B = Matrix([[x, y], [y, x]])
    C = Matrix([[4, 8], [2, 9]])

    assert A.lower_triangular_solve(B) == B
    assert A.lower_triangular_solve(C) == C


def test_upper_triangular_solve():

    raises(NonSquareMatrixError,
        lambda: Matrix([1, 0]).upper_triangular_solve(Matrix([0, 1])))
    raises(ShapeError,
        lambda: Matrix([[1, 0], [0, 1]]).upper_triangular_solve(Matrix([1])))
    raises(TypeError,
        lambda: Matrix([[2, 1], [1, 2]]).upper_triangular_solve(
            Matrix([[1, 0], [0, 1]])))

    A = Matrix([[1, 0], [0, 1]])
    B = Matrix([[x, y], [y, x]])
    C = Matrix([[2, 4], [3, 8]])

    assert A.upper_triangular_solve(B) == B
    assert A.upper_triangular_solve(C) == C


def test_diagonal_solve():
    raises(TypeError, lambda: Matrix([1, 1]).diagonal_solve(Matrix([1])))
    A = Matrix([[1, 0], [0, 1]])*2
    B = Matrix([[x, y], [y, x]])
    assert A.diagonal_solve(B) == B/2

    A = Matrix([[1, 0], [1, 2]])
    raises(TypeError, lambda: A.diagonal_solve(B))

def test_pinv_solve():
    # Fully determined system (unique result, identical to other solvers).
    A = Matrix([[1, 5], [7, 9]])
    B = Matrix([12, 13])
    assert A.pinv_solve(B) == A.cholesky_solve(B)
    assert A.pinv_solve(B) == A.LDLsolve(B)
    assert A.pinv_solve(B) == Matrix([sympify('-43/26'), sympify('71/26')])
    assert A * A.pinv() * B == B
    # Fully determined, with two-dimensional B matrix.
    B = Matrix([[12, 13, 14], [15, 16, 17]])
    assert A.pinv_solve(B) == A.cholesky_solve(B)
    assert A.pinv_solve(B) == A.LDLsolve(B)
    assert A.pinv_solve(B) == Matrix([[-33, -37, -41], [69, 75, 81]]) / 26
    assert A * A.pinv() * B == B
    # Underdetermined system (infinite results).
    A = Matrix([[1, 0, 1], [0, 1, 1]])
    B = Matrix([5, 7])
    solution = A.pinv_solve(B)
    w = {}
    for s in solution.atoms(Symbol):
        # Extract dummy symbols used in the solution.
        w[s.name] = s
    assert solution == Matrix([[w['w0_0']/3 + w['w1_0']/3 - w['w2_0']/3 + 1],
                               [w['w0_0']/3 + w['w1_0']/3 - w['w2_0']/3 + 3],
                               [-w['w0_0']/3 - w['w1_0']/3 + w['w2_0']/3 + 4]])
    assert A * A.pinv() * B == B
    # Overdetermined system (least squares results).
    A = Matrix([[1, 0], [0, 0], [0, 1]])
    B = Matrix([3, 2, 1])
    assert A.pinv_solve(B) == Matrix([3, 1])
    # Proof the solution is not exact.
    assert A * A.pinv() * B != B

def test_pinv_rank_deficient():
    # Test the four properties of the pseudoinverse for various matrices.
    As = [Matrix([[1, 1, 1], [2, 2, 2]]),
          Matrix([[1, 0], [0, 0]]),
          Matrix([[1, 2], [2, 4], [3, 6]])]

    for A in As:
        A_pinv = A.pinv(method="RD")
        AAp = A * A_pinv
        ApA = A_pinv * A
        assert simplify(AAp * A) == A
        assert simplify(ApA * A_pinv) == A_pinv
        assert AAp.H == AAp
        assert ApA.H == ApA

    for A in As:
        A_pinv = A.pinv(method="ED")
        AAp = A * A_pinv
        ApA = A_pinv * A
        assert simplify(AAp * A) == A
        assert simplify(ApA * A_pinv) == A_pinv
        assert AAp.H == AAp
        assert ApA.H == ApA

    # Test solving with rank-deficient matrices.
    A = Matrix([[1, 0], [0, 0]])
    # Exact, non-unique solution.
    B = Matrix([3, 0])
    solution = A.pinv_solve(B)
    w1 = solution.atoms(Symbol).pop()
    assert w1.name == 'w1_0'
    assert solution == Matrix([3, w1])
    assert A * A.pinv() * B == B
    # Least squares, non-unique solution.
    B = Matrix([3, 1])
    solution = A.pinv_solve(B)
    w1 = solution.atoms(Symbol).pop()
    assert w1.name == 'w1_0'
    assert solution == Matrix([3, w1])
    assert A * A.pinv() * B != B

def test_gauss_jordan_solve():

    # Square, full rank, unique solution
    A = Matrix([[1, 2, 3], [4, 5, 6], [7, 8, 10]])
    b = Matrix([3, 6, 9])
    sol, params = A.gauss_jordan_solve(b)
    assert sol == Matrix([[-1], [2], [0]])
    assert params == Matrix(0, 1, [])

    # Square, full rank, unique solution, B has more columns than rows
    A = eye(3)
    B = Matrix([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
    sol, params = A.gauss_jordan_solve(B)
    assert sol == B
    assert params == Matrix(0, 4, [])

    # Square, reduced rank, parametrized solution
    A = Matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    b = Matrix([3, 6, 9])
    sol, params, freevar = A.gauss_jordan_solve(b, freevar=True)
    w = {}
    for s in sol.atoms(Symbol):
        # Extract dummy symbols used in the solution.
        w[s.name] = s
    assert sol == Matrix([[w['tau0'] - 1], [-2*w['tau0'] + 2], [w['tau0']]])
    assert params == Matrix([[w['tau0']]])
    assert freevar == [2]

    # Square, reduced rank, parametrized solution, B has two columns
    A = Matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    B = Matrix([[3, 4], [6, 8], [9, 12]])
    sol, params, freevar = A.gauss_jordan_solve(B, freevar=True)
    w = {}
    for s in sol.atoms(Symbol):
        # Extract dummy symbols used in the solution.
        w[s.name] = s
    assert sol == Matrix([[w['tau0'] - 1, w['tau1'] - Rational(4, 3)],
                          [-2*w['tau0'] + 2, -2*w['tau1'] + Rational(8, 3)],
                          [w['tau0'], w['tau1']],])
    assert params == Matrix([[w['tau0'], w['tau1']]])
    assert freevar == [2]

    # Square, reduced rank, parametrized solution
    A = Matrix([[1, 2, 3], [2, 4, 6], [3, 6, 9]])
    b = Matrix([0, 0, 0])
    sol, params = A.gauss_jordan_solve(b)
    w = {}
    for s in sol.atoms(Symbol):
        w[s.name] = s
    assert sol == Matrix([[-2*w['tau0'] - 3*w['tau1']],
                         [w['tau0']], [w['tau1']]])
    assert params == Matrix([[w['tau0']], [w['tau1']]])

    # Square, reduced rank, parametrized solution
    A = Matrix([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
    b = Matrix([0, 0, 0])
    sol, params = A.gauss_jordan_solve(b)
    w = {}
    for s in sol.atoms(Symbol):
        w[s.name] = s
    assert sol == Matrix([[w['tau0']], [w['tau1']], [w['tau2']]])
    assert params == Matrix([[w['tau0']], [w['tau1']], [w['tau2']]])

    # Square, reduced rank, no solution
    A = Matrix([[1, 2, 3], [2, 4, 6], [3, 6, 9]])
    b = Matrix([0, 0, 1])
    raises(ValueError, lambda: A.gauss_jordan_solve(b))

    # Rectangular, tall, full rank, unique solution
    A = Matrix([[1, 5, 3], [2, 1, 6], [1, 7, 9], [1, 4, 3]])
    b = Matrix([0, 0, 1, 0])
    sol, params = A.gauss_jordan_solve(b)
    assert sol == Matrix([[Rational(-1, 2)], [0], [Rational(1, 6)]])
    assert params == Matrix(0, 1, [])

    # Rectangular, tall, full rank, unique solution, B has less columns than rows
    A = Matrix([[1, 5, 3], [2, 1, 6], [1, 7, 9], [1, 4, 3]])
    B = Matrix([[0,0], [0, 0], [1, 2], [0, 0]])
    sol, params = A.gauss_jordan_solve(B)
    assert sol == Matrix([[Rational(-1, 2), Rational(-2, 2)], [0, 0], [Rational(1, 6), Rational(2, 6)]])
    assert params == Matrix(0, 2, [])

    # Rectangular, tall, full rank, no solution
    A = Matrix([[1, 5, 3], [2, 1, 6], [1, 7, 9], [1, 4, 3]])
    b = Matrix([0, 0, 0, 1])
    raises(ValueError, lambda: A.gauss_jordan_solve(b))

    # Rectangular, tall, full rank, no solution, B has two columns (2nd has no solution)
    A = Matrix([[1, 5, 3], [2, 1, 6], [1, 7, 9], [1, 4, 3]])
    B = Matrix([[0,0], [0, 0], [1, 0], [0, 1]])
    raises(ValueError, lambda: A.gauss_jordan_solve(B))

    # Rectangular, tall, full rank, no solution, B has two columns (1st has no solution)
    A = Matrix([[1, 5, 3], [2, 1, 6], [1, 7, 9], [1, 4, 3]])
    B = Matrix([[0,0], [0, 0], [0, 1], [1, 0]])
    raises(ValueError, lambda: A.gauss_jordan_solve(B))

    # Rectangular, tall, reduced rank, parametrized solution
    A = Matrix([[1, 5, 3], [2, 10, 6], [3, 15, 9], [1, 4, 3]])
    b = Matrix([0, 0, 0, 1])
    sol, params = A.gauss_jordan_solve(b)
    w = {}
    for s in sol.atoms(Symbol):
        w[s.name] = s
    assert sol == Matrix([[-3*w['tau0'] + 5], [-1], [w['tau0']]])
    assert params == Matrix([[w['tau0']]])

    # Rectangular, tall, reduced rank, no solution
    A = Matrix([[1, 5, 3], [2, 10, 6], [3, 15, 9], [1, 4, 3]])
    b = Matrix([0, 0, 1, 1])
    raises(ValueError, lambda: A.gauss_jordan_solve(b))

    # Rectangular, wide, full rank, parametrized solution
    A = Matrix([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 1, 12]])
    b = Matrix([1, 1, 1])
    sol, params = A.gauss_jordan_solve(b)
    w = {}
    for s in sol.atoms(Symbol):
        w[s.name] = s
    assert sol == Matrix([[2*w['tau0'] - 1], [-3*w['tau0'] + 1], [0],
                         [w['tau0']]])
    assert params == Matrix([[w['tau0']]])

    # Rectangular, wide, reduced rank, parametrized solution
    A = Matrix([[1, 2, 3, 4], [5, 6, 7, 8], [2, 4, 6, 8]])
    b = Matrix([0, 1, 0])
    sol, params = A.gauss_jordan_solve(b)
    w = {}
    for s in sol.atoms(Symbol):
        w[s.name] = s
    assert sol == Matrix([[w['tau0'] + 2*w['tau1'] + S.Half],
                         [-2*w['tau0'] - 3*w['tau1'] - Rational(1, 4)],
                         [w['tau0']], [w['tau1']]])
    assert params == Matrix([[w['tau0']], [w['tau1']]])
    # watch out for clashing symbols
    x0, x1, x2, _x0 = symbols('_tau0 _tau1 _tau2 tau1')
    M = Matrix([[0, 1, 0, 0, 0, 0], [0, 0, 0, 1, 0, _x0]])
    A = M[:, :-1]
    b = M[:, -1:]
    sol, params = A.gauss_jordan_solve(b)
    assert params == Matrix(3, 1, [x0, x1, x2])
    assert sol == Matrix(5, 1, [x0, 0, x1, _x0, x2])

    # Rectangular, wide, reduced rank, no solution
    A = Matrix([[1, 2, 3, 4], [5, 6, 7, 8], [2, 4, 6, 8]])
    b = Matrix([1, 1, 1])
    raises(ValueError, lambda: A.gauss_jordan_solve(b))

    # Test for immutable matrix
    A = ImmutableMatrix([[1, 0], [0, 1]])
    B = ImmutableMatrix([1, 2])
    sol, params = A.gauss_jordan_solve(B)
    assert sol == ImmutableMatrix([1, 2])
    assert params == ImmutableMatrix(0, 1, [])
    assert sol.__class__ == ImmutableDenseMatrix
    assert params.__class__ == ImmutableDenseMatrix

    # Test placement of free variables
    A = Matrix([[1, 0, 0, 0], [0, 0, 0, 1]])
    b = Matrix([1, 1])
    sol, params = A.gauss_jordan_solve(b)
    w = {}
    for s in sol.atoms(Symbol):
        w[s.name] = s
    assert sol == Matrix([[1], [w['tau0']], [w['tau1']], [1]])
    assert params == Matrix([[w['tau0']], [w['tau1']]])


def test_linsolve_underdetermined_AND_gauss_jordan_solve():
    #Test placement of free variables as per issue 19815
    A = Matrix([[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1]])
    B  = Matrix([1, 2, 1, 1, 1, 1, 1, 2])
    sol, params = A.gauss_jordan_solve(B)
    w = {}
    for s in sol.atoms(Symbol):
        w[s.name] = s
    assert params == Matrix([[w['tau0']], [w['tau1']], [w['tau2']],
                             [w['tau3']], [w['tau4']], [w['tau5']]])
    assert sol == Matrix([[1 - 1*w['tau2']],
                          [w['tau2']],
                          [1 - 1*w['tau0'] + w['tau1']],
                          [w['tau0']],
                          [w['tau3'] + w['tau4']],
                          [-1*w['tau3'] - 1*w['tau4'] - 1*w['tau1']],
                          [1 - 1*w['tau2']],
                          [w['tau1']],
                          [w['tau2']],
                          [w['tau3']],
                          [w['tau4']],
                          [1 - 1*w['tau5']],
                          [w['tau5']],
                          [1]])

    from sympy.abc import j,f
    # https://github.com/sympy/sympy/issues/20046
    A = Matrix([
    [1,  1, 1,  1, 1,  1, 1,  1,  1],
    [0, -1, 0, -1, 0, -1, 0, -1, -j],
    [0,  0, 0,  0, 1,  1, 1,  1,  f]
    ])

    sol_1=Matrix(list(linsolve(A))[0])

    tau0, tau1, tau2, tau3, tau4 = symbols('tau:5')

    assert sol_1 == Matrix([[-f - j - tau0 + tau2 + tau4 + 1],
                          [j - tau1 - tau2 - tau4],
                          [tau0],
                          [tau1],
                          [f - tau2 - tau3 - tau4],
                          [tau2],
                          [tau3],
                          [tau4]])

    # https://github.com/sympy/sympy/issues/19815
    sol_2 = A[:, : -1 ] * sol_1 - A[:, -1 ]
    assert sol_2 == Matrix([[0], [0], [0]])


@pytest.mark.parametrize("det_method", ["bird", "laplace"])
@pytest.mark.parametrize("M, rhs", [
    (Matrix([[2, 3, 5], [3, 6, 2], [8, 3, 6]]), Matrix(3, 1, [3, 7, 5])),
    (Matrix([[2, 3, 5], [3, 6, 2], [8, 3, 6]]),
     Matrix([[1, 2], [3, 4], [5, 6]])),
    (Matrix(2, 2, symbols("a:4")), Matrix(2, 1, symbols("b:2"))),
])
def test_cramer_solve(det_method, M, rhs):
    assert simplify(M.cramer_solve(rhs, det_method=det_method) - M.LUsolve(rhs)
                    ) == Matrix.zeros(M.rows, rhs.cols)


@pytest.mark.parametrize("det_method, error", [
    ("bird", DMShapeError), (_det_laplace, NonSquareMatrixError)])
def test_cramer_solve_errors(det_method, error):
    # Non-square matrix
    A = Matrix([[0, -1, 2], [5, 10, 7]])
    b = Matrix([-2, 15])
    raises(error, lambda: A.cramer_solve(b, det_method=det_method))


def test_solve():
    A = Matrix([[1,2], [2,4]])
    b = Matrix([[3], [4]])
    raises(ValueError, lambda: A.solve(b)) #no solution
    b = Matrix([[ 4], [8]])
    raises(ValueError, lambda: A.solve(b)) #infinite solution
