from sympy.core import (S, pi, oo, symbols, Function, Rational, Integer,
                        Tuple, Symbol, Eq, Ne, Le, Lt, Gt, Ge)
from sympy.core import EulerGamma, GoldenRatio, Catalan, Lambda, Mul, Pow
from sympy.functions import Piecewise, sqrt, ceiling, exp, sin, cos, sinc, lucas
from sympy.testing.pytest import raises
from sympy.utilities.lambdify import implemented_function
from sympy.matrices import (eye, Matrix, MatrixSymbol, Identity,
                            HadamardProduct, SparseMatrix)
from sympy.functions.special.bessel import besseli

from sympy.printing.maple import maple_code

x, y, z = symbols('x,y,z')


def test_Integer():
    assert maple_code(Integer(67)) == "67"
    assert maple_code(Integer(-1)) == "-1"


def test_Rational():
    assert maple_code(Rational(3, 7)) == "3/7"
    assert maple_code(Rational(18, 9)) == "2"
    assert maple_code(Rational(3, -7)) == "-3/7"
    assert maple_code(Rational(-3, -7)) == "3/7"
    assert maple_code(x + Rational(3, 7)) == "x + 3/7"
    assert maple_code(Rational(3, 7) * x) == '(3/7)*x'


def test_Relational():
    assert maple_code(Eq(x, y)) == "x = y"
    assert maple_code(Ne(x, y)) == "x <> y"
    assert maple_code(Le(x, y)) == "x <= y"
    assert maple_code(Lt(x, y)) == "x < y"
    assert maple_code(Gt(x, y)) == "x > y"
    assert maple_code(Ge(x, y)) == "x >= y"


def test_Function():
    assert maple_code(sin(x) ** cos(x)) == "sin(x)^cos(x)"
    assert maple_code(abs(x)) == "abs(x)"
    assert maple_code(ceiling(x)) == "ceil(x)"


def test_Pow():
    assert maple_code(x ** 3) == "x^3"
    assert maple_code(x ** (y ** 3)) == "x^(y^3)"

    assert maple_code((x ** 3) ** y) == "(x^3)^y"
    assert maple_code(x ** Rational(2, 3)) == 'x^(2/3)'

    g = implemented_function('g', Lambda(x, 2 * x))
    assert maple_code(1 / (g(x) * 3.5) ** (x - y ** x) / (x ** 2 + y)) == \
           "(3.5*2*x)^(-x + y^x)/(x^2 + y)"
    # For issue 14160
    assert maple_code(Mul(-2, x, Pow(Mul(y, y, evaluate=False), -1, evaluate=False),
                          evaluate=False)) == '-2*x/(y*y)'


def test_basic_ops():
    assert maple_code(x * y) == "x*y"
    assert maple_code(x + y) == "x + y"
    assert maple_code(x - y) == "x - y"
    assert maple_code(-x) == "-x"


def test_1_over_x_and_sqrt():
    # 1.0 and 0.5 would do something different in regular StrPrinter,
    # but these are exact in IEEE floating point so no different here.
    assert maple_code(1 / x) == '1/x'
    assert maple_code(x ** -1) == maple_code(x ** -1.0) == '1/x'
    assert maple_code(1 / sqrt(x)) == '1/sqrt(x)'
    assert maple_code(x ** -S.Half) == maple_code(x ** -0.5) == '1/sqrt(x)'
    assert maple_code(sqrt(x)) == 'sqrt(x)'
    assert maple_code(x ** S.Half) == maple_code(x ** 0.5) == 'sqrt(x)'
    assert maple_code(1 / pi) == '1/Pi'
    assert maple_code(pi ** -1) == maple_code(pi ** -1.0) == '1/Pi'
    assert maple_code(pi ** -0.5) == '1/sqrt(Pi)'


def test_mix_number_mult_symbols():
    assert maple_code(3 * x) == "3*x"
    assert maple_code(pi * x) == "Pi*x"
    assert maple_code(3 / x) == "3/x"
    assert maple_code(pi / x) == "Pi/x"
    assert maple_code(x / 3) == '(1/3)*x'
    assert maple_code(x / pi) == "x/Pi"
    assert maple_code(x * y) == "x*y"
    assert maple_code(3 * x * y) == "3*x*y"
    assert maple_code(3 * pi * x * y) == "3*Pi*x*y"
    assert maple_code(x / y) == "x/y"
    assert maple_code(3 * x / y) == "3*x/y"
    assert maple_code(x * y / z) == "x*y/z"
    assert maple_code(x / y * z) == "x*z/y"
    assert maple_code(1 / x / y) == "1/(x*y)"
    assert maple_code(2 * pi * x / y / z) == "2*Pi*x/(y*z)"
    assert maple_code(3 * pi / x) == "3*Pi/x"
    assert maple_code(S(3) / 5) == "3/5"
    assert maple_code(S(3) / 5 * x) == '(3/5)*x'
    assert maple_code(x / y / z) == "x/(y*z)"
    assert maple_code((x + y) / z) == "(x + y)/z"
    assert maple_code((x + y) / (z + x)) == "(x + y)/(x + z)"
    assert maple_code((x + y) / EulerGamma) == '(x + y)/gamma'
    assert maple_code(x / 3 / pi) == '(1/3)*x/Pi'
    assert maple_code(S(3) / 5 * x * y / pi) == '(3/5)*x*y/Pi'


def test_mix_number_pow_symbols():
    assert maple_code(pi ** 3) == 'Pi^3'
    assert maple_code(x ** 2) == 'x^2'

    assert maple_code(x ** (pi ** 3)) == 'x^(Pi^3)'
    assert maple_code(x ** y) == 'x^y'

    assert maple_code(x ** (y ** z)) == 'x^(y^z)'
    assert maple_code((x ** y) ** z) == '(x^y)^z'


def test_imag():
    I = S('I')
    assert maple_code(I) == "I"
    assert maple_code(5 * I) == "5*I"

    assert maple_code((S(3) / 2) * I) == "(3/2)*I"
    assert maple_code(3 + 4 * I) == "3 + 4*I"


def test_constants():
    assert maple_code(pi) == "Pi"
    assert maple_code(oo) == "infinity"
    assert maple_code(-oo) == "-infinity"
    assert maple_code(S.NegativeInfinity) == "-infinity"
    assert maple_code(S.NaN) == "undefined"
    assert maple_code(S.Exp1) == "exp(1)"
    assert maple_code(exp(1)) == "exp(1)"


def test_constants_other():
    assert maple_code(2 * GoldenRatio) == '2*(1/2 + (1/2)*sqrt(5))'
    assert maple_code(2 * Catalan) == '2*Catalan'
    assert maple_code(2 * EulerGamma) == "2*gamma"


def test_boolean():
    assert maple_code(x & y) == "x and y"
    assert maple_code(x | y) == "x or y"
    assert maple_code(~x) == "not x"
    assert maple_code(x & y & z) == "x and y and z"
    assert maple_code(x | y | z) == "x or y or z"
    assert maple_code((x & y) | z) == "z or x and y"
    assert maple_code((x | y) & z) == "z and (x or y)"


def test_Matrices():
    assert maple_code(Matrix(1, 1, [10])) == \
           'Matrix([[10]], storage = rectangular)'

    A = Matrix([[1, sin(x / 2), abs(x)],
                [0, 1, pi],
                [0, exp(1), ceiling(x)]])
    expected = \
        'Matrix(' \
        '[[1, sin((1/2)*x), abs(x)],' \
        ' [0, 1, Pi],' \
        ' [0, exp(1), ceil(x)]], ' \
        'storage = rectangular)'
    assert maple_code(A) == expected

    # row and columns
    assert maple_code(A[:, 0]) == \
           'Matrix([[1], [0], [0]], storage = rectangular)'
    assert maple_code(A[0, :]) == \
           'Matrix([[1, sin((1/2)*x), abs(x)]], storage = rectangular)'
    assert maple_code(Matrix([[x, x - y, -y]])) == \
           'Matrix([[x, x - y, -y]], storage = rectangular)'

    # empty matrices
    assert maple_code(Matrix(0, 0, [])) == \
           'Matrix([], storage = rectangular)'
    assert maple_code(Matrix(0, 3, [])) == \
           'Matrix([], storage = rectangular)'

def test_SparseMatrices():
    assert maple_code(SparseMatrix(Identity(2))) == 'Matrix([[1, 0], [0, 1]], storage = sparse)'


def test_vector_entries_hadamard():
    # For a row or column, user might to use the other dimension
    A = Matrix([[1, sin(2 / x), 3 * pi / x / 5]])
    assert maple_code(A) == \
           'Matrix([[1, sin(2/x), (3/5)*Pi/x]], storage = rectangular)'
    assert maple_code(A.T) == \
           'Matrix([[1], [sin(2/x)], [(3/5)*Pi/x]], storage = rectangular)'


def test_Matrices_entries_not_hadamard():
    A = Matrix([[1, sin(2 / x), 3 * pi / x / 5], [1, 2, x * y]])
    expected = \
        'Matrix([[1, sin(2/x), (3/5)*Pi/x], [1, 2, x*y]], ' \
        'storage = rectangular)'
    assert maple_code(A) == expected


def test_MatrixSymbol():
    n = Symbol('n', integer=True)
    A = MatrixSymbol('A', n, n)
    B = MatrixSymbol('B', n, n)
    assert maple_code(A * B) == "A.B"
    assert maple_code(B * A) == "B.A"
    assert maple_code(2 * A * B) == "2*A.B"
    assert maple_code(B * 2 * A) == "2*B.A"

    assert maple_code(
        A * (B + 3 * Identity(n))) == "A.(3*Matrix(n, shape = identity) + B)"

    assert maple_code(A ** (x ** 2)) == "MatrixPower(A, x^2)"
    assert maple_code(A ** 3) == "MatrixPower(A, 3)"
    assert maple_code(A ** (S.Half)) == "MatrixPower(A, 1/2)"


def test_special_matrices():
    assert maple_code(6 * Identity(3)) == "6*Matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]], storage = sparse)"
    assert maple_code(Identity(x)) == 'Matrix(x, shape = identity)'


def test_containers():
    assert maple_code([1, 2, 3, [4, 5, [6, 7]], 8, [9, 10], 11]) == \
           "[1, 2, 3, [4, 5, [6, 7]], 8, [9, 10], 11]"

    assert maple_code((1, 2, (3, 4))) == "[1, 2, [3, 4]]"
    assert maple_code([1]) == "[1]"
    assert maple_code((1,)) == "[1]"
    assert maple_code(Tuple(*[1, 2, 3])) == "[1, 2, 3]"
    assert maple_code((1, x * y, (3, x ** 2))) == "[1, x*y, [3, x^2]]"
    # scalar, matrix, empty matrix and empty list

    assert maple_code((1, eye(3), Matrix(0, 0, []), [])) == \
           "[1, Matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]], storage = rectangular), Matrix([], storage = rectangular), []]"


def test_maple_noninline():
    source = maple_code((x + y)/Catalan, assign_to='me', inline=False)
    expected = "me := (x + y)/Catalan"

    assert source == expected


def test_maple_matrix_assign_to():
    A = Matrix([[1, 2, 3]])
    assert maple_code(A, assign_to='a') == "a := Matrix([[1, 2, 3]], storage = rectangular)"
    A = Matrix([[1, 2], [3, 4]])
    assert maple_code(A, assign_to='A') == "A := Matrix([[1, 2], [3, 4]], storage = rectangular)"


def test_maple_matrix_assign_to_more():
    # assigning to Symbol or MatrixSymbol requires lhs/rhs match
    A = Matrix([[1, 2, 3]])
    B = MatrixSymbol('B', 1, 3)
    C = MatrixSymbol('C', 2, 3)
    assert maple_code(A, assign_to=B) == "B := Matrix([[1, 2, 3]], storage = rectangular)"
    raises(ValueError, lambda: maple_code(A, assign_to=x))
    raises(ValueError, lambda: maple_code(A, assign_to=C))


def test_maple_matrix_1x1():
    A = Matrix([[3]])
    assert maple_code(A, assign_to='B') == "B := Matrix([[3]], storage = rectangular)"


def test_maple_matrix_elements():
    A = Matrix([[x, 2, x * y]])

    assert maple_code(A[0, 0] ** 2 + A[0, 1] + A[0, 2]) == "x^2 + x*y + 2"
    AA = MatrixSymbol('AA', 1, 3)
    assert maple_code(AA) == "AA"

    assert maple_code(AA[0, 0] ** 2 + sin(AA[0, 1]) + AA[0, 2]) == \
           "sin(AA[1, 2]) + AA[1, 1]^2 + AA[1, 3]"
    assert maple_code(sum(AA)) == "AA[1, 1] + AA[1, 2] + AA[1, 3]"


def test_maple_boolean():
    assert maple_code(True) == "true"
    assert maple_code(S.true) == "true"
    assert maple_code(False) == "false"
    assert maple_code(S.false) == "false"


def test_sparse():
    M = SparseMatrix(5, 6, {})
    M[2, 2] = 10
    M[1, 2] = 20
    M[1, 3] = 22
    M[0, 3] = 30
    M[3, 0] = x * y
    assert maple_code(M) == \
           'Matrix([[0, 0, 0, 30, 0, 0],' \
           ' [0, 0, 20, 22, 0, 0],' \
           ' [0, 0, 10, 0, 0, 0],' \
           ' [x*y, 0, 0, 0, 0, 0],' \
           ' [0, 0, 0, 0, 0, 0]], ' \
           'storage = sparse)'

# Not an important point.
def test_maple_not_supported():
    with raises(NotImplementedError):
        maple_code(S.ComplexInfinity)


def test_MatrixElement_printing():
    # test cases for issue #11821
    A = MatrixSymbol("A", 1, 3)
    B = MatrixSymbol("B", 1, 3)

    assert (maple_code(A[0, 0]) == "A[1, 1]")
    assert (maple_code(3 * A[0, 0]) == "3*A[1, 1]")

    F = A-B

    assert (maple_code(F[0,0]) == "A[1, 1] - B[1, 1]")


def test_hadamard():
    A = MatrixSymbol('A', 3, 3)
    B = MatrixSymbol('B', 3, 3)
    v = MatrixSymbol('v', 3, 1)
    h = MatrixSymbol('h', 1, 3)
    C = HadamardProduct(A, B)
    assert maple_code(C) == "A*B"

    assert maple_code(C * v) == "(A*B).v"
    # HadamardProduct is higher than dot product.

    assert maple_code(h * C * v) == "h.(A*B).v"

    assert maple_code(C * A) == "(A*B).A"
    # mixing Hadamard and scalar strange b/c we vectorize scalars

    assert maple_code(C * x * y) == "x*y*(A*B)"


def test_maple_piecewise():
    expr = Piecewise((x, x < 1), (x ** 2, True))

    assert maple_code(expr) == "piecewise(x < 1, x, x^2)"
    assert maple_code(expr, assign_to="r") == (
        "r := piecewise(x < 1, x, x^2)")

    expr = Piecewise((x ** 2, x < 1), (x ** 3, x < 2), (x ** 4, x < 3), (x ** 5, True))
    expected = "piecewise(x < 1, x^2, x < 2, x^3, x < 3, x^4, x^5)"
    assert maple_code(expr) == expected
    assert maple_code(expr, assign_to="r") == "r := " + expected

    # Check that Piecewise without a True (default) condition error
    expr = Piecewise((x, x < 1), (x ** 2, x > 1), (sin(x), x > 0))
    raises(ValueError, lambda: maple_code(expr))


def test_maple_piecewise_times_const():
    pw = Piecewise((x, x < 1), (x ** 2, True))

    assert maple_code(2 * pw) == "2*piecewise(x < 1, x, x^2)"
    assert maple_code(pw / x) == "piecewise(x < 1, x, x^2)/x"
    assert maple_code(pw / (x * y)) == "piecewise(x < 1, x, x^2)/(x*y)"
    assert maple_code(pw / 3) == "(1/3)*piecewise(x < 1, x, x^2)"


def test_maple_derivatives():
    f = Function('f')
    assert maple_code(f(x).diff(x)) == 'diff(f(x), x)'
    assert maple_code(f(x).diff(x, 2)) == 'diff(f(x), x$2)'


def test_automatic_rewrites():
    assert maple_code(lucas(x)) == '(2^(-x)*((1 - sqrt(5))^x + (1 + sqrt(5))^x))'
    assert maple_code(sinc(x)) == '(piecewise(x <> 0, sin(x)/x, 1))'


def test_specfun():
    assert maple_code('asin(x)') == 'arcsin(x)'
    assert maple_code(besseli(x, y)) == 'BesselI(x, y)'
