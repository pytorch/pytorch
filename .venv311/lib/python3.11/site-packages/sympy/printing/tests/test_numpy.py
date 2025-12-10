from sympy.concrete.summations import Sum
from sympy.core.mod import Mod
from sympy.core.relational import (Equality, Unequality)
from sympy.core.symbol import Symbol
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.piecewise import Piecewise
from sympy.functions.special.gamma_functions import polygamma
from sympy.functions.special.error_functions import (Si, Ci)
from sympy.matrices import Matrix
from sympy.matrices.expressions.blockmatrix import BlockMatrix
from sympy.matrices.expressions.matexpr import MatrixSymbol
from sympy.matrices.expressions.special import Identity
from sympy.utilities.lambdify import lambdify
from sympy import symbols, Min, Max

from sympy.abc import x, i, j, a, b, c, d
from sympy.core import Pow
from sympy.codegen.matrix_nodes import MatrixSolve
from sympy.codegen.numpy_nodes import logaddexp, logaddexp2
from sympy.codegen.cfunctions import log1p, expm1, hypot, log10, exp2, log2, Sqrt
from sympy.tensor.array import Array
from sympy.tensor.array.expressions.array_expressions import ArrayTensorProduct, ArrayAdd, \
    PermuteDims, ArrayDiagonal
from sympy.printing.numpy import NumPyPrinter, SciPyPrinter, _numpy_known_constants, \
    _numpy_known_functions, _scipy_known_constants, _scipy_known_functions
from sympy.tensor.array.expressions.from_matrix_to_array import convert_matrix_to_array

from sympy.testing.pytest import skip, raises
from sympy.external import import_module

np = import_module('numpy')
jax = import_module('jax')

if np:
    deafult_float_info = np.finfo(np.array([]).dtype)
    NUMPY_DEFAULT_EPSILON = deafult_float_info.eps

def test_numpy_piecewise_regression():
    """
    NumPyPrinter needs to print Piecewise()'s choicelist as a list to avoid
    breaking compatibility with numpy 1.8. This is not necessary in numpy 1.9+.
    See gh-9747 and gh-9749 for details.
    """
    printer = NumPyPrinter()
    p = Piecewise((1, x < 0), (0, True))
    assert printer.doprint(p) == \
        'numpy.select([numpy.less(x, 0),True], [1,0], default=numpy.nan)'
    assert printer.module_imports == {'numpy': {'select', 'less', 'nan'}}

def test_numpy_logaddexp():
    lae = logaddexp(a, b)
    assert NumPyPrinter().doprint(lae) == 'numpy.logaddexp(a, b)'
    lae2 = logaddexp2(a, b)
    assert NumPyPrinter().doprint(lae2) == 'numpy.logaddexp2(a, b)'


def test_sum():
    if not np:
        skip("NumPy not installed")

    s = Sum(x ** i, (i, a, b))
    f = lambdify((a, b, x), s, 'numpy')

    a_, b_ = 0, 10
    x_ = np.linspace(-1, +1, 10)
    assert np.allclose(f(a_, b_, x_), sum(x_ ** i_ for i_ in range(a_, b_ + 1)))

    s = Sum(i * x, (i, a, b))
    f = lambdify((a, b, x), s, 'numpy')

    a_, b_ = 0, 10
    x_ = np.linspace(-1, +1, 10)
    assert np.allclose(f(a_, b_, x_), sum(i_ * x_ for i_ in range(a_, b_ + 1)))


def test_multiple_sums():
    if not np:
        skip("NumPy not installed")

    s = Sum((x + j) * i, (i, a, b), (j, c, d))
    f = lambdify((a, b, c, d, x), s, 'numpy')

    a_, b_ = 0, 10
    c_, d_ = 11, 21
    x_ = np.linspace(-1, +1, 10)
    assert np.allclose(f(a_, b_, c_, d_, x_),
                       sum((x_ + j_) * i_ for i_ in range(a_, b_ + 1) for j_ in range(c_, d_ + 1)))


def test_codegen_einsum():
    if not np:
        skip("NumPy not installed")

    M = MatrixSymbol("M", 2, 2)
    N = MatrixSymbol("N", 2, 2)

    cg = convert_matrix_to_array(M * N)
    f = lambdify((M, N), cg, 'numpy')

    ma = np.array([[1, 2], [3, 4]])
    mb = np.array([[1,-2], [-1, 3]])
    assert (f(ma, mb) == np.matmul(ma, mb)).all()


def test_codegen_extra():
    if not np:
        skip("NumPy not installed")

    M = MatrixSymbol("M", 2, 2)
    N = MatrixSymbol("N", 2, 2)
    P = MatrixSymbol("P", 2, 2)
    Q = MatrixSymbol("Q", 2, 2)
    ma = np.array([[1, 2], [3, 4]])
    mb = np.array([[1,-2], [-1, 3]])
    mc = np.array([[2, 0], [1, 2]])
    md = np.array([[1,-1], [4, 7]])

    cg = ArrayTensorProduct(M, N)
    f = lambdify((M, N), cg, 'numpy')
    assert (f(ma, mb) == np.einsum(ma, [0, 1], mb, [2, 3])).all()

    cg = ArrayAdd(M, N)
    f = lambdify((M, N), cg, 'numpy')
    assert (f(ma, mb) == ma+mb).all()

    cg = ArrayAdd(M, N, P)
    f = lambdify((M, N, P), cg, 'numpy')
    assert (f(ma, mb, mc) == ma+mb+mc).all()

    cg = ArrayAdd(M, N, P, Q)
    f = lambdify((M, N, P, Q), cg, 'numpy')
    assert (f(ma, mb, mc, md) == ma+mb+mc+md).all()

    cg = PermuteDims(M, [1, 0])
    f = lambdify((M,), cg, 'numpy')
    assert (f(ma) == ma.T).all()

    cg = PermuteDims(ArrayTensorProduct(M, N), [1, 2, 3, 0])
    f = lambdify((M, N), cg, 'numpy')
    assert (f(ma, mb) == np.transpose(np.einsum(ma, [0, 1], mb, [2, 3]), (1, 2, 3, 0))).all()

    cg = ArrayDiagonal(ArrayTensorProduct(M, N), (1, 2))
    f = lambdify((M, N), cg, 'numpy')
    assert (f(ma, mb) == np.diagonal(np.einsum(ma, [0, 1], mb, [2, 3]), axis1=1, axis2=2)).all()


def test_relational():
    if not np:
        skip("NumPy not installed")

    e = Equality(x, 1)

    f = lambdify((x,), e)
    x_ = np.array([0, 1, 2])
    assert np.array_equal(f(x_), [False, True, False])

    e = Unequality(x, 1)

    f = lambdify((x,), e)
    x_ = np.array([0, 1, 2])
    assert np.array_equal(f(x_), [True, False, True])

    e = (x < 1)

    f = lambdify((x,), e)
    x_ = np.array([0, 1, 2])
    assert np.array_equal(f(x_), [True, False, False])

    e = (x <= 1)

    f = lambdify((x,), e)
    x_ = np.array([0, 1, 2])
    assert np.array_equal(f(x_), [True, True, False])

    e = (x > 1)

    f = lambdify((x,), e)
    x_ = np.array([0, 1, 2])
    assert np.array_equal(f(x_), [False, False, True])

    e = (x >= 1)

    f = lambdify((x,), e)
    x_ = np.array([0, 1, 2])
    assert np.array_equal(f(x_), [False, True, True])


def test_mod():
    if not np:
        skip("NumPy not installed")

    e = Mod(a, b)
    f = lambdify((a, b), e)

    a_ = np.array([0, 1, 2, 3])
    b_ = 2
    assert np.array_equal(f(a_, b_), [0, 1, 0, 1])

    a_ = np.array([0, 1, 2, 3])
    b_ = np.array([2, 2, 2, 2])
    assert np.array_equal(f(a_, b_), [0, 1, 0, 1])

    a_ = np.array([2, 3, 4, 5])
    b_ = np.array([2, 3, 4, 5])
    assert np.array_equal(f(a_, b_), [0, 0, 0, 0])


def test_pow():
    if not np:
        skip('NumPy not installed')

    expr = Pow(2, -1, evaluate=False)
    f = lambdify([], expr, 'numpy')
    assert f() == 0.5


def test_expm1():
    if not np:
        skip("NumPy not installed")

    f = lambdify((a,), expm1(a), 'numpy')
    assert abs(f(1e-10) - 1e-10 - 5e-21) <= 1e-10 * NUMPY_DEFAULT_EPSILON


def test_log1p():
    if not np:
        skip("NumPy not installed")

    f = lambdify((a,), log1p(a), 'numpy')
    assert abs(f(1e-99) - 1e-99) <= 1e-99 * NUMPY_DEFAULT_EPSILON

def test_hypot():
    if not np:
        skip("NumPy not installed")
    assert abs(lambdify((a, b), hypot(a, b), 'numpy')(3, 4) - 5) <= NUMPY_DEFAULT_EPSILON

def test_log10():
    if not np:
        skip("NumPy not installed")
    assert abs(lambdify((a,), log10(a), 'numpy')(100) - 2) <= NUMPY_DEFAULT_EPSILON


def test_exp2():
    if not np:
        skip("NumPy not installed")
    assert abs(lambdify((a,), exp2(a), 'numpy')(5) - 32) <= NUMPY_DEFAULT_EPSILON


def test_log2():
    if not np:
        skip("NumPy not installed")
    assert abs(lambdify((a,), log2(a), 'numpy')(256) - 8) <= NUMPY_DEFAULT_EPSILON


def test_Sqrt():
    if not np:
        skip("NumPy not installed")
    assert abs(lambdify((a,), Sqrt(a), 'numpy')(4) - 2) <= NUMPY_DEFAULT_EPSILON


def test_sqrt():
    if not np:
        skip("NumPy not installed")
    assert abs(lambdify((a,), sqrt(a), 'numpy')(4) - 2) <= NUMPY_DEFAULT_EPSILON


def test_matsolve():
    if not np:
        skip("NumPy not installed")

    M = MatrixSymbol("M", 3, 3)
    x = MatrixSymbol("x", 3, 1)

    expr = M**(-1) * x + x
    matsolve_expr = MatrixSolve(M, x) + x

    f = lambdify((M, x), expr)
    f_matsolve = lambdify((M, x), matsolve_expr)

    m0 = np.array([[1, 2, 3], [3, 2, 5], [5, 6, 7]])
    assert np.linalg.matrix_rank(m0) == 3

    x0 = np.array([3, 4, 5])

    assert np.allclose(f_matsolve(m0, x0), f(m0, x0))


def test_16857():
    if not np:
        skip("NumPy not installed")

    a_1 = MatrixSymbol('a_1', 10, 3)
    a_2 = MatrixSymbol('a_2', 10, 3)
    a_3 = MatrixSymbol('a_3', 10, 3)
    a_4 = MatrixSymbol('a_4', 10, 3)
    A = BlockMatrix([[a_1, a_2], [a_3, a_4]])
    assert A.shape == (20, 6)

    printer = NumPyPrinter()
    assert printer.doprint(A) == 'numpy.block([[a_1, a_2], [a_3, a_4]])'


def test_issue_17006():
    if not np:
        skip("NumPy not installed")

    M = MatrixSymbol("M", 2, 2)

    f = lambdify(M, M + Identity(2))
    ma = np.array([[1, 2], [3, 4]])
    mr = np.array([[2, 2], [3, 5]])

    assert (f(ma) == mr).all()

    from sympy.core.symbol import symbols
    n = symbols('n', integer=True)
    N = MatrixSymbol("M", n, n)
    raises(NotImplementedError, lambda: lambdify(N, N + Identity(n)))

def test_jax_tuple_compatibility():
    if not jax:
        skip("Jax not installed")

    x, y, z = symbols('x y z')
    expr = Max(x, y, z) + Min(x, y, z)
    func = lambdify((x, y, z), expr, 'jax')
    input_tuple1, input_tuple2 = (1, 2, 3), (4, 5, 6)
    input_array1, input_array2 = jax.numpy.asarray(input_tuple1), jax.numpy.asarray(input_tuple2)
    assert np.allclose(func(*input_tuple1), func(*input_array1))
    assert np.allclose(func(*input_tuple2), func(*input_array2))

def test_numpy_array():
    p = NumPyPrinter()
    assert p.doprint(Array([[1, 2], [3, 5]])) == 'numpy.array([[1, 2], [3, 5]])'
    assert p.doprint(Array([1, 2])) == 'numpy.array([1, 2])'
    assert p.doprint(Array([[[1, 2, 3]]])) == 'numpy.array([[[1, 2, 3]]])'
    assert p.doprint(Array([], (0,))) == 'numpy.zeros((0,))'
    assert p.doprint(Array([], (0, 0))) == 'numpy.zeros((0, 0))'
    assert p.doprint(Array([], (0, 1))) == 'numpy.zeros((0, 1))'
    assert p.doprint(Array([], (1, 0))) == 'numpy.zeros((1, 0))'
    assert p.doprint(Array([1], ())) == 'numpy.array(1)'

def test_numpy_matrix():
    p = NumPyPrinter()
    assert p.doprint(Matrix([[1, 2], [3, 5]])) == 'numpy.array([[1, 2], [3, 5]])'
    assert p.doprint(Matrix([1, 2])) == 'numpy.array([[1], [2]])'
    assert p.doprint(Matrix(0, 0, [])) == 'numpy.zeros((0, 0))'
    assert p.doprint(Matrix(0, 1, [])) == 'numpy.zeros((0, 1))'
    assert p.doprint(Matrix(1, 0, [])) == 'numpy.zeros((1, 0))'

def test_numpy_known_funcs_consts():
    assert _numpy_known_constants['NaN'] == 'numpy.nan'
    assert _numpy_known_constants['EulerGamma'] == 'numpy.euler_gamma'

    assert _numpy_known_functions['acos'] == 'numpy.arccos'
    assert _numpy_known_functions['log'] == 'numpy.log'

def test_scipy_known_funcs_consts():
    assert _scipy_known_constants['GoldenRatio'] == 'scipy.constants.golden_ratio'
    assert _scipy_known_constants['Pi'] == 'scipy.constants.pi'

    assert _scipy_known_functions['erf'] == 'scipy.special.erf'
    assert _scipy_known_functions['factorial'] == 'scipy.special.factorial'

def test_numpy_print_methods():
    prntr = NumPyPrinter()
    assert hasattr(prntr, '_print_acos')
    assert hasattr(prntr, '_print_log')

def test_scipy_print_methods():
    prntr = SciPyPrinter()
    assert hasattr(prntr, '_print_acos')
    assert hasattr(prntr, '_print_log')
    assert hasattr(prntr, '_print_erf')
    assert hasattr(prntr, '_print_factorial')
    assert hasattr(prntr, '_print_chebyshevt')
    k = Symbol('k', integer=True, nonnegative=True)
    x = Symbol('x', real=True)
    assert prntr.doprint(polygamma(k, x)) == "scipy.special.polygamma(k, x)"
    assert prntr.doprint(Si(x)) == "scipy.special.sici(x)[0]"
    assert prntr.doprint(Ci(x)) == "scipy.special.sici(x)[1]"
