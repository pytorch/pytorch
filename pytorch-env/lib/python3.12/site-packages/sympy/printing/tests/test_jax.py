from sympy.concrete.summations import Sum
from sympy.core.mod import Mod
from sympy.core.relational import (Equality, Unequality)
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.piecewise import Piecewise
from sympy.matrices.expressions.blockmatrix import BlockMatrix
from sympy.matrices.expressions.matexpr import MatrixSymbol
from sympy.matrices.expressions.special import Identity
from sympy.utilities.lambdify import lambdify

from sympy.abc import x, i, j, a, b, c, d
from sympy.core import Function, Pow, Symbol
from sympy.codegen.matrix_nodes import MatrixSolve
from sympy.codegen.numpy_nodes import logaddexp, logaddexp2
from sympy.codegen.cfunctions import log1p, expm1, hypot, log10, exp2, log2, Sqrt
from sympy.tensor.array import Array
from sympy.tensor.array.expressions.array_expressions import ArrayTensorProduct, ArrayAdd, \
    PermuteDims, ArrayDiagonal
from sympy.printing.numpy import JaxPrinter, _jax_known_constants, _jax_known_functions
from sympy.tensor.array.expressions.from_matrix_to_array import convert_matrix_to_array

from sympy.testing.pytest import skip, raises
from sympy.external import import_module

# Unlike NumPy which will aggressively promote operands to double precision,
# jax always uses single precision. Double precision in jax can be
# configured before the call to `import jax`, however this must be explicitly
# configured and is not fully supported. Thus, the tests here have been modified
# from the tests in test_numpy.py, only in the fact that they assert lambdify
# function accuracy to only single precision accuracy.
# https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html#double-64bit-precision

jax = import_module('jax')

if jax:
    deafult_float_info = jax.numpy.finfo(jax.numpy.array([]).dtype)
    JAX_DEFAULT_EPSILON = deafult_float_info.eps


def test_jax_piecewise_regression():
    """
    NumPyPrinter needs to print Piecewise()'s choicelist as a list to avoid
    breaking compatibility with numpy 1.8. This is not necessary in numpy 1.9+.
    See gh-9747 and gh-9749 for details.
    """
    printer = JaxPrinter()
    p = Piecewise((1, x < 0), (0, True))
    assert printer.doprint(p) == \
        'jax.numpy.select([jax.numpy.less(x, 0),True], [1,0], default=jax.numpy.nan)'
    assert printer.module_imports == {'jax.numpy': {'select', 'less', 'nan'}}


def test_jax_logaddexp():
    lae = logaddexp(a, b)
    assert JaxPrinter().doprint(lae) == 'jax.numpy.logaddexp(a, b)'
    lae2 = logaddexp2(a, b)
    assert JaxPrinter().doprint(lae2) == 'jax.numpy.logaddexp2(a, b)'


def test_jax_sum():
    if not jax:
        skip("JAX not installed")

    s = Sum(x ** i, (i, a, b))
    f = lambdify((a, b, x), s, 'jax')

    a_, b_ = 0, 10
    x_ = jax.numpy.linspace(-1, +1, 10)
    assert jax.numpy.allclose(f(a_, b_, x_), sum(x_ ** i_ for i_ in range(a_, b_ + 1)))

    s = Sum(i * x, (i, a, b))
    f = lambdify((a, b, x), s, 'jax')

    a_, b_ = 0, 10
    x_ = jax.numpy.linspace(-1, +1, 10)
    assert jax.numpy.allclose(f(a_, b_, x_), sum(i_ * x_ for i_ in range(a_, b_ + 1)))


def test_jax_multiple_sums():
    if not jax:
        skip("JAX not installed")

    s = Sum((x + j) * i, (i, a, b), (j, c, d))
    f = lambdify((a, b, c, d, x), s, 'jax')

    a_, b_ = 0, 10
    c_, d_ = 11, 21
    x_ = jax.numpy.linspace(-1, +1, 10)
    assert jax.numpy.allclose(f(a_, b_, c_, d_, x_),
                       sum((x_ + j_) * i_ for i_ in range(a_, b_ + 1) for j_ in range(c_, d_ + 1)))


def test_jax_codegen_einsum():
    if not jax:
        skip("JAX not installed")

    M = MatrixSymbol("M", 2, 2)
    N = MatrixSymbol("N", 2, 2)

    cg = convert_matrix_to_array(M * N)
    f = lambdify((M, N), cg, 'jax')

    ma = jax.numpy.array([[1, 2], [3, 4]])
    mb = jax.numpy.array([[1,-2], [-1, 3]])
    assert (f(ma, mb) == jax.numpy.matmul(ma, mb)).all()


def test_jax_codegen_extra():
    if not jax:
        skip("JAX not installed")

    M = MatrixSymbol("M", 2, 2)
    N = MatrixSymbol("N", 2, 2)
    P = MatrixSymbol("P", 2, 2)
    Q = MatrixSymbol("Q", 2, 2)
    ma = jax.numpy.array([[1, 2], [3, 4]])
    mb = jax.numpy.array([[1,-2], [-1, 3]])
    mc = jax.numpy.array([[2, 0], [1, 2]])
    md = jax.numpy.array([[1,-1], [4, 7]])

    cg = ArrayTensorProduct(M, N)
    f = lambdify((M, N), cg, 'jax')
    assert (f(ma, mb) == jax.numpy.einsum(ma, [0, 1], mb, [2, 3])).all()

    cg = ArrayAdd(M, N)
    f = lambdify((M, N), cg, 'jax')
    assert (f(ma, mb) == ma+mb).all()

    cg = ArrayAdd(M, N, P)
    f = lambdify((M, N, P), cg, 'jax')
    assert (f(ma, mb, mc) == ma+mb+mc).all()

    cg = ArrayAdd(M, N, P, Q)
    f = lambdify((M, N, P, Q), cg, 'jax')
    assert (f(ma, mb, mc, md) == ma+mb+mc+md).all()

    cg = PermuteDims(M, [1, 0])
    f = lambdify((M,), cg, 'jax')
    assert (f(ma) == ma.T).all()

    cg = PermuteDims(ArrayTensorProduct(M, N), [1, 2, 3, 0])
    f = lambdify((M, N), cg, 'jax')
    assert (f(ma, mb) == jax.numpy.transpose(jax.numpy.einsum(ma, [0, 1], mb, [2, 3]), (1, 2, 3, 0))).all()

    cg = ArrayDiagonal(ArrayTensorProduct(M, N), (1, 2))
    f = lambdify((M, N), cg, 'jax')
    assert (f(ma, mb) == jax.numpy.diagonal(jax.numpy.einsum(ma, [0, 1], mb, [2, 3]), axis1=1, axis2=2)).all()


def test_jax_relational():
    if not jax:
        skip("JAX not installed")

    e = Equality(x, 1)

    f = lambdify((x,), e, 'jax')
    x_ = jax.numpy.array([0, 1, 2])
    assert jax.numpy.array_equal(f(x_), [False, True, False])

    e = Unequality(x, 1)

    f = lambdify((x,), e, 'jax')
    x_ = jax.numpy.array([0, 1, 2])
    assert jax.numpy.array_equal(f(x_), [True, False, True])

    e = (x < 1)

    f = lambdify((x,), e, 'jax')
    x_ = jax.numpy.array([0, 1, 2])
    assert jax.numpy.array_equal(f(x_), [True, False, False])

    e = (x <= 1)

    f = lambdify((x,), e, 'jax')
    x_ = jax.numpy.array([0, 1, 2])
    assert jax.numpy.array_equal(f(x_), [True, True, False])

    e = (x > 1)

    f = lambdify((x,), e, 'jax')
    x_ = jax.numpy.array([0, 1, 2])
    assert jax.numpy.array_equal(f(x_), [False, False, True])

    e = (x >= 1)

    f = lambdify((x,), e, 'jax')
    x_ = jax.numpy.array([0, 1, 2])
    assert jax.numpy.array_equal(f(x_), [False, True, True])

    # Multi-condition expressions
    e = (x >= 1) & (x < 2)
    f = lambdify((x,), e, 'jax')
    x_ = jax.numpy.array([0, 1, 2])
    assert jax.numpy.array_equal(f(x_), [False, True, False])

    e = (x >= 1) | (x < 2)
    f = lambdify((x,), e, 'jax')
    x_ = jax.numpy.array([0, 1, 2])
    assert jax.numpy.array_equal(f(x_), [True, True, True])

def test_jax_mod():
    if not jax:
        skip("JAX not installed")

    e = Mod(a, b)
    f = lambdify((a, b), e, 'jax')

    a_ = jax.numpy.array([0, 1, 2, 3])
    b_ = 2
    assert jax.numpy.array_equal(f(a_, b_), [0, 1, 0, 1])

    a_ = jax.numpy.array([0, 1, 2, 3])
    b_ = jax.numpy.array([2, 2, 2, 2])
    assert jax.numpy.array_equal(f(a_, b_), [0, 1, 0, 1])

    a_ = jax.numpy.array([2, 3, 4, 5])
    b_ = jax.numpy.array([2, 3, 4, 5])
    assert jax.numpy.array_equal(f(a_, b_), [0, 0, 0, 0])


def test_jax_pow():
    if not jax:
        skip('JAX not installed')

    expr = Pow(2, -1, evaluate=False)
    f = lambdify([], expr, 'jax')
    assert f() == 0.5


def test_jax_expm1():
    if not jax:
        skip("JAX not installed")

    f = lambdify((a,), expm1(a), 'jax')
    assert abs(f(1e-10) - 1e-10 - 5e-21) <= 1e-10 * JAX_DEFAULT_EPSILON


def test_jax_log1p():
    if not jax:
        skip("JAX not installed")

    f = lambdify((a,), log1p(a), 'jax')
    assert abs(f(1e-99) - 1e-99) <= 1e-99 * JAX_DEFAULT_EPSILON

def test_jax_hypot():
    if not jax:
        skip("JAX not installed")
    assert abs(lambdify((a, b), hypot(a, b), 'jax')(3, 4) - 5) <= JAX_DEFAULT_EPSILON

def test_jax_log10():
    if not jax:
        skip("JAX not installed")

    assert abs(lambdify((a,), log10(a), 'jax')(100) - 2) <= JAX_DEFAULT_EPSILON


def test_jax_exp2():
    if not jax:
        skip("JAX not installed")
    assert abs(lambdify((a,), exp2(a), 'jax')(5) - 32) <= JAX_DEFAULT_EPSILON


def test_jax_log2():
    if not jax:
        skip("JAX not installed")
    assert abs(lambdify((a,), log2(a), 'jax')(256) - 8) <= JAX_DEFAULT_EPSILON


def test_jax_Sqrt():
    if not jax:
        skip("JAX not installed")
    assert abs(lambdify((a,), Sqrt(a), 'jax')(4) - 2) <= JAX_DEFAULT_EPSILON


def test_jax_sqrt():
    if not jax:
        skip("JAX not installed")
    assert abs(lambdify((a,), sqrt(a), 'jax')(4) - 2) <= JAX_DEFAULT_EPSILON


def test_jax_matsolve():
    if not jax:
        skip("JAX not installed")

    M = MatrixSymbol("M", 3, 3)
    x = MatrixSymbol("x", 3, 1)

    expr = M**(-1) * x + x
    matsolve_expr = MatrixSolve(M, x) + x

    f = lambdify((M, x), expr, 'jax')
    f_matsolve = lambdify((M, x), matsolve_expr, 'jax')

    m0 = jax.numpy.array([[1, 2, 3], [3, 2, 5], [5, 6, 7]])
    assert jax.numpy.linalg.matrix_rank(m0) == 3

    x0 = jax.numpy.array([3, 4, 5])

    assert jax.numpy.allclose(f_matsolve(m0, x0), f(m0, x0))


def test_16857():
    if not jax:
        skip("JAX not installed")

    a_1 = MatrixSymbol('a_1', 10, 3)
    a_2 = MatrixSymbol('a_2', 10, 3)
    a_3 = MatrixSymbol('a_3', 10, 3)
    a_4 = MatrixSymbol('a_4', 10, 3)
    A = BlockMatrix([[a_1, a_2], [a_3, a_4]])
    assert A.shape == (20, 6)

    printer = JaxPrinter()
    assert printer.doprint(A) == 'jax.numpy.block([[a_1, a_2], [a_3, a_4]])'


def test_issue_17006():
    if not jax:
        skip("JAX not installed")

    M = MatrixSymbol("M", 2, 2)

    f = lambdify(M, M + Identity(2), 'jax')
    ma = jax.numpy.array([[1, 2], [3, 4]])
    mr = jax.numpy.array([[2, 2], [3, 5]])

    assert (f(ma) == mr).all()

    from sympy.core.symbol import symbols
    n = symbols('n', integer=True)
    N = MatrixSymbol("M", n, n)
    raises(NotImplementedError, lambda: lambdify(N, N + Identity(n), 'jax'))


def test_jax_array():
    assert JaxPrinter().doprint(Array(((1, 2), (3, 5)))) == 'jax.numpy.array([[1, 2], [3, 5]])'
    assert JaxPrinter().doprint(Array((1, 2))) == 'jax.numpy.array((1, 2))'


def test_jax_known_funcs_consts():
    assert _jax_known_constants['NaN'] == 'jax.numpy.nan'
    assert _jax_known_constants['EulerGamma'] == 'jax.numpy.euler_gamma'

    assert _jax_known_functions['acos'] == 'jax.numpy.arccos'
    assert _jax_known_functions['log'] == 'jax.numpy.log'


def test_jax_print_methods():
    prntr = JaxPrinter()
    assert hasattr(prntr, '_print_acos')
    assert hasattr(prntr, '_print_log')


def test_jax_printmethod():
    printer = JaxPrinter()
    assert hasattr(printer, 'printmethod')
    assert printer.printmethod == '_jaxcode'


def test_jax_custom_print_method():

    class expm1(Function):

        def _jaxcode(self, printer):
            x, = self.args
            function = f'expm1({printer._print(x)})'
            return printer._module_format(printer._module + '.' + function)

    printer = JaxPrinter()
    assert printer.doprint(expm1(Symbol('x'))) == 'jax.numpy.expm1(x)'
