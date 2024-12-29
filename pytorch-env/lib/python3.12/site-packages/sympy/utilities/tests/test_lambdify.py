from itertools import product
import math
import inspect



import mpmath
from sympy.testing.pytest import raises, warns_deprecated_sympy
from sympy.concrete.summations import Sum
from sympy.core.function import (Function, Lambda, diff)
from sympy.core.numbers import (E, Float, I, Rational, all_close, oo, pi)
from sympy.core.relational import Eq
from sympy.core.singleton import S
from sympy.core.symbol import (Dummy, symbols)
from sympy.functions.combinatorial.factorials import (RisingFactorial, factorial)
from sympy.functions.combinatorial.numbers import bernoulli, harmonic
from sympy.functions.elementary.complexes import Abs
from sympy.functions.elementary.exponential import exp, log
from sympy.functions.elementary.hyperbolic import acosh
from sympy.functions.elementary.integers import floor
from sympy.functions.elementary.miscellaneous import (Max, Min, sqrt)
from sympy.functions.elementary.piecewise import Piecewise
from sympy.functions.elementary.trigonometric import (acos, cos, cot, sin,
                                                      sinc, tan)
from sympy.functions.special.bessel import (besseli, besselj, besselk, bessely, jn, yn)
from sympy.functions.special.beta_functions import (beta, betainc, betainc_regularized)
from sympy.functions.special.delta_functions import (Heaviside)
from sympy.functions.special.error_functions import (Ei, erf, erfc, fresnelc, fresnels, Si, Ci)
from sympy.functions.special.gamma_functions import (digamma, gamma, loggamma, polygamma)
from sympy.integrals.integrals import Integral
from sympy.logic.boolalg import (And, false, ITE, Not, Or, true)
from sympy.matrices.expressions.dotproduct import DotProduct
from sympy.simplify.cse_main import cse
from sympy.tensor.array import derive_by_array, Array
from sympy.tensor.indexed import IndexedBase
from sympy.utilities.lambdify import lambdify
from sympy.utilities.iterables import numbered_symbols
from sympy.vector import CoordSys3D
from sympy.core.expr import UnevaluatedExpr
from sympy.codegen.cfunctions import expm1, log1p, exp2, log2, log10, hypot
from sympy.codegen.numpy_nodes import logaddexp, logaddexp2
from sympy.codegen.scipy_nodes import cosm1, powm1
from sympy.functions.elementary.complexes import re, im, arg
from sympy.functions.special.polynomials import \
    chebyshevt, chebyshevu, legendre, hermite, laguerre, gegenbauer, \
    assoc_legendre, assoc_laguerre, jacobi
from sympy.matrices import Matrix, MatrixSymbol, SparseMatrix
from sympy.printing.lambdarepr import LambdaPrinter
from sympy.printing.numpy import NumPyPrinter
from sympy.utilities.lambdify import implemented_function, lambdastr
from sympy.testing.pytest import skip
from sympy.utilities.decorator import conserve_mpmath_dps
from sympy.utilities.exceptions import ignore_warnings
from sympy.external import import_module
from sympy.functions.special.gamma_functions import uppergamma, lowergamma


import sympy


MutableDenseMatrix = Matrix

numpy = import_module('numpy')
scipy = import_module('scipy', import_kwargs={'fromlist': ['sparse']})
numexpr = import_module('numexpr')
tensorflow = import_module('tensorflow')
cupy = import_module('cupy')
jax = import_module('jax')
numba = import_module('numba')

if tensorflow:
    # Hide Tensorflow warnings
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

w, x, y, z = symbols('w,x,y,z')

#================== Test different arguments =======================


def test_no_args():
    f = lambdify([], 1)
    raises(TypeError, lambda: f(-1))
    assert f() == 1


def test_single_arg():
    f = lambdify(x, 2*x)
    assert f(1) == 2


def test_list_args():
    f = lambdify([x, y], x + y)
    assert f(1, 2) == 3


def test_nested_args():
    f1 = lambdify([[w, x]], [w, x])
    assert f1([91, 2]) == [91, 2]
    raises(TypeError, lambda: f1(1, 2))

    f2 = lambdify([(w, x), (y, z)], [w, x, y, z])
    assert f2((18, 12), (73, 4)) == [18, 12, 73, 4]
    raises(TypeError, lambda: f2(3, 4))

    f3 = lambdify([w, [[[x]], y], z], [w, x, y, z])
    assert f3(10, [[[52]], 31], 44) == [10, 52, 31, 44]


def test_str_args():
    f = lambdify('x,y,z', 'z,y,x')
    assert f(3, 2, 1) == (1, 2, 3)
    assert f(1.0, 2.0, 3.0) == (3.0, 2.0, 1.0)
    # make sure correct number of args required
    raises(TypeError, lambda: f(0))


def test_own_namespace_1():
    myfunc = lambda x: 1
    f = lambdify(x, sin(x), {"sin": myfunc})
    assert f(0.1) == 1
    assert f(100) == 1


def test_own_namespace_2():
    def myfunc(x):
        return 1
    f = lambdify(x, sin(x), {'sin': myfunc})
    assert f(0.1) == 1
    assert f(100) == 1


def test_own_module():
    f = lambdify(x, sin(x), math)
    assert f(0) == 0.0

    p, q, r = symbols("p q r", real=True)
    ae = abs(exp(p+UnevaluatedExpr(q+r)))
    f = lambdify([p, q, r], [ae, ae], modules=math)
    results = f(1.0, 1e18, -1e18)
    refvals = [math.exp(1.0)]*2
    for res, ref in zip(results, refvals):
        assert abs((res-ref)/ref) < 1e-15


def test_bad_args():
    # no vargs given
    raises(TypeError, lambda: lambdify(1))
    # same with vector exprs
    raises(TypeError, lambda: lambdify([1, 2]))


def test_atoms():
    # Non-Symbol atoms should not be pulled out from the expression namespace
    f = lambdify(x, pi + x, {"pi": 3.14})
    assert f(0) == 3.14
    f = lambdify(x, I + x, {"I": 1j})
    assert f(1) == 1 + 1j

#================== Test different modules =========================

# high precision output of sin(0.2*pi) is used to detect if precision is lost unwanted


@conserve_mpmath_dps
def test_sympy_lambda():
    mpmath.mp.dps = 50
    sin02 = mpmath.mpf("0.19866933079506121545941262711838975037020672954020")
    f = lambdify(x, sin(x), "sympy")
    assert f(x) == sin(x)
    prec = 1e-15
    assert -prec < f(Rational(1, 5)).evalf() - Float(str(sin02)) < prec
    # arctan is in numpy module and should not be available
    # The arctan below gives NameError. What is this supposed to test?
    # raises(NameError, lambda: lambdify(x, arctan(x), "sympy"))


@conserve_mpmath_dps
def test_math_lambda():
    mpmath.mp.dps = 50
    sin02 = mpmath.mpf("0.19866933079506121545941262711838975037020672954020")
    f = lambdify(x, sin(x), "math")
    prec = 1e-15
    assert -prec < f(0.2) - sin02 < prec
    raises(TypeError, lambda: f(x))
           # if this succeeds, it can't be a Python math function


@conserve_mpmath_dps
def test_mpmath_lambda():
    mpmath.mp.dps = 50
    sin02 = mpmath.mpf("0.19866933079506121545941262711838975037020672954020")
    f = lambdify(x, sin(x), "mpmath")
    prec = 1e-49  # mpmath precision is around 50 decimal places
    assert -prec < f(mpmath.mpf("0.2")) - sin02 < prec
    raises(TypeError, lambda: f(x))
           # if this succeeds, it can't be a mpmath function

    ref2 = (mpmath.mpf("1e-30")
            - mpmath.mpf("1e-45")/2
            + 5*mpmath.mpf("1e-60")/6
            - 3*mpmath.mpf("1e-75")/4
            + 33*mpmath.mpf("1e-90")/40
            )
    f2a = lambdify((x, y), x**y - 1, "mpmath")
    f2b = lambdify((x, y), powm1(x, y), "mpmath")
    f2c = lambdify((x,), expm1(x*log1p(x)), "mpmath")
    ans2a = f2a(mpmath.mpf("1")+mpmath.mpf("1e-15"), mpmath.mpf("1e-15"))
    ans2b = f2b(mpmath.mpf("1")+mpmath.mpf("1e-15"), mpmath.mpf("1e-15"))
    ans2c = f2c(mpmath.mpf("1e-15"))
    assert abs(ans2a - ref2) < 1e-51
    assert abs(ans2b - ref2) < 1e-67
    assert abs(ans2c - ref2) < 1e-80


@conserve_mpmath_dps
def test_number_precision():
    mpmath.mp.dps = 50
    sin02 = mpmath.mpf("0.19866933079506121545941262711838975037020672954020")
    f = lambdify(x, sin02, "mpmath")
    prec = 1e-49  # mpmath precision is around 50 decimal places
    assert -prec < f(0) - sin02 < prec

@conserve_mpmath_dps
def test_mpmath_precision():
    mpmath.mp.dps = 100
    assert str(lambdify((), pi.evalf(100), 'mpmath')()) == str(pi.evalf(100))

#================== Test Translations ==============================
# We can only check if all translated functions are valid. It has to be checked
# by hand if they are complete.


def test_math_transl():
    from sympy.utilities.lambdify import MATH_TRANSLATIONS
    for sym, mat in MATH_TRANSLATIONS.items():
        assert sym in sympy.__dict__
        assert mat in math.__dict__


def test_mpmath_transl():
    from sympy.utilities.lambdify import MPMATH_TRANSLATIONS
    for sym, mat in MPMATH_TRANSLATIONS.items():
        assert sym in sympy.__dict__ or sym == 'Matrix'
        assert mat in mpmath.__dict__


def test_numpy_transl():
    if not numpy:
        skip("numpy not installed.")

    from sympy.utilities.lambdify import NUMPY_TRANSLATIONS
    for sym, nump in NUMPY_TRANSLATIONS.items():
        assert sym in sympy.__dict__
        assert nump in numpy.__dict__


def test_scipy_transl():
    if not scipy:
        skip("scipy not installed.")

    from sympy.utilities.lambdify import SCIPY_TRANSLATIONS
    for sym, scip in SCIPY_TRANSLATIONS.items():
        assert sym in sympy.__dict__
        assert scip in scipy.__dict__ or scip in scipy.special.__dict__


def test_numpy_translation_abs():
    if not numpy:
        skip("numpy not installed.")

    f = lambdify(x, Abs(x), "numpy")
    assert f(-1) == 1
    assert f(1) == 1


def test_numexpr_printer():
    if not numexpr:
        skip("numexpr not installed.")

    # if translation/printing is done incorrectly then evaluating
    # a lambdified numexpr expression will throw an exception
    from sympy.printing.lambdarepr import NumExprPrinter

    blacklist = ('where', 'complex', 'contains')
    arg_tuple = (x, y, z) # some functions take more than one argument
    for sym in NumExprPrinter._numexpr_functions.keys():
        if sym in blacklist:
            continue
        ssym = S(sym)
        if hasattr(ssym, '_nargs'):
            nargs = ssym._nargs[0]
        else:
            nargs = 1
        args = arg_tuple[:nargs]
        f = lambdify(args, ssym(*args), modules='numexpr')
        assert f(*(1, )*nargs) is not None


def test_issue_9334():
    if not numexpr:
        skip("numexpr not installed.")
    if not numpy:
        skip("numpy not installed.")
    expr = S('b*a - sqrt(a**2)')
    a, b = sorted(expr.free_symbols, key=lambda s: s.name)
    func_numexpr = lambdify((a,b), expr, modules=[numexpr], dummify=False)
    foo, bar = numpy.random.random((2, 4))
    func_numexpr(foo, bar)


def test_issue_12984():
    if not numexpr:
        skip("numexpr not installed.")
    func_numexpr = lambdify((x,y,z), Piecewise((y, x >= 0), (z, x > -1)), numexpr)
    with ignore_warnings(RuntimeWarning):
        assert func_numexpr(1, 24, 42) == 24
        assert str(func_numexpr(-1, 24, 42)) == 'nan'


def test_empty_modules():
    x, y = symbols('x y')
    expr = -(x % y)

    no_modules = lambdify([x, y], expr)
    empty_modules = lambdify([x, y], expr, modules=[])
    assert no_modules(3, 7) == empty_modules(3, 7)
    assert no_modules(3, 7) == -3


def test_exponentiation():
    f = lambdify(x, x**2)
    assert f(-1) == 1
    assert f(0) == 0
    assert f(1) == 1
    assert f(-2) == 4
    assert f(2) == 4
    assert f(2.5) == 6.25


def test_sqrt():
    f = lambdify(x, sqrt(x))
    assert f(0) == 0.0
    assert f(1) == 1.0
    assert f(4) == 2.0
    assert abs(f(2) - 1.414) < 0.001
    assert f(6.25) == 2.5


def test_trig():
    f = lambdify([x], [cos(x), sin(x)], 'math')
    d = f(pi)
    prec = 1e-11
    assert -prec < d[0] + 1 < prec
    assert -prec < d[1] < prec
    d = f(3.14159)
    prec = 1e-5
    assert -prec < d[0] + 1 < prec
    assert -prec < d[1] < prec


def test_integral():
    if numpy and not scipy:
        skip("scipy not installed.")
    f = Lambda(x, exp(-x**2))
    l = lambdify(y, Integral(f(x), (x, y, oo)))
    d = l(-oo)
    assert 1.77245385 < d < 1.772453851


def test_double_integral():
    if numpy and not scipy:
        skip("scipy not installed.")
    # example from http://mpmath.org/doc/current/calculus/integration.html
    i = Integral(1/(1 - x**2*y**2), (x, 0, 1), (y, 0, z))
    l = lambdify([z], i)
    d = l(1)
    assert 1.23370055 < d < 1.233700551

def test_spherical_bessel():
    if numpy and not scipy:
        skip("scipy not installed.")
    test_point = 4.2 #randomly selected
    x = symbols("x")
    jtest = jn(2, x)
    assert abs(lambdify(x,jtest)(test_point) -
            jtest.subs(x,test_point).evalf()) < 1e-8
    ytest = yn(2, x)
    assert abs(lambdify(x,ytest)(test_point) -
            ytest.subs(x,test_point).evalf()) < 1e-8


#================== Test vectors ===================================


def test_vector_simple():
    f = lambdify((x, y, z), (z, y, x))
    assert f(3, 2, 1) == (1, 2, 3)
    assert f(1.0, 2.0, 3.0) == (3.0, 2.0, 1.0)
    # make sure correct number of args required
    raises(TypeError, lambda: f(0))


def test_vector_discontinuous():
    f = lambdify(x, (-1/x, 1/x))
    raises(ZeroDivisionError, lambda: f(0))
    assert f(1) == (-1.0, 1.0)
    assert f(2) == (-0.5, 0.5)
    assert f(-2) == (0.5, -0.5)


def test_trig_symbolic():
    f = lambdify([x], [cos(x), sin(x)], 'math')
    d = f(pi)
    assert abs(d[0] + 1) < 0.0001
    assert abs(d[1] - 0) < 0.0001


def test_trig_float():
    f = lambdify([x], [cos(x), sin(x)])
    d = f(3.14159)
    assert abs(d[0] + 1) < 0.0001
    assert abs(d[1] - 0) < 0.0001


def test_docs():
    f = lambdify(x, x**2)
    assert f(2) == 4
    f = lambdify([x, y, z], [z, y, x])
    assert f(1, 2, 3) == [3, 2, 1]
    f = lambdify(x, sqrt(x))
    assert f(4) == 2.0
    f = lambdify((x, y), sin(x*y)**2)
    assert f(0, 5) == 0


def test_math():
    f = lambdify((x, y), sin(x), modules="math")
    assert f(0, 5) == 0


def test_sin():
    f = lambdify(x, sin(x)**2)
    assert isinstance(f(2), float)
    f = lambdify(x, sin(x)**2, modules="math")
    assert isinstance(f(2), float)


def test_matrix():
    A = Matrix([[x, x*y], [sin(z) + 4, x**z]])
    sol = Matrix([[1, 2], [sin(3) + 4, 1]])
    f = lambdify((x, y, z), A, modules="sympy")
    assert f(1, 2, 3) == sol
    f = lambdify((x, y, z), (A, [A]), modules="sympy")
    assert f(1, 2, 3) == (sol, [sol])
    J = Matrix((x, x + y)).jacobian((x, y))
    v = Matrix((x, y))
    sol = Matrix([[1, 0], [1, 1]])
    assert lambdify(v, J, modules='sympy')(1, 2) == sol
    assert lambdify(v.T, J, modules='sympy')(1, 2) == sol


def test_numpy_matrix():
    if not numpy:
        skip("numpy not installed.")
    A = Matrix([[x, x*y], [sin(z) + 4, x**z]])
    sol_arr = numpy.array([[1, 2], [numpy.sin(3) + 4, 1]])
    #Lambdify array first, to ensure return to array as default
    f = lambdify((x, y, z), A, ['numpy'])
    numpy.testing.assert_allclose(f(1, 2, 3), sol_arr)
    #Check that the types are arrays and matrices
    assert isinstance(f(1, 2, 3), numpy.ndarray)

    # gh-15071
    class dot(Function):
        pass
    x_dot_mtx = dot(x, Matrix([[2], [1], [0]]))
    f_dot1 = lambdify(x, x_dot_mtx)
    inp = numpy.zeros((17, 3))
    assert numpy.all(f_dot1(inp) == 0)

    strict_kw = {"allow_unknown_functions": False, "inline": True, "fully_qualified_modules": False}
    p2 = NumPyPrinter(dict(user_functions={'dot': 'dot'}, **strict_kw))
    f_dot2 = lambdify(x, x_dot_mtx, printer=p2)
    assert numpy.all(f_dot2(inp) == 0)

    p3 = NumPyPrinter(strict_kw)
    # The line below should probably fail upon construction (before calling with "(inp)"):
    raises(Exception, lambda: lambdify(x, x_dot_mtx, printer=p3)(inp))


def test_numpy_transpose():
    if not numpy:
        skip("numpy not installed.")
    A = Matrix([[1, x], [0, 1]])
    f = lambdify((x), A.T, modules="numpy")
    numpy.testing.assert_array_equal(f(2), numpy.array([[1, 0], [2, 1]]))


def test_numpy_dotproduct():
    if not numpy:
        skip("numpy not installed")
    A = Matrix([x, y, z])
    f1 = lambdify([x, y, z], DotProduct(A, A), modules='numpy')
    f2 = lambdify([x, y, z], DotProduct(A, A.T), modules='numpy')
    f3 = lambdify([x, y, z], DotProduct(A.T, A), modules='numpy')
    f4 = lambdify([x, y, z], DotProduct(A, A.T), modules='numpy')

    assert f1(1, 2, 3) == \
           f2(1, 2, 3) == \
           f3(1, 2, 3) == \
           f4(1, 2, 3) == \
           numpy.array([14])


def test_numpy_inverse():
    if not numpy:
        skip("numpy not installed.")
    A = Matrix([[1, x], [0, 1]])
    f = lambdify((x), A**-1, modules="numpy")
    numpy.testing.assert_array_equal(f(2), numpy.array([[1, -2], [0,  1]]))


def test_numpy_old_matrix():
    if not numpy:
        skip("numpy not installed.")
    A = Matrix([[x, x*y], [sin(z) + 4, x**z]])
    sol_arr = numpy.array([[1, 2], [numpy.sin(3) + 4, 1]])
    f = lambdify((x, y, z), A, [{'ImmutableDenseMatrix': numpy.matrix}, 'numpy'])
    with ignore_warnings(PendingDeprecationWarning):
        numpy.testing.assert_allclose(f(1, 2, 3), sol_arr)
        assert isinstance(f(1, 2, 3), numpy.matrix)


def test_scipy_sparse_matrix():
    if not scipy:
        skip("scipy not installed.")
    A = SparseMatrix([[x, 0], [0, y]])
    f = lambdify((x, y), A, modules="scipy")
    B = f(1, 2)
    assert isinstance(B, scipy.sparse.coo_matrix)


def test_python_div_zero_issue_11306():
    if not numpy:
        skip("numpy not installed.")
    p = Piecewise((1 / x, y < -1), (x, y < 1), (1 / x, True))
    f = lambdify([x, y], p, modules='numpy')
    with numpy.errstate(divide='ignore'):
        assert float(f(numpy.array(0), numpy.array(0.5))) == 0
        assert float(f(numpy.array(0), numpy.array(1))) == float('inf')


def test_issue9474():
    mods = [None, 'math']
    if numpy:
        mods.append('numpy')
    if mpmath:
        mods.append('mpmath')
    for mod in mods:
        f = lambdify(x, S.One/x, modules=mod)
        assert f(2) == 0.5
        f = lambdify(x, floor(S.One/x), modules=mod)
        assert f(2) == 0

    for absfunc, modules in product([Abs, abs], mods):
        f = lambdify(x, absfunc(x), modules=modules)
        assert f(-1) == 1
        assert f(1) == 1
        assert f(3+4j) == 5


def test_issue_9871():
    if not numexpr:
        skip("numexpr not installed.")
    if not numpy:
        skip("numpy not installed.")

    r = sqrt(x**2 + y**2)
    expr = diff(1/r, x)

    xn = yn = numpy.linspace(1, 10, 16)
    # expr(xn, xn) = -xn/(sqrt(2)*xn)^3
    fv_exact = -numpy.sqrt(2.)**-3 * xn**-2

    fv_numpy = lambdify((x, y), expr, modules='numpy')(xn, yn)
    fv_numexpr = lambdify((x, y), expr, modules='numexpr')(xn, yn)
    numpy.testing.assert_allclose(fv_numpy, fv_exact, rtol=1e-10)
    numpy.testing.assert_allclose(fv_numexpr, fv_exact, rtol=1e-10)


def test_numpy_piecewise():
    if not numpy:
        skip("numpy not installed.")
    pieces = Piecewise((x, x < 3), (x**2, x > 5), (0, True))
    f = lambdify(x, pieces, modules="numpy")
    numpy.testing.assert_array_equal(f(numpy.arange(10)),
                                     numpy.array([0, 1, 2, 0, 0, 0, 36, 49, 64, 81]))
    # If we evaluate somewhere all conditions are False, we should get back NaN
    nodef_func = lambdify(x, Piecewise((x, x > 0), (-x, x < 0)))
    numpy.testing.assert_array_equal(nodef_func(numpy.array([-1, 0, 1])),
                                     numpy.array([1, numpy.nan, 1]))


def test_numpy_logical_ops():
    if not numpy:
        skip("numpy not installed.")
    and_func = lambdify((x, y), And(x, y), modules="numpy")
    and_func_3 = lambdify((x, y, z), And(x, y, z), modules="numpy")
    or_func = lambdify((x, y), Or(x, y), modules="numpy")
    or_func_3 = lambdify((x, y, z), Or(x, y, z), modules="numpy")
    not_func = lambdify((x), Not(x), modules="numpy")
    arr1 = numpy.array([True, True])
    arr2 = numpy.array([False, True])
    arr3 = numpy.array([True, False])
    numpy.testing.assert_array_equal(and_func(arr1, arr2), numpy.array([False, True]))
    numpy.testing.assert_array_equal(and_func_3(arr1, arr2, arr3), numpy.array([False, False]))
    numpy.testing.assert_array_equal(or_func(arr1, arr2), numpy.array([True, True]))
    numpy.testing.assert_array_equal(or_func_3(arr1, arr2, arr3), numpy.array([True, True]))
    numpy.testing.assert_array_equal(not_func(arr2), numpy.array([True, False]))


def test_numpy_matmul():
    if not numpy:
        skip("numpy not installed.")
    xmat = Matrix([[x, y], [z, 1+z]])
    ymat = Matrix([[x**2], [Abs(x)]])
    mat_func = lambdify((x, y, z), xmat*ymat, modules="numpy")
    numpy.testing.assert_array_equal(mat_func(0.5, 3, 4), numpy.array([[1.625], [3.5]]))
    numpy.testing.assert_array_equal(mat_func(-0.5, 3, 4), numpy.array([[1.375], [3.5]]))
    # Multiple matrices chained together in multiplication
    f = lambdify((x, y, z), xmat*xmat*xmat, modules="numpy")
    numpy.testing.assert_array_equal(f(0.5, 3, 4), numpy.array([[72.125, 119.25],
                                                                [159, 251]]))


def test_numpy_numexpr():
    if not numpy:
        skip("numpy not installed.")
    if not numexpr:
        skip("numexpr not installed.")
    a, b, c = numpy.random.randn(3, 128, 128)
    # ensure that numpy and numexpr return same value for complicated expression
    expr = sin(x) + cos(y) + tan(z)**2 + Abs(z-y)*acos(sin(y*z)) + \
           Abs(y-z)*acosh(2+exp(y-x))- sqrt(x**2+I*y**2)
    npfunc = lambdify((x, y, z), expr, modules='numpy')
    nefunc = lambdify((x, y, z), expr, modules='numexpr')
    assert numpy.allclose(npfunc(a, b, c), nefunc(a, b, c))


def test_numexpr_userfunctions():
    if not numpy:
        skip("numpy not installed.")
    if not numexpr:
        skip("numexpr not installed.")
    a, b = numpy.random.randn(2, 10)
    uf = type('uf', (Function, ),
              {'eval' : classmethod(lambda x, y : y**2+1)})
    func = lambdify(x, 1-uf(x), modules='numexpr')
    assert numpy.allclose(func(a), -(a**2))

    uf = implemented_function(Function('uf'), lambda x, y : 2*x*y+1)
    func = lambdify((x, y), uf(x, y), modules='numexpr')
    assert numpy.allclose(func(a, b), 2*a*b+1)


def test_tensorflow_basic_math():
    if not tensorflow:
        skip("tensorflow not installed.")
    expr = Max(sin(x), Abs(1/(x+2)))
    func = lambdify(x, expr, modules="tensorflow")

    with tensorflow.compat.v1.Session() as s:
        a = tensorflow.constant(0, dtype=tensorflow.float32)
        assert func(a).eval(session=s) == 0.5


def test_tensorflow_placeholders():
    if not tensorflow:
        skip("tensorflow not installed.")
    expr = Max(sin(x), Abs(1/(x+2)))
    func = lambdify(x, expr, modules="tensorflow")

    with tensorflow.compat.v1.Session() as s:
        a = tensorflow.compat.v1.placeholder(dtype=tensorflow.float32)
        assert func(a).eval(session=s, feed_dict={a: 0}) == 0.5


def test_tensorflow_variables():
    if not tensorflow:
        skip("tensorflow not installed.")
    expr = Max(sin(x), Abs(1/(x+2)))
    func = lambdify(x, expr, modules="tensorflow")

    with tensorflow.compat.v1.Session() as s:
        a = tensorflow.Variable(0, dtype=tensorflow.float32)
        s.run(a.initializer)
        assert func(a).eval(session=s, feed_dict={a: 0}) == 0.5


def test_tensorflow_logical_operations():
    if not tensorflow:
        skip("tensorflow not installed.")
    expr = Not(And(Or(x, y), y))
    func = lambdify([x, y], expr, modules="tensorflow")

    with tensorflow.compat.v1.Session() as s:
        assert func(False, True).eval(session=s) == False


def test_tensorflow_piecewise():
    if not tensorflow:
        skip("tensorflow not installed.")
    expr = Piecewise((0, Eq(x,0)), (-1, x < 0), (1, x > 0))
    func = lambdify(x, expr, modules="tensorflow")

    with tensorflow.compat.v1.Session() as s:
        assert func(-1).eval(session=s) == -1
        assert func(0).eval(session=s) == 0
        assert func(1).eval(session=s) == 1


def test_tensorflow_multi_max():
    if not tensorflow:
        skip("tensorflow not installed.")
    expr = Max(x, -x, x**2)
    func = lambdify(x, expr, modules="tensorflow")

    with tensorflow.compat.v1.Session() as s:
        assert func(-2).eval(session=s) == 4


def test_tensorflow_multi_min():
    if not tensorflow:
        skip("tensorflow not installed.")
    expr = Min(x, -x, x**2)
    func = lambdify(x, expr, modules="tensorflow")

    with tensorflow.compat.v1.Session() as s:
        assert func(-2).eval(session=s) == -2


def test_tensorflow_relational():
    if not tensorflow:
        skip("tensorflow not installed.")
    expr = x >= 0
    func = lambdify(x, expr, modules="tensorflow")

    with tensorflow.compat.v1.Session() as s:
        assert func(1).eval(session=s) == True


def test_tensorflow_complexes():
    if not tensorflow:
        skip("tensorflow not installed")

    func1 = lambdify(x, re(x), modules="tensorflow")
    func2 = lambdify(x, im(x), modules="tensorflow")
    func3 = lambdify(x, Abs(x), modules="tensorflow")
    func4 = lambdify(x, arg(x), modules="tensorflow")

    with tensorflow.compat.v1.Session() as s:
        # For versions before
        # https://github.com/tensorflow/tensorflow/issues/30029
        # resolved, using Python numeric types may not work
        a = tensorflow.constant(1+2j)
        assert func1(a).eval(session=s) == 1
        assert func2(a).eval(session=s) == 2

        tensorflow_result = func3(a).eval(session=s)
        sympy_result = Abs(1 + 2j).evalf()
        assert abs(tensorflow_result-sympy_result) < 10**-6

        tensorflow_result = func4(a).eval(session=s)
        sympy_result = arg(1 + 2j).evalf()
        assert abs(tensorflow_result-sympy_result) < 10**-6


def test_tensorflow_array_arg():
    # Test for issue 14655 (tensorflow part)
    if not tensorflow:
        skip("tensorflow not installed.")

    f = lambdify([[x, y]], x*x + y, 'tensorflow')

    with tensorflow.compat.v1.Session() as s:
        fcall = f(tensorflow.constant([2.0, 1.0]))
        assert fcall.eval(session=s) == 5.0


#================== Test symbolic ==================================


def test_sym_single_arg():
    f = lambdify(x, x * y)
    assert f(z) == z * y


def test_sym_list_args():
    f = lambdify([x, y], x + y + z)
    assert f(1, 2) == 3 + z


def test_sym_integral():
    f = Lambda(x, exp(-x**2))
    l = lambdify(x, Integral(f(x), (x, -oo, oo)), modules="sympy")
    assert l(y) == Integral(exp(-y**2), (y, -oo, oo))
    assert l(y).doit() == sqrt(pi)


def test_namespace_order():
    # lambdify had a bug, such that module dictionaries or cached module
    # dictionaries would pull earlier namespaces into themselves.
    # Because the module dictionaries form the namespace of the
    # generated lambda, this meant that the behavior of a previously
    # generated lambda function could change as a result of later calls
    # to lambdify.
    n1 = {'f': lambda x: 'first f'}
    n2 = {'f': lambda x: 'second f',
          'g': lambda x: 'function g'}
    f = sympy.Function('f')
    g = sympy.Function('g')
    if1 = lambdify(x, f(x), modules=(n1, "sympy"))
    assert if1(1) == 'first f'
    if2 = lambdify(x, g(x), modules=(n2, "sympy"))
    # previously gave 'second f'
    assert if1(1) == 'first f'

    assert if2(1) == 'function g'


def test_imps():
    # Here we check if the default returned functions are anonymous - in
    # the sense that we can have more than one function with the same name
    f = implemented_function('f', lambda x: 2*x)
    g = implemented_function('f', lambda x: math.sqrt(x))
    l1 = lambdify(x, f(x))
    l2 = lambdify(x, g(x))
    assert str(f(x)) == str(g(x))
    assert l1(3) == 6
    assert l2(3) == math.sqrt(3)
    # check that we can pass in a Function as input
    func = sympy.Function('myfunc')
    assert not hasattr(func, '_imp_')
    my_f = implemented_function(func, lambda x: 2*x)
    assert hasattr(my_f, '_imp_')
    # Error for functions with same name and different implementation
    f2 = implemented_function("f", lambda x: x + 101)
    raises(ValueError, lambda: lambdify(x, f(f2(x))))


def test_imps_errors():
    # Test errors that implemented functions can return, and still be able to
    # form expressions.
    # See: https://github.com/sympy/sympy/issues/10810
    #
    # XXX: Removed AttributeError here. This test was added due to issue 10810
    # but that issue was about ValueError. It doesn't seem reasonable to
    # "support" catching AttributeError in the same context...
    for val, error_class in product((0, 0., 2, 2.0), (TypeError, ValueError)):

        def myfunc(a):
            if a == 0:
                raise error_class
            return 1

        f = implemented_function('f', myfunc)
        expr = f(val)
        assert expr == f(val)


def test_imps_wrong_args():
    raises(ValueError, lambda: implemented_function(sin, lambda x: x))


def test_lambdify_imps():
    # Test lambdify with implemented functions
    # first test basic (sympy) lambdify
    f = sympy.cos
    assert lambdify(x, f(x))(0) == 1
    assert lambdify(x, 1 + f(x))(0) == 2
    assert lambdify((x, y), y + f(x))(0, 1) == 2
    # make an implemented function and test
    f = implemented_function("f", lambda x: x + 100)
    assert lambdify(x, f(x))(0) == 100
    assert lambdify(x, 1 + f(x))(0) == 101
    assert lambdify((x, y), y + f(x))(0, 1) == 101
    # Can also handle tuples, lists, dicts as expressions
    lam = lambdify(x, (f(x), x))
    assert lam(3) == (103, 3)
    lam = lambdify(x, [f(x), x])
    assert lam(3) == [103, 3]
    lam = lambdify(x, [f(x), (f(x), x)])
    assert lam(3) == [103, (103, 3)]
    lam = lambdify(x, {f(x): x})
    assert lam(3) == {103: 3}
    lam = lambdify(x, {f(x): x})
    assert lam(3) == {103: 3}
    lam = lambdify(x, {x: f(x)})
    assert lam(3) == {3: 103}
    # Check that imp preferred to other namespaces by default
    d = {'f': lambda x: x + 99}
    lam = lambdify(x, f(x), d)
    assert lam(3) == 103
    # Unless flag passed
    lam = lambdify(x, f(x), d, use_imps=False)
    assert lam(3) == 102


def test_dummification():
    t = symbols('t')
    F = Function('F')
    G = Function('G')
    #"\alpha" is not a valid Python variable name
    #lambdify should sub in a dummy for it, and return
    #without a syntax error
    alpha = symbols(r'\alpha')
    some_expr = 2 * F(t)**2 / G(t)
    lam = lambdify((F(t), G(t)), some_expr)
    assert lam(3, 9) == 2
    lam = lambdify(sin(t), 2 * sin(t)**2)
    assert lam(F(t)) == 2 * F(t)**2
    #Test that \alpha was properly dummified
    lam = lambdify((alpha, t), 2*alpha + t)
    assert lam(2, 1) == 5
    raises(SyntaxError, lambda: lambdify(F(t) * G(t), F(t) * G(t) + 5))
    raises(SyntaxError, lambda: lambdify(2 * F(t), 2 * F(t) + 5))
    raises(SyntaxError, lambda: lambdify(2 * F(t), 4 * F(t) + 5))


def test_lambdify__arguments_with_invalid_python_identifiers():
    # see sympy/sympy#26690
    N = CoordSys3D('N')
    xn, yn, zn = N.base_scalars()
    expr = xn + yn
    f = lambdify([xn, yn], expr)
    res = f(0.2, 0.3)
    ref = 0.2 + 0.3
    assert abs(res-ref) < 1e-15


def test_curly_matrix_symbol():
    # Issue #15009
    curlyv = sympy.MatrixSymbol("{v}", 2, 1)
    lam = lambdify(curlyv, curlyv)
    assert lam(1)==1
    lam = lambdify(curlyv, curlyv, dummify=True)
    assert lam(1)==1


def test_python_keywords():
    # Test for issue 7452. The automatic dummification should ensure use of
    # Python reserved keywords as symbol names will create valid lambda
    # functions. This is an additional regression test.
    python_if = symbols('if')
    expr = python_if / 2
    f = lambdify(python_if, expr)
    assert f(4.0) == 2.0


def test_lambdify_docstring():
    func = lambdify((w, x, y, z), w + x + y + z)
    ref = (
        "Created with lambdify. Signature:\n\n"
        "func(w, x, y, z)\n\n"
        "Expression:\n\n"
        "w + x + y + z"
    ).splitlines()
    assert func.__doc__.splitlines()[:len(ref)] == ref
    syms = symbols('a1:26')
    func = lambdify(syms, sum(syms))
    ref = (
        "Created with lambdify. Signature:\n\n"
        "func(a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15,\n"
        "        a16, a17, a18, a19, a20, a21, a22, a23, a24, a25)\n\n"
        "Expression:\n\n"
        "a1 + a10 + a11 + a12 + a13 + a14 + a15 + a16 + a17 + a18 + a19 + a2 + a20 +..."
    ).splitlines()
    assert func.__doc__.splitlines()[:len(ref)] == ref


#================== Test special printers ==========================


def test_special_printers():
    from sympy.printing.lambdarepr import IntervalPrinter

    def intervalrepr(expr):
        return IntervalPrinter().doprint(expr)

    expr = sqrt(sqrt(2) + sqrt(3)) + S.Half

    func0 = lambdify((), expr, modules="mpmath", printer=intervalrepr)
    func1 = lambdify((), expr, modules="mpmath", printer=IntervalPrinter)
    func2 = lambdify((), expr, modules="mpmath", printer=IntervalPrinter())

    mpi = type(mpmath.mpi(1, 2))

    assert isinstance(func0(), mpi)
    assert isinstance(func1(), mpi)
    assert isinstance(func2(), mpi)

    # To check Is lambdify loggamma works for mpmath or not
    exp1 = lambdify(x, loggamma(x), 'mpmath')(5)
    exp2 = lambdify(x, loggamma(x), 'mpmath')(1.8)
    exp3 = lambdify(x, loggamma(x), 'mpmath')(15)
    exp_ls = [exp1, exp2, exp3]

    sol1 = mpmath.loggamma(5)
    sol2 = mpmath.loggamma(1.8)
    sol3 = mpmath.loggamma(15)
    sol_ls = [sol1, sol2, sol3]

    assert exp_ls == sol_ls


def test_true_false():
    # We want exact is comparison here, not just ==
    assert lambdify([], true)() is True
    assert lambdify([], false)() is False


def test_issue_2790():
    assert lambdify((x, (y, z)), x + y)(1, (2, 4)) == 3
    assert lambdify((x, (y, (w, z))), w + x + y + z)(1, (2, (3, 4))) == 10
    assert lambdify(x, x + 1, dummify=False)(1) == 2


def test_issue_12092():
    f = implemented_function('f', lambda x: x**2)
    assert f(f(2)).evalf() == Float(16)


def test_issue_14911():
    class Variable(sympy.Symbol):
        def _sympystr(self, printer):
            return printer.doprint(self.name)

        _lambdacode = _sympystr
        _numpycode = _sympystr

    x = Variable('x')
    y = 2 * x
    code = LambdaPrinter().doprint(y)
    assert code.replace(' ', '') == '2*x'


def test_ITE():
    assert lambdify((x, y, z), ITE(x, y, z))(True, 5, 3) == 5
    assert lambdify((x, y, z), ITE(x, y, z))(False, 5, 3) == 3


def test_Min_Max():
    # see gh-10375
    assert lambdify((x, y, z), Min(x, y, z))(1, 2, 3) == 1
    assert lambdify((x, y, z), Max(x, y, z))(1, 2, 3) == 3


def test_Indexed():
    # Issue #10934
    if not numpy:
        skip("numpy not installed")

    a = IndexedBase('a')
    i, j = symbols('i j')
    b = numpy.array([[1, 2], [3, 4]])
    assert lambdify(a, Sum(a[x, y], (x, 0, 1), (y, 0, 1)))(b) == 10


def test_issue_12173():
    #test for issue 12173
    expr1 = lambdify((x, y), uppergamma(x, y),"mpmath")(1, 2)
    expr2 = lambdify((x, y), lowergamma(x, y),"mpmath")(1, 2)
    assert expr1 == uppergamma(1, 2).evalf()
    assert expr2 == lowergamma(1, 2).evalf()


def test_issue_13642():
    if not numpy:
        skip("numpy not installed")
    f = lambdify(x, sinc(x))
    assert Abs(f(1) - sinc(1)).n() < 1e-15


def test_sinc_mpmath():
    f = lambdify(x, sinc(x), "mpmath")
    assert Abs(f(1) - sinc(1)).n() < 1e-15


def test_lambdify_dummy_arg():
    d1 = Dummy()
    f1 = lambdify(d1, d1 + 1, dummify=False)
    assert f1(2) == 3
    f1b = lambdify(d1, d1 + 1)
    assert f1b(2) == 3
    d2 = Dummy('x')
    f2 = lambdify(d2, d2 + 1)
    assert f2(2) == 3
    f3 = lambdify([[d2]], d2 + 1)
    assert f3([2]) == 3


def test_lambdify_mixed_symbol_dummy_args():
    d = Dummy()
    # Contrived example of name clash
    dsym = symbols(str(d))
    f = lambdify([d, dsym], d - dsym)
    assert f(4, 1) == 3


def test_numpy_array_arg():
    # Test for issue 14655 (numpy part)
    if not numpy:
        skip("numpy not installed")

    f = lambdify([[x, y]], x*x + y, 'numpy')

    assert f(numpy.array([2.0, 1.0])) == 5


def test_scipy_fns():
    if not scipy:
        skip("scipy not installed")

    single_arg_sympy_fns = [Ei, erf, erfc, factorial, gamma, loggamma, digamma, Si, Ci]
    single_arg_scipy_fns = [scipy.special.expi, scipy.special.erf, scipy.special.erfc,
        scipy.special.factorial, scipy.special.gamma, scipy.special.gammaln,
                            scipy.special.psi, scipy.special.sici, scipy.special.sici]
    numpy.random.seed(0)
    for (sympy_fn, scipy_fn) in zip(single_arg_sympy_fns, single_arg_scipy_fns):
        f = lambdify(x, sympy_fn(x), modules="scipy")
        for i in range(20):
            tv = numpy.random.uniform(-10, 10) + 1j*numpy.random.uniform(-5, 5)
            # SciPy thinks that factorial(z) is 0 when re(z) < 0 and
            # does not support complex numbers.
            # SymPy does not think so.
            if sympy_fn == factorial:
                tv = numpy.abs(tv)
            # SciPy supports gammaln for real arguments only,
            # and there is also a branch cut along the negative real axis
            if sympy_fn == loggamma:
                tv = numpy.abs(tv)
            # SymPy's digamma evaluates as polygamma(0, z)
            # which SciPy supports for real arguments only
            if sympy_fn == digamma:
                tv = numpy.real(tv)
            sympy_result = sympy_fn(tv).evalf()
            scipy_result = scipy_fn(tv)
            # SciPy's sici returns a tuple with both Si and Ci present in it
            # which needs to be unpacked
            if sympy_fn == Si:
                scipy_result = scipy_fn(tv)[0]
            if sympy_fn == Ci:
                scipy_result = scipy_fn(tv)[1]
            assert abs(f(tv) - sympy_result) < 1e-13*(1 + abs(sympy_result))
            assert abs(f(tv) - scipy_result) < 1e-13*(1 + abs(sympy_result))

    double_arg_sympy_fns = [RisingFactorial, besselj, bessely, besseli,
                            besselk, polygamma]
    double_arg_scipy_fns = [scipy.special.poch, scipy.special.jv,
                            scipy.special.yv, scipy.special.iv, scipy.special.kv, scipy.special.polygamma]
    for (sympy_fn, scipy_fn) in zip(double_arg_sympy_fns, double_arg_scipy_fns):
        f = lambdify((x, y), sympy_fn(x, y), modules="scipy")
        for i in range(20):
            # SciPy supports only real orders of Bessel functions
            tv1 = numpy.random.uniform(-10, 10)
            tv2 = numpy.random.uniform(-10, 10) + 1j*numpy.random.uniform(-5, 5)
            # SciPy requires a real valued 2nd argument for: poch, polygamma
            if sympy_fn in (RisingFactorial, polygamma):
                tv2 = numpy.real(tv2)
            if sympy_fn == polygamma:
                tv1 = abs(int(tv1))  # first argument to polygamma must be a non-negative integer.
            sympy_result = sympy_fn(tv1, tv2).evalf()
            assert abs(f(tv1, tv2) - sympy_result) < 1e-13*(1 + abs(sympy_result))
            assert abs(f(tv1, tv2) - scipy_fn(tv1, tv2)) < 1e-13*(1 + abs(sympy_result))


def test_scipy_polys():
    if not scipy:
        skip("scipy not installed")
    numpy.random.seed(0)

    params = symbols('n k a b')
    # list polynomials with the number of parameters
    polys = [
        (chebyshevt, 1),
        (chebyshevu, 1),
        (legendre, 1),
        (hermite, 1),
        (laguerre, 1),
        (gegenbauer, 2),
        (assoc_legendre, 2),
        (assoc_laguerre, 2),
        (jacobi, 3)
    ]

    msg = \
        "The random test of the function {func} with the arguments " \
        "{args} had failed because the SymPy result {sympy_result} " \
        "and SciPy result {scipy_result} had failed to converge " \
        "within the tolerance {tol} " \
        "(Actual absolute difference : {diff})"

    for sympy_fn, num_params in polys:
        args = params[:num_params] + (x,)
        f = lambdify(args, sympy_fn(*args))
        for _ in range(10):
            tn = numpy.random.randint(3, 10)
            tparams = tuple(numpy.random.uniform(0, 5, size=num_params-1))
            tv = numpy.random.uniform(-10, 10) + 1j*numpy.random.uniform(-5, 5)
            # SciPy supports hermite for real arguments only
            if sympy_fn == hermite:
                tv = numpy.real(tv)
            # assoc_legendre needs x in (-1, 1) and integer param at most n
            if sympy_fn == assoc_legendre:
                tv = numpy.random.uniform(-1, 1)
                tparams = tuple(numpy.random.randint(1, tn, size=1))

            vals = (tn,) + tparams + (tv,)
            scipy_result = f(*vals)
            sympy_result = sympy_fn(*vals).evalf()
            atol = 1e-9*(1 + abs(sympy_result))
            diff = abs(scipy_result - sympy_result)
            try:
                assert diff < atol
            except TypeError:
                raise AssertionError(
                    msg.format(
                        func=repr(sympy_fn),
                        args=repr(vals),
                        sympy_result=repr(sympy_result),
                        scipy_result=repr(scipy_result),
                        diff=diff,
                        tol=atol)
                    )


def test_lambdify_inspect():
    f = lambdify(x, x**2)
    # Test that inspect.getsource works but don't hard-code implementation
    # details
    assert 'x**2' in inspect.getsource(f)


def test_issue_14941():
    x, y = Dummy(), Dummy()

    # test dict
    f1 = lambdify([x, y], {x: 3, y: 3}, 'sympy')
    assert f1(2, 3) == {2: 3, 3: 3}

    # test tuple
    f2 = lambdify([x, y], (y, x), 'sympy')
    assert f2(2, 3) == (3, 2)
    f2b = lambdify([], (1,))  # gh-23224
    assert f2b() == (1,)

    # test list
    f3 = lambdify([x, y], [y, x], 'sympy')
    assert f3(2, 3) == [3, 2]


def test_lambdify_Derivative_arg_issue_16468():
    f = Function('f')(x)
    fx = f.diff()
    assert lambdify((f, fx), f + fx)(10, 5) == 15
    assert eval(lambdastr((f, fx), f/fx))(10, 5) == 2
    raises(Exception, lambda:
        eval(lambdastr((f, fx), f/fx, dummify=False)))
    assert eval(lambdastr((f, fx), f/fx, dummify=True))(10, 5) == 2
    assert eval(lambdastr((fx, f), f/fx, dummify=True))(S(10), 5) == S.Half
    assert lambdify(fx, 1 + fx)(41) == 42
    assert eval(lambdastr(fx, 1 + fx, dummify=True))(41) == 42


def test_imag_real():
    f_re = lambdify([z], sympy.re(z))
    val = 3+2j
    assert f_re(val) == val.real

    f_im = lambdify([z], sympy.im(z))  # see #15400
    assert f_im(val) == val.imag


def test_MatrixSymbol_issue_15578():
    if not numpy:
        skip("numpy not installed")
    A = MatrixSymbol('A', 2, 2)
    A0 = numpy.array([[1, 2], [3, 4]])
    f = lambdify(A, A**(-1))
    assert numpy.allclose(f(A0), numpy.array([[-2., 1.], [1.5, -0.5]]))
    g = lambdify(A, A**3)
    assert numpy.allclose(g(A0), numpy.array([[37, 54], [81, 118]]))


def test_issue_15654():
    if not scipy:
        skip("scipy not installed")
    from sympy.abc import n, l, r, Z
    from sympy.physics import hydrogen
    nv, lv, rv, Zv = 1, 0, 3, 1
    sympy_value = hydrogen.R_nl(nv, lv, rv, Zv).evalf()
    f = lambdify((n, l, r, Z), hydrogen.R_nl(n, l, r, Z))
    scipy_value = f(nv, lv, rv, Zv)
    assert abs(sympy_value - scipy_value) < 1e-15


def test_issue_15827():
    if not numpy:
        skip("numpy not installed")
    A = MatrixSymbol("A", 3, 3)
    B = MatrixSymbol("B", 2, 3)
    C = MatrixSymbol("C", 3, 4)
    D = MatrixSymbol("D", 4, 5)
    k=symbols("k")
    f = lambdify(A, (2*k)*A)
    g = lambdify(A, (2+k)*A)
    h = lambdify(A, 2*A)
    i = lambdify((B, C, D), 2*B*C*D)
    assert numpy.array_equal(f(numpy.array([[1, 2, 3], [1, 2, 3], [1, 2, 3]])), \
    numpy.array([[2*k, 4*k, 6*k], [2*k, 4*k, 6*k], [2*k, 4*k, 6*k]], dtype=object))

    assert numpy.array_equal(g(numpy.array([[1, 2, 3], [1, 2, 3], [1, 2, 3]])), \
    numpy.array([[k + 2, 2*k + 4, 3*k + 6], [k + 2, 2*k + 4, 3*k + 6], \
    [k + 2, 2*k + 4, 3*k + 6]], dtype=object))

    assert numpy.array_equal(h(numpy.array([[1, 2, 3], [1, 2, 3], [1, 2, 3]])), \
    numpy.array([[2, 4, 6], [2, 4, 6], [2, 4, 6]]))

    assert numpy.array_equal(i(numpy.array([[1, 2, 3], [1, 2, 3]]), numpy.array([[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]]), \
    numpy.array([[1, 2, 3, 4, 5], [1, 2, 3, 4, 5], [1, 2, 3, 4, 5], [1, 2, 3, 4, 5]])), numpy.array([[ 120, 240, 360, 480, 600], \
    [ 120, 240, 360, 480, 600]]))


def test_issue_16930():
    if not scipy:
        skip("scipy not installed")

    x = symbols("x")
    f = lambda x:  S.GoldenRatio * x**2
    f_ = lambdify(x, f(x), modules='scipy')
    assert f_(1) == scipy.constants.golden_ratio

def test_issue_17898():
    if not scipy:
        skip("scipy not installed")
    x = symbols("x")
    f_ = lambdify([x], sympy.LambertW(x,-1), modules='scipy')
    assert f_(0.1) == mpmath.lambertw(0.1, -1)

def test_issue_13167_21411():
    if not numpy:
        skip("numpy not installed")
    f1 = lambdify(x, sympy.Heaviside(x))
    f2 = lambdify(x, sympy.Heaviside(x, 1))
    res1 = f1([-1, 0, 1])
    res2 = f2([-1, 0, 1])
    assert Abs(res1[0]).n() < 1e-15        # First functionality: only one argument passed
    assert Abs(res1[1] - 1/2).n() < 1e-15
    assert Abs(res1[2] - 1).n() < 1e-15
    assert Abs(res2[0]).n() < 1e-15        # Second functionality: two arguments passed
    assert Abs(res2[1] - 1).n() < 1e-15
    assert Abs(res2[2] - 1).n() < 1e-15

def test_single_e():
    f = lambdify(x, E)
    assert f(23) == exp(1.0)

def test_issue_16536():
    if not scipy:
        skip("scipy not installed")

    a = symbols('a')
    f1 = lowergamma(a, x)
    F = lambdify((a, x), f1, modules='scipy')
    assert abs(lowergamma(1, 3) - F(1, 3)) <= 1e-10

    f2 = uppergamma(a, x)
    F = lambdify((a, x), f2, modules='scipy')
    assert abs(uppergamma(1, 3) - F(1, 3)) <= 1e-10


def test_issue_22726():
    if not numpy:
        skip("numpy not installed")

    x1, x2 = symbols('x1 x2')
    f = Max(S.Zero, Min(x1, x2))
    g = derive_by_array(f, (x1, x2))
    G = lambdify((x1, x2), g, modules='numpy')
    point = {x1: 1, x2: 2}
    assert (abs(g.subs(point) - G(*point.values())) <= 1e-10).all()


def test_issue_22739():
    if not numpy:
        skip("numpy not installed")

    x1, x2 = symbols('x1 x2')
    f = Heaviside(Min(x1, x2))
    F = lambdify((x1, x2), f, modules='numpy')
    point = {x1: 1, x2: 2}
    assert abs(f.subs(point) - F(*point.values())) <= 1e-10


def test_issue_22992():
    if not numpy:
        skip("numpy not installed")

    a, t = symbols('a t')
    expr = a*(log(cot(t/2)) - cos(t))
    F = lambdify([a, t], expr, 'numpy')

    point = {a: 10, t: 2}

    assert abs(expr.subs(point) - F(*point.values())) <= 1e-10

    # Standard math
    F = lambdify([a, t], expr)

    assert abs(expr.subs(point) - F(*point.values())) <= 1e-10


def test_issue_19764():
    if not numpy:
        skip("numpy not installed")

    expr = Array([x, x**2])
    f = lambdify(x, expr, 'numpy')

    assert f(1).__class__ == numpy.ndarray

def test_issue_20070():
    if not numba:
        skip("numba not installed")

    f = lambdify(x, sin(x), 'numpy')
    assert numba.jit(f, nopython=True)(1)==0.8414709848078965


def test_fresnel_integrals_scipy():
    if not scipy:
        skip("scipy not installed")

    f1 = fresnelc(x)
    f2 = fresnels(x)
    F1 = lambdify(x, f1, modules='scipy')
    F2 = lambdify(x, f2, modules='scipy')

    assert abs(fresnelc(1.3) - F1(1.3)) <= 1e-10
    assert abs(fresnels(1.3) - F2(1.3)) <= 1e-10


def test_beta_scipy():
    if not scipy:
        skip("scipy not installed")

    f = beta(x, y)
    F = lambdify((x, y), f, modules='scipy')

    assert abs(beta(1.3, 2.3) - F(1.3, 2.3)) <= 1e-10


def test_beta_math():
    f = beta(x, y)
    F = lambdify((x, y), f, modules='math')

    assert abs(beta(1.3, 2.3) - F(1.3, 2.3)) <= 1e-10


def test_betainc_scipy():
    if not scipy:
        skip("scipy not installed")

    f = betainc(w, x, y, z)
    F = lambdify((w, x, y, z), f, modules='scipy')

    assert abs(betainc(1.4, 3.1, 0.1, 0.5) - F(1.4, 3.1, 0.1, 0.5)) <= 1e-10


def test_betainc_regularized_scipy():
    if not scipy:
        skip("scipy not installed")

    f = betainc_regularized(w, x, y, z)
    F = lambdify((w, x, y, z), f, modules='scipy')

    assert abs(betainc_regularized(0.2, 3.5, 0.1, 1) - F(0.2, 3.5, 0.1, 1)) <= 1e-10


def test_numpy_special_math():
    if not numpy:
        skip("numpy not installed")

    funcs = [expm1, log1p, exp2, log2, log10, hypot, logaddexp, logaddexp2]
    for func in funcs:
        if 2 in func.nargs:
            expr = func(x, y)
            args = (x, y)
            num_args = (0.3, 0.4)
        elif 1 in func.nargs:
            expr = func(x)
            args = (x,)
            num_args = (0.3,)
        else:
            raise NotImplementedError("Need to handle other than unary & binary functions in test")
        f = lambdify(args, expr)
        result = f(*num_args)
        reference = expr.subs(dict(zip(args, num_args))).evalf()
        assert numpy.allclose(result, float(reference))

    lae2 = lambdify((x, y), logaddexp2(log2(x), log2(y)))
    assert abs(2.0**lae2(1e-50, 2.5e-50) - 3.5e-50) < 1e-62  # from NumPy's docstring


def test_scipy_special_math():
    if not scipy:
        skip("scipy not installed")

    cm1 = lambdify((x,), cosm1(x), modules='scipy')
    assert abs(cm1(1e-20) + 5e-41) < 1e-200

    have_scipy_1_10plus = tuple(map(int, scipy.version.version.split('.')[:2])) >= (1, 10)

    if have_scipy_1_10plus:
        cm2 = lambdify((x, y), powm1(x, y), modules='scipy')
        assert abs(cm2(1.2, 1e-9) - 1.82321557e-10)  < 1e-17


def test_scipy_bernoulli():
    if not scipy:
        skip("scipy not installed")

    bern = lambdify((x,), bernoulli(x), modules='scipy')
    assert bern(1) == 0.5


def test_scipy_harmonic():
    if not scipy:
        skip("scipy not installed")

    hn = lambdify((x,), harmonic(x), modules='scipy')
    assert hn(2) == 1.5
    hnm = lambdify((x, y), harmonic(x, y), modules='scipy')
    assert hnm(2, 2) == 1.25


def test_cupy_array_arg():
    if not cupy:
        skip("CuPy not installed")

    f = lambdify([[x, y]], x*x + y, 'cupy')
    result = f(cupy.array([2.0, 1.0]))
    assert result == 5
    assert "cupy" in str(type(result))


def test_cupy_array_arg_using_numpy():
    # numpy functions can be run on cupy arrays
    # unclear if we can "officially" support this,
    # depends on numpy __array_function__ support
    if not cupy:
        skip("CuPy not installed")

    f = lambdify([[x, y]], x*x + y, 'numpy')
    result = f(cupy.array([2.0, 1.0]))
    assert result == 5
    assert "cupy" in str(type(result))

def test_cupy_dotproduct():
    if not cupy:
        skip("CuPy not installed")

    A = Matrix([x, y, z])
    f1 = lambdify([x, y, z], DotProduct(A, A), modules='cupy')
    f2 = lambdify([x, y, z], DotProduct(A, A.T), modules='cupy')
    f3 = lambdify([x, y, z], DotProduct(A.T, A), modules='cupy')
    f4 = lambdify([x, y, z], DotProduct(A, A.T), modules='cupy')

    assert f1(1, 2, 3) == \
        f2(1, 2, 3) == \
        f3(1, 2, 3) == \
        f4(1, 2, 3) == \
        cupy.array([14])


def test_jax_array_arg():
    if not jax:
        skip("JAX not installed")

    f = lambdify([[x, y]], x*x + y, 'jax')
    result = f(jax.numpy.array([2.0, 1.0]))
    assert result == 5
    assert "jax" in str(type(result))


def test_jax_array_arg_using_numpy():
    if not jax:
        skip("JAX not installed")

    f = lambdify([[x, y]], x*x + y, 'numpy')
    result = f(jax.numpy.array([2.0, 1.0]))
    assert result == 5
    assert "jax" in str(type(result))


def test_jax_dotproduct():
    if not jax:
        skip("JAX not installed")

    A = Matrix([x, y, z])
    f1 = lambdify([x, y, z], DotProduct(A, A), modules='jax')
    f2 = lambdify([x, y, z], DotProduct(A, A.T), modules='jax')
    f3 = lambdify([x, y, z], DotProduct(A.T, A), modules='jax')
    f4 = lambdify([x, y, z], DotProduct(A, A.T), modules='jax')

    assert f1(1, 2, 3) == \
        f2(1, 2, 3) == \
        f3(1, 2, 3) == \
        f4(1, 2, 3) == \
        jax.numpy.array([14])


def test_lambdify_cse():
    def no_op_cse(exprs):
        return (), exprs

    def dummy_cse(exprs):
        from sympy.simplify.cse_main import cse
        return cse(exprs, symbols=numbered_symbols(cls=Dummy))

    def minmem(exprs):
        from sympy.simplify.cse_main import cse_release_variables, cse
        return cse(exprs, postprocess=cse_release_variables)

    class Case:
        def __init__(self, *, args, exprs, num_args, requires_numpy=False):
            self.args = args
            self.exprs = exprs
            self.num_args = num_args
            subs_dict = dict(zip(self.args, self.num_args))
            self.ref = [e.subs(subs_dict).evalf() for e in exprs]
            self.requires_numpy = requires_numpy

        def lambdify(self, *, cse):
            return lambdify(self.args, self.exprs, cse=cse)

        def assertAllClose(self, result, *, abstol=1e-15, reltol=1e-15):
            if self.requires_numpy:
                assert all(numpy.allclose(result[i], numpy.asarray(r, dtype=float),
                                          rtol=reltol, atol=abstol)
                           for i, r in enumerate(self.ref))
                return

            for i, r in enumerate(self.ref):
                abs_err = abs(result[i] - r)
                if r == 0:
                    assert abs_err < abstol
                else:
                    assert abs_err/abs(r) < reltol

    cases = [
        Case(
            args=(x, y, z),
            exprs=[
             x + y + z,
             x + y - z,
             2*x + 2*y - z,
             (x+y)**2 + (y+z)**2,
            ],
            num_args=(2., 3., 4.)
        ),
        Case(
            args=(x, y, z),
            exprs=[
            x + sympy.Heaviside(x),
            y + sympy.Heaviside(x),
            z + sympy.Heaviside(x, 1),
            z/sympy.Heaviside(x, 1)
            ],
            num_args=(0., 3., 4.)
        ),
        Case(
            args=(x, y, z),
            exprs=[
            x + sinc(y),
            y + sinc(y),
            z - sinc(y)
            ],
            num_args=(0.1, 0.2, 0.3)
        ),
        Case(
            args=(x, y, z),
            exprs=[
                Matrix([[x, x*y], [sin(z) + 4, x**z]]),
                x*y+sin(z)-x**z,
                Matrix([x*x, sin(z), x**z])
            ],
            num_args=(1.,2.,3.),
            requires_numpy=True
        ),
        Case(
            args=(x, y),
            exprs=[(x + y - 1)**2, x, x + y,
            (x + y)/(2*x + 1) + (x + y - 1)**2, (2*x + 1)**(x + y)],
            num_args=(1,2)
        )
    ]
    for case in cases:
        if not numpy and case.requires_numpy:
            continue
        for _cse in [False, True, minmem, no_op_cse, dummy_cse]:
            f = case.lambdify(cse=_cse)
            result = f(*case.num_args)
            case.assertAllClose(result)

def test_issue_25288():
    syms = numbered_symbols(cls=Dummy)
    ok = lambdify(x, [x**2, sin(x**2)], cse=lambda e: cse(e, symbols=syms))(2)
    assert ok


def test_deprecated_set():
    with warns_deprecated_sympy():
        lambdify({x, y}, x + y)

def test_issue_13881():
    if not numpy:
        skip("numpy not installed.")

    X = MatrixSymbol('X', 3, 1)

    f = lambdify(X, X.T*X, 'numpy')
    assert f(numpy.array([1, 2, 3])) == 14
    assert f(numpy.array([3, 2, 1])) == 14

    f = lambdify(X, X*X.T, 'numpy')
    assert f(numpy.array([1, 2, 3])) == 14
    assert f(numpy.array([3, 2, 1])) == 14

    f = lambdify(X, (X*X.T)*X, 'numpy')
    arr1 = numpy.array([[1], [2], [3]])
    arr2 = numpy.array([[14],[28],[42]])

    assert numpy.array_equal(f(arr1), arr2)


def test_23536_lambdify_cse_dummy():

    f = Function('x')(y)
    g = Function('w')(y)
    expr = z + (f**4 + g**5)*(f**3 + (g*f)**3)
    expr = expr.expand()
    eval_expr = lambdify(((f, g), z), expr, cse=True)
    ans = eval_expr((1.0, 2.0), 3.0)  # shouldn't raise NameError
    assert ans == 300.0  # not a list and value is 300


class LambdifyDocstringTestCase:
    SIGNATURE = None
    EXPR = None
    SRC = None

    def __init__(self, docstring_limit, expected_redacted):
        self.docstring_limit = docstring_limit
        self.expected_redacted = expected_redacted

    @property
    def expected_expr(self):
        expr_redacted_msg = "EXPRESSION REDACTED DUE TO LENGTH, (see lambdify's `docstring_limit`)"
        return self.EXPR if not self.expected_redacted else expr_redacted_msg

    @property
    def expected_src(self):
        src_redacted_msg = "SOURCE CODE REDACTED DUE TO LENGTH, (see lambdify's `docstring_limit`)"
        return self.SRC if not self.expected_redacted else src_redacted_msg

    @property
    def expected_docstring(self):
        expected_docstring = (
            f'Created with lambdify. Signature:\n\n'
            f'func({self.SIGNATURE})\n\n'
            f'Expression:\n\n'
            f'{self.expected_expr}\n\n'
            f'Source code:\n\n'
            f'{self.expected_src}\n\n'
            f'Imported modules:\n\n'
        )
        return expected_docstring

    def __len__(self):
        return len(self.expected_docstring)

    def __repr__(self):
        return (
            f'{self.__class__.__name__}('
            f'docstring_limit={self.docstring_limit}, '
            f'expected_redacted={self.expected_redacted})'
        )


def test_lambdify_docstring_size_limit_simple_symbol():

    class SimpleSymbolTestCase(LambdifyDocstringTestCase):
        SIGNATURE = 'x'
        EXPR = 'x'
        SRC = (
            'def _lambdifygenerated(x):\n'
            '    return x\n'
        )

    x = symbols('x')

    test_cases = (
        SimpleSymbolTestCase(docstring_limit=None, expected_redacted=False),
        SimpleSymbolTestCase(docstring_limit=100, expected_redacted=False),
        SimpleSymbolTestCase(docstring_limit=1, expected_redacted=False),
        SimpleSymbolTestCase(docstring_limit=0, expected_redacted=True),
        SimpleSymbolTestCase(docstring_limit=-1, expected_redacted=True),
    )
    for test_case in test_cases:
        lambdified_expr = lambdify(
            [x],
            x,
            'sympy',
            docstring_limit=test_case.docstring_limit,
        )
        assert lambdified_expr.__doc__ == test_case.expected_docstring


def test_lambdify_docstring_size_limit_nested_expr():

    class ExprListTestCase(LambdifyDocstringTestCase):
        SIGNATURE = 'x, y, z'
        EXPR = (
            '[x, [y], z, x**3 + 3*x**2*y + 3*x**2*z + 3*x*y**2 + 6*x*y*z '
            '+ 3*x*z**2 +...'
        )
        SRC = (
            'def _lambdifygenerated(x, y, z):\n'
            '    return [x, [y], z, x**3 + 3*x**2*y + 3*x**2*z + 3*x*y**2 '
            '+ 6*x*y*z + 3*x*z**2 + y**3 + 3*y**2*z + 3*y*z**2 + z**3]\n'
        )

    x, y, z = symbols('x, y, z')
    expr = [x, [y], z, ((x + y + z)**3).expand()]

    test_cases = (
        ExprListTestCase(docstring_limit=None, expected_redacted=False),
        ExprListTestCase(docstring_limit=200, expected_redacted=False),
        ExprListTestCase(docstring_limit=50, expected_redacted=True),
        ExprListTestCase(docstring_limit=0, expected_redacted=True),
        ExprListTestCase(docstring_limit=-1, expected_redacted=True),
    )
    for test_case in test_cases:
        lambdified_expr = lambdify(
            [x, y, z],
            expr,
            'sympy',
            docstring_limit=test_case.docstring_limit,
        )
        assert lambdified_expr.__doc__ == test_case.expected_docstring


def test_lambdify_docstring_size_limit_matrix():

    class MatrixTestCase(LambdifyDocstringTestCase):
        SIGNATURE = 'x, y, z'
        EXPR = (
            'Matrix([[0, x], [x + y + z, x**3 + 3*x**2*y + 3*x**2*z + 3*x*y**2 '
            '+ 6*x*y*z...'
        )
        SRC = (
            'def _lambdifygenerated(x, y, z):\n'
            '    return ImmutableDenseMatrix([[0, x], [x + y + z, x**3 '
            '+ 3*x**2*y + 3*x**2*z + 3*x*y**2 + 6*x*y*z + 3*x*z**2 + y**3 '
            '+ 3*y**2*z + 3*y*z**2 + z**3]])\n'
        )

    x, y, z = symbols('x, y, z')
    expr = Matrix([[S.Zero, x], [x + y + z, ((x + y + z)**3).expand()]])

    test_cases = (
        MatrixTestCase(docstring_limit=None, expected_redacted=False),
        MatrixTestCase(docstring_limit=200, expected_redacted=False),
        MatrixTestCase(docstring_limit=50, expected_redacted=True),
        MatrixTestCase(docstring_limit=0, expected_redacted=True),
        MatrixTestCase(docstring_limit=-1, expected_redacted=True),
    )
    for test_case in test_cases:
        lambdified_expr = lambdify(
            [x, y, z],
            expr,
            'sympy',
            docstring_limit=test_case.docstring_limit,
        )
        assert lambdified_expr.__doc__ == test_case.expected_docstring


def test_lambdify_empty_tuple():
    a = symbols("a")
    expr = ((), (a,))
    f = lambdify(a, expr)
    result = f(1)
    assert result == ((), (1,)), "Lambdify did not handle the empty tuple correctly."

def test_assoc_legendre_numerical_evaluation():

    tol = 1e-10

    sympy_result_integer = assoc_legendre(1, 1/2, 0.1).evalf()
    sympy_result_complex = assoc_legendre(2, 1, 3).evalf()
    mpmath_result_integer = -0.474572528387641
    mpmath_result_complex = -25.45584412271571*I

    assert all_close(sympy_result_integer, mpmath_result_integer, tol)
    assert all_close(sympy_result_complex, mpmath_result_complex, tol)
