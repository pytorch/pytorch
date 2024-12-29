import tempfile
from sympy.core.numbers import pi, Rational
from sympy.core.power import Pow
from sympy.core.singleton import S
from sympy.core.symbol import Symbol
from sympy.functions.elementary.complexes import Abs
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.trigonometric import (cos, sin, sinc)
from sympy.matrices.expressions.matexpr import MatrixSymbol
from sympy.assumptions import assuming, Q
from sympy.external import import_module
from sympy.printing.codeprinter import ccode
from sympy.codegen.matrix_nodes import MatrixSolve
from sympy.codegen.cfunctions import log2, exp2, expm1, log1p
from sympy.codegen.numpy_nodes import logaddexp, logaddexp2
from sympy.codegen.scipy_nodes import cosm1, powm1
from sympy.codegen.rewriting import (
    optimize, cosm1_opt, log2_opt, exp2_opt, expm1_opt, log1p_opt, powm1_opt, optims_c99,
    create_expand_pow_optimization, matinv_opt, logaddexp_opt, logaddexp2_opt,
    optims_numpy, optims_scipy, sinc_opts, FuncMinusOneOptim
)
from sympy.testing.pytest import XFAIL, skip
from sympy.utilities import lambdify
from sympy.utilities._compilation import compile_link_import_strings, has_c
from sympy.utilities._compilation.util import may_xfail

cython = import_module('cython')
numpy = import_module('numpy')
scipy = import_module('scipy')


def test_log2_opt():
    x = Symbol('x')
    expr1 = 7*log(3*x + 5)/(log(2))
    opt1 = optimize(expr1, [log2_opt])
    assert opt1 == 7*log2(3*x + 5)
    assert opt1.rewrite(log) == expr1

    expr2 = 3*log(5*x + 7)/(13*log(2))
    opt2 = optimize(expr2, [log2_opt])
    assert opt2 == 3*log2(5*x + 7)/13
    assert opt2.rewrite(log) == expr2

    expr3 = log(x)/log(2)
    opt3 = optimize(expr3, [log2_opt])
    assert opt3 == log2(x)
    assert opt3.rewrite(log) == expr3

    expr4 = log(x)/log(2) + log(x+1)
    opt4 = optimize(expr4, [log2_opt])
    assert opt4 == log2(x) + log(2)*log2(x+1)
    assert opt4.rewrite(log) == expr4

    expr5 = log(17)
    opt5 = optimize(expr5, [log2_opt])
    assert opt5 == expr5

    expr6 = log(x + 3)/log(2)
    opt6 = optimize(expr6, [log2_opt])
    assert str(opt6) == 'log2(x + 3)'
    assert opt6.rewrite(log) == expr6


def test_exp2_opt():
    x = Symbol('x')
    expr1 = 1 + 2**x
    opt1 = optimize(expr1, [exp2_opt])
    assert opt1 == 1 + exp2(x)
    assert opt1.rewrite(Pow) == expr1

    expr2 = 1 + 3**x
    assert expr2 == optimize(expr2, [exp2_opt])


def test_expm1_opt():
    x = Symbol('x')

    expr1 = exp(x) - 1
    opt1 = optimize(expr1, [expm1_opt])
    assert expm1(x) - opt1 == 0
    assert opt1.rewrite(exp) == expr1

    expr2 = 3*exp(x) - 3
    opt2 = optimize(expr2, [expm1_opt])
    assert 3*expm1(x) == opt2
    assert opt2.rewrite(exp) == expr2

    expr3 = 3*exp(x) - 5
    opt3 = optimize(expr3, [expm1_opt])
    assert 3*expm1(x) - 2 == opt3
    assert opt3.rewrite(exp) == expr3
    expm1_opt_non_opportunistic = FuncMinusOneOptim(exp, expm1, opportunistic=False)
    assert expr3 == optimize(expr3, [expm1_opt_non_opportunistic])
    assert opt1 == optimize(expr1, [expm1_opt_non_opportunistic])
    assert opt2 == optimize(expr2, [expm1_opt_non_opportunistic])

    expr4 = 3*exp(x) + log(x) - 3
    opt4 = optimize(expr4, [expm1_opt])
    assert 3*expm1(x) + log(x) == opt4
    assert opt4.rewrite(exp) == expr4

    expr5 = 3*exp(2*x) - 3
    opt5 = optimize(expr5, [expm1_opt])
    assert 3*expm1(2*x) == opt5
    assert opt5.rewrite(exp) == expr5

    expr6 = (2*exp(x) + 1)/(exp(x) + 1) + 1
    opt6 = optimize(expr6, [expm1_opt])
    assert opt6.count_ops() <= expr6.count_ops()

    def ev(e):
        return e.subs(x, 3).evalf()
    assert abs(ev(expr6) - ev(opt6)) < 1e-15

    y = Symbol('y')
    expr7 = (2*exp(x) - 1)/(1 - exp(y)) - 1/(1-exp(y))
    opt7 = optimize(expr7, [expm1_opt])
    assert -2*expm1(x)/expm1(y) == opt7
    assert (opt7.rewrite(exp) - expr7).factor() == 0

    expr8 = (1+exp(x))**2 - 4
    opt8 = optimize(expr8, [expm1_opt])
    tgt8a = (exp(x) + 3)*expm1(x)
    tgt8b = 2*expm1(x) + expm1(2*x)
    # Both tgt8a & tgt8b seem to give full precision (~16 digits for double)
    # for x=1e-7 (compare with expr8 which only achieves ~8 significant digits).
    # If we can show that either tgt8a or tgt8b is preferable, we can
    # change this test to ensure the preferable version is returned.
    assert (tgt8a - tgt8b).rewrite(exp).factor() == 0
    assert opt8 in (tgt8a, tgt8b)
    assert (opt8.rewrite(exp) - expr8).factor() == 0

    expr9 = sin(expr8)
    opt9 = optimize(expr9, [expm1_opt])
    tgt9a = sin(tgt8a)
    tgt9b = sin(tgt8b)
    assert opt9 in (tgt9a, tgt9b)
    assert (opt9.rewrite(exp) - expr9.rewrite(exp)).factor().is_zero


def test_expm1_two_exp_terms():
    x, y = map(Symbol, 'x y'.split())
    expr1 = exp(x) + exp(y) - 2
    opt1 = optimize(expr1, [expm1_opt])
    assert opt1 == expm1(x) + expm1(y)


def test_cosm1_opt():
    x = Symbol('x')

    expr1 = cos(x) - 1
    opt1 = optimize(expr1, [cosm1_opt])
    assert cosm1(x) - opt1 == 0
    assert opt1.rewrite(cos) == expr1

    expr2 = 3*cos(x) - 3
    opt2 = optimize(expr2, [cosm1_opt])
    assert 3*cosm1(x) == opt2
    assert opt2.rewrite(cos) == expr2

    expr3 = 3*cos(x) - 5
    opt3 = optimize(expr3, [cosm1_opt])
    assert 3*cosm1(x) - 2 == opt3
    assert opt3.rewrite(cos) == expr3
    cosm1_opt_non_opportunistic = FuncMinusOneOptim(cos, cosm1, opportunistic=False)
    assert expr3 == optimize(expr3, [cosm1_opt_non_opportunistic])
    assert opt1 == optimize(expr1, [cosm1_opt_non_opportunistic])
    assert opt2 == optimize(expr2, [cosm1_opt_non_opportunistic])

    expr4 = 3*cos(x) + log(x) - 3
    opt4 = optimize(expr4, [cosm1_opt])
    assert 3*cosm1(x) + log(x) == opt4
    assert opt4.rewrite(cos) == expr4

    expr5 = 3*cos(2*x) - 3
    opt5 = optimize(expr5, [cosm1_opt])
    assert 3*cosm1(2*x) == opt5
    assert opt5.rewrite(cos) == expr5

    expr6 = 2 - 2*cos(x)
    opt6 = optimize(expr6, [cosm1_opt])
    assert -2*cosm1(x) == opt6
    assert opt6.rewrite(cos) == expr6


def test_cosm1_two_cos_terms():
    x, y = map(Symbol, 'x y'.split())
    expr1 = cos(x) + cos(y) - 2
    opt1 = optimize(expr1, [cosm1_opt])
    assert opt1 == cosm1(x) + cosm1(y)


def test_expm1_cosm1_mixed():
    x = Symbol('x')
    expr1 = exp(x) + cos(x) - 2
    opt1 = optimize(expr1, [expm1_opt, cosm1_opt])
    assert opt1 == cosm1(x) + expm1(x)


def _check_num_lambdify(expr, opt, val_subs, approx_ref, lambdify_kw=None, poorness=1e10):
    """ poorness=1e10 signifies that `expr` loses precision of at least ten decimal digits. """
    num_ref = expr.subs(val_subs).evalf()
    eps = numpy.finfo(numpy.float64).eps
    assert abs(num_ref - approx_ref) < approx_ref*eps
    f1 = lambdify(list(val_subs.keys()), opt, **(lambdify_kw or {}))
    args_float = tuple(map(float, val_subs.values()))
    num_err1 = abs(f1(*args_float) - approx_ref)
    assert num_err1 < abs(num_ref*eps)
    f2 = lambdify(list(val_subs.keys()), expr, **(lambdify_kw or {}))
    num_err2 = abs(f2(*args_float) - approx_ref)
    assert num_err2 > abs(num_ref*eps*poorness)   # this only ensures that the *test* works as intended


def test_cosm1_apart():
    x = Symbol('x')

    expr1 = 1/cos(x) - 1
    opt1 = optimize(expr1, [cosm1_opt])
    assert opt1 == -cosm1(x)/cos(x)
    if scipy:
        _check_num_lambdify(expr1, opt1, {x: S(10)**-30}, 5e-61, lambdify_kw={"modules": 'scipy'})

    expr2 = 2/cos(x) - 2
    opt2 = optimize(expr2, optims_scipy)
    assert opt2 == -2*cosm1(x)/cos(x)
    if scipy:
        _check_num_lambdify(expr2, opt2, {x: S(10)**-30}, 1e-60, lambdify_kw={"modules": 'scipy'})

    expr3 = pi/cos(3*x) - pi
    opt3 = optimize(expr3, [cosm1_opt])
    assert opt3 == -pi*cosm1(3*x)/cos(3*x)
    if scipy:
        _check_num_lambdify(expr3, opt3, {x: S(10)**-30/3}, float(5e-61*pi), lambdify_kw={"modules": 'scipy'})


def test_powm1():
    args = x, y = map(Symbol, "xy")

    expr1 = x**y - 1
    opt1 = optimize(expr1, [powm1_opt])
    assert opt1 == powm1(x, y)
    for arg in args:
        assert expr1.diff(arg) == opt1.diff(arg)
    if scipy and tuple(map(int, scipy.version.version.split('.')[:3])) >= (1, 10, 0):
        subs1_a = {x: Rational(*(1.0+1e-13).as_integer_ratio()), y: pi}
        ref1_f64_a = 3.139081648208105e-13
        _check_num_lambdify(expr1, opt1, subs1_a, ref1_f64_a, lambdify_kw={"modules": 'scipy'}, poorness=10**11)

        subs1_b = {x: pi, y: Rational(*(1e-10).as_integer_ratio())}
        ref1_f64_b = 1.1447298859149205e-10
        _check_num_lambdify(expr1, opt1, subs1_b, ref1_f64_b, lambdify_kw={"modules": 'scipy'}, poorness=10**9)


def test_log1p_opt():
    x = Symbol('x')
    expr1 = log(x + 1)
    opt1 = optimize(expr1, [log1p_opt])
    assert log1p(x) - opt1 == 0
    assert opt1.rewrite(log) == expr1

    expr2 = log(3*x + 3)
    opt2 = optimize(expr2, [log1p_opt])
    assert log1p(x) + log(3) == opt2
    assert (opt2.rewrite(log) - expr2).simplify() == 0

    expr3 = log(2*x + 1)
    opt3 = optimize(expr3, [log1p_opt])
    assert log1p(2*x) - opt3 == 0
    assert opt3.rewrite(log) == expr3

    expr4 = log(x+3)
    opt4 = optimize(expr4, [log1p_opt])
    assert str(opt4) == 'log(x + 3)'


def test_optims_c99():
    x = Symbol('x')

    expr1 = 2**x + log(x)/log(2) + log(x + 1) + exp(x) - 1
    opt1 = optimize(expr1, optims_c99).simplify()
    assert opt1 == exp2(x) + log2(x) + log1p(x) + expm1(x)
    assert opt1.rewrite(exp).rewrite(log).rewrite(Pow) == expr1

    expr2 = log(x)/log(2) + log(x + 1)
    opt2 = optimize(expr2, optims_c99)
    assert opt2 == log2(x) + log1p(x)
    assert opt2.rewrite(log) == expr2

    expr3 = log(x)/log(2) + log(17*x + 17)
    opt3 = optimize(expr3, optims_c99)
    delta3 = opt3 - (log2(x) + log(17) + log1p(x))
    assert delta3 == 0
    assert (opt3.rewrite(log) - expr3).simplify() == 0

    expr4 = 2**x + 3*log(5*x + 7)/(13*log(2)) + 11*exp(x) - 11 + log(17*x + 17)
    opt4 = optimize(expr4, optims_c99).simplify()
    delta4 = opt4 - (exp2(x) + 3*log2(5*x + 7)/13 + 11*expm1(x) + log(17) + log1p(x))
    assert delta4 == 0
    assert (opt4.rewrite(exp).rewrite(log).rewrite(Pow) - expr4).simplify() == 0

    expr5 = 3*exp(2*x) - 3
    opt5 = optimize(expr5, optims_c99)
    delta5 = opt5 - 3*expm1(2*x)
    assert delta5 == 0
    assert opt5.rewrite(exp) == expr5

    expr6 = exp(2*x) - 3
    opt6 = optimize(expr6, optims_c99)
    assert opt6 in (expm1(2*x) - 2, expr6)  # expm1(2*x) - 2 is not better or worse

    expr7 = log(3*x + 3)
    opt7 = optimize(expr7, optims_c99)
    delta7 = opt7 - (log(3) + log1p(x))
    assert delta7 == 0
    assert (opt7.rewrite(log) - expr7).simplify() == 0

    expr8 = log(2*x + 3)
    opt8 = optimize(expr8, optims_c99)
    assert opt8 == expr8


def test_create_expand_pow_optimization():
    cc = lambda x: ccode(
        optimize(x, [create_expand_pow_optimization(4)]))
    x = Symbol('x')
    assert cc(x**4) == 'x*x*x*x'
    assert cc(x**4 + x**2) == 'x*x + x*x*x*x'
    assert cc(x**5 + x**4) == 'pow(x, 5) + x*x*x*x'
    assert cc(sin(x)**4) == 'pow(sin(x), 4)'
    # gh issue 15335
    assert cc(x**(-4)) == '1.0/(x*x*x*x)'
    assert cc(x**(-5)) == 'pow(x, -5)'
    assert cc(-x**4) == '-(x*x*x*x)'
    assert cc(x**4 - x**2) == '-(x*x) + x*x*x*x'
    i = Symbol('i', integer=True)
    assert cc(x**i - x**2) == 'pow(x, i) - (x*x)'
    y = Symbol('y', real=True)
    assert cc(Abs(exp(y**4))) == "exp(y*y*y*y)"

    # gh issue 20753
    cc2 = lambda x: ccode(optimize(x, [create_expand_pow_optimization(
        4, base_req=lambda b: b.is_Function)]))
    assert cc2(x**3 + sin(x)**3) == "pow(x, 3) + sin(x)*sin(x)*sin(x)"


def test_matsolve():
    n = Symbol('n', integer=True)
    A = MatrixSymbol('A', n, n)
    x = MatrixSymbol('x', n, 1)

    with assuming(Q.fullrank(A)):
        assert optimize(A**(-1) * x, [matinv_opt]) == MatrixSolve(A, x)
        assert optimize(A**(-1) * x + x, [matinv_opt]) == MatrixSolve(A, x) + x


def test_logaddexp_opt():
    x, y = map(Symbol, 'x y'.split())
    expr1 = log(exp(x) + exp(y))
    opt1 = optimize(expr1, [logaddexp_opt])
    assert logaddexp(x, y) - opt1 == 0
    assert logaddexp(y, x) - opt1 == 0
    assert opt1.rewrite(log) == expr1


def test_logaddexp2_opt():
    x, y = map(Symbol, 'x y'.split())
    expr1 = log(2**x + 2**y)/log(2)
    opt1 = optimize(expr1, [logaddexp2_opt])
    assert logaddexp2(x, y) - opt1 == 0
    assert logaddexp2(y, x) - opt1 == 0
    assert opt1.rewrite(log) == expr1


def test_sinc_opts():
    def check(d):
        for k, v in d.items():
            assert optimize(k, sinc_opts) == v

    x = Symbol('x')
    check({
        sin(x)/x       : sinc(x),
        sin(2*x)/(2*x) : sinc(2*x),
        sin(3*x)/x     : 3*sinc(3*x),
        x*sin(x)       : x*sin(x)
    })

    y = Symbol('y')
    check({
        sin(x*y)/(x*y)       : sinc(x*y),
        y*sin(x/y)/x         : sinc(x/y),
        sin(sin(x))/sin(x)   : sinc(sin(x)),
        sin(3*sin(x))/sin(x) : 3*sinc(3*sin(x)),
        sin(x)/y             : sin(x)/y
    })


def test_optims_numpy():
    def check(d):
        for k, v in d.items():
            assert optimize(k, optims_numpy) == v

    x = Symbol('x')
    check({
        sin(2*x)/(2*x) + exp(2*x) - 1: sinc(2*x) + expm1(2*x),
        log(x+3)/log(2) + log(x**2 + 1): log1p(x**2) + log2(x+3)
    })


@XFAIL  # room for improvement, ideally this test case should pass.
def test_optims_numpy_TODO():
    def check(d):
        for k, v in d.items():
            assert optimize(k, optims_numpy) == v

    x, y = map(Symbol, 'x y'.split())
    check({
        log(x*y)*sin(x*y)*log(x*y+1)/(log(2)*x*y): log2(x*y)*sinc(x*y)*log1p(x*y),
        exp(x*sin(y)/y) - 1: expm1(x*sinc(y))
    })


@may_xfail
def test_compiled_ccode_with_rewriting():
    if not cython:
        skip("cython not installed.")
    if not has_c():
        skip("No C compiler found.")

    x = Symbol('x')
    about_two = 2**(58/S(117))*3**(97/S(117))*5**(4/S(39))*7**(92/S(117))/S(30)*pi
    # about_two: 1.999999999999581826
    unchanged = 2*exp(x) - about_two
    xval = S(10)**-11
    ref = unchanged.subs(x, xval).n(19) # 2.0418173913673213e-11

    rewritten = optimize(2*exp(x) - about_two, [expm1_opt])

    # Unfortunately, we need to call ``.n()`` on our expressions before we hand them
    # to ``ccode``, and we need to request a large number of significant digits.
    # In this test, results converged for double precision when the following number
    # of significant digits were chosen:
    NUMBER_OF_DIGITS = 25   # TODO: this should ideally be automatically handled.

    func_c = '''
#include <math.h>

double func_unchanged(double x) {
    return %(unchanged)s;
}
double func_rewritten(double x) {
    return %(rewritten)s;
}
''' % {"unchanged": ccode(unchanged.n(NUMBER_OF_DIGITS)),
           "rewritten": ccode(rewritten.n(NUMBER_OF_DIGITS))}

    func_pyx = '''
#cython: language_level=3
cdef extern double func_unchanged(double)
cdef extern double func_rewritten(double)
def py_unchanged(x):
    return func_unchanged(x)
def py_rewritten(x):
    return func_rewritten(x)
'''
    with tempfile.TemporaryDirectory() as folder:
        mod, info = compile_link_import_strings(
            [('func.c', func_c), ('_func.pyx', func_pyx)],
            build_dir=folder, compile_kwargs={"std": 'c99'}
        )
        err_rewritten = abs(mod.py_rewritten(1e-11) - ref)
        err_unchanged = abs(mod.py_unchanged(1e-11) - ref)
        assert 1e-27 < err_rewritten < 1e-25  # highly accurate.
        assert 1e-19 < err_unchanged < 1e-16  # quite poor.

    # Tolerances used above were determined as follows:
    # >>> no_opt = unchanged.subs(x, xval.evalf()).evalf()
    # >>> with_opt = rewritten.n(25).subs(x, 1e-11).evalf()
    # >>> with_opt - ref, no_opt - ref
    # (1.1536301877952077e-26, 1.6547074214222335e-18)
