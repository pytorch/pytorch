import tempfile
from sympy import log, Min, Max, sqrt
from sympy.core.numbers import Float
from sympy.core.symbol import Symbol, symbols
from sympy.functions.elementary.trigonometric import cos
from sympy.codegen.ast import Assignment, Raise, RuntimeError_, QuotedString
from sympy.codegen.algorithms import newtons_method, newtons_method_function
from sympy.codegen.cfunctions import expm1
from sympy.codegen.fnodes import bind_C
from sympy.codegen.futils import render_as_module as f_module
from sympy.codegen.pyutils import render_as_module as py_module
from sympy.external import import_module
from sympy.printing.codeprinter import ccode
from sympy.utilities._compilation import compile_link_import_strings, has_c, has_fortran
from sympy.utilities._compilation.util import may_xfail
from sympy.testing.pytest import skip, raises, skip_under_pyodide

cython = import_module('cython')
wurlitzer = import_module('wurlitzer')

def test_newtons_method():
    x, dx, atol = symbols('x dx atol')
    expr = cos(x) - x**3
    algo = newtons_method(expr, x, atol, dx)
    assert algo.has(Assignment(dx, -expr/expr.diff(x)))


@may_xfail
def test_newtons_method_function__ccode():
    x = Symbol('x', real=True)
    expr = cos(x) - x**3
    func = newtons_method_function(expr, x)

    if not cython:
        skip("cython not installed.")
    if not has_c():
        skip("No C compiler found.")

    compile_kw = {"std": 'c99'}
    with tempfile.TemporaryDirectory() as folder:
        mod, info = compile_link_import_strings([
            ('newton.c', ('#include <math.h>\n'
                          '#include <stdio.h>\n') + ccode(func)),
            ('_newton.pyx', ("#cython: language_level={}\n".format("3") +
                             "cdef extern double newton(double)\n"
                             "def py_newton(x):\n"
                             "    return newton(x)\n"))
        ], build_dir=folder, compile_kwargs=compile_kw)
        assert abs(mod.py_newton(0.5) - 0.865474033102) < 1e-12


@may_xfail
def test_newtons_method_function__fcode():
    x = Symbol('x', real=True)
    expr = cos(x) - x**3
    func = newtons_method_function(expr, x, attrs=[bind_C(name='newton')])

    if not cython:
        skip("cython not installed.")
    if not has_fortran():
        skip("No Fortran compiler found.")

    f_mod = f_module([func], 'mod_newton')
    with tempfile.TemporaryDirectory() as folder:
        mod, info = compile_link_import_strings([
            ('newton.f90', f_mod),
            ('_newton.pyx', ("#cython: language_level={}\n".format("3") +
                             "cdef extern double newton(double*)\n"
                             "def py_newton(double x):\n"
                             "    return newton(&x)\n"))
        ], build_dir=folder)
        assert abs(mod.py_newton(0.5) - 0.865474033102) < 1e-12


def test_newtons_method_function__pycode():
    x = Symbol('x', real=True)
    expr = cos(x) - x**3
    func = newtons_method_function(expr, x)
    py_mod = py_module(func)
    namespace = {}
    exec(py_mod, namespace, namespace)
    res = eval('newton(0.5)', namespace)
    assert abs(res - 0.865474033102) < 1e-12


@may_xfail
@skip_under_pyodide("Emscripten does not support process spawning")
def test_newtons_method_function__ccode_parameters():
    args = x, A, k, p = symbols('x A k p')
    expr = A*cos(k*x) - p*x**3
    raises(ValueError, lambda: newtons_method_function(expr, x))
    use_wurlitzer = wurlitzer

    func = newtons_method_function(expr, x, args, debug=use_wurlitzer)

    if not has_c():
        skip("No C compiler found.")
    if not cython:
        skip("cython not installed.")

    compile_kw = {"std": 'c99'}
    with tempfile.TemporaryDirectory() as folder:
        mod, info = compile_link_import_strings([
            ('newton_par.c', ('#include <math.h>\n'
                          '#include <stdio.h>\n') + ccode(func)),
            ('_newton_par.pyx', ("#cython: language_level={}\n".format("3") +
                                 "cdef extern double newton(double, double, double, double)\n"
                             "def py_newton(x, A=1, k=1, p=1):\n"
                             "    return newton(x, A, k, p)\n"))
        ], compile_kwargs=compile_kw, build_dir=folder)

        if use_wurlitzer:
            with wurlitzer.pipes() as (out, err):
                result = mod.py_newton(0.5)
        else:
            result = mod.py_newton(0.5)

        assert abs(result - 0.865474033102) < 1e-12

        if not use_wurlitzer:
            skip("C-level output only tested when package 'wurlitzer' is available.")

        out, err = out.read(), err.read()
        assert err == ''
        assert out == """\
x=         0.5
x=      1.1121 d_x=     0.61214
x=     0.90967 d_x=    -0.20247
x=     0.86726 d_x=   -0.042409
x=     0.86548 d_x=  -0.0017867
x=     0.86547 d_x= -3.1022e-06
x=     0.86547 d_x= -9.3421e-12
x=     0.86547 d_x=  3.6902e-17
"""  # try to run tests with LC_ALL=C if this assertion fails


def test_newtons_method_function__rtol_cse_nan():
    a, b, c, N_geo, N_tot = symbols('a b c N_geo N_tot', real=True, nonnegative=True)
    i = Symbol('i', integer=True, nonnegative=True)
    N_ari = N_tot - N_geo - 1
    delta_ari = (c-b)/N_ari
    ln_delta_geo = log(b) + log(-expm1((log(a)-log(b))/N_geo))
    eqb_log = ln_delta_geo - log(delta_ari)

    def _clamp(low, expr, high):
        return Min(Max(low, expr), high)

    meth_kw = {
        'clamped_newton': {'delta_fn': lambda e, x: _clamp(
            (sqrt(a*x)-x)*0.99,
            -e/e.diff(x),
            (sqrt(c*x)-x)*0.99
        )},
        'halley': {'delta_fn': lambda e, x: (-2*(e*e.diff(x))/(2*e.diff(x)**2 - e*e.diff(x, 2)))},
        'halley_alt': {'delta_fn': lambda e, x: (-e/e.diff(x)/(1-e/e.diff(x)*e.diff(x,2)/2/e.diff(x)))},
    }
    args = eqb_log, b
    for use_cse in [False, True]:
        kwargs = {
            'params': (b, a, c, N_geo, N_tot), 'itermax': 60, 'debug': True, 'cse': use_cse,
            'counter': i, 'atol': 1e-100, 'rtol': 2e-16, 'bounds': (a,c),
            'handle_nan': Raise(RuntimeError_(QuotedString("encountered NaN.")))
        }
        func = {k: newtons_method_function(*args, func_name=f"{k}_b", **dict(kwargs, **kw)) for k, kw in meth_kw.items()}
        py_mod = {k: py_module(v) for k, v in func.items()}
        namespace = {}
        root_find_b = {}
        for k, v in py_mod.items():
            ns = namespace[k] = {}
            exec(v, ns, ns)
            root_find_b[k] = ns[f'{k}_b']
        ref = Float('13.2261515064168768938151923226496')
        reftol = {'clamped_newton': 2e-16, 'halley': 2e-16, 'halley_alt': 3e-16}
        guess = 4.0
        for meth, func in root_find_b.items():
            result = func(guess, 1e-2, 1e2, 50, 100)
            req = ref*reftol[meth]
            if use_cse:
                req *= 2
            assert abs(result - ref) < req
