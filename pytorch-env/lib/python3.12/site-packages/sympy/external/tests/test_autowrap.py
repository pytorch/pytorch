import sympy
import tempfile
import os
from sympy.core.mod import Mod
from sympy.core.relational import Eq
from sympy.core.symbol import symbols
from sympy.external import import_module
from sympy.tensor import IndexedBase, Idx
from sympy.utilities.autowrap import autowrap, ufuncify, CodeWrapError
from sympy.testing.pytest import skip

numpy = import_module('numpy', min_module_version='1.6.1')
Cython = import_module('Cython', min_module_version='0.15.1')
f2py = import_module('numpy.f2py', import_kwargs={'fromlist': ['f2py']})

f2pyworks = False
if f2py:
    try:
        autowrap(symbols('x'), 'f95', 'f2py')
    except (CodeWrapError, ImportError, OSError):
        f2pyworks = False
    else:
        f2pyworks = True

a, b, c = symbols('a b c')
n, m, d = symbols('n m d', integer=True)
A, B, C = symbols('A B C', cls=IndexedBase)
i = Idx('i', m)
j = Idx('j', n)
k = Idx('k', d)


def has_module(module):
    """
    Return True if module exists, otherwise run skip().

    module should be a string.
    """
    # To give a string of the module name to skip(), this function takes a
    # string.  So we don't waste time running import_module() more than once,
    # just map the three modules tested here in this dict.
    modnames = {'numpy': numpy, 'Cython': Cython, 'f2py': f2py}

    if modnames[module]:
        if module == 'f2py' and not f2pyworks:
            skip("Couldn't run f2py.")
        return True
    skip("Couldn't import %s." % module)

#
# test runners used by several language-backend combinations
#

def runtest_autowrap_twice(language, backend):
    f = autowrap((((a + b)/c)**5).expand(), language, backend)
    g = autowrap((((a + b)/c)**4).expand(), language, backend)

    # check that autowrap updates the module name.  Else, g gives the same as f
    assert f(1, -2, 1) == -1.0
    assert g(1, -2, 1) == 1.0


def runtest_autowrap_trace(language, backend):
    has_module('numpy')
    trace = autowrap(A[i, i], language, backend)
    assert trace(numpy.eye(100)) == 100


def runtest_autowrap_matrix_vector(language, backend):
    has_module('numpy')
    x, y = symbols('x y', cls=IndexedBase)
    expr = Eq(y[i], A[i, j]*x[j])
    mv = autowrap(expr, language, backend)

    # compare with numpy's dot product
    M = numpy.random.rand(10, 20)
    x = numpy.random.rand(20)
    y = numpy.dot(M, x)
    assert numpy.sum(numpy.abs(y - mv(M, x))) < 1e-13


def runtest_autowrap_matrix_matrix(language, backend):
    has_module('numpy')
    expr = Eq(C[i, j], A[i, k]*B[k, j])
    matmat = autowrap(expr, language, backend)

    # compare with numpy's dot product
    M1 = numpy.random.rand(10, 20)
    M2 = numpy.random.rand(20, 15)
    M3 = numpy.dot(M1, M2)
    assert numpy.sum(numpy.abs(M3 - matmat(M1, M2))) < 1e-13


def runtest_ufuncify(language, backend):
    has_module('numpy')
    a, b, c = symbols('a b c')
    fabc = ufuncify([a, b, c], a*b + c, backend=backend)
    facb = ufuncify([a, c, b], a*b + c, backend=backend)
    grid = numpy.linspace(-2, 2, 50)
    b = numpy.linspace(-5, 4, 50)
    c = numpy.linspace(-1, 1, 50)
    expected = grid*b + c
    numpy.testing.assert_allclose(fabc(grid, b, c), expected)
    numpy.testing.assert_allclose(facb(grid, c, b), expected)


def runtest_issue_10274(language, backend):
    expr = (a - b + c)**(13)
    tmp = tempfile.mkdtemp()
    f = autowrap(expr, language, backend, tempdir=tmp,
                 helpers=('helper', a - b + c, (a, b, c)))
    assert f(1, 1, 1) == 1

    for file in os.listdir(tmp):
        if not (file.startswith("wrapped_code_") and file.endswith(".c")):
            continue

        with open(tmp + '/' + file) as fil:
            lines = fil.readlines()
            assert lines[0] == "/******************************************************************************\n"
            assert "Code generated with SymPy " + sympy.__version__ in lines[1]
            assert lines[2:] == [
                " *                                                                            *\n",
                " *              See http://www.sympy.org/ for more information.               *\n",
                " *                                                                            *\n",
                " *                      This file is part of 'autowrap'                       *\n",
                " ******************************************************************************/\n",
                "#include " + '"' + file[:-1]+ 'h"' + "\n",
                "#include <math.h>\n",
                "\n",
                "double helper(double a, double b, double c) {\n",
                "\n",
                "   double helper_result;\n",
                "   helper_result = a - b + c;\n",
                "   return helper_result;\n",
                "\n",
                "}\n",
                "\n",
                "double autofunc(double a, double b, double c) {\n",
                "\n",
                "   double autofunc_result;\n",
                "   autofunc_result = pow(helper(a, b, c), 13);\n",
                "   return autofunc_result;\n",
                "\n",
                "}\n",
                ]


def runtest_issue_15337(language, backend):
    has_module('numpy')
    # NOTE : autowrap was originally designed to only accept an iterable for
    # the kwarg "helpers", but in issue 10274 the user mistakenly thought that
    # if there was only a single helper it did not need to be passed via an
    # iterable that wrapped the helper tuple. There were no tests for this
    # behavior so when the code was changed to accept a single tuple it broke
    # the original behavior. These tests below ensure that both now work.
    a, b, c, d, e = symbols('a, b, c, d, e')
    expr = (a - b + c - d + e)**13
    exp_res = (1. - 2. + 3. - 4. + 5.)**13

    f = autowrap(expr, language, backend, args=(a, b, c, d, e),
                 helpers=('f1', a - b + c, (a, b, c)))
    numpy.testing.assert_allclose(f(1, 2, 3, 4, 5), exp_res)

    f = autowrap(expr, language, backend, args=(a, b, c, d, e),
                 helpers=(('f1', a - b, (a, b)), ('f2', c - d, (c, d))))
    numpy.testing.assert_allclose(f(1, 2, 3, 4, 5), exp_res)


def test_issue_15230():
    has_module('f2py')

    x, y = symbols('x, y')
    expr = Mod(x, 3.0) - Mod(y, -2.0)
    f = autowrap(expr, args=[x, y], language='F95')
    exp_res = float(expr.xreplace({x: 3.5, y: 2.7}).evalf())
    assert abs(f(3.5, 2.7) - exp_res) < 1e-14

    x, y = symbols('x, y', integer=True)
    expr = Mod(x, 3) - Mod(y, -2)
    f = autowrap(expr, args=[x, y], language='F95')
    assert f(3, 2) == expr.xreplace({x: 3, y: 2})

#
# tests of language-backend combinations
#

# f2py


def test_wrap_twice_f95_f2py():
    has_module('f2py')
    runtest_autowrap_twice('f95', 'f2py')


def test_autowrap_trace_f95_f2py():
    has_module('f2py')
    runtest_autowrap_trace('f95', 'f2py')


def test_autowrap_matrix_vector_f95_f2py():
    has_module('f2py')
    runtest_autowrap_matrix_vector('f95', 'f2py')


def test_autowrap_matrix_matrix_f95_f2py():
    has_module('f2py')
    runtest_autowrap_matrix_matrix('f95', 'f2py')


def test_ufuncify_f95_f2py():
    has_module('f2py')
    runtest_ufuncify('f95', 'f2py')


def test_issue_15337_f95_f2py():
    has_module('f2py')
    runtest_issue_15337('f95', 'f2py')

# Cython


def test_wrap_twice_c_cython():
    has_module('Cython')
    runtest_autowrap_twice('C', 'cython')


def test_autowrap_trace_C_Cython():
    has_module('Cython')
    runtest_autowrap_trace('C99', 'cython')


def test_autowrap_matrix_vector_C_cython():
    has_module('Cython')
    runtest_autowrap_matrix_vector('C99', 'cython')


def test_autowrap_matrix_matrix_C_cython():
    has_module('Cython')
    runtest_autowrap_matrix_matrix('C99', 'cython')


def test_ufuncify_C_Cython():
    has_module('Cython')
    runtest_ufuncify('C99', 'cython')


def test_issue_10274_C_cython():
    has_module('Cython')
    runtest_issue_10274('C89', 'cython')


def test_issue_15337_C_cython():
    has_module('Cython')
    runtest_issue_15337('C89', 'cython')


def test_autowrap_custom_printer():
    has_module('Cython')

    from sympy.core.numbers import pi
    from sympy.utilities.codegen import C99CodeGen
    from sympy.printing.c import C99CodePrinter

    class PiPrinter(C99CodePrinter):
        def _print_Pi(self, expr):
            return "S_PI"

    printer = PiPrinter()
    gen = C99CodeGen(printer=printer)
    gen.preprocessor_statements.append('#include "shortpi.h"')

    expr = pi * a

    expected = (
        '#include "%s"\n'
        '#include <math.h>\n'
        '#include "shortpi.h"\n'
        '\n'
        'double autofunc(double a) {\n'
        '\n'
        '   double autofunc_result;\n'
        '   autofunc_result = S_PI*a;\n'
        '   return autofunc_result;\n'
        '\n'
        '}\n'
    )

    tmpdir = tempfile.mkdtemp()
    # write a trivial header file to use in the generated code
    with open(os.path.join(tmpdir, 'shortpi.h'), 'w') as f:
        f.write('#define S_PI 3.14')

    func = autowrap(expr, backend='cython', tempdir=tmpdir, code_gen=gen)

    assert func(4.2) == 3.14 * 4.2

    # check that the generated code is correct
    for filename in os.listdir(tmpdir):
        if filename.startswith('wrapped_code') and filename.endswith('.c'):
            with open(os.path.join(tmpdir, filename)) as f:
                lines = f.readlines()
                expected = expected % filename.replace('.c', '.h')
                assert ''.join(lines[7:]) == expected


# Numpy

def test_ufuncify_numpy():
    # This test doesn't use Cython, but if Cython works, then there is a valid
    # C compiler, which is needed.
    has_module('Cython')
    runtest_ufuncify('C99', 'numpy')
