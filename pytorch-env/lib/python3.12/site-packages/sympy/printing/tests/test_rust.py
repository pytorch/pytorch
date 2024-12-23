from sympy.core import (S, pi, oo, symbols, Rational, Integer,
                        GoldenRatio, EulerGamma, Catalan, Lambda, Dummy,
                        Eq, Ne, Le, Lt, Gt, Ge, Mod)
from sympy.functions import (Piecewise, sin, cos, Abs, exp, ceiling, sqrt,
                             sign, floor)
from sympy.logic import ITE
from sympy.testing.pytest import raises
from sympy.utilities.lambdify import implemented_function
from sympy.tensor import IndexedBase, Idx
from sympy.matrices import MatrixSymbol, SparseMatrix, Matrix

from sympy.printing.rust import rust_code

x, y, z = symbols('x,y,z')


def test_Integer():
    assert rust_code(Integer(42)) == "42"
    assert rust_code(Integer(-56)) == "-56"


def test_Relational():
    assert rust_code(Eq(x, y)) == "x == y"
    assert rust_code(Ne(x, y)) == "x != y"
    assert rust_code(Le(x, y)) == "x <= y"
    assert rust_code(Lt(x, y)) == "x < y"
    assert rust_code(Gt(x, y)) == "x > y"
    assert rust_code(Ge(x, y)) == "x >= y"


def test_Rational():
    assert rust_code(Rational(3, 7)) == "3_f64/7.0"
    assert rust_code(Rational(18, 9)) == "2"
    assert rust_code(Rational(3, -7)) == "-3_f64/7.0"
    assert rust_code(Rational(-3, -7)) == "3_f64/7.0"
    assert rust_code(x + Rational(3, 7)) == "x + 3_f64/7.0"
    assert rust_code(Rational(3, 7)*x) == "(3_f64/7.0)*x"


def test_basic_ops():
    assert rust_code(x + y) == "x + y"
    assert rust_code(x - y) == "x - y"
    assert rust_code(x * y) == "x*y"
    assert rust_code(x / y) == "x/y"
    assert rust_code(-x) == "-x"


def test_printmethod():
    class fabs(Abs):
        def _rust_code(self, printer):
            return "%s.fabs()" % printer._print(self.args[0])
    assert rust_code(fabs(x)) == "x.fabs()"
    a = MatrixSymbol("a", 1, 3)
    assert rust_code(a[0,0]) == 'a[0]'


def test_Functions():
    assert rust_code(sin(x) ** cos(x)) == "x.sin().powf(x.cos())"
    assert rust_code(abs(x)) == "x.abs()"
    assert rust_code(ceiling(x)) == "x.ceil()"
    assert rust_code(floor(x)) == "x.floor()"

    # Automatic rewrite
    assert rust_code(Mod(x, 3)) == 'x - 3*((1_f64/3.0)*x).floor()'


def test_Pow():
    assert rust_code(1/x) == "x.recip()"
    assert rust_code(x**-1) == rust_code(x**-1.0) == "x.recip()"
    assert rust_code(sqrt(x)) == "x.sqrt()"
    assert rust_code(x**S.Half) == rust_code(x**0.5) == "x.sqrt()"

    assert rust_code(1/sqrt(x)) == "x.sqrt().recip()"
    assert rust_code(x**-S.Half) == rust_code(x**-0.5) == "x.sqrt().recip()"

    assert rust_code(1/pi) == "PI.recip()"
    assert rust_code(pi**-1) == rust_code(pi**-1.0) == "PI.recip()"
    assert rust_code(pi**-0.5) == "PI.sqrt().recip()"

    assert rust_code(x**Rational(1, 3)) == "x.cbrt()"
    assert rust_code(2**x) == "x.exp2()"
    assert rust_code(exp(x)) == "x.exp()"
    assert rust_code(x**3) == "x.powi(3)"
    assert rust_code(x**(y**3)) == "x.powf(y.powi(3))"
    assert rust_code(x**Rational(2, 3)) == "x.powf(2_f64/3.0)"

    g = implemented_function('g', Lambda(x, 2*x))
    assert rust_code(1/(g(x)*3.5)**(x - y**x)/(x**2 + y)) == \
        "(3.5*2*x).powf(-x + y.powf(x))/(x.powi(2) + y)"
    _cond_cfunc = [(lambda base, exp: exp.is_integer, "dpowi", 1),
                   (lambda base, exp: not exp.is_integer, "pow", 1)]
    assert rust_code(x**3, user_functions={'Pow': _cond_cfunc}) == 'x.dpowi(3)'
    assert rust_code(x**3.2, user_functions={'Pow': _cond_cfunc}) == 'x.pow(3.2)'


def test_constants():
    assert rust_code(pi) == "PI"
    assert rust_code(oo) == "INFINITY"
    assert rust_code(S.Infinity) == "INFINITY"
    assert rust_code(-oo) == "NEG_INFINITY"
    assert rust_code(S.NegativeInfinity) == "NEG_INFINITY"
    assert rust_code(S.NaN) == "NAN"
    assert rust_code(exp(1)) == "E"
    assert rust_code(S.Exp1) == "E"


def test_constants_other():
    assert rust_code(2*GoldenRatio) == "const GoldenRatio: f64 = %s;\n2*GoldenRatio" % GoldenRatio.evalf(17)
    assert rust_code(
            2*Catalan) == "const Catalan: f64 = %s;\n2*Catalan" % Catalan.evalf(17)
    assert rust_code(2*EulerGamma) == "const EulerGamma: f64 = %s;\n2*EulerGamma" % EulerGamma.evalf(17)


def test_boolean():
    assert rust_code(True) == "true"
    assert rust_code(S.true) == "true"
    assert rust_code(False) == "false"
    assert rust_code(S.false) == "false"
    assert rust_code(x & y) == "x && y"
    assert rust_code(x | y) == "x || y"
    assert rust_code(~x) == "!x"
    assert rust_code(x & y & z) == "x && y && z"
    assert rust_code(x | y | z) == "x || y || z"
    assert rust_code((x & y) | z) == "z || x && y"
    assert rust_code((x | y) & z) == "z && (x || y)"


def test_Piecewise():
    expr = Piecewise((x, x < 1), (x + 2, True))
    assert rust_code(expr) == (
            "if (x < 1) {\n"
            "    x\n"
            "} else {\n"
            "    x + 2\n"
            "}")
    assert rust_code(expr, assign_to="r") == (
        "r = if (x < 1) {\n"
        "    x\n"
        "} else {\n"
        "    x + 2\n"
        "};")
    assert rust_code(expr, assign_to="r", inline=True) == (
        "r = if (x < 1) { x } else { x + 2 };")
    expr = Piecewise((x, x < 1), (x + 1, x < 5), (x + 2, True))
    assert rust_code(expr, inline=True) == (
        "if (x < 1) { x } else if (x < 5) { x + 1 } else { x + 2 }")
    assert rust_code(expr, assign_to="r", inline=True) == (
        "r = if (x < 1) { x } else if (x < 5) { x + 1 } else { x + 2 };")
    assert rust_code(expr, assign_to="r") == (
        "r = if (x < 1) {\n"
        "    x\n"
        "} else if (x < 5) {\n"
        "    x + 1\n"
        "} else {\n"
        "    x + 2\n"
        "};")
    expr = 2*Piecewise((x, x < 1), (x + 1, x < 5), (x + 2, True))
    assert rust_code(expr, inline=True) == (
        "2*if (x < 1) { x } else if (x < 5) { x + 1 } else { x + 2 }")
    expr = 2*Piecewise((x, x < 1), (x + 1, x < 5), (x + 2, True)) - 42
    assert rust_code(expr, inline=True) == (
        "2*if (x < 1) { x } else if (x < 5) { x + 1 } else { x + 2 } - 42")
    # Check that Piecewise without a True (default) condition error
    expr = Piecewise((x, x < 1), (x**2, x > 1), (sin(x), x > 0))
    raises(ValueError, lambda: rust_code(expr))


def test_dereference_printing():
    expr = x + y + sin(z) + z
    assert rust_code(expr, dereference=[z]) == "x + y + (*z) + (*z).sin()"


def test_sign():
    expr = sign(x) * y
    assert rust_code(expr) == "y*x.signum()"
    assert rust_code(expr, assign_to='r') == "r = y*x.signum();"

    expr = sign(x + y) + 42
    assert rust_code(expr) == "(x + y).signum() + 42"
    assert rust_code(expr, assign_to='r') == "r = (x + y).signum() + 42;"

    expr = sign(cos(x))
    assert rust_code(expr) == "x.cos().signum()"


def test_reserved_words():

    x, y = symbols("x if")

    expr = sin(y)
    assert rust_code(expr) == "if_.sin()"
    assert rust_code(expr, dereference=[y]) == "(*if_).sin()"
    assert rust_code(expr, reserved_word_suffix='_unreserved') == "if_unreserved.sin()"

    with raises(ValueError):
        rust_code(expr, error_on_reserved=True)


def test_ITE():
    expr = ITE(x < 1, y, z)
    assert rust_code(expr) == (
            "if (x < 1) {\n"
            "    y\n"
            "} else {\n"
            "    z\n"
            "}")


def test_Indexed():
    n, m, o = symbols('n m o', integer=True)
    i, j, k = Idx('i', n), Idx('j', m), Idx('k', o)

    x = IndexedBase('x')[j]
    assert rust_code(x) == "x[j]"

    A = IndexedBase('A')[i, j]
    assert rust_code(A) == "A[m*i + j]"

    B = IndexedBase('B')[i, j, k]
    assert rust_code(B) == "B[m*o*i + o*j + k]"


def test_dummy_loops():
    i, m = symbols('i m', integer=True, cls=Dummy)
    x = IndexedBase('x')
    y = IndexedBase('y')
    i = Idx(i, m)

    assert rust_code(x[i], assign_to=y[i]) == (
        "for i in 0..m {\n"
        "    y[i] = x[i];\n"
        "}")


def test_loops():
    m, n = symbols('m n', integer=True)
    A = IndexedBase('A')
    x = IndexedBase('x')
    y = IndexedBase('y')
    z = IndexedBase('z')
    i = Idx('i', m)
    j = Idx('j', n)

    assert rust_code(A[i, j]*x[j], assign_to=y[i]) == (
        "for i in 0..m {\n"
        "    y[i] = 0;\n"
        "}\n"
        "for i in 0..m {\n"
        "    for j in 0..n {\n"
        "        y[i] = A[n*i + j]*x[j] + y[i];\n"
        "    }\n"
        "}")

    assert rust_code(A[i, j]*x[j] + x[i] + z[i], assign_to=y[i]) == (
        "for i in 0..m {\n"
        "    y[i] = x[i] + z[i];\n"
        "}\n"
        "for i in 0..m {\n"
        "    for j in 0..n {\n"
        "        y[i] = A[n*i + j]*x[j] + y[i];\n"
        "    }\n"
        "}")


def test_loops_multiple_contractions():
    n, m, o, p = symbols('n m o p', integer=True)
    a = IndexedBase('a')
    b = IndexedBase('b')
    y = IndexedBase('y')
    i = Idx('i', m)
    j = Idx('j', n)
    k = Idx('k', o)
    l = Idx('l', p)

    assert rust_code(b[j, k, l]*a[i, j, k, l], assign_to=y[i]) == (
        "for i in 0..m {\n"
        "    y[i] = 0;\n"
        "}\n"
        "for i in 0..m {\n"
        "    for j in 0..n {\n"
        "        for k in 0..o {\n"
        "            for l in 0..p {\n"
        "                y[i] = a[%s]*b[%s] + y[i];\n" % (i*n*o*p + j*o*p + k*p + l, j*o*p + k*p + l) +\
        "            }\n"
        "        }\n"
        "    }\n"
        "}")


def test_loops_addfactor():
    m, n, o, p = symbols('m n o p', integer=True)
    a = IndexedBase('a')
    b = IndexedBase('b')
    c = IndexedBase('c')
    y = IndexedBase('y')
    i = Idx('i', m)
    j = Idx('j', n)
    k = Idx('k', o)
    l = Idx('l', p)

    code = rust_code((a[i, j, k, l] + b[i, j, k, l])*c[j, k, l], assign_to=y[i])
    assert code == (
        "for i in 0..m {\n"
        "    y[i] = 0;\n"
        "}\n"
        "for i in 0..m {\n"
        "    for j in 0..n {\n"
        "        for k in 0..o {\n"
        "            for l in 0..p {\n"
        "                y[i] = (a[%s] + b[%s])*c[%s] + y[i];\n" % (i*n*o*p + j*o*p + k*p + l, i*n*o*p + j*o*p + k*p + l, j*o*p + k*p + l) +\
        "            }\n"
        "        }\n"
        "    }\n"
        "}")


def test_settings():
    raises(TypeError, lambda: rust_code(sin(x), method="garbage"))


def test_inline_function():
    x = symbols('x')
    g = implemented_function('g', Lambda(x, 2*x))
    assert rust_code(g(x)) == "2*x"

    g = implemented_function('g', Lambda(x, 2*x/Catalan))
    assert rust_code(g(x)) == (
        "const Catalan: f64 = %s;\n2*x/Catalan" % Catalan.evalf(17))

    A = IndexedBase('A')
    i = Idx('i', symbols('n', integer=True))
    g = implemented_function('g', Lambda(x, x*(1 + x)*(2 + x)))
    assert rust_code(g(A[i]), assign_to=A[i]) == (
        "for i in 0..n {\n"
        "    A[i] = (A[i] + 1)*(A[i] + 2)*A[i];\n"
        "}")


def test_user_functions():
    x = symbols('x', integer=False)
    n = symbols('n', integer=True)
    custom_functions = {
        "ceiling": "ceil",
        "Abs": [(lambda x: not x.is_integer, "fabs", 4), (lambda x: x.is_integer, "abs", 4)],
    }
    assert rust_code(ceiling(x), user_functions=custom_functions) == "x.ceil()"
    assert rust_code(Abs(x), user_functions=custom_functions) == "fabs(x)"
    assert rust_code(Abs(n), user_functions=custom_functions) == "abs(n)"


def test_matrix():
    assert rust_code(Matrix([1, 2, 3])) == '[1, 2, 3]'
    with raises(ValueError):
        rust_code(Matrix([[1, 2, 3]]))


def test_sparse_matrix():
    # gh-15791
    with raises(NotImplementedError):
        rust_code(SparseMatrix([[1, 2, 3]]))
