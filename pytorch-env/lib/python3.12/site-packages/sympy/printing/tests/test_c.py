from sympy.core import (
    S, pi, oo, Symbol, symbols, Rational, Integer, Float, Function, Mod, GoldenRatio, EulerGamma, Catalan,
    Lambda, Dummy, nan, Mul, Pow, UnevaluatedExpr
)
from sympy.core.relational import (Eq, Ge, Gt, Le, Lt, Ne)
from sympy.functions import (
    Abs, acos, acosh, asin, asinh, atan, atanh, atan2, ceiling, cos, cosh, erf,
    erfc, exp, floor, gamma, log, loggamma, Max, Min, Piecewise, sign, sin, sinh,
    sqrt, tan, tanh, fibonacci, lucas
)
from sympy.sets import Range
from sympy.logic import ITE, Implies, Equivalent
from sympy.codegen import For, aug_assign, Assignment
from sympy.testing.pytest import raises, XFAIL
from sympy.printing.codeprinter import PrintMethodNotImplementedError
from sympy.printing.c import C89CodePrinter, C99CodePrinter, get_math_macros
from sympy.codegen.ast import (
    AddAugmentedAssignment, Element, Type, FloatType, Declaration, Pointer, Variable, value_const, pointer_const,
    While, Scope, Print, FunctionPrototype, FunctionDefinition, FunctionCall, Return,
    real, float32, float64, float80, float128, intc, Comment, CodeBlock, stderr, QuotedString
)
from sympy.codegen.cfunctions import expm1, log1p, exp2, log2, fma, log10, Cbrt, hypot, Sqrt
from sympy.codegen.cnodes import restrict
from sympy.utilities.lambdify import implemented_function
from sympy.tensor import IndexedBase, Idx
from sympy.matrices import Matrix, MatrixSymbol, SparseMatrix

from sympy.printing.codeprinter import ccode

x, y, z = symbols('x,y,z')


def test_printmethod():
    class fabs(Abs):
        def _ccode(self, printer):
            return "fabs(%s)" % printer._print(self.args[0])

    assert ccode(fabs(x)) == "fabs(x)"


def test_ccode_sqrt():
    assert ccode(sqrt(x)) == "sqrt(x)"
    assert ccode(x**0.5) == "sqrt(x)"
    assert ccode(sqrt(x)) == "sqrt(x)"


def test_ccode_Pow():
    assert ccode(x**3) == "pow(x, 3)"
    assert ccode(x**(y**3)) == "pow(x, pow(y, 3))"
    g = implemented_function('g', Lambda(x, 2*x))
    assert ccode(1/(g(x)*3.5)**(x - y**x)/(x**2 + y)) == \
        "pow(3.5*2*x, -x + pow(y, x))/(pow(x, 2) + y)"
    assert ccode(x**-1.0) == '1.0/x'
    assert ccode(x**Rational(2, 3)) == 'pow(x, 2.0/3.0)'
    assert ccode(x**Rational(2, 3), type_aliases={real: float80}) == 'powl(x, 2.0L/3.0L)'
    _cond_cfunc = [(lambda base, exp: exp.is_integer, "dpowi"),
                   (lambda base, exp: not exp.is_integer, "pow")]
    assert ccode(x**3, user_functions={'Pow': _cond_cfunc}) == 'dpowi(x, 3)'
    assert ccode(x**0.5, user_functions={'Pow': _cond_cfunc}) == 'pow(x, 0.5)'
    assert ccode(x**Rational(16, 5), user_functions={'Pow': _cond_cfunc}) == 'pow(x, 16.0/5.0)'
    _cond_cfunc2 = [(lambda base, exp: base == 2, lambda base, exp: 'exp2(%s)' % exp),
                    (lambda base, exp: base != 2, 'pow')]
    # Related to gh-11353
    assert ccode(2**x, user_functions={'Pow': _cond_cfunc2}) == 'exp2(x)'
    assert ccode(x**2, user_functions={'Pow': _cond_cfunc2}) == 'pow(x, 2)'
    # For issue 14160
    assert ccode(Mul(-2, x, Pow(Mul(y,y,evaluate=False), -1, evaluate=False),
                                                evaluate=False)) == '-2*x/(y*y)'


def test_ccode_Max():
    # Test for gh-11926
    assert ccode(Max(x,x*x),user_functions={"Max":"my_max", "Pow":"my_pow"}) == 'my_max(x, my_pow(x, 2))'


def test_ccode_Min_performance():
    #Shouldn't take more than a few seconds
    big_min = Min(*symbols('a[0:50]'))
    for curr_standard in ('c89', 'c99', 'c11'):
        output = ccode(big_min, standard=curr_standard)
        assert output.count('(') == output.count(')')


def test_ccode_constants_mathh():
    assert ccode(exp(1)) == "M_E"
    assert ccode(pi) == "M_PI"
    assert ccode(oo, standard='c89') == "HUGE_VAL"
    assert ccode(-oo, standard='c89') == "-HUGE_VAL"
    assert ccode(oo) == "INFINITY"
    assert ccode(-oo, standard='c99') == "-INFINITY"
    assert ccode(pi, type_aliases={real: float80}) == "M_PIl"


def test_ccode_constants_other():
    assert ccode(2*GoldenRatio) == "const double GoldenRatio = %s;\n2*GoldenRatio" % GoldenRatio.evalf(17)
    assert ccode(
        2*Catalan) == "const double Catalan = %s;\n2*Catalan" % Catalan.evalf(17)
    assert ccode(2*EulerGamma) == "const double EulerGamma = %s;\n2*EulerGamma" % EulerGamma.evalf(17)


def test_ccode_Rational():
    assert ccode(Rational(3, 7)) == "3.0/7.0"
    assert ccode(Rational(3, 7), type_aliases={real: float80}) == "3.0L/7.0L"
    assert ccode(Rational(18, 9)) == "2"
    assert ccode(Rational(3, -7)) == "-3.0/7.0"
    assert ccode(Rational(3, -7), type_aliases={real: float80}) == "-3.0L/7.0L"
    assert ccode(Rational(-3, -7)) == "3.0/7.0"
    assert ccode(Rational(-3, -7), type_aliases={real: float80}) == "3.0L/7.0L"
    assert ccode(x + Rational(3, 7)) == "x + 3.0/7.0"
    assert ccode(x + Rational(3, 7), type_aliases={real: float80}) == "x + 3.0L/7.0L"
    assert ccode(Rational(3, 7)*x) == "(3.0/7.0)*x"
    assert ccode(Rational(3, 7)*x, type_aliases={real: float80}) == "(3.0L/7.0L)*x"


def test_ccode_Integer():
    assert ccode(Integer(67)) == "67"
    assert ccode(Integer(-1)) == "-1"


def test_ccode_functions():
    assert ccode(sin(x) ** cos(x)) == "pow(sin(x), cos(x))"


def test_ccode_inline_function():
    x = symbols('x')
    g = implemented_function('g', Lambda(x, 2*x))
    assert ccode(g(x)) == "2*x"
    g = implemented_function('g', Lambda(x, 2*x/Catalan))
    assert ccode(
        g(x)) == "const double Catalan = %s;\n2*x/Catalan" % Catalan.evalf(17)
    A = IndexedBase('A')
    i = Idx('i', symbols('n', integer=True))
    g = implemented_function('g', Lambda(x, x*(1 + x)*(2 + x)))
    assert ccode(g(A[i]), assign_to=A[i]) == (
        "for (int i=0; i<n; i++){\n"
        "   A[i] = (A[i] + 1)*(A[i] + 2)*A[i];\n"
        "}"
    )


def test_ccode_exceptions():
    assert ccode(gamma(x), standard='C99') == "tgamma(x)"
    with raises(PrintMethodNotImplementedError):
        ccode(gamma(x), standard='C89')
    with raises(PrintMethodNotImplementedError):
        ccode(gamma(x), standard='C89', allow_unknown_functions=False)

    ccode(gamma(x), standard='C89', allow_unknown_functions=True)



def test_ccode_functions2():
    assert ccode(ceiling(x)) == "ceil(x)"
    assert ccode(Abs(x)) == "fabs(x)"
    assert ccode(gamma(x)) == "tgamma(x)"
    r, s = symbols('r,s', real=True)
    assert ccode(Mod(ceiling(r), ceiling(s))) == '((ceil(r) % ceil(s)) + '\
                                                 'ceil(s)) % ceil(s)'
    assert ccode(Mod(r, s)) == "fmod(r, s)"
    p1, p2 = symbols('p1 p2', integer=True, positive=True)
    assert ccode(Mod(p1, p2)) == 'p1 % p2'
    assert ccode(Mod(p1, p2 + 3)) == 'p1 % (p2 + 3)'
    assert ccode(Mod(-3, -7, evaluate=False)) == '(-3) % (-7)'
    assert ccode(-Mod(3, 7, evaluate=False)) == '-(3 % 7)'
    assert ccode(r*Mod(p1, p2)) == 'r*(p1 % p2)'
    assert ccode(Mod(p1, p2)**s) == 'pow(p1 % p2, s)'
    n = symbols('n', integer=True, negative=True)
    assert ccode(Mod(-n, p2)) == '(-n) % p2'
    assert ccode(fibonacci(n)) == '((1.0/5.0)*pow(2, -n)*sqrt(5)*(-pow(1 - sqrt(5), n) + pow(1 + sqrt(5), n)))'
    assert ccode(lucas(n)) == '(pow(2, -n)*(pow(1 - sqrt(5), n) + pow(1 + sqrt(5), n)))'


def test_ccode_user_functions():
    x = symbols('x', integer=False)
    n = symbols('n', integer=True)
    custom_functions = {
        "ceiling": "ceil",
        "Abs": [(lambda x: not x.is_integer, "fabs"), (lambda x: x.is_integer, "abs")],
    }
    assert ccode(ceiling(x), user_functions=custom_functions) == "ceil(x)"
    assert ccode(Abs(x), user_functions=custom_functions) == "fabs(x)"
    assert ccode(Abs(n), user_functions=custom_functions) == "abs(n)"

    expr = Symbol('a')
    muladd = Function('muladd')
    for i in range(0, 100):
        # the large number of terms acts as a regression test for gh-23839
        expr = muladd(Rational(1, 2), Symbol(f'a{i}'), expr)
    out = ccode(expr, user_functions={'muladd':'muladd'})
    assert 'a99' in out
    assert out.count('muladd') == 100


def test_ccode_boolean():
    assert ccode(True) == "true"
    assert ccode(S.true) == "true"
    assert ccode(False) == "false"
    assert ccode(S.false) == "false"
    assert ccode(x & y) == "x && y"
    assert ccode(x | y) == "x || y"
    assert ccode(~x) == "!x"
    assert ccode(x & y & z) == "x && y && z"
    assert ccode(x | y | z) == "x || y || z"
    assert ccode((x & y) | z) == "z || x && y"
    assert ccode((x | y) & z) == "z && (x || y)"
    # Automatic rewrites
    assert ccode(x ^ y) == '(x || y) && (!x || !y)'
    assert ccode((x ^ y) ^ z) == '(x || y || z) && (x || !y || !z) && (y || !x || !z) && (z || !x || !y)'
    assert ccode(Implies(x, y)) == 'y || !x'
    assert ccode(Equivalent(x, z ^ y, Implies(z, x))) == '(x || (y || !z) && (z || !y)) && (z && !x || (y || z) && (!y || !z))'


def test_ccode_Relational():
    assert ccode(Eq(x, y)) == "x == y"
    assert ccode(Ne(x, y)) == "x != y"
    assert ccode(Le(x, y)) == "x <= y"
    assert ccode(Lt(x, y)) == "x < y"
    assert ccode(Gt(x, y)) == "x > y"
    assert ccode(Ge(x, y)) == "x >= y"


def test_ccode_Piecewise():
    expr = Piecewise((x, x < 1), (x**2, True))
    assert ccode(expr) == (
            "((x < 1) ? (\n"
            "   x\n"
            ")\n"
            ": (\n"
            "   pow(x, 2)\n"
            "))")
    assert ccode(expr, assign_to="c") == (
            "if (x < 1) {\n"
            "   c = x;\n"
            "}\n"
            "else {\n"
            "   c = pow(x, 2);\n"
            "}")
    expr = Piecewise((x, x < 1), (x + 1, x < 2), (x**2, True))
    assert ccode(expr) == (
            "((x < 1) ? (\n"
            "   x\n"
            ")\n"
            ": ((x < 2) ? (\n"
            "   x + 1\n"
            ")\n"
            ": (\n"
            "   pow(x, 2)\n"
            ")))")
    assert ccode(expr, assign_to='c') == (
            "if (x < 1) {\n"
            "   c = x;\n"
            "}\n"
            "else if (x < 2) {\n"
            "   c = x + 1;\n"
            "}\n"
            "else {\n"
            "   c = pow(x, 2);\n"
            "}")
    # Check that Piecewise without a True (default) condition error
    expr = Piecewise((x, x < 1), (x**2, x > 1), (sin(x), x > 0))
    raises(ValueError, lambda: ccode(expr))


def test_ccode_sinc():
    from sympy.functions.elementary.trigonometric import sinc
    expr = sinc(x)
    assert ccode(expr) == (
            "(((x != 0) ? (\n"
            "   sin(x)/x\n"
            ")\n"
            ": (\n"
            "   1\n"
            ")))")


def test_ccode_Piecewise_deep():
    p = ccode(2*Piecewise((x, x < 1), (x + 1, x < 2), (x**2, True)))
    assert p == (
            "2*((x < 1) ? (\n"
            "   x\n"
            ")\n"
            ": ((x < 2) ? (\n"
            "   x + 1\n"
            ")\n"
            ": (\n"
            "   pow(x, 2)\n"
            ")))")
    expr = x*y*z + x**2 + y**2 + Piecewise((0, x < 0.5), (1, True)) + cos(z) - 1
    assert ccode(expr) == (
            "pow(x, 2) + x*y*z + pow(y, 2) + ((x < 0.5) ? (\n"
            "   0\n"
            ")\n"
            ": (\n"
            "   1\n"
            ")) + cos(z) - 1")
    assert ccode(expr, assign_to='c') == (
            "c = pow(x, 2) + x*y*z + pow(y, 2) + ((x < 0.5) ? (\n"
            "   0\n"
            ")\n"
            ": (\n"
            "   1\n"
            ")) + cos(z) - 1;")


def test_ccode_ITE():
    expr = ITE(x < 1, y, z)
    assert ccode(expr) == (
            "((x < 1) ? (\n"
            "   y\n"
            ")\n"
            ": (\n"
            "   z\n"
            "))")


def test_ccode_settings():
    raises(TypeError, lambda: ccode(sin(x), method="garbage"))


def test_ccode_Indexed():
    s, n, m, o = symbols('s n m o', integer=True)
    i, j, k = Idx('i', n), Idx('j', m), Idx('k', o)

    x = IndexedBase('x')[j]
    A = IndexedBase('A')[i, j]
    B = IndexedBase('B')[i, j, k]

    p = C99CodePrinter()

    assert p._print_Indexed(x) == 'x[j]'
    assert p._print_Indexed(A) == 'A[%s]' % (m*i+j)
    assert p._print_Indexed(B) == 'B[%s]' % (i*o*m+j*o+k)

    A = IndexedBase('A', shape=(5,3))[i, j]
    assert p._print_Indexed(A) == 'A[%s]' % (3*i + j)

    A = IndexedBase('A', shape=(5,3), strides='F')[i, j]
    assert ccode(A) == 'A[%s]' % (i + 5*j)

    A = IndexedBase('A', shape=(29,29), strides=(1, s), offset=o)[i, j]
    assert ccode(A) == 'A[o + s*j + i]'

    Abase = IndexedBase('A', strides=(s, m, n), offset=o)
    assert ccode(Abase[i, j, k]) == 'A[m*j + n*k + o + s*i]'
    assert ccode(Abase[2, 3, k]) == 'A[3*m + n*k + o + 2*s]'


def test_Element():
    assert ccode(Element('x', 'ij')) == 'x[i][j]'
    assert ccode(Element('x', 'ij', strides='kl', offset='o')) == 'x[i*k + j*l + o]'
    assert ccode(Element('x', (3,))) == 'x[3]'
    assert ccode(Element('x', (3,4,5))) == 'x[3][4][5]'


def test_ccode_Indexed_without_looking_for_contraction():
    len_y = 5
    y = IndexedBase('y', shape=(len_y,))
    x = IndexedBase('x', shape=(len_y,))
    Dy = IndexedBase('Dy', shape=(len_y-1,))
    i = Idx('i', len_y-1)
    e = Eq(Dy[i], (y[i+1]-y[i])/(x[i+1]-x[i]))
    code0 = ccode(e.rhs, assign_to=e.lhs, contract=False)
    assert code0 == 'Dy[i] = (y[%s] - y[i])/(x[%s] - x[i]);' % (i + 1, i + 1)


def test_ccode_loops_matrix_vector():
    n, m = symbols('n m', integer=True)
    A = IndexedBase('A')
    x = IndexedBase('x')
    y = IndexedBase('y')
    i = Idx('i', m)
    j = Idx('j', n)

    s = (
        'for (int i=0; i<m; i++){\n'
        '   y[i] = 0;\n'
        '}\n'
        'for (int i=0; i<m; i++){\n'
        '   for (int j=0; j<n; j++){\n'
        '      y[i] = A[%s]*x[j] + y[i];\n' % (i*n + j) +\
        '   }\n'
        '}'
    )
    assert ccode(A[i, j]*x[j], assign_to=y[i]) == s


def test_dummy_loops():
    i, m = symbols('i m', integer=True, cls=Dummy)
    x = IndexedBase('x')
    y = IndexedBase('y')
    i = Idx(i, m)

    expected = (
        'for (int i_%(icount)i=0; i_%(icount)i<m_%(mcount)i; i_%(icount)i++){\n'
        '   y[i_%(icount)i] = x[i_%(icount)i];\n'
        '}'
    ) % {'icount': i.label.dummy_index, 'mcount': m.dummy_index}

    assert ccode(x[i], assign_to=y[i]) == expected


def test_ccode_loops_add():
    n, m = symbols('n m', integer=True)
    A = IndexedBase('A')
    x = IndexedBase('x')
    y = IndexedBase('y')
    z = IndexedBase('z')
    i = Idx('i', m)
    j = Idx('j', n)

    s = (
        'for (int i=0; i<m; i++){\n'
        '   y[i] = x[i] + z[i];\n'
        '}\n'
        'for (int i=0; i<m; i++){\n'
        '   for (int j=0; j<n; j++){\n'
        '      y[i] = A[%s]*x[j] + y[i];\n' % (i*n + j) +\
        '   }\n'
        '}'
    )
    assert ccode(A[i, j]*x[j] + x[i] + z[i], assign_to=y[i]) == s


def test_ccode_loops_multiple_contractions():
    n, m, o, p = symbols('n m o p', integer=True)
    a = IndexedBase('a')
    b = IndexedBase('b')
    y = IndexedBase('y')
    i = Idx('i', m)
    j = Idx('j', n)
    k = Idx('k', o)
    l = Idx('l', p)

    s = (
        'for (int i=0; i<m; i++){\n'
        '   y[i] = 0;\n'
        '}\n'
        'for (int i=0; i<m; i++){\n'
        '   for (int j=0; j<n; j++){\n'
        '      for (int k=0; k<o; k++){\n'
        '         for (int l=0; l<p; l++){\n'
        '            y[i] = a[%s]*b[%s] + y[i];\n' % (i*n*o*p + j*o*p + k*p + l, j*o*p + k*p + l) +\
        '         }\n'
        '      }\n'
        '   }\n'
        '}'
    )
    assert ccode(b[j, k, l]*a[i, j, k, l], assign_to=y[i]) == s


def test_ccode_loops_addfactor():
    n, m, o, p = symbols('n m o p', integer=True)
    a = IndexedBase('a')
    b = IndexedBase('b')
    c = IndexedBase('c')
    y = IndexedBase('y')
    i = Idx('i', m)
    j = Idx('j', n)
    k = Idx('k', o)
    l = Idx('l', p)

    s = (
        'for (int i=0; i<m; i++){\n'
        '   y[i] = 0;\n'
        '}\n'
        'for (int i=0; i<m; i++){\n'
        '   for (int j=0; j<n; j++){\n'
        '      for (int k=0; k<o; k++){\n'
        '         for (int l=0; l<p; l++){\n'
        '            y[i] = (a[%s] + b[%s])*c[%s] + y[i];\n' % (i*n*o*p + j*o*p + k*p + l, i*n*o*p + j*o*p + k*p + l, j*o*p + k*p + l) +\
        '         }\n'
        '      }\n'
        '   }\n'
        '}'
    )
    assert ccode((a[i, j, k, l] + b[i, j, k, l])*c[j, k, l], assign_to=y[i]) == s


def test_ccode_loops_multiple_terms():
    n, m, o, p = symbols('n m o p', integer=True)
    a = IndexedBase('a')
    b = IndexedBase('b')
    c = IndexedBase('c')
    y = IndexedBase('y')
    i = Idx('i', m)
    j = Idx('j', n)
    k = Idx('k', o)

    s0 = (
        'for (int i=0; i<m; i++){\n'
        '   y[i] = 0;\n'
        '}\n'
    )
    s1 = (
        'for (int i=0; i<m; i++){\n'
        '   for (int j=0; j<n; j++){\n'
        '      for (int k=0; k<o; k++){\n'
        '         y[i] = b[j]*b[k]*c[%s] + y[i];\n' % (i*n*o + j*o + k) +\
        '      }\n'
        '   }\n'
        '}\n'
    )
    s2 = (
        'for (int i=0; i<m; i++){\n'
        '   for (int k=0; k<o; k++){\n'
        '      y[i] = a[%s]*b[k] + y[i];\n' % (i*o + k) +\
        '   }\n'
        '}\n'
    )
    s3 = (
        'for (int i=0; i<m; i++){\n'
        '   for (int j=0; j<n; j++){\n'
        '      y[i] = a[%s]*b[j] + y[i];\n' % (i*n + j) +\
        '   }\n'
        '}\n'
    )
    c = ccode(b[j]*a[i, j] + b[k]*a[i, k] + b[j]*b[k]*c[i, j, k], assign_to=y[i])
    assert (c == s0 + s1 + s2 + s3[:-1] or
            c == s0 + s1 + s3 + s2[:-1] or
            c == s0 + s2 + s1 + s3[:-1] or
            c == s0 + s2 + s3 + s1[:-1] or
            c == s0 + s3 + s1 + s2[:-1] or
            c == s0 + s3 + s2 + s1[:-1])


def test_dereference_printing():
    expr = x + y + sin(z) + z
    assert ccode(expr, dereference=[z]) == "x + y + (*z) + sin((*z))"


def test_Matrix_printing():
    # Test returning a Matrix
    mat = Matrix([x*y, Piecewise((2 + x, y>0), (y, True)), sin(z)])
    A = MatrixSymbol('A', 3, 1)
    assert ccode(mat, A) == (
        "A[0] = x*y;\n"
        "if (y > 0) {\n"
        "   A[1] = x + 2;\n"
        "}\n"
        "else {\n"
        "   A[1] = y;\n"
        "}\n"
        "A[2] = sin(z);")
    # Test using MatrixElements in expressions
    expr = Piecewise((2*A[2, 0], x > 0), (A[2, 0], True)) + sin(A[1, 0]) + A[0, 0]
    assert ccode(expr) == (
        "((x > 0) ? (\n"
        "   2*A[2]\n"
        ")\n"
        ": (\n"
        "   A[2]\n"
        ")) + sin(A[1]) + A[0]")
    # Test using MatrixElements in a Matrix
    q = MatrixSymbol('q', 5, 1)
    M = MatrixSymbol('M', 3, 3)
    m = Matrix([[sin(q[1,0]), 0, cos(q[2,0])],
        [q[1,0] + q[2,0], q[3, 0], 5],
        [2*q[4, 0]/q[1,0], sqrt(q[0,0]) + 4, 0]])
    assert ccode(m, M) == (
        "M[0] = sin(q[1]);\n"
        "M[1] = 0;\n"
        "M[2] = cos(q[2]);\n"
        "M[3] = q[1] + q[2];\n"
        "M[4] = q[3];\n"
        "M[5] = 5;\n"
        "M[6] = 2*q[4]/q[1];\n"
        "M[7] = sqrt(q[0]) + 4;\n"
        "M[8] = 0;")


def test_sparse_matrix():
    # gh-15791
    with raises(PrintMethodNotImplementedError):
        ccode(SparseMatrix([[1, 2, 3]]))

    assert 'Not supported in C' in C89CodePrinter({'strict': False}).doprint(SparseMatrix([[1, 2, 3]]))



def test_ccode_reserved_words():
    x, y = symbols('x, if')
    with raises(ValueError):
        ccode(y**2, error_on_reserved=True, standard='C99')
    assert ccode(y**2) == 'pow(if_, 2)'
    assert ccode(x * y**2, dereference=[y]) == 'pow((*if_), 2)*x'
    assert ccode(y**2, reserved_word_suffix='_unreserved') == 'pow(if_unreserved, 2)'


def test_ccode_sign():
    expr1, ref1 = sign(x) * y, 'y*(((x) > 0) - ((x) < 0))'
    expr2, ref2 = sign(cos(x)), '(((cos(x)) > 0) - ((cos(x)) < 0))'
    expr3, ref3 = sign(2 * x + x**2) * x + x**2, 'pow(x, 2) + x*(((pow(x, 2) + 2*x) > 0) - ((pow(x, 2) + 2*x) < 0))'
    assert ccode(expr1) == ref1
    assert ccode(expr1, 'z') == 'z = %s;' % ref1
    assert ccode(expr2) == ref2
    assert ccode(expr3) == ref3

def test_ccode_Assignment():
    assert ccode(Assignment(x, y + z)) == 'x = y + z;'
    assert ccode(aug_assign(x, '+', y + z)) == 'x += y + z;'


def test_ccode_For():
    f = For(x, Range(0, 10, 2), [aug_assign(y, '*', x)])
    assert ccode(f) == ("for (x = 0; x < 10; x += 2) {\n"
                        "   y *= x;\n"
                        "}")

def test_ccode_Max_Min():
    assert ccode(Max(x, 0), standard='C89') == '((0 > x) ? 0 : x)'
    assert ccode(Max(x, 0), standard='C99') == 'fmax(0, x)'
    assert ccode(Min(x, 0, sqrt(x)), standard='c89') == (
        '((0 < ((x < sqrt(x)) ? x : sqrt(x))) ? 0 : ((x < sqrt(x)) ? x : sqrt(x)))'
    )

def test_ccode_standard():
    assert ccode(expm1(x), standard='c99') == 'expm1(x)'
    assert ccode(nan, standard='c99') == 'NAN'
    assert ccode(float('nan'), standard='c99') == 'NAN'


def test_C89CodePrinter():
    c89printer = C89CodePrinter()
    assert c89printer.language == 'C'
    assert c89printer.standard == 'C89'
    assert 'void' in c89printer.reserved_words
    assert 'template' not in c89printer.reserved_words


def test_C99CodePrinter():
    assert C99CodePrinter().doprint(expm1(x)) == 'expm1(x)'
    assert C99CodePrinter().doprint(log1p(x)) == 'log1p(x)'
    assert C99CodePrinter().doprint(exp2(x)) == 'exp2(x)'
    assert C99CodePrinter().doprint(log2(x)) == 'log2(x)'
    assert C99CodePrinter().doprint(fma(x, y, -z)) == 'fma(x, y, -z)'
    assert C99CodePrinter().doprint(log10(x)) == 'log10(x)'
    assert C99CodePrinter().doprint(Cbrt(x)) == 'cbrt(x)'  # note Cbrt due to cbrt already taken.
    assert C99CodePrinter().doprint(hypot(x, y)) == 'hypot(x, y)'
    assert C99CodePrinter().doprint(loggamma(x)) == 'lgamma(x)'
    assert C99CodePrinter().doprint(Max(x, 3, x**2)) == 'fmax(3, fmax(x, pow(x, 2)))'
    assert C99CodePrinter().doprint(Min(x, 3)) == 'fmin(3, x)'
    c99printer = C99CodePrinter()
    assert c99printer.language == 'C'
    assert c99printer.standard == 'C99'
    assert 'restrict' in c99printer.reserved_words
    assert 'using' not in c99printer.reserved_words


@XFAIL
def test_C99CodePrinter__precision_f80():
    f80_printer = C99CodePrinter({"type_aliases": {real: float80}})
    assert f80_printer.doprint(sin(x + Float('2.1'))) == 'sinl(x + 2.1L)'


def test_C99CodePrinter__precision():
    n = symbols('n', integer=True)
    p = symbols('p', integer=True, positive=True)
    f32_printer = C99CodePrinter({"type_aliases": {real: float32}})
    f64_printer = C99CodePrinter({"type_aliases": {real: float64}})
    f80_printer = C99CodePrinter({"type_aliases": {real: float80}})
    assert f32_printer.doprint(sin(x+2.1)) == 'sinf(x + 2.1F)'
    assert f64_printer.doprint(sin(x+2.1)) == 'sin(x + 2.1000000000000001)'
    assert f80_printer.doprint(sin(x+Float('2.0'))) == 'sinl(x + 2.0L)'

    for printer, suffix in zip([f32_printer, f64_printer, f80_printer], ['f', '', 'l']):
        def check(expr, ref):
            assert printer.doprint(expr) == ref.format(s=suffix, S=suffix.upper())
        check(Abs(n), 'abs(n)')
        check(Abs(x + 2.0), 'fabs{s}(x + 2.0{S})')
        check(sin(x + 4.0)**cos(x - 2.0), 'pow{s}(sin{s}(x + 4.0{S}), cos{s}(x - 2.0{S}))')
        check(exp(x*8.0), 'exp{s}(8.0{S}*x)')
        check(exp2(x), 'exp2{s}(x)')
        check(expm1(x*4.0), 'expm1{s}(4.0{S}*x)')
        check(Mod(p, 2), 'p % 2')
        check(Mod(2*p + 3, 3*p + 5, evaluate=False), '(2*p + 3) % (3*p + 5)')
        check(Mod(x + 2.0, 3.0), 'fmod{s}(1.0{S}*x + 2.0{S}, 3.0{S})')
        check(Mod(x, 2.0*x + 3.0), 'fmod{s}(1.0{S}*x, 2.0{S}*x + 3.0{S})')
        check(log(x/2), 'log{s}((1.0{S}/2.0{S})*x)')
        check(log10(3*x/2), 'log10{s}((3.0{S}/2.0{S})*x)')
        check(log2(x*8.0), 'log2{s}(8.0{S}*x)')
        check(log1p(x), 'log1p{s}(x)')
        check(2**x, 'pow{s}(2, x)')
        check(2.0**x, 'pow{s}(2.0{S}, x)')
        check(x**3, 'pow{s}(x, 3)')
        check(x**4.0, 'pow{s}(x, 4.0{S})')
        check(sqrt(3+x), 'sqrt{s}(x + 3)')
        check(Cbrt(x-2.0), 'cbrt{s}(x - 2.0{S})')
        check(hypot(x, y), 'hypot{s}(x, y)')
        check(sin(3.*x + 2.), 'sin{s}(3.0{S}*x + 2.0{S})')
        check(cos(3.*x - 1.), 'cos{s}(3.0{S}*x - 1.0{S})')
        check(tan(4.*y + 2.), 'tan{s}(4.0{S}*y + 2.0{S})')
        check(asin(3.*x + 2.), 'asin{s}(3.0{S}*x + 2.0{S})')
        check(acos(3.*x + 2.), 'acos{s}(3.0{S}*x + 2.0{S})')
        check(atan(3.*x + 2.), 'atan{s}(3.0{S}*x + 2.0{S})')
        check(atan2(3.*x, 2.*y), 'atan2{s}(3.0{S}*x, 2.0{S}*y)')

        check(sinh(3.*x + 2.), 'sinh{s}(3.0{S}*x + 2.0{S})')
        check(cosh(3.*x - 1.), 'cosh{s}(3.0{S}*x - 1.0{S})')
        check(tanh(4.0*y + 2.), 'tanh{s}(4.0{S}*y + 2.0{S})')
        check(asinh(3.*x + 2.), 'asinh{s}(3.0{S}*x + 2.0{S})')
        check(acosh(3.*x + 2.), 'acosh{s}(3.0{S}*x + 2.0{S})')
        check(atanh(3.*x + 2.), 'atanh{s}(3.0{S}*x + 2.0{S})')
        check(erf(42.*x), 'erf{s}(42.0{S}*x)')
        check(erfc(42.*x), 'erfc{s}(42.0{S}*x)')
        check(gamma(x), 'tgamma{s}(x)')
        check(loggamma(x), 'lgamma{s}(x)')

        check(ceiling(x + 2.), "ceil{s}(x) + 2")
        check(floor(x + 2.), "floor{s}(x) + 2")
        check(fma(x, y, -z), 'fma{s}(x, y, -z)')
        check(Max(x, 8.0, x**4.0), 'fmax{s}(8.0{S}, fmax{s}(x, pow{s}(x, 4.0{S})))')
        check(Min(x, 2.0), 'fmin{s}(2.0{S}, x)')


def test_get_math_macros():
    macros = get_math_macros()
    assert macros[exp(1)] == 'M_E'
    assert macros[1/Sqrt(2)] == 'M_SQRT1_2'


def test_ccode_Declaration():
    i = symbols('i', integer=True)
    var1 = Variable(i, type=Type.from_expr(i))
    dcl1 = Declaration(var1)
    assert ccode(dcl1) == 'int i'

    var2 = Variable(x, type=float32, attrs={value_const})
    dcl2a = Declaration(var2)
    assert ccode(dcl2a) == 'const float x'
    dcl2b = var2.as_Declaration(value=pi)
    assert ccode(dcl2b) == 'const float x = M_PI'

    var3 = Variable(y, type=Type('bool'))
    dcl3 = Declaration(var3)
    printer = C89CodePrinter()
    assert 'stdbool.h' not in printer.headers
    assert printer.doprint(dcl3) == 'bool y'
    assert 'stdbool.h' in printer.headers

    u = symbols('u', real=True)
    ptr4 = Pointer.deduced(u, attrs={pointer_const, restrict})
    dcl4 = Declaration(ptr4)
    assert ccode(dcl4) == 'double * const restrict u'

    var5 = Variable(x, Type('__float128'), attrs={value_const})
    dcl5a = Declaration(var5)
    assert ccode(dcl5a) == 'const __float128 x'
    var5b = Variable(var5.symbol, var5.type, pi, attrs=var5.attrs)
    dcl5b = Declaration(var5b)
    assert ccode(dcl5b) == 'const __float128 x = M_PI'


def test_C99CodePrinter_custom_type():
    # We will look at __float128 (new in glibc 2.26)
    f128 = FloatType('_Float128', float128.nbits, float128.nmant, float128.nexp)
    p128 = C99CodePrinter({
        "type_aliases": {real: f128},
        "type_literal_suffixes": {f128: 'Q'},
        "type_func_suffixes": {f128: 'f128'},
        "type_math_macro_suffixes": {
            real: 'f128',
            f128: 'f128'
        },
        "type_macros": {
            f128: ('__STDC_WANT_IEC_60559_TYPES_EXT__',)
        }
    })
    assert p128.doprint(x) == 'x'
    assert not p128.headers
    assert not p128.libraries
    assert not p128.macros
    assert p128.doprint(2.0) == '2.0Q'
    assert not p128.headers
    assert not p128.libraries
    assert p128.macros == {'__STDC_WANT_IEC_60559_TYPES_EXT__'}

    assert p128.doprint(Rational(1, 2)) == '1.0Q/2.0Q'
    assert p128.doprint(sin(x)) == 'sinf128(x)'
    assert p128.doprint(cos(2., evaluate=False)) == 'cosf128(2.0Q)'
    assert p128.doprint(x**-1.0) == '1.0Q/x'

    var5 = Variable(x, f128, attrs={value_const})

    dcl5a = Declaration(var5)
    assert ccode(dcl5a) == 'const _Float128 x'
    var5b = Variable(x, f128, pi, attrs={value_const})
    dcl5b = Declaration(var5b)
    assert p128.doprint(dcl5b) == 'const _Float128 x = M_PIf128'
    var5b = Variable(x, f128, value=Catalan.evalf(38), attrs={value_const})
    dcl5c = Declaration(var5b)
    assert p128.doprint(dcl5c) == 'const _Float128 x = %sQ' % Catalan.evalf(f128.decimal_dig)


def test_MatrixElement_printing():
    # test cases for issue #11821
    A = MatrixSymbol("A", 1, 3)
    B = MatrixSymbol("B", 1, 3)
    C = MatrixSymbol("C", 1, 3)

    assert(ccode(A[0, 0]) == "A[0]")
    assert(ccode(3 * A[0, 0]) == "3*A[0]")

    F = C[0, 0].subs(C, A - B)
    assert(ccode(F) == "(A - B)[0]")

def test_ccode_math_macros():
    assert ccode(z + exp(1)) == 'z + M_E'
    assert ccode(z + log2(exp(1))) == 'z + M_LOG2E'
    assert ccode(z + 1/log(2)) == 'z + M_LOG2E'
    assert ccode(z + log(2)) == 'z + M_LN2'
    assert ccode(z + log(10)) == 'z + M_LN10'
    assert ccode(z + pi) == 'z + M_PI'
    assert ccode(z + pi/2) == 'z + M_PI_2'
    assert ccode(z + pi/4) == 'z + M_PI_4'
    assert ccode(z + 1/pi) == 'z + M_1_PI'
    assert ccode(z + 2/pi) == 'z + M_2_PI'
    assert ccode(z + 2/sqrt(pi)) == 'z + M_2_SQRTPI'
    assert ccode(z + 2/Sqrt(pi)) == 'z + M_2_SQRTPI'
    assert ccode(z + sqrt(2)) == 'z + M_SQRT2'
    assert ccode(z + Sqrt(2)) == 'z + M_SQRT2'
    assert ccode(z + 1/sqrt(2)) == 'z + M_SQRT1_2'
    assert ccode(z + 1/Sqrt(2)) == 'z + M_SQRT1_2'


def test_ccode_Type():
    assert ccode(Type('float')) == 'float'
    assert ccode(intc) == 'int'


def test_ccode_codegen_ast():
    # Note that C only allows comments of the form /* ... */, double forward
    # slash is not standard C, and some C compilers will grind to a halt upon
    # encountering them.
    assert ccode(Comment("this is a comment")) == "/* this is a comment */"  # not //
    assert ccode(While(abs(x) > 1, [aug_assign(x, '-', 1)])) == (
        'while (fabs(x) > 1) {\n'
        '   x -= 1;\n'
        '}'
    )
    assert ccode(Scope([AddAugmentedAssignment(x, 1)])) == (
        '{\n'
        '   x += 1;\n'
        '}'
    )
    inp_x = Declaration(Variable(x, type=real))
    assert ccode(FunctionPrototype(real, 'pwer', [inp_x])) == 'double pwer(double x)'
    assert ccode(FunctionDefinition(real, 'pwer', [inp_x], [Assignment(x, x**2)])) == (
        'double pwer(double x){\n'
        '   x = pow(x, 2);\n'
        '}'
    )

    # Elements of CodeBlock are formatted as statements:
    block = CodeBlock(
        x,
        Print([x, y], "%d %d"),
        Print([QuotedString('hello'), y], "%s %d", file=stderr),
        FunctionCall('pwer', [x]),
        Return(x),
    )
    assert ccode(block) == '\n'.join([
        'x;',
        'printf("%d %d", x, y);',
        'fprintf(stderr, "%s %d", "hello", y);',
        'pwer(x);',
        'return x;',
    ])

def test_ccode_UnevaluatedExpr():
    assert ccode(UnevaluatedExpr(y * x) + z) == "z + x*y"
    assert ccode(UnevaluatedExpr(y + x) + z) == "z + (x + y)"  # gh-21955
    w = symbols('w')
    assert ccode(UnevaluatedExpr(y + x) + UnevaluatedExpr(z + w)) == "(w + z) + (x + y)"

    p, q, r = symbols("p q r", real=True)
    q_r = UnevaluatedExpr(q + r)
    expr = abs(exp(p+q_r))
    assert ccode(expr) == "exp(p + (q + r))"


def test_ccode_array_like_containers():
    assert ccode([2,3,4]) == "{2, 3, 4}"
    assert ccode((2,3,4)) == "{2, 3, 4}"
