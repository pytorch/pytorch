from sympy.core import (S, pi, oo, Symbol, symbols, Rational, Integer,
                        GoldenRatio, EulerGamma, Catalan, Lambda, Dummy)
from sympy.functions import (Piecewise, sin, cos, Abs, exp, ceiling, sqrt,
                             gamma, sign, Max, Min, factorial, beta)
from sympy.core.relational import (Eq, Ge, Gt, Le, Lt, Ne)
from sympy.sets import Range
from sympy.logic import ITE
from sympy.codegen import For, aug_assign, Assignment
from sympy.testing.pytest import raises
from sympy.printing.rcode import RCodePrinter
from sympy.utilities.lambdify import implemented_function
from sympy.tensor import IndexedBase, Idx
from sympy.matrices import Matrix, MatrixSymbol

from sympy.printing.rcode import rcode

x, y, z = symbols('x,y,z')


def test_printmethod():
    class fabs(Abs):
        def _rcode(self, printer):
            return "abs(%s)" % printer._print(self.args[0])

    assert rcode(fabs(x)) == "abs(x)"


def test_rcode_sqrt():
    assert rcode(sqrt(x)) == "sqrt(x)"
    assert rcode(x**0.5) == "sqrt(x)"
    assert rcode(sqrt(x)) == "sqrt(x)"


def test_rcode_Pow():
    assert rcode(x**3) == "x^3"
    assert rcode(x**(y**3)) == "x^(y^3)"
    g = implemented_function('g', Lambda(x, 2*x))
    assert rcode(1/(g(x)*3.5)**(x - y**x)/(x**2 + y)) == \
        "(3.5*2*x)^(-x + y^x)/(x^2 + y)"
    assert rcode(x**-1.0) == '1.0/x'
    assert rcode(x**Rational(2, 3)) == 'x^(2.0/3.0)'
    _cond_cfunc = [(lambda base, exp: exp.is_integer, "dpowi"),
                   (lambda base, exp: not exp.is_integer, "pow")]
    assert rcode(x**3, user_functions={'Pow': _cond_cfunc}) == 'dpowi(x, 3)'
    assert rcode(x**3.2, user_functions={'Pow': _cond_cfunc}) == 'pow(x, 3.2)'


def test_rcode_Max():
    # Test for gh-11926
    assert rcode(Max(x,x*x),user_functions={"Max":"my_max", "Pow":"my_pow"}) == 'my_max(x, my_pow(x, 2))'


def test_rcode_constants_mathh():
    assert rcode(exp(1)) == "exp(1)"
    assert rcode(pi) == "pi"
    assert rcode(oo) == "Inf"
    assert rcode(-oo) == "-Inf"


def test_rcode_constants_other():
    assert rcode(2*GoldenRatio) == "GoldenRatio = 1.61803398874989;\n2*GoldenRatio"
    assert rcode(
        2*Catalan) == "Catalan = 0.915965594177219;\n2*Catalan"
    assert rcode(2*EulerGamma) == "EulerGamma = 0.577215664901533;\n2*EulerGamma"


def test_rcode_Rational():
    assert rcode(Rational(3, 7)) == "3.0/7.0"
    assert rcode(Rational(18, 9)) == "2"
    assert rcode(Rational(3, -7)) == "-3.0/7.0"
    assert rcode(Rational(-3, -7)) == "3.0/7.0"
    assert rcode(x + Rational(3, 7)) == "x + 3.0/7.0"
    assert rcode(Rational(3, 7)*x) == "(3.0/7.0)*x"


def test_rcode_Integer():
    assert rcode(Integer(67)) == "67"
    assert rcode(Integer(-1)) == "-1"


def test_rcode_functions():
    assert rcode(sin(x) ** cos(x)) == "sin(x)^cos(x)"
    assert rcode(factorial(x) + gamma(y)) == "factorial(x) + gamma(y)"
    assert rcode(beta(Min(x, y), Max(x, y))) == "beta(min(x, y), max(x, y))"


def test_rcode_inline_function():
    x = symbols('x')
    g = implemented_function('g', Lambda(x, 2*x))
    assert rcode(g(x)) == "2*x"
    g = implemented_function('g', Lambda(x, 2*x/Catalan))
    assert rcode(
        g(x)) == "Catalan = %s;\n2*x/Catalan" % Catalan.n()
    A = IndexedBase('A')
    i = Idx('i', symbols('n', integer=True))
    g = implemented_function('g', Lambda(x, x*(1 + x)*(2 + x)))
    res=rcode(g(A[i]), assign_to=A[i])
    ref=(
        "for (i in 1:n){\n"
        "   A[i] = (A[i] + 1)*(A[i] + 2)*A[i];\n"
        "}"
    )
    assert res == ref


def test_rcode_exceptions():
    assert rcode(ceiling(x)) == "ceiling(x)"
    assert rcode(Abs(x)) == "abs(x)"
    assert rcode(gamma(x)) == "gamma(x)"


def test_rcode_user_functions():
    x = symbols('x', integer=False)
    n = symbols('n', integer=True)
    custom_functions = {
        "ceiling": "myceil",
        "Abs": [(lambda x: not x.is_integer, "fabs"), (lambda x: x.is_integer, "abs")],
    }
    assert rcode(ceiling(x), user_functions=custom_functions) == "myceil(x)"
    assert rcode(Abs(x), user_functions=custom_functions) == "fabs(x)"
    assert rcode(Abs(n), user_functions=custom_functions) == "abs(n)"


def test_rcode_boolean():
    assert rcode(True) == "True"
    assert rcode(S.true) == "True"
    assert rcode(False) == "False"
    assert rcode(S.false) == "False"
    assert rcode(x & y) == "x & y"
    assert rcode(x | y) == "x | y"
    assert rcode(~x) == "!x"
    assert rcode(x & y & z) == "x & y & z"
    assert rcode(x | y | z) == "x | y | z"
    assert rcode((x & y) | z) == "z | x & y"
    assert rcode((x | y) & z) == "z & (x | y)"

def test_rcode_Relational():
    assert rcode(Eq(x, y)) == "x == y"
    assert rcode(Ne(x, y)) == "x != y"
    assert rcode(Le(x, y)) == "x <= y"
    assert rcode(Lt(x, y)) == "x < y"
    assert rcode(Gt(x, y)) == "x > y"
    assert rcode(Ge(x, y)) == "x >= y"


def test_rcode_Piecewise():
    expr = Piecewise((x, x < 1), (x**2, True))
    res=rcode(expr)
    ref="ifelse(x < 1,x,x^2)"
    assert res == ref
    tau=Symbol("tau")
    res=rcode(expr,tau)
    ref="tau = ifelse(x < 1,x,x^2);"
    assert res == ref

    expr = 2*Piecewise((x, x < 1), (x**2, x<2), (x**3,True))
    assert rcode(expr) == "2*ifelse(x < 1,x,ifelse(x < 2,x^2,x^3))"
    res = rcode(expr, assign_to='c')
    assert res == "c = 2*ifelse(x < 1,x,ifelse(x < 2,x^2,x^3));"

    # Check that Piecewise without a True (default) condition error
    #expr = Piecewise((x, x < 1), (x**2, x > 1), (sin(x), x > 0))
    #raises(ValueError, lambda: rcode(expr))
    expr = 2*Piecewise((x, x < 1), (x**2, x<2))
    assert(rcode(expr))== "2*ifelse(x < 1,x,ifelse(x < 2,x^2,NA))"


def test_rcode_sinc():
    from sympy.functions.elementary.trigonometric import sinc
    expr = sinc(x)
    res = rcode(expr)
    ref = "(ifelse(x != 0,sin(x)/x,1))"
    assert res == ref


def test_rcode_Piecewise_deep():
    p = rcode(2*Piecewise((x, x < 1), (x + 1, x < 2), (x**2, True)))
    assert p == "2*ifelse(x < 1,x,ifelse(x < 2,x + 1,x^2))"
    expr = x*y*z + x**2 + y**2 + Piecewise((0, x < 0.5), (1, True)) + cos(z) - 1
    p = rcode(expr)
    ref="x^2 + x*y*z + y^2 + ifelse(x < 0.5,0,1) + cos(z) - 1"
    assert p == ref

    ref="c = x^2 + x*y*z + y^2 + ifelse(x < 0.5,0,1) + cos(z) - 1;"
    p = rcode(expr, assign_to='c')
    assert p == ref


def test_rcode_ITE():
    expr = ITE(x < 1, y, z)
    p = rcode(expr)
    ref="ifelse(x < 1,y,z)"
    assert p == ref


def test_rcode_settings():
    raises(TypeError, lambda: rcode(sin(x), method="garbage"))


def test_rcode_Indexed():
    n, m, o = symbols('n m o', integer=True)
    i, j, k = Idx('i', n), Idx('j', m), Idx('k', o)
    p = RCodePrinter()
    p._not_r = set()

    x = IndexedBase('x')[j]
    assert p._print_Indexed(x) == 'x[j]'
    A = IndexedBase('A')[i, j]
    assert p._print_Indexed(A) == 'A[i, j]'
    B = IndexedBase('B')[i, j, k]
    assert p._print_Indexed(B) == 'B[i, j, k]'

    assert p._not_r == set()

def test_rcode_Indexed_without_looking_for_contraction():
    len_y = 5
    y = IndexedBase('y', shape=(len_y,))
    x = IndexedBase('x', shape=(len_y,))
    Dy = IndexedBase('Dy', shape=(len_y-1,))
    i = Idx('i', len_y-1)
    e=Eq(Dy[i], (y[i+1]-y[i])/(x[i+1]-x[i]))
    code0 = rcode(e.rhs, assign_to=e.lhs, contract=False)
    assert code0 == 'Dy[i] = (y[%s] - y[i])/(x[%s] - x[i]);' % (i + 1, i + 1)


def test_rcode_loops_matrix_vector():
    n, m = symbols('n m', integer=True)
    A = IndexedBase('A')
    x = IndexedBase('x')
    y = IndexedBase('y')
    i = Idx('i', m)
    j = Idx('j', n)

    s = (
        'for (i in 1:m){\n'
        '   y[i] = 0;\n'
        '}\n'
        'for (i in 1:m){\n'
        '   for (j in 1:n){\n'
        '      y[i] = A[i, j]*x[j] + y[i];\n'
        '   }\n'
        '}'
    )
    c = rcode(A[i, j]*x[j], assign_to=y[i])
    assert c == s


def test_dummy_loops():
    # the following line could also be
    # [Dummy(s, integer=True) for s in 'im']
    # or [Dummy(integer=True) for s in 'im']
    i, m = symbols('i m', integer=True, cls=Dummy)
    x = IndexedBase('x')
    y = IndexedBase('y')
    i = Idx(i, m)

    expected = (
            'for (i_%(icount)i in 1:m_%(mcount)i){\n'
        '   y[i_%(icount)i] = x[i_%(icount)i];\n'
        '}'
    ) % {'icount': i.label.dummy_index, 'mcount': m.dummy_index}
    code = rcode(x[i], assign_to=y[i])
    assert code == expected


def test_rcode_loops_add():
    n, m = symbols('n m', integer=True)
    A = IndexedBase('A')
    x = IndexedBase('x')
    y = IndexedBase('y')
    z = IndexedBase('z')
    i = Idx('i', m)
    j = Idx('j', n)

    s = (
        'for (i in 1:m){\n'
        '   y[i] = x[i] + z[i];\n'
        '}\n'
        'for (i in 1:m){\n'
        '   for (j in 1:n){\n'
        '      y[i] = A[i, j]*x[j] + y[i];\n'
        '   }\n'
        '}'
    )
    c = rcode(A[i, j]*x[j] + x[i] + z[i], assign_to=y[i])
    assert c == s


def test_rcode_loops_multiple_contractions():
    n, m, o, p = symbols('n m o p', integer=True)
    a = IndexedBase('a')
    b = IndexedBase('b')
    y = IndexedBase('y')
    i = Idx('i', m)
    j = Idx('j', n)
    k = Idx('k', o)
    l = Idx('l', p)

    s = (
        'for (i in 1:m){\n'
        '   y[i] = 0;\n'
        '}\n'
        'for (i in 1:m){\n'
        '   for (j in 1:n){\n'
        '      for (k in 1:o){\n'
        '         for (l in 1:p){\n'
        '            y[i] = a[i, j, k, l]*b[j, k, l] + y[i];\n'
        '         }\n'
        '      }\n'
        '   }\n'
        '}'
    )
    c = rcode(b[j, k, l]*a[i, j, k, l], assign_to=y[i])
    assert c == s


def test_rcode_loops_addfactor():
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
        'for (i in 1:m){\n'
        '   y[i] = 0;\n'
        '}\n'
        'for (i in 1:m){\n'
        '   for (j in 1:n){\n'
        '      for (k in 1:o){\n'
        '         for (l in 1:p){\n'
        '            y[i] = (a[i, j, k, l] + b[i, j, k, l])*c[j, k, l] + y[i];\n'
        '         }\n'
        '      }\n'
        '   }\n'
        '}'
    )
    c = rcode((a[i, j, k, l] + b[i, j, k, l])*c[j, k, l], assign_to=y[i])
    assert c == s


def test_rcode_loops_multiple_terms():
    n, m, o, p = symbols('n m o p', integer=True)
    a = IndexedBase('a')
    b = IndexedBase('b')
    c = IndexedBase('c')
    y = IndexedBase('y')
    i = Idx('i', m)
    j = Idx('j', n)
    k = Idx('k', o)

    s0 = (
        'for (i in 1:m){\n'
        '   y[i] = 0;\n'
        '}\n'
    )
    s1 = (
        'for (i in 1:m){\n'
        '   for (j in 1:n){\n'
        '      for (k in 1:o){\n'
        '         y[i] = b[j]*b[k]*c[i, j, k] + y[i];\n'
        '      }\n'
        '   }\n'
        '}\n'
    )
    s2 = (
        'for (i in 1:m){\n'
        '   for (k in 1:o){\n'
        '      y[i] = a[i, k]*b[k] + y[i];\n'
        '   }\n'
        '}\n'
    )
    s3 = (
        'for (i in 1:m){\n'
        '   for (j in 1:n){\n'
        '      y[i] = a[i, j]*b[j] + y[i];\n'
        '   }\n'
        '}\n'
    )
    c = rcode(
        b[j]*a[i, j] + b[k]*a[i, k] + b[j]*b[k]*c[i, j, k], assign_to=y[i])

    ref={}
    ref[0] = s0 + s1 + s2 + s3[:-1]
    ref[1] = s0 + s1 + s3 + s2[:-1]
    ref[2] = s0 + s2 + s1 + s3[:-1]
    ref[3] = s0 + s2 + s3 + s1[:-1]
    ref[4] = s0 + s3 + s1 + s2[:-1]
    ref[5] = s0 + s3 + s2 + s1[:-1]

    assert (c == ref[0] or
            c == ref[1] or
            c == ref[2] or
            c == ref[3] or
            c == ref[4] or
            c == ref[5])


def test_dereference_printing():
    expr = x + y + sin(z) + z
    assert rcode(expr, dereference=[z]) == "x + y + (*z) + sin((*z))"


def test_Matrix_printing():
    # Test returning a Matrix
    mat = Matrix([x*y, Piecewise((2 + x, y>0), (y, True)), sin(z)])
    A = MatrixSymbol('A', 3, 1)
    p = rcode(mat, A)
    assert p == (
        "A[0] = x*y;\n"
        "A[1] = ifelse(y > 0,x + 2,y);\n"
        "A[2] = sin(z);")
    # Test using MatrixElements in expressions
    expr = Piecewise((2*A[2, 0], x > 0), (A[2, 0], True)) + sin(A[1, 0]) + A[0, 0]
    p = rcode(expr)
    assert p  == ("ifelse(x > 0,2*A[2],A[2]) + sin(A[1]) + A[0]")
    # Test using MatrixElements in a Matrix
    q = MatrixSymbol('q', 5, 1)
    M = MatrixSymbol('M', 3, 3)
    m = Matrix([[sin(q[1,0]), 0, cos(q[2,0])],
        [q[1,0] + q[2,0], q[3, 0], 5],
        [2*q[4, 0]/q[1,0], sqrt(q[0,0]) + 4, 0]])
    assert rcode(m, M) == (
        "M[0] = sin(q[1]);\n"
        "M[1] = 0;\n"
        "M[2] = cos(q[2]);\n"
        "M[3] = q[1] + q[2];\n"
        "M[4] = q[3];\n"
        "M[5] = 5;\n"
        "M[6] = 2*q[4]/q[1];\n"
        "M[7] = sqrt(q[0]) + 4;\n"
        "M[8] = 0;")


def test_rcode_sgn():

    expr = sign(x) * y
    assert rcode(expr) == 'y*sign(x)'
    p = rcode(expr, 'z')
    assert p  == 'z = y*sign(x);'

    p = rcode(sign(2 * x + x**2) * x + x**2)
    assert p  == "x^2 + x*sign(x^2 + 2*x)"

    expr = sign(cos(x))
    p = rcode(expr)
    assert p == 'sign(cos(x))'

def test_rcode_Assignment():
    assert rcode(Assignment(x, y + z)) == 'x = y + z;'
    assert rcode(aug_assign(x, '+', y + z)) == 'x += y + z;'


def test_rcode_For():
    f = For(x, Range(0, 10, 2), [aug_assign(y, '*', x)])
    sol = rcode(f)
    assert sol == ("for(x in seq(from=0, to=9, by=2){\n"
                   "   y *= x;\n"
                   "}")


def test_MatrixElement_printing():
    # test cases for issue #11821
    A = MatrixSymbol("A", 1, 3)
    B = MatrixSymbol("B", 1, 3)
    C = MatrixSymbol("C", 1, 3)

    assert(rcode(A[0, 0]) == "A[0]")
    assert(rcode(3 * A[0, 0]) == "3*A[0]")

    F = C[0, 0].subs(C, A - B)
    assert(rcode(F) == "(A - B)[0]")
