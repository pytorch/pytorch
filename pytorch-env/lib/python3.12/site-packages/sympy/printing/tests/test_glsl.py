from sympy.core import (pi, symbols, Rational, Integer, GoldenRatio, EulerGamma,
                        Catalan, Lambda, Dummy, Eq, Ne, Le, Lt, Gt, Ge)
from sympy.functions import Piecewise, sin, cos, Abs, exp, ceiling, sqrt
from sympy.testing.pytest import raises, warns_deprecated_sympy
from sympy.printing.glsl import GLSLPrinter
from sympy.printing.str import StrPrinter
from sympy.utilities.lambdify import implemented_function
from sympy.tensor import IndexedBase, Idx
from sympy.matrices import Matrix, MatrixSymbol
from sympy.core import Tuple
from sympy.printing.glsl import glsl_code
import textwrap

x, y, z = symbols('x,y,z')


def test_printmethod():
    assert glsl_code(Abs(x)) == "abs(x)"

def test_print_without_operators():
    assert glsl_code(x*y,use_operators = False) == 'mul(x, y)'
    assert glsl_code(x**y+z,use_operators = False) == 'add(pow(x, y), z)'
    assert glsl_code(x*(y+z),use_operators = False) == 'mul(x, add(y, z))'
    assert glsl_code(x*(y+z),use_operators = False) == 'mul(x, add(y, z))'
    assert glsl_code(x*(y+z**y**0.5),use_operators = False) == 'mul(x, add(y, pow(z, sqrt(y))))'
    assert glsl_code(-x-y, use_operators=False, zero='zero()') == 'sub(zero(), add(x, y))'
    assert glsl_code(-x-y, use_operators=False) == 'sub(0.0, add(x, y))'

def test_glsl_code_sqrt():
    assert glsl_code(sqrt(x)) == "sqrt(x)"
    assert glsl_code(x**0.5) == "sqrt(x)"
    assert glsl_code(sqrt(x)) == "sqrt(x)"


def test_glsl_code_Pow():
    g = implemented_function('g', Lambda(x, 2*x))
    assert glsl_code(x**3) == "pow(x, 3.0)"
    assert glsl_code(x**(y**3)) == "pow(x, pow(y, 3.0))"
    assert glsl_code(1/(g(x)*3.5)**(x - y**x)/(x**2 + y)) == \
        "pow(3.5*2*x, -x + pow(y, x))/(pow(x, 2.0) + y)"
    assert glsl_code(x**-1.0) == '1.0/x'


def test_glsl_code_Relational():
    assert glsl_code(Eq(x, y)) == "x == y"
    assert glsl_code(Ne(x, y)) == "x != y"
    assert glsl_code(Le(x, y)) == "x <= y"
    assert glsl_code(Lt(x, y)) == "x < y"
    assert glsl_code(Gt(x, y)) == "x > y"
    assert glsl_code(Ge(x, y)) == "x >= y"


def test_glsl_code_constants_mathh():
    assert glsl_code(exp(1)) == "float E = 2.71828183;\nE"
    assert glsl_code(pi) == "float pi = 3.14159265;\npi"
    # assert glsl_code(oo) == "Number.POSITIVE_INFINITY"
    # assert glsl_code(-oo) == "Number.NEGATIVE_INFINITY"


def test_glsl_code_constants_other():
    assert glsl_code(2*GoldenRatio) == "float GoldenRatio = 1.61803399;\n2*GoldenRatio"
    assert glsl_code(2*Catalan) == "float Catalan = 0.915965594;\n2*Catalan"
    assert glsl_code(2*EulerGamma) == "float EulerGamma = 0.577215665;\n2*EulerGamma"


def test_glsl_code_Rational():
    assert glsl_code(Rational(3, 7)) == "3.0/7.0"
    assert glsl_code(Rational(18, 9)) == "2"
    assert glsl_code(Rational(3, -7)) == "-3.0/7.0"
    assert glsl_code(Rational(-3, -7)) == "3.0/7.0"


def test_glsl_code_Integer():
    assert glsl_code(Integer(67)) == "67"
    assert glsl_code(Integer(-1)) == "-1"


def test_glsl_code_functions():
    assert glsl_code(sin(x) ** cos(x)) == "pow(sin(x), cos(x))"


def test_glsl_code_inline_function():
    x = symbols('x')
    g = implemented_function('g', Lambda(x, 2*x))
    assert glsl_code(g(x)) == "2*x"
    g = implemented_function('g', Lambda(x, 2*x/Catalan))
    assert glsl_code(g(x)) == "float Catalan = 0.915965594;\n2*x/Catalan"
    A = IndexedBase('A')
    i = Idx('i', symbols('n', integer=True))
    g = implemented_function('g', Lambda(x, x*(1 + x)*(2 + x)))
    assert glsl_code(g(A[i]), assign_to=A[i]) == (
        "for (int i=0; i<n; i++){\n"
        "   A[i] = (A[i] + 1)*(A[i] + 2)*A[i];\n"
        "}"
    )


def test_glsl_code_exceptions():
    assert glsl_code(ceiling(x)) == "ceil(x)"
    assert glsl_code(Abs(x)) == "abs(x)"


def test_glsl_code_boolean():
    assert glsl_code(x & y) == "x && y"
    assert glsl_code(x | y) == "x || y"
    assert glsl_code(~x) == "!x"
    assert glsl_code(x & y & z) == "x && y && z"
    assert glsl_code(x | y | z) == "x || y || z"
    assert glsl_code((x & y) | z) == "z || x && y"
    assert glsl_code((x | y) & z) == "z && (x || y)"


def test_glsl_code_Piecewise():
    expr = Piecewise((x, x < 1), (x**2, True))
    p = glsl_code(expr)
    s = \
"""\
((x < 1) ? (
   x
)
: (
   pow(x, 2.0)
))\
"""
    assert p == s
    assert glsl_code(expr, assign_to="c") == (
    "if (x < 1) {\n"
    "   c = x;\n"
    "}\n"
    "else {\n"
    "   c = pow(x, 2.0);\n"
    "}")
    # Check that Piecewise without a True (default) condition error
    expr = Piecewise((x, x < 1), (x**2, x > 1), (sin(x), x > 0))
    raises(ValueError, lambda: glsl_code(expr))


def test_glsl_code_Piecewise_deep():
    p = glsl_code(2*Piecewise((x, x < 1), (x**2, True)))
    s = \
"""\
2*((x < 1) ? (
   x
)
: (
   pow(x, 2.0)
))\
"""
    assert p == s


def test_glsl_code_settings():
    raises(TypeError, lambda: glsl_code(sin(x), method="garbage"))


def test_glsl_code_Indexed():
    n, m, o = symbols('n m o', integer=True)
    i, j, k = Idx('i', n), Idx('j', m), Idx('k', o)
    p = GLSLPrinter()
    p._not_c = set()

    x = IndexedBase('x')[j]
    assert p._print_Indexed(x) == 'x[j]'
    A = IndexedBase('A')[i, j]
    assert p._print_Indexed(A) == 'A[%s]' % (m*i+j)
    B = IndexedBase('B')[i, j, k]
    assert p._print_Indexed(B) == 'B[%s]' % (i*o*m+j*o+k)

    assert p._not_c == set()

def test_glsl_code_list_tuple_Tuple():
    assert glsl_code([1,2,3,4]) == 'vec4(1, 2, 3, 4)'
    assert glsl_code([1,2,3],glsl_types=False) == 'float[3](1, 2, 3)'
    assert glsl_code([1,2,3]) == glsl_code((1,2,3))
    assert glsl_code([1,2,3]) == glsl_code(Tuple(1,2,3))

    m = MatrixSymbol('A',3,4)
    assert glsl_code([m[0],m[1]])

def test_glsl_code_loops_matrix_vector():
    n, m = symbols('n m', integer=True)
    A = IndexedBase('A')
    x = IndexedBase('x')
    y = IndexedBase('y')
    i = Idx('i', m)
    j = Idx('j', n)

    s = (
        'for (int i=0; i<m; i++){\n'
        '   y[i] = 0.0;\n'
        '}\n'
        'for (int i=0; i<m; i++){\n'
        '   for (int j=0; j<n; j++){\n'
        '      y[i] = A[n*i + j]*x[j] + y[i];\n'
        '   }\n'
        '}'
    )

    c = glsl_code(A[i, j]*x[j], assign_to=y[i])
    assert c == s


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
    code = glsl_code(x[i], assign_to=y[i])
    assert code == expected


def test_glsl_code_loops_add():
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
        '      y[i] = A[n*i + j]*x[j] + y[i];\n'
        '   }\n'
        '}'
    )
    c = glsl_code(A[i, j]*x[j] + x[i] + z[i], assign_to=y[i])
    assert c == s


def test_glsl_code_loops_multiple_contractions():
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
        '   y[i] = 0.0;\n'
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
    c = glsl_code(b[j, k, l]*a[i, j, k, l], assign_to=y[i])
    assert c == s


def test_glsl_code_loops_addfactor():
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
        '   y[i] = 0.0;\n'
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
    c = glsl_code((a[i, j, k, l] + b[i, j, k, l])*c[j, k, l], assign_to=y[i])
    assert c == s


def test_glsl_code_loops_multiple_terms():
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
        '   y[i] = 0.0;\n'
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
    c = glsl_code(
        b[j]*a[i, j] + b[k]*a[i, k] + b[j]*b[k]*c[i, j, k], assign_to=y[i])
    assert (c == s0 + s1 + s2 + s3[:-1] or
            c == s0 + s1 + s3 + s2[:-1] or
            c == s0 + s2 + s1 + s3[:-1] or
            c == s0 + s2 + s3 + s1[:-1] or
            c == s0 + s3 + s1 + s2[:-1] or
            c == s0 + s3 + s2 + s1[:-1])


def test_Matrix_printing():
    # Test returning a Matrix

    mat = Matrix([x*y, Piecewise((2 + x, y>0), (y, True)), sin(z)])
    A = MatrixSymbol('A', 3, 1)
    assert glsl_code(mat, assign_to=A) == (
'''A[0][0] = x*y;
if (y > 0) {
   A[1][0] = x + 2;
}
else {
   A[1][0] = y;
}
A[2][0] = sin(z);''' )
    assert glsl_code(Matrix([A[0],A[1]]))
    # Test using MatrixElements in expressions
    expr = Piecewise((2*A[2, 0], x > 0), (A[2, 0], True)) + sin(A[1, 0]) + A[0, 0]
    assert glsl_code(expr) == (
'''((x > 0) ? (
   2*A[2][0]
)
: (
   A[2][0]
)) + sin(A[1][0]) + A[0][0]''' )

    # Test using MatrixElements in a Matrix
    q = MatrixSymbol('q', 5, 1)
    M = MatrixSymbol('M', 3, 3)
    m = Matrix([[sin(q[1,0]), 0, cos(q[2,0])],
        [q[1,0] + q[2,0], q[3, 0], 5],
        [2*q[4, 0]/q[1,0], sqrt(q[0,0]) + 4, 0]])
    assert glsl_code(m,M) == (
'''M[0][0] = sin(q[1]);
M[0][1] = 0;
M[0][2] = cos(q[2]);
M[1][0] = q[1] + q[2];
M[1][1] = q[3];
M[1][2] = 5;
M[2][0] = 2*q[4]/q[1];
M[2][1] = sqrt(q[0]) + 4;
M[2][2] = 0;'''
        )

def test_Matrices_1x7():
    gl = glsl_code
    A = Matrix([1,2,3,4,5,6,7])
    assert gl(A) == 'float[7](1, 2, 3, 4, 5, 6, 7)'
    assert gl(A.transpose()) == 'float[7](1, 2, 3, 4, 5, 6, 7)'

def test_Matrices_1x7_array_type_int():
    gl = glsl_code
    A = Matrix([1,2,3,4,5,6,7])
    assert gl(A, array_type='int') == 'int[7](1, 2, 3, 4, 5, 6, 7)'

def test_Tuple_array_type_custom():
    gl = glsl_code
    A = symbols('a b c')
    assert gl(A, array_type='AbcType', glsl_types=False) == 'AbcType[3](a, b, c)'

def test_Matrices_1x7_spread_assign_to_symbols():
    gl = glsl_code
    A = Matrix([1,2,3,4,5,6,7])
    assign_to = symbols('x.a x.b x.c x.d x.e x.f x.g')
    assert gl(A, assign_to=assign_to) == textwrap.dedent('''\
        x.a = 1;
        x.b = 2;
        x.c = 3;
        x.d = 4;
        x.e = 5;
        x.f = 6;
        x.g = 7;'''
    )

def test_spread_assign_to_nested_symbols():
    gl = glsl_code
    expr = ((1,2,3), (1,2,3))
    assign_to = (symbols('a b c'), symbols('x y z'))
    assert gl(expr, assign_to=assign_to) == textwrap.dedent('''\
        a = 1;
        b = 2;
        c = 3;
        x = 1;
        y = 2;
        z = 3;'''
    )

def test_spread_assign_to_deeply_nested_symbols():
    gl = glsl_code
    a, b, c, x, y, z = symbols('a b c x y z')
    expr = (((1,2),3), ((1,2),3))
    assign_to = (((a, b), c), ((x, y), z))
    assert gl(expr, assign_to=assign_to) == textwrap.dedent('''\
        a = 1;
        b = 2;
        c = 3;
        x = 1;
        y = 2;
        z = 3;'''
    )

def test_matrix_of_tuples_spread_assign_to_symbols():
    gl = glsl_code
    with warns_deprecated_sympy():
        expr = Matrix([[(1,2),(3,4)],[(5,6),(7,8)]])
    assign_to = (symbols('a b'), symbols('c d'), symbols('e f'), symbols('g h'))
    assert gl(expr, assign_to) == textwrap.dedent('''\
        a = 1;
        b = 2;
        c = 3;
        d = 4;
        e = 5;
        f = 6;
        g = 7;
        h = 8;'''
    )

def test_cannot_assign_to_cause_mismatched_length():
    expr = (1, 2)
    assign_to = symbols('x y z')
    raises(ValueError, lambda: glsl_code(expr, assign_to))

def test_matrix_4x4_assign():
    gl = glsl_code
    expr = MatrixSymbol('A',4,4) * MatrixSymbol('B',4,4) + MatrixSymbol('C',4,4)
    assign_to = MatrixSymbol('X',4,4)
    assert gl(expr, assign_to=assign_to) == textwrap.dedent('''\
        X[0][0] = A[0][0]*B[0][0] + A[0][1]*B[1][0] + A[0][2]*B[2][0] + A[0][3]*B[3][0] + C[0][0];
        X[0][1] = A[0][0]*B[0][1] + A[0][1]*B[1][1] + A[0][2]*B[2][1] + A[0][3]*B[3][1] + C[0][1];
        X[0][2] = A[0][0]*B[0][2] + A[0][1]*B[1][2] + A[0][2]*B[2][2] + A[0][3]*B[3][2] + C[0][2];
        X[0][3] = A[0][0]*B[0][3] + A[0][1]*B[1][3] + A[0][2]*B[2][3] + A[0][3]*B[3][3] + C[0][3];
        X[1][0] = A[1][0]*B[0][0] + A[1][1]*B[1][0] + A[1][2]*B[2][0] + A[1][3]*B[3][0] + C[1][0];
        X[1][1] = A[1][0]*B[0][1] + A[1][1]*B[1][1] + A[1][2]*B[2][1] + A[1][3]*B[3][1] + C[1][1];
        X[1][2] = A[1][0]*B[0][2] + A[1][1]*B[1][2] + A[1][2]*B[2][2] + A[1][3]*B[3][2] + C[1][2];
        X[1][3] = A[1][0]*B[0][3] + A[1][1]*B[1][3] + A[1][2]*B[2][3] + A[1][3]*B[3][3] + C[1][3];
        X[2][0] = A[2][0]*B[0][0] + A[2][1]*B[1][0] + A[2][2]*B[2][0] + A[2][3]*B[3][0] + C[2][0];
        X[2][1] = A[2][0]*B[0][1] + A[2][1]*B[1][1] + A[2][2]*B[2][1] + A[2][3]*B[3][1] + C[2][1];
        X[2][2] = A[2][0]*B[0][2] + A[2][1]*B[1][2] + A[2][2]*B[2][2] + A[2][3]*B[3][2] + C[2][2];
        X[2][3] = A[2][0]*B[0][3] + A[2][1]*B[1][3] + A[2][2]*B[2][3] + A[2][3]*B[3][3] + C[2][3];
        X[3][0] = A[3][0]*B[0][0] + A[3][1]*B[1][0] + A[3][2]*B[2][0] + A[3][3]*B[3][0] + C[3][0];
        X[3][1] = A[3][0]*B[0][1] + A[3][1]*B[1][1] + A[3][2]*B[2][1] + A[3][3]*B[3][1] + C[3][1];
        X[3][2] = A[3][0]*B[0][2] + A[3][1]*B[1][2] + A[3][2]*B[2][2] + A[3][3]*B[3][2] + C[3][2];
        X[3][3] = A[3][0]*B[0][3] + A[3][1]*B[1][3] + A[3][2]*B[2][3] + A[3][3]*B[3][3] + C[3][3];'''
    )

def test_1xN_vecs():
    gl = glsl_code
    for i in range(1,10):
        A = Matrix(range(i))
        assert gl(A.transpose()) == gl(A)
        assert gl(A,mat_transpose=True) == gl(A)
        if i > 1:
            if i <= 4:
                assert gl(A) == 'vec%s(%s)' % (i,', '.join(str(s) for s in range(i)))
            else:
                assert gl(A) == 'float[%s](%s)' % (i,', '.join(str(s) for s in range(i)))

def test_MxN_mats():
    generatedAssertions='def test_misc_mats():\n'
    for i in range(1,6):
        for j in range(1,6):
            A = Matrix([[x + y*j for x in range(j)] for y in range(i)])
            gl = glsl_code(A)
            glTransposed = glsl_code(A,mat_transpose=True)
            generatedAssertions+='    mat = '+StrPrinter()._print(A)+'\n\n'
            generatedAssertions+='    gl = \'\'\''+gl+'\'\'\'\n'
            generatedAssertions+='    glTransposed = \'\'\''+glTransposed+'\'\'\'\n\n'
            generatedAssertions+='    assert glsl_code(mat) == gl\n'
            generatedAssertions+='    assert glsl_code(mat,mat_transpose=True) == glTransposed\n'
            if i == 1 and j == 1:
                assert gl == '0'
            elif i <= 4 and j <= 4 and i>1 and j>1:
                assert gl.startswith('mat%s' % j)
                assert glTransposed.startswith('mat%s' % i)
            elif i == 1 and j <= 4:
                assert gl.startswith('vec')
            elif j == 1 and i <= 4:
                assert gl.startswith('vec')
            elif i == 1:
                assert gl.startswith('float[%s]('% j*i)
                assert glTransposed.startswith('float[%s]('% j*i)
            elif j == 1:
                assert gl.startswith('float[%s]('% i*j)
                assert glTransposed.startswith('float[%s]('% i*j)
            else:
                assert gl.startswith('float[%s](' % (i*j))
                assert glTransposed.startswith('float[%s](' % (i*j))
                glNested = glsl_code(A,mat_nested=True)
                glNestedTransposed = glsl_code(A,mat_transpose=True,mat_nested=True)
                assert glNested.startswith('float[%s][%s]' % (i,j))
                assert glNestedTransposed.startswith('float[%s][%s]' % (j,i))
                generatedAssertions+='    glNested = \'\'\''+glNested+'\'\'\'\n'
                generatedAssertions+='    glNestedTransposed = \'\'\''+glNestedTransposed+'\'\'\'\n\n'
                generatedAssertions+='    assert glsl_code(mat,mat_nested=True) == glNested\n'
                generatedAssertions+='    assert glsl_code(mat,mat_nested=True,mat_transpose=True) == glNestedTransposed\n\n'
    generateAssertions = False # set this to true to write bake these generated tests to a file
    if generateAssertions:
        gen = open('test_glsl_generated_matrices.py','w')
        gen.write(generatedAssertions)
        gen.close()


# these assertions were generated from the previous function
# glsl has complicated rules and this makes it easier to look over all the cases
def test_misc_mats():

    mat = Matrix([[0]])

    gl = '''0'''
    glTransposed = '''0'''

    assert glsl_code(mat) == gl
    assert glsl_code(mat,mat_transpose=True) == glTransposed

    mat = Matrix([[0, 1]])

    gl = '''vec2(0, 1)'''
    glTransposed = '''vec2(0, 1)'''

    assert glsl_code(mat) == gl
    assert glsl_code(mat,mat_transpose=True) == glTransposed

    mat = Matrix([[0, 1, 2]])

    gl = '''vec3(0, 1, 2)'''
    glTransposed = '''vec3(0, 1, 2)'''

    assert glsl_code(mat) == gl
    assert glsl_code(mat,mat_transpose=True) == glTransposed

    mat = Matrix([[0, 1, 2, 3]])

    gl = '''vec4(0, 1, 2, 3)'''
    glTransposed = '''vec4(0, 1, 2, 3)'''

    assert glsl_code(mat) == gl
    assert glsl_code(mat,mat_transpose=True) == glTransposed

    mat = Matrix([[0, 1, 2, 3, 4]])

    gl = '''float[5](0, 1, 2, 3, 4)'''
    glTransposed = '''float[5](0, 1, 2, 3, 4)'''

    assert glsl_code(mat) == gl
    assert glsl_code(mat,mat_transpose=True) == glTransposed

    mat = Matrix([
[0],
[1]])

    gl = '''vec2(0, 1)'''
    glTransposed = '''vec2(0, 1)'''

    assert glsl_code(mat) == gl
    assert glsl_code(mat,mat_transpose=True) == glTransposed

    mat = Matrix([
[0, 1],
[2, 3]])

    gl = '''mat2(0, 1, 2, 3)'''
    glTransposed = '''mat2(0, 2, 1, 3)'''

    assert glsl_code(mat) == gl
    assert glsl_code(mat,mat_transpose=True) == glTransposed

    mat = Matrix([
[0, 1, 2],
[3, 4, 5]])

    gl = '''mat3x2(0, 1, 2, 3, 4, 5)'''
    glTransposed = '''mat2x3(0, 3, 1, 4, 2, 5)'''

    assert glsl_code(mat) == gl
    assert glsl_code(mat,mat_transpose=True) == glTransposed

    mat = Matrix([
[0, 1, 2, 3],
[4, 5, 6, 7]])

    gl = '''mat4x2(0, 1, 2, 3, 4, 5, 6, 7)'''
    glTransposed = '''mat2x4(0, 4, 1, 5, 2, 6, 3, 7)'''

    assert glsl_code(mat) == gl
    assert glsl_code(mat,mat_transpose=True) == glTransposed

    mat = Matrix([
[0, 1, 2, 3, 4],
[5, 6, 7, 8, 9]])

    gl = '''float[10](
   0, 1, 2, 3, 4,
   5, 6, 7, 8, 9
) /* a 2x5 matrix */'''
    glTransposed = '''float[10](
   0, 5,
   1, 6,
   2, 7,
   3, 8,
   4, 9
) /* a 5x2 matrix */'''

    assert glsl_code(mat) == gl
    assert glsl_code(mat,mat_transpose=True) == glTransposed
    glNested = '''float[2][5](
   float[](0, 1, 2, 3, 4),
   float[](5, 6, 7, 8, 9)
)'''
    glNestedTransposed = '''float[5][2](
   float[](0, 5),
   float[](1, 6),
   float[](2, 7),
   float[](3, 8),
   float[](4, 9)
)'''

    assert glsl_code(mat,mat_nested=True) == glNested
    assert glsl_code(mat,mat_nested=True,mat_transpose=True) == glNestedTransposed

    mat = Matrix([
[0],
[1],
[2]])

    gl = '''vec3(0, 1, 2)'''
    glTransposed = '''vec3(0, 1, 2)'''

    assert glsl_code(mat) == gl
    assert glsl_code(mat,mat_transpose=True) == glTransposed

    mat = Matrix([
[0, 1],
[2, 3],
[4, 5]])

    gl = '''mat2x3(0, 1, 2, 3, 4, 5)'''
    glTransposed = '''mat3x2(0, 2, 4, 1, 3, 5)'''

    assert glsl_code(mat) == gl
    assert glsl_code(mat,mat_transpose=True) == glTransposed

    mat = Matrix([
[0, 1, 2],
[3, 4, 5],
[6, 7, 8]])

    gl = '''mat3(0, 1, 2, 3, 4, 5, 6, 7, 8)'''
    glTransposed = '''mat3(0, 3, 6, 1, 4, 7, 2, 5, 8)'''

    assert glsl_code(mat) == gl
    assert glsl_code(mat,mat_transpose=True) == glTransposed

    mat = Matrix([
[0, 1,  2,  3],
[4, 5,  6,  7],
[8, 9, 10, 11]])

    gl = '''mat4x3(0, 1,  2,  3, 4, 5,  6,  7, 8, 9, 10, 11)'''
    glTransposed = '''mat3x4(0, 4,  8, 1, 5,  9, 2, 6, 10, 3, 7, 11)'''

    assert glsl_code(mat) == gl
    assert glsl_code(mat,mat_transpose=True) == glTransposed

    mat = Matrix([
[ 0,  1,  2,  3,  4],
[ 5,  6,  7,  8,  9],
[10, 11, 12, 13, 14]])

    gl = '''float[15](
   0,  1,  2,  3,  4,
   5,  6,  7,  8,  9,
   10, 11, 12, 13, 14
) /* a 3x5 matrix */'''
    glTransposed = '''float[15](
   0, 5, 10,
   1, 6, 11,
   2, 7, 12,
   3, 8, 13,
   4, 9, 14
) /* a 5x3 matrix */'''

    assert glsl_code(mat) == gl
    assert glsl_code(mat,mat_transpose=True) == glTransposed
    glNested = '''float[3][5](
   float[]( 0,  1,  2,  3,  4),
   float[]( 5,  6,  7,  8,  9),
   float[](10, 11, 12, 13, 14)
)'''
    glNestedTransposed = '''float[5][3](
   float[](0, 5, 10),
   float[](1, 6, 11),
   float[](2, 7, 12),
   float[](3, 8, 13),
   float[](4, 9, 14)
)'''

    assert glsl_code(mat,mat_nested=True) == glNested
    assert glsl_code(mat,mat_nested=True,mat_transpose=True) == glNestedTransposed

    mat = Matrix([
[0],
[1],
[2],
[3]])

    gl = '''vec4(0, 1, 2, 3)'''
    glTransposed = '''vec4(0, 1, 2, 3)'''

    assert glsl_code(mat) == gl
    assert glsl_code(mat,mat_transpose=True) == glTransposed

    mat = Matrix([
[0, 1],
[2, 3],
[4, 5],
[6, 7]])

    gl = '''mat2x4(0, 1, 2, 3, 4, 5, 6, 7)'''
    glTransposed = '''mat4x2(0, 2, 4, 6, 1, 3, 5, 7)'''

    assert glsl_code(mat) == gl
    assert glsl_code(mat,mat_transpose=True) == glTransposed

    mat = Matrix([
[0,  1,  2],
[3,  4,  5],
[6,  7,  8],
[9, 10, 11]])

    gl = '''mat3x4(0,  1,  2, 3,  4,  5, 6,  7,  8, 9, 10, 11)'''
    glTransposed = '''mat4x3(0, 3, 6,  9, 1, 4, 7, 10, 2, 5, 8, 11)'''

    assert glsl_code(mat) == gl
    assert glsl_code(mat,mat_transpose=True) == glTransposed

    mat = Matrix([
[ 0,  1,  2,  3],
[ 4,  5,  6,  7],
[ 8,  9, 10, 11],
[12, 13, 14, 15]])

    gl = '''mat4( 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15)'''
    glTransposed = '''mat4(0, 4,  8, 12, 1, 5,  9, 13, 2, 6, 10, 14, 3, 7, 11, 15)'''

    assert glsl_code(mat) == gl
    assert glsl_code(mat,mat_transpose=True) == glTransposed

    mat = Matrix([
[ 0,  1,  2,  3,  4],
[ 5,  6,  7,  8,  9],
[10, 11, 12, 13, 14],
[15, 16, 17, 18, 19]])

    gl = '''float[20](
   0,  1,  2,  3,  4,
   5,  6,  7,  8,  9,
   10, 11, 12, 13, 14,
   15, 16, 17, 18, 19
) /* a 4x5 matrix */'''
    glTransposed = '''float[20](
   0, 5, 10, 15,
   1, 6, 11, 16,
   2, 7, 12, 17,
   3, 8, 13, 18,
   4, 9, 14, 19
) /* a 5x4 matrix */'''

    assert glsl_code(mat) == gl
    assert glsl_code(mat,mat_transpose=True) == glTransposed
    glNested = '''float[4][5](
   float[]( 0,  1,  2,  3,  4),
   float[]( 5,  6,  7,  8,  9),
   float[](10, 11, 12, 13, 14),
   float[](15, 16, 17, 18, 19)
)'''
    glNestedTransposed = '''float[5][4](
   float[](0, 5, 10, 15),
   float[](1, 6, 11, 16),
   float[](2, 7, 12, 17),
   float[](3, 8, 13, 18),
   float[](4, 9, 14, 19)
)'''

    assert glsl_code(mat,mat_nested=True) == glNested
    assert glsl_code(mat,mat_nested=True,mat_transpose=True) == glNestedTransposed

    mat = Matrix([
[0],
[1],
[2],
[3],
[4]])

    gl = '''float[5](0, 1, 2, 3, 4)'''
    glTransposed = '''float[5](0, 1, 2, 3, 4)'''

    assert glsl_code(mat) == gl
    assert glsl_code(mat,mat_transpose=True) == glTransposed

    mat = Matrix([
[0, 1],
[2, 3],
[4, 5],
[6, 7],
[8, 9]])

    gl = '''float[10](
   0, 1,
   2, 3,
   4, 5,
   6, 7,
   8, 9
) /* a 5x2 matrix */'''
    glTransposed = '''float[10](
   0, 2, 4, 6, 8,
   1, 3, 5, 7, 9
) /* a 2x5 matrix */'''

    assert glsl_code(mat) == gl
    assert glsl_code(mat,mat_transpose=True) == glTransposed
    glNested = '''float[5][2](
   float[](0, 1),
   float[](2, 3),
   float[](4, 5),
   float[](6, 7),
   float[](8, 9)
)'''
    glNestedTransposed = '''float[2][5](
   float[](0, 2, 4, 6, 8),
   float[](1, 3, 5, 7, 9)
)'''

    assert glsl_code(mat,mat_nested=True) == glNested
    assert glsl_code(mat,mat_nested=True,mat_transpose=True) == glNestedTransposed

    mat = Matrix([
[ 0,  1,  2],
[ 3,  4,  5],
[ 6,  7,  8],
[ 9, 10, 11],
[12, 13, 14]])

    gl = '''float[15](
   0,  1,  2,
   3,  4,  5,
   6,  7,  8,
   9, 10, 11,
   12, 13, 14
) /* a 5x3 matrix */'''
    glTransposed = '''float[15](
   0, 3, 6,  9, 12,
   1, 4, 7, 10, 13,
   2, 5, 8, 11, 14
) /* a 3x5 matrix */'''

    assert glsl_code(mat) == gl
    assert glsl_code(mat,mat_transpose=True) == glTransposed
    glNested = '''float[5][3](
   float[]( 0,  1,  2),
   float[]( 3,  4,  5),
   float[]( 6,  7,  8),
   float[]( 9, 10, 11),
   float[](12, 13, 14)
)'''
    glNestedTransposed = '''float[3][5](
   float[](0, 3, 6,  9, 12),
   float[](1, 4, 7, 10, 13),
   float[](2, 5, 8, 11, 14)
)'''

    assert glsl_code(mat,mat_nested=True) == glNested
    assert glsl_code(mat,mat_nested=True,mat_transpose=True) == glNestedTransposed

    mat = Matrix([
[ 0,  1,  2,  3],
[ 4,  5,  6,  7],
[ 8,  9, 10, 11],
[12, 13, 14, 15],
[16, 17, 18, 19]])

    gl = '''float[20](
   0,  1,  2,  3,
   4,  5,  6,  7,
   8,  9, 10, 11,
   12, 13, 14, 15,
   16, 17, 18, 19
) /* a 5x4 matrix */'''
    glTransposed = '''float[20](
   0, 4,  8, 12, 16,
   1, 5,  9, 13, 17,
   2, 6, 10, 14, 18,
   3, 7, 11, 15, 19
) /* a 4x5 matrix */'''

    assert glsl_code(mat) == gl
    assert glsl_code(mat,mat_transpose=True) == glTransposed
    glNested = '''float[5][4](
   float[]( 0,  1,  2,  3),
   float[]( 4,  5,  6,  7),
   float[]( 8,  9, 10, 11),
   float[](12, 13, 14, 15),
   float[](16, 17, 18, 19)
)'''
    glNestedTransposed = '''float[4][5](
   float[](0, 4,  8, 12, 16),
   float[](1, 5,  9, 13, 17),
   float[](2, 6, 10, 14, 18),
   float[](3, 7, 11, 15, 19)
)'''

    assert glsl_code(mat,mat_nested=True) == glNested
    assert glsl_code(mat,mat_nested=True,mat_transpose=True) == glNestedTransposed

    mat = Matrix([
[ 0,  1,  2,  3,  4],
[ 5,  6,  7,  8,  9],
[10, 11, 12, 13, 14],
[15, 16, 17, 18, 19],
[20, 21, 22, 23, 24]])

    gl = '''float[25](
   0,  1,  2,  3,  4,
   5,  6,  7,  8,  9,
   10, 11, 12, 13, 14,
   15, 16, 17, 18, 19,
   20, 21, 22, 23, 24
) /* a 5x5 matrix */'''
    glTransposed = '''float[25](
   0, 5, 10, 15, 20,
   1, 6, 11, 16, 21,
   2, 7, 12, 17, 22,
   3, 8, 13, 18, 23,
   4, 9, 14, 19, 24
) /* a 5x5 matrix */'''

    assert glsl_code(mat) == gl
    assert glsl_code(mat,mat_transpose=True) == glTransposed
    glNested = '''float[5][5](
   float[]( 0,  1,  2,  3,  4),
   float[]( 5,  6,  7,  8,  9),
   float[](10, 11, 12, 13, 14),
   float[](15, 16, 17, 18, 19),
   float[](20, 21, 22, 23, 24)
)'''
    glNestedTransposed = '''float[5][5](
   float[](0, 5, 10, 15, 20),
   float[](1, 6, 11, 16, 21),
   float[](2, 7, 12, 17, 22),
   float[](3, 8, 13, 18, 23),
   float[](4, 9, 14, 19, 24)
)'''

    assert glsl_code(mat,mat_nested=True) == glNested
    assert glsl_code(mat,mat_nested=True,mat_transpose=True) == glNestedTransposed
