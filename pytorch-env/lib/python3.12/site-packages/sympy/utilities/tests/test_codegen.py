from io import StringIO

from sympy.core import symbols, Eq, pi, Catalan, Lambda, Dummy
from sympy.core.relational import Equality
from sympy.core.symbol import Symbol
from sympy.functions.special.error_functions import erf
from sympy.integrals.integrals import Integral
from sympy.matrices import Matrix, MatrixSymbol
from sympy.utilities.codegen import (
    codegen, make_routine, CCodeGen, C89CodeGen, C99CodeGen, InputArgument,
    CodeGenError, FCodeGen, CodeGenArgumentListError, OutputArgument,
    InOutArgument)
from sympy.testing.pytest import raises
from sympy.utilities.lambdify import implemented_function

#FIXME: Fails due to circular import in with core
# from sympy import codegen


def get_string(dump_fn, routines, prefix="file", header=False, empty=False):
    """Wrapper for dump_fn. dump_fn writes its results to a stream object and
       this wrapper returns the contents of that stream as a string. This
       auxiliary function is used by many tests below.

       The header and the empty lines are not generated to facilitate the
       testing of the output.
    """
    output = StringIO()
    dump_fn(routines, output, prefix, header, empty)
    source = output.getvalue()
    output.close()
    return source


def test_Routine_argument_order():
    a, x, y, z = symbols('a x y z')
    expr = (x + y)*z
    raises(CodeGenArgumentListError, lambda: make_routine("test", expr,
           argument_sequence=[z, x]))
    raises(CodeGenArgumentListError, lambda: make_routine("test", Eq(a,
           expr), argument_sequence=[z, x, y]))
    r = make_routine('test', Eq(a, expr), argument_sequence=[z, x, a, y])
    assert [ arg.name for arg in r.arguments ] == [z, x, a, y]
    assert [ type(arg) for arg in r.arguments ] == [
        InputArgument, InputArgument, OutputArgument, InputArgument  ]
    r = make_routine('test', Eq(z, expr), argument_sequence=[z, x, y])
    assert [ type(arg) for arg in r.arguments ] == [
        InOutArgument, InputArgument, InputArgument ]

    from sympy.tensor import IndexedBase, Idx
    A, B = map(IndexedBase, ['A', 'B'])
    m = symbols('m', integer=True)
    i = Idx('i', m)
    r = make_routine('test', Eq(A[i], B[i]), argument_sequence=[B, A, m])
    assert [ arg.name for arg in r.arguments ] == [B.label, A.label, m]

    expr = Integral(x*y*z, (x, 1, 2), (y, 1, 3))
    r = make_routine('test', Eq(a, expr), argument_sequence=[z, x, a, y])
    assert [ arg.name for arg in r.arguments ] == [z, x, a, y]


def test_empty_c_code():
    code_gen = C89CodeGen()
    source = get_string(code_gen.dump_c, [])
    assert source == "#include \"file.h\"\n#include <math.h>\n"


def test_empty_c_code_with_comment():
    code_gen = C89CodeGen()
    source = get_string(code_gen.dump_c, [], header=True)
    assert source[:82] == (
        "/******************************************************************************\n *"
    )
          #   "                    Code generated with SymPy 0.7.2-git                    "
    assert source[158:] == (                                                              "*\n"
            " *                                                                            *\n"
            " *              See http://www.sympy.org/ for more information.               *\n"
            " *                                                                            *\n"
            " *                       This file is part of 'project'                       *\n"
            " ******************************************************************************/\n"
            "#include \"file.h\"\n"
            "#include <math.h>\n"
            )


def test_empty_c_header():
    code_gen = C99CodeGen()
    source = get_string(code_gen.dump_h, [])
    assert source == "#ifndef PROJECT__FILE__H\n#define PROJECT__FILE__H\n#endif\n"


def test_simple_c_code():
    x, y, z = symbols('x,y,z')
    expr = (x + y)*z
    routine = make_routine("test", expr)
    code_gen = C89CodeGen()
    source = get_string(code_gen.dump_c, [routine])
    expected = (
        "#include \"file.h\"\n"
        "#include <math.h>\n"
        "double test(double x, double y, double z) {\n"
        "   double test_result;\n"
        "   test_result = z*(x + y);\n"
        "   return test_result;\n"
        "}\n"
    )
    assert source == expected


def test_c_code_reserved_words():
    x, y, z = symbols('if, typedef, while')
    expr = (x + y) * z
    routine = make_routine("test", expr)
    code_gen = C99CodeGen()
    source = get_string(code_gen.dump_c, [routine])
    expected = (
        "#include \"file.h\"\n"
        "#include <math.h>\n"
        "double test(double if_, double typedef_, double while_) {\n"
        "   double test_result;\n"
        "   test_result = while_*(if_ + typedef_);\n"
        "   return test_result;\n"
        "}\n"
    )
    assert source == expected


def test_numbersymbol_c_code():
    routine = make_routine("test", pi**Catalan)
    code_gen = C89CodeGen()
    source = get_string(code_gen.dump_c, [routine])
    expected = (
        "#include \"file.h\"\n"
        "#include <math.h>\n"
        "double test() {\n"
        "   double test_result;\n"
        "   double const Catalan = %s;\n"
        "   test_result = pow(M_PI, Catalan);\n"
        "   return test_result;\n"
        "}\n"
    ) % Catalan.evalf(17)
    assert source == expected


def test_c_code_argument_order():
    x, y, z = symbols('x,y,z')
    expr = x + y
    routine = make_routine("test", expr, argument_sequence=[z, x, y])
    code_gen = C89CodeGen()
    source = get_string(code_gen.dump_c, [routine])
    expected = (
        "#include \"file.h\"\n"
        "#include <math.h>\n"
        "double test(double z, double x, double y) {\n"
        "   double test_result;\n"
        "   test_result = x + y;\n"
        "   return test_result;\n"
        "}\n"
    )
    assert source == expected


def test_simple_c_header():
    x, y, z = symbols('x,y,z')
    expr = (x + y)*z
    routine = make_routine("test", expr)
    code_gen = C89CodeGen()
    source = get_string(code_gen.dump_h, [routine])
    expected = (
        "#ifndef PROJECT__FILE__H\n"
        "#define PROJECT__FILE__H\n"
        "double test(double x, double y, double z);\n"
        "#endif\n"
    )
    assert source == expected


def test_simple_c_codegen():
    x, y, z = symbols('x,y,z')
    expr = (x + y)*z
    expected = [
        ("file.c",
        "#include \"file.h\"\n"
        "#include <math.h>\n"
        "double test(double x, double y, double z) {\n"
        "   double test_result;\n"
        "   test_result = z*(x + y);\n"
        "   return test_result;\n"
        "}\n"),
        ("file.h",
        "#ifndef PROJECT__FILE__H\n"
        "#define PROJECT__FILE__H\n"
        "double test(double x, double y, double z);\n"
        "#endif\n")
    ]
    result = codegen(("test", expr), "C", "file", header=False, empty=False)
    assert result == expected


def test_multiple_results_c():
    x, y, z = symbols('x,y,z')
    expr1 = (x + y)*z
    expr2 = (x - y)*z
    routine = make_routine(
        "test",
        [expr1, expr2]
    )
    code_gen = C99CodeGen()
    raises(CodeGenError, lambda: get_string(code_gen.dump_h, [routine]))


def test_no_results_c():
    raises(ValueError, lambda: make_routine("test", []))


def test_ansi_math1_codegen():
    # not included: log10
    from sympy.functions.elementary.complexes import Abs
    from sympy.functions.elementary.exponential import log
    from sympy.functions.elementary.hyperbolic import (cosh, sinh, tanh)
    from sympy.functions.elementary.integers import (ceiling, floor)
    from sympy.functions.elementary.miscellaneous import sqrt
    from sympy.functions.elementary.trigonometric import (acos, asin, atan, cos, sin, tan)
    x = symbols('x')
    name_expr = [
        ("test_fabs", Abs(x)),
        ("test_acos", acos(x)),
        ("test_asin", asin(x)),
        ("test_atan", atan(x)),
        ("test_ceil", ceiling(x)),
        ("test_cos", cos(x)),
        ("test_cosh", cosh(x)),
        ("test_floor", floor(x)),
        ("test_log", log(x)),
        ("test_ln", log(x)),
        ("test_sin", sin(x)),
        ("test_sinh", sinh(x)),
        ("test_sqrt", sqrt(x)),
        ("test_tan", tan(x)),
        ("test_tanh", tanh(x)),
    ]
    result = codegen(name_expr, "C89", "file", header=False, empty=False)
    assert result[0][0] == "file.c"
    assert result[0][1] == (
        '#include "file.h"\n#include <math.h>\n'
        'double test_fabs(double x) {\n   double test_fabs_result;\n   test_fabs_result = fabs(x);\n   return test_fabs_result;\n}\n'
        'double test_acos(double x) {\n   double test_acos_result;\n   test_acos_result = acos(x);\n   return test_acos_result;\n}\n'
        'double test_asin(double x) {\n   double test_asin_result;\n   test_asin_result = asin(x);\n   return test_asin_result;\n}\n'
        'double test_atan(double x) {\n   double test_atan_result;\n   test_atan_result = atan(x);\n   return test_atan_result;\n}\n'
        'double test_ceil(double x) {\n   double test_ceil_result;\n   test_ceil_result = ceil(x);\n   return test_ceil_result;\n}\n'
        'double test_cos(double x) {\n   double test_cos_result;\n   test_cos_result = cos(x);\n   return test_cos_result;\n}\n'
        'double test_cosh(double x) {\n   double test_cosh_result;\n   test_cosh_result = cosh(x);\n   return test_cosh_result;\n}\n'
        'double test_floor(double x) {\n   double test_floor_result;\n   test_floor_result = floor(x);\n   return test_floor_result;\n}\n'
        'double test_log(double x) {\n   double test_log_result;\n   test_log_result = log(x);\n   return test_log_result;\n}\n'
        'double test_ln(double x) {\n   double test_ln_result;\n   test_ln_result = log(x);\n   return test_ln_result;\n}\n'
        'double test_sin(double x) {\n   double test_sin_result;\n   test_sin_result = sin(x);\n   return test_sin_result;\n}\n'
        'double test_sinh(double x) {\n   double test_sinh_result;\n   test_sinh_result = sinh(x);\n   return test_sinh_result;\n}\n'
        'double test_sqrt(double x) {\n   double test_sqrt_result;\n   test_sqrt_result = sqrt(x);\n   return test_sqrt_result;\n}\n'
        'double test_tan(double x) {\n   double test_tan_result;\n   test_tan_result = tan(x);\n   return test_tan_result;\n}\n'
        'double test_tanh(double x) {\n   double test_tanh_result;\n   test_tanh_result = tanh(x);\n   return test_tanh_result;\n}\n'
    )
    assert result[1][0] == "file.h"
    assert result[1][1] == (
        '#ifndef PROJECT__FILE__H\n#define PROJECT__FILE__H\n'
        'double test_fabs(double x);\ndouble test_acos(double x);\n'
        'double test_asin(double x);\ndouble test_atan(double x);\n'
        'double test_ceil(double x);\ndouble test_cos(double x);\n'
        'double test_cosh(double x);\ndouble test_floor(double x);\n'
        'double test_log(double x);\ndouble test_ln(double x);\n'
        'double test_sin(double x);\ndouble test_sinh(double x);\n'
        'double test_sqrt(double x);\ndouble test_tan(double x);\n'
        'double test_tanh(double x);\n#endif\n'
    )


def test_ansi_math2_codegen():
    # not included: frexp, ldexp, modf, fmod
    from sympy.functions.elementary.trigonometric import atan2
    x, y = symbols('x,y')
    name_expr = [
        ("test_atan2", atan2(x, y)),
        ("test_pow", x**y),
    ]
    result = codegen(name_expr, "C89", "file", header=False, empty=False)
    assert result[0][0] == "file.c"
    assert result[0][1] == (
        '#include "file.h"\n#include <math.h>\n'
        'double test_atan2(double x, double y) {\n   double test_atan2_result;\n   test_atan2_result = atan2(x, y);\n   return test_atan2_result;\n}\n'
        'double test_pow(double x, double y) {\n   double test_pow_result;\n   test_pow_result = pow(x, y);\n   return test_pow_result;\n}\n'
    )
    assert result[1][0] == "file.h"
    assert result[1][1] == (
        '#ifndef PROJECT__FILE__H\n#define PROJECT__FILE__H\n'
        'double test_atan2(double x, double y);\n'
        'double test_pow(double x, double y);\n'
        '#endif\n'
    )


def test_complicated_codegen():
    from sympy.functions.elementary.trigonometric import (cos, sin, tan)
    x, y, z = symbols('x,y,z')
    name_expr = [
        ("test1", ((sin(x) + cos(y) + tan(z))**7).expand()),
        ("test2", cos(cos(cos(cos(cos(cos(cos(cos(x + y + z))))))))),
    ]
    result = codegen(name_expr, "C89", "file", header=False, empty=False)
    assert result[0][0] == "file.c"
    assert result[0][1] == (
        '#include "file.h"\n#include <math.h>\n'
        'double test1(double x, double y, double z) {\n'
        '   double test1_result;\n'
        '   test1_result = '
        'pow(sin(x), 7) + '
        '7*pow(sin(x), 6)*cos(y) + '
        '7*pow(sin(x), 6)*tan(z) + '
        '21*pow(sin(x), 5)*pow(cos(y), 2) + '
        '42*pow(sin(x), 5)*cos(y)*tan(z) + '
        '21*pow(sin(x), 5)*pow(tan(z), 2) + '
        '35*pow(sin(x), 4)*pow(cos(y), 3) + '
        '105*pow(sin(x), 4)*pow(cos(y), 2)*tan(z) + '
        '105*pow(sin(x), 4)*cos(y)*pow(tan(z), 2) + '
        '35*pow(sin(x), 4)*pow(tan(z), 3) + '
        '35*pow(sin(x), 3)*pow(cos(y), 4) + '
        '140*pow(sin(x), 3)*pow(cos(y), 3)*tan(z) + '
        '210*pow(sin(x), 3)*pow(cos(y), 2)*pow(tan(z), 2) + '
        '140*pow(sin(x), 3)*cos(y)*pow(tan(z), 3) + '
        '35*pow(sin(x), 3)*pow(tan(z), 4) + '
        '21*pow(sin(x), 2)*pow(cos(y), 5) + '
        '105*pow(sin(x), 2)*pow(cos(y), 4)*tan(z) + '
        '210*pow(sin(x), 2)*pow(cos(y), 3)*pow(tan(z), 2) + '
        '210*pow(sin(x), 2)*pow(cos(y), 2)*pow(tan(z), 3) + '
        '105*pow(sin(x), 2)*cos(y)*pow(tan(z), 4) + '
        '21*pow(sin(x), 2)*pow(tan(z), 5) + '
        '7*sin(x)*pow(cos(y), 6) + '
        '42*sin(x)*pow(cos(y), 5)*tan(z) + '
        '105*sin(x)*pow(cos(y), 4)*pow(tan(z), 2) + '
        '140*sin(x)*pow(cos(y), 3)*pow(tan(z), 3) + '
        '105*sin(x)*pow(cos(y), 2)*pow(tan(z), 4) + '
        '42*sin(x)*cos(y)*pow(tan(z), 5) + '
        '7*sin(x)*pow(tan(z), 6) + '
        'pow(cos(y), 7) + '
        '7*pow(cos(y), 6)*tan(z) + '
        '21*pow(cos(y), 5)*pow(tan(z), 2) + '
        '35*pow(cos(y), 4)*pow(tan(z), 3) + '
        '35*pow(cos(y), 3)*pow(tan(z), 4) + '
        '21*pow(cos(y), 2)*pow(tan(z), 5) + '
        '7*cos(y)*pow(tan(z), 6) + '
        'pow(tan(z), 7);\n'
        '   return test1_result;\n'
        '}\n'
        'double test2(double x, double y, double z) {\n'
        '   double test2_result;\n'
        '   test2_result = cos(cos(cos(cos(cos(cos(cos(cos(x + y + z))))))));\n'
        '   return test2_result;\n'
        '}\n'
    )
    assert result[1][0] == "file.h"
    assert result[1][1] == (
        '#ifndef PROJECT__FILE__H\n'
        '#define PROJECT__FILE__H\n'
        'double test1(double x, double y, double z);\n'
        'double test2(double x, double y, double z);\n'
        '#endif\n'
    )


def test_loops_c():
    from sympy.tensor import IndexedBase, Idx
    from sympy.core.symbol import symbols
    n, m = symbols('n m', integer=True)
    A = IndexedBase('A')
    x = IndexedBase('x')
    y = IndexedBase('y')
    i = Idx('i', m)
    j = Idx('j', n)

    (f1, code), (f2, interface) = codegen(
        ('matrix_vector', Eq(y[i], A[i, j]*x[j])), "C99", "file", header=False, empty=False)

    assert f1 == 'file.c'
    expected = (
        '#include "file.h"\n'
        '#include <math.h>\n'
        'void matrix_vector(double *A, int m, int n, double *x, double *y) {\n'
        '   for (int i=0; i<m; i++){\n'
        '      y[i] = 0;\n'
        '   }\n'
        '   for (int i=0; i<m; i++){\n'
        '      for (int j=0; j<n; j++){\n'
        '         y[i] = %(rhs)s + y[i];\n'
        '      }\n'
        '   }\n'
        '}\n'
    )

    assert (code == expected % {'rhs': 'A[%s]*x[j]' % (i*n + j)} or
            code == expected % {'rhs': 'A[%s]*x[j]' % (j + i*n)} or
            code == expected % {'rhs': 'x[j]*A[%s]' % (i*n + j)} or
            code == expected % {'rhs': 'x[j]*A[%s]' % (j + i*n)})
    assert f2 == 'file.h'
    assert interface == (
        '#ifndef PROJECT__FILE__H\n'
        '#define PROJECT__FILE__H\n'
        'void matrix_vector(double *A, int m, int n, double *x, double *y);\n'
        '#endif\n'
    )


def test_dummy_loops_c():
    from sympy.tensor import IndexedBase, Idx
    i, m = symbols('i m', integer=True, cls=Dummy)
    x = IndexedBase('x')
    y = IndexedBase('y')
    i = Idx(i, m)
    expected = (
        '#include "file.h"\n'
        '#include <math.h>\n'
        'void test_dummies(int m_%(mno)i, double *x, double *y) {\n'
        '   for (int i_%(ino)i=0; i_%(ino)i<m_%(mno)i; i_%(ino)i++){\n'
        '      y[i_%(ino)i] = x[i_%(ino)i];\n'
        '   }\n'
        '}\n'
    ) % {'ino': i.label.dummy_index, 'mno': m.dummy_index}
    r = make_routine('test_dummies', Eq(y[i], x[i]))
    c89 = C89CodeGen()
    c99 = C99CodeGen()
    code = get_string(c99.dump_c, [r])
    assert code == expected
    with raises(NotImplementedError):
        get_string(c89.dump_c, [r])

def test_partial_loops_c():
    # check that loop boundaries are determined by Idx, and array strides
    # determined by shape of IndexedBase object.
    from sympy.tensor import IndexedBase, Idx
    from sympy.core.symbol import symbols
    n, m, o, p = symbols('n m o p', integer=True)
    A = IndexedBase('A', shape=(m, p))
    x = IndexedBase('x')
    y = IndexedBase('y')
    i = Idx('i', (o, m - 5))  # Note: bounds are inclusive
    j = Idx('j', n)          # dimension n corresponds to bounds (0, n - 1)

    (f1, code), (f2, interface) = codegen(
        ('matrix_vector', Eq(y[i], A[i, j]*x[j])), "C99", "file", header=False, empty=False)

    assert f1 == 'file.c'
    expected = (
        '#include "file.h"\n'
        '#include <math.h>\n'
        'void matrix_vector(double *A, int m, int n, int o, int p, double *x, double *y) {\n'
        '   for (int i=o; i<%(upperi)s; i++){\n'
        '      y[i] = 0;\n'
        '   }\n'
        '   for (int i=o; i<%(upperi)s; i++){\n'
        '      for (int j=0; j<n; j++){\n'
        '         y[i] = %(rhs)s + y[i];\n'
        '      }\n'
        '   }\n'
        '}\n'
    ) % {'upperi': m - 4, 'rhs': '%(rhs)s'}

    assert (code == expected % {'rhs': 'A[%s]*x[j]' % (i*p + j)} or
            code == expected % {'rhs': 'A[%s]*x[j]' % (j + i*p)} or
            code == expected % {'rhs': 'x[j]*A[%s]' % (i*p + j)} or
            code == expected % {'rhs': 'x[j]*A[%s]' % (j + i*p)})
    assert f2 == 'file.h'
    assert interface == (
        '#ifndef PROJECT__FILE__H\n'
        '#define PROJECT__FILE__H\n'
        'void matrix_vector(double *A, int m, int n, int o, int p, double *x, double *y);\n'
        '#endif\n'
    )


def test_output_arg_c():
    from sympy.core.relational import Equality
    from sympy.functions.elementary.trigonometric import (cos, sin)
    x, y, z = symbols("x,y,z")
    r = make_routine("foo", [Equality(y, sin(x)), cos(x)])
    c = C89CodeGen()
    result = c.write([r], "test", header=False, empty=False)
    assert result[0][0] == "test.c"
    expected = (
        '#include "test.h"\n'
        '#include <math.h>\n'
        'double foo(double x, double *y) {\n'
        '   (*y) = sin(x);\n'
        '   double foo_result;\n'
        '   foo_result = cos(x);\n'
        '   return foo_result;\n'
        '}\n'
    )
    assert result[0][1] == expected


def test_output_arg_c_reserved_words():
    from sympy.core.relational import Equality
    from sympy.functions.elementary.trigonometric import (cos, sin)
    x, y, z = symbols("if, while, z")
    r = make_routine("foo", [Equality(y, sin(x)), cos(x)])
    c = C89CodeGen()
    result = c.write([r], "test", header=False, empty=False)
    assert result[0][0] == "test.c"
    expected = (
        '#include "test.h"\n'
        '#include <math.h>\n'
        'double foo(double if_, double *while_) {\n'
        '   (*while_) = sin(if_);\n'
        '   double foo_result;\n'
        '   foo_result = cos(if_);\n'
        '   return foo_result;\n'
        '}\n'
    )
    assert result[0][1] == expected


def test_multidim_c_argument_cse():
    A_sym = MatrixSymbol('A', 3, 3)
    b_sym = MatrixSymbol('b', 3, 1)
    A = Matrix(A_sym)
    b = Matrix(b_sym)
    c = A*b
    cgen = CCodeGen(project="test", cse=True)
    r = cgen.routine("c", c)
    r.arguments[-1].result_var = "out"
    r.arguments[-1]._name = "out"
    code = get_string(cgen.dump_c, [r], prefix="test")
    expected = (
        '#include "test.h"\n'
        "#include <math.h>\n"
        "void c(double *A, double *b, double *out) {\n"
        "   out[0] = A[0]*b[0] + A[1]*b[1] + A[2]*b[2];\n"
        "   out[1] = A[3]*b[0] + A[4]*b[1] + A[5]*b[2];\n"
        "   out[2] = A[6]*b[0] + A[7]*b[1] + A[8]*b[2];\n"
        "}\n"
    )
    assert code == expected


def test_ccode_results_named_ordered():
    x, y, z = symbols('x,y,z')
    B, C = symbols('B,C')
    A = MatrixSymbol('A', 1, 3)
    expr1 = Equality(A, Matrix([[1, 2, x]]))
    expr2 = Equality(C, (x + y)*z)
    expr3 = Equality(B, 2*x)
    name_expr = ("test", [expr1, expr2, expr3])
    expected = (
        '#include "test.h"\n'
        '#include <math.h>\n'
        'void test(double x, double *C, double z, double y, double *A, double *B) {\n'
        '   (*C) = z*(x + y);\n'
        '   A[0] = 1;\n'
        '   A[1] = 2;\n'
        '   A[2] = x;\n'
        '   (*B) = 2*x;\n'
        '}\n'
    )

    result = codegen(name_expr, "c", "test", header=False, empty=False,
                     argument_sequence=(x, C, z, y, A, B))
    source = result[0][1]
    assert source == expected


def test_ccode_matrixsymbol_slice():
    A = MatrixSymbol('A', 5, 3)
    B = MatrixSymbol('B', 1, 3)
    C = MatrixSymbol('C', 1, 3)
    D = MatrixSymbol('D', 5, 1)
    name_expr = ("test", [Equality(B, A[0, :]),
                          Equality(C, A[1, :]),
                          Equality(D, A[:, 2])])
    result = codegen(name_expr, "c99", "test", header=False, empty=False)
    source = result[0][1]
    expected = (
        '#include "test.h"\n'
        '#include <math.h>\n'
        'void test(double *A, double *B, double *C, double *D) {\n'
        '   B[0] = A[0];\n'
        '   B[1] = A[1];\n'
        '   B[2] = A[2];\n'
        '   C[0] = A[3];\n'
        '   C[1] = A[4];\n'
        '   C[2] = A[5];\n'
        '   D[0] = A[2];\n'
        '   D[1] = A[5];\n'
        '   D[2] = A[8];\n'
        '   D[3] = A[11];\n'
        '   D[4] = A[14];\n'
        '}\n'
    )
    assert source == expected

def test_ccode_cse():
    a, b, c, d = symbols('a b c d')
    e = MatrixSymbol('e', 3, 1)
    name_expr = ("test", [Equality(e, Matrix([[a*b], [a*b + c*d], [a*b*c*d]]))])
    generator = CCodeGen(cse=True)
    result = codegen(name_expr, code_gen=generator, header=False, empty=False)
    source = result[0][1]
    expected = (
        '#include "test.h"\n'
        '#include <math.h>\n'
        'void test(double a, double b, double c, double d, double *e) {\n'
        '   const double x0 = a*b;\n'
        '   const double x1 = c*d;\n'
        '   e[0] = x0;\n'
        '   e[1] = x0 + x1;\n'
        '   e[2] = x0*x1;\n'
        '}\n'
    )
    assert source == expected

def test_ccode_unused_array_arg():
    x = MatrixSymbol('x', 2, 1)
    # x does not appear in output
    name_expr = ("test", 1.0)
    generator = CCodeGen()
    result = codegen(name_expr, code_gen=generator, header=False, empty=False, argument_sequence=(x,))
    source = result[0][1]
    # note: x should appear as (double *)
    expected = (
        '#include "test.h"\n'
        '#include <math.h>\n'
        'double test(double *x) {\n'
        '   double test_result;\n'
        '   test_result = 1.0;\n'
        '   return test_result;\n'
        '}\n'
    )
    assert source == expected

def test_ccode_unused_array_arg_func():
    # issue 16689
    X = MatrixSymbol('X',3,1)
    Y = MatrixSymbol('Y',3,1)
    z = symbols('z',integer = True)
    name_expr = ('testBug', X[0] + X[1])
    result = codegen(name_expr, language='C', header=False, empty=False, argument_sequence=(X, Y, z))
    source = result[0][1]
    expected = (
        '#include "testBug.h"\n'
        '#include <math.h>\n'
        'double testBug(double *X, double *Y, int z) {\n'
        '   double testBug_result;\n'
        '   testBug_result = X[0] + X[1];\n'
        '   return testBug_result;\n'
        '}\n'
    )
    assert source == expected

def test_empty_f_code():
    code_gen = FCodeGen()
    source = get_string(code_gen.dump_f95, [])
    assert source == ""


def test_empty_f_code_with_header():
    code_gen = FCodeGen()
    source = get_string(code_gen.dump_f95, [], header=True)
    assert source[:82] == (
        "!******************************************************************************\n!*"
    )
          #   "                    Code generated with SymPy 0.7.2-git                    "
    assert source[158:] == (                                                              "*\n"
            "!*                                                                            *\n"
            "!*              See http://www.sympy.org/ for more information.               *\n"
            "!*                                                                            *\n"
            "!*                       This file is part of 'project'                       *\n"
            "!******************************************************************************\n"
            )


def test_empty_f_header():
    code_gen = FCodeGen()
    source = get_string(code_gen.dump_h, [])
    assert source == ""


def test_simple_f_code():
    x, y, z = symbols('x,y,z')
    expr = (x + y)*z
    routine = make_routine("test", expr)
    code_gen = FCodeGen()
    source = get_string(code_gen.dump_f95, [routine])
    expected = (
        "REAL*8 function test(x, y, z)\n"
        "implicit none\n"
        "REAL*8, intent(in) :: x\n"
        "REAL*8, intent(in) :: y\n"
        "REAL*8, intent(in) :: z\n"
        "test = z*(x + y)\n"
        "end function\n"
    )
    assert source == expected


def test_numbersymbol_f_code():
    routine = make_routine("test", pi**Catalan)
    code_gen = FCodeGen()
    source = get_string(code_gen.dump_f95, [routine])
    expected = (
        "REAL*8 function test()\n"
        "implicit none\n"
        "REAL*8, parameter :: Catalan = %sd0\n"
        "REAL*8, parameter :: pi = %sd0\n"
        "test = pi**Catalan\n"
        "end function\n"
    ) % (Catalan.evalf(17), pi.evalf(17))
    assert source == expected

def test_erf_f_code():
    x = symbols('x')
    routine = make_routine("test", erf(x) - erf(-2 * x))
    code_gen = FCodeGen()
    source = get_string(code_gen.dump_f95, [routine])
    expected = (
        "REAL*8 function test(x)\n"
        "implicit none\n"
        "REAL*8, intent(in) :: x\n"
        "test = erf(x) + erf(2.0d0*x)\n"
        "end function\n"
    )
    assert source == expected, source

def test_f_code_argument_order():
    x, y, z = symbols('x,y,z')
    expr = x + y
    routine = make_routine("test", expr, argument_sequence=[z, x, y])
    code_gen = FCodeGen()
    source = get_string(code_gen.dump_f95, [routine])
    expected = (
        "REAL*8 function test(z, x, y)\n"
        "implicit none\n"
        "REAL*8, intent(in) :: z\n"
        "REAL*8, intent(in) :: x\n"
        "REAL*8, intent(in) :: y\n"
        "test = x + y\n"
        "end function\n"
    )
    assert source == expected


def test_simple_f_header():
    x, y, z = symbols('x,y,z')
    expr = (x + y)*z
    routine = make_routine("test", expr)
    code_gen = FCodeGen()
    source = get_string(code_gen.dump_h, [routine])
    expected = (
        "interface\n"
        "REAL*8 function test(x, y, z)\n"
        "implicit none\n"
        "REAL*8, intent(in) :: x\n"
        "REAL*8, intent(in) :: y\n"
        "REAL*8, intent(in) :: z\n"
        "end function\n"
        "end interface\n"
    )
    assert source == expected


def test_simple_f_codegen():
    x, y, z = symbols('x,y,z')
    expr = (x + y)*z
    result = codegen(
        ("test", expr), "F95", "file", header=False, empty=False)
    expected = [
        ("file.f90",
        "REAL*8 function test(x, y, z)\n"
        "implicit none\n"
        "REAL*8, intent(in) :: x\n"
        "REAL*8, intent(in) :: y\n"
        "REAL*8, intent(in) :: z\n"
        "test = z*(x + y)\n"
        "end function\n"),
        ("file.h",
        "interface\n"
        "REAL*8 function test(x, y, z)\n"
        "implicit none\n"
        "REAL*8, intent(in) :: x\n"
        "REAL*8, intent(in) :: y\n"
        "REAL*8, intent(in) :: z\n"
        "end function\n"
        "end interface\n")
    ]
    assert result == expected


def test_multiple_results_f():
    x, y, z = symbols('x,y,z')
    expr1 = (x + y)*z
    expr2 = (x - y)*z
    routine = make_routine(
        "test",
        [expr1, expr2]
    )
    code_gen = FCodeGen()
    raises(CodeGenError, lambda: get_string(code_gen.dump_h, [routine]))


def test_no_results_f():
    raises(ValueError, lambda: make_routine("test", []))


def test_intrinsic_math_codegen():
    # not included: log10
    from sympy.functions.elementary.complexes import Abs
    from sympy.functions.elementary.exponential import log
    from sympy.functions.elementary.hyperbolic import (cosh, sinh, tanh)
    from sympy.functions.elementary.miscellaneous import sqrt
    from sympy.functions.elementary.trigonometric import (acos, asin, atan, cos, sin, tan)
    x = symbols('x')
    name_expr = [
        ("test_abs", Abs(x)),
        ("test_acos", acos(x)),
        ("test_asin", asin(x)),
        ("test_atan", atan(x)),
        ("test_cos", cos(x)),
        ("test_cosh", cosh(x)),
        ("test_log", log(x)),
        ("test_ln", log(x)),
        ("test_sin", sin(x)),
        ("test_sinh", sinh(x)),
        ("test_sqrt", sqrt(x)),
        ("test_tan", tan(x)),
        ("test_tanh", tanh(x)),
    ]
    result = codegen(name_expr, "F95", "file", header=False, empty=False)
    assert result[0][0] == "file.f90"
    expected = (
        'REAL*8 function test_abs(x)\n'
        'implicit none\n'
        'REAL*8, intent(in) :: x\n'
        'test_abs = abs(x)\n'
        'end function\n'
        'REAL*8 function test_acos(x)\n'
        'implicit none\n'
        'REAL*8, intent(in) :: x\n'
        'test_acos = acos(x)\n'
        'end function\n'
        'REAL*8 function test_asin(x)\n'
        'implicit none\n'
        'REAL*8, intent(in) :: x\n'
        'test_asin = asin(x)\n'
        'end function\n'
        'REAL*8 function test_atan(x)\n'
        'implicit none\n'
        'REAL*8, intent(in) :: x\n'
        'test_atan = atan(x)\n'
        'end function\n'
        'REAL*8 function test_cos(x)\n'
        'implicit none\n'
        'REAL*8, intent(in) :: x\n'
        'test_cos = cos(x)\n'
        'end function\n'
        'REAL*8 function test_cosh(x)\n'
        'implicit none\n'
        'REAL*8, intent(in) :: x\n'
        'test_cosh = cosh(x)\n'
        'end function\n'
        'REAL*8 function test_log(x)\n'
        'implicit none\n'
        'REAL*8, intent(in) :: x\n'
        'test_log = log(x)\n'
        'end function\n'
        'REAL*8 function test_ln(x)\n'
        'implicit none\n'
        'REAL*8, intent(in) :: x\n'
        'test_ln = log(x)\n'
        'end function\n'
        'REAL*8 function test_sin(x)\n'
        'implicit none\n'
        'REAL*8, intent(in) :: x\n'
        'test_sin = sin(x)\n'
        'end function\n'
        'REAL*8 function test_sinh(x)\n'
        'implicit none\n'
        'REAL*8, intent(in) :: x\n'
        'test_sinh = sinh(x)\n'
        'end function\n'
        'REAL*8 function test_sqrt(x)\n'
        'implicit none\n'
        'REAL*8, intent(in) :: x\n'
        'test_sqrt = sqrt(x)\n'
        'end function\n'
        'REAL*8 function test_tan(x)\n'
        'implicit none\n'
        'REAL*8, intent(in) :: x\n'
        'test_tan = tan(x)\n'
        'end function\n'
        'REAL*8 function test_tanh(x)\n'
        'implicit none\n'
        'REAL*8, intent(in) :: x\n'
        'test_tanh = tanh(x)\n'
        'end function\n'
    )
    assert result[0][1] == expected

    assert result[1][0] == "file.h"
    expected = (
        'interface\n'
        'REAL*8 function test_abs(x)\n'
        'implicit none\n'
        'REAL*8, intent(in) :: x\n'
        'end function\n'
        'end interface\n'
        'interface\n'
        'REAL*8 function test_acos(x)\n'
        'implicit none\n'
        'REAL*8, intent(in) :: x\n'
        'end function\n'
        'end interface\n'
        'interface\n'
        'REAL*8 function test_asin(x)\n'
        'implicit none\n'
        'REAL*8, intent(in) :: x\n'
        'end function\n'
        'end interface\n'
        'interface\n'
        'REAL*8 function test_atan(x)\n'
        'implicit none\n'
        'REAL*8, intent(in) :: x\n'
        'end function\n'
        'end interface\n'
        'interface\n'
        'REAL*8 function test_cos(x)\n'
        'implicit none\n'
        'REAL*8, intent(in) :: x\n'
        'end function\n'
        'end interface\n'
        'interface\n'
        'REAL*8 function test_cosh(x)\n'
        'implicit none\n'
        'REAL*8, intent(in) :: x\n'
        'end function\n'
        'end interface\n'
        'interface\n'
        'REAL*8 function test_log(x)\n'
        'implicit none\n'
        'REAL*8, intent(in) :: x\n'
        'end function\n'
        'end interface\n'
        'interface\n'
        'REAL*8 function test_ln(x)\n'
        'implicit none\n'
        'REAL*8, intent(in) :: x\n'
        'end function\n'
        'end interface\n'
        'interface\n'
        'REAL*8 function test_sin(x)\n'
        'implicit none\n'
        'REAL*8, intent(in) :: x\n'
        'end function\n'
        'end interface\n'
        'interface\n'
        'REAL*8 function test_sinh(x)\n'
        'implicit none\n'
        'REAL*8, intent(in) :: x\n'
        'end function\n'
        'end interface\n'
        'interface\n'
        'REAL*8 function test_sqrt(x)\n'
        'implicit none\n'
        'REAL*8, intent(in) :: x\n'
        'end function\n'
        'end interface\n'
        'interface\n'
        'REAL*8 function test_tan(x)\n'
        'implicit none\n'
        'REAL*8, intent(in) :: x\n'
        'end function\n'
        'end interface\n'
        'interface\n'
        'REAL*8 function test_tanh(x)\n'
        'implicit none\n'
        'REAL*8, intent(in) :: x\n'
        'end function\n'
        'end interface\n'
    )
    assert result[1][1] == expected


def test_intrinsic_math2_codegen():
    # not included: frexp, ldexp, modf, fmod
    from sympy.functions.elementary.trigonometric import atan2
    x, y = symbols('x,y')
    name_expr = [
        ("test_atan2", atan2(x, y)),
        ("test_pow", x**y),
    ]
    result = codegen(name_expr, "F95", "file", header=False, empty=False)
    assert result[0][0] == "file.f90"
    expected = (
        'REAL*8 function test_atan2(x, y)\n'
        'implicit none\n'
        'REAL*8, intent(in) :: x\n'
        'REAL*8, intent(in) :: y\n'
        'test_atan2 = atan2(x, y)\n'
        'end function\n'
        'REAL*8 function test_pow(x, y)\n'
        'implicit none\n'
        'REAL*8, intent(in) :: x\n'
        'REAL*8, intent(in) :: y\n'
        'test_pow = x**y\n'
        'end function\n'
    )
    assert result[0][1] == expected

    assert result[1][0] == "file.h"
    expected = (
        'interface\n'
        'REAL*8 function test_atan2(x, y)\n'
        'implicit none\n'
        'REAL*8, intent(in) :: x\n'
        'REAL*8, intent(in) :: y\n'
        'end function\n'
        'end interface\n'
        'interface\n'
        'REAL*8 function test_pow(x, y)\n'
        'implicit none\n'
        'REAL*8, intent(in) :: x\n'
        'REAL*8, intent(in) :: y\n'
        'end function\n'
        'end interface\n'
    )
    assert result[1][1] == expected


def test_complicated_codegen_f95():
    from sympy.functions.elementary.trigonometric import (cos, sin, tan)
    x, y, z = symbols('x,y,z')
    name_expr = [
        ("test1", ((sin(x) + cos(y) + tan(z))**7).expand()),
        ("test2", cos(cos(cos(cos(cos(cos(cos(cos(x + y + z))))))))),
    ]
    result = codegen(name_expr, "F95", "file", header=False, empty=False)
    assert result[0][0] == "file.f90"
    expected = (
        'REAL*8 function test1(x, y, z)\n'
        'implicit none\n'
        'REAL*8, intent(in) :: x\n'
        'REAL*8, intent(in) :: y\n'
        'REAL*8, intent(in) :: z\n'
        'test1 = sin(x)**7 + 7*sin(x)**6*cos(y) + 7*sin(x)**6*tan(z) + 21*sin(x) &\n'
        '      **5*cos(y)**2 + 42*sin(x)**5*cos(y)*tan(z) + 21*sin(x)**5*tan(z) &\n'
        '      **2 + 35*sin(x)**4*cos(y)**3 + 105*sin(x)**4*cos(y)**2*tan(z) + &\n'
        '      105*sin(x)**4*cos(y)*tan(z)**2 + 35*sin(x)**4*tan(z)**3 + 35*sin( &\n'
        '      x)**3*cos(y)**4 + 140*sin(x)**3*cos(y)**3*tan(z) + 210*sin(x)**3* &\n'
        '      cos(y)**2*tan(z)**2 + 140*sin(x)**3*cos(y)*tan(z)**3 + 35*sin(x) &\n'
        '      **3*tan(z)**4 + 21*sin(x)**2*cos(y)**5 + 105*sin(x)**2*cos(y)**4* &\n'
        '      tan(z) + 210*sin(x)**2*cos(y)**3*tan(z)**2 + 210*sin(x)**2*cos(y) &\n'
        '      **2*tan(z)**3 + 105*sin(x)**2*cos(y)*tan(z)**4 + 21*sin(x)**2*tan &\n'
        '      (z)**5 + 7*sin(x)*cos(y)**6 + 42*sin(x)*cos(y)**5*tan(z) + 105* &\n'
        '      sin(x)*cos(y)**4*tan(z)**2 + 140*sin(x)*cos(y)**3*tan(z)**3 + 105 &\n'
        '      *sin(x)*cos(y)**2*tan(z)**4 + 42*sin(x)*cos(y)*tan(z)**5 + 7*sin( &\n'
        '      x)*tan(z)**6 + cos(y)**7 + 7*cos(y)**6*tan(z) + 21*cos(y)**5*tan( &\n'
        '      z)**2 + 35*cos(y)**4*tan(z)**3 + 35*cos(y)**3*tan(z)**4 + 21*cos( &\n'
        '      y)**2*tan(z)**5 + 7*cos(y)*tan(z)**6 + tan(z)**7\n'
        'end function\n'
        'REAL*8 function test2(x, y, z)\n'
        'implicit none\n'
        'REAL*8, intent(in) :: x\n'
        'REAL*8, intent(in) :: y\n'
        'REAL*8, intent(in) :: z\n'
        'test2 = cos(cos(cos(cos(cos(cos(cos(cos(x + y + z))))))))\n'
        'end function\n'
    )
    assert result[0][1] == expected
    assert result[1][0] == "file.h"
    expected = (
        'interface\n'
        'REAL*8 function test1(x, y, z)\n'
        'implicit none\n'
        'REAL*8, intent(in) :: x\n'
        'REAL*8, intent(in) :: y\n'
        'REAL*8, intent(in) :: z\n'
        'end function\n'
        'end interface\n'
        'interface\n'
        'REAL*8 function test2(x, y, z)\n'
        'implicit none\n'
        'REAL*8, intent(in) :: x\n'
        'REAL*8, intent(in) :: y\n'
        'REAL*8, intent(in) :: z\n'
        'end function\n'
        'end interface\n'
    )
    assert result[1][1] == expected


def test_loops():
    from sympy.tensor import IndexedBase, Idx
    from sympy.core.symbol import symbols

    n, m = symbols('n,m', integer=True)
    A, x, y = map(IndexedBase, 'Axy')
    i = Idx('i', m)
    j = Idx('j', n)

    (f1, code), (f2, interface) = codegen(
        ('matrix_vector', Eq(y[i], A[i, j]*x[j])), "F95", "file", header=False, empty=False)

    assert f1 == 'file.f90'
    expected = (
        'subroutine matrix_vector(A, m, n, x, y)\n'
        'implicit none\n'
        'INTEGER*4, intent(in) :: m\n'
        'INTEGER*4, intent(in) :: n\n'
        'REAL*8, intent(in), dimension(1:m, 1:n) :: A\n'
        'REAL*8, intent(in), dimension(1:n) :: x\n'
        'REAL*8, intent(out), dimension(1:m) :: y\n'
        'INTEGER*4 :: i\n'
        'INTEGER*4 :: j\n'
        'do i = 1, m\n'
        '   y(i) = 0\n'
        'end do\n'
        'do i = 1, m\n'
        '   do j = 1, n\n'
        '      y(i) = %(rhs)s + y(i)\n'
        '   end do\n'
        'end do\n'
        'end subroutine\n'
    )

    assert code == expected % {'rhs': 'A(i, j)*x(j)'} or\
        code == expected % {'rhs': 'x(j)*A(i, j)'}
    assert f2 == 'file.h'
    assert interface == (
        'interface\n'
        'subroutine matrix_vector(A, m, n, x, y)\n'
        'implicit none\n'
        'INTEGER*4, intent(in) :: m\n'
        'INTEGER*4, intent(in) :: n\n'
        'REAL*8, intent(in), dimension(1:m, 1:n) :: A\n'
        'REAL*8, intent(in), dimension(1:n) :: x\n'
        'REAL*8, intent(out), dimension(1:m) :: y\n'
        'end subroutine\n'
        'end interface\n'
    )


def test_dummy_loops_f95():
    from sympy.tensor import IndexedBase, Idx
    i, m = symbols('i m', integer=True, cls=Dummy)
    x = IndexedBase('x')
    y = IndexedBase('y')
    i = Idx(i, m)
    expected = (
        'subroutine test_dummies(m_%(mcount)i, x, y)\n'
        'implicit none\n'
        'INTEGER*4, intent(in) :: m_%(mcount)i\n'
        'REAL*8, intent(in), dimension(1:m_%(mcount)i) :: x\n'
        'REAL*8, intent(out), dimension(1:m_%(mcount)i) :: y\n'
        'INTEGER*4 :: i_%(icount)i\n'
        'do i_%(icount)i = 1, m_%(mcount)i\n'
        '   y(i_%(icount)i) = x(i_%(icount)i)\n'
        'end do\n'
        'end subroutine\n'
    ) % {'icount': i.label.dummy_index, 'mcount': m.dummy_index}
    r = make_routine('test_dummies', Eq(y[i], x[i]))
    c = FCodeGen()
    code = get_string(c.dump_f95, [r])
    assert code == expected


def test_loops_InOut():
    from sympy.tensor import IndexedBase, Idx
    from sympy.core.symbol import symbols

    i, j, n, m = symbols('i,j,n,m', integer=True)
    A, x, y = symbols('A,x,y')
    A = IndexedBase(A)[Idx(i, m), Idx(j, n)]
    x = IndexedBase(x)[Idx(j, n)]
    y = IndexedBase(y)[Idx(i, m)]

    (f1, code), (f2, interface) = codegen(
        ('matrix_vector', Eq(y, y + A*x)), "F95", "file", header=False, empty=False)

    assert f1 == 'file.f90'
    expected = (
        'subroutine matrix_vector(A, m, n, x, y)\n'
        'implicit none\n'
        'INTEGER*4, intent(in) :: m\n'
        'INTEGER*4, intent(in) :: n\n'
        'REAL*8, intent(in), dimension(1:m, 1:n) :: A\n'
        'REAL*8, intent(in), dimension(1:n) :: x\n'
        'REAL*8, intent(inout), dimension(1:m) :: y\n'
        'INTEGER*4 :: i\n'
        'INTEGER*4 :: j\n'
        'do i = 1, m\n'
        '   do j = 1, n\n'
        '      y(i) = %(rhs)s + y(i)\n'
        '   end do\n'
        'end do\n'
        'end subroutine\n'
    )

    assert (code == expected % {'rhs': 'A(i, j)*x(j)'} or
            code == expected % {'rhs': 'x(j)*A(i, j)'})
    assert f2 == 'file.h'
    assert interface == (
        'interface\n'
        'subroutine matrix_vector(A, m, n, x, y)\n'
        'implicit none\n'
        'INTEGER*4, intent(in) :: m\n'
        'INTEGER*4, intent(in) :: n\n'
        'REAL*8, intent(in), dimension(1:m, 1:n) :: A\n'
        'REAL*8, intent(in), dimension(1:n) :: x\n'
        'REAL*8, intent(inout), dimension(1:m) :: y\n'
        'end subroutine\n'
        'end interface\n'
    )


def test_partial_loops_f():
    # check that loop boundaries are determined by Idx, and array strides
    # determined by shape of IndexedBase object.
    from sympy.tensor import IndexedBase, Idx
    from sympy.core.symbol import symbols
    n, m, o, p = symbols('n m o p', integer=True)
    A = IndexedBase('A', shape=(m, p))
    x = IndexedBase('x')
    y = IndexedBase('y')
    i = Idx('i', (o, m - 5))  # Note: bounds are inclusive
    j = Idx('j', n)          # dimension n corresponds to bounds (0, n - 1)

    (f1, code), (f2, interface) = codegen(
        ('matrix_vector', Eq(y[i], A[i, j]*x[j])), "F95", "file", header=False, empty=False)

    expected = (
        'subroutine matrix_vector(A, m, n, o, p, x, y)\n'
        'implicit none\n'
        'INTEGER*4, intent(in) :: m\n'
        'INTEGER*4, intent(in) :: n\n'
        'INTEGER*4, intent(in) :: o\n'
        'INTEGER*4, intent(in) :: p\n'
        'REAL*8, intent(in), dimension(1:m, 1:p) :: A\n'
        'REAL*8, intent(in), dimension(1:n) :: x\n'
        'REAL*8, intent(out), dimension(1:%(iup-ilow)s) :: y\n'
        'INTEGER*4 :: i\n'
        'INTEGER*4 :: j\n'
        'do i = %(ilow)s, %(iup)s\n'
        '   y(i) = 0\n'
        'end do\n'
        'do i = %(ilow)s, %(iup)s\n'
        '   do j = 1, n\n'
        '      y(i) = %(rhs)s + y(i)\n'
        '   end do\n'
        'end do\n'
        'end subroutine\n'
    ) % {
        'rhs': '%(rhs)s',
        'iup': str(m - 4),
        'ilow': str(1 + o),
        'iup-ilow': str(m - 4 - o)
    }

    assert code == expected % {'rhs': 'A(i, j)*x(j)'} or\
        code == expected % {'rhs': 'x(j)*A(i, j)'}


def test_output_arg_f():
    from sympy.core.relational import Equality
    from sympy.functions.elementary.trigonometric import (cos, sin)
    x, y, z = symbols("x,y,z")
    r = make_routine("foo", [Equality(y, sin(x)), cos(x)])
    c = FCodeGen()
    result = c.write([r], "test", header=False, empty=False)
    assert result[0][0] == "test.f90"
    assert result[0][1] == (
        'REAL*8 function foo(x, y)\n'
        'implicit none\n'
        'REAL*8, intent(in) :: x\n'
        'REAL*8, intent(out) :: y\n'
        'y = sin(x)\n'
        'foo = cos(x)\n'
        'end function\n'
    )


def test_inline_function():
    from sympy.tensor import IndexedBase, Idx
    from sympy.core.symbol import symbols
    n, m = symbols('n m', integer=True)
    A, x, y = map(IndexedBase, 'Axy')
    i = Idx('i', m)
    p = FCodeGen()
    func = implemented_function('func', Lambda(n, n*(n + 1)))
    routine = make_routine('test_inline', Eq(y[i], func(x[i])))
    code = get_string(p.dump_f95, [routine])
    expected = (
        'subroutine test_inline(m, x, y)\n'
        'implicit none\n'
        'INTEGER*4, intent(in) :: m\n'
        'REAL*8, intent(in), dimension(1:m) :: x\n'
        'REAL*8, intent(out), dimension(1:m) :: y\n'
        'INTEGER*4 :: i\n'
        'do i = 1, m\n'
        '   y(i) = %s*%s\n'
        'end do\n'
        'end subroutine\n'
    )
    args = ('x(i)', '(x(i) + 1)')
    assert code == expected % args or\
        code == expected % args[::-1]


def test_f_code_call_signature_wrap():
    # Issue #7934
    x = symbols('x:20')
    expr = 0
    for sym in x:
        expr += sym
    routine = make_routine("test", expr)
    code_gen = FCodeGen()
    source = get_string(code_gen.dump_f95, [routine])
    expected = """\
REAL*8 function test(x0, x1, x10, x11, x12, x13, x14, x15, x16, x17, x18, &
      x19, x2, x3, x4, x5, x6, x7, x8, x9)
implicit none
REAL*8, intent(in) :: x0
REAL*8, intent(in) :: x1
REAL*8, intent(in) :: x10
REAL*8, intent(in) :: x11
REAL*8, intent(in) :: x12
REAL*8, intent(in) :: x13
REAL*8, intent(in) :: x14
REAL*8, intent(in) :: x15
REAL*8, intent(in) :: x16
REAL*8, intent(in) :: x17
REAL*8, intent(in) :: x18
REAL*8, intent(in) :: x19
REAL*8, intent(in) :: x2
REAL*8, intent(in) :: x3
REAL*8, intent(in) :: x4
REAL*8, intent(in) :: x5
REAL*8, intent(in) :: x6
REAL*8, intent(in) :: x7
REAL*8, intent(in) :: x8
REAL*8, intent(in) :: x9
test = x0 + x1 + x10 + x11 + x12 + x13 + x14 + x15 + x16 + x17 + x18 + &
      x19 + x2 + x3 + x4 + x5 + x6 + x7 + x8 + x9
end function
"""
    assert source == expected


def test_check_case():
    x, X = symbols('x,X')
    raises(CodeGenError, lambda: codegen(('test', x*X), 'f95', 'prefix'))


def test_check_case_false_positive():
    # The upper case/lower case exception should not be triggered by SymPy
    # objects that differ only because of assumptions.  (It may be useful to
    # have a check for that as well, but here we only want to test against
    # false positives with respect to case checking.)
    x1 = symbols('x')
    x2 = symbols('x', my_assumption=True)
    try:
        codegen(('test', x1*x2), 'f95', 'prefix')
    except CodeGenError as e:
        if e.args[0].startswith("Fortran ignores case."):
            raise AssertionError("This exception should not be raised!")


def test_c_fortran_omit_routine_name():
    x, y = symbols("x,y")
    name_expr = [("foo", 2*x)]
    result = codegen(name_expr, "F95", header=False, empty=False)
    expresult = codegen(name_expr, "F95", "foo", header=False, empty=False)
    assert result[0][1] == expresult[0][1]

    name_expr = ("foo", x*y)
    result = codegen(name_expr, "F95", header=False, empty=False)
    expresult = codegen(name_expr, "F95", "foo", header=False, empty=False)
    assert result[0][1] == expresult[0][1]

    name_expr = ("foo", Matrix([[x, y], [x+y, x-y]]))
    result = codegen(name_expr, "C89", header=False, empty=False)
    expresult = codegen(name_expr, "C89", "foo", header=False, empty=False)
    assert result[0][1] == expresult[0][1]


def test_fcode_matrix_output():
    x, y, z = symbols('x,y,z')
    e1 = x + y
    e2 = Matrix([[x, y], [z, 16]])
    name_expr = ("test", (e1, e2))
    result = codegen(name_expr, "f95", "test", header=False, empty=False)
    source = result[0][1]
    expected = (
        "REAL*8 function test(x, y, z, out_%(hash)s)\n"
        "implicit none\n"
        "REAL*8, intent(in) :: x\n"
        "REAL*8, intent(in) :: y\n"
        "REAL*8, intent(in) :: z\n"
        "REAL*8, intent(out), dimension(1:2, 1:2) :: out_%(hash)s\n"
        "out_%(hash)s(1, 1) = x\n"
        "out_%(hash)s(2, 1) = z\n"
        "out_%(hash)s(1, 2) = y\n"
        "out_%(hash)s(2, 2) = 16\n"
        "test = x + y\n"
        "end function\n"
    )
    # look for the magic number
    a = source.splitlines()[5]
    b = a.split('_')
    out = b[1]
    expected = expected % {'hash': out}
    assert source == expected


def test_fcode_results_named_ordered():
    x, y, z = symbols('x,y,z')
    B, C = symbols('B,C')
    A = MatrixSymbol('A', 1, 3)
    expr1 = Equality(A, Matrix([[1, 2, x]]))
    expr2 = Equality(C, (x + y)*z)
    expr3 = Equality(B, 2*x)
    name_expr = ("test", [expr1, expr2, expr3])
    result = codegen(name_expr, "f95", "test", header=False, empty=False,
                     argument_sequence=(x, z, y, C, A, B))
    source = result[0][1]
    expected = (
        "subroutine test(x, z, y, C, A, B)\n"
        "implicit none\n"
        "REAL*8, intent(in) :: x\n"
        "REAL*8, intent(in) :: z\n"
        "REAL*8, intent(in) :: y\n"
        "REAL*8, intent(out) :: C\n"
        "REAL*8, intent(out) :: B\n"
        "REAL*8, intent(out), dimension(1:1, 1:3) :: A\n"
        "C = z*(x + y)\n"
        "A(1, 1) = 1\n"
        "A(1, 2) = 2\n"
        "A(1, 3) = x\n"
        "B = 2*x\n"
        "end subroutine\n"
    )
    assert source == expected


def test_fcode_matrixsymbol_slice():
    A = MatrixSymbol('A', 2, 3)
    B = MatrixSymbol('B', 1, 3)
    C = MatrixSymbol('C', 1, 3)
    D = MatrixSymbol('D', 2, 1)
    name_expr = ("test", [Equality(B, A[0, :]),
                          Equality(C, A[1, :]),
                          Equality(D, A[:, 2])])
    result = codegen(name_expr, "f95", "test", header=False, empty=False)
    source = result[0][1]
    expected = (
        "subroutine test(A, B, C, D)\n"
        "implicit none\n"
        "REAL*8, intent(in), dimension(1:2, 1:3) :: A\n"
        "REAL*8, intent(out), dimension(1:1, 1:3) :: B\n"
        "REAL*8, intent(out), dimension(1:1, 1:3) :: C\n"
        "REAL*8, intent(out), dimension(1:2, 1:1) :: D\n"
        "B(1, 1) = A(1, 1)\n"
        "B(1, 2) = A(1, 2)\n"
        "B(1, 3) = A(1, 3)\n"
        "C(1, 1) = A(2, 1)\n"
        "C(1, 2) = A(2, 2)\n"
        "C(1, 3) = A(2, 3)\n"
        "D(1, 1) = A(1, 3)\n"
        "D(2, 1) = A(2, 3)\n"
        "end subroutine\n"
    )
    assert source == expected


def test_fcode_matrixsymbol_slice_autoname():
    # see issue #8093
    A = MatrixSymbol('A', 2, 3)
    name_expr = ("test", A[:, 1])
    result = codegen(name_expr, "f95", "test", header=False, empty=False)
    source = result[0][1]
    expected = (
        "subroutine test(A, out_%(hash)s)\n"
        "implicit none\n"
        "REAL*8, intent(in), dimension(1:2, 1:3) :: A\n"
        "REAL*8, intent(out), dimension(1:2, 1:1) :: out_%(hash)s\n"
        "out_%(hash)s(1, 1) = A(1, 2)\n"
        "out_%(hash)s(2, 1) = A(2, 2)\n"
        "end subroutine\n"
    )
    # look for the magic number
    a = source.splitlines()[3]
    b = a.split('_')
    out = b[1]
    expected = expected % {'hash': out}
    assert source == expected


def test_global_vars():
    x, y, z, t = symbols("x y z t")
    result = codegen(('f', x*y), "F95", header=False, empty=False,
                     global_vars=(y,))
    source = result[0][1]
    expected = (
        "REAL*8 function f(x)\n"
        "implicit none\n"
        "REAL*8, intent(in) :: x\n"
        "f = x*y\n"
        "end function\n"
        )
    assert source == expected

    expected = (
        '#include "f.h"\n'
        '#include <math.h>\n'
        'double f(double x, double y) {\n'
        '   double f_result;\n'
        '   f_result = x*y + z;\n'
        '   return f_result;\n'
        '}\n'
    )
    result = codegen(('f', x*y+z), "C", header=False, empty=False,
                     global_vars=(z, t))
    source = result[0][1]
    assert source == expected

def test_custom_codegen():
    from sympy.printing.c import C99CodePrinter
    from sympy.functions.elementary.exponential import exp

    printer = C99CodePrinter(settings={'user_functions': {'exp': 'fastexp'}})

    x, y = symbols('x y')
    expr = exp(x + y)

    # replace math.h with a different header
    gen = C99CodeGen(printer=printer,
                     preprocessor_statements=['#include "fastexp.h"'])

    expected = (
        '#include "expr.h"\n'
        '#include "fastexp.h"\n'
        'double expr(double x, double y) {\n'
        '   double expr_result;\n'
        '   expr_result = fastexp(x + y);\n'
        '   return expr_result;\n'
        '}\n'
    )

    result = codegen(('expr', expr), header=False, empty=False, code_gen=gen)
    source = result[0][1]
    assert source == expected

    # use both math.h and an external header
    gen = C99CodeGen(printer=printer)
    gen.preprocessor_statements.append('#include "fastexp.h"')

    expected = (
        '#include "expr.h"\n'
        '#include <math.h>\n'
        '#include "fastexp.h"\n'
        'double expr(double x, double y) {\n'
        '   double expr_result;\n'
        '   expr_result = fastexp(x + y);\n'
        '   return expr_result;\n'
        '}\n'
    )

    result = codegen(('expr', expr), header=False, empty=False, code_gen=gen)
    source = result[0][1]
    assert source == expected

def test_c_with_printer():
    # issue 13586
    from sympy.printing.c import C99CodePrinter
    class CustomPrinter(C99CodePrinter):
        def _print_Pow(self, expr):
            return "fastpow({}, {})".format(self._print(expr.base),
                                            self._print(expr.exp))

    x = symbols('x')
    expr = x**3
    expected =[
        ("file.c",
        "#include \"file.h\"\n"
        "#include <math.h>\n"
        "double test(double x) {\n"
        "   double test_result;\n"
        "   test_result = fastpow(x, 3);\n"
        "   return test_result;\n"
        "}\n"),
        ("file.h",
        "#ifndef PROJECT__FILE__H\n"
        "#define PROJECT__FILE__H\n"
        "double test(double x);\n"
        "#endif\n")
    ]
    result = codegen(("test", expr), "C","file", header=False, empty=False, printer = CustomPrinter())
    assert result == expected


def test_fcode_complex():
    import sympy.utilities.codegen
    sympy.utilities.codegen.COMPLEX_ALLOWED = True
    x = Symbol('x', real=True)
    y = Symbol('y',real=True)
    result = codegen(('test',x+y), 'f95', 'test', header=False, empty=False)
    source = (result[0][1])
    expected = (
        "REAL*8 function test(x, y)\n"
        "implicit none\n"
        "REAL*8, intent(in) :: x\n"
        "REAL*8, intent(in) :: y\n"
        "test = x + y\n"
        "end function\n")
    assert source == expected
    x = Symbol('x')
    y = Symbol('y',real=True)
    result = codegen(('test',x+y), 'f95', 'test', header=False, empty=False)
    source = (result[0][1])
    expected = (
        "COMPLEX*16 function test(x, y)\n"
        "implicit none\n"
        "COMPLEX*16, intent(in) :: x\n"
        "REAL*8, intent(in) :: y\n"
        "test = x + y\n"
        "end function\n"
        )
    assert source==expected
    sympy.utilities.codegen.COMPLEX_ALLOWED = False
