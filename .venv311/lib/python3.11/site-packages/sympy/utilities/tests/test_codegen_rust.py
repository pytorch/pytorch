from io import StringIO

from sympy.core import S, symbols, pi, Catalan, EulerGamma, Function
from sympy.core.relational import Equality
from sympy.functions.elementary.piecewise import Piecewise
from sympy.utilities.codegen import RustCodeGen, codegen, make_routine
from sympy.testing.pytest import XFAIL
import sympy


x, y, z = symbols('x,y,z')


def test_empty_rust_code():
    code_gen = RustCodeGen()
    output = StringIO()
    code_gen.dump_rs([], output, "file", header=False, empty=False)
    source = output.getvalue()
    assert source == ""


def test_simple_rust_code():
    name_expr = ("test", (x + y)*z)
    result, = codegen(name_expr, "Rust", header=False, empty=False)
    assert result[0] == "test.rs"
    source = result[1]
    expected = (
        "fn test(x: f64, y: f64, z: f64) -> f64 {\n"
        "    let out1 = z*(x + y);\n"
        "    out1\n"
        "}\n"
    )
    assert source == expected


def test_simple_code_with_header():
    name_expr = ("test", (x + y)*z)
    result, = codegen(name_expr, "Rust", header=True, empty=False)
    assert result[0] == "test.rs"
    source = result[1]
    version_str = "Code generated with SymPy %s" % sympy.__version__
    version_line = version_str.center(76).rstrip()
    expected = (
        "/*\n"
        " *%(version_line)s\n"
        " *\n"
        " *              See http://www.sympy.org/ for more information.\n"
        " *\n"
        " *                       This file is part of 'project'\n"
        " */\n"
        "fn test(x: f64, y: f64, z: f64) -> f64 {\n"
        "    let out1 = z*(x + y);\n"
        "    out1\n"
        "}\n"
    ) % {'version_line': version_line}
    assert source == expected


def test_simple_code_nameout():
    expr = Equality(z, (x + y))
    name_expr = ("test", expr)
    result, = codegen(name_expr, "Rust", header=False, empty=False)
    source = result[1]
    expected = (
        "fn test(x: f64, y: f64) -> f64 {\n"
        "    let z = x + y;\n"
        "    z\n"
        "}\n"
    )
    assert source == expected


def test_numbersymbol():
    name_expr = ("test", pi**Catalan)
    result, = codegen(name_expr, "Rust", header=False, empty=False)
    source = result[1]
    expected = (
        "fn test() -> f64 {\n"
        "    const Catalan: f64 = %s;\n"
        "    let out1 = PI.powf(Catalan);\n"
        "    out1\n"
        "}\n"
    ) % Catalan.evalf(17)
    assert source == expected


@XFAIL
def test_numbersymbol_inline():
    # FIXME: how to pass inline to the RustCodePrinter?
    name_expr = ("test", [pi**Catalan, EulerGamma])
    result, = codegen(name_expr, "Rust", header=False,
                      empty=False, inline=True)
    source = result[1]
    expected = (
        "fn test() -> (f64, f64) {\n"
        "    const Catalan: f64 = %s;\n"
        "    const EulerGamma: f64 = %s;\n"
        "    let out1 = PI.powf(Catalan);\n"
        "    let out2 = EulerGamma);\n"
        "    (out1, out2)\n"
        "}\n"
    ) % (Catalan.evalf(17), EulerGamma.evalf(17))
    assert source == expected


def test_argument_order():
    expr = x + y
    routine = make_routine("test", expr, argument_sequence=[z, x, y], language="rust")
    code_gen = RustCodeGen()
    output = StringIO()
    code_gen.dump_rs([routine], output, "test", header=False, empty=False)
    source = output.getvalue()
    expected = (
        "fn test(z: f64, x: f64, y: f64) -> f64 {\n"
        "    let out1 = x + y;\n"
        "    out1\n"
        "}\n"
    )
    assert source == expected


def test_multiple_results_rust():
    # Here the output order is the input order
    expr1 = (x + y)*z
    expr2 = (x - y)*z
    name_expr = ("test", [expr1, expr2])
    result, = codegen(name_expr, "Rust", header=False, empty=False)
    source = result[1]
    expected = (
        "fn test(x: f64, y: f64, z: f64) -> (f64, f64) {\n"
        "    let out1 = z*(x + y);\n"
        "    let out2 = z*(x - y);\n"
        "    (out1, out2)\n"
        "}\n"
    )
    assert source == expected


def test_results_named_unordered():
    # Here output order is based on name_expr
    A, B, C = symbols('A,B,C')
    expr1 = Equality(C, (x + y)*z)
    expr2 = Equality(A, (x - y)*z)
    expr3 = Equality(B, 2*x)
    name_expr = ("test", [expr1, expr2, expr3])
    result, = codegen(name_expr, "Rust", header=False, empty=False)
    source = result[1]
    expected = (
        "fn test(x: f64, y: f64, z: f64) -> (f64, f64, f64) {\n"
        "    let C = z*(x + y);\n"
        "    let A = z*(x - y);\n"
        "    let B = 2*x;\n"
        "    (C, A, B)\n"
        "}\n"
    )
    assert source == expected


def test_results_named_ordered():
    A, B, C = symbols('A,B,C')
    expr1 = Equality(C, (x + y)*z)
    expr2 = Equality(A, (x - y)*z)
    expr3 = Equality(B, 2*x)
    name_expr = ("test", [expr1, expr2, expr3])
    result = codegen(name_expr, "Rust", header=False, empty=False,
                     argument_sequence=(x, z, y))
    assert result[0][0] == "test.rs"
    source = result[0][1]
    expected = (
        "fn test(x: f64, z: f64, y: f64) -> (f64, f64, f64) {\n"
        "    let C = z*(x + y);\n"
        "    let A = z*(x - y);\n"
        "    let B = 2*x;\n"
        "    (C, A, B)\n"
        "}\n"
    )
    assert source == expected


def test_complicated_rs_codegen():
    from sympy.functions.elementary.trigonometric import (cos, sin, tan)
    name_expr = ("testlong",
            [ ((sin(x) + cos(y) + tan(z))**3).expand(),
            cos(cos(cos(cos(cos(cos(cos(cos(x + y + z))))))))
    ])
    result = codegen(name_expr, "Rust", header=False, empty=False)
    assert result[0][0] == "testlong.rs"
    source = result[0][1]
    expected = (
        "fn testlong(x: f64, y: f64, z: f64) -> (f64, f64) {\n"
        "    let out1 = x.sin().powi(3) + 3*x.sin().powi(2)*y.cos()"
        " + 3*x.sin().powi(2)*z.tan() + 3*x.sin()*y.cos().powi(2)"
        " + 6*x.sin()*y.cos()*z.tan() + 3*x.sin()*z.tan().powi(2)"
        " + y.cos().powi(3) + 3*y.cos().powi(2)*z.tan()"
        " + 3*y.cos()*z.tan().powi(2) + z.tan().powi(3);\n"
        "    let out2 = (x + y + z).cos().cos().cos().cos()"
        ".cos().cos().cos().cos();\n"
        "    (out1, out2)\n"
        "}\n"
    )
    assert source == expected


def test_output_arg_mixed_unordered():
    # named outputs are alphabetical, unnamed output appear in the given order
    from sympy.functions.elementary.trigonometric import (cos, sin)
    a = symbols("a")
    name_expr = ("foo", [cos(2*x), Equality(y, sin(x)), cos(x), Equality(a, sin(2*x))])
    result, = codegen(name_expr, "Rust", header=False, empty=False)
    assert result[0] == "foo.rs"
    source = result[1]
    expected = (
        "fn foo(x: f64) -> (f64, f64, f64, f64) {\n"
        "    let out1 = (2*x).cos();\n"
        "    let y = x.sin();\n"
        "    let out3 = x.cos();\n"
        "    let a = (2*x).sin();\n"
        "    (out1, y, out3, a)\n"
        "}\n"
    )
    assert source == expected


def test_piecewise_():
    pw = Piecewise((0, x < -1), (x**2, x <= 1), (-x+2, x > 1), (1, True), evaluate=False)
    name_expr = ("pwtest", pw)
    result, = codegen(name_expr, "Rust", header=False, empty=False)
    source = result[1]
    expected = (
        "fn pwtest(x: f64) -> f64 {\n"
        "    let out1 = if (x < -1.0) {\n"
        "        0\n"
        "    } else if (x <= 1.0) {\n"
        "        x.powi(2)\n"
        "    } else if (x > 1.0) {\n"
        "        2 - x\n"
        "    } else {\n"
        "        1\n"
        "    };\n"
        "    out1\n"
        "}\n"
    )
    assert source == expected


@XFAIL
def test_piecewise_inline():
    # FIXME: how to pass inline to the RustCodePrinter?
    pw = Piecewise((0, x < -1), (x**2, x <= 1), (-x+2, x > 1), (1, True))
    name_expr = ("pwtest", pw)
    result, = codegen(name_expr, "Rust", header=False, empty=False,
                      inline=True)
    source = result[1]
    expected = (
        "fn pwtest(x: f64) -> f64 {\n"
        "    let out1 = if (x < -1) { 0 } else if (x <= 1) { x.powi(2) }"
        " else if (x > 1) { -x + 2 } else { 1 };\n"
        "    out1\n"
        "}\n"
    )
    assert source == expected


def test_multifcns_per_file():
    name_expr = [ ("foo", [2*x, 3*y]), ("bar", [y**2, 4*y]) ]
    result = codegen(name_expr, "Rust", header=False, empty=False)
    assert result[0][0] == "foo.rs"
    source = result[0][1]
    expected = (
        "fn foo(x: f64, y: f64) -> (f64, f64) {\n"
        "    let out1 = 2*x;\n"
        "    let out2 = 3*y;\n"
        "    (out1, out2)\n"
        "}\n"
        "fn bar(y: f64) -> (f64, f64) {\n"
        "    let out1 = y.powi(2);\n"
        "    let out2 = 4*y;\n"
        "    (out1, out2)\n"
        "}\n"
    )
    assert source == expected


def test_multifcns_per_file_w_header():
    name_expr = [ ("foo", [2*x, 3*y]), ("bar", [y**2, 4*y]) ]
    result = codegen(name_expr, "Rust", header=True, empty=False)
    assert result[0][0] == "foo.rs"
    source = result[0][1]
    version_str = "Code generated with SymPy %s" % sympy.__version__
    version_line = version_str.center(76).rstrip()
    expected = (
        "/*\n"
        " *%(version_line)s\n"
        " *\n"
        " *              See http://www.sympy.org/ for more information.\n"
        " *\n"
        " *                       This file is part of 'project'\n"
        " */\n"
        "fn foo(x: f64, y: f64) -> (f64, f64) {\n"
        "    let out1 = 2*x;\n"
        "    let out2 = 3*y;\n"
        "    (out1, out2)\n"
        "}\n"
        "fn bar(y: f64) -> (f64, f64) {\n"
        "    let out1 = y.powi(2);\n"
        "    let out2 = 4*y;\n"
        "    (out1, out2)\n"
        "}\n"
    ) % {'version_line': version_line}
    assert source == expected


def test_filename_match_prefix():
    name_expr = [ ("foo", [2*x, 3*y]), ("bar", [y**2, 4*y]) ]
    result, = codegen(name_expr, "Rust", prefix="baz", header=False,
                     empty=False)
    assert result[0] == "baz.rs"


def test_InOutArgument():
    expr = Equality(x, x**2)
    name_expr = ("mysqr", expr)
    result, = codegen(name_expr, "Rust", header=False, empty=False)
    source = result[1]
    expected = (
        "fn mysqr(x: f64) -> f64 {\n"
        "    let x = x.powi(2);\n"
        "    x\n"
        "}\n"
    )
    assert source == expected


def test_InOutArgument_order():
    # can specify the order as (x, y)
    expr = Equality(x, x**2 + y)
    name_expr = ("test", expr)
    result, = codegen(name_expr, "Rust", header=False,
                      empty=False, argument_sequence=(x,y))
    source = result[1]
    expected = (
        "fn test(x: f64, y: f64) -> f64 {\n"
        "    let x = x.powi(2) + y;\n"
        "    x\n"
        "}\n"
    )
    assert source == expected
    # make sure it gives (x, y) not (y, x)
    expr = Equality(x, x**2 + y)
    name_expr = ("test", expr)
    result, = codegen(name_expr, "Rust", header=False, empty=False)
    source = result[1]
    expected = (
        "fn test(x: f64, y: f64) -> f64 {\n"
        "    let x = x.powi(2) + y;\n"
        "    x\n"
        "}\n"
    )
    assert source == expected


def test_not_supported():
    f = Function('f')
    name_expr = ("test", [f(x).diff(x), S.ComplexInfinity])
    result, = codegen(name_expr, "Rust", header=False, empty=False)
    source = result[1]
    expected = (
        "fn test(x: f64) -> (f64, f64) {\n"
        "    // unsupported: Derivative(f(x), x)\n"
        "    // unsupported: zoo\n"
        "    let out1 = Derivative(f(x), x);\n"
        "    let out2 = zoo;\n"
        "    (out1, out2)\n"
        "}\n"
    )
    assert source == expected


def test_global_vars_rust():
    x, y, z, t = symbols("x y z t")
    result = codegen(('f', x*y), "Rust", header=False, empty=False,
                     global_vars=(y,))
    source = result[0][1]
    expected = (
        "fn f(x: f64) -> f64 {\n"
        "    let out1 = x*y;\n"
        "    out1\n"
        "}\n"
        )
    assert source == expected

    result = codegen(('f', x*y+z), "Rust", header=False, empty=False,
                     argument_sequence=(x, y), global_vars=(z, t))
    source = result[0][1]
    expected = (
        "fn f(x: f64, y: f64) -> f64 {\n"
        "    let out1 = x*y + z;\n"
        "    out1\n"
        "}\n"
    )
    assert source == expected
