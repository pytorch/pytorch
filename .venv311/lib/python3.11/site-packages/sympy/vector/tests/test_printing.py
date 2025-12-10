# -*- coding: utf-8 -*-
from sympy.core.function import Function
from sympy.integrals.integrals import Integral
from sympy.printing.latex import latex
from sympy.printing.pretty import pretty as xpretty
from sympy.vector import CoordSys3D, Del, Vector, express
from sympy.abc import a, b, c
from sympy.testing.pytest import XFAIL


def pretty(expr):
    """ASCII pretty-printing"""
    return xpretty(expr, use_unicode=False, wrap_line=False)


def upretty(expr):
    """Unicode pretty-printing"""
    return xpretty(expr, use_unicode=True, wrap_line=False)


# Initialize the basic and tedious vector/dyadic expressions
# needed for testing.
# Some of the pretty forms shown denote how the expressions just
# above them should look with pretty printing.
N = CoordSys3D('N')
C = N.orient_new_axis('C', a, N.k)  # type: ignore
v = []
d = []
v.append(Vector.zero)
v.append(N.i)  # type: ignore
v.append(-N.i)  # type: ignore
v.append(N.i + N.j)  # type: ignore
v.append(a*N.i)  # type: ignore
v.append(a*N.i - b*N.j)  # type: ignore
v.append((a**2 + N.x)*N.i + N.k)  # type: ignore
v.append((a**2 + b)*N.i + 3*(C.y - c)*N.k)  # type: ignore
f = Function('f')
v.append(N.j - (Integral(f(b)) - C.x**2)*N.k)  # type: ignore
upretty_v_8 = """\
      ⎛   2   ⌠        ⎞    \n\
j_N + ⎜x_C  - ⎮ f(b) db⎟ k_N\n\
      ⎝       ⌡        ⎠    \
"""
pretty_v_8 = """\
j_N + /         /       \\\n\
      |   2    |        |\n\
      |x_C  -  | f(b) db|\n\
      |        |        |\n\
      \\       /         / \
"""

v.append(N.i + C.k)  # type: ignore
v.append(express(N.i, C))  # type: ignore
v.append((a**2 + b)*N.i + (Integral(f(b)))*N.k)  # type: ignore
upretty_v_11 = """\
⎛ 2    ⎞        ⎛⌠        ⎞    \n\
⎝a  + b⎠ i_N  + ⎜⎮ f(b) db⎟ k_N\n\
                ⎝⌡        ⎠    \
"""
pretty_v_11 = """\
/ 2    \\ + /  /       \\\n\
\\a  + b/ i_N| |        |\n\
           | | f(b) db|\n\
           | |        |\n\
           \\/         / \
"""

for x in v:
    d.append(x | N.k)  # type: ignore
s = 3*N.x**2*C.y  # type: ignore
upretty_s = """\
         2\n\
3⋅y_C⋅x_N \
"""
pretty_s = """\
         2\n\
3*y_C*x_N \
"""

# This is the pretty form for ((a**2 + b)*N.i + 3*(C.y - c)*N.k) | N.k
upretty_d_7 = """\
⎛ 2    ⎞                                     \n\
⎝a  + b⎠ (i_N|k_N)  + (3⋅y_C - 3⋅c) (k_N|k_N)\
"""
pretty_d_7 = """\
/ 2    \\ (i_N|k_N) + (3*y_C - 3*c) (k_N|k_N)\n\
\\a  + b/                                    \
"""


def test_str_printing():
    assert str(v[0]) == '0'
    assert str(v[1]) == 'N.i'
    assert str(v[2]) == '(-1)*N.i'
    assert str(v[3]) == 'N.i + N.j'
    assert str(v[8]) == 'N.j + (C.x**2 - Integral(f(b), b))*N.k'
    assert str(v[9]) == 'C.k + N.i'
    assert str(s) == '3*C.y*N.x**2'
    assert str(d[0]) == '0'
    assert str(d[1]) == '(N.i|N.k)'
    assert str(d[4]) == 'a*(N.i|N.k)'
    assert str(d[5]) == 'a*(N.i|N.k) + (-b)*(N.j|N.k)'
    assert str(d[8]) == ('(N.j|N.k) + (C.x**2 - ' +
                         'Integral(f(b), b))*(N.k|N.k)')


@XFAIL
def test_pretty_printing_ascii():
    assert pretty(v[0]) == '0'
    assert pretty(v[1]) == 'i_N'
    assert pretty(v[5]) == '(a) i_N + (-b) j_N'
    assert pretty(v[8]) == pretty_v_8
    assert pretty(v[2]) == '(-1) i_N'
    assert pretty(v[11]) == pretty_v_11
    assert pretty(s) == pretty_s
    assert pretty(d[0]) == '(0|0)'
    assert pretty(d[5]) == '(a) (i_N|k_N) + (-b) (j_N|k_N)'
    assert pretty(d[7]) == pretty_d_7
    assert pretty(d[10]) == '(cos(a)) (i_C|k_N) + (-sin(a)) (j_C|k_N)'


def test_pretty_print_unicode_v():
    assert upretty(v[0]) == '0'
    assert upretty(v[1]) == 'i_N'
    assert upretty(v[5]) == '(a) i_N + (-b) j_N'
    # Make sure the printing works in other objects
    assert upretty(v[5].args) == '((a) i_N, (-b) j_N)'
    assert upretty(v[8]) == upretty_v_8
    assert upretty(v[2]) == '(-1) i_N'
    assert upretty(v[11]) == upretty_v_11
    assert upretty(s) == upretty_s
    assert upretty(d[0]) == '(0|0)'
    assert upretty(d[5]) == '(a) (i_N|k_N) + (-b) (j_N|k_N)'
    assert upretty(d[7]) == upretty_d_7
    assert upretty(d[10]) == '(cos(a)) (i_C|k_N) + (-sin(a)) (j_C|k_N)'


def test_latex_printing():
    assert latex(v[0]) == '\\mathbf{\\hat{0}}'
    assert latex(v[1]) == '\\mathbf{\\hat{i}_{N}}'
    assert latex(v[2]) == '- \\mathbf{\\hat{i}_{N}}'
    assert latex(v[5]) == ('\\left(a\\right)\\mathbf{\\hat{i}_{N}} + ' +
                           '\\left(- b\\right)\\mathbf{\\hat{j}_{N}}')
    assert latex(v[6]) == ('\\left(\\mathbf{{x}_{N}} + a^{2}\\right)\\mathbf{\\hat{i}_' +
                          '{N}} + \\mathbf{\\hat{k}_{N}}')
    assert latex(v[8]) == ('\\mathbf{\\hat{j}_{N}} + \\left(\\mathbf{{x}_' +
                           '{C}}^{2} - \\int f{\\left(b \\right)}\\,' +
                           ' db\\right)\\mathbf{\\hat{k}_{N}}')
    assert latex(s) == '3 \\mathbf{{y}_{C}} \\mathbf{{x}_{N}}^{2}'
    assert latex(d[0]) == '(\\mathbf{\\hat{0}}|\\mathbf{\\hat{0}})'
    assert latex(d[4]) == ('\\left(a\\right)\\left(\\mathbf{\\hat{i}_{N}}{\\middle|}' +
                           '\\mathbf{\\hat{k}_{N}}\\right)')
    assert latex(d[9]) == ('\\left(\\mathbf{\\hat{k}_{C}}{\\middle|}' +
                           '\\mathbf{\\hat{k}_{N}}\\right) + \\left(' +
                           '\\mathbf{\\hat{i}_{N}}{\\middle|}\\mathbf{' +
                           '\\hat{k}_{N}}\\right)')
    assert latex(d[11]) == ('\\left(a^{2} + b\\right)\\left(\\mathbf{\\hat{i}_{N}}' +
                            '{\\middle|}\\mathbf{\\hat{k}_{N}}\\right) + ' +
                            '\\left(\\int f{\\left(b \\right)}\\, db\\right)\\left(' +
                            '\\mathbf{\\hat{k}_{N}}{\\middle|}\\mathbf{' +
                            '\\hat{k}_{N}}\\right)')

def test_issue_23058():
    from sympy import symbols, sin, cos, pi, UnevaluatedExpr

    delop = Del()
    CC_   = CoordSys3D("C")
    y     = CC_.y
    xhat  = CC_.i

    t = symbols("t")
    ten = symbols("10", positive=True)
    eps, mu = 4*pi*ten**(-11), ten**(-5)

    Bx = 2 * ten**(-4) * cos(ten**5 * t) * sin(ten**(-3) * y)
    vecB = Bx * xhat
    vecE = (1/eps) * Integral(delop.cross(vecB/mu).doit(), t)
    vecE = vecE.doit()

    vecB_str = """\
⎛     ⎛y_C⎞    ⎛  5  ⎞⎞    \n\
⎜2⋅sin⎜───⎟⋅cos⎝10 ⋅t⎠⎟ i_C\n\
⎜     ⎜  3⎟           ⎟    \n\
⎜     ⎝10 ⎠           ⎟    \n\
⎜─────────────────────⎟    \n\
⎜           4         ⎟    \n\
⎝         10          ⎠    \
"""
    vecE_str = """\
⎛   4    ⎛  5  ⎞    ⎛y_C⎞ ⎞    \n\
⎜-10 ⋅sin⎝10 ⋅t⎠⋅cos⎜───⎟ ⎟ k_C\n\
⎜                   ⎜  3⎟ ⎟    \n\
⎜                   ⎝10 ⎠ ⎟    \n\
⎜─────────────────────────⎟    \n\
⎝           2⋅π           ⎠    \
"""

    assert upretty(vecB) == vecB_str
    assert upretty(vecE) == vecE_str

    ten = UnevaluatedExpr(10)
    eps, mu = 4*pi*ten**(-11), ten**(-5)

    Bx = 2 * ten**(-4) * cos(ten**5 * t) * sin(ten**(-3) * y)
    vecB = Bx * xhat

    vecB_str = """\
⎛    -4    ⎛    5⎞    ⎛      -3⎞⎞     \n\
⎝2⋅10  ⋅cos⎝t⋅10 ⎠⋅sin⎝y_C⋅10  ⎠⎠ i_C \
"""
    assert upretty(vecB) == vecB_str

def test_custom_names():
    A = CoordSys3D('A', vector_names=['x', 'y', 'z'],
                   variable_names=['i', 'j', 'k'])
    assert A.i.__str__() == 'A.i'
    assert A.x.__str__() == 'A.x'
    assert A.i._pretty_form == 'i_A'
    assert A.x._pretty_form == 'x_A'
    assert A.i._latex_form == r'\mathbf{{i}_{A}}'
    assert A.x._latex_form == r"\mathbf{\hat{x}_{A}}"
