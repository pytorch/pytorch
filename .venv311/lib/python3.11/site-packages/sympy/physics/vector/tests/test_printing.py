# -*- coding: utf-8 -*-

from sympy.core.function import Function
from sympy.core.symbol import symbols
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (asin, cos, sin)
from sympy.physics.vector import ReferenceFrame, dynamicsymbols, Dyadic
from sympy.physics.vector.printing import (VectorLatexPrinter, vpprint,
                                           vsprint, vsstrrepr, vlatex)


a, b, c = symbols('a, b, c')
alpha, omega, beta = dynamicsymbols('alpha, omega, beta')

A = ReferenceFrame('A')
N = ReferenceFrame('N')

v = a ** 2 * N.x + b * N.y + c * sin(alpha) * N.z
w = alpha * N.x + sin(omega) * N.y + alpha * beta * N.z
ww = alpha * N.x + asin(omega) * N.y - alpha.diff() * beta * N.z
o = a/b * N.x + (c+b)/a * N.y + c**2/b * N.z

y = a ** 2 * (N.x | N.y) + b * (N.y | N.y) + c * sin(alpha) * (N.z | N.y)
x = alpha * (N.x | N.x) + sin(omega) * (N.y | N.z) + alpha * beta * (N.z | N.x)
xx = N.x | (-N.y - N.z)
xx2 = N.x | (N.y + N.z)

def ascii_vpretty(expr):
    return vpprint(expr, use_unicode=False, wrap_line=False)


def unicode_vpretty(expr):
    return vpprint(expr, use_unicode=True, wrap_line=False)


def test_latex_printer():
    r = Function('r')('t')
    assert VectorLatexPrinter().doprint(r ** 2) == "r^{2}"
    r2 = Function('r^2')('t')
    assert VectorLatexPrinter().doprint(r2.diff()) == r'\dot{r^{2}}'
    ra = Function('r__a')('t')
    assert VectorLatexPrinter().doprint(ra.diff().diff()) == r'\ddot{r^{a}}'


def test_vector_pretty_print():

    # TODO : The unit vectors should print with subscripts but they just
    # print as `n_x` instead of making `x` a subscript with unicode.

    # TODO : The pretty print division does not print correctly here:
    # w = alpha * N.x + sin(omega) * N.y + alpha / beta * N.z

    expected = """\
 2                               \n\
a  n_x + b n_y + c*sin(alpha) n_z\
"""
    uexpected = """\
 2                           \n\
a  n_x + b n_y + c⋅sin(α) n_z\
"""

    assert ascii_vpretty(v) == expected
    assert unicode_vpretty(v) == uexpected

    expected = 'alpha n_x + sin(omega) n_y + alpha*beta n_z'
    uexpected = 'α n_x + sin(ω) n_y + α⋅β n_z'

    assert ascii_vpretty(w) == expected
    assert unicode_vpretty(w) == uexpected

    expected = """\
                     2    \n\
a       b + c       c     \n\
- n_x + ----- n_y + -- n_z\n\
b         a         b     \
"""
    uexpected = """\
                     2    \n\
a       b + c       c     \n\
─ n_x + ───── n_y + ── n_z\n\
b         a         b     \
"""

    assert ascii_vpretty(o) == expected
    assert unicode_vpretty(o) == uexpected

    # https://github.com/sympy/sympy/issues/26731
    assert ascii_vpretty(-A.x) == '-a_x'
    assert unicode_vpretty(-A.x) == '-a_x'

    # https://github.com/sympy/sympy/issues/26799
    assert ascii_vpretty(0*A.x) == '0'
    assert unicode_vpretty(0*A.x) == '0'


def test_vector_latex():

    a, b, c, d, omega = symbols('a, b, c, d, omega')

    v = (a ** 2 + b / c) * A.x + sqrt(d) * A.y + cos(omega) * A.z

    assert vlatex(v) == (r'(a^{2} + \frac{b}{c})\mathbf{\hat{a}_x} + '
                         r'\sqrt{d}\mathbf{\hat{a}_y} + '
                         r'\cos{\left(\omega \right)}'
                         r'\mathbf{\hat{a}_z}')

    theta, omega, alpha, q = dynamicsymbols('theta, omega, alpha, q')

    v = theta * A.x + omega * omega * A.y + (q * alpha) * A.z

    assert vlatex(v) == (r'\theta\mathbf{\hat{a}_x} + '
                         r'\omega^{2}\mathbf{\hat{a}_y} + '
                         r'\alpha q\mathbf{\hat{a}_z}')

    phi1, phi2, phi3 = dynamicsymbols('phi1, phi2, phi3')
    theta1, theta2, theta3 = symbols('theta1, theta2, theta3')

    v = (sin(theta1) * A.x +
         cos(phi1) * cos(phi2) * A.y +
         cos(theta1 + phi3) * A.z)

    assert vlatex(v) == (r'\sin{\left(\theta_{1} \right)}'
                         r'\mathbf{\hat{a}_x} + \cos{'
                         r'\left(\phi_{1} \right)} \cos{'
                         r'\left(\phi_{2} \right)}\mathbf{\hat{a}_y} + '
                         r'\cos{\left(\theta_{1} + '
                         r'\phi_{3} \right)}\mathbf{\hat{a}_z}')

    N = ReferenceFrame('N')

    a, b, c, d, omega = symbols('a, b, c, d, omega')

    v = (a ** 2 + b / c) * N.x + sqrt(d) * N.y + cos(omega) * N.z

    expected = (r'(a^{2} + \frac{b}{c})\mathbf{\hat{n}_x} + '
                r'\sqrt{d}\mathbf{\hat{n}_y} + '
                r'\cos{\left(\omega \right)}'
                r'\mathbf{\hat{n}_z}')

    assert vlatex(v) == expected

    # Try custom unit vectors.

    N = ReferenceFrame('N', latexs=(r'\hat{i}', r'\hat{j}', r'\hat{k}'))

    v = (a ** 2 + b / c) * N.x + sqrt(d) * N.y + cos(omega) * N.z

    expected = (r'(a^{2} + \frac{b}{c})\hat{i} + '
                r'\sqrt{d}\hat{j} + '
                r'\cos{\left(\omega \right)}\hat{k}')
    assert vlatex(v) == expected

    expected = r'\alpha\mathbf{\hat{n}_x} + \operatorname{asin}{\left(\omega ' \
        r'\right)}\mathbf{\hat{n}_y} -  \beta \dot{\alpha}\mathbf{\hat{n}_z}'
    assert vlatex(ww) == expected

    expected = r'- \mathbf{\hat{n}_x}\otimes \mathbf{\hat{n}_y} - ' \
        r'\mathbf{\hat{n}_x}\otimes \mathbf{\hat{n}_z}'
    assert vlatex(xx) == expected

    expected = r'\mathbf{\hat{n}_x}\otimes \mathbf{\hat{n}_y} + ' \
        r'\mathbf{\hat{n}_x}\otimes \mathbf{\hat{n}_z}'
    assert vlatex(xx2) == expected


def test_vector_latex_arguments():
    assert vlatex(N.x * 3.0, full_prec=False) == r'3.0\mathbf{\hat{n}_x}'
    assert vlatex(N.x * 3.0, full_prec=True) == r'3.00000000000000\mathbf{\hat{n}_x}'


def test_vector_latex_with_functions():

    N = ReferenceFrame('N')

    omega, alpha = dynamicsymbols('omega, alpha')

    v = omega.diff() * N.x

    assert vlatex(v) == r'\dot{\omega}\mathbf{\hat{n}_x}'

    v = omega.diff() ** alpha * N.x

    assert vlatex(v) == (r'\dot{\omega}^{\alpha}'
                          r'\mathbf{\hat{n}_x}')


def test_dyadic_pretty_print():

    expected = """\
 2
a  n_x|n_y + b n_y|n_y + c*sin(alpha) n_z|n_y\
"""

    uexpected = """\
 2
a  n_x⊗n_y + b n_y⊗n_y + c⋅sin(α) n_z⊗n_y\
"""
    assert ascii_vpretty(y) == expected
    assert unicode_vpretty(y) == uexpected

    expected = 'alpha n_x|n_x + sin(omega) n_y|n_z + alpha*beta n_z|n_x'
    uexpected = 'α n_x⊗n_x + sin(ω) n_y⊗n_z + α⋅β n_z⊗n_x'
    assert ascii_vpretty(x) == expected
    assert unicode_vpretty(x) == uexpected

    assert ascii_vpretty(Dyadic([])) == '0'
    assert unicode_vpretty(Dyadic([])) == '0'

    assert ascii_vpretty(xx) == '- n_x|n_y - n_x|n_z'
    assert unicode_vpretty(xx) == '- n_x⊗n_y - n_x⊗n_z'

    assert ascii_vpretty(xx2) == 'n_x|n_y + n_x|n_z'
    assert unicode_vpretty(xx2) == 'n_x⊗n_y + n_x⊗n_z'


def test_dyadic_latex():

    expected = (r'a^{2}\mathbf{\hat{n}_x}\otimes \mathbf{\hat{n}_y} + '
                r'b\mathbf{\hat{n}_y}\otimes \mathbf{\hat{n}_y} + '
                r'c \sin{\left(\alpha \right)}'
                r'\mathbf{\hat{n}_z}\otimes \mathbf{\hat{n}_y}')

    assert vlatex(y) == expected

    expected = (r'\alpha\mathbf{\hat{n}_x}\otimes \mathbf{\hat{n}_x} + '
                r'\sin{\left(\omega \right)}\mathbf{\hat{n}_y}'
                r'\otimes \mathbf{\hat{n}_z} + '
                r'\alpha \beta\mathbf{\hat{n}_z}\otimes \mathbf{\hat{n}_x}')

    assert vlatex(x) == expected

    assert vlatex(Dyadic([])) == '0'


def test_dyadic_str():
    assert vsprint(Dyadic([])) == '0'
    assert vsprint(y) == 'a**2*(N.x|N.y) + b*(N.y|N.y) + c*sin(alpha)*(N.z|N.y)'
    assert vsprint(x) == 'alpha*(N.x|N.x) + sin(omega)*(N.y|N.z) + alpha*beta*(N.z|N.x)'
    assert vsprint(ww) == "alpha*N.x + asin(omega)*N.y - beta*alpha'*N.z"
    assert vsprint(xx) == '- (N.x|N.y) - (N.x|N.z)'
    assert vsprint(xx2) == '(N.x|N.y) + (N.x|N.z)'


def test_vlatex(): # vlatex is broken #12078
    from sympy.physics.vector import vlatex

    x = symbols('x')
    J = symbols('J')

    f = Function('f')
    g = Function('g')
    h = Function('h')

    expected = r'J \left(\frac{d}{d x} g{\left(x \right)} - \frac{d}{d x} h{\left(x \right)}\right)'

    expr = J*f(x).diff(x).subs(f(x), g(x)-h(x))

    assert vlatex(expr) == expected


def test_issue_13354():
    """
    Test for proper pretty printing of physics vectors with ADD
    instances in arguments.

    Test is exactly the one suggested in the original bug report by
    @moorepants.
    """

    a, b, c = symbols('a, b, c')
    A = ReferenceFrame('A')
    v = a * A.x + b * A.y + c * A.z
    w = b * A.x + c * A.y + a * A.z
    z = w + v

    expected = """(a + b) a_x + (b + c) a_y + (a + c) a_z"""

    assert ascii_vpretty(z) == expected


def test_vector_derivative_printing():
    # First order
    v = omega.diff() * N.x
    assert unicode_vpretty(v) == 'ω̇ n_x'
    assert ascii_vpretty(v) == "omega'(t) n_x"

    # Second order
    v = omega.diff().diff() * N.x

    assert vlatex(v) == r'\ddot{\omega}\mathbf{\hat{n}_x}'
    assert unicode_vpretty(v) == 'ω̈ n_x'
    assert ascii_vpretty(v) == "omega''(t) n_x"

    # Third order
    v = omega.diff().diff().diff() * N.x

    assert vlatex(v) == r'\dddot{\omega}\mathbf{\hat{n}_x}'
    assert unicode_vpretty(v) == 'ω⃛ n_x'
    assert ascii_vpretty(v) == "omega'''(t) n_x"

    # Fourth order
    v = omega.diff().diff().diff().diff() * N.x

    assert vlatex(v) == r'\ddddot{\omega}\mathbf{\hat{n}_x}'
    assert unicode_vpretty(v) == 'ω⃜ n_x'
    assert ascii_vpretty(v) == "omega''''(t) n_x"

    # Fifth order
    v = omega.diff().diff().diff().diff().diff() * N.x

    assert vlatex(v) == r'\frac{d^{5}}{d t^{5}} \omega\mathbf{\hat{n}_x}'
    expected = '''\
 5            \n\
d             \n\
---(omega) n_x\n\
  5           \n\
dt            \
'''
    uexpected = '''\
 5        \n\
d         \n\
───(ω) n_x\n\
  5       \n\
dt        \
'''
    assert unicode_vpretty(v) == uexpected
    assert ascii_vpretty(v) == expected


def test_vector_str_printing():
    assert vsprint(w) == 'alpha*N.x + sin(omega)*N.y + alpha*beta*N.z'
    assert vsprint(omega.diff() * N.x) == "omega'*N.x"
    assert vsstrrepr(w) == 'alpha*N.x + sin(omega)*N.y + alpha*beta*N.z'


def test_vector_str_arguments():
    assert vsprint(N.x * 3.0, full_prec=False) == '3.0*N.x'
    assert vsprint(N.x * 3.0, full_prec=True) == '3.00000000000000*N.x'


def test_issue_14041():
    import sympy.physics.mechanics as me

    A_frame = me.ReferenceFrame('A')
    thetad, phid = me.dynamicsymbols('theta, phi', 1)
    L = symbols('L')

    assert vlatex(L*(phid + thetad)**2*A_frame.x) == \
        r"L \left(\dot{\phi} + \dot{\theta}\right)^{2}\mathbf{\hat{a}_x}"
    assert vlatex((phid + thetad)**2*A_frame.x) == \
        r"\left(\dot{\phi} + \dot{\theta}\right)^{2}\mathbf{\hat{a}_x}"
    assert vlatex((phid*thetad)**a*A_frame.x) == \
        r"\left(\dot{\phi} \dot{\theta}\right)^{a}\mathbf{\hat{a}_x}"
