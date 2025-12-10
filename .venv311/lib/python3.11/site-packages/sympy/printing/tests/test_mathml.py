from sympy.calculus.accumulationbounds import AccumBounds
from sympy.concrete.summations import Sum
from sympy.core.basic import Basic
from sympy.core.containers import Tuple
from sympy.core.function import Derivative, Lambda, diff, Function
from sympy.core.numbers import (zoo, Float, Integer, I, oo, pi, E,
    Rational)
from sympy.core.relational import Lt, Ge, Ne, Eq
from sympy.core.singleton import S
from sympy.core.symbol import symbols, Symbol
from sympy.core.sympify import sympify
from sympy.functions.combinatorial.factorials import (factorial2,
    binomial, factorial)
from sympy.functions.combinatorial.numbers import (lucas, bell,
    catalan, euler, tribonacci, fibonacci, bernoulli, primenu, primeomega,
    totient, reduced_totient)
from sympy.functions.elementary.complexes import re, im, conjugate, Abs
from sympy.functions.elementary.exponential import exp, LambertW, log
from sympy.functions.elementary.hyperbolic import (tanh, acoth, atanh,
    coth, asinh, acsch, asech, acosh, csch, sinh, cosh, sech)
from sympy.functions.elementary.integers import ceiling, floor
from sympy.functions.elementary.miscellaneous import Max, Min
from sympy.functions.elementary.trigonometric import (csc, sec, tan,
    atan, sin, asec, cot, cos, acot, acsc, asin, acos)
from sympy.functions.special.delta_functions import Heaviside
from sympy.functions.special.elliptic_integrals import (elliptic_pi,
    elliptic_f, elliptic_k, elliptic_e)
from sympy.functions.special.error_functions import (fresnelc,
    fresnels, Ei, expint)
from sympy.functions.special.gamma_functions import (gamma, uppergamma,
    lowergamma)
from sympy.functions.special.mathieu_functions import (mathieusprime,
    mathieus, mathieucprime, mathieuc)
from sympy.functions.special.polynomials import (jacobi, chebyshevu,
    chebyshevt, hermite, assoc_legendre, gegenbauer, assoc_laguerre,
    legendre, laguerre)
from sympy.functions.special.singularity_functions import SingularityFunction
from sympy.functions.special.zeta_functions import (polylog, stieltjes,
    lerchphi, dirichlet_eta, zeta)
from sympy.integrals.integrals import Integral
from sympy.logic.boolalg import (Xor, Or, false, true, And, Equivalent,
    Implies, Not)
from sympy.matrices.dense import Matrix
from sympy.matrices.expressions.determinant import Determinant
from sympy.matrices.expressions.matexpr import MatrixSymbol
from sympy.physics.quantum import (ComplexSpace, FockSpace, hbar,
    HilbertSpace, Dagger)
from sympy.printing.mathml import (MathMLPresentationPrinter,
    MathMLPrinter, MathMLContentPrinter, mathml)
from sympy.series.limits import Limit
from sympy.sets.contains import Contains
from sympy.sets.fancysets import Range
from sympy.sets.sets import (Interval, Union, SymmetricDifference,
    Complement, FiniteSet, Intersection, ProductSet)
from sympy.stats.rv import RandomSymbol
from sympy.tensor.indexed import IndexedBase
from sympy.vector import (Divergence, CoordSys3D, Cross, Curl, Dot,
    Laplacian, Gradient)
from sympy.testing.pytest import raises, XFAIL

x, y, z, a, b, c, d, e, n = symbols('x:z a:e n')
mp = MathMLContentPrinter()
mpp = MathMLPresentationPrinter()


def test_mathml_printer():
    m = MathMLPrinter()
    assert m.doprint(1+x) == mp.doprint(1+x)


def test_content_printmethod():
    assert mp.doprint(1 + x) == '<apply><plus/><ci>x</ci><cn>1</cn></apply>'


def test_content_mathml_core():
    mml_1 = mp._print(1 + x)
    assert mml_1.nodeName == 'apply'
    nodes = mml_1.childNodes
    assert len(nodes) == 3
    assert nodes[0].nodeName == 'plus'
    assert nodes[0].hasChildNodes() is False
    assert nodes[0].nodeValue is None
    assert nodes[1].nodeName in ['cn', 'ci']
    if nodes[1].nodeName == 'cn':
        assert nodes[1].childNodes[0].nodeValue == '1'
        assert nodes[2].childNodes[0].nodeValue == 'x'
    else:
        assert nodes[1].childNodes[0].nodeValue == 'x'
        assert nodes[2].childNodes[0].nodeValue == '1'

    mml_2 = mp._print(x**2)
    assert mml_2.nodeName == 'apply'
    nodes = mml_2.childNodes
    assert nodes[1].childNodes[0].nodeValue == 'x'
    assert nodes[2].childNodes[0].nodeValue == '2'

    mml_3 = mp._print(2*x)
    assert mml_3.nodeName == 'apply'
    nodes = mml_3.childNodes
    assert nodes[0].nodeName == 'times'
    assert nodes[1].childNodes[0].nodeValue == '2'
    assert nodes[2].childNodes[0].nodeValue == 'x'

    mml = mp._print(Float(1.0, 2)*x)
    assert mml.nodeName == 'apply'
    nodes = mml.childNodes
    assert nodes[0].nodeName == 'times'
    assert nodes[1].childNodes[0].nodeValue == '1.0'
    assert nodes[2].childNodes[0].nodeValue == 'x'


def test_content_mathml_functions():
    mml_1 = mp._print(sin(x))
    assert mml_1.nodeName == 'apply'
    assert mml_1.childNodes[0].nodeName == 'sin'
    assert mml_1.childNodes[1].nodeName == 'ci'

    mml_2 = mp._print(diff(sin(x), x, evaluate=False))
    assert mml_2.nodeName == 'apply'
    assert mml_2.childNodes[0].nodeName == 'diff'
    assert mml_2.childNodes[1].nodeName == 'bvar'
    assert mml_2.childNodes[1].childNodes[
        0].nodeName == 'ci'  # below bvar there's <ci>x/ci>

    mml_3 = mp._print(diff(cos(x*y), x, evaluate=False))
    assert mml_3.nodeName == 'apply'
    assert mml_3.childNodes[0].nodeName == 'partialdiff'
    assert mml_3.childNodes[1].nodeName == 'bvar'
    assert mml_3.childNodes[1].childNodes[
        0].nodeName == 'ci'  # below bvar there's <ci>x/ci>

    mml_4 = mp._print(Lambda((x, y), x * y))
    assert mml_4.nodeName == 'lambda'
    assert mml_4.childNodes[0].nodeName == 'bvar'
    assert mml_4.childNodes[0].childNodes[
        0].nodeName == 'ci'  # below bvar there's <ci>x/ci>
    assert mml_4.childNodes[1].nodeName == 'bvar'
    assert mml_4.childNodes[1].childNodes[
        0].nodeName == 'ci'  # below bvar there's <ci>y/ci>
    assert mml_4.childNodes[2].nodeName == 'apply'


def test_content_mathml_limits():
    # XXX No unevaluated limits
    lim_fun = sin(x)/x
    mml_1 = mp._print(Limit(lim_fun, x, 0))
    assert mml_1.childNodes[0].nodeName == 'limit'
    assert mml_1.childNodes[1].nodeName == 'bvar'
    assert mml_1.childNodes[2].nodeName == 'lowlimit'
    assert mml_1.childNodes[3].toxml() == mp._print(lim_fun).toxml()


def test_content_mathml_integrals():
    integrand = x
    mml_1 = mp._print(Integral(integrand, (x, 0, 1)))
    assert mml_1.childNodes[0].nodeName == 'int'
    assert mml_1.childNodes[1].nodeName == 'bvar'
    assert mml_1.childNodes[2].nodeName == 'lowlimit'
    assert mml_1.childNodes[3].nodeName == 'uplimit'
    assert mml_1.childNodes[4].toxml() == mp._print(integrand).toxml()


def test_content_mathml_matrices():
    A = Matrix([1, 2, 3])
    B = Matrix([[0, 5, 4], [2, 3, 1], [9, 7, 9]])
    mll_1 = mp._print(A)
    assert mll_1.childNodes[0].nodeName == 'matrixrow'
    assert mll_1.childNodes[0].childNodes[0].nodeName == 'cn'
    assert mll_1.childNodes[0].childNodes[0].childNodes[0].nodeValue == '1'
    assert mll_1.childNodes[1].nodeName == 'matrixrow'
    assert mll_1.childNodes[1].childNodes[0].nodeName == 'cn'
    assert mll_1.childNodes[1].childNodes[0].childNodes[0].nodeValue == '2'
    assert mll_1.childNodes[2].nodeName == 'matrixrow'
    assert mll_1.childNodes[2].childNodes[0].nodeName == 'cn'
    assert mll_1.childNodes[2].childNodes[0].childNodes[0].nodeValue == '3'
    mll_2 = mp._print(B)
    assert mll_2.childNodes[0].nodeName == 'matrixrow'
    assert mll_2.childNodes[0].childNodes[0].nodeName == 'cn'
    assert mll_2.childNodes[0].childNodes[0].childNodes[0].nodeValue == '0'
    assert mll_2.childNodes[0].childNodes[1].nodeName == 'cn'
    assert mll_2.childNodes[0].childNodes[1].childNodes[0].nodeValue == '5'
    assert mll_2.childNodes[0].childNodes[2].nodeName == 'cn'
    assert mll_2.childNodes[0].childNodes[2].childNodes[0].nodeValue == '4'
    assert mll_2.childNodes[1].nodeName == 'matrixrow'
    assert mll_2.childNodes[1].childNodes[0].nodeName == 'cn'
    assert mll_2.childNodes[1].childNodes[0].childNodes[0].nodeValue == '2'
    assert mll_2.childNodes[1].childNodes[1].nodeName == 'cn'
    assert mll_2.childNodes[1].childNodes[1].childNodes[0].nodeValue == '3'
    assert mll_2.childNodes[1].childNodes[2].nodeName == 'cn'
    assert mll_2.childNodes[1].childNodes[2].childNodes[0].nodeValue == '1'
    assert mll_2.childNodes[2].nodeName == 'matrixrow'
    assert mll_2.childNodes[2].childNodes[0].nodeName == 'cn'
    assert mll_2.childNodes[2].childNodes[0].childNodes[0].nodeValue == '9'
    assert mll_2.childNodes[2].childNodes[1].nodeName == 'cn'
    assert mll_2.childNodes[2].childNodes[1].childNodes[0].nodeValue == '7'
    assert mll_2.childNodes[2].childNodes[2].nodeName == 'cn'
    assert mll_2.childNodes[2].childNodes[2].childNodes[0].nodeValue == '9'


def test_content_mathml_sums():
    summand = x
    mml_1 = mp._print(Sum(summand, (x, 1, 10)))
    assert mml_1.childNodes[0].nodeName == 'sum'
    assert mml_1.childNodes[1].nodeName == 'bvar'
    assert mml_1.childNodes[2].nodeName == 'lowlimit'
    assert mml_1.childNodes[3].nodeName == 'uplimit'
    assert mml_1.childNodes[4].toxml() == mp._print(summand).toxml()


def test_content_mathml_tuples():
    mml_1 = mp._print([2])
    assert mml_1.nodeName == 'list'
    assert mml_1.childNodes[0].nodeName == 'cn'
    assert len(mml_1.childNodes) == 1

    mml_2 = mp._print([2, Integer(1)])
    assert mml_2.nodeName == 'list'
    assert mml_2.childNodes[0].nodeName == 'cn'
    assert mml_2.childNodes[1].nodeName == 'cn'
    assert len(mml_2.childNodes) == 2


def test_content_mathml_add():
    mml = mp._print(x**5 - x**4 + x)
    assert mml.childNodes[0].nodeName == 'plus'
    assert mml.childNodes[1].childNodes[0].nodeName == 'minus'
    assert mml.childNodes[1].childNodes[1].nodeName == 'apply'


def test_content_mathml_Rational():
    mml_1 = mp._print(Rational(1, 1))
    """should just return a number"""
    assert mml_1.nodeName == 'cn'

    mml_2 = mp._print(Rational(2, 5))
    assert mml_2.childNodes[0].nodeName == 'divide'


def test_content_mathml_constants():
    mml = mp._print(I)
    assert mml.nodeName == 'imaginaryi'

    mml = mp._print(E)
    assert mml.nodeName == 'exponentiale'

    mml = mp._print(oo)
    assert mml.nodeName == 'infinity'

    mml = mp._print(pi)
    assert mml.nodeName == 'pi'

    assert mathml(hbar) == '<hbar/>'
    assert mathml(S.TribonacciConstant) == '<tribonacciconstant/>'
    assert mathml(S.GoldenRatio) == '<cn>&#966;</cn>'
    mml = mathml(S.EulerGamma)
    assert mml == '<eulergamma/>'

    mml = mathml(S.EmptySet)
    assert mml == '<emptyset/>'

    mml = mathml(S.true)
    assert mml == '<true/>'

    mml = mathml(S.false)
    assert mml == '<false/>'

    mml = mathml(S.NaN)
    assert mml == '<notanumber/>'


def test_content_mathml_trig():
    mml = mp._print(sin(x))
    assert mml.childNodes[0].nodeName == 'sin'

    mml = mp._print(cos(x))
    assert mml.childNodes[0].nodeName == 'cos'

    mml = mp._print(tan(x))
    assert mml.childNodes[0].nodeName == 'tan'

    mml = mp._print(cot(x))
    assert mml.childNodes[0].nodeName == 'cot'

    mml = mp._print(csc(x))
    assert mml.childNodes[0].nodeName == 'csc'

    mml = mp._print(sec(x))
    assert mml.childNodes[0].nodeName == 'sec'

    mml = mp._print(asin(x))
    assert mml.childNodes[0].nodeName == 'arcsin'

    mml = mp._print(acos(x))
    assert mml.childNodes[0].nodeName == 'arccos'

    mml = mp._print(atan(x))
    assert mml.childNodes[0].nodeName == 'arctan'

    mml = mp._print(acot(x))
    assert mml.childNodes[0].nodeName == 'arccot'

    mml = mp._print(acsc(x))
    assert mml.childNodes[0].nodeName == 'arccsc'

    mml = mp._print(asec(x))
    assert mml.childNodes[0].nodeName == 'arcsec'

    mml = mp._print(sinh(x))
    assert mml.childNodes[0].nodeName == 'sinh'

    mml = mp._print(cosh(x))
    assert mml.childNodes[0].nodeName == 'cosh'

    mml = mp._print(tanh(x))
    assert mml.childNodes[0].nodeName == 'tanh'

    mml = mp._print(coth(x))
    assert mml.childNodes[0].nodeName == 'coth'

    mml = mp._print(csch(x))
    assert mml.childNodes[0].nodeName == 'csch'

    mml = mp._print(sech(x))
    assert mml.childNodes[0].nodeName == 'sech'

    mml = mp._print(asinh(x))
    assert mml.childNodes[0].nodeName == 'arcsinh'

    mml = mp._print(atanh(x))
    assert mml.childNodes[0].nodeName == 'arctanh'

    mml = mp._print(acosh(x))
    assert mml.childNodes[0].nodeName == 'arccosh'

    mml = mp._print(acoth(x))
    assert mml.childNodes[0].nodeName == 'arccoth'

    mml = mp._print(acsch(x))
    assert mml.childNodes[0].nodeName == 'arccsch'

    mml = mp._print(asech(x))
    assert mml.childNodes[0].nodeName == 'arcsech'


def test_content_mathml_relational():
    mml_1 = mp._print(Eq(x, 1))
    assert mml_1.nodeName == 'apply'
    assert mml_1.childNodes[0].nodeName == 'eq'
    assert mml_1.childNodes[1].nodeName == 'ci'
    assert mml_1.childNodes[1].childNodes[0].nodeValue == 'x'
    assert mml_1.childNodes[2].nodeName == 'cn'
    assert mml_1.childNodes[2].childNodes[0].nodeValue == '1'

    mml_2 = mp._print(Ne(1, x))
    assert mml_2.nodeName == 'apply'
    assert mml_2.childNodes[0].nodeName == 'neq'
    assert mml_2.childNodes[1].nodeName == 'cn'
    assert mml_2.childNodes[1].childNodes[0].nodeValue == '1'
    assert mml_2.childNodes[2].nodeName == 'ci'
    assert mml_2.childNodes[2].childNodes[0].nodeValue == 'x'

    mml_3 = mp._print(Ge(1, x))
    assert mml_3.nodeName == 'apply'
    assert mml_3.childNodes[0].nodeName == 'geq'
    assert mml_3.childNodes[1].nodeName == 'cn'
    assert mml_3.childNodes[1].childNodes[0].nodeValue == '1'
    assert mml_3.childNodes[2].nodeName == 'ci'
    assert mml_3.childNodes[2].childNodes[0].nodeValue == 'x'

    mml_4 = mp._print(Lt(1, x))
    assert mml_4.nodeName == 'apply'
    assert mml_4.childNodes[0].nodeName == 'lt'
    assert mml_4.childNodes[1].nodeName == 'cn'
    assert mml_4.childNodes[1].childNodes[0].nodeValue == '1'
    assert mml_4.childNodes[2].nodeName == 'ci'
    assert mml_4.childNodes[2].childNodes[0].nodeValue == 'x'


def test_content_symbol():
    mml = mp._print(x)
    assert mml.nodeName == 'ci'
    assert mml.childNodes[0].nodeValue == 'x'
    del mml

    mml = mp._print(Symbol("x^2"))
    assert mml.nodeName == 'ci'
    assert mml.childNodes[0].nodeName == 'mml:msup'
    assert mml.childNodes[0].childNodes[0].nodeName == 'mml:mi'
    assert mml.childNodes[0].childNodes[0].childNodes[0].nodeValue == 'x'
    assert mml.childNodes[0].childNodes[1].nodeName == 'mml:mi'
    assert mml.childNodes[0].childNodes[1].childNodes[0].nodeValue == '2'
    del mml

    mml = mp._print(Symbol("x__2"))
    assert mml.nodeName == 'ci'
    assert mml.childNodes[0].nodeName == 'mml:msup'
    assert mml.childNodes[0].childNodes[0].nodeName == 'mml:mi'
    assert mml.childNodes[0].childNodes[0].childNodes[0].nodeValue == 'x'
    assert mml.childNodes[0].childNodes[1].nodeName == 'mml:mi'
    assert mml.childNodes[0].childNodes[1].childNodes[0].nodeValue == '2'
    del mml

    mml = mp._print(Symbol("x_2"))
    assert mml.nodeName == 'ci'
    assert mml.childNodes[0].nodeName == 'mml:msub'
    assert mml.childNodes[0].childNodes[0].nodeName == 'mml:mi'
    assert mml.childNodes[0].childNodes[0].childNodes[0].nodeValue == 'x'
    assert mml.childNodes[0].childNodes[1].nodeName == 'mml:mi'
    assert mml.childNodes[0].childNodes[1].childNodes[0].nodeValue == '2'
    del mml

    mml = mp._print(Symbol("x^3_2"))
    assert mml.nodeName == 'ci'
    assert mml.childNodes[0].nodeName == 'mml:msubsup'
    assert mml.childNodes[0].childNodes[0].nodeName == 'mml:mi'
    assert mml.childNodes[0].childNodes[0].childNodes[0].nodeValue == 'x'
    assert mml.childNodes[0].childNodes[1].nodeName == 'mml:mi'
    assert mml.childNodes[0].childNodes[1].childNodes[0].nodeValue == '2'
    assert mml.childNodes[0].childNodes[2].nodeName == 'mml:mi'
    assert mml.childNodes[0].childNodes[2].childNodes[0].nodeValue == '3'
    del mml

    mml = mp._print(Symbol("x__3_2"))
    assert mml.nodeName == 'ci'
    assert mml.childNodes[0].nodeName == 'mml:msubsup'
    assert mml.childNodes[0].childNodes[0].nodeName == 'mml:mi'
    assert mml.childNodes[0].childNodes[0].childNodes[0].nodeValue == 'x'
    assert mml.childNodes[0].childNodes[1].nodeName == 'mml:mi'
    assert mml.childNodes[0].childNodes[1].childNodes[0].nodeValue == '2'
    assert mml.childNodes[0].childNodes[2].nodeName == 'mml:mi'
    assert mml.childNodes[0].childNodes[2].childNodes[0].nodeValue == '3'
    del mml

    mml = mp._print(Symbol("x_2_a"))
    assert mml.nodeName == 'ci'
    assert mml.childNodes[0].nodeName == 'mml:msub'
    assert mml.childNodes[0].childNodes[0].nodeName == 'mml:mi'
    assert mml.childNodes[0].childNodes[0].childNodes[0].nodeValue == 'x'
    assert mml.childNodes[0].childNodes[1].nodeName == 'mml:mrow'
    assert mml.childNodes[0].childNodes[1].childNodes[0].nodeName == 'mml:mi'
    assert mml.childNodes[0].childNodes[1].childNodes[0].childNodes[
        0].nodeValue == '2'
    assert mml.childNodes[0].childNodes[1].childNodes[1].nodeName == 'mml:mo'
    assert mml.childNodes[0].childNodes[1].childNodes[1].childNodes[
        0].nodeValue == ' '
    assert mml.childNodes[0].childNodes[1].childNodes[2].nodeName == 'mml:mi'
    assert mml.childNodes[0].childNodes[1].childNodes[2].childNodes[
        0].nodeValue == 'a'
    del mml

    mml = mp._print(Symbol("x^2^a"))
    assert mml.nodeName == 'ci'
    assert mml.childNodes[0].nodeName == 'mml:msup'
    assert mml.childNodes[0].childNodes[0].nodeName == 'mml:mi'
    assert mml.childNodes[0].childNodes[0].childNodes[0].nodeValue == 'x'
    assert mml.childNodes[0].childNodes[1].nodeName == 'mml:mrow'
    assert mml.childNodes[0].childNodes[1].childNodes[0].nodeName == 'mml:mi'
    assert mml.childNodes[0].childNodes[1].childNodes[0].childNodes[
        0].nodeValue == '2'
    assert mml.childNodes[0].childNodes[1].childNodes[1].nodeName == 'mml:mo'
    assert mml.childNodes[0].childNodes[1].childNodes[1].childNodes[
        0].nodeValue == ' '
    assert mml.childNodes[0].childNodes[1].childNodes[2].nodeName == 'mml:mi'
    assert mml.childNodes[0].childNodes[1].childNodes[2].childNodes[
        0].nodeValue == 'a'
    del mml

    mml = mp._print(Symbol("x__2__a"))
    assert mml.nodeName == 'ci'
    assert mml.childNodes[0].nodeName == 'mml:msup'
    assert mml.childNodes[0].childNodes[0].nodeName == 'mml:mi'
    assert mml.childNodes[0].childNodes[0].childNodes[0].nodeValue == 'x'
    assert mml.childNodes[0].childNodes[1].nodeName == 'mml:mrow'
    assert mml.childNodes[0].childNodes[1].childNodes[0].nodeName == 'mml:mi'
    assert mml.childNodes[0].childNodes[1].childNodes[0].childNodes[
        0].nodeValue == '2'
    assert mml.childNodes[0].childNodes[1].childNodes[1].nodeName == 'mml:mo'
    assert mml.childNodes[0].childNodes[1].childNodes[1].childNodes[
        0].nodeValue == ' '
    assert mml.childNodes[0].childNodes[1].childNodes[2].nodeName == 'mml:mi'
    assert mml.childNodes[0].childNodes[1].childNodes[2].childNodes[
        0].nodeValue == 'a'
    del mml


def test_content_mathml_greek():
    mml = mp._print(Symbol('alpha'))
    assert mml.nodeName == 'ci'
    assert mml.childNodes[0].nodeValue == '\N{GREEK SMALL LETTER ALPHA}'

    assert mp.doprint(Symbol('alpha')) == '<ci>&#945;</ci>'
    assert mp.doprint(Symbol('beta')) == '<ci>&#946;</ci>'
    assert mp.doprint(Symbol('gamma')) == '<ci>&#947;</ci>'
    assert mp.doprint(Symbol('delta')) == '<ci>&#948;</ci>'
    assert mp.doprint(Symbol('epsilon')) == '<ci>&#949;</ci>'
    assert mp.doprint(Symbol('zeta')) == '<ci>&#950;</ci>'
    assert mp.doprint(Symbol('eta')) == '<ci>&#951;</ci>'
    assert mp.doprint(Symbol('theta')) == '<ci>&#952;</ci>'
    assert mp.doprint(Symbol('iota')) == '<ci>&#953;</ci>'
    assert mp.doprint(Symbol('kappa')) == '<ci>&#954;</ci>'
    assert mp.doprint(Symbol('lambda')) == '<ci>&#955;</ci>'
    assert mp.doprint(Symbol('mu')) == '<ci>&#956;</ci>'
    assert mp.doprint(Symbol('nu')) == '<ci>&#957;</ci>'
    assert mp.doprint(Symbol('xi')) == '<ci>&#958;</ci>'
    assert mp.doprint(Symbol('omicron')) == '<ci>&#959;</ci>'
    assert mp.doprint(Symbol('pi')) == '<ci>&#960;</ci>'
    assert mp.doprint(Symbol('rho')) == '<ci>&#961;</ci>'
    assert mp.doprint(Symbol('varsigma')) == '<ci>&#962;</ci>'
    assert mp.doprint(Symbol('sigma')) == '<ci>&#963;</ci>'
    assert mp.doprint(Symbol('tau')) == '<ci>&#964;</ci>'
    assert mp.doprint(Symbol('upsilon')) == '<ci>&#965;</ci>'
    assert mp.doprint(Symbol('phi')) == '<ci>&#966;</ci>'
    assert mp.doprint(Symbol('chi')) == '<ci>&#967;</ci>'
    assert mp.doprint(Symbol('psi')) == '<ci>&#968;</ci>'
    assert mp.doprint(Symbol('omega')) == '<ci>&#969;</ci>'

    assert mp.doprint(Symbol('Alpha')) == '<ci>&#913;</ci>'
    assert mp.doprint(Symbol('Beta')) == '<ci>&#914;</ci>'
    assert mp.doprint(Symbol('Gamma')) == '<ci>&#915;</ci>'
    assert mp.doprint(Symbol('Delta')) == '<ci>&#916;</ci>'
    assert mp.doprint(Symbol('Epsilon')) == '<ci>&#917;</ci>'
    assert mp.doprint(Symbol('Zeta')) == '<ci>&#918;</ci>'
    assert mp.doprint(Symbol('Eta')) == '<ci>&#919;</ci>'
    assert mp.doprint(Symbol('Theta')) == '<ci>&#920;</ci>'
    assert mp.doprint(Symbol('Iota')) == '<ci>&#921;</ci>'
    assert mp.doprint(Symbol('Kappa')) == '<ci>&#922;</ci>'
    assert mp.doprint(Symbol('Lambda')) == '<ci>&#923;</ci>'
    assert mp.doprint(Symbol('Mu')) == '<ci>&#924;</ci>'
    assert mp.doprint(Symbol('Nu')) == '<ci>&#925;</ci>'
    assert mp.doprint(Symbol('Xi')) == '<ci>&#926;</ci>'
    assert mp.doprint(Symbol('Omicron')) == '<ci>&#927;</ci>'
    assert mp.doprint(Symbol('Pi')) == '<ci>&#928;</ci>'
    assert mp.doprint(Symbol('Rho')) == '<ci>&#929;</ci>'
    assert mp.doprint(Symbol('Sigma')) == '<ci>&#931;</ci>'
    assert mp.doprint(Symbol('Tau')) == '<ci>&#932;</ci>'
    assert mp.doprint(Symbol('Upsilon')) == '<ci>&#933;</ci>'
    assert mp.doprint(Symbol('Phi')) == '<ci>&#934;</ci>'
    assert mp.doprint(Symbol('Chi')) == '<ci>&#935;</ci>'
    assert mp.doprint(Symbol('Psi')) == '<ci>&#936;</ci>'
    assert mp.doprint(Symbol('Omega')) == '<ci>&#937;</ci>'


def test_content_mathml_order():
    expr = x**3 + x**2*y + 3*x*y**3 + y**4

    mp = MathMLContentPrinter({'order': 'lex'})
    mml = mp._print(expr)

    assert mml.childNodes[1].childNodes[0].nodeName == 'power'
    assert mml.childNodes[1].childNodes[1].childNodes[0].data == 'x'
    assert mml.childNodes[1].childNodes[2].childNodes[0].data == '3'

    assert mml.childNodes[4].childNodes[0].nodeName == 'power'
    assert mml.childNodes[4].childNodes[1].childNodes[0].data == 'y'
    assert mml.childNodes[4].childNodes[2].childNodes[0].data == '4'

    mp = MathMLContentPrinter({'order': 'rev-lex'})
    mml = mp._print(expr)

    assert mml.childNodes[1].childNodes[0].nodeName == 'power'
    assert mml.childNodes[1].childNodes[1].childNodes[0].data == 'y'
    assert mml.childNodes[1].childNodes[2].childNodes[0].data == '4'

    assert mml.childNodes[4].childNodes[0].nodeName == 'power'
    assert mml.childNodes[4].childNodes[1].childNodes[0].data == 'x'
    assert mml.childNodes[4].childNodes[2].childNodes[0].data == '3'


def test_content_settings():
    raises(TypeError, lambda: mathml(x, method="garbage"))


def test_content_mathml_logic():
    assert mathml(And(x, y)) == '<apply><and/><ci>x</ci><ci>y</ci></apply>'
    assert mathml(Or(x, y)) == '<apply><or/><ci>x</ci><ci>y</ci></apply>'
    assert mathml(Xor(x, y)) == '<apply><xor/><ci>x</ci><ci>y</ci></apply>'
    assert mathml(Implies(x, y)) == '<apply><implies/><ci>x</ci><ci>y</ci></apply>'
    assert mathml(Not(x)) == '<apply><not/><ci>x</ci></apply>'


def test_content_finite_sets():
    assert mathml(FiniteSet(a)) == '<set><ci>a</ci></set>'
    assert mathml(FiniteSet(a, b)) == '<set><ci>a</ci><ci>b</ci></set>'
    assert mathml(FiniteSet(FiniteSet(a, b), c)) == \
        '<set><ci>c</ci><set><ci>a</ci><ci>b</ci></set></set>'

    A = FiniteSet(a)
    B = FiniteSet(b)
    C = FiniteSet(c)
    D = FiniteSet(d)

    U1 = Union(A, B, evaluate=False)
    U2 = Union(C, D, evaluate=False)
    I1 = Intersection(A, B, evaluate=False)
    I2 = Intersection(C, D, evaluate=False)
    C1 = Complement(A, B, evaluate=False)
    C2 = Complement(C, D, evaluate=False)
    # XXX ProductSet does not support evaluate keyword
    P1 = ProductSet(A, B)
    P2 = ProductSet(C, D)

    assert mathml(U1) == \
        '<apply><union/><set><ci>a</ci></set><set><ci>b</ci></set></apply>'
    assert mathml(I1) == \
        '<apply><intersect/><set><ci>a</ci></set><set><ci>b</ci></set>' \
        '</apply>'
    assert mathml(C1) == \
        '<apply><setdiff/><set><ci>a</ci></set><set><ci>b</ci></set></apply>'
    assert mathml(P1) == \
        '<apply><cartesianproduct/><set><ci>a</ci></set><set><ci>b</ci>' \
        '</set></apply>'

    assert mathml(Intersection(A, U2, evaluate=False)) == \
        '<apply><intersect/><set><ci>a</ci></set><apply><union/><set>' \
        '<ci>c</ci></set><set><ci>d</ci></set></apply></apply>'
    assert mathml(Intersection(U1, U2, evaluate=False)) == \
        '<apply><intersect/><apply><union/><set><ci>a</ci></set><set>' \
        '<ci>b</ci></set></apply><apply><union/><set><ci>c</ci></set>' \
        '<set><ci>d</ci></set></apply></apply>'

    # XXX Does the parenthesis appear correctly for these examples in mathjax?
    assert mathml(Intersection(C1, C2, evaluate=False)) == \
        '<apply><intersect/><apply><setdiff/><set><ci>a</ci></set><set>' \
        '<ci>b</ci></set></apply><apply><setdiff/><set><ci>c</ci></set>' \
        '<set><ci>d</ci></set></apply></apply>'
    assert mathml(Intersection(P1, P2, evaluate=False)) == \
        '<apply><intersect/><apply><cartesianproduct/><set><ci>a</ci></set>' \
        '<set><ci>b</ci></set></apply><apply><cartesianproduct/><set>' \
        '<ci>c</ci></set><set><ci>d</ci></set></apply></apply>'

    assert mathml(Union(A, I2, evaluate=False)) == \
        '<apply><union/><set><ci>a</ci></set><apply><intersect/><set>' \
        '<ci>c</ci></set><set><ci>d</ci></set></apply></apply>'
    assert mathml(Union(I1, I2, evaluate=False)) == \
        '<apply><union/><apply><intersect/><set><ci>a</ci></set><set>' \
        '<ci>b</ci></set></apply><apply><intersect/><set><ci>c</ci></set>' \
        '<set><ci>d</ci></set></apply></apply>'
    assert mathml(Union(C1, C2, evaluate=False)) == \
        '<apply><union/><apply><setdiff/><set><ci>a</ci></set><set>' \
        '<ci>b</ci></set></apply><apply><setdiff/><set><ci>c</ci></set>' \
        '<set><ci>d</ci></set></apply></apply>'
    assert mathml(Union(P1, P2, evaluate=False)) == \
        '<apply><union/><apply><cartesianproduct/><set><ci>a</ci></set>' \
        '<set><ci>b</ci></set></apply><apply><cartesianproduct/><set>' \
        '<ci>c</ci></set><set><ci>d</ci></set></apply></apply>'

    assert mathml(Complement(A, C2, evaluate=False)) == \
        '<apply><setdiff/><set><ci>a</ci></set><apply><setdiff/><set>' \
        '<ci>c</ci></set><set><ci>d</ci></set></apply></apply>'
    assert mathml(Complement(U1, U2, evaluate=False)) == \
        '<apply><setdiff/><apply><union/><set><ci>a</ci></set><set>' \
        '<ci>b</ci></set></apply><apply><union/><set><ci>c</ci></set>' \
        '<set><ci>d</ci></set></apply></apply>'
    assert mathml(Complement(I1, I2, evaluate=False)) == \
        '<apply><setdiff/><apply><intersect/><set><ci>a</ci></set><set>' \
        '<ci>b</ci></set></apply><apply><intersect/><set><ci>c</ci></set>' \
        '<set><ci>d</ci></set></apply></apply>'
    assert mathml(Complement(P1, P2, evaluate=False)) == \
        '<apply><setdiff/><apply><cartesianproduct/><set><ci>a</ci></set>' \
        '<set><ci>b</ci></set></apply><apply><cartesianproduct/><set>' \
        '<ci>c</ci></set><set><ci>d</ci></set></apply></apply>'

    assert mathml(ProductSet(A, P2)) == \
        '<apply><cartesianproduct/><set><ci>a</ci></set>' \
        '<apply><cartesianproduct/><set><ci>c</ci></set>' \
        '<set><ci>d</ci></set></apply></apply>'
    assert mathml(ProductSet(U1, U2)) == \
        '<apply><cartesianproduct/><apply><union/><set><ci>a</ci></set>' \
        '<set><ci>b</ci></set></apply><apply><union/><set><ci>c</ci></set>' \
        '<set><ci>d</ci></set></apply></apply>'
    assert mathml(ProductSet(I1, I2)) == \
        '<apply><cartesianproduct/><apply><intersect/><set><ci>a</ci></set>' \
        '<set><ci>b</ci></set></apply><apply><intersect/><set>' \
        '<ci>c</ci></set><set><ci>d</ci></set></apply></apply>'
    assert mathml(ProductSet(C1, C2)) == \
        '<apply><cartesianproduct/><apply><setdiff/><set><ci>a</ci></set>' \
        '<set><ci>b</ci></set></apply><apply><setdiff/><set>' \
        '<ci>c</ci></set><set><ci>d</ci></set></apply></apply>'


def test_presentation_printmethod():
    assert mpp.doprint(1 + x) == '<mrow><mi>x</mi><mo>+</mo><mn>1</mn></mrow>'
    assert mpp.doprint(x**2) == '<msup><mi>x</mi><mn>2</mn></msup>'
    assert mpp.doprint(x**-1) == '<mfrac><mn>1</mn><mi>x</mi></mfrac>'
    assert mpp.doprint(x**-2) == \
        '<mfrac><mn>1</mn><msup><mi>x</mi><mn>2</mn></msup></mfrac>'
    assert mpp.doprint(2*x) == \
        '<mrow><mn>2</mn><mo>&InvisibleTimes;</mo><mi>x</mi></mrow>'


def test_presentation_mathml_core():
    mml_1 = mpp._print(1 + x)
    assert mml_1.nodeName == 'mrow'
    nodes = mml_1.childNodes
    assert len(nodes) == 3
    assert nodes[0].nodeName in ['mi', 'mn']
    assert nodes[1].nodeName == 'mo'
    if nodes[0].nodeName == 'mn':
        assert nodes[0].childNodes[0].nodeValue == '1'
        assert nodes[2].childNodes[0].nodeValue == 'x'
    else:
        assert nodes[0].childNodes[0].nodeValue == 'x'
        assert nodes[2].childNodes[0].nodeValue == '1'

    mml_2 = mpp._print(x**2)
    assert mml_2.nodeName == 'msup'
    nodes = mml_2.childNodes
    assert nodes[0].childNodes[0].nodeValue == 'x'
    assert nodes[1].childNodes[0].nodeValue == '2'

    mml_3 = mpp._print(2*x)
    assert mml_3.nodeName == 'mrow'
    nodes = mml_3.childNodes
    assert nodes[0].childNodes[0].nodeValue == '2'
    assert nodes[1].childNodes[0].nodeValue == '&InvisibleTimes;'
    assert nodes[2].childNodes[0].nodeValue == 'x'

    mml = mpp._print(Float(1.0, 2)*x)
    assert mml.nodeName == 'mrow'
    nodes = mml.childNodes
    assert nodes[0].childNodes[0].nodeValue == '1.0'
    assert nodes[1].childNodes[0].nodeValue == '&InvisibleTimes;'
    assert nodes[2].childNodes[0].nodeValue == 'x'


def test_presentation_mathml_functions():
    mml_1 = mpp._print(sin(x))
    assert mml_1.childNodes[0].childNodes[0
        ].nodeValue == 'sin'
    assert mml_1.childNodes[1].childNodes[1
        ].childNodes[0].nodeValue == 'x'

    mml_2 = mpp._print(diff(sin(x), x, evaluate=False))
    assert mml_2.nodeName == 'mrow'
    assert mml_2.childNodes[0].childNodes[0
        ].childNodes[0].childNodes[0].nodeValue == '&dd;'
    assert mml_2.childNodes[1].childNodes[1
        ].nodeName == 'mrow'
    assert mml_2.childNodes[0].childNodes[1
        ].childNodes[0].childNodes[0].nodeValue == '&dd;'

    mml_3 = mpp._print(diff(cos(x*y), x, evaluate=False))
    assert mml_3.childNodes[0].nodeName == 'mfrac'
    assert mml_3.childNodes[0].childNodes[0
        ].childNodes[0].childNodes[0].nodeValue == '&#x2202;'
    assert mml_3.childNodes[1].childNodes[0
        ].childNodes[0].nodeValue == 'cos'


def test_print_derivative():
    f = Function('f')
    d = Derivative(f(x, y, z), x, z, x, z, z, y)
    assert mathml(d) == \
        '<apply><partialdiff/><bvar><ci>y</ci><ci>z</ci><degree><cn>2</cn></degree><ci>x</ci><ci>z</ci><ci>x</ci></bvar><apply><f/><ci>x</ci><ci>y</ci><ci>z</ci></apply></apply>'
    assert mathml(d, printer='presentation') == \
        '<mrow><mfrac><mrow><msup><mo>&#x2202;</mo><mn>6</mn></msup></mrow><mrow><mo>&#x2202;</mo><mi>y</mi><msup><mo>&#x2202;</mo><mn>2</mn></msup><mi>z</mi><mo>&#x2202;</mo><mi>x</mi><mo>&#x2202;</mo><mi>z</mi><mo>&#x2202;</mo><mi>x</mi></mrow></mfrac><mrow><mi>f</mi><mrow><mo>(</mo><mi>x</mi><mo>,</mo><mi>y</mi><mo>,</mo><mi>z</mi><mo>)</mo></mrow></mrow></mrow>'


def test_presentation_mathml_limits():
    lim_fun = sin(x)/x
    mml_1 = mpp._print(Limit(lim_fun, x, 0))
    assert mml_1.childNodes[0].nodeName == 'munder'
    assert mml_1.childNodes[0].childNodes[0
        ].childNodes[0].nodeValue == 'lim'
    assert mml_1.childNodes[0].childNodes[1
        ].childNodes[0].childNodes[0
        ].nodeValue == 'x'
    assert mml_1.childNodes[0].childNodes[1
        ].childNodes[1].childNodes[0
        ].nodeValue == '&#x2192;'
    assert mml_1.childNodes[0].childNodes[1
        ].childNodes[2].childNodes[0
        ].nodeValue == '0'


def test_presentation_mathml_integrals():
    assert mpp.doprint(Integral(x, (x, 0, 1))) == \
        '<mrow><msubsup><mo>&#x222B;</mo><mn>0</mn><mn>1</mn></msubsup>'\
        '<mi>x</mi><mo>&dd;</mo><mi>x</mi></mrow>'
    assert mpp.doprint(Integral(log(x), x)) == \
        '<mrow><mo>&#x222B;</mo><mrow><mi>log</mi><mrow><mo>(</mo><mi>x</mi>' \
        '<mo>)</mo></mrow></mrow><mo>&dd;</mo><mi>x</mi></mrow>'
    assert mpp.doprint(Integral(x*y, x, y)) == \
        '<mrow><mo>&#x222C;</mo><mrow><mi>x</mi><mo>&InvisibleTimes;</mo>'\
        '<mi>y</mi></mrow><mo>&dd;</mo><mi>y</mi><mo>&dd;</mo><mi>x</mi></mrow>'
    z, w = symbols('z w')
    assert mpp.doprint(Integral(x*y*z, x, y, z)) == \
        '<mrow><mo>&#x222D;</mo><mrow><mi>x</mi><mo>&InvisibleTimes;</mo>'\
        '<mi>y</mi><mo>&InvisibleTimes;</mo><mi>z</mi></mrow><mo>&dd;</mo>'\
        '<mi>z</mi><mo>&dd;</mo><mi>y</mi><mo>&dd;</mo><mi>x</mi></mrow>'
    assert mpp.doprint(Integral(x*y*z*w, x, y, z, w)) == \
        '<mrow><mo>&#x222B;</mo><mo>&#x222B;</mo><mo>&#x222B;</mo>'\
        '<mo>&#x222B;</mo><mrow><mi>w</mi><mo>&InvisibleTimes;</mo>'\
        '<mi>x</mi><mo>&InvisibleTimes;</mo><mi>y</mi>'\
        '<mo>&InvisibleTimes;</mo><mi>z</mi></mrow><mo>&dd;</mo><mi>w</mi>'\
        '<mo>&dd;</mo><mi>z</mi><mo>&dd;</mo><mi>y</mi><mo>&dd;</mo><mi>x</mi></mrow>'
    assert mpp.doprint(Integral(x, x, y, (z, 0, 1))) == \
        '<mrow><msubsup><mo>&#x222B;</mo><mn>0</mn><mn>1</mn></msubsup>'\
        '<mo>&#x222B;</mo><mo>&#x222B;</mo><mi>x</mi><mo>&dd;</mo><mi>z</mi>'\
        '<mo>&dd;</mo><mi>y</mi><mo>&dd;</mo><mi>x</mi></mrow>'
    assert mpp.doprint(Integral(x, (x, 0))) == \
        '<mrow><msup><mo>&#x222B;</mo><mn>0</mn></msup><mi>x</mi><mo>&dd;</mo>'\
        '<mi>x</mi></mrow>'


def test_presentation_mathml_matrices():
    A = Matrix([1, 2, 3])
    B = Matrix([[0, 5, 4], [2, 3, 1], [9, 7, 9]])
    mll_1 = mpp._print(A)
    assert mll_1.childNodes[1].nodeName == 'mtable'
    assert mll_1.childNodes[1].childNodes[0].nodeName == 'mtr'
    assert len(mll_1.childNodes[1].childNodes) == 3
    assert mll_1.childNodes[1].childNodes[0].childNodes[0].nodeName == 'mtd'
    assert len(mll_1.childNodes[1].childNodes[0].childNodes) == 1
    assert mll_1.childNodes[1].childNodes[0].childNodes[0
        ].childNodes[0].childNodes[0].nodeValue == '1'
    assert mll_1.childNodes[1].childNodes[1].childNodes[0
        ].childNodes[0].childNodes[0].nodeValue == '2'
    assert mll_1.childNodes[1].childNodes[2].childNodes[0
        ].childNodes[0].childNodes[0].nodeValue == '3'
    mll_2 = mpp._print(B)
    assert mll_2.childNodes[1].nodeName == 'mtable'
    assert mll_2.childNodes[1].childNodes[0].nodeName == 'mtr'
    assert len(mll_2.childNodes[1].childNodes) == 3
    assert mll_2.childNodes[1].childNodes[0].childNodes[0].nodeName == 'mtd'
    assert len(mll_2.childNodes[1].childNodes[0].childNodes) == 3
    assert mll_2.childNodes[1].childNodes[0].childNodes[0
        ].childNodes[0].childNodes[0].nodeValue == '0'
    assert mll_2.childNodes[1].childNodes[0].childNodes[1
        ].childNodes[0].childNodes[0].nodeValue == '5'
    assert mll_2.childNodes[1].childNodes[0].childNodes[2
        ].childNodes[0].childNodes[0].nodeValue == '4'
    assert mll_2.childNodes[1].childNodes[1].childNodes[0
        ].childNodes[0].childNodes[0].nodeValue == '2'
    assert mll_2.childNodes[1].childNodes[1].childNodes[1
        ].childNodes[0].childNodes[0].nodeValue == '3'
    assert mll_2.childNodes[1].childNodes[1].childNodes[2
        ].childNodes[0].childNodes[0].nodeValue == '1'
    assert mll_2.childNodes[1].childNodes[2].childNodes[0
        ].childNodes[0].childNodes[0].nodeValue == '9'
    assert mll_2.childNodes[1].childNodes[2].childNodes[1
        ].childNodes[0].childNodes[0].nodeValue == '7'
    assert mll_2.childNodes[1].childNodes[2].childNodes[2
        ].childNodes[0].childNodes[0].nodeValue == '9'


def test_presentation_mathml_sums():
    mml_1 = mpp._print(Sum(x, (x, 1, 10)))
    assert mml_1.childNodes[0].nodeName == 'munderover'
    assert len(mml_1.childNodes[0].childNodes) == 3
    assert mml_1.childNodes[0].childNodes[0].childNodes[0
        ].nodeValue == '&#x2211;'
    assert len(mml_1.childNodes[0].childNodes[1].childNodes) == 3
    assert mml_1.childNodes[0].childNodes[2].childNodes[0
        ].nodeValue == '10'
    assert mml_1.childNodes[1].childNodes[0].nodeValue == 'x'

    assert mpp.doprint(Sum(x, (x, 1, 10))) == \
        '<mrow><munderover><mo>&#x2211;</mo><mrow><mi>x</mi><mo>=</mo><mn>1</mn></mrow><mn>10</mn></munderover><mi>x</mi></mrow>'
    assert mpp.doprint(Sum(x + y, (x, 1, 10))) == \
        '<mrow><munderover><mo>&#x2211;</mo><mrow><mi>x</mi><mo>=</mo><mn>1</mn></mrow><mn>10</mn></munderover><mrow><mo>(</mo><mrow><mi>x</mi><mo>+</mo><mi>y</mi></mrow><mo>)</mo></mrow></mrow>'


def test_presentation_mathml_add():
    mml = mpp._print(x**5 - x**4 + x)
    assert len(mml.childNodes) == 5
    assert mml.childNodes[0].childNodes[0].childNodes[0
        ].nodeValue == 'x'
    assert mml.childNodes[0].childNodes[1].childNodes[0
        ].nodeValue == '5'
    assert mml.childNodes[1].childNodes[0].nodeValue == '-'
    assert mml.childNodes[2].childNodes[0].childNodes[0
        ].nodeValue == 'x'
    assert mml.childNodes[2].childNodes[1].childNodes[0
        ].nodeValue == '4'
    assert mml.childNodes[3].childNodes[0].nodeValue == '+'
    assert mml.childNodes[4].childNodes[0].nodeValue == 'x'


def test_presentation_mathml_Rational():
    mml_1 = mpp._print(Rational(1, 1))
    assert mml_1.nodeName == 'mn'

    mml_2 = mpp._print(Rational(2, 5))
    assert mml_2.nodeName == 'mfrac'
    assert mml_2.childNodes[0].childNodes[0].nodeValue == '2'
    assert mml_2.childNodes[1].childNodes[0].nodeValue == '5'


def test_presentation_mathml_constants():
    mml = mpp._print(I)
    assert mml.childNodes[0].nodeValue == '&ImaginaryI;'

    mml = mpp._print(E)
    assert mml.childNodes[0].nodeValue == '&ExponentialE;'

    mml = mpp._print(oo)
    assert mml.childNodes[0].nodeValue == '&#x221E;'

    mml = mpp._print(pi)
    assert mml.childNodes[0].nodeValue == '&pi;'

    assert mathml(hbar, printer='presentation') == '<mi>&#x210F;</mi>'
    assert mathml(S.TribonacciConstant, printer='presentation'
        ) == '<mi>TribonacciConstant</mi>'
    assert mathml(S.EulerGamma, printer='presentation'
        ) == '<mi>&#x3B3;</mi>'
    assert mathml(S.GoldenRatio, printer='presentation'
        ) == '<mi>&#x3A6;</mi>'

    assert mathml(zoo, printer='presentation') == \
        '<mover><mo>&#x221E;</mo><mo>~</mo></mover>'

    assert mathml(S.NaN, printer='presentation') == '<mi>NaN</mi>'

def test_presentation_mathml_trig():
    mml = mpp._print(sin(x))
    assert mml.childNodes[0].childNodes[0].nodeValue == 'sin'

    mml = mpp._print(cos(x))
    assert mml.childNodes[0].childNodes[0].nodeValue == 'cos'

    mml = mpp._print(tan(x))
    assert mml.childNodes[0].childNodes[0].nodeValue == 'tan'

    mml = mpp._print(asin(x))
    assert mml.childNodes[0].childNodes[0].nodeValue == 'arcsin'

    mml = mpp._print(acos(x))
    assert mml.childNodes[0].childNodes[0].nodeValue == 'arccos'

    mml = mpp._print(atan(x))
    assert mml.childNodes[0].childNodes[0].nodeValue == 'arctan'

    mml = mpp._print(sinh(x))
    assert mml.childNodes[0].childNodes[0].nodeValue == 'sinh'

    mml = mpp._print(cosh(x))
    assert mml.childNodes[0].childNodes[0].nodeValue == 'cosh'

    mml = mpp._print(tanh(x))
    assert mml.childNodes[0].childNodes[0].nodeValue == 'tanh'

    mml = mpp._print(asinh(x))
    assert mml.childNodes[0].childNodes[0].nodeValue == 'arcsinh'

    mml = mpp._print(atanh(x))
    assert mml.childNodes[0].childNodes[0].nodeValue == 'arctanh'

    mml = mpp._print(acosh(x))
    assert mml.childNodes[0].childNodes[0].nodeValue == 'arccosh'


def test_presentation_mathml_relational():
    mml_1 = mpp._print(Eq(x, 1))
    assert len(mml_1.childNodes) == 3
    assert mml_1.childNodes[0].nodeName == 'mi'
    assert mml_1.childNodes[0].childNodes[0].nodeValue == 'x'
    assert mml_1.childNodes[1].nodeName == 'mo'
    assert mml_1.childNodes[1].childNodes[0].nodeValue == '='
    assert mml_1.childNodes[2].nodeName == 'mn'
    assert mml_1.childNodes[2].childNodes[0].nodeValue == '1'

    mml_2 = mpp._print(Ne(1, x))
    assert len(mml_2.childNodes) == 3
    assert mml_2.childNodes[0].nodeName == 'mn'
    assert mml_2.childNodes[0].childNodes[0].nodeValue == '1'
    assert mml_2.childNodes[1].nodeName == 'mo'
    assert mml_2.childNodes[1].childNodes[0].nodeValue == '&#x2260;'
    assert mml_2.childNodes[2].nodeName == 'mi'
    assert mml_2.childNodes[2].childNodes[0].nodeValue == 'x'

    mml_3 = mpp._print(Ge(1, x))
    assert len(mml_3.childNodes) == 3
    assert mml_3.childNodes[0].nodeName == 'mn'
    assert mml_3.childNodes[0].childNodes[0].nodeValue == '1'
    assert mml_3.childNodes[1].nodeName == 'mo'
    assert mml_3.childNodes[1].childNodes[0].nodeValue == '&#x2265;'
    assert mml_3.childNodes[2].nodeName == 'mi'
    assert mml_3.childNodes[2].childNodes[0].nodeValue == 'x'

    mml_4 = mpp._print(Lt(1, x))
    assert len(mml_4.childNodes) == 3
    assert mml_4.childNodes[0].nodeName == 'mn'
    assert mml_4.childNodes[0].childNodes[0].nodeValue == '1'
    assert mml_4.childNodes[1].nodeName == 'mo'
    assert mml_4.childNodes[1].childNodes[0].nodeValue == '<'
    assert mml_4.childNodes[2].nodeName == 'mi'
    assert mml_4.childNodes[2].childNodes[0].nodeValue == 'x'


def test_presentation_symbol():
    mml = mpp._print(x)
    assert mml.nodeName == 'mi'
    assert mml.childNodes[0].nodeValue == 'x'
    del mml

    mml = mpp._print(Symbol("x^2"))
    assert mml.nodeName == 'msup'
    assert mml.childNodes[0].nodeName == 'mi'
    assert mml.childNodes[0].childNodes[0].nodeValue == 'x'
    assert mml.childNodes[1].nodeName == 'mi'
    assert mml.childNodes[1].childNodes[0].nodeValue == '2'
    del mml

    mml = mpp._print(Symbol("x__2"))
    assert mml.nodeName == 'msup'
    assert mml.childNodes[0].nodeName == 'mi'
    assert mml.childNodes[0].childNodes[0].nodeValue == 'x'
    assert mml.childNodes[1].nodeName == 'mi'
    assert mml.childNodes[1].childNodes[0].nodeValue == '2'
    del mml

    mml = mpp._print(Symbol("x_2"))
    assert mml.nodeName == 'msub'
    assert mml.childNodes[0].nodeName == 'mi'
    assert mml.childNodes[0].childNodes[0].nodeValue == 'x'
    assert mml.childNodes[1].nodeName == 'mi'
    assert mml.childNodes[1].childNodes[0].nodeValue == '2'
    del mml

    mml = mpp._print(Symbol("x^3_2"))
    assert mml.nodeName == 'msubsup'
    assert mml.childNodes[0].nodeName == 'mi'
    assert mml.childNodes[0].childNodes[0].nodeValue == 'x'
    assert mml.childNodes[1].nodeName == 'mi'
    assert mml.childNodes[1].childNodes[0].nodeValue == '2'
    assert mml.childNodes[2].nodeName == 'mi'
    assert mml.childNodes[2].childNodes[0].nodeValue == '3'
    del mml

    mml = mpp._print(Symbol("x__3_2"))
    assert mml.nodeName == 'msubsup'
    assert mml.childNodes[0].nodeName == 'mi'
    assert mml.childNodes[0].childNodes[0].nodeValue == 'x'
    assert mml.childNodes[1].nodeName == 'mi'
    assert mml.childNodes[1].childNodes[0].nodeValue == '2'
    assert mml.childNodes[2].nodeName == 'mi'
    assert mml.childNodes[2].childNodes[0].nodeValue == '3'
    del mml

    mml = mpp._print(Symbol("x_2_a"))
    assert mml.nodeName == 'msub'
    assert mml.childNodes[0].nodeName == 'mi'
    assert mml.childNodes[0].childNodes[0].nodeValue == 'x'
    assert mml.childNodes[1].nodeName == 'mrow'
    assert mml.childNodes[1].childNodes[0].nodeName == 'mi'
    assert mml.childNodes[1].childNodes[0].childNodes[0].nodeValue == '2'
    assert mml.childNodes[1].childNodes[1].nodeName == 'mo'
    assert mml.childNodes[1].childNodes[1].childNodes[0].nodeValue == ' '
    assert mml.childNodes[1].childNodes[2].nodeName == 'mi'
    assert mml.childNodes[1].childNodes[2].childNodes[0].nodeValue == 'a'
    del mml

    mml = mpp._print(Symbol("x^2^a"))
    assert mml.nodeName == 'msup'
    assert mml.childNodes[0].nodeName == 'mi'
    assert mml.childNodes[0].childNodes[0].nodeValue == 'x'
    assert mml.childNodes[1].nodeName == 'mrow'
    assert mml.childNodes[1].childNodes[0].nodeName == 'mi'
    assert mml.childNodes[1].childNodes[0].childNodes[0].nodeValue == '2'
    assert mml.childNodes[1].childNodes[1].nodeName == 'mo'
    assert mml.childNodes[1].childNodes[1].childNodes[0].nodeValue == ' '
    assert mml.childNodes[1].childNodes[2].nodeName == 'mi'
    assert mml.childNodes[1].childNodes[2].childNodes[0].nodeValue == 'a'
    del mml

    mml = mpp._print(Symbol("x__2__a"))
    assert mml.nodeName == 'msup'
    assert mml.childNodes[0].nodeName == 'mi'
    assert mml.childNodes[0].childNodes[0].nodeValue == 'x'
    assert mml.childNodes[1].nodeName == 'mrow'
    assert mml.childNodes[1].childNodes[0].nodeName == 'mi'
    assert mml.childNodes[1].childNodes[0].childNodes[0].nodeValue == '2'
    assert mml.childNodes[1].childNodes[1].nodeName == 'mo'
    assert mml.childNodes[1].childNodes[1].childNodes[0].nodeValue == ' '
    assert mml.childNodes[1].childNodes[2].nodeName == 'mi'
    assert mml.childNodes[1].childNodes[2].childNodes[0].nodeValue == 'a'
    del mml


def test_presentation_mathml_greek():
    mml = mpp._print(Symbol('alpha'))
    assert mml.nodeName == 'mi'
    assert mml.childNodes[0].nodeValue == '\N{GREEK SMALL LETTER ALPHA}'

    assert mpp.doprint(Symbol('alpha')) == '<mi>&#945;</mi>'
    assert mpp.doprint(Symbol('beta')) == '<mi>&#946;</mi>'
    assert mpp.doprint(Symbol('gamma')) == '<mi>&#947;</mi>'
    assert mpp.doprint(Symbol('delta')) == '<mi>&#948;</mi>'
    assert mpp.doprint(Symbol('epsilon')) == '<mi>&#949;</mi>'
    assert mpp.doprint(Symbol('zeta')) == '<mi>&#950;</mi>'
    assert mpp.doprint(Symbol('eta')) == '<mi>&#951;</mi>'
    assert mpp.doprint(Symbol('theta')) == '<mi>&#952;</mi>'
    assert mpp.doprint(Symbol('iota')) == '<mi>&#953;</mi>'
    assert mpp.doprint(Symbol('kappa')) == '<mi>&#954;</mi>'
    assert mpp.doprint(Symbol('lambda')) == '<mi>&#955;</mi>'
    assert mpp.doprint(Symbol('mu')) == '<mi>&#956;</mi>'
    assert mpp.doprint(Symbol('nu')) == '<mi>&#957;</mi>'
    assert mpp.doprint(Symbol('xi')) == '<mi>&#958;</mi>'
    assert mpp.doprint(Symbol('omicron')) == '<mi>&#959;</mi>'
    assert mpp.doprint(Symbol('pi')) == '<mi>&#960;</mi>'
    assert mpp.doprint(Symbol('rho')) == '<mi>&#961;</mi>'
    assert mpp.doprint(Symbol('varsigma')) == '<mi>&#962;</mi>'
    assert mpp.doprint(Symbol('sigma')) == '<mi>&#963;</mi>'
    assert mpp.doprint(Symbol('tau')) == '<mi>&#964;</mi>'
    assert mpp.doprint(Symbol('upsilon')) == '<mi>&#965;</mi>'
    assert mpp.doprint(Symbol('phi')) == '<mi>&#966;</mi>'
    assert mpp.doprint(Symbol('chi')) == '<mi>&#967;</mi>'
    assert mpp.doprint(Symbol('psi')) == '<mi>&#968;</mi>'
    assert mpp.doprint(Symbol('omega')) == '<mi>&#969;</mi>'

    assert mpp.doprint(Symbol('Alpha')) == '<mi>&#913;</mi>'
    assert mpp.doprint(Symbol('Beta')) == '<mi>&#914;</mi>'
    assert mpp.doprint(Symbol('Gamma')) == '<mi>&#915;</mi>'
    assert mpp.doprint(Symbol('Delta')) == '<mi>&#916;</mi>'
    assert mpp.doprint(Symbol('Epsilon')) == '<mi>&#917;</mi>'
    assert mpp.doprint(Symbol('Zeta')) == '<mi>&#918;</mi>'
    assert mpp.doprint(Symbol('Eta')) == '<mi>&#919;</mi>'
    assert mpp.doprint(Symbol('Theta')) == '<mi>&#920;</mi>'
    assert mpp.doprint(Symbol('Iota')) == '<mi>&#921;</mi>'
    assert mpp.doprint(Symbol('Kappa')) == '<mi>&#922;</mi>'
    assert mpp.doprint(Symbol('Lambda')) == '<mi>&#923;</mi>'
    assert mpp.doprint(Symbol('Mu')) == '<mi>&#924;</mi>'
    assert mpp.doprint(Symbol('Nu')) == '<mi>&#925;</mi>'
    assert mpp.doprint(Symbol('Xi')) == '<mi>&#926;</mi>'
    assert mpp.doprint(Symbol('Omicron')) == '<mi>&#927;</mi>'
    assert mpp.doprint(Symbol('Pi')) == '<mi>&#928;</mi>'
    assert mpp.doprint(Symbol('Rho')) == '<mi>&#929;</mi>'
    assert mpp.doprint(Symbol('Sigma')) == '<mi>&#931;</mi>'
    assert mpp.doprint(Symbol('Tau')) == '<mi>&#932;</mi>'
    assert mpp.doprint(Symbol('Upsilon')) == '<mi>&#933;</mi>'
    assert mpp.doprint(Symbol('Phi')) == '<mi>&#934;</mi>'
    assert mpp.doprint(Symbol('Chi')) == '<mi>&#935;</mi>'
    assert mpp.doprint(Symbol('Psi')) == '<mi>&#936;</mi>'
    assert mpp.doprint(Symbol('Omega')) == '<mi>&#937;</mi>'


def test_presentation_mathml_order():
    expr = x**3 + x**2*y + 3*x*y**3 + y**4

    mp = MathMLPresentationPrinter({'order': 'lex'})
    mml = mp._print(expr)
    assert mml.childNodes[0].nodeName == 'msup'
    assert mml.childNodes[0].childNodes[0].childNodes[0].nodeValue == 'x'
    assert mml.childNodes[0].childNodes[1].childNodes[0].nodeValue == '3'

    assert mml.childNodes[6].nodeName == 'msup'
    assert mml.childNodes[6].childNodes[0].childNodes[0].nodeValue == 'y'
    assert mml.childNodes[6].childNodes[1].childNodes[0].nodeValue == '4'

    mp = MathMLPresentationPrinter({'order': 'rev-lex'})
    mml = mp._print(expr)

    assert mml.childNodes[0].nodeName == 'msup'
    assert mml.childNodes[0].childNodes[0].childNodes[0].nodeValue == 'y'
    assert mml.childNodes[0].childNodes[1].childNodes[0].nodeValue == '4'

    assert mml.childNodes[6].nodeName == 'msup'
    assert mml.childNodes[6].childNodes[0].childNodes[0].nodeValue == 'x'
    assert mml.childNodes[6].childNodes[1].childNodes[0].nodeValue == '3'


def test_print_intervals():
    a = Symbol('a', real=True)
    assert mpp.doprint(Interval(0, a)) == \
        '<mrow><mo>[</mo><mn>0</mn><mo>,</mo><mi>a</mi><mo>]</mo></mrow>'
    assert mpp.doprint(Interval(0, a, False, False)) == \
        '<mrow><mo>[</mo><mn>0</mn><mo>,</mo><mi>a</mi><mo>]</mo></mrow>'
    assert mpp.doprint(Interval(0, a, True, False)) == \
        '<mrow><mo>(</mo><mn>0</mn><mo>,</mo><mi>a</mi><mo>]</mo></mrow>'
    assert mpp.doprint(Interval(0, a, False, True)) == \
        '<mrow><mo>[</mo><mn>0</mn><mo>,</mo><mi>a</mi><mo>)</mo></mrow>'
    assert mpp.doprint(Interval(0, a, True, True)) == \
        '<mrow><mo>(</mo><mn>0</mn><mo>,</mo><mi>a</mi><mo>)</mo></mrow>'


def test_print_tuples():
    assert mpp.doprint(Tuple(0,)) == \
        '<mrow><mo>(</mo><mn>0</mn><mo>)</mo></mrow>'
    assert mpp.doprint(Tuple(0, a)) == \
        '<mrow><mo>(</mo><mn>0</mn><mo>,</mo><mi>a</mi><mo>)</mo></mrow>'
    assert mpp.doprint(Tuple(0, a, a)) == \
        '<mrow><mo>(</mo><mn>0</mn><mo>,</mo><mi>a</mi><mo>,</mo><mi>a</mi><mo>)</mo></mrow>'
    assert mpp.doprint(Tuple(0, 1, 2, 3, 4)) == \
        '<mrow><mo>(</mo><mn>0</mn><mo>,</mo><mn>1</mn><mo>,</mo><mn>2</mn><mo>,</mo><mn>3</mn><mo>,</mo><mn>4</mn><mo>)</mo></mrow>'
    assert mpp.doprint(Tuple(0, 1, Tuple(2, 3, 4))) == \
        '<mrow><mo>(</mo><mn>0</mn><mo>,</mo><mn>1</mn><mo>,</mo><mrow><mo>(</mo><mn>2</mn><mo>,</mo><mn>3'\
        '</mn><mo>,</mo><mn>4</mn><mo>)</mo></mrow><mo>)</mo></mrow>'


def test_print_re_im():
    assert mpp.doprint(re(x)) == \
        '<mrow><mi>&#8476;</mi><mrow><mo>(</mo><mi>x</mi><mo>)</mo></mrow></mrow>'
    assert mpp.doprint(im(x)) == \
        '<mrow><mi>&#8465;</mi><mrow><mo>(</mo><mi>x</mi><mo>)</mo></mrow></mrow>'
    assert mpp.doprint(re(x + 1, evaluate=False)) == \
        '<mrow><mi>&#8476;</mi><mrow><mo>(</mo><mrow><mi>x</mi><mo>+</mo><mn>1</mn></mrow><mo>)</mo></mrow></mrow>'
    assert mpp.doprint(im(x + 1, evaluate=False)) == \
        '<mrow><mi>&#8465;</mi><mrow><mo>(</mo><mrow><mi>x</mi><mo>+</mo><mn>1</mn></mrow><mo>)</mo></mrow></mrow>'


def test_print_Abs():
    assert mpp.doprint(Abs(x)) == \
        '<mrow><mo>|</mo><mi>x</mi><mo>|</mo></mrow>'
    assert mpp.doprint(Abs(x + 1)) == \
        '<mrow><mo>|</mo><mrow><mi>x</mi><mo>+</mo><mn>1</mn></mrow><mo>|</mo></mrow>'


def test_print_Determinant():
    assert mpp.doprint(Determinant(Matrix([[1, 2], [3, 4]]))) == \
        '<mrow><mo>|</mo><mrow><mo>[</mo><mtable><mtr><mtd><mn>1</mn></mtd><mtd><mn>2</mn></mtd></mtr><mtr><mtd><mn>3</mn></mtd><mtd><mn>4</mn></mtd></mtr></mtable><mo>]</mo></mrow><mo>|</mo></mrow>'


def test_presentation_settings():
    raises(TypeError, lambda: mathml(x, printer='presentation',
                                     method="garbage"))


def test_print_domains():
    from sympy.sets import Integers, Naturals, Naturals0, Reals, Complexes

    assert mpp.doprint(Complexes) == '<mi mathvariant="normal">&#x2102;</mi>'
    assert mpp.doprint(Integers) == '<mi mathvariant="normal">&#x2124;</mi>'
    assert mpp.doprint(Naturals) == '<mi mathvariant="normal">&#x2115;</mi>'
    assert mpp.doprint(Naturals0) == \
        '<msub><mi mathvariant="normal">&#x2115;</mi><mn>0</mn></msub>'
    assert mpp.doprint(Reals) == '<mi mathvariant="normal">&#x211D;</mi>'


def test_print_expression_with_minus():
    assert mpp.doprint(-x) == '<mrow><mo>-</mo><mi>x</mi></mrow>'
    assert mpp.doprint(-x/y) == \
        '<mrow><mo>-</mo><mfrac><mi>x</mi><mi>y</mi></mfrac></mrow>'
    assert mpp.doprint(-Rational(1, 2)) == \
        '<mrow><mo>-</mo><mfrac><mn>1</mn><mn>2</mn></mfrac></mrow>'


def test_print_AssocOp():
    from sympy.core.operations import AssocOp

    class TestAssocOp(AssocOp):
        identity = 0

    expr = TestAssocOp(1, 2)
    assert mpp.doprint(expr) == \
        '<mrow><mi>testassocop</mi><mn>1</mn><mn>2</mn></mrow>'


def test_print_basic():
    expr = Basic(S(1), S(2))
    assert mpp.doprint(expr) == \
        '<mrow><mi>basic</mi><mrow><mo>(</mo><mn>1</mn><mo>,</mo><mn>2</mn><mo>)</mo></mrow></mrow>'
    assert mp.doprint(expr) == '<basic><cn>1</cn><cn>2</cn></basic>'


def test_mat_delim_print():
    expr = Matrix([[1, 2], [3, 4]])
    assert mathml(expr, printer='presentation', mat_delim='[') == \
        '<mrow><mo>[</mo><mtable><mtr><mtd><mn>1</mn></mtd><mtd>'\
        '<mn>2</mn></mtd></mtr><mtr><mtd><mn>3</mn></mtd><mtd><mn>4</mn>'\
        '</mtd></mtr></mtable><mo>]</mo></mrow>'
    assert mathml(expr, printer='presentation', mat_delim='(') == \
        '<mrow><mo>(</mo><mtable><mtr><mtd><mn>1</mn></mtd><mtd><mn>2</mn></mtd>'\
        '</mtr><mtr><mtd><mn>3</mn></mtd><mtd><mn>4</mn></mtd></mtr></mtable><mo>)</mo></mrow>'
    assert mathml(expr, printer='presentation', mat_delim='') == \
        '<mtable><mtr><mtd><mn>1</mn></mtd><mtd><mn>2</mn></mtd></mtr><mtr>'\
        '<mtd><mn>3</mn></mtd><mtd><mn>4</mn></mtd></mtr></mtable>'


def test_ln_notation_print():
    expr = log(x)
    assert mathml(expr, printer='presentation') == \
        '<mrow><mi>log</mi><mrow><mo>(</mo><mi>x</mi><mo>)</mo></mrow></mrow>'
    assert mathml(expr, printer='presentation', ln_notation=False) == \
        '<mrow><mi>log</mi><mrow><mo>(</mo><mi>x</mi><mo>)</mo></mrow></mrow>'
    assert mathml(expr, printer='presentation', ln_notation=True) == \
        '<mrow><mi>ln</mi><mrow><mo>(</mo><mi>x</mi><mo>)</mo></mrow></mrow>'


def test_mul_symbol_print():
    expr = x * y
    assert mathml(expr, printer='presentation') == \
        '<mrow><mi>x</mi><mo>&InvisibleTimes;</mo><mi>y</mi></mrow>'
    assert mathml(expr, printer='presentation', mul_symbol=None) == \
        '<mrow><mi>x</mi><mo>&InvisibleTimes;</mo><mi>y</mi></mrow>'
    assert mathml(expr, printer='presentation', mul_symbol='dot') == \
        '<mrow><mi>x</mi><mo>&#xB7;</mo><mi>y</mi></mrow>'
    assert mathml(expr, printer='presentation', mul_symbol='ldot') == \
        '<mrow><mi>x</mi><mo>&#x2024;</mo><mi>y</mi></mrow>'
    assert mathml(expr, printer='presentation', mul_symbol='times') == \
        '<mrow><mi>x</mi><mo>&#xD7;</mo><mi>y</mi></mrow>'


def test_print_lerchphi():
    assert mpp.doprint(lerchphi(1, 2, 3)) == \
        '<mrow><mi>&#x3A6;</mi><mrow><mo>(</mo><mn>1</mn><mo>,</mo><mn>2</mn><mo>,</mo><mn>3</mn><mo>)</mo></mrow></mrow>'


def test_print_polylog():
    assert mp.doprint(polylog(x, y)) == \
        '<apply><polylog/><ci>x</ci><ci>y</ci></apply>'
    assert mpp.doprint(polylog(x, y)) == \
        '<mrow><msub><mi>Li</mi><mi>x</mi></msub><mrow><mo>(</mo><mi>y</mi><mo>)</mo></mrow></mrow>'


def test_print_set_frozenset():
    f = frozenset({1, 5, 3})
    assert mpp.doprint(f) == \
        '<mrow><mo>{</mo><mn>1</mn><mo>,</mo><mn>3</mn><mo>,</mo><mn>5</mn><mo>}</mo></mrow>'
    s = set({1, 2, 3})
    assert mpp.doprint(s) == \
        '<mrow><mo>{</mo><mn>1</mn><mo>,</mo><mn>2</mn><mo>,</mo><mn>3</mn><mo>}</mo></mrow>'


def test_print_FiniteSet():
    f1 = FiniteSet(x, 1, 3)
    assert mpp.doprint(f1) == \
        '<mrow><mo>{</mo><mn>1</mn><mo>,</mo><mn>3</mn><mo>,</mo><mi>x</mi><mo>}</mo></mrow>'


def test_print_LambertW():
    assert mpp.doprint(LambertW(x)) == '<mrow><mi>W</mi><mrow><mo>(</mo><mi>x</mi><mo>)</mo></mrow></mrow>'
    assert mpp.doprint(LambertW(x, y)) == '<mrow><mi>W</mi><mrow><mo>(</mo><mi>x</mi><mo>,</mo><mi>y</mi><mo>)</mo></mrow></mrow>'


def test_print_EmptySet():
    assert mpp.doprint(S.EmptySet) == '<mo>&#x2205;</mo>'


def test_print_UniversalSet():
    assert mpp.doprint(S.UniversalSet) == '<mo>&#x1D54C;</mo>'


def test_print_spaces():
    assert mpp.doprint(HilbertSpace()) == '<mi>&#x210B;</mi>'
    assert mpp.doprint(ComplexSpace(2)) == '<msup>&#x1D49E;<mn>2</mn></msup>'
    assert mpp.doprint(FockSpace()) == '<mi>&#x2131;</mi>'


def test_print_constants():
    assert mpp.doprint(hbar) == '<mi>&#x210F;</mi>'
    assert mpp.doprint(S.TribonacciConstant) == '<mi>TribonacciConstant</mi>'
    assert mpp.doprint(S.GoldenRatio) == '<mi>&#x3A6;</mi>'
    assert mpp.doprint(S.EulerGamma) == '<mi>&#x3B3;</mi>'


def test_print_Contains():
    assert mpp.doprint(Contains(x, S.Naturals)) == \
        '<mrow><mi>x</mi><mo>&#x2208;</mo><mi mathvariant="normal">&#x2115;</mi></mrow>'


def test_print_Dagger():
    x = symbols('x', commutative=False)
    assert mpp.doprint(Dagger(x)) == '<msup><mi>x</mi>&#x2020;</msup>'


def test_print_SetOp():
    f1 = FiniteSet(x, 1, 3)
    f2 = FiniteSet(y, 2, 4)

    prntr = lambda x: mathml(x, printer='presentation')

    assert prntr(Union(f1, f2, evaluate=False)) == \
    '<mrow><mrow><mo>{</mo><mn>1</mn><mo>,</mo><mn>3</mn><mo>,</mo><mi>x</mi>'\
    '<mo>}</mo></mrow><mo>&#x222A;</mo><mrow><mo>{</mo><mn>2</mn><mo>,</mo>'\
    '<mn>4</mn><mo>,</mo><mi>y</mi><mo>}</mo></mrow></mrow>'
    assert prntr(Intersection(f1, f2, evaluate=False)) == \
    '<mrow><mrow><mo>{</mo><mn>1</mn><mo>,</mo><mn>3</mn><mo>,</mo><mi>x</mi>'\
    '<mo>}</mo></mrow><mo>&#x2229;</mo><mrow><mo>{</mo><mn>2</mn>'\
    '<mo>,</mo><mn>4</mn><mo>,</mo><mi>y</mi><mo>}</mo></mrow></mrow>'
    assert prntr(Complement(f1, f2, evaluate=False)) == \
    '<mrow><mrow><mo>{</mo><mn>1</mn><mo>,</mo><mn>3</mn><mo>,</mo><mi>x</mi>'\
    '<mo>}</mo></mrow><mo>&#x2216;</mo><mrow><mo>{</mo><mn>2</mn>'\
    '<mo>,</mo><mn>4</mn><mo>,</mo><mi>y</mi><mo>}</mo></mrow></mrow>'
    assert prntr(SymmetricDifference(f1, f2, evaluate=False)) == \
    '<mrow><mrow><mo>{</mo><mn>1</mn><mo>,</mo><mn>3</mn><mo>,</mo><mi>x</mi>'\
    '<mo>}</mo></mrow><mo>&#x2206;</mo><mrow><mo>{</mo><mn>2</mn>'\
    '<mo>,</mo><mn>4</mn><mo>,</mo><mi>y</mi><mo>}</mo></mrow></mrow>'

    A = FiniteSet(a)
    C = FiniteSet(c)
    D = FiniteSet(d)

    U1 = Union(C, D, evaluate=False)
    I1 = Intersection(C, D, evaluate=False)
    C1 = Complement(C, D, evaluate=False)
    D1 = SymmetricDifference(C, D, evaluate=False)
    # XXX ProductSet does not support evaluate keyword
    P1 = ProductSet(C, D)

    assert prntr(Union(A, I1, evaluate=False)) == \
        '<mrow><mrow><mo>{</mo><mi>a</mi><mo>}</mo></mrow>' \
        '<mo>&#x222A;</mo><mrow><mo>(</mo><mrow><mrow><mo>{</mo>' \
        '<mi>c</mi><mo>}</mo></mrow><mo>&#x2229;</mo><mrow><mo>{</mo>' \
        '<mi>d</mi><mo>}</mo></mrow></mrow><mo>)</mo></mrow></mrow>'
    assert prntr(Intersection(A, C1, evaluate=False)) == \
        '<mrow><mrow><mo>{</mo><mi>a</mi><mo>}</mo></mrow>' \
        '<mo>&#x2229;</mo><mrow><mo>(</mo><mrow><mrow><mo>{</mo>' \
        '<mi>c</mi><mo>}</mo></mrow><mo>&#x2216;</mo><mrow><mo>{</mo>' \
        '<mi>d</mi><mo>}</mo></mrow></mrow><mo>)</mo></mrow></mrow>'
    assert prntr(Complement(A, D1, evaluate=False)) == \
        '<mrow><mrow><mo>{</mo><mi>a</mi><mo>}</mo></mrow>' \
        '<mo>&#x2216;</mo><mrow><mo>(</mo><mrow><mrow><mo>{</mo>' \
        '<mi>c</mi><mo>}</mo></mrow><mo>&#x2206;</mo><mrow><mo>{</mo>' \
        '<mi>d</mi><mo>}</mo></mrow></mrow><mo>)</mo></mrow></mrow>'
    assert prntr(SymmetricDifference(A, P1, evaluate=False)) == \
        '<mrow><mrow><mo>{</mo><mi>a</mi><mo>}</mo></mrow>' \
        '<mo>&#x2206;</mo><mrow><mo>(</mo><mrow><mrow><mo>{</mo>' \
        '<mi>c</mi><mo>}</mo></mrow><mo>&#x00d7;</mo><mrow><mo>{</mo>' \
        '<mi>d</mi><mo>}</mo></mrow></mrow><mo>)</mo></mrow></mrow>'
    assert prntr(ProductSet(A, U1)) == \
        '<mrow><mrow><mo>{</mo><mi>a</mi><mo>}</mo></mrow>' \
        '<mo>&#x00d7;</mo><mrow><mo>(</mo><mrow><mrow><mo>{</mo>' \
        '<mi>c</mi><mo>}</mo></mrow><mo>&#x222A;</mo><mrow><mo>{</mo>' \
        '<mi>d</mi><mo>}</mo></mrow></mrow><mo>)</mo></mrow></mrow>'


def test_print_logic():
    assert mpp.doprint(And(x, y)) == \
        '<mrow><mi>x</mi><mo>&#x2227;</mo><mi>y</mi></mrow>'
    assert mpp.doprint(Or(x, y)) == \
        '<mrow><mi>x</mi><mo>&#x2228;</mo><mi>y</mi></mrow>'
    assert mpp.doprint(Xor(x, y)) == \
        '<mrow><mi>x</mi><mo>&#x22BB;</mo><mi>y</mi></mrow>'
    assert mpp.doprint(Implies(x, y)) == \
        '<mrow><mi>x</mi><mo>&#x21D2;</mo><mi>y</mi></mrow>'
    assert mpp.doprint(Equivalent(x, y)) == \
        '<mrow><mi>x</mi><mo>&#x21D4;</mo><mi>y</mi></mrow>'

    assert mpp.doprint(And(Eq(x, y), x > 4)) == \
        '<mrow><mrow><mi>x</mi><mo>=</mo><mi>y</mi></mrow><mo>&#x2227;</mo>'\
        '<mrow><mi>x</mi><mo>></mo><mn>4</mn></mrow></mrow>'
    assert mpp.doprint(And(Eq(x, 3), y < 3, x > y + 1)) == \
        '<mrow><mrow><mi>x</mi><mo>=</mo><mn>3</mn></mrow><mo>&#x2227;</mo>'\
        '<mrow><mi>x</mi><mo>></mo><mrow><mi>y</mi><mo>+</mo><mn>1</mn></mrow>'\
        '</mrow><mo>&#x2227;</mo><mrow><mi>y</mi><mo><</mo><mn>3</mn></mrow></mrow>'
    assert mpp.doprint(Or(Eq(x, y), x > 4)) == \
        '<mrow><mrow><mi>x</mi><mo>=</mo><mi>y</mi></mrow><mo>&#x2228;</mo>'\
        '<mrow><mi>x</mi><mo>></mo><mn>4</mn></mrow></mrow>'
    assert mpp.doprint(And(Eq(x, 3), Or(y < 3, x > y + 1))) == \
        '<mrow><mrow><mi>x</mi><mo>=</mo><mn>3</mn></mrow>'\
        '<mo>&#x2227;</mo><mrow><mo>(</mo><mrow><mrow><mi>x</mi><mo>></mo><mrow>'\
        '<mi>y</mi><mo>+</mo><mn>1</mn></mrow></mrow><mo>&#x2228;</mo><mrow>'\
        '<mi>y</mi><mo><</mo><mn>3</mn></mrow></mrow><mo>)</mo></mrow></mrow>'

    assert mpp.doprint(Not(x)) == '<mrow><mo>&#xAC;</mo><mi>x</mi></mrow>'
    assert mpp.doprint(Not(And(x, y))) == \
        '<mrow><mo>&#xAC;</mo><mrow><mo>(</mo><mrow><mi>x</mi><mo>&#x2227;</mo><mi>y</mi></mrow><mo>)</mo></mrow></mrow>'


def test_root_notation_print():
    assert mathml(x**(S.One/3), printer='presentation') == \
        '<mroot><mi>x</mi><mn>3</mn></mroot>'
    assert mathml(x**(S.One/3), printer='presentation', root_notation=False) ==\
        '<msup><mi>x</mi><mfrac><mn>1</mn><mn>3</mn></mfrac></msup>'
    assert mathml(x**(S.One/3), printer='content') == \
        '<apply><root/><degree><cn>3</cn></degree><ci>x</ci></apply>'
    assert mathml(x**(S.One/3), printer='content', root_notation=False) == \
        '<apply><power/><ci>x</ci><apply><divide/><cn>1</cn><cn>3</cn></apply></apply>'
    assert mathml(x**(Rational(-1, 3)), printer='presentation') == \
        '<mfrac><mn>1</mn><mroot><mi>x</mi><mn>3</mn></mroot></mfrac>'
    assert mathml(x**(Rational(-1, 3)), printer='presentation', root_notation=False) \
        == '<mfrac><mn>1</mn><msup><mi>x</mi><mfrac><mn>1</mn><mn>3</mn></mfrac></msup></mfrac>'


def test_fold_frac_powers_print():
    expr = x ** Rational(5, 2)
    assert mathml(expr, printer='presentation') == \
        '<msup><mi>x</mi><mfrac><mn>5</mn><mn>2</mn></mfrac></msup>'
    assert mathml(expr, printer='presentation', fold_frac_powers=True) == \
        '<msup><mi>x</mi><mfrac bevelled="true"><mn>5</mn><mn>2</mn></mfrac></msup>'
    assert mathml(expr, printer='presentation', fold_frac_powers=False) == \
        '<msup><mi>x</mi><mfrac><mn>5</mn><mn>2</mn></mfrac></msup>'


def test_fold_short_frac_print():
    expr = Rational(2, 5)
    assert mathml(expr, printer='presentation') == \
        '<mfrac><mn>2</mn><mn>5</mn></mfrac>'
    assert mathml(expr, printer='presentation', fold_short_frac=True) == \
        '<mfrac bevelled="true"><mn>2</mn><mn>5</mn></mfrac>'
    assert mathml(expr, printer='presentation', fold_short_frac=False) == \
        '<mfrac><mn>2</mn><mn>5</mn></mfrac>'


def test_print_factorials():
    assert mpp.doprint(factorial(x)) == '<mrow><mi>x</mi><mo>!</mo></mrow>'
    assert mpp.doprint(factorial(x + 1)) == \
        '<mrow><mrow><mo>(</mo><mrow><mi>x</mi><mo>+</mo><mn>1</mn></mrow><mo>)</mo></mrow><mo>!</mo></mrow>'
    assert mpp.doprint(factorial2(x)) == '<mrow><mi>x</mi><mo>!!</mo></mrow>'
    assert mpp.doprint(factorial2(x + 1)) == \
        '<mrow><mrow><mo>(</mo><mrow><mi>x</mi><mo>+</mo><mn>1</mn></mrow><mo>)</mo></mrow><mo>!!</mo></mrow>'
    assert mpp.doprint(binomial(x, y)) == \
        '<mrow><mo>(</mo><mfrac linethickness="0"><mi>x</mi><mi>y</mi></mfrac><mo>)</mo></mrow>'
    assert mpp.doprint(binomial(4, x + y)) == \
        '<mrow><mo>(</mo><mfrac linethickness="0"><mn>4</mn><mrow><mi>x</mi>'\
        '<mo>+</mo><mi>y</mi></mrow></mfrac><mo>)</mo></mrow>'


def test_print_floor():
    expr = floor(x)
    assert mathml(expr, printer='presentation') == \
        '<mrow><mo>&#8970;</mo><mi>x</mi><mo>&#8971;</mo></mrow>'


def test_print_ceiling():
    expr = ceiling(x)
    assert mathml(expr, printer='presentation') == \
        '<mrow><mo>&#8968;</mo><mi>x</mi><mo>&#8969;</mo></mrow>'


def test_print_Lambda():
    expr = Lambda(x, x+1)
    assert mathml(expr, printer='presentation') == \
        '<mrow><mo>(</mo><mi>x</mi><mo>&#x21A6;</mo><mrow><mi>x</mi><mo>+</mo><mn>1</mn></mrow><mo>)</mo></mrow>'
    expr = Lambda((x, y), x + y)
    assert mathml(expr, printer='presentation') == \
        '<mrow><mo>(</mo><mrow><mo>(</mo><mi>x</mi><mo>,</mo><mi>y</mi><mo>)</mo></mrow><mo>&#x21A6;</mo><mrow><mi>x</mi><mo>+</mo><mi>y</mi></mrow><mo>)</mo></mrow>'


def test_print_conjugate():
    assert mpp.doprint(conjugate(x)) == \
        '<menclose notation="top"><mi>x</mi></menclose>'
    assert mpp.doprint(conjugate(x + 1)) == \
        '<mrow><menclose notation="top"><mi>x</mi></menclose><mo>+</mo><mn>1</mn></mrow>'


def test_print_AccumBounds():
    a = Symbol('a', real=True)
    assert mpp.doprint(AccumBounds(0, 1)) == '<mrow><mo>&#10216;</mo><mn>0</mn><mo>,</mo><mn>1</mn><mo>&#10217;</mo></mrow>'
    assert mpp.doprint(AccumBounds(0, a)) == '<mrow><mo>&#10216;</mo><mn>0</mn><mo>,</mo><mi>a</mi><mo>&#10217;</mo></mrow>'
    assert mpp.doprint(AccumBounds(a + 1, a + 2)) == '<mrow><mo>&#10216;</mo><mrow><mi>a</mi><mo>+</mo><mn>1</mn></mrow><mo>,</mo><mrow><mi>a</mi><mo>+</mo><mn>2</mn></mrow><mo>&#10217;</mo></mrow>'


def test_print_Float():
    assert mpp.doprint(Float(1e100)) == '<mrow><mn>1.0</mn><mo>&#xB7;</mo><msup><mn>10</mn><mn>100</mn></msup></mrow>'
    assert mpp.doprint(Float(1e-100)) == '<mrow><mn>1.0</mn><mo>&#xB7;</mo><msup><mn>10</mn><mn>-100</mn></msup></mrow>'
    assert mpp.doprint(Float(-1e100)) == '<mrow><mn>-1.0</mn><mo>&#xB7;</mo><msup><mn>10</mn><mn>100</mn></msup></mrow>'
    assert mpp.doprint(Float(1.0*oo)) == '<mi>&#x221E;</mi>'
    assert mpp.doprint(Float(-1.0*oo)) == '<mrow><mo>-</mo><mi>&#x221E;</mi></mrow>'


def test_print_different_functions():
    assert mpp.doprint(gamma(x)) == '<mrow><mi>&#x393;</mi><mrow><mo>(</mo><mi>x</mi><mo>)</mo></mrow></mrow>'
    assert mpp.doprint(lowergamma(x, y)) == '<mrow><mi>&#x3B3;</mi><mrow><mo>(</mo><mi>x</mi><mo>,</mo><mi>y</mi><mo>)</mo></mrow></mrow>'
    assert mpp.doprint(uppergamma(x, y)) == '<mrow><mi>&#x393;</mi><mrow><mo>(</mo><mi>x</mi><mo>,</mo><mi>y</mi><mo>)</mo></mrow></mrow>'
    assert mpp.doprint(zeta(x)) == '<mrow><mi>&#x3B6;</mi><mrow><mo>(</mo><mi>x</mi><mo>)</mo></mrow></mrow>'
    assert mpp.doprint(zeta(x, y)) == '<mrow><mi>&#x3B6;</mi><mrow><mo>(</mo><mi>x</mi><mo>,</mo><mi>y</mi><mo>)</mo></mrow></mrow>'
    assert mpp.doprint(dirichlet_eta(x)) ==  '<mrow><mi>&#x3B7;</mi><mrow><mo>(</mo><mi>x</mi><mo>)</mo></mrow></mrow>'
    assert mpp.doprint(elliptic_k(x)) == '<mrow><mi>&#x39A;</mi><mrow><mo>(</mo><mi>x</mi><mo>)</mo></mrow></mrow>'
    assert mpp.doprint(totient(x)) == '<mrow><mi>&#x3D5;</mi><mrow><mo>(</mo><mi>x</mi><mo>)</mo></mrow></mrow>'
    assert mpp.doprint(reduced_totient(x)) == '<mrow><mi>&#x3BB;</mi><mrow><mo>(</mo><mi>x</mi><mo>)</mo></mrow></mrow>'
    assert mpp.doprint(primenu(x)) == '<mrow><mi>&#x3BD;</mi><mrow><mo>(</mo><mi>x</mi><mo>)</mo></mrow></mrow>'
    assert mpp.doprint(primeomega(x)) == '<mrow><mi>&#x3A9;</mi><mrow><mo>(</mo><mi>x</mi><mo>)</mo></mrow></mrow>'
    assert mpp.doprint(fresnels(x)) == '<mrow><mi>S</mi><mrow><mo>(</mo><mi>x</mi><mo>)</mo></mrow></mrow>'
    assert mpp.doprint(fresnelc(x)) ==  '<mrow><mi>C</mi><mrow><mo>(</mo><mi>x</mi><mo>)</mo></mrow></mrow>'
    assert mpp.doprint(Heaviside(x)) == '<mrow><mi>&#x398;</mi><mrow><mo>(</mo><mi>x</mi><mo>,</mo><mfrac><mn>1</mn><mn>2</mn></mfrac><mo>)</mo></mrow></mrow>'


def test_mathml_builtins():
    assert mpp.doprint(None) == '<mi>None</mi>'
    assert mpp.doprint(true) == '<mi>True</mi>'
    assert mpp.doprint(false) == '<mi>False</mi>'


def test_mathml_Range():
    assert mpp.doprint(Range(1, 51)) == \
        '<mrow><mo>{</mo><mn>1</mn><mo>,</mo><mn>2</mn><mo>,</mo><mi>&#8230;</mi><mo>,</mo><mn>50</mn><mo>}</mo></mrow>'
    assert mpp.doprint(Range(1, 4)) == \
        '<mrow><mo>{</mo><mn>1</mn><mo>,</mo><mn>2</mn><mo>,</mo><mn>3</mn><mo>}</mo></mrow>'
    assert mpp.doprint(Range(0, 3, 1)) == \
        '<mrow><mo>{</mo><mn>0</mn><mo>,</mo><mn>1</mn><mo>,</mo><mn>2</mn><mo>}</mo></mrow>'
    assert mpp.doprint(Range(0, 30, 1)) == \
        '<mrow><mo>{</mo><mn>0</mn><mo>,</mo><mn>1</mn><mo>,</mo><mi>&#8230;</mi><mo>,</mo><mn>29</mn><mo>}</mo></mrow>'
    assert mpp.doprint(Range(30, 1, -1)) == \
        '<mrow><mo>{</mo><mn>30</mn><mo>,</mo><mn>29</mn><mo>,</mo><mi>&#8230;</mi><mo>,</mo><mn>2</mn><mo>}</mo></mrow>'
    assert mpp.doprint(Range(0, oo, 2)) == \
        '<mrow><mo>{</mo><mn>0</mn><mo>,</mo><mn>2</mn><mo>,</mo><mi>&#8230;</mi><mo>}</mo></mrow>'
    assert mpp.doprint(Range(oo, -2, -2)) == \
        '<mrow><mo>{</mo><mi>&#8230;</mi><mo>,</mo><mn>2</mn><mo>,</mo><mn>0</mn><mo>}</mo></mrow>'
    assert mpp.doprint(Range(-2, -oo, -1)) == \
        '<mrow><mo>{</mo><mn>-2</mn><mo>,</mo><mn>-3</mn><mo>,</mo><mi>&#8230;</mi><mo>}</mo></mrow>'


def test_print_exp():
    assert mpp.doprint(exp(x)) == \
        '<msup><mi>&ExponentialE;</mi><mi>x</mi></msup>'
    assert mpp.doprint(exp(1) + exp(2)) == \
        '<mrow><mi>&ExponentialE;</mi><mo>+</mo><msup><mi>&ExponentialE;</mi><mn>2</mn></msup></mrow>'


def test_print_MinMax():
    assert mpp.doprint(Min(x, y)) == \
        '<mrow><mo>min</mo><mrow><mo>(</mo><mi>x</mi><mo>,</mo><mi>y</mi><mo>)</mo></mrow></mrow>'
    assert mpp.doprint(Min(x, 2, x**3)) == \
        '<mrow><mo>min</mo><mrow><mo>(</mo><mn>2</mn><mo>,</mo><mi>x</mi><mo>,</mo><msup><mi>x</mi><mn>3</mn></msup><mo>)</mo></mrow></mrow>'
    assert mpp.doprint(Max(x, y)) == \
        '<mrow><mo>max</mo><mrow><mo>(</mo><mi>x</mi><mo>,</mo><mi>y</mi><mo>)</mo></mrow></mrow>'
    assert mpp.doprint(Max(x, 2, x**3)) == \
        '<mrow><mo>max</mo><mrow><mo>(</mo><mn>2</mn><mo>,</mo><mi>x</mi><mo>,</mo><msup><mi>x</mi><mn>3</mn></msup><mo>)</mo></mrow></mrow>'


def test_mathml_presentation_numbers():
    n = Symbol('n')
    assert mathml(catalan(n), printer='presentation') == \
        '<msub><mi>C</mi><mi>n</mi></msub>'
    assert mathml(bernoulli(n), printer='presentation') == \
        '<msub><mi>B</mi><mi>n</mi></msub>'
    assert mathml(bell(n), printer='presentation') == \
        '<msub><mi>B</mi><mi>n</mi></msub>'
    assert mathml(euler(n), printer='presentation') == \
        '<msub><mi>E</mi><mi>n</mi></msub>'
    assert mathml(fibonacci(n), printer='presentation') == \
        '<msub><mi>F</mi><mi>n</mi></msub>'
    assert mathml(lucas(n), printer='presentation') == \
        '<msub><mi>L</mi><mi>n</mi></msub>'
    assert mathml(tribonacci(n), printer='presentation') == \
        '<msub><mi>T</mi><mi>n</mi></msub>'
    assert mathml(bernoulli(n, x), printer='presentation') == \
        mathml(bell(n, x), printer='presentation') == \
        '<mrow><msub><mi>B</mi><mi>n</mi></msub><mrow><mo>(</mo><mi>x</mi><mo>)</mo></mrow></mrow>'
    assert mathml(euler(n, x), printer='presentation') == \
        '<mrow><msub><mi>E</mi><mi>n</mi></msub><mrow><mo>(</mo><mi>x</mi><mo>)</mo></mrow></mrow>'
    assert mathml(fibonacci(n, x), printer='presentation') == \
        '<mrow><msub><mi>F</mi><mi>n</mi></msub><mrow><mo>(</mo><mi>x</mi><mo>)</mo></mrow></mrow>'
    assert mathml(tribonacci(n, x), printer='presentation') == \
        '<mrow><msub><mi>T</mi><mi>n</mi></msub><mrow><mo>(</mo><mi>x</mi><mo>)</mo></mrow></mrow>'


def test_mathml_presentation_mathieu():
    assert mathml(mathieuc(x, y, z), printer='presentation') == \
        '<mrow><mi>C</mi><mrow><mo>(</mo><mi>x</mi><mo>,</mo><mi>y</mi><mo>,</mo><mi>z</mi><mo>)</mo></mrow></mrow>'
    assert mathml(mathieus(x, y, z), printer='presentation') == \
        '<mrow><mi>S</mi><mrow><mo>(</mo><mi>x</mi><mo>,</mo><mi>y</mi><mo>,</mo><mi>z</mi><mo>)</mo></mrow></mrow>'
    assert mathml(mathieucprime(x, y, z), printer='presentation') == \
        '<mrow><mi>C&#x2032;</mi><mrow><mo>(</mo><mi>x</mi><mo>,</mo><mi>y</mi><mo>,</mo><mi>z</mi><mo>)</mo></mrow></mrow>'
    assert mathml(mathieusprime(x, y, z), printer='presentation') == \
        '<mrow><mi>S&#x2032;</mi><mrow><mo>(</mo><mi>x</mi><mo>,</mo><mi>y</mi><mo>,</mo><mi>z</mi><mo>)</mo></mrow></mrow>'


def test_mathml_presentation_stieltjes():
    assert mathml(stieltjes(n), printer='presentation') == \
         '<msub><mi>&#x03B3;</mi><mi>n</mi></msub>'
    assert mathml(stieltjes(n, x), printer='presentation') == \
         '<mrow><msub><mi>&#x03B3;</mi><mi>n</mi></msub><mrow><mo>(</mo><mi>x</mi><mo>)</mo></mrow></mrow>'


def test_print_matrix_symbol():
    A = MatrixSymbol('A', 1, 2)
    assert mpp.doprint(A) == '<mi>A</mi>'
    assert mp.doprint(A) == '<ci>A</ci>'
    assert mathml(A, printer='presentation', mat_symbol_style="bold") == \
        '<mi mathvariant="bold">A</mi>'
    # No effect in content printer
    assert mathml(A, mat_symbol_style="bold") == '<ci>A</ci>'


def test_print_hadamard():
    from sympy.matrices.expressions import HadamardProduct
    from sympy.matrices.expressions import Transpose

    X = MatrixSymbol('X', 2, 2)
    Y = MatrixSymbol('Y', 2, 2)

    assert mathml(HadamardProduct(X, Y*Y), printer="presentation") == \
        '<mrow>' \
        '<mi>X</mi>' \
        '<mo>&#x2218;</mo>' \
        '<msup><mi>Y</mi><mn>2</mn></msup>' \
        '</mrow>'

    assert mathml(HadamardProduct(X, Y)*Y, printer="presentation") == \
        '<mrow>' \
        '<mrow><mo>(</mo>' \
        '<mrow><mi>X</mi><mo>&#x2218;</mo><mi>Y</mi></mrow>' \
        '<mo>)</mo></mrow>' \
        '<mo>&InvisibleTimes;</mo><mi>Y</mi>' \
        '</mrow>'

    assert mathml(HadamardProduct(X, Y, Y), printer="presentation") == \
        '<mrow>' \
        '<mi>X</mi><mo>&#x2218;</mo>' \
        '<mi>Y</mi><mo>&#x2218;</mo>' \
        '<mi>Y</mi>' \
        '</mrow>'

    assert mathml(
        Transpose(HadamardProduct(X, Y)), printer="presentation") == \
            '<msup>' \
            '<mrow><mo>(</mo>' \
            '<mrow><mi>X</mi><mo>&#x2218;</mo><mi>Y</mi></mrow>' \
            '<mo>)</mo></mrow>' \
            '<mo>T</mo>' \
            '</msup>'


def test_print_random_symbol():
    R = RandomSymbol(Symbol('R'))
    assert mpp.doprint(R) == '<mi>R</mi>'
    assert mp.doprint(R) == '<ci>R</ci>'


def test_print_IndexedBase():
    assert mathml(IndexedBase(a)[b], printer='presentation') == \
        '<msub><mi>a</mi><mi>b</mi></msub>'
    assert mathml(IndexedBase(a)[b, c, d], printer='presentation') == \
        '<msub><mi>a</mi><mrow><mo>(</mo><mi>b</mi><mo>,</mo><mi>c</mi><mo>,</mo><mi>d</mi><mo>)</mo></mrow></msub>'
    assert mathml(IndexedBase(a)[b]*IndexedBase(c)[d]*IndexedBase(e),
                  printer='presentation') == \
                  '<mrow><msub><mi>a</mi><mi>b</mi></msub><mo>&InvisibleTimes;'\
                  '</mo><msub><mi>c</mi><mi>d</mi></msub><mo>&InvisibleTimes;</mo><mi>e</mi></mrow>'


def test_print_Indexed():
    assert mathml(IndexedBase(a), printer='presentation') == '<mi>a</mi>'
    assert mathml(IndexedBase(a/b), printer='presentation') == \
        '<mrow><mfrac><mi>a</mi><mi>b</mi></mfrac></mrow>'
    assert mathml(IndexedBase((a, b)), printer='presentation') == \
        '<mrow><mo>(</mo><mi>a</mi><mo>,</mo><mi>b</mi><mo>)</mo></mrow>'

def test_print_MatrixElement():
    i, j = symbols('i j')
    A = MatrixSymbol('A', i, j)
    assert mathml(A[0,0],printer = 'presentation') == \
        '<msub><mi>A</mi><mrow><mn>0</mn><mo>,</mo><mn>0</mn></mrow></msub>'
    assert mathml(A[i,j], printer = 'presentation') == \
        '<msub><mi>A</mi><mrow><mi>i</mi><mo>,</mo><mi>j</mi></mrow></msub>'
    assert mathml(A[i*j,0], printer = 'presentation') == \
        '<msub><mi>A</mi><mrow><mrow><mi>i</mi><mo>&InvisibleTimes;</mo><mi>j</mi></mrow><mo>,</mo><mn>0</mn></mrow></msub>'


def test_print_Vector():
    ACS = CoordSys3D('A')
    assert mathml(Cross(ACS.i, ACS.j*ACS.x*3 + ACS.k), printer='presentation') == \
        '<mrow><msub><mover><mi mathvariant="bold">i</mi><mo>^</mo></mover>'\
        '<mi mathvariant="bold">A</mi></msub><mo>&#xD7;</mo><mrow><mo>(</mo><mrow>'\
        '<mrow><mo>(</mo><mrow><mn>3</mn><mo>&InvisibleTimes;</mo><msub>'\
        '<mi mathvariant="bold">x</mi><mi mathvariant="bold">A</mi></msub>'\
        '</mrow><mo>)</mo></mrow><mo>&InvisibleTimes;</mo><msub><mover>'\
        '<mi mathvariant="bold">j</mi><mo>^</mo></mover>'\
        '<mi mathvariant="bold">A</mi></msub><mo>+</mo><msub><mover>'\
        '<mi mathvariant="bold">k</mi><mo>^</mo></mover><mi mathvariant="bold">'\
        'A</mi></msub></mrow><mo>)</mo></mrow></mrow>'
    assert mathml(Cross(ACS.i, ACS.j), printer='presentation') == \
        '<mrow><msub><mover><mi mathvariant="bold">i</mi><mo>^</mo></mover>'\
        '<mi mathvariant="bold">A</mi></msub><mo>&#xD7;</mo><msub><mover>'\
        '<mi mathvariant="bold">j</mi><mo>^</mo></mover>'\
        '<mi mathvariant="bold">A</mi></msub></mrow>'
    assert mathml(x*Cross(ACS.i, ACS.j), printer='presentation') == \
        '<mrow><mi>x</mi><mo>&InvisibleTimes;</mo><mrow><mo>(</mo><mrow><msub><mover>'\
        '<mi mathvariant="bold">i</mi><mo>^</mo></mover>'\
        '<mi mathvariant="bold">A</mi></msub><mo>&#xD7;</mo><msub><mover>'\
        '<mi mathvariant="bold">j</mi><mo>^</mo></mover>'\
        '<mi mathvariant="bold">A</mi></msub></mrow><mo>)</mo></mrow></mrow>'
    assert mathml(Cross(x*ACS.i, ACS.j), printer='presentation') == \
        '<mrow><mo>-</mo><mrow><msub><mover><mi mathvariant="bold">j</mi>'\
        '<mo>^</mo></mover><mi mathvariant="bold">A</mi></msub>'\
        '<mo>&#xD7;</mo><mrow><mo>(</mo><mrow><mrow><mo>(</mo><mi>x</mi><mo>)</mo></mrow>'\
        '<mo>&InvisibleTimes;</mo><msub><mover><mi mathvariant="bold">i</mi>'\
        '<mo>^</mo></mover><mi mathvariant="bold">A</mi></msub></mrow>'\
        '<mo>)</mo></mrow></mrow></mrow>'
    assert mathml(Curl(3*ACS.x*ACS.j), printer='presentation') == \
        '<mrow><mo>&#x2207;</mo><mo>&#xD7;</mo><mrow><mo>(</mo><mrow><mrow><mo>(</mo>'\
        '<mrow><mn>3</mn><mo>&InvisibleTimes;</mo><msub><mi mathvariant="bold">x</mi>'\
        '<mi mathvariant="bold">A</mi></msub></mrow><mo>)</mo></mrow><mo>&InvisibleTimes;</mo>'\
        '<msub><mover><mi mathvariant="bold">j</mi><mo>^</mo></mover>'\
        '<mi mathvariant="bold">A</mi></msub></mrow><mo>)</mo></mrow></mrow>'
    assert mathml(Curl(3*x*ACS.x*ACS.j), printer='presentation') == \
        '<mrow><mo>&#x2207;</mo><mo>&#xD7;</mo><mrow><mo>(</mo><mrow><mrow><mo>(</mo><mrow>'\
        '<mn>3</mn><mo>&InvisibleTimes;</mo><msub><mi mathvariant="bold">x'\
        '</mi><mi mathvariant="bold">A</mi></msub><mo>&InvisibleTimes;</mo>'\
        '<mi>x</mi></mrow><mo>)</mo></mrow><mo>&InvisibleTimes;</mo><msub><mover>'\
        '<mi mathvariant="bold">j</mi><mo>^</mo></mover>'\
        '<mi mathvariant="bold">A</mi></msub></mrow><mo>)</mo></mrow></mrow>'
    assert mathml(x*Curl(3*ACS.x*ACS.j), printer='presentation') == \
        '<mrow><mi>x</mi><mo>&InvisibleTimes;</mo><mrow><mo>(</mo><mrow><mo>&#x2207;</mo>'\
        '<mo>&#xD7;</mo><mrow><mo>(</mo><mrow><mrow><mo>(</mo><mrow><mn>3</mn>'\
        '<mo>&InvisibleTimes;</mo><msub><mi mathvariant="bold">x</mi>'\
        '<mi mathvariant="bold">A</mi></msub></mrow><mo>)</mo></mrow>'\
        '<mo>&InvisibleTimes;</mo><msub><mover><mi mathvariant="bold">j</mi>'\
        '<mo>^</mo></mover><mi mathvariant="bold">A</mi></msub></mrow><mo>)</mo>'\
        '</mrow></mrow><mo>)</mo></mrow></mrow>'
    assert mathml(Curl(3*x*ACS.x*ACS.j + ACS.i), printer='presentation') == \
        '<mrow><mo>&#x2207;</mo><mo>&#xD7;</mo><mrow><mo>(</mo><mrow><msub><mover>'\
        '<mi mathvariant="bold">i</mi><mo>^</mo></mover>'\
        '<mi mathvariant="bold">A</mi></msub><mo>+</mo><mrow><mo>(</mo><mrow>'\
        '<mn>3</mn><mo>&InvisibleTimes;</mo><msub><mi mathvariant="bold">x'\
        '</mi><mi mathvariant="bold">A</mi></msub><mo>&InvisibleTimes;</mo>'\
        '<mi>x</mi></mrow><mo>)</mo></mrow><mo>&InvisibleTimes;</mo><msub><mover>'\
        '<mi mathvariant="bold">j</mi><mo>^</mo></mover>'\
        '<mi mathvariant="bold">A</mi></msub></mrow><mo>)</mo></mrow></mrow>'
    assert mathml(Divergence(3*ACS.x*ACS.j), printer='presentation') == \
        '<mrow><mo>&#x2207;</mo><mo>&#xB7;</mo><mrow><mo>(</mo><mrow><mrow><mo>(</mo>'\
        '<mrow><mn>3</mn><mo>&InvisibleTimes;</mo><msub><mi mathvariant="bold">x'\
        '</mi><mi mathvariant="bold">A</mi></msub></mrow><mo>)</mo></mrow>'\
        '<mo>&InvisibleTimes;</mo><msub><mover><mi mathvariant="bold">j</mi>'\
        '<mo>^</mo></mover><mi mathvariant="bold">A</mi></msub></mrow><mo>)</mo></mrow></mrow>'
    assert mathml(x*Divergence(3*ACS.x*ACS.j), printer='presentation') == \
        '<mrow><mi>x</mi><mo>&InvisibleTimes;</mo><mrow><mo>(</mo><mrow><mo>&#x2207;</mo>'\
        '<mo>&#xB7;</mo><mrow><mo>(</mo><mrow><mrow><mo>(</mo><mrow><mn>3</mn>'\
        '<mo>&InvisibleTimes;</mo><msub><mi mathvariant="bold">x</mi>'\
        '<mi mathvariant="bold">A</mi></msub></mrow><mo>)</mo></mrow>'\
        '<mo>&InvisibleTimes;</mo><msub><mover><mi mathvariant="bold">j</mi>'\
        '<mo>^</mo></mover><mi mathvariant="bold">A</mi></msub></mrow>'\
        '<mo>)</mo></mrow></mrow><mo>)</mo></mrow></mrow>'
    assert mathml(Divergence(3*x*ACS.x*ACS.j + ACS.i), printer='presentation') == \
        '<mrow><mo>&#x2207;</mo><mo>&#xB7;</mo><mrow><mo>(</mo><mrow><msub><mover>'\
        '<mi mathvariant="bold">i</mi><mo>^</mo></mover>'\
        '<mi mathvariant="bold">A</mi></msub><mo>+</mo><mrow><mo>(</mo><mrow>'\
        '<mn>3</mn><mo>&InvisibleTimes;</mo><msub>'\
        '<mi mathvariant="bold">x</mi><mi mathvariant="bold">A</mi></msub>'\
        '<mo>&InvisibleTimes;</mo><mi>x</mi></mrow><mo>)</mo></mrow>'\
        '<mo>&InvisibleTimes;</mo><msub><mover><mi mathvariant="bold">j</mi>'\
        '<mo>^</mo></mover><mi mathvariant="bold">A</mi></msub></mrow>'\
        '<mo>)</mo></mrow></mrow>'
    assert mathml(Dot(ACS.i, ACS.j*ACS.x*3+ACS.k), printer='presentation') == \
        '<mrow><msub><mover><mi mathvariant="bold">i</mi><mo>^</mo></mover>'\
        '<mi mathvariant="bold">A</mi></msub><mo>&#xB7;</mo><mrow><mo>(</mo><mrow>'\
        '<mrow><mo>(</mo><mrow><mn>3</mn><mo>&InvisibleTimes;</mo><msub>'\
        '<mi mathvariant="bold">x</mi><mi mathvariant="bold">A</mi></msub>'\
        '</mrow><mo>)</mo></mrow><mo>&InvisibleTimes;</mo><msub><mover>'\
        '<mi mathvariant="bold">j</mi><mo>^</mo></mover>'\
        '<mi mathvariant="bold">A</mi></msub><mo>+</mo><msub><mover>'\
        '<mi mathvariant="bold">k</mi><mo>^</mo></mover>'\
        '<mi mathvariant="bold">A</mi></msub></mrow><mo>)</mo></mrow></mrow>'
    assert mathml(Dot(ACS.i, ACS.j), printer='presentation') == \
        '<mrow><msub><mover><mi mathvariant="bold">i</mi><mo>^</mo></mover>'\
        '<mi mathvariant="bold">A</mi></msub><mo>&#xB7;</mo><msub><mover>'\
        '<mi mathvariant="bold">j</mi><mo>^</mo></mover>'\
        '<mi mathvariant="bold">A</mi></msub></mrow>'
    assert mathml(Dot(x*ACS.i, ACS.j), printer='presentation') == \
        '<mrow><msub><mover><mi mathvariant="bold">j</mi><mo>^</mo></mover>'\
        '<mi mathvariant="bold">A</mi></msub><mo>&#xB7;</mo><mrow><mo>(</mo><mrow>'\
        '<mrow><mo>(</mo><mi>x</mi><mo>)</mo></mrow><mo>&InvisibleTimes;</mo><msub><mover>'\
        '<mi mathvariant="bold">i</mi><mo>^</mo></mover>'\
        '<mi mathvariant="bold">A</mi></msub></mrow><mo>)</mo></mrow></mrow>'
    assert mathml(x*Dot(ACS.i, ACS.j), printer='presentation') == \
        '<mrow><mi>x</mi><mo>&InvisibleTimes;</mo><mrow><mo>(</mo><mrow><msub><mover>'\
        '<mi mathvariant="bold">i</mi><mo>^</mo></mover>'\
        '<mi mathvariant="bold">A</mi></msub><mo>&#xB7;</mo><msub><mover>'\
        '<mi mathvariant="bold">j</mi><mo>^</mo></mover>'\
        '<mi mathvariant="bold">A</mi></msub></mrow><mo>)</mo></mrow></mrow>'
    assert mathml(Gradient(ACS.x), printer='presentation') == \
        '<mrow><mo>&#x2207;</mo><msub><mi mathvariant="bold">x</mi>'\
        '<mi mathvariant="bold">A</mi></msub></mrow>'
    assert mathml(Gradient(ACS.x + 3*ACS.y), printer='presentation') == \
        '<mrow><mo>&#x2207;</mo><mrow><mo>(</mo><mrow><msub><mi mathvariant="bold">'\
        'x</mi><mi mathvariant="bold">A</mi></msub><mo>+</mo><mrow><mn>3</mn>'\
        '<mo>&InvisibleTimes;</mo><msub><mi mathvariant="bold">y</mi>'\
        '<mi mathvariant="bold">A</mi></msub></mrow></mrow><mo>)</mo></mrow></mrow>'
    assert mathml(x*Gradient(ACS.x), printer='presentation') == \
        '<mrow><mi>x</mi><mo>&InvisibleTimes;</mo><mrow><mo>(</mo><mrow><mo>&#x2207;</mo>'\
        '<msub><mi mathvariant="bold">x</mi><mi mathvariant="bold">A</mi>'\
        '</msub></mrow><mo>)</mo></mrow></mrow>'
    assert mathml(Gradient(x*ACS.x), printer='presentation') == \
        '<mrow><mo>&#x2207;</mo><mrow><mo>(</mo><mrow><msub><mi mathvariant="bold">'\
        'x</mi><mi mathvariant="bold">A</mi></msub><mo>&InvisibleTimes;</mo>'\
        '<mi>x</mi></mrow><mo>)</mo></mrow></mrow>'
    assert mathml(Cross(ACS.z, ACS.x), printer='presentation') == \
        '<mrow><mo>-</mo><mrow><msub><mi mathvariant="bold">x</mi>'\
        '<mi mathvariant="bold">A</mi></msub><mo>&#xD7;</mo><msub>'\
        '<mi mathvariant="bold">z</mi><mi mathvariant="bold">A</mi></msub></mrow></mrow>'
    assert mathml(Laplacian(ACS.x), printer='presentation') == \
        '<mrow><mo>&#x2206;</mo><msub><mi mathvariant="bold">x</mi>'\
        '<mi mathvariant="bold">A</mi></msub></mrow>'
    assert mathml(Laplacian(ACS.x + 3*ACS.y), printer='presentation') == \
        '<mrow><mo>&#x2206;</mo><mrow><mo>(</mo><mrow><msub><mi mathvariant="bold">'\
        'x</mi><mi mathvariant="bold">A</mi></msub><mo>+</mo><mrow><mn>3</mn>'\
        '<mo>&InvisibleTimes;</mo><msub><mi mathvariant="bold">y</mi>'\
        '<mi mathvariant="bold">A</mi></msub></mrow></mrow><mo>)</mo></mrow></mrow>'
    assert mathml(x*Laplacian(ACS.x), printer='presentation') == \
        '<mrow><mi>x</mi><mo>&InvisibleTimes;</mo><mrow><mo>(</mo><mrow><mo>&#x2206;</mo>'\
        '<msub><mi mathvariant="bold">x</mi><mi mathvariant="bold">A</mi>'\
        '</msub></mrow><mo>)</mo></mrow></mrow>'
    assert mathml(Laplacian(x*ACS.x), printer='presentation') == \
        '<mrow><mo>&#x2206;</mo><mrow><mo>(</mo><mrow><msub><mi mathvariant="bold">'\
        'x</mi><mi mathvariant="bold">A</mi></msub><mo>&InvisibleTimes;</mo>'\
        '<mi>x</mi></mrow><mo>)</mo></mrow></mrow>'

@XFAIL
def test_vector_cross_xfail():
    ACS = CoordSys3D('A')
    assert mathml(Cross(ACS.x, ACS.z) + Cross(ACS.z, ACS.x), printer='presentation') == \
        '<mover><mi mathvariant="bold">0</mi><mo>^</mo></mover>'

def test_print_elliptic_f():
    assert mathml(elliptic_f(x, y), printer = 'presentation') == \
        '<mrow><mi>&#x1d5a5;</mi><mrow><mo>(</mo><mi>x</mi><mo>|</mo><mi>y</mi><mo>)</mo></mrow></mrow>'
    assert mathml(elliptic_f(x/y, y), printer = 'presentation') == \
        '<mrow><mi>&#x1d5a5;</mi><mrow><mo>(</mo><mrow><mfrac><mi>x</mi><mi>y</mi></mfrac></mrow><mo>|</mo><mi>y</mi><mo>)</mo></mrow></mrow>'

def test_print_elliptic_e():
    assert mathml(elliptic_e(x), printer = 'presentation') == \
        '<mrow><mi>&#x1d5a4;</mi><mrow><mo>(</mo><mi>x</mi><mo>)</mo></mrow></mrow>'
    assert mathml(elliptic_e(x, y), printer = 'presentation') == \
        '<mrow><mi>&#x1d5a4;</mi><mrow><mo>(</mo><mi>x</mi><mo>|</mo><mi>y</mi><mo>)</mo></mrow></mrow>'

def test_print_elliptic_pi():
    assert mathml(elliptic_pi(x, y), printer = 'presentation') == \
        '<mrow><mi>&#x1d6f1;</mi><mrow><mo>(</mo><mi>x</mi><mo>|</mo><mi>y</mi><mo>)</mo></mrow></mrow>'
    assert mathml(elliptic_pi(x, y, z), printer = 'presentation') == \
        '<mrow><mi>&#x1d6f1;</mi><mrow><mo>(</mo><mi>x</mi><mo>;</mo><mi>y</mi><mo>|</mo><mi>z</mi><mo>)</mo></mrow></mrow>'

def test_print_Ei():
    assert mathml(Ei(x), printer = 'presentation') == \
        '<mrow><mi>Ei</mi><mrow><mo>(</mo><mi>x</mi><mo>)</mo></mrow></mrow>'
    assert mathml(Ei(x**y), printer = 'presentation') == \
        '<mrow><mi>Ei</mi><mrow><mo>(</mo><msup><mi>x</mi><mi>y</mi></msup><mo>)</mo></mrow></mrow>'

def test_print_expint():
    assert mathml(expint(x, y), printer = 'presentation') == \
        '<mrow><msub><mo>E</mo><mi>x</mi></msub><mrow><mo>(</mo><mi>y</mi><mo>)</mo></mrow></mrow>'
    assert mathml(expint(IndexedBase(x)[1], IndexedBase(x)[2]), printer = 'presentation') == \
        '<mrow><msub><mo>E</mo><msub><mi>x</mi><mn>1</mn></msub></msub><mrow><mo>(</mo><msub><mi>x</mi><mn>2</mn></msub><mo>)</mo></mrow></mrow>'

def test_print_jacobi():
    assert mathml(jacobi(n, a, b, x), printer = 'presentation') == \
        '<mrow><msubsup><mo>P</mo><mi>n</mi><mrow><mo>(</mo><mi>a</mi><mo>,</mo><mi>b</mi><mo>)</mo></mrow></msubsup><mrow><mo>(</mo><mi>x</mi><mo>)</mo></mrow></mrow>'

def test_print_gegenbauer():
    assert mathml(gegenbauer(n, a, x), printer = 'presentation') == \
        '<mrow><msubsup><mo>C</mo><mi>n</mi><mrow><mo>(</mo><mi>a</mi><mo>)</mo></mrow></msubsup><mrow><mo>(</mo><mi>x</mi><mo>)</mo></mrow></mrow>'

def test_print_chebyshevt():
    assert mathml(chebyshevt(n, x), printer = 'presentation') == \
        '<mrow><msub><mo>T</mo><mi>n</mi></msub><mrow><mo>(</mo><mi>x</mi><mo>)</mo></mrow></mrow>'

def test_print_chebyshevu():
    assert mathml(chebyshevu(n, x), printer = 'presentation') == \
        '<mrow><msub><mo>U</mo><mi>n</mi></msub><mrow><mo>(</mo><mi>x</mi><mo>)</mo></mrow></mrow>'

def test_print_legendre():
    assert mathml(legendre(n, x), printer = 'presentation') == \
        '<mrow><msub><mo>P</mo><mi>n</mi></msub><mrow><mo>(</mo><mi>x</mi><mo>)</mo></mrow></mrow>'

def test_print_assoc_legendre():
    assert mathml(assoc_legendre(n, a, x), printer = 'presentation') == \
        '<mrow><msubsup><mo>P</mo><mi>n</mi><mrow><mo>(</mo><mi>a</mi><mo>)</mo></mrow></msubsup><mrow><mo>(</mo><mi>x</mi><mo>)</mo></mrow></mrow>'

def test_print_laguerre():
    assert mathml(laguerre(n, x), printer = 'presentation') == \
        '<mrow><msub><mo>L</mo><mi>n</mi></msub><mrow><mo>(</mo><mi>x</mi><mo>)</mo></mrow></mrow>'

def test_print_assoc_laguerre():
    assert mathml(assoc_laguerre(n, a, x), printer = 'presentation') == \
        '<mrow><msubsup><mo>L</mo><mi>n</mi><mrow><mo>(</mo><mi>a</mi><mo>)</mo></mrow></msubsup><mrow><mo>(</mo><mi>x</mi><mo>)</mo></mrow></mrow>'

def test_print_hermite():
    assert mathml(hermite(n, x), printer = 'presentation') == \
        '<mrow><msub><mo>H</mo><mi>n</mi></msub><mrow><mo>(</mo><mi>x</mi><mo>)</mo></mrow></mrow>'

def test_mathml_SingularityFunction():
    assert mathml(SingularityFunction(x, 4, 5), printer='presentation') == \
        '<msup><mrow><mo>&#10216;</mo><mrow><mi>x</mi><mo>-</mo><mn>4</mn></mrow><mo>&#10217;</mo></mrow><mn>5</mn></msup>'
    assert mathml(SingularityFunction(x, -3, 4), printer='presentation') == \
        '<msup><mrow><mo>&#10216;</mo><mrow><mi>x</mi><mo>+</mo><mn>3</mn></mrow><mo>&#10217;</mo></mrow><mn>4</mn></msup>'
    assert mathml(SingularityFunction(x, 0, 4), printer='presentation') == \
        '<msup><mrow><mo>&#10216;</mo><mi>x</mi><mo>&#10217;</mo></mrow><mn>4</mn></msup>'
    assert mathml(SingularityFunction(x, a, n), printer='presentation') == \
        '<msup><mrow><mo>&#10216;</mo><mrow><mrow><mo>-</mo><mi>a</mi></mrow><mo>+</mo><mi>x</mi></mrow><mo>&#10217;</mo></mrow><mi>n</mi></msup>'
    assert mathml(SingularityFunction(x, 4, -2), printer='presentation') == \
        '<msup><mrow><mo>&#10216;</mo><mrow><mi>x</mi><mo>-</mo><mn>4</mn></mrow><mo>&#10217;</mo></mrow><mn>-2</mn></msup>'
    assert mathml(SingularityFunction(x, 4, -1), printer='presentation') == \
        '<msup><mrow><mo>&#10216;</mo><mrow><mi>x</mi><mo>-</mo><mn>4</mn></mrow><mo>&#10217;</mo></mrow><mn>-1</mn></msup>'


def test_mathml_matrix_functions():
    from sympy.matrices import Adjoint, Inverse, Transpose
    X = MatrixSymbol('X', 2, 2)
    Y = MatrixSymbol('Y', 2, 2)
    assert mathml(Adjoint(X), printer='presentation') == \
        '<msup><mi>X</mi><mo>&#x2020;</mo></msup>'
    assert mathml(Adjoint(X + Y), printer='presentation') == \
        '<msup><mrow><mo>(</mo><mrow><mi>X</mi><mo>+</mo><mi>Y</mi></mrow><mo>)</mo></mrow><mo>&#x2020;</mo></msup>'
    assert mathml(Adjoint(X) + Adjoint(Y), printer='presentation') == \
        '<mrow><msup><mi>X</mi><mo>&#x2020;</mo></msup><mo>+</mo><msup>' \
        '<mi>Y</mi><mo>&#x2020;</mo></msup></mrow>'
    assert mathml(Adjoint(X*Y), printer='presentation') == \
        '<msup><mrow><mo>(</mo><mrow><mi>X</mi><mo>&InvisibleTimes;</mo>' \
        '<mi>Y</mi></mrow><mo>)</mo></mrow><mo>&#x2020;</mo></msup>'
    assert mathml(Adjoint(Y)*Adjoint(X), printer='presentation') == \
        '<mrow><msup><mi>Y</mi><mo>&#x2020;</mo></msup><mo>&InvisibleTimes;' \
        '</mo><msup><mi>X</mi><mo>&#x2020;</mo></msup></mrow>'
    assert mathml(Adjoint(X**2), printer='presentation') == \
        '<msup><mrow><mo>(</mo><msup><mi>X</mi><mn>2</mn></msup><mo>)</mo></mrow><mo>&#x2020;</mo></msup>'
    assert mathml(Adjoint(X)**2, printer='presentation') == \
        '<msup><mrow><mo>(</mo><msup><mi>X</mi><mo>&#x2020;</mo></msup><mo>)</mo></mrow><mn>2</mn></msup>'
    assert mathml(Adjoint(Inverse(X)), printer='presentation') == \
        '<msup><mrow><mo>(</mo><msup><mi>X</mi><mn>-1</mn></msup><mo>)</mo></mrow><mo>&#x2020;</mo></msup>'
    assert mathml(Inverse(Adjoint(X)), printer='presentation') == \
        '<msup><mrow><mo>(</mo><msup><mi>X</mi><mo>&#x2020;</mo></msup><mo>)</mo></mrow><mn>-1</mn></msup>'
    assert mathml(Adjoint(Transpose(X)), printer='presentation') == \
        '<msup><mrow><mo>(</mo><msup><mi>X</mi><mo>T</mo></msup><mo>)</mo></mrow><mo>&#x2020;</mo></msup>'
    assert mathml(Transpose(Adjoint(X)), printer='presentation') ==  \
        '<msup><mrow><mo>(</mo><msup><mi>X</mi><mo>&#x2020;</mo></msup><mo>)</mo></mrow><mo>T</mo></msup>'
    assert mathml(Transpose(Adjoint(X) + Y), printer='presentation') ==  \
        '<msup><mrow><mo>(</mo><mrow><msup><mi>X</mi><mo>&#x2020;</mo></msup>' \
        '<mo>+</mo><mi>Y</mi></mrow><mo>)</mo></mrow><mo>T</mo></msup>'
    assert mathml(Transpose(X), printer='presentation') == \
        '<msup><mi>X</mi><mo>T</mo></msup>'
    assert mathml(Transpose(X + Y), printer='presentation') == \
        '<msup><mrow><mo>(</mo><mrow><mi>X</mi><mo>+</mo><mi>Y</mi></mrow><mo>)</mo></mrow><mo>T</mo></msup>'


def test_mathml_special_matrices():
    from sympy.matrices import Identity, ZeroMatrix, OneMatrix
    assert mathml(Identity(4), printer='presentation') == '<mi>&#x1D540;</mi>'
    assert mathml(ZeroMatrix(2, 2), printer='presentation') == '<mn>&#x1D7D8</mn>'
    assert mathml(OneMatrix(2, 2), printer='presentation') == '<mn>&#x1D7D9</mn>'

def test_mathml_piecewise():
    from sympy.functions.elementary.piecewise import Piecewise
    # Content MathML
    assert mathml(Piecewise((x, x <= 1), (x**2, True))) == \
        '<piecewise><piece><ci>x</ci><apply><leq/><ci>x</ci><cn>1</cn></apply></piece><otherwise><apply><power/><ci>x</ci><cn>2</cn></apply></otherwise></piecewise>'

    raises(ValueError, lambda: mathml(Piecewise((x, x <= 1))))


def test_issue_17857():
    assert mathml(Range(-oo, oo), printer='presentation') == \
        '<mrow><mo>{</mo><mi>&#8230;</mi><mo>,</mo><mn>-1</mn><mo>,</mo><mn>0</mn><mo>,</mo><mn>1</mn><mo>,</mo><mi>&#8230;</mi><mo>}</mo></mrow>'
    assert mathml(Range(oo, -oo, -1), printer='presentation') == \
        '<mrow><mo>{</mo><mi>&#8230;</mi><mo>,</mo><mn>1</mn><mo>,</mo><mn>0</mn><mo>,</mo><mn>-1</mn><mo>,</mo><mi>&#8230;</mi><mo>}</mo></mrow>'


def test_float_roundtrip():
    x = sympify(0.8975979010256552)
    y = float(mp.doprint(x).strip('</cn>'))
    assert x == y


def test_content_mathml_disable_split_super_sub():
    mp = MathMLContentPrinter()
    assert mp.doprint(Symbol('u_b')) == '<ci><mml:msub><mml:mi>u</mml:mi><mml:mi>b</mml:mi></mml:msub></ci>'
    mp = MathMLContentPrinter({'disable_split_super_sub': False})
    assert mp.doprint(Symbol('u_b')) == '<ci><mml:msub><mml:mi>u</mml:mi><mml:mi>b</mml:mi></mml:msub></ci>'
    mp = MathMLContentPrinter({'disable_split_super_sub': True})
    assert mp.doprint(Symbol('u_b')) == '<ci>u_b</ci>'

def test_presentation_mathml_disable_split_super_sub():
    mpp = MathMLPresentationPrinter()
    assert mpp.doprint(Symbol('u_b')) == '<msub><mi>u</mi><mi>b</mi></msub>'
    mpp = MathMLPresentationPrinter({'disable_split_super_sub': False})
    assert mpp.doprint(Symbol('u_b')) == '<msub><mi>u</mi><mi>b</mi></msub>'
    mpp = MathMLPresentationPrinter({'disable_split_super_sub': True})
    assert mpp.doprint(Symbol('u_b')) == '<mi>u_b</mi>'
