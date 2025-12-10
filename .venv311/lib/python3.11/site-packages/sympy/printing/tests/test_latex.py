from sympy import MatAdd, MatMul, Array
from sympy.algebras.quaternion import Quaternion
from sympy.calculus.accumulationbounds import AccumBounds
from sympy.combinatorics.permutations import Cycle, Permutation, AppliedPermutation
from sympy.concrete.products import Product
from sympy.concrete.summations import Sum
from sympy.core.containers import Tuple, Dict
from sympy.core.expr import UnevaluatedExpr
from sympy.core.function import (Derivative, Function, Lambda, Subs, diff)
from sympy.core.mod import Mod
from sympy.core.mul import Mul
from sympy.core.numbers import (AlgebraicNumber, Float, I, Integer, Rational, oo, pi)
from sympy.core.parameters import evaluate
from sympy.core.power import Pow
from sympy.core.relational import Eq, Ne
from sympy.core.singleton import S
from sympy.core.symbol import (Symbol, Wild, symbols)
from sympy.functions.combinatorial.factorials import (FallingFactorial, RisingFactorial, binomial, factorial, factorial2, subfactorial)
from sympy.functions.combinatorial.numbers import (bernoulli, bell, catalan, euler, genocchi,
                                                   lucas, fibonacci, tribonacci, divisor_sigma, udivisor_sigma,
                                                   mobius, primenu, primeomega,
                                                   totient, reduced_totient)
from sympy.functions.elementary.complexes import (Abs, arg, conjugate, im, polar_lift, re)
from sympy.functions.elementary.exponential import (LambertW, exp, log)
from sympy.functions.elementary.hyperbolic import (asinh, coth)
from sympy.functions.elementary.integers import (ceiling, floor, frac)
from sympy.functions.elementary.miscellaneous import (Max, Min, root, sqrt)
from sympy.functions.elementary.piecewise import Piecewise
from sympy.functions.elementary.trigonometric import (acsc, asin, cos, cot, sin, tan)
from sympy.functions.special.beta_functions import beta
from sympy.functions.special.delta_functions import (DiracDelta, Heaviside)
from sympy.functions.special.elliptic_integrals import (elliptic_e, elliptic_f, elliptic_k, elliptic_pi)
from sympy.functions.special.error_functions import (Chi, Ci, Ei, Shi, Si, expint)
from sympy.functions.special.gamma_functions import (gamma, uppergamma)
from sympy.functions.special.hyper import (hyper, meijerg)
from sympy.functions.special.mathieu_functions import (mathieuc, mathieucprime, mathieus, mathieusprime)
from sympy.functions.special.polynomials import (assoc_laguerre, assoc_legendre, chebyshevt, chebyshevu, gegenbauer, hermite, jacobi, laguerre, legendre)
from sympy.functions.special.singularity_functions import SingularityFunction
from sympy.functions.special.spherical_harmonics import (Ynm, Znm)
from sympy.functions.special.tensor_functions import (KroneckerDelta, LeviCivita)
from sympy.functions.special.zeta_functions import (dirichlet_eta, lerchphi, polylog, stieltjes, zeta)
from sympy.integrals.integrals import Integral
from sympy.integrals.transforms import (CosineTransform, FourierTransform, InverseCosineTransform, InverseFourierTransform, InverseLaplaceTransform, InverseMellinTransform, InverseSineTransform, LaplaceTransform, MellinTransform, SineTransform)
from sympy.logic import Implies
from sympy.logic.boolalg import (And, Or, Xor, Equivalent, false, Not, true)
from sympy.matrices.dense import Matrix
from sympy.matrices.expressions.kronecker import KroneckerProduct
from sympy.matrices.expressions.matexpr import MatrixSymbol
from sympy.matrices.expressions.permutation import PermutationMatrix
from sympy.matrices.expressions.slice import MatrixSlice
from sympy.matrices.expressions.dotproduct import DotProduct
from sympy.physics.control.lti import TransferFunction, Series, Parallel, Feedback, TransferFunctionMatrix, MIMOSeries, MIMOParallel, MIMOFeedback
from sympy.physics.quantum import Commutator, Operator
from sympy.physics.quantum.trace import Tr
from sympy.physics.units import meter, gibibyte, gram, microgram, second, milli, micro
from sympy.polys.domains.integerring import ZZ
from sympy.polys.fields import field
from sympy.polys.polytools import Poly
from sympy.polys.rings import ring
from sympy.polys.rootoftools import (RootSum, rootof)
from sympy.series.formal import fps
from sympy.series.fourier import fourier_series
from sympy.series.limits import Limit
from sympy.series.order import Order
from sympy.series.sequences import (SeqAdd, SeqFormula, SeqMul, SeqPer)
from sympy.sets.conditionset import ConditionSet
from sympy.sets.contains import Contains
from sympy.sets.fancysets import (ComplexRegion, ImageSet, Range)
from sympy.sets.ordinals import Ordinal, OrdinalOmega, OmegaPower
from sympy.sets.powerset import PowerSet
from sympy.sets.sets import (FiniteSet, Interval, Union, Intersection, Complement, SymmetricDifference, ProductSet)
from sympy.sets.setexpr import SetExpr
from sympy.stats.crv_types import Normal
from sympy.stats.symbolic_probability import (Covariance, Expectation,
                                              Probability, Variance)
from sympy.tensor.array import (ImmutableDenseNDimArray,
                                ImmutableSparseNDimArray,
                                MutableSparseNDimArray,
                                MutableDenseNDimArray,
                                tensorproduct)
from sympy.tensor.array.expressions.array_expressions import ArraySymbol, ArrayElement
from sympy.tensor.indexed import (Idx, Indexed, IndexedBase)
from sympy.tensor.toperators import PartialDerivative
from sympy.vector import CoordSys3D, Cross, Curl, Dot, Divergence, Gradient, Laplacian


from sympy.testing.pytest import (XFAIL, raises, _both_exp_pow,
                                  warns_deprecated_sympy)
from sympy.printing.latex import (latex, translate, greek_letters_set,
                                  tex_greek_dictionary, multiline_latex,
                                  latex_escape, LatexPrinter)

import sympy as sym

from sympy.abc import mu, tau


class lowergamma(sym.lowergamma):
    pass   # testing notation inheritance by a subclass with same name


x, y, z, t, w, a, b, c, s, p = symbols('x y z t w a b c s p')
k, m, n = symbols('k m n', integer=True)


def test_printmethod():
    class R(Abs):
        def _latex(self, printer):
            return "foo(%s)" % printer._print(self.args[0])
    assert latex(R(x)) == r"foo(x)"

    class R(Abs):
        def _latex(self, printer):
            return "foo"
    assert latex(R(x)) == r"foo"


def test_latex_basic():
    assert latex(1 + x) == r"x + 1"
    assert latex(x**2) == r"x^{2}"
    assert latex(x**(1 + x)) == r"x^{x + 1}"
    assert latex(x**3 + x + 1 + x**2) == r"x^{3} + x^{2} + x + 1"

    assert latex(2*x*y) == r"2 x y"
    assert latex(2*x*y, mul_symbol='dot') == r"2 \cdot x \cdot y"
    assert latex(3*x**2*y, mul_symbol='\\,') == r"3\,x^{2}\,y"
    assert latex(1.5*3**x, mul_symbol='\\,') == r"1.5 \cdot 3^{x}"

    assert latex(x**S.Half**5) == r"\sqrt[32]{x}"
    assert latex(Mul(S.Half, x**2, -5, evaluate=False)) == r"\frac{1}{2} x^{2} \left(-5\right)"
    assert latex(Mul(S.Half, x**2, 5, evaluate=False)) == r"\frac{1}{2} x^{2} \cdot 5"
    assert latex(Mul(-5, -5, evaluate=False)) == r"\left(-5\right) \left(-5\right)"
    assert latex(Mul(5, -5, evaluate=False)) == r"5 \left(-5\right)"
    assert latex(Mul(S.Half, -5, S.Half, evaluate=False)) == r"\frac{1}{2} \left(-5\right) \frac{1}{2}"
    assert latex(Mul(5, I, 5, evaluate=False)) == r"5 i 5"
    assert latex(Mul(5, I, -5, evaluate=False)) == r"5 i \left(-5\right)"
    assert latex(Mul(Pow(x, 2), S.Half*x + 1)) == r"x^{2} \left(\frac{x}{2} + 1\right)"
    assert latex(Mul(Pow(x, 3), Rational(2, 3)*x + 1)) == r"x^{3} \left(\frac{2 x}{3} + 1\right)"
    assert latex(Mul(Pow(x, 11), 2*x + 1)) == r"x^{11} \left(2 x + 1\right)"

    assert latex(Mul(0, 1, evaluate=False)) == r'0 \cdot 1'
    assert latex(Mul(1, 0, evaluate=False)) == r'1 \cdot 0'
    assert latex(Mul(1, 1, evaluate=False)) == r'1 \cdot 1'
    assert latex(Mul(-1, 1, evaluate=False)) == r'\left(-1\right) 1'
    assert latex(Mul(1, 1, 1, evaluate=False)) == r'1 \cdot 1 \cdot 1'
    assert latex(Mul(1, 2, evaluate=False)) == r'1 \cdot 2'
    assert latex(Mul(1, S.Half, evaluate=False)) == r'1 \cdot \frac{1}{2}'
    assert latex(Mul(1, 1, S.Half, evaluate=False)) == \
        r'1 \cdot 1 \cdot \frac{1}{2}'
    assert latex(Mul(1, 1, 2, 3, x, evaluate=False)) == \
        r'1 \cdot 1 \cdot 2 \cdot 3 x'
    assert latex(Mul(1, -1, evaluate=False)) == r'1 \left(-1\right)'
    assert latex(Mul(4, 3, 2, 1, 0, y, x, evaluate=False)) == \
        r'4 \cdot 3 \cdot 2 \cdot 1 \cdot 0 y x'
    assert latex(Mul(4, 3, 2, 1+z, 0, y, x, evaluate=False)) == \
        r'4 \cdot 3 \cdot 2 \left(z + 1\right) 0 y x'
    assert latex(Mul(Rational(2, 3), Rational(5, 7), evaluate=False)) == \
        r'\frac{2}{3} \cdot \frac{5}{7}'

    assert latex(1/x) == r"\frac{1}{x}"
    assert latex(1/x, fold_short_frac=True) == r"1 / x"
    assert latex(-S(3)/2) == r"- \frac{3}{2}"
    assert latex(-S(3)/2, fold_short_frac=True) == r"- 3 / 2"
    assert latex(1/x**2) == r"\frac{1}{x^{2}}"
    assert latex(1/(x + y)/2) == r"\frac{1}{2 \left(x + y\right)}"
    assert latex(x/2) == r"\frac{x}{2}"
    assert latex(x/2, fold_short_frac=True) == r"x / 2"
    assert latex((x + y)/(2*x)) == r"\frac{x + y}{2 x}"
    assert latex((x + y)/(2*x), fold_short_frac=True) == \
        r"\left(x + y\right) / 2 x"
    assert latex((x + y)/(2*x), long_frac_ratio=0) == \
        r"\frac{1}{2 x} \left(x + y\right)"
    assert latex((x + y)/x) == r"\frac{x + y}{x}"
    assert latex((x + y)/x, long_frac_ratio=3) == r"\frac{x + y}{x}"
    assert latex((2*sqrt(2)*x)/3) == r"\frac{2 \sqrt{2} x}{3}"
    assert latex((2*sqrt(2)*x)/3, long_frac_ratio=2) == \
        r"\frac{2 x}{3} \sqrt{2}"
    assert latex(binomial(x, y)) == r"{\binom{x}{y}}"

    x_star = Symbol('x^*')
    f = Function('f')
    assert latex(x_star**2) == r"\left(x^{*}\right)^{2}"
    assert latex(x_star**2, parenthesize_super=False) == r"{x^{*}}^{2}"
    assert latex(Derivative(f(x_star), x_star,2)) == r"\frac{d^{2}}{d \left(x^{*}\right)^{2}} f{\left(x^{*} \right)}"
    assert latex(Derivative(f(x_star), x_star,2), parenthesize_super=False) == r"\frac{d^{2}}{d {x^{*}}^{2}} f{\left(x^{*} \right)}"

    assert latex(2*Integral(x, x)/3) == r"\frac{2 \int x\, dx}{3}"
    assert latex(2*Integral(x, x)/3, fold_short_frac=True) == \
        r"\left(2 \int x\, dx\right) / 3"

    assert latex(sqrt(x)) == r"\sqrt{x}"
    assert latex(x**Rational(1, 3)) == r"\sqrt[3]{x}"
    assert latex(x**Rational(1, 3), root_notation=False) == r"x^{\frac{1}{3}}"
    assert latex(sqrt(x)**3) == r"x^{\frac{3}{2}}"
    assert latex(sqrt(x), itex=True) == r"\sqrt{x}"
    assert latex(x**Rational(1, 3), itex=True) == r"\root{3}{x}"
    assert latex(sqrt(x)**3, itex=True) == r"x^{\frac{3}{2}}"
    assert latex(x**Rational(3, 4)) == r"x^{\frac{3}{4}}"
    assert latex(x**Rational(3, 4), fold_frac_powers=True) == r"x^{3/4}"
    assert latex((x + 1)**Rational(3, 4)) == \
        r"\left(x + 1\right)^{\frac{3}{4}}"
    assert latex((x + 1)**Rational(3, 4), fold_frac_powers=True) == \
        r"\left(x + 1\right)^{3/4}"
    assert latex(AlgebraicNumber(sqrt(2))) == r"\sqrt{2}"
    assert latex(AlgebraicNumber(sqrt(2), [3, -7])) == r"-7 + 3 \sqrt{2}"
    assert latex(AlgebraicNumber(sqrt(2), alias='alpha')) == r"\alpha"
    assert latex(AlgebraicNumber(sqrt(2), [3, -7], alias='alpha')) == \
        r"3 \alpha - 7"
    assert latex(AlgebraicNumber(2**(S(1)/3), [1, 3, -7], alias='beta')) == \
        r"\beta^{2} + 3 \beta - 7"

    k = ZZ.cyclotomic_field(5)
    assert latex(k.ext.field_element([1, 2, 3, 4])) == \
        r"\zeta^{3} + 2 \zeta^{2} + 3 \zeta + 4"
    assert latex(k.ext.field_element([1, 2, 3, 4]), order='old') == \
        r"4 + 3 \zeta + 2 \zeta^{2} + \zeta^{3}"
    assert latex(k.primes_above(19)[0]) == \
        r"\left(19, \zeta^{2} + 5 \zeta + 1\right)"
    assert latex(k.primes_above(19)[0], order='old') == \
           r"\left(19, 1 + 5 \zeta + \zeta^{2}\right)"
    assert latex(k.primes_above(7)[0]) == r"\left(7\right)"

    assert latex(1.5e20*x) == r"1.5 \cdot 10^{20} x"
    assert latex(1.5e20*x, mul_symbol='dot') == r"1.5 \cdot 10^{20} \cdot x"
    assert latex(1.5e20*x, mul_symbol='times') == \
        r"1.5 \times 10^{20} \times x"

    assert latex(1/sin(x)) == r"\frac{1}{\sin{\left(x \right)}}"
    assert latex(sin(x)**-1) == r"\frac{1}{\sin{\left(x \right)}}"
    assert latex(sin(x)**Rational(3, 2)) == \
        r"\sin^{\frac{3}{2}}{\left(x \right)}"
    assert latex(sin(x)**Rational(3, 2), fold_frac_powers=True) == \
        r"\sin^{3/2}{\left(x \right)}"

    assert latex(~x) == r"\neg x"
    assert latex(x & y) == r"x \wedge y"
    assert latex(x & y & z) == r"x \wedge y \wedge z"
    assert latex(x | y) == r"x \vee y"
    assert latex(x | y | z) == r"x \vee y \vee z"
    assert latex((x & y) | z) == r"z \vee \left(x \wedge y\right)"
    assert latex(Implies(x, y)) == r"x \Rightarrow y"
    assert latex(~(x >> ~y)) == r"x \not\Rightarrow \neg y"
    assert latex(Implies(Or(x,y), z)) == r"\left(x \vee y\right) \Rightarrow z"
    assert latex(Implies(z, Or(x,y))) == r"z \Rightarrow \left(x \vee y\right)"
    assert latex(~(x & y)) == r"\neg \left(x \wedge y\right)"

    assert latex(~x, symbol_names={x: "x_i"}) == r"\neg x_i"
    assert latex(x & y, symbol_names={x: "x_i", y: "y_i"}) == \
        r"x_i \wedge y_i"
    assert latex(x & y & z, symbol_names={x: "x_i", y: "y_i", z: "z_i"}) == \
        r"x_i \wedge y_i \wedge z_i"
    assert latex(x | y, symbol_names={x: "x_i", y: "y_i"}) == r"x_i \vee y_i"
    assert latex(x | y | z, symbol_names={x: "x_i", y: "y_i", z: "z_i"}) == \
        r"x_i \vee y_i \vee z_i"
    assert latex((x & y) | z, symbol_names={x: "x_i", y: "y_i", z: "z_i"}) == \
        r"z_i \vee \left(x_i \wedge y_i\right)"
    assert latex(Implies(x, y), symbol_names={x: "x_i", y: "y_i"}) == \
        r"x_i \Rightarrow y_i"
    assert latex(Pow(Rational(1, 3), -1, evaluate=False)) == r"\frac{1}{\frac{1}{3}}"
    assert latex(Pow(Rational(1, 3), -2, evaluate=False)) == r"\frac{1}{(\frac{1}{3})^{2}}"
    assert latex(Pow(Integer(1)/100, -1, evaluate=False)) == r"\frac{1}{\frac{1}{100}}"

    p = Symbol('p', positive=True)
    assert latex(exp(-p)*log(p)) == r"e^{- p} \log{\left(p \right)}"

    assert latex(Pow(Rational(2, 3), -1, evaluate=False)) == r'\frac{1}{\frac{2}{3}}'
    assert latex(Pow(Rational(4, 3), -1, evaluate=False)) == r'\frac{1}{\frac{4}{3}}'
    assert latex(Pow(Rational(-3, 4), -1, evaluate=False)) == r'\frac{1}{- \frac{3}{4}}'
    assert latex(Pow(Rational(-4, 4), -1, evaluate=False)) == r'\frac{1}{-1}'
    assert latex(Pow(Rational(1, 3), -1, evaluate=False)) == r'\frac{1}{\frac{1}{3}}'
    assert latex(Pow(Rational(-1, 3), -1, evaluate=False)) == r'\frac{1}{- \frac{1}{3}}'


def test_latex_builtins():
    assert latex(True) == r"\text{True}"
    assert latex(False) == r"\text{False}"
    assert latex(None) == r"\text{None}"
    assert latex(true) == r"\text{True}"
    assert latex(false) == r'\text{False}'


def test_latex_SingularityFunction():
    assert latex(SingularityFunction(x, 4, 5)) == \
        r"{\left\langle x - 4 \right\rangle}^{5}"
    assert latex(SingularityFunction(x, -3, 4)) == \
        r"{\left\langle x + 3 \right\rangle}^{4}"
    assert latex(SingularityFunction(x, 0, 4)) == \
        r"{\left\langle x \right\rangle}^{4}"
    assert latex(SingularityFunction(x, a, n)) == \
        r"{\left\langle - a + x \right\rangle}^{n}"
    assert latex(SingularityFunction(x, 4, -2)) == \
        r"{\left\langle x - 4 \right\rangle}^{-2}"
    assert latex(SingularityFunction(x, 4, -1)) == \
        r"{\left\langle x - 4 \right\rangle}^{-1}"

    assert latex(SingularityFunction(x, 4, 5)**3) == \
        r"{\left({\langle x - 4 \rangle}^{5}\right)}^{3}"
    assert latex(SingularityFunction(x, -3, 4)**3) == \
        r"{\left({\langle x + 3 \rangle}^{4}\right)}^{3}"
    assert latex(SingularityFunction(x, 0, 4)**3) == \
        r"{\left({\langle x \rangle}^{4}\right)}^{3}"
    assert latex(SingularityFunction(x, a, n)**3) == \
        r"{\left({\langle - a + x \rangle}^{n}\right)}^{3}"
    assert latex(SingularityFunction(x, 4, -2)**3) == \
        r"{\left({\langle x - 4 \rangle}^{-2}\right)}^{3}"
    assert latex((SingularityFunction(x, 4, -1)**3)**3) == \
        r"{\left({\langle x - 4 \rangle}^{-1}\right)}^{9}"


def test_latex_cycle():
    assert latex(Cycle(1, 2, 4)) == r"\left( 1\; 2\; 4\right)"
    assert latex(Cycle(1, 2)(4, 5, 6)) == \
        r"\left( 1\; 2\right)\left( 4\; 5\; 6\right)"
    assert latex(Cycle()) == r"\left( \right)"


def test_latex_permutation():
    assert latex(Permutation(1, 2, 4)) == r"\left( 1\; 2\; 4\right)"
    assert latex(Permutation(1, 2)(4, 5, 6)) == \
        r"\left( 1\; 2\right)\left( 4\; 5\; 6\right)"
    assert latex(Permutation()) == r"\left( \right)"
    assert latex(Permutation(2, 4)*Permutation(5)) == \
        r"\left( 2\; 4\right)\left( 5\right)"
    assert latex(Permutation(5)) == r"\left( 5\right)"

    assert latex(Permutation(0, 1), perm_cyclic=False) == \
        r"\begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix}"
    assert latex(Permutation(0, 1)(2, 3), perm_cyclic=False) == \
        r"\begin{pmatrix} 0 & 1 & 2 & 3 \\ 1 & 0 & 3 & 2 \end{pmatrix}"
    assert latex(Permutation(), perm_cyclic=False) == \
        r"\left( \right)"

    with warns_deprecated_sympy():
        old_print_cyclic = Permutation.print_cyclic
        Permutation.print_cyclic = False
        assert latex(Permutation(0, 1)(2, 3)) == \
            r"\begin{pmatrix} 0 & 1 & 2 & 3 \\ 1 & 0 & 3 & 2 \end{pmatrix}"
        Permutation.print_cyclic = old_print_cyclic

def test_latex_Float():
    assert latex(Float(1.0e100)) == r"1.0 \cdot 10^{100}"
    assert latex(Float(1.0e-100)) == r"1.0 \cdot 10^{-100}"
    assert latex(Float(1.0e-100), mul_symbol="times") == \
        r"1.0 \times 10^{-100}"
    assert latex(Float('10000.0'), full_prec=False, min=-2, max=2) == \
        r"1.0 \cdot 10^{4}"
    assert latex(Float('10000.0'), full_prec=False, min=-2, max=4) == \
        r"1.0 \cdot 10^{4}"
    assert latex(Float('10000.0'), full_prec=False, min=-2, max=5) == \
        r"10000.0"
    assert latex(Float('0.099999'), full_prec=True,  min=-2, max=5) == \
        r"9.99990000000000 \cdot 10^{-2}"


def test_latex_vector_expressions():
    A = CoordSys3D('A')

    assert latex(Cross(A.i, A.j*A.x*3+A.k)) == \
        r"\mathbf{\hat{i}_{A}} \times \left(\left(3 \mathbf{{x}_{A}}\right)\mathbf{\hat{j}_{A}} + \mathbf{\hat{k}_{A}}\right)"
    assert latex(Cross(A.i, A.j)) == \
        r"\mathbf{\hat{i}_{A}} \times \mathbf{\hat{j}_{A}}"
    assert latex(x*Cross(A.i, A.j)) == \
        r"x \left(\mathbf{\hat{i}_{A}} \times \mathbf{\hat{j}_{A}}\right)"
    assert latex(Cross(x*A.i, A.j)) == \
        r'- \mathbf{\hat{j}_{A}} \times \left(\left(x\right)\mathbf{\hat{i}_{A}}\right)'

    assert latex(Curl(3*A.x*A.j)) == \
        r"\nabla\times \left(\left(3 \mathbf{{x}_{A}}\right)\mathbf{\hat{j}_{A}}\right)"
    assert latex(Curl(3*A.x*A.j+A.i)) == \
        r"\nabla\times \left(\mathbf{\hat{i}_{A}} + \left(3 \mathbf{{x}_{A}}\right)\mathbf{\hat{j}_{A}}\right)"
    assert latex(Curl(3*x*A.x*A.j)) == \
        r"\nabla\times \left(\left(3 \mathbf{{x}_{A}} x\right)\mathbf{\hat{j}_{A}}\right)"
    assert latex(x*Curl(3*A.x*A.j)) == \
        r"x \left(\nabla\times \left(\left(3 \mathbf{{x}_{A}}\right)\mathbf{\hat{j}_{A}}\right)\right)"

    assert latex(Divergence(3*A.x*A.j+A.i)) == \
        r"\nabla\cdot \left(\mathbf{\hat{i}_{A}} + \left(3 \mathbf{{x}_{A}}\right)\mathbf{\hat{j}_{A}}\right)"
    assert latex(Divergence(3*A.x*A.j)) == \
        r"\nabla\cdot \left(\left(3 \mathbf{{x}_{A}}\right)\mathbf{\hat{j}_{A}}\right)"
    assert latex(x*Divergence(3*A.x*A.j)) == \
        r"x \left(\nabla\cdot \left(\left(3 \mathbf{{x}_{A}}\right)\mathbf{\hat{j}_{A}}\right)\right)"

    assert latex(Dot(A.i, A.j*A.x*3+A.k)) == \
        r"\mathbf{\hat{i}_{A}} \cdot \left(\left(3 \mathbf{{x}_{A}}\right)\mathbf{\hat{j}_{A}} + \mathbf{\hat{k}_{A}}\right)"
    assert latex(Dot(A.i, A.j)) == \
        r"\mathbf{\hat{i}_{A}} \cdot \mathbf{\hat{j}_{A}}"
    assert latex(Dot(x*A.i, A.j)) == \
        r"\mathbf{\hat{j}_{A}} \cdot \left(\left(x\right)\mathbf{\hat{i}_{A}}\right)"
    assert latex(x*Dot(A.i, A.j)) == \
        r"x \left(\mathbf{\hat{i}_{A}} \cdot \mathbf{\hat{j}_{A}}\right)"

    assert latex(Gradient(A.x)) == r"\nabla \mathbf{{x}_{A}}"
    assert latex(Gradient(A.x + 3*A.y)) == \
        r"\nabla \left(\mathbf{{x}_{A}} + 3 \mathbf{{y}_{A}}\right)"
    assert latex(x*Gradient(A.x)) == r"x \left(\nabla \mathbf{{x}_{A}}\right)"
    assert latex(Gradient(x*A.x)) == r"\nabla \left(\mathbf{{x}_{A}} x\right)"

    assert latex(Laplacian(A.x)) == r"\Delta \mathbf{{x}_{A}}"
    assert latex(Laplacian(A.x + 3*A.y)) == \
        r"\Delta \left(\mathbf{{x}_{A}} + 3 \mathbf{{y}_{A}}\right)"
    assert latex(x*Laplacian(A.x)) == r"x \left(\Delta \mathbf{{x}_{A}}\right)"
    assert latex(Laplacian(x*A.x)) == r"\Delta \left(\mathbf{{x}_{A}} x\right)"

def test_latex_symbols():
    Gamma, lmbda, rho = symbols('Gamma, lambda, rho')
    tau, Tau, TAU, taU = symbols('tau, Tau, TAU, taU')
    assert latex(tau) == r"\tau"
    assert latex(Tau) == r"\mathrm{T}"
    assert latex(TAU) == r"\tau"
    assert latex(taU) == r"\tau"
    # Check that all capitalized greek letters are handled explicitly
    capitalized_letters = {l.capitalize() for l in greek_letters_set}
    assert len(capitalized_letters - set(tex_greek_dictionary.keys())) == 0
    assert latex(Gamma + lmbda) == r"\Gamma + \lambda"
    assert latex(Gamma * lmbda) == r"\Gamma \lambda"
    assert latex(Symbol('q1')) == r"q_{1}"
    assert latex(Symbol('q21')) == r"q_{21}"
    assert latex(Symbol('epsilon0')) == r"\epsilon_{0}"
    assert latex(Symbol('omega1')) == r"\omega_{1}"
    assert latex(Symbol('91')) == r"91"
    assert latex(Symbol('alpha_new')) == r"\alpha_{new}"
    assert latex(Symbol('C^orig')) == r"C^{orig}"
    assert latex(Symbol('x^alpha')) == r"x^{\alpha}"
    assert latex(Symbol('beta^alpha')) == r"\beta^{\alpha}"
    assert latex(Symbol('e^Alpha')) == r"e^{\mathrm{A}}"
    assert latex(Symbol('omega_alpha^beta')) == r"\omega^{\beta}_{\alpha}"
    assert latex(Symbol('omega') ** Symbol('beta')) == r"\omega^{\beta}"


@XFAIL
def test_latex_symbols_failing():
    rho, mass, volume = symbols('rho, mass, volume')
    assert latex(
        volume * rho == mass) == r"\rho \mathrm{volume} = \mathrm{mass}"
    assert latex(volume / mass * rho == 1) == \
        r"\rho \mathrm{volume} {\mathrm{mass}}^{(-1)} = 1"
    assert latex(mass**3 * volume**3) == \
        r"{\mathrm{mass}}^{3} \cdot {\mathrm{volume}}^{3}"


@_both_exp_pow
def test_latex_functions():
    assert latex(exp(x)) == r"e^{x}"
    assert latex(exp(1) + exp(2)) == r"e + e^{2}"

    f = Function('f')
    assert latex(f(x)) == r'f{\left(x \right)}'
    assert latex(f) == r'f'

    g = Function('g')
    assert latex(g(x, y)) == r'g{\left(x,y \right)}'
    assert latex(g) == r'g'

    h = Function('h')
    assert latex(h(x, y, z)) == r'h{\left(x,y,z \right)}'
    assert latex(h) == r'h'

    Li = Function('Li')
    assert latex(Li) == r'\operatorname{Li}'
    assert latex(Li(x)) == r'\operatorname{Li}{\left(x \right)}'

    mybeta = Function('beta')
    # not to be confused with the beta function
    assert latex(mybeta(x, y, z)) == r"\beta{\left(x,y,z \right)}"
    assert latex(beta(x, y)) == r'\operatorname{B}\left(x, y\right)'
    assert latex(beta(x, evaluate=False)) == r'\operatorname{B}\left(x, x\right)'
    assert latex(beta(x, y)**2) == r'\operatorname{B}^{2}\left(x, y\right)'
    assert latex(mybeta(x)) == r"\beta{\left(x \right)}"
    assert latex(mybeta) == r"\beta"

    g = Function('gamma')
    # not to be confused with the gamma function
    assert latex(g(x, y, z)) == r"\gamma{\left(x,y,z \right)}"
    assert latex(g(x)) == r"\gamma{\left(x \right)}"
    assert latex(g) == r"\gamma"

    a_1 = Function('a_1')
    assert latex(a_1) == r"a_{1}"
    assert latex(a_1(x)) == r"a_{1}{\left(x \right)}"
    assert latex(Function('a_1')) == r"a_{1}"

    # Issue #16925
    # multi letter function names
    # > simple
    assert latex(Function('ab')) == r"\operatorname{ab}"
    assert latex(Function('ab1')) == r"\operatorname{ab}_{1}"
    assert latex(Function('ab12')) == r"\operatorname{ab}_{12}"
    assert latex(Function('ab_1')) == r"\operatorname{ab}_{1}"
    assert latex(Function('ab_12')) == r"\operatorname{ab}_{12}"
    assert latex(Function('ab_c')) == r"\operatorname{ab}_{c}"
    assert latex(Function('ab_cd')) == r"\operatorname{ab}_{cd}"
    # > with argument
    assert latex(Function('ab')(Symbol('x'))) == r"\operatorname{ab}{\left(x \right)}"
    assert latex(Function('ab1')(Symbol('x'))) == r"\operatorname{ab}_{1}{\left(x \right)}"
    assert latex(Function('ab12')(Symbol('x'))) == r"\operatorname{ab}_{12}{\left(x \right)}"
    assert latex(Function('ab_1')(Symbol('x'))) == r"\operatorname{ab}_{1}{\left(x \right)}"
    assert latex(Function('ab_c')(Symbol('x'))) == r"\operatorname{ab}_{c}{\left(x \right)}"
    assert latex(Function('ab_cd')(Symbol('x'))) == r"\operatorname{ab}_{cd}{\left(x \right)}"

    # > with power
    #   does not work on functions without brackets

    # > with argument and power combined
    assert latex(Function('ab')()**2) == r"\operatorname{ab}^{2}{\left( \right)}"
    assert latex(Function('ab1')()**2) == r"\operatorname{ab}_{1}^{2}{\left( \right)}"
    assert latex(Function('ab12')()**2) == r"\operatorname{ab}_{12}^{2}{\left( \right)}"
    assert latex(Function('ab_1')()**2) == r"\operatorname{ab}_{1}^{2}{\left( \right)}"
    assert latex(Function('ab_12')()**2) == r"\operatorname{ab}_{12}^{2}{\left( \right)}"
    assert latex(Function('ab')(Symbol('x'))**2) == r"\operatorname{ab}^{2}{\left(x \right)}"
    assert latex(Function('ab1')(Symbol('x'))**2) == r"\operatorname{ab}_{1}^{2}{\left(x \right)}"
    assert latex(Function('ab12')(Symbol('x'))**2) == r"\operatorname{ab}_{12}^{2}{\left(x \right)}"
    assert latex(Function('ab_1')(Symbol('x'))**2) == r"\operatorname{ab}_{1}^{2}{\left(x \right)}"
    assert latex(Function('ab_12')(Symbol('x'))**2) == \
        r"\operatorname{ab}_{12}^{2}{\left(x \right)}"

    # single letter function names
    # > simple
    assert latex(Function('a')) == r"a"
    assert latex(Function('a1')) == r"a_{1}"
    assert latex(Function('a12')) == r"a_{12}"
    assert latex(Function('a_1')) == r"a_{1}"
    assert latex(Function('a_12')) == r"a_{12}"

    # > with argument
    assert latex(Function('a')()) == r"a{\left( \right)}"
    assert latex(Function('a1')()) == r"a_{1}{\left( \right)}"
    assert latex(Function('a12')()) == r"a_{12}{\left( \right)}"
    assert latex(Function('a_1')()) == r"a_{1}{\left( \right)}"
    assert latex(Function('a_12')()) == r"a_{12}{\left( \right)}"

    # > with power
    #   does not work on functions without brackets

    # > with argument and power combined
    assert latex(Function('a')()**2) == r"a^{2}{\left( \right)}"
    assert latex(Function('a1')()**2) == r"a_{1}^{2}{\left( \right)}"
    assert latex(Function('a12')()**2) == r"a_{12}^{2}{\left( \right)}"
    assert latex(Function('a_1')()**2) == r"a_{1}^{2}{\left( \right)}"
    assert latex(Function('a_12')()**2) == r"a_{12}^{2}{\left( \right)}"
    assert latex(Function('a')(Symbol('x'))**2) == r"a^{2}{\left(x \right)}"
    assert latex(Function('a1')(Symbol('x'))**2) == r"a_{1}^{2}{\left(x \right)}"
    assert latex(Function('a12')(Symbol('x'))**2) == r"a_{12}^{2}{\left(x \right)}"
    assert latex(Function('a_1')(Symbol('x'))**2) == r"a_{1}^{2}{\left(x \right)}"
    assert latex(Function('a_12')(Symbol('x'))**2) == r"a_{12}^{2}{\left(x \right)}"

    assert latex(Function('a')()**32) == r"a^{32}{\left( \right)}"
    assert latex(Function('a1')()**32) == r"a_{1}^{32}{\left( \right)}"
    assert latex(Function('a12')()**32) == r"a_{12}^{32}{\left( \right)}"
    assert latex(Function('a_1')()**32) == r"a_{1}^{32}{\left( \right)}"
    assert latex(Function('a_12')()**32) == r"a_{12}^{32}{\left( \right)}"
    assert latex(Function('a')(Symbol('x'))**32) == r"a^{32}{\left(x \right)}"
    assert latex(Function('a1')(Symbol('x'))**32) == r"a_{1}^{32}{\left(x \right)}"
    assert latex(Function('a12')(Symbol('x'))**32) == r"a_{12}^{32}{\left(x \right)}"
    assert latex(Function('a_1')(Symbol('x'))**32) == r"a_{1}^{32}{\left(x \right)}"
    assert latex(Function('a_12')(Symbol('x'))**32) == r"a_{12}^{32}{\left(x \right)}"

    assert latex(Function('a')()**a) == r"a^{a}{\left( \right)}"
    assert latex(Function('a1')()**a) == r"a_{1}^{a}{\left( \right)}"
    assert latex(Function('a12')()**a) == r"a_{12}^{a}{\left( \right)}"
    assert latex(Function('a_1')()**a) == r"a_{1}^{a}{\left( \right)}"
    assert latex(Function('a_12')()**a) == r"a_{12}^{a}{\left( \right)}"
    assert latex(Function('a')(Symbol('x'))**a) == r"a^{a}{\left(x \right)}"
    assert latex(Function('a1')(Symbol('x'))**a) == r"a_{1}^{a}{\left(x \right)}"
    assert latex(Function('a12')(Symbol('x'))**a) == r"a_{12}^{a}{\left(x \right)}"
    assert latex(Function('a_1')(Symbol('x'))**a) == r"a_{1}^{a}{\left(x \right)}"
    assert latex(Function('a_12')(Symbol('x'))**a) == r"a_{12}^{a}{\left(x \right)}"

    ab = Symbol('ab')
    assert latex(Function('a')()**ab) == r"a^{ab}{\left( \right)}"
    assert latex(Function('a1')()**ab) == r"a_{1}^{ab}{\left( \right)}"
    assert latex(Function('a12')()**ab) == r"a_{12}^{ab}{\left( \right)}"
    assert latex(Function('a_1')()**ab) == r"a_{1}^{ab}{\left( \right)}"
    assert latex(Function('a_12')()**ab) == r"a_{12}^{ab}{\left( \right)}"
    assert latex(Function('a')(Symbol('x'))**ab) == r"a^{ab}{\left(x \right)}"
    assert latex(Function('a1')(Symbol('x'))**ab) == r"a_{1}^{ab}{\left(x \right)}"
    assert latex(Function('a12')(Symbol('x'))**ab) == r"a_{12}^{ab}{\left(x \right)}"
    assert latex(Function('a_1')(Symbol('x'))**ab) == r"a_{1}^{ab}{\left(x \right)}"
    assert latex(Function('a_12')(Symbol('x'))**ab) == r"a_{12}^{ab}{\left(x \right)}"

    assert latex(Function('a^12')(x)) == R"a^{12}{\left(x \right)}"
    assert latex(Function('a^12')(x) ** ab) == R"\left(a^{12}\right)^{ab}{\left(x \right)}"
    assert latex(Function('a__12')(x)) == R"a^{12}{\left(x \right)}"
    assert latex(Function('a__12')(x) ** ab) == R"\left(a^{12}\right)^{ab}{\left(x \right)}"
    assert latex(Function('a_1__1_2')(x)) == R"a^{1}_{1 2}{\left(x \right)}"

    # issue 5868
    omega1 = Function('omega1')
    assert latex(omega1) == r"\omega_{1}"
    assert latex(omega1(x)) == r"\omega_{1}{\left(x \right)}"

    assert latex(sin(x)) == r"\sin{\left(x \right)}"
    assert latex(sin(x), fold_func_brackets=True) == r"\sin {x}"
    assert latex(sin(2*x**2), fold_func_brackets=True) == \
        r"\sin {2 x^{2}}"
    assert latex(sin(x**2), fold_func_brackets=True) == \
        r"\sin {x^{2}}"

    assert latex(asin(x)**2) == r"\operatorname{asin}^{2}{\left(x \right)}"
    assert latex(asin(x)**2, inv_trig_style="full") == \
        r"\arcsin^{2}{\left(x \right)}"
    assert latex(asin(x)**2, inv_trig_style="power") == \
        r"\sin^{-1}{\left(x \right)}^{2}"
    assert latex(asin(x**2), inv_trig_style="power",
                 fold_func_brackets=True) == \
        r"\sin^{-1} {x^{2}}"
    assert latex(acsc(x), inv_trig_style="full") == \
        r"\operatorname{arccsc}{\left(x \right)}"
    assert latex(asinh(x), inv_trig_style="full") == \
        r"\operatorname{arsinh}{\left(x \right)}"

    assert latex(factorial(k)) == r"k!"
    assert latex(factorial(-k)) == r"\left(- k\right)!"
    assert latex(factorial(k)**2) == r"k!^{2}"

    assert latex(subfactorial(k)) == r"!k"
    assert latex(subfactorial(-k)) == r"!\left(- k\right)"
    assert latex(subfactorial(k)**2) == r"\left(!k\right)^{2}"

    assert latex(factorial2(k)) == r"k!!"
    assert latex(factorial2(-k)) == r"\left(- k\right)!!"
    assert latex(factorial2(k)**2) == r"k!!^{2}"

    assert latex(binomial(2, k)) == r"{\binom{2}{k}}"
    assert latex(binomial(2, k)**2) == r"{\binom{2}{k}}^{2}"

    assert latex(FallingFactorial(3, k)) == r"{\left(3\right)}_{k}"
    assert latex(RisingFactorial(3, k)) == r"{3}^{\left(k\right)}"

    assert latex(floor(x)) == r"\left\lfloor{x}\right\rfloor"
    assert latex(ceiling(x)) == r"\left\lceil{x}\right\rceil"
    assert latex(frac(x)) == r"\operatorname{frac}{\left(x\right)}"
    assert latex(floor(x)**2) == r"\left\lfloor{x}\right\rfloor^{2}"
    assert latex(ceiling(x)**2) == r"\left\lceil{x}\right\rceil^{2}"
    assert latex(frac(x)**2) == r"\operatorname{frac}{\left(x\right)}^{2}"

    assert latex(Min(x, 2, x**3)) == r"\min\left(2, x, x^{3}\right)"
    assert latex(Min(x, y)**2) == r"\min\left(x, y\right)^{2}"
    assert latex(Max(x, 2, x**3)) == r"\max\left(2, x, x^{3}\right)"
    assert latex(Max(x, y)**2) == r"\max\left(x, y\right)^{2}"
    assert latex(Abs(x)) == r"\left|{x}\right|"
    assert latex(Abs(x)**2) == r"\left|{x}\right|^{2}"
    assert latex(re(x)) == r"\operatorname{re}{\left(x\right)}"
    assert latex(re(x + y)) == \
        r"\operatorname{re}{\left(x\right)} + \operatorname{re}{\left(y\right)}"
    assert latex(im(x)) == r"\operatorname{im}{\left(x\right)}"
    assert latex(conjugate(x)) == r"\overline{x}"
    assert latex(conjugate(x)**2) == r"\overline{x}^{2}"
    assert latex(conjugate(x**2)) == r"\overline{x}^{2}"
    assert latex(gamma(x)) == r"\Gamma\left(x\right)"
    w = Wild('w')
    assert latex(gamma(w)) == r"\Gamma\left(w\right)"
    assert latex(Order(x)) == r"O\left(x\right)"
    assert latex(Order(x, x)) == r"O\left(x\right)"
    assert latex(Order(x, (x, 0))) == r"O\left(x\right)"
    assert latex(Order(x, (x, oo))) == r"O\left(x; x\rightarrow \infty\right)"
    assert latex(Order(x - y, (x, y))) == \
        r"O\left(x - y; x\rightarrow y\right)"
    assert latex(Order(x, x, y)) == \
        r"O\left(x; \left( x, \  y\right)\rightarrow \left( 0, \  0\right)\right)"
    assert latex(Order(x, x, y)) == \
        r"O\left(x; \left( x, \  y\right)\rightarrow \left( 0, \  0\right)\right)"
    assert latex(Order(x, (x, oo), (y, oo))) == \
        r"O\left(x; \left( x, \  y\right)\rightarrow \left( \infty, \  \infty\right)\right)"
    assert latex(lowergamma(x, y)) == r'\gamma\left(x, y\right)'
    assert latex(lowergamma(x, y)**2) == r'\gamma^{2}\left(x, y\right)'
    assert latex(uppergamma(x, y)) == r'\Gamma\left(x, y\right)'
    assert latex(uppergamma(x, y)**2) == r'\Gamma^{2}\left(x, y\right)'

    assert latex(cot(x)) == r'\cot{\left(x \right)}'
    assert latex(coth(x)) == r'\coth{\left(x \right)}'
    assert latex(re(x)) == r'\operatorname{re}{\left(x\right)}'
    assert latex(im(x)) == r'\operatorname{im}{\left(x\right)}'
    assert latex(root(x, y)) == r'x^{\frac{1}{y}}'
    assert latex(arg(x)) == r'\arg{\left(x \right)}'

    assert latex(zeta(x)) == r"\zeta\left(x\right)"
    assert latex(zeta(x)**2) == r"\zeta^{2}\left(x\right)"
    assert latex(zeta(x, y)) == r"\zeta\left(x, y\right)"
    assert latex(zeta(x, y)**2) == r"\zeta^{2}\left(x, y\right)"
    assert latex(dirichlet_eta(x)) == r"\eta\left(x\right)"
    assert latex(dirichlet_eta(x)**2) == r"\eta^{2}\left(x\right)"
    assert latex(polylog(x, y)) == r"\operatorname{Li}_{x}\left(y\right)"
    assert latex(
        polylog(x, y)**2) == r"\operatorname{Li}_{x}^{2}\left(y\right)"
    assert latex(lerchphi(x, y, n)) == r"\Phi\left(x, y, n\right)"
    assert latex(lerchphi(x, y, n)**2) == r"\Phi^{2}\left(x, y, n\right)"
    assert latex(stieltjes(x)) == r"\gamma_{x}"
    assert latex(stieltjes(x)**2) == r"\gamma_{x}^{2}"
    assert latex(stieltjes(x, y)) == r"\gamma_{x}\left(y\right)"
    assert latex(stieltjes(x, y)**2) == r"\gamma_{x}\left(y\right)^{2}"

    assert latex(elliptic_k(z)) == r"K\left(z\right)"
    assert latex(elliptic_k(z)**2) == r"K^{2}\left(z\right)"
    assert latex(elliptic_f(x, y)) == r"F\left(x\middle| y\right)"
    assert latex(elliptic_f(x, y)**2) == r"F^{2}\left(x\middle| y\right)"
    assert latex(elliptic_e(x, y)) == r"E\left(x\middle| y\right)"
    assert latex(elliptic_e(x, y)**2) == r"E^{2}\left(x\middle| y\right)"
    assert latex(elliptic_e(z)) == r"E\left(z\right)"
    assert latex(elliptic_e(z)**2) == r"E^{2}\left(z\right)"
    assert latex(elliptic_pi(x, y, z)) == r"\Pi\left(x; y\middle| z\right)"
    assert latex(elliptic_pi(x, y, z)**2) == \
        r"\Pi^{2}\left(x; y\middle| z\right)"
    assert latex(elliptic_pi(x, y)) == r"\Pi\left(x\middle| y\right)"
    assert latex(elliptic_pi(x, y)**2) == r"\Pi^{2}\left(x\middle| y\right)"

    assert latex(Ei(x)) == r'\operatorname{Ei}{\left(x \right)}'
    assert latex(Ei(x)**2) == r'\operatorname{Ei}^{2}{\left(x \right)}'
    assert latex(expint(x, y)) == r'\operatorname{E}_{x}\left(y\right)'
    assert latex(expint(x, y)**2) == r'\operatorname{E}_{x}^{2}\left(y\right)'
    assert latex(Shi(x)**2) == r'\operatorname{Shi}^{2}{\left(x \right)}'
    assert latex(Si(x)**2) == r'\operatorname{Si}^{2}{\left(x \right)}'
    assert latex(Ci(x)**2) == r'\operatorname{Ci}^{2}{\left(x \right)}'
    assert latex(Chi(x)**2) == r'\operatorname{Chi}^{2}\left(x\right)'
    assert latex(Chi(x)) == r'\operatorname{Chi}\left(x\right)'
    assert latex(jacobi(n, a, b, x)) == \
        r'P_{n}^{\left(a,b\right)}\left(x\right)'
    assert latex(jacobi(n, a, b, x)**2) == \
        r'\left(P_{n}^{\left(a,b\right)}\left(x\right)\right)^{2}'
    assert latex(gegenbauer(n, a, x)) == \
        r'C_{n}^{\left(a\right)}\left(x\right)'
    assert latex(gegenbauer(n, a, x)**2) == \
        r'\left(C_{n}^{\left(a\right)}\left(x\right)\right)^{2}'
    assert latex(chebyshevt(n, x)) == r'T_{n}\left(x\right)'
    assert latex(chebyshevt(n, x)**2) == \
        r'\left(T_{n}\left(x\right)\right)^{2}'
    assert latex(chebyshevu(n, x)) == r'U_{n}\left(x\right)'
    assert latex(chebyshevu(n, x)**2) == \
        r'\left(U_{n}\left(x\right)\right)^{2}'
    assert latex(legendre(n, x)) == r'P_{n}\left(x\right)'
    assert latex(legendre(n, x)**2) == r'\left(P_{n}\left(x\right)\right)^{2}'
    assert latex(assoc_legendre(n, a, x)) == \
        r'P_{n}^{\left(a\right)}\left(x\right)'
    assert latex(assoc_legendre(n, a, x)**2) == \
        r'\left(P_{n}^{\left(a\right)}\left(x\right)\right)^{2}'
    assert latex(laguerre(n, x)) == r'L_{n}\left(x\right)'
    assert latex(laguerre(n, x)**2) == r'\left(L_{n}\left(x\right)\right)^{2}'
    assert latex(assoc_laguerre(n, a, x)) == \
        r'L_{n}^{\left(a\right)}\left(x\right)'
    assert latex(assoc_laguerre(n, a, x)**2) == \
        r'\left(L_{n}^{\left(a\right)}\left(x\right)\right)^{2}'
    assert latex(hermite(n, x)) == r'H_{n}\left(x\right)'
    assert latex(hermite(n, x)**2) == r'\left(H_{n}\left(x\right)\right)^{2}'

    theta = Symbol("theta", real=True)
    phi = Symbol("phi", real=True)
    assert latex(Ynm(n, m, theta, phi)) == r'Y_{n}^{m}\left(\theta,\phi\right)'
    assert latex(Ynm(n, m, theta, phi)**3) == \
        r'\left(Y_{n}^{m}\left(\theta,\phi\right)\right)^{3}'
    assert latex(Znm(n, m, theta, phi)) == r'Z_{n}^{m}\left(\theta,\phi\right)'
    assert latex(Znm(n, m, theta, phi)**3) == \
        r'\left(Z_{n}^{m}\left(\theta,\phi\right)\right)^{3}'

    # Test latex printing of function names with "_"
    assert latex(polar_lift(0)) == \
        r"\operatorname{polar\_lift}{\left(0 \right)}"
    assert latex(polar_lift(0)**3) == \
        r"\operatorname{polar\_lift}^{3}{\left(0 \right)}"

    assert latex(totient(n)) == r'\phi\left(n\right)'
    assert latex(totient(n) ** 2) == r'\left(\phi\left(n\right)\right)^{2}'

    assert latex(reduced_totient(n)) == r'\lambda\left(n\right)'
    assert latex(reduced_totient(n) ** 2) == \
        r'\left(\lambda\left(n\right)\right)^{2}'

    assert latex(divisor_sigma(x)) == r"\sigma\left(x\right)"
    assert latex(divisor_sigma(x)**2) == r"\sigma^{2}\left(x\right)"
    assert latex(divisor_sigma(x, y)) == r"\sigma_y\left(x\right)"
    assert latex(divisor_sigma(x, y)**2) == r"\sigma^{2}_y\left(x\right)"

    assert latex(udivisor_sigma(x)) == r"\sigma^*\left(x\right)"
    assert latex(udivisor_sigma(x)**2) == r"\sigma^*^{2}\left(x\right)"
    assert latex(udivisor_sigma(x, y)) == r"\sigma^*_y\left(x\right)"
    assert latex(udivisor_sigma(x, y)**2) == r"\sigma^*^{2}_y\left(x\right)"

    assert latex(primenu(n)) == r'\nu\left(n\right)'
    assert latex(primenu(n) ** 2) == r'\left(\nu\left(n\right)\right)^{2}'

    assert latex(primeomega(n)) == r'\Omega\left(n\right)'
    assert latex(primeomega(n) ** 2) == \
        r'\left(\Omega\left(n\right)\right)^{2}'

    assert latex(LambertW(n)) == r'W\left(n\right)'
    assert latex(LambertW(n, -1)) == r'W_{-1}\left(n\right)'
    assert latex(LambertW(n, k)) == r'W_{k}\left(n\right)'
    assert latex(LambertW(n) * LambertW(n)) == r"W^{2}\left(n\right)"
    assert latex(Pow(LambertW(n), 2)) == r"W^{2}\left(n\right)"
    assert latex(LambertW(n)**k) == r"W^{k}\left(n\right)"
    assert latex(LambertW(n, k)**p) == r"W^{p}_{k}\left(n\right)"

    assert latex(Mod(x, 7)) == r'x \bmod 7'
    assert latex(Mod(x + 1, 7)) == r'\left(x + 1\right) \bmod 7'
    assert latex(Mod(7, x + 1)) == r'7 \bmod \left(x + 1\right)'
    assert latex(Mod(2 * x, 7)) == r'2 x \bmod 7'
    assert latex(Mod(7, 2 * x)) == r'7 \bmod 2 x'
    assert latex(Mod(x, 7) + 1) == r'\left(x \bmod 7\right) + 1'
    assert latex(2 * Mod(x, 7)) == r'2 \left(x \bmod 7\right)'
    assert latex(Mod(7, 2 * x)**n) == r'\left(7 \bmod 2 x\right)^{n}'

    # some unknown function name should get rendered with \operatorname
    fjlkd = Function('fjlkd')
    assert latex(fjlkd(x)) == r'\operatorname{fjlkd}{\left(x \right)}'
    # even when it is referred to without an argument
    assert latex(fjlkd) == r'\operatorname{fjlkd}'


# test that notation passes to subclasses of the same name only
def test_function_subclass_different_name():
    class mygamma(gamma):
        pass
    assert latex(mygamma) == r"\operatorname{mygamma}"
    assert latex(mygamma(x)) == r"\operatorname{mygamma}{\left(x \right)}"


def test_hyper_printing():
    from sympy.abc import x, z

    assert latex(meijerg(Tuple(pi, pi, x), Tuple(1),
                         (0, 1), Tuple(1, 2, 3/pi), z)) == \
        r'{G_{4, 5}^{2, 3}\left(\begin{matrix} \pi, \pi, x & 1 \\0, 1 & 1, 2, '\
        r'\frac{3}{\pi} \end{matrix} \middle| {z} \right)}'
    assert latex(meijerg(Tuple(), Tuple(1), (0,), Tuple(), z)) == \
        r'{G_{1, 1}^{1, 0}\left(\begin{matrix}  & 1 \\0 &  \end{matrix} \middle| {z} \right)}'
    assert latex(hyper((x, 2), (3,), z)) == \
        r'{{}_{2}F_{1}\left(\begin{matrix} 2, x ' \
        r'\\ 3 \end{matrix}\middle| {z} \right)}'
    assert latex(hyper(Tuple(), Tuple(1), z)) == \
        r'{{}_{0}F_{1}\left(\begin{matrix}  ' \
        r'\\ 1 \end{matrix}\middle| {z} \right)}'


def test_latex_bessel():
    from sympy.functions.special.bessel import (besselj, bessely, besseli,
                                                besselk, hankel1, hankel2,
                                                jn, yn, hn1, hn2)
    from sympy.abc import z
    assert latex(besselj(n, z**2)**k) == r'J^{k}_{n}\left(z^{2}\right)'
    assert latex(bessely(n, z)) == r'Y_{n}\left(z\right)'
    assert latex(besseli(n, z)) == r'I_{n}\left(z\right)'
    assert latex(besselk(n, z)) == r'K_{n}\left(z\right)'
    assert latex(hankel1(n, z**2)**2) == \
        r'\left(H^{(1)}_{n}\left(z^{2}\right)\right)^{2}'
    assert latex(hankel2(n, z)) == r'H^{(2)}_{n}\left(z\right)'
    assert latex(jn(n, z)) == r'j_{n}\left(z\right)'
    assert latex(yn(n, z)) == r'y_{n}\left(z\right)'
    assert latex(hn1(n, z)) == r'h^{(1)}_{n}\left(z\right)'
    assert latex(hn2(n, z)) == r'h^{(2)}_{n}\left(z\right)'


def test_latex_fresnel():
    from sympy.functions.special.error_functions import (fresnels, fresnelc)
    from sympy.abc import z
    assert latex(fresnels(z)) == r'S\left(z\right)'
    assert latex(fresnelc(z)) == r'C\left(z\right)'
    assert latex(fresnels(z)**2) == r'S^{2}\left(z\right)'
    assert latex(fresnelc(z)**2) == r'C^{2}\left(z\right)'


def test_latex_brackets():
    assert latex((-1)**x) == r"\left(-1\right)^{x}"


def test_latex_indexed():
    Psi_symbol = Symbol('Psi_0', complex=True, real=False)
    Psi_indexed = IndexedBase(Symbol('Psi', complex=True, real=False))
    symbol_latex = latex(Psi_symbol * conjugate(Psi_symbol))
    indexed_latex = latex(Psi_indexed[0] * conjugate(Psi_indexed[0]))
    # \\overline{{\\Psi}_{0}} {\\Psi}_{0}  vs.  \\Psi_{0} \\overline{\\Psi_{0}}
    assert symbol_latex == r'\Psi_{0} \overline{\Psi_{0}}'
    assert indexed_latex == r'\overline{{\Psi}_{0}} {\Psi}_{0}'

    # Symbol('gamma') gives r'\gamma'
    interval = '\\mathrel{..}\\nobreak '
    assert latex(Indexed('x1', Symbol('i'))) == r'{x_{1}}_{i}'
    assert latex(Indexed('x2', Idx('i'))) == r'{x_{2}}_{i}'
    assert latex(Indexed('x3', Idx('i', Symbol('N')))) == r'{x_{3}}_{{i}_{0'+interval+'N - 1}}'
    assert latex(Indexed('x3', Idx('i', Symbol('N')+1))) == r'{x_{3}}_{{i}_{0'+interval+'N}}'
    assert latex(Indexed('x4', Idx('i', (Symbol('a'),Symbol('b'))))) == r'{x_{4}}_{{i}_{a'+interval+'b}}'
    assert latex(IndexedBase('gamma')) == r'\gamma'
    assert latex(IndexedBase('a b')) == r'a b'
    assert latex(IndexedBase('a_b')) == r'a_{b}'


def test_latex_derivatives():
    # regular "d" for ordinary derivatives
    assert latex(diff(x**3, x, evaluate=False)) == \
        r"\frac{d}{d x} x^{3}"
    assert latex(diff(sin(x) + x**2, x, evaluate=False)) == \
        r"\frac{d}{d x} \left(x^{2} + \sin{\left(x \right)}\right)"
    assert latex(diff(diff(sin(x) + x**2, x, evaluate=False), evaluate=False))\
        == \
        r"\frac{d^{2}}{d x^{2}} \left(x^{2} + \sin{\left(x \right)}\right)"
    assert latex(diff(diff(diff(sin(x) + x**2, x, evaluate=False), evaluate=False), evaluate=False)) == \
        r"\frac{d^{3}}{d x^{3}} \left(x^{2} + \sin{\left(x \right)}\right)"

    # \partial for partial derivatives
    assert latex(diff(sin(x * y), x, evaluate=False)) == \
        r"\frac{\partial}{\partial x} \sin{\left(x y \right)}"
    assert latex(diff(sin(x * y) + x**2, x, evaluate=False)) == \
        r"\frac{\partial}{\partial x} \left(x^{2} + \sin{\left(x y \right)}\right)"
    assert latex(diff(diff(sin(x*y) + x**2, x, evaluate=False), x, evaluate=False)) == \
        r"\frac{\partial^{2}}{\partial x^{2}} \left(x^{2} + \sin{\left(x y \right)}\right)"
    assert latex(diff(diff(diff(sin(x*y) + x**2, x, evaluate=False), x, evaluate=False), x, evaluate=False)) == \
        r"\frac{\partial^{3}}{\partial x^{3}} \left(x^{2} + \sin{\left(x y \right)}\right)"

    # mixed partial derivatives
    f = Function("f")
    assert latex(diff(diff(f(x, y), x, evaluate=False), y, evaluate=False)) == \
        r"\frac{\partial^{2}}{\partial y\partial x} " + latex(f(x, y))

    assert latex(diff(diff(diff(f(x, y), x, evaluate=False), x, evaluate=False), y, evaluate=False)) == \
        r"\frac{\partial^{3}}{\partial y\partial x^{2}} " + latex(f(x, y))

    # for negative nested Derivative
    assert latex(diff(-diff(y**2,x,evaluate=False),x,evaluate=False)) == r'\frac{d}{d x} \left(- \frac{d}{d x} y^{2}\right)'
    assert latex(diff(diff(-diff(diff(y,x,evaluate=False),x,evaluate=False),x,evaluate=False),x,evaluate=False)) == \
        r'\frac{d^{2}}{d x^{2}} \left(- \frac{d^{2}}{d x^{2}} y\right)'

    # use ordinary d when one of the variables has been integrated out
    assert latex(diff(Integral(exp(-x*y), (x, 0, oo)), y, evaluate=False)) == \
        r"\frac{d}{d y} \int\limits_{0}^{\infty} e^{- x y}\, dx"

    # Derivative wrapped in power:
    assert latex(diff(x, x, evaluate=False)**2) == \
        r"\left(\frac{d}{d x} x\right)^{2}"

    assert latex(diff(f(x), x)**2) == \
        r"\left(\frac{d}{d x} f{\left(x \right)}\right)^{2}"

    assert latex(diff(f(x), (x, n))) == \
        r"\frac{d^{n}}{d x^{n}} f{\left(x \right)}"

    x1 = Symbol('x1')
    x2 = Symbol('x2')
    assert latex(diff(f(x1, x2), x1)) == r'\frac{\partial}{\partial x_{1}} f{\left(x_{1},x_{2} \right)}'

    n1 = Symbol('n1')
    assert latex(diff(f(x), (x, n1))) == r'\frac{d^{n_{1}}}{d x^{n_{1}}} f{\left(x \right)}'

    n2 = Symbol('n2')
    assert latex(diff(f(x), (x, Max(n1, n2)))) == \
        r'\frac{d^{\max\left(n_{1}, n_{2}\right)}}{d x^{\max\left(n_{1}, n_{2}\right)}} f{\left(x \right)}'

    # set diff operator
    assert latex(diff(f(x), x), diff_operator="rd") == r'\frac{\mathrm{d}}{\mathrm{d} x} f{\left(x \right)}'


def test_latex_subs():
    assert latex(Subs(x*y, (x, y), (1, 2))) == r'\left. x y \right|_{\substack{ x=1\\ y=2 }}'


def test_latex_integrals():
    assert latex(Integral(log(x), x)) == r"\int \log{\left(x \right)}\, dx"
    assert latex(Integral(x**2, (x, 0, 1))) == \
        r"\int\limits_{0}^{1} x^{2}\, dx"
    assert latex(Integral(x**2, (x, 10, 20))) == \
        r"\int\limits_{10}^{20} x^{2}\, dx"
    assert latex(Integral(y*x**2, (x, 0, 1), y)) == \
        r"\int\int\limits_{0}^{1} x^{2} y\, dx\, dy"
    assert latex(Integral(y*x**2, (x, 0, 1), y), mode='equation*') == \
        r"\begin{equation*}\int\int\limits_{0}^{1} x^{2} y\, dx\, dy\end{equation*}"
    assert latex(Integral(y*x**2, (x, 0, 1), y), mode='equation*', itex=True) \
        == r"$$\int\int_{0}^{1} x^{2} y\, dx\, dy$$"
    assert latex(Integral(x, (x, 0))) == r"\int\limits^{0} x\, dx"
    assert latex(Integral(x*y, x, y)) == r"\iint x y\, dx\, dy"
    assert latex(Integral(x*y*z, x, y, z)) == r"\iiint x y z\, dx\, dy\, dz"
    assert latex(Integral(x*y*z*t, x, y, z, t)) == \
        r"\iiiint t x y z\, dx\, dy\, dz\, dt"
    assert latex(Integral(x, x, x, x, x, x, x)) == \
        r"\int\int\int\int\int\int x\, dx\, dx\, dx\, dx\, dx\, dx"
    assert latex(Integral(x, x, y, (z, 0, 1))) == \
        r"\int\limits_{0}^{1}\int\int x\, dx\, dy\, dz"

    # for negative nested Integral
    assert latex(Integral(-Integral(y**2,x),x)) == \
        r'\int \left(- \int y^{2}\, dx\right)\, dx'
    assert latex(Integral(-Integral(-Integral(y,x),x),x)) == \
        r'\int \left(- \int \left(- \int y\, dx\right)\, dx\right)\, dx'

    # fix issue #10806
    assert latex(Integral(z, z)**2) == r"\left(\int z\, dz\right)^{2}"
    assert latex(Integral(x + z, z)) == r"\int \left(x + z\right)\, dz"
    assert latex(Integral(x+z/2, z)) == \
        r"\int \left(x + \frac{z}{2}\right)\, dz"
    assert latex(Integral(x**y, z)) == r"\int x^{y}\, dz"

    # set diff operator
    assert latex(Integral(x, x), diff_operator="rd") == r'\int x\, \mathrm{d}x'
    assert latex(Integral(x, (x, 0, 1)), diff_operator="rd") == r'\int\limits_{0}^{1} x\, \mathrm{d}x'


def test_latex_sets():
    for s in (frozenset, set):
        assert latex(s([x*y, x**2])) == r"\left\{x^{2}, x y\right\}"
        assert latex(s(range(1, 6))) == r"\left\{1, 2, 3, 4, 5\right\}"
        assert latex(s(range(1, 13))) == \
            r"\left\{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12\right\}"

    s = FiniteSet
    assert latex(s(*[x*y, x**2])) == r"\left\{x^{2}, x y\right\}"
    assert latex(s(*range(1, 6))) == r"\left\{1, 2, 3, 4, 5\right\}"
    assert latex(s(*range(1, 13))) == \
        r"\left\{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12\right\}"


def test_latex_SetExpr():
    iv = Interval(1, 3)
    se = SetExpr(iv)
    assert latex(se) == r"SetExpr\left(\left[1, 3\right]\right)"


def test_latex_Range():
    assert latex(Range(1, 51)) == r'\left\{1, 2, \ldots, 50\right\}'
    assert latex(Range(1, 4)) == r'\left\{1, 2, 3\right\}'
    assert latex(Range(0, 3, 1)) == r'\left\{0, 1, 2\right\}'
    assert latex(Range(0, 30, 1)) == r'\left\{0, 1, \ldots, 29\right\}'
    assert latex(Range(30, 1, -1)) == r'\left\{30, 29, \ldots, 2\right\}'
    assert latex(Range(0, oo, 2)) == r'\left\{0, 2, \ldots\right\}'
    assert latex(Range(oo, -2, -2)) == r'\left\{\ldots, 2, 0\right\}'
    assert latex(Range(-2, -oo, -1)) == r'\left\{-2, -3, \ldots\right\}'
    assert latex(Range(-oo, oo)) == r'\left\{\ldots, -1, 0, 1, \ldots\right\}'
    assert latex(Range(oo, -oo, -1)) == r'\left\{\ldots, 1, 0, -1, \ldots\right\}'

    a, b, c = symbols('a:c')
    assert latex(Range(a, b, c)) == r'\text{Range}\left(a, b, c\right)'
    assert latex(Range(a, 10, 1)) == r'\text{Range}\left(a, 10\right)'
    assert latex(Range(0, b, 1)) == r'\text{Range}\left(b\right)'
    assert latex(Range(0, 10, c)) == r'\text{Range}\left(0, 10, c\right)'

    i = Symbol('i', integer=True)
    n = Symbol('n', negative=True, integer=True)
    p = Symbol('p', positive=True, integer=True)

    assert latex(Range(i, i + 3)) == r'\left\{i, i + 1, i + 2\right\}'
    assert latex(Range(-oo, n, 2)) == r'\left\{\ldots, n - 4, n - 2\right\}'
    assert latex(Range(p, oo)) == r'\left\{p, p + 1, \ldots\right\}'
    # The following will work if __iter__ is improved
    # assert latex(Range(-3, p + 7)) == r'\left\{-3, -2,  \ldots, p + 6\right\}'
    # Must have integer assumptions
    assert latex(Range(a, a + 3)) == r'\text{Range}\left(a, a + 3\right)'


def test_latex_sequences():
    s1 = SeqFormula(a**2, (0, oo))
    s2 = SeqPer((1, 2))

    latex_str = r'\left[0, 1, 4, 9, \ldots\right]'
    assert latex(s1) == latex_str

    latex_str = r'\left[1, 2, 1, 2, \ldots\right]'
    assert latex(s2) == latex_str

    s3 = SeqFormula(a**2, (0, 2))
    s4 = SeqPer((1, 2), (0, 2))

    latex_str = r'\left[0, 1, 4\right]'
    assert latex(s3) == latex_str

    latex_str = r'\left[1, 2, 1\right]'
    assert latex(s4) == latex_str

    s5 = SeqFormula(a**2, (-oo, 0))
    s6 = SeqPer((1, 2), (-oo, 0))

    latex_str = r'\left[\ldots, 9, 4, 1, 0\right]'
    assert latex(s5) == latex_str

    latex_str = r'\left[\ldots, 2, 1, 2, 1\right]'
    assert latex(s6) == latex_str

    latex_str = r'\left[1, 3, 5, 11, \ldots\right]'
    assert latex(SeqAdd(s1, s2)) == latex_str

    latex_str = r'\left[1, 3, 5\right]'
    assert latex(SeqAdd(s3, s4)) == latex_str

    latex_str = r'\left[\ldots, 11, 5, 3, 1\right]'
    assert latex(SeqAdd(s5, s6)) == latex_str

    latex_str = r'\left[0, 2, 4, 18, \ldots\right]'
    assert latex(SeqMul(s1, s2)) == latex_str

    latex_str = r'\left[0, 2, 4\right]'
    assert latex(SeqMul(s3, s4)) == latex_str

    latex_str = r'\left[\ldots, 18, 4, 2, 0\right]'
    assert latex(SeqMul(s5, s6)) == latex_str

    # Sequences with symbolic limits, issue 12629
    s7 = SeqFormula(a**2, (a, 0, x))
    latex_str = r'\left\{a^{2}\right\}_{a=0}^{x}'
    assert latex(s7) == latex_str

    b = Symbol('b')
    s8 = SeqFormula(b*a**2, (a, 0, 2))
    latex_str = r'\left[0, b, 4 b\right]'
    assert latex(s8) == latex_str


def test_latex_FourierSeries():
    latex_str = \
        r'2 \sin{\left(x \right)} - \sin{\left(2 x \right)} + \frac{2 \sin{\left(3 x \right)}}{3} + \ldots'
    assert latex(fourier_series(x, (x, -pi, pi))) == latex_str


def test_latex_FormalPowerSeries():
    latex_str = r'\sum_{k=1}^{\infty} - \frac{\left(-1\right)^{- k} x^{k}}{k}'
    assert latex(fps(log(1 + x))) == latex_str


def test_latex_intervals():
    a = Symbol('a', real=True)
    assert latex(Interval(0, 0)) == r"\left\{0\right\}"
    assert latex(Interval(0, a)) == r"\left[0, a\right]"
    assert latex(Interval(0, a, False, False)) == r"\left[0, a\right]"
    assert latex(Interval(0, a, True, False)) == r"\left(0, a\right]"
    assert latex(Interval(0, a, False, True)) == r"\left[0, a\right)"
    assert latex(Interval(0, a, True, True)) == r"\left(0, a\right)"


def test_latex_AccumuBounds():
    a = Symbol('a', real=True)
    assert latex(AccumBounds(0, 1)) == r"\left\langle 0, 1\right\rangle"
    assert latex(AccumBounds(0, a)) == r"\left\langle 0, a\right\rangle"
    assert latex(AccumBounds(a + 1, a + 2)) == \
        r"\left\langle a + 1, a + 2\right\rangle"


def test_latex_emptyset():
    assert latex(S.EmptySet) == r"\emptyset"


def test_latex_universalset():
    assert latex(S.UniversalSet) == r"\mathbb{U}"


def test_latex_commutator():
    A = Operator('A')
    B = Operator('B')
    comm = Commutator(B, A)
    assert latex(comm.doit()) == r"- (A B - B A)"


def test_latex_union():
    assert latex(Union(Interval(0, 1), Interval(2, 3))) == \
        r"\left[0, 1\right] \cup \left[2, 3\right]"
    assert latex(Union(Interval(1, 1), Interval(2, 2), Interval(3, 4))) == \
        r"\left\{1, 2\right\} \cup \left[3, 4\right]"


def test_latex_intersection():
    assert latex(Intersection(Interval(0, 1), Interval(x, y))) == \
        r"\left[0, 1\right] \cap \left[x, y\right]"


def test_latex_symmetric_difference():
    assert latex(SymmetricDifference(Interval(2, 5), Interval(4, 7),
                                     evaluate=False)) == \
        r'\left[2, 5\right] \triangle \left[4, 7\right]'


def test_latex_Complement():
    assert latex(Complement(S.Reals, S.Naturals)) == \
        r"\mathbb{R} \setminus \mathbb{N}"


def test_latex_productset():
    line = Interval(0, 1)
    bigline = Interval(0, 10)
    fset = FiniteSet(1, 2, 3)
    assert latex(line**2) == r"%s^{2}" % latex(line)
    assert latex(line**10) == r"%s^{10}" % latex(line)
    assert latex((line * bigline * fset).flatten()) == r"%s \times %s \times %s" % (
        latex(line), latex(bigline), latex(fset))


def test_latex_powerset():
    fset = FiniteSet(1, 2, 3)
    assert latex(PowerSet(fset)) == r'\mathcal{P}\left(\left\{1, 2, 3\right\}\right)'


def test_latex_ordinals():
    w = OrdinalOmega()
    assert latex(w) == r"\omega"
    wp = OmegaPower(2, 3)
    assert latex(wp) == r'3 \omega^{2}'
    assert latex(Ordinal(wp, OmegaPower(1, 1))) == r'3 \omega^{2} + \omega'
    assert latex(Ordinal(OmegaPower(2, 1), OmegaPower(1, 2))) == r'\omega^{2} + 2 \omega'


def test_set_operators_parenthesis():
    a, b, c, d = symbols('a:d')
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
    D1 = SymmetricDifference(A, B, evaluate=False)
    D2 = SymmetricDifference(C, D, evaluate=False)
    # XXX ProductSet does not support evaluate keyword
    P1 = ProductSet(A, B)
    P2 = ProductSet(C, D)

    assert latex(Intersection(A, U2, evaluate=False)) == \
        r'\left\{a\right\} \cap ' \
        r'\left(\left\{c\right\} \cup \left\{d\right\}\right)'
    assert latex(Intersection(U1, U2, evaluate=False)) == \
        r'\left(\left\{a\right\} \cup \left\{b\right\}\right) ' \
        r'\cap \left(\left\{c\right\} \cup \left\{d\right\}\right)'
    assert latex(Intersection(C1, C2, evaluate=False)) == \
        r'\left(\left\{a\right\} \setminus ' \
        r'\left\{b\right\}\right) \cap \left(\left\{c\right\} ' \
        r'\setminus \left\{d\right\}\right)'
    assert latex(Intersection(D1, D2, evaluate=False)) == \
        r'\left(\left\{a\right\} \triangle ' \
        r'\left\{b\right\}\right) \cap \left(\left\{c\right\} ' \
        r'\triangle \left\{d\right\}\right)'
    assert latex(Intersection(P1, P2, evaluate=False)) == \
        r'\left(\left\{a\right\} \times \left\{b\right\}\right) ' \
        r'\cap \left(\left\{c\right\} \times ' \
        r'\left\{d\right\}\right)'

    assert latex(Union(A, I2, evaluate=False)) == \
        r'\left\{a\right\} \cup ' \
        r'\left(\left\{c\right\} \cap \left\{d\right\}\right)'
    assert latex(Union(I1, I2, evaluate=False)) == \
        r'\left(\left\{a\right\} \cap \left\{b\right\}\right) ' \
        r'\cup \left(\left\{c\right\} \cap \left\{d\right\}\right)'
    assert latex(Union(C1, C2, evaluate=False)) == \
        r'\left(\left\{a\right\} \setminus ' \
        r'\left\{b\right\}\right) \cup \left(\left\{c\right\} ' \
        r'\setminus \left\{d\right\}\right)'
    assert latex(Union(D1, D2, evaluate=False)) == \
        r'\left(\left\{a\right\} \triangle ' \
        r'\left\{b\right\}\right) \cup \left(\left\{c\right\} ' \
        r'\triangle \left\{d\right\}\right)'
    assert latex(Union(P1, P2, evaluate=False)) == \
        r'\left(\left\{a\right\} \times \left\{b\right\}\right) ' \
        r'\cup \left(\left\{c\right\} \times ' \
        r'\left\{d\right\}\right)'

    assert latex(Complement(A, C2, evaluate=False)) == \
        r'\left\{a\right\} \setminus \left(\left\{c\right\} ' \
        r'\setminus \left\{d\right\}\right)'
    assert latex(Complement(U1, U2, evaluate=False)) == \
        r'\left(\left\{a\right\} \cup \left\{b\right\}\right) ' \
        r'\setminus \left(\left\{c\right\} \cup ' \
        r'\left\{d\right\}\right)'
    assert latex(Complement(I1, I2, evaluate=False)) == \
        r'\left(\left\{a\right\} \cap \left\{b\right\}\right) ' \
        r'\setminus \left(\left\{c\right\} \cap ' \
        r'\left\{d\right\}\right)'
    assert latex(Complement(D1, D2, evaluate=False)) == \
        r'\left(\left\{a\right\} \triangle ' \
        r'\left\{b\right\}\right) \setminus ' \
        r'\left(\left\{c\right\} \triangle \left\{d\right\}\right)'
    assert latex(Complement(P1, P2, evaluate=False)) == \
        r'\left(\left\{a\right\} \times \left\{b\right\}\right) '\
        r'\setminus \left(\left\{c\right\} \times '\
        r'\left\{d\right\}\right)'

    assert latex(SymmetricDifference(A, D2, evaluate=False)) == \
        r'\left\{a\right\} \triangle \left(\left\{c\right\} ' \
        r'\triangle \left\{d\right\}\right)'
    assert latex(SymmetricDifference(U1, U2, evaluate=False)) == \
        r'\left(\left\{a\right\} \cup \left\{b\right\}\right) ' \
        r'\triangle \left(\left\{c\right\} \cup ' \
        r'\left\{d\right\}\right)'
    assert latex(SymmetricDifference(I1, I2, evaluate=False)) == \
        r'\left(\left\{a\right\} \cap \left\{b\right\}\right) ' \
        r'\triangle \left(\left\{c\right\} \cap ' \
        r'\left\{d\right\}\right)'
    assert latex(SymmetricDifference(C1, C2, evaluate=False)) == \
        r'\left(\left\{a\right\} \setminus ' \
        r'\left\{b\right\}\right) \triangle ' \
        r'\left(\left\{c\right\} \setminus \left\{d\right\}\right)'
    assert latex(SymmetricDifference(P1, P2, evaluate=False)) == \
        r'\left(\left\{a\right\} \times \left\{b\right\}\right) ' \
        r'\triangle \left(\left\{c\right\} \times ' \
        r'\left\{d\right\}\right)'

    # XXX This can be incorrect since cartesian product is not associative
    assert latex(ProductSet(A, P2).flatten()) == \
        r'\left\{a\right\} \times \left\{c\right\} \times ' \
        r'\left\{d\right\}'
    assert latex(ProductSet(U1, U2)) == \
        r'\left(\left\{a\right\} \cup \left\{b\right\}\right) ' \
        r'\times \left(\left\{c\right\} \cup ' \
        r'\left\{d\right\}\right)'
    assert latex(ProductSet(I1, I2)) == \
        r'\left(\left\{a\right\} \cap \left\{b\right\}\right) ' \
        r'\times \left(\left\{c\right\} \cap ' \
        r'\left\{d\right\}\right)'
    assert latex(ProductSet(C1, C2)) == \
        r'\left(\left\{a\right\} \setminus ' \
        r'\left\{b\right\}\right) \times \left(\left\{c\right\} ' \
        r'\setminus \left\{d\right\}\right)'
    assert latex(ProductSet(D1, D2)) == \
        r'\left(\left\{a\right\} \triangle ' \
        r'\left\{b\right\}\right) \times \left(\left\{c\right\} ' \
        r'\triangle \left\{d\right\}\right)'


def test_latex_Complexes():
    assert latex(S.Complexes) == r"\mathbb{C}"


def test_latex_Naturals():
    assert latex(S.Naturals) == r"\mathbb{N}"


def test_latex_Naturals0():
    assert latex(S.Naturals0) == r"\mathbb{N}_0"


def test_latex_Integers():
    assert latex(S.Integers) == r"\mathbb{Z}"


def test_latex_ImageSet():
    x = Symbol('x')
    assert latex(ImageSet(Lambda(x, x**2), S.Naturals)) == \
        r"\left\{x^{2}\; \middle|\; x \in \mathbb{N}\right\}"

    y = Symbol('y')
    imgset = ImageSet(Lambda((x, y), x + y), {1, 2, 3}, {3, 4})
    assert latex(imgset) == \
        r"\left\{x + y\; \middle|\; x \in \left\{1, 2, 3\right\}, y \in \left\{3, 4\right\}\right\}"

    imgset = ImageSet(Lambda(((x, y),), x + y), ProductSet({1, 2, 3}, {3, 4}))
    assert latex(imgset) == \
        r"\left\{x + y\; \middle|\; \left( x, \  y\right) \in \left\{1, 2, 3\right\} \times \left\{3, 4\right\}\right\}"


def test_latex_ConditionSet():
    x = Symbol('x')
    assert latex(ConditionSet(x, Eq(x**2, 1), S.Reals)) == \
        r"\left\{x\; \middle|\; x \in \mathbb{R} \wedge x^{2} = 1 \right\}"
    assert latex(ConditionSet(x, Eq(x**2, 1), S.UniversalSet)) == \
        r"\left\{x\; \middle|\; x^{2} = 1 \right\}"


def test_latex_ComplexRegion():
    assert latex(ComplexRegion(Interval(3, 5)*Interval(4, 6))) == \
        r"\left\{x + y i\; \middle|\; x, y \in \left[3, 5\right] \times \left[4, 6\right] \right\}"
    assert latex(ComplexRegion(Interval(0, 1)*Interval(0, 2*pi), polar=True)) == \
        r"\left\{r \left(i \sin{\left(\theta \right)} + \cos{\left(\theta "\
        r"\right)}\right)\; \middle|\; r, \theta \in \left[0, 1\right] \times \left[0, 2 \pi\right) \right\}"


def test_latex_Contains():
    x = Symbol('x')
    assert latex(Contains(x, S.Naturals)) == r"x \in \mathbb{N}"


def test_latex_sum():
    assert latex(Sum(x*y**2, (x, -2, 2), (y, -5, 5))) == \
        r"\sum_{\substack{-2 \leq x \leq 2\\-5 \leq y \leq 5}} x y^{2}"
    assert latex(Sum(x**2, (x, -2, 2))) == \
        r"\sum_{x=-2}^{2} x^{2}"
    assert latex(Sum(x**2 + y, (x, -2, 2))) == \
        r"\sum_{x=-2}^{2} \left(x^{2} + y\right)"
    assert latex(Sum(x**2 + y, (x, -2, 2))**2) == \
        r"\left(\sum_{x=-2}^{2} \left(x^{2} + y\right)\right)^{2}"


def test_latex_product():
    assert latex(Product(x*y**2, (x, -2, 2), (y, -5, 5))) == \
        r"\prod_{\substack{-2 \leq x \leq 2\\-5 \leq y \leq 5}} x y^{2}"
    assert latex(Product(x**2, (x, -2, 2))) == \
        r"\prod_{x=-2}^{2} x^{2}"
    assert latex(Product(x**2 + y, (x, -2, 2))) == \
        r"\prod_{x=-2}^{2} \left(x^{2} + y\right)"

    assert latex(Product(x, (x, -2, 2))**2) == \
        r"\left(\prod_{x=-2}^{2} x\right)^{2}"


def test_latex_limits():
    assert latex(Limit(x, x, oo)) == r"\lim_{x \to \infty} x"

    # issue 8175
    f = Function('f')
    assert latex(Limit(f(x), x, 0)) == r"\lim_{x \to 0^+} f{\left(x \right)}"
    assert latex(Limit(f(x), x, 0, "-")) == \
        r"\lim_{x \to 0^-} f{\left(x \right)}"

    # issue #10806
    assert latex(Limit(f(x), x, 0)**2) == \
        r"\left(\lim_{x \to 0^+} f{\left(x \right)}\right)^{2}"
    # bi-directional limit
    assert latex(Limit(f(x), x, 0, dir='+-')) == \
        r"\lim_{x \to 0} f{\left(x \right)}"


def test_latex_log():
    assert latex(log(x)) == r"\log{\left(x \right)}"
    assert latex(log(x), ln_notation=True) == r"\ln{\left(x \right)}"
    assert latex(log(x) + log(y)) == \
        r"\log{\left(x \right)} + \log{\left(y \right)}"
    assert latex(log(x) + log(y), ln_notation=True) == \
        r"\ln{\left(x \right)} + \ln{\left(y \right)}"
    assert latex(pow(log(x), x)) == r"\log{\left(x \right)}^{x}"
    assert latex(pow(log(x), x), ln_notation=True) == \
        r"\ln{\left(x \right)}^{x}"


def test_issue_3568():
    beta = Symbol(r'\beta')
    y = beta + x
    assert latex(y) in [r'\beta + x', r'x + \beta']

    beta = Symbol(r'beta')
    y = beta + x
    assert latex(y) in [r'\beta + x', r'x + \beta']


def test_latex():
    assert latex((2*tau)**Rational(7, 2)) == r"8 \sqrt{2} \tau^{\frac{7}{2}}"
    assert latex((2*mu)**Rational(7, 2), mode='equation*') == \
        r"\begin{equation*}8 \sqrt{2} \mu^{\frac{7}{2}}\end{equation*}"
    assert latex((2*mu)**Rational(7, 2), mode='equation', itex=True) == \
        r"$$8 \sqrt{2} \mu^{\frac{7}{2}}$$"
    assert latex([2/x, y]) == r"\left[ \frac{2}{x}, \  y\right]"


def test_latex_dict():
    d = {Rational(1): 1, x**2: 2, x: 3, x**3: 4}
    assert latex(d) == \
        r'\left\{ 1 : 1, \  x : 3, \  x^{2} : 2, \  x^{3} : 4\right\}'
    D = Dict(d)
    assert latex(D) == \
        r'\left\{ 1 : 1, \  x : 3, \  x^{2} : 2, \  x^{3} : 4\right\}'


def test_latex_list():
    ll = [Symbol('omega1'), Symbol('a'), Symbol('alpha')]
    assert latex(ll) == r'\left[ \omega_{1}, \  a, \  \alpha\right]'


def test_latex_NumberSymbols():
    assert latex(S.Catalan) == "G"
    assert latex(S.EulerGamma) == r"\gamma"
    assert latex(S.Exp1) == "e"
    assert latex(S.GoldenRatio) == r"\phi"
    assert latex(S.Pi) == r"\pi"
    assert latex(S.TribonacciConstant) == r"\text{TribonacciConstant}"


def test_latex_rational():
    # tests issue 3973
    assert latex(-Rational(1, 2)) == r"- \frac{1}{2}"
    assert latex(Rational(-1, 2)) == r"- \frac{1}{2}"
    assert latex(Rational(1, -2)) == r"- \frac{1}{2}"
    assert latex(-Rational(-1, 2)) == r"\frac{1}{2}"
    assert latex(-Rational(1, 2)*x) == r"- \frac{x}{2}"
    assert latex(-Rational(1, 2)*x + Rational(-2, 3)*y) == \
        r"- \frac{x}{2} - \frac{2 y}{3}"


def test_latex_inverse():
    # tests issue 4129
    assert latex(1/x) == r"\frac{1}{x}"
    assert latex(1/(x + y)) == r"\frac{1}{x + y}"


def test_latex_DiracDelta():
    assert latex(DiracDelta(x)) == r"\delta\left(x\right)"
    assert latex(DiracDelta(x)**2) == r"\left(\delta\left(x\right)\right)^{2}"
    assert latex(DiracDelta(x, 0)) == r"\delta\left(x\right)"
    assert latex(DiracDelta(x, 5)) == \
        r"\delta^{\left( 5 \right)}\left( x \right)"
    assert latex(DiracDelta(x, 5)**2) == \
        r"\left(\delta^{\left( 5 \right)}\left( x \right)\right)^{2}"


def test_latex_Heaviside():
    assert latex(Heaviside(x)) == r"\theta\left(x\right)"
    assert latex(Heaviside(x)**2) == r"\left(\theta\left(x\right)\right)^{2}"


def test_latex_KroneckerDelta():
    assert latex(KroneckerDelta(x, y)) == r"\delta_{x y}"
    assert latex(KroneckerDelta(x, y + 1)) == r"\delta_{x, y + 1}"
    # issue 6578
    assert latex(KroneckerDelta(x + 1, y)) == r"\delta_{y, x + 1}"
    assert latex(Pow(KroneckerDelta(x, y), 2, evaluate=False)) == \
        r"\left(\delta_{x y}\right)^{2}"


def test_latex_LeviCivita():
    assert latex(LeviCivita(x, y, z)) == r"\varepsilon_{x y z}"
    assert latex(LeviCivita(x, y, z)**2) == \
        r"\left(\varepsilon_{x y z}\right)^{2}"
    assert latex(LeviCivita(x, y, z + 1)) == r"\varepsilon_{x, y, z + 1}"
    assert latex(LeviCivita(x, y + 1, z)) == r"\varepsilon_{x, y + 1, z}"
    assert latex(LeviCivita(x + 1, y, z)) == r"\varepsilon_{x + 1, y, z}"


def test_mode():
    expr = x + y
    assert latex(expr) == r'x + y'
    assert latex(expr, mode='plain') == r'x + y'
    assert latex(expr, mode='inline') == r'$x + y$'
    assert latex(
        expr, mode='equation*') == r'\begin{equation*}x + y\end{equation*}'
    assert latex(
        expr, mode='equation') == r'\begin{equation}x + y\end{equation}'
    raises(ValueError, lambda: latex(expr, mode='foo'))


def test_latex_mathieu():
    assert latex(mathieuc(x, y, z)) == r"C\left(x, y, z\right)"
    assert latex(mathieus(x, y, z)) == r"S\left(x, y, z\right)"
    assert latex(mathieuc(x, y, z)**2) == r"C\left(x, y, z\right)^{2}"
    assert latex(mathieus(x, y, z)**2) == r"S\left(x, y, z\right)^{2}"
    assert latex(mathieucprime(x, y, z)) == r"C^{\prime}\left(x, y, z\right)"
    assert latex(mathieusprime(x, y, z)) == r"S^{\prime}\left(x, y, z\right)"
    assert latex(mathieucprime(x, y, z)**2) == r"C^{\prime}\left(x, y, z\right)^{2}"
    assert latex(mathieusprime(x, y, z)**2) == r"S^{\prime}\left(x, y, z\right)^{2}"

def test_latex_Piecewise():
    p = Piecewise((x, x < 1), (x**2, True))
    assert latex(p) == r"\begin{cases} x & \text{for}\: x < 1 \\x^{2} &" \
                       r" \text{otherwise} \end{cases}"
    assert latex(p, itex=True) == \
        r"\begin{cases} x & \text{for}\: x \lt 1 \\x^{2} &" \
        r" \text{otherwise} \end{cases}"
    p = Piecewise((x, x < 0), (0, x >= 0))
    assert latex(p) == r'\begin{cases} x & \text{for}\: x < 0 \\0 &' \
                       r' \text{otherwise} \end{cases}'
    A, B = symbols("A B", commutative=False)
    p = Piecewise((A**2, Eq(A, B)), (A*B, True))
    s = r"\begin{cases} A^{2} & \text{for}\: A = B \\A B & \text{otherwise} \end{cases}"
    assert latex(p) == s
    assert latex(A*p) == r"A \left(%s\right)" % s
    assert latex(p*A) == r"\left(%s\right) A" % s
    assert latex(Piecewise((x, x < 1), (x**2, x < 2))) == \
        r'\begin{cases} x & ' \
        r'\text{for}\: x < 1 \\x^{2} & \text{for}\: x < 2 \end{cases}'


def test_latex_Matrix():
    M = Matrix([[1 + x, y], [y, x - 1]])
    assert latex(M) == \
        r'\left[\begin{matrix}x + 1 & y\\y & x - 1\end{matrix}\right]'
    assert latex(M, mode='inline') == \
        r'$\left[\begin{smallmatrix}x + 1 & y\\' \
        r'y & x - 1\end{smallmatrix}\right]$'
    assert latex(M, mat_str='array') == \
        r'\left[\begin{array}{cc}x + 1 & y\\y & x - 1\end{array}\right]'
    assert latex(M, mat_str='bmatrix') == \
        r'\left[\begin{bmatrix}x + 1 & y\\y & x - 1\end{bmatrix}\right]'
    assert latex(M, mat_delim=None, mat_str='bmatrix') == \
        r'\begin{bmatrix}x + 1 & y\\y & x - 1\end{bmatrix}'

    M2 = Matrix(1, 11, range(11))
    assert latex(M2) == \
        r'\left[\begin{array}{ccccccccccc}' \
        r'0 & 1 & 2 & 3 & 4 & 5 & 6 & 7 & 8 & 9 & 10\end{array}\right]'


def test_latex_matrix_with_functions():
    t = symbols('t')
    theta1 = symbols('theta1', cls=Function)

    M = Matrix([[sin(theta1(t)), cos(theta1(t))],
                [cos(theta1(t).diff(t)), sin(theta1(t).diff(t))]])

    expected = (r'\left[\begin{matrix}\sin{\left('
                r'\theta_{1}{\left(t \right)} \right)} & '
                r'\cos{\left(\theta_{1}{\left(t \right)} \right)'
                r'}\\\cos{\left(\frac{d}{d t} \theta_{1}{\left(t '
                r'\right)} \right)} & \sin{\left(\frac{d}{d t} '
                r'\theta_{1}{\left(t \right)} \right'
                r')}\end{matrix}\right]')

    assert latex(M) == expected


def test_latex_NDimArray():
    x, y, z, w = symbols("x y z w")

    for ArrayType in (ImmutableDenseNDimArray, ImmutableSparseNDimArray,
                      MutableDenseNDimArray, MutableSparseNDimArray):
        # Basic: scalar array
        M = ArrayType(x)

        assert latex(M) == r"x"

        M = ArrayType([[1 / x, y], [z, w]])
        M1 = ArrayType([1 / x, y, z])

        M2 = tensorproduct(M1, M)
        M3 = tensorproduct(M, M)

        assert latex(M) == \
            r'\left[\begin{matrix}\frac{1}{x} & y\\z & w\end{matrix}\right]'
        assert latex(M1) == \
            r"\left[\begin{matrix}\frac{1}{x} & y & z\end{matrix}\right]"
        assert latex(M2) == \
            r"\left[\begin{matrix}" \
            r"\left[\begin{matrix}\frac{1}{x^{2}} & \frac{y}{x}\\\frac{z}{x} & \frac{w}{x}\end{matrix}\right] & " \
            r"\left[\begin{matrix}\frac{y}{x} & y^{2}\\y z & w y\end{matrix}\right] & " \
            r"\left[\begin{matrix}\frac{z}{x} & y z\\z^{2} & w z\end{matrix}\right]" \
            r"\end{matrix}\right]"
        assert latex(M3) == \
            r"""\left[\begin{matrix}"""\
            r"""\left[\begin{matrix}\frac{1}{x^{2}} & \frac{y}{x}\\\frac{z}{x} & \frac{w}{x}\end{matrix}\right] & """\
            r"""\left[\begin{matrix}\frac{y}{x} & y^{2}\\y z & w y\end{matrix}\right]\\"""\
            r"""\left[\begin{matrix}\frac{z}{x} & y z\\z^{2} & w z\end{matrix}\right] & """\
            r"""\left[\begin{matrix}\frac{w}{x} & w y\\w z & w^{2}\end{matrix}\right]"""\
            r"""\end{matrix}\right]"""

        Mrow = ArrayType([[x, y, 1/z]])
        Mcolumn = ArrayType([[x], [y], [1/z]])
        Mcol2 = ArrayType([Mcolumn.tolist()])

        assert latex(Mrow) == \
            r"\left[\left[\begin{matrix}x & y & \frac{1}{z}\end{matrix}\right]\right]"
        assert latex(Mcolumn) == \
            r"\left[\begin{matrix}x\\y\\\frac{1}{z}\end{matrix}\right]"
        assert latex(Mcol2) == \
            r'\left[\begin{matrix}\left[\begin{matrix}x\\y\\\frac{1}{z}\end{matrix}\right]\end{matrix}\right]'


def test_latex_mul_symbol():
    assert latex(4*4**x, mul_symbol='times') == r"4 \times 4^{x}"
    assert latex(4*4**x, mul_symbol='dot') == r"4 \cdot 4^{x}"
    assert latex(4*4**x, mul_symbol='ldot') == r"4 \,.\, 4^{x}"

    assert latex(4*x, mul_symbol='times') == r"4 \times x"
    assert latex(4*x, mul_symbol='dot') == r"4 \cdot x"
    assert latex(4*x, mul_symbol='ldot') == r"4 \,.\, x"


def test_latex_issue_4381():
    y = 4*4**log(2)
    assert latex(y) == r'4 \cdot 4^{\log{\left(2 \right)}}'
    assert latex(1/y) == r'\frac{1}{4 \cdot 4^{\log{\left(2 \right)}}}'


def test_latex_issue_4576():
    assert latex(Symbol("beta_13_2")) == r"\beta_{13 2}"
    assert latex(Symbol("beta_132_20")) == r"\beta_{132 20}"
    assert latex(Symbol("beta_13")) == r"\beta_{13}"
    assert latex(Symbol("x_a_b")) == r"x_{a b}"
    assert latex(Symbol("x_1_2_3")) == r"x_{1 2 3}"
    assert latex(Symbol("x_a_b1")) == r"x_{a b1}"
    assert latex(Symbol("x_a_1")) == r"x_{a 1}"
    assert latex(Symbol("x_1_a")) == r"x_{1 a}"
    assert latex(Symbol("x_1^aa")) == r"x^{aa}_{1}"
    assert latex(Symbol("x_1__aa")) == r"x^{aa}_{1}"
    assert latex(Symbol("x_11^a")) == r"x^{a}_{11}"
    assert latex(Symbol("x_11__a")) == r"x^{a}_{11}"
    assert latex(Symbol("x_a_a_a_a")) == r"x_{a a a a}"
    assert latex(Symbol("x_a_a^a^a")) == r"x^{a a}_{a a}"
    assert latex(Symbol("x_a_a__a__a")) == r"x^{a a}_{a a}"
    assert latex(Symbol("alpha_11")) == r"\alpha_{11}"
    assert latex(Symbol("alpha_11_11")) == r"\alpha_{11 11}"
    assert latex(Symbol("alpha_alpha")) == r"\alpha_{\alpha}"
    assert latex(Symbol("alpha^aleph")) == r"\alpha^{\aleph}"
    assert latex(Symbol("alpha__aleph")) == r"\alpha^{\aleph}"


def test_latex_pow_fraction():
    x = Symbol('x')
    # Testing exp
    assert r'e^{-x}' in latex(exp(-x)/2).replace(' ', '')  # Remove Whitespace

    # Testing e^{-x} in case future changes alter behavior of muls or fracs
    # In particular current output is \frac{1}{2}e^{- x} but perhaps this will
    # change to \frac{e^{-x}}{2}

    # Testing general, non-exp, power
    assert r'3^{-x}' in latex(3**-x/2).replace(' ', '')


def test_noncommutative():
    A, B, C = symbols('A,B,C', commutative=False)

    assert latex(A*B*C**-1) == r"A B C^{-1}"
    assert latex(C**-1*A*B) == r"C^{-1} A B"
    assert latex(A*C**-1*B) == r"A C^{-1} B"


def test_latex_order():
    expr = x**3 + x**2*y + y**4 + 3*x*y**3

    assert latex(expr, order='lex') == r"x^{3} + x^{2} y + 3 x y^{3} + y^{4}"
    assert latex(
        expr, order='rev-lex') == r"y^{4} + 3 x y^{3} + x^{2} y + x^{3}"
    assert latex(expr, order='none') == r"x^{3} + y^{4} + y x^{2} + 3 x y^{3}"


def test_latex_Lambda():
    assert latex(Lambda(x, x + 1)) == r"\left( x \mapsto x + 1 \right)"
    assert latex(Lambda((x, y), x + 1)) == r"\left( \left( x, \  y\right) \mapsto x + 1 \right)"
    assert latex(Lambda(x, x)) == r"\left( x \mapsto x \right)"

def test_latex_PolyElement():
    Ruv, u, v = ring("u,v", ZZ)
    Rxyz, x, y, z = ring("x,y,z", Ruv)

    assert latex(x - x) == r"0"
    assert latex(x - 1) == r"x - 1"
    assert latex(x + 1) == r"x + 1"

    assert latex((u**2 + 3*u*v + 1)*x**2*y + u + 1) == \
        r"\left({u}^{2} + 3 u v + 1\right) {x}^{2} y + u + 1"
    assert latex((u**2 + 3*u*v + 1)*x**2*y + (u + 1)*x) == \
        r"\left({u}^{2} + 3 u v + 1\right) {x}^{2} y + \left(u + 1\right) x"
    assert latex((u**2 + 3*u*v + 1)*x**2*y + (u + 1)*x + 1) == \
        r"\left({u}^{2} + 3 u v + 1\right) {x}^{2} y + \left(u + 1\right) x + 1"
    assert latex((-u**2 + 3*u*v - 1)*x**2*y - (u + 1)*x - 1) == \
        r"-\left({u}^{2} - 3 u v + 1\right) {x}^{2} y - \left(u + 1\right) x - 1"

    assert latex(-(v**2 + v + 1)*x + 3*u*v + 1) == \
        r"-\left({v}^{2} + v + 1\right) x + 3 u v + 1"
    assert latex(-(v**2 + v + 1)*x - 3*u*v + 1) == \
        r"-\left({v}^{2} + v + 1\right) x - 3 u v + 1"


def test_latex_FracElement():
    Fuv, u, v = field("u,v", ZZ)
    Fxyzt, x, y, z, t = field("x,y,z,t", Fuv)

    assert latex(x - x) == r"0"
    assert latex(x - 1) == r"x - 1"
    assert latex(x + 1) == r"x + 1"

    assert latex(x/3) == r"\frac{x}{3}"
    assert latex(x/z) == r"\frac{x}{z}"
    assert latex(x*y/z) == r"\frac{x y}{z}"
    assert latex(x/(z*t)) == r"\frac{x}{z t}"
    assert latex(x*y/(z*t)) == r"\frac{x y}{z t}"

    assert latex((x - 1)/y) == r"\frac{x - 1}{y}"
    assert latex((x + 1)/y) == r"\frac{x + 1}{y}"
    assert latex((-x - 1)/y) == r"\frac{-x - 1}{y}"
    assert latex((x + 1)/(y*z)) == r"\frac{x + 1}{y z}"
    assert latex(-y/(x + 1)) == r"\frac{-y}{x + 1}"
    assert latex(y*z/(x + 1)) == r"\frac{y z}{x + 1}"

    assert latex(((u + 1)*x*y + 1)/((v - 1)*z - 1)) == \
        r"\frac{\left(u + 1\right) x y + 1}{\left(v - 1\right) z - 1}"
    assert latex(((u + 1)*x*y + 1)/((v - 1)*z - t*u*v - 1)) == \
        r"\frac{\left(u + 1\right) x y + 1}{\left(v - 1\right) z - u v t - 1}"


def test_latex_Poly():
    assert latex(Poly(x**2 + 2 * x, x)) == \
        r"\operatorname{Poly}{\left( x^{2} + 2 x, x, domain=\mathbb{Z} \right)}"
    assert latex(Poly(x/y, x)) == \
        r"\operatorname{Poly}{\left( \frac{1}{y} x, x, domain=\mathbb{Z}\left(y\right) \right)}"
    assert latex(Poly(2.0*x + y)) == \
        r"\operatorname{Poly}{\left( 2.0 x + 1.0 y, x, y, domain=\mathbb{R} \right)}"


def test_latex_Poly_order():
    assert latex(Poly([a, 1, b, 2, c, 3], x)) == \
        r'\operatorname{Poly}{\left( a x^{5} + x^{4} + b x^{3} + 2 x^{2} + c'\
        r' x + 3, x, domain=\mathbb{Z}\left[a, b, c\right] \right)}'
    assert latex(Poly([a, 1, b+c, 2, 3], x)) == \
        r'\operatorname{Poly}{\left( a x^{4} + x^{3} + \left(b + c\right) '\
        r'x^{2} + 2 x + 3, x, domain=\mathbb{Z}\left[a, b, c\right] \right)}'
    assert latex(Poly(a*x**3 + x**2*y - x*y - c*y**3 - b*x*y**2 + y - a*x + b,
                      (x, y))) == \
        r'\operatorname{Poly}{\left( a x^{3} + x^{2}y -  b xy^{2} - xy -  '\
        r'a x -  c y^{3} + y + b, x, y, domain=\mathbb{Z}\left[a, b, c\right] \right)}'


def test_latex_ComplexRootOf():
    assert latex(rootof(x**5 + x + 3, 0)) == \
        r"\operatorname{CRootOf} {\left(x^{5} + x + 3, 0\right)}"


def test_latex_RootSum():
    assert latex(RootSum(x**5 + x + 3, sin)) == \
        r"\operatorname{RootSum} {\left(x^{5} + x + 3, \left( x \mapsto \sin{\left(x \right)} \right)\right)}"


def test_settings():
    raises(TypeError, lambda: latex(x*y, method="garbage"))


def test_latex_numbers():
    assert latex(catalan(n)) == r"C_{n}"
    assert latex(catalan(n)**2) == r"C_{n}^{2}"
    assert latex(bernoulli(n)) == r"B_{n}"
    assert latex(bernoulli(n, x)) == r"B_{n}\left(x\right)"
    assert latex(bernoulli(n)**2) == r"B_{n}^{2}"
    assert latex(bernoulli(n, x)**2) == r"B_{n}^{2}\left(x\right)"
    assert latex(genocchi(n)) == r"G_{n}"
    assert latex(genocchi(n, x)) == r"G_{n}\left(x\right)"
    assert latex(genocchi(n)**2) == r"G_{n}^{2}"
    assert latex(genocchi(n, x)**2) == r"G_{n}^{2}\left(x\right)"
    assert latex(bell(n)) == r"B_{n}"
    assert latex(bell(n, x)) == r"B_{n}\left(x\right)"
    assert latex(bell(n, m, (x, y))) == r"B_{n, m}\left(x, y\right)"
    assert latex(bell(n)**2) == r"B_{n}^{2}"
    assert latex(bell(n, x)**2) == r"B_{n}^{2}\left(x\right)"
    assert latex(bell(n, m, (x, y))**2) == r"B_{n, m}^{2}\left(x, y\right)"
    assert latex(fibonacci(n)) == r"F_{n}"
    assert latex(fibonacci(n, x)) == r"F_{n}\left(x\right)"
    assert latex(fibonacci(n)**2) == r"F_{n}^{2}"
    assert latex(fibonacci(n, x)**2) == r"F_{n}^{2}\left(x\right)"
    assert latex(lucas(n)) == r"L_{n}"
    assert latex(lucas(n)**2) == r"L_{n}^{2}"
    assert latex(tribonacci(n)) == r"T_{n}"
    assert latex(tribonacci(n, x)) == r"T_{n}\left(x\right)"
    assert latex(tribonacci(n)**2) == r"T_{n}^{2}"
    assert latex(tribonacci(n, x)**2) == r"T_{n}^{2}\left(x\right)"
    assert latex(mobius(n)) == r"\mu\left(n\right)"
    assert latex(mobius(n)**2) == r"\mu^{2}\left(n\right)"


def test_latex_euler():
    assert latex(euler(n)) == r"E_{n}"
    assert latex(euler(n, x)) == r"E_{n}\left(x\right)"
    assert latex(euler(n, x)**2) == r"E_{n}^{2}\left(x\right)"


def test_lamda():
    assert latex(Symbol('lamda')) == r"\lambda"
    assert latex(Symbol('Lamda')) == r"\Lambda"


def test_custom_symbol_names():
    x = Symbol('x')
    y = Symbol('y')
    assert latex(x) == r"x"
    assert latex(x, symbol_names={x: "x_i"}) == r"x_i"
    assert latex(x + y, symbol_names={x: "x_i"}) == r"x_i + y"
    assert latex(x**2, symbol_names={x: "x_i"}) == r"x_i^{2}"
    assert latex(x + y, symbol_names={x: "x_i", y: "y_j"}) == r"x_i + y_j"


def test_matAdd():
    C = MatrixSymbol('C', 5, 5)
    B = MatrixSymbol('B', 5, 5)

    n = symbols("n")
    h = MatrixSymbol("h", 1, 1)

    assert latex(C - 2*B) in [r'- 2 B + C', r'C -2 B']
    assert latex(C + 2*B) in [r'2 B + C', r'C + 2 B']
    assert latex(B - 2*C) in [r'B - 2 C', r'- 2 C + B']
    assert latex(B + 2*C) in [r'B + 2 C', r'2 C + B']

    assert latex(n * h - (-h + h.T) * (h + h.T)) == 'n h - \\left(- h + h^{T}\\right) \\left(h + h^{T}\\right)'
    assert latex(MatAdd(MatAdd(h, h), MatAdd(h, h))) == '\\left(h + h\\right) + \\left(h + h\\right)'
    assert latex(MatMul(MatMul(h, h), MatMul(h, h))) == '\\left(h h\\right) \\left(h h\\right)'


def test_matMul():
    A = MatrixSymbol('A', 5, 5)
    B = MatrixSymbol('B', 5, 5)
    x = Symbol('x')
    assert latex(2*A) == r'2 A'
    assert latex(2*x*A) == r'2 x A'
    assert latex(-2*A) == r'- 2 A'
    assert latex(1.5*A) == r'1.5 A'
    assert latex(sqrt(2)*A) == r'\sqrt{2} A'
    assert latex(-sqrt(2)*A) == r'- \sqrt{2} A'
    assert latex(2*sqrt(2)*x*A) == r'2 \sqrt{2} x A'
    assert latex(-2*A*(A + 2*B)) in [r'- 2 A \left(A + 2 B\right)',
                                        r'- 2 A \left(2 B + A\right)']


def test_latex_MatrixSlice():
    n = Symbol('n', integer=True)
    x, y, z, w, t, = symbols('x y z w t')
    X = MatrixSymbol('X', n, n)
    Y = MatrixSymbol('Y', 10, 10)
    Z = MatrixSymbol('Z', 10, 10)

    assert latex(MatrixSlice(X, (None, None, None), (None, None, None))) == r'X\left[:, :\right]'
    assert latex(X[x:x + 1, y:y + 1]) == r'X\left[x:x + 1, y:y + 1\right]'
    assert latex(X[x:x + 1:2, y:y + 1:2]) == r'X\left[x:x + 1:2, y:y + 1:2\right]'
    assert latex(X[:x, y:]) == r'X\left[:x, y:\right]'
    assert latex(X[:x, y:]) == r'X\left[:x, y:\right]'
    assert latex(X[x:, :y]) == r'X\left[x:, :y\right]'
    assert latex(X[x:y, z:w]) == r'X\left[x:y, z:w\right]'
    assert latex(X[x:y:t, w:t:x]) == r'X\left[x:y:t, w:t:x\right]'
    assert latex(X[x::y, t::w]) == r'X\left[x::y, t::w\right]'
    assert latex(X[:x:y, :t:w]) == r'X\left[:x:y, :t:w\right]'
    assert latex(X[::x, ::y]) == r'X\left[::x, ::y\right]'
    assert latex(MatrixSlice(X, (0, None, None), (0, None, None))) == r'X\left[:, :\right]'
    assert latex(MatrixSlice(X, (None, n, None), (None, n, None))) == r'X\left[:, :\right]'
    assert latex(MatrixSlice(X, (0, n, None), (0, n, None))) == r'X\left[:, :\right]'
    assert latex(MatrixSlice(X, (0, n, 2), (0, n, 2))) == r'X\left[::2, ::2\right]'
    assert latex(X[1:2:3, 4:5:6]) == r'X\left[1:2:3, 4:5:6\right]'
    assert latex(X[1:3:5, 4:6:8]) == r'X\left[1:3:5, 4:6:8\right]'
    assert latex(X[1:10:2]) == r'X\left[1:10:2, :\right]'
    assert latex(Y[:5, 1:9:2]) == r'Y\left[:5, 1:9:2\right]'
    assert latex(Y[:5, 1:10:2]) == r'Y\left[:5, 1::2\right]'
    assert latex(Y[5, :5:2]) == r'Y\left[5:6, :5:2\right]'
    assert latex(X[0:1, 0:1]) == r'X\left[:1, :1\right]'
    assert latex(X[0:1:2, 0:1:2]) == r'X\left[:1:2, :1:2\right]'
    assert latex((Y + Z)[2:, 2:]) == r'\left(Y + Z\right)\left[2:, 2:\right]'


def test_latex_RandomDomain():
    from sympy.stats import Normal, Die, Exponential, pspace, where
    from sympy.stats.rv import RandomDomain

    X = Normal('x1', 0, 1)
    assert latex(where(X > 0)) == r"\text{Domain: }0 < x_{1} \wedge x_{1} < \infty"

    D = Die('d1', 6)
    assert latex(where(D > 4)) == r"\text{Domain: }d_{1} = 5 \vee d_{1} = 6"

    A = Exponential('a', 1)
    B = Exponential('b', 1)
    assert latex(
        pspace(Tuple(A, B)).domain) == \
        r"\text{Domain: }0 \leq a \wedge 0 \leq b \wedge a < \infty \wedge b < \infty"

    assert latex(RandomDomain(FiniteSet(x), FiniteSet(1, 2))) == \
        r'\text{Domain: }\left\{x\right\} \in \left\{1, 2\right\}'

def test_PrettyPoly():
    from sympy.polys.domains import QQ
    F = QQ.frac_field(x, y)
    R = QQ[x, y]

    assert latex(F.convert(x/(x + y))) == latex(x/(x + y))
    assert latex(R.convert(x + y)) == latex(x + y)


def test_integral_transforms():
    x = Symbol("x")
    k = Symbol("k")
    f = Function("f")
    a = Symbol("a")
    b = Symbol("b")

    assert latex(MellinTransform(f(x), x, k)) == \
        r"\mathcal{M}_{x}\left[f{\left(x \right)}\right]\left(k\right)"
    assert latex(InverseMellinTransform(f(k), k, x, a, b)) == \
        r"\mathcal{M}^{-1}_{k}\left[f{\left(k \right)}\right]\left(x\right)"

    assert latex(LaplaceTransform(f(x), x, k)) == \
        r"\mathcal{L}_{x}\left[f{\left(x \right)}\right]\left(k\right)"
    assert latex(InverseLaplaceTransform(f(k), k, x, (a, b))) == \
        r"\mathcal{L}^{-1}_{k}\left[f{\left(k \right)}\right]\left(x\right)"

    assert latex(FourierTransform(f(x), x, k)) == \
        r"\mathcal{F}_{x}\left[f{\left(x \right)}\right]\left(k\right)"
    assert latex(InverseFourierTransform(f(k), k, x)) == \
        r"\mathcal{F}^{-1}_{k}\left[f{\left(k \right)}\right]\left(x\right)"

    assert latex(CosineTransform(f(x), x, k)) == \
        r"\mathcal{COS}_{x}\left[f{\left(x \right)}\right]\left(k\right)"
    assert latex(InverseCosineTransform(f(k), k, x)) == \
        r"\mathcal{COS}^{-1}_{k}\left[f{\left(k \right)}\right]\left(x\right)"

    assert latex(SineTransform(f(x), x, k)) == \
        r"\mathcal{SIN}_{x}\left[f{\left(x \right)}\right]\left(k\right)"
    assert latex(InverseSineTransform(f(k), k, x)) == \
        r"\mathcal{SIN}^{-1}_{k}\left[f{\left(k \right)}\right]\left(x\right)"


def test_PolynomialRingBase():
    from sympy.polys.domains import QQ
    assert latex(QQ.old_poly_ring(x, y)) == r"\mathbb{Q}\left[x, y\right]"
    assert latex(QQ.old_poly_ring(x, y, order="ilex")) == \
        r"S_<^{-1}\mathbb{Q}\left[x, y\right]"


def test_categories():
    from sympy.categories import (Object, IdentityMorphism,
                                  NamedMorphism, Category, Diagram,
                                  DiagramGrid)

    A1 = Object("A1")
    A2 = Object("A2")
    A3 = Object("A3")

    f1 = NamedMorphism(A1, A2, "f1")
    f2 = NamedMorphism(A2, A3, "f2")
    id_A1 = IdentityMorphism(A1)

    K1 = Category("K1")

    assert latex(A1) == r"A_{1}"
    assert latex(f1) == r"f_{1}:A_{1}\rightarrow A_{2}"
    assert latex(id_A1) == r"id:A_{1}\rightarrow A_{1}"
    assert latex(f2*f1) == r"f_{2}\circ f_{1}:A_{1}\rightarrow A_{3}"

    assert latex(K1) == r"\mathbf{K_{1}}"

    d = Diagram()
    assert latex(d) == r"\emptyset"

    d = Diagram({f1: "unique", f2: S.EmptySet})
    assert latex(d) == r"\left\{ f_{2}\circ f_{1}:A_{1}" \
        r"\rightarrow A_{3} : \emptyset, \  id:A_{1}\rightarrow " \
        r"A_{1} : \emptyset, \  id:A_{2}\rightarrow A_{2} : " \
        r"\emptyset, \  id:A_{3}\rightarrow A_{3} : \emptyset, " \
        r"\  f_{1}:A_{1}\rightarrow A_{2} : \left\{unique\right\}, " \
        r"\  f_{2}:A_{2}\rightarrow A_{3} : \emptyset\right\}"

    d = Diagram({f1: "unique", f2: S.EmptySet}, {f2 * f1: "unique"})
    assert latex(d) == r"\left\{ f_{2}\circ f_{1}:A_{1}" \
        r"\rightarrow A_{3} : \emptyset, \  id:A_{1}\rightarrow " \
        r"A_{1} : \emptyset, \  id:A_{2}\rightarrow A_{2} : " \
        r"\emptyset, \  id:A_{3}\rightarrow A_{3} : \emptyset, " \
        r"\  f_{1}:A_{1}\rightarrow A_{2} : \left\{unique\right\}," \
        r" \  f_{2}:A_{2}\rightarrow A_{3} : \emptyset\right\}" \
        r"\Longrightarrow \left\{ f_{2}\circ f_{1}:A_{1}" \
        r"\rightarrow A_{3} : \left\{unique\right\}\right\}"

    # A linear diagram.
    A = Object("A")
    B = Object("B")
    C = Object("C")
    f = NamedMorphism(A, B, "f")
    g = NamedMorphism(B, C, "g")
    d = Diagram([f, g])
    grid = DiagramGrid(d)

    assert latex(grid) == r"\begin{array}{cc}" + "\n" \
        r"A & B \\" + "\n" \
        r" & C " + "\n" \
        r"\end{array}" + "\n"


def test_Modules():
    from sympy.polys.domains import QQ
    from sympy.polys.agca import homomorphism

    R = QQ.old_poly_ring(x, y)
    F = R.free_module(2)
    M = F.submodule([x, y], [1, x**2])

    assert latex(F) == r"{\mathbb{Q}\left[x, y\right]}^{2}"
    assert latex(M) == \
        r"\left\langle {\left[ {x},{y} \right]},{\left[ {1},{x^{2}} \right]} \right\rangle"

    I = R.ideal(x**2, y)
    assert latex(I) == r"\left\langle {x^{2}},{y} \right\rangle"

    Q = F / M
    assert latex(Q) == \
        r"\frac{{\mathbb{Q}\left[x, y\right]}^{2}}{\left\langle {\left[ {x},"\
        r"{y} \right]},{\left[ {1},{x^{2}} \right]} \right\rangle}"
    assert latex(Q.submodule([1, x**3/2], [2, y])) == \
        r"\left\langle {{\left[ {1},{\frac{x^{3}}{2}} \right]} + {\left"\
        r"\langle {\left[ {x},{y} \right]},{\left[ {1},{x^{2}} \right]} "\
        r"\right\rangle}},{{\left[ {2},{y} \right]} + {\left\langle {\left[ "\
        r"{x},{y} \right]},{\left[ {1},{x^{2}} \right]} \right\rangle}} \right\rangle"

    h = homomorphism(QQ.old_poly_ring(x).free_module(2),
                     QQ.old_poly_ring(x).free_module(2), [0, 0])

    assert latex(h) == \
        r"{\left[\begin{matrix}0 & 0\\0 & 0\end{matrix}\right]} : "\
        r"{{\mathbb{Q}\left[x\right]}^{2}} \to {{\mathbb{Q}\left[x\right]}^{2}}"


def test_QuotientRing():
    from sympy.polys.domains import QQ
    R = QQ.old_poly_ring(x)/[x**2 + 1]

    assert latex(R) == \
        r"\frac{\mathbb{Q}\left[x\right]}{\left\langle {x^{2} + 1} \right\rangle}"
    assert latex(R.one) == r"{1} + {\left\langle {x^{2} + 1} \right\rangle}"


def test_Tr():
    #TODO: Handle indices
    A, B = symbols('A B', commutative=False)
    t = Tr(A*B)
    assert latex(t) == r'\operatorname{tr}\left(A B\right)'


def test_Determinant():
    from sympy.matrices import Determinant, Inverse, BlockMatrix, OneMatrix, ZeroMatrix
    m = Matrix(((1, 2), (3, 4)))
    assert latex(Determinant(m)) == '\\left|{\\begin{matrix}1 & 2\\\\3 & 4\\end{matrix}}\\right|'
    assert latex(Determinant(Inverse(m))) == \
        '\\left|{\\left[\\begin{matrix}1 & 2\\\\3 & 4\\end{matrix}\\right]^{-1}}\\right|'
    X = MatrixSymbol('X', 2, 2)
    assert latex(Determinant(X)) == '\\left|{X}\\right|'
    assert latex(Determinant(X + m)) == \
        '\\left|{\\left[\\begin{matrix}1 & 2\\\\3 & 4\\end{matrix}\\right] + X}\\right|'
    assert latex(Determinant(BlockMatrix(((OneMatrix(2, 2), X),
                                          (m, ZeroMatrix(2, 2)))))) == \
        '\\left|{\\begin{matrix}1 & X\\\\\\left[\\begin{matrix}1 & 2\\\\3 & 4\\end{matrix}\\right] & 0\\end{matrix}}\\right|'


def test_Adjoint():
    from sympy.matrices import Adjoint, Inverse, Transpose
    X = MatrixSymbol('X', 2, 2)
    Y = MatrixSymbol('Y', 2, 2)
    assert latex(Adjoint(X)) == r'X^{\dagger}'
    assert latex(Adjoint(X + Y)) == r'\left(X + Y\right)^{\dagger}'
    assert latex(Adjoint(X) + Adjoint(Y)) == r'X^{\dagger} + Y^{\dagger}'
    assert latex(Adjoint(X*Y)) == r'\left(X Y\right)^{\dagger}'
    assert latex(Adjoint(Y)*Adjoint(X)) == r'Y^{\dagger} X^{\dagger}'
    assert latex(Adjoint(X**2)) == r'\left(X^{2}\right)^{\dagger}'
    assert latex(Adjoint(X)**2) == r'\left(X^{\dagger}\right)^{2}'
    assert latex(Adjoint(Inverse(X))) == r'\left(X^{-1}\right)^{\dagger}'
    assert latex(Inverse(Adjoint(X))) == r'\left(X^{\dagger}\right)^{-1}'
    assert latex(Adjoint(Transpose(X))) == r'\left(X^{T}\right)^{\dagger}'
    assert latex(Transpose(Adjoint(X))) == r'\left(X^{\dagger}\right)^{T}'
    assert latex(Transpose(Adjoint(X) + Y)) == r'\left(X^{\dagger} + Y\right)^{T}'
    m = Matrix(((1, 2), (3, 4)))
    assert latex(Adjoint(m)) == '\\left[\\begin{matrix}1 & 2\\\\3 & 4\\end{matrix}\\right]^{\\dagger}'
    assert latex(Adjoint(m+X)) == \
        '\\left(\\left[\\begin{matrix}1 & 2\\\\3 & 4\\end{matrix}\\right] + X\\right)^{\\dagger}'
    from sympy.matrices import BlockMatrix, OneMatrix, ZeroMatrix
    assert latex(Adjoint(BlockMatrix(((OneMatrix(2, 2), X),
                                      (m, ZeroMatrix(2, 2)))))) == \
        '\\left[\\begin{matrix}1 & X\\\\\\left[\\begin{matrix}1 & 2\\\\3 & 4\\end{matrix}\\right] & 0\\end{matrix}\\right]^{\\dagger}'
    # Issue 20959
    Mx = MatrixSymbol('M^x', 2, 2)
    assert latex(Adjoint(Mx)) == r'\left(M^{x}\right)^{\dagger}'

    # adjoint style
    assert latex(Adjoint(X), adjoint_style="star") == r'X^{\ast}'
    assert latex(Adjoint(X + Y), adjoint_style="hermitian") == r'\left(X + Y\right)^{\mathsf{H}}'
    assert latex(Adjoint(X) + Adjoint(Y), adjoint_style="dagger") == r'X^{\dagger} + Y^{\dagger}'
    assert latex(Adjoint(Y)*Adjoint(X)) == r'Y^{\dagger} X^{\dagger}'
    assert latex(Adjoint(X**2), adjoint_style="star") == r'\left(X^{2}\right)^{\ast}'
    assert latex(Adjoint(X)**2, adjoint_style="hermitian") == r'\left(X^{\mathsf{H}}\right)^{2}'

def test_Transpose():
    from sympy.matrices import Transpose, MatPow, HadamardPower
    X = MatrixSymbol('X', 2, 2)
    Y = MatrixSymbol('Y', 2, 2)
    assert latex(Transpose(X)) == r'X^{T}'
    assert latex(Transpose(X + Y)) == r'\left(X + Y\right)^{T}'

    assert latex(Transpose(HadamardPower(X, 2))) == r'\left(X^{\circ {2}}\right)^{T}'
    assert latex(HadamardPower(Transpose(X), 2)) == r'\left(X^{T}\right)^{\circ {2}}'
    assert latex(Transpose(MatPow(X, 2))) == r'\left(X^{2}\right)^{T}'
    assert latex(MatPow(Transpose(X), 2)) == r'\left(X^{T}\right)^{2}'
    m = Matrix(((1, 2), (3, 4)))
    assert latex(Transpose(m)) == '\\left[\\begin{matrix}1 & 2\\\\3 & 4\\end{matrix}\\right]^{T}'
    assert latex(Transpose(m+X)) == \
        '\\left(\\left[\\begin{matrix}1 & 2\\\\3 & 4\\end{matrix}\\right] + X\\right)^{T}'
    from sympy.matrices import BlockMatrix, OneMatrix, ZeroMatrix
    assert latex(Transpose(BlockMatrix(((OneMatrix(2, 2), X),
                                        (m, ZeroMatrix(2, 2)))))) == \
        '\\left[\\begin{matrix}1 & X\\\\\\left[\\begin{matrix}1 & 2\\\\3 & 4\\end{matrix}\\right] & 0\\end{matrix}\\right]^{T}'
    # Issue 20959
    Mx = MatrixSymbol('M^x', 2, 2)
    assert latex(Transpose(Mx)) == r'\left(M^{x}\right)^{T}'


def test_Hadamard():
    from sympy.matrices import HadamardProduct, HadamardPower
    from sympy.matrices.expressions import MatAdd, MatMul, MatPow
    X = MatrixSymbol('X', 2, 2)
    Y = MatrixSymbol('Y', 2, 2)
    assert latex(HadamardProduct(X, Y*Y)) == r'X \circ Y^{2}'
    assert latex(HadamardProduct(X, Y)*Y) == r'\left(X \circ Y\right) Y'

    assert latex(HadamardPower(X, 2)) == r'X^{\circ {2}}'
    assert latex(HadamardPower(X, -1)) == r'X^{\circ \left({-1}\right)}'
    assert latex(HadamardPower(MatAdd(X, Y), 2)) == \
        r'\left(X + Y\right)^{\circ {2}}'
    assert latex(HadamardPower(MatMul(X, Y), 2)) == \
        r'\left(X Y\right)^{\circ {2}}'

    assert latex(HadamardPower(MatPow(X, -1), -1)) == \
        r'\left(X^{-1}\right)^{\circ \left({-1}\right)}'
    assert latex(MatPow(HadamardPower(X, -1), -1)) == \
        r'\left(X^{\circ \left({-1}\right)}\right)^{-1}'

    assert latex(HadamardPower(X, n+1)) == \
        r'X^{\circ \left({n + 1}\right)}'


def test_MatPow():
    from sympy.matrices.expressions import MatPow
    X = MatrixSymbol('X', 2, 2)
    Y = MatrixSymbol('Y', 2, 2)
    assert latex(MatPow(X, 2)) == 'X^{2}'
    assert latex(MatPow(X*X, 2)) == '\\left(X^{2}\\right)^{2}'
    assert latex(MatPow(X*Y, 2)) == '\\left(X Y\\right)^{2}'
    assert latex(MatPow(X + Y, 2)) == '\\left(X + Y\\right)^{2}'
    assert latex(MatPow(X + X, 2)) == '\\left(2 X\\right)^{2}'
    # Issue 20959
    Mx = MatrixSymbol('M^x', 2, 2)
    assert latex(MatPow(Mx, 2)) == r'\left(M^{x}\right)^{2}'


def test_ElementwiseApplyFunction():
    X = MatrixSymbol('X', 2, 2)
    expr = (X.T*X).applyfunc(sin)
    assert latex(expr) == r"{\left( d \mapsto \sin{\left(d \right)} \right)}_{\circ}\left({X^{T} X}\right)"
    expr = X.applyfunc(Lambda(x, 1/x))
    assert latex(expr) == r'{\left( x \mapsto \frac{1}{x} \right)}_{\circ}\left({X}\right)'


def test_ZeroMatrix():
    from sympy.matrices.expressions.special import ZeroMatrix
    assert latex(ZeroMatrix(1, 1), mat_symbol_style='plain') == r"0"
    assert latex(ZeroMatrix(1, 1), mat_symbol_style='bold') == r"\mathbf{0}"


def test_OneMatrix():
    from sympy.matrices.expressions.special import OneMatrix
    assert latex(OneMatrix(3, 4), mat_symbol_style='plain') == r"1"
    assert latex(OneMatrix(3, 4), mat_symbol_style='bold') == r"\mathbf{1}"


def test_Identity():
    from sympy.matrices.expressions.special import Identity
    assert latex(Identity(1), mat_symbol_style='plain') == r"\mathbb{I}"
    assert latex(Identity(1), mat_symbol_style='bold') == r"\mathbf{I}"


def test_latex_DFT_IDFT():
    from sympy.matrices.expressions.fourier import DFT, IDFT
    assert latex(DFT(13)) == r"\text{DFT}_{13}"
    assert latex(IDFT(x)) == r"\text{IDFT}_{x}"


def test_boolean_args_order():
    syms = symbols('a:f')

    expr = And(*syms)
    assert latex(expr) == r'a \wedge b \wedge c \wedge d \wedge e \wedge f'

    expr = Or(*syms)
    assert latex(expr) == r'a \vee b \vee c \vee d \vee e \vee f'

    expr = Equivalent(*syms)
    assert latex(expr) == \
        r'a \Leftrightarrow b \Leftrightarrow c \Leftrightarrow d \Leftrightarrow e \Leftrightarrow f'

    expr = Xor(*syms)
    assert latex(expr) == \
        r'a \veebar b \veebar c \veebar d \veebar e \veebar f'


def test_imaginary():
    i = sqrt(-1)
    assert latex(i) == r'i'


def test_builtins_without_args():
    assert latex(sin) == r'\sin'
    assert latex(cos) == r'\cos'
    assert latex(tan) == r'\tan'
    assert latex(log) == r'\log'
    assert latex(Ei) == r'\operatorname{Ei}'
    assert latex(zeta) == r'\zeta'


def test_latex_greek_functions():
    # bug because capital greeks that have roman equivalents should not use
    # \Alpha, \Beta, \Eta, etc.
    s = Function('Alpha')
    assert latex(s) == r'\mathrm{A}'
    assert latex(s(x)) == r'\mathrm{A}{\left(x \right)}'
    s = Function('Beta')
    assert latex(s) == r'\mathrm{B}'
    s = Function('Eta')
    assert latex(s) == r'\mathrm{H}'
    assert latex(s(x)) == r'\mathrm{H}{\left(x \right)}'

    # bug because sympy.core.numbers.Pi is special
    p = Function('Pi')
    # assert latex(p(x)) == r'\Pi{\left(x \right)}'
    assert latex(p) == r'\Pi'

    # bug because not all greeks are included
    c = Function('chi')
    assert latex(c(x)) == r'\chi{\left(x \right)}'
    assert latex(c) == r'\chi'


def test_translate():
    s = 'Alpha'
    assert translate(s) == r'\mathrm{A}'
    s = 'Beta'
    assert translate(s) == r'\mathrm{B}'
    s = 'Eta'
    assert translate(s) == r'\mathrm{H}'
    s = 'omicron'
    assert translate(s) == r'o'
    s = 'Pi'
    assert translate(s) == r'\Pi'
    s = 'pi'
    assert translate(s) == r'\pi'
    s = 'LamdaHatDOT'
    assert translate(s) == r'\dot{\hat{\Lambda}}'


def test_other_symbols():
    from sympy.printing.latex import other_symbols
    for s in other_symbols:
        assert latex(symbols(s)) == r"" "\\" + s


def test_modifiers():
    # Test each modifier individually in the simplest case
    # (with funny capitalizations)
    assert latex(symbols("xMathring")) == r"\mathring{x}"
    assert latex(symbols("xCheck")) == r"\check{x}"
    assert latex(symbols("xBreve")) == r"\breve{x}"
    assert latex(symbols("xAcute")) == r"\acute{x}"
    assert latex(symbols("xGrave")) == r"\grave{x}"
    assert latex(symbols("xTilde")) == r"\tilde{x}"
    assert latex(symbols("xPrime")) == r"{x}'"
    assert latex(symbols("xddDDot")) == r"\ddddot{x}"
    assert latex(symbols("xDdDot")) == r"\dddot{x}"
    assert latex(symbols("xDDot")) == r"\ddot{x}"
    assert latex(symbols("xBold")) == r"\boldsymbol{x}"
    assert latex(symbols("xnOrM")) == r"\left\|{x}\right\|"
    assert latex(symbols("xAVG")) == r"\left\langle{x}\right\rangle"
    assert latex(symbols("xHat")) == r"\hat{x}"
    assert latex(symbols("xDot")) == r"\dot{x}"
    assert latex(symbols("xBar")) == r"\bar{x}"
    assert latex(symbols("xVec")) == r"\vec{x}"
    assert latex(symbols("xAbs")) == r"\left|{x}\right|"
    assert latex(symbols("xMag")) == r"\left|{x}\right|"
    assert latex(symbols("xPrM")) == r"{x}'"
    assert latex(symbols("xBM")) == r"\boldsymbol{x}"
    # Test strings that are *only* the names of modifiers
    assert latex(symbols("Mathring")) == r"Mathring"
    assert latex(symbols("Check")) == r"Check"
    assert latex(symbols("Breve")) == r"Breve"
    assert latex(symbols("Acute")) == r"Acute"
    assert latex(symbols("Grave")) == r"Grave"
    assert latex(symbols("Tilde")) == r"Tilde"
    assert latex(symbols("Prime")) == r"Prime"
    assert latex(symbols("DDot")) == r"\dot{D}"
    assert latex(symbols("Bold")) == r"Bold"
    assert latex(symbols("NORm")) == r"NORm"
    assert latex(symbols("AVG")) == r"AVG"
    assert latex(symbols("Hat")) == r"Hat"
    assert latex(symbols("Dot")) == r"Dot"
    assert latex(symbols("Bar")) == r"Bar"
    assert latex(symbols("Vec")) == r"Vec"
    assert latex(symbols("Abs")) == r"Abs"
    assert latex(symbols("Mag")) == r"Mag"
    assert latex(symbols("PrM")) == r"PrM"
    assert latex(symbols("BM")) == r"BM"
    assert latex(symbols("hbar")) == r"\hbar"
    # Check a few combinations
    assert latex(symbols("xvecdot")) == r"\dot{\vec{x}}"
    assert latex(symbols("xDotVec")) == r"\vec{\dot{x}}"
    assert latex(symbols("xHATNorm")) == r"\left\|{\hat{x}}\right\|"
    # Check a couple big, ugly combinations
    assert latex(symbols('xMathringBm_yCheckPRM__zbreveAbs')) == \
        r"\boldsymbol{\mathring{x}}^{\left|{\breve{z}}\right|}_{{\check{y}}'}"
    assert latex(symbols('alphadothat_nVECDOT__tTildePrime')) == \
        r"\hat{\dot{\alpha}}^{{\tilde{t}}'}_{\dot{\vec{n}}}"


def test_greek_symbols():
    assert latex(Symbol('alpha'))   == r'\alpha'
    assert latex(Symbol('beta'))    == r'\beta'
    assert latex(Symbol('gamma'))   == r'\gamma'
    assert latex(Symbol('delta'))   == r'\delta'
    assert latex(Symbol('epsilon')) == r'\epsilon'
    assert latex(Symbol('zeta'))    == r'\zeta'
    assert latex(Symbol('eta'))     == r'\eta'
    assert latex(Symbol('theta'))   == r'\theta'
    assert latex(Symbol('iota'))    == r'\iota'
    assert latex(Symbol('kappa'))   == r'\kappa'
    assert latex(Symbol('lambda'))  == r'\lambda'
    assert latex(Symbol('mu'))      == r'\mu'
    assert latex(Symbol('nu'))      == r'\nu'
    assert latex(Symbol('xi'))      == r'\xi'
    assert latex(Symbol('omicron')) == r'o'
    assert latex(Symbol('pi'))      == r'\pi'
    assert latex(Symbol('rho'))     == r'\rho'
    assert latex(Symbol('sigma'))   == r'\sigma'
    assert latex(Symbol('tau'))     == r'\tau'
    assert latex(Symbol('upsilon')) == r'\upsilon'
    assert latex(Symbol('phi'))     == r'\phi'
    assert latex(Symbol('chi'))     == r'\chi'
    assert latex(Symbol('psi'))     == r'\psi'
    assert latex(Symbol('omega'))   == r'\omega'

    assert latex(Symbol('Alpha'))   == r'\mathrm{A}'
    assert latex(Symbol('Beta'))    == r'\mathrm{B}'
    assert latex(Symbol('Gamma'))   == r'\Gamma'
    assert latex(Symbol('Delta'))   == r'\Delta'
    assert latex(Symbol('Epsilon')) == r'\mathrm{E}'
    assert latex(Symbol('Zeta'))    == r'\mathrm{Z}'
    assert latex(Symbol('Eta'))     == r'\mathrm{H}'
    assert latex(Symbol('Theta'))   == r'\Theta'
    assert latex(Symbol('Iota'))    == r'\mathrm{I}'
    assert latex(Symbol('Kappa'))   == r'\mathrm{K}'
    assert latex(Symbol('Lambda'))  == r'\Lambda'
    assert latex(Symbol('Mu'))      == r'\mathrm{M}'
    assert latex(Symbol('Nu'))      == r'\mathrm{N}'
    assert latex(Symbol('Xi'))      == r'\Xi'
    assert latex(Symbol('Omicron')) == r'\mathrm{O}'
    assert latex(Symbol('Pi'))      == r'\Pi'
    assert latex(Symbol('Rho'))     == r'\mathrm{P}'
    assert latex(Symbol('Sigma'))   == r'\Sigma'
    assert latex(Symbol('Tau'))     == r'\mathrm{T}'
    assert latex(Symbol('Upsilon')) == r'\Upsilon'
    assert latex(Symbol('Phi'))     == r'\Phi'
    assert latex(Symbol('Chi'))     == r'\mathrm{X}'
    assert latex(Symbol('Psi'))     == r'\Psi'
    assert latex(Symbol('Omega'))   == r'\Omega'

    assert latex(Symbol('varepsilon')) == r'\varepsilon'
    assert latex(Symbol('varkappa')) == r'\varkappa'
    assert latex(Symbol('varphi')) == r'\varphi'
    assert latex(Symbol('varpi')) == r'\varpi'
    assert latex(Symbol('varrho')) == r'\varrho'
    assert latex(Symbol('varsigma')) == r'\varsigma'
    assert latex(Symbol('vartheta')) == r'\vartheta'


def test_fancyset_symbols():
    assert latex(S.Rationals) == r'\mathbb{Q}'
    assert latex(S.Naturals) == r'\mathbb{N}'
    assert latex(S.Naturals0) == r'\mathbb{N}_0'
    assert latex(S.Integers) == r'\mathbb{Z}'
    assert latex(S.Reals) == r'\mathbb{R}'
    assert latex(S.Complexes) == r'\mathbb{C}'


@XFAIL
def test_builtin_without_args_mismatched_names():
    assert latex(CosineTransform) == r'\mathcal{COS}'


def test_builtin_no_args():
    assert latex(Chi) == r'\operatorname{Chi}'
    assert latex(beta) == r'\operatorname{B}'
    assert latex(gamma) == r'\Gamma'
    assert latex(KroneckerDelta) == r'\delta'
    assert latex(DiracDelta) == r'\delta'
    assert latex(lowergamma) == r'\gamma'


def test_issue_6853():
    p = Function('Pi')
    assert latex(p(x)) == r"\Pi{\left(x \right)}"


def test_Mul():
    e = Mul(-2, x + 1, evaluate=False)
    assert latex(e) == r'- 2 \left(x + 1\right)'
    e = Mul(2, x + 1, evaluate=False)
    assert latex(e) == r'2 \left(x + 1\right)'
    e = Mul(S.Half, x + 1, evaluate=False)
    assert latex(e) == r'\frac{x + 1}{2}'
    e = Mul(y, x + 1, evaluate=False)
    assert latex(e) == r'y \left(x + 1\right)'
    e = Mul(-y, x + 1, evaluate=False)
    assert latex(e) == r'- y \left(x + 1\right)'
    e = Mul(-2, x + 1)
    assert latex(e) == r'- 2 x - 2'
    e = Mul(2, x + 1)
    assert latex(e) == r'2 x + 2'


def test_Pow():
    e = Pow(2, 2, evaluate=False)
    assert latex(e) == r'2^{2}'
    assert latex(x**(Rational(-1, 3))) == r'\frac{1}{\sqrt[3]{x}}'
    x2 = Symbol(r'x^2')
    assert latex(x2**2) == r'\left(x^{2}\right)^{2}'
    # Issue 11011
    assert latex(S('1.453e4500')**x) == r'{1.453 \cdot 10^{4500}}^{x}'


def test_issue_7180():
    assert latex(Equivalent(x, y)) == r"x \Leftrightarrow y"
    assert latex(Not(Equivalent(x, y))) == r"x \not\Leftrightarrow y"


def test_issue_8409():
    assert latex(S.Half**n) == r"\left(\frac{1}{2}\right)^{n}"


def test_issue_8470():
    from sympy.parsing.sympy_parser import parse_expr
    e = parse_expr("-B*A", evaluate=False)
    assert latex(e) == r"A \left(- B\right)"


def test_issue_15439():
    x = MatrixSymbol('x', 2, 2)
    y = MatrixSymbol('y', 2, 2)
    assert latex((x * y).subs(y, -y)) == r"x \left(- y\right)"
    assert latex((x * y).subs(y, -2*y)) == r"x \left(- 2 y\right)"
    assert latex((x * y).subs(x, -x)) == r"\left(- x\right) y"


def test_issue_2934():
    assert latex(Symbol(r'\frac{a_1}{b_1}')) == r'\frac{a_1}{b_1}'


def test_issue_10489():
    latexSymbolWithBrace = r'C_{x_{0}}'
    s = Symbol(latexSymbolWithBrace)
    assert latex(s) == latexSymbolWithBrace
    assert latex(cos(s)) == r'\cos{\left(C_{x_{0}} \right)}'


def test_issue_12886():
    m__1, l__1 = symbols('m__1, l__1')
    assert latex(m__1**2 + l__1**2) == \
        r'\left(l^{1}\right)^{2} + \left(m^{1}\right)^{2}'


def test_issue_13559():
    from sympy.parsing.sympy_parser import parse_expr
    expr = parse_expr('5/1', evaluate=False)
    assert latex(expr) == r"\frac{5}{1}"


def test_issue_13651():
    expr = c + Mul(-1, a + b, evaluate=False)
    assert latex(expr) == r"c - \left(a + b\right)"


def test_latex_UnevaluatedExpr():
    x = symbols("x")
    he = UnevaluatedExpr(1/x)
    assert latex(he) == latex(1/x) == r"\frac{1}{x}"
    assert latex(he**2) == r"\left(\frac{1}{x}\right)^{2}"
    assert latex(he + 1) == r"1 + \frac{1}{x}"
    assert latex(x*he) == r"x \frac{1}{x}"


def test_MatrixElement_printing():
    # test cases for issue #11821
    A = MatrixSymbol("A", 1, 3)
    B = MatrixSymbol("B", 1, 3)
    C = MatrixSymbol("C", 1, 3)

    assert latex(A[0, 0]) == r"{A}_{0,0}"
    assert latex(3 * A[0, 0]) == r"3 {A}_{0,0}"

    F = C[0, 0].subs(C, A - B)
    assert latex(F) == r"{\left(A - B\right)}_{0,0}"

    i, j, k = symbols("i j k")
    M = MatrixSymbol("M", k, k)
    N = MatrixSymbol("N", k, k)
    assert latex((M*N)[i, j]) == \
        r'\sum_{i_{1}=0}^{k - 1} {M}_{i,i_{1}} {N}_{i_{1},j}'

    X_a = MatrixSymbol('X_a', 3, 3)
    assert latex(X_a[0, 0]) == r"{X_{a}}_{0,0}"


def test_MatrixSymbol_printing():
    # test cases for issue #14237
    A = MatrixSymbol("A", 3, 3)
    B = MatrixSymbol("B", 3, 3)
    C = MatrixSymbol("C", 3, 3)

    assert latex(-A) == r"- A"
    assert latex(A - A*B - B) == r"A - A B - B"
    assert latex(-A*B - A*B*C - B) == r"- A B - A B C - B"


def test_DotProduct_printing():
    X = MatrixSymbol('X', 3, 1)
    Y = MatrixSymbol('Y', 3, 1)
    a = Symbol('a')
    assert latex(DotProduct(X, Y)) == r"X \cdot Y"
    assert latex(DotProduct(a * X, Y)) == r"a X \cdot Y"
    assert latex(a * DotProduct(X, Y)) == r"a \left(X \cdot Y\right)"


def test_KroneckerProduct_printing():
    A = MatrixSymbol('A', 3, 3)
    B = MatrixSymbol('B', 2, 2)
    assert latex(KroneckerProduct(A, B)) == r'A \otimes B'


def test_Series_printing():
    tf1 = TransferFunction(x*y**2 - z, y**3 - t**3, y)
    tf2 = TransferFunction(x - y, x + y, y)
    tf3 = TransferFunction(t*x**2 - t**w*x + w, t - y, y)
    assert latex(Series(tf1, tf2)) == \
        r'\left(\frac{x y^{2} - z}{- t^{3} + y^{3}}\right) \left(\frac{x - y}{x + y}\right)'
    assert latex(Series(tf1, tf2, tf3)) == \
        r'\left(\frac{x y^{2} - z}{- t^{3} + y^{3}}\right) \left(\frac{x - y}{x + y}\right) \left(\frac{t x^{2} - t^{w} x + w}{t - y}\right)'
    assert latex(Series(-tf2, tf1)) == \
        r'\left(\frac{- x + y}{x + y}\right) \left(\frac{x y^{2} - z}{- t^{3} + y^{3}}\right)'

    M_1 = Matrix([[5/s], [5/(2*s)]])
    T_1 = TransferFunctionMatrix.from_Matrix(M_1, s)
    M_2 = Matrix([[5, 6*s**3]])
    T_2 = TransferFunctionMatrix.from_Matrix(M_2, s)
    # Brackets
    assert latex(T_1*(T_2 + T_2)) == \
        r'\left[\begin{matrix}\frac{5}{s}\\\frac{5}{2 s}\end{matrix}\right]_\tau\cdot\left(\left[\begin{matrix}\frac{5}{1} &' \
        r' \frac{6 s^{3}}{1}\end{matrix}\right]_\tau + \left[\begin{matrix}\frac{5}{1} & \frac{6 s^{3}}{1}\end{matrix}\right]_\tau\right)' \
            == latex(MIMOSeries(MIMOParallel(T_2, T_2), T_1))
    # No Brackets
    M_3 = Matrix([[5, 6], [6, 5/s]])
    T_3 = TransferFunctionMatrix.from_Matrix(M_3, s)
    assert latex(T_1*T_2 + T_3) == r'\left[\begin{matrix}\frac{5}{s}\\\frac{5}{2 s}\end{matrix}\right]_\tau\cdot\left[\begin{matrix}' \
        r'\frac{5}{1} & \frac{6 s^{3}}{1}\end{matrix}\right]_\tau + \left[\begin{matrix}\frac{5}{1} & \frac{6}{1}\\\frac{6}{1} & ' \
            r'\frac{5}{s}\end{matrix}\right]_\tau' == latex(MIMOParallel(MIMOSeries(T_2, T_1), T_3))


def test_TransferFunction_printing():
    tf1 = TransferFunction(x - 1, x + 1, x)
    assert latex(tf1) == r"\frac{x - 1}{x + 1}"
    tf2 = TransferFunction(x + 1, 2 - y, x)
    assert latex(tf2) == r"\frac{x + 1}{2 - y}"
    tf3 = TransferFunction(y, y**2 + 2*y + 3, y)
    assert latex(tf3) == r"\frac{y}{y^{2} + 2 y + 3}"


def test_Parallel_printing():
    tf1 = TransferFunction(x*y**2 - z, y**3 - t**3, y)
    tf2 = TransferFunction(x - y, x + y, y)
    assert latex(Parallel(tf1, tf2)) == \
        r'\frac{x y^{2} - z}{- t^{3} + y^{3}} + \frac{x - y}{x + y}'
    assert latex(Parallel(-tf2, tf1)) == \
        r'\frac{- x + y}{x + y} + \frac{x y^{2} - z}{- t^{3} + y^{3}}'

    M_1 = Matrix([[5, 6], [6, 5/s]])
    T_1 = TransferFunctionMatrix.from_Matrix(M_1, s)
    M_2 = Matrix([[5/s, 6], [6, 5/(s - 1)]])
    T_2 = TransferFunctionMatrix.from_Matrix(M_2, s)
    M_3 = Matrix([[6, 5/(s*(s - 1))], [5, 6]])
    T_3 = TransferFunctionMatrix.from_Matrix(M_3, s)
    assert latex(T_1 + T_2 + T_3) == r'\left[\begin{matrix}\frac{5}{1} & \frac{6}{1}\\\frac{6}{1} & \frac{5}{s}\end{matrix}\right]' \
        r'_\tau + \left[\begin{matrix}\frac{5}{s} & \frac{6}{1}\\\frac{6}{1} & \frac{5}{s - 1}\end{matrix}\right]_\tau + \left[\begin{matrix}' \
            r'\frac{6}{1} & \frac{5}{s \left(s - 1\right)}\\\frac{5}{1} & \frac{6}{1}\end{matrix}\right]_\tau' \
                == latex(MIMOParallel(T_1, T_2, T_3)) == latex(MIMOParallel(T_1, MIMOParallel(T_2, T_3))) == latex(MIMOParallel(MIMOParallel(T_1, T_2), T_3))


def test_TransferFunctionMatrix_printing():
    tf1 = TransferFunction(p, p + x, p)
    tf2 = TransferFunction(-s + p, p + s, p)
    tf3 = TransferFunction(p, y**2 + 2*y + 3, p)
    assert latex(TransferFunctionMatrix([[tf1], [tf2]])) == \
        r'\left[\begin{matrix}\frac{p}{p + x}\\\frac{p - s}{p + s}\end{matrix}\right]_\tau'
    assert latex(TransferFunctionMatrix([[tf1, tf2], [tf3, -tf1]])) == \
        r'\left[\begin{matrix}\frac{p}{p + x} & \frac{p - s}{p + s}\\\frac{p}{y^{2} + 2 y + 3} & \frac{\left(-1\right) p}{p + x}\end{matrix}\right]_\tau'


def test_Feedback_printing():
    tf1 = TransferFunction(p, p + x, p)
    tf2 = TransferFunction(-s + p, p + s, p)
    # Negative Feedback (Default)
    assert latex(Feedback(tf1, tf2)) == \
        r'\frac{\frac{p}{p + x}}{\frac{1}{1} + \left(\frac{p}{p + x}\right) \left(\frac{p - s}{p + s}\right)}'
    assert latex(Feedback(tf1*tf2, TransferFunction(1, 1, p))) == \
        r'\frac{\left(\frac{p}{p + x}\right) \left(\frac{p - s}{p + s}\right)}{\frac{1}{1} + \left(\frac{p}{p + x}\right) \left(\frac{p - s}{p + s}\right)}'
    # Positive Feedback
    assert latex(Feedback(tf1, tf2, 1)) == \
        r'\frac{\frac{p}{p + x}}{\frac{1}{1} - \left(\frac{p}{p + x}\right) \left(\frac{p - s}{p + s}\right)}'
    assert latex(Feedback(tf1*tf2, sign=1)) == \
        r'\frac{\left(\frac{p}{p + x}\right) \left(\frac{p - s}{p + s}\right)}{\frac{1}{1} - \left(\frac{p}{p + x}\right) \left(\frac{p - s}{p + s}\right)}'


def test_MIMOFeedback_printing():
    tf1 = TransferFunction(1, s, s)
    tf2 = TransferFunction(s, s**2 - 1, s)
    tf3 = TransferFunction(s, s - 1, s)
    tf4 = TransferFunction(s**2, s**2 - 1, s)

    tfm_1 = TransferFunctionMatrix([[tf1, tf2], [tf3, tf4]])
    tfm_2 = TransferFunctionMatrix([[tf4, tf3], [tf2, tf1]])

    # Negative Feedback (Default)
    assert latex(MIMOFeedback(tfm_1, tfm_2)) == \
        r'\left(I_{\tau} + \left[\begin{matrix}\frac{1}{s} & \frac{s}{s^{2} - 1}\\\frac{s}{s - 1} & \frac{s^{2}}{s^{2} - 1}\end{matrix}\right]_\tau\cdot\left[' \
        r'\begin{matrix}\frac{s^{2}}{s^{2} - 1} & \frac{s}{s - 1}\\\frac{s}{s^{2} - 1} & \frac{1}{s}\end{matrix}\right]_\tau\right)^{-1} \cdot \left[\begin{matrix}' \
        r'\frac{1}{s} & \frac{s}{s^{2} - 1}\\\frac{s}{s - 1} & \frac{s^{2}}{s^{2} - 1}\end{matrix}\right]_\tau'

    # Positive Feedback
    assert latex(MIMOFeedback(tfm_1*tfm_2, tfm_1, 1)) == \
        r'\left(I_{\tau} - \left[\begin{matrix}\frac{1}{s} & \frac{s}{s^{2} - 1}\\\frac{s}{s - 1} & \frac{s^{2}}{s^{2} - 1}\end{matrix}\right]_\tau\cdot\left' \
        r'[\begin{matrix}\frac{s^{2}}{s^{2} - 1} & \frac{s}{s - 1}\\\frac{s}{s^{2} - 1} & \frac{1}{s}\end{matrix}\right]_\tau\cdot\left[\begin{matrix}\frac{1}{s} & \frac{s}{s^{2} - 1}' \
        r'\\\frac{s}{s - 1} & \frac{s^{2}}{s^{2} - 1}\end{matrix}\right]_\tau\right)^{-1} \cdot \left[\begin{matrix}\frac{1}{s} & \frac{s}{s^{2} - 1}' \
        r'\\\frac{s}{s - 1} & \frac{s^{2}}{s^{2} - 1}\end{matrix}\right]_\tau\cdot\left[\begin{matrix}\frac{s^{2}}{s^{2} - 1} & \frac{s}{s - 1}\\\frac{s}{s^{2} - 1}' \
        r' & \frac{1}{s}\end{matrix}\right]_\tau'


def test_Quaternion_latex_printing():
    q = Quaternion(x, y, z, t)
    assert latex(q) == r"x + y i + z j + t k"
    q = Quaternion(x, y, z, x*t)
    assert latex(q) == r"x + y i + z j + t x k"
    q = Quaternion(x, y, z, x + t)
    assert latex(q) == r"x + y i + z j + \left(t + x\right) k"


def test_TensorProduct_printing():
    from sympy.tensor.functions import TensorProduct
    A = MatrixSymbol("A", 3, 3)
    B = MatrixSymbol("B", 3, 3)
    assert latex(TensorProduct(A, B)) == r"A \otimes B"


def test_WedgeProduct_printing():
    from sympy.diffgeom.rn import R2
    from sympy.diffgeom import WedgeProduct
    wp = WedgeProduct(R2.dx, R2.dy)
    assert latex(wp) == r"\operatorname{d}x \wedge \operatorname{d}y"


def test_issue_9216():
    expr_1 = Pow(1, -1, evaluate=False)
    assert latex(expr_1) == r"1^{-1}"

    expr_2 = Pow(1, Pow(1, -1, evaluate=False), evaluate=False)
    assert latex(expr_2) == r"1^{1^{-1}}"

    expr_3 = Pow(3, -2, evaluate=False)
    assert latex(expr_3) == r"\frac{1}{9}"

    expr_4 = Pow(1, -2, evaluate=False)
    assert latex(expr_4) == r"1^{-2}"


def test_latex_printer_tensor():
    from sympy.tensor.tensor import TensorIndexType, tensor_indices, TensorHead, tensor_heads
    L = TensorIndexType("L")
    i, j, k, l = tensor_indices("i j k l", L)
    i0 = tensor_indices("i_0", L)
    A, B, C, D = tensor_heads("A B C D", [L])
    H = TensorHead("H", [L, L])
    K = TensorHead("K", [L, L, L, L])

    assert latex(i) == r"{}^{i}"
    assert latex(-i) == r"{}_{i}"

    expr = A(i)
    assert latex(expr) == r"A{}^{i}"

    expr = A(i0)
    assert latex(expr) == r"A{}^{i_{0}}"

    expr = A(-i)
    assert latex(expr) == r"A{}_{i}"

    expr = -3*A(i)
    assert latex(expr) == r"-3A{}^{i}"

    expr = K(i, j, -k, -i0)
    assert latex(expr) == r"K{}^{ij}{}_{ki_{0}}"

    expr = K(i, -j, -k, i0)
    assert latex(expr) == r"K{}^{i}{}_{jk}{}^{i_{0}}"

    expr = K(i, -j, k, -i0)
    assert latex(expr) == r"K{}^{i}{}_{j}{}^{k}{}_{i_{0}}"

    expr = H(i, -j)
    assert latex(expr) == r"H{}^{i}{}_{j}"

    expr = H(i, j)
    assert latex(expr) == r"H{}^{ij}"

    expr = H(-i, -j)
    assert latex(expr) == r"H{}_{ij}"

    expr = (1+x)*A(i)
    assert latex(expr) == r"\left(x + 1\right)A{}^{i}"

    expr = H(i, -i)
    assert latex(expr) == r"H{}^{L_{0}}{}_{L_{0}}"

    expr = H(i, -j)*A(j)*B(k)
    assert latex(expr) == r"H{}^{i}{}_{L_{0}}A{}^{L_{0}}B{}^{k}"

    expr = A(i) + 3*B(i)
    assert latex(expr) == r"3B{}^{i} + A{}^{i}"

    # Test ``TensorElement``:
    from sympy.tensor.tensor import TensorElement

    expr = TensorElement(K(i, j, k, l), {i: 3, k: 2})
    assert latex(expr) == r'K{}^{i=3,j,k=2,l}'

    expr = TensorElement(K(i, j, k, l), {i: 3})
    assert latex(expr) == r'K{}^{i=3,jkl}'

    expr = TensorElement(K(i, -j, k, l), {i: 3, k: 2})
    assert latex(expr) == r'K{}^{i=3}{}_{j}{}^{k=2,l}'

    expr = TensorElement(K(i, -j, k, -l), {i: 3, k: 2})
    assert latex(expr) == r'K{}^{i=3}{}_{j}{}^{k=2}{}_{l}'

    expr = TensorElement(K(i, j, -k, -l), {i: 3, -k: 2})
    assert latex(expr) == r'K{}^{i=3,j}{}_{k=2,l}'

    expr = TensorElement(K(i, j, -k, -l), {i: 3})
    assert latex(expr) == r'K{}^{i=3,j}{}_{kl}'

    expr = PartialDerivative(A(i), A(i))
    assert latex(expr) == r"\frac{\partial}{\partial {A{}^{L_{0}}}}{A{}^{L_{0}}}"

    expr = PartialDerivative(A(-i), A(-j))
    assert latex(expr) == r"\frac{\partial}{\partial {A{}_{j}}}{A{}_{i}}"

    expr = PartialDerivative(K(i, j, -k, -l), A(m), A(-n))
    assert latex(expr) == r"\frac{\partial^{2}}{\partial {A{}^{m}} \partial {A{}_{n}}}{K{}^{ij}{}_{kl}}"

    expr = PartialDerivative(B(-i) + A(-i), A(-j), A(-n))
    assert latex(expr) == r"\frac{\partial^{2}}{\partial {A{}_{j}} \partial {A{}_{n}}}{\left(A{}_{i} + B{}_{i}\right)}"

    expr = PartialDerivative(3*A(-i), A(-j), A(-n))
    assert latex(expr) == r"\frac{\partial^{2}}{\partial {A{}_{j}} \partial {A{}_{n}}}{\left(3A{}_{i}\right)}"


def test_multiline_latex():
    a, b, c, d, e, f = symbols('a b c d e f')
    expr = -a + 2*b -3*c +4*d -5*e
    expected = r"\begin{eqnarray}" + "\n"\
        r"f & = &- a \nonumber\\" + "\n"\
        r"& & + 2 b \nonumber\\" + "\n"\
        r"& & - 3 c \nonumber\\" + "\n"\
        r"& & + 4 d \nonumber\\" + "\n"\
        r"& & - 5 e " + "\n"\
        r"\end{eqnarray}"
    assert multiline_latex(f, expr, environment="eqnarray") == expected

    expected2 = r'\begin{eqnarray}' + '\n'\
        r'f & = &- a + 2 b \nonumber\\' + '\n'\
        r'& & - 3 c + 4 d \nonumber\\' + '\n'\
        r'& & - 5 e ' + '\n'\
        r'\end{eqnarray}'

    assert multiline_latex(f, expr, 2, environment="eqnarray") == expected2

    expected3 = r'\begin{eqnarray}' + '\n'\
        r'f & = &- a + 2 b - 3 c \nonumber\\'+ '\n'\
        r'& & + 4 d - 5 e ' + '\n'\
        r'\end{eqnarray}'

    assert multiline_latex(f, expr, 3, environment="eqnarray") == expected3

    expected3dots = r'\begin{eqnarray}' + '\n'\
        r'f & = &- a + 2 b - 3 c \dots\nonumber\\'+ '\n'\
        r'& & + 4 d - 5 e ' + '\n'\
        r'\end{eqnarray}'

    assert multiline_latex(f, expr, 3, environment="eqnarray", use_dots=True) == expected3dots

    expected3align = r'\begin{align*}' + '\n'\
        r'f = &- a + 2 b - 3 c \\'+ '\n'\
        r'& + 4 d - 5 e ' + '\n'\
        r'\end{align*}'

    assert multiline_latex(f, expr, 3) == expected3align
    assert multiline_latex(f, expr, 3, environment='align*') == expected3align

    expected2ieee = r'\begin{IEEEeqnarray}{rCl}' + '\n'\
        r'f & = &- a + 2 b \nonumber\\' + '\n'\
        r'& & - 3 c + 4 d \nonumber\\' + '\n'\
        r'& & - 5 e ' + '\n'\
        r'\end{IEEEeqnarray}'

    assert multiline_latex(f, expr, 2, environment="IEEEeqnarray") == expected2ieee

    raises(ValueError, lambda: multiline_latex(f, expr, environment="foo"))

def test_issue_15353():
    a, x = symbols('a x')
    # Obtained from nonlinsolve([(sin(a*x)),cos(a*x)],[x,a])
    sol = ConditionSet(
        Tuple(x, a), Eq(sin(a*x), 0) & Eq(cos(a*x), 0), S.Complexes**2)
    assert latex(sol) == \
        r'\left\{\left( x, \  a\right)\; \middle|\; \left( x, \  a\right) \in ' \
        r'\mathbb{C}^{2} \wedge \sin{\left(a x \right)} = 0 \wedge ' \
        r'\cos{\left(a x \right)} = 0 \right\}'


def test_latex_symbolic_probability():
    mu = symbols("mu")
    sigma = symbols("sigma", positive=True)
    X = Normal("X", mu, sigma)
    assert latex(Expectation(X)) == r'\operatorname{E}\left[X\right]'
    assert latex(Variance(X)) == r'\operatorname{Var}\left(X\right)'
    assert latex(Probability(X > 0)) == r'\operatorname{P}\left(X > 0\right)'
    Y = Normal("Y", mu, sigma)
    assert latex(Covariance(X, Y)) == r'\operatorname{Cov}\left(X, Y\right)'


def test_trace():
    # Issue 15303
    from sympy.matrices.expressions.trace import trace
    A = MatrixSymbol("A", 2, 2)
    assert latex(trace(A)) == r"\operatorname{tr}\left(A \right)"
    assert latex(trace(A**2)) == r"\operatorname{tr}\left(A^{2} \right)"


def test_print_basic():
    # Issue 15303
    from sympy.core.basic import Basic
    from sympy.core.expr import Expr

    # dummy class for testing printing where the function is not
    # implemented in latex.py
    class UnimplementedExpr(Expr):
        def __new__(cls, e):
            return Basic.__new__(cls, e)

    # dummy function for testing
    def unimplemented_expr(expr):
        return UnimplementedExpr(expr).doit()

    # override class name to use superscript / subscript
    def unimplemented_expr_sup_sub(expr):
        result = UnimplementedExpr(expr)
        result.__class__.__name__ = 'UnimplementedExpr_x^1'
        return result

    assert latex(unimplemented_expr(x)) == r'\operatorname{UnimplementedExpr}\left(x\right)'
    assert latex(unimplemented_expr(x**2)) == \
        r'\operatorname{UnimplementedExpr}\left(x^{2}\right)'
    assert latex(unimplemented_expr_sup_sub(x)) == \
        r'\operatorname{UnimplementedExpr^{1}_{x}}\left(x\right)'


def test_MatrixSymbol_bold():
    # Issue #15871
    from sympy.matrices.expressions.trace import trace
    A = MatrixSymbol("A", 2, 2)
    assert latex(trace(A), mat_symbol_style='bold') == \
        r"\operatorname{tr}\left(\mathbf{A} \right)"
    assert latex(trace(A), mat_symbol_style='plain') == \
        r"\operatorname{tr}\left(A \right)"

    A = MatrixSymbol("A", 3, 3)
    B = MatrixSymbol("B", 3, 3)
    C = MatrixSymbol("C", 3, 3)

    assert latex(-A, mat_symbol_style='bold') == r"- \mathbf{A}"
    assert latex(A - A*B - B, mat_symbol_style='bold') == \
        r"\mathbf{A} - \mathbf{A} \mathbf{B} - \mathbf{B}"
    assert latex(-A*B - A*B*C - B, mat_symbol_style='bold') == \
        r"- \mathbf{A} \mathbf{B} - \mathbf{A} \mathbf{B} \mathbf{C} - \mathbf{B}"

    A_k = MatrixSymbol("A_k", 3, 3)
    assert latex(A_k, mat_symbol_style='bold') == r"\mathbf{A}_{k}"

    A = MatrixSymbol(r"\nabla_k", 3, 3)
    assert latex(A, mat_symbol_style='bold') == r"\mathbf{\nabla}_{k}"

def test_AppliedPermutation():
    p = Permutation(0, 1, 2)
    x = Symbol('x')
    assert latex(AppliedPermutation(p, x)) == \
        r'\sigma_{\left( 0\; 1\; 2\right)}(x)'


def test_PermutationMatrix():
    p = Permutation(0, 1, 2)
    assert latex(PermutationMatrix(p)) == r'P_{\left( 0\; 1\; 2\right)}'
    p = Permutation(0, 3)(1, 2)
    assert latex(PermutationMatrix(p)) == \
        r'P_{\left( 0\; 3\right)\left( 1\; 2\right)}'


def test_issue_21758():
    from sympy.functions.elementary.piecewise import piecewise_fold
    from sympy.series.fourier import FourierSeries
    x = Symbol('x')
    k, n = symbols('k n')
    fo = FourierSeries(x, (x, -pi, pi), (0, SeqFormula(0, (k, 1, oo)), SeqFormula(
        Piecewise((-2*pi*cos(n*pi)/n + 2*sin(n*pi)/n**2, (n > -oo) & (n < oo) & Ne(n, 0)),
                  (0, True))*sin(n*x)/pi, (n, 1, oo))))
    assert latex(piecewise_fold(fo)) == '\\begin{cases} 2 \\sin{\\left(x \\right)}' \
            ' - \\sin{\\left(2 x \\right)} + \\frac{2 \\sin{\\left(3 x \\right)}}{3} +' \
            ' \\ldots & \\text{for}\\: n > -\\infty \\wedge n < \\infty \\wedge ' \
                'n \\neq 0 \\\\0 & \\text{otherwise} \\end{cases}'
    assert latex(FourierSeries(x, (x, -pi, pi), (0, SeqFormula(0, (k, 1, oo)),
                                                 SeqFormula(0, (n, 1, oo))))) == '0'


def test_imaginary_unit():
    assert latex(1 + I) == r'1 + i'
    assert latex(1 + I, imaginary_unit='i') == r'1 + i'
    assert latex(1 + I, imaginary_unit='j') == r'1 + j'
    assert latex(1 + I, imaginary_unit='foo') == r'1 + foo'
    assert latex(I, imaginary_unit="ti") == r'\text{i}'
    assert latex(I, imaginary_unit="tj") == r'\text{j}'


def test_text_re_im():
    assert latex(im(x), gothic_re_im=True) == r'\Im{\left(x\right)}'
    assert latex(im(x), gothic_re_im=False) == r'\operatorname{im}{\left(x\right)}'
    assert latex(re(x), gothic_re_im=True) == r'\Re{\left(x\right)}'
    assert latex(re(x), gothic_re_im=False) == r'\operatorname{re}{\left(x\right)}'


def test_latex_diffgeom():
    from sympy.diffgeom import Manifold, Patch, CoordSystem, BaseScalarField, Differential
    from sympy.diffgeom.rn import R2
    x,y = symbols('x y', real=True)
    m = Manifold('M', 2)
    assert latex(m) == r'\text{M}'
    p = Patch('P', m)
    assert latex(p) == r'\text{P}_{\text{M}}'
    rect = CoordSystem('rect', p, [x, y])
    assert latex(rect) == r'\text{rect}^{\text{P}}_{\text{M}}'
    b = BaseScalarField(rect, 0)
    assert latex(b) == r'\mathbf{x}'

    g = Function('g')
    s_field = g(R2.x, R2.y)
    assert latex(Differential(s_field)) == \
        r'\operatorname{d}\left(g{\left(\mathbf{x},\mathbf{y} \right)}\right)'


def test_unit_printing():
    assert latex(5*meter) == r'5 \text{m}'
    assert latex(3*gibibyte) == r'3 \text{gibibyte}'
    assert latex(4*microgram/second) == r'\frac{4 \mu\text{g}}{\text{s}}'
    assert latex(4*micro*gram/second) == r'\frac{4 \mu \text{g}}{\text{s}}'
    assert latex(5*milli*meter) == r'5 \text{m} \text{m}'
    assert latex(milli) == r'\text{m}'


def test_issue_17092():
    x_star = Symbol('x^*')
    assert latex(Derivative(x_star, x_star,2)) == r'\frac{d^{2}}{d \left(x^{*}\right)^{2}} x^{*}'


def test_latex_decimal_separator():

    x, y, z, t = symbols('x y z t')
    k, m, n = symbols('k m n', integer=True)
    f, g, h = symbols('f g h', cls=Function)

    # comma decimal_separator
    assert(latex([1, 2.3, 4.5], decimal_separator='comma') == r'\left[ 1; \  2{,}3; \  4{,}5\right]')
    assert(latex(FiniteSet(1, 2.3, 4.5), decimal_separator='comma') == r'\left\{1; 2{,}3; 4{,}5\right\}')
    assert(latex((1, 2.3, 4.6), decimal_separator = 'comma') == r'\left( 1; \  2{,}3; \  4{,}6\right)')
    assert(latex((1,), decimal_separator='comma') == r'\left( 1;\right)')

    # period decimal_separator
    assert(latex([1, 2.3, 4.5], decimal_separator='period') == r'\left[ 1, \  2.3, \  4.5\right]' )
    assert(latex(FiniteSet(1, 2.3, 4.5), decimal_separator='period') == r'\left\{1, 2.3, 4.5\right\}')
    assert(latex((1, 2.3, 4.6), decimal_separator = 'period') == r'\left( 1, \  2.3, \  4.6\right)')
    assert(latex((1,), decimal_separator='period') == r'\left( 1,\right)')

    # default decimal_separator
    assert(latex([1, 2.3, 4.5]) == r'\left[ 1, \  2.3, \  4.5\right]')
    assert(latex(FiniteSet(1, 2.3, 4.5)) == r'\left\{1, 2.3, 4.5\right\}')
    assert(latex((1, 2.3, 4.6)) == r'\left( 1, \  2.3, \  4.6\right)')
    assert(latex((1,)) == r'\left( 1,\right)')

    assert(latex(Mul(3.4,5.3), decimal_separator = 'comma') == r'18{,}02')
    assert(latex(3.4*5.3, decimal_separator = 'comma') == r'18{,}02')
    x = symbols('x')
    y = symbols('y')
    z = symbols('z')
    assert(latex(x*5.3 + 2**y**3.4 + 4.5 + z, decimal_separator = 'comma') == r'2^{y^{3{,}4}} + 5{,}3 x + z + 4{,}5')

    assert(latex(0.987, decimal_separator='comma') == r'0{,}987')
    assert(latex(S(0.987), decimal_separator='comma') == r'0{,}987')
    assert(latex(.3, decimal_separator='comma') == r'0{,}3')
    assert(latex(S(.3), decimal_separator='comma') == r'0{,}3')


    assert(latex(5.8*10**(-7), decimal_separator='comma') == r'5{,}8 \cdot 10^{-7}')
    assert(latex(S(5.7)*10**(-7), decimal_separator='comma') == r'5{,}7 \cdot 10^{-7}')
    assert(latex(S(5.7*10**(-7)), decimal_separator='comma') == r'5{,}7 \cdot 10^{-7}')

    x = symbols('x')
    assert(latex(1.2*x+3.4, decimal_separator='comma') == r'1{,}2 x + 3{,}4')
    assert(latex(FiniteSet(1, 2.3, 4.5), decimal_separator='period') == r'\left\{1, 2.3, 4.5\right\}')

    # Error Handling tests
    raises(ValueError, lambda: latex([1,2.3,4.5], decimal_separator='non_existing_decimal_separator_in_list'))
    raises(ValueError, lambda: latex(FiniteSet(1,2.3,4.5), decimal_separator='non_existing_decimal_separator_in_set'))
    raises(ValueError, lambda: latex((1,2.3,4.5), decimal_separator='non_existing_decimal_separator_in_tuple'))

def test_Str():
    from sympy.core.symbol import Str
    assert str(Str('x')) == r'x'

def test_latex_escape():
    assert latex_escape(r"~^\&%$#_{}") == "".join([
        r'\textasciitilde',
        r'\textasciicircum',
        r'\textbackslash',
        r'\&',
        r'\%',
        r'\$',
        r'\#',
        r'\_',
        r'\{',
        r'\}',
    ])

def test_emptyPrinter():
    class MyObject:
        def __repr__(self):
            return "<MyObject with {...}>"

    # unknown objects are monospaced
    assert latex(MyObject()) == r"\mathtt{\text{<MyObject with \{...\}>}}"

    # even if they are nested within other objects
    assert latex((MyObject(),)) == r"\left( \mathtt{\text{<MyObject with \{...\}>}},\right)"

def test_global_settings():
    import inspect

    # settings should be visible in the signature of `latex`
    assert inspect.signature(latex).parameters['imaginary_unit'].default == r'i'
    assert latex(I) == r'i'
    try:
        # but changing the defaults...
        LatexPrinter.set_global_settings(imaginary_unit='j')
        # ... should change the signature
        assert inspect.signature(latex).parameters['imaginary_unit'].default == r'j'
        assert latex(I) == r'j'
    finally:
        # there's no public API to undo this, but we need to make sure we do
        # so as not to impact other tests
        del LatexPrinter._global_settings['imaginary_unit']

    # check we really did undo it
    assert inspect.signature(latex).parameters['imaginary_unit'].default == r'i'
    assert latex(I) == r'i'

def test_pickleable():
    # this tests that the _PrintFunction instance is pickleable
    import pickle
    assert pickle.loads(pickle.dumps(latex)) is latex

def test_printing_latex_array_expressions():
    assert latex(ArraySymbol("A", (2, 3, 4))) == "A"
    assert latex(ArrayElement("A", (2, 1/(1-x), 0))) == "{{A}_{2, \\frac{1}{1 - x}, 0}}"
    M = MatrixSymbol("M", 3, 3)
    N = MatrixSymbol("N", 3, 3)
    assert latex(ArrayElement(M*N, [x, 0])) == "{{\\left(M N\\right)}_{x, 0}}"

def test_Array():
    arr = Array(range(10))
    assert latex(arr) == r'\left[\begin{matrix}0 & 1 & 2 & 3 & 4 & 5 & 6 & 7 & 8 & 9\end{matrix}\right]'

    arr = Array(range(11))
    # fill the empty argument with a bunch of 'c' to avoid latex errors
    assert latex(arr) == r'\left[\begin{array}{ccccccccccc}0 & 1 & 2 & 3 & 4 & 5 & 6 & 7 & 8 & 9 & 10\end{array}\right]'

def test_latex_with_unevaluated():
    with evaluate(False):
        assert latex(a * a) == r"a a"


def test_latex_disable_split_super_sub():
    assert latex(Symbol('u^a_b')) == 'u^{a}_{b}'
    assert latex(Symbol('u^a_b'), disable_split_super_sub=False) == 'u^{a}_{b}'
    assert latex(Symbol('u^a_b'), disable_split_super_sub=True) == 'u\\^a\\_b'
