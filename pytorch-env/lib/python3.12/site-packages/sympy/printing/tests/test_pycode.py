from sympy.codegen import Assignment
from sympy.codegen.ast import none
from sympy.codegen.cfunctions import expm1, log1p
from sympy.codegen.scipy_nodes import cosm1
from sympy.codegen.matrix_nodes import MatrixSolve
from sympy.core import Expr, Mod, symbols, Eq, Le, Gt, zoo, oo, Rational, Pow
from sympy.core.numbers import pi
from sympy.core.singleton import S
from sympy.functions import acos, KroneckerDelta, Piecewise, sign, sqrt, Min, Max, cot, acsch, asec, coth, sec
from sympy.logic import And, Or
from sympy.matrices import SparseMatrix, MatrixSymbol, Identity
from sympy.printing.pycode import (
    MpmathPrinter, PythonCodePrinter, pycode, SymPyPrinter
)
from sympy.printing.tensorflow import TensorflowPrinter
from sympy.printing.numpy import NumPyPrinter, SciPyPrinter
from sympy.testing.pytest import raises, skip
from sympy.tensor import IndexedBase, Idx
from sympy.tensor.array.expressions.array_expressions import ArraySymbol, ArrayDiagonal, ArrayContraction, ZeroArray, OneArray
from sympy.external import import_module
from sympy.functions.special.gamma_functions import loggamma


x, y, z = symbols('x y z')
p = IndexedBase("p")


def test_PythonCodePrinter():
    prntr = PythonCodePrinter()

    assert not prntr.module_imports

    assert prntr.doprint(x**y) == 'x**y'
    assert prntr.doprint(Mod(x, 2)) == 'x % 2'
    assert prntr.doprint(-Mod(x, y)) == '-(x % y)'
    assert prntr.doprint(Mod(-x, y)) == '(-x) % y'
    assert prntr.doprint(And(x, y)) == 'x and y'
    assert prntr.doprint(Or(x, y)) == 'x or y'
    assert prntr.doprint(1/(x+y)) == '1/(x + y)'
    assert not prntr.module_imports

    assert prntr.doprint(pi) == 'math.pi'
    assert prntr.module_imports == {'math': {'pi'}}

    assert prntr.doprint(x**Rational(1, 2)) == 'math.sqrt(x)'
    assert prntr.doprint(sqrt(x)) == 'math.sqrt(x)'
    assert prntr.module_imports == {'math': {'pi', 'sqrt'}}

    assert prntr.doprint(acos(x)) == 'math.acos(x)'
    assert prntr.doprint(cot(x)) == '(1/math.tan(x))'
    assert prntr.doprint(coth(x)) == '((math.exp(x) + math.exp(-x))/(math.exp(x) - math.exp(-x)))'
    assert prntr.doprint(asec(x)) == '(math.acos(1/x))'
    assert prntr.doprint(acsch(x)) == '(math.log(math.sqrt(1 + x**(-2)) + 1/x))'

    assert prntr.doprint(Assignment(x, 2)) == 'x = 2'
    assert prntr.doprint(Piecewise((1, Eq(x, 0)),
                        (2, x>6))) == '((1) if (x == 0) else (2) if (x > 6) else None)'
    assert prntr.doprint(Piecewise((2, Le(x, 0)),
                        (3, Gt(x, 0)), evaluate=False)) == '((2) if (x <= 0) else'\
                                                        ' (3) if (x > 0) else None)'
    assert prntr.doprint(sign(x)) == '(0.0 if x == 0 else math.copysign(1, x))'
    assert prntr.doprint(p[0, 1]) == 'p[0, 1]'
    assert prntr.doprint(KroneckerDelta(x,y)) == '(1 if x == y else 0)'

    assert prntr.doprint((2,3)) == "(2, 3)"
    assert prntr.doprint([2,3]) == "[2, 3]"

    assert prntr.doprint(Min(x, y)) == "min(x, y)"
    assert prntr.doprint(Max(x, y)) == "max(x, y)"


def test_PythonCodePrinter_standard():
    prntr = PythonCodePrinter()

    assert prntr.standard == 'python3'

    raises(ValueError, lambda: PythonCodePrinter({'standard':'python4'}))


def test_MpmathPrinter():
    p = MpmathPrinter()
    assert p.doprint(sign(x)) == 'mpmath.sign(x)'
    assert p.doprint(Rational(1, 2)) == 'mpmath.mpf(1)/mpmath.mpf(2)'

    assert p.doprint(S.Exp1) == 'mpmath.e'
    assert p.doprint(S.Pi) == 'mpmath.pi'
    assert p.doprint(S.GoldenRatio) == 'mpmath.phi'
    assert p.doprint(S.EulerGamma) == 'mpmath.euler'
    assert p.doprint(S.NaN) == 'mpmath.nan'
    assert p.doprint(S.Infinity) == 'mpmath.inf'
    assert p.doprint(S.NegativeInfinity) == 'mpmath.ninf'
    assert p.doprint(loggamma(x)) == 'mpmath.loggamma(x)'


def test_NumPyPrinter():
    from sympy.core.function import Lambda
    from sympy.matrices.expressions.adjoint import Adjoint
    from sympy.matrices.expressions.diagonal import (DiagMatrix, DiagonalMatrix, DiagonalOf)
    from sympy.matrices.expressions.funcmatrix import FunctionMatrix
    from sympy.matrices.expressions.hadamard import HadamardProduct
    from sympy.matrices.expressions.kronecker import KroneckerProduct
    from sympy.matrices.expressions.special import (OneMatrix, ZeroMatrix)
    from sympy.abc import a, b
    p = NumPyPrinter()
    assert p.doprint(sign(x)) == 'numpy.sign(x)'
    A = MatrixSymbol("A", 2, 2)
    B = MatrixSymbol("B", 2, 2)
    C = MatrixSymbol("C", 1, 5)
    D = MatrixSymbol("D", 3, 4)
    assert p.doprint(A**(-1)) == "numpy.linalg.inv(A)"
    assert p.doprint(A**5) == "numpy.linalg.matrix_power(A, 5)"
    assert p.doprint(Identity(3)) == "numpy.eye(3)"

    u = MatrixSymbol('x', 2, 1)
    v = MatrixSymbol('y', 2, 1)
    assert p.doprint(MatrixSolve(A, u)) == 'numpy.linalg.solve(A, x)'
    assert p.doprint(MatrixSolve(A, u) + v) == 'numpy.linalg.solve(A, x) + y'

    assert p.doprint(ZeroMatrix(2, 3)) == "numpy.zeros((2, 3))"
    assert p.doprint(OneMatrix(2, 3)) == "numpy.ones((2, 3))"
    assert p.doprint(FunctionMatrix(4, 5, Lambda((a, b), a + b))) == \
        "numpy.fromfunction(lambda a, b: a + b, (4, 5))"
    assert p.doprint(HadamardProduct(A, B)) == "numpy.multiply(A, B)"
    assert p.doprint(KroneckerProduct(A, B)) == "numpy.kron(A, B)"
    assert p.doprint(Adjoint(A)) == "numpy.conjugate(numpy.transpose(A))"
    assert p.doprint(DiagonalOf(A)) == "numpy.reshape(numpy.diag(A), (-1, 1))"
    assert p.doprint(DiagMatrix(C)) == "numpy.diagflat(C)"
    assert p.doprint(DiagonalMatrix(D)) == "numpy.multiply(D, numpy.eye(3, 4))"

    # Workaround for numpy negative integer power errors
    assert p.doprint(x**-1) == 'x**(-1.0)'
    assert p.doprint(x**-2) == 'x**(-2.0)'

    expr = Pow(2, -1, evaluate=False)
    assert p.doprint(expr) == "2**(-1.0)"

    assert p.doprint(S.Exp1) == 'numpy.e'
    assert p.doprint(S.Pi) == 'numpy.pi'
    assert p.doprint(S.EulerGamma) == 'numpy.euler_gamma'
    assert p.doprint(S.NaN) == 'numpy.nan'
    assert p.doprint(S.Infinity) == 'numpy.inf'
    assert p.doprint(S.NegativeInfinity) == '-numpy.inf'

    # Function rewriting operator precedence fix
    assert p.doprint(sec(x)**2) == '(numpy.cos(x)**(-1.0))**2'


def test_issue_18770():
    numpy = import_module('numpy')
    if not numpy:
        skip("numpy not installed.")

    from sympy.functions.elementary.miscellaneous import (Max, Min)
    from sympy.utilities.lambdify import lambdify

    expr1 = Min(0.1*x + 3, x + 1, 0.5*x + 1)
    func = lambdify(x, expr1, "numpy")
    assert (func(numpy.linspace(0, 3, 3)) == [1.0, 1.75, 2.5 ]).all()
    assert  func(4) == 3

    expr1 = Max(x**2, x**3)
    func = lambdify(x,expr1, "numpy")
    assert (func(numpy.linspace(-1, 2, 4)) == [1, 0, 1, 8] ).all()
    assert func(4) == 64


def test_SciPyPrinter():
    p = SciPyPrinter()
    expr = acos(x)
    assert 'numpy' not in p.module_imports
    assert p.doprint(expr) == 'numpy.arccos(x)'
    assert 'numpy' in p.module_imports
    assert not any(m.startswith('scipy') for m in p.module_imports)
    smat = SparseMatrix(2, 5, {(0, 1): 3})
    assert p.doprint(smat) == \
        'scipy.sparse.coo_matrix(([3], ([0], [1])), shape=(2, 5))'
    assert 'scipy.sparse' in p.module_imports

    assert p.doprint(S.GoldenRatio) == 'scipy.constants.golden_ratio'
    assert p.doprint(S.Pi) == 'scipy.constants.pi'
    assert p.doprint(S.Exp1) == 'numpy.e'


def test_pycode_reserved_words():
    s1, s2 = symbols('if else')
    raises(ValueError, lambda: pycode(s1 + s2, error_on_reserved=True))
    py_str = pycode(s1 + s2)
    assert py_str in ('else_ + if_', 'if_ + else_')


def test_issue_20762():
    # Make sure pycode removes curly braces from subscripted variables
    a_b, b, a_11 = symbols('a_{b} b a_{11}')
    expr = a_b*b
    assert pycode(expr) == 'a_b*b'
    expr = a_11*b
    assert pycode(expr) == 'a_11*b'


def test_sqrt():
    prntr = PythonCodePrinter()
    assert prntr._print_Pow(sqrt(x), rational=False) == 'math.sqrt(x)'
    assert prntr._print_Pow(1/sqrt(x), rational=False) == '1/math.sqrt(x)'

    prntr = PythonCodePrinter({'standard' : 'python3'})
    assert prntr._print_Pow(sqrt(x), rational=True) == 'x**(1/2)'
    assert prntr._print_Pow(1/sqrt(x), rational=True) == 'x**(-1/2)'

    prntr = MpmathPrinter()
    assert prntr._print_Pow(sqrt(x), rational=False) == 'mpmath.sqrt(x)'
    assert prntr._print_Pow(sqrt(x), rational=True) == \
        "x**(mpmath.mpf(1)/mpmath.mpf(2))"

    prntr = NumPyPrinter()
    assert prntr._print_Pow(sqrt(x), rational=False) == 'numpy.sqrt(x)'
    assert prntr._print_Pow(sqrt(x), rational=True) == 'x**(1/2)'

    prntr = SciPyPrinter()
    assert prntr._print_Pow(sqrt(x), rational=False) == 'numpy.sqrt(x)'
    assert prntr._print_Pow(sqrt(x), rational=True) == 'x**(1/2)'

    prntr = SymPyPrinter()
    assert prntr._print_Pow(sqrt(x), rational=False) == 'sympy.sqrt(x)'
    assert prntr._print_Pow(sqrt(x), rational=True) == 'x**(1/2)'


def test_frac():
    from sympy.functions.elementary.integers import frac

    expr = frac(x)
    prntr = NumPyPrinter()
    assert prntr.doprint(expr) == 'numpy.mod(x, 1)'

    prntr = SciPyPrinter()
    assert prntr.doprint(expr) == 'numpy.mod(x, 1)'

    prntr = PythonCodePrinter()
    assert prntr.doprint(expr) == 'x % 1'

    prntr = MpmathPrinter()
    assert prntr.doprint(expr) == 'mpmath.frac(x)'

    prntr = SymPyPrinter()
    assert prntr.doprint(expr) == 'sympy.functions.elementary.integers.frac(x)'


class CustomPrintedObject(Expr):
    def _numpycode(self, printer):
        return 'numpy'

    def _mpmathcode(self, printer):
        return 'mpmath'


def test_printmethod():
    obj = CustomPrintedObject()
    assert NumPyPrinter().doprint(obj) == 'numpy'
    assert MpmathPrinter().doprint(obj) == 'mpmath'


def test_codegen_ast_nodes():
    assert pycode(none) == 'None'


def test_issue_14283():
    prntr = PythonCodePrinter()

    assert prntr.doprint(zoo) == "math.nan"
    assert prntr.doprint(-oo) == "float('-inf')"


def test_NumPyPrinter_print_seq():
    n = NumPyPrinter()

    assert n._print_seq(range(2)) == '(0, 1,)'


def test_issue_16535_16536():
    from sympy.functions.special.gamma_functions import (lowergamma, uppergamma)

    a = symbols('a')
    expr1 = lowergamma(a, x)
    expr2 = uppergamma(a, x)

    prntr = SciPyPrinter()
    assert prntr.doprint(expr1) == 'scipy.special.gamma(a)*scipy.special.gammainc(a, x)'
    assert prntr.doprint(expr2) == 'scipy.special.gamma(a)*scipy.special.gammaincc(a, x)'

    p_numpy = NumPyPrinter()
    p_pycode = PythonCodePrinter({'strict': False})

    for expr in [expr1, expr2]:
        with raises(NotImplementedError):
            p_numpy.doprint(expr1)
        assert "Not supported" in p_pycode.doprint(expr)


def test_Integral():
    from sympy.functions.elementary.exponential import exp
    from sympy.integrals.integrals import Integral

    single = Integral(exp(-x), (x, 0, oo))
    double = Integral(x**2*exp(x*y), (x, -z, z), (y, 0, z))
    indefinite = Integral(x**2, x)
    evaluateat = Integral(x**2, (x, 1))

    prntr = SciPyPrinter()
    assert prntr.doprint(single) == 'scipy.integrate.quad(lambda x: numpy.exp(-x), 0, numpy.inf)[0]'
    assert prntr.doprint(double) == 'scipy.integrate.nquad(lambda x, y: x**2*numpy.exp(x*y), ((-z, z), (0, z)))[0]'
    raises(NotImplementedError, lambda: prntr.doprint(indefinite))
    raises(NotImplementedError, lambda: prntr.doprint(evaluateat))

    prntr = MpmathPrinter()
    assert prntr.doprint(single) == 'mpmath.quad(lambda x: mpmath.exp(-x), (0, mpmath.inf))'
    assert prntr.doprint(double) == 'mpmath.quad(lambda x, y: x**2*mpmath.exp(x*y), (-z, z), (0, z))'
    raises(NotImplementedError, lambda: prntr.doprint(indefinite))
    raises(NotImplementedError, lambda: prntr.doprint(evaluateat))


def test_fresnel_integrals():
    from sympy.functions.special.error_functions import (fresnelc, fresnels)

    expr1 = fresnelc(x)
    expr2 = fresnels(x)

    prntr = SciPyPrinter()
    assert prntr.doprint(expr1) == 'scipy.special.fresnel(x)[1]'
    assert prntr.doprint(expr2) == 'scipy.special.fresnel(x)[0]'

    p_numpy = NumPyPrinter()
    p_pycode = PythonCodePrinter()
    p_mpmath = MpmathPrinter()
    for expr in [expr1, expr2]:
        with raises(NotImplementedError):
            p_numpy.doprint(expr)
        with raises(NotImplementedError):
            p_pycode.doprint(expr)

    assert p_mpmath.doprint(expr1) == 'mpmath.fresnelc(x)'
    assert p_mpmath.doprint(expr2) == 'mpmath.fresnels(x)'


def test_beta():
    from sympy.functions.special.beta_functions import beta

    expr = beta(x, y)

    prntr = SciPyPrinter()
    assert prntr.doprint(expr) == 'scipy.special.beta(x, y)'

    prntr = NumPyPrinter()
    assert prntr.doprint(expr) == '(math.gamma(x)*math.gamma(y)/math.gamma(x + y))'

    prntr = PythonCodePrinter()
    assert prntr.doprint(expr) == '(math.gamma(x)*math.gamma(y)/math.gamma(x + y))'

    prntr = PythonCodePrinter({'allow_unknown_functions': True})
    assert prntr.doprint(expr) == '(math.gamma(x)*math.gamma(y)/math.gamma(x + y))'

    prntr = MpmathPrinter()
    assert prntr.doprint(expr) ==  'mpmath.beta(x, y)'

def test_airy():
    from sympy.functions.special.bessel import (airyai, airybi)

    expr1 = airyai(x)
    expr2 = airybi(x)

    prntr = SciPyPrinter()
    assert prntr.doprint(expr1) == 'scipy.special.airy(x)[0]'
    assert prntr.doprint(expr2) == 'scipy.special.airy(x)[2]'

    prntr = NumPyPrinter({'strict': False})
    assert "Not supported" in prntr.doprint(expr1)
    assert "Not supported" in prntr.doprint(expr2)

    prntr = PythonCodePrinter({'strict': False})
    assert "Not supported" in prntr.doprint(expr1)
    assert "Not supported" in prntr.doprint(expr2)

def test_airy_prime():
    from sympy.functions.special.bessel import (airyaiprime, airybiprime)

    expr1 = airyaiprime(x)
    expr2 = airybiprime(x)

    prntr = SciPyPrinter()
    assert prntr.doprint(expr1) == 'scipy.special.airy(x)[1]'
    assert prntr.doprint(expr2) == 'scipy.special.airy(x)[3]'

    prntr = NumPyPrinter({'strict': False})
    assert "Not supported" in prntr.doprint(expr1)
    assert "Not supported" in prntr.doprint(expr2)

    prntr = PythonCodePrinter({'strict': False})
    assert "Not supported" in prntr.doprint(expr1)
    assert "Not supported" in prntr.doprint(expr2)


def test_numerical_accuracy_functions():
    prntr = SciPyPrinter()
    assert prntr.doprint(expm1(x)) == 'numpy.expm1(x)'
    assert prntr.doprint(log1p(x)) == 'numpy.log1p(x)'
    assert prntr.doprint(cosm1(x)) == 'scipy.special.cosm1(x)'

def test_array_printer():
    A = ArraySymbol('A', (4,4,6,6,6))
    I = IndexedBase('I')
    i,j,k = Idx('i', (0,1)), Idx('j', (2,3)), Idx('k', (4,5))

    prntr = NumPyPrinter()
    assert prntr.doprint(ZeroArray(5)) == 'numpy.zeros((5,))'
    assert prntr.doprint(OneArray(5)) == 'numpy.ones((5,))'
    assert prntr.doprint(ArrayContraction(A, [2,3])) == 'numpy.einsum("abccd->abd", A)'
    assert prntr.doprint(I) == 'I'
    assert prntr.doprint(ArrayDiagonal(A, [2,3,4])) == 'numpy.einsum("abccc->abc", A)'
    assert prntr.doprint(ArrayDiagonal(A, [0,1], [2,3])) == 'numpy.einsum("aabbc->cab", A)'
    assert prntr.doprint(ArrayContraction(A, [2], [3])) == 'numpy.einsum("abcde->abe", A)'
    assert prntr.doprint(Assignment(I[i,j,k], I[i,j,k])) == 'I = I'

    prntr = TensorflowPrinter()
    assert prntr.doprint(ZeroArray(5)) == 'tensorflow.zeros((5,))'
    assert prntr.doprint(OneArray(5)) == 'tensorflow.ones((5,))'
    assert prntr.doprint(ArrayContraction(A, [2,3])) == 'tensorflow.linalg.einsum("abccd->abd", A)'
    assert prntr.doprint(I) == 'I'
    assert prntr.doprint(ArrayDiagonal(A, [2,3,4])) == 'tensorflow.linalg.einsum("abccc->abc", A)'
    assert prntr.doprint(ArrayDiagonal(A, [0,1], [2,3])) == 'tensorflow.linalg.einsum("aabbc->cab", A)'
    assert prntr.doprint(ArrayContraction(A, [2], [3])) == 'tensorflow.linalg.einsum("abcde->abe", A)'
    assert prntr.doprint(Assignment(I[i,j,k], I[i,j,k])) == 'I = I'
