from sympy.core import (S, pi, oo, symbols, Function, Rational, Integer, Tuple,
                        Derivative, Eq, Ne, Le, Lt, Gt, Ge)
from sympy.integrals import Integral
from sympy.concrete import Sum
from sympy.functions import (exp, sin, cos, fresnelc, fresnels, conjugate, Max,
                             Min, gamma, polygamma, loggamma, erf, erfi, erfc,
                             erf2, expint, erfinv, erfcinv, Ei, Si, Ci, li,
                             Shi, Chi, uppergamma, beta, subfactorial, erf2inv,
                             factorial, factorial2, catalan, RisingFactorial,
                             FallingFactorial, harmonic, atan2, sec, acsc,
                             hermite, laguerre, assoc_laguerre, jacobi,
                             gegenbauer, chebyshevt, chebyshevu, legendre,
                             assoc_legendre, Li, LambertW)

from sympy.printing.mathematica import mathematica_code as mcode

x, y, z, w = symbols('x,y,z,w')
f = Function('f')


def test_Integer():
    assert mcode(Integer(67)) == "67"
    assert mcode(Integer(-1)) == "-1"


def test_Rational():
    assert mcode(Rational(3, 7)) == "3/7"
    assert mcode(Rational(18, 9)) == "2"
    assert mcode(Rational(3, -7)) == "-3/7"
    assert mcode(Rational(-3, -7)) == "3/7"
    assert mcode(x + Rational(3, 7)) == "x + 3/7"
    assert mcode(Rational(3, 7)*x) == "(3/7)*x"


def test_Relational():
    assert mcode(Eq(x, y)) == "x == y"
    assert mcode(Ne(x, y)) == "x != y"
    assert mcode(Le(x, y)) == "x <= y"
    assert mcode(Lt(x, y)) == "x < y"
    assert mcode(Gt(x, y)) == "x > y"
    assert mcode(Ge(x, y)) == "x >= y"


def test_Function():
    assert mcode(f(x, y, z)) == "f[x, y, z]"
    assert mcode(sin(x) ** cos(x)) == "Sin[x]^Cos[x]"
    assert mcode(sec(x) * acsc(x)) == "ArcCsc[x]*Sec[x]"
    assert mcode(atan2(y, x)) == "ArcTan[x, y]"
    assert mcode(conjugate(x)) == "Conjugate[x]"
    assert mcode(Max(x, y, z)*Min(y, z)) == "Max[x, y, z]*Min[y, z]"
    assert mcode(fresnelc(x)) == "FresnelC[x]"
    assert mcode(fresnels(x)) == "FresnelS[x]"
    assert mcode(gamma(x)) == "Gamma[x]"
    assert mcode(uppergamma(x, y)) == "Gamma[x, y]"
    assert mcode(polygamma(x, y)) == "PolyGamma[x, y]"
    assert mcode(loggamma(x)) == "LogGamma[x]"
    assert mcode(erf(x)) == "Erf[x]"
    assert mcode(erfc(x)) == "Erfc[x]"
    assert mcode(erfi(x)) == "Erfi[x]"
    assert mcode(erf2(x, y)) == "Erf[x, y]"
    assert mcode(expint(x, y)) == "ExpIntegralE[x, y]"
    assert mcode(erfcinv(x)) == "InverseErfc[x]"
    assert mcode(erfinv(x)) == "InverseErf[x]"
    assert mcode(erf2inv(x, y)) == "InverseErf[x, y]"
    assert mcode(Ei(x)) == "ExpIntegralEi[x]"
    assert mcode(Ci(x)) == "CosIntegral[x]"
    assert mcode(li(x)) == "LogIntegral[x]"
    assert mcode(Si(x)) == "SinIntegral[x]"
    assert mcode(Shi(x)) == "SinhIntegral[x]"
    assert mcode(Chi(x)) == "CoshIntegral[x]"
    assert mcode(beta(x, y)) == "Beta[x, y]"
    assert mcode(factorial(x)) == "Factorial[x]"
    assert mcode(factorial2(x)) == "Factorial2[x]"
    assert mcode(subfactorial(x)) == "Subfactorial[x]"
    assert mcode(FallingFactorial(x, y)) == "FactorialPower[x, y]"
    assert mcode(RisingFactorial(x, y)) == "Pochhammer[x, y]"
    assert mcode(catalan(x)) == "CatalanNumber[x]"
    assert mcode(harmonic(x)) == "HarmonicNumber[x]"
    assert mcode(harmonic(x, y)) == "HarmonicNumber[x, y]"
    assert mcode(Li(x)) == "LogIntegral[x] - LogIntegral[2]"
    assert mcode(LambertW(x)) == "ProductLog[x]"
    assert mcode(LambertW(x, -1)) == "ProductLog[-1, x]"
    assert mcode(LambertW(x, y)) == "ProductLog[y, x]"


def test_special_polynomials():
    assert mcode(hermite(x, y)) == "HermiteH[x, y]"
    assert mcode(laguerre(x, y)) == "LaguerreL[x, y]"
    assert mcode(assoc_laguerre(x, y, z)) == "LaguerreL[x, y, z]"
    assert mcode(jacobi(x, y, z, w)) == "JacobiP[x, y, z, w]"
    assert mcode(gegenbauer(x, y, z)) == "GegenbauerC[x, y, z]"
    assert mcode(chebyshevt(x, y)) == "ChebyshevT[x, y]"
    assert mcode(chebyshevu(x, y)) == "ChebyshevU[x, y]"
    assert mcode(legendre(x, y)) == "LegendreP[x, y]"
    assert mcode(assoc_legendre(x, y, z)) == "LegendreP[x, y, z]"


def test_Pow():
    assert mcode(x**3) == "x^3"
    assert mcode(x**(y**3)) == "x^(y^3)"
    assert mcode(1/(f(x)*3.5)**(x - y**x)/(x**2 + y)) == \
        "(3.5*f[x])^(-x + y^x)/(x^2 + y)"
    assert mcode(x**-1.0) == 'x^(-1.0)'
    assert mcode(x**Rational(2, 3)) == 'x^(2/3)'


def test_Mul():
    A, B, C, D = symbols('A B C D', commutative=False)
    assert mcode(x*y*z) == "x*y*z"
    assert mcode(x*y*A) == "x*y*A"
    assert mcode(x*y*A*B) == "x*y*A**B"
    assert mcode(x*y*A*B*C) == "x*y*A**B**C"
    assert mcode(x*A*B*(C + D)*A*y) == "x*y*A**B**(C + D)**A"


def test_constants():
    assert mcode(S.Zero) == "0"
    assert mcode(S.One) == "1"
    assert mcode(S.NegativeOne) == "-1"
    assert mcode(S.Half) == "1/2"
    assert mcode(S.ImaginaryUnit) == "I"

    assert mcode(oo) == "Infinity"
    assert mcode(S.NegativeInfinity) == "-Infinity"
    assert mcode(S.ComplexInfinity) == "ComplexInfinity"
    assert mcode(S.NaN) == "Indeterminate"

    assert mcode(S.Exp1) == "E"
    assert mcode(pi) == "Pi"
    assert mcode(S.GoldenRatio) == "GoldenRatio"
    assert mcode(S.TribonacciConstant) == \
        "(1/3 + (1/3)*(19 - 3*33^(1/2))^(1/3) + " \
        "(1/3)*(3*33^(1/2) + 19)^(1/3))"
    assert mcode(2*S.TribonacciConstant) == \
        "2*(1/3 + (1/3)*(19 - 3*33^(1/2))^(1/3) + " \
        "(1/3)*(3*33^(1/2) + 19)^(1/3))"
    assert mcode(S.EulerGamma) == "EulerGamma"
    assert mcode(S.Catalan) == "Catalan"


def test_containers():
    assert mcode([1, 2, 3, [4, 5, [6, 7]], 8, [9, 10], 11]) == \
        "{1, 2, 3, {4, 5, {6, 7}}, 8, {9, 10}, 11}"
    assert mcode((1, 2, (3, 4))) == "{1, 2, {3, 4}}"
    assert mcode([1]) == "{1}"
    assert mcode((1,)) == "{1}"
    assert mcode(Tuple(*[1, 2, 3])) == "{1, 2, 3}"


def test_matrices():
    from sympy.matrices import MutableDenseMatrix, MutableSparseMatrix, \
        ImmutableDenseMatrix, ImmutableSparseMatrix
    A = MutableDenseMatrix(
        [[1, -1, 0, 0],
         [0, 1, -1, 0],
         [0, 0, 1, -1],
         [0, 0, 0, 1]]
    )
    B = MutableSparseMatrix(A)
    C = ImmutableDenseMatrix(A)
    D = ImmutableSparseMatrix(A)

    assert mcode(C) == mcode(A) == \
        "{{1, -1, 0, 0}, " \
        "{0, 1, -1, 0}, " \
        "{0, 0, 1, -1}, " \
        "{0, 0, 0, 1}}"

    assert mcode(D) == mcode(B) == \
        "SparseArray[{" \
        "{1, 1} -> 1, {1, 2} -> -1, {2, 2} -> 1, {2, 3} -> -1, " \
        "{3, 3} -> 1, {3, 4} -> -1, {4, 4} -> 1" \
        "}, {4, 4}]"

    # Trivial cases of matrices
    assert mcode(MutableDenseMatrix(0, 0, [])) == '{}'
    assert mcode(MutableSparseMatrix(0, 0, [])) == 'SparseArray[{}, {0, 0}]'
    assert mcode(MutableDenseMatrix(0, 3, [])) == '{}'
    assert mcode(MutableSparseMatrix(0, 3, [])) == 'SparseArray[{}, {0, 3}]'
    assert mcode(MutableDenseMatrix(3, 0, [])) == '{{}, {}, {}}'
    assert mcode(MutableSparseMatrix(3, 0, [])) == 'SparseArray[{}, {3, 0}]'

def test_NDArray():
    from sympy.tensor.array import (
        MutableDenseNDimArray, ImmutableDenseNDimArray,
        MutableSparseNDimArray, ImmutableSparseNDimArray)

    example = MutableDenseNDimArray(
        [[[1, 2, 3, 4],
          [5, 6, 7, 8],
          [9, 10, 11, 12]],
         [[13, 14, 15, 16],
          [17, 18, 19, 20],
          [21, 22, 23, 24]]]
    )

    assert mcode(example) == \
    "{{{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}}, " \
    "{{13, 14, 15, 16}, {17, 18, 19, 20}, {21, 22, 23, 24}}}"

    example = ImmutableDenseNDimArray(example)

    assert mcode(example) == \
    "{{{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}}, " \
    "{{13, 14, 15, 16}, {17, 18, 19, 20}, {21, 22, 23, 24}}}"

    example = MutableSparseNDimArray(example)

    assert mcode(example) == \
    "SparseArray[{" \
        "{1, 1, 1} -> 1, {1, 1, 2} -> 2, {1, 1, 3} -> 3, " \
        "{1, 1, 4} -> 4, {1, 2, 1} -> 5, {1, 2, 2} -> 6, " \
        "{1, 2, 3} -> 7, {1, 2, 4} -> 8, {1, 3, 1} -> 9, " \
        "{1, 3, 2} -> 10, {1, 3, 3} -> 11, {1, 3, 4} -> 12, " \
        "{2, 1, 1} -> 13, {2, 1, 2} -> 14, {2, 1, 3} -> 15, " \
        "{2, 1, 4} -> 16, {2, 2, 1} -> 17, {2, 2, 2} -> 18, " \
        "{2, 2, 3} -> 19, {2, 2, 4} -> 20, {2, 3, 1} -> 21, " \
        "{2, 3, 2} -> 22, {2, 3, 3} -> 23, {2, 3, 4} -> 24" \
        "}, {2, 3, 4}]"

    example = ImmutableSparseNDimArray(example)

    assert mcode(example) == \
    "SparseArray[{" \
        "{1, 1, 1} -> 1, {1, 1, 2} -> 2, {1, 1, 3} -> 3, " \
        "{1, 1, 4} -> 4, {1, 2, 1} -> 5, {1, 2, 2} -> 6, " \
        "{1, 2, 3} -> 7, {1, 2, 4} -> 8, {1, 3, 1} -> 9, " \
        "{1, 3, 2} -> 10, {1, 3, 3} -> 11, {1, 3, 4} -> 12, " \
        "{2, 1, 1} -> 13, {2, 1, 2} -> 14, {2, 1, 3} -> 15, " \
        "{2, 1, 4} -> 16, {2, 2, 1} -> 17, {2, 2, 2} -> 18, " \
        "{2, 2, 3} -> 19, {2, 2, 4} -> 20, {2, 3, 1} -> 21, " \
        "{2, 3, 2} -> 22, {2, 3, 3} -> 23, {2, 3, 4} -> 24" \
        "}, {2, 3, 4}]"


def test_Integral():
    assert mcode(Integral(sin(sin(x)), x)) == "Hold[Integrate[Sin[Sin[x]], x]]"
    assert mcode(Integral(exp(-x**2 - y**2),
                          (x, -oo, oo),
                          (y, -oo, oo))) == \
        "Hold[Integrate[Exp[-x^2 - y^2], {x, -Infinity, Infinity}, " \
        "{y, -Infinity, Infinity}]]"


def test_Derivative():
    assert mcode(Derivative(sin(x), x)) == "Hold[D[Sin[x], x]]"
    assert mcode(Derivative(x, x)) == "Hold[D[x, x]]"
    assert mcode(Derivative(sin(x)*y**4, x, 2)) == "Hold[D[y^4*Sin[x], {x, 2}]]"
    assert mcode(Derivative(sin(x)*y**4, x, y, x)) == "Hold[D[y^4*Sin[x], x, y, x]]"
    assert mcode(Derivative(sin(x)*y**4, x, y, 3, x)) == "Hold[D[y^4*Sin[x], x, {y, 3}, x]]"


def test_Sum():
    assert mcode(Sum(sin(x), (x, 0, 10))) == "Hold[Sum[Sin[x], {x, 0, 10}]]"
    assert mcode(Sum(exp(-x**2 - y**2),
                     (x, -oo, oo),
                     (y, -oo, oo))) == \
        "Hold[Sum[Exp[-x^2 - y^2], {x, -Infinity, Infinity}, " \
        "{y, -Infinity, Infinity}]]"


def test_comment():
    from sympy.printing.mathematica import MCodePrinter
    assert MCodePrinter()._get_comment("Hello World") == \
        "(* Hello World *)"


def test_userfuncs():
    # Dictionary mutation test
    some_function = symbols("some_function", cls=Function)
    my_user_functions = {"some_function": "SomeFunction"}
    assert mcode(
        some_function(z),
        user_functions=my_user_functions) == \
        'SomeFunction[z]'
    assert mcode(
        some_function(z),
        user_functions=my_user_functions) == \
        'SomeFunction[z]'

    # List argument test
    my_user_functions = \
        {"some_function": [(lambda x: True, "SomeOtherFunction")]}
    assert mcode(
        some_function(z),
        user_functions=my_user_functions) == \
        'SomeOtherFunction[z]'
