from sympy.core.add import Add
from sympy.core.expr import Expr
from sympy.core.function import (Function, Lambda, diff)
from sympy.core.mod import Mod
from sympy.core import (Catalan, EulerGamma, GoldenRatio)
from sympy.core.numbers import (E, Float, I, Integer, Rational, pi)
from sympy.core.relational import Eq
from sympy.core.singleton import S
from sympy.core.symbol import (Dummy, symbols)
from sympy.functions.combinatorial.factorials import factorial
from sympy.functions.elementary.complexes import (conjugate, sign)
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.piecewise import Piecewise
from sympy.functions.elementary.trigonometric import (atan2, cos, sin)
from sympy.functions.special.gamma_functions import gamma
from sympy.integrals.integrals import Integral
from sympy.sets.fancysets import Range

from sympy.codegen import For, Assignment, aug_assign
from sympy.codegen.ast import Declaration, Variable, float32, float64, \
        value_const, real, bool_, While, FunctionPrototype, FunctionDefinition, \
        integer, Return, Element
from sympy.core.expr import UnevaluatedExpr
from sympy.core.relational import Relational
from sympy.logic.boolalg import And, Or, Not, Equivalent, Xor
from sympy.matrices import Matrix, MatrixSymbol
from sympy.printing.fortran import fcode, FCodePrinter
from sympy.tensor import IndexedBase, Idx
from sympy.tensor.array.expressions import ArraySymbol, ArrayElement
from sympy.utilities.lambdify import implemented_function
from sympy.testing.pytest import raises


def test_UnevaluatedExpr():
    p, q, r = symbols("p q r", real=True)
    q_r = UnevaluatedExpr(q + r)
    expr = abs(exp(p+q_r))
    assert fcode(expr, source_format="free") == "exp(p + (q + r))"
    x, y, z = symbols("x y z")
    y_z = UnevaluatedExpr(y + z)
    expr2 = abs(exp(x+y_z))
    assert fcode(expr2, human=False)[2].lstrip() == "exp(re(x) + re(y + z))"
    assert fcode(expr2, user_functions={"re": "realpart"}).lstrip() == "exp(realpart(x) + realpart(y + z))"


def test_printmethod():
    x = symbols('x')

    class nint(Function):
        def _fcode(self, printer):
            return "nint(%s)" % printer._print(self.args[0])
    assert fcode(nint(x)) == "      nint(x)"


def test_fcode_sign():  #issue 12267
    x=symbols('x')
    y=symbols('y', integer=True)
    z=symbols('z', complex=True)
    assert fcode(sign(x), standard=95, source_format='free') == "merge(0d0, dsign(1d0, x), x == 0d0)"
    assert fcode(sign(y), standard=95, source_format='free') == "merge(0, isign(1, y), y == 0)"
    assert fcode(sign(z), standard=95, source_format='free') == "merge(cmplx(0d0, 0d0), z/abs(z), abs(z) == 0d0)"
    raises(NotImplementedError, lambda: fcode(sign(x)))


def test_fcode_Pow():
    x, y = symbols('x,y')
    n = symbols('n', integer=True)

    assert fcode(x**3) == "      x**3"
    assert fcode(x**(y**3)) == "      x**(y**3)"
    assert fcode(1/(sin(x)*3.5)**(x - y**x)/(x**2 + y)) == \
        "      (3.5d0*sin(x))**(-x + y**x)/(x**2 + y)"
    assert fcode(sqrt(x)) == '      sqrt(x)'
    assert fcode(sqrt(n)) == '      sqrt(dble(n))'
    assert fcode(x**0.5) == '      sqrt(x)'
    assert fcode(sqrt(x)) == '      sqrt(x)'
    assert fcode(sqrt(10)) == '      sqrt(10.0d0)'
    assert fcode(x**-1.0) == '      1d0/x'
    assert fcode(x**-2.0, 'y', source_format='free') == 'y = x**(-2.0d0)'  # 2823
    assert fcode(x**Rational(3, 7)) == '      x**(3.0d0/7.0d0)'


def test_fcode_Rational():
    x = symbols('x')
    assert fcode(Rational(3, 7)) == "      3.0d0/7.0d0"
    assert fcode(Rational(18, 9)) == "      2"
    assert fcode(Rational(3, -7)) == "      -3.0d0/7.0d0"
    assert fcode(Rational(-3, -7)) == "      3.0d0/7.0d0"
    assert fcode(x + Rational(3, 7)) == "      x + 3.0d0/7.0d0"
    assert fcode(Rational(3, 7)*x) == "      (3.0d0/7.0d0)*x"


def test_fcode_Integer():
    assert fcode(Integer(67)) == "      67"
    assert fcode(Integer(-1)) == "      -1"


def test_fcode_Float():
    assert fcode(Float(42.0)) == "      42.0000000000000d0"
    assert fcode(Float(-1e20)) == "      -1.00000000000000d+20"


def test_fcode_functions():
    x, y = symbols('x,y')
    assert fcode(sin(x) ** cos(y)) == "      sin(x)**cos(y)"
    raises(NotImplementedError, lambda: fcode(Mod(x, y), standard=66))
    raises(NotImplementedError, lambda: fcode(x % y, standard=66))
    raises(NotImplementedError, lambda: fcode(Mod(x, y), standard=77))
    raises(NotImplementedError, lambda: fcode(x % y, standard=77))
    for standard in [90, 95, 2003, 2008]:
        assert fcode(Mod(x, y), standard=standard) == "      modulo(x, y)"
        assert fcode(x % y, standard=standard) == "      modulo(x, y)"


def test_case():
    ob = FCodePrinter()
    x,x_,x__,y,X,X_,Y = symbols('x,x_,x__,y,X,X_,Y')
    assert fcode(exp(x_) + sin(x*y) + cos(X*Y)) == \
                        '      exp(x_) + sin(x*y) + cos(X__*Y_)'
    assert fcode(exp(x__) + 2*x*Y*X_**Rational(7, 2)) == \
                        '      2*X_**(7.0d0/2.0d0)*Y*x + exp(x__)'
    assert fcode(exp(x_) + sin(x*y) + cos(X*Y), name_mangling=False) == \
                        '      exp(x_) + sin(x*y) + cos(X*Y)'
    assert fcode(x - cos(X), name_mangling=False) == '      x - cos(X)'
    assert ob.doprint(X*sin(x) + x_, assign_to='me') == '      me = X*sin(x_) + x__'
    assert ob.doprint(X*sin(x), assign_to='mu') == '      mu = X*sin(x_)'
    assert ob.doprint(x_, assign_to='ad') == '      ad = x__'
    n, m = symbols('n,m', integer=True)
    A = IndexedBase('A')
    x = IndexedBase('x')
    y = IndexedBase('y')
    i = Idx('i', m)
    I = Idx('I', n)
    assert fcode(A[i, I]*x[I], assign_to=y[i], source_format='free') == (
                                            "do i = 1, m\n"
                                            "   y(i) = 0\n"
                                            "end do\n"
                                            "do i = 1, m\n"
                                            "   do I_ = 1, n\n"
                                            "      y(i) = A(i, I_)*x(I_) + y(i)\n"
                                            "   end do\n"
                                            "end do" )


#issue 6814
def test_fcode_functions_with_integers():
    x= symbols('x')
    log10_17 = log(10).evalf(17)
    loglog10_17 = '0.8340324452479558d0'
    assert fcode(x * log(10)) == "      x*%sd0" % log10_17
    assert fcode(x * log(10)) == "      x*%sd0" % log10_17
    assert fcode(x * log(S(10))) == "      x*%sd0" % log10_17
    assert fcode(log(S(10))) == "      %sd0" % log10_17
    assert fcode(exp(10)) == "      %sd0" % exp(10).evalf(17)
    assert fcode(x * log(log(10))) == "      x*%s" % loglog10_17
    assert fcode(x * log(log(S(10)))) == "      x*%s" % loglog10_17


def test_fcode_NumberSymbol():
    prec = 17
    p = FCodePrinter()
    assert fcode(Catalan) == '      parameter (Catalan = %sd0)\n      Catalan' % Catalan.evalf(prec)
    assert fcode(EulerGamma) == '      parameter (EulerGamma = %sd0)\n      EulerGamma' % EulerGamma.evalf(prec)
    assert fcode(E) == '      parameter (E = %sd0)\n      E' % E.evalf(prec)
    assert fcode(GoldenRatio) == '      parameter (GoldenRatio = %sd0)\n      GoldenRatio' % GoldenRatio.evalf(prec)
    assert fcode(pi) == '      parameter (pi = %sd0)\n      pi' % pi.evalf(prec)
    assert fcode(
        pi, precision=5) == '      parameter (pi = %sd0)\n      pi' % pi.evalf(5)
    assert fcode(Catalan, human=False) == ({
        (Catalan, p._print(Catalan.evalf(prec)))}, set(), '      Catalan')
    assert fcode(EulerGamma, human=False) == ({(EulerGamma, p._print(
        EulerGamma.evalf(prec)))}, set(), '      EulerGamma')
    assert fcode(E, human=False) == (
        {(E, p._print(E.evalf(prec)))}, set(), '      E')
    assert fcode(GoldenRatio, human=False) == ({(GoldenRatio, p._print(
        GoldenRatio.evalf(prec)))}, set(), '      GoldenRatio')
    assert fcode(pi, human=False) == (
        {(pi, p._print(pi.evalf(prec)))}, set(), '      pi')
    assert fcode(pi, precision=5, human=False) == (
        {(pi, p._print(pi.evalf(5)))}, set(), '      pi')


def test_fcode_complex():
    assert fcode(I) == "      cmplx(0,1)"
    x = symbols('x')
    assert fcode(4*I) == "      cmplx(0,4)"
    assert fcode(3 + 4*I) == "      cmplx(3,4)"
    assert fcode(3 + 4*I + x) == "      cmplx(3,4) + x"
    assert fcode(I*x) == "      cmplx(0,1)*x"
    assert fcode(3 + 4*I - x) == "      cmplx(3,4) - x"
    x = symbols('x', imaginary=True)
    assert fcode(5*x) == "      5*x"
    assert fcode(I*x) == "      cmplx(0,1)*x"
    assert fcode(3 + x) == "      x + 3"


def test_implicit():
    x, y = symbols('x,y')
    assert fcode(sin(x)) == "      sin(x)"
    assert fcode(atan2(x, y)) == "      atan2(x, y)"
    assert fcode(conjugate(x)) == "      conjg(x)"


def test_not_fortran():
    x = symbols('x')
    g = Function('g')
    with raises(NotImplementedError):
        fcode(gamma(x))
    assert fcode(Integral(sin(x)), strict=False) == "C     Not supported in Fortran:\nC     Integral\n      Integral(sin(x), x)"
    with raises(NotImplementedError):
        fcode(g(x))


def test_user_functions():
    x = symbols('x')
    assert fcode(sin(x), user_functions={"sin": "zsin"}) == "      zsin(x)"
    x = symbols('x')
    assert fcode(
        gamma(x), user_functions={"gamma": "mygamma"}) == "      mygamma(x)"
    g = Function('g')
    assert fcode(g(x), user_functions={"g": "great"}) == "      great(x)"
    n = symbols('n', integer=True)
    assert fcode(
        factorial(n), user_functions={"factorial": "fct"}) == "      fct(n)"


def test_inline_function():
    x = symbols('x')
    g = implemented_function('g', Lambda(x, 2*x))
    assert fcode(g(x)) == "      2*x"
    g = implemented_function('g', Lambda(x, 2*pi/x))
    assert fcode(g(x)) == (
        "      parameter (pi = %sd0)\n"
        "      2*pi/x"
    ) % pi.evalf(17)
    A = IndexedBase('A')
    i = Idx('i', symbols('n', integer=True))
    g = implemented_function('g', Lambda(x, x*(1 + x)*(2 + x)))
    assert fcode(g(A[i]), assign_to=A[i]) == (
        "      do i = 1, n\n"
        "         A(i) = (A(i) + 1)*(A(i) + 2)*A(i)\n"
        "      end do"
    )


def test_assign_to():
    x = symbols('x')
    assert fcode(sin(x), assign_to="s") == "      s = sin(x)"


def test_line_wrapping():
    x, y = symbols('x,y')
    assert fcode(((x + y)**10).expand(), assign_to="var") == (
        "      var = x**10 + 10*x**9*y + 45*x**8*y**2 + 120*x**7*y**3 + 210*x**6*\n"
        "     @ y**4 + 252*x**5*y**5 + 210*x**4*y**6 + 120*x**3*y**7 + 45*x**2*y\n"
        "     @ **8 + 10*x*y**9 + y**10"
    )
    e = [x**i for i in range(11)]
    assert fcode(Add(*e)) == (
        "      x**10 + x**9 + x**8 + x**7 + x**6 + x**5 + x**4 + x**3 + x**2 + x\n"
        "     @ + 1"
    )


def test_fcode_precedence():
    x, y = symbols("x y")
    assert fcode(And(x < y, y < x + 1), source_format="free") == \
        "x < y .and. y < x + 1"
    assert fcode(Or(x < y, y < x + 1), source_format="free") == \
        "x < y .or. y < x + 1"
    assert fcode(Xor(x < y, y < x + 1, evaluate=False),
        source_format="free") == "x < y .neqv. y < x + 1"
    assert fcode(Equivalent(x < y, y < x + 1), source_format="free") == \
        "x < y .eqv. y < x + 1"


def test_fcode_Logical():
    x, y, z = symbols("x y z")
    # unary Not
    assert fcode(Not(x), source_format="free") == ".not. x"
    # binary And
    assert fcode(And(x, y), source_format="free") == "x .and. y"
    assert fcode(And(x, Not(y)), source_format="free") == "x .and. .not. y"
    assert fcode(And(Not(x), y), source_format="free") == "y .and. .not. x"
    assert fcode(And(Not(x), Not(y)), source_format="free") == \
        ".not. x .and. .not. y"
    assert fcode(Not(And(x, y), evaluate=False), source_format="free") == \
        ".not. (x .and. y)"
    # binary Or
    assert fcode(Or(x, y), source_format="free") == "x .or. y"
    assert fcode(Or(x, Not(y)), source_format="free") == "x .or. .not. y"
    assert fcode(Or(Not(x), y), source_format="free") == "y .or. .not. x"
    assert fcode(Or(Not(x), Not(y)), source_format="free") == \
        ".not. x .or. .not. y"
    assert fcode(Not(Or(x, y), evaluate=False), source_format="free") == \
        ".not. (x .or. y)"
    # mixed And/Or
    assert fcode(And(Or(y, z), x), source_format="free") == "x .and. (y .or. z)"
    assert fcode(And(Or(z, x), y), source_format="free") == "y .and. (x .or. z)"
    assert fcode(And(Or(x, y), z), source_format="free") == "z .and. (x .or. y)"
    assert fcode(Or(And(y, z), x), source_format="free") == "x .or. y .and. z"
    assert fcode(Or(And(z, x), y), source_format="free") == "y .or. x .and. z"
    assert fcode(Or(And(x, y), z), source_format="free") == "z .or. x .and. y"
    # trinary And
    assert fcode(And(x, y, z), source_format="free") == "x .and. y .and. z"
    assert fcode(And(x, y, Not(z)), source_format="free") == \
        "x .and. y .and. .not. z"
    assert fcode(And(x, Not(y), z), source_format="free") == \
        "x .and. z .and. .not. y"
    assert fcode(And(Not(x), y, z), source_format="free") == \
        "y .and. z .and. .not. x"
    assert fcode(Not(And(x, y, z), evaluate=False), source_format="free") == \
        ".not. (x .and. y .and. z)"
    # trinary Or
    assert fcode(Or(x, y, z), source_format="free") == "x .or. y .or. z"
    assert fcode(Or(x, y, Not(z)), source_format="free") == \
        "x .or. y .or. .not. z"
    assert fcode(Or(x, Not(y), z), source_format="free") == \
        "x .or. z .or. .not. y"
    assert fcode(Or(Not(x), y, z), source_format="free") == \
        "y .or. z .or. .not. x"
    assert fcode(Not(Or(x, y, z), evaluate=False), source_format="free") == \
        ".not. (x .or. y .or. z)"


def test_fcode_Xlogical():
    x, y, z = symbols("x y z")
    # binary Xor
    assert fcode(Xor(x, y, evaluate=False), source_format="free") == \
        "x .neqv. y"
    assert fcode(Xor(x, Not(y), evaluate=False), source_format="free") == \
        "x .neqv. .not. y"
    assert fcode(Xor(Not(x), y, evaluate=False), source_format="free") == \
        "y .neqv. .not. x"
    assert fcode(Xor(Not(x), Not(y), evaluate=False),
        source_format="free") == ".not. x .neqv. .not. y"
    assert fcode(Not(Xor(x, y, evaluate=False), evaluate=False),
        source_format="free") == ".not. (x .neqv. y)"
    # binary Equivalent
    assert fcode(Equivalent(x, y), source_format="free") == "x .eqv. y"
    assert fcode(Equivalent(x, Not(y)), source_format="free") == \
        "x .eqv. .not. y"
    assert fcode(Equivalent(Not(x), y), source_format="free") == \
        "y .eqv. .not. x"
    assert fcode(Equivalent(Not(x), Not(y)), source_format="free") == \
        ".not. x .eqv. .not. y"
    assert fcode(Not(Equivalent(x, y), evaluate=False),
        source_format="free") == ".not. (x .eqv. y)"
    # mixed And/Equivalent
    assert fcode(Equivalent(And(y, z), x), source_format="free") == \
        "x .eqv. y .and. z"
    assert fcode(Equivalent(And(z, x), y), source_format="free") == \
        "y .eqv. x .and. z"
    assert fcode(Equivalent(And(x, y), z), source_format="free") == \
        "z .eqv. x .and. y"
    assert fcode(And(Equivalent(y, z), x), source_format="free") == \
        "x .and. (y .eqv. z)"
    assert fcode(And(Equivalent(z, x), y), source_format="free") == \
        "y .and. (x .eqv. z)"
    assert fcode(And(Equivalent(x, y), z), source_format="free") == \
        "z .and. (x .eqv. y)"
    # mixed Or/Equivalent
    assert fcode(Equivalent(Or(y, z), x), source_format="free") == \
        "x .eqv. y .or. z"
    assert fcode(Equivalent(Or(z, x), y), source_format="free") == \
        "y .eqv. x .or. z"
    assert fcode(Equivalent(Or(x, y), z), source_format="free") == \
        "z .eqv. x .or. y"
    assert fcode(Or(Equivalent(y, z), x), source_format="free") == \
        "x .or. (y .eqv. z)"
    assert fcode(Or(Equivalent(z, x), y), source_format="free") == \
        "y .or. (x .eqv. z)"
    assert fcode(Or(Equivalent(x, y), z), source_format="free") == \
        "z .or. (x .eqv. y)"
    # mixed Xor/Equivalent
    assert fcode(Equivalent(Xor(y, z, evaluate=False), x),
        source_format="free") == "x .eqv. (y .neqv. z)"
    assert fcode(Equivalent(Xor(z, x, evaluate=False), y),
        source_format="free") == "y .eqv. (x .neqv. z)"
    assert fcode(Equivalent(Xor(x, y, evaluate=False), z),
        source_format="free") == "z .eqv. (x .neqv. y)"
    assert fcode(Xor(Equivalent(y, z), x, evaluate=False),
        source_format="free") == "x .neqv. (y .eqv. z)"
    assert fcode(Xor(Equivalent(z, x), y, evaluate=False),
        source_format="free") == "y .neqv. (x .eqv. z)"
    assert fcode(Xor(Equivalent(x, y), z, evaluate=False),
        source_format="free") == "z .neqv. (x .eqv. y)"
    # mixed And/Xor
    assert fcode(Xor(And(y, z), x, evaluate=False), source_format="free") == \
        "x .neqv. y .and. z"
    assert fcode(Xor(And(z, x), y, evaluate=False), source_format="free") == \
        "y .neqv. x .and. z"
    assert fcode(Xor(And(x, y), z, evaluate=False), source_format="free") == \
        "z .neqv. x .and. y"
    assert fcode(And(Xor(y, z, evaluate=False), x), source_format="free") == \
        "x .and. (y .neqv. z)"
    assert fcode(And(Xor(z, x, evaluate=False), y), source_format="free") == \
        "y .and. (x .neqv. z)"
    assert fcode(And(Xor(x, y, evaluate=False), z), source_format="free") == \
        "z .and. (x .neqv. y)"
    # mixed Or/Xor
    assert fcode(Xor(Or(y, z), x, evaluate=False), source_format="free") == \
        "x .neqv. y .or. z"
    assert fcode(Xor(Or(z, x), y, evaluate=False), source_format="free") == \
        "y .neqv. x .or. z"
    assert fcode(Xor(Or(x, y), z, evaluate=False), source_format="free") == \
        "z .neqv. x .or. y"
    assert fcode(Or(Xor(y, z, evaluate=False), x), source_format="free") == \
        "x .or. (y .neqv. z)"
    assert fcode(Or(Xor(z, x, evaluate=False), y), source_format="free") == \
        "y .or. (x .neqv. z)"
    assert fcode(Or(Xor(x, y, evaluate=False), z), source_format="free") == \
        "z .or. (x .neqv. y)"
    # trinary Xor
    assert fcode(Xor(x, y, z, evaluate=False), source_format="free") == \
        "x .neqv. y .neqv. z"
    assert fcode(Xor(x, y, Not(z), evaluate=False), source_format="free") == \
        "x .neqv. y .neqv. .not. z"
    assert fcode(Xor(x, Not(y), z, evaluate=False), source_format="free") == \
        "x .neqv. z .neqv. .not. y"
    assert fcode(Xor(Not(x), y, z, evaluate=False), source_format="free") == \
        "y .neqv. z .neqv. .not. x"


def test_fcode_Relational():
    x, y = symbols("x y")
    assert fcode(Relational(x, y, "=="), source_format="free") == "x == y"
    assert fcode(Relational(x, y, "!="), source_format="free") == "x /= y"
    assert fcode(Relational(x, y, ">="), source_format="free") == "x >= y"
    assert fcode(Relational(x, y, "<="), source_format="free") == "x <= y"
    assert fcode(Relational(x, y, ">"), source_format="free") == "x > y"
    assert fcode(Relational(x, y, "<"), source_format="free") == "x < y"


def test_fcode_Piecewise():
    x = symbols('x')
    expr = Piecewise((x, x < 1), (x**2, True))
    # Check that inline conditional (merge) fails if standard isn't 95+
    raises(NotImplementedError, lambda: fcode(expr))
    code = fcode(expr, standard=95)
    expected = "      merge(x, x**2, x < 1)"
    assert code == expected
    assert fcode(Piecewise((x, x < 1), (x**2, True)), assign_to="var") == (
        "      if (x < 1) then\n"
        "         var = x\n"
        "      else\n"
        "         var = x**2\n"
        "      end if"
    )
    a = cos(x)/x
    b = sin(x)/x
    for i in range(10):
        a = diff(a, x)
        b = diff(b, x)
    expected = (
        "      if (x < 0) then\n"
        "         weird_name = -cos(x)/x + 10*sin(x)/x**2 + 90*cos(x)/x**3 - 720*\n"
        "     @ sin(x)/x**4 - 5040*cos(x)/x**5 + 30240*sin(x)/x**6 + 151200*cos(x\n"
        "     @ )/x**7 - 604800*sin(x)/x**8 - 1814400*cos(x)/x**9 + 3628800*sin(x\n"
        "     @ )/x**10 + 3628800*cos(x)/x**11\n"
        "      else\n"
        "         weird_name = -sin(x)/x - 10*cos(x)/x**2 + 90*sin(x)/x**3 + 720*\n"
        "     @ cos(x)/x**4 - 5040*sin(x)/x**5 - 30240*cos(x)/x**6 + 151200*sin(x\n"
        "     @ )/x**7 + 604800*cos(x)/x**8 - 1814400*sin(x)/x**9 - 3628800*cos(x\n"
        "     @ )/x**10 + 3628800*sin(x)/x**11\n"
        "      end if"
    )
    code = fcode(Piecewise((a, x < 0), (b, True)), assign_to="weird_name")
    assert code == expected
    code = fcode(Piecewise((x, x < 1), (x**2, x > 1), (sin(x), True)), standard=95)
    expected = "      merge(x, merge(x**2, sin(x), x > 1), x < 1)"
    assert code == expected
    # Check that Piecewise without a True (default) condition error
    expr = Piecewise((x, x < 1), (x**2, x > 1), (sin(x), x > 0))
    raises(ValueError, lambda: fcode(expr))


def test_wrap_fortran():
    #   "########################################################################"
    printer = FCodePrinter()
    lines = [
        "C     This is a long comment on a single line that must be wrapped properly to produce nice output",
        "      this = is + a + long + and + nasty + fortran + statement + that * must + be + wrapped + properly",
        "      this = is + a + long + and + nasty + fortran + statement +  that * must + be + wrapped + properly",
        "      this = is + a + long + and + nasty + fortran + statement +   that * must + be + wrapped + properly",
        "      this = is + a + long + and + nasty + fortran + statement + that*must + be + wrapped + properly",
        "      this = is + a + long + and + nasty + fortran + statement +   that*must + be + wrapped + properly",
        "      this = is + a + long + and + nasty + fortran + statement +    that*must + be + wrapped + properly",
        "      this = is + a + long + and + nasty + fortran + statement +     that*must + be + wrapped + properly",
        "      this = is + a + long + and + nasty + fortran + statement + that**must + be + wrapped + properly",
        "      this = is + a + long + and + nasty + fortran + statement +  that**must + be + wrapped + properly",
        "      this = is + a + long + and + nasty + fortran + statement +   that**must + be + wrapped + properly",
        "      this = is + a + long + and + nasty + fortran + statement +    that**must + be + wrapped + properly",
        "      this = is + a + long + and + nasty + fortran + statement +     that**must + be + wrapped + properly",
        "      this = is + a + long + and + nasty + fortran + statement(that)/must + be + wrapped + properly",
        "      this = is + a + long + and + nasty + fortran +     statement(that)/must + be + wrapped + properly",
    ]
    wrapped_lines = printer._wrap_fortran(lines)
    expected_lines = [
        "C     This is a long comment on a single line that must be wrapped",
        "C     properly to produce nice output",
        "      this = is + a + long + and + nasty + fortran + statement + that *",
        "     @ must + be + wrapped + properly",
        "      this = is + a + long + and + nasty + fortran + statement +  that *",
        "     @ must + be + wrapped + properly",
        "      this = is + a + long + and + nasty + fortran + statement +   that",
        "     @ * must + be + wrapped + properly",
        "      this = is + a + long + and + nasty + fortran + statement + that*",
        "     @ must + be + wrapped + properly",
        "      this = is + a + long + and + nasty + fortran + statement +   that*",
        "     @ must + be + wrapped + properly",
        "      this = is + a + long + and + nasty + fortran + statement +    that",
        "     @ *must + be + wrapped + properly",
        "      this = is + a + long + and + nasty + fortran + statement +",
        "     @ that*must + be + wrapped + properly",
        "      this = is + a + long + and + nasty + fortran + statement + that**",
        "     @ must + be + wrapped + properly",
        "      this = is + a + long + and + nasty + fortran + statement +  that**",
        "     @ must + be + wrapped + properly",
        "      this = is + a + long + and + nasty + fortran + statement +   that",
        "     @ **must + be + wrapped + properly",
        "      this = is + a + long + and + nasty + fortran + statement +    that",
        "     @ **must + be + wrapped + properly",
        "      this = is + a + long + and + nasty + fortran + statement +",
        "     @ that**must + be + wrapped + properly",
        "      this = is + a + long + and + nasty + fortran + statement(that)/",
        "     @ must + be + wrapped + properly",
        "      this = is + a + long + and + nasty + fortran +     statement(that)",
        "     @ /must + be + wrapped + properly",
    ]
    for line in wrapped_lines:
        assert len(line) <= 72
    for w, e in zip(wrapped_lines, expected_lines):
        assert w == e
    assert len(wrapped_lines) == len(expected_lines)


def test_wrap_fortran_keep_d0():
    printer = FCodePrinter()
    lines = [
        '      this_variable_is_very_long_because_we_try_to_test_line_break=1.0d0',
        '      this_variable_is_very_long_because_we_try_to_test_line_break =1.0d0',
        '      this_variable_is_very_long_because_we_try_to_test_line_break  = 1.0d0',
        '      this_variable_is_very_long_because_we_try_to_test_line_break   = 1.0d0',
        '      this_variable_is_very_long_because_we_try_to_test_line_break    = 1.0d0',
        '      this_variable_is_very_long_because_we_try_to_test_line_break = 10.0d0'
    ]
    expected = [
        '      this_variable_is_very_long_because_we_try_to_test_line_break=1.0d0',
        '      this_variable_is_very_long_because_we_try_to_test_line_break =',
        '     @ 1.0d0',
        '      this_variable_is_very_long_because_we_try_to_test_line_break  =',
        '     @ 1.0d0',
        '      this_variable_is_very_long_because_we_try_to_test_line_break   =',
        '     @ 1.0d0',
        '      this_variable_is_very_long_because_we_try_to_test_line_break    =',
        '     @ 1.0d0',
        '      this_variable_is_very_long_because_we_try_to_test_line_break =',
        '     @ 10.0d0'
    ]
    assert printer._wrap_fortran(lines) == expected


def test_settings():
    raises(TypeError, lambda: fcode(S(4), method="garbage"))


def test_free_form_code_line():
    x, y = symbols('x,y')
    assert fcode(cos(x) + sin(y), source_format='free') == "sin(y) + cos(x)"


def test_free_form_continuation_line():
    x, y = symbols('x,y')
    result = fcode(((cos(x) + sin(y))**(7)).expand(), source_format='free')
    expected = (
        'sin(y)**7 + 7*sin(y)**6*cos(x) + 21*sin(y)**5*cos(x)**2 + 35*sin(y)**4* &\n'
        '      cos(x)**3 + 35*sin(y)**3*cos(x)**4 + 21*sin(y)**2*cos(x)**5 + 7* &\n'
        '      sin(y)*cos(x)**6 + cos(x)**7'
    )
    assert result == expected


def test_free_form_comment_line():
    printer = FCodePrinter({'source_format': 'free'})
    lines = [ "! This is a long comment on a single line that must be wrapped properly to produce nice output"]
    expected = [
        '! This is a long comment on a single line that must be wrapped properly',
        '! to produce nice output']
    assert printer._wrap_fortran(lines) == expected


def test_loops():
    n, m = symbols('n,m', integer=True)
    A = IndexedBase('A')
    x = IndexedBase('x')
    y = IndexedBase('y')
    i = Idx('i', m)
    j = Idx('j', n)

    expected = (
        'do i = 1, m\n'
        '   y(i) = 0\n'
        'end do\n'
        'do i = 1, m\n'
        '   do j = 1, n\n'
        '      y(i) = %(rhs)s\n'
        '   end do\n'
        'end do'
    )

    code = fcode(A[i, j]*x[j], assign_to=y[i], source_format='free')
    assert (code == expected % {'rhs': 'y(i) + A(i, j)*x(j)'} or
            code == expected % {'rhs': 'y(i) + x(j)*A(i, j)'} or
            code == expected % {'rhs': 'x(j)*A(i, j) + y(i)'} or
            code == expected % {'rhs': 'A(i, j)*x(j) + y(i)'})


def test_dummy_loops():
    i, m = symbols('i m', integer=True, cls=Dummy)
    x = IndexedBase('x')
    y = IndexedBase('y')
    i = Idx(i, m)

    expected = (
        'do i_%(icount)i = 1, m_%(mcount)i\n'
        '   y(i_%(icount)i) = x(i_%(icount)i)\n'
        'end do'
    ) % {'icount': i.label.dummy_index, 'mcount': m.dummy_index}
    code = fcode(x[i], assign_to=y[i], source_format='free')
    assert code == expected


def test_fcode_Indexed_without_looking_for_contraction():
    len_y = 5
    y = IndexedBase('y', shape=(len_y,))
    x = IndexedBase('x', shape=(len_y,))
    Dy = IndexedBase('Dy', shape=(len_y-1,))
    i = Idx('i', len_y-1)
    e=Eq(Dy[i], (y[i+1]-y[i])/(x[i+1]-x[i]))
    code0 = fcode(e.rhs, assign_to=e.lhs, contract=False)
    assert code0.endswith('Dy(i) = (y(i + 1) - y(i))/(x(i + 1) - x(i))')


def test_element_like_objects():
    len_y = 5
    y = ArraySymbol('y', shape=(len_y,))
    x = ArraySymbol('x', shape=(len_y,))
    Dy = ArraySymbol('Dy', shape=(len_y-1,))
    i = Idx('i', len_y-1)
    e=Eq(Dy[i], (y[i+1]-y[i])/(x[i+1]-x[i]))
    code0 = fcode(Assignment(e.lhs, e.rhs))
    assert code0.endswith('Dy(i) = (y(i + 1) - y(i))/(x(i + 1) - x(i))')

    class ElementExpr(Element, Expr):
        pass

    e = e.subs((a, ElementExpr(a.name, a.indices)) for a in e.atoms(ArrayElement)  )
    e=Eq(Dy[i], (y[i+1]-y[i])/(x[i+1]-x[i]))
    code0 = fcode(Assignment(e.lhs, e.rhs))
    assert code0.endswith('Dy(i) = (y(i + 1) - y(i))/(x(i + 1) - x(i))')


def test_derived_classes():
    class MyFancyFCodePrinter(FCodePrinter):
        _default_settings = FCodePrinter._default_settings.copy()

    printer = MyFancyFCodePrinter()
    x = symbols('x')
    assert printer.doprint(sin(x), "bork") == "      bork = sin(x)"


def test_indent():
    codelines = (
        'subroutine test(a)\n'
        'integer :: a, i, j\n'
        '\n'
        'do\n'
        'do \n'
        'do j = 1, 5\n'
        'if (a>b) then\n'
        'if(b>0) then\n'
        'a = 3\n'
        'donot_indent_me = 2\n'
        'do_not_indent_me_either = 2\n'
        'ifIam_indented_something_went_wrong = 2\n'
        'if_I_am_indented_something_went_wrong = 2\n'
        'end should not be unindented here\n'
        'end if\n'
        'endif\n'
        'end do\n'
        'end do\n'
        'enddo\n'
        'end subroutine\n'
        '\n'
        'subroutine test2(a)\n'
        'integer :: a\n'
        'do\n'
        'a = a + 1\n'
        'end do \n'
        'end subroutine\n'
    )
    expected = (
        'subroutine test(a)\n'
        'integer :: a, i, j\n'
        '\n'
        'do\n'
        '   do \n'
        '      do j = 1, 5\n'
        '         if (a>b) then\n'
        '            if(b>0) then\n'
        '               a = 3\n'
        '               donot_indent_me = 2\n'
        '               do_not_indent_me_either = 2\n'
        '               ifIam_indented_something_went_wrong = 2\n'
        '               if_I_am_indented_something_went_wrong = 2\n'
        '               end should not be unindented here\n'
        '            end if\n'
        '         endif\n'
        '      end do\n'
        '   end do\n'
        'enddo\n'
        'end subroutine\n'
        '\n'
        'subroutine test2(a)\n'
        'integer :: a\n'
        'do\n'
        '   a = a + 1\n'
        'end do \n'
        'end subroutine\n'
    )
    p = FCodePrinter({'source_format': 'free'})
    result = p.indent_code(codelines)
    assert result == expected

def test_Matrix_printing():
    x, y, z = symbols('x,y,z')
    # Test returning a Matrix
    mat = Matrix([x*y, Piecewise((2 + x, y>0), (y, True)), sin(z)])
    A = MatrixSymbol('A', 3, 1)
    assert fcode(mat, A) == (
        "      A(1, 1) = x*y\n"
        "      if (y > 0) then\n"
        "         A(2, 1) = x + 2\n"
        "      else\n"
        "         A(2, 1) = y\n"
        "      end if\n"
        "      A(3, 1) = sin(z)")
    # Test using MatrixElements in expressions
    expr = Piecewise((2*A[2, 0], x > 0), (A[2, 0], True)) + sin(A[1, 0]) + A[0, 0]
    assert fcode(expr, standard=95) == (
        "      merge(2*A(3, 1), A(3, 1), x > 0) + sin(A(2, 1)) + A(1, 1)")
    # Test using MatrixElements in a Matrix
    q = MatrixSymbol('q', 5, 1)
    M = MatrixSymbol('M', 3, 3)
    m = Matrix([[sin(q[1,0]), 0, cos(q[2,0])],
        [q[1,0] + q[2,0], q[3, 0], 5],
        [2*q[4, 0]/q[1,0], sqrt(q[0,0]) + 4, 0]])
    assert fcode(m, M) == (
        "      M(1, 1) = sin(q(2, 1))\n"
        "      M(2, 1) = q(2, 1) + q(3, 1)\n"
        "      M(3, 1) = 2*q(5, 1)/q(2, 1)\n"
        "      M(1, 2) = 0\n"
        "      M(2, 2) = q(4, 1)\n"
        "      M(3, 2) = sqrt(q(1, 1)) + 4\n"
        "      M(1, 3) = cos(q(3, 1))\n"
        "      M(2, 3) = 5\n"
        "      M(3, 3) = 0")


def test_fcode_For():
    x, y = symbols('x y')

    f = For(x, Range(0, 10, 2), [Assignment(y, x * y)])
    sol = fcode(f)
    assert sol == ("      do x = 0, 9, 2\n"
                   "         y = x*y\n"
                   "      end do")


def test_fcode_Declaration():
    def check(expr, ref, **kwargs):
        assert fcode(expr, standard=95, source_format='free', **kwargs) == ref

    i = symbols('i', integer=True)
    var1 = Variable.deduced(i)
    dcl1 = Declaration(var1)
    check(dcl1, "integer*4 :: i")


    x, y = symbols('x y')
    var2 = Variable(x, float32, value=42, attrs={value_const})
    dcl2b = Declaration(var2)
    check(dcl2b, 'real*4, parameter :: x = 42')

    var3 = Variable(y, type=bool_)
    dcl3 = Declaration(var3)
    check(dcl3, 'logical :: y')

    check(float32, "real*4")
    check(float64, "real*8")
    check(real, "real*4", type_aliases={real: float32})
    check(real, "real*8", type_aliases={real: float64})


def test_MatrixElement_printing():
    # test cases for issue #11821
    A = MatrixSymbol("A", 1, 3)
    B = MatrixSymbol("B", 1, 3)
    C = MatrixSymbol("C", 1, 3)

    assert(fcode(A[0, 0]) == "      A(1, 1)")
    assert(fcode(3 * A[0, 0]) == "      3*A(1, 1)")

    F = C[0, 0].subs(C, A - B)
    assert(fcode(F) == "      (A - B)(1, 1)")


def test_aug_assign():
    x = symbols('x')
    assert fcode(aug_assign(x, '+', 1), source_format='free') == 'x = x + 1'


def test_While():
    x = symbols('x')
    assert fcode(While(abs(x) > 1, [aug_assign(x, '-', 1)]), source_format='free') == (
        'do while (abs(x) > 1)\n'
        '   x = x - 1\n'
        'end do'
    )


def test_FunctionPrototype_print():
    x = symbols('x')
    n = symbols('n', integer=True)
    vx = Variable(x, type=real)
    vn = Variable(n, type=integer)
    fp1 = FunctionPrototype(real, 'power', [vx, vn])
    # Should be changed to proper test once multi-line generation is working
    # see https://github.com/sympy/sympy/issues/15824
    raises(NotImplementedError, lambda: fcode(fp1))


def test_FunctionDefinition_print():
    x = symbols('x')
    n = symbols('n', integer=True)
    vx = Variable(x, type=real)
    vn = Variable(n, type=integer)
    body = [Assignment(x, x**n), Return(x)]
    fd1 = FunctionDefinition(real, 'power', [vx, vn], body)
    # Should be changed to proper test once multi-line generation is working
    # see https://github.com/sympy/sympy/issues/15824
    raises(NotImplementedError, lambda: fcode(fd1))
