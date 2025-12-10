from sympy.concrete.summations import Sum
from sympy.core.basic import Basic, _aresame
from sympy.core.cache import clear_cache
from sympy.core.containers import Dict, Tuple
from sympy.core.expr import Expr, unchanged
from sympy.core.function import (Subs, Function, diff, Lambda, expand,
    nfloat, Derivative)
from sympy.core.numbers import E, Float, zoo, Rational, pi, I, oo, nan
from sympy.core.power import Pow
from sympy.core.relational import Eq
from sympy.core.singleton import S
from sympy.core.symbol import symbols, Dummy, Symbol
from sympy.functions.elementary.complexes import im, re
from sympy.functions.elementary.exponential import log, exp
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.piecewise import Piecewise
from sympy.functions.elementary.trigonometric import sin, cos, acos
from sympy.functions.special.error_functions import expint
from sympy.functions.special.gamma_functions import loggamma, polygamma
from sympy.matrices.dense import Matrix
from sympy.printing.str import sstr
from sympy.series.order import O
from sympy.tensor.indexed import Indexed
from sympy.core.function import (PoleError, _mexpand, arity,
        BadSignatureError, BadArgumentsError)
from sympy.core.parameters import _exp_is_pow
from sympy.core.sympify import sympify, SympifyError
from sympy.matrices import MutableMatrix, ImmutableMatrix
from sympy.sets.sets import FiniteSet
from sympy.solvers.solveset import solveset
from sympy.tensor.array import NDimArray
from sympy.utilities.iterables import subsets, variations
from sympy.testing.pytest import XFAIL, raises, warns_deprecated_sympy, _both_exp_pow

from sympy.abc import t, w, x, y, z
f, g, h = symbols('f g h', cls=Function)
_xi_1, _xi_2, _xi_3 = [Dummy() for i in range(3)]

def test_f_expand_complex():
    x = Symbol('x', real=True)

    assert f(x).expand(complex=True) == I*im(f(x)) + re(f(x))
    assert exp(x).expand(complex=True) == exp(x)
    assert exp(I*x).expand(complex=True) == cos(x) + I*sin(x)
    assert exp(z).expand(complex=True) == cos(im(z))*exp(re(z)) + \
        I*sin(im(z))*exp(re(z))


def test_bug1():
    e = sqrt(-log(w))
    assert e.subs(log(w), -x) == sqrt(x)

    e = sqrt(-5*log(w))
    assert e.subs(log(w), -x) == sqrt(5*x)


def test_general_function():
    nu = Function('nu')

    e = nu(x)
    edx = e.diff(x)
    edy = e.diff(y)
    edxdx = e.diff(x).diff(x)
    edxdy = e.diff(x).diff(y)
    assert e == nu(x)
    assert edx != nu(x)
    assert edx == diff(nu(x), x)
    assert edy == 0
    assert edxdx == diff(diff(nu(x), x), x)
    assert edxdy == 0

def test_general_function_nullary():
    nu = Function('nu')

    e = nu()
    edx = e.diff(x)
    edxdx = e.diff(x).diff(x)
    assert e == nu()
    assert edx != nu()
    assert edx == 0
    assert edxdx == 0


def test_derivative_subs_bug():
    e = diff(g(x), x)
    assert e.subs(g(x), f(x)) != e
    assert e.subs(g(x), f(x)) == Derivative(f(x), x)
    assert e.subs(g(x), -f(x)) == Derivative(-f(x), x)

    assert e.subs(x, y) == Derivative(g(y), y)


def test_derivative_subs_self_bug():
    d = diff(f(x), x)

    assert d.subs(d, y) == y


def test_derivative_linearity():
    assert diff(-f(x), x) == -diff(f(x), x)
    assert diff(8*f(x), x) == 8*diff(f(x), x)
    assert diff(8*f(x), x) != 7*diff(f(x), x)
    assert diff(8*f(x)*x, x) == 8*f(x) + 8*x*diff(f(x), x)
    assert diff(8*f(x)*y*x, x).expand() == 8*y*f(x) + 8*y*x*diff(f(x), x)


def test_derivative_evaluate():
    assert Derivative(sin(x), x) != diff(sin(x), x)
    assert Derivative(sin(x), x).doit() == diff(sin(x), x)

    assert Derivative(Derivative(f(x), x), x) == diff(f(x), x, x)
    assert Derivative(sin(x), x, 0) == sin(x)
    assert Derivative(sin(x), (x, y), (x, -y)) == sin(x)


def test_diff_symbols():
    assert diff(f(x, y, z), x, y, z) == Derivative(f(x, y, z), x, y, z)
    assert diff(f(x, y, z), x, x, x) == Derivative(f(x, y, z), x, x, x) == Derivative(f(x, y, z), (x, 3))
    assert diff(f(x, y, z), x, 3) == Derivative(f(x, y, z), x, 3)

    # issue 5028
    assert [diff(-z + x/y, sym) for sym in (z, x, y)] == [-1, 1/y, -x/y**2]
    assert diff(f(x, y, z), x, y, z, 2) == Derivative(f(x, y, z), x, y, z, z)
    assert diff(f(x, y, z), x, y, z, 2, evaluate=False) == \
        Derivative(f(x, y, z), x, y, z, z)
    assert Derivative(f(x, y, z), x, y, z)._eval_derivative(z) == \
        Derivative(f(x, y, z), x, y, z, z)
    assert Derivative(Derivative(f(x, y, z), x), y)._eval_derivative(z) == \
        Derivative(f(x, y, z), x, y, z)

    raises(TypeError, lambda: cos(x).diff((x, y)).variables)
    assert cos(x).diff((x, y))._wrt_variables == [x]

    # issue 23222
    assert sympify("a*x+b").diff("x") == sympify("a")

def test_Function():
    class myfunc(Function):
        @classmethod
        def eval(cls):  # zero args
            return

    assert myfunc.nargs == FiniteSet(0)
    assert myfunc().nargs == FiniteSet(0)
    raises(TypeError, lambda: myfunc(x).nargs)

    class myfunc(Function):
        @classmethod
        def eval(cls, x):  # one arg
            return

    assert myfunc.nargs == FiniteSet(1)
    assert myfunc(x).nargs == FiniteSet(1)
    raises(TypeError, lambda: myfunc(x, y).nargs)

    class myfunc(Function):
        @classmethod
        def eval(cls, *x):  # star args
            return

    assert myfunc.nargs == S.Naturals0
    assert myfunc(x).nargs == S.Naturals0


def test_nargs():
    f = Function('f')
    assert f.nargs == S.Naturals0
    assert f(1).nargs == S.Naturals0
    assert Function('f', nargs=2)(1, 2).nargs == FiniteSet(2)
    assert sin.nargs == FiniteSet(1)
    assert sin(2).nargs == FiniteSet(1)
    assert log.nargs == FiniteSet(1, 2)
    assert log(2).nargs == FiniteSet(1, 2)
    assert Function('f', nargs=2).nargs == FiniteSet(2)
    assert Function('f', nargs=0).nargs == FiniteSet(0)
    assert Function('f', nargs=(0, 1)).nargs == FiniteSet(0, 1)
    assert Function('f', nargs=None).nargs == S.Naturals0
    raises(ValueError, lambda: Function('f', nargs=()))

def test_nargs_inheritance():
    class f1(Function):
        nargs = 2
    class f2(f1):
        pass
    class f3(f2):
        pass
    class f4(f3):
        nargs = 1,2
    class f5(f4):
        pass
    class f6(f5):
        pass
    class f7(f6):
        nargs=None
    class f8(f7):
        pass
    class f9(f8):
        pass
    class f10(f9):
        nargs = 1
    class f11(f10):
        pass
    assert f1.nargs == FiniteSet(2)
    assert f2.nargs == FiniteSet(2)
    assert f3.nargs == FiniteSet(2)
    assert f4.nargs == FiniteSet(1, 2)
    assert f5.nargs == FiniteSet(1, 2)
    assert f6.nargs == FiniteSet(1, 2)
    assert f7.nargs == S.Naturals0
    assert f8.nargs == S.Naturals0
    assert f9.nargs == S.Naturals0
    assert f10.nargs == FiniteSet(1)
    assert f11.nargs == FiniteSet(1)

def test_arity():
    f = lambda x, y: 1
    assert arity(f) == 2
    def f(x, y, z=None):
        pass
    assert arity(f) == (2, 3)
    assert arity(lambda *x: x) is None
    assert arity(log) == (1, 2)


def test_Lambda():
    e = Lambda(x, x**2)
    assert e(4) == 16
    assert e(x) == x**2
    assert e(y) == y**2

    assert Lambda((), 42)() == 42
    assert unchanged(Lambda, (), 42)
    assert Lambda((), 42) != Lambda((), 43)
    assert Lambda((), f(x))() == f(x)
    assert Lambda((), 42).nargs == FiniteSet(0)

    assert unchanged(Lambda, (x,), x**2)
    assert Lambda(x, x**2) == Lambda((x,), x**2)
    assert Lambda(x, x**2) != Lambda(x, x**2 + 1)
    assert Lambda((x, y), x**y) != Lambda((y, x), y**x)
    assert Lambda((x, y), x**y) != Lambda((x, y), y**x)

    assert Lambda((x, y), x**y)(x, y) == x**y
    assert Lambda((x, y), x**y)(3, 3) == 3**3
    assert Lambda((x, y), x**y)(x, 3) == x**3
    assert Lambda((x, y), x**y)(3, y) == 3**y
    assert Lambda(x, f(x))(x) == f(x)
    assert Lambda(x, x**2)(e(x)) == x**4
    assert e(e(x)) == x**4

    x1, x2 = (Indexed('x', i) for i in (1, 2))
    assert Lambda((x1, x2), x1 + x2)(x, y) == x + y

    assert Lambda((x, y), x + y).nargs == FiniteSet(2)

    p = x, y, z, t
    assert Lambda(p, t*(x + y + z))(*p) == t * (x + y + z)

    eq = Lambda(x, 2*x) + Lambda(y, 2*y)
    assert eq != 2*Lambda(x, 2*x)
    assert eq.as_dummy() == 2*Lambda(x, 2*x).as_dummy()
    assert Lambda(x, 2*x) not in [ Lambda(x, x) ]
    raises(BadSignatureError, lambda: Lambda(1, x))
    assert Lambda(x, 1)(1) is S.One

    raises(BadSignatureError, lambda: Lambda((x, x), x + 2))
    raises(BadSignatureError, lambda: Lambda(((x, x), y), x))
    raises(BadSignatureError, lambda: Lambda(((y, x), x), x))
    raises(BadSignatureError, lambda: Lambda(((y, 1), 2), x))

    with warns_deprecated_sympy():
        assert Lambda([x, y], x+y) == Lambda((x, y), x+y)

    flam = Lambda(((x, y),), x + y)
    assert flam((2, 3)) == 5
    flam = Lambda(((x, y), z), x + y + z)
    assert flam((2, 3), 1) == 6
    flam = Lambda((((x, y), z),), x + y + z)
    assert flam(((2, 3), 1)) == 6
    raises(BadArgumentsError, lambda: flam(1, 2, 3))
    flam = Lambda( (x,), (x, x))
    assert flam(1,) == (1, 1)
    assert flam((1,)) == ((1,), (1,))
    flam = Lambda( ((x,),), (x, x))
    raises(BadArgumentsError, lambda: flam(1))
    assert flam((1,)) == (1, 1)

    # Previously TypeError was raised so this is potentially needed for
    # backwards compatibility.
    assert issubclass(BadSignatureError, TypeError)
    assert issubclass(BadArgumentsError, TypeError)

    # These are tested to see they don't raise:
    hash(Lambda(x, 2*x))
    hash(Lambda(x, x))  # IdentityFunction subclass


def test_IdentityFunction():
    assert Lambda(x, x) is Lambda(y, y) is S.IdentityFunction
    assert Lambda(x, 2*x) is not S.IdentityFunction
    assert Lambda((x, y), x) is not S.IdentityFunction


def test_Lambda_symbols():
    assert Lambda(x, 2*x).free_symbols == set()
    assert Lambda(x, x*y).free_symbols == {y}
    assert Lambda((), 42).free_symbols == set()
    assert Lambda((), x*y).free_symbols == {x,y}


def test_functionclas_symbols():
    assert f.free_symbols == set()


def test_Lambda_arguments():
    raises(TypeError, lambda: Lambda(x, 2*x)(x, y))
    raises(TypeError, lambda: Lambda((x, y), x + y)(x))
    raises(TypeError, lambda: Lambda((), 42)(x))


def test_Lambda_equality():
    assert Lambda((x, y), 2*x) == Lambda((x, y), 2*x)
    # these, of course, should never be equal
    assert Lambda(x, 2*x) != Lambda((x, y), 2*x)
    assert Lambda(x, 2*x) != 2*x
    # But it is tempting to want expressions that differ only
    # in bound symbols to compare the same.  But this is not what
    # Python's `==` is intended to do; two objects that compare
    # as equal means that they are indistibguishable and cache to the
    # same value.  We wouldn't want to expression that are
    # mathematically the same but written in different variables to be
    # interchanged else what is the point of allowing for different
    # variable names?
    assert Lambda(x, 2*x) != Lambda(y, 2*y)


def test_Subs():
    assert Subs(1, (), ()) is S.One
    # check null subs influence on hashing
    assert Subs(x, y, z) != Subs(x, y, 1)
    # neutral subs works
    assert Subs(x, x, 1).subs(x, y).has(y)
    # self mapping var/point
    assert Subs(Derivative(f(x), (x, 2)), x, x).doit() == f(x).diff(x, x)
    assert Subs(x, x, 0).has(x)  # it's a structural answer
    assert not Subs(x, x, 0).free_symbols
    assert Subs(Subs(x + y, x, 2), y, 1) == Subs(x + y, (x, y), (2, 1))
    assert Subs(x, (x,), (0,)) == Subs(x, x, 0)
    assert Subs(x, x, 0) == Subs(y, y, 0)
    assert Subs(x, x, 0).subs(x, 1) == Subs(x, x, 0)
    assert Subs(y, x, 0).subs(y, 1) == Subs(1, x, 0)
    assert Subs(f(x), x, 0).doit() == f(0)
    assert Subs(f(x**2), x**2, 0).doit() == f(0)
    assert Subs(f(x, y, z), (x, y, z), (0, 1, 1)) != \
        Subs(f(x, y, z), (x, y, z), (0, 0, 1))
    assert Subs(x, y, 2).subs(x, y).doit() == 2
    assert Subs(f(x, y), (x, y, z), (0, 1, 1)) != \
        Subs(f(x, y) + z, (x, y, z), (0, 1, 0))
    assert Subs(f(x, y), (x, y), (0, 1)).doit() == f(0, 1)
    assert Subs(Subs(f(x, y), x, 0), y, 1).doit() == f(0, 1)
    raises(ValueError, lambda: Subs(f(x, y), (x, y), (0, 0, 1)))
    raises(ValueError, lambda: Subs(f(x, y), (x, x, y), (0, 0, 1)))

    assert len(Subs(f(x, y), (x, y), (0, 1)).variables) == 2
    assert Subs(f(x, y), (x, y), (0, 1)).point == Tuple(0, 1)

    assert Subs(f(x), x, 0) == Subs(f(y), y, 0)
    assert Subs(f(x, y), (x, y), (0, 1)) == Subs(f(x, y), (y, x), (1, 0))
    assert Subs(f(x)*y, (x, y), (0, 1)) == Subs(f(y)*x, (y, x), (0, 1))
    assert Subs(f(x)*y, (x, y), (1, 1)) == Subs(f(y)*x, (x, y), (1, 1))

    assert Subs(f(x), x, 0).subs(x, 1).doit() == f(0)
    assert Subs(f(x), x, y).subs(y, 0) == Subs(f(x), x, 0)
    assert Subs(y*f(x), x, y).subs(y, 2) == Subs(2*f(x), x, 2)
    assert (2 * Subs(f(x), x, 0)).subs(Subs(f(x), x, 0), y) == 2*y

    assert Subs(f(x), x, 0).free_symbols == set()
    assert Subs(f(x, y), x, z).free_symbols == {y, z}

    assert Subs(f(x).diff(x), x, 0).doit(), Subs(f(x).diff(x), x, 0)
    assert Subs(1 + f(x).diff(x), x, 0).doit(), 1 + Subs(f(x).diff(x), x, 0)
    assert Subs(y*f(x, y).diff(x), (x, y), (0, 2)).doit() == \
        2*Subs(Derivative(f(x, 2), x), x, 0)
    assert Subs(y**2*f(x), x, 0).diff(y) == 2*y*f(0)

    e = Subs(y**2*f(x), x, y)
    assert e.diff(y) == e.doit().diff(y) == y**2*Derivative(f(y), y) + 2*y*f(y)

    assert Subs(f(x), x, 0) + Subs(f(x), x, 0) == 2*Subs(f(x), x, 0)
    e1 = Subs(z*f(x), x, 1)
    e2 = Subs(z*f(y), y, 1)
    assert e1 + e2 == 2*e1
    assert e1.__hash__() == e2.__hash__()
    assert Subs(z*f(x + 1), x, 1) not in [ e1, e2 ]
    assert Derivative(f(x), x).subs(x, g(x)) == Derivative(f(g(x)), g(x))
    assert Derivative(f(x), x).subs(x, x + y) == Subs(Derivative(f(x), x),
        x, x + y)
    assert Subs(f(x)*cos(y) + z, (x, y), (0, pi/3)).n(2) == \
        Subs(f(x)*cos(y) + z, (x, y), (0, pi/3)).evalf(2) == \
        z + Rational('1/2').n(2)*f(0)

    assert f(x).diff(x).subs(x, 0).subs(x, y) == f(x).diff(x).subs(x, 0)
    assert (x*f(x).diff(x).subs(x, 0)).subs(x, y) == y*f(x).diff(x).subs(x, 0)
    assert Subs(Derivative(g(x)**2, g(x), x), g(x), exp(x)
        ).doit() == 2*exp(x)
    assert Subs(Derivative(g(x)**2, g(x), x), g(x), exp(x)
        ).doit(deep=False) == 2*Derivative(exp(x), x)
    assert Derivative(f(x, g(x)), x).doit() == Derivative(
        f(x, g(x)), g(x))*Derivative(g(x), x) + Subs(Derivative(
        f(y, g(x)), y), y, x)

def test_doitdoit():
    done = Derivative(f(x, g(x)), x, g(x)).doit()
    assert done == done.doit()


@XFAIL
def test_Subs2():
    # this reflects a limitation of subs(), probably won't fix
    assert Subs(f(x), x**2, x).doit() == f(sqrt(x))


def test_expand_function():
    assert expand(x + y) == x + y
    assert expand(x + y, complex=True) == I*im(x) + I*im(y) + re(x) + re(y)
    assert expand((x + y)**11, modulus=11) == x**11 + y**11


def test_function_comparable():
    assert sin(x).is_comparable is False
    assert cos(x).is_comparable is False

    assert sin(Float('0.1')).is_comparable is True
    assert cos(Float('0.1')).is_comparable is True

    assert sin(E).is_comparable is True
    assert cos(E).is_comparable is True

    assert sin(Rational(1, 3)).is_comparable is True
    assert cos(Rational(1, 3)).is_comparable is True


def test_function_comparable_infinities():
    assert sin(oo).is_comparable is False
    assert sin(-oo).is_comparable is False
    assert sin(zoo).is_comparable is False
    assert sin(nan).is_comparable is False


def test_deriv1():
    # These all require derivatives evaluated at a point (issue 4719) to work.
    # See issue 4624
    assert f(2*x).diff(x) == 2*Subs(Derivative(f(x), x), x, 2*x)
    assert (f(x)**3).diff(x) == 3*f(x)**2*f(x).diff(x)
    assert (f(2*x)**3).diff(x) == 6*f(2*x)**2*Subs(
        Derivative(f(x), x), x, 2*x)

    assert f(2 + x).diff(x) == Subs(Derivative(f(x), x), x, x + 2)
    assert f(2 + 3*x).diff(x) == 3*Subs(
        Derivative(f(x), x), x, 3*x + 2)
    assert f(3*sin(x)).diff(x) == 3*cos(x)*Subs(
        Derivative(f(x), x), x, 3*sin(x))

    # See issue 8510
    assert f(x, x + z).diff(x) == (
        Subs(Derivative(f(y, x + z), y), y, x) +
        Subs(Derivative(f(x, y), y), y, x + z))
    assert f(x, x**2).diff(x) == (
        2*x*Subs(Derivative(f(x, y), y), y, x**2) +
        Subs(Derivative(f(y, x**2), y), y, x))
    # but Subs is not always necessary
    assert f(x, g(y)).diff(g(y)) == Derivative(f(x, g(y)), g(y))


def test_deriv2():
    assert (x**3).diff(x) == 3*x**2
    assert (x**3).diff(x, evaluate=False) != 3*x**2
    assert (x**3).diff(x, evaluate=False) == Derivative(x**3, x)

    assert diff(x**3, x) == 3*x**2
    assert diff(x**3, x, evaluate=False) != 3*x**2
    assert diff(x**3, x, evaluate=False) == Derivative(x**3, x)


def test_func_deriv():
    assert f(x).diff(x) == Derivative(f(x), x)
    # issue 4534
    assert f(x, y).diff(x, y) - f(x, y).diff(y, x) == 0
    assert Derivative(f(x, y), x, y).args[1:] == ((x, 1), (y, 1))
    assert Derivative(f(x, y), y, x).args[1:] == ((y, 1), (x, 1))
    assert (Derivative(f(x, y), x, y) - Derivative(f(x, y), y, x)).doit() == 0


def test_suppressed_evaluation():
    a = sin(0, evaluate=False)
    assert a != 0
    assert a.func is sin
    assert a.args == (0,)


def test_function_evalf():
    def eq(a, b, eps):
        return abs(a - b) < eps
    assert eq(sin(1).evalf(15), Float("0.841470984807897"), 1e-13)
    assert eq(
        sin(2).evalf(25), Float("0.9092974268256816953960199", 25), 1e-23)
    assert eq(sin(1 + I).evalf(
        15), Float("1.29845758141598") + Float("0.634963914784736")*I, 1e-13)
    assert eq(exp(1 + I).evalf(15), Float(
        "1.46869393991588") + Float("2.28735528717884239")*I, 1e-13)
    assert eq(exp(-0.5 + 1.5*I).evalf(15), Float(
        "0.0429042815937374") + Float("0.605011292285002")*I, 1e-13)
    assert eq(log(pi + sqrt(2)*I).evalf(
        15), Float("1.23699044022052") + Float("0.422985442737893")*I, 1e-13)
    assert eq(cos(100).evalf(15), Float("0.86231887228768"), 1e-13)


def test_extensibility_eval():
    class MyFunc(Function):
        @classmethod
        def eval(cls, *args):
            return (0, 0, 0)
    assert MyFunc(0) == (0, 0, 0)


@_both_exp_pow
def test_function_non_commutative():
    x = Symbol('x', commutative=False)
    assert f(x).is_commutative is False
    assert sin(x).is_commutative is False
    assert exp(x).is_commutative is False
    assert log(x).is_commutative is False
    assert f(x).is_complex is False
    assert sin(x).is_complex is False
    assert exp(x).is_complex is False
    assert log(x).is_complex is False


def test_function_complex():
    x = Symbol('x', complex=True)
    xzf = Symbol('x', complex=True, zero=False)
    assert f(x).is_commutative is True
    assert sin(x).is_commutative is True
    assert exp(x).is_commutative is True
    assert log(x).is_commutative is True
    assert f(x).is_complex is None
    assert sin(x).is_complex is True
    assert exp(x).is_complex is True
    assert log(x).is_complex is None
    assert log(xzf).is_complex is True


def test_function__eval_nseries():
    n = Symbol('n')

    assert sin(x)._eval_nseries(x, 2, None) == x + O(x**2)
    assert sin(x + 1)._eval_nseries(x, 2, None) == x*cos(1) + sin(1) + O(x**2)
    assert sin(pi*(1 - x))._eval_nseries(x, 2, None) == pi*x + O(x**2)
    assert acos(1 - x**2)._eval_nseries(x, 2, None) == sqrt(2)*sqrt(x**2) + O(x**2)
    assert polygamma(n, x + 1)._eval_nseries(x, 2, None) == \
        polygamma(n, 1) + polygamma(n + 1, 1)*x + O(x**2)
    raises(PoleError, lambda: sin(1/x)._eval_nseries(x, 2, None))
    assert acos(1 - x)._eval_nseries(x, 2, None) == sqrt(2)*sqrt(x) + sqrt(2)*x**(S(3)/2)/12 + O(x**2)
    assert acos(1 + x)._eval_nseries(x, 2, None) == sqrt(2)*sqrt(-x) + sqrt(2)*(-x)**(S(3)/2)/12 + O(x**2)
    assert loggamma(1/x)._eval_nseries(x, 0, None) == \
        log(x)/2 - log(x)/x - 1/x + O(1, x)
    assert loggamma(log(1/x)).nseries(x, n=1, logx=y) == loggamma(-y)

    # issue 6725:
    assert expint(Rational(3, 2), -x)._eval_nseries(x, 5, None) == \
        2 - 2*x - x**2/3 - x**3/15 - x**4/84 - 2*I*sqrt(pi)*sqrt(x) + O(x**5)
    assert sin(sqrt(x))._eval_nseries(x, 3, None) == \
        sqrt(x) - x**Rational(3, 2)/6 + x**Rational(5, 2)/120 + O(x**3)

    # issue 19065:
    s1 = f(x,y).series(y, n=2)
    assert {i.name for i in s1.atoms(Symbol)} == {'x', 'xi', 'y'}
    xi = Symbol('xi')
    s2 = f(xi, y).series(y, n=2)
    assert {i.name for i in s2.atoms(Symbol)} == {'xi', 'xi0', 'y'}

def test_doit():
    n = Symbol('n', integer=True)
    f = Sum(2 * n * x, (n, 1, 3))
    d = Derivative(f, x)
    assert d.doit() == 12
    assert d.doit(deep=False) == Sum(2*n, (n, 1, 3))


def test_evalf_default():
    from sympy.functions.special.gamma_functions import polygamma
    assert type(sin(4.0)) == Float
    assert type(re(sin(I + 1.0))) == Float
    assert type(im(sin(I + 1.0))) == Float
    assert type(sin(4)) == sin
    assert type(polygamma(2.0, 4.0)) == Float
    assert type(sin(Rational(1, 4))) == sin


def test_issue_5399():
    args = [x, y, S(2), S.Half]

    def ok(a):
        """Return True if the input args for diff are ok"""
        if not a:
            return False
        if a[0].is_Symbol is False:
            return False
        s_at = [i for i in range(len(a)) if a[i].is_Symbol]
        n_at = [i for i in range(len(a)) if not a[i].is_Symbol]
        # every symbol is followed by symbol or int
        # every number is followed by a symbol
        return (all(a[i + 1].is_Symbol or a[i + 1].is_Integer
            for i in s_at if i + 1 < len(a)) and
            all(a[i + 1].is_Symbol
            for i in n_at if i + 1 < len(a)))
    eq = x**10*y**8
    for a in subsets(args):
        for v in variations(a, len(a)):
            if ok(v):
                eq.diff(*v) # does not raise
            else:
                raises(ValueError, lambda: eq.diff(*v))


def test_derivative_numerically():
    z0 = x._random()
    assert abs(Derivative(sin(x), x).doit_numerically(z0) - cos(z0)) < 1e-15


def test_fdiff_argument_index_error():
    from sympy.core.function import ArgumentIndexError

    class myfunc(Function):
        nargs = 1  # define since there is no eval routine

        def fdiff(self, idx):
            raise ArgumentIndexError
    mf = myfunc(x)
    assert mf.diff(x) == Derivative(mf, x)
    raises(TypeError, lambda: myfunc(x, x))


def test_deriv_wrt_function():
    x = f(t)
    xd = diff(x, t)
    xdd = diff(xd, t)
    y = g(t)
    yd = diff(y, t)

    assert diff(x, t) == xd
    assert diff(2 * x + 4, t) == 2 * xd
    assert diff(2 * x + 4 + y, t) == 2 * xd + yd
    assert diff(2 * x + 4 + y * x, t) == 2 * xd + x * yd + xd * y
    assert diff(2 * x + 4 + y * x, x) == 2 + y
    assert (diff(4 * x**2 + 3 * x + x * y, t) == 3 * xd + x * yd + xd * y +
            8 * x * xd)
    assert (diff(4 * x**2 + 3 * xd + x * y, t) == 3 * xdd + x * yd + xd * y +
            8 * x * xd)
    assert diff(4 * x**2 + 3 * xd + x * y, xd) == 3
    assert diff(4 * x**2 + 3 * xd + x * y, xdd) == 0
    assert diff(sin(x), t) == xd * cos(x)
    assert diff(exp(x), t) == xd * exp(x)
    assert diff(sqrt(x), t) == xd / (2 * sqrt(x))


def test_diff_wrt_value():
    assert Expr()._diff_wrt is False
    assert x._diff_wrt is True
    assert f(x)._diff_wrt is True
    assert Derivative(f(x), x)._diff_wrt is True
    assert Derivative(x**2, x)._diff_wrt is False


def test_diff_wrt():
    fx = f(x)
    dfx = diff(f(x), x)
    ddfx = diff(f(x), x, x)

    assert diff(sin(fx) + fx**2, fx) == cos(fx) + 2*fx
    assert diff(sin(dfx) + dfx**2, dfx) == cos(dfx) + 2*dfx
    assert diff(sin(ddfx) + ddfx**2, ddfx) == cos(ddfx) + 2*ddfx
    assert diff(fx**2, dfx) == 0
    assert diff(fx**2, ddfx) == 0
    assert diff(dfx**2, fx) == 0
    assert diff(dfx**2, ddfx) == 0
    assert diff(ddfx**2, dfx) == 0

    assert diff(fx*dfx*ddfx, fx) == dfx*ddfx
    assert diff(fx*dfx*ddfx, dfx) == fx*ddfx
    assert diff(fx*dfx*ddfx, ddfx) == fx*dfx

    assert diff(f(x), x).diff(f(x)) == 0
    assert (sin(f(x)) - cos(diff(f(x), x))).diff(f(x)) == cos(f(x))

    assert diff(sin(fx), fx, x) == diff(sin(fx), x, fx)

    # Chain rule cases
    assert f(g(x)).diff(x) == \
        Derivative(g(x), x)*Derivative(f(g(x)), g(x))
    assert diff(f(g(x), h(y)), x) == \
        Derivative(g(x), x)*Derivative(f(g(x), h(y)), g(x))
    assert diff(f(g(x), h(x)), x) == (
        Derivative(f(g(x), h(x)), g(x))*Derivative(g(x), x) +
        Derivative(f(g(x), h(x)), h(x))*Derivative(h(x), x))
    assert f(
        sin(x)).diff(x) == cos(x)*Subs(Derivative(f(x), x), x, sin(x))

    assert diff(f(g(x)), g(x)) == Derivative(f(g(x)), g(x))


def test_diff_wrt_func_subs():
    assert f(g(x)).diff(x).subs(g, Lambda(x, 2*x)).doit() == f(2*x).diff(x)


def test_subs_in_derivative():
    expr = sin(x*exp(y))
    u = Function('u')
    v = Function('v')
    assert Derivative(expr, y).subs(expr, y) == Derivative(y, y)
    assert Derivative(expr, y).subs(y, x).doit() == \
        Derivative(expr, y).doit().subs(y, x)
    assert Derivative(f(x, y), y).subs(y, x) == Subs(Derivative(f(x, y), y), y, x)
    assert Derivative(f(x, y), y).subs(x, y) == Subs(Derivative(f(x, y), y), x, y)
    assert Derivative(f(x, y), y).subs(y, g(x, y)) == Subs(Derivative(f(x, y), y), y, g(x, y)).doit()
    assert Derivative(f(x, y), y).subs(x, g(x, y)) == Subs(Derivative(f(x, y), y), x, g(x, y))
    assert Derivative(f(x, y), g(y)).subs(x, g(x, y)) == Derivative(f(g(x, y), y), g(y))
    assert Derivative(f(u(x), h(y)), h(y)).subs(h(y), g(x, y)) == \
        Subs(Derivative(f(u(x), h(y)), h(y)), h(y), g(x, y)).doit()
    assert Derivative(f(x, y), y).subs(y, z) == Derivative(f(x, z), z)
    assert Derivative(f(x, y), y).subs(y, g(y)) == Derivative(f(x, g(y)), g(y))
    assert Derivative(f(g(x), h(y)), h(y)).subs(h(y), u(y)) == \
        Derivative(f(g(x), u(y)), u(y))
    assert Derivative(f(x, f(x, x)), f(x, x)).subs(
        f, Lambda((x, y), x + y)) == Subs(
        Derivative(z + x, z), z, 2*x)
    assert Subs(Derivative(f(f(x)), x), f, cos).doit() == sin(x)*sin(cos(x))
    assert Subs(Derivative(f(f(x)), f(x)), f, cos).doit() == -sin(cos(x))
    # Issue 13791. No comparison (it's a long formula) but this used to raise an exception.
    assert isinstance(v(x, y, u(x, y)).diff(y).diff(x).diff(y), Expr)
    # This is also related to issues 13791 and 13795; issue 15190
    F = Lambda((x, y), exp(2*x + 3*y))
    abstract = f(x, f(x, x)).diff(x, 2)
    concrete = F(x, F(x, x)).diff(x, 2)
    assert (abstract.subs(f, F).doit() - concrete).simplify() == 0
    # don't introduce a new symbol if not necessary
    assert x in f(x).diff(x).subs(x, 0).atoms()
    # case (4)
    assert Derivative(f(x,f(x,y)), x, y).subs(x, g(y)
        ) == Subs(Derivative(f(x, f(x, y)), x, y), x, g(y))

    assert Derivative(f(x, x), x).subs(x, 0
        ) == Subs(Derivative(f(x, x), x), x, 0)
    # issue 15194
    assert Derivative(f(y, g(x)), (x, z)).subs(z, x
        ) == Derivative(f(y, g(x)), (x, x))

    df = f(x).diff(x)
    assert df.subs(df, 1) is S.One
    assert df.diff(df) is S.One
    dxy = Derivative(f(x, y), x, y)
    dyx = Derivative(f(x, y), y, x)
    assert dxy.subs(Derivative(f(x, y), y, x), 1) is S.One
    assert dxy.diff(dyx) is S.One
    assert Derivative(f(x, y), x, 2, y, 3).subs(
        dyx, g(x, y)) == Derivative(g(x, y), x, 1, y, 2)
    assert Derivative(f(x, x - y), y).subs(x, x + y) == Subs(
        Derivative(f(x, x - y), y), x, x + y)


def test_diff_wrt_not_allowed():
    # issue 7027 included
    for wrt in (
            cos(x), re(x), x**2, x*y, 1 + x,
            Derivative(cos(x), x), Derivative(f(f(x)), x)):
        raises(ValueError, lambda: diff(f(x), wrt))
    # if we don't differentiate wrt then don't raise error
    assert diff(exp(x*y), x*y, 0) == exp(x*y)


def test_diff_wrt_intlike():
    class Two:
        def __int__(self):
            return 2

    assert cos(x).diff(x, Two()) == -cos(x)


def test_klein_gordon_lagrangian():
    m = Symbol('m')
    phi = f(x, t)

    L = -(diff(phi, t)**2 - diff(phi, x)**2 - m**2*phi**2)/2
    eqna = Eq(
        diff(L, phi) - diff(L, diff(phi, x), x) - diff(L, diff(phi, t), t), 0)
    eqnb = Eq(diff(phi, t, t) - diff(phi, x, x) + m**2*phi, 0)
    assert eqna == eqnb


def test_sho_lagrangian():
    m = Symbol('m')
    k = Symbol('k')
    x = f(t)

    L = m*diff(x, t)**2/2 - k*x**2/2
    eqna = Eq(diff(L, x), diff(L, diff(x, t), t))
    eqnb = Eq(-k*x, m*diff(x, t, t))
    assert eqna == eqnb

    assert diff(L, x, t) == diff(L, t, x)
    assert diff(L, diff(x, t), t) == m*diff(x, t, 2)
    assert diff(L, t, diff(x, t)) == -k*x + m*diff(x, t, 2)


def test_straight_line():
    F = f(x)
    Fd = F.diff(x)
    L = sqrt(1 + Fd**2)
    assert diff(L, F) == 0
    assert diff(L, Fd) == Fd/sqrt(1 + Fd**2)


def test_sort_variable():
    vsort = Derivative._sort_variable_count
    def vsort0(*v, reverse=False):
        return [i[0] for i in vsort([(i, 0) for i in (
            reversed(v) if reverse else v)])]

    for R in range(2):
        assert vsort0(y, x, reverse=R) == [x, y]
        assert vsort0(f(x), x, reverse=R) == [x, f(x)]
        assert vsort0(f(y), f(x), reverse=R) == [f(x), f(y)]
        assert vsort0(g(x), f(y), reverse=R) == [f(y), g(x)]
        assert vsort0(f(x, y), f(x), reverse=R) == [f(x), f(x, y)]
        fx = f(x).diff(x)
        assert vsort0(fx, y, reverse=R) == [y, fx]
        fy = f(y).diff(y)
        assert vsort0(fy, fx, reverse=R) == [fx, fy]
        fxx = fx.diff(x)
        assert vsort0(fxx, fx, reverse=R) == [fx, fxx]
        assert vsort0(Basic(x), f(x), reverse=R) == [f(x), Basic(x)]
        assert vsort0(Basic(y), Basic(x), reverse=R) == [Basic(x), Basic(y)]
        assert vsort0(Basic(y, z), Basic(x), reverse=R) == [
            Basic(x), Basic(y, z)]
        assert vsort0(fx, x, reverse=R) == [
            x, fx] if R else [fx, x]
        assert vsort0(Basic(x), x, reverse=R) == [
            x, Basic(x)] if R else [Basic(x), x]
        assert vsort0(Basic(f(x)), f(x), reverse=R) == [
            f(x), Basic(f(x))] if R else [Basic(f(x)), f(x)]
        assert vsort0(Basic(x, z), Basic(x), reverse=R) == [
            Basic(x), Basic(x, z)] if R else [Basic(x, z), Basic(x)]
    assert vsort([]) == []
    assert _aresame(vsort([(x, 1)]), [Tuple(x, 1)])
    assert vsort([(x, y), (x, z)]) == [(x, y + z)]
    assert vsort([(y, 1), (x, 1 + y)]) == [(x, 1 + y), (y, 1)]
    # coverage complete; legacy tests below
    assert vsort([(x, 3), (y, 2), (z, 1)]) == [(x, 3), (y, 2), (z, 1)]
    assert vsort([(h(x), 1), (g(x), 1), (f(x), 1)]) == [
        (f(x), 1), (g(x), 1), (h(x), 1)]
    assert vsort([(z, 1), (y, 2), (x, 3), (h(x), 1), (g(x), 1),
        (f(x), 1)]) == [(x, 3), (y, 2), (z, 1), (f(x), 1), (g(x), 1),
        (h(x), 1)]
    assert vsort([(x, 1), (f(x), 1), (y, 1), (f(y), 1)]) == [(x, 1),
        (y, 1), (f(x), 1), (f(y), 1)]
    assert vsort([(y, 1), (x, 2), (g(x), 1), (f(x), 1), (z, 1),
        (h(x), 1), (y, 2), (x, 1)]) == [(x, 3), (y, 3), (z, 1),
        (f(x), 1), (g(x), 1), (h(x), 1)]
    assert vsort([(z, 1), (y, 1), (f(x), 1), (x, 1), (f(x), 1),
        (g(x), 1)]) == [(x, 1), (y, 1), (z, 1), (f(x), 2), (g(x), 1)]
    assert vsort([(z, 1), (y, 2), (f(x), 1), (x, 2), (f(x), 2),
        (g(x), 1), (z, 2), (z, 1), (y, 1), (x, 1)]) == [(x, 3), (y, 3),
        (z, 4), (f(x), 3), (g(x), 1)]
    assert vsort(((y, 2), (x, 1), (y, 1), (x, 1))) == [(x, 2), (y, 3)]
    assert isinstance(vsort([(x, 3), (y, 2), (z, 1)])[0], Tuple)
    assert vsort([(x, 1), (f(x), 1), (x, 1)]) == [(x, 2), (f(x), 1)]
    assert vsort([(y, 2), (x, 3), (z, 1)]) == [(x, 3), (y, 2), (z, 1)]
    assert vsort([(h(y), 1), (g(x), 1), (f(x), 1)]) == [
        (f(x), 1), (g(x), 1), (h(y), 1)]
    assert vsort([(x, 1), (y, 1), (x, 1)]) == [(x, 2), (y, 1)]
    assert vsort([(f(x), 1), (f(y), 1), (f(x), 1)]) == [
        (f(x), 2), (f(y), 1)]
    dfx = f(x).diff(x)
    self = [(dfx, 1), (x, 1)]
    assert vsort(self) == self
    assert vsort([
        (dfx, 1), (y, 1), (f(x), 1), (x, 1), (f(y), 1), (x, 1)]) == [
        (y, 1), (f(x), 1), (f(y), 1), (dfx, 1), (x, 2)]
    dfy = f(y).diff(y)
    assert vsort([(dfy, 1), (dfx, 1)]) == [(dfx, 1), (dfy, 1)]
    d2fx = dfx.diff(x)
    assert vsort([(d2fx, 1), (dfx, 1)]) == [(dfx, 1), (d2fx, 1)]


def test_multiple_derivative():
    # Issue #15007
    assert f(x, y).diff(y, y, x, y, x
        ) == Derivative(f(x, y), (x, 2), (y, 3))


def test_unhandled():
    class MyExpr(Expr):
        def _eval_derivative(self, s):
            if not s.name.startswith('xi'):
                return self
            else:
                return None

    eq = MyExpr(f(x), y, z)
    assert diff(eq, x, y, f(x), z) == Derivative(eq, f(x))
    assert diff(eq, f(x), x) == Derivative(eq, f(x))
    assert f(x, y).diff(x,(y, z)) == Derivative(f(x, y), x, (y, z))
    assert f(x, y).diff(x,(y, 0)) == Derivative(f(x, y), x)


def test_nfloat():
    from sympy.core.basic import _aresame
    from sympy.polys.rootoftools import rootof

    x = Symbol("x")
    eq = x**Rational(4, 3) + 4*x**(S.One/3)/3
    assert _aresame(nfloat(eq), x**Rational(4, 3) + (4.0/3)*x**(S.One/3))
    assert _aresame(nfloat(eq, exponent=True), x**(4.0/3) + (4.0/3)*x**(1.0/3))
    eq = x**Rational(4, 3) + 4*x**(x/3)/3
    assert _aresame(nfloat(eq), x**Rational(4, 3) + (4.0/3)*x**(x/3))
    big = 12345678901234567890
    # specify precision to match value used in nfloat
    Float_big = Float(big, 15)
    assert _aresame(nfloat(big), Float_big)
    assert _aresame(nfloat(big*x), Float_big*x)
    assert _aresame(nfloat(x**big, exponent=True), x**Float_big)
    assert nfloat(cos(x + sqrt(2))) == cos(x + nfloat(sqrt(2)))

    # issue 6342
    f = S('x*lamda + lamda**3*(x/2 + 1/2) + lamda**2 + 1/4')
    assert not any(a.free_symbols for a in solveset(f.subs(x, -0.139)))

    # issue 6632
    assert nfloat(-100000*sqrt(2500000001) + 5000000001) == \
        9.99999999800000e-11

    # issue 7122
    eq = cos(3*x**4 + y)*rootof(x**5 + 3*x**3 + 1, 0)
    assert str(nfloat(eq, exponent=False, n=1)) == '-0.7*cos(3.0*x**4 + y)'

    # issue 10933
    for ti in (dict, Dict):
        d = ti({S.Half: S.Half})
        n = nfloat(d)
        assert isinstance(n, ti)
        assert _aresame(list(n.items()).pop(), (S.Half, Float(.5)))
    for ti in (dict, Dict):
        d = ti({S.Half: S.Half})
        n = nfloat(d, dkeys=True)
        assert isinstance(n, ti)
        assert _aresame(list(n.items()).pop(), (Float(.5), Float(.5)))
    d = [S.Half]
    n = nfloat(d)
    assert type(n) is list
    assert _aresame(n[0], Float(.5))
    assert _aresame(nfloat(Eq(x, S.Half)).rhs, Float(.5))
    assert _aresame(nfloat(S(True)), S(True))
    assert _aresame(nfloat(Tuple(S.Half))[0], Float(.5))
    assert nfloat(Eq((3 - I)**2/2 + I, 0)) == S.false
    # pass along kwargs
    assert nfloat([{S.Half: x}], dkeys=True) == [{Float(0.5): x}]

    # Issue 17706
    A = MutableMatrix([[1, 2], [3, 4]])
    B = MutableMatrix(
        [[Float('1.0', precision=53), Float('2.0', precision=53)],
        [Float('3.0', precision=53), Float('4.0', precision=53)]])
    assert _aresame(nfloat(A), B)
    A = ImmutableMatrix([[1, 2], [3, 4]])
    B = ImmutableMatrix(
        [[Float('1.0', precision=53), Float('2.0', precision=53)],
        [Float('3.0', precision=53), Float('4.0', precision=53)]])
    assert _aresame(nfloat(A), B)

    # issue 22524
    f = Function('f')
    assert not nfloat(f(2)).atoms(Float)


def test_issue_7068():
    from sympy.abc import a, b
    f = Function('f')
    y1 = Dummy('y')
    y2 = Dummy('y')
    func1 = f(a + y1 * b)
    func2 = f(a + y2 * b)
    func1_y = func1.diff(y1)
    func2_y = func2.diff(y2)
    assert func1_y != func2_y
    z1 = Subs(f(a), a, y1)
    z2 = Subs(f(a), a, y2)
    assert z1 != z2


def test_issue_7231():
    from sympy.abc import a
    ans1 = f(x).series(x, a)
    res = (f(a) + (-a + x)*Subs(Derivative(f(y), y), y, a) +
           (-a + x)**2*Subs(Derivative(f(y), y, y), y, a)/2 +
           (-a + x)**3*Subs(Derivative(f(y), y, y, y),
                            y, a)/6 +
           (-a + x)**4*Subs(Derivative(f(y), y, y, y, y),
                            y, a)/24 +
           (-a + x)**5*Subs(Derivative(f(y), y, y, y, y, y),
                            y, a)/120 + O((-a + x)**6, (x, a)))
    assert res == ans1
    ans2 = f(x).series(x, a)
    assert res == ans2


def test_issue_7687():
    from sympy.core.function import Function
    from sympy.abc import x
    f = Function('f')(x)
    ff = Function('f')(x)
    match_with_cache = ff.matches(f)
    assert isinstance(f, type(ff))
    clear_cache()
    ff = Function('f')(x)
    assert isinstance(f, type(ff))
    assert match_with_cache == ff.matches(f)


def test_issue_7688():
    from sympy.core.function import Function, UndefinedFunction

    f = Function('f')  # actually an UndefinedFunction
    clear_cache()
    class A(UndefinedFunction):
        pass
    a = A('f')
    assert isinstance(a, type(f))


def test_mexpand():
    from sympy.abc import x
    assert _mexpand(None) is None
    assert _mexpand(1) is S.One
    assert _mexpand(x*(x + 1)**2) == (x*(x + 1)**2).expand()


def test_issue_8469():
    # This should not take forever to run
    N = 40
    def g(w, theta):
        return 1/(1+exp(w-theta))

    ws = symbols(['w%i'%i for i in range(N)])
    import functools
    expr = functools.reduce(g, ws)
    assert isinstance(expr, Pow)


def test_issue_12996():
    # foo=True imitates the sort of arguments that Derivative can get
    # from Integral when it passes doit to the expression
    assert Derivative(im(x), x).doit(foo=True) == Derivative(im(x), x)


def test_should_evalf():
    # This should not take forever to run (see #8506)
    assert isinstance(sin((1.0 + 1.0*I)**10000 + 1), sin)


def test_Derivative_as_finite_difference():
    # Central 1st derivative at gridpoint
    x, h = symbols('x h', real=True)
    dfdx = f(x).diff(x)
    assert (dfdx.as_finite_difference([x-2, x-1, x, x+1, x+2]) -
            (S.One/12*(f(x-2)-f(x+2)) + Rational(2, 3)*(f(x+1)-f(x-1)))).simplify() == 0

    # Central 1st derivative "half-way"
    assert (dfdx.as_finite_difference() -
            (f(x + S.Half)-f(x - S.Half))).simplify() == 0
    assert (dfdx.as_finite_difference(h) -
            (f(x + h/S(2))-f(x - h/S(2)))/h).simplify() == 0
    assert (dfdx.as_finite_difference([x - 3*h, x-h, x+h, x + 3*h]) -
            (S(9)/(8*2*h)*(f(x+h) - f(x-h)) +
             S.One/(24*2*h)*(f(x - 3*h) - f(x + 3*h)))).simplify() == 0

    # One sided 1st derivative at gridpoint
    assert (dfdx.as_finite_difference([0, 1, 2], 0) -
            (Rational(-3, 2)*f(0) + 2*f(1) - f(2)/2)).simplify() == 0
    assert (dfdx.as_finite_difference([x, x+h], x) -
            (f(x+h) - f(x))/h).simplify() == 0
    assert (dfdx.as_finite_difference([x-h, x, x+h], x-h) -
            (-S(3)/(2*h)*f(x-h) + 2/h*f(x) -
             S.One/(2*h)*f(x+h))).simplify() == 0

    # One sided 1st derivative "half-way"
    assert (dfdx.as_finite_difference([x-h, x+h, x + 3*h, x + 5*h, x + 7*h])
            - 1/(2*h)*(-S(11)/(12)*f(x-h) + S(17)/(24)*f(x+h)
                       + Rational(3, 8)*f(x + 3*h) - Rational(5, 24)*f(x + 5*h)
                       + S.One/24*f(x + 7*h))).simplify() == 0

    d2fdx2 = f(x).diff(x, 2)
    # Central 2nd derivative at gridpoint
    assert (d2fdx2.as_finite_difference([x-h, x, x+h]) -
            h**-2 * (f(x-h) + f(x+h) - 2*f(x))).simplify() == 0

    assert (d2fdx2.as_finite_difference([x - 2*h, x-h, x, x+h, x + 2*h]) -
            h**-2 * (Rational(-1, 12)*(f(x - 2*h) + f(x + 2*h)) +
                     Rational(4, 3)*(f(x+h) + f(x-h)) - Rational(5, 2)*f(x))).simplify() == 0

    # Central 2nd derivative "half-way"
    assert (d2fdx2.as_finite_difference([x - 3*h, x-h, x+h, x + 3*h]) -
            (2*h)**-2 * (S.Half*(f(x - 3*h) + f(x + 3*h)) -
                         S.Half*(f(x+h) + f(x-h)))).simplify() == 0

    # One sided 2nd derivative at gridpoint
    assert (d2fdx2.as_finite_difference([x, x+h, x + 2*h, x + 3*h]) -
            h**-2 * (2*f(x) - 5*f(x+h) +
                     4*f(x+2*h) - f(x+3*h))).simplify() == 0

    # One sided 2nd derivative at "half-way"
    assert (d2fdx2.as_finite_difference([x-h, x+h, x + 3*h, x + 5*h]) -
            (2*h)**-2 * (Rational(3, 2)*f(x-h) - Rational(7, 2)*f(x+h) + Rational(5, 2)*f(x + 3*h) -
                         S.Half*f(x + 5*h))).simplify() == 0

    d3fdx3 = f(x).diff(x, 3)
    # Central 3rd derivative at gridpoint
    assert (d3fdx3.as_finite_difference() -
            (-f(x - Rational(3, 2)) + 3*f(x - S.Half) -
             3*f(x + S.Half) + f(x + Rational(3, 2)))).simplify() == 0

    assert (d3fdx3.as_finite_difference(
        [x - 3*h, x - 2*h, x-h, x, x+h, x + 2*h, x + 3*h]) -
        h**-3 * (S.One/8*(f(x - 3*h) - f(x + 3*h)) - f(x - 2*h) +
                 f(x + 2*h) + Rational(13, 8)*(f(x-h) - f(x+h)))).simplify() == 0

    # Central 3rd derivative at "half-way"
    assert (d3fdx3.as_finite_difference([x - 3*h, x-h, x+h, x + 3*h]) -
            (2*h)**-3 * (f(x + 3*h)-f(x - 3*h) +
                         3*(f(x-h)-f(x+h)))).simplify() == 0

    # One sided 3rd derivative at gridpoint
    assert (d3fdx3.as_finite_difference([x, x+h, x + 2*h, x + 3*h]) -
            h**-3 * (f(x + 3*h)-f(x) + 3*(f(x+h)-f(x + 2*h)))).simplify() == 0

    # One sided 3rd derivative at "half-way"
    assert (d3fdx3.as_finite_difference([x-h, x+h, x + 3*h, x + 5*h]) -
            (2*h)**-3 * (f(x + 5*h)-f(x-h) +
                         3*(f(x+h)-f(x + 3*h)))).simplify() == 0

    # issue 11007
    y = Symbol('y', real=True)
    d2fdxdy = f(x, y).diff(x, y)

    ref0 = Derivative(f(x + S.Half, y), y) - Derivative(f(x - S.Half, y), y)
    assert (d2fdxdy.as_finite_difference(wrt=x) - ref0).simplify() == 0

    half = S.Half
    xm, xp, ym, yp = x-half, x+half, y-half, y+half
    ref2 = f(xm, ym) + f(xp, yp) - f(xp, ym) - f(xm, yp)
    assert (d2fdxdy.as_finite_difference() - ref2).simplify() == 0


def test_issue_11159():
    # Tests Application._eval_subs
    with _exp_is_pow(False):
        expr1 = E
        expr0 = expr1 * expr1
        expr1 = expr0.subs(expr1,expr0)
        assert expr0 == expr1
    with _exp_is_pow(True):
        expr1 = E
        expr0 = expr1 * expr1
        expr2 = expr0.subs(expr1, expr0)
        assert expr2 == E ** 4


def test_issue_12005():
    e1 = Subs(Derivative(f(x), x), x, x)
    assert e1.diff(x) == Derivative(f(x), x, x)
    e2 = Subs(Derivative(f(x), x), x, x**2 + 1)
    assert e2.diff(x) == 2*x*Subs(Derivative(f(x), x, x), x, x**2 + 1)
    e3 = Subs(Derivative(f(x) + y**2 - y, y), y, y**2)
    assert e3.diff(y) == 4*y
    e4 = Subs(Derivative(f(x + y), y), y, (x**2))
    assert e4.diff(y) is S.Zero
    e5 = Subs(Derivative(f(x), x), (y, z), (y, z))
    assert e5.diff(x) == Derivative(f(x), x, x)
    assert f(g(x)).diff(g(x), g(x)) == Derivative(f(g(x)), g(x), g(x))


def test_issue_13843():
    x = symbols('x')
    f = Function('f')
    m, n = symbols('m n', integer=True)
    assert Derivative(Derivative(f(x), (x, m)), (x, n)) == Derivative(f(x), (x, m + n))
    assert Derivative(Derivative(f(x), (x, m+5)), (x, n+3)) == Derivative(f(x), (x, m + n + 8))

    assert Derivative(f(x), (x, n)).doit() == Derivative(f(x), (x, n))


def test_order_could_be_zero():
    x, y = symbols('x, y')
    n = symbols('n', integer=True, nonnegative=True)
    m = symbols('m', integer=True, positive=True)
    assert diff(y, (x, n)) == Piecewise((y, Eq(n, 0)), (0, True))
    assert diff(y, (x, n + 1)) is S.Zero
    assert diff(y, (x, m)) is S.Zero


def test_undefined_function_eq():
    f = Function('f')
    f2 = Function('f')
    g = Function('g')
    f_real = Function('f', is_real=True)

    # This test may only be meaningful if the cache is turned off
    assert f == f2
    assert hash(f) == hash(f2)
    assert f == f

    assert f != g

    assert f != f_real


def test_function_assumptions():
    x = Symbol('x')
    f = Function('f')
    f_real = Function('f', real=True)
    f_real1 = Function('f', real=1)
    f_real_inherit = Function(Symbol('f', real=True))

    assert f_real == f_real1  # assumptions are sanitized
    assert f != f_real
    assert f(x) != f_real(x)

    assert f(x).is_real is None
    assert f_real(x).is_real is True
    assert f_real_inherit(x).is_real is True and f_real_inherit.name == 'f'

    # Can also do it this way, but it won't be equal to f_real because of the
    # way UndefinedFunction.__new__ works. Any non-recognized assumptions
    # are just added literally as something which is used in the hash
    f_real2 = Function('f', is_real=True)
    assert f_real2(x).is_real is True


def test_undef_fcn_float_issue_6938():
    f = Function('ceil')
    assert not f(0.3).is_number
    f = Function('sin')
    assert not f(0.3).is_number
    assert not f(pi).evalf().is_number
    x = Symbol('x')
    assert not f(x).evalf(subs={x:1.2}).is_number


def test_undefined_function_eval():
    # Issue 15170. Make sure UndefinedFunction with eval defined works
    # properly.

    fdiff = lambda self, argindex=1: cos(self.args[argindex - 1])
    eval = classmethod(lambda cls, t: None)
    _imp_ = classmethod(lambda cls, t: sin(t))

    temp = Function('temp', fdiff=fdiff, eval=eval, _imp_=_imp_)

    expr = temp(t)
    assert sympify(expr) == expr
    assert type(sympify(expr)).fdiff.__name__ == "<lambda>"
    assert expr.diff(t) == cos(t)


def test_issue_15241():
    F = f(x)
    Fx = F.diff(x)
    assert (F + x*Fx).diff(x, Fx) == 2
    assert (F + x*Fx).diff(Fx, x) == 1
    assert (x*F + x*Fx*F).diff(F, x) == x*Fx.diff(x) + Fx + 1
    assert (x*F + x*Fx*F).diff(x, F) == x*Fx.diff(x) + Fx + 1
    y = f(x)
    G = f(y)
    Gy = G.diff(y)
    assert (G + y*Gy).diff(y, Gy) == 2
    assert (G + y*Gy).diff(Gy, y) == 1
    assert (y*G + y*Gy*G).diff(G, y) == y*Gy.diff(y) + Gy + 1
    assert (y*G + y*Gy*G).diff(y, G) == y*Gy.diff(y) + Gy + 1


def test_issue_15226():
    assert Subs(Derivative(f(y), x, y), y, g(x)).doit() != 0


def test_issue_7027():
    for wrt in (cos(x), re(x), Derivative(cos(x), x)):
        raises(ValueError, lambda: diff(f(x), wrt))


def test_derivative_quick_exit():
    assert f(x).diff(y) == 0
    assert f(x).diff(y, f(x)) == 0
    assert f(x).diff(x, f(y)) == 0
    assert f(f(x)).diff(x, f(x), f(y)) == 0
    assert f(f(x)).diff(x, f(x), y) == 0
    assert f(x).diff(g(x)) == 0
    assert f(x).diff(x, f(x).diff(x)) == 1
    df = f(x).diff(x)
    assert f(x).diff(df) == 0
    dg = g(x).diff(x)
    assert dg.diff(df).doit() == 0


def test_issue_15084_13166():
    eq = f(x, g(x))
    assert eq.diff((g(x), y)) == Derivative(f(x, g(x)), (g(x), y))
    # issue 13166
    assert eq.diff(x, 2).doit() == (
        (Derivative(f(x, g(x)), (g(x), 2))*Derivative(g(x), x) +
        Subs(Derivative(f(x, _xi_2), _xi_2, x), _xi_2, g(x)))*Derivative(g(x),
        x) + Derivative(f(x, g(x)), g(x))*Derivative(g(x), (x, 2)) +
        Derivative(g(x), x)*Subs(Derivative(f(_xi_1, g(x)), _xi_1, g(x)),
        _xi_1, x) + Subs(Derivative(f(_xi_1, g(x)), (_xi_1, 2)), _xi_1, x))
    # issue 6681
    assert diff(f(x, t, g(x, t)), x).doit() == (
        Derivative(f(x, t, g(x, t)), g(x, t))*Derivative(g(x, t), x) +
        Subs(Derivative(f(_xi_1, t, g(x, t)), _xi_1), _xi_1, x))
    # make sure the order doesn't matter when using diff
    assert eq.diff(x, g(x)) == eq.diff(g(x), x)


def test_negative_counts():
    # issue 13873
    raises(ValueError, lambda: sin(x).diff(x, -1))


def test_Derivative__new__():
    raises(TypeError, lambda: f(x).diff((x, 2), 0))
    assert f(x, y).diff([(x, y), 0]) == f(x, y)
    assert f(x, y).diff([(x, y), 1]) == NDimArray([
        Derivative(f(x, y), x), Derivative(f(x, y), y)])
    assert f(x,y).diff(y, (x, z), y, x) == Derivative(
        f(x, y), (x, z + 1), (y, 2))
    assert Matrix([x]).diff(x, 2) == Matrix([0])  # is_zero exit


def test_issue_14719_10150():
    class V(Expr):
        _diff_wrt = True
        is_scalar = False
    assert V().diff(V()) == Derivative(V(), V())
    assert (2*V()).diff(V()) == 2*Derivative(V(), V())
    class X(Expr):
        _diff_wrt = True
    assert X().diff(X()) == 1
    assert (2*X()).diff(X()) == 2


def test_noncommutative_issue_15131():
    x = Symbol('x', commutative=False)
    t = Symbol('t', commutative=False)
    fx = Function('Fx', commutative=False)(x)
    ft = Function('Ft', commutative=False)(t)
    A = Symbol('A', commutative=False)
    eq = fx * A * ft
    eqdt = eq.diff(t)
    assert eqdt.args[-1] == ft.diff(t)


def test_Subs_Derivative():
    a = Derivative(f(g(x), h(x)), g(x), h(x),x)
    b = Derivative(Derivative(f(g(x), h(x)), g(x), h(x)),x)
    c = f(g(x), h(x)).diff(g(x), h(x), x)
    d = f(g(x), h(x)).diff(g(x), h(x)).diff(x)
    e = Derivative(f(g(x), h(x)), x)
    eqs = (a, b, c, d, e)
    subs = lambda arg: arg.subs(f, Lambda((x, y), exp(x + y))
        ).subs(g(x), 1/x).subs(h(x), x**3)
    ans = 3*x**2*exp(1/x)*exp(x**3) - exp(1/x)*exp(x**3)/x**2
    assert all(subs(i).doit().expand() == ans for i in eqs)
    assert all(subs(i.doit()).doit().expand() == ans for i in eqs)

def test_issue_15360():
    f = Function('f')
    assert f.name == 'f'


def test_issue_15947():
    assert f._diff_wrt is False
    raises(TypeError, lambda: f(f))
    raises(TypeError, lambda: f(x).diff(f))


def test_Derivative_free_symbols():
    f = Function('f')
    n = Symbol('n', integer=True, positive=True)
    assert diff(f(x), (x, n)).free_symbols == {n, x}


def test_issue_20683():
    x = Symbol('x')
    y = Symbol('y')
    z = Symbol('z')
    y = Derivative(z, x).subs(x,0)
    assert y.doit() == 0
    y = Derivative(8, x).subs(x,0)
    assert y.doit() == 0


def test_issue_10503():
    f = exp(x**3)*cos(x**6)
    assert f.series(x, 0, 14) == 1 + x**3 + x**6/2 + x**9/6 - 11*x**12/24 + O(x**14)


def test_issue_17382():
    # copied from sympy/core/tests/test_evalf.py
    def NS(e, n=15, **options):
        return sstr(sympify(e).evalf(n, **options), full_prec=True)

    x = Symbol('x')
    expr = solveset(2 * cos(x) * cos(2 * x) - 1, x, S.Reals)
    expected = "Union(" \
               "ImageSet(Lambda(_n, 6.28318530717959*_n + 5.79812359592087), Integers), " \
               "ImageSet(Lambda(_n, 6.28318530717959*_n + 0.485061711258717), Integers))"
    assert NS(expr) == expected

def test_eval_sympified():
    # Check both arguments and return types from eval are sympified

    class F(Function):
        @classmethod
        def eval(cls, x):
            assert x is S.One
            return 1

    assert F(1) is S.One

    # String arguments are not allowed
    class F2(Function):
        @classmethod
        def eval(cls, x):
            if x == 0:
                return '1'

    raises(SympifyError, lambda: F2(0))
    F2(1) # Doesn't raise

    # TODO: Disable string inputs (https://github.com/sympy/sympy/issues/11003)
    # raises(SympifyError, lambda: F2('2'))

def test_eval_classmethod_check():
    with raises(TypeError):
        class F(Function):
            def eval(self, x):
                pass


def test_issue_27163():
    # https://github.com/sympy/sympy/issues/27163
    raises(TypeError, lambda: Derivative(f, t))
