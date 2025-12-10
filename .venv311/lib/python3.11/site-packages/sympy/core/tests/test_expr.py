from sympy.assumptions.refine import refine
from sympy.concrete.summations import Sum
from sympy.core.add import Add
from sympy.core.basic import Basic
from sympy.core.containers import Tuple
from sympy.core.expr import (ExprBuilder, unchanged, Expr,
    UnevaluatedExpr)
from sympy.core.function import (Function, expand, WildFunction,
    AppliedUndef, Derivative, diff, Subs)
from sympy.core.mul import Mul, _unevaluated_Mul
from sympy.core.numbers import (NumberSymbol, E, zoo, oo, Float, I,
    Rational, nan, Integer, Number, pi, _illegal)
from sympy.core.power import Pow
from sympy.core.relational import Ge, Lt, Gt, Le
from sympy.core.singleton import S
from sympy.core.sorting import default_sort_key
from sympy.core.symbol import Symbol, symbols, Dummy, Wild
from sympy.core.sympify import sympify
from sympy.functions.combinatorial.factorials import factorial
from sympy.functions.elementary.exponential import exp_polar, exp, log
from sympy.functions.elementary.hyperbolic import sinh, tanh
from sympy.functions.elementary.miscellaneous import sqrt, Max
from sympy.functions.elementary.piecewise import Piecewise
from sympy.functions.elementary.trigonometric import tan, sin, cos
from sympy.functions.special.delta_functions import (Heaviside,
    DiracDelta)
from sympy.functions.special.error_functions import Si
from sympy.functions.special.gamma_functions import gamma
from sympy.integrals.integrals import integrate, Integral
from sympy.physics.secondquant import FockState
from sympy.polys.partfrac import apart
from sympy.polys.polytools import factor, cancel, Poly
from sympy.polys.rationaltools import together
from sympy.series.order import O
from sympy.sets.sets import FiniteSet
from sympy.simplify.combsimp import combsimp
from sympy.simplify.gammasimp import gammasimp
from sympy.simplify.powsimp import powsimp
from sympy.simplify.radsimp import collect, radsimp
from sympy.simplify.ratsimp import ratsimp
from sympy.simplify.simplify import simplify, nsimplify
from sympy.simplify.trigsimp import trigsimp
from sympy.tensor.indexed import Indexed
from sympy.physics.units import meter

from sympy.testing.pytest import raises, XFAIL

from sympy.abc import a, b, c, n, t, u, x, y, z


f, g, h = symbols('f,g,h', cls=Function)


class DummyNumber:
    """
    Minimal implementation of a number that works with SymPy.

    If one has a Number class (e.g. Sage Integer, or some other custom class)
    that one wants to work well with SymPy, one has to implement at least the
    methods of this class DummyNumber, resp. its subclasses I5 and F1_1.

    Basically, one just needs to implement either __int__() or __float__() and
    then one needs to make sure that the class works with Python integers and
    with itself.
    """

    def __radd__(self, a):
        if isinstance(a, (int, float)):
            return a + self.number
        return NotImplemented

    def __add__(self, a):
        if isinstance(a, (int, float, DummyNumber)):
            return self.number + a
        return NotImplemented

    def __rsub__(self, a):
        if isinstance(a, (int, float)):
            return a - self.number
        return NotImplemented

    def __sub__(self, a):
        if isinstance(a, (int, float, DummyNumber)):
            return self.number - a
        return NotImplemented

    def __rmul__(self, a):
        if isinstance(a, (int, float)):
            return a * self.number
        return NotImplemented

    def __mul__(self, a):
        if isinstance(a, (int, float, DummyNumber)):
            return self.number * a
        return NotImplemented

    def __rtruediv__(self, a):
        if isinstance(a, (int, float)):
            return a / self.number
        return NotImplemented

    def __truediv__(self, a):
        if isinstance(a, (int, float, DummyNumber)):
            return self.number / a
        return NotImplemented

    def __rpow__(self, a):
        if isinstance(a, (int, float)):
            return a ** self.number
        return NotImplemented

    def __pow__(self, a):
        if isinstance(a, (int, float, DummyNumber)):
            return self.number ** a
        return NotImplemented

    def __pos__(self):
        return self.number

    def __neg__(self):
        return - self.number


class I5(DummyNumber):
    number = 5

    def __int__(self):
        return self.number


class F1_1(DummyNumber):
    number = 1.1

    def __float__(self):
        return self.number

i5 = I5()
f1_1 = F1_1()

# basic SymPy objects
basic_objs = [
    Rational(2),
    Float("1.3"),
    x,
    y,
    pow(x, y)*y,
]

# all supported objects
all_objs = basic_objs + [
    5,
    5.5,
    i5,
    f1_1
]


def dotest(s):
    for xo in all_objs:
        for yo in all_objs:
            s(xo, yo)
    return True


def test_basic():
    def j(a, b):
        x = a
        x = +a
        x = -a
        x = a + b
        x = a - b
        x = a*b
        x = a/b
        x = a**b
        del x
    assert dotest(j)


def test_ibasic():
    def s(a, b):
        x = a
        x += b
        x = a
        x -= b
        x = a
        x *= b
        x = a
        x /= b
    assert dotest(s)


class NonBasic:
    '''This class represents an object that knows how to implement binary
    operations like +, -, etc with Expr but is not a subclass of Basic itself.
    The NonExpr subclass below does subclass Basic but not Expr.

    For both NonBasic and NonExpr it should be possible for them to override
    Expr.__add__ etc because Expr.__add__ should be returning NotImplemented
    for non Expr classes. Otherwise Expr.__add__ would create meaningless
    objects like Add(Integer(1), FiniteSet(2)) and it wouldn't be possible for
    other classes to override these operations when interacting with Expr.
    '''
    def __add__(self, other):
        return SpecialOp('+', self, other)

    def __radd__(self, other):
        return SpecialOp('+', other, self)

    def __sub__(self, other):
        return SpecialOp('-', self, other)

    def __rsub__(self, other):
        return SpecialOp('-', other, self)

    def __mul__(self, other):
        return SpecialOp('*', self, other)

    def __rmul__(self, other):
        return SpecialOp('*', other, self)

    def __truediv__(self, other):
        return SpecialOp('/', self, other)

    def __rtruediv__(self, other):
        return SpecialOp('/', other, self)

    def __floordiv__(self, other):
        return SpecialOp('//', self, other)

    def __rfloordiv__(self, other):
        return SpecialOp('//', other, self)

    def __mod__(self, other):
        return SpecialOp('%', self, other)

    def __rmod__(self, other):
        return SpecialOp('%', other, self)

    def __divmod__(self, other):
        return SpecialOp('divmod', self, other)

    def __rdivmod__(self, other):
        return SpecialOp('divmod', other, self)

    def __pow__(self, other):
        return SpecialOp('**', self, other)

    def __rpow__(self, other):
        return SpecialOp('**', other, self)

    def __lt__(self, other):
        return SpecialOp('<', self, other)

    def __gt__(self, other):
        return SpecialOp('>', self, other)

    def __le__(self, other):
        return SpecialOp('<=', self, other)

    def __ge__(self, other):
        return SpecialOp('>=', self, other)


class NonExpr(Basic, NonBasic):
    '''Like NonBasic above except this is a subclass of Basic but not Expr'''
    pass


class SpecialOp():
    '''Represents the results of operations with NonBasic and NonExpr'''
    def __new__(cls, op, arg1, arg2):
        obj = object.__new__(cls)
        obj.args = (op, arg1, arg2)
        return obj


class NonArithmetic(Basic):
    '''Represents a Basic subclass that does not support arithmetic operations'''
    pass


def test_cooperative_operations():
    '''Tests that Expr uses binary operations cooperatively.

    In particular it should be possible for non-Expr classes to override
    binary operators like +, - etc when used with Expr instances. This should
    work for non-Expr classes whether they are Basic subclasses or not. Also
    non-Expr classes that do not define binary operators with Expr should give
    TypeError.
    '''
    # A bunch of instances of Expr subclasses
    exprs = [
        Expr(),
        S.Zero,
        S.One,
        S.Infinity,
        S.NegativeInfinity,
        S.ComplexInfinity,
        S.Half,
        Float(0.5),
        Integer(2),
        Symbol('x'),
        Mul(2, Symbol('x')),
        Add(2, Symbol('x')),
        Pow(2, Symbol('x')),
    ]

    for e in exprs:
        # Test that these classes can override arithmetic operations in
        # combination with various Expr types.
        for ne in [NonBasic(), NonExpr()]:

            results = [
                (ne + e, ('+', ne, e)),
                (e + ne, ('+', e, ne)),
                (ne - e, ('-', ne, e)),
                (e - ne, ('-', e, ne)),
                (ne * e, ('*', ne, e)),
                (e * ne, ('*', e, ne)),
                (ne / e, ('/', ne, e)),
                (e / ne, ('/', e, ne)),
                (ne // e, ('//', ne, e)),
                (e // ne, ('//', e, ne)),
                (ne % e, ('%', ne, e)),
                (e % ne, ('%', e, ne)),
                (divmod(ne, e), ('divmod', ne, e)),
                (divmod(e, ne), ('divmod', e, ne)),
                (ne ** e, ('**', ne, e)),
                (e ** ne, ('**', e, ne)),
                (e < ne, ('>', ne, e)),
                (ne < e, ('<', ne, e)),
                (e > ne, ('<', ne, e)),
                (ne > e, ('>', ne, e)),
                (e <= ne, ('>=', ne, e)),
                (ne <= e, ('<=', ne, e)),
                (e >= ne, ('<=', ne, e)),
                (ne >= e, ('>=', ne, e)),
            ]

            for res, args in results:
                assert type(res) is SpecialOp and res.args == args

        # These classes do not support binary operators with Expr. Every
        # operation should raise in combination with any of the Expr types.
        for na in [NonArithmetic(), object()]:

            raises(TypeError, lambda : e + na)
            raises(TypeError, lambda : na + e)
            raises(TypeError, lambda : e - na)
            raises(TypeError, lambda : na - e)
            raises(TypeError, lambda : e * na)
            raises(TypeError, lambda : na * e)
            raises(TypeError, lambda : e / na)
            raises(TypeError, lambda : na / e)
            raises(TypeError, lambda : e // na)
            raises(TypeError, lambda : na // e)
            raises(TypeError, lambda : e % na)
            raises(TypeError, lambda : na % e)
            raises(TypeError, lambda : divmod(e, na))
            raises(TypeError, lambda : divmod(na, e))
            raises(TypeError, lambda : e ** na)
            raises(TypeError, lambda : na ** e)
            raises(TypeError, lambda : e > na)
            raises(TypeError, lambda : na > e)
            raises(TypeError, lambda : e < na)
            raises(TypeError, lambda : na < e)
            raises(TypeError, lambda : e >= na)
            raises(TypeError, lambda : na >= e)
            raises(TypeError, lambda : e <= na)
            raises(TypeError, lambda : na <= e)


def test_relational():
    from sympy.core.relational import Lt
    assert (pi < 3) is S.false
    assert (pi <= 3) is S.false
    assert (pi > 3) is S.true
    assert (pi >= 3) is S.true
    assert (-pi < 3) is S.true
    assert (-pi <= 3) is S.true
    assert (-pi > 3) is S.false
    assert (-pi >= 3) is S.false
    r = Symbol('r', real=True)
    assert (r - 2 < r - 3) is S.false
    assert Lt(x + I, x + I + 2).func == Lt  # issue 8288


def test_relational_assumptions():
    m1 = Symbol("m1", nonnegative=False)
    m2 = Symbol("m2", positive=False)
    m3 = Symbol("m3", nonpositive=False)
    m4 = Symbol("m4", negative=False)
    assert (m1 < 0) == Lt(m1, 0)
    assert (m2 <= 0) == Le(m2, 0)
    assert (m3 > 0) == Gt(m3, 0)
    assert (m4 >= 0) == Ge(m4, 0)
    m1 = Symbol("m1", nonnegative=False, real=True)
    m2 = Symbol("m2", positive=False, real=True)
    m3 = Symbol("m3", nonpositive=False, real=True)
    m4 = Symbol("m4", negative=False, real=True)
    assert (m1 < 0) is S.true
    assert (m2 <= 0) is S.true
    assert (m3 > 0) is S.true
    assert (m4 >= 0) is S.true
    m1 = Symbol("m1", negative=True)
    m2 = Symbol("m2", nonpositive=True)
    m3 = Symbol("m3", positive=True)
    m4 = Symbol("m4", nonnegative=True)
    assert (m1 < 0) is S.true
    assert (m2 <= 0) is S.true
    assert (m3 > 0) is S.true
    assert (m4 >= 0) is S.true
    m1 = Symbol("m1", negative=False, real=True)
    m2 = Symbol("m2", nonpositive=False, real=True)
    m3 = Symbol("m3", positive=False, real=True)
    m4 = Symbol("m4", nonnegative=False, real=True)
    assert (m1 < 0) is S.false
    assert (m2 <= 0) is S.false
    assert (m3 > 0) is S.false
    assert (m4 >= 0) is S.false


# See https://github.com/sympy/sympy/issues/17708
#def test_relational_noncommutative():
#    from sympy import Lt, Gt, Le, Ge
#    A, B = symbols('A,B', commutative=False)
#    assert (A < B) == Lt(A, B)
#    assert (A <= B) == Le(A, B)
#    assert (A > B) == Gt(A, B)
#    assert (A >= B) == Ge(A, B)


def test_basic_nostr():
    for obj in basic_objs:
        raises(TypeError, lambda: obj + '1')
        raises(TypeError, lambda: obj - '1')
        if obj == 2:
            assert obj * '1' == '11'
        else:
            raises(TypeError, lambda: obj * '1')
        raises(TypeError, lambda: obj / '1')
        raises(TypeError, lambda: obj ** '1')


def test_series_expansion_for_uniform_order():
    assert (1/x + y + x).series(x, 0, 0) == 1/x + O(1, x)
    assert (1/x + y + x).series(x, 0, 1) == 1/x + y + O(x)
    assert (1/x + 1 + x).series(x, 0, 0) == 1/x + O(1, x)
    assert (1/x + 1 + x).series(x, 0, 1) == 1/x + 1 + O(x)
    assert (1/x + x).series(x, 0, 0) == 1/x + O(1, x)
    assert (1/x + y + y*x + x).series(x, 0, 0) == 1/x + O(1, x)
    assert (1/x + y + y*x + x).series(x, 0, 1) == 1/x + y + O(x)


def test_leadterm():
    assert (3 + 2*x**(log(3)/log(2) - 1)).leadterm(x) == (3, 0)

    assert (1/x**2 + 1 + x + x**2).leadterm(x)[1] == -2
    assert (1/x + 1 + x + x**2).leadterm(x)[1] == -1
    assert (x**2 + 1/x).leadterm(x)[1] == -1
    assert (1 + x**2).leadterm(x)[1] == 0
    assert (x + 1).leadterm(x)[1] == 0
    assert (x + x**2).leadterm(x)[1] == 1
    assert (x**2).leadterm(x)[1] == 2


def test_as_leading_term():
    assert (3 + 2*x**(log(3)/log(2) - 1)).as_leading_term(x) == 3
    assert (1/x**2 + 1 + x + x**2).as_leading_term(x) == 1/x**2
    assert (1/x + 1 + x + x**2).as_leading_term(x) == 1/x
    assert (x**2 + 1/x).as_leading_term(x) == 1/x
    assert (1 + x**2).as_leading_term(x) == 1
    assert (x + 1).as_leading_term(x) == 1
    assert (x + x**2).as_leading_term(x) == x
    assert (x**2).as_leading_term(x) == x**2
    assert (x + oo).as_leading_term(x) is oo

    raises(ValueError, lambda: (x + 1).as_leading_term(1))

    # https://github.com/sympy/sympy/issues/21177
    e = -3*x + (x + Rational(3, 2) - sqrt(3)*S.ImaginaryUnit/2)**2\
        - Rational(3, 2) + 3*sqrt(3)*S.ImaginaryUnit/2
    assert e.as_leading_term(x) == -sqrt(3)*I*x

    # https://github.com/sympy/sympy/issues/21245
    e = 1 - x - x**2
    d = (1 + sqrt(5))/2
    assert e.subs(x, y + 1/d).as_leading_term(y) == \
        (-40*y - 16*sqrt(5)*y)/(16 + 8*sqrt(5))

    # https://github.com/sympy/sympy/issues/26991
    assert sinh(tanh(3/(100*x))).as_leading_term(x, cdir = 1) == sinh(1)


def test_leadterm2():
    assert (x*cos(1)*cos(1 + sin(1)) + sin(1 + sin(1))).leadterm(x) == \
           (sin(1 + sin(1)), 0)


def test_leadterm3():
    assert (y + z + x).leadterm(x) == (y + z, 0)


def test_as_leading_term2():
    assert (x*cos(1)*cos(1 + sin(1)) + sin(1 + sin(1))).as_leading_term(x) == \
        sin(1 + sin(1))


def test_as_leading_term3():
    assert (2 + pi + x).as_leading_term(x) == 2 + pi
    assert (2*x + pi*x + x**2).as_leading_term(x) == 2*x + pi*x


def test_as_leading_term4():
    # see issue 6843
    n = Symbol('n', integer=True, positive=True)
    r = -n**3/(2*n**2 + 4*n + 2) - n**2/(n**2 + 2*n + 1) + \
        n**2/(n + 1) - n/(2*n**2 + 4*n + 2) + n/(n*x + x) + 2*n/(n + 1) - \
        1 + 1/(n*x + x) + 1/(n + 1) - 1/x
    assert r.as_leading_term(x).cancel() == n/2


def test_as_leading_term_stub():
    class foo(Function):
        pass
    assert foo(1/x).as_leading_term(x) == foo(1/x)
    assert foo(1).as_leading_term(x) == foo(1)
    raises(NotImplementedError, lambda: foo(x).as_leading_term(x))


def test_as_leading_term_deriv_integral():
    # related to issue 11313
    assert Derivative(x ** 3, x).as_leading_term(x) == 3*x**2
    assert Derivative(x ** 3, y).as_leading_term(x) == 0

    assert Integral(x ** 3, x).as_leading_term(x) == x**4/4
    assert Integral(x ** 3, y).as_leading_term(x) == y*x**3

    assert Derivative(exp(x), x).as_leading_term(x) == 1
    assert Derivative(log(x), x).as_leading_term(x) == (1/x).as_leading_term(x)


def test_atoms():
    assert x.atoms() == {x}
    assert (1 + x).atoms() == {x, S.One}

    assert (1 + 2*cos(x)).atoms(Symbol) == {x}
    assert (1 + 2*cos(x)).atoms(Symbol, Number) == {S.One, S(2), x}

    assert (2*(x**(y**x))).atoms() == {S(2), x, y}

    assert S.Half.atoms() == {S.Half}
    assert S.Half.atoms(Symbol) == set()

    assert sin(oo).atoms(oo) == set()

    assert Poly(0, x).atoms() == {S.Zero, x}
    assert Poly(1, x).atoms() == {S.One, x}

    assert Poly(x, x).atoms() == {x}
    assert Poly(x, x, y).atoms() == {x, y}
    assert Poly(x + y, x, y).atoms() == {x, y}
    assert Poly(x + y, x, y, z).atoms() == {x, y, z}
    assert Poly(x + y*t, x, y, z).atoms() == {t, x, y, z}

    assert (I*pi).atoms(NumberSymbol) == {pi}
    assert (I*pi).atoms(NumberSymbol, I) == \
        (I*pi).atoms(I, NumberSymbol) == {pi, I}

    assert exp(exp(x)).atoms(exp) == {exp(exp(x)), exp(x)}
    assert (1 + x*(2 + y) + exp(3 + z)).atoms(Add) == \
        {1 + x*(2 + y) + exp(3 + z), 2 + y, 3 + z}

    # issue 6132
    e = (f(x) + sin(x) + 2)
    assert e.atoms(AppliedUndef) == \
        {f(x)}
    assert e.atoms(AppliedUndef, Function) == \
        {f(x), sin(x)}
    assert e.atoms(Function) == \
        {f(x), sin(x)}
    assert e.atoms(AppliedUndef, Number) == \
        {f(x), S(2)}
    assert e.atoms(Function, Number) == \
        {S(2), sin(x), f(x)}


def test_is_polynomial():
    k = Symbol('k', nonnegative=True, integer=True)

    assert Rational(2).is_polynomial(x, y, z) is True
    assert (S.Pi).is_polynomial(x, y, z) is True

    assert x.is_polynomial(x) is True
    assert x.is_polynomial(y) is True

    assert (x**2).is_polynomial(x) is True
    assert (x**2).is_polynomial(y) is True

    assert (x**(-2)).is_polynomial(x) is False
    assert (x**(-2)).is_polynomial(y) is True

    assert (2**x).is_polynomial(x) is False
    assert (2**x).is_polynomial(y) is True

    assert (x**k).is_polynomial(x) is False
    assert (x**k).is_polynomial(k) is False
    assert (x**x).is_polynomial(x) is False
    assert (k**k).is_polynomial(k) is False
    assert (k**x).is_polynomial(k) is False

    assert (x**(-k)).is_polynomial(x) is False
    assert ((2*x)**k).is_polynomial(x) is False

    assert (x**2 + 3*x - 8).is_polynomial(x) is True
    assert (x**2 + 3*x - 8).is_polynomial(y) is True

    assert (x**2 + 3*x - 8).is_polynomial() is True

    assert sqrt(x).is_polynomial(x) is False
    assert (sqrt(x)**3).is_polynomial(x) is False

    assert (x**2 + 3*x*sqrt(y) - 8).is_polynomial(x) is True
    assert (x**2 + 3*x*sqrt(y) - 8).is_polynomial(y) is False

    assert ((x**2)*(y**2) + x*(y**2) + y*x + exp(2)).is_polynomial() is True
    assert ((x**2)*(y**2) + x*(y**2) + y*x + exp(x)).is_polynomial() is False

    assert (
        (x**2)*(y**2) + x*(y**2) + y*x + exp(2)).is_polynomial(x, y) is True
    assert (
        (x**2)*(y**2) + x*(y**2) + y*x + exp(x)).is_polynomial(x, y) is False

    assert (1/f(x) + 1).is_polynomial(f(x)) is False


def test_is_rational_function():
    assert Integer(1).is_rational_function() is True
    assert Integer(1).is_rational_function(x) is True

    assert Rational(17, 54).is_rational_function() is True
    assert Rational(17, 54).is_rational_function(x) is True

    assert (12/x).is_rational_function() is True
    assert (12/x).is_rational_function(x) is True

    assert (x/y).is_rational_function() is True
    assert (x/y).is_rational_function(x) is True
    assert (x/y).is_rational_function(x, y) is True

    assert (x**2 + 1/x/y).is_rational_function() is True
    assert (x**2 + 1/x/y).is_rational_function(x) is True
    assert (x**2 + 1/x/y).is_rational_function(x, y) is True

    assert (sin(y)/x).is_rational_function() is False
    assert (sin(y)/x).is_rational_function(y) is False
    assert (sin(y)/x).is_rational_function(x) is True
    assert (sin(y)/x).is_rational_function(x, y) is False

    for i in _illegal:
        assert not i.is_rational_function()
        for d in (1, x):
            assert not (i/d).is_rational_function()


def test_is_meromorphic():
    f = a/x**2 + b + x + c*x**2
    assert f.is_meromorphic(x, 0) is True
    assert f.is_meromorphic(x, 1) is True
    assert f.is_meromorphic(x, zoo) is True

    g = 3 + 2*x**(log(3)/log(2) - 1)
    assert g.is_meromorphic(x, 0) is False
    assert g.is_meromorphic(x, 1) is True
    assert g.is_meromorphic(x, zoo) is False

    n = Symbol('n', integer=True)
    e = sin(1/x)**n*x
    assert e.is_meromorphic(x, 0) is False
    assert e.is_meromorphic(x, 1) is True
    assert e.is_meromorphic(x, zoo) is False

    e = log(x)**pi
    assert e.is_meromorphic(x, 0) is False
    assert e.is_meromorphic(x, 1) is False
    assert e.is_meromorphic(x, 2) is True
    assert e.is_meromorphic(x, zoo) is False

    assert (log(x)**a).is_meromorphic(x, 0) is False
    assert (log(x)**a).is_meromorphic(x, 1) is False
    assert (a**log(x)).is_meromorphic(x, 0) is None
    assert (3**log(x)).is_meromorphic(x, 0) is False
    assert (3**log(x)).is_meromorphic(x, 1) is True

def test_is_algebraic_expr():
    assert sqrt(3).is_algebraic_expr(x) is True
    assert sqrt(3).is_algebraic_expr() is True

    eq = ((1 + x**2)/(1 - y**2))**(S.One/3)
    assert eq.is_algebraic_expr(x) is True
    assert eq.is_algebraic_expr(y) is True

    assert (sqrt(x) + y**(S(2)/3)).is_algebraic_expr(x) is True
    assert (sqrt(x) + y**(S(2)/3)).is_algebraic_expr(y) is True
    assert (sqrt(x) + y**(S(2)/3)).is_algebraic_expr() is True

    assert (cos(y)/sqrt(x)).is_algebraic_expr() is False
    assert (cos(y)/sqrt(x)).is_algebraic_expr(x) is True
    assert (cos(y)/sqrt(x)).is_algebraic_expr(y) is False
    assert (cos(y)/sqrt(x)).is_algebraic_expr(x, y) is False


def test_SAGE1():
    #see https://github.com/sympy/sympy/issues/3346
    class MyInt:
        def _sympy_(self):
            return Integer(5)
    m = MyInt()
    e = Rational(2)*m
    assert e == 10

    raises(TypeError, lambda: Rational(2)*MyInt)


def test_SAGE2():
    class MyInt:
        def __int__(self):
            return 5
    assert sympify(MyInt()) == 5
    e = Rational(2)*MyInt()
    assert e == 10

    raises(TypeError, lambda: Rational(2)*MyInt)


def test_SAGE3():
    class MySymbol:
        def __rmul__(self, other):
            return ('mys', other, self)

    o = MySymbol()
    e = x*o

    assert e == ('mys', x, o)


def test_len():
    e = x*y
    assert len(e.args) == 2
    e = x + y + z
    assert len(e.args) == 3


def test_doit():
    a = Integral(x**2, x)

    assert isinstance(a.doit(), Integral) is False

    assert isinstance(a.doit(integrals=True), Integral) is False
    assert isinstance(a.doit(integrals=False), Integral) is True

    assert (2*Integral(x, x)).doit() == x**2


def test_attribute_error():
    raises(AttributeError, lambda: x.cos())
    raises(AttributeError, lambda: x.sin())
    raises(AttributeError, lambda: x.exp())


def test_args():
    assert (x*y).args in ((x, y), (y, x))
    assert (x + y).args in ((x, y), (y, x))
    assert (x*y + 1).args in ((x*y, 1), (1, x*y))
    assert sin(x*y).args == (x*y,)
    assert sin(x*y).args[0] == x*y
    assert (x**y).args == (x, y)
    assert (x**y).args[0] == x
    assert (x**y).args[1] == y


def test_noncommutative_expand_issue_3757():
    A, B, C = symbols('A,B,C', commutative=False)
    assert A*B - B*A != 0
    assert (A*(A + B)*B).expand() == A**2*B + A*B**2
    assert (A*(A + B + C)*B).expand() == A**2*B + A*B**2 + A*C*B


def test_as_numer_denom():
    a, b, c = symbols('a, b, c')

    assert nan.as_numer_denom() == (nan, 1)
    assert oo.as_numer_denom() == (oo, 1)
    assert (-oo).as_numer_denom() == (-oo, 1)
    assert zoo.as_numer_denom() == (zoo, 1)
    assert (-zoo).as_numer_denom() == (zoo, 1)

    assert x.as_numer_denom() == (x, 1)
    assert (1/x).as_numer_denom() == (1, x)
    assert (x/y).as_numer_denom() == (x, y)
    assert (x/2).as_numer_denom() == (x, 2)
    assert (x*y/z).as_numer_denom() == (x*y, z)
    assert (x/(y*z)).as_numer_denom() == (x, y*z)
    assert S.Half.as_numer_denom() == (1, 2)
    assert (1/y**2).as_numer_denom() == (1, y**2)
    assert (x/y**2).as_numer_denom() == (x, y**2)
    assert ((x**2 + 1)/y).as_numer_denom() == (x**2 + 1, y)
    assert (x*(y + 1)/y**7).as_numer_denom() == (x*(y + 1), y**7)
    assert (x**-2).as_numer_denom() == (1, x**2)
    assert (a/x + b/2/x + c/3/x).as_numer_denom() == \
        (6*a + 3*b + 2*c, 6*x)
    assert (a/x + b/2/x + c/3/y).as_numer_denom() == \
        (2*c*x + y*(6*a + 3*b), 6*x*y)
    assert (a/x + b/2/x + c/.5/x).as_numer_denom() == \
        (2*a + b + 4.0*c, 2*x)
    # this should take no more than a few seconds
    assert int(log(Add(*[Dummy()/i/x for i in range(1, 705)]
                       ).as_numer_denom()[1]/x).n(4)) == 705
    for i in [S.Infinity, S.NegativeInfinity, S.ComplexInfinity]:
        assert (i + x/3).as_numer_denom() == \
            (x + i, 3)
    assert (S.Infinity + x/3 + y/4).as_numer_denom() == \
        (4*x + 3*y + S.Infinity, 12)
    assert (oo*x + zoo*y).as_numer_denom() == \
        (zoo*y + oo*x, 1)

    A, B, C = symbols('A,B,C', commutative=False)

    assert (A*B*C**-1).as_numer_denom() == (A*B*C**-1, 1)
    assert (A*B*C**-1/x).as_numer_denom() == (A*B*C**-1, x)
    assert (C**-1*A*B).as_numer_denom() == (C**-1*A*B, 1)
    assert (C**-1*A*B/x).as_numer_denom() == (C**-1*A*B, x)
    assert ((A*B*C)**-1).as_numer_denom() == ((A*B*C)**-1, 1)
    assert ((A*B*C)**-1/x).as_numer_denom() == ((A*B*C)**-1, x)

    # the following morphs from Add to Mul during processing
    assert Add(0, (x + y)/z/-2, evaluate=False).as_numer_denom(
        ) == (-x - y, 2*z)


def test_trunc():
    import math
    x, y = symbols('x y')
    assert math.trunc(2) == 2
    assert math.trunc(4.57) == 4
    assert math.trunc(-5.79) == -5
    assert math.trunc(pi) == 3
    assert math.trunc(log(7)) == 1
    assert math.trunc(exp(5)) == 148
    assert math.trunc(cos(pi)) == -1
    assert math.trunc(sin(5)) == 0

    raises(TypeError, lambda: math.trunc(x))
    raises(TypeError, lambda: math.trunc(x + y**2))
    raises(TypeError, lambda: math.trunc(oo))


def test_as_independent():
    assert S.Zero.as_independent(x, as_Add=True) == (0, 0)
    assert S.Zero.as_independent(x, as_Add=False) == (0, 0)
    assert (2*x*sin(x) + y + x).as_independent(x) == (y, x + 2*x*sin(x))
    assert (2*x*sin(x) + y + x).as_independent(y) == (x + 2*x*sin(x), y)

    assert (2*x*sin(x) + y + x).as_independent(x, y) == (0, y + x + 2*x*sin(x))

    assert (x*sin(x)*cos(y)).as_independent(x) == (cos(y), x*sin(x))
    assert (x*sin(x)*cos(y)).as_independent(y) == (x*sin(x), cos(y))

    assert (x*sin(x)*cos(y)).as_independent(x, y) == (1, x*sin(x)*cos(y))

    assert (sin(x)).as_independent(x) == (1, sin(x))
    assert (sin(x)).as_independent(y) == (sin(x), 1)

    assert (2*sin(x)).as_independent(x) == (2, sin(x))
    assert (2*sin(x)).as_independent(y) == (2*sin(x), 1)

    # issue 4903 = 1766b
    n1, n2, n3 = symbols('n1 n2 n3', commutative=False)
    assert (n1 + n1*n2).as_independent(n2) == (n1, n1*n2)
    assert (n2*n1 + n1*n2).as_independent(n2) == (0, n1*n2 + n2*n1)
    assert (n1*n2*n1).as_independent(n2) == (n1, n2*n1)
    assert (n1*n2*n1).as_independent(n1) == (1, n1*n2*n1)

    assert (3*x).as_independent(x, as_Add=True) == (0, 3*x)
    assert (3*x).as_independent(x, as_Add=False) == (3, x)
    assert (3 + x).as_independent(x, as_Add=True) == (3, x)
    assert (3 + x).as_independent(x, as_Add=False) == (1, 3 + x)

    # issue 5479
    assert (3*x).as_independent(Symbol) == (3, x)

    # issue 5648
    assert (n1*x*y).as_independent(x) == (n1*y, x)
    assert ((x + n1)*(x - y)).as_independent(x) == (1, (x + n1)*(x - y))
    assert ((x + n1)*(x - y)).as_independent(y) == (x + n1, x - y)
    assert (DiracDelta(x - n1)*DiracDelta(x - y)).as_independent(x) \
        == (1, DiracDelta(x - n1)*DiracDelta(x - y))
    assert (x*y*n1*n2*n3).as_independent(n2) == (x*y*n1, n2*n3)
    assert (x*y*n1*n2*n3).as_independent(n1) == (x*y, n1*n2*n3)
    assert (x*y*n1*n2*n3).as_independent(n3) == (x*y*n1*n2, n3)
    assert (DiracDelta(x - n1)*DiracDelta(y - n1)*DiracDelta(x - n2)).as_independent(y) == \
           (DiracDelta(x - n1)*DiracDelta(x - n2), DiracDelta(y - n1))

    # issue 5784
    assert (x + Integral(x, (x, 1, 2))).as_independent(x, strict=True) == \
           (Integral(x, (x, 1, 2)), x)

    eq = Add(x, -x, 2, -3, evaluate=False)
    assert eq.as_independent(x) == (-1, Add(x, -x, evaluate=False))
    eq = Mul(x, 1/x, 2, -3, evaluate=False)
    assert eq.as_independent(x) == (-6, Mul(x, 1/x, evaluate=False))

    assert (x*y).as_independent(z, as_Add=True) == (x*y, 0)

@XFAIL
def test_call_2():
    # TODO UndefinedFunction does not subclass Expr
    assert (2*f)(x) == 2*f(x)


def test_replace():
    e = log(sin(x)) + tan(sin(x**2))

    assert e.replace(sin, cos) == log(cos(x)) + tan(cos(x**2))
    assert e.replace(
        sin, lambda a: sin(2*a)) == log(sin(2*x)) + tan(sin(2*x**2))

    a = Wild('a')
    b = Wild('b')

    assert e.replace(sin(a), cos(a)) == log(cos(x)) + tan(cos(x**2))
    assert e.replace(
        sin(a), lambda a: sin(2*a)) == log(sin(2*x)) + tan(sin(2*x**2))
    # test exact
    assert (2*x).replace(a*x + b, b - a, exact=True) == 2*x
    assert (2*x).replace(a*x + b, b - a) == 2*x
    assert (2*x).replace(a*x + b, b - a, exact=False) == 2/x
    assert (2*x).replace(a*x + b, lambda a, b: b - a, exact=True) == 2*x
    assert (2*x).replace(a*x + b, lambda a, b: b - a) == 2*x
    assert (2*x).replace(a*x + b, lambda a, b: b - a, exact=False) == 2/x

    g = 2*sin(x**3)

    assert g.replace(
        lambda expr: expr.is_Number, lambda expr: expr**2) == 4*sin(x**9)

    assert cos(x).replace(cos, sin, map=True) == (sin(x), {cos(x): sin(x)})
    assert sin(x).replace(cos, sin) == sin(x)

    cond, func = lambda x: x.is_Mul, lambda x: 2*x
    assert (x*y).replace(cond, func, map=True) == (2*x*y, {x*y: 2*x*y})
    assert (x*(1 + x*y)).replace(cond, func, map=True) == \
        (2*x*(2*x*y + 1), {x*(2*x*y + 1): 2*x*(2*x*y + 1), x*y: 2*x*y})
    assert (y*sin(x)).replace(sin, lambda expr: sin(expr)/y, map=True) == \
        (sin(x), {sin(x): sin(x)/y})
    # if not simultaneous then y*sin(x) -> y*sin(x)/y = sin(x) -> sin(x)/y
    assert (y*sin(x)).replace(sin, lambda expr: sin(expr)/y,
        simultaneous=False) == sin(x)/y
    assert (x**2 + O(x**3)).replace(Pow, lambda b, e: b**e/e
        ) == x**2/2 + O(x**3)
    assert (x**2 + O(x**3)).replace(Pow, lambda b, e: b**e/e,
        simultaneous=False) == x**2/2 + O(x**3)
    assert (x*(x*y + 3)).replace(lambda x: x.is_Mul, lambda x: 2 + x) == \
        x*(x*y + 5) + 2
    e = (x*y + 1)*(2*x*y + 1) + 1
    assert e.replace(cond, func, map=True) == (
        2*((2*x*y + 1)*(4*x*y + 1)) + 1,
        {2*x*y: 4*x*y, x*y: 2*x*y, (2*x*y + 1)*(4*x*y + 1):
        2*((2*x*y + 1)*(4*x*y + 1))})
    assert x.replace(x, y) == y
    assert (x + 1).replace(1, 2) == x + 2

    # https://groups.google.com/forum/#!topic/sympy/8wCgeC95tz0
    n1, n2, n3 = symbols('n1:4', commutative=False)
    assert (n1*f(n2)).replace(f, lambda x: x) == n1*n2
    assert (n3*f(n2)).replace(f, lambda x: x) == n3*n2

    # issue 16725
    assert S.Zero.replace(Wild('x'), 1) == 1
    # let the user override the default decision of False
    assert S.Zero.replace(Wild('x'), 1, exact=True) == 0


def test_replace_integral():
    # https://github.com/sympy/sympy/issues/27142
    q, p, s, t = symbols('q p s t', cls=Wild)
    a, b, c, d = symbols('a b c d')
    i = Integral(a + b, (b, c, d))
    pattern = Integral(q, (p, s, t))
    assert i.replace(pattern, q) == a + b


def test_find():
    expr = (x + y + 2 + sin(3*x))

    assert expr.find(lambda u: u.is_Integer) == {S(2), S(3)}
    assert expr.find(lambda u: u.is_Symbol) == {x, y}

    assert expr.find(lambda u: u.is_Integer, group=True) == {S(2): 1, S(3): 1}
    assert expr.find(lambda u: u.is_Symbol, group=True) == {x: 2, y: 1}

    assert expr.find(Integer) == {S(2), S(3)}
    assert expr.find(Symbol) == {x, y}

    assert expr.find(Integer, group=True) == {S(2): 1, S(3): 1}
    assert expr.find(Symbol, group=True) == {x: 2, y: 1}

    a = Wild('a')

    expr = sin(sin(x)) + sin(x) + cos(x) + x

    assert expr.find(lambda u: type(u) is sin) == {sin(x), sin(sin(x))}
    assert expr.find(
        lambda u: type(u) is sin, group=True) == {sin(x): 2, sin(sin(x)): 1}

    assert expr.find(sin(a)) == {sin(x), sin(sin(x))}
    assert expr.find(sin(a), group=True) == {sin(x): 2, sin(sin(x)): 1}

    assert expr.find(sin) == {sin(x), sin(sin(x))}
    assert expr.find(sin, group=True) == {sin(x): 2, sin(sin(x)): 1}


def test_count():
    expr = (x + y + 2 + sin(3*x))

    assert expr.count(lambda u: u.is_Integer) == 2
    assert expr.count(lambda u: u.is_Symbol) == 3

    assert expr.count(Integer) == 2
    assert expr.count(Symbol) == 3
    assert expr.count(2) == 1

    a = Wild('a')

    assert expr.count(sin) == 1
    assert expr.count(sin(a)) == 1
    assert expr.count(lambda u: type(u) is sin) == 1

    assert f(x).count(f(x)) == 1
    assert f(x).diff(x).count(f(x)) == 1
    assert f(x).diff(x).count(x) == 2


def test_has_basics():
    p = Wild('p')

    assert sin(x).has(x)
    assert sin(x).has(sin)
    assert not sin(x).has(y)
    assert not sin(x).has(cos)
    assert f(x).has(x)
    assert f(x).has(f)
    assert not f(x).has(y)
    assert not f(x).has(g)

    assert f(x).diff(x).has(x)
    assert f(x).diff(x).has(f)
    assert f(x).diff(x).has(Derivative)
    assert not f(x).diff(x).has(y)
    assert not f(x).diff(x).has(g)
    assert not f(x).diff(x).has(sin)

    assert (x**2).has(Symbol)
    assert not (x**2).has(Wild)
    assert (2*p).has(Wild)

    assert not x.has()

    # see issue at https://github.com/sympy/sympy/issues/5190
    assert not S(1).has(Wild)
    assert not x.has(Wild)


def test_has_multiple():
    f = x**2*y + sin(2**t + log(z))

    assert f.has(x)
    assert f.has(y)
    assert f.has(z)
    assert f.has(t)

    assert not f.has(u)

    assert f.has(x, y, z, t)
    assert f.has(x, y, z, t, u)

    i = Integer(4400)

    assert not i.has(x)

    assert (i*x**i).has(x)
    assert not (i*y**i).has(x)
    assert (i*y**i).has(x, y)
    assert not (i*y**i).has(x, z)


def test_has_piecewise():
    f = (x*y + 3/y)**(3 + 2)
    p = Piecewise((g(x), x < -1), (1, x <= 1), (f, True))

    assert p.has(x)
    assert p.has(y)
    assert not p.has(z)
    assert p.has(1)
    assert p.has(3)
    assert not p.has(4)
    assert p.has(f)
    assert p.has(g)
    assert not p.has(h)


def test_has_iterative():
    A, B, C = symbols('A,B,C', commutative=False)
    f = x*gamma(x)*sin(x)*exp(x*y)*A*B*C*cos(x*A*B)

    assert f.has(x)
    assert f.has(x*y)
    assert f.has(x*sin(x))
    assert not f.has(x*sin(y))
    assert f.has(x*A)
    assert f.has(x*A*B)
    assert not f.has(x*A*C)
    assert f.has(x*A*B*C)
    assert not f.has(x*A*C*B)
    assert f.has(x*sin(x)*A*B*C)
    assert not f.has(x*sin(x)*A*C*B)
    assert not f.has(x*sin(y)*A*B*C)
    assert f.has(x*gamma(x))
    assert not f.has(x + sin(x))

    assert (x & y & z).has(x & z)


def test_has_integrals():
    f = Integral(x**2 + sin(x*y*z), (x, 0, x + y + z))

    assert f.has(x + y)
    assert f.has(x + z)
    assert f.has(y + z)

    assert f.has(x*y)
    assert f.has(x*z)
    assert f.has(y*z)

    assert not f.has(2*x + y)
    assert not f.has(2*x*y)


def test_has_tuple():
    assert Tuple(x, y).has(x)
    assert not Tuple(x, y).has(z)
    assert Tuple(f(x), g(x)).has(x)
    assert not Tuple(f(x), g(x)).has(y)
    assert Tuple(f(x), g(x)).has(f)
    assert Tuple(f(x), g(x)).has(f(x))
    # XXX to be deprecated
    #assert not Tuple(f, g).has(x)
    #assert Tuple(f, g).has(f)
    #assert not Tuple(f, g).has(h)
    assert Tuple(True).has(True)
    assert Tuple(True).has(S.true)
    assert not Tuple(True).has(1)


def test_has_units():
    from sympy.physics.units import m, s

    assert (x*m/s).has(x)
    assert (x*m/s).has(y, z) is False


def test_has_polys():
    poly = Poly(x**2 + x*y*sin(z), x, y, t)

    assert poly.has(x)
    assert poly.has(x, y, z)
    assert poly.has(x, y, z, t)


def test_has_physics():
    assert FockState((x, y)).has(x)


def test_as_poly_as_expr():
    f = x**2 + 2*x*y

    assert f.as_poly().as_expr() == f
    assert f.as_poly(x, y).as_expr() == f

    assert (f + sin(x)).as_poly(x, y) is None

    p = Poly(f, x, y)

    assert p.as_poly() == p

    # https://github.com/sympy/sympy/issues/20610
    assert S(2).as_poly() is None
    assert sqrt(2).as_poly(extension=True) is None

    raises(AttributeError, lambda: Tuple(x, x).as_poly(x))
    raises(AttributeError, lambda: Tuple(x ** 2, x, y).as_poly(x))


def test_nonzero():
    assert bool(S.Zero) is False
    assert bool(S.One) is True
    assert bool(x) is True
    assert bool(x + y) is True
    assert bool(x - x) is False
    assert bool(x*y) is True
    assert bool(x*1) is True
    assert bool(x*0) is False


def test_is_number():
    assert Float(3.14).is_number is True
    assert Integer(737).is_number is True
    assert Rational(3, 2).is_number is True
    assert Rational(8).is_number is True
    assert x.is_number is False
    assert (2*x).is_number is False
    assert (x + y).is_number is False
    assert log(2).is_number is True
    assert log(x).is_number is False
    assert (2 + log(2)).is_number is True
    assert (8 + log(2)).is_number is True
    assert (2 + log(x)).is_number is False
    assert (8 + log(2) + x).is_number is False
    assert (1 + x**2/x - x).is_number is True
    assert Tuple(Integer(1)).is_number is False
    assert Add(2, x).is_number is False
    assert Mul(3, 4).is_number is True
    assert Pow(log(2), 2).is_number is True
    assert oo.is_number is True
    g = WildFunction('g')
    assert g.is_number is False
    assert (2*g).is_number is False
    assert (x**2).subs(x, 3).is_number is True

    # test extensibility of .is_number
    # on subinstances of Basic
    class A(Basic):
        pass
    a = A()
    assert a.is_number is False


def test_as_coeff_add():
    assert S(2).as_coeff_add() == (2, ())
    assert S(3.0).as_coeff_add() == (0, (S(3.0),))
    assert S(-3.0).as_coeff_add() == (0, (S(-3.0),))
    assert x.as_coeff_add() == (0, (x,))
    assert (x - 1).as_coeff_add() == (-1, (x,))
    assert (x + 1).as_coeff_add() == (1, (x,))
    assert (x + 2).as_coeff_add() == (2, (x,))
    assert (x + y).as_coeff_add(y) == (x, (y,))
    assert (3*x).as_coeff_add(y) == (3*x, ())
    # don't do expansion
    e = (x + y)**2
    assert e.as_coeff_add(y) == (0, (e,))


def test_as_coeff_mul():
    assert S(2).as_coeff_mul() == (2, ())
    assert S(3.0).as_coeff_mul() == (1, (S(3.0),))
    assert S(-3.0).as_coeff_mul() == (-1, (S(3.0),))
    assert S(-3.0).as_coeff_mul(rational=False) == (-S(3.0), ())
    assert x.as_coeff_mul() == (1, (x,))
    assert (-x).as_coeff_mul() == (-1, (x,))
    assert (2*x).as_coeff_mul() == (2, (x,))
    assert (x*y).as_coeff_mul(y) == (x, (y,))
    assert (3 + x).as_coeff_mul() == (1, (3 + x,))
    assert (3 + x).as_coeff_mul(y) == (3 + x, ())
    # don't do expansion
    e = exp(x + y)
    assert e.as_coeff_mul(y) == (1, (e,))
    e = 2**(x + y)
    assert e.as_coeff_mul(y) == (1, (e,))
    assert (1.1*x).as_coeff_mul(rational=False) == (1.1, (x,))
    assert (1.1*x).as_coeff_mul() == (1, (1.1, x))
    assert (-oo*x).as_coeff_mul(rational=True) == (-1, (oo, x))


def test_as_coeff_exponent():
    assert (3*x**4).as_coeff_exponent(x) == (3, 4)
    assert (2*x**3).as_coeff_exponent(x) == (2, 3)
    assert (4*x**2).as_coeff_exponent(x) == (4, 2)
    assert (6*x**1).as_coeff_exponent(x) == (6, 1)
    assert (3*x**0).as_coeff_exponent(x) == (3, 0)
    assert (2*x**0).as_coeff_exponent(x) == (2, 0)
    assert (1*x**0).as_coeff_exponent(x) == (1, 0)
    assert (0*x**0).as_coeff_exponent(x) == (0, 0)
    assert (-1*x**0).as_coeff_exponent(x) == (-1, 0)
    assert (-2*x**0).as_coeff_exponent(x) == (-2, 0)
    assert (2*x**3 + pi*x**3).as_coeff_exponent(x) == (2 + pi, 3)
    assert (x*log(2)/(2*x + pi*x)).as_coeff_exponent(x) == \
        (log(2)/(2 + pi), 0)
    # issue 4784
    D = Derivative
    fx = D(f(x), x)
    assert fx.as_coeff_exponent(f(x)) == (fx, 0)


def test_extractions():
    for base in (2, S.Exp1):
        assert Pow(base**x, 3, evaluate=False
            ).extract_multiplicatively(base**x) == base**(2*x)
        assert (base**(5*x)).extract_multiplicatively(
            base**(3*x)) == base**(2*x)
    assert ((x*y)**3).extract_multiplicatively(x**2 * y) == x*y**2
    assert ((x*y)**3).extract_multiplicatively(x**4 * y) is None
    assert (2*x).extract_multiplicatively(2) == x
    assert (2*x).extract_multiplicatively(3) is None
    assert (2*x).extract_multiplicatively(-1) is None
    assert (S.Half*x).extract_multiplicatively(3) == x/6
    assert (sqrt(x)).extract_multiplicatively(x) is None
    assert (sqrt(x)).extract_multiplicatively(1/x) is None
    assert x.extract_multiplicatively(-x) is None
    assert (-2 - 4*I).extract_multiplicatively(-2) == 1 + 2*I
    assert (-2 - 4*I).extract_multiplicatively(3) is None
    assert (-2*x - 4*y - 8).extract_multiplicatively(-2) == x + 2*y + 4
    assert (-2*x*y - 4*x**2*y).extract_multiplicatively(-2*y) == 2*x**2 + x
    assert (2*x*y + 4*x**2*y).extract_multiplicatively(2*y) == 2*x**2 + x
    assert (-4*y**2*x).extract_multiplicatively(-3*y) is None
    assert (2*x).extract_multiplicatively(1) == 2*x
    assert (-oo).extract_multiplicatively(5) is -oo
    assert (oo).extract_multiplicatively(5) is oo

    assert ((x*y)**3).extract_additively(1) is None
    assert (x + 1).extract_additively(x) == 1
    assert (x + 1).extract_additively(2*x) is None
    assert (x + 1).extract_additively(-x) is None
    assert (-x + 1).extract_additively(2*x) is None
    assert (2*x + 3).extract_additively(x) == x + 3
    assert (2*x + 3).extract_additively(2) == 2*x + 1
    assert (2*x + 3).extract_additively(3) == 2*x
    assert (2*x + 3).extract_additively(-2) is None
    assert (2*x + 3).extract_additively(3*x) is None
    assert (2*x + 3).extract_additively(2*x) == 3
    assert x.extract_additively(0) == x
    assert S(2).extract_additively(x) is None
    assert S(2.).extract_additively(2.) is S.Zero
    assert S(2.).extract_additively(2) is S.Zero
    assert S(2*x + 3).extract_additively(x + 1) == x + 2
    assert S(2*x + 3).extract_additively(y + 1) is None
    assert S(2*x - 3).extract_additively(x + 1) is None
    assert S(2*x - 3).extract_additively(y + z) is None
    assert ((a + 1)*x*4 + y).extract_additively(x).expand() == \
        4*a*x + 3*x + y
    assert ((a + 1)*x*4 + 3*y).extract_additively(x + 2*y).expand() == \
        4*a*x + 3*x + y
    assert (y*(x + 1)).extract_additively(x + 1) is None
    assert ((y + 1)*(x + 1) + 3).extract_additively(x + 1) == \
        y*(x + 1) + 3
    assert ((x + y)*(x + 1) + x + y + 3).extract_additively(x + y) == \
        x*(x + y) + 3
    assert (x + y + 2*((x + y)*(x + 1)) + 3).extract_additively((x + y)*(x + 1)) == \
        x + y + (x + 1)*(x + y) + 3
    assert ((y + 1)*(x + 2*y + 1) + 3).extract_additively(y + 1) == \
        (x + 2*y)*(y + 1) + 3
    assert (-x - x*I).extract_additively(-x) == -I*x
    # extraction does not leave artificats, now
    assert (4*x*(y + 1) + y).extract_additively(x) == x*(4*y + 3) + y

    n = Symbol("n", integer=True)
    assert (Integer(-3)).could_extract_minus_sign() is True
    assert (-n*x + x).could_extract_minus_sign() != \
        (n*x - x).could_extract_minus_sign()
    assert (x - y).could_extract_minus_sign() != \
        (-x + y).could_extract_minus_sign()
    assert (1 - x - y).could_extract_minus_sign() is True
    assert (1 - x + y).could_extract_minus_sign() is False
    assert ((-x - x*y)/y).could_extract_minus_sign() is False
    assert ((x + x*y)/(-y)).could_extract_minus_sign() is True
    assert ((x + x*y)/y).could_extract_minus_sign() is False
    assert ((-x - y)/(x + y)).could_extract_minus_sign() is False

    class sign_invariant(Function, Expr):
        nargs = 1
        def __neg__(self):
            return self
    foo = sign_invariant(x)
    assert foo == -foo
    assert foo.could_extract_minus_sign() is False
    assert (x - y).could_extract_minus_sign() is False
    assert (-x + y).could_extract_minus_sign() is True
    assert (x - 1).could_extract_minus_sign() is False
    assert (1 - x).could_extract_minus_sign() is True
    assert (sqrt(2) - 1).could_extract_minus_sign() is True
    assert (1 - sqrt(2)).could_extract_minus_sign() is False
    # check that result is canonical
    eq = (3*x + 15*y).extract_multiplicatively(3)
    assert eq.args == eq.func(*eq.args).args


def test_nan_extractions():
    for r in (1, 0, I, nan):
        assert nan.extract_additively(r) is None
        assert nan.extract_multiplicatively(r) is None


def test_coeff():
    assert (x + 1).coeff(x + 1) == 1
    assert (3*x).coeff(0) == 0
    assert (z*(1 + x)*x**2).coeff(1 + x) == z*x**2
    assert (1 + 2*x*x**(1 + x)).coeff(x*x**(1 + x)) == 2
    assert (1 + 2*x**(y + z)).coeff(x**(y + z)) == 2
    assert (3 + 2*x + 4*x**2).coeff(1) == 0
    assert (3 + 2*x + 4*x**2).coeff(-1) == 0
    assert (3 + 2*x + 4*x**2).coeff(x) == 2
    assert (3 + 2*x + 4*x**2).coeff(x**2) == 4
    assert (3 + 2*x + 4*x**2).coeff(x**3) == 0

    assert (-x/8 + x*y).coeff(x) == Rational(-1, 8) + y
    assert (-x/8 + x*y).coeff(-x) == S.One/8
    assert (4*x).coeff(2*x) == 0
    assert (2*x).coeff(2*x) == 1
    assert (-oo*x).coeff(x*oo) == -1
    assert (10*x).coeff(x, 0) == 0
    assert (10*x).coeff(10*x, 0) == 0

    n1, n2 = symbols('n1 n2', commutative=False)
    assert (n1*n2).coeff(n1) == 1
    assert (n1*n2).coeff(n2) == n1
    assert (n1*n2 + x*n1).coeff(n1) == 1  # 1*n1*(n2+x)
    assert (n2*n1 + x*n1).coeff(n1) == n2 + x
    assert (n2*n1 + x*n1**2).coeff(n1) == n2
    assert (n1**x).coeff(n1) == 0
    assert (n1*n2 + n2*n1).coeff(n1) == 0
    assert (2*(n1 + n2)*n2).coeff(n1 + n2, right=1) == n2
    assert (2*(n1 + n2)*n2).coeff(n1 + n2, right=0) == 2

    assert (2*f(x) + 3*f(x).diff(x)).coeff(f(x)) == 2

    expr = z*(x + y)**2
    expr2 = z*(x + y)**2 + z*(2*x + 2*y)**2
    assert expr.coeff(z) == (x + y)**2
    assert expr.coeff(x + y) == 0
    assert expr2.coeff(z) == (x + y)**2 + (2*x + 2*y)**2

    assert (x + y + 3*z).coeff(1) == x + y
    assert (-x + 2*y).coeff(-1) == x
    assert (x - 2*y).coeff(-1) == 2*y
    assert (3 + 2*x + 4*x**2).coeff(1) == 0
    assert (-x - 2*y).coeff(2) == -y
    assert (x + sqrt(2)*x).coeff(sqrt(2)) == x
    assert (3 + 2*x + 4*x**2).coeff(x) == 2
    assert (3 + 2*x + 4*x**2).coeff(x**2) == 4
    assert (3 + 2*x + 4*x**2).coeff(x**3) == 0
    assert (z*(x + y)**2).coeff((x + y)**2) == z
    assert (z*(x + y)**2).coeff(x + y) == 0
    assert (2 + 2*x + (x + 1)*y).coeff(x + 1) == y

    assert (x + 2*y + 3).coeff(1) == x
    assert (x + 2*y + 3).coeff(x, 0) == 2*y + 3
    assert (x**2 + 2*y + 3*x).coeff(x**2, 0) == 2*y + 3*x
    assert x.coeff(0, 0) == 0
    assert x.coeff(x, 0) == 0

    n, m, o, l = symbols('n m o l', commutative=False)
    assert n.coeff(n) == 1
    assert y.coeff(n) == 0
    assert (3*n).coeff(n) == 3
    assert (2 + n).coeff(x*m) == 0
    assert (2*x*n*m).coeff(x) == 2*n*m
    assert (2 + n).coeff(x*m*n + y) == 0
    assert (2*x*n*m).coeff(3*n) == 0
    assert (n*m + m*n*m).coeff(n) == 1 + m
    assert (n*m + m*n*m).coeff(n, right=True) == m  # = (1 + m)*n*m
    assert (n*m + m*n).coeff(n) == 0
    assert (n*m + o*m*n).coeff(m*n) == o
    assert (n*m + o*m*n).coeff(m*n, right=True) == 1
    assert (n*m + n*m*n).coeff(n*m, right=True) == 1 + n  # = n*m*(n + 1)

    assert (x*y).coeff(z, 0) == x*y

    assert (x*n + y*n + z*m).coeff(n) == x + y
    assert (n*m + n*o + o*l).coeff(n, right=True) == m + o
    assert (x*n*m*n + y*n*m*o + z*l).coeff(m, right=True) == x*n + y*o
    assert (x*n*m*n + x*n*m*o + z*l).coeff(m, right=True) == n + o
    assert (x*n*m*n + x*n*m*o + z*l).coeff(m) == x*n


def test_coeff2():
    r, kappa = symbols('r, kappa')
    psi = Function("psi")
    g = 1/r**2 * (2*r*psi(r).diff(r, 1) + r**2 * psi(r).diff(r, 2))
    g = g.expand()
    assert g.coeff(psi(r).diff(r)) == 2/r


def test_coeff2_0():
    r, kappa = symbols('r, kappa')
    psi = Function("psi")
    g = 1/r**2 * (2*r*psi(r).diff(r, 1) + r**2 * psi(r).diff(r, 2))
    g = g.expand()

    assert g.coeff(psi(r).diff(r, 2)) == 1


def test_coeff_expand():
    expr = z*(x + y)**2
    expr2 = z*(x + y)**2 + z*(2*x + 2*y)**2
    assert expr.coeff(z) == (x + y)**2
    assert expr2.coeff(z) == (x + y)**2 + (2*x + 2*y)**2


def test_integrate():
    assert x.integrate(x) == x**2/2
    assert x.integrate((x, 0, 1)) == S.Half


def test_as_base_exp():
    assert x.as_base_exp() == (x, S.One)
    assert (x*y*z).as_base_exp() == (x*y*z, S.One)
    assert (x + y + z).as_base_exp() == (x + y + z, S.One)
    assert ((x + y)**z).as_base_exp() == (x + y, z)
    assert (x**2*y**2).as_base_exp() == (x*y, 2)
    assert (x**z*y**z).as_base_exp() == (x**z*y**z, S.One)


def test_issue_4963():
    assert hasattr(Mul(x, y), "is_commutative")
    assert hasattr(Mul(x, y, evaluate=False), "is_commutative")
    assert hasattr(Pow(x, y), "is_commutative")
    assert hasattr(Pow(x, y, evaluate=False), "is_commutative")
    expr = Mul(Pow(2, 2, evaluate=False), 3, evaluate=False) + 1
    assert hasattr(expr, "is_commutative")


def test_action_verbs():
    assert nsimplify(1/(exp(3*pi*x/5) + 1)) == \
        (1/(exp(3*pi*x/5) + 1)).nsimplify()
    assert ratsimp(1/x + 1/y) == (1/x + 1/y).ratsimp()
    assert trigsimp(log(x), deep=True) == (log(x)).trigsimp(deep=True)
    assert radsimp(1/(2 + sqrt(2))) == (1/(2 + sqrt(2))).radsimp()
    assert radsimp(1/(a + b*sqrt(c)), symbolic=False) == \
        (1/(a + b*sqrt(c))).radsimp(symbolic=False)
    assert powsimp(x**y*x**z*y**z, combine='all') == \
        (x**y*x**z*y**z).powsimp(combine='all')
    assert (x**t*y**t).powsimp(force=True) == (x*y)**t
    assert simplify(x**y*x**z*y**z) == (x**y*x**z*y**z).simplify()
    assert together(1/x + 1/y) == (1/x + 1/y).together()
    assert collect(a*x**2 + b*x**2 + a*x - b*x + c, x) == \
        (a*x**2 + b*x**2 + a*x - b*x + c).collect(x)
    assert apart(y/(y + 2)/(y + 1), y) == (y/(y + 2)/(y + 1)).apart(y)
    assert combsimp(y/(x + 2)/(x + 1)) == (y/(x + 2)/(x + 1)).combsimp()
    assert gammasimp(gamma(x)/gamma(x-5)) == (gamma(x)/gamma(x-5)).gammasimp()
    assert factor(x**2 + 5*x + 6) == (x**2 + 5*x + 6).factor()
    assert refine(sqrt(x**2)) == sqrt(x**2).refine()
    assert cancel((x**2 + 5*x + 6)/(x + 2)) == ((x**2 + 5*x + 6)/(x + 2)).cancel()


def test_as_powers_dict():
    assert x.as_powers_dict() == {x: 1}
    assert (x**y*z).as_powers_dict() == {x: y, z: 1}
    assert Mul(2, 2, evaluate=False).as_powers_dict() == {S(2): S(2)}
    assert (x*y).as_powers_dict()[z] == 0
    assert (x + y).as_powers_dict()[z] == 0


def test_as_coefficients_dict():
    check = [S.One, x, y, x*y, 1]
    assert [Add(3*x, 2*x, y, 3).as_coefficients_dict()[i] for i in check] == \
        [3, 5, 1, 0, 3]
    assert [Add(3*x, 2*x, y, 3, evaluate=False).as_coefficients_dict()[i]
            for i in check] == [3, 5, 1, 0, 3]
    assert [(3*x*y).as_coefficients_dict()[i] for i in check] == \
        [0, 0, 0, 3, 0]
    assert [(3.0*x*y).as_coefficients_dict()[i] for i in check] == \
        [0, 0, 0, 3.0, 0]
    assert (3.0*x*y).as_coefficients_dict()[3.0*x*y] == 0
    eq = x*(x + 1)*a + x*b + c/x
    assert eq.as_coefficients_dict(x) == {x: b, 1/x: c,
        x*(x + 1): a}
    assert eq.expand().as_coefficients_dict(x) == {x**2: a, x: a + b, 1/x: c}
    assert x.as_coefficients_dict() == {x: S.One}


def test_args_cnc():
    A = symbols('A', commutative=False)
    assert (x + A).args_cnc() == \
        [[], [x + A]]
    assert (x + a).args_cnc() == \
        [[a + x], []]
    assert (x*a).args_cnc() == \
        [[a, x], []]
    assert (x*y*A*(A + 1)).args_cnc(cset=True) == \
        [{x, y}, [A, 1 + A]]
    assert Mul(x, x, evaluate=False).args_cnc(cset=True, warn=False) == \
        [{x}, []]
    assert Mul(x, x**2, evaluate=False).args_cnc(cset=True, warn=False) == \
        [{x, x**2}, []]
    raises(ValueError, lambda: Mul(x, x, evaluate=False).args_cnc(cset=True))
    assert Mul(x, y, x, evaluate=False).args_cnc() == \
        [[x, y, x], []]
    # always split -1 from leading number
    assert (-1.*x).args_cnc() == [[-1, 1.0, x], []]


def test_new_rawargs():
    n = Symbol('n', commutative=False)
    a = x + n
    assert a.is_commutative is False
    assert a._new_rawargs(x).is_commutative
    assert a._new_rawargs(x, y).is_commutative
    assert a._new_rawargs(x, n).is_commutative is False
    assert a._new_rawargs(x, y, n).is_commutative is False
    m = x*n
    assert m.is_commutative is False
    assert m._new_rawargs(x).is_commutative
    assert m._new_rawargs(n).is_commutative is False
    assert m._new_rawargs(x, y).is_commutative
    assert m._new_rawargs(x, n).is_commutative is False
    assert m._new_rawargs(x, y, n).is_commutative is False

    assert m._new_rawargs(x, n, reeval=False).is_commutative is False
    assert m._new_rawargs(S.One) is S.One


def test_issue_5226():
    assert Add(evaluate=False) == 0
    assert Mul(evaluate=False) == 1
    assert Mul(x + y, evaluate=False).is_Add


def test_free_symbols():
    # free_symbols should return the free symbols of an object
    assert S.One.free_symbols == set()
    assert x.free_symbols == {x}
    assert Integral(x, (x, 1, y)).free_symbols == {y}
    assert (-Integral(x, (x, 1, y))).free_symbols == {y}
    assert meter.free_symbols == set()
    assert (meter**x).free_symbols == {x}


def test_has_free():
    assert x.has_free(x)
    assert not x.has_free(y)
    assert (x + y).has_free(x)
    assert (x + y).has_free(*(x, z))
    assert f(x).has_free(x)
    assert f(x).has_free(f(x))
    assert Integral(f(x), (f(x), 1, y)).has_free(y)
    assert not Integral(f(x), (f(x), 1, y)).has_free(x)
    assert not Integral(f(x), (f(x), 1, y)).has_free(f(x))
    # simple extraction
    assert (x + 1 + y).has_free(x + 1)
    assert not (x + 2 + y).has_free(x + 1)
    assert (2 + 3*x*y).has_free(3*x)
    raises(TypeError, lambda: x.has_free({x, y}))
    s = FiniteSet(1, 2)
    assert Piecewise((s, x > 3), (4, True)).has_free(s)
    assert not Piecewise((1, x > 3), (4, True)).has_free(s)
    # can't make set of these, but fallback will handle
    raises(TypeError, lambda: x.has_free(y, []))


def test_has_xfree():
    assert (x + 1).has_xfree({x})
    assert ((x + 1)**2).has_xfree({x + 1})
    assert not (x + y + 1).has_xfree({x + 1})
    raises(TypeError, lambda: x.has_xfree(x))
    raises(TypeError, lambda: x.has_xfree([x]))


def test_issue_5300():
    x = Symbol('x', commutative=False)
    assert x*sqrt(2)/sqrt(6) == x*sqrt(3)/3


def test_floordiv():
    from sympy.functions.elementary.integers import floor
    assert x // y == floor(x / y)


def test_as_coeff_Mul():
    assert Integer(3).as_coeff_Mul() == (Integer(3), Integer(1))
    assert Rational(3, 4).as_coeff_Mul() == (Rational(3, 4), Integer(1))
    assert Float(5.0).as_coeff_Mul() == (Float(5.0), Integer(1))
    assert Float(0.0).as_coeff_Mul() == (Float(0.0), Integer(1))

    assert (Integer(3)*x).as_coeff_Mul() == (Integer(3), x)
    assert (Rational(3, 4)*x).as_coeff_Mul() == (Rational(3, 4), x)
    assert (Float(5.0)*x).as_coeff_Mul() == (Float(5.0), x)

    assert (Integer(3)*x*y).as_coeff_Mul() == (Integer(3), x*y)
    assert (Rational(3, 4)*x*y).as_coeff_Mul() == (Rational(3, 4), x*y)
    assert (Float(5.0)*x*y).as_coeff_Mul() == (Float(5.0), x*y)

    assert (x).as_coeff_Mul() == (S.One, x)
    assert (x*y).as_coeff_Mul() == (S.One, x*y)
    assert (-oo*x).as_coeff_Mul(rational=True) == (-1, oo*x)


def test_as_coeff_Add():
    assert Integer(3).as_coeff_Add() == (Integer(3), Integer(0))
    assert Rational(3, 4).as_coeff_Add() == (Rational(3, 4), Integer(0))
    assert Float(5.0).as_coeff_Add() == (Float(5.0), Integer(0))

    assert (Integer(3) + x).as_coeff_Add() == (Integer(3), x)
    assert (Rational(3, 4) + x).as_coeff_Add() == (Rational(3, 4), x)
    assert (Float(5.0) + x).as_coeff_Add() == (Float(5.0), x)
    assert (Float(5.0) + x).as_coeff_Add(rational=True) == (0, Float(5.0) + x)

    assert (Integer(3) + x + y).as_coeff_Add() == (Integer(3), x + y)
    assert (Rational(3, 4) + x + y).as_coeff_Add() == (Rational(3, 4), x + y)
    assert (Float(5.0) + x + y).as_coeff_Add() == (Float(5.0), x + y)

    assert (x).as_coeff_Add() == (S.Zero, x)
    assert (x*y).as_coeff_Add() == (S.Zero, x*y)


def test_expr_sorting():

    exprs = [1/x**2, 1/x, sqrt(sqrt(x)), sqrt(x), x, sqrt(x)**3, x**2]
    assert sorted(exprs, key=default_sort_key) == exprs

    exprs = [x, 2*x, 2*x**2, 2*x**3, x**n, 2*x**n, sin(x), sin(x)**n,
             sin(x**2), cos(x), cos(x**2), tan(x)]
    assert sorted(exprs, key=default_sort_key) == exprs

    exprs = [x + 1, x**2 + x + 1, x**3 + x**2 + x + 1]
    assert sorted(exprs, key=default_sort_key) == exprs

    exprs = [S(4), x - 3*I/2, x + 3*I/2, x - 4*I + 1, x + 4*I + 1]
    assert sorted(exprs, key=default_sort_key) == exprs

    exprs = [f(1), f(2), f(3), f(1, 2, 3), g(1), g(2), g(3), g(1, 2, 3)]
    assert sorted(exprs, key=default_sort_key) == exprs

    exprs = [f(x), g(x), exp(x), sin(x), cos(x), factorial(x)]
    assert sorted(exprs, key=default_sort_key) == exprs

    exprs = [Tuple(x, y), Tuple(x, z), Tuple(x, y, z)]
    assert sorted(exprs, key=default_sort_key) == exprs

    exprs = [[3], [1, 2]]
    assert sorted(exprs, key=default_sort_key) == exprs

    exprs = [[1, 2], [2, 3]]
    assert sorted(exprs, key=default_sort_key) == exprs

    exprs = [[1, 2], [1, 2, 3]]
    assert sorted(exprs, key=default_sort_key) == exprs

    exprs = [{x: -y}, {x: y}]
    assert sorted(exprs, key=default_sort_key) == exprs

    exprs = [{1}, {1, 2}]
    assert sorted(exprs, key=default_sort_key) == exprs

    a, b = exprs = [Dummy('x'), Dummy('x')]
    assert sorted([b, a], key=default_sort_key) == exprs


def test_as_ordered_factors():

    assert x.as_ordered_factors() == [x]
    assert (2*x*x**n*sin(x)*cos(x)).as_ordered_factors() \
        == [Integer(2), x, x**n, sin(x), cos(x)]

    args = [f(1), f(2), f(3), f(1, 2, 3), g(1), g(2), g(3), g(1, 2, 3)]
    expr = Mul(*args)

    assert expr.as_ordered_factors() == args

    A, B = symbols('A,B', commutative=False)

    assert (A*B).as_ordered_factors() == [A, B]
    assert (B*A).as_ordered_factors() == [B, A]


def test_as_ordered_terms():

    assert x.as_ordered_terms() == [x]
    assert (sin(x)**2*cos(x) + sin(x)*cos(x)**2 + 1).as_ordered_terms() \
        == [sin(x)**2*cos(x), sin(x)*cos(x)**2, 1]

    args = [f(1), f(2), f(3), f(1, 2, 3), g(1), g(2), g(3), g(1, 2, 3)]
    expr = Add(*args)

    assert expr.as_ordered_terms() == args

    assert (1 + 4*sqrt(3)*pi*x).as_ordered_terms() == [4*pi*x*sqrt(3), 1]

    assert ( 2 + 3*I).as_ordered_terms() == [2, 3*I]
    assert (-2 + 3*I).as_ordered_terms() == [-2, 3*I]
    assert ( 2 - 3*I).as_ordered_terms() == [2, -3*I]
    assert (-2 - 3*I).as_ordered_terms() == [-2, -3*I]

    assert ( 4 + 3*I).as_ordered_terms() == [4, 3*I]
    assert (-4 + 3*I).as_ordered_terms() == [-4, 3*I]
    assert ( 4 - 3*I).as_ordered_terms() == [4, -3*I]
    assert (-4 - 3*I).as_ordered_terms() == [-4, -3*I]

    e = x**2*y**2 + x*y**4 + y + 2

    assert e.as_ordered_terms(order="lex") == [x**2*y**2, x*y**4, y, 2]
    assert e.as_ordered_terms(order="grlex") == [x*y**4, x**2*y**2, y, 2]
    assert e.as_ordered_terms(order="rev-lex") == [2, y, x*y**4, x**2*y**2]
    assert e.as_ordered_terms(order="rev-grlex") == [2, y, x**2*y**2, x*y**4]

    k = symbols('k')
    assert k.as_ordered_terms(data=True) == ([(k, ((1.0, 0.0), (1,), ()))], [k])


def test_sort_key_atomic_expr():
    from sympy.physics.units import m, s
    assert sorted([-m, s], key=lambda arg: arg.sort_key()) == [-m, s]


def test_eval_interval():
    assert exp(x)._eval_interval(*Tuple(x, 0, 1)) == exp(1) - exp(0)

    # issue 4199
    a = x/y
    raises(NotImplementedError, lambda: a._eval_interval(x, S.Zero, oo)._eval_interval(y, oo, S.Zero))
    raises(NotImplementedError, lambda: a._eval_interval(x, S.Zero, oo)._eval_interval(y, S.Zero, oo))
    a = x - y
    raises(NotImplementedError, lambda: a._eval_interval(x, S.One, oo)._eval_interval(y, oo, S.One))
    raises(ValueError, lambda: x._eval_interval(x, None, None))
    a = -y*Heaviside(x - y)
    assert a._eval_interval(x, -oo, oo) == -y
    assert a._eval_interval(x, oo, -oo) == y


def test_eval_interval_zoo():
    # Test that limit is used when zoo is returned
    assert Si(1/x)._eval_interval(x, S.Zero, S.One) == -pi/2 + Si(1)


def test_primitive():
    assert (3*(x + 1)**2).primitive() == (3, (x + 1)**2)
    assert (6*x + 2).primitive() == (2, 3*x + 1)
    assert (x/2 + 3).primitive() == (S.Half, x + 6)
    eq = (6*x + 2)*(x/2 + 3)
    assert eq.primitive()[0] == 1
    eq = (2 + 2*x)**2
    assert eq.primitive()[0] == 1
    assert (4.0*x).primitive() == (1, 4.0*x)
    assert (4.0*x + y/2).primitive() == (S.Half, 8.0*x + y)
    assert (-2*x).primitive() == (2, -x)
    assert Add(5*z/7, 0.5*x, 3*y/2, evaluate=False).primitive() == \
        (S.One/14, 7.0*x + 21*y + 10*z)
    for i in [S.Infinity, S.NegativeInfinity, S.ComplexInfinity]:
        assert (i + x/3).primitive() == \
            (S.One/3, i + x)
    assert (S.Infinity + 2*x/3 + 4*y/7).primitive() == \
        (S.One/21, 14*x + 12*y + oo)
    assert S.Zero.primitive() == (S.One, S.Zero)


def test_issue_5843():
    a = 1 + x
    assert (2*a).extract_multiplicatively(a) == 2
    assert (4*a).extract_multiplicatively(2*a) == 2
    assert ((3*a)*(2*a)).extract_multiplicatively(a) == 6*a


def test_is_constant():
    from sympy.solvers.solvers import checksol
    assert Sum(x, (x, 1, 10)).is_constant() is True
    assert Sum(x, (x, 1, n)).is_constant() is False
    assert Sum(x, (x, 1, n)).is_constant(y) is True
    assert Sum(x, (x, 1, n)).is_constant(n) is False
    assert Sum(x, (x, 1, n)).is_constant(x) is True
    eq = a*cos(x)**2 + a*sin(x)**2 - a
    assert eq.is_constant() is True
    assert eq.subs({x: pi, a: 2}) == eq.subs({x: pi, a: 3}) == 0
    assert x.is_constant() is False
    assert x.is_constant(y) is True
    assert log(x/y).is_constant() is False

    assert checksol(x, x, Sum(x, (x, 1, n))) is False
    assert checksol(x, x, Sum(x, (x, 1, n))) is False
    assert f(1).is_constant
    assert checksol(x, x, f(x)) is False

    assert Pow(x, S.Zero, evaluate=False).is_constant() is True  # == 1
    assert Pow(S.Zero, x, evaluate=False).is_constant() is False  # == 0 or 1
    assert (2**x).is_constant() is False
    assert Pow(S(2), S(3), evaluate=False).is_constant() is True

    z1, z2 = symbols('z1 z2', zero=True)
    assert (z1 + 2*z2).is_constant() is True

    assert meter.is_constant() is True
    assert (3*meter).is_constant() is True
    assert (x*meter).is_constant() is False


def test_equals():
    assert (-3 - sqrt(5) + (-sqrt(10)/2 - sqrt(2)/2)**2).equals(0)
    assert (x**2 - 1).equals((x + 1)*(x - 1))
    assert (cos(x)**2 + sin(x)**2).equals(1)
    assert (a*cos(x)**2 + a*sin(x)**2).equals(a)
    r = sqrt(2)
    assert (-1/(r + r*x) + 1/r/(1 + x)).equals(0)
    assert factorial(x + 1).equals((x + 1)*factorial(x))
    assert sqrt(3).equals(2*sqrt(3)) is False
    assert (sqrt(5)*sqrt(3)).equals(sqrt(3)) is False
    assert (sqrt(5) + sqrt(3)).equals(0) is False
    assert (sqrt(5) + pi).equals(0) is False
    assert meter.equals(0) is False
    assert (3*meter**2).equals(0) is False
    eq = -(-1)**(S(3)/4)*6**(S.One/4) + (-6)**(S.One/4)*I
    if eq != 0:  # if canonicalization makes this zero, skip the test
        assert eq.equals(0)
    assert sqrt(x).equals(0) is False

    # from integrate(x*sqrt(1 + 2*x), x);
    # diff is zero only when assumptions allow
    i = 2*sqrt(2)*x**(S(5)/2)*(1 + 1/(2*x))**(S(5)/2)/5 + \
        2*sqrt(2)*x**(S(3)/2)*(1 + 1/(2*x))**(S(5)/2)/(-6 - 3/x)
    ans = sqrt(2*x + 1)*(6*x**2 + x - 1)/15
    diff = i - ans
    assert diff.equals(0) is None  # should be False, but previously this was False due to wrong intermediate result
    assert diff.subs(x, Rational(-1, 2)/2) == 7*sqrt(2)/120
    # there are regions for x for which the expression is True, for
    # example, when x < -1/2 or x > 0 the expression is zero
    p = Symbol('p', positive=True)
    assert diff.subs(x, p).equals(0) is True
    assert diff.subs(x, -1).equals(0) is True

    # prove via minimal_polynomial or self-consistency
    eq = sqrt(1 + sqrt(3)) + sqrt(3 + 3*sqrt(3)) - sqrt(10 + 6*sqrt(3))
    assert eq.equals(0)
    q = 3**Rational(1, 3) + 3
    p = expand(q**3)**Rational(1, 3)
    assert (p - q).equals(0)

    # issue 6829
    # eq = q*x + q/4 + x**4 + x**3 + 2*x**2 - S.One/3
    # z = eq.subs(x, solve(eq, x)[0])
    q = symbols('q')
    z = (q*(-sqrt(-2*(-(q - S(7)/8)**S(2)/8 - S(2197)/13824)**(S.One/3) -
    S(13)/12)/2 - sqrt((2*q - S(7)/4)/sqrt(-2*(-(q - S(7)/8)**S(2)/8 -
    S(2197)/13824)**(S.One/3) - S(13)/12) + 2*(-(q - S(7)/8)**S(2)/8 -
    S(2197)/13824)**(S.One/3) - S(13)/6)/2 - S.One/4) + q/4 + (-sqrt(-2*(-(q
    - S(7)/8)**S(2)/8 - S(2197)/13824)**(S.One/3) - S(13)/12)/2 - sqrt((2*q
    - S(7)/4)/sqrt(-2*(-(q - S(7)/8)**S(2)/8 - S(2197)/13824)**(S.One/3) -
    S(13)/12) + 2*(-(q - S(7)/8)**S(2)/8 - S(2197)/13824)**(S.One/3) -
    S(13)/6)/2 - S.One/4)**4 + (-sqrt(-2*(-(q - S(7)/8)**S(2)/8 -
    S(2197)/13824)**(S.One/3) - S(13)/12)/2 - sqrt((2*q -
    S(7)/4)/sqrt(-2*(-(q - S(7)/8)**S(2)/8 - S(2197)/13824)**(S.One/3) -
    S(13)/12) + 2*(-(q - S(7)/8)**S(2)/8 - S(2197)/13824)**(S.One/3) -
    S(13)/6)/2 - S.One/4)**3 + 2*(-sqrt(-2*(-(q - S(7)/8)**S(2)/8 -
    S(2197)/13824)**(S.One/3) - S(13)/12)/2 - sqrt((2*q -
    S(7)/4)/sqrt(-2*(-(q - S(7)/8)**S(2)/8 - S(2197)/13824)**(S.One/3) -
    S(13)/12) + 2*(-(q - S(7)/8)**S(2)/8 - S(2197)/13824)**(S.One/3) -
    S(13)/6)/2 - S.One/4)**2 - Rational(1, 3))
    assert z.equals(0)


def test_random():
    from sympy.functions.combinatorial.numbers import lucas
    from sympy.simplify.simplify import posify
    assert posify(x)[0]._random() is not None
    assert lucas(n)._random(2, -2, 0, -1, 1) is None

    # issue 8662
    assert Piecewise((Max(x, y), z))._random() is None


def test_round():
    assert str(Float('0.1249999').round(2)) == '0.12'
    d20 = 12345678901234567890
    ans = S(d20).round(2)
    assert ans.is_Integer and ans == d20
    ans = S(d20).round(-2)
    assert ans.is_Integer and ans == 12345678901234567900
    assert str(S('1/7').round(4)) == '0.1429'
    assert str(S('.[12345]').round(4)) == '0.1235'
    assert str(S('.1349').round(2)) == '0.13'
    n = S(12345)
    ans = n.round()
    assert ans.is_Integer
    assert ans == n
    ans = n.round(1)
    assert ans.is_Integer
    assert ans == n
    ans = n.round(4)
    assert ans.is_Integer
    assert ans == n
    assert n.round(-1) == 12340

    r = Float(str(n)).round(-4)
    assert r == 10000.0

    assert n.round(-5) == 0

    assert str((pi + sqrt(2)).round(2)) == '4.56'
    assert (10*(pi + sqrt(2))).round(-1) == 50.0
    raises(TypeError, lambda: round(x + 2, 2))
    assert str(S(2.3).round(1)) == '2.3'
    # rounding in SymPy (as in Decimal) should be
    # exact for the given precision; we check here
    # that when a 5 follows the last digit that
    # the rounded digit will be even.
    for i in range(-99, 100):
        # construct a decimal that ends in 5, e.g. 123 -> 0.1235
        s = str(abs(i))
        p = len(s)  # we are going to round to the last digit of i
        n = '0.%s5' % s  # put a 5 after i's digits
        j = p + 2  # 2 for '0.'
        if i < 0:  # 1 for '-'
            j += 1
            n = '-' + n
        v = str(Float(n).round(p))[:j]  # pertinent digits
        if v.endswith('.'):
            continue  # it ends with 0 which is even
        L = int(v[-1])  # last digit
        assert L % 2 == 0, (n, '->', v)

    assert (Float(.3, 3) + 2*pi).round() == 7
    assert (Float(.3, 3) + 2*pi*100).round() == 629
    assert (pi + 2*E*I).round() == 3 + 5*I
    # don't let request for extra precision give more than
    # what is known (in this case, only 3 digits)
    assert str((Float(.03, 3) + 2*pi/100).round(5)) == '0.0928'
    assert str((Float(.03, 3) + 2*pi/100).round(4)) == '0.0928'

    assert S.Zero.round() == 0

    a = (Add(1, Float('1.' + '9'*27, ''), evaluate=False))
    assert a.round(10) == Float('3.000000000000000000000000000', '')
    assert a.round(25) == Float('3.000000000000000000000000000', '')
    assert a.round(26) == Float('3.000000000000000000000000000', '')
    assert a.round(27) == Float('2.999999999999999999999999999', '')
    assert a.round(30) == Float('2.999999999999999999999999999', '')
    #assert a.round(10) == Float('3.0000000000', '')
    #assert a.round(25) == Float('3.0000000000000000000000000', '')
    #assert a.round(26) == Float('3.00000000000000000000000000', '')
    #assert a.round(27) == Float('2.999999999999999999999999999', '')
    #assert a.round(30) == Float('2.999999999999999999999999999', '')

    # XXX: Should round set the precision of the result?
    #      The previous version of the tests above is this but they only pass
    #      because Floats with unequal precision compare equal:
    #
    # assert a.round(10) == Float('3.0000000000', '')
    # assert a.round(25) == Float('3.0000000000000000000000000', '')
    # assert a.round(26) == Float('3.00000000000000000000000000', '')
    # assert a.round(27) == Float('2.999999999999999999999999999', '')
    # assert a.round(30) == Float('2.999999999999999999999999999', '')

    raises(TypeError, lambda: x.round())
    raises(TypeError, lambda: f(1).round())

    # exact magnitude of 10
    assert str(S.One.round()) == '1'
    assert str(S(100).round()) == '100'

    # applied to real and imaginary portions
    assert (2*pi + E*I).round() == 6 + 3*I
    assert (2*pi + I/10).round() == 6
    assert (pi/10 + 2*I).round() == 2*I
    # the lhs re and im parts are Float with dps of 2
    # and those on the right have dps of 15 so they won't compare
    # equal unless we use string or compare components (which will
    # then coerce the floats to the same precision) or re-create
    # the floats
    assert str((pi/10 + E*I).round(2)) == '0.31 + 2.72*I'
    assert str((pi/10 + E*I).round(2).as_real_imag()) == '(0.31, 2.72)'
    assert str((pi/10 + E*I).round(2)) == '0.31 + 2.72*I'

    # issue 6914
    assert (I**(I + 3)).round(3) == Float('-0.208', '')*I

    # issue 8720
    assert S(-123.6).round() == -124
    assert S(-1.5).round() == -2
    assert S(-100.5).round() == -100
    assert S(-1.5 - 10.5*I).round() == -2 - 10*I

    # issue 7961
    assert str(S(0.006).round(2)) == '0.01'
    assert str(S(0.00106).round(4)) == '0.0011'

    # issue 8147
    assert S.NaN.round() is S.NaN
    assert S.Infinity.round() is S.Infinity
    assert S.NegativeInfinity.round() is S.NegativeInfinity
    assert S.ComplexInfinity.round() is S.ComplexInfinity

    # check that types match
    for i in range(2):
        fi = float(i)
        # 2 args
        assert all(type(round(i, p)) is int for p in (-1, 0, 1))
        assert all(S(i).round(p).is_Integer for p in (-1, 0, 1))
        assert all(type(round(fi, p)) is float for p in (-1, 0, 1))
        assert all(S(fi).round(p).is_Float for p in (-1, 0, 1))
        # 1 arg (p is None)
        assert type(round(i)) is int
        assert S(i).round().is_Integer
        assert type(round(fi)) is int
        assert S(fi).round().is_Integer

        # issue 25698
        n = 6000002
        assert int(n*(log(n) + log(log(n)))) == 110130079
        one = cos(2)**2 + sin(2)**2
        eq = exp(one*I*pi)
        qr, qi = eq.as_real_imag()
        assert qi.round(2) == 0.0
        assert eq.round(2) == -1.0
        eq = one - 1/S(10**120)
        assert S.true not in (eq > 1, eq < 1)
        assert int(eq) == int(.9) == 0
        assert int(-eq) == int(-.9) == 0


def test_held_expression_UnevaluatedExpr():
    x = symbols("x")
    he = UnevaluatedExpr(1/x)
    e1 = x*he

    assert isinstance(e1, Mul)
    assert e1.args == (x, he)
    assert e1.doit() == 1
    assert UnevaluatedExpr(Derivative(x, x)).doit(deep=False
        ) == Derivative(x, x)
    assert UnevaluatedExpr(Derivative(x, x)).doit() == 1

    xx = Mul(x, x, evaluate=False)
    assert xx != x**2

    ue2 = UnevaluatedExpr(xx)
    assert isinstance(ue2, UnevaluatedExpr)
    assert ue2.args == (xx,)
    assert ue2.doit() == x**2
    assert ue2.doit(deep=False) == xx

    x2 = UnevaluatedExpr(2)*2
    assert type(x2) is Mul
    assert x2.args == (2, UnevaluatedExpr(2))

def test_round_exception_nostr():
    # Don't use the string form of the expression in the round exception, as
    # it's too slow
    s = Symbol('bad')
    try:
        s.round()
    except TypeError as e:
        assert 'bad' not in str(e)
    else:
        # Did not raise
        raise AssertionError("Did not raise")


def test_extract_branch_factor():
    assert exp_polar(2.0*I*pi).extract_branch_factor() == (1, 1)


def test_identity_removal():
    assert Add.make_args(x + 0) == (x,)
    assert Mul.make_args(x*1) == (x,)


def test_float_0():
    assert Float(0.0) + 1 == Float(1.0)


@XFAIL
def test_float_0_fail():
    assert Float(0.0)*x == Float(0.0)
    assert (x + Float(0.0)).is_Add


def test_issue_6325():
    ans = (b**2 + z**2 - (b*(a + b*t) + z*(c + t*z))**2/(
        (a + b*t)**2 + (c + t*z)**2))/sqrt((a + b*t)**2 + (c + t*z)**2)
    e = sqrt((a + b*t)**2 + (c + z*t)**2)
    assert diff(e, t, 2) == ans
    assert e.diff(t, 2) == ans
    assert diff(e, t, 2, simplify=False) != ans


def test_issue_7426():
    f1 = a % c
    f2 = x % z
    assert f1.equals(f2) is None


def test_issue_11122():
    x = Symbol('x', extended_positive=False)
    assert unchanged(Gt, x, 0)  # (x > 0)
    # (x > 0) should remain unevaluated after PR #16956

    x = Symbol('x', positive=False, real=True)
    assert (x > 0) is S.false


def test_issue_10651():
    x = Symbol('x', real=True)
    e1 = (-1 + x)/(1 - x)
    e3 = (4*x**2 - 4)/((1 - x)*(1 + x))
    e4 = 1/(cos(x)**2) - (tan(x))**2
    x = Symbol('x', positive=True)
    e5 = (1 + x)/x
    assert e1.is_constant() is None
    assert e3.is_constant() is None
    assert e4.is_constant() is None
    assert e5.is_constant() is False


def test_issue_10161():
    x = symbols('x', real=True)
    assert x*abs(x)*abs(x) == x**3


def test_issue_10755():
    x = symbols('x')
    raises(TypeError, lambda: int(log(x)))
    raises(TypeError, lambda: log(x).round(2))


def test_issue_11877():
    x = symbols('x')
    assert integrate(log(S.Half - x), (x, 0, S.Half)) == Rational(-1, 2) -log(2)/2


def test_normal():
    x = symbols('x')
    e = Mul(S.Half, 1 + x, evaluate=False)
    assert e.normal() == e


def test_expr():
    x = symbols('x')
    raises(TypeError, lambda: tan(x).series(x, 2, oo, "+"))


def test_ExprBuilder():
    eb = ExprBuilder(Mul)
    eb.args.extend([x, x])
    assert eb.build() == x**2


def test_issue_22020():
    from sympy.parsing.sympy_parser import parse_expr
    x = parse_expr("log((2*V/3-V)/C)/-(R+r)*C")
    y = parse_expr("log((2*V/3-V)/C)/-(R+r)*2")
    assert x.equals(y) is False


def test_non_string_equality():
    # Expressions should not compare equal to strings
    x = symbols('x')
    one = sympify(1)
    assert (x == 'x') is False
    assert (x != 'x') is True
    assert (one == '1') is False
    assert (one != '1') is True
    assert (x + 1 == 'x + 1') is False
    assert (x + 1 != 'x + 1') is True

    # Make sure == doesn't try to convert the resulting expression to a string
    # (e.g., by calling sympify() instead of _sympify())

    class BadRepr:
        def __repr__(self):
            raise RuntimeError

    assert (x == BadRepr()) is False
    assert (x != BadRepr()) is True


def test_21494():
    from sympy.testing.pytest import warns_deprecated_sympy

    with warns_deprecated_sympy():
        assert x.expr_free_symbols == {x}

    with warns_deprecated_sympy():
        assert Basic().expr_free_symbols == set()

    with warns_deprecated_sympy():
        assert S(2).expr_free_symbols == {S(2)}

    with warns_deprecated_sympy():
        assert Indexed("A", x).expr_free_symbols == {Indexed("A", x)}

    with warns_deprecated_sympy():
        assert Subs(x, x, 0).expr_free_symbols == set()


def test_Expr__eq__iterable_handling():
    assert x != range(3)


def test_format():
    assert '{:1.2f}'.format(S.Zero) == '0.00'
    assert '{:+3.0f}'.format(S(3)) == ' +3'
    assert '{:23.20f}'.format(pi) == ' 3.14159265358979323846'
    assert '{:50.48f}'.format(exp(sin(1))) == '2.319776824715853173956590377503266813254904772376'


def test_issue_24045():
    assert powsimp(exp(a)/((c*a - c*b)*(Float(1.0)*c*a - Float(1.0)*c*b)))  # doesn't raise


def test__unevaluated_Mul():
    A, B = symbols('A B', commutative=False)
    assert _unevaluated_Mul(x, A, B, S(2), A).args == (2, x, A, B, A)
    assert _unevaluated_Mul(-x*A*B, S(2), A).args == (-2, x, A, B, A)


def test_Float_zero_division_error():
    # issue 27165
    assert Float('1.7567e-1417').round(15) == Float(0)
