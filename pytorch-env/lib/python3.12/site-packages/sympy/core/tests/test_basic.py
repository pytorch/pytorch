"""This tests sympy/core/basic.py with (ideally) no reference to subclasses
of Basic or Atom."""
import collections
from typing import TypeVar, Generic

from sympy.assumptions.ask import Q
from sympy.core.basic import (Basic, Atom, as_Basic,
    _atomic, _aresame)
from sympy.core.containers import Tuple
from sympy.core.function import Function, Lambda
from sympy.core.numbers import I, pi, Float
from sympy.core.singleton import S
from sympy.core.symbol import symbols, Symbol, Dummy
from sympy.concrete.summations import Sum
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.functions.special.gamma_functions import gamma
from sympy.integrals.integrals import Integral
from sympy.functions.elementary.exponential import exp
from sympy.testing.pytest import raises, warns_deprecated_sympy

b1 = Basic()
b2 = Basic(b1)
b3 = Basic(b2)
b21 = Basic(b2, b1)
T = TypeVar('T')


def test__aresame():
    assert not _aresame(Basic(Tuple()), Basic())
    for i, j in [(S(2), S(2.)), (1., Float(1))]:
        for do in range(2):
            assert not _aresame(Basic(i), Basic(j))
            assert not _aresame(i, j)
            i, j = j, i


def test_structure():
    assert b21.args == (b2, b1)
    assert b21.func(*b21.args) == b21
    assert bool(b1)


def test_immutable():
    assert not hasattr(b1, '__dict__')
    with raises(AttributeError):
        b1.x = 1


def test_equality():
    instances = [b1, b2, b3, b21, Basic(b1, b1, b1), Basic]
    for i, b_i in enumerate(instances):
        for j, b_j in enumerate(instances):
            assert (b_i == b_j) == (i == j)
            assert (b_i != b_j) == (i != j)

    assert Basic() != []
    assert not(Basic() == [])
    assert Basic() != 0
    assert not(Basic() == 0)

    class Foo:
        """
        Class that is unaware of Basic, and relies on both classes returning
        the NotImplemented singleton for equivalence to evaluate to False.

        """

    b = Basic()
    foo = Foo()

    assert b != foo
    assert foo != b
    assert not b == foo
    assert not foo == b

    class Bar:
        """
        Class that considers itself equal to any instance of Basic, and relies
        on Basic returning the NotImplemented singleton in order to achieve
        a symmetric equivalence relation.

        """
        def __eq__(self, other):
            if isinstance(other, Basic):
                return True
            return NotImplemented

        def __ne__(self, other):
            return not self == other

    bar = Bar()

    assert b == bar
    assert bar == b
    assert not b != bar
    assert not bar != b


def test_matches_basic():
    instances = [Basic(b1, b1, b2), Basic(b1, b2, b1), Basic(b2, b1, b1),
                 Basic(b1, b2), Basic(b2, b1), b2, b1]
    for i, b_i in enumerate(instances):
        for j, b_j in enumerate(instances):
            if i == j:
                assert b_i.matches(b_j) == {}
            else:
                assert b_i.matches(b_j) is None
    assert b1.match(b1) == {}


def test_has():
    assert b21.has(b1)
    assert b21.has(b3, b1)
    assert b21.has(Basic)
    assert not b1.has(b21, b3)
    assert not b21.has()
    assert not b21.has(str)
    assert not Symbol("x").has("x")


def test_subs():
    assert b21.subs(b2, b1) == Basic(b1, b1)
    assert b21.subs(b2, b21) == Basic(b21, b1)
    assert b3.subs(b2, b1) == b2

    assert b21.subs([(b2, b1), (b1, b2)]) == Basic(b2, b2)

    assert b21.subs({b1: b2, b2: b1}) == Basic(b2, b2)
    assert b21.subs(collections.ChainMap({b1: b2}, {b2: b1})) == Basic(b2, b2)
    assert b21.subs(collections.OrderedDict([(b2, b1), (b1, b2)])) == Basic(b2, b2)

    raises(ValueError, lambda: b21.subs('bad arg'))
    raises(ValueError, lambda: b21.subs(b1, b2, b3))
    # dict(b1=foo) creates a string 'b1' but leaves foo unchanged; subs
    # will convert the first to a symbol but will raise an error if foo
    # cannot be sympified; sympification is strict if foo is not string
    raises(ValueError, lambda: b21.subs(b1='bad arg'))

    assert Symbol("text").subs({"text": b1}) == b1
    assert Symbol("s").subs({"s": 1}) == 1


def test_subs_with_unicode_symbols():
    expr = Symbol('var1')
    replaced = expr.subs('var1', 'x')
    assert replaced.name == 'x'

    replaced = expr.subs('var1', 'x')
    assert replaced.name == 'x'


def test_atoms():
    assert b21.atoms() == {Basic()}


def test_free_symbols_empty():
    assert b21.free_symbols == set()


def test_doit():
    assert b21.doit() == b21
    assert b21.doit(deep=False) == b21


def test_S():
    assert repr(S) == 'S'


def test_xreplace():
    assert b21.xreplace({b2: b1}) == Basic(b1, b1)
    assert b21.xreplace({b2: b21}) == Basic(b21, b1)
    assert b3.xreplace({b2: b1}) == b2
    assert Basic(b1, b2).xreplace({b1: b2, b2: b1}) == Basic(b2, b1)
    assert Atom(b1).xreplace({b1: b2}) == Atom(b1)
    assert Atom(b1).xreplace({Atom(b1): b2}) == b2
    raises(TypeError, lambda: b1.xreplace())
    raises(TypeError, lambda: b1.xreplace([b1, b2]))
    for f in (exp, Function('f')):
        assert f.xreplace({}) == f
        assert f.xreplace({}, hack2=True) == f
        assert f.xreplace({f: b1}) == b1
        assert f.xreplace({f: b1}, hack2=True) == b1


def test_sorted_args():
    x = symbols('x')
    assert b21._sorted_args == b21.args
    raises(AttributeError, lambda: x._sorted_args)

def test_call():
    x, y = symbols('x y')
    # See the long history of this in issues 5026 and 5105.

    raises(TypeError, lambda: sin(x)({ x : 1, sin(x) : 2}))
    raises(TypeError, lambda: sin(x)(1))

    # No effect as there are no callables
    assert sin(x).rcall(1) == sin(x)
    assert (1 + sin(x)).rcall(1) == 1 + sin(x)

    # Effect in the pressence of callables
    l = Lambda(x, 2*x)
    assert (l + x).rcall(y) == 2*y + x
    assert (x**l).rcall(2) == x**4
    # TODO UndefinedFunction does not subclass Expr
    #f = Function('f')
    #assert (2*f)(x) == 2*f(x)

    assert (Q.real & Q.positive).rcall(x) == Q.real(x) & Q.positive(x)


def test_rewrite():
    x, y, z = symbols('x y z')
    a, b = symbols('a b')
    f1 = sin(x) + cos(x)
    assert f1.rewrite(cos,exp) == exp(I*x)/2 + sin(x) + exp(-I*x)/2
    assert f1.rewrite([cos],sin) == sin(x) + sin(x + pi/2, evaluate=False)
    f2 = sin(x) + cos(y)/gamma(z)
    assert f2.rewrite(sin,exp) == -I*(exp(I*x) - exp(-I*x))/2 + cos(y)/gamma(z)

    assert f1.rewrite() == f1

def test_literal_evalf_is_number_is_zero_is_comparable():
    x = symbols('x')
    f = Function('f')

    # issue 5033
    assert f.is_number is False
    # issue 6646
    assert f(1).is_number is False
    i = Integral(0, (x, x, x))
    # expressions that are symbolically 0 can be difficult to prove
    # so in case there is some easy way to know if something is 0
    # it should appear in the is_zero property for that object;
    # if is_zero is true evalf should always be able to compute that
    # zero
    assert i.n() == 0
    assert i.is_zero
    assert i.is_number is False
    assert i.evalf(2, strict=False) == 0

    # issue 10268
    n = sin(1)**2 + cos(1)**2 - 1
    assert n.is_comparable is False
    assert n.n(2).is_comparable is False
    assert n.n(2).n(2).is_comparable


def test_as_Basic():
    assert as_Basic(1) is S.One
    assert as_Basic(()) == Tuple()
    raises(TypeError, lambda: as_Basic([]))


def test_atomic():
    g, h = map(Function, 'gh')
    x = symbols('x')
    assert _atomic(g(x + h(x))) == {g(x + h(x))}
    assert _atomic(g(x + h(x)), recursive=True) == {h(x), x, g(x + h(x))}
    assert _atomic(1) == set()
    assert _atomic(Basic(S(1), S(2))) == set()


def test_as_dummy():
    u, v, x, y, z, _0, _1 = symbols('u v x y z _0 _1')
    assert Lambda(x, x + 1).as_dummy() == Lambda(_0, _0 + 1)
    assert Lambda(x, x + _0).as_dummy() == Lambda(_1, _0 + _1)
    eq = (1 + Sum(x, (x, 1, x)))
    ans = 1 + Sum(_0, (_0, 1, x))
    once = eq.as_dummy()
    assert once == ans
    twice = once.as_dummy()
    assert twice == ans
    assert Integral(x + _0, (x, x + 1), (_0, 1, 2)
        ).as_dummy() == Integral(_0 + _1, (_0, x + 1), (_1, 1, 2))
    for T in (Symbol, Dummy):
        d = T('x', real=True)
        D = d.as_dummy()
        assert D != d and D.func == Dummy and D.is_real is None
    assert Dummy().as_dummy().is_commutative
    assert Dummy(commutative=False).as_dummy().is_commutative is False


def test_canonical_variables():
    x, i0, i1 = symbols('x _:2')
    assert Integral(x, (x, x + 1)).canonical_variables == {x: i0}
    assert Integral(x, (x, x + 1), (i0, 1, 2)).canonical_variables == {
        x: i0, i0: i1}
    assert Integral(x, (x, x + i0)).canonical_variables == {x: i1}


def test_replace_exceptions():
    from sympy.core.symbol import Wild
    x, y = symbols('x y')
    e = (x**2 + x*y)
    raises(TypeError, lambda: e.replace(sin, 2))
    b = Wild('b')
    c = Wild('c')
    raises(TypeError, lambda: e.replace(b*c, c.is_real))
    raises(TypeError, lambda: e.replace(b.is_real, 1))
    raises(TypeError, lambda: e.replace(lambda d: d.is_Number, 1))


def test_ManagedProperties():
    # ManagedProperties is now deprecated. Here we do our best to check that if
    # someone is using it then it does work in the way that it previously did
    # but gives a deprecation warning.
    from sympy.core.assumptions import ManagedProperties

    myclasses = []

    class MyMeta(ManagedProperties):
        def __init__(cls, *args, **kwargs):
            myclasses.append('executed')
            super().__init__(*args, **kwargs)

    code = """
class MySubclass(Basic, metaclass=MyMeta):
    pass
"""
    with warns_deprecated_sympy():
        exec(code)

    assert myclasses == ['executed']


def test_generic():
    # https://github.com/sympy/sympy/issues/25399
    class A(Symbol, Generic[T]):
        pass

    class B(A[T]):
        pass
