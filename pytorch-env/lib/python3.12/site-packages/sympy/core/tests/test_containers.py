from collections import defaultdict

from sympy.core.basic import Basic
from sympy.core.containers import (Dict, Tuple)
from sympy.core.numbers import Integer
from sympy.core.kind import NumberKind
from sympy.matrices.kind import MatrixKind
from sympy.core.singleton import S
from sympy.core.symbol import symbols
from sympy.core.sympify import sympify
from sympy.matrices.dense import Matrix
from sympy.sets.sets import FiniteSet
from sympy.core.containers import tuple_wrapper, TupleKind
from sympy.core.expr import unchanged
from sympy.core.function import Function, Lambda
from sympy.core.relational import Eq
from sympy.testing.pytest import raises
from sympy.utilities.iterables import is_sequence, iterable

from sympy.abc import x, y


def test_Tuple():
    t = (1, 2, 3, 4)
    st = Tuple(*t)
    assert set(sympify(t)) == set(st)
    assert len(t) == len(st)
    assert set(sympify(t[:2])) == set(st[:2])
    assert isinstance(st[:], Tuple)
    assert st == Tuple(1, 2, 3, 4)
    assert st.func(*st.args) == st
    p, q, r, s = symbols('p q r s')
    t2 = (p, q, r, s)
    st2 = Tuple(*t2)
    assert st2.atoms() == set(t2)
    assert st == st2.subs({p: 1, q: 2, r: 3, s: 4})
    # issue 5505
    assert all(isinstance(arg, Basic) for arg in st.args)
    assert Tuple(p, 1).subs(p, 0) == Tuple(0, 1)
    assert Tuple(p, Tuple(p, 1)).subs(p, 0) == Tuple(0, Tuple(0, 1))

    assert Tuple(t2) == Tuple(Tuple(*t2))
    assert Tuple.fromiter(t2) == Tuple(*t2)
    assert Tuple.fromiter(x for x in range(4)) == Tuple(0, 1, 2, 3)
    assert st2.fromiter(st2.args) == st2


def test_Tuple_contains():
    t1, t2 = Tuple(1), Tuple(2)
    assert t1 in Tuple(1, 2, 3, t1, Tuple(t2))
    assert t2 not in Tuple(1, 2, 3, t1, Tuple(t2))


def test_Tuple_concatenation():
    assert Tuple(1, 2) + Tuple(3, 4) == Tuple(1, 2, 3, 4)
    assert (1, 2) + Tuple(3, 4) == Tuple(1, 2, 3, 4)
    assert Tuple(1, 2) + (3, 4) == Tuple(1, 2, 3, 4)
    raises(TypeError, lambda: Tuple(1, 2) + 3)
    raises(TypeError, lambda: 1 + Tuple(2, 3))

    #the Tuple case in __radd__ is only reached when a subclass is involved
    class Tuple2(Tuple):
        def __radd__(self, other):
            return Tuple.__radd__(self, other + other)
    assert Tuple(1, 2) + Tuple2(3, 4) == Tuple(1, 2, 1, 2, 3, 4)
    assert Tuple2(1, 2) + Tuple(3, 4) == Tuple(1, 2, 3, 4)


def test_Tuple_equality():
    assert not isinstance(Tuple(1, 2), tuple)
    assert (Tuple(1, 2) == (1, 2)) is True
    assert (Tuple(1, 2) != (1, 2)) is False
    assert (Tuple(1, 2) == (1, 3)) is False
    assert (Tuple(1, 2) != (1, 3)) is True
    assert (Tuple(1, 2) == Tuple(1, 2)) is True
    assert (Tuple(1, 2) != Tuple(1, 2)) is False
    assert (Tuple(1, 2) == Tuple(1, 3)) is False
    assert (Tuple(1, 2) != Tuple(1, 3)) is True


def test_Tuple_Eq():
    assert Eq(Tuple(), Tuple()) is S.true
    assert Eq(Tuple(1), 1) is S.false
    assert Eq(Tuple(1, 2), Tuple(1)) is S.false
    assert Eq(Tuple(1), Tuple(1)) is S.true
    assert Eq(Tuple(1, 2), Tuple(1, 3)) is S.false
    assert Eq(Tuple(1, 2), Tuple(1, 2)) is S.true
    assert unchanged(Eq, Tuple(1, x), Tuple(1, 2))
    assert Eq(Tuple(1, x), Tuple(1, 2)).subs(x, 2) is S.true
    assert unchanged(Eq, Tuple(1, 2), x)
    f = Function('f')
    assert unchanged(Eq, Tuple(1), f(x))
    assert Eq(Tuple(1), f(x)).subs(x, 1).subs(f, Lambda(y, (y,))) is S.true


def test_Tuple_comparision():
    assert (Tuple(1, 3) >= Tuple(-10, 30)) is S.true
    assert (Tuple(1, 3) <= Tuple(-10, 30)) is S.false
    assert (Tuple(1, 3) >= Tuple(1, 3)) is S.true
    assert (Tuple(1, 3) <= Tuple(1, 3)) is S.true


def test_Tuple_tuple_count():
    assert Tuple(0, 1, 2, 3).tuple_count(4) == 0
    assert Tuple(0, 4, 1, 2, 3).tuple_count(4) == 1
    assert Tuple(0, 4, 1, 4, 2, 3).tuple_count(4) == 2
    assert Tuple(0, 4, 1, 4, 2, 4, 3).tuple_count(4) == 3


def test_Tuple_index():
    assert Tuple(4, 0, 1, 2, 3).index(4) == 0
    assert Tuple(0, 4, 1, 2, 3).index(4) == 1
    assert Tuple(0, 1, 4, 2, 3).index(4) == 2
    assert Tuple(0, 1, 2, 4, 3).index(4) == 3
    assert Tuple(0, 1, 2, 3, 4).index(4) == 4

    raises(ValueError, lambda: Tuple(0, 1, 2, 3).index(4))
    raises(ValueError, lambda: Tuple(4, 0, 1, 2, 3).index(4, 1))
    raises(ValueError, lambda: Tuple(0, 1, 2, 3, 4).index(4, 1, 4))


def test_Tuple_mul():
    assert Tuple(1, 2, 3)*2 == Tuple(1, 2, 3, 1, 2, 3)
    assert 2*Tuple(1, 2, 3) == Tuple(1, 2, 3, 1, 2, 3)
    assert Tuple(1, 2, 3)*Integer(2) == Tuple(1, 2, 3, 1, 2, 3)
    assert Integer(2)*Tuple(1, 2, 3) == Tuple(1, 2, 3, 1, 2, 3)

    raises(TypeError, lambda: Tuple(1, 2, 3)*S.Half)
    raises(TypeError, lambda: S.Half*Tuple(1, 2, 3))


def test_tuple_wrapper():

    @tuple_wrapper
    def wrap_tuples_and_return(*t):
        return t

    p = symbols('p')
    assert wrap_tuples_and_return(p, 1) == (p, 1)
    assert wrap_tuples_and_return((p, 1)) == (Tuple(p, 1),)
    assert wrap_tuples_and_return(1, (p, 2), 3) == (1, Tuple(p, 2), 3)


def test_iterable_is_sequence():
    ordered = [[], (), Tuple(), Matrix([[]])]
    unordered = [set()]
    not_sympy_iterable = [{}, '', '']
    assert all(is_sequence(i) for i in ordered)
    assert all(not is_sequence(i) for i in unordered)
    assert all(iterable(i) for i in ordered + unordered)
    assert all(not iterable(i) for i in not_sympy_iterable)
    assert all(iterable(i, exclude=None) for i in not_sympy_iterable)


def test_TupleKind():
    kind = TupleKind(NumberKind, MatrixKind(NumberKind))
    assert Tuple(1, Matrix([1, 2])).kind is kind
    assert Tuple(1, 2).kind is TupleKind(NumberKind, NumberKind)
    assert Tuple(1, 2).kind.element_kind == (NumberKind, NumberKind)


def test_Dict():
    x, y, z = symbols('x y z')
    d = Dict({x: 1, y: 2, z: 3})
    assert d[x] == 1
    assert d[y] == 2
    raises(KeyError, lambda: d[2])
    raises(KeyError, lambda: d['2'])
    assert len(d) == 3
    assert set(d.keys()) == {x, y, z}
    assert set(d.values()) == {S.One, S(2), S(3)}
    assert d.get(5, 'default') == 'default'
    assert d.get('5', 'default') == 'default'
    assert x in d and z in d and 5 not in d and '5' not in d
    assert d.has(x) and d.has(1)  # SymPy Basic .has method

    # Test input types
    # input - a Python dict
    # input - items as args - SymPy style
    assert (Dict({x: 1, y: 2, z: 3}) ==
            Dict((x, 1), (y, 2), (z, 3)))

    raises(TypeError, lambda: Dict(((x, 1), (y, 2), (z, 3))))
    with raises(NotImplementedError):
        d[5] = 6  # assert immutability

    assert set(
        d.items()) == {Tuple(x, S.One), Tuple(y, S(2)), Tuple(z, S(3))}
    assert set(d) == {x, y, z}
    assert str(d) == '{x: 1, y: 2, z: 3}'
    assert d.__repr__() == '{x: 1, y: 2, z: 3}'

    # Test creating a Dict from a Dict.
    d = Dict({x: 1, y: 2, z: 3})
    assert d == Dict(d)

    # Test for supporting defaultdict
    d = defaultdict(int)
    assert d[x] == 0
    assert d[y] == 0
    assert d[z] == 0
    assert Dict(d)
    d = Dict(d)
    assert len(d) == 3
    assert set(d.keys()) == {x, y, z}
    assert set(d.values()) == {S.Zero, S.Zero, S.Zero}


def test_issue_5788():
    args = [(1, 2), (2, 1)]
    for o in [Dict, Tuple, FiniteSet]:
        # __eq__ and arg handling
        if o != Tuple:
            assert o(*args) == o(*reversed(args))
        pair = [o(*args), o(*reversed(args))]
        assert sorted(pair) == sorted(pair)
        assert set(o(*args))  # doesn't fail
