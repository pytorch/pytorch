from sympy.core.expr import unchanged
from sympy.core.singleton import S
from sympy.core.symbol import Symbol
from sympy.sets.contains import Contains
from sympy.sets.fancysets import Interval
from sympy.sets.powerset import PowerSet
from sympy.sets.sets import FiniteSet
from sympy.testing.pytest import raises, XFAIL


def test_powerset_creation():
    assert unchanged(PowerSet, FiniteSet(1, 2))
    assert unchanged(PowerSet, S.EmptySet)
    raises(ValueError, lambda: PowerSet(123))
    assert unchanged(PowerSet, S.Reals)
    assert unchanged(PowerSet, S.Integers)


def test_powerset_rewrite_FiniteSet():
    assert PowerSet(FiniteSet(1, 2)).rewrite(FiniteSet) == \
        FiniteSet(S.EmptySet, FiniteSet(1), FiniteSet(2), FiniteSet(1, 2))
    assert PowerSet(S.EmptySet).rewrite(FiniteSet) == FiniteSet(S.EmptySet)
    assert PowerSet(S.Naturals).rewrite(FiniteSet) == PowerSet(S.Naturals)


def test_finiteset_rewrite_powerset():
    assert FiniteSet(S.EmptySet).rewrite(PowerSet) == PowerSet(S.EmptySet)
    assert FiniteSet(
        S.EmptySet, FiniteSet(1),
        FiniteSet(2), FiniteSet(1, 2)).rewrite(PowerSet) == \
            PowerSet(FiniteSet(1, 2))
    assert FiniteSet(1, 2, 3).rewrite(PowerSet) == FiniteSet(1, 2, 3)


def test_powerset__contains__():
    subset_series = [
        S.EmptySet,
        FiniteSet(1, 2),
        S.Naturals,
        S.Naturals0,
        S.Integers,
        S.Rationals,
        S.Reals,
        S.Complexes]

    l = len(subset_series)
    for i in range(l):
        for j in range(l):
            if i <= j:
                assert subset_series[i] in \
                    PowerSet(subset_series[j], evaluate=False)
            else:
                assert subset_series[i] not in \
                    PowerSet(subset_series[j], evaluate=False)


@XFAIL
def test_failing_powerset__contains__():
    # XXX These are failing when evaluate=True,
    # but using unevaluated PowerSet works fine.
    assert FiniteSet(1, 2) not in PowerSet(S.EmptySet).rewrite(FiniteSet)
    assert S.Naturals not in PowerSet(S.EmptySet).rewrite(FiniteSet)
    assert S.Naturals not in PowerSet(FiniteSet(1, 2)).rewrite(FiniteSet)
    assert S.Naturals0 not in PowerSet(S.EmptySet).rewrite(FiniteSet)
    assert S.Naturals0 not in PowerSet(FiniteSet(1, 2)).rewrite(FiniteSet)
    assert S.Integers not in PowerSet(S.EmptySet).rewrite(FiniteSet)
    assert S.Integers not in PowerSet(FiniteSet(1, 2)).rewrite(FiniteSet)
    assert S.Rationals not in PowerSet(S.EmptySet).rewrite(FiniteSet)
    assert S.Rationals not in PowerSet(FiniteSet(1, 2)).rewrite(FiniteSet)
    assert S.Reals not in PowerSet(S.EmptySet).rewrite(FiniteSet)
    assert S.Reals not in PowerSet(FiniteSet(1, 2)).rewrite(FiniteSet)
    assert S.Complexes not in PowerSet(S.EmptySet).rewrite(FiniteSet)
    assert S.Complexes not in PowerSet(FiniteSet(1, 2)).rewrite(FiniteSet)


def test_powerset__len__():
    A = PowerSet(S.EmptySet, evaluate=False)
    assert len(A) == 1
    A = PowerSet(A, evaluate=False)
    assert len(A) == 2
    A = PowerSet(A, evaluate=False)
    assert len(A) == 4
    A = PowerSet(A, evaluate=False)
    assert len(A) == 16


def test_powerset__iter__():
    a = PowerSet(FiniteSet(1, 2)).__iter__()
    assert next(a) == S.EmptySet
    assert next(a) == FiniteSet(1)
    assert next(a) == FiniteSet(2)
    assert next(a) == FiniteSet(1, 2)

    a = PowerSet(S.Naturals).__iter__()
    assert next(a) == S.EmptySet
    assert next(a) == FiniteSet(1)
    assert next(a) == FiniteSet(2)
    assert next(a) == FiniteSet(1, 2)
    assert next(a) == FiniteSet(3)
    assert next(a) == FiniteSet(1, 3)
    assert next(a) == FiniteSet(2, 3)
    assert next(a) == FiniteSet(1, 2, 3)


def test_powerset_contains():
    A = PowerSet(FiniteSet(1), evaluate=False)
    assert A.contains(2) == Contains(2, A)

    x = Symbol('x')

    A = PowerSet(FiniteSet(x), evaluate=False)
    assert A.contains(FiniteSet(1)) == Contains(FiniteSet(1), A)


def test_powerset_method():
    # EmptySet
    A = FiniteSet()
    pset = A.powerset()
    assert len(pset) == 1
    assert pset ==  FiniteSet(S.EmptySet)

    # FiniteSets
    A = FiniteSet(1, 2)
    pset = A.powerset()
    assert len(pset) == 2**len(A)
    assert pset == FiniteSet(FiniteSet(), FiniteSet(1),
                             FiniteSet(2), A)
    # Not finite sets
    A = Interval(0, 1)
    assert A.powerset() == PowerSet(A)

def test_is_subset():
    # covers line 101-102
    # initialize powerset(1), which is a subset of powerset(1,2)
    subset = PowerSet(FiniteSet(1))
    pset = PowerSet(FiniteSet(1, 2))
    bad_set = PowerSet(FiniteSet(2, 3))
    # assert "subset" is subset of pset == True
    assert subset.is_subset(pset)
    # assert "bad_set" is subset of pset == False
    assert not pset.is_subset(bad_set)
