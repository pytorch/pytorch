from sympy.sandbox.indexed_integrals import IndexedIntegral
from sympy.core.symbol import symbols
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.tensor.indexed import (Idx, IndexedBase)


def test_indexed_integrals():
    A = IndexedBase('A')
    i, j = symbols('i j', integer=True)
    a1, a2 = symbols('a1:3', cls=Idx)
    assert isinstance(a1, Idx)

    assert IndexedIntegral(1, A[i]).doit() == A[i]
    assert IndexedIntegral(A[i], A[i]).doit() == A[i] ** 2 / 2
    assert IndexedIntegral(A[j], A[i]).doit() == A[i] * A[j]
    assert IndexedIntegral(A[i] * A[j], A[i]).doit() == A[i] ** 2 * A[j] / 2
    assert IndexedIntegral(sin(A[i]), A[i]).doit() == -cos(A[i])
    assert IndexedIntegral(sin(A[j]), A[i]).doit() == sin(A[j]) * A[i]

    assert IndexedIntegral(1, A[a1]).doit() == A[a1]
    assert IndexedIntegral(A[a1], A[a1]).doit() == A[a1] ** 2 / 2
    assert IndexedIntegral(A[a2], A[a1]).doit() == A[a1] * A[a2]
    assert IndexedIntegral(A[a1] * A[a2], A[a1]).doit() == A[a1] ** 2 * A[a2] / 2
    assert IndexedIntegral(sin(A[a1]), A[a1]).doit() == -cos(A[a1])
    assert IndexedIntegral(sin(A[a2]), A[a1]).doit() == sin(A[a2]) * A[a1]
