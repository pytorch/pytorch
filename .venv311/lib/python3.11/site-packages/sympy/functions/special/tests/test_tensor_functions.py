from sympy.core.relational import Ne
from sympy.core.symbol import (Dummy, Symbol, symbols)
from sympy.functions.elementary.complexes import (adjoint, conjugate, transpose)
from sympy.functions.elementary.piecewise import Piecewise
from sympy.functions.special.tensor_functions import (Eijk, KroneckerDelta, LeviCivita)

from sympy.physics.secondquant import evaluate_deltas, F

x, y = symbols('x y')


def test_levicivita():
    assert Eijk(1, 2, 3) == LeviCivita(1, 2, 3)
    assert LeviCivita(1, 2, 3) == 1
    assert LeviCivita(int(1), int(2), int(3)) == 1
    assert LeviCivita(1, 3, 2) == -1
    assert LeviCivita(1, 2, 2) == 0
    i, j, k = symbols('i j k')
    assert LeviCivita(i, j, k) == LeviCivita(i, j, k, evaluate=False)
    assert LeviCivita(i, j, i) == 0
    assert LeviCivita(1, i, i) == 0
    assert LeviCivita(i, j, k).doit() == (j - i)*(k - i)*(k - j)/2
    assert LeviCivita(1, 2, 3, 1) == 0
    assert LeviCivita(4, 5, 1, 2, 3) == 1
    assert LeviCivita(4, 5, 2, 1, 3) == -1

    assert LeviCivita(i, j, k).is_integer is True

    assert adjoint(LeviCivita(i, j, k)) == LeviCivita(i, j, k)
    assert conjugate(LeviCivita(i, j, k)) == LeviCivita(i, j, k)
    assert transpose(LeviCivita(i, j, k)) == LeviCivita(i, j, k)


def test_kronecker_delta():
    i, j = symbols('i j')
    k = Symbol('k', nonzero=True)
    assert KroneckerDelta(1, 1) == 1
    assert KroneckerDelta(1, 2) == 0
    assert KroneckerDelta(k, 0) == 0
    assert KroneckerDelta(x, x) == 1
    assert KroneckerDelta(x**2 - y**2, x**2 - y**2) == 1
    assert KroneckerDelta(i, i) == 1
    assert KroneckerDelta(i, i + 1) == 0
    assert KroneckerDelta(0, 0) == 1
    assert KroneckerDelta(0, 1) == 0
    assert KroneckerDelta(i + k, i) == 0
    assert KroneckerDelta(i + k, i + k) == 1
    assert KroneckerDelta(i + k, i + 1 + k) == 0
    assert KroneckerDelta(i, j).subs({"i": 1, "j": 0}) == 0
    assert KroneckerDelta(i, j).subs({"i": 3, "j": 3}) == 1

    assert KroneckerDelta(i, j)**0 == 1
    for n in range(1, 10):
        assert KroneckerDelta(i, j)**n == KroneckerDelta(i, j)
        assert KroneckerDelta(i, j)**-n == 1/KroneckerDelta(i, j)

    assert KroneckerDelta(i, j).is_integer is True

    assert adjoint(KroneckerDelta(i, j)) == KroneckerDelta(i, j)
    assert conjugate(KroneckerDelta(i, j)) == KroneckerDelta(i, j)
    assert transpose(KroneckerDelta(i, j)) == KroneckerDelta(i, j)
    # to test if canonical
    assert (KroneckerDelta(i, j) == KroneckerDelta(j, i)) == True

    assert KroneckerDelta(i, j).rewrite(Piecewise) == Piecewise((0, Ne(i, j)), (1, True))

    # Tests with range:
    assert KroneckerDelta(i, j, (0, i)).args == (i, j, (0, i))
    assert KroneckerDelta(i, j, (-j, i)).delta_range == (-j, i)

    # If index is out of range, return zero:
    assert KroneckerDelta(i, j, (0, i-1)) == 0
    assert KroneckerDelta(-1, j, (0, i-1)) == 0
    assert KroneckerDelta(j, -1, (0, i-1)) == 0
    assert KroneckerDelta(j, i, (0, i-1)) == 0


def test_kronecker_delta_secondquant():
    """secondquant-specific methods"""
    D = KroneckerDelta
    i, j, v, w = symbols('i j v w', below_fermi=True, cls=Dummy)
    a, b, t, u = symbols('a b t u', above_fermi=True, cls=Dummy)
    p, q, r, s = symbols('p q r s', cls=Dummy)

    assert D(i, a) == 0
    assert D(i, t) == 0

    assert D(i, j).is_above_fermi is False
    assert D(a, b).is_above_fermi is True
    assert D(p, q).is_above_fermi is True
    assert D(i, q).is_above_fermi is False
    assert D(q, i).is_above_fermi is False
    assert D(q, v).is_above_fermi is False
    assert D(a, q).is_above_fermi is True

    assert D(i, j).is_below_fermi is True
    assert D(a, b).is_below_fermi is False
    assert D(p, q).is_below_fermi is True
    assert D(p, j).is_below_fermi is True
    assert D(q, b).is_below_fermi is False

    assert D(i, j).is_only_above_fermi is False
    assert D(a, b).is_only_above_fermi is True
    assert D(p, q).is_only_above_fermi is False
    assert D(i, q).is_only_above_fermi is False
    assert D(q, i).is_only_above_fermi is False
    assert D(a, q).is_only_above_fermi is True

    assert D(i, j).is_only_below_fermi is True
    assert D(a, b).is_only_below_fermi is False
    assert D(p, q).is_only_below_fermi is False
    assert D(p, j).is_only_below_fermi is True
    assert D(q, b).is_only_below_fermi is False

    assert not D(i, q).indices_contain_equal_information
    assert not D(a, q).indices_contain_equal_information
    assert D(p, q).indices_contain_equal_information
    assert D(a, b).indices_contain_equal_information
    assert D(i, j).indices_contain_equal_information

    assert D(q, b).preferred_index == b
    assert D(q, b).killable_index == q
    assert D(q, t).preferred_index == t
    assert D(q, t).killable_index == q
    assert D(q, i).preferred_index == i
    assert D(q, i).killable_index == q
    assert D(q, v).preferred_index == v
    assert D(q, v).killable_index == q
    assert D(q, p).preferred_index == p
    assert D(q, p).killable_index == q

    EV = evaluate_deltas
    assert EV(D(a, q)*F(q)) == F(a)
    assert EV(D(i, q)*F(q)) == F(i)
    assert EV(D(a, q)*F(a)) == D(a, q)*F(a)
    assert EV(D(i, q)*F(i)) == D(i, q)*F(i)
    assert EV(D(a, b)*F(a)) == F(b)
    assert EV(D(a, b)*F(b)) == F(a)
    assert EV(D(i, j)*F(i)) == F(j)
    assert EV(D(i, j)*F(j)) == F(i)
    assert EV(D(p, q)*F(q)) == F(p)
    assert EV(D(p, q)*F(p)) == F(q)
    assert EV(D(p, j)*D(p, i)*F(i)) == F(j)
    assert EV(D(p, j)*D(p, i)*F(j)) == F(i)
    assert EV(D(p, q)*D(p, i))*F(i) == D(q, i)*F(i)
