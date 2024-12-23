from math import prod

from sympy.core.numbers import Rational
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.physics.quantum import Dagger, Commutator, qapply
from sympy.physics.quantum.boson import BosonOp
from sympy.physics.quantum.boson import (
    BosonFockKet, BosonFockBra, BosonCoherentKet, BosonCoherentBra)


def test_bosonoperator():
    a = BosonOp('a')
    b = BosonOp('b')

    assert isinstance(a, BosonOp)
    assert isinstance(Dagger(a), BosonOp)

    assert a.is_annihilation
    assert not Dagger(a).is_annihilation

    assert BosonOp("a") == BosonOp("a", True)
    assert BosonOp("a") != BosonOp("c")
    assert BosonOp("a", True) != BosonOp("a", False)

    assert Commutator(a, Dagger(a)).doit() == 1

    assert Commutator(a, Dagger(b)).doit() == a * Dagger(b) - Dagger(b) * a

    assert Dagger(exp(a)) == exp(Dagger(a))


def test_boson_states():
    a = BosonOp("a")

    # Fock states
    n = 3
    assert (BosonFockBra(0) * BosonFockKet(1)).doit() == 0
    assert (BosonFockBra(1) * BosonFockKet(1)).doit() == 1
    assert qapply(BosonFockBra(n) * Dagger(a)**n * BosonFockKet(0)) \
        == sqrt(prod(range(1, n+1)))

    # Coherent states
    alpha1, alpha2 = 1.2, 4.3
    assert (BosonCoherentBra(alpha1) * BosonCoherentKet(alpha1)).doit() == 1
    assert (BosonCoherentBra(alpha2) * BosonCoherentKet(alpha2)).doit() == 1
    assert abs((BosonCoherentBra(alpha1) * BosonCoherentKet(alpha2)).doit() -
               exp((alpha1 - alpha2) ** 2 * Rational(-1, 2))) < 1e-12
    assert qapply(a * BosonCoherentKet(alpha1)) == \
        alpha1 * BosonCoherentKet(alpha1)
