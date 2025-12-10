from sympy.physics.quantum import Dagger
from sympy.physics.quantum.boson import BosonOp
from sympy.physics.quantum.fermion import FermionOp
from sympy.physics.quantum.operatorordering import (normal_order,
                                                 normal_ordered_form)


def test_normal_order():
    a = BosonOp('a')

    c = FermionOp('c')

    assert normal_order(a * Dagger(a)) == Dagger(a) * a
    assert normal_order(Dagger(a) * a) == Dagger(a) * a
    assert normal_order(a * Dagger(a) ** 2) == Dagger(a) ** 2 * a

    assert normal_order(c * Dagger(c)) == - Dagger(c) * c
    assert normal_order(Dagger(c) * c) == Dagger(c) * c
    assert normal_order(c * Dagger(c) ** 2) == Dagger(c) ** 2 * c


def test_normal_ordered_form():
    a = BosonOp('a')
    b = BosonOp('b')

    c = FermionOp('c')
    d = FermionOp('d')

    assert normal_ordered_form(Dagger(a) * a) == Dagger(a) * a
    assert normal_ordered_form(a * Dagger(a)) == 1 + Dagger(a) * a
    assert normal_ordered_form(a ** 2 * Dagger(a)) == \
        2 * a + Dagger(a) * a ** 2
    assert normal_ordered_form(a ** 3 * Dagger(a)) == \
        3 * a ** 2 + Dagger(a) * a ** 3

    assert normal_ordered_form(Dagger(c) * c) == Dagger(c) * c
    assert normal_ordered_form(c * Dagger(c)) == 1 - Dagger(c) * c
    assert normal_ordered_form(c ** 2 * Dagger(c)) == Dagger(c) * c ** 2
    assert normal_ordered_form(c ** 3 * Dagger(c)) == \
        c ** 2 - Dagger(c) * c ** 3

    assert normal_ordered_form(a * Dagger(b), True) == Dagger(b) * a
    assert normal_ordered_form(Dagger(a) * b, True) == Dagger(a) * b
    assert normal_ordered_form(b * a, True) == a * b
    assert normal_ordered_form(Dagger(b) * Dagger(a), True) == Dagger(a) * Dagger(b)

    assert normal_ordered_form(c * Dagger(d), True) == -Dagger(d) * c
    assert normal_ordered_form(Dagger(c) * d, True) == Dagger(c) * d
    assert normal_ordered_form(d * c, True) == -c * d
    assert normal_ordered_form(Dagger(d) * Dagger(c), True) == -Dagger(c) * Dagger(d)
