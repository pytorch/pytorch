from pytest import raises

import sympy
from sympy.physics.quantum import Dagger, AntiCommutator, qapply
from sympy.physics.quantum.fermion import FermionOp
from sympy.physics.quantum.fermion import FermionFockKet, FermionFockBra
from sympy import Symbol


def test_fermionoperator():
    c = FermionOp('c')
    d = FermionOp('d')

    assert isinstance(c, FermionOp)
    assert isinstance(Dagger(c), FermionOp)

    assert c.is_annihilation
    assert not Dagger(c).is_annihilation

    assert FermionOp("c") == FermionOp("c", True)
    assert FermionOp("c") != FermionOp("d")
    assert FermionOp("c", True) != FermionOp("c", False)

    assert AntiCommutator(c, Dagger(c)).doit() == 1

    assert AntiCommutator(c, Dagger(d)).doit() == c * Dagger(d) + Dagger(d) * c


def test_fermion_states():
    c = FermionOp("c")

    # Fock states
    assert (FermionFockBra(0) * FermionFockKet(1)).doit() == 0
    assert (FermionFockBra(1) * FermionFockKet(1)).doit() == 1

    assert qapply(c * FermionFockKet(1)) == FermionFockKet(0)
    assert qapply(c * FermionFockKet(0)) == 0

    assert qapply(Dagger(c) * FermionFockKet(0)) == FermionFockKet(1)
    assert qapply(Dagger(c) * FermionFockKet(1)) == 0


def test_power():
    c = FermionOp("c")
    assert c**0 == 1
    assert c**1 == c
    assert c**2 == 0
    assert c**3 == 0
    assert Dagger(c)**1 == Dagger(c)
    assert Dagger(c)**2 == 0

    assert (c**Symbol('a')).func == sympy.core.power.Pow
    assert (c**Symbol('a')).args == (c, Symbol('a'))

    with raises(ValueError):
        c**-1

    with raises(ValueError):
        c**3.2

    with raises(TypeError):
        c**1j
