"""Tests for sympy.physics.quantum.kind."""

from sympy.core.kind import NumberKind, UndefinedKind
from sympy.core.symbol import symbols

from sympy.physics.quantum.kind import (
    OperatorKind, KetKind, BraKind
)
from sympy.physics.quantum.anticommutator import AntiCommutator
from sympy.physics.quantum.commutator import Commutator
from sympy.physics.quantum.dagger import Dagger
from sympy.physics.quantum.operator import Operator
from sympy.physics.quantum.state import Ket, Bra
from sympy.physics.quantum.tensorproduct import TensorProduct

k = Ket('k')
b = Bra('k')
A = Operator('A')
B = Operator('B')
x, y, z = symbols('x y z', integer=True)

def test_bra_ket():
    assert k.kind == KetKind
    assert b.kind == BraKind
    assert (b*k).kind == NumberKind # inner product
    assert (x*k).kind == KetKind
    assert (x*b).kind == BraKind


def test_operator_kind():
    assert A.kind == OperatorKind
    assert (A*B).kind == OperatorKind
    assert (x*A).kind == OperatorKind
    assert (x*A*B).kind == OperatorKind
    assert (x*k*b).kind == OperatorKind # outer product


def test_undefind_kind():
    # Because of limitations in the kind dispatcher API, we are currently
    # unable to have OperatorKind*KetKind -> KetKind (and similar for bras).
    assert (A*k).kind == UndefinedKind
    assert (b*A).kind == UndefinedKind
    assert (x*b*A*k).kind == UndefinedKind


def test_dagger_kind():
    assert Dagger(k).kind == BraKind
    assert Dagger(b).kind == KetKind
    assert Dagger(A).kind == OperatorKind


def test_commutator_kind():
    assert Commutator(A, B).kind == OperatorKind
    assert Commutator(A, x*B).kind == OperatorKind
    assert Commutator(x*A, B).kind == OperatorKind
    assert Commutator(x*A, x*B).kind == OperatorKind


def test_anticommutator_kind():
    assert AntiCommutator(A, B).kind == OperatorKind
    assert AntiCommutator(A, x*B).kind == OperatorKind
    assert AntiCommutator(x*A, B).kind == OperatorKind
    assert AntiCommutator(x*A, x*B).kind == OperatorKind


def test_tensorproduct_kind():
    assert TensorProduct(k,k).kind == KetKind
    assert TensorProduct(b,b).kind == BraKind
    assert TensorProduct(x*k,y*k).kind == KetKind
    assert TensorProduct(x*b,y*b).kind == BraKind
    assert TensorProduct(x*b*k, y*b*k).kind == NumberKind
    assert TensorProduct(x*k*b, y*k*b).kind == OperatorKind
    assert TensorProduct(A, B).kind == OperatorKind
    assert TensorProduct(A, x*B).kind == OperatorKind
    assert TensorProduct(x*A, B).kind == OperatorKind
