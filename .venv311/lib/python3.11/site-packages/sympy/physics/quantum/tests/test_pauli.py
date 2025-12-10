from sympy.core.mul import Mul
from sympy.core.numbers import I
from sympy.matrices.dense import Matrix
from sympy.printing.latex import latex
from sympy.physics.quantum import (Dagger, Commutator, AntiCommutator, qapply,
                                   Operator, represent)
from sympy.physics.quantum.pauli import (SigmaOpBase, SigmaX, SigmaY, SigmaZ,
                                         SigmaMinus, SigmaPlus,
                                         qsimplify_pauli)
from sympy.physics.quantum.pauli import SigmaZKet, SigmaZBra
from sympy.testing.pytest import raises


sx, sy, sz = SigmaX(), SigmaY(), SigmaZ()
sx1, sy1, sz1 = SigmaX(1), SigmaY(1), SigmaZ(1)
sx2, sy2, sz2 = SigmaX(2), SigmaY(2), SigmaZ(2)

sm, sp = SigmaMinus(), SigmaPlus()
sm1, sp1 = SigmaMinus(1), SigmaPlus(1)
A, B = Operator("A"), Operator("B")


def test_pauli_operators_types():

    assert isinstance(sx, SigmaOpBase) and isinstance(sx, SigmaX)
    assert isinstance(sy, SigmaOpBase) and isinstance(sy, SigmaY)
    assert isinstance(sz, SigmaOpBase) and isinstance(sz, SigmaZ)
    assert isinstance(sm, SigmaOpBase) and isinstance(sm, SigmaMinus)
    assert isinstance(sp, SigmaOpBase) and isinstance(sp, SigmaPlus)


def test_pauli_operators_commutator():

    assert Commutator(sx, sy).doit() == 2 * I * sz
    assert Commutator(sy, sz).doit() == 2 * I * sx
    assert Commutator(sz, sx).doit() == 2 * I * sy


def test_pauli_operators_commutator_with_labels():

    assert Commutator(sx1, sy1).doit() == 2 * I * sz1
    assert Commutator(sy1, sz1).doit() == 2 * I * sx1
    assert Commutator(sz1, sx1).doit() == 2 * I * sy1

    assert Commutator(sx2, sy2).doit() == 2 * I * sz2
    assert Commutator(sy2, sz2).doit() == 2 * I * sx2
    assert Commutator(sz2, sx2).doit() == 2 * I * sy2

    assert Commutator(sx1, sy2).doit() == 0
    assert Commutator(sy1, sz2).doit() == 0
    assert Commutator(sz1, sx2).doit() == 0


def test_pauli_operators_anticommutator():

    assert AntiCommutator(sy, sz).doit() == 0
    assert AntiCommutator(sz, sx).doit() == 0
    assert AntiCommutator(sx, sm).doit() == 1
    assert AntiCommutator(sx, sp).doit() == 1


def test_pauli_operators_adjoint():

    assert Dagger(sx) == sx
    assert Dagger(sy) == sy
    assert Dagger(sz) == sz


def test_pauli_operators_adjoint_with_labels():

    assert Dagger(sx1) == sx1
    assert Dagger(sy1) == sy1
    assert Dagger(sz1) == sz1

    assert Dagger(sx1) != sx2
    assert Dagger(sy1) != sy2
    assert Dagger(sz1) != sz2


def test_pauli_operators_multiplication():

    assert qsimplify_pauli(sx * sx) == 1
    assert qsimplify_pauli(sy * sy) == 1
    assert qsimplify_pauli(sz * sz) == 1

    assert qsimplify_pauli(sx * sy) == I * sz
    assert qsimplify_pauli(sy * sz) == I * sx
    assert qsimplify_pauli(sz * sx) == I * sy

    assert qsimplify_pauli(sy * sx) == - I * sz
    assert qsimplify_pauli(sz * sy) == - I * sx
    assert qsimplify_pauli(sx * sz) == - I * sy


def test_pauli_operators_multiplication_with_labels():

    assert qsimplify_pauli(sx1 * sx1) == 1
    assert qsimplify_pauli(sy1 * sy1) == 1
    assert qsimplify_pauli(sz1 * sz1) == 1

    assert isinstance(sx1 * sx2, Mul)
    assert isinstance(sy1 * sy2, Mul)
    assert isinstance(sz1 * sz2, Mul)

    assert qsimplify_pauli(sx1 * sy1 * sx2 * sy2) == - sz1 * sz2
    assert qsimplify_pauli(sy1 * sz1 * sz2 * sx2) == - sx1 * sy2


def test_pauli_states():
    sx, sz = SigmaX(), SigmaZ()

    up = SigmaZKet(0)
    down = SigmaZKet(1)

    assert qapply(sx * up) == down
    assert qapply(sx * down) == up
    assert qapply(sz * up) == up
    assert qapply(sz * down) == - down

    up = SigmaZBra(0)
    down = SigmaZBra(1)

    assert qapply(up * sx, dagger=True) == down
    assert qapply(down * sx, dagger=True) == up
    assert qapply(up * sz, dagger=True) == up
    assert qapply(down * sz, dagger=True) == - down

    assert Dagger(SigmaZKet(0)) == SigmaZBra(0)
    assert Dagger(SigmaZBra(1)) == SigmaZKet(1)
    raises(ValueError, lambda: SigmaZBra(2))
    raises(ValueError, lambda: SigmaZKet(2))


def test_use_name():
    assert sm.use_name is False
    assert sm1.use_name is True
    assert sx.use_name is False
    assert sx1.use_name is True


def test_printing():
    assert latex(sx) == r'{\sigma_x}'
    assert latex(sx1) == r'{\sigma_x^{(1)}}'
    assert latex(sy) == r'{\sigma_y}'
    assert latex(sy1) == r'{\sigma_y^{(1)}}'
    assert latex(sz) == r'{\sigma_z}'
    assert latex(sz1) == r'{\sigma_z^{(1)}}'
    assert latex(sm) == r'{\sigma_-}'
    assert latex(sm1) == r'{\sigma_-^{(1)}}'
    assert latex(sp) == r'{\sigma_+}'
    assert latex(sp1) == r'{\sigma_+^{(1)}}'


def test_represent():
    assert represent(sx) == Matrix([[0, 1], [1, 0]])
    assert represent(sy) == Matrix([[0, -I], [I, 0]])
    assert represent(sz) == Matrix([[1, 0], [0, -1]])
    assert represent(sm) == Matrix([[0, 0], [1, 0]])
    assert represent(sp) == Matrix([[0, 1], [0, 0]])
