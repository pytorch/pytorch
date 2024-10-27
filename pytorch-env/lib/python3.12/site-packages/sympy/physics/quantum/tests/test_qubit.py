import random

from sympy.core.numbers import (Integer, Rational)
from sympy.core.singleton import S
from sympy.core.symbol import symbols
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.matrices.dense import Matrix
from sympy.physics.quantum.qubit import (measure_all, measure_partial,
                                         matrix_to_qubit, matrix_to_density,
                                         qubit_to_matrix, IntQubit,
                                         IntQubitBra, QubitBra)
from sympy.physics.quantum.gate import (HadamardGate, CNOT, XGate, YGate,
                                        ZGate, PhaseGate)
from sympy.physics.quantum.qapply import qapply
from sympy.physics.quantum.represent import represent
from sympy.physics.quantum.shor import Qubit
from sympy.testing.pytest import raises
from sympy.physics.quantum.density import Density
from sympy.physics.quantum.trace import Tr

x, y = symbols('x,y')

epsilon = .000001


def test_Qubit():
    array = [0, 0, 1, 1, 0]
    qb = Qubit('00110')
    assert qb.flip(0) == Qubit('00111')
    assert qb.flip(1) == Qubit('00100')
    assert qb.flip(4) == Qubit('10110')
    assert qb.qubit_values == (0, 0, 1, 1, 0)
    assert qb.dimension == 5
    for i in range(5):
        assert qb[i] == array[4 - i]
    assert len(qb) == 5
    qb = Qubit('110')


def test_QubitBra():
    qb = Qubit(0)
    qb_bra = QubitBra(0)
    assert qb.dual_class() == QubitBra
    assert qb_bra.dual_class() == Qubit

    qb = Qubit(1, 1, 0)
    qb_bra = QubitBra(1, 1, 0)
    assert represent(qb, nqubits=3).H == represent(qb_bra, nqubits=3)

    qb = Qubit(0, 1)
    qb_bra = QubitBra(1,0)
    assert qb._eval_innerproduct_QubitBra(qb_bra) == Integer(0)

    qb_bra = QubitBra(0, 1)
    assert qb._eval_innerproduct_QubitBra(qb_bra) == Integer(1)


def test_IntQubit():
    # issue 9136
    iqb = IntQubit(0, nqubits=1)
    assert qubit_to_matrix(Qubit('0')) == qubit_to_matrix(iqb)

    qb = Qubit('1010')
    assert qubit_to_matrix(IntQubit(qb)) == qubit_to_matrix(qb)

    iqb = IntQubit(1, nqubits=1)
    assert qubit_to_matrix(Qubit('1')) == qubit_to_matrix(iqb)
    assert qubit_to_matrix(IntQubit(1)) == qubit_to_matrix(iqb)

    iqb = IntQubit(7, nqubits=4)
    assert qubit_to_matrix(Qubit('0111')) == qubit_to_matrix(iqb)
    assert qubit_to_matrix(IntQubit(7, 4)) == qubit_to_matrix(iqb)

    iqb = IntQubit(8)
    assert iqb.as_int() == 8
    assert iqb.qubit_values == (1, 0, 0, 0)

    iqb = IntQubit(7, 4)
    assert iqb.qubit_values == (0, 1, 1, 1)
    assert IntQubit(3) == IntQubit(3, 2)

    #test Dual Classes
    iqb = IntQubit(3)
    iqb_bra = IntQubitBra(3)
    assert iqb.dual_class() == IntQubitBra
    assert iqb_bra.dual_class() == IntQubit

    iqb = IntQubit(5)
    iqb_bra = IntQubitBra(5)
    assert iqb._eval_innerproduct_IntQubitBra(iqb_bra) == Integer(1)

    iqb = IntQubit(4)
    iqb_bra = IntQubitBra(5)
    assert iqb._eval_innerproduct_IntQubitBra(iqb_bra) == Integer(0)
    raises(ValueError, lambda: IntQubit(4, 1))

    raises(ValueError, lambda: IntQubit('5'))
    raises(ValueError, lambda: IntQubit(5, '5'))
    raises(ValueError, lambda: IntQubit(5, nqubits='5'))
    raises(TypeError, lambda: IntQubit(5, bad_arg=True))

def test_superposition_of_states():
    state = 1/sqrt(2)*Qubit('01') + 1/sqrt(2)*Qubit('10')
    state_gate = CNOT(0, 1)*HadamardGate(0)*state
    state_expanded = Qubit('01')/2 + Qubit('00')/2 - Qubit('11')/2 + Qubit('10')/2
    assert qapply(state_gate).expand() == state_expanded
    assert matrix_to_qubit(represent(state_gate, nqubits=2)) == state_expanded


#test apply methods
def test_apply_represent_equality():
    gates = [HadamardGate(int(3*random.random())),
     XGate(int(3*random.random())), ZGate(int(3*random.random())),
        YGate(int(3*random.random())), ZGate(int(3*random.random())),
        PhaseGate(int(3*random.random()))]

    circuit = Qubit(int(random.random()*2), int(random.random()*2),
    int(random.random()*2), int(random.random()*2), int(random.random()*2),
        int(random.random()*2))
    for i in range(int(random.random()*6)):
        circuit = gates[int(random.random()*6)]*circuit

    mat = represent(circuit, nqubits=6)
    states = qapply(circuit)
    state_rep = matrix_to_qubit(mat)
    states = states.expand()
    state_rep = state_rep.expand()
    assert state_rep == states


def test_matrix_to_qubits():
    qb = Qubit(0, 0, 0, 0)
    mat = Matrix([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    assert matrix_to_qubit(mat) == qb
    assert qubit_to_matrix(qb) == mat

    state = 2*sqrt(2)*(Qubit(0, 0, 0) + Qubit(0, 0, 1) + Qubit(0, 1, 0) +
                       Qubit(0, 1, 1) + Qubit(1, 0, 0) + Qubit(1, 0, 1) +
                       Qubit(1, 1, 0) + Qubit(1, 1, 1))
    ones = sqrt(2)*2*Matrix([1, 1, 1, 1, 1, 1, 1, 1])
    assert matrix_to_qubit(ones) == state.expand()
    assert qubit_to_matrix(state) == ones


def test_measure_normalize():
    a, b = symbols('a b')
    state = a*Qubit('110') + b*Qubit('111')
    assert measure_partial(state, (0,), normalize=False) == \
        [(a*Qubit('110'), a*a.conjugate()), (b*Qubit('111'), b*b.conjugate())]
    assert measure_all(state, normalize=False) == \
        [(Qubit('110'), a*a.conjugate()), (Qubit('111'), b*b.conjugate())]


def test_measure_partial():
    #Basic test of collapse of entangled two qubits (Bell States)
    state = Qubit('01') + Qubit('10')
    assert measure_partial(state, (0,)) == \
        [(Qubit('10'), S.Half), (Qubit('01'), S.Half)]
    assert measure_partial(state, int(0)) == \
        [(Qubit('10'), S.Half), (Qubit('01'), S.Half)]
    assert measure_partial(state, (0,)) == \
        measure_partial(state, (1,))[::-1]

    #Test of more complex collapse and probability calculation
    state1 = sqrt(2)/sqrt(3)*Qubit('00001') + 1/sqrt(3)*Qubit('11111')
    assert measure_partial(state1, (0,)) == \
        [(sqrt(2)/sqrt(3)*Qubit('00001') + 1/sqrt(3)*Qubit('11111'), 1)]
    assert measure_partial(state1, (1, 2)) == measure_partial(state1, (3, 4))
    assert measure_partial(state1, (1, 2, 3)) == \
        [(Qubit('00001'), Rational(2, 3)), (Qubit('11111'), Rational(1, 3))]

    #test of measuring multiple bits at once
    state2 = Qubit('1111') + Qubit('1101') + Qubit('1011') + Qubit('1000')
    assert measure_partial(state2, (0, 1, 3)) == \
        [(Qubit('1000'), Rational(1, 4)), (Qubit('1101'), Rational(1, 4)),
         (Qubit('1011')/sqrt(2) + Qubit('1111')/sqrt(2), S.Half)]
    assert measure_partial(state2, (0,)) == \
        [(Qubit('1000'), Rational(1, 4)),
         (Qubit('1111')/sqrt(3) + Qubit('1101')/sqrt(3) +
          Qubit('1011')/sqrt(3), Rational(3, 4))]


def test_measure_all():
    assert measure_all(Qubit('11')) == [(Qubit('11'), 1)]
    state = Qubit('11') + Qubit('10')
    assert measure_all(state) == [(Qubit('10'), S.Half),
           (Qubit('11'), S.Half)]
    state2 = Qubit('11')/sqrt(5) + 2*Qubit('00')/sqrt(5)
    assert measure_all(state2) == \
        [(Qubit('00'), Rational(4, 5)), (Qubit('11'), Rational(1, 5))]

    # from issue #12585
    assert measure_all(qapply(Qubit('0'))) == [(Qubit('0'), 1)]


def test_eval_trace():
    q1 = Qubit('10110')
    q2 = Qubit('01010')
    d = Density([q1, 0.6], [q2, 0.4])

    t = Tr(d)
    assert t.doit() == 1.0

    # extreme bits
    t = Tr(d, 0)
    assert t.doit() == (0.4*Density([Qubit('0101'), 1]) +
                        0.6*Density([Qubit('1011'), 1]))
    t = Tr(d, 4)
    assert t.doit() == (0.4*Density([Qubit('1010'), 1]) +
                        0.6*Density([Qubit('0110'), 1]))
    # index somewhere in between
    t = Tr(d, 2)
    assert t.doit() == (0.4*Density([Qubit('0110'), 1]) +
                        0.6*Density([Qubit('1010'), 1]))
    #trace all indices
    t = Tr(d, [0, 1, 2, 3, 4])
    assert t.doit() == 1.0

    # trace some indices, initialized in
    # non-canonical order
    t = Tr(d, [2, 1, 3])
    assert t.doit() == (0.4*Density([Qubit('00'), 1]) +
                        0.6*Density([Qubit('10'), 1]))

    # mixed states
    q = (1/sqrt(2)) * (Qubit('00') + Qubit('11'))
    d = Density( [q, 1.0] )
    t = Tr(d, 0)
    assert t.doit() == (0.5*Density([Qubit('0'), 1]) +
                        0.5*Density([Qubit('1'), 1]))


def test_matrix_to_density():
    mat = Matrix([[0, 0], [0, 1]])
    assert matrix_to_density(mat) == Density([Qubit('1'), 1])

    mat = Matrix([[1, 0], [0, 0]])
    assert matrix_to_density(mat) == Density([Qubit('0'), 1])

    mat = Matrix([[0, 0], [0, 0]])
    assert matrix_to_density(mat) == 0

    mat = Matrix([[0, 0, 0, 0],
                  [0, 0, 0, 0],
                  [0, 0, 1, 0],
                  [0, 0, 0, 0]])

    assert matrix_to_density(mat) == Density([Qubit('10'), 1])

    mat = Matrix([[1, 0, 0, 0],
                  [0, 0, 0, 0],
                  [0, 0, 0, 0],
                  [0, 0, 0, 0]])

    assert matrix_to_density(mat) == Density([Qubit('00'), 1])
