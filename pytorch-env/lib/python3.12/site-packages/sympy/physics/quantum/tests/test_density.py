from sympy.core.numbers import Rational
from sympy.core.singleton import S
from sympy.core.symbol import symbols
from sympy.functions.elementary.exponential import log
from sympy.external import import_module
from sympy.physics.quantum.density import Density, entropy, fidelity
from sympy.physics.quantum.state import Ket, TimeDepKet
from sympy.physics.quantum.qubit import Qubit
from sympy.physics.quantum.represent import represent
from sympy.physics.quantum.dagger import Dagger
from sympy.physics.quantum.cartesian import XKet, PxKet, PxOp, XOp
from sympy.physics.quantum.spin import JzKet
from sympy.physics.quantum.operator import OuterProduct
from sympy.physics.quantum.trace import Tr
from sympy.functions import sqrt
from sympy.testing.pytest import raises
from sympy.physics.quantum.matrixutils import scipy_sparse_matrix
from sympy.physics.quantum.tensorproduct import TensorProduct


def test_eval_args():
    # check instance created
    assert isinstance(Density([Ket(0), 0.5], [Ket(1), 0.5]), Density)
    assert isinstance(Density([Qubit('00'), 1/sqrt(2)],
                              [Qubit('11'), 1/sqrt(2)]), Density)

    #test if Qubit object type preserved
    d = Density([Qubit('00'), 1/sqrt(2)], [Qubit('11'), 1/sqrt(2)])
    for (state, prob) in d.args:
        assert isinstance(state, Qubit)

    # check for value error, when prob is not provided
    raises(ValueError, lambda: Density([Ket(0)], [Ket(1)]))


def test_doit():

    x, y = symbols('x y')
    A, B, C, D, E, F = symbols('A B C D E F', commutative=False)
    d = Density([XKet(), 0.5], [PxKet(), 0.5])
    assert (0.5*(PxKet()*Dagger(PxKet())) +
            0.5*(XKet()*Dagger(XKet()))) == d.doit()

    # check for kets with expr in them
    d_with_sym = Density([XKet(x*y), 0.5], [PxKet(x*y), 0.5])
    assert (0.5*(PxKet(x*y)*Dagger(PxKet(x*y))) +
            0.5*(XKet(x*y)*Dagger(XKet(x*y)))) == d_with_sym.doit()

    d = Density([(A + B)*C, 1.0])
    assert d.doit() == (1.0*A*C*Dagger(C)*Dagger(A) +
                        1.0*A*C*Dagger(C)*Dagger(B) +
                        1.0*B*C*Dagger(C)*Dagger(A) +
                        1.0*B*C*Dagger(C)*Dagger(B))

    #  With TensorProducts as args
    # Density with simple tensor products as args
    t = TensorProduct(A, B, C)
    d = Density([t, 1.0])
    assert d.doit() == \
        1.0 * TensorProduct(A*Dagger(A), B*Dagger(B), C*Dagger(C))

    # Density with multiple Tensorproducts as states
    t2 = TensorProduct(A, B)
    t3 = TensorProduct(C, D)

    d = Density([t2, 0.5], [t3, 0.5])
    assert d.doit() == (0.5 * TensorProduct(A*Dagger(A), B*Dagger(B)) +
                        0.5 * TensorProduct(C*Dagger(C), D*Dagger(D)))

    #Density with mixed states
    d = Density([t2 + t3, 1.0])
    assert d.doit() == (1.0 * TensorProduct(A*Dagger(A), B*Dagger(B)) +
                        1.0 * TensorProduct(A*Dagger(C), B*Dagger(D)) +
                        1.0 * TensorProduct(C*Dagger(A), D*Dagger(B)) +
                        1.0 * TensorProduct(C*Dagger(C), D*Dagger(D)))

    #Density operators with spin states
    tp1 = TensorProduct(JzKet(1, 1), JzKet(1, -1))
    d = Density([tp1, 1])

    # full trace
    t = Tr(d)
    assert t.doit() == 1

    #Partial trace on density operators with spin states
    t = Tr(d, [0])
    assert t.doit() == JzKet(1, -1) * Dagger(JzKet(1, -1))
    t = Tr(d, [1])
    assert t.doit() == JzKet(1, 1) * Dagger(JzKet(1, 1))

    # with another spin state
    tp2 = TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2)))
    d = Density([tp2, 1])

    #full trace
    t = Tr(d)
    assert t.doit() == 1

    #Partial trace on density operators with spin states
    t = Tr(d, [0])
    assert t.doit() == JzKet(S.Half, Rational(-1, 2)) * Dagger(JzKet(S.Half, Rational(-1, 2)))
    t = Tr(d, [1])
    assert t.doit() == JzKet(S.Half, S.Half) * Dagger(JzKet(S.Half, S.Half))


def test_apply_op():
    d = Density([Ket(0), 0.5], [Ket(1), 0.5])
    assert d.apply_op(XOp()) == Density([XOp()*Ket(0), 0.5],
                                        [XOp()*Ket(1), 0.5])


def test_represent():
    x, y = symbols('x y')
    d = Density([XKet(), 0.5], [PxKet(), 0.5])
    assert (represent(0.5*(PxKet()*Dagger(PxKet()))) +
            represent(0.5*(XKet()*Dagger(XKet())))) == represent(d)

    # check for kets with expr in them
    d_with_sym = Density([XKet(x*y), 0.5], [PxKet(x*y), 0.5])
    assert (represent(0.5*(PxKet(x*y)*Dagger(PxKet(x*y)))) +
            represent(0.5*(XKet(x*y)*Dagger(XKet(x*y))))) == \
        represent(d_with_sym)

    # check when given explicit basis
    assert (represent(0.5*(XKet()*Dagger(XKet())), basis=PxOp()) +
            represent(0.5*(PxKet()*Dagger(PxKet())), basis=PxOp())) == \
        represent(d, basis=PxOp())


def test_states():
    d = Density([Ket(0), 0.5], [Ket(1), 0.5])
    states = d.states()
    assert states[0] == Ket(0) and states[1] == Ket(1)


def test_probs():
    d = Density([Ket(0), .75], [Ket(1), 0.25])
    probs = d.probs()
    assert probs[0] == 0.75 and probs[1] == 0.25

    #probs can be symbols
    x, y = symbols('x y')
    d = Density([Ket(0), x], [Ket(1), y])
    probs = d.probs()
    assert probs[0] == x and probs[1] == y


def test_get_state():
    x, y = symbols('x y')
    d = Density([Ket(0), x], [Ket(1), y])
    states = (d.get_state(0), d.get_state(1))
    assert states[0] == Ket(0) and states[1] == Ket(1)


def test_get_prob():
    x, y = symbols('x y')
    d = Density([Ket(0), x], [Ket(1), y])
    probs = (d.get_prob(0), d.get_prob(1))
    assert probs[0] == x and probs[1] == y


def test_entropy():
    up = JzKet(S.Half, S.Half)
    down = JzKet(S.Half, Rational(-1, 2))
    d = Density((up, S.Half), (down, S.Half))

    # test for density object
    ent = entropy(d)
    assert entropy(d) == log(2)/2
    assert d.entropy() == log(2)/2

    np = import_module('numpy', min_module_version='1.4.0')
    if np:
        #do this test only if 'numpy' is available on test machine
        np_mat = represent(d, format='numpy')
        ent = entropy(np_mat)
        assert isinstance(np_mat, np.ndarray)
        assert ent.real == 0.69314718055994529
        assert ent.imag == 0

    scipy = import_module('scipy', import_kwargs={'fromlist': ['sparse']})
    if scipy and np:
        #do this test only if numpy and scipy are available
        mat = represent(d, format="scipy.sparse")
        assert isinstance(mat, scipy_sparse_matrix)
        assert ent.real == 0.69314718055994529
        assert ent.imag == 0


def test_eval_trace():
    up = JzKet(S.Half, S.Half)
    down = JzKet(S.Half, Rational(-1, 2))
    d = Density((up, 0.5), (down, 0.5))

    t = Tr(d)
    assert t.doit() == 1.0

    #test dummy time dependent states
    class TestTimeDepKet(TimeDepKet):
        def _eval_trace(self, bra, **options):
            return 1

    x, t = symbols('x t')
    k1 = TestTimeDepKet(0, 0.5)
    k2 = TestTimeDepKet(0, 1)
    d = Density([k1, 0.5], [k2, 0.5])
    assert d.doit() == (0.5 * OuterProduct(k1, k1.dual) +
                        0.5 * OuterProduct(k2, k2.dual))

    t = Tr(d)
    assert t.doit() == 1.0


def test_fidelity():
    #test with kets
    up = JzKet(S.Half, S.Half)
    down = JzKet(S.Half, Rational(-1, 2))
    updown = (S.One/sqrt(2))*up + (S.One/sqrt(2))*down

    #check with matrices
    up_dm = represent(up * Dagger(up))
    down_dm = represent(down * Dagger(down))
    updown_dm = represent(updown * Dagger(updown))

    assert abs(fidelity(up_dm, up_dm) - 1) < 1e-3
    assert fidelity(up_dm, down_dm) < 1e-3
    assert abs(fidelity(up_dm, updown_dm) - (S.One/sqrt(2))) < 1e-3
    assert abs(fidelity(updown_dm, down_dm) - (S.One/sqrt(2))) < 1e-3

    #check with density
    up_dm = Density([up, 1.0])
    down_dm = Density([down, 1.0])
    updown_dm = Density([updown, 1.0])

    assert abs(fidelity(up_dm, up_dm) - 1) < 1e-3
    assert abs(fidelity(up_dm, down_dm)) < 1e-3
    assert abs(fidelity(up_dm, updown_dm) - (S.One/sqrt(2))) < 1e-3
    assert abs(fidelity(updown_dm, down_dm) - (S.One/sqrt(2))) < 1e-3

    #check mixed states with density
    updown2 = sqrt(3)/2*up + S.Half*down
    d1 = Density([updown, 0.25], [updown2, 0.75])
    d2 = Density([updown, 0.75], [updown2, 0.25])
    assert abs(fidelity(d1, d2) - 0.991) < 1e-3
    assert abs(fidelity(d2, d1) - fidelity(d1, d2)) < 1e-3

    #using qubits/density(pure states)
    state1 = Qubit('0')
    state2 = Qubit('1')
    state3 = S.One/sqrt(2)*state1 + S.One/sqrt(2)*state2
    state4 = sqrt(Rational(2, 3))*state1 + S.One/sqrt(3)*state2

    state1_dm = Density([state1, 1])
    state2_dm = Density([state2, 1])
    state3_dm = Density([state3, 1])

    assert fidelity(state1_dm, state1_dm) == 1
    assert fidelity(state1_dm, state2_dm) == 0
    assert abs(fidelity(state1_dm, state3_dm) - 1/sqrt(2)) < 1e-3
    assert abs(fidelity(state3_dm, state2_dm) - 1/sqrt(2)) < 1e-3

    #using qubits/density(mixed states)
    d1 = Density([state3, 0.70], [state4, 0.30])
    d2 = Density([state3, 0.20], [state4, 0.80])
    assert abs(fidelity(d1, d1) - 1) < 1e-3
    assert abs(fidelity(d1, d2) - 0.996) < 1e-3
    assert abs(fidelity(d1, d2) - fidelity(d2, d1)) < 1e-3

    #TODO: test for invalid arguments
    # non-square matrix
    mat1 = [[0, 0],
            [0, 0],
            [0, 0]]

    mat2 = [[0, 0],
            [0, 0]]
    raises(ValueError, lambda: fidelity(mat1, mat2))

    # unequal dimensions
    mat1 = [[0, 0],
            [0, 0]]
    mat2 = [[0, 0, 0],
            [0, 0, 0],
            [0, 0, 0]]
    raises(ValueError, lambda: fidelity(mat1, mat2))

    # unsupported data-type
    x, y = 1, 2  # random values that is not a matrix
    raises(ValueError, lambda: fidelity(x, y))
