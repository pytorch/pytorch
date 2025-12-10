from sympy.external import import_module
from sympy.core.mul import Mul
from sympy.core.numbers import Integer
from sympy.physics.quantum.dagger import Dagger
from sympy.physics.quantum.gate import (X, Y, Z, H, CNOT,
        IdentityGate, CGate, PhaseGate, TGate)
from sympy.physics.quantum.identitysearch import (generate_gate_rules,
        generate_equivalent_ids, GateIdentity, bfs_identity_search,
        is_scalar_sparse_matrix,
        is_scalar_nonsparse_matrix, is_degenerate, is_reducible)
from sympy.testing.pytest import skip


def create_gate_sequence(qubit=0):
    gates = (X(qubit), Y(qubit), Z(qubit), H(qubit))
    return gates


def test_generate_gate_rules_1():
    # Test with tuples
    (x, y, z, h) = create_gate_sequence()
    ph = PhaseGate(0)
    cgate_t = CGate(0, TGate(1))

    assert generate_gate_rules((x,)) == {((x,), ())}

    gate_rules = {((x, x), ()),
                      ((x,), (x,))}
    assert generate_gate_rules((x, x)) == gate_rules

    gate_rules = {((x, y, x), ()),
                      ((y, x, x), ()),
                      ((x, x, y), ()),
                      ((y, x), (x,)),
                      ((x, y), (x,)),
                      ((y,), (x, x))}
    assert generate_gate_rules((x, y, x)) == gate_rules

    gate_rules = {((x, y, z), ()), ((y, z, x), ()), ((z, x, y), ()),
                      ((), (x, z, y)), ((), (y, x, z)), ((), (z, y, x)),
                      ((x,), (z, y)), ((y, z), (x,)), ((y,), (x, z)),
                      ((z, x), (y,)), ((z,), (y, x)), ((x, y), (z,))}
    actual = generate_gate_rules((x, y, z))
    assert actual == gate_rules

    gate_rules = {
        ((), (h, z, y, x)), ((), (x, h, z, y)), ((), (y, x, h, z)),
         ((), (z, y, x, h)), ((h,), (z, y, x)), ((x,), (h, z, y)),
         ((y,), (x, h, z)), ((z,), (y, x, h)), ((h, x), (z, y)),
         ((x, y), (h, z)), ((y, z), (x, h)), ((z, h), (y, x)),
         ((h, x, y), (z,)), ((x, y, z), (h,)), ((y, z, h), (x,)),
         ((z, h, x), (y,)), ((h, x, y, z), ()), ((x, y, z, h), ()),
         ((y, z, h, x), ()), ((z, h, x, y), ())}
    actual = generate_gate_rules((x, y, z, h))
    assert actual == gate_rules

    gate_rules = {((), (cgate_t**(-1), ph**(-1), x)),
                      ((), (ph**(-1), x, cgate_t**(-1))),
                      ((), (x, cgate_t**(-1), ph**(-1))),
                      ((cgate_t,), (ph**(-1), x)),
                      ((ph,), (x, cgate_t**(-1))),
                      ((x,), (cgate_t**(-1), ph**(-1))),
                      ((cgate_t, x), (ph**(-1),)),
                      ((ph, cgate_t), (x,)),
                      ((x, ph), (cgate_t**(-1),)),
                      ((cgate_t, x, ph), ()),
                      ((ph, cgate_t, x), ()),
                      ((x, ph, cgate_t), ())}
    actual = generate_gate_rules((x, ph, cgate_t))
    assert actual == gate_rules

    gate_rules = {(Integer(1), cgate_t**(-1)*ph**(-1)*x),
                      (Integer(1), ph**(-1)*x*cgate_t**(-1)),
                      (Integer(1), x*cgate_t**(-1)*ph**(-1)),
                      (cgate_t, ph**(-1)*x),
                      (ph, x*cgate_t**(-1)),
                      (x, cgate_t**(-1)*ph**(-1)),
                      (cgate_t*x, ph**(-1)),
                      (ph*cgate_t, x),
                      (x*ph, cgate_t**(-1)),
                      (cgate_t*x*ph, Integer(1)),
                      (ph*cgate_t*x, Integer(1)),
                      (x*ph*cgate_t, Integer(1))}
    actual = generate_gate_rules((x, ph, cgate_t), return_as_muls=True)
    assert actual == gate_rules


def test_generate_gate_rules_2():
    # Test with Muls
    (x, y, z, h) = create_gate_sequence()
    ph = PhaseGate(0)
    cgate_t = CGate(0, TGate(1))

    # Note: 1 (type int) is not the same as 1 (type One)
    expected = {(x, Integer(1))}
    assert generate_gate_rules((x,), return_as_muls=True) == expected

    expected = {(Integer(1), Integer(1))}
    assert generate_gate_rules(x*x, return_as_muls=True) == expected

    expected = {((), ())}
    assert generate_gate_rules(x*x, return_as_muls=False) == expected

    gate_rules = {(x*y*x, Integer(1)),
                      (y, Integer(1)),
                      (y*x, x),
                      (x*y, x)}
    assert generate_gate_rules(x*y*x, return_as_muls=True) == gate_rules

    gate_rules = {(x*y*z, Integer(1)),
                      (y*z*x, Integer(1)),
                      (z*x*y, Integer(1)),
                      (Integer(1), x*z*y),
                      (Integer(1), y*x*z),
                      (Integer(1), z*y*x),
                      (x, z*y),
                      (y*z, x),
                      (y, x*z),
                      (z*x, y),
                      (z, y*x),
                      (x*y, z)}
    actual = generate_gate_rules(x*y*z, return_as_muls=True)
    assert actual == gate_rules

    gate_rules = {(Integer(1), h*z*y*x),
                      (Integer(1), x*h*z*y),
                      (Integer(1), y*x*h*z),
                      (Integer(1), z*y*x*h),
                      (h, z*y*x), (x, h*z*y),
                      (y, x*h*z), (z, y*x*h),
                      (h*x, z*y), (z*h, y*x),
                      (x*y, h*z), (y*z, x*h),
                      (h*x*y, z), (x*y*z, h),
                      (y*z*h, x), (z*h*x, y),
                      (h*x*y*z, Integer(1)),
                      (x*y*z*h, Integer(1)),
                      (y*z*h*x, Integer(1)),
                      (z*h*x*y, Integer(1))}
    actual = generate_gate_rules(x*y*z*h, return_as_muls=True)
    assert actual == gate_rules

    gate_rules = {(Integer(1), cgate_t**(-1)*ph**(-1)*x),
                      (Integer(1), ph**(-1)*x*cgate_t**(-1)),
                      (Integer(1), x*cgate_t**(-1)*ph**(-1)),
                      (cgate_t, ph**(-1)*x),
                      (ph, x*cgate_t**(-1)),
                      (x, cgate_t**(-1)*ph**(-1)),
                      (cgate_t*x, ph**(-1)),
                      (ph*cgate_t, x),
                      (x*ph, cgate_t**(-1)),
                      (cgate_t*x*ph, Integer(1)),
                      (ph*cgate_t*x, Integer(1)),
                      (x*ph*cgate_t, Integer(1))}
    actual = generate_gate_rules(x*ph*cgate_t, return_as_muls=True)
    assert actual == gate_rules

    gate_rules = {((), (cgate_t**(-1), ph**(-1), x)),
                      ((), (ph**(-1), x, cgate_t**(-1))),
                      ((), (x, cgate_t**(-1), ph**(-1))),
                      ((cgate_t,), (ph**(-1), x)),
                      ((ph,), (x, cgate_t**(-1))),
                      ((x,), (cgate_t**(-1), ph**(-1))),
                      ((cgate_t, x), (ph**(-1),)),
                      ((ph, cgate_t), (x,)),
                      ((x, ph), (cgate_t**(-1),)),
                      ((cgate_t, x, ph), ()),
                      ((ph, cgate_t, x), ()),
                      ((x, ph, cgate_t), ())}
    actual = generate_gate_rules(x*ph*cgate_t)
    assert actual == gate_rules


def test_generate_equivalent_ids_1():
    # Test with tuples
    (x, y, z, h) = create_gate_sequence()

    assert generate_equivalent_ids((x,)) == {(x,)}
    assert generate_equivalent_ids((x, x)) == {(x, x)}
    assert generate_equivalent_ids((x, y)) == {(x, y), (y, x)}

    gate_seq = (x, y, z)
    gate_ids = {(x, y, z), (y, z, x), (z, x, y), (z, y, x),
                    (y, x, z), (x, z, y)}
    assert generate_equivalent_ids(gate_seq) == gate_ids

    gate_ids = {Mul(x, y, z), Mul(y, z, x), Mul(z, x, y),
                    Mul(z, y, x), Mul(y, x, z), Mul(x, z, y)}
    assert generate_equivalent_ids(gate_seq, return_as_muls=True) == gate_ids

    gate_seq = (x, y, z, h)
    gate_ids = {(x, y, z, h), (y, z, h, x),
                    (h, x, y, z), (h, z, y, x),
                    (z, y, x, h), (y, x, h, z),
                    (z, h, x, y), (x, h, z, y)}
    assert generate_equivalent_ids(gate_seq) == gate_ids

    gate_seq = (x, y, x, y)
    gate_ids = {(x, y, x, y), (y, x, y, x)}
    assert generate_equivalent_ids(gate_seq) == gate_ids

    cgate_y = CGate((1,), y)
    gate_seq = (y, cgate_y, y, cgate_y)
    gate_ids = {(y, cgate_y, y, cgate_y), (cgate_y, y, cgate_y, y)}
    assert generate_equivalent_ids(gate_seq) == gate_ids

    cnot = CNOT(1, 0)
    cgate_z = CGate((0,), Z(1))
    gate_seq = (cnot, h, cgate_z, h)
    gate_ids = {(cnot, h, cgate_z, h), (h, cgate_z, h, cnot),
                    (h, cnot, h, cgate_z), (cgate_z, h, cnot, h)}
    assert generate_equivalent_ids(gate_seq) == gate_ids


def test_generate_equivalent_ids_2():
    # Test with Muls
    (x, y, z, h) = create_gate_sequence()

    assert generate_equivalent_ids((x,), return_as_muls=True) == {x}

    gate_ids = {Integer(1)}
    assert generate_equivalent_ids(x*x, return_as_muls=True) == gate_ids

    gate_ids = {x*y, y*x}
    assert generate_equivalent_ids(x*y, return_as_muls=True) == gate_ids

    gate_ids = {(x, y), (y, x)}
    assert generate_equivalent_ids(x*y) == gate_ids

    circuit = Mul(*(x, y, z))
    gate_ids = {x*y*z, y*z*x, z*x*y, z*y*x,
                    y*x*z, x*z*y}
    assert generate_equivalent_ids(circuit, return_as_muls=True) == gate_ids

    circuit = Mul(*(x, y, z, h))
    gate_ids = {x*y*z*h, y*z*h*x,
                    h*x*y*z, h*z*y*x,
                    z*y*x*h, y*x*h*z,
                    z*h*x*y, x*h*z*y}
    assert generate_equivalent_ids(circuit, return_as_muls=True) == gate_ids

    circuit = Mul(*(x, y, x, y))
    gate_ids = {x*y*x*y, y*x*y*x}
    assert generate_equivalent_ids(circuit, return_as_muls=True) == gate_ids

    cgate_y = CGate((1,), y)
    circuit = Mul(*(y, cgate_y, y, cgate_y))
    gate_ids = {y*cgate_y*y*cgate_y, cgate_y*y*cgate_y*y}
    assert generate_equivalent_ids(circuit, return_as_muls=True) == gate_ids

    cnot = CNOT(1, 0)
    cgate_z = CGate((0,), Z(1))
    circuit = Mul(*(cnot, h, cgate_z, h))
    gate_ids = {cnot*h*cgate_z*h, h*cgate_z*h*cnot,
                    h*cnot*h*cgate_z, cgate_z*h*cnot*h}
    assert generate_equivalent_ids(circuit, return_as_muls=True) == gate_ids


def test_is_scalar_nonsparse_matrix():
    numqubits = 2
    id_only = False

    id_gate = (IdentityGate(1),)
    actual = is_scalar_nonsparse_matrix(id_gate, numqubits, id_only)
    assert actual is True

    x0 = X(0)
    xx_circuit = (x0, x0)
    actual = is_scalar_nonsparse_matrix(xx_circuit, numqubits, id_only)
    assert actual is True

    x1 = X(1)
    y1 = Y(1)
    xy_circuit = (x1, y1)
    actual = is_scalar_nonsparse_matrix(xy_circuit, numqubits, id_only)
    assert actual is False

    z1 = Z(1)
    xyz_circuit = (x1, y1, z1)
    actual = is_scalar_nonsparse_matrix(xyz_circuit, numqubits, id_only)
    assert actual is True

    cnot = CNOT(1, 0)
    cnot_circuit = (cnot, cnot)
    actual = is_scalar_nonsparse_matrix(cnot_circuit, numqubits, id_only)
    assert actual is True

    h = H(0)
    hh_circuit = (h, h)
    actual = is_scalar_nonsparse_matrix(hh_circuit, numqubits, id_only)
    assert actual is True

    h1 = H(1)
    xhzh_circuit = (x1, h1, z1, h1)
    actual = is_scalar_nonsparse_matrix(xhzh_circuit, numqubits, id_only)
    assert actual is True

    id_only = True
    actual = is_scalar_nonsparse_matrix(xhzh_circuit, numqubits, id_only)
    assert actual is True
    actual = is_scalar_nonsparse_matrix(xyz_circuit, numqubits, id_only)
    assert actual is False
    actual = is_scalar_nonsparse_matrix(cnot_circuit, numqubits, id_only)
    assert actual is True
    actual = is_scalar_nonsparse_matrix(hh_circuit, numqubits, id_only)
    assert actual is True


def test_is_scalar_sparse_matrix():
    np = import_module('numpy')
    if not np:
        skip("numpy not installed.")

    scipy = import_module('scipy', import_kwargs={'fromlist': ['sparse']})
    if not scipy:
        skip("scipy not installed.")

    numqubits = 2
    id_only = False

    id_gate = (IdentityGate(1),)
    assert is_scalar_sparse_matrix(id_gate, numqubits, id_only) is True

    x0 = X(0)
    xx_circuit = (x0, x0)
    assert is_scalar_sparse_matrix(xx_circuit, numqubits, id_only) is True

    x1 = X(1)
    y1 = Y(1)
    xy_circuit = (x1, y1)
    assert is_scalar_sparse_matrix(xy_circuit, numqubits, id_only) is False

    z1 = Z(1)
    xyz_circuit = (x1, y1, z1)
    assert is_scalar_sparse_matrix(xyz_circuit, numqubits, id_only) is True

    cnot = CNOT(1, 0)
    cnot_circuit = (cnot, cnot)
    assert is_scalar_sparse_matrix(cnot_circuit, numqubits, id_only) is True

    h = H(0)
    hh_circuit = (h, h)
    assert is_scalar_sparse_matrix(hh_circuit, numqubits, id_only) is True

    # NOTE:
    # The elements of the sparse matrix for the following circuit
    # is actually 1.0000000000000002+0.0j.
    h1 = H(1)
    xhzh_circuit = (x1, h1, z1, h1)
    assert is_scalar_sparse_matrix(xhzh_circuit, numqubits, id_only) is True

    id_only = True
    assert is_scalar_sparse_matrix(xhzh_circuit, numqubits, id_only) is True
    assert is_scalar_sparse_matrix(xyz_circuit, numqubits, id_only) is False
    assert is_scalar_sparse_matrix(cnot_circuit, numqubits, id_only) is True
    assert is_scalar_sparse_matrix(hh_circuit, numqubits, id_only) is True


def test_is_degenerate():
    (x, y, z, h) = create_gate_sequence()

    gate_id = GateIdentity(x, y, z)
    ids = {gate_id}

    another_id = (z, y, x)
    assert is_degenerate(ids, another_id) is True


def test_is_reducible():
    nqubits = 2
    (x, y, z, h) = create_gate_sequence()

    circuit = (x, y, y)
    assert is_reducible(circuit, nqubits, 1, 3) is True

    circuit = (x, y, x)
    assert is_reducible(circuit, nqubits, 1, 3) is False

    circuit = (x, y, y, x)
    assert is_reducible(circuit, nqubits, 0, 4) is True

    circuit = (x, y, y, x)
    assert is_reducible(circuit, nqubits, 1, 3) is True

    circuit = (x, y, z, y, y)
    assert is_reducible(circuit, nqubits, 1, 5) is True


def test_bfs_identity_search():
    assert bfs_identity_search([], 1) == set()

    (x, y, z, h) = create_gate_sequence()

    gate_list = [x]
    id_set = {GateIdentity(x, x)}
    assert bfs_identity_search(gate_list, 1, max_depth=2) == id_set

    # Set should not contain degenerate quantum circuits
    gate_list = [x, y, z]
    id_set = {GateIdentity(x, x),
                  GateIdentity(y, y),
                  GateIdentity(z, z),
                  GateIdentity(x, y, z)}
    assert bfs_identity_search(gate_list, 1) == id_set

    id_set = {GateIdentity(x, x),
                  GateIdentity(y, y),
                  GateIdentity(z, z),
                  GateIdentity(x, y, z),
                  GateIdentity(x, y, x, y),
                  GateIdentity(x, z, x, z),
                  GateIdentity(y, z, y, z)}
    assert bfs_identity_search(gate_list, 1, max_depth=4) == id_set
    assert bfs_identity_search(gate_list, 1, max_depth=5) == id_set

    gate_list = [x, y, z, h]
    id_set = {GateIdentity(x, x),
                  GateIdentity(y, y),
                  GateIdentity(z, z),
                  GateIdentity(h, h),
                  GateIdentity(x, y, z),
                  GateIdentity(x, y, x, y),
                  GateIdentity(x, z, x, z),
                  GateIdentity(x, h, z, h),
                  GateIdentity(y, z, y, z),
                  GateIdentity(y, h, y, h)}
    assert bfs_identity_search(gate_list, 1) == id_set

    id_set = {GateIdentity(x, x),
                  GateIdentity(y, y),
                  GateIdentity(z, z),
                  GateIdentity(h, h)}
    assert id_set == bfs_identity_search(gate_list, 1, max_depth=3,
                                         identity_only=True)

    id_set = {GateIdentity(x, x),
                  GateIdentity(y, y),
                  GateIdentity(z, z),
                  GateIdentity(h, h),
                  GateIdentity(x, y, z),
                  GateIdentity(x, y, x, y),
                  GateIdentity(x, z, x, z),
                  GateIdentity(x, h, z, h),
                  GateIdentity(y, z, y, z),
                  GateIdentity(y, h, y, h),
                  GateIdentity(x, y, h, x, h),
                  GateIdentity(x, z, h, y, h),
                  GateIdentity(y, z, h, z, h)}
    assert bfs_identity_search(gate_list, 1, max_depth=5) == id_set

    id_set = {GateIdentity(x, x),
                  GateIdentity(y, y),
                  GateIdentity(z, z),
                  GateIdentity(h, h),
                  GateIdentity(x, h, z, h)}
    assert id_set == bfs_identity_search(gate_list, 1, max_depth=4,
                                         identity_only=True)

    cnot = CNOT(1, 0)
    gate_list = [x, cnot]
    id_set = {GateIdentity(x, x),
                  GateIdentity(cnot, cnot),
                  GateIdentity(x, cnot, x, cnot)}
    assert bfs_identity_search(gate_list, 2, max_depth=4) == id_set

    cgate_x = CGate((1,), x)
    gate_list = [x, cgate_x]
    id_set = {GateIdentity(x, x),
                  GateIdentity(cgate_x, cgate_x),
                  GateIdentity(x, cgate_x, x, cgate_x)}
    assert bfs_identity_search(gate_list, 2, max_depth=4) == id_set

    cgate_z = CGate((0,), Z(1))
    gate_list = [cnot, cgate_z, h]
    id_set = {GateIdentity(h, h),
                  GateIdentity(cgate_z, cgate_z),
                  GateIdentity(cnot, cnot),
                  GateIdentity(cnot, h, cgate_z, h)}
    assert bfs_identity_search(gate_list, 2, max_depth=4) == id_set

    s = PhaseGate(0)
    t = TGate(0)
    gate_list = [s, t]
    id_set = {GateIdentity(s, s, s, s)}
    assert bfs_identity_search(gate_list, 1, max_depth=4) == id_set


def test_bfs_identity_search_xfail():
    s = PhaseGate(0)
    t = TGate(0)
    gate_list = [Dagger(s), t]
    id_set = {GateIdentity(Dagger(s), t, t)}
    assert bfs_identity_search(gate_list, 1, max_depth=3) == id_set
