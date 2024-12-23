from sympy.core.mul import Mul
from sympy.core.numbers import Integer
from sympy.core.symbol import Symbol
from sympy.utilities import numbered_symbols
from sympy.physics.quantum.gate import X, Y, Z, H, CNOT, CGate
from sympy.physics.quantum.identitysearch import bfs_identity_search
from sympy.physics.quantum.circuitutils import (kmp_table, find_subcircuit,
        replace_subcircuit, convert_to_symbolic_indices,
        convert_to_real_indices, random_reduce, random_insert,
        flatten_ids)
from sympy.testing.pytest import slow


def create_gate_sequence(qubit=0):
    gates = (X(qubit), Y(qubit), Z(qubit), H(qubit))
    return gates


def test_kmp_table():
    word = ('a', 'b', 'c', 'd', 'a', 'b', 'd')
    expected_table = [-1, 0, 0, 0, 0, 1, 2]
    assert expected_table == kmp_table(word)

    word = ('P', 'A', 'R', 'T', 'I', 'C', 'I', 'P', 'A', 'T', 'E', ' ',
            'I', 'N', ' ', 'P', 'A', 'R', 'A', 'C', 'H', 'U', 'T', 'E')
    expected_table = [-1, 0, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0,
                      0, 0, 0, 0, 1, 2, 3, 0, 0, 0, 0, 0]
    assert expected_table == kmp_table(word)

    x = X(0)
    y = Y(0)
    z = Z(0)
    h = H(0)
    word = (x, y, y, x, z)
    expected_table = [-1, 0, 0, 0, 1]
    assert expected_table == kmp_table(word)

    word = (x, x, y, h, z)
    expected_table = [-1, 0, 1, 0, 0]
    assert expected_table == kmp_table(word)


def test_find_subcircuit():
    x = X(0)
    y = Y(0)
    z = Z(0)
    h = H(0)
    x1 = X(1)
    y1 = Y(1)

    i0 = Symbol('i0')
    x_i0 = X(i0)
    y_i0 = Y(i0)
    z_i0 = Z(i0)
    h_i0 = H(i0)

    circuit = (x, y, z)

    assert find_subcircuit(circuit, (x,)) == 0
    assert find_subcircuit(circuit, (x1,)) == -1
    assert find_subcircuit(circuit, (y,)) == 1
    assert find_subcircuit(circuit, (h,)) == -1
    assert find_subcircuit(circuit, Mul(x, h)) == -1
    assert find_subcircuit(circuit, Mul(x, y, z)) == 0
    assert find_subcircuit(circuit, Mul(y, z)) == 1
    assert find_subcircuit(Mul(*circuit), (x, y, z, h)) == -1
    assert find_subcircuit(Mul(*circuit), (z, y, x)) == -1
    assert find_subcircuit(circuit, (x,), start=2, end=1) == -1

    circuit = (x, y, x, y, z)
    assert find_subcircuit(Mul(*circuit), Mul(x, y, z)) == 2
    assert find_subcircuit(circuit, (x,), start=1) == 2
    assert find_subcircuit(circuit, (x, y), start=1, end=2) == -1
    assert find_subcircuit(Mul(*circuit), (x, y), start=1, end=3) == -1
    assert find_subcircuit(circuit, (x, y), start=1, end=4) == 2
    assert find_subcircuit(circuit, (x, y), start=2, end=4) == 2

    circuit = (x, y, z, x1, x, y, z, h, x, y, x1,
               x, y, z, h, y1, h)
    assert find_subcircuit(circuit, (x, y, z, h, y1)) == 11

    circuit = (x, y, x_i0, y_i0, z_i0, z)
    assert find_subcircuit(circuit, (x_i0, y_i0, z_i0)) == 2

    circuit = (x_i0, y_i0, z_i0, x_i0, y_i0, h_i0)
    subcircuit = (x_i0, y_i0, z_i0)
    result = find_subcircuit(circuit, subcircuit)
    assert result == 0


def test_replace_subcircuit():
    x = X(0)
    y = Y(0)
    z = Z(0)
    h = H(0)
    cnot = CNOT(1, 0)
    cgate_z = CGate((0,), Z(1))

    # Standard cases
    circuit = (z, y, x, x)
    remove = (z, y, x)
    assert replace_subcircuit(circuit, Mul(*remove)) == (x,)
    assert replace_subcircuit(circuit, remove + (x,)) == ()
    assert replace_subcircuit(circuit, remove, pos=1) == circuit
    assert replace_subcircuit(circuit, remove, pos=0) == (x,)
    assert replace_subcircuit(circuit, (x, x), pos=2) == (z, y)
    assert replace_subcircuit(circuit, (h,)) == circuit

    circuit = (x, y, x, y, z)
    remove = (x, y, z)
    assert replace_subcircuit(Mul(*circuit), Mul(*remove)) == (x, y)
    remove = (x, y, x, y)
    assert replace_subcircuit(circuit, remove) == (z,)

    circuit = (x, h, cgate_z, h, cnot)
    remove = (x, h, cgate_z)
    assert replace_subcircuit(circuit, Mul(*remove), pos=-1) == (h, cnot)
    assert replace_subcircuit(circuit, remove, pos=1) == circuit
    remove = (h, h)
    assert replace_subcircuit(circuit, remove) == circuit
    remove = (h, cgate_z, h, cnot)
    assert replace_subcircuit(circuit, remove) == (x,)

    replace = (h, x)
    actual = replace_subcircuit(circuit, remove,
                     replace=replace)
    assert actual == (x, h, x)

    circuit = (x, y, h, x, y, z)
    remove = (x, y)
    replace = (cnot, cgate_z)
    actual = replace_subcircuit(circuit, remove,
                     replace=Mul(*replace))
    assert actual == (cnot, cgate_z, h, x, y, z)

    actual = replace_subcircuit(circuit, remove,
                     replace=replace, pos=1)
    assert actual == (x, y, h, cnot, cgate_z, z)


def test_convert_to_symbolic_indices():
    (x, y, z, h) = create_gate_sequence()

    i0 = Symbol('i0')
    exp_map = {i0: Integer(0)}
    actual, act_map, sndx, gen = convert_to_symbolic_indices((x,))
    assert actual == (X(i0),)
    assert act_map == exp_map

    expected = (X(i0), Y(i0), Z(i0), H(i0))
    exp_map = {i0: Integer(0)}
    actual, act_map, sndx, gen = convert_to_symbolic_indices((x, y, z, h))
    assert actual == expected
    assert exp_map == act_map

    (x1, y1, z1, h1) = create_gate_sequence(1)
    i1 = Symbol('i1')

    expected = (X(i0), Y(i0), Z(i0), H(i0))
    exp_map = {i0: Integer(1)}
    actual, act_map, sndx, gen = convert_to_symbolic_indices((x1, y1, z1, h1))
    assert actual == expected
    assert act_map == exp_map

    expected = (X(i0), Y(i0), Z(i0), H(i0), X(i1), Y(i1), Z(i1), H(i1))
    exp_map = {i0: Integer(0), i1: Integer(1)}
    actual, act_map, sndx, gen = convert_to_symbolic_indices((x, y, z, h,
                                         x1, y1, z1, h1))
    assert actual == expected
    assert act_map == exp_map

    exp_map = {i0: Integer(1), i1: Integer(0)}
    actual, act_map, sndx, gen = convert_to_symbolic_indices(Mul(x1, y1,
                                         z1, h1, x, y, z, h))
    assert actual == expected
    assert act_map == exp_map

    expected = (X(i0), X(i1), Y(i0), Y(i1), Z(i0), Z(i1), H(i0), H(i1))
    exp_map = {i0: Integer(0), i1: Integer(1)}
    actual, act_map, sndx, gen = convert_to_symbolic_indices(Mul(x, x1,
                                         y, y1, z, z1, h, h1))
    assert actual == expected
    assert act_map == exp_map

    exp_map = {i0: Integer(1), i1: Integer(0)}
    actual, act_map, sndx, gen = convert_to_symbolic_indices((x1, x, y1, y,
                                         z1, z, h1, h))
    assert actual == expected
    assert act_map == exp_map

    cnot_10 = CNOT(1, 0)
    cnot_01 = CNOT(0, 1)
    cgate_z_10 = CGate(1, Z(0))
    cgate_z_01 = CGate(0, Z(1))

    expected = (X(i0), X(i1), Y(i0), Y(i1), Z(i0), Z(i1),
                H(i0), H(i1), CNOT(i1, i0), CNOT(i0, i1),
                CGate(i1, Z(i0)), CGate(i0, Z(i1)))
    exp_map = {i0: Integer(0), i1: Integer(1)}
    args = (x, x1, y, y1, z, z1, h, h1, cnot_10, cnot_01,
            cgate_z_10, cgate_z_01)
    actual, act_map, sndx, gen = convert_to_symbolic_indices(args)
    assert actual == expected
    assert act_map == exp_map

    args = (x1, x, y1, y, z1, z, h1, h, cnot_10, cnot_01,
            cgate_z_10, cgate_z_01)
    expected = (X(i0), X(i1), Y(i0), Y(i1), Z(i0), Z(i1),
                H(i0), H(i1), CNOT(i0, i1), CNOT(i1, i0),
                CGate(i0, Z(i1)), CGate(i1, Z(i0)))
    exp_map = {i0: Integer(1), i1: Integer(0)}
    actual, act_map, sndx, gen = convert_to_symbolic_indices(args)
    assert actual == expected
    assert act_map == exp_map

    args = (cnot_10, h, cgate_z_01, h)
    expected = (CNOT(i0, i1), H(i1), CGate(i1, Z(i0)), H(i1))
    exp_map = {i0: Integer(1), i1: Integer(0)}
    actual, act_map, sndx, gen = convert_to_symbolic_indices(args)
    assert actual == expected
    assert act_map == exp_map

    args = (cnot_01, h1, cgate_z_10, h1)
    exp_map = {i0: Integer(0), i1: Integer(1)}
    actual, act_map, sndx, gen = convert_to_symbolic_indices(args)
    assert actual == expected
    assert act_map == exp_map

    args = (cnot_10, h1, cgate_z_01, h1)
    expected = (CNOT(i0, i1), H(i0), CGate(i1, Z(i0)), H(i0))
    exp_map = {i0: Integer(1), i1: Integer(0)}
    actual, act_map, sndx, gen = convert_to_symbolic_indices(args)
    assert actual == expected
    assert act_map == exp_map

    i2 = Symbol('i2')
    ccgate_z = CGate(0, CGate(1, Z(2)))
    ccgate_x = CGate(1, CGate(2, X(0)))
    args = (ccgate_z, ccgate_x)

    expected = (CGate(i0, CGate(i1, Z(i2))), CGate(i1, CGate(i2, X(i0))))
    exp_map = {i0: Integer(0), i1: Integer(1), i2: Integer(2)}
    actual, act_map, sndx, gen = convert_to_symbolic_indices(args)
    assert actual == expected
    assert act_map == exp_map

    ndx_map = {i0: Integer(0)}
    index_gen = numbered_symbols(prefix='i', start=1)
    actual, act_map, sndx, gen = convert_to_symbolic_indices(args,
                                         qubit_map=ndx_map,
                                         start=i0,
                                         gen=index_gen)
    assert actual == expected
    assert act_map == exp_map

    i3 = Symbol('i3')
    cgate_x0_c321 = CGate((3, 2, 1), X(0))
    exp_map = {i0: Integer(3), i1: Integer(2),
               i2: Integer(1), i3: Integer(0)}
    expected = (CGate((i0, i1, i2), X(i3)),)
    args = (cgate_x0_c321,)
    actual, act_map, sndx, gen = convert_to_symbolic_indices(args)
    assert actual == expected
    assert act_map == exp_map


def test_convert_to_real_indices():
    i0 = Symbol('i0')
    i1 = Symbol('i1')

    (x, y, z, h) = create_gate_sequence()

    x_i0 = X(i0)
    y_i0 = Y(i0)
    z_i0 = Z(i0)

    qubit_map = {i0: 0}
    args = (z_i0, y_i0, x_i0)
    expected = (z, y, x)
    actual = convert_to_real_indices(args, qubit_map)
    assert actual == expected

    cnot_10 = CNOT(1, 0)
    cnot_01 = CNOT(0, 1)
    cgate_z_10 = CGate(1, Z(0))
    cgate_z_01 = CGate(0, Z(1))

    cnot_i1_i0 = CNOT(i1, i0)
    cnot_i0_i1 = CNOT(i0, i1)
    cgate_z_i1_i0 = CGate(i1, Z(i0))

    qubit_map = {i0: 0, i1: 1}
    args = (cnot_i1_i0,)
    expected = (cnot_10,)
    actual = convert_to_real_indices(args, qubit_map)
    assert actual == expected

    args = (cgate_z_i1_i0,)
    expected = (cgate_z_10,)
    actual = convert_to_real_indices(args, qubit_map)
    assert actual == expected

    args = (cnot_i0_i1,)
    expected = (cnot_01,)
    actual = convert_to_real_indices(args, qubit_map)
    assert actual == expected

    qubit_map = {i0: 1, i1: 0}
    args = (cgate_z_i1_i0,)
    expected = (cgate_z_01,)
    actual = convert_to_real_indices(args, qubit_map)
    assert actual == expected

    i2 = Symbol('i2')
    ccgate_z = CGate(i0, CGate(i1, Z(i2)))
    ccgate_x = CGate(i1, CGate(i2, X(i0)))

    qubit_map = {i0: 0, i1: 1, i2: 2}
    args = (ccgate_z, ccgate_x)
    expected = (CGate(0, CGate(1, Z(2))), CGate(1, CGate(2, X(0))))
    actual = convert_to_real_indices(Mul(*args), qubit_map)
    assert actual == expected

    qubit_map = {i0: 1, i2: 0, i1: 2}
    args = (ccgate_x, ccgate_z)
    expected = (CGate(2, CGate(0, X(1))), CGate(1, CGate(2, Z(0))))
    actual = convert_to_real_indices(args, qubit_map)
    assert actual == expected


@slow
def test_random_reduce():
    x = X(0)
    y = Y(0)
    z = Z(0)
    h = H(0)
    cnot = CNOT(1, 0)
    cgate_z = CGate((0,), Z(1))

    gate_list = [x, y, z]
    ids = list(bfs_identity_search(gate_list, 1, max_depth=4))

    circuit = (x, y, h, z, cnot)
    assert random_reduce(circuit, []) == circuit
    assert random_reduce(circuit, ids) == circuit

    seq = [2, 11, 9, 3, 5]
    circuit = (x, y, z, x, y, h)
    assert random_reduce(circuit, ids, seed=seq) == (x, y, h)

    circuit = (x, x, y, y, z, z)
    assert random_reduce(circuit, ids, seed=seq) == (x, x, y, y)

    seq = [14, 13, 0]
    assert random_reduce(circuit, ids, seed=seq) == (y, y, z, z)

    gate_list = [x, y, z, h, cnot, cgate_z]
    ids = list(bfs_identity_search(gate_list, 2, max_depth=4))

    seq = [25]
    circuit = (x, y, z, y, h, y, h, cgate_z, h, cnot)
    expected = (x, y, z, cgate_z, h, cnot)
    assert random_reduce(circuit, ids, seed=seq) == expected
    circuit = Mul(*circuit)
    assert random_reduce(circuit, ids, seed=seq) == expected


@slow
def test_random_insert():
    x = X(0)
    y = Y(0)
    z = Z(0)
    h = H(0)
    cnot = CNOT(1, 0)
    cgate_z = CGate((0,), Z(1))

    choices = [(x, x)]
    circuit = (y, y)
    loc, choice = 0, 0
    actual = random_insert(circuit, choices, seed=[loc, choice])
    assert actual == (x, x, y, y)

    circuit = (x, y, z, h)
    choices = [(h, h), (x, y, z)]
    expected = (x, x, y, z, y, z, h)
    loc, choice = 1, 1
    actual = random_insert(circuit, choices, seed=[loc, choice])
    assert actual == expected

    gate_list = [x, y, z, h, cnot, cgate_z]
    ids = list(bfs_identity_search(gate_list, 2, max_depth=4))

    eq_ids = flatten_ids(ids)

    circuit = (x, y, h, cnot, cgate_z)
    expected = (x, z, x, z, x, y, h, cnot, cgate_z)
    loc, choice = 1, 30
    actual = random_insert(circuit, eq_ids, seed=[loc, choice])
    assert actual == expected
    circuit = Mul(*circuit)
    actual = random_insert(circuit, eq_ids, seed=[loc, choice])
    assert actual == expected
