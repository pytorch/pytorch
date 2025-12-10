"""Primitive circuit operations on quantum circuits."""

from functools import reduce

from sympy.core.sorting import default_sort_key
from sympy.core.containers import Tuple
from sympy.core.mul import Mul
from sympy.core.symbol import Symbol
from sympy.core.sympify import sympify
from sympy.utilities import numbered_symbols
from sympy.physics.quantum.gate import Gate

__all__ = [
    'kmp_table',
    'find_subcircuit',
    'replace_subcircuit',
    'convert_to_symbolic_indices',
    'convert_to_real_indices',
    'random_reduce',
    'random_insert'
]


def kmp_table(word):
    """Build the 'partial match' table of the Knuth-Morris-Pratt algorithm.

    Note: This is applicable to strings or
    quantum circuits represented as tuples.
    """

    # Current position in subcircuit
    pos = 2
    # Beginning position of candidate substring that
    # may reappear later in word
    cnd = 0
    # The 'partial match' table that helps one determine
    # the next location to start substring search
    table = []
    table.append(-1)
    table.append(0)

    while pos < len(word):
        if word[pos - 1] == word[cnd]:
            cnd = cnd + 1
            table.append(cnd)
            pos = pos + 1
        elif cnd > 0:
            cnd = table[cnd]
        else:
            table.append(0)
            pos = pos + 1

    return table


def find_subcircuit(circuit, subcircuit, start=0, end=0):
    """Finds the subcircuit in circuit, if it exists.

    Explanation
    ===========

    If the subcircuit exists, the index of the start of
    the subcircuit in circuit is returned; otherwise,
    -1 is returned.  The algorithm that is implemented
    is the Knuth-Morris-Pratt algorithm.

    Parameters
    ==========

    circuit : tuple, Gate or Mul
        A tuple of Gates or Mul representing a quantum circuit
    subcircuit : tuple, Gate or Mul
        A tuple of Gates or Mul to find in circuit
    start : int
        The location to start looking for subcircuit.
        If start is the same or past end, -1 is returned.
    end : int
        The last place to look for a subcircuit.  If end
        is less than 1 (one), then the length of circuit
        is taken to be end.

    Examples
    ========

    Find the first instance of a subcircuit:

    >>> from sympy.physics.quantum.circuitutils import find_subcircuit
    >>> from sympy.physics.quantum.gate import X, Y, Z, H
    >>> circuit = X(0)*Z(0)*Y(0)*H(0)
    >>> subcircuit = Z(0)*Y(0)
    >>> find_subcircuit(circuit, subcircuit)
    1

    Find the first instance starting at a specific position:

    >>> find_subcircuit(circuit, subcircuit, start=1)
    1

    >>> find_subcircuit(circuit, subcircuit, start=2)
    -1

    >>> circuit = circuit*subcircuit
    >>> find_subcircuit(circuit, subcircuit, start=2)
    4

    Find the subcircuit within some interval:

    >>> find_subcircuit(circuit, subcircuit, start=2, end=2)
    -1
    """

    if isinstance(circuit, Mul):
        circuit = circuit.args

    if isinstance(subcircuit, Mul):
        subcircuit = subcircuit.args

    if len(subcircuit) == 0 or len(subcircuit) > len(circuit):
        return -1

    if end < 1:
        end = len(circuit)

    # Location in circuit
    pos = start
    # Location in the subcircuit
    index = 0
    # 'Partial match' table
    table = kmp_table(subcircuit)

    while (pos + index) < end:
        if subcircuit[index] == circuit[pos + index]:
            index = index + 1
        else:
            pos = pos + index - table[index]
            index = table[index] if table[index] > -1 else 0

        if index == len(subcircuit):
            return pos

    return -1


def replace_subcircuit(circuit, subcircuit, replace=None, pos=0):
    """Replaces a subcircuit with another subcircuit in circuit,
    if it exists.

    Explanation
    ===========

    If multiple instances of subcircuit exists, the first instance is
    replaced.  The position to being searching from (if different from
    0) may be optionally given.  If subcircuit cannot be found, circuit
    is returned.

    Parameters
    ==========

    circuit : tuple, Gate or Mul
        A quantum circuit.
    subcircuit : tuple, Gate or Mul
        The circuit to be replaced.
    replace : tuple, Gate or Mul
        The replacement circuit.
    pos : int
        The location to start search and replace
        subcircuit, if it exists.  This may be used
        if it is known beforehand that multiple
        instances exist, and it is desirable to
        replace a specific instance.  If a negative number
        is given, pos will be defaulted to 0.

    Examples
    ========

    Find and remove the subcircuit:

    >>> from sympy.physics.quantum.circuitutils import replace_subcircuit
    >>> from sympy.physics.quantum.gate import X, Y, Z, H
    >>> circuit = X(0)*Z(0)*Y(0)*H(0)*X(0)*H(0)*Y(0)
    >>> subcircuit = Z(0)*Y(0)
    >>> replace_subcircuit(circuit, subcircuit)
    (X(0), H(0), X(0), H(0), Y(0))

    Remove the subcircuit given a starting search point:

    >>> replace_subcircuit(circuit, subcircuit, pos=1)
    (X(0), H(0), X(0), H(0), Y(0))

    >>> replace_subcircuit(circuit, subcircuit, pos=2)
    (X(0), Z(0), Y(0), H(0), X(0), H(0), Y(0))

    Replace the subcircuit:

    >>> replacement = H(0)*Z(0)
    >>> replace_subcircuit(circuit, subcircuit, replace=replacement)
    (X(0), H(0), Z(0), H(0), X(0), H(0), Y(0))
    """

    if pos < 0:
        pos = 0

    if isinstance(circuit, Mul):
        circuit = circuit.args

    if isinstance(subcircuit, Mul):
        subcircuit = subcircuit.args

    if isinstance(replace, Mul):
        replace = replace.args
    elif replace is None:
        replace = ()

    # Look for the subcircuit starting at pos
    loc = find_subcircuit(circuit, subcircuit, start=pos)

    # If subcircuit was found
    if loc > -1:
        # Get the gates to the left of subcircuit
        left = circuit[0:loc]
        # Get the gates to the right of subcircuit
        right = circuit[loc + len(subcircuit):len(circuit)]
        # Recombine the left and right side gates into a circuit
        circuit = left + replace + right

    return circuit


def _sympify_qubit_map(mapping):
    new_map = {}
    for key in mapping:
        new_map[key] = sympify(mapping[key])
    return new_map


def convert_to_symbolic_indices(seq, start=None, gen=None, qubit_map=None):
    """Returns the circuit with symbolic indices and the
    dictionary mapping symbolic indices to real indices.

    The mapping is 1 to 1 and onto (bijective).

    Parameters
    ==========

    seq : tuple, Gate/Integer/tuple or Mul
        A tuple of Gate, Integer, or tuple objects, or a Mul
    start : Symbol
        An optional starting symbolic index
    gen : object
        An optional numbered symbol generator
    qubit_map : dict
        An existing mapping of symbolic indices to real indices

    All symbolic indices have the format 'i#', where # is
    some number >= 0.
    """

    if isinstance(seq, Mul):
        seq = seq.args

    # A numbered symbol generator
    index_gen = numbered_symbols(prefix='i', start=-1)
    cur_ndx = next(index_gen)

    # keys are symbolic indices; values are real indices
    ndx_map = {}

    def create_inverse_map(symb_to_real_map):
        rev_items = lambda item: (item[1], item[0])
        return dict(map(rev_items, symb_to_real_map.items()))

    if start is not None:
        if not isinstance(start, Symbol):
            msg = 'Expected Symbol for starting index, got %r.' % start
            raise TypeError(msg)
        cur_ndx = start

    if gen is not None:
        if not isinstance(gen, numbered_symbols().__class__):
            msg = 'Expected a generator, got %r.' % gen
            raise TypeError(msg)
        index_gen = gen

    if qubit_map is not None:
        if not isinstance(qubit_map, dict):
            msg = ('Expected dict for existing map, got ' +
                   '%r.' % qubit_map)
            raise TypeError(msg)
        ndx_map = qubit_map

    ndx_map = _sympify_qubit_map(ndx_map)
    # keys are real indices; keys are symbolic indices
    inv_map = create_inverse_map(ndx_map)

    sym_seq = ()
    for item in seq:
        # Nested items, so recurse
        if isinstance(item, Gate):
            result = convert_to_symbolic_indices(item.args,
                                                 qubit_map=ndx_map,
                                                 start=cur_ndx,
                                                 gen=index_gen)
            sym_item, new_map, cur_ndx, index_gen = result
            ndx_map.update(new_map)
            inv_map = create_inverse_map(ndx_map)

        elif isinstance(item, (tuple, Tuple)):
            result = convert_to_symbolic_indices(item,
                                                 qubit_map=ndx_map,
                                                 start=cur_ndx,
                                                 gen=index_gen)
            sym_item, new_map, cur_ndx, index_gen = result
            ndx_map.update(new_map)
            inv_map = create_inverse_map(ndx_map)

        elif item in inv_map:
            sym_item = inv_map[item]

        else:
            cur_ndx = next(gen)
            ndx_map[cur_ndx] = item
            inv_map[item] = cur_ndx
            sym_item = cur_ndx

        if isinstance(item, Gate):
            sym_item = item.__class__(*sym_item)

        sym_seq = sym_seq + (sym_item,)

    return sym_seq, ndx_map, cur_ndx, index_gen


def convert_to_real_indices(seq, qubit_map):
    """Returns the circuit with real indices.

    Parameters
    ==========

    seq : tuple, Gate/Integer/tuple or Mul
        A tuple of Gate, Integer, or tuple objects or a Mul
    qubit_map : dict
        A dictionary mapping symbolic indices to real indices.

    Examples
    ========

    Change the symbolic indices to real integers:

    >>> from sympy import symbols
    >>> from sympy.physics.quantum.circuitutils import convert_to_real_indices
    >>> from sympy.physics.quantum.gate import X, Y, H
    >>> i0, i1 = symbols('i:2')
    >>> index_map = {i0 : 0, i1 : 1}
    >>> convert_to_real_indices(X(i0)*Y(i1)*H(i0)*X(i1), index_map)
    (X(0), Y(1), H(0), X(1))
    """

    if isinstance(seq, Mul):
        seq = seq.args

    if not isinstance(qubit_map, dict):
        msg = 'Expected dict for qubit_map, got %r.' % qubit_map
        raise TypeError(msg)

    qubit_map = _sympify_qubit_map(qubit_map)
    real_seq = ()
    for item in seq:
        # Nested items, so recurse
        if isinstance(item, Gate):
            real_item = convert_to_real_indices(item.args, qubit_map)

        elif isinstance(item, (tuple, Tuple)):
            real_item = convert_to_real_indices(item, qubit_map)

        else:
            real_item = qubit_map[item]

        if isinstance(item, Gate):
            real_item = item.__class__(*real_item)

        real_seq = real_seq + (real_item,)

    return real_seq


def random_reduce(circuit, gate_ids, seed=None):
    """Shorten the length of a quantum circuit.

    Explanation
    ===========

    random_reduce looks for circuit identities in circuit, randomly chooses
    one to remove, and returns a shorter yet equivalent circuit.  If no
    identities are found, the same circuit is returned.

    Parameters
    ==========

    circuit : Gate tuple of Mul
        A tuple of Gates representing a quantum circuit
    gate_ids : list, GateIdentity
        List of gate identities to find in circuit
    seed : int or list
        seed used for _randrange; to override the random selection, provide a
        list of integers: the elements of gate_ids will be tested in the order
        given by the list

    """
    from sympy.core.random import _randrange

    if not gate_ids:
        return circuit

    if isinstance(circuit, Mul):
        circuit = circuit.args

    ids = flatten_ids(gate_ids)

    # Create the random integer generator with the seed
    randrange = _randrange(seed)

    # Look for an identity in the circuit
    while ids:
        i = randrange(len(ids))
        id = ids.pop(i)
        if find_subcircuit(circuit, id) != -1:
            break
    else:
        # no identity was found
        return circuit

    # return circuit with the identity removed
    return replace_subcircuit(circuit, id)


def random_insert(circuit, choices, seed=None):
    """Insert a circuit into another quantum circuit.

    Explanation
    ===========

    random_insert randomly chooses a location in the circuit to insert
    a randomly selected circuit from amongst the given choices.

    Parameters
    ==========

    circuit : Gate tuple or Mul
        A tuple or Mul of Gates representing a quantum circuit
    choices : list
        Set of circuit choices
    seed : int or list
        seed used for _randrange; to override the random selections, give
        a list two integers, [i, j] where i is the circuit location where
        choice[j] will be inserted.

    Notes
    =====

    Indices for insertion should be [0, n] if n is the length of the
    circuit.
    """
    from sympy.core.random import _randrange

    if not choices:
        return circuit

    if isinstance(circuit, Mul):
        circuit = circuit.args

    # get the location in the circuit and the element to insert from choices
    randrange = _randrange(seed)
    loc = randrange(len(circuit) + 1)
    choice = choices[randrange(len(choices))]

    circuit = list(circuit)
    circuit[loc: loc] = choice
    return tuple(circuit)

# Flatten the GateIdentity objects (with gate rules) into one single list


def flatten_ids(ids):
    collapse = lambda acc, an_id: acc + sorted(an_id.equivalent_ids,
                                        key=default_sort_key)
    ids = reduce(collapse, ids, [])
    ids.sort(key=default_sort_key)
    return ids
