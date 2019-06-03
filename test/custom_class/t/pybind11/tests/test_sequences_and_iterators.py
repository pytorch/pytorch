import pytest
from pybind11_tests import sequences_and_iterators as m
from pybind11_tests import ConstructorStats


def isclose(a, b, rel_tol=1e-05, abs_tol=0.0):
    """Like math.isclose() from Python 3.5"""
    return abs(a - b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)


def allclose(a_list, b_list, rel_tol=1e-05, abs_tol=0.0):
    return all(isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol) for a, b in zip(a_list, b_list))


def test_generalized_iterators():
    assert list(m.IntPairs([(1, 2), (3, 4), (0, 5)]).nonzero()) == [(1, 2), (3, 4)]
    assert list(m.IntPairs([(1, 2), (2, 0), (0, 3), (4, 5)]).nonzero()) == [(1, 2)]
    assert list(m.IntPairs([(0, 3), (1, 2), (3, 4)]).nonzero()) == []

    assert list(m.IntPairs([(1, 2), (3, 4), (0, 5)]).nonzero_keys()) == [1, 3]
    assert list(m.IntPairs([(1, 2), (2, 0), (0, 3), (4, 5)]).nonzero_keys()) == [1]
    assert list(m.IntPairs([(0, 3), (1, 2), (3, 4)]).nonzero_keys()) == []

    # __next__ must continue to raise StopIteration
    it = m.IntPairs([(0, 0)]).nonzero()
    for _ in range(3):
        with pytest.raises(StopIteration):
            next(it)

    it = m.IntPairs([(0, 0)]).nonzero_keys()
    for _ in range(3):
        with pytest.raises(StopIteration):
            next(it)


def test_sequence():
    cstats = ConstructorStats.get(m.Sequence)

    s = m.Sequence(5)
    assert cstats.values() == ['of size', '5']

    assert "Sequence" in repr(s)
    assert len(s) == 5
    assert s[0] == 0 and s[3] == 0
    assert 12.34 not in s
    s[0], s[3] = 12.34, 56.78
    assert 12.34 in s
    assert isclose(s[0], 12.34) and isclose(s[3], 56.78)

    rev = reversed(s)
    assert cstats.values() == ['of size', '5']

    rev2 = s[::-1]
    assert cstats.values() == ['of size', '5']

    it = iter(m.Sequence(0))
    for _ in range(3):  # __next__ must continue to raise StopIteration
        with pytest.raises(StopIteration):
            next(it)
    assert cstats.values() == ['of size', '0']

    expected = [0, 56.78, 0, 0, 12.34]
    assert allclose(rev, expected)
    assert allclose(rev2, expected)
    assert rev == rev2

    rev[0::2] = m.Sequence([2.0, 2.0, 2.0])
    assert cstats.values() == ['of size', '3', 'from std::vector']

    assert allclose(rev, [2, 56.78, 2, 0, 2])

    assert cstats.alive() == 4
    del it
    assert cstats.alive() == 3
    del s
    assert cstats.alive() == 2
    del rev
    assert cstats.alive() == 1
    del rev2
    assert cstats.alive() == 0

    assert cstats.values() == []
    assert cstats.default_constructions == 0
    assert cstats.copy_constructions == 0
    assert cstats.move_constructions >= 1
    assert cstats.copy_assignments == 0
    assert cstats.move_assignments == 0


def test_map_iterator():
    sm = m.StringMap({'hi': 'bye', 'black': 'white'})
    assert sm['hi'] == 'bye'
    assert len(sm) == 2
    assert sm['black'] == 'white'

    with pytest.raises(KeyError):
        assert sm['orange']
    sm['orange'] = 'banana'
    assert sm['orange'] == 'banana'

    expected = {'hi': 'bye', 'black': 'white', 'orange': 'banana'}
    for k in sm:
        assert sm[k] == expected[k]
    for k, v in sm.items():
        assert v == expected[k]

    it = iter(m.StringMap({}))
    for _ in range(3):  # __next__ must continue to raise StopIteration
        with pytest.raises(StopIteration):
            next(it)


def test_python_iterator_in_cpp():
    t = (1, 2, 3)
    assert m.object_to_list(t) == [1, 2, 3]
    assert m.object_to_list(iter(t)) == [1, 2, 3]
    assert m.iterator_to_list(iter(t)) == [1, 2, 3]

    with pytest.raises(TypeError) as excinfo:
        m.object_to_list(1)
    assert "object is not iterable" in str(excinfo.value)

    with pytest.raises(TypeError) as excinfo:
        m.iterator_to_list(1)
    assert "incompatible function arguments" in str(excinfo.value)

    def bad_next_call():
        raise RuntimeError("py::iterator::advance() should propagate errors")

    with pytest.raises(RuntimeError) as excinfo:
        m.iterator_to_list(iter(bad_next_call, None))
    assert str(excinfo.value) == "py::iterator::advance() should propagate errors"

    lst = [1, None, 0, None]
    assert m.count_none(lst) == 2
    assert m.find_none(lst) is True
    assert m.count_nonzeros({"a": 0, "b": 1, "c": 2}) == 2

    r = range(5)
    assert all(m.tuple_iterator(tuple(r)))
    assert all(m.list_iterator(list(r)))
    assert all(m.sequence_iterator(r))


def test_iterator_passthrough():
    """#181: iterator passthrough did not compile"""
    from pybind11_tests.sequences_and_iterators import iterator_passthrough

    assert list(iterator_passthrough(iter([3, 5, 7, 9, 11, 13, 15]))) == [3, 5, 7, 9, 11, 13, 15]


def test_iterator_rvp():
    """#388: Can't make iterators via make_iterator() with different r/v policies """
    import pybind11_tests.sequences_and_iterators as m

    assert list(m.make_iterator_1()) == [1, 2, 3]
    assert list(m.make_iterator_2()) == [1, 2, 3]
    assert not isinstance(m.make_iterator_1(), type(m.make_iterator_2()))
