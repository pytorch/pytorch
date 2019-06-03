import pytest
import sys
from pybind11_tests import stl_binders as m

with pytest.suppress(ImportError):
    import numpy as np


def test_vector_int():
    v_int = m.VectorInt([0, 0])
    assert len(v_int) == 2
    assert bool(v_int) is True

    v_int2 = m.VectorInt([0, 0])
    assert v_int == v_int2
    v_int2[1] = 1
    assert v_int != v_int2

    v_int2.append(2)
    v_int2.insert(0, 1)
    v_int2.insert(0, 2)
    v_int2.insert(0, 3)
    v_int2.insert(6, 3)
    assert str(v_int2) == "VectorInt[3, 2, 1, 0, 1, 2, 3]"
    with pytest.raises(IndexError):
        v_int2.insert(8, 4)

    v_int.append(99)
    v_int2[2:-2] = v_int
    assert v_int2 == m.VectorInt([3, 2, 0, 0, 99, 2, 3])
    del v_int2[1:3]
    assert v_int2 == m.VectorInt([3, 0, 99, 2, 3])
    del v_int2[0]
    assert v_int2 == m.VectorInt([0, 99, 2, 3])


# related to the PyPy's buffer protocol.
@pytest.unsupported_on_pypy
def test_vector_buffer():
    b = bytearray([1, 2, 3, 4])
    v = m.VectorUChar(b)
    assert v[1] == 2
    v[2] = 5
    mv = memoryview(v)  # We expose the buffer interface
    if sys.version_info.major > 2:
        assert mv[2] == 5
        mv[2] = 6
    else:
        assert mv[2] == '\x05'
        mv[2] = '\x06'
    assert v[2] == 6

    with pytest.raises(RuntimeError) as excinfo:
        m.create_undeclstruct()  # Undeclared struct contents, no buffer interface
    assert "NumPy type info missing for " in str(excinfo.value)


@pytest.unsupported_on_pypy
@pytest.requires_numpy
def test_vector_buffer_numpy():
    a = np.array([1, 2, 3, 4], dtype=np.int32)
    with pytest.raises(TypeError):
        m.VectorInt(a)

    a = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]], dtype=np.uintc)
    v = m.VectorInt(a[0, :])
    assert len(v) == 4
    assert v[2] == 3
    ma = np.asarray(v)
    ma[2] = 5
    assert v[2] == 5

    v = m.VectorInt(a[:, 1])
    assert len(v) == 3
    assert v[2] == 10

    v = m.get_vectorstruct()
    assert v[0].x == 5
    ma = np.asarray(v)
    ma[1]['x'] = 99
    assert v[1].x == 99

    v = m.VectorStruct(np.zeros(3, dtype=np.dtype([('w', 'bool'), ('x', 'I'),
                                                   ('y', 'float64'), ('z', 'bool')], align=True)))
    assert len(v) == 3


def test_vector_bool():
    import pybind11_cross_module_tests as cm

    vv_c = cm.VectorBool()
    for i in range(10):
        vv_c.append(i % 2 == 0)
    for i in range(10):
        assert vv_c[i] == (i % 2 == 0)
    assert str(vv_c) == "VectorBool[1, 0, 1, 0, 1, 0, 1, 0, 1, 0]"


def test_vector_custom():
    v_a = m.VectorEl()
    v_a.append(m.El(1))
    v_a.append(m.El(2))
    assert str(v_a) == "VectorEl[El{1}, El{2}]"

    vv_a = m.VectorVectorEl()
    vv_a.append(v_a)
    vv_b = vv_a[0]
    assert str(vv_b) == "VectorEl[El{1}, El{2}]"


def test_map_string_double():
    mm = m.MapStringDouble()
    mm['a'] = 1
    mm['b'] = 2.5

    assert list(mm) == ['a', 'b']
    assert list(mm.items()) == [('a', 1), ('b', 2.5)]
    assert str(mm) == "MapStringDouble{a: 1, b: 2.5}"

    um = m.UnorderedMapStringDouble()
    um['ua'] = 1.1
    um['ub'] = 2.6

    assert sorted(list(um)) == ['ua', 'ub']
    assert sorted(list(um.items())) == [('ua', 1.1), ('ub', 2.6)]
    assert "UnorderedMapStringDouble" in str(um)


def test_map_string_double_const():
    mc = m.MapStringDoubleConst()
    mc['a'] = 10
    mc['b'] = 20.5
    assert str(mc) == "MapStringDoubleConst{a: 10, b: 20.5}"

    umc = m.UnorderedMapStringDoubleConst()
    umc['a'] = 11
    umc['b'] = 21.5

    str(umc)


def test_noncopyable_containers():
    # std::vector
    vnc = m.get_vnc(5)
    for i in range(0, 5):
        assert vnc[i].value == i + 1

    for i, j in enumerate(vnc, start=1):
        assert j.value == i

    # std::deque
    dnc = m.get_dnc(5)
    for i in range(0, 5):
        assert dnc[i].value == i + 1

    i = 1
    for j in dnc:
        assert(j.value == i)
        i += 1

    # std::map
    mnc = m.get_mnc(5)
    for i in range(1, 6):
        assert mnc[i].value == 10 * i

    vsum = 0
    for k, v in mnc.items():
        assert v.value == 10 * k
        vsum += v.value

    assert vsum == 150

    # std::unordered_map
    mnc = m.get_umnc(5)
    for i in range(1, 6):
        assert mnc[i].value == 10 * i

    vsum = 0
    for k, v in mnc.items():
        assert v.value == 10 * k
        vsum += v.value

    assert vsum == 150


def test_map_delitem():
    mm = m.MapStringDouble()
    mm['a'] = 1
    mm['b'] = 2.5

    assert list(mm) == ['a', 'b']
    assert list(mm.items()) == [('a', 1), ('b', 2.5)]
    del mm['a']
    assert list(mm) == ['b']
    assert list(mm.items()) == [('b', 2.5)]

    um = m.UnorderedMapStringDouble()
    um['ua'] = 1.1
    um['ub'] = 2.6

    assert sorted(list(um)) == ['ua', 'ub']
    assert sorted(list(um.items())) == [('ua', 1.1), ('ub', 2.6)]
    del um['ua']
    assert sorted(list(um)) == ['ub']
    assert sorted(list(um.items())) == [('ub', 2.6)]
