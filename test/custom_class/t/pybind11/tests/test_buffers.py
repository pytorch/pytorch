import struct
import pytest
from pybind11_tests import buffers as m
from pybind11_tests import ConstructorStats

pytestmark = pytest.requires_numpy

with pytest.suppress(ImportError):
    import numpy as np


def test_from_python():
    with pytest.raises(RuntimeError) as excinfo:
        m.Matrix(np.array([1, 2, 3]))  # trying to assign a 1D array
    assert str(excinfo.value) == "Incompatible buffer format!"

    m3 = np.array([[1, 2, 3], [4, 5, 6]]).astype(np.float32)
    m4 = m.Matrix(m3)

    for i in range(m4.rows()):
        for j in range(m4.cols()):
            assert m3[i, j] == m4[i, j]

    cstats = ConstructorStats.get(m.Matrix)
    assert cstats.alive() == 1
    del m3, m4
    assert cstats.alive() == 0
    assert cstats.values() == ["2x3 matrix"]
    assert cstats.copy_constructions == 0
    # assert cstats.move_constructions >= 0  # Don't invoke any
    assert cstats.copy_assignments == 0
    assert cstats.move_assignments == 0


# PyPy: Memory leak in the "np.array(m, copy=False)" call
# https://bitbucket.org/pypy/pypy/issues/2444
@pytest.unsupported_on_pypy
def test_to_python():
    mat = m.Matrix(5, 4)
    assert memoryview(mat).shape == (5, 4)

    assert mat[2, 3] == 0
    mat[2, 3] = 4.0
    mat[3, 2] = 7.0
    assert mat[2, 3] == 4
    assert mat[3, 2] == 7
    assert struct.unpack_from('f', mat, (3 * 4 + 2) * 4) == (7, )
    assert struct.unpack_from('f', mat, (2 * 4 + 3) * 4) == (4, )

    mat2 = np.array(mat, copy=False)
    assert mat2.shape == (5, 4)
    assert abs(mat2).sum() == 11
    assert mat2[2, 3] == 4 and mat2[3, 2] == 7
    mat2[2, 3] = 5
    assert mat2[2, 3] == 5

    cstats = ConstructorStats.get(m.Matrix)
    assert cstats.alive() == 1
    del mat
    pytest.gc_collect()
    assert cstats.alive() == 1
    del mat2  # holds a mat reference
    pytest.gc_collect()
    assert cstats.alive() == 0
    assert cstats.values() == ["5x4 matrix"]
    assert cstats.copy_constructions == 0
    # assert cstats.move_constructions >= 0  # Don't invoke any
    assert cstats.copy_assignments == 0
    assert cstats.move_assignments == 0


@pytest.unsupported_on_pypy
def test_inherited_protocol():
    """SquareMatrix is derived from Matrix and inherits the buffer protocol"""

    matrix = m.SquareMatrix(5)
    assert memoryview(matrix).shape == (5, 5)
    assert np.asarray(matrix).shape == (5, 5)


@pytest.unsupported_on_pypy
def test_pointer_to_member_fn():
    for cls in [m.Buffer, m.ConstBuffer, m.DerivedBuffer]:
        buf = cls()
        buf.value = 0x12345678
        value = struct.unpack('i', bytearray(buf))[0]
        assert value == 0x12345678
