# doctest
r''' Test the .npy file format.

Set up:

    >>> import sys
    >>> from io import BytesIO
    >>> from numpy.lib import format
    >>>
    >>> scalars = [
    ...     np.uint8,
    ...     np.int8,
    ...     np.uint16,
    ...     np.int16,
    ...     np.uint32,
    ...     np.int32,
    ...     np.uint64,
    ...     np.int64,
    ...     np.float32,
    ...     np.float64,
    ...     np.complex64,
    ...     np.complex128,
    ...     object,
    ... ]
    >>>
    >>> basic_arrays = []
    >>>
    >>> for scalar in scalars:
    ...     for endian in '<>':
    ...         dtype = np.dtype(scalar).newbyteorder(endian)
    ...         basic = np.arange(15).astype(dtype)
    ...         basic_arrays.extend([
    ...             np.array([], dtype=dtype),
    ...             np.array(10, dtype=dtype),
    ...             basic,
    ...             basic.reshape((3,5)),
    ...             basic.reshape((3,5)).T,
    ...             basic.reshape((3,5))[::-1,::2],
    ...         ])
    ...
    >>>
    >>> Pdescr = [
    ...     ('x', 'i4', (2,)),
    ...     ('y', 'f8', (2, 2)),
    ...     ('z', 'u1')]
    >>>
    >>>
    >>> PbufferT = [
    ...     ([3,2], [[6.,4.],[6.,4.]], 8),
    ...     ([4,3], [[7.,5.],[7.,5.]], 9),
    ...     ]
    >>>
    >>>
    >>> Ndescr = [
    ...     ('x', 'i4', (2,)),
    ...     ('Info', [
    ...         ('value', 'c16'),
    ...         ('y2', 'f8'),
    ...         ('Info2', [
    ...             ('name', 'S2'),
    ...             ('value', 'c16', (2,)),
    ...             ('y3', 'f8', (2,)),
    ...             ('z3', 'u4', (2,))]),
    ...         ('name', 'S2'),
    ...         ('z2', 'b1')]),
    ...     ('color', 'S2'),
    ...     ('info', [
    ...         ('Name', 'U8'),
    ...         ('Value', 'c16')]),
    ...     ('y', 'f8', (2, 2)),
    ...     ('z', 'u1')]
    >>>
    >>>
    >>> NbufferT = [
    ...     ([3,2], (6j, 6., ('nn', [6j,4j], [6.,4.], [1,2]), 'NN', True), 'cc', ('NN', 6j), [[6.,4.],[6.,4.]], 8),
    ...     ([4,3], (7j, 7., ('oo', [7j,5j], [7.,5.], [2,1]), 'OO', False), 'dd', ('OO', 7j), [[7.,5.],[7.,5.]], 9),
    ...     ]
    >>>
    >>>
    >>> record_arrays = [
    ...     np.array(PbufferT, dtype=np.dtype(Pdescr).newbyteorder('<')),
    ...     np.array(NbufferT, dtype=np.dtype(Ndescr).newbyteorder('<')),
    ...     np.array(PbufferT, dtype=np.dtype(Pdescr).newbyteorder('>')),
    ...     np.array(NbufferT, dtype=np.dtype(Ndescr).newbyteorder('>')),
    ... ]

Test the magic string writing.

    >>> format.magic(1, 0)
    '\x93NUMPY\x01\x00'
    >>> format.magic(0, 0)
    '\x93NUMPY\x00\x00'
    >>> format.magic(255, 255)
    '\x93NUMPY\xff\xff'
    >>> format.magic(2, 5)
    '\x93NUMPY\x02\x05'

Test the magic string reading.

    >>> format.read_magic(BytesIO(format.magic(1, 0)))
    (1, 0)
    >>> format.read_magic(BytesIO(format.magic(0, 0)))
    (0, 0)
    >>> format.read_magic(BytesIO(format.magic(255, 255)))
    (255, 255)
    >>> format.read_magic(BytesIO(format.magic(2, 5)))
    (2, 5)

Test the header writing.

    >>> for arr in basic_arrays + record_arrays:
    ...     f = BytesIO()
    ...     format.write_array_header_1_0(f, arr)   # XXX: arr is not a dict, items gets called on it
    ...     print(repr(f.getvalue()))
    ...
    "F\x00{'descr': '|u1', 'fortran_order': False, 'shape': (0,)}              \n"
    "F\x00{'descr': '|u1', 'fortran_order': False, 'shape': ()}                \n"
    "F\x00{'descr': '|u1', 'fortran_order': False, 'shape': (15,)}             \n"
    "F\x00{'descr': '|u1', 'fortran_order': False, 'shape': (3, 5)}            \n"
    "F\x00{'descr': '|u1', 'fortran_order': True, 'shape': (5, 3)}             \n"
    "F\x00{'descr': '|u1', 'fortran_order': False, 'shape': (3, 3)}            \n"
    "F\x00{'descr': '|u1', 'fortran_order': False, 'shape': (0,)}              \n"
    "F\x00{'descr': '|u1', 'fortran_order': False, 'shape': ()}                \n"
    "F\x00{'descr': '|u1', 'fortran_order': False, 'shape': (15,)}             \n"
    "F\x00{'descr': '|u1', 'fortran_order': False, 'shape': (3, 5)}            \n"
    "F\x00{'descr': '|u1', 'fortran_order': True, 'shape': (5, 3)}             \n"
    "F\x00{'descr': '|u1', 'fortran_order': False, 'shape': (3, 3)}            \n"
    "F\x00{'descr': '|i1', 'fortran_order': False, 'shape': (0,)}              \n"
    "F\x00{'descr': '|i1', 'fortran_order': False, 'shape': ()}                \n"
    "F\x00{'descr': '|i1', 'fortran_order': False, 'shape': (15,)}             \n"
    "F\x00{'descr': '|i1', 'fortran_order': False, 'shape': (3, 5)}            \n"
    "F\x00{'descr': '|i1', 'fortran_order': True, 'shape': (5, 3)}             \n"
    "F\x00{'descr': '|i1', 'fortran_order': False, 'shape': (3, 3)}            \n"
    "F\x00{'descr': '|i1', 'fortran_order': False, 'shape': (0,)}              \n"
    "F\x00{'descr': '|i1', 'fortran_order': False, 'shape': ()}                \n"
    "F\x00{'descr': '|i1', 'fortran_order': False, 'shape': (15,)}             \n"
    "F\x00{'descr': '|i1', 'fortran_order': False, 'shape': (3, 5)}            \n"
    "F\x00{'descr': '|i1', 'fortran_order': True, 'shape': (5, 3)}             \n"
    "F\x00{'descr': '|i1', 'fortran_order': False, 'shape': (3, 3)}            \n"
    "F\x00{'descr': '<u2', 'fortran_order': False, 'shape': (0,)}              \n"
    "F\x00{'descr': '<u2', 'fortran_order': False, 'shape': ()}                \n"
    "F\x00{'descr': '<u2', 'fortran_order': False, 'shape': (15,)}             \n"
    "F\x00{'descr': '<u2', 'fortran_order': False, 'shape': (3, 5)}            \n"
    "F\x00{'descr': '<u2', 'fortran_order': True, 'shape': (5, 3)}             \n"
    "F\x00{'descr': '<u2', 'fortran_order': False, 'shape': (3, 3)}            \n"
    "F\x00{'descr': '>u2', 'fortran_order': False, 'shape': (0,)}              \n"
    "F\x00{'descr': '>u2', 'fortran_order': False, 'shape': ()}                \n"
    "F\x00{'descr': '>u2', 'fortran_order': False, 'shape': (15,)}             \n"
    "F\x00{'descr': '>u2', 'fortran_order': False, 'shape': (3, 5)}            \n"
    "F\x00{'descr': '>u2', 'fortran_order': True, 'shape': (5, 3)}             \n"
    "F\x00{'descr': '>u2', 'fortran_order': False, 'shape': (3, 3)}            \n"
    "F\x00{'descr': '<i2', 'fortran_order': False, 'shape': (0,)}              \n"
    "F\x00{'descr': '<i2', 'fortran_order': False, 'shape': ()}                \n"
    "F\x00{'descr': '<i2', 'fortran_order': False, 'shape': (15,)}             \n"
    "F\x00{'descr': '<i2', 'fortran_order': False, 'shape': (3, 5)}            \n"
    "F\x00{'descr': '<i2', 'fortran_order': True, 'shape': (5, 3)}             \n"
    "F\x00{'descr': '<i2', 'fortran_order': False, 'shape': (3, 3)}            \n"
    "F\x00{'descr': '>i2', 'fortran_order': False, 'shape': (0,)}              \n"
    "F\x00{'descr': '>i2', 'fortran_order': False, 'shape': ()}                \n"
    "F\x00{'descr': '>i2', 'fortran_order': False, 'shape': (15,)}             \n"
    "F\x00{'descr': '>i2', 'fortran_order': False, 'shape': (3, 5)}            \n"
    "F\x00{'descr': '>i2', 'fortran_order': True, 'shape': (5, 3)}             \n"
    "F\x00{'descr': '>i2', 'fortran_order': False, 'shape': (3, 3)}            \n"
    "F\x00{'descr': '<u4', 'fortran_order': False, 'shape': (0,)}              \n"
    "F\x00{'descr': '<u4', 'fortran_order': False, 'shape': ()}                \n"
    "F\x00{'descr': '<u4', 'fortran_order': False, 'shape': (15,)}             \n"
    "F\x00{'descr': '<u4', 'fortran_order': False, 'shape': (3, 5)}            \n"
    "F\x00{'descr': '<u4', 'fortran_order': True, 'shape': (5, 3)}             \n"
    "F\x00{'descr': '<u4', 'fortran_order': False, 'shape': (3, 3)}            \n"
    "F\x00{'descr': '>u4', 'fortran_order': False, 'shape': (0,)}              \n"
    "F\x00{'descr': '>u4', 'fortran_order': False, 'shape': ()}                \n"
    "F\x00{'descr': '>u4', 'fortran_order': False, 'shape': (15,)}             \n"
    "F\x00{'descr': '>u4', 'fortran_order': False, 'shape': (3, 5)}            \n"
    "F\x00{'descr': '>u4', 'fortran_order': True, 'shape': (5, 3)}             \n"
    "F\x00{'descr': '>u4', 'fortran_order': False, 'shape': (3, 3)}            \n"
    "F\x00{'descr': '<i4', 'fortran_order': False, 'shape': (0,)}              \n"
    "F\x00{'descr': '<i4', 'fortran_order': False, 'shape': ()}                \n"
    "F\x00{'descr': '<i4', 'fortran_order': False, 'shape': (15,)}             \n"
    "F\x00{'descr': '<i4', 'fortran_order': False, 'shape': (3, 5)}            \n"
    "F\x00{'descr': '<i4', 'fortran_order': True, 'shape': (5, 3)}             \n"
    "F\x00{'descr': '<i4', 'fortran_order': False, 'shape': (3, 3)}            \n"
    "F\x00{'descr': '>i4', 'fortran_order': False, 'shape': (0,)}              \n"
    "F\x00{'descr': '>i4', 'fortran_order': False, 'shape': ()}                \n"
    "F\x00{'descr': '>i4', 'fortran_order': False, 'shape': (15,)}             \n"
    "F\x00{'descr': '>i4', 'fortran_order': False, 'shape': (3, 5)}            \n"
    "F\x00{'descr': '>i4', 'fortran_order': True, 'shape': (5, 3)}             \n"
    "F\x00{'descr': '>i4', 'fortran_order': False, 'shape': (3, 3)}            \n"
    "F\x00{'descr': '<u8', 'fortran_order': False, 'shape': (0,)}              \n"
    "F\x00{'descr': '<u8', 'fortran_order': False, 'shape': ()}                \n"
    "F\x00{'descr': '<u8', 'fortran_order': False, 'shape': (15,)}             \n"
    "F\x00{'descr': '<u8', 'fortran_order': False, 'shape': (3, 5)}            \n"
    "F\x00{'descr': '<u8', 'fortran_order': True, 'shape': (5, 3)}             \n"
    "F\x00{'descr': '<u8', 'fortran_order': False, 'shape': (3, 3)}            \n"
    "F\x00{'descr': '>u8', 'fortran_order': False, 'shape': (0,)}              \n"
    "F\x00{'descr': '>u8', 'fortran_order': False, 'shape': ()}                \n"
    "F\x00{'descr': '>u8', 'fortran_order': False, 'shape': (15,)}             \n"
    "F\x00{'descr': '>u8', 'fortran_order': False, 'shape': (3, 5)}            \n"
    "F\x00{'descr': '>u8', 'fortran_order': True, 'shape': (5, 3)}             \n"
    "F\x00{'descr': '>u8', 'fortran_order': False, 'shape': (3, 3)}            \n"
    "F\x00{'descr': '<i8', 'fortran_order': False, 'shape': (0,)}              \n"
    "F\x00{'descr': '<i8', 'fortran_order': False, 'shape': ()}                \n"
    "F\x00{'descr': '<i8', 'fortran_order': False, 'shape': (15,)}             \n"
    "F\x00{'descr': '<i8', 'fortran_order': False, 'shape': (3, 5)}            \n"
    "F\x00{'descr': '<i8', 'fortran_order': True, 'shape': (5, 3)}             \n"
    "F\x00{'descr': '<i8', 'fortran_order': False, 'shape': (3, 3)}            \n"
    "F\x00{'descr': '>i8', 'fortran_order': False, 'shape': (0,)}              \n"
    "F\x00{'descr': '>i8', 'fortran_order': False, 'shape': ()}                \n"
    "F\x00{'descr': '>i8', 'fortran_order': False, 'shape': (15,)}             \n"
    "F\x00{'descr': '>i8', 'fortran_order': False, 'shape': (3, 5)}            \n"
    "F\x00{'descr': '>i8', 'fortran_order': True, 'shape': (5, 3)}             \n"
    "F\x00{'descr': '>i8', 'fortran_order': False, 'shape': (3, 3)}            \n"
    "F\x00{'descr': '<f4', 'fortran_order': False, 'shape': (0,)}              \n"
    "F\x00{'descr': '<f4', 'fortran_order': False, 'shape': ()}                \n"
    "F\x00{'descr': '<f4', 'fortran_order': False, 'shape': (15,)}             \n"
    "F\x00{'descr': '<f4', 'fortran_order': False, 'shape': (3, 5)}            \n"
    "F\x00{'descr': '<f4', 'fortran_order': True, 'shape': (5, 3)}             \n"
    "F\x00{'descr': '<f4', 'fortran_order': False, 'shape': (3, 3)}            \n"
    "F\x00{'descr': '>f4', 'fortran_order': False, 'shape': (0,)}              \n"
    "F\x00{'descr': '>f4', 'fortran_order': False, 'shape': ()}                \n"
    "F\x00{'descr': '>f4', 'fortran_order': False, 'shape': (15,)}             \n"
    "F\x00{'descr': '>f4', 'fortran_order': False, 'shape': (3, 5)}            \n"
    "F\x00{'descr': '>f4', 'fortran_order': True, 'shape': (5, 3)}             \n"
    "F\x00{'descr': '>f4', 'fortran_order': False, 'shape': (3, 3)}            \n"
    "F\x00{'descr': '<f8', 'fortran_order': False, 'shape': (0,)}              \n"
    "F\x00{'descr': '<f8', 'fortran_order': False, 'shape': ()}                \n"
    "F\x00{'descr': '<f8', 'fortran_order': False, 'shape': (15,)}             \n"
    "F\x00{'descr': '<f8', 'fortran_order': False, 'shape': (3, 5)}            \n"
    "F\x00{'descr': '<f8', 'fortran_order': True, 'shape': (5, 3)}             \n"
    "F\x00{'descr': '<f8', 'fortran_order': False, 'shape': (3, 3)}            \n"
    "F\x00{'descr': '>f8', 'fortran_order': False, 'shape': (0,)}              \n"
    "F\x00{'descr': '>f8', 'fortran_order': False, 'shape': ()}                \n"
    "F\x00{'descr': '>f8', 'fortran_order': False, 'shape': (15,)}             \n"
    "F\x00{'descr': '>f8', 'fortran_order': False, 'shape': (3, 5)}            \n"
    "F\x00{'descr': '>f8', 'fortran_order': True, 'shape': (5, 3)}             \n"
    "F\x00{'descr': '>f8', 'fortran_order': False, 'shape': (3, 3)}            \n"
    "F\x00{'descr': '<c8', 'fortran_order': False, 'shape': (0,)}              \n"
    "F\x00{'descr': '<c8', 'fortran_order': False, 'shape': ()}                \n"
    "F\x00{'descr': '<c8', 'fortran_order': False, 'shape': (15,)}             \n"
    "F\x00{'descr': '<c8', 'fortran_order': False, 'shape': (3, 5)}            \n"
    "F\x00{'descr': '<c8', 'fortran_order': True, 'shape': (5, 3)}             \n"
    "F\x00{'descr': '<c8', 'fortran_order': False, 'shape': (3, 3)}            \n"
    "F\x00{'descr': '>c8', 'fortran_order': False, 'shape': (0,)}              \n"
    "F\x00{'descr': '>c8', 'fortran_order': False, 'shape': ()}                \n"
    "F\x00{'descr': '>c8', 'fortran_order': False, 'shape': (15,)}             \n"
    "F\x00{'descr': '>c8', 'fortran_order': False, 'shape': (3, 5)}            \n"
    "F\x00{'descr': '>c8', 'fortran_order': True, 'shape': (5, 3)}             \n"
    "F\x00{'descr': '>c8', 'fortran_order': False, 'shape': (3, 3)}            \n"
    "F\x00{'descr': '<c16', 'fortran_order': False, 'shape': (0,)}             \n"
    "F\x00{'descr': '<c16', 'fortran_order': False, 'shape': ()}               \n"
    "F\x00{'descr': '<c16', 'fortran_order': False, 'shape': (15,)}            \n"
    "F\x00{'descr': '<c16', 'fortran_order': False, 'shape': (3, 5)}           \n"
    "F\x00{'descr': '<c16', 'fortran_order': True, 'shape': (5, 3)}            \n"
    "F\x00{'descr': '<c16', 'fortran_order': False, 'shape': (3, 3)}           \n"
    "F\x00{'descr': '>c16', 'fortran_order': False, 'shape': (0,)}             \n"
    "F\x00{'descr': '>c16', 'fortran_order': False, 'shape': ()}               \n"
    "F\x00{'descr': '>c16', 'fortran_order': False, 'shape': (15,)}            \n"
    "F\x00{'descr': '>c16', 'fortran_order': False, 'shape': (3, 5)}           \n"
    "F\x00{'descr': '>c16', 'fortran_order': True, 'shape': (5, 3)}            \n"
    "F\x00{'descr': '>c16', 'fortran_order': False, 'shape': (3, 3)}           \n"
    "F\x00{'descr': 'O', 'fortran_order': False, 'shape': (0,)}              \n"
    "F\x00{'descr': 'O', 'fortran_order': False, 'shape': ()}                \n"
    "F\x00{'descr': 'O', 'fortran_order': False, 'shape': (15,)}             \n"
    "F\x00{'descr': 'O', 'fortran_order': False, 'shape': (3, 5)}            \n"
    "F\x00{'descr': 'O', 'fortran_order': True, 'shape': (5, 3)}             \n"
    "F\x00{'descr': 'O', 'fortran_order': False, 'shape': (3, 3)}            \n"
    "F\x00{'descr': 'O', 'fortran_order': False, 'shape': (0,)}              \n"
    "F\x00{'descr': 'O', 'fortran_order': False, 'shape': ()}                \n"
    "F\x00{'descr': 'O', 'fortran_order': False, 'shape': (15,)}             \n"
    "F\x00{'descr': 'O', 'fortran_order': False, 'shape': (3, 5)}            \n"
    "F\x00{'descr': 'O', 'fortran_order': True, 'shape': (5, 3)}             \n"
    "F\x00{'descr': 'O', 'fortran_order': False, 'shape': (3, 3)}            \n"
    "v\x00{'descr': [('x', '<i4', (2,)), ('y', '<f8', (2, 2)), ('z', '|u1')],\n 'fortran_order': False,\n 'shape': (2,)}         \n"
    "\x16\x02{'descr': [('x', '<i4', (2,)),\n           ('Info',\n            [('value', '<c16'),\n             ('y2', '<f8'),\n             ('Info2',\n              [('name', '|S2'),\n               ('value', '<c16', (2,)),\n               ('y3', '<f8', (2,)),\n               ('z3', '<u4', (2,))]),\n             ('name', '|S2'),\n             ('z2', '|b1')]),\n           ('color', '|S2'),\n           ('info', [('Name', '<U8'), ('Value', '<c16')]),\n           ('y', '<f8', (2, 2)),\n           ('z', '|u1')],\n 'fortran_order': False,\n 'shape': (2,)}      \n"
    "v\x00{'descr': [('x', '>i4', (2,)), ('y', '>f8', (2, 2)), ('z', '|u1')],\n 'fortran_order': False,\n 'shape': (2,)}         \n"
    "\x16\x02{'descr': [('x', '>i4', (2,)),\n           ('Info',\n            [('value', '>c16'),\n             ('y2', '>f8'),\n             ('Info2',\n              [('name', '|S2'),\n               ('value', '>c16', (2,)),\n               ('y3', '>f8', (2,)),\n               ('z3', '>u4', (2,))]),\n             ('name', '|S2'),\n             ('z2', '|b1')]),\n           ('color', '|S2'),\n           ('info', [('Name', '>U8'), ('Value', '>c16')]),\n           ('y', '>f8', (2, 2)),\n           ('z', '|u1')],\n 'fortran_order': False,\n 'shape': (2,)}      \n"
'''
import sys
import os
import shutil
import tempfile
import warnings
import pytest
from io import BytesIO

import numpy as np
from numpy.testing import (
    assert_, assert_array_equal, assert_raises, assert_raises_regex,
    assert_warns
    )
from numpy.lib import format


tempdir = None

# Module-level setup.


def setup_module():
    global tempdir
    tempdir = tempfile.mkdtemp()


def teardown_module():
    global tempdir
    if tempdir is not None and os.path.isdir(tempdir):
        shutil.rmtree(tempdir)
        tempdir = None


# Generate some basic arrays to test with.
scalars = [
    np.uint8,
    np.int8,
    np.uint16,
    np.int16,
    np.uint32,
    np.int32,
    np.uint64,
    np.int64,
    np.float32,
    np.float64,
    np.complex64,
    np.complex128,
    object,
]
basic_arrays = []
for scalar in scalars:
    for endian in '<>':
        dtype = np.dtype(scalar).newbyteorder(endian)
        basic = np.arange(1500).astype(dtype)
        basic_arrays.extend([
            # Empty
            np.array([], dtype=dtype),
            # Rank-0
            np.array(10, dtype=dtype),
            # 1-D
            basic,
            # 2-D C-contiguous
            basic.reshape((30, 50)),
            # 2-D F-contiguous
            basic.reshape((30, 50)).T,
            # 2-D non-contiguous
            basic.reshape((30, 50))[::-1, ::2],
        ])

# More complicated record arrays.
# This is the structure of the table used for plain objects:
#
# +-+-+-+
# |x|y|z|
# +-+-+-+

# Structure of a plain array description:
Pdescr = [
    ('x', 'i4', (2,)),
    ('y', 'f8', (2, 2)),
    ('z', 'u1')]

# A plain list of tuples with values for testing:
PbufferT = [
    # x     y                  z
    ([3, 2], [[6., 4.], [6., 4.]], 8),
    ([4, 3], [[7., 5.], [7., 5.]], 9),
    ]


# This is the structure of the table used for nested objects (DON'T PANIC!):
#
# +-+---------------------------------+-----+----------+-+-+
# |x|Info                             |color|info      |y|z|
# | +-----+--+----------------+----+--+     +----+-----+ | |
# | |value|y2|Info2           |name|z2|     |Name|Value| | |
# | |     |  +----+-----+--+--+    |  |     |    |     | | |
# | |     |  |name|value|y3|z3|    |  |     |    |     | | |
# +-+-----+--+----+-----+--+--+----+--+-----+----+-----+-+-+
#

# The corresponding nested array description:
Ndescr = [
    ('x', 'i4', (2,)),
    ('Info', [
        ('value', 'c16'),
        ('y2', 'f8'),
        ('Info2', [
            ('name', 'S2'),
            ('value', 'c16', (2,)),
            ('y3', 'f8', (2,)),
            ('z3', 'u4', (2,))]),
        ('name', 'S2'),
        ('z2', 'b1')]),
    ('color', 'S2'),
    ('info', [
        ('Name', 'U8'),
        ('Value', 'c16')]),
    ('y', 'f8', (2, 2)),
    ('z', 'u1')]

NbufferT = [
    # x     Info                                                color info        y                  z
    #       value y2 Info2                            name z2         Name Value
    #                name   value    y3       z3
    ([3, 2], (6j, 6., ('nn', [6j, 4j], [6., 4.], [1, 2]), 'NN', True),
     'cc', ('NN', 6j), [[6., 4.], [6., 4.]], 8),
    ([4, 3], (7j, 7., ('oo', [7j, 5j], [7., 5.], [2, 1]), 'OO', False),
     'dd', ('OO', 7j), [[7., 5.], [7., 5.]], 9),
    ]

record_arrays = [
    np.array(PbufferT, dtype=np.dtype(Pdescr).newbyteorder('<')),
    np.array(NbufferT, dtype=np.dtype(Ndescr).newbyteorder('<')),
    np.array(PbufferT, dtype=np.dtype(Pdescr).newbyteorder('>')),
    np.array(NbufferT, dtype=np.dtype(Ndescr).newbyteorder('>')),
    np.zeros(1, dtype=[('c', ('<f8', (5,)), (2,))])
]


#BytesIO that reads a random number of bytes at a time
class BytesIOSRandomSize(BytesIO):
    def read(self, size=None):
        import random
        size = random.randint(1, size)
        return super(BytesIOSRandomSize, self).read(size)


def roundtrip(arr):
    f = BytesIO()
    format.write_array(f, arr)
    f2 = BytesIO(f.getvalue())
    arr2 = format.read_array(f2, allow_pickle=True)
    return arr2


def roundtrip_randsize(arr):
    f = BytesIO()
    format.write_array(f, arr)
    f2 = BytesIOSRandomSize(f.getvalue())
    arr2 = format.read_array(f2)
    return arr2


def roundtrip_truncated(arr):
    f = BytesIO()
    format.write_array(f, arr)
    #BytesIO is one byte short
    f2 = BytesIO(f.getvalue()[0:-1])
    arr2 = format.read_array(f2)
    return arr2


def assert_equal_(o1, o2):
    assert_(o1 == o2)


def test_roundtrip():
    for arr in basic_arrays + record_arrays:
        arr2 = roundtrip(arr)
        assert_array_equal(arr, arr2)


def test_roundtrip_randsize():
    for arr in basic_arrays + record_arrays:
        if arr.dtype != object:
            arr2 = roundtrip_randsize(arr)
            assert_array_equal(arr, arr2)


def test_roundtrip_truncated():
    for arr in basic_arrays:
        if arr.dtype != object:
            assert_raises(ValueError, roundtrip_truncated, arr)


def test_long_str():
    # check items larger than internal buffer size, gh-4027
    long_str_arr = np.ones(1, dtype=np.dtype((str, format.BUFFER_SIZE + 1)))
    long_str_arr2 = roundtrip(long_str_arr)
    assert_array_equal(long_str_arr, long_str_arr2)


@pytest.mark.slow
def test_memmap_roundtrip():
    # Fixme: used to crash on windows
    if not (sys.platform == 'win32' or sys.platform == 'cygwin'):
        for arr in basic_arrays + record_arrays:
            if arr.dtype.hasobject:
                # Skip these since they can't be mmap'ed.
                continue
            # Write it out normally and through mmap.
            nfn = os.path.join(tempdir, 'normal.npy')
            mfn = os.path.join(tempdir, 'memmap.npy')
            fp = open(nfn, 'wb')
            try:
                format.write_array(fp, arr)
            finally:
                fp.close()

            fortran_order = (
                arr.flags.f_contiguous and not arr.flags.c_contiguous)
            ma = format.open_memmap(mfn, mode='w+', dtype=arr.dtype,
                                    shape=arr.shape, fortran_order=fortran_order)
            ma[...] = arr
            del ma

            # Check that both of these files' contents are the same.
            fp = open(nfn, 'rb')
            normal_bytes = fp.read()
            fp.close()
            fp = open(mfn, 'rb')
            memmap_bytes = fp.read()
            fp.close()
            assert_equal_(normal_bytes, memmap_bytes)

            # Check that reading the file using memmap works.
            ma = format.open_memmap(nfn, mode='r')
            del ma


def test_compressed_roundtrip():
    arr = np.random.rand(200, 200)
    npz_file = os.path.join(tempdir, 'compressed.npz')
    np.savez_compressed(npz_file, arr=arr)
    arr1 = np.load(npz_file)['arr']
    assert_array_equal(arr, arr1)


# aligned
dt1 = np.dtype('i1, i4, i1', align=True)
# non-aligned, explicit offsets
dt2 = np.dtype({'names': ['a', 'b'], 'formats': ['i4', 'i4'],
                'offsets': [1, 6]})
# nested struct-in-struct
dt3 = np.dtype({'names': ['c', 'd'], 'formats': ['i4', dt2]})
# field with '' name
dt4 = np.dtype({'names': ['a', '', 'b'], 'formats': ['i4']*3})
# titles
dt5 = np.dtype({'names': ['a', 'b'], 'formats': ['i4', 'i4'],
                'offsets': [1, 6], 'titles': ['aa', 'bb']})
# empty
dt6 = np.dtype({'names': [], 'formats': [], 'itemsize': 8})

@pytest.mark.parametrize("dt", [dt1, dt2, dt3, dt4, dt5, dt6])
def test_load_padded_dtype(dt):
    arr = np.zeros(3, dt)
    for i in range(3):
        arr[i] = i + 5
    npz_file = os.path.join(tempdir, 'aligned.npz')
    np.savez(npz_file, arr=arr)
    arr1 = np.load(npz_file)['arr']
    assert_array_equal(arr, arr1)


def test_python2_python3_interoperability():
    fname = 'win64python2.npy'
    path = os.path.join(os.path.dirname(__file__), 'data', fname)
    data = np.load(path)
    assert_array_equal(data, np.ones(2))

def test_pickle_python2_python3():
    # Test that loading object arrays saved on Python 2 works both on
    # Python 2 and Python 3 and vice versa
    data_dir = os.path.join(os.path.dirname(__file__), 'data')

    expected = np.array([None, range, u'\u512a\u826f',
                         b'\xe4\xb8\x8d\xe8\x89\xaf'],
                        dtype=object)

    for fname in ['py2-objarr.npy', 'py2-objarr.npz',
                  'py3-objarr.npy', 'py3-objarr.npz']:
        path = os.path.join(data_dir, fname)

        for encoding in ['bytes', 'latin1']:
            data_f = np.load(path, allow_pickle=True, encoding=encoding)
            if fname.endswith('.npz'):
                data = data_f['x']
                data_f.close()
            else:
                data = data_f

            if encoding == 'latin1' and fname.startswith('py2'):
                assert_(isinstance(data[3], str))
                assert_array_equal(data[:-1], expected[:-1])
                # mojibake occurs
                assert_array_equal(data[-1].encode(encoding), expected[-1])
            else:
                assert_(isinstance(data[3], bytes))
                assert_array_equal(data, expected)

        if fname.startswith('py2'):
            if fname.endswith('.npz'):
                data = np.load(path, allow_pickle=True)
                assert_raises(UnicodeError, data.__getitem__, 'x')
                data.close()
                data = np.load(path, allow_pickle=True, fix_imports=False,
                               encoding='latin1')
                assert_raises(ImportError, data.__getitem__, 'x')
                data.close()
            else:
                assert_raises(UnicodeError, np.load, path,
                              allow_pickle=True)
                assert_raises(ImportError, np.load, path,
                              allow_pickle=True, fix_imports=False,
                              encoding='latin1')


def test_pickle_disallow():
    data_dir = os.path.join(os.path.dirname(__file__), 'data')

    path = os.path.join(data_dir, 'py2-objarr.npy')
    assert_raises(ValueError, np.load, path,
                  allow_pickle=False, encoding='latin1')

    path = os.path.join(data_dir, 'py2-objarr.npz')
    f = np.load(path, allow_pickle=False, encoding='latin1')
    assert_raises(ValueError, f.__getitem__, 'x')

    path = os.path.join(tempdir, 'pickle-disabled.npy')
    assert_raises(ValueError, np.save, path, np.array([None], dtype=object),
                  allow_pickle=False)

@pytest.mark.parametrize('dt', [
    np.dtype(np.dtype([('a', np.int8),
                       ('b', np.int16),
                       ('c', np.int32),
                      ], align=True),
             (3,)),
    np.dtype([('x', np.dtype({'names':['a','b'],
                              'formats':['i1','i1'],
                              'offsets':[0,4],
                              'itemsize':8,
                             },
                    (3,)),
               (4,),
             )]),
    np.dtype([('x',
                   ('<f8', (5,)),
                   (2,),
               )]),
    np.dtype([('x', np.dtype((
        np.dtype((
            np.dtype({'names':['a','b'],
                      'formats':['i1','i1'],
                      'offsets':[0,4],
                      'itemsize':8}),
            (3,)
            )),
        (4,)
        )))
        ]),
    np.dtype([
        ('a', np.dtype((
            np.dtype((
                np.dtype((
                    np.dtype([
                        ('a', int),
                        ('b', np.dtype({'names':['a','b'],
                                        'formats':['i1','i1'],
                                        'offsets':[0,4],
                                        'itemsize':8})),
                    ]),
                    (3,),
                )),
                (4,),
            )),
            (5,),
        )))
        ]),
    ])

def test_descr_to_dtype(dt):
    dt1 = format.descr_to_dtype(dt.descr)
    assert_equal_(dt1, dt)
    arr1 = np.zeros(3, dt)
    arr2 = roundtrip(arr1)
    assert_array_equal(arr1, arr2)

def test_version_2_0():
    f = BytesIO()
    # requires more than 2 byte for header
    dt = [(("%d" % i) * 100, float) for i in range(500)]
    d = np.ones(1000, dtype=dt)

    format.write_array(f, d, version=(2, 0))
    with warnings.catch_warnings(record=True) as w:
        warnings.filterwarnings('always', '', UserWarning)
        format.write_array(f, d)
        assert_(w[0].category is UserWarning)

    # check alignment of data portion
    f.seek(0)
    header = f.readline()
    assert_(len(header) % format.ARRAY_ALIGN == 0)

    f.seek(0)
    n = format.read_array(f)
    assert_array_equal(d, n)

    # 1.0 requested but data cannot be saved this way
    assert_raises(ValueError, format.write_array, f, d, (1, 0))


@pytest.mark.slow
def test_version_2_0_memmap():
    # requires more than 2 byte for header
    dt = [(("%d" % i) * 100, float) for i in range(500)]
    d = np.ones(1000, dtype=dt)
    tf = tempfile.mktemp('', 'mmap', dir=tempdir)

    # 1.0 requested but data cannot be saved this way
    assert_raises(ValueError, format.open_memmap, tf, mode='w+', dtype=d.dtype,
                            shape=d.shape, version=(1, 0))

    ma = format.open_memmap(tf, mode='w+', dtype=d.dtype,
                            shape=d.shape, version=(2, 0))
    ma[...] = d
    del ma

    with warnings.catch_warnings(record=True) as w:
        warnings.filterwarnings('always', '', UserWarning)
        ma = format.open_memmap(tf, mode='w+', dtype=d.dtype,
                                shape=d.shape, version=None)
        assert_(w[0].category is UserWarning)
        ma[...] = d
        del ma

    ma = format.open_memmap(tf, mode='r')
    assert_array_equal(ma, d)


def test_write_version():
    f = BytesIO()
    arr = np.arange(1)
    # These should pass.
    format.write_array(f, arr, version=(1, 0))
    format.write_array(f, arr)

    format.write_array(f, arr, version=None)
    format.write_array(f, arr)

    format.write_array(f, arr, version=(2, 0))
    format.write_array(f, arr)

    # These should all fail.
    bad_versions = [
        (1, 1),
        (0, 0),
        (0, 1),
        (2, 2),
        (255, 255),
    ]
    for version in bad_versions:
        with assert_raises_regex(ValueError,
                                 'we only support format version.*'):
            format.write_array(f, arr, version=version)


bad_version_magic = [
    b'\x93NUMPY\x01\x01',
    b'\x93NUMPY\x00\x00',
    b'\x93NUMPY\x00\x01',
    b'\x93NUMPY\x02\x00',
    b'\x93NUMPY\x02\x02',
    b'\x93NUMPY\xff\xff',
]
malformed_magic = [
    b'\x92NUMPY\x01\x00',
    b'\x00NUMPY\x01\x00',
    b'\x93numpy\x01\x00',
    b'\x93MATLB\x01\x00',
    b'\x93NUMPY\x01',
    b'\x93NUMPY',
    b'',
]

def test_read_magic():
    s1 = BytesIO()
    s2 = BytesIO()

    arr = np.ones((3, 6), dtype=float)

    format.write_array(s1, arr, version=(1, 0))
    format.write_array(s2, arr, version=(2, 0))

    s1.seek(0)
    s2.seek(0)

    version1 = format.read_magic(s1)
    version2 = format.read_magic(s2)

    assert_(version1 == (1, 0))
    assert_(version2 == (2, 0))

    assert_(s1.tell() == format.MAGIC_LEN)
    assert_(s2.tell() == format.MAGIC_LEN)

def test_read_magic_bad_magic():
    for magic in malformed_magic:
        f = BytesIO(magic)
        assert_raises(ValueError, format.read_array, f)


def test_read_version_1_0_bad_magic():
    for magic in bad_version_magic + malformed_magic:
        f = BytesIO(magic)
        assert_raises(ValueError, format.read_array, f)


def test_bad_magic_args():
    assert_raises(ValueError, format.magic, -1, 1)
    assert_raises(ValueError, format.magic, 256, 1)
    assert_raises(ValueError, format.magic, 1, -1)
    assert_raises(ValueError, format.magic, 1, 256)


def test_large_header():
    s = BytesIO()
    d = {'a': 1, 'b': 2}
    format.write_array_header_1_0(s, d)

    s = BytesIO()
    d = {'a': 1, 'b': 2, 'c': 'x'*256*256}
    assert_raises(ValueError, format.write_array_header_1_0, s, d)


def test_read_array_header_1_0():
    s = BytesIO()

    arr = np.ones((3, 6), dtype=float)
    format.write_array(s, arr, version=(1, 0))

    s.seek(format.MAGIC_LEN)
    shape, fortran, dtype = format.read_array_header_1_0(s)

    assert_(s.tell() % format.ARRAY_ALIGN == 0)
    assert_((shape, fortran, dtype) == ((3, 6), False, float))


def test_read_array_header_2_0():
    s = BytesIO()

    arr = np.ones((3, 6), dtype=float)
    format.write_array(s, arr, version=(2, 0))

    s.seek(format.MAGIC_LEN)
    shape, fortran, dtype = format.read_array_header_2_0(s)

    assert_(s.tell() % format.ARRAY_ALIGN == 0)
    assert_((shape, fortran, dtype) == ((3, 6), False, float))


def test_bad_header():
    # header of length less than 2 should fail
    s = BytesIO()
    assert_raises(ValueError, format.read_array_header_1_0, s)
    s = BytesIO(b'1')
    assert_raises(ValueError, format.read_array_header_1_0, s)

    # header shorter than indicated size should fail
    s = BytesIO(b'\x01\x00')
    assert_raises(ValueError, format.read_array_header_1_0, s)

    # headers without the exact keys required should fail
    d = {"shape": (1, 2),
         "descr": "x"}
    s = BytesIO()
    format.write_array_header_1_0(s, d)
    assert_raises(ValueError, format.read_array_header_1_0, s)

    d = {"shape": (1, 2),
         "fortran_order": False,
         "descr": "x",
         "extrakey": -1}
    s = BytesIO()
    format.write_array_header_1_0(s, d)
    assert_raises(ValueError, format.read_array_header_1_0, s)


def test_large_file_support():
    if (sys.platform == 'win32' or sys.platform == 'cygwin'):
        pytest.skip("Unknown if Windows has sparse filesystems")
    # try creating a large sparse file
    tf_name = os.path.join(tempdir, 'sparse_file')
    try:
        # seek past end would work too, but linux truncate somewhat
        # increases the chances that we have a sparse filesystem and can
        # avoid actually writing 5GB
        import subprocess as sp
        sp.check_call(["truncate", "-s", "5368709120", tf_name])
    except Exception:
        pytest.skip("Could not create 5GB large file")
    # write a small array to the end
    with open(tf_name, "wb") as f:
        f.seek(5368709120)
        d = np.arange(5)
        np.save(f, d)
    # read it back
    with open(tf_name, "rb") as f:
        f.seek(5368709120)
        r = np.load(f)
    assert_array_equal(r, d)


@pytest.mark.skipif(np.dtype(np.intp).itemsize < 8,
                    reason="test requires 64-bit system")
@pytest.mark.slow
def test_large_archive():
    # Regression test for product of saving arrays with dimensions of array
    # having a product that doesn't fit in int32.  See gh-7598 for details.
    try:
        a = np.empty((2**30, 2), dtype=np.uint8)
    except MemoryError:
        pytest.skip("Could not create large file")

    fname = os.path.join(tempdir, "large_archive")

    with open(fname, "wb") as f:
        np.savez(f, arr=a)

    with open(fname, "rb") as f:
        new_a = np.load(f)["arr"]

    assert_(a.shape == new_a.shape)


def test_empty_npz():
    # Test for gh-9989
    fname = os.path.join(tempdir, "nothing.npz")
    np.savez(fname)
    np.load(fname)


def test_unicode_field_names():
    # gh-7391
    arr = np.array([
        (1, 3),
        (1, 2),
        (1, 3),
        (1, 2)
    ], dtype=[
        ('int', int),
        (u'\N{CJK UNIFIED IDEOGRAPH-6574}\N{CJK UNIFIED IDEOGRAPH-5F62}', int)
    ])
    fname = os.path.join(tempdir, "unicode.npy")
    with open(fname, 'wb') as f:
        format.write_array(f, arr, version=(3, 0))
    with open(fname, 'rb') as f:
        arr2 = format.read_array(f)
    assert_array_equal(arr, arr2)

    # notifies the user that 3.0 is selected
    with open(fname, 'wb') as f:
        with assert_warns(UserWarning):
            format.write_array(f, arr, version=None)


@pytest.mark.parametrize('dt, fail', [
    (np.dtype({'names': ['a', 'b'], 'formats':  [float, np.dtype('S3',
                 metadata={'some': 'stuff'})]}), True),
    (np.dtype(int, metadata={'some': 'stuff'}), False),
    (np.dtype([('subarray', (int, (2,)))], metadata={'some': 'stuff'}), False),
    # recursive: metadata on the field of a dtype
    (np.dtype({'names': ['a', 'b'], 'formats': [
        float, np.dtype({'names': ['c'], 'formats': [np.dtype(int, metadata={})]})
    ]}), False)
    ])
def test_metadata_dtype(dt, fail):
    # gh-14142
    arr = np.ones(10, dtype=dt)
    buf = BytesIO()
    with assert_warns(UserWarning):
        np.save(buf, arr)
    buf.seek(0)
    if fail:
        with assert_raises(ValueError):
            np.load(buf)
    else:
        arr2 = np.load(buf)
        # BUG: assert_array_equal does not check metadata
        from numpy.lib.format import _has_metadata
        assert_array_equal(arr, arr2)
        assert _has_metadata(arr.dtype)
        assert not _has_metadata(arr2.dtype)

