import sys
import operator
import pytest
import ctypes
import gc

import numpy as np
from numpy.core._rational_tests import rational
from numpy.testing import (
    assert_, assert_equal, assert_array_equal, assert_raises, HAS_REFCOUNT)
from numpy.compat import pickle
from itertools import permutations

def assert_dtype_equal(a, b):
    assert_equal(a, b)
    assert_equal(hash(a), hash(b),
                 "two equivalent types do not hash to the same value !")

def assert_dtype_not_equal(a, b):
    assert_(a != b)
    assert_(hash(a) != hash(b),
            "two different types hash to the same value !")

class TestBuiltin:
    @pytest.mark.parametrize('t', [int, float, complex, np.int32, str, object,
                                   np.compat.unicode])
    def test_run(self, t):
        """Only test hash runs at all."""
        dt = np.dtype(t)
        hash(dt)

    @pytest.mark.parametrize('t', [int, float])
    def test_dtype(self, t):
        # Make sure equivalent byte order char hash the same (e.g. < and = on
        # little endian)
        dt = np.dtype(t)
        dt2 = dt.newbyteorder("<")
        dt3 = dt.newbyteorder(">")
        if dt == dt2:
            assert_(dt.byteorder != dt2.byteorder, "bogus test")
            assert_dtype_equal(dt, dt2)
        else:
            assert_(dt.byteorder != dt3.byteorder, "bogus test")
            assert_dtype_equal(dt, dt3)

    def test_equivalent_dtype_hashing(self):
        # Make sure equivalent dtypes with different type num hash equal
        uintp = np.dtype(np.uintp)
        if uintp.itemsize == 4:
            left = uintp
            right = np.dtype(np.uint32)
        else:
            left = uintp
            right = np.dtype(np.ulonglong)
        assert_(left == right)
        assert_(hash(left) == hash(right))

    def test_invalid_types(self):
        # Make sure invalid type strings raise an error

        assert_raises(TypeError, np.dtype, 'O3')
        assert_raises(TypeError, np.dtype, 'O5')
        assert_raises(TypeError, np.dtype, 'O7')
        assert_raises(TypeError, np.dtype, 'b3')
        assert_raises(TypeError, np.dtype, 'h4')
        assert_raises(TypeError, np.dtype, 'I5')
        assert_raises(TypeError, np.dtype, 'e3')
        assert_raises(TypeError, np.dtype, 'f5')

        if np.dtype('g').itemsize == 8 or np.dtype('g').itemsize == 16:
            assert_raises(TypeError, np.dtype, 'g12')
        elif np.dtype('g').itemsize == 12:
            assert_raises(TypeError, np.dtype, 'g16')

        if np.dtype('l').itemsize == 8:
            assert_raises(TypeError, np.dtype, 'l4')
            assert_raises(TypeError, np.dtype, 'L4')
        else:
            assert_raises(TypeError, np.dtype, 'l8')
            assert_raises(TypeError, np.dtype, 'L8')

        if np.dtype('q').itemsize == 8:
            assert_raises(TypeError, np.dtype, 'q4')
            assert_raises(TypeError, np.dtype, 'Q4')
        else:
            assert_raises(TypeError, np.dtype, 'q8')
            assert_raises(TypeError, np.dtype, 'Q8')

    @pytest.mark.parametrize(
        'value',
        ['m8', 'M8', 'datetime64', 'timedelta64',
         'i4, (2,3)f8, f4', 'a3, 3u8, (3,4)a10',
         '>f', '<f', '=f', '|f',
        ])
    def test_dtype_bytes_str_equivalence(self, value):
        bytes_value = value.encode('ascii')
        from_bytes = np.dtype(bytes_value)
        from_str = np.dtype(value)
        assert_dtype_equal(from_bytes, from_str)

    def test_dtype_from_bytes(self):
        # Empty bytes object
        assert_raises(TypeError, np.dtype, b'')
        # Byte order indicator, but no type
        assert_raises(TypeError, np.dtype, b'|')

        # Single character with ordinal < NPY_NTYPES returns
        # type by index into _builtin_descrs
        assert_dtype_equal(np.dtype(bytes([0])), np.dtype('bool'))
        assert_dtype_equal(np.dtype(bytes([17])), np.dtype(object))

        # Single character where value is a valid type code
        assert_dtype_equal(np.dtype(b'f'), np.dtype('float32'))

        # Bytes with non-ascii values raise errors
        assert_raises(TypeError, np.dtype, b'\xff')
        assert_raises(TypeError, np.dtype, b's\xff')

    def test_bad_param(self):
        # Can't give a size that's too small
        assert_raises(ValueError, np.dtype,
                        {'names':['f0', 'f1'],
                         'formats':['i4', 'i1'],
                         'offsets':[0, 4],
                         'itemsize':4})
        # If alignment is enabled, the alignment (4) must divide the itemsize
        assert_raises(ValueError, np.dtype,
                        {'names':['f0', 'f1'],
                         'formats':['i4', 'i1'],
                         'offsets':[0, 4],
                         'itemsize':9}, align=True)
        # If alignment is enabled, the individual fields must be aligned
        assert_raises(ValueError, np.dtype,
                        {'names':['f0', 'f1'],
                         'formats':['i1', 'f4'],
                         'offsets':[0, 2]}, align=True)

    def test_field_order_equality(self):
        x = np.dtype({'names': ['A', 'B'],
                      'formats': ['i4', 'f4'],
                      'offsets': [0, 4]})
        y = np.dtype({'names': ['B', 'A'],
                      'formats': ['f4', 'i4'],
                      'offsets': [4, 0]})
        assert_equal(x == y, False)

class TestRecord:
    def test_equivalent_record(self):
        """Test whether equivalent record dtypes hash the same."""
        a = np.dtype([('yo', int)])
        b = np.dtype([('yo', int)])
        assert_dtype_equal(a, b)

    def test_different_names(self):
        # In theory, they may hash the same (collision) ?
        a = np.dtype([('yo', int)])
        b = np.dtype([('ye', int)])
        assert_dtype_not_equal(a, b)

    def test_different_titles(self):
        # In theory, they may hash the same (collision) ?
        a = np.dtype({'names': ['r', 'b'],
                      'formats': ['u1', 'u1'],
                      'titles': ['Red pixel', 'Blue pixel']})
        b = np.dtype({'names': ['r', 'b'],
                      'formats': ['u1', 'u1'],
                      'titles': ['RRed pixel', 'Blue pixel']})
        assert_dtype_not_equal(a, b)

    @pytest.mark.skipif(not HAS_REFCOUNT, reason="Python lacks refcounts")
    def test_refcount_dictionary_setting(self):
        names = ["name1"]
        formats = ["f8"]
        titles = ["t1"]
        offsets = [0]
        d = dict(names=names, formats=formats, titles=titles, offsets=offsets)
        refcounts = {k: sys.getrefcount(i) for k, i in d.items()}
        np.dtype(d)
        refcounts_new = {k: sys.getrefcount(i) for k, i in d.items()}
        assert refcounts == refcounts_new

    def test_mutate(self):
        # Mutating a dtype should reset the cached hash value
        a = np.dtype([('yo', int)])
        b = np.dtype([('yo', int)])
        c = np.dtype([('ye', int)])
        assert_dtype_equal(a, b)
        assert_dtype_not_equal(a, c)
        a.names = ['ye']
        assert_dtype_equal(a, c)
        assert_dtype_not_equal(a, b)
        state = b.__reduce__()[2]
        a.__setstate__(state)
        assert_dtype_equal(a, b)
        assert_dtype_not_equal(a, c)

    def test_not_lists(self):
        """Test if an appropriate exception is raised when passing bad values to
        the dtype constructor.
        """
        assert_raises(TypeError, np.dtype,
                      dict(names={'A', 'B'}, formats=['f8', 'i4']))
        assert_raises(TypeError, np.dtype,
                      dict(names=['A', 'B'], formats={'f8', 'i4'}))

    def test_aligned_size(self):
        # Check that structured dtypes get padded to an aligned size
        dt = np.dtype('i4, i1', align=True)
        assert_equal(dt.itemsize, 8)
        dt = np.dtype([('f0', 'i4'), ('f1', 'i1')], align=True)
        assert_equal(dt.itemsize, 8)
        dt = np.dtype({'names':['f0', 'f1'],
                       'formats':['i4', 'u1'],
                       'offsets':[0, 4]}, align=True)
        assert_equal(dt.itemsize, 8)
        dt = np.dtype({'f0': ('i4', 0), 'f1':('u1', 4)}, align=True)
        assert_equal(dt.itemsize, 8)
        # Nesting should preserve that alignment
        dt1 = np.dtype([('f0', 'i4'),
                       ('f1', [('f1', 'i1'), ('f2', 'i4'), ('f3', 'i1')]),
                       ('f2', 'i1')], align=True)
        assert_equal(dt1.itemsize, 20)
        dt2 = np.dtype({'names':['f0', 'f1', 'f2'],
                       'formats':['i4',
                                  [('f1', 'i1'), ('f2', 'i4'), ('f3', 'i1')],
                                  'i1'],
                       'offsets':[0, 4, 16]}, align=True)
        assert_equal(dt2.itemsize, 20)
        dt3 = np.dtype({'f0': ('i4', 0),
                       'f1': ([('f1', 'i1'), ('f2', 'i4'), ('f3', 'i1')], 4),
                       'f2': ('i1', 16)}, align=True)
        assert_equal(dt3.itemsize, 20)
        assert_equal(dt1, dt2)
        assert_equal(dt2, dt3)
        # Nesting should preserve packing
        dt1 = np.dtype([('f0', 'i4'),
                       ('f1', [('f1', 'i1'), ('f2', 'i4'), ('f3', 'i1')]),
                       ('f2', 'i1')], align=False)
        assert_equal(dt1.itemsize, 11)
        dt2 = np.dtype({'names':['f0', 'f1', 'f2'],
                       'formats':['i4',
                                  [('f1', 'i1'), ('f2', 'i4'), ('f3', 'i1')],
                                  'i1'],
                       'offsets':[0, 4, 10]}, align=False)
        assert_equal(dt2.itemsize, 11)
        dt3 = np.dtype({'f0': ('i4', 0),
                       'f1': ([('f1', 'i1'), ('f2', 'i4'), ('f3', 'i1')], 4),
                       'f2': ('i1', 10)}, align=False)
        assert_equal(dt3.itemsize, 11)
        assert_equal(dt1, dt2)
        assert_equal(dt2, dt3)
        # Array of subtype should preserve alignment
        dt1 = np.dtype([('a', '|i1'),
                        ('b', [('f0', '<i2'),
                        ('f1', '<f4')], 2)], align=True)
        assert_equal(dt1.descr, [('a', '|i1'), ('', '|V3'),
                                 ('b', [('f0', '<i2'), ('', '|V2'),
                                 ('f1', '<f4')], (2,))])

    def test_union_struct(self):
        # Should be able to create union dtypes
        dt = np.dtype({'names':['f0', 'f1', 'f2'], 'formats':['<u4', '<u2', '<u2'],
                        'offsets':[0, 0, 2]}, align=True)
        assert_equal(dt.itemsize, 4)
        a = np.array([3], dtype='<u4').view(dt)
        a['f1'] = 10
        a['f2'] = 36
        assert_equal(a['f0'], 10 + 36*256*256)
        # Should be able to specify fields out of order
        dt = np.dtype({'names':['f0', 'f1', 'f2'], 'formats':['<u4', '<u2', '<u2'],
                        'offsets':[4, 0, 2]}, align=True)
        assert_equal(dt.itemsize, 8)
        # field name should not matter: assignment is by position
        dt2 = np.dtype({'names':['f2', 'f0', 'f1'],
                        'formats':['<u4', '<u2', '<u2'],
                        'offsets':[4, 0, 2]}, align=True)
        vals = [(0, 1, 2), (3, -1, 4)]
        vals2 = [(0, 1, 2), (3, -1, 4)]
        a = np.array(vals, dt)
        b = np.array(vals2, dt2)
        assert_equal(a.astype(dt2), b)
        assert_equal(b.astype(dt), a)
        assert_equal(a.view(dt2), b)
        assert_equal(b.view(dt), a)
        # Should not be able to overlap objects with other types
        assert_raises(TypeError, np.dtype,
                {'names':['f0', 'f1'],
                 'formats':['O', 'i1'],
                 'offsets':[0, 2]})
        assert_raises(TypeError, np.dtype,
                {'names':['f0', 'f1'],
                 'formats':['i4', 'O'],
                 'offsets':[0, 3]})
        assert_raises(TypeError, np.dtype,
                {'names':['f0', 'f1'],
                 'formats':[[('a', 'O')], 'i1'],
                 'offsets':[0, 2]})
        assert_raises(TypeError, np.dtype,
                {'names':['f0', 'f1'],
                 'formats':['i4', [('a', 'O')]],
                 'offsets':[0, 3]})
        # Out of order should still be ok, however
        dt = np.dtype({'names':['f0', 'f1'],
                       'formats':['i1', 'O'],
                       'offsets':[np.dtype('intp').itemsize, 0]})

    def test_comma_datetime(self):
        dt = np.dtype('M8[D],datetime64[Y],i8')
        assert_equal(dt, np.dtype([('f0', 'M8[D]'),
                                   ('f1', 'datetime64[Y]'),
                                   ('f2', 'i8')]))

    def test_from_dictproxy(self):
        # Tests for PR #5920
        dt = np.dtype({'names': ['a', 'b'], 'formats': ['i4', 'f4']})
        assert_dtype_equal(dt, np.dtype(dt.fields))
        dt2 = np.dtype((np.void, dt.fields))
        assert_equal(dt2.fields, dt.fields)

    def test_from_dict_with_zero_width_field(self):
        # Regression test for #6430 / #2196
        dt = np.dtype([('val1', np.float32, (0,)), ('val2', int)])
        dt2 = np.dtype({'names': ['val1', 'val2'],
                        'formats': [(np.float32, (0,)), int]})

        assert_dtype_equal(dt, dt2)
        assert_equal(dt.fields['val1'][0].itemsize, 0)
        assert_equal(dt.itemsize, dt.fields['val2'][0].itemsize)

    def test_bool_commastring(self):
        d = np.dtype('?,?,?')  # raises?
        assert_equal(len(d.names), 3)
        for n in d.names:
            assert_equal(d.fields[n][0], np.dtype('?'))

    def test_nonint_offsets(self):
        # gh-8059
        def make_dtype(off):
            return np.dtype({'names': ['A'], 'formats': ['i4'],
                             'offsets': [off]})

        assert_raises(TypeError, make_dtype, 'ASD')
        assert_raises(OverflowError, make_dtype, 2**70)
        assert_raises(TypeError, make_dtype, 2.3)
        assert_raises(ValueError, make_dtype, -10)

        # no errors here:
        dt = make_dtype(np.uint32(0))
        np.zeros(1, dtype=dt)[0].item()

    def test_fields_by_index(self):
        dt = np.dtype([('a', np.int8), ('b', np.float32, 3)])
        assert_dtype_equal(dt[0], np.dtype(np.int8))
        assert_dtype_equal(dt[1], np.dtype((np.float32, 3)))
        assert_dtype_equal(dt[-1], dt[1])
        assert_dtype_equal(dt[-2], dt[0])
        assert_raises(IndexError, lambda: dt[-3])

        assert_raises(TypeError, operator.getitem, dt, 3.0)

        assert_equal(dt[1], dt[np.int8(1)])

    @pytest.mark.parametrize('align_flag',[False, True])
    def test_multifield_index(self, align_flag):
        # indexing with a list produces subfields
        # the align flag should be preserved
        dt = np.dtype([
            (('title', 'col1'), '<U20'), ('A', '<f8'), ('B', '<f8')
        ], align=align_flag)

        dt_sub = dt[['B', 'col1']]
        assert_equal(
            dt_sub,
            np.dtype({
                'names': ['B', 'col1'],
                'formats': ['<f8', '<U20'],
                'offsets': [88, 0],
                'titles': [None, 'title'],
                'itemsize': 96
            })
        )
        assert_equal(dt_sub.isalignedstruct, align_flag)

        dt_sub = dt[['B']]
        assert_equal(
            dt_sub,
            np.dtype({
                'names': ['B'],
                'formats': ['<f8'],
                'offsets': [88],
                'itemsize': 96
            })
        )
        assert_equal(dt_sub.isalignedstruct, align_flag)

        dt_sub = dt[[]]
        assert_equal(
            dt_sub,
            np.dtype({
                'names': [],
                'formats': [],
                'offsets': [],
                'itemsize': 96
            })
        )
        assert_equal(dt_sub.isalignedstruct, align_flag)

        assert_raises(TypeError, operator.getitem, dt, ())
        assert_raises(TypeError, operator.getitem, dt, [1, 2, 3])
        assert_raises(TypeError, operator.getitem, dt, ['col1', 2])
        assert_raises(KeyError, operator.getitem, dt, ['fake'])
        assert_raises(KeyError, operator.getitem, dt, ['title'])
        assert_raises(ValueError, operator.getitem, dt, ['col1', 'col1'])

    def test_partial_dict(self):
        # 'names' is missing
        assert_raises(ValueError, np.dtype,
                {'formats': ['i4', 'i4'], 'f0': ('i4', 0), 'f1':('i4', 4)})

    def test_fieldless_views(self):
        a = np.zeros(2, dtype={'names':[], 'formats':[], 'offsets':[],
                               'itemsize':8})
        assert_raises(ValueError, a.view, np.dtype([]))

        d = np.dtype((np.dtype([]), 10))
        assert_equal(d.shape, (10,))
        assert_equal(d.itemsize, 0)
        assert_equal(d.base, np.dtype([]))

        arr = np.fromiter((() for i in range(10)), [])
        assert_equal(arr.dtype, np.dtype([]))
        assert_raises(ValueError, np.frombuffer, b'', dtype=[])
        assert_equal(np.frombuffer(b'', dtype=[], count=2),
                     np.empty(2, dtype=[]))

        assert_raises(ValueError, np.dtype, ([], 'f8'))
        assert_raises(ValueError, np.zeros(1, dtype='i4').view, [])

        assert_equal(np.zeros(2, dtype=[]) == np.zeros(2, dtype=[]),
                     np.ones(2, dtype=bool))

        assert_equal(np.zeros((1, 2), dtype=[]) == a,
                     np.ones((1, 2), dtype=bool))


class TestSubarray:
    def test_single_subarray(self):
        a = np.dtype((int, (2)))
        b = np.dtype((int, (2,)))
        assert_dtype_equal(a, b)

        assert_equal(type(a.subdtype[1]), tuple)
        assert_equal(type(b.subdtype[1]), tuple)

    def test_equivalent_record(self):
        """Test whether equivalent subarray dtypes hash the same."""
        a = np.dtype((int, (2, 3)))
        b = np.dtype((int, (2, 3)))
        assert_dtype_equal(a, b)

    def test_nonequivalent_record(self):
        """Test whether different subarray dtypes hash differently."""
        a = np.dtype((int, (2, 3)))
        b = np.dtype((int, (3, 2)))
        assert_dtype_not_equal(a, b)

        a = np.dtype((int, (2, 3)))
        b = np.dtype((int, (2, 2)))
        assert_dtype_not_equal(a, b)

        a = np.dtype((int, (1, 2, 3)))
        b = np.dtype((int, (1, 2)))
        assert_dtype_not_equal(a, b)

    def test_shape_equal(self):
        """Test some data types that are equal"""
        assert_dtype_equal(np.dtype('f8'), np.dtype(('f8', tuple())))
        # FutureWarning during deprecation period; after it is passed this
        # should instead check that "(1)f8" == "1f8" == ("f8", 1).
        with pytest.warns(FutureWarning):
            assert_dtype_equal(np.dtype('f8'), np.dtype(('f8', 1)))
        assert_dtype_equal(np.dtype((int, 2)), np.dtype((int, (2,))))
        assert_dtype_equal(np.dtype(('<f4', (3, 2))), np.dtype(('<f4', (3, 2))))
        d = ([('a', 'f4', (1, 2)), ('b', 'f8', (3, 1))], (3, 2))
        assert_dtype_equal(np.dtype(d), np.dtype(d))

    def test_shape_simple(self):
        """Test some simple cases that shouldn't be equal"""
        assert_dtype_not_equal(np.dtype('f8'), np.dtype(('f8', (1,))))
        assert_dtype_not_equal(np.dtype(('f8', (1,))), np.dtype(('f8', (1, 1))))
        assert_dtype_not_equal(np.dtype(('f4', (3, 2))), np.dtype(('f4', (2, 3))))

    def test_shape_monster(self):
        """Test some more complicated cases that shouldn't be equal"""
        assert_dtype_not_equal(
            np.dtype(([('a', 'f4', (2, 1)), ('b', 'f8', (1, 3))], (2, 2))),
            np.dtype(([('a', 'f4', (1, 2)), ('b', 'f8', (1, 3))], (2, 2))))
        assert_dtype_not_equal(
            np.dtype(([('a', 'f4', (2, 1)), ('b', 'f8', (1, 3))], (2, 2))),
            np.dtype(([('a', 'f4', (2, 1)), ('b', 'i8', (1, 3))], (2, 2))))
        assert_dtype_not_equal(
            np.dtype(([('a', 'f4', (2, 1)), ('b', 'f8', (1, 3))], (2, 2))),
            np.dtype(([('e', 'f8', (1, 3)), ('d', 'f4', (2, 1))], (2, 2))))
        assert_dtype_not_equal(
            np.dtype(([('a', [('a', 'i4', 6)], (2, 1)), ('b', 'f8', (1, 3))], (2, 2))),
            np.dtype(([('a', [('a', 'u4', 6)], (2, 1)), ('b', 'f8', (1, 3))], (2, 2))))

    def test_shape_sequence(self):
        # Any sequence of integers should work as shape, but the result
        # should be a tuple (immutable) of base type integers.
        a = np.array([1, 2, 3], dtype=np.int16)
        l = [1, 2, 3]
        # Array gets converted
        dt = np.dtype([('a', 'f4', a)])
        assert_(isinstance(dt['a'].shape, tuple))
        assert_(isinstance(dt['a'].shape[0], int))
        # List gets converted
        dt = np.dtype([('a', 'f4', l)])
        assert_(isinstance(dt['a'].shape, tuple))
        #

        class IntLike:
            def __index__(self):
                return 3

            def __int__(self):
                # (a PyNumber_Check fails without __int__)
                return 3

        dt = np.dtype([('a', 'f4', IntLike())])
        assert_(isinstance(dt['a'].shape, tuple))
        assert_(isinstance(dt['a'].shape[0], int))
        dt = np.dtype([('a', 'f4', (IntLike(),))])
        assert_(isinstance(dt['a'].shape, tuple))
        assert_(isinstance(dt['a'].shape[0], int))

    def test_shape_matches_ndim(self):
        dt = np.dtype([('a', 'f4', ())])
        assert_equal(dt['a'].shape, ())
        assert_equal(dt['a'].ndim, 0)

        dt = np.dtype([('a', 'f4')])
        assert_equal(dt['a'].shape, ())
        assert_equal(dt['a'].ndim, 0)

        dt = np.dtype([('a', 'f4', 4)])
        assert_equal(dt['a'].shape, (4,))
        assert_equal(dt['a'].ndim, 1)

        dt = np.dtype([('a', 'f4', (1, 2, 3))])
        assert_equal(dt['a'].shape, (1, 2, 3))
        assert_equal(dt['a'].ndim, 3)

    def test_shape_invalid(self):
        # Check that the shape is valid.
        max_int = np.iinfo(np.intc).max
        max_intp = np.iinfo(np.intp).max
        # Too large values (the datatype is part of this)
        assert_raises(ValueError, np.dtype, [('a', 'f4', max_int // 4 + 1)])
        assert_raises(ValueError, np.dtype, [('a', 'f4', max_int + 1)])
        assert_raises(ValueError, np.dtype, [('a', 'f4', (max_int, 2))])
        # Takes a different code path (fails earlier:
        assert_raises(ValueError, np.dtype, [('a', 'f4', max_intp + 1)])
        # Negative values
        assert_raises(ValueError, np.dtype, [('a', 'f4', -1)])
        assert_raises(ValueError, np.dtype, [('a', 'f4', (-1, -1))])

    def test_alignment(self):
        #Check that subarrays are aligned
        t1 = np.dtype('(1,)i4', align=True)
        t2 = np.dtype('2i4', align=True)
        assert_equal(t1.alignment, t2.alignment)


def iter_struct_object_dtypes():
    """
    Iterates over a few complex dtypes and object pattern which
    fill the array with a given object (defaults to a singleton).

    Yields
    ------
    dtype : dtype
    pattern : tuple
        Structured tuple for use with `np.array`.
    count : int
        Number of objects stored in the dtype.
    singleton : object
        A singleton object. The returned pattern is constructed so that
        all objects inside the datatype are set to the singleton.
    """
    obj = object()

    dt = np.dtype([('b', 'O', (2, 3))])
    p = ([[obj] * 3] * 2,)
    yield pytest.param(dt, p, 6, obj, id="<subarray>")

    dt = np.dtype([('a', 'i4'), ('b', 'O', (2, 3))])
    p = (0, [[obj] * 3] * 2)
    yield pytest.param(dt, p, 6, obj, id="<subarray in field>")

    dt = np.dtype([('a', 'i4'),
                   ('b', [('ba', 'O'), ('bb', 'i1')], (2, 3))])
    p = (0, [[(obj, 0)] * 3] * 2)
    yield pytest.param(dt, p, 6, obj, id="<structured subarray 1>")

    dt = np.dtype([('a', 'i4'),
                   ('b', [('ba', 'O'), ('bb', 'O')], (2, 3))])
    p = (0, [[(obj, obj)] * 3] * 2)
    yield pytest.param(dt, p, 12, obj, id="<structured subarray 2>")


@pytest.mark.skipif(not HAS_REFCOUNT, reason="Python lacks refcounts")
class TestStructuredObjectRefcounting:
    """These tests cover various uses of complicated structured types which
    include objects and thus require reference counting.
    """
    @pytest.mark.parametrize(['dt', 'pat', 'count', 'singleton'],
                             iter_struct_object_dtypes())
    @pytest.mark.parametrize(["creation_func", "creation_obj"], [
        pytest.param(np.empty, None,
             # None is probably used for too many things
             marks=pytest.mark.skip("unreliable due to python's behaviour")),
        (np.ones, 1),
        (np.zeros, 0)])
    def test_structured_object_create_delete(self, dt, pat, count, singleton,
                                             creation_func, creation_obj):
        """Structured object reference counting in creation and deletion"""
        # The test assumes that 0, 1, and None are singletons.
        gc.collect()
        before = sys.getrefcount(creation_obj)
        arr = creation_func(3, dt)

        now = sys.getrefcount(creation_obj)
        assert now - before == count * 3
        del arr
        now = sys.getrefcount(creation_obj)
        assert now == before

    @pytest.mark.parametrize(['dt', 'pat', 'count', 'singleton'],
                             iter_struct_object_dtypes())
    def test_structured_object_item_setting(self, dt, pat, count, singleton):
        """Structured object reference counting for simple item setting"""
        one = 1

        gc.collect()
        before = sys.getrefcount(singleton)
        arr = np.array([pat] * 3, dt)
        assert sys.getrefcount(singleton) - before == count * 3
        # Fill with `1` and check that it was replaced correctly:
        before2 = sys.getrefcount(one)
        arr[...] = one
        after2 = sys.getrefcount(one)
        assert after2 - before2 == count * 3
        del arr
        gc.collect()
        assert sys.getrefcount(one) == before2
        assert sys.getrefcount(singleton) == before

    @pytest.mark.parametrize(['dt', 'pat', 'count', 'singleton'],
                             iter_struct_object_dtypes())
    @pytest.mark.parametrize(
        ['shape', 'index', 'items_changed'],
        [((3,), ([0, 2],), 2),
         ((3, 2), ([0, 2], slice(None)), 4),
         ((3, 2), ([0, 2], [1]), 2),
         ((3,), ([True, False, True]), 2)])
    def test_structured_object_indexing(self, shape, index, items_changed,
                                        dt, pat, count, singleton):
        """Structured object reference counting for advanced indexing."""
        zero = 0
        one = 1

        arr = np.zeros(shape, dt)

        gc.collect()
        before_zero = sys.getrefcount(zero)
        before_one = sys.getrefcount(one)
        # Test item getting:
        part = arr[index]
        after_zero = sys.getrefcount(zero)
        assert after_zero - before_zero == count * items_changed
        del part
        # Test item setting:
        arr[index] = one
        gc.collect()
        after_zero = sys.getrefcount(zero)
        after_one = sys.getrefcount(one)
        assert before_zero - after_zero == count * items_changed
        assert after_one - before_one == count * items_changed

    @pytest.mark.parametrize(['dt', 'pat', 'count', 'singleton'],
                             iter_struct_object_dtypes())
    def test_structured_object_take_and_repeat(self, dt, pat, count, singleton):
        """Structured object reference counting for specialized functions.
        The older functions such as take and repeat use different code paths
        then item setting (when writing this).
        """
        indices = [0, 1]

        arr = np.array([pat] * 3, dt)
        gc.collect()
        before = sys.getrefcount(singleton)
        res = arr.take(indices)
        after = sys.getrefcount(singleton)
        assert after - before == count * 2
        new = res.repeat(10)
        gc.collect()
        after_repeat = sys.getrefcount(singleton)
        assert after_repeat - after == count * 2 * 10


class TestStructuredDtypeSparseFields:
    """Tests subarray fields which contain sparse dtypes so that
    not all memory is used by the dtype work. Such dtype's should
    leave the underlying memory unchanged.
    """
    dtype = np.dtype([('a', {'names':['aa', 'ab'], 'formats':['f', 'f'],
                             'offsets':[0, 4]}, (2, 3))])
    sparse_dtype = np.dtype([('a', {'names':['ab'], 'formats':['f'],
                                    'offsets':[4]}, (2, 3))])

    @pytest.mark.xfail(reason="inaccessible data is changed see gh-12686.")
    @pytest.mark.valgrind_error(reason="reads from uninitialized buffers.")
    def test_sparse_field_assignment(self):
        arr = np.zeros(3, self.dtype)
        sparse_arr = arr.view(self.sparse_dtype)

        sparse_arr[...] = np.finfo(np.float32).max
        # dtype is reduced when accessing the field, so shape is (3, 2, 3):
        assert_array_equal(arr["a"]["aa"], np.zeros((3, 2, 3)))

    def test_sparse_field_assignment_fancy(self):
        # Fancy assignment goes to the copyswap function for complex types:
        arr = np.zeros(3, self.dtype)
        sparse_arr = arr.view(self.sparse_dtype)

        sparse_arr[[0, 1, 2]] = np.finfo(np.float32).max
        # dtype is reduced when accessing the field, so shape is (3, 2, 3):
        assert_array_equal(arr["a"]["aa"], np.zeros((3, 2, 3)))


class TestMonsterType:
    """Test deeply nested subtypes."""

    def test1(self):
        simple1 = np.dtype({'names': ['r', 'b'], 'formats': ['u1', 'u1'],
            'titles': ['Red pixel', 'Blue pixel']})
        a = np.dtype([('yo', int), ('ye', simple1),
            ('yi', np.dtype((int, (3, 2))))])
        b = np.dtype([('yo', int), ('ye', simple1),
            ('yi', np.dtype((int, (3, 2))))])
        assert_dtype_equal(a, b)

        c = np.dtype([('yo', int), ('ye', simple1),
            ('yi', np.dtype((a, (3, 2))))])
        d = np.dtype([('yo', int), ('ye', simple1),
            ('yi', np.dtype((a, (3, 2))))])
        assert_dtype_equal(c, d)

class TestMetadata:
    def test_no_metadata(self):
        d = np.dtype(int)
        assert_(d.metadata is None)

    def test_metadata_takes_dict(self):
        d = np.dtype(int, metadata={'datum': 1})
        assert_(d.metadata == {'datum': 1})

    def test_metadata_rejects_nondict(self):
        assert_raises(TypeError, np.dtype, int, metadata='datum')
        assert_raises(TypeError, np.dtype, int, metadata=1)
        assert_raises(TypeError, np.dtype, int, metadata=None)

    def test_nested_metadata(self):
        d = np.dtype([('a', np.dtype(int, metadata={'datum': 1}))])
        assert_(d['a'].metadata == {'datum': 1})

    def test_base_metadata_copied(self):
        d = np.dtype((np.void, np.dtype('i4,i4', metadata={'datum': 1})))
        assert_(d.metadata == {'datum': 1})

class TestString:
    def test_complex_dtype_str(self):
        dt = np.dtype([('top', [('tiles', ('>f4', (64, 64)), (1,)),
                                ('rtile', '>f4', (64, 36))], (3,)),
                       ('bottom', [('bleft', ('>f4', (8, 64)), (1,)),
                                   ('bright', '>f4', (8, 36))])])
        assert_equal(str(dt),
                     "[('top', [('tiles', ('>f4', (64, 64)), (1,)), "
                     "('rtile', '>f4', (64, 36))], (3,)), "
                     "('bottom', [('bleft', ('>f4', (8, 64)), (1,)), "
                     "('bright', '>f4', (8, 36))])]")

        # If the sticky aligned flag is set to True, it makes the
        # str() function use a dict representation with an 'aligned' flag
        dt = np.dtype([('top', [('tiles', ('>f4', (64, 64)), (1,)),
                                ('rtile', '>f4', (64, 36))],
                                (3,)),
                       ('bottom', [('bleft', ('>f4', (8, 64)), (1,)),
                                   ('bright', '>f4', (8, 36))])],
                       align=True)
        assert_equal(str(dt),
                    "{'names':['top','bottom'], "
                     "'formats':[([('tiles', ('>f4', (64, 64)), (1,)), "
                                  "('rtile', '>f4', (64, 36))], (3,)),"
                                 "[('bleft', ('>f4', (8, 64)), (1,)), "
                                  "('bright', '>f4', (8, 36))]], "
                     "'offsets':[0,76800], "
                     "'itemsize':80000, "
                     "'aligned':True}")
        assert_equal(np.dtype(eval(str(dt))), dt)

        dt = np.dtype({'names': ['r', 'g', 'b'], 'formats': ['u1', 'u1', 'u1'],
                        'offsets': [0, 1, 2],
                        'titles': ['Red pixel', 'Green pixel', 'Blue pixel']})
        assert_equal(str(dt),
                    "[(('Red pixel', 'r'), 'u1'), "
                    "(('Green pixel', 'g'), 'u1'), "
                    "(('Blue pixel', 'b'), 'u1')]")

        dt = np.dtype({'names': ['rgba', 'r', 'g', 'b'],
                       'formats': ['<u4', 'u1', 'u1', 'u1'],
                       'offsets': [0, 0, 1, 2],
                       'titles': ['Color', 'Red pixel',
                                  'Green pixel', 'Blue pixel']})
        assert_equal(str(dt),
                    "{'names':['rgba','r','g','b'],"
                    " 'formats':['<u4','u1','u1','u1'],"
                    " 'offsets':[0,0,1,2],"
                    " 'titles':['Color','Red pixel',"
                              "'Green pixel','Blue pixel'],"
                    " 'itemsize':4}")

        dt = np.dtype({'names': ['r', 'b'], 'formats': ['u1', 'u1'],
                        'offsets': [0, 2],
                        'titles': ['Red pixel', 'Blue pixel']})
        assert_equal(str(dt),
                    "{'names':['r','b'],"
                    " 'formats':['u1','u1'],"
                    " 'offsets':[0,2],"
                    " 'titles':['Red pixel','Blue pixel'],"
                    " 'itemsize':3}")

        dt = np.dtype([('a', '<m8[D]'), ('b', '<M8[us]')])
        assert_equal(str(dt),
                    "[('a', '<m8[D]'), ('b', '<M8[us]')]")

    def test_repr_structured(self):
        dt = np.dtype([('top', [('tiles', ('>f4', (64, 64)), (1,)),
                                ('rtile', '>f4', (64, 36))], (3,)),
                       ('bottom', [('bleft', ('>f4', (8, 64)), (1,)),
                                   ('bright', '>f4', (8, 36))])])
        assert_equal(repr(dt),
                     "dtype([('top', [('tiles', ('>f4', (64, 64)), (1,)), "
                     "('rtile', '>f4', (64, 36))], (3,)), "
                     "('bottom', [('bleft', ('>f4', (8, 64)), (1,)), "
                     "('bright', '>f4', (8, 36))])])")

        dt = np.dtype({'names': ['r', 'g', 'b'], 'formats': ['u1', 'u1', 'u1'],
                        'offsets': [0, 1, 2],
                        'titles': ['Red pixel', 'Green pixel', 'Blue pixel']},
                        align=True)
        assert_equal(repr(dt),
                    "dtype([(('Red pixel', 'r'), 'u1'), "
                    "(('Green pixel', 'g'), 'u1'), "
                    "(('Blue pixel', 'b'), 'u1')], align=True)")

    def test_repr_structured_not_packed(self):
        dt = np.dtype({'names': ['rgba', 'r', 'g', 'b'],
                       'formats': ['<u4', 'u1', 'u1', 'u1'],
                       'offsets': [0, 0, 1, 2],
                       'titles': ['Color', 'Red pixel',
                                  'Green pixel', 'Blue pixel']}, align=True)
        assert_equal(repr(dt),
                    "dtype({'names':['rgba','r','g','b'],"
                    " 'formats':['<u4','u1','u1','u1'],"
                    " 'offsets':[0,0,1,2],"
                    " 'titles':['Color','Red pixel',"
                              "'Green pixel','Blue pixel'],"
                    " 'itemsize':4}, align=True)")

        dt = np.dtype({'names': ['r', 'b'], 'formats': ['u1', 'u1'],
                        'offsets': [0, 2],
                        'titles': ['Red pixel', 'Blue pixel'],
                        'itemsize': 4})
        assert_equal(repr(dt),
                    "dtype({'names':['r','b'], "
                    "'formats':['u1','u1'], "
                    "'offsets':[0,2], "
                    "'titles':['Red pixel','Blue pixel'], "
                    "'itemsize':4})")

    def test_repr_structured_datetime(self):
        dt = np.dtype([('a', '<M8[D]'), ('b', '<m8[us]')])
        assert_equal(repr(dt),
                    "dtype([('a', '<M8[D]'), ('b', '<m8[us]')])")

    def test_repr_str_subarray(self):
        dt = np.dtype(('<i2', (1,)))
        assert_equal(repr(dt), "dtype(('<i2', (1,)))")
        assert_equal(str(dt), "('<i2', (1,))")

    def test_base_dtype_with_object_type(self):
        # Issue gh-2798, should not error.
        np.array(['a'], dtype="O").astype(("O", [("name", "O")]))

    def test_empty_string_to_object(self):
        # Pull request #4722
        np.array(["", ""]).astype(object)

    def test_void_subclass_unsized(self):
        dt = np.dtype(np.record)
        assert_equal(repr(dt), "dtype('V')")
        assert_equal(str(dt), '|V0')
        assert_equal(dt.name, 'record')

    def test_void_subclass_sized(self):
        dt = np.dtype((np.record, 2))
        assert_equal(repr(dt), "dtype('V2')")
        assert_equal(str(dt), '|V2')
        assert_equal(dt.name, 'record16')

    def test_void_subclass_fields(self):
        dt = np.dtype((np.record, [('a', '<u2')]))
        assert_equal(repr(dt), "dtype((numpy.record, [('a', '<u2')]))")
        assert_equal(str(dt), "(numpy.record, [('a', '<u2')])")
        assert_equal(dt.name, 'record16')


class TestDtypeAttributeDeletion:

    def test_dtype_non_writable_attributes_deletion(self):
        dt = np.dtype(np.double)
        attr = ["subdtype", "descr", "str", "name", "base", "shape",
                "isbuiltin", "isnative", "isalignedstruct", "fields",
                "metadata", "hasobject"]

        for s in attr:
            assert_raises(AttributeError, delattr, dt, s)

    def test_dtype_writable_attributes_deletion(self):
        dt = np.dtype(np.double)
        attr = ["names"]
        for s in attr:
            assert_raises(AttributeError, delattr, dt, s)


class TestDtypeAttributes:
    def test_descr_has_trailing_void(self):
        # see gh-6359
        dtype = np.dtype({
            'names': ['A', 'B'],
            'formats': ['f4', 'f4'],
            'offsets': [0, 8],
            'itemsize': 16})
        new_dtype = np.dtype(dtype.descr)
        assert_equal(new_dtype.itemsize, 16)

    def test_name_dtype_subclass(self):
        # Ticket #4357
        class user_def_subcls(np.void):
            pass
        assert_equal(np.dtype(user_def_subcls).name, 'user_def_subcls')


class TestPickling:

    def check_pickling(self, dtype):
        for proto in range(pickle.HIGHEST_PROTOCOL + 1):
            pickled = pickle.loads(pickle.dumps(dtype, proto))
            assert_equal(pickled, dtype)
            assert_equal(pickled.descr, dtype.descr)
            if dtype.metadata is not None:
                assert_equal(pickled.metadata, dtype.metadata)
            # Check the reconstructed dtype is functional
            x = np.zeros(3, dtype=dtype)
            y = np.zeros(3, dtype=pickled)
            assert_equal(x, y)
            assert_equal(x[0], y[0])

    @pytest.mark.parametrize('t', [int, float, complex, np.int32, str, object,
                                   np.compat.unicode, bool])
    def test_builtin(self, t):
        self.check_pickling(np.dtype(t))

    def test_structured(self):
        dt = np.dtype(([('a', '>f4', (2, 1)), ('b', '<f8', (1, 3))], (2, 2)))
        self.check_pickling(dt)

    def test_structured_aligned(self):
        dt = np.dtype('i4, i1', align=True)
        self.check_pickling(dt)

    def test_structured_unaligned(self):
        dt = np.dtype('i4, i1', align=False)
        self.check_pickling(dt)

    def test_structured_padded(self):
        dt = np.dtype({
            'names': ['A', 'B'],
            'formats': ['f4', 'f4'],
            'offsets': [0, 8],
            'itemsize': 16})
        self.check_pickling(dt)

    def test_structured_titles(self):
        dt = np.dtype({'names': ['r', 'b'],
                       'formats': ['u1', 'u1'],
                       'titles': ['Red pixel', 'Blue pixel']})
        self.check_pickling(dt)

    @pytest.mark.parametrize('base', ['m8', 'M8'])
    @pytest.mark.parametrize('unit', ['', 'Y', 'M', 'W', 'D', 'h', 'm', 's',
                                      'ms', 'us', 'ns', 'ps', 'fs', 'as'])
    def test_datetime(self, base, unit):
        dt = np.dtype('%s[%s]' % (base, unit) if unit else base)
        self.check_pickling(dt)
        if unit:
            dt = np.dtype('%s[7%s]' % (base, unit))
            self.check_pickling(dt)

    def test_metadata(self):
        dt = np.dtype(int, metadata={'datum': 1})
        self.check_pickling(dt)


def test_rational_dtype():
    # test for bug gh-5719
    a = np.array([1111], dtype=rational).astype
    assert_raises(OverflowError, a, 'int8')

    # test that dtype detection finds user-defined types
    x = rational(1)
    assert_equal(np.array([x,x]).dtype, np.dtype(rational))


def test_dtypes_are_true():
    # test for gh-6294
    assert bool(np.dtype('f8'))
    assert bool(np.dtype('i8'))
    assert bool(np.dtype([('a', 'i8'), ('b', 'f4')]))


def test_invalid_dtype_string():
    # test for gh-10440
    assert_raises(TypeError, np.dtype, 'f8,i8,[f8,i8]')
    assert_raises(TypeError, np.dtype, u'Fl\xfcgel')


class TestFromDTypeAttribute:
    def test_simple(self):
        class dt:
            dtype = "f8"

        assert np.dtype(dt) == np.float64
        assert np.dtype(dt()) == np.float64

    def test_recursion(self):
        class dt:
            pass

        dt.dtype = dt
        with pytest.raises(RecursionError):
            np.dtype(dt)

        dt_instance = dt()
        dt_instance.dtype = dt
        with pytest.raises(RecursionError):
            np.dtype(dt_instance)

    def test_void_subtype(self):
        class dt(np.void):
            # This code path is fully untested before, so it is unclear
            # what this should be useful for. Note that if np.void is used
            # numpy will think we are deallocating a base type [1.17, 2019-02].
            dtype = np.dtype("f,f")
            pass

        np.dtype(dt)
        np.dtype(dt(1))

    def test_void_subtype_recursion(self):
        class dt(np.void):
            pass

        dt.dtype = dt

        with pytest.raises(RecursionError):
            np.dtype(dt)

        with pytest.raises(RecursionError):
            np.dtype(dt(1))

class TestFromCTypes:

    @staticmethod
    def check(ctype, dtype):
        dtype = np.dtype(dtype)
        assert_equal(np.dtype(ctype), dtype)
        assert_equal(np.dtype(ctype()), dtype)

    def test_array(self):
        c8 = ctypes.c_uint8
        self.check(     3 * c8,  (np.uint8, (3,)))
        self.check(     1 * c8,  (np.uint8, (1,)))
        self.check(     0 * c8,  (np.uint8, (0,)))
        self.check(1 * (3 * c8), ((np.uint8, (3,)), (1,)))
        self.check(3 * (1 * c8), ((np.uint8, (1,)), (3,)))

    def test_padded_structure(self):
        class PaddedStruct(ctypes.Structure):
            _fields_ = [
                ('a', ctypes.c_uint8),
                ('b', ctypes.c_uint16)
            ]
        expected = np.dtype([
            ('a', np.uint8),
            ('b', np.uint16)
        ], align=True)
        self.check(PaddedStruct, expected)

    def test_bit_fields(self):
        class BitfieldStruct(ctypes.Structure):
            _fields_ = [
                ('a', ctypes.c_uint8, 7),
                ('b', ctypes.c_uint8, 1)
            ]
        assert_raises(TypeError, np.dtype, BitfieldStruct)
        assert_raises(TypeError, np.dtype, BitfieldStruct())

    def test_pointer(self):
        p_uint8 = ctypes.POINTER(ctypes.c_uint8)
        assert_raises(TypeError, np.dtype, p_uint8)

    def test_void_pointer(self):
        self.check(ctypes.c_void_p, np.uintp)

    def test_union(self):
        class Union(ctypes.Union):
            _fields_ = [
                ('a', ctypes.c_uint8),
                ('b', ctypes.c_uint16),
            ]
        expected = np.dtype(dict(
            names=['a', 'b'],
            formats=[np.uint8, np.uint16],
            offsets=[0, 0],
            itemsize=2
        ))
        self.check(Union, expected)

    def test_union_with_struct_packed(self):
        class Struct(ctypes.Structure):
            _pack_ = 1
            _fields_ = [
                ('one', ctypes.c_uint8),
                ('two', ctypes.c_uint32)
            ]

        class Union(ctypes.Union):
            _fields_ = [
                ('a', ctypes.c_uint8),
                ('b', ctypes.c_uint16),
                ('c', ctypes.c_uint32),
                ('d', Struct),
            ]
        expected = np.dtype(dict(
            names=['a', 'b', 'c', 'd'],
            formats=['u1', np.uint16, np.uint32, [('one', 'u1'), ('two', np.uint32)]],
            offsets=[0, 0, 0, 0],
            itemsize=ctypes.sizeof(Union)
        ))
        self.check(Union, expected)

    def test_union_packed(self):
        class Struct(ctypes.Structure):
            _fields_ = [
                ('one', ctypes.c_uint8),
                ('two', ctypes.c_uint32)
            ]
            _pack_ = 1
        class Union(ctypes.Union):
            _pack_ = 1
            _fields_ = [
                ('a', ctypes.c_uint8),
                ('b', ctypes.c_uint16),
                ('c', ctypes.c_uint32),
                ('d', Struct),
            ]
        expected = np.dtype(dict(
            names=['a', 'b', 'c', 'd'],
            formats=['u1', np.uint16, np.uint32, [('one', 'u1'), ('two', np.uint32)]],
            offsets=[0, 0, 0, 0],
            itemsize=ctypes.sizeof(Union)
        ))
        self.check(Union, expected)

    def test_packed_structure(self):
        class PackedStructure(ctypes.Structure):
            _pack_ = 1
            _fields_ = [
                ('a', ctypes.c_uint8),
                ('b', ctypes.c_uint16)
            ]
        expected = np.dtype([
            ('a', np.uint8),
            ('b', np.uint16)
        ])
        self.check(PackedStructure, expected)

    def test_large_packed_structure(self):
        class PackedStructure(ctypes.Structure):
            _pack_ = 2
            _fields_ = [
                ('a', ctypes.c_uint8),
                ('b', ctypes.c_uint16),
                ('c', ctypes.c_uint8),
                ('d', ctypes.c_uint16),
                ('e', ctypes.c_uint32),
                ('f', ctypes.c_uint32),
                ('g', ctypes.c_uint8)
                ]
        expected = np.dtype(dict(
            formats=[np.uint8, np.uint16, np.uint8, np.uint16, np.uint32, np.uint32, np.uint8 ],
            offsets=[0, 2, 4, 6, 8, 12, 16],
            names=['a', 'b', 'c', 'd', 'e', 'f', 'g'],
            itemsize=18))
        self.check(PackedStructure, expected)

    def test_big_endian_structure_packed(self):
        class BigEndStruct(ctypes.BigEndianStructure):
            _fields_ = [
                ('one', ctypes.c_uint8),
                ('two', ctypes.c_uint32)
            ]
            _pack_ = 1
        expected = np.dtype([('one', 'u1'), ('two', '>u4')])
        self.check(BigEndStruct, expected)

    def test_little_endian_structure_packed(self):
        class LittleEndStruct(ctypes.LittleEndianStructure):
            _fields_ = [
                ('one', ctypes.c_uint8),
                ('two', ctypes.c_uint32)
            ]
            _pack_ = 1
        expected = np.dtype([('one', 'u1'), ('two', '<u4')])
        self.check(LittleEndStruct, expected)

    def test_little_endian_structure(self):
        class PaddedStruct(ctypes.LittleEndianStructure):
            _fields_ = [
                ('a', ctypes.c_uint8),
                ('b', ctypes.c_uint16)
            ]
        expected = np.dtype([
            ('a', '<B'),
            ('b', '<H')
        ], align=True)
        self.check(PaddedStruct, expected)

    def test_big_endian_structure(self):
        class PaddedStruct(ctypes.BigEndianStructure):
            _fields_ = [
                ('a', ctypes.c_uint8),
                ('b', ctypes.c_uint16)
            ]
        expected = np.dtype([
            ('a', '>B'),
            ('b', '>H')
        ], align=True)
        self.check(PaddedStruct, expected)

    def test_simple_endian_types(self):
        self.check(ctypes.c_uint16.__ctype_le__, np.dtype('<u2'))
        self.check(ctypes.c_uint16.__ctype_be__, np.dtype('>u2'))
        self.check(ctypes.c_uint8.__ctype_le__, np.dtype('u1'))
        self.check(ctypes.c_uint8.__ctype_be__, np.dtype('u1'))

    all_types = set(np.typecodes['All'])
    all_pairs = permutations(all_types, 2)

    @pytest.mark.parametrize("pair", all_pairs)
    def test_pairs(self, pair):
        """
        Check that np.dtype('x,y') matches [np.dtype('x'), np.dtype('y')]
        Example: np.dtype('d,I') -> dtype([('f0', '<f8'), ('f1', '<u4')])
        """
        # gh-5645: check that np.dtype('i,L') can be used
        pair_type = np.dtype('{},{}'.format(*pair))
        expected = np.dtype([('f0', pair[0]), ('f1', pair[1])])
        assert_equal(pair_type, expected)

