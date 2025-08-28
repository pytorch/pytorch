# Owner(s): ["module: dynamo"]

import functools
import operator
import re
import sys
import warnings
from itertools import product
from unittest import expectedFailure as xfail, skipIf as skipif, SkipTest

import pytest
from pytest import raises as assert_raises

from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    skipIfTorchDynamo,
    TEST_WITH_TORCHDYNAMO,
    TestCase,
    xpassIfTorchDynamo_np,
)


if TEST_WITH_TORCHDYNAMO:
    import numpy as np
    from numpy.testing import (
        assert_,
        assert_array_equal,
        assert_equal,
        assert_warns,
        HAS_REFCOUNT,
    )
else:
    import torch._numpy as np
    from torch._numpy.testing import (
        assert_,
        assert_array_equal,
        assert_equal,
        assert_warns,
        HAS_REFCOUNT,
    )


skip = functools.partial(skipif, True)


@instantiate_parametrized_tests
class TestIndexing(TestCase):
    def test_index_no_floats(self):
        a = np.array([[[5]]])

        assert_raises(IndexError, lambda: a[0.0])
        assert_raises(IndexError, lambda: a[0, 0.0])
        assert_raises(IndexError, lambda: a[0.0, 0])
        assert_raises(IndexError, lambda: a[0.0, :])
        assert_raises(IndexError, lambda: a[:, 0.0])
        assert_raises(IndexError, lambda: a[:, 0.0, :])
        assert_raises(IndexError, lambda: a[0.0, :, :])
        assert_raises(IndexError, lambda: a[0, 0, 0.0])
        assert_raises(IndexError, lambda: a[0.0, 0, 0])
        assert_raises(IndexError, lambda: a[0, 0.0, 0])
        assert_raises(IndexError, lambda: a[-1.4])
        assert_raises(IndexError, lambda: a[0, -1.4])
        assert_raises(IndexError, lambda: a[-1.4, 0])
        assert_raises(IndexError, lambda: a[-1.4, :])
        assert_raises(IndexError, lambda: a[:, -1.4])
        assert_raises(IndexError, lambda: a[:, -1.4, :])
        assert_raises(IndexError, lambda: a[-1.4, :, :])
        assert_raises(IndexError, lambda: a[0, 0, -1.4])
        assert_raises(IndexError, lambda: a[-1.4, 0, 0])
        assert_raises(IndexError, lambda: a[0, -1.4, 0])
        # Note torch validates index arguments "depth-first", so will prioritise
        # raising TypeError over IndexError, e.g.
        #
        #     >>> a = np.array([[[5]]])
        #     >>> a[0.0:, 0.0]
        #     IndexError: only integers, slices (`:`), ellipsis (`...`),
        #     numpy.newaxis #     (`None`) and integer or boolean arrays are
        #     valid indices
        #     >>> t = torch.as_tensor([[[5]]])  # identical to a
        #     >>> t[0.0:, 0.0]
        #     TypeError: slice indices must be integers or None or have an
        #     __index__ method
        #
        assert_raises((IndexError, TypeError), lambda: a[0.0:, 0.0])
        assert_raises((IndexError, TypeError), lambda: a[0.0:, 0.0, :])

    def test_slicing_no_floats(self):
        a = np.array([[5]])

        # start as float.
        assert_raises(TypeError, lambda: a[0.0:])
        assert_raises(TypeError, lambda: a[0:, 0.0:2])
        assert_raises(TypeError, lambda: a[0.0::2, :0])
        assert_raises(TypeError, lambda: a[0.0:1:2, :])
        assert_raises(TypeError, lambda: a[:, 0.0:])
        # stop as float.
        assert_raises(TypeError, lambda: a[:0.0])
        assert_raises(TypeError, lambda: a[:0, 1:2.0])
        assert_raises(TypeError, lambda: a[:0.0:2, :0])
        assert_raises(TypeError, lambda: a[:0.0, :])
        assert_raises(TypeError, lambda: a[:, 0:4.0:2])
        # step as float.
        assert_raises(TypeError, lambda: a[::1.0])
        assert_raises(TypeError, lambda: a[0:, :2:2.0])
        assert_raises(TypeError, lambda: a[1::4.0, :0])
        assert_raises(TypeError, lambda: a[::5.0, :])
        assert_raises(TypeError, lambda: a[:, 0:4:2.0])
        # mixed.
        assert_raises(TypeError, lambda: a[1.0:2:2.0])
        assert_raises(TypeError, lambda: a[1.0::2.0])
        assert_raises(TypeError, lambda: a[0:, :2.0:2.0])
        assert_raises(TypeError, lambda: a[1.0:1:4.0, :0])
        assert_raises(TypeError, lambda: a[1.0:5.0:5.0, :])
        assert_raises(TypeError, lambda: a[:, 0.4:4.0:2.0])
        # should still get the DeprecationWarning if step = 0.
        assert_raises(TypeError, lambda: a[::0.0])

    @skip(reason="torch allows slicing with non-0d array components")
    def test_index_no_array_to_index(self):
        # No non-scalar arrays.
        a = np.array([[[1]]])

        assert_raises(TypeError, lambda: a[a:a:a])
        # Conversely, using scalars doesn't raise in NumPy, e.g.
        #
        #     >>> i = np.int64(1)
        #     >>> a[i:i:i]
        #     array([], shape=(0, 1, 1), dtype=int64)
        #

    def test_none_index(self):
        # `None` index adds newaxis
        a = np.array([1, 2, 3])
        assert_equal(a[None], a[np.newaxis])
        assert_equal(a[None].ndim, a.ndim + 1)

    @skip
    def test_empty_tuple_index(self):
        # Empty tuple index creates a view
        a = np.array([1, 2, 3])
        assert_equal(a[()], a)
        assert_(a[()].tensor._base is a.tensor)
        a = np.array(0)
        assert_(isinstance(a[()], np.int_))

    def test_same_kind_index_casting(self):
        # Indexes should be cast with same-kind and not safe, even if that
        # is somewhat unsafe. So test various different code paths.
        index = np.arange(5)
        u_index = index.astype(np.uint8)  # i.e. cast to default uint indexing dtype
        arr = np.arange(10)

        assert_array_equal(arr[index], arr[u_index])
        arr[u_index] = np.arange(5)
        assert_array_equal(arr, np.arange(10))

        arr = np.arange(10).reshape(5, 2)
        assert_array_equal(arr[index], arr[u_index])

        arr[u_index] = np.arange(5)[:, None]
        assert_array_equal(arr, np.arange(5)[:, None].repeat(2, axis=1))

        arr = np.arange(25).reshape(5, 5)
        assert_array_equal(arr[u_index, u_index], arr[index, index])

    def test_empty_fancy_index(self):
        # Empty list index creates an empty array
        # with the same dtype (but with weird shape)
        a = np.array([1, 2, 3])
        assert_equal(a[[]], [])
        assert_equal(a[[]].dtype, a.dtype)

        b = np.array([], dtype=np.intp)
        assert_equal(a[[]], [])
        assert_equal(a[[]].dtype, a.dtype)

        b = np.array([])
        assert_raises(IndexError, a.__getitem__, b)

    def test_ellipsis_index(self):
        a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        assert_(a[...] is not a)
        assert_equal(a[...], a)
        # `a[...]` was `a` in numpy <1.9.

        # Slicing with ellipsis can skip an
        # arbitrary number of dimensions
        assert_equal(a[0, ...], a[0])
        assert_equal(a[0, ...], a[0, :])
        assert_equal(a[..., 0], a[:, 0])

        # Slicing with ellipsis always results
        # in an array, not a scalar
        assert_equal(a[0, ..., 1], np.array(2))

        # Assignment with `(Ellipsis,)` on 0-d arrays
        b = np.array(1)
        b[(Ellipsis,)] = 2
        assert_equal(b, 2)

    @xpassIfTorchDynamo_np  # 'torch_.np.array() does not have base attribute.
    def test_ellipsis_index_2(self):
        a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        assert_(a[...] is not a)
        assert_equal(a[...], a)
        # `a[...]` was `a` in numpy <1.9.
        assert_(a[...].base is a)

    def test_single_int_index(self):
        # Single integer index selects one row
        a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

        assert_equal(a[0], [1, 2, 3])
        assert_equal(a[-1], [7, 8, 9])

        # Index out of bounds produces IndexError
        assert_raises(IndexError, a.__getitem__, 1 << 30)
        # Index overflow produces IndexError
        # Note torch raises RuntimeError here
        assert_raises((IndexError, ValueError), a.__getitem__, 1 << 64)

    def test_single_bool_index(self):
        # Single boolean index
        a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

        assert_equal(a[np.array(True)], a[None])
        assert_equal(a[np.array(False)], a[None][0:0])

    def test_boolean_shape_mismatch(self):
        arr = np.ones((5, 4, 3))

        index = np.array([True])
        assert_raises(IndexError, arr.__getitem__, index)

        index = np.array([False] * 6)
        assert_raises(IndexError, arr.__getitem__, index)

        index = np.zeros((4, 4), dtype=bool)
        assert_raises(IndexError, arr.__getitem__, index)

        assert_raises(IndexError, arr.__getitem__, (slice(None), index))

    def test_boolean_indexing_onedim(self):
        # Indexing a 2-dimensional array with
        # boolean array of length one
        a = np.array([[0.0, 0.0, 0.0]])
        b = np.array([True], dtype=bool)
        assert_equal(a[b], a)
        # boolean assignment
        a[b] = 1.0
        assert_equal(a, [[1.0, 1.0, 1.0]])

    @skip(reason="NP_VER: fails on CI")
    def test_boolean_assignment_value_mismatch(self):
        # A boolean assignment should fail when the shape of the values
        # cannot be broadcast to the subscription. (see also gh-3458)
        a = np.arange(4)

        def f(a, v):
            a[a > -1] = v

        assert_raises((RuntimeError, ValueError, TypeError), f, a, [])
        assert_raises((RuntimeError, ValueError, TypeError), f, a, [1, 2, 3])
        assert_raises((RuntimeError, ValueError, TypeError), f, a[:1], [1, 2, 3])

    def test_boolean_indexing_twodim(self):
        # Indexing a 2-dimensional array with
        # 2-dimensional boolean array
        a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        b = np.array([[True, False, True], [False, True, False], [True, False, True]])
        assert_equal(a[b], [1, 3, 5, 7, 9])
        assert_equal(a[b[1]], [[4, 5, 6]])
        assert_equal(a[b[0]], a[b[2]])

        # boolean assignment
        a[b] = 0
        assert_equal(a, [[0, 2, 0], [4, 0, 6], [0, 8, 0]])

    def test_boolean_indexing_list(self):
        # Regression test for #13715. It's a use-after-free bug which the
        # test won't directly catch, but it will show up in valgrind.
        a = np.array([1, 2, 3])
        b = [True, False, True]
        # Two variants of the test because the first takes a fast path
        assert_equal(a[b], [1, 3])
        assert_equal(a[None, b], [[1, 3]])

    def test_reverse_strides_and_subspace_bufferinit(self):
        # This tests that the strides are not reversed for simple and
        # subspace fancy indexing.
        a = np.ones(5)
        b = np.zeros(5, dtype=np.intp)[::-1]
        c = np.arange(5)[::-1]

        a[b] = c
        # If the strides are not reversed, the 0 in the arange comes last.
        assert_equal(a[0], 0)

        # This also tests that the subspace buffer is initialized:
        a = np.ones((5, 2))
        c = np.arange(10).reshape(5, 2)[::-1]
        a[b, :] = c
        assert_equal(a[0], [0, 1])

    def test_reversed_strides_result_allocation(self):
        # Test a bug when calculating the output strides for a result array
        # when the subspace size was 1 (and test other cases as well)
        a = np.arange(10)[:, None]
        i = np.arange(10)[::-1]
        assert_array_equal(a[i], a[i.copy("C")])

        a = np.arange(20).reshape(-1, 2)

    def test_uncontiguous_subspace_assignment(self):
        # During development there was a bug activating a skip logic
        # based on ndim instead of size.
        a = np.full((3, 4, 2), -1)
        b = np.full((3, 4, 2), -1)

        a[[0, 1]] = np.arange(2 * 4 * 2).reshape(2, 4, 2).T
        b[[0, 1]] = np.arange(2 * 4 * 2).reshape(2, 4, 2).T.copy()

        assert_equal(a, b)

    @skip(reason="torch does not limit dims to 32")
    def test_too_many_fancy_indices_special_case(self):
        # Just documents behaviour, this is a small limitation.
        a = np.ones((1,) * 32)  # 32 is NPY_MAXDIMS
        assert_raises(IndexError, a.__getitem__, (np.array([0]),) * 32)

    def test_scalar_array_bool(self):
        # NumPy bools can be used as boolean index (python ones as of yet not)
        a = np.array(1)
        assert_equal(a[np.bool_(True)], a[np.array(True)])
        assert_equal(a[np.bool_(False)], a[np.array(False)])

        # After deprecating bools as integers:
        # a = np.array([0,1,2])
        # assert_equal(a[True, :], a[None, :])
        # assert_equal(a[:, True], a[:, None])
        #
        # assert_(not np.may_share_memory(a, a[True, :]))

    def test_everything_returns_views(self):
        # Before `...` would return a itself.
        a = np.arange(5)

        assert_(a is not a[()])
        assert_(a is not a[...])
        assert_(a is not a[:])

    def test_broaderrors_indexing(self):
        a = np.zeros((5, 5))
        assert_raises(IndexError, a.__getitem__, ([0, 1], [0, 1, 2]))
        assert_raises(IndexError, a.__setitem__, ([0, 1], [0, 1, 2]), 0)

    def test_trivial_fancy_out_of_bounds(self):
        a = np.zeros(5)
        ind = np.ones(20, dtype=np.intp)
        ind[-1] = 10
        assert_raises(IndexError, a.__getitem__, ind)
        assert_raises((IndexError, RuntimeError), a.__setitem__, ind, 0)
        ind = np.ones(20, dtype=np.intp)
        ind[0] = 11
        assert_raises(IndexError, a.__getitem__, ind)
        assert_raises((IndexError, RuntimeError), a.__setitem__, ind, 0)

    def test_trivial_fancy_not_possible(self):
        # Test that the fast path for trivial assignment is not incorrectly
        # used when the index is not contiguous or 1D, see also gh-11467.
        a = np.arange(6)
        idx = np.arange(6, dtype=np.intp).reshape(2, 1, 3)[:, :, 0]
        assert_array_equal(a[idx], idx)

        # this case must not go into the fast path, note that idx is
        # a non-contiuguous none 1D array here.
        a[idx] = -1
        res = np.arange(6)
        res[0] = -1
        res[3] = -1
        assert_array_equal(a, res)

    def test_memory_order(self):
        # This is not necessary to preserve. Memory layouts for
        # more complex indices are not as simple.
        a = np.arange(10)
        b = np.arange(10).reshape(5, 2).T
        assert_(a[b].flags.f_contiguous)

        # Takes a different implementation branch:
        a = a.reshape(-1, 1)
        assert_(a[b, 0].flags.f_contiguous)

    @skipIfTorchDynamo()  # XXX: flaky, depends on implementation details
    def test_small_regressions(self):
        # Reference count of intp for index checks
        a = np.array([0])
        if HAS_REFCOUNT:
            refcount = sys.getrefcount(np.dtype(np.intp))
        # item setting always checks indices in separate function:
        a[np.array([0], dtype=np.intp)] = 1
        a[np.array([0], dtype=np.uint8)] = 1
        assert_raises(IndexError, a.__setitem__, np.array([1], dtype=np.intp), 1)
        assert_raises(IndexError, a.__setitem__, np.array([1], dtype=np.uint8), 1)

        if HAS_REFCOUNT:
            assert_equal(sys.getrefcount(np.dtype(np.intp)), refcount)

    def test_tuple_subclass(self):
        arr = np.ones((5, 5))

        # A tuple subclass should also be an nd-index
        class TupleSubclass(tuple):
            __slots__ = ()

        index = ([1], [1])
        index = TupleSubclass(index)
        assert_(arr[index].shape == (1,))
        # Unlike the non nd-index:
        assert_(arr[index,].shape != (1,))

    @xpassIfTorchDynamo_np  # (reason="XXX: low-prio behaviour to support")
    def test_broken_sequence_not_nd_index(self):
        # See https://github.com/numpy/numpy/issues/5063
        # If we have an object which claims to be a sequence, but fails
        # on item getting, this should not be converted to an nd-index (tuple)
        # If this object happens to be a valid index otherwise, it should work
        # This object here is very dubious and probably bad though:
        class SequenceLike:
            def __index__(self):
                return 0

            def __len__(self):
                return 1

            def __getitem__(self, item):
                raise IndexError("Not possible")

        arr = np.arange(10)
        assert_array_equal(arr[SequenceLike()], arr[SequenceLike(),])

        # also test that field indexing does not segfault
        # for a similar reason, by indexing a structured array
        arr = np.zeros((1,), dtype=[("f1", "i8"), ("f2", "i8")])
        assert_array_equal(arr[SequenceLike()], arr[SequenceLike(),])

    def test_indexing_array_weird_strides(self):
        # See also gh-6221
        # the shapes used here come from the issue and create the correct
        # size for the iterator buffering size.
        x = np.ones(10)
        x2 = np.ones((10, 2))
        ind = np.arange(10)[:, None, None, None]
        ind = np.broadcast_to(ind, (10, 55, 4, 4))

        # single advanced index case
        assert_array_equal(x[ind], x[ind.copy()])
        # higher dimensional advanced index
        zind = np.zeros(4, dtype=np.intp)
        assert_array_equal(x2[ind, zind], x2[ind.copy(), zind])

    def test_indexing_array_negative_strides(self):
        # From gh-8264,
        # core dumps if negative strides are used in iteration
        arro = np.zeros((4, 4))
        arr = arro[::-1, ::-1]

        slices = (slice(None), [0, 1, 2, 3])
        arr[slices] = 10
        assert_array_equal(arr, 10.0)

    @parametrize("index", [True, False, np.array([0])])
    @parametrize("num", [32, 40])
    @parametrize("original_ndim", [1, 32])
    def test_too_many_advanced_indices(self, index, num, original_ndim):
        # These are limitations based on the number of arguments we can process.
        # For `num=32` (and all boolean cases), the result is actually define;
        # but the use of NpyIter (NPY_MAXARGS) limits it for technical reasons.
        if not (isinstance(index, np.ndarray) and original_ndim < num):
            # unskipped cases fail because of assigning too many indices
            raise SkipTest("torch does not limit dims to 32")

        arr = np.ones((1,) * original_ndim)
        with pytest.raises(IndexError):
            arr[(index,) * num]
        with pytest.raises(IndexError):
            arr[(index,) * num] = 1.0

    def test_nontuple_ndindex(self):
        a = np.arange(25).reshape((5, 5))
        assert_equal(a[[0, 1]], np.array([a[0], a[1]]))
        assert_equal(a[[0, 1], [0, 1]], np.array([0, 6]))
        raise SkipTest(
            "torch happily consumes non-tuple sequences with multi-axis "
            "indices (i.e. slices) as an index, whereas NumPy invalidates "
            "them, assumedly to keep things simple. This invalidation "
            "behaviour is just too niche to bother emulating."
        )
        assert_raises(IndexError, a.__getitem__, [slice(None)])


@instantiate_parametrized_tests
class TestBroadcastedAssignments(TestCase):
    def assign(self, a, ind, val):
        a[ind] = val
        return a

    def test_prepending_ones(self):
        a = np.zeros((3, 2))

        a[...] = np.ones((1, 3, 2))
        # Fancy with subspace with and without transpose
        a[[0, 1, 2], :] = np.ones((1, 3, 2))
        a[:, [0, 1]] = np.ones((1, 3, 2))
        # Fancy without subspace (with broadcasting)
        a[[[0], [1], [2]], [0, 1]] = np.ones((1, 3, 2))

    def test_prepend_not_one(self):
        assign = self.assign
        s_ = np.s_
        a = np.zeros(5)

        # Too large and not only ones.
        try:
            assign(a, s_[...], np.ones((2, 1)))
        except Exception as e:
            self.assertTrue(isinstance(e, (ValueError, RuntimeError)))
        assert_raises(
            (ValueError, RuntimeError), assign, a, s_[[1, 2, 3],], np.ones((2, 1))
        )
        assert_raises(
            (ValueError, RuntimeError), assign, a, s_[[[1], [2]],], np.ones((2, 2, 1))
        )

    def test_simple_broadcasting_errors(self):
        assign = self.assign
        s_ = np.s_
        a = np.zeros((5, 1))

        try:
            assign(a, s_[...], np.zeros((5, 2)))
        except Exception as e:
            self.assertTrue(isinstance(e, (ValueError, RuntimeError)))
        try:
            assign(a, s_[...], np.zeros((5, 0)))
        except Exception as e:
            self.assertTrue(isinstance(e, (ValueError, RuntimeError)))
        assert_raises(
            (ValueError, RuntimeError), assign, a, s_[:, [0]], np.zeros((5, 2))
        )
        assert_raises(
            (ValueError, RuntimeError), assign, a, s_[:, [0]], np.zeros((5, 0))
        )
        assert_raises(
            (ValueError, RuntimeError), assign, a, s_[[0], :], np.zeros((2, 1))
        )

    @parametrize(
        "index", [(..., [1, 2], slice(None)), ([0, 1], ..., 0), (..., [1, 2], [1, 2])]
    )
    def test_broadcast_error_reports_correct_shape(self, index):
        values = np.zeros((100, 100))  # will never broadcast below

        arr = np.zeros((3, 4, 5, 6, 7))

        with pytest.raises((ValueError, RuntimeError)) as e:
            arr[index] = values

        shape = arr[index].shape
        r_inner_shape = "".join(f"{side}, ?" for side in shape[:-1]) + str(shape[-1])
        assert re.search(rf"[\(\[]{r_inner_shape}[\]\)]$", str(e.value))

    def test_index_is_larger(self):
        # Simple case of fancy index broadcasting of the index.
        a = np.zeros((5, 5))
        a[[[0], [1], [2]], [0, 1, 2]] = [2, 3, 4]

        assert_((a[:3, :3] == [2, 3, 4]).all())

    def test_broadcast_subspace(self):
        a = np.zeros((100, 100))
        v = np.arange(100)[:, None]
        b = np.arange(100)[::-1]
        a[b] = v
        assert_((a[::-1] == v).all())


class TestFancyIndexingCast(TestCase):
    @xpassIfTorchDynamo_np  # (
    #    reason="XXX: low-prio to support assigning complex values on floating arrays"
    # )
    def test_boolean_index_cast_assign(self):
        # Setup the boolean index and float arrays.
        shape = (8, 63)
        bool_index = np.zeros(shape).astype(bool)
        bool_index[0, 1] = True
        zero_array = np.zeros(shape)

        # Assigning float is fine.
        zero_array[bool_index] = np.array([1])
        assert_equal(zero_array[0, 1], 1)

        # np.ComplexWarning moved to np.exceptions in numpy>=2.0.0
        # np.exceptions only available in numpy>=1.25.0
        has_exceptions_ns = hasattr(np, "exceptions")
        ComplexWarning = (
            np.exceptions.ComplexWarning if has_exceptions_ns else np.ComplexWarning
        )

        # Fancy indexing works, although we get a cast warning.
        assert_warns(
            ComplexWarning, zero_array.__setitem__, ([0], [1]), np.array([2 + 1j])
        )
        assert_equal(zero_array[0, 1], 2)  # No complex part

        # Cast complex to float, throwing away the imaginary portion.
        assert_warns(ComplexWarning, zero_array.__setitem__, bool_index, np.array([1j]))
        assert_equal(zero_array[0, 1], 0)


@xfail  # (reason="XXX: requires broadcast() and broadcast_to()")
class TestMultiIndexingAutomated(TestCase):
    """
    These tests use code to mimic the C-Code indexing for selection.

    NOTE:

        * This still lacks tests for complex item setting.
        * If you change behavior of indexing, you might want to modify
          these tests to try more combinations.
        * Behavior was written to match numpy version 1.8. (though a
          first version matched 1.7.)
        * Only tuple indices are supported by the mimicking code.
          (and tested as of writing this)
        * Error types should match most of the time as long as there
          is only one error. For multiple errors, what gets raised
          will usually not be the same one. They are *not* tested.

    Update 2016-11-30: It is probably not worth maintaining this test
    indefinitely and it can be dropped if maintenance becomes a burden.

    """

    def setupUp(self):
        self.a = np.arange(np.prod([3, 1, 5, 6])).reshape(3, 1, 5, 6)
        self.b = np.empty((3, 0, 5, 6))
        self.complex_indices = [
            "skip",
            Ellipsis,
            0,
            # Boolean indices, up to 3-d for some special cases of eating up
            # dimensions, also need to test all False
            np.array([True, False, False]),
            np.array([[True, False], [False, True]]),
            np.array([[[False, False], [False, False]]]),
            # Some slices:
            slice(-5, 5, 2),
            slice(1, 1, 100),
            slice(4, -1, -2),
            slice(None, None, -3),
            # Some Fancy indexes:
            np.empty((0, 1, 1), dtype=np.intp),  # empty and can be broadcast
            np.array([0, 1, -2]),
            np.array([[2], [0], [1]]),
            np.array([[0, -1], [0, 1]], dtype=np.dtype("intp")),
            np.array([2, -1], dtype=np.int8),
            np.zeros([1] * 31, dtype=int),  # trigger too large array.
            np.array([0.0, 1.0]),
        ]  # invalid datatype
        # Some simpler indices that still cover a bit more
        self.simple_indices = [Ellipsis, None, -1, [1], np.array([True]), "skip"]
        # Very simple ones to fill the rest:
        self.fill_indices = [slice(None, None), 0]

    def _get_multi_index(self, arr, indices):
        """Mimic multi dimensional indexing.

        Parameters
        ----------
        arr : ndarray
            Array to be indexed.
        indices : tuple of index objects

        Returns
        -------
        out : ndarray
            An array equivalent to the indexing operation (but always a copy).
            `arr[indices]` should be identical.
        no_copy : bool
            Whether the indexing operation requires a copy. If this is `True`,
            `np.may_share_memory(arr, arr[indices])` should be `True` (with
            some exceptions for scalars and possibly 0-d arrays).

        Notes
        -----
        While the function may mostly match the errors of normal indexing this
        is generally not the case.
        """
        in_indices = list(indices)
        indices = []
        # if False, this is a fancy or boolean index
        no_copy = True
        # number of fancy/scalar indexes that are not consecutive
        num_fancy = 0
        # number of dimensions indexed by a "fancy" index
        fancy_dim = 0
        # NOTE: This is a funny twist (and probably OK to change).
        # The boolean array has illegal indexes, but this is
        # allowed if the broadcast fancy-indices are 0-sized.
        # This variable is to catch that case.
        error_unless_broadcast_to_empty = False

        # We need to handle Ellipsis and make arrays from indices, also
        # check if this is fancy indexing (set no_copy).
        ndim = 0
        ellipsis_pos = None  # define here mostly to replace all but first.
        for i, indx in enumerate(in_indices):
            if indx is None:
                continue
            if isinstance(indx, np.ndarray) and indx.dtype == bool:
                no_copy = False
                if indx.ndim == 0:
                    raise IndexError
                # boolean indices can have higher dimensions
                ndim += indx.ndim
                fancy_dim += indx.ndim
                continue
            if indx is Ellipsis:
                if ellipsis_pos is None:
                    ellipsis_pos = i
                    continue  # do not increment ndim counter
                raise IndexError
            if isinstance(indx, slice):
                ndim += 1
                continue
            if not isinstance(indx, np.ndarray):
                # This could be open for changes in numpy.
                # numpy should maybe raise an error if casting to intp
                # is not safe. It rejects np.array([1., 2.]) but not
                # [1., 2.] as index (same for ie. np.take).
                # (Note the importance of empty lists if changing this here)
                try:
                    indx = np.array(indx, dtype=np.intp)
                except ValueError:
                    raise IndexError from None
                in_indices[i] = indx
            elif indx.dtype.kind != "b" and indx.dtype.kind != "i":
                raise IndexError(
                    "arrays used as indices must be of integer (or boolean) type"
                )
            if indx.ndim != 0:
                no_copy = False
            ndim += 1
            fancy_dim += 1

        if arr.ndim - ndim < 0:
            # we can't take more dimensions then we have, not even for 0-d
            # arrays.  since a[()] makes sense, but not a[(),]. We will
            # raise an error later on, unless a broadcasting error occurs
            # first.
            raise IndexError

        if ndim == 0 and None not in in_indices:
            # Well we have no indexes or one Ellipsis. This is legal.
            return arr.copy(), no_copy

        if ellipsis_pos is not None:
            in_indices[ellipsis_pos : ellipsis_pos + 1] = [slice(None, None)] * (
                arr.ndim - ndim
            )

        for ax, indx in enumerate(in_indices):
            if isinstance(indx, slice):
                # convert to an index array
                indx = np.arange(*indx.indices(arr.shape[ax]))
                indices.append(["s", indx])
                continue
            elif indx is None:
                # this is like taking a slice with one element from a new axis:
                indices.append(["n", np.array([0], dtype=np.intp)])
                arr = arr.reshape(arr.shape[:ax] + (1,) + arr.shape[ax:])
                continue
            if isinstance(indx, np.ndarray) and indx.dtype == bool:
                if indx.shape != arr.shape[ax : ax + indx.ndim]:
                    raise IndexError

                try:
                    flat_indx = np.ravel_multi_index(
                        np.nonzero(indx), arr.shape[ax : ax + indx.ndim], mode="raise"
                    )
                except Exception:
                    error_unless_broadcast_to_empty = True
                    # fill with 0s instead, and raise error later
                    flat_indx = np.array([0] * indx.sum(), dtype=np.intp)
                # concatenate axis into a single one:
                if indx.ndim != 0:
                    arr = arr.reshape(
                        arr.shape[:ax]
                        + (np.prod(arr.shape[ax : ax + indx.ndim]),)
                        + arr.shape[ax + indx.ndim :]
                    )
                    indx = flat_indx
                else:
                    # This could be changed, a 0-d boolean index can
                    # make sense (even outside the 0-d indexed array case)
                    # Note that originally this is could be interpreted as
                    # integer in the full integer special case.
                    raise IndexError
            else:
                # If the index is a singleton, the bounds check is done
                # before the broadcasting. This used to be different in <1.9
                if indx.ndim == 0:
                    if indx >= arr.shape[ax] or indx < -arr.shape[ax]:
                        raise IndexError
            if indx.ndim == 0:
                # The index is a scalar. This used to be two fold, but if
                # fancy indexing was active, the check was done later,
                # possibly after broadcasting it away (1.7. or earlier).
                # Now it is always done.
                if indx >= arr.shape[ax] or indx < -arr.shape[ax]:
                    raise IndexError
            if len(indices) > 0 and indices[-1][0] == "f" and ax != ellipsis_pos:
                # NOTE: There could still have been a 0-sized Ellipsis
                # between them. Checked that with ellipsis_pos.
                indices[-1].append(indx)
            else:
                # We have a fancy index that is not after an existing one.
                # NOTE: A 0-d array triggers this as well, while one may
                # expect it to not trigger it, since a scalar would not be
                # considered fancy indexing.
                num_fancy += 1
                indices.append(["f", indx])

        if num_fancy > 1 and not no_copy:
            # We have to flush the fancy indexes left
            new_indices = indices[:]
            axes = list(range(arr.ndim))
            fancy_axes = []
            new_indices.insert(0, ["f"])
            ni = 0
            ai = 0
            for indx in indices:
                ni += 1
                if indx[0] == "f":
                    new_indices[0].extend(indx[1:])
                    del new_indices[ni]
                    ni -= 1
                    for ax in range(ai, ai + len(indx[1:])):
                        fancy_axes.append(ax)
                        axes.remove(ax)
                ai += len(indx) - 1  # axis we are at
            indices = new_indices
            # and now we need to transpose arr:
            arr = arr.transpose(*(fancy_axes + axes))

        # We only have one 'f' index now and arr is transposed accordingly.
        # Now handle newaxis by reshaping...
        ax = 0
        for indx in indices:
            if indx[0] == "f":
                if len(indx) == 1:
                    continue
                # First of all, reshape arr to combine fancy axes into one:
                orig_shape = arr.shape
                orig_slice = orig_shape[ax : ax + len(indx[1:])]
                arr = arr.reshape(
                    arr.shape[:ax]
                    + (np.prod(orig_slice).astype(int),)
                    + arr.shape[ax + len(indx[1:]) :]
                )

                # Check if broadcasting works
                res = np.broadcast(*indx[1:])
                # unfortunately the indices might be out of bounds. So check
                # that first, and use mode='wrap' then. However only if
                # there are any indices...
                if res.size != 0:
                    if error_unless_broadcast_to_empty:
                        raise IndexError
                    for _indx, _size in zip(indx[1:], orig_slice):
                        if _indx.size == 0:
                            continue
                        if np.any(_indx >= _size) or np.any(_indx < -_size):
                            raise IndexError
                if len(indx[1:]) == len(orig_slice):
                    if np.prod(orig_slice) == 0:
                        # Work around for a crash or IndexError with 'wrap'
                        # in some 0-sized cases.
                        try:
                            mi = np.ravel_multi_index(
                                indx[1:], orig_slice, mode="raise"
                            )
                        except Exception as exc:
                            # This happens with 0-sized orig_slice (sometimes?)
                            # here it is a ValueError, but indexing gives a:
                            raise IndexError("invalid index into 0-sized") from exc
                    else:
                        mi = np.ravel_multi_index(indx[1:], orig_slice, mode="wrap")
                else:
                    # Maybe never happens...
                    raise ValueError
                arr = arr.take(mi.ravel(), axis=ax)
                try:
                    arr = arr.reshape(arr.shape[:ax] + mi.shape + arr.shape[ax + 1 :])
                except ValueError:
                    # too many dimensions, probably
                    raise IndexError from None
                ax += mi.ndim
                continue

            # If we are here, we have a 1D array for take:
            arr = arr.take(indx[1], axis=ax)
            ax += 1

        return arr, no_copy

    def _check_multi_index(self, arr, index):
        """Check a multi index item getting and simple setting.

        Parameters
        ----------
        arr : ndarray
            Array to be indexed, must be a reshaped arange.
        index : tuple of indexing objects
            Index being tested.
        """
        # Test item getting
        try:
            mimic_get, no_copy = self._get_multi_index(arr, index)
        except Exception as e:
            if HAS_REFCOUNT:
                prev_refcount = sys.getrefcount(arr)
            assert_raises(type(e), arr.__getitem__, index)
            assert_raises(type(e), arr.__setitem__, index, 0)
            if HAS_REFCOUNT:
                assert_equal(prev_refcount, sys.getrefcount(arr))
            return

        self._compare_index_result(arr, index, mimic_get, no_copy)

    def _check_single_index(self, arr, index):
        """Check a single index item getting and simple setting.

        Parameters
        ----------
        arr : ndarray
            Array to be indexed, must be an arange.
        index : indexing object
            Index being tested. Must be a single index and not a tuple
            of indexing objects (see also `_check_multi_index`).
        """
        try:
            mimic_get, no_copy = self._get_multi_index(arr, (index,))
        except Exception as e:
            if HAS_REFCOUNT:
                prev_refcount = sys.getrefcount(arr)
            assert_raises(type(e), arr.__getitem__, index)
            assert_raises(type(e), arr.__setitem__, index, 0)
            if HAS_REFCOUNT:
                assert_equal(prev_refcount, sys.getrefcount(arr))
            return

        self._compare_index_result(arr, index, mimic_get, no_copy)

    def _compare_index_result(self, arr, index, mimic_get, no_copy):
        """Compare mimicked result to indexing result."""
        raise SkipTest("torch does not support subclassing")
        arr = arr.copy()
        indexed_arr = arr[index]
        assert_array_equal(indexed_arr, mimic_get)
        # Check if we got a view, unless its a 0-sized or 0-d array.
        # (then its not a view, and that does not matter)
        if indexed_arr.size != 0 and indexed_arr.ndim != 0:
            assert_(np.may_share_memory(indexed_arr, arr) == no_copy)
            # Check reference count of the original array
            if HAS_REFCOUNT:
                if no_copy:
                    # refcount increases by one:
                    assert_equal(sys.getrefcount(arr), 3)
                else:
                    assert_equal(sys.getrefcount(arr), 2)

        # Test non-broadcast setitem:
        b = arr.copy()
        b[index] = mimic_get + 1000
        if b.size == 0:
            return  # nothing to compare here...
        if no_copy and indexed_arr.ndim != 0:
            # change indexed_arr in-place to manipulate original:
            indexed_arr += 1000
            assert_array_equal(arr, b)
            return
        # Use the fact that the array is originally an arange:
        arr.flat[indexed_arr.ravel()] += 1000
        assert_array_equal(arr, b)

    def test_boolean(self):
        a = np.array(5)
        assert_equal(a[np.array(True)], 5)
        a[np.array(True)] = 1
        assert_equal(a, 1)
        # NOTE: This is different from normal broadcasting, as
        # arr[boolean_array] works like in a multi index. Which means
        # it is aligned to the left. This is probably correct for
        # consistency with arr[boolean_array,] also no broadcasting
        # is done at all
        self._check_multi_index(self.a, (np.zeros_like(self.a, dtype=bool),))
        self._check_multi_index(self.a, (np.zeros_like(self.a, dtype=bool)[..., 0],))
        self._check_multi_index(self.a, (np.zeros_like(self.a, dtype=bool)[None, ...],))

    def test_multidim(self):
        # Automatically test combinations with complex indexes on 2nd (or 1st)
        # spot and the simple ones in one other spot.
        with warnings.catch_warnings():
            # This is so that np.array(True) is not accepted in a full integer
            # index, when running the file separately.
            warnings.filterwarnings("error", "", DeprecationWarning)
            # np.VisibleDeprecationWarning moved to np.exceptions in numpy>=2.0.0
            # np.exceptions only available in numpy>=1.25.0
            has_exceptions_ns = hasattr(np, "exceptions")
            VisibleDeprecationWarning = (  # noqa: F841
                np.exceptions.VisibleDeprecationWarning
                if has_exceptions_ns
                else np.VisibleDeprecationWarning
            )
            # FIXME(rec): should this use VisibleDeprecationWarning instead?
            warnings.filterwarnings("error", "", np.VisibleDeprecationWarning)

            def isskip(idx):
                return isinstance(idx, str) and idx == "skip"

            for simple_pos in [0, 2, 3]:
                tocheck = [
                    self.fill_indices,
                    self.complex_indices,
                    self.fill_indices,
                    self.fill_indices,
                ]
                tocheck[simple_pos] = self.simple_indices
                for index in product(*tocheck):
                    index = tuple(i for i in index if not isskip(i))
                    self._check_multi_index(self.a, index)
                    self._check_multi_index(self.b, index)

        # Check very simple item getting:
        self._check_multi_index(self.a, (0, 0, 0, 0))
        self._check_multi_index(self.b, (0, 0, 0, 0))
        # Also check (simple cases of) too many indices:
        assert_raises(IndexError, self.a.__getitem__, (0, 0, 0, 0, 0))
        assert_raises(IndexError, self.a.__setitem__, (0, 0, 0, 0, 0), 0)
        assert_raises(IndexError, self.a.__getitem__, (0, 0, [1], 0, 0))
        assert_raises(IndexError, self.a.__setitem__, (0, 0, [1], 0, 0), 0)

    def test_1d(self):
        a = np.arange(10)
        for index in self.complex_indices:
            self._check_single_index(a, index)


class TestFloatNonIntegerArgument(TestCase):
    """
    These test that ``TypeError`` is raised when you try to use
    non-integers as arguments to for indexing and slicing e.g. ``a[0.0:5]``
    and ``a[0.5]``, or other functions like ``array.reshape(1., -1)``.

    """

    def test_valid_indexing(self):
        # These should raise no errors.
        a = np.array([[[5]]])

        a[np.array([0])]
        a[[0, 0]]
        a[:, [0, 0]]
        a[:, 0, :]
        a[:, :, :]

    def test_valid_slicing(self):
        # These should raise no errors.
        a = np.array([[[5]]])

        a[::]
        a[0:]
        a[:2]
        a[0:2]
        a[::2]
        a[1::2]
        a[:2:2]
        a[1:2:2]

    def test_non_integer_argument_errors(self):
        a = np.array([[5]])

        assert_raises(TypeError, np.reshape, a, (1.0, 1.0, -1))
        assert_raises(TypeError, np.reshape, a, (np.array(1.0), -1))
        assert_raises(TypeError, np.take, a, [0], 1.0)
        assert_raises((TypeError, RuntimeError), np.take, a, [0], np.float64(1.0))

    @skip(
        reason=("torch doesn't have scalar types with distinct element-wise behaviours")
    )
    def test_non_integer_sequence_multiplication(self):
        # NumPy scalar sequence multiply should not work with non-integers
        def mult(a, b):
            return a * b

        assert_raises(TypeError, mult, [1], np.float64(3))
        # following should be OK
        mult([1], np.int_(3))

    def test_reduce_axis_float_index(self):
        d = np.zeros((3, 3, 3))
        assert_raises(TypeError, np.min, d, 0.5)
        assert_raises(TypeError, np.min, d, (0.5, 1))
        assert_raises(TypeError, np.min, d, (1, 2.2))
        assert_raises(TypeError, np.min, d, (0.2, 1.2))


class TestBooleanIndexing(TestCase):
    # Using a boolean as integer argument/indexing is an error.
    def test_bool_as_int_argument_errors(self):
        a = np.array([[[1]]])

        assert_raises(TypeError, np.reshape, a, (True, -1))
        # Note that operator.index(np.array(True)) does not work, a boolean
        # array is thus also deprecated, but not with the same message:
        # assert_warns(DeprecationWarning, operator.index, np.True_)

        assert_raises(TypeError, np.take, args=(a, [0], False))

        raise SkipTest("torch consumes boolean tensors as ints, no bother raising here")
        assert_raises(TypeError, np.reshape, a, (np.bool_(True), -1))
        assert_raises(TypeError, operator.index, np.array(True))

    def test_boolean_indexing_weirdness(self):
        # Weird boolean indexing things
        a = np.ones((2, 3, 4))
        assert a[False, True, ...].shape == (0, 2, 3, 4)
        assert a[True, [0, 1], True, True, [1], [[2]]].shape == (1, 2)
        assert_raises(IndexError, lambda: a[False, [0, 1], ...])

    def test_boolean_indexing_fast_path(self):
        # These used to either give the wrong error, or incorrectly give no
        # error.
        a = np.ones((3, 3))

        # This used to incorrectly work (and give an array of shape (0,))
        idx1 = np.array([[False] * 9])
        with pytest.raises(IndexError):
            a[idx1]

        # This used to incorrectly give a ValueError: operands could not be broadcast together
        idx2 = np.array([[False] * 8 + [True]])
        with pytest.raises(IndexError):
            a[idx2]

        # This is the same as it used to be. The above two should work like this.
        idx3 = np.array([[False] * 10])
        with pytest.raises(IndexError):
            a[idx3]

        # This used to give ValueError: non-broadcastable operand
        a = np.ones((1, 1, 2))
        idx = np.array([[[True], [False]]])
        with pytest.raises(IndexError):
            a[idx]


class TestArrayToIndexDeprecation(TestCase):
    """Creating an index from array not 0-D is an error."""

    def test_array_to_index_error(self):
        # so no exception is expected. The raising is effectively tested above.
        a = np.array([[[1]]])

        assert_raises((TypeError, RuntimeError), np.take, a, [0], a)

        raise SkipTest(
            "Multi-dimensional tensors are indexable just as long as they only "
            "contain a single element, no bother raising here"
        )
        assert_raises(TypeError, operator.index, np.array([1]))

        raise SkipTest("torch consumes tensors as ints, no bother raising here")
        assert_raises(TypeError, np.reshape, a, (a, -1))


class TestNonIntegerArrayLike(TestCase):
    """Tests that array_likes only valid if can safely cast to integer.

    For instance, lists give IndexError when they cannot be safely cast to
    an integer.

    """

    @skip(
        reason=(
            "torch consumes floats by way of falling back on its deprecated "
            "__index__ behaviour, no bother raising here"
        )
    )
    def test_basic(self):
        a = np.arange(10)

        assert_raises(IndexError, a.__getitem__, [0.5, 1.5])
        assert_raises(IndexError, a.__getitem__, (["1", "2"],))

        # The following is valid
        a.__getitem__([])


class TestMultipleEllipsisError(TestCase):
    """An index can only have a single ellipsis."""

    @xfail  # (
    #    reason=(
    #        "torch currently consumes multiple ellipsis, no bother raising "
    #        "here. See https://github.com/pytorch/pytorch/issues/59787#issue-917252204"
    #    )
    # )
    def test_basic(self):
        a = np.arange(10)
        assert_raises(IndexError, lambda: a[..., ...])
        assert_raises(IndexError, a.__getitem__, ((Ellipsis,) * 2,))
        assert_raises(IndexError, a.__getitem__, ((Ellipsis,) * 3,))


if __name__ == "__main__":
    run_tests()
