from common_utils import TestCase, run_tests
import torch
import warnings
from torch import tensor
import unittest


class TestIndexing(TestCase):
    def test_single_int(self):
        v = torch.randn(5, 7, 3)
        self.assertEqual(v[4].shape, (7, 3))

    def test_multiple_int(self):
        v = torch.randn(5, 7, 3)
        self.assertEqual(v[4].shape, (7, 3))
        self.assertEqual(v[4, :, 1].shape, (7,))

    def test_none(self):
        v = torch.randn(5, 7, 3)
        self.assertEqual(v[None].shape, (1, 5, 7, 3))
        self.assertEqual(v[:, None].shape, (5, 1, 7, 3))
        self.assertEqual(v[:, None, None].shape, (5, 1, 1, 7, 3))
        self.assertEqual(v[..., None].shape, (5, 7, 3, 1))

    def test_step(self):
        v = torch.arange(10)
        self.assertEqual(v[::1], v)
        self.assertEqual(v[::2].tolist(), [0, 2, 4, 6, 8])
        self.assertEqual(v[::3].tolist(), [0, 3, 6, 9])
        self.assertEqual(v[::11].tolist(), [0])
        self.assertEqual(v[1:6:2].tolist(), [1, 3, 5])

    def test_step_assignment(self):
        v = torch.zeros(4, 4)
        v[0, 1::2] = torch.tensor([3., 4.])
        self.assertEqual(v[0].tolist(), [0, 3, 0, 4])
        self.assertEqual(v[1:].sum(), 0)

    def test_byte_mask(self):
        v = torch.randn(5, 7, 3)
        mask = torch.ByteTensor([1, 0, 1, 1, 0])
        self.assertEqual(v[mask].shape, (3, 7, 3))
        self.assertEqual(v[mask], torch.stack([v[0], v[2], v[3]]))

        v = torch.tensor([1.])
        self.assertEqual(v[v == 0], torch.tensor([]))

    def test_byte_mask_accumulate(self):
        mask = torch.zeros(size=(10, ), dtype=torch.uint8)
        y = torch.ones(size=(10, 10))
        y.index_put_((mask, ), y[mask], accumulate=True)
        self.assertEqual(y, torch.ones(size=(10, 10)))

    def test_multiple_byte_mask(self):
        v = torch.randn(5, 7, 3)
        # note: these broadcast together and are transposed to the first dim
        mask1 = torch.ByteTensor([1, 0, 1, 1, 0])
        mask2 = torch.ByteTensor([1, 1, 1])
        self.assertEqual(v[mask1, :, mask2].shape, (3, 7))

    def test_byte_mask2d(self):
        v = torch.randn(5, 7, 3)
        c = torch.randn(5, 7)
        num_ones = (c > 0).sum()
        r = v[c > 0]
        self.assertEqual(r.shape, (num_ones, 3))

    def test_int_indices(self):
        v = torch.randn(5, 7, 3)
        self.assertEqual(v[[0, 4, 2]].shape, (3, 7, 3))
        self.assertEqual(v[:, [0, 4, 2]].shape, (5, 3, 3))
        self.assertEqual(v[:, [[0, 1], [4, 3]]].shape, (5, 2, 2, 3))

    def test_int_indices2d(self):
        # From the NumPy indexing example
        x = torch.arange(0, 12).view(4, 3)
        rows = torch.tensor([[0, 0], [3, 3]])
        columns = torch.tensor([[0, 2], [0, 2]])
        self.assertEqual(x[rows, columns].tolist(), [[0, 2], [9, 11]])

    def test_int_indices_broadcast(self):
        # From the NumPy indexing example
        x = torch.arange(0, 12).view(4, 3)
        rows = torch.tensor([0, 3])
        columns = torch.tensor([0, 2])
        result = x[rows[:, None], columns]
        self.assertEqual(result.tolist(), [[0, 2], [9, 11]])

    def test_empty_index(self):
        x = torch.arange(0, 12).view(4, 3)
        idx = torch.tensor([], dtype=torch.long)
        self.assertEqual(x[idx].numel(), 0)

        # empty assignment should have no effect but not throw an exception
        y = x.clone()
        y[idx] = -1
        self.assertEqual(x, y)

        mask = torch.zeros(4, 3).byte()
        y[mask] = -1
        self.assertEqual(x, y)

    def test_empty_ndim_index(self):
        devices = ['cpu'] if not torch.cuda.is_available() else ['cpu', 'cuda']
        for device in devices:
            x = torch.randn(5, device=device)
            self.assertEqual(torch.empty(0, 2, device=device), x[torch.empty(0, 2, dtype=torch.int64, device=device)])

            x = torch.randn(2, 3, 4, 5, device=device)
            self.assertEqual(torch.empty(2, 0, 6, 4, 5, device=device),
                             x[:, torch.empty(0, 6, dtype=torch.int64, device=device)])

        x = torch.empty(10, 0)
        self.assertEqual(x[[1, 2]].shape, (2, 0))
        self.assertEqual(x[[], []].shape, (0,))
        with self.assertRaisesRegex(IndexError, 'for dimension with size 0'):
            x[:, [0, 1]]

    def test_empty_ndim_index_bool(self):
        devices = ['cpu'] if not torch.cuda.is_available() else ['cpu', 'cuda']
        for device in devices:
            x = torch.randn(5, device=device)
            self.assertRaises(IndexError, lambda: x[torch.empty(0, 2, dtype=torch.uint8, device=device)])

    def test_empty_slice(self):
        devices = ['cpu'] if not torch.cuda.is_available() else ['cpu', 'cuda']
        for device in devices:
            x = torch.randn(2, 3, 4, 5, device=device)
            y = x[:, :, :, 1]
            z = y[:, 1:1, :]
            self.assertEqual((2, 0, 4), z.shape)
            # this isn't technically necessary, but matches NumPy stride calculations.
            self.assertEqual((60, 20, 5), z.stride())
            self.assertTrue(z.is_contiguous())

    def test_index_getitem_copy_bools_slices(self):
        true = torch.tensor(1, dtype=torch.uint8)
        false = torch.tensor(0, dtype=torch.uint8)

        tensors = [torch.randn(2, 3), torch.tensor(3)]

        for a in tensors:
            self.assertNotEqual(a.data_ptr(), a[True].data_ptr())
            self.assertEqual(torch.empty(0, *a.shape), a[False])
            self.assertNotEqual(a.data_ptr(), a[true].data_ptr())
            self.assertEqual(torch.empty(0, *a.shape), a[false])
            self.assertEqual(a.data_ptr(), a[None].data_ptr())
            self.assertEqual(a.data_ptr(), a[...].data_ptr())

    def test_index_setitem_bools_slices(self):
        true = torch.tensor(1, dtype=torch.uint8)
        false = torch.tensor(0, dtype=torch.uint8)

        tensors = [torch.randn(2, 3), torch.tensor(3)]

        for a in tensors:
            # prefix with a 1,1, to ensure we are compatible with numpy which cuts off prefix 1s
            # (some of these ops already prefix a 1 to the size)
            neg_ones = torch.ones_like(a) * -1
            neg_ones_expanded = neg_ones.unsqueeze(0).unsqueeze(0)
            a[True] = neg_ones_expanded
            self.assertEqual(a, neg_ones)
            a[False] = 5
            self.assertEqual(a, neg_ones)
            a[true] = neg_ones_expanded * 2
            self.assertEqual(a, neg_ones * 2)
            a[false] = 5
            self.assertEqual(a, neg_ones * 2)
            a[None] = neg_ones_expanded * 3
            self.assertEqual(a, neg_ones * 3)
            a[...] = neg_ones_expanded * 4
            self.assertEqual(a, neg_ones * 4)
            if a.dim() == 0:
                with self.assertRaises(IndexError):
                    a[:] = neg_ones_expanded * 5

    def test_setitem_expansion_error(self):
        true = torch.tensor(True)
        a = torch.randn(2, 3)
        # check prefix with  non-1s doesn't work
        a_expanded = a.expand(torch.Size([5, 1]) + a.size())
        # NumPy: ValueError
        with self.assertRaises(RuntimeError):
            a[True] = a_expanded
        with self.assertRaises(RuntimeError):
            a[true] = a_expanded

    def test_getitem_scalars(self):
        zero = torch.tensor(0, dtype=torch.int64)
        one = torch.tensor(1, dtype=torch.int64)

        # non-scalar indexed with scalars
        a = torch.randn(2, 3)
        self.assertEqual(a[0], a[zero])
        self.assertEqual(a[0][1], a[zero][one])
        self.assertEqual(a[0, 1], a[zero, one])
        self.assertEqual(a[0, one], a[zero, 1])

        # indexing by a scalar should slice (not copy)
        self.assertEqual(a[0, 1].data_ptr(), a[zero, one].data_ptr())
        self.assertEqual(a[1].data_ptr(), a[one.int()].data_ptr())
        self.assertEqual(a[1].data_ptr(), a[one.short()].data_ptr())

        # scalar indexed with scalar
        r = torch.randn(())
        with self.assertRaises(IndexError):
            r[:]
        with self.assertRaises(IndexError):
            r[zero]
        self.assertEqual(r, r[...])

    def test_setitem_scalars(self):
        zero = torch.tensor(0, dtype=torch.int64)

        # non-scalar indexed with scalars
        a = torch.randn(2, 3)
        a_set_with_number = a.clone()
        a_set_with_scalar = a.clone()
        b = torch.randn(3)

        a_set_with_number[0] = b
        a_set_with_scalar[zero] = b
        self.assertEqual(a_set_with_number, a_set_with_scalar)
        a[1, zero] = 7.7
        self.assertEqual(7.7, a[1, 0])

        # scalar indexed with scalars
        r = torch.randn(())
        with self.assertRaises(IndexError):
            r[:] = 8.8
        with self.assertRaises(IndexError):
            r[zero] = 8.8
        r[...] = 9.9
        self.assertEqual(9.9, r)

    def test_basic_advanced_combined(self):
        # From the NumPy indexing example
        x = torch.arange(0, 12).view(4, 3)
        self.assertEqual(x[1:2, 1:3], x[1:2, [1, 2]])
        self.assertEqual(x[1:2, 1:3].tolist(), [[4, 5]])

        # Check that it is a copy
        unmodified = x.clone()
        x[1:2, [1, 2]].zero_()
        self.assertEqual(x, unmodified)

        # But assignment should modify the original
        unmodified = x.clone()
        x[1:2, [1, 2]] = 0
        self.assertNotEqual(x, unmodified)

    def test_int_assignment(self):
        x = torch.arange(0, 4).view(2, 2)
        x[1] = 5
        self.assertEqual(x.tolist(), [[0, 1], [5, 5]])

        x = torch.arange(0, 4).view(2, 2)
        x[1] = torch.arange(5, 7)
        self.assertEqual(x.tolist(), [[0, 1], [5, 6]])

    def test_byte_tensor_assignment(self):
        x = torch.arange(0., 16).view(4, 4)
        b = torch.ByteTensor([True, False, True, False])
        value = torch.tensor([3., 4., 5., 6.])
        x[b] = value
        self.assertEqual(x[0], value)
        self.assertEqual(x[1], torch.arange(4, 8))
        self.assertEqual(x[2], value)
        self.assertEqual(x[3], torch.arange(12, 16))

    def test_variable_slicing(self):
        x = torch.arange(0, 16).view(4, 4)
        indices = torch.IntTensor([0, 1])
        i, j = indices
        self.assertEqual(x[i:j], x[0:1])

    def test_ellipsis_tensor(self):
        x = torch.arange(0, 9).view(3, 3)
        idx = torch.tensor([0, 2])
        self.assertEqual(x[..., idx].tolist(), [[0, 2],
                                                [3, 5],
                                                [6, 8]])
        self.assertEqual(x[idx, ...].tolist(), [[0, 1, 2],
                                                [6, 7, 8]])

    def test_invalid_index(self):
        x = torch.arange(0, 16).view(4, 4)
        self.assertRaisesRegex(TypeError, 'slice indices', lambda: x["0":"1"])

    def test_out_of_bound_index(self):
        x = torch.arange(0, 100).view(2, 5, 10)
        self.assertRaisesRegex(IndexError, 'index 5 is out of bounds for dimension 1 with size 5', lambda: x[0, 5])
        self.assertRaisesRegex(IndexError, 'index 4 is out of bounds for dimension 0 with size 2', lambda: x[4, 5])
        self.assertRaisesRegex(IndexError, 'index 15 is out of bounds for dimension 2 with size 10',
                               lambda: x[0, 1, 15])
        self.assertRaisesRegex(IndexError, 'index 12 is out of bounds for dimension 2 with size 10',
                               lambda: x[:, :, 12])

    def test_zero_dim_index(self):
        x = torch.tensor(10)
        self.assertEqual(x, x.item())

        def runner():
            print(x[0])
            return x[0]

        self.assertRaisesRegex(IndexError, 'invalid index', runner)


# The tests below are from NumPy test_indexing.py with some modifications to
# make them compatible with PyTorch. It's licensed under the BDS license below:
#
# Copyright (c) 2005-2017, NumPy Developers.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met:
#
#     * Redistributions of source code must retain the above copyright
#        notice, this list of conditions and the following disclaimer.
#
#     * Redistributions in binary form must reproduce the above
#        copyright notice, this list of conditions and the following
#        disclaimer in the documentation and/or other materials provided
#        with the distribution.
#
#     * Neither the name of the NumPy Developers nor the names of any
#        contributors may be used to endorse or promote products derived
#        from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


class NumpyTests(TestCase):
    def test_index_no_floats(self):
        a = torch.tensor([[[5.]]])

        self.assertRaises(IndexError, lambda: a[0.0])
        self.assertRaises(IndexError, lambda: a[0, 0.0])
        self.assertRaises(IndexError, lambda: a[0.0, 0])
        self.assertRaises(IndexError, lambda: a[0.0, :])
        self.assertRaises(IndexError, lambda: a[:, 0.0])
        self.assertRaises(IndexError, lambda: a[:, 0.0, :])
        self.assertRaises(IndexError, lambda: a[0.0, :, :])
        self.assertRaises(IndexError, lambda: a[0, 0, 0.0])
        self.assertRaises(IndexError, lambda: a[0.0, 0, 0])
        self.assertRaises(IndexError, lambda: a[0, 0.0, 0])
        self.assertRaises(IndexError, lambda: a[-1.4])
        self.assertRaises(IndexError, lambda: a[0, -1.4])
        self.assertRaises(IndexError, lambda: a[-1.4, 0])
        self.assertRaises(IndexError, lambda: a[-1.4, :])
        self.assertRaises(IndexError, lambda: a[:, -1.4])
        self.assertRaises(IndexError, lambda: a[:, -1.4, :])
        self.assertRaises(IndexError, lambda: a[-1.4, :, :])
        self.assertRaises(IndexError, lambda: a[0, 0, -1.4])
        self.assertRaises(IndexError, lambda: a[-1.4, 0, 0])
        self.assertRaises(IndexError, lambda: a[0, -1.4, 0])
        # self.assertRaises(IndexError, lambda: a[0.0:, 0.0])
        # self.assertRaises(IndexError, lambda: a[0.0:, 0.0,:])

    def test_none_index(self):
        # `None` index adds newaxis
        a = tensor([1, 2, 3])
        self.assertEqual(a[None].dim(), a.dim() + 1)

    def test_empty_tuple_index(self):
        # Empty tuple index creates a view
        a = tensor([1, 2, 3])
        self.assertEqual(a[()], a)
        self.assertEqual(a[()].data_ptr(), a.data_ptr())

    def test_empty_fancy_index(self):
        # Empty list index creates an empty array
        a = tensor([1, 2, 3])
        self.assertEqual(a[[]], torch.tensor([]))

        b = tensor([]).long()
        self.assertEqual(a[[]], torch.tensor([], dtype=torch.long))

        b = tensor([]).float()
        self.assertRaises(IndexError, lambda: a[b])

    def test_ellipsis_index(self):
        a = tensor([[1, 2, 3],
                    [4, 5, 6],
                    [7, 8, 9]])
        self.assertIsNot(a[...], a)
        self.assertEqual(a[...], a)
        # `a[...]` was `a` in numpy <1.9.
        self.assertEqual(a[...].data_ptr(), a.data_ptr())

        # Slicing with ellipsis can skip an
        # arbitrary number of dimensions
        self.assertEqual(a[0, ...], a[0])
        self.assertEqual(a[0, ...], a[0, :])
        self.assertEqual(a[..., 0], a[:, 0])

        # In NumPy, slicing with ellipsis results in a 0-dim array. In PyTorch
        # we don't have separate 0-dim arrays and scalars.
        self.assertEqual(a[0, ..., 1], torch.tensor(2))

        # Assignment with `(Ellipsis,)` on 0-d arrays
        b = torch.tensor(1)
        b[(Ellipsis,)] = 2
        self.assertEqual(b, 2)

    def test_single_int_index(self):
        # Single integer index selects one row
        a = tensor([[1, 2, 3],
                    [4, 5, 6],
                    [7, 8, 9]])

        self.assertEqual(a[0], [1, 2, 3])
        self.assertEqual(a[-1], [7, 8, 9])

        # Index out of bounds produces IndexError
        self.assertRaises(IndexError, a.__getitem__, 1 << 30)
        # Index overflow produces Exception  NB: different exception type
        self.assertRaises(Exception, a.__getitem__, 1 << 64)

    def test_single_bool_index(self):
        # Single boolean index
        a = tensor([[1, 2, 3],
                    [4, 5, 6],
                    [7, 8, 9]])

        self.assertEqual(a[True], a[None])
        self.assertEqual(a[False], a[None][0:0])

    def test_boolean_shape_mismatch(self):
        arr = torch.ones((5, 4, 3))

        index = tensor([True])
        self.assertRaisesRegex(IndexError, 'mask', lambda: arr[index])

        index = tensor([False] * 6)
        self.assertRaisesRegex(IndexError, 'mask', lambda: arr[index])

        index = torch.ByteTensor(4, 4).zero_()
        self.assertRaisesRegex(IndexError, 'mask', lambda: arr[index])

        self.assertRaisesRegex(IndexError, 'mask', lambda: arr[(slice(None), index)])

    def test_boolean_indexing_onedim(self):
        # Indexing a 2-dimensional array with
        # boolean array of length one
        a = tensor([[0., 0., 0.]])
        b = tensor([True])
        self.assertEqual(a[b], a)
        # boolean assignment
        a[b] = 1.
        self.assertEqual(a, tensor([[1., 1., 1.]]))

    def test_boolean_assignment_value_mismatch(self):
        # A boolean assignment should fail when the shape of the values
        # cannot be broadcast to the subscription. (see also gh-3458)
        a = torch.arange(0, 4)

        def f(a, v):
            a[a > -1] = tensor(v)

        self.assertRaisesRegex(Exception, 'shape mismatch', f, a, [])
        self.assertRaisesRegex(Exception, 'shape mismatch', f, a, [1, 2, 3])
        self.assertRaisesRegex(Exception, 'shape mismatch', f, a[:1], [1, 2, 3])

    def test_boolean_indexing_twodim(self):
        # Indexing a 2-dimensional array with
        # 2-dimensional boolean array
        a = tensor([[1, 2, 3],
                    [4, 5, 6],
                    [7, 8, 9]])
        b = tensor([[True, False, True],
                    [False, True, False],
                    [True, False, True]])
        self.assertEqual(a[b], tensor([1, 3, 5, 7, 9]))
        self.assertEqual(a[b[1]], tensor([[4, 5, 6]]))
        self.assertEqual(a[b[0]], a[b[2]])

        # boolean assignment
        a[b] = 0
        self.assertEqual(a, tensor([[0, 2, 0],
                                    [4, 0, 6],
                                    [0, 8, 0]]))

    def test_boolean_indexing_weirdness(self):
        # Weird boolean indexing things
        a = torch.ones((2, 3, 4))
        self.assertEqual((0, 2, 3, 4), a[False, True, ...].shape)
        self.assertEqual(torch.ones(1, 2), a[True, [0, 1], True, True, [1], [[2]]])
        self.assertRaises(IndexError, lambda: a[False, [0, 1], ...])

    def test_boolean_indexing_weirdness_tensors(self):
        # Weird boolean indexing things
        false = torch.tensor(False)
        true = torch.tensor(True)
        a = torch.ones((2, 3, 4))
        self.assertEqual((0, 2, 3, 4), a[False, True, ...].shape)
        self.assertEqual(torch.ones(1, 2), a[true, [0, 1], true, true, [1], [[2]]])
        self.assertRaises(IndexError, lambda: a[false, [0, 1], ...])

    def test_boolean_indexing_alldims(self):
        true = torch.tensor(True)
        a = torch.ones((2, 3))
        self.assertEqual((1, 2, 3), a[True, True].shape)
        self.assertEqual((1, 2, 3), a[true, true].shape)

    def test_boolean_list_indexing(self):
        # Indexing a 2-dimensional array with
        # boolean lists
        a = tensor([[1, 2, 3],
                    [4, 5, 6],
                    [7, 8, 9]])
        b = [True, False, False]
        c = [True, True, False]
        self.assertEqual(a[b], tensor([[1, 2, 3]]))
        self.assertEqual(a[b, b], tensor([1]))
        self.assertEqual(a[c], tensor([[1, 2, 3], [4, 5, 6]]))
        self.assertEqual(a[c, c], tensor([1, 5]))

    def test_everything_returns_views(self):
        # Before `...` would return a itself.
        a = tensor([5])

        self.assertIsNot(a, a[()])
        self.assertIsNot(a, a[...])
        self.assertIsNot(a, a[:])

    def test_broaderrors_indexing(self):
        a = torch.zeros(5, 5)
        self.assertRaisesRegex(IndexError, 'shape mismatch', a.__getitem__, ([0, 1], [0, 1, 2]))
        self.assertRaisesRegex(IndexError, 'shape mismatch', a.__setitem__, ([0, 1], [0, 1, 2]), 0)

    def test_trivial_fancy_out_of_bounds(self):
        a = torch.zeros(5)
        ind = torch.ones(20, dtype=torch.int64)
        if a.is_cuda:
            raise unittest.SkipTest('CUDA asserts instead of raising an exception')
        ind[-1] = 10
        self.assertRaises(IndexError, a.__getitem__, ind)
        self.assertRaises(IndexError, a.__setitem__, ind, 0)
        ind = torch.ones(20, dtype=torch.int64)
        ind[0] = 11
        self.assertRaises(IndexError, a.__getitem__, ind)
        self.assertRaises(IndexError, a.__setitem__, ind, 0)

    def test_index_is_larger(self):
        # Simple case of fancy index broadcasting of the index.
        a = torch.zeros((5, 5))
        a[[[0], [1], [2]], [0, 1, 2]] = tensor([2., 3., 4.])

        self.assertTrue((a[:3, :3] == tensor([2., 3., 4.])).all())

    def test_broadcast_subspace(self):
        a = torch.zeros((100, 100))
        v = torch.arange(0., 100)[:, None]
        b = torch.arange(99, -1, -1).long()
        a[b] = v
        expected = b.double().unsqueeze(1).expand(100, 100)
        self.assertEqual(a, expected)


if __name__ == '__main__':
    run_tests()
