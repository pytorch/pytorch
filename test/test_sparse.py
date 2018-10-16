import torch
from torch import sparse

import itertools
import functools
import random
import unittest
from common import TestCase, run_tests, skipIfRocm
from common_cuda import TEST_CUDA
from test_torch import TestTorch
from numbers import Number


def cpu_only(inner):
    @functools.wraps(inner)
    def outer(self, *args, **kwargs):
        if self.is_cuda:
            raise unittest.SkipTest("Test is CPU-only")
        inner(self, *args, **kwargs)
    return outer


def cuda_only(inner):
    @functools.wraps(inner)
    def outer(self, *args, **kwargs):
        if not self.is_cuda:
            raise unittest.SkipTest("Test is GPU-only")
        inner(self, *args, **kwargs)
    return outer


class TestSparse(TestCase):

    def setUp(self):
        # These parameters control the various ways we can run the test.
        # We will subclass and override this method to implement CUDA
        # tests
        self.is_cuda = False
        self.is_uncoalesced = False
        self.device = 'cpu'
        self.IndexTensor = torch.LongTensor
        self.ValueTensor = torch.DoubleTensor
        self.value_dtype = torch.float64
        self.SparseTensor = torch.sparse.DoubleTensor
        super(TestSparse, self).setUp()

    def _gen_sparse(self, sparse_dims, nnz, with_size):
        # TODO: Consider implementing this in the CUDA case by directly
        # performing the operations on the GPU.  You won't be able to
        # use torch.rand/torch.randn in this case because they are
        # CPU-only.  If you do this, you can remove the is_cuda branch
        # at the end.
        #
        # If you do this, be sure to update assert_uncoalesced too

        if isinstance(with_size, Number):
            with_size = [with_size] * sparse_dims

        if self.is_uncoalesced:
            # We want to generate a tensor with a lot of uncoalesced
            # entries to stress test whether or not we handle this
            # (subtle) case correctly
            v_size = [nnz * 2] + list(with_size[sparse_dims:])
            v = torch.randn(*v_size)
            r = torch.rand(sparse_dims, nnz)
            # Repeat the indexes, so every position shows up twice
            i = torch.cat([r, r], dim=1)
            if nnz > 0:
                i *= torch.Tensor(with_size[:sparse_dims]).repeat(nnz * 2, 1).transpose(0, 1)
            i = i.type(torch.LongTensor)
            x = torch.sparse.DoubleTensor(i, v, torch.Size(with_size))
            self.assert_uncoalesced(x)
        else:
            # Generate a sparse tensor with sparse_dims sparse dimensions; the
            # rest the dimensions with_size[sparse_dims:] are dense.
            v_size = [nnz] + list(with_size[sparse_dims:])
            v = torch.randn(*v_size)
            i = torch.rand(sparse_dims, nnz)
            if nnz > 0:
                i *= torch.Tensor(with_size[:sparse_dims]).repeat(nnz, 1).transpose(0, 1)
            i = i.type(torch.LongTensor)
            x = torch.sparse.DoubleTensor(i, v, torch.Size(with_size))

        if self.is_cuda:
            return x.cuda(), i.cuda(), v.cuda()
        else:
            return x, i.clone(), v.clone()

    def assert_uncoalesced(self, x):
        """
        Test if a CPU tensor is uncoalesced.  This is used to ensure
        correctness of the uncoalesced tensor generation algorithm.
        """
        assert not x.is_coalesced()
        existing_indices = set()
        for i in range(x._nnz()):
            index = str(x._indices()[:, i])
            if index in existing_indices:
                return True
            else:
                existing_indices.add(index)

    def randn(self, *args, **kwargs):
        """
        Variant of torch.randn that also works in the TEST_CUDA case.
        """
        # TODO: Put this in torch.cuda.randn
        return self.ValueTensor(*args, **kwargs).normal_()

    @skipIfRocm  # ROCm stack doesn't like the x + x call
    def test_print(self):
        shape_sparseDim_nnz = [
            ((), 0, 2),
            ((0,), 0, 10),
            ((2,), 0, 3),
            ((100, 3), 1, 3),
            ((100, 20, 3), 2, 0),
            ((10, 0, 3), 0, 3),
            ((10, 0, 3), 0, 0),
        ]

        printed = []
        for shape, sparseDim, nnz in shape_sparseDim_nnz:
            indices_shape = torch.Size((sparseDim, nnz))
            values_shape = torch.Size((nnz,) + shape[sparseDim:])
            printed.append("# shape: {}".format(torch.Size(shape)))
            printed.append("# nnz: {}".format(nnz))
            printed.append("# sparseDim: {}".format(sparseDim))
            printed.append("# indices shape: {}".format(indices_shape))
            printed.append("# values shape: {}".format(values_shape))

            indices = torch.arange(indices_shape.numel(), dtype=self.IndexTensor.dtype,
                                   device=self.device).view(indices_shape)
            for d in range(sparseDim):
                indices[d].clamp_(max=(shape[d] - 1))  # make it valid index
            if self.is_uncoalesced and indices.numel() > 0:
                indices[:, -1] = indices[:, 0]  # make it uncoalesced
            values_numel = values_shape.numel()
            values = torch.arange(values_numel, dtype=self.ValueTensor.dtype,
                                  device=self.device).view(values_shape).div_(values_numel / 2.)
            sp_tensor = self.SparseTensor(indices, values, shape)

            dtypes = [torch.int32]
            if values.dtype == torch.double:
                dtypes.append(torch.float)
            else:
                dtypes.append(torch.double)
            for dtype in dtypes:
                printed.append("########## {} ##########".format(dtype))
                x = sp_tensor.detach().to(dtype)
                printed.append("# sparse tensor")
                printed.append(str(x))
                if x.dtype.is_floating_point:
                    printed.append("# after requires_grad_")
                    printed.append(str(x.requires_grad_()))
                    printed.append("# after addition")
                    printed.append(str(x + x))
                printed.append("# _indices")
                printed.append(str(x._indices()))
                printed.append("# _values")
                printed.append(str(x._values()))
            printed.append('')
        self.assertExpected('\n'.join(printed))

    @skipIfRocm
    def test_basic(self):
        def test_shape(sparse_dims, nnz, with_size):
            if isinstance(with_size, Number):
                with_size = [with_size] * sparse_dims
            x, i, v = self._gen_sparse(sparse_dims, nnz, with_size)
            self.assertEqual(i, x._indices())
            self.assertEqual(v, x._values())
            self.assertEqual(x.ndimension(), len(with_size))
            self.assertEqual(self.safeCoalesce(x)._nnz(), nnz)
            self.assertEqual(list(x.size()), with_size)

        test_shape(3, 10, 100)
        test_shape(3, 10, [100, 100, 100])
        test_shape(3, 10, [100, 100, 100, 5, 5, 5, 0])
        test_shape(3, 0, [0, 0, 100, 5, 5, 5, 0])

        # Make sure that coalesce handles duplicate indices correctly
        i = self.IndexTensor([[9, 0, 0, 0, 8, 1, 1, 1, 2, 7, 2, 2, 3, 4, 6, 9]])
        v = self.ValueTensor([[idx**2, idx] for idx in range(i.size(1))])
        x = self.SparseTensor(i, v, torch.Size([10, 2]))
        self.assertEqual(self.safeCoalesce(x)._nnz(), 9)

        # Make sure we can access empty indices / values
        x = self.SparseTensor()
        self.assertEqual(x._indices().numel(), 0)
        self.assertEqual(x._values().numel(), 0)

    def test_ctor_size_checks(self):
        indices = self.IndexTensor([
            [0, 0, 0],
            [0, 3, 0],
            [0, 0, 0],
            [0, 0, 0],
        ])
        values = self.ValueTensor([2, 1, 3, 4])

        # indices inconsistent with size
        self.assertRaises(
            RuntimeError,
            lambda: self.SparseTensor(indices, values, torch.Size([2, 1, 1])))

        # values inconsistent with size
        values = self.ValueTensor([
            [2, 1, 2, 1],
            [1, 0, 5, 2],
        ])
        self.assertRaises(
            RuntimeError,
            lambda: self.SparseTensor(indices, values, torch.Size([2, 4, 2, 1])))

    def test_to_dense(self):
        def test_tensor(x, res):
            x.to_dense()  # Tests triple to_dense for memory corruption
            x.to_dense()
            x.to_dense()
            self.assertEqual(res, x.to_dense())
            self.assertEqual(res, self.safeToDense(x))

        i = self.IndexTensor([
            [0, 1, 2, 2],
            [0, 0, 0, 3],
            [0, 0, 1, 4],
        ])
        v = self.ValueTensor([2, 1, 3, 4])
        x = self.SparseTensor(i, v, torch.Size([3, 4, 5]))
        res = self.ValueTensor([
            [[2, 0, 0, 0, 0],
             [0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0]],
            [[1, 0, 0, 0, 0],
             [0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0]],
            [[0, 3, 0, 0, 0],
             [0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0],
             [0, 0, 0, 0, 4]],
        ])
        test_tensor(x, res)

        i = self.IndexTensor([
            [0, 1, 2, 2],
            [0, 0, 0, 3],
            [0, 0, 1, 4],
        ])
        v = self.ValueTensor(4, 0)
        x = self.SparseTensor(i, v, torch.Size([3, 4, 5, 0]))
        res = self.ValueTensor(3, 4, 5, 0)
        test_tensor(x, res)

    def test_shared(self):
        i = self.IndexTensor([[2]])
        v = self.ValueTensor([5])
        x = self.SparseTensor(i, v, torch.Size([3]))
        v[0] = 6
        self.assertEqual(self.ValueTensor([0, 0, 6]), self.safeToDense(x))
        i[0][0] = 0
        self.assertEqual(self.ValueTensor([6, 0, 0]), self.safeToDense(x))

        i = self.IndexTensor([[2]])
        v = self.ValueTensor(1, 0)
        x = self.SparseTensor(i, v, torch.Size([3, 0]))
        i[0][0] = 0
        self.assertEqual(self.ValueTensor(3, 0), self.safeToDense(x))

    def test_to_dense_hybrid(self):
        def test_tensor(x, res):
            x.to_dense()  # Tests double to_dense for memory corruption
            x.to_dense()
            x.to_dense()
            self.assertEqual(res, x.to_dense())
            self.assertEqual(res, self.safeToDense(x))

        i = self.IndexTensor([
            [0, 1, 2, 2],
            [0, 0, 0, 3],
        ])
        v = self.ValueTensor([[2, 3], [1, 2], [3, 4], [4, 5]])
        x = self.SparseTensor(i, v, torch.Size([3, 4, 2]))
        res = self.ValueTensor([
            [[2, 3],
             [0, 0],
             [0, 0],
             [0, 0]],
            [[1, 2],
             [0, 0],
             [0, 0],
             [0, 0]],
            [[3, 4],
             [0, 0],
             [0, 0],
             [4, 5]],
        ])
        test_tensor(x, res)

        i = self.IndexTensor([
            [0, 1, 2, 2],
            [0, 0, 0, 3],
        ])
        v = self.ValueTensor(4, 2, 0)
        x = self.SparseTensor(i, v, torch.Size([3, 4, 2, 0]))
        res = self.ValueTensor(3, 4, 2, 0)
        test_tensor(x, res)

    @skipIfRocm
    def test_contig(self):
        def test_tensor(x, exp_i, exp_v):
            x = self.safeCoalesce(x)
            self.assertEqual(exp_i, x._indices())
            self.assertEqual(exp_v, x._values())

        i = self.IndexTensor([
            [1, 0, 35, 14, 39, 6, 71, 66, 40, 27],
            [92, 31, 62, 50, 22, 65, 89, 74, 56, 34],
        ])
        v = self.ValueTensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        x = self.SparseTensor(i, v, torch.Size([100, 100]))
        exp_i = self.IndexTensor([
            [0, 1, 6, 14, 27, 35, 39, 40, 66, 71],
            [31, 92, 65, 50, 34, 62, 22, 56, 74, 89],
        ])
        exp_v = self.ValueTensor([2, 1, 6, 4, 10, 3, 5, 9, 8, 7])
        test_tensor(x, exp_i, exp_v)

        i = self.IndexTensor([
            [2, 0, 2, 1],
            [0, 0, 3, 0],
            [1, 0, 4, 0],
        ])
        v = self.ValueTensor([3, 2, 4, 1])
        x = self.SparseTensor(i, v, torch.Size([3, 4, 5]))
        exp_i = self.IndexTensor([
            [0, 1, 2, 2],
            [0, 0, 0, 3],
            [0, 0, 1, 4],
        ])
        exp_v = self.ValueTensor([2, 1, 3, 4])
        test_tensor(x, exp_i, exp_v)

        i = self.IndexTensor([
            [2, 0, 2, 1],
            [0, 0, 3, 0],
            [1, 0, 4, 0],
        ])
        v = self.ValueTensor(4, 0)
        x = self.SparseTensor(i, v, torch.Size([3, 4, 5, 0]))
        exp_i = self.IndexTensor([
            [0, 1, 2, 2],
            [0, 0, 0, 3],
            [0, 0, 1, 4],
        ])
        exp_v = self.ValueTensor(4, 0)
        test_tensor(x, exp_i, exp_v)

        # Duplicate indices
        i = self.IndexTensor([
            [0, 0, 2, 0],
            [0, 0, 3, 0],
            [0, 0, 4, 0],
        ])
        v = self.ValueTensor([3, 2, 4, 1])
        x = self.SparseTensor(i, v, torch.Size([3, 4, 5]))
        exp_i = self.IndexTensor([
            [0, 2],
            [0, 3],
            [0, 4],
        ])
        exp_v = self.ValueTensor([6, 4])
        test_tensor(x, exp_i, exp_v)

        i = self.IndexTensor([
            [0, 0, 2, 0],
            [0, 0, 3, 0],
            [0, 0, 4, 0],
        ])
        v = self.ValueTensor(4, 0)
        x = self.SparseTensor(i, v, torch.Size([3, 4, 5, 0]))
        exp_i = self.IndexTensor([
            [0, 2],
            [0, 3],
            [0, 4],
        ])
        exp_v = self.ValueTensor(2, 0)
        test_tensor(x, exp_i, exp_v)

    @skipIfRocm
    def test_contig_hybrid(self):
        def test_tensor(x, exp_i, exp_v):
            x = self.safeCoalesce(x)
            self.assertEqual(exp_i, x._indices())
            self.assertEqual(exp_v, x._values())

        i = self.IndexTensor([
            [1, 0, 35, 14, 39, 6, 71, 66, 40, 27],
            [92, 31, 62, 50, 22, 65, 89, 74, 56, 34],
        ])
        v = self.ValueTensor([
            [1, 2], [2, 3], [3, 4], [4, 5], [5, 6],
            [6, 7], [7, 8], [8, 9], [9, 10], [10, 11],
        ])
        x = self.SparseTensor(i, v, torch.Size([100, 100, 2]))
        exp_i = self.IndexTensor([
            [0, 1, 6, 14, 27, 35, 39, 40, 66, 71],
            [31, 92, 65, 50, 34, 62, 22, 56, 74, 89],
        ])
        exp_v = self.ValueTensor([
            [2, 3], [1, 2], [6, 7], [4, 5], [10, 11],
            [3, 4], [5, 6], [9, 10], [8, 9], [7, 8],
        ])
        test_tensor(x, exp_i, exp_v)

        i = self.IndexTensor([
            [2, 0, 2, 1],
            [0, 0, 3, 0],
            [1, 0, 4, 0],
        ])
        v = self.ValueTensor([[3, 3, 3], [2, 2, 2], [4, 4, 4], [1, 1, 1]])
        x = self.SparseTensor(i, v, torch.Size([3, 4, 5, 3]))
        exp_i = self.IndexTensor([
            [0, 1, 2, 2],
            [0, 0, 0, 3],
            [0, 0, 1, 4],
        ])
        exp_v = self.ValueTensor([[2, 2, 2], [1, 1, 1], [3, 3, 3], [4, 4, 4]])
        test_tensor(x, exp_i, exp_v)

        i = self.IndexTensor([
            [2, 0, 2, 1],
            [0, 0, 3, 0],
            [1, 0, 4, 0],
        ])
        v = self.ValueTensor(4, 3, 0)
        x = self.SparseTensor(i, v, torch.Size([3, 4, 5, 3, 0]))
        exp_i = self.IndexTensor([
            [0, 1, 2, 2],
            [0, 0, 0, 3],
            [0, 0, 1, 4],
        ])
        exp_v = self.ValueTensor(4, 3, 0)
        test_tensor(x, exp_i, exp_v)

        # Duplicate indices
        i = self.IndexTensor([
            [0, 0, 2, 0],
            [0, 0, 3, 0],
            [0, 0, 4, 0],
        ])
        v = self.ValueTensor([[3, 2, 3], [2, 1, 1], [4, 3, 4], [1, 1, 1]])
        x = self.SparseTensor(i, v, torch.Size([3, 4, 5, 3]))
        exp_i = self.IndexTensor([
            [0, 2],
            [0, 3],
            [0, 4],
        ])
        exp_v = self.ValueTensor([[6, 4, 5], [4, 3, 4]])
        test_tensor(x, exp_i, exp_v)

        i = self.IndexTensor([
            [0, 0, 2, 0],
            [0, 0, 3, 0],
            [0, 0, 4, 0],
        ])
        v = self.ValueTensor(4, 3, 0)
        x = self.SparseTensor(i, v, torch.Size([3, 4, 5, 3, 0]))
        exp_i = self.IndexTensor([
            [0, 2],
            [0, 3],
            [0, 4],
        ])
        exp_v = self.ValueTensor(2, 3, 0)
        test_tensor(x, exp_i, exp_v)

    def test_clone(self):
        def test_shape(sparse_dims, nnz, with_size):
            x = self._gen_sparse(sparse_dims, nnz, with_size)[0]
            if self.is_uncoalesced:
                self.assertFalse(x.is_coalesced())
                y = x.clone()
                self.assertFalse(y.is_coalesced())
            x = x.coalesce()
            self.assertTrue(x.is_coalesced())
            y = x.clone()
            self.assertTrue(y.is_coalesced())

        test_shape(4, 20, 5)
        test_shape(3, 10, [100, 100, 100, 5, 5, 5, 0])
        test_shape(3, 0, [0, 0, 100, 5, 5, 5, 0])

    @skipIfRocm
    def test_Sparse_to_Sparse_copy_(self):
        # This is for testing torch.copy_(SparseTensor, SparseTensor)
        sparse_dims = 3
        nnz = 10
        sizes = [2, 3, 4, 5]  # hybrid sparse
        x1, _, _ = self._gen_sparse(sparse_dims, nnz, sizes)
        x2, _, _ = self._gen_sparse(sparse_dims, nnz + 10, sizes)

        # test copy
        x2_dense = x2.to_dense()
        x1.copy_(x2)
        self.assertEqual(x2_dense, x1.to_dense())

        # test type conversion (when x1.copy_(x2), x1.dtype should stay the same)
        x1 = x1.to(torch.float32)
        x2 = x2.to(torch.float64)
        x1_dtype = x1.dtype
        x1.copy_(x2)
        self.assertEqual(x1_dtype, x1.dtype)

        # test no broadcast
        self.assertRaises(RuntimeError, lambda: x1.copy_(x2.narrow_copy(0, 0, 1)))

        # test raise error on copy_() between dense and sparse Tensors
        self.assertRaises(RuntimeError, lambda: x1.copy_(torch.randn(5, 5)))

        # test autograd
        x1, _, _ = self._gen_sparse(sparse_dims, nnz, sizes)
        x2, _, _ = self._gen_sparse(sparse_dims, nnz + 10, sizes)
        x2.requires_grad_(True)
        x1.copy_(x2)
        y = x1 * 2
        x2_clone = x2.clone()
        y.backward(x2_clone)
        expected_grad = x2_clone * 2
        self.assertEqual(expected_grad.to_dense(), x2.grad.to_dense())
        self.assertEqual(None, x1.grad)

    @unittest.skipIf(torch.cuda.device_count() < 2, "no multi-GPU")
    @skipIfRocm
    def test_Sparse_to_Sparse_copy_multi_gpu(self):
        # This is for testing torch.copy_(SparseTensor, SparseTensor) across GPU devices
        sparse_dims = 3
        nnz = 10
        sizes = [2, 3, 4, 5]  # hybrid sparse
        x1, _, _ = self._gen_sparse(sparse_dims, nnz, sizes)
        x2, _, _ = self._gen_sparse(sparse_dims, nnz + 10, sizes)
        x1 = x1.to('cuda:0')

        def test_cross_device(x1, x2):
            x1_device = x1.device
            x1.copy_(x2)
            self.assertEqual(x2.to('cuda:0').to_dense(), x1.to_dense())
            self.assertEqual(x1_device, x1.device)

        test_cross_device(x1, x2.to('cuda:1'))  # test across gpu devices
        test_cross_device(x1, x2.to('cpu'))  # test between cpu and gpu

        # test autograd
        x2 = x2.to('cuda:1')
        x2.requires_grad_(True)
        x1.copy_(x2)
        y = x1 * 2
        x2_clone = x2.clone().to('cuda:0')
        y.backward(x2_clone)
        expected_grad = x2_clone * 2
        self.assertEqual(expected_grad.to_dense(), x2.grad.to('cuda:0').to_dense())
        self.assertEqual(None, x1.grad)

    @cuda_only
    def test_cuda_empty(self):
        def test_tensor(x):
            y = x.cuda(0)
            self.assertEqual(x._sparseDims(), y._sparseDims())
            self.assertEqual(x._denseDims(), y._denseDims())
            x = y.cpu()
            self.assertEqual(y._sparseDims(), x._sparseDims())
            self.assertEqual(y._denseDims(), x._denseDims())

        x = torch.sparse.FloatTensor(2, 3, 4)
        test_tensor(x)

        x = torch.sparse.FloatTensor(2, 3, 4, 0)
        test_tensor(x)

    @skipIfRocm
    def test_transpose(self):
        def test_shape(sparse_dims, nnz, with_size):
            x = self._gen_sparse(sparse_dims, nnz, with_size)[0]
            y = self.safeToDense(x)

            for i, j in itertools.combinations(range(4), 2):
                x = x.transpose_(i, j)
                y = y.transpose(i, j)
                self.assertEqual(self.safeToDense(x), y)

                x = x.transpose(i, j)
                y = y.transpose(i, j)
                self.assertEqual(self.safeToDense(x), y)

        test_shape(4, 20, 5)
        test_shape(4, 10, [100, 100, 100, 5, 5, 5, 0])
        test_shape(4, 0, [0, 0, 100, 5, 5, 5, 0])

    @cpu_only
    def test_coalesce_transpose_mm(self):
        def test_shape(di, dj, dk, nnz):
            x, _, _ = self._gen_sparse(2, nnz, [dj, di])
            y = torch.randn(dj, dk)

            x_coalesced = x.coalesce()
            self.assertTrue(x_coalesced.is_coalesced())

            x_coalesced_t = x.t()
            self.assertFalse(x_coalesced_t.is_coalesced())

            res = torch.mm(x_coalesced_t, y)
            expected = torch.mm(self.safeToDense(x_coalesced_t), y)
            self.assertEqual(res, expected)

        test_shape(10, 20, 30, 20)
        test_shape(0, 20, 30, 0)
        test_shape(10, 0, 30, 0)
        test_shape(10, 20, 0, 0)
        test_shape(10, 20, 0, 20)

    def test_t_empty(self):
        def test_in_place(x):
            shape_original = x.shape
            x.t_()
            self.assertEqual(torch.Size([shape_original[1], shape_original[0]]), x.size())
            self.assertEqual(0, x._indices().numel())
            self.assertEqual(0, x._values().numel())
            self.assertEqual(x._sparseDims(), 2)
            self.assertEqual(x._denseDims(), 0)

        def test_not_in_place(x):
            shape_original = x.shape
            y = x.t()
            self.assertEqual(torch.Size([shape_original[1], shape_original[0]]), y.size())
            self.assertEqual(0, y._indices().numel())
            self.assertEqual(0, y._values().numel())
            self.assertEqual(x._sparseDims(), 2)
            self.assertEqual(x._denseDims(), 0)

        x = self.SparseTensor(2, 3)
        test_in_place(x)
        test_not_in_place(x)

        x = self.SparseTensor(2, 0)
        test_in_place(x)
        test_not_in_place(x)

    @skipIfRocm
    def test_add_zeros(self):
        def test_shape(sparse_dims, nnz, sizes):
            x, _, _ = self._gen_sparse(sparse_dims, nnz, sizes)
            zeros = torch.zeros(sizes, layout=torch.sparse_coo).to(x.device)
            r1 = zeros + x
            r2 = x + zeros
            self.assertEqual(r1, x)
            self.assertEqual(r2, x)

        test_shape(1, 20, [1])
        test_shape(4, 20, [3, 17, 19, 5])
        test_shape(2, 20, [3, 17, 19, 5])
        test_shape(2, 20, [3, 17, 19, 0])

    @cpu_only
    def test_mm(self):
        def test_shape(di, dj, dk, nnz):
            x, _, _ = self._gen_sparse(2, nnz, [di, dj])
            t = torch.randn(di, dk)
            y = torch.randn(dj, dk)
            alpha = random.random()
            beta = random.random()

            res = torch.addmm(alpha, t, beta, x, y)
            expected = torch.addmm(alpha, t, beta, self.safeToDense(x), y)
            self.assertEqual(res, expected)

            res = torch.addmm(t, x, y)
            expected = torch.addmm(t, self.safeToDense(x), y)
            self.assertEqual(res, expected)

            res = torch.mm(x, y)
            expected = torch.mm(self.safeToDense(x), y)
            self.assertEqual(res, expected)

        test_shape(10, 100, 100, 20)
        test_shape(100, 1000, 200, 20)
        test_shape(64, 10000, 300, 20)
        test_shape(0, 100, 100, 0)
        test_shape(10, 0, 100, 0)
        test_shape(10, 100, 0, 0)
        test_shape(10, 100, 0, 20)

    @cpu_only
    def test_saddmm(self):
        def test_shape(di, dj, dk, nnz):
            x = self._gen_sparse(2, nnz, [di, dj])[0]
            t = self._gen_sparse(2, nnz, [di, dk])[0]
            y = torch.randn(dj, dk)
            alpha = random.random()
            beta = random.random()

            res = torch.saddmm(alpha, t, beta, x, y)
            expected = torch.addmm(alpha, self.safeToDense(t), beta, self.safeToDense(x), y)
            self.assertEqual(self.safeToDense(res), expected)

            res = torch.saddmm(t, x, y)
            expected = torch.addmm(self.safeToDense(t), self.safeToDense(x), y)
            self.assertEqual(self.safeToDense(res), expected)

            res = torch.smm(x, y)
            expected = torch.mm(self.safeToDense(x), y)
            self.assertEqual(self.safeToDense(res), expected)

        test_shape(7, 5, 3, 20)
        test_shape(1000, 100, 100, 20)
        test_shape(3000, 64, 300, 20)
        test_shape(0, 100, 100, 0)
        test_shape(1000, 0, 100, 0)
        test_shape(1000, 100, 0, 0)

    @skipIfRocm
    def test_dsmm(self):
        def test_shape(di, dj, dk, nnz):
            x = self._gen_sparse(2, nnz, [di, dj])[0]
            y = self.randn(dj, dk)

            res = torch.dsmm(x, y)
            expected = torch.mm(self.safeToDense(x), y)
            self.assertEqual(res, expected)

        test_shape(7, 5, 3, 20)
        test_shape(1000, 100, 100, 20)
        test_shape(3000, 64, 300, 20)
        test_shape(0, 100, 100, 0)
        test_shape(1000, 0, 100, 0)
        test_shape(1000, 100, 0, 0)
        test_shape(1000, 100, 0, 20)

    @skipIfRocm
    def test_hsmm(self):
        def test_shape(di, dj, dk, nnz):
            x = self._gen_sparse(2, nnz, [di, dj])[0]
            y = self.randn(dj, dk)

            res = torch.hsmm(x, y)
            expected = torch.mm(self.safeToDense(x), y)
            self.assertEqual(res.to_dense(), expected)

        test_shape(7, 5, 3, 20)
        test_shape(1000, 100, 100, 20)
        test_shape(3000, 64, 300, 20)
        test_shape(0, 100, 100, 0)
        test_shape(1000, 0, 100, 0)
        test_shape(1000, 100, 0, 0)
        test_shape(1000, 100, 0, 20)

    def _test_spadd_shape(self, nnz, shape_i, shape_v=None):
        shape = shape_i + (shape_v or [])
        x, _, _ = self._gen_sparse(len(shape_i), nnz, shape)
        y = self.randn(*shape)
        r = random.random()

        res = torch.add(y, r, x)
        expected = y + r * self.safeToDense(x)

        self.assertEqual(res, expected)

        # Non contiguous dense tensor
        s = list(shape)
        s[0] = shape[-1]
        s[-1] = shape[0]
        y = self.randn(*s)
        y.transpose_(0, len(s) - 1)
        r = random.random()

        res = torch.add(y, r, x)
        expected = y + r * self.safeToDense(x)

        self.assertEqual(res, expected)

        x, i, v = self._gen_sparse(len(shape_i), nnz, shape)
        nnz = i.size(1)

        # Non contiguous sparse indices tensor
        x_ = self.SparseTensor(i[:, ::2], v[:int(nnz / 2)], x.shape)
        res = torch.add(y, r, x_)
        expected = y + r * self.safeToDense(x_)
        self.assertEqual(res, expected)

        # Non contiguous sparse values tensor
        x_ = self.SparseTensor(i[:, :int(nnz / 2)], v[::2], x.shape)
        res = torch.add(y, r, x_)
        expected = y + r * self.safeToDense(x_)
        self.assertEqual(res, expected)

        # Non contiguous sparse indices and values tensors
        x_ = self.SparseTensor(i[:, 1::2], v[1::2], x.shape)
        res = torch.add(y, r, x_)
        expected = y + r * self.safeToDense(x_)
        self.assertEqual(res, expected)

    @skipIfRocm
    def test_spadd(self):
        self._test_spadd_shape(10, [5, 6])
        self._test_spadd_shape(10, [10, 10, 10])
        self._test_spadd_shape(10, [50, 30, 20])
        self._test_spadd_shape(10, [5, 5, 5, 5, 5, 5])
        self._test_spadd_shape(0, [0, 30, 20])
        self._test_spadd_shape(0, [50, 0, 20])
        self._test_spadd_shape(0, [50, 30, 0])

    @skipIfRocm
    def test_spadd_hybrid(self):
        self._test_spadd_shape(10, [5, 6], [2, 3])
        self._test_spadd_shape(10, [10, 10, 10], [3])
        self._test_spadd_shape(10, [50, 30, 20], [2])
        self._test_spadd_shape(10, [5, 5, 5, 5, 5, 5], [2])
        self._test_spadd_shape(0, [0, 30, 20], [2, 0])
        self._test_spadd_shape(0, [50, 0, 20], [2, 0])
        self._test_spadd_shape(0, [50, 30, 0], [2, 0])
        self._test_spadd_shape(10, [50, 30, 20], [2, 0])

    def test_norm(self):
        def test_shape(sparse_dims, nnz, with_size):
            x, _, _ = self._gen_sparse(sparse_dims, nnz, with_size)
            y = x.coalesce()
            self.assertEqual(x.norm(), y._values().norm())

        test_shape(3, 10, 100)
        test_shape(4, 10, [100, 100, 100, 5, 5, 5, 0])
        test_shape(4, 0, [0, 0, 100, 5, 5, 5, 0])

    def _test_basic_ops_shape(self, nnz_x1, nnz_x2, shape_i, shape_v=None):
        shape = shape_i + (shape_v or [])
        x1, _, _ = self._gen_sparse(len(shape_i), nnz_x1, shape)
        x2, _, _ = self._gen_sparse(len(shape_i), nnz_x2, shape)

        y1 = x1 + x2
        y2 = x1.clone()
        y2.add_(x2)
        expected = self.safeToDense(x1) + self.safeToDense(x2)
        self.assertEqual(self.safeToDense(y1), expected)
        self.assertEqual(self.safeToDense(y2), expected)

        y1 = x1 - x2
        y2 = x1.clone()
        y2.sub_(x2)
        expected = self.safeToDense(x1) - self.safeToDense(x2)
        self.assertEqual(self.safeToDense(y1), expected)
        self.assertEqual(self.safeToDense(y2), expected)

        y1 = x1 * x2
        y2 = x1.clone()
        y2.mul_(x2)
        expected = self.safeToDense(x1) * self.safeToDense(x2)
        self.assertEqual(self.safeToDense(y1), expected)
        self.assertEqual(self.safeToDense(y2), expected)

        y1 = x1 * 37.5
        y2 = x1.clone()
        y2.mul_(37.5)
        expected = self.safeToDense(x1) * 37.5
        self.assertEqual(self.safeToDense(y1), expected)
        self.assertEqual(self.safeToDense(y2), expected)

        y1 = x1 / 37.5
        y2 = x1.clone()
        y2.div_(37.5)
        expected = self.safeToDense(x1) / 37.5
        self.assertEqual(self.safeToDense(y1), expected)
        self.assertEqual(self.safeToDense(y2), expected)

        # TODO: add back inplace support
        y1 = x1 ** 2
        y2 = x1.clone()
        y2 = y2.pow(2)
        expected = self.safeToDense(x1) ** 2
        self.assertEqual(self.safeToDense(y1), expected)
        self.assertEqual(self.safeToDense(y2), expected)

        y = x1.clone()
        y.zero_()
        expected = torch.zeros(x1.size())
        self.assertEqual(self.safeToDense(y), expected)

        self.assertFalse(x1.is_coalesced())
        y = x1.coalesce()
        z = x1.coalesce()
        self.assertFalse(x1.is_coalesced())
        self.assertTrue(y.is_coalesced())
        self.assertEqual(x1, y)
        # check that coalesce is out of place
        y._values().add_(1)
        self.assertEqual(z._values() + 1, y._values())

    @skipIfRocm
    def test_basic_ops(self):
        self._test_basic_ops_shape(9, 12, [5, 6])
        self._test_basic_ops_shape(9, 12, [10, 10, 10])
        self._test_basic_ops_shape(9, 12, [50, 30, 20])
        self._test_basic_ops_shape(9, 12, [5, 5, 5, 5, 5, 5])
        self._test_basic_ops_shape(0, 12, [10, 10, 10])
        self._test_basic_ops_shape(9, 0, [10, 10, 10])
        self._test_basic_ops_shape(0, 0, [10, 10, 10])
        self._test_basic_ops_shape(0, 0, [10, 10, 0])

    @skipIfRocm
    def test_basic_ops_hybrid(self):
        self._test_basic_ops_shape(9, 12, [5, 6], [2, 3])
        self._test_basic_ops_shape(9, 12, [10, 10, 10], [3])
        self._test_basic_ops_shape(9, 12, [50, 30, 20], [2])
        self._test_basic_ops_shape(9, 12, [5, 5, 5, 5, 5, 5], [2])
        self._test_basic_ops_shape(0, 12, [10, 10, 10], [2])
        self._test_basic_ops_shape(9, 0, [10, 10, 10], [2])
        self._test_basic_ops_shape(0, 0, [10, 10, 10], [2])
        self._test_basic_ops_shape(9, 12, [10, 10, 10], [2, 0])
        self._test_basic_ops_shape(0, 12, [10, 10, 10], [2, 0])
        self._test_basic_ops_shape(9, 0, [10, 10, 10], [2, 0])
        self._test_basic_ops_shape(0, 0, [10, 10, 10], [2, 0])
        self._test_basic_ops_shape(0, 0, [10, 10, 0], [2, 0])

    def test_add_dense_sparse_mismatch(self):
        def test_shape(dense_size, sparse_dims_shape, dense_dims_shape, sparse_size):
            x = torch.zeros(dense_size, dtype=self.value_dtype, device=self.device)
            sparse_y = self.SparseTensor(torch.zeros(sparse_dims_shape, dtype=torch.int64, device=self.device),
                                         torch.randn(dense_dims_shape, dtype=self.value_dtype, device=self.device),
                                         torch.Size(sparse_size))
            with self.assertRaisesRegex(
                    RuntimeError,
                    "add: expected 'self' and 'other' to have same size"):
                x + sparse_y

        test_shape([3, 4], [1, 4], [4, 4, 4], [3, 4, 4])
        test_shape([3, 4, 0], [1, 4], [4, 4, 4, 0], [3, 4, 4, 0])

    def _test_sparse_mask_shape(self, nnz_x1, nnz_x2, shape_i, shape_v=None):
        shape = shape_i + (shape_v or [])
        x1, _, _ = self._gen_sparse(len(shape_i), nnz_x1, shape)
        x2, _, _ = self._gen_sparse(len(shape_i), nnz_x2, shape)

        y1 = x1 + x2
        y2 = x1.clone()
        y2.add_(x2)
        expected = self.safeToDense(x1) + self.safeToDense(x2)
        self.assertEqual(self.safeToDense(y1), expected)
        self.assertEqual(self.safeToDense(y2), expected)

    def _test_sparse_mask_fixed(self):
        i = self.IndexTensor([
            [1, 3, 0, 4],
            [2, 1, 2, 3],
        ])
        v = self.ValueTensor([1, 2, 3, 4])
        x = self.SparseTensor(i, v, torch.Size([5, 4])).coalesce()
        dense = self.ValueTensor([
            [1, 2, 3, 4],
            [5, 6, 7, 8],
            [9, 10, 11, 12],
            [13, 14, 15, 16],
            [17, 18, 19, 20],
        ])
        exp_v = self.ValueTensor([7, 14, 3, 20])
        res = dense.sparse_mask(x)
        expected = self.SparseTensor(i, exp_v, torch.Size([5, 4]))
        self.assertEqual(res, expected)

        i = self.IndexTensor([
            [1, 3, 0, 4],
            [2, 1, 2, 3],
        ])
        v = self.ValueTensor(4, 0)
        x = self.SparseTensor(i, v, torch.Size([5, 4, 0])).coalesce()
        dense = self.ValueTensor(5, 4, 0)
        exp_v = self.ValueTensor(4, 0)
        res = dense.sparse_mask(x)
        expected = self.SparseTensor(i, exp_v, torch.Size([5, 4, 0]))
        self.assertEqual(res, expected)

    @skipIfRocm
    def test_sparse_mask(self):
        self._test_sparse_mask_fixed()

        self._test_sparse_mask_shape(9, 12, [5, 6])
        self._test_sparse_mask_shape(9, 12, [10, 10, 10])
        self._test_sparse_mask_shape(9, 12, [50, 30, 20])
        self._test_sparse_mask_shape(9, 12, [5, 5, 5, 5, 5, 5])
        self._test_sparse_mask_shape(0, 12, [10, 10, 10])
        self._test_sparse_mask_shape(9, 0, [10, 10, 10])
        self._test_sparse_mask_shape(0, 0, [10, 10, 10])
        self._test_sparse_mask_shape(0, 0, [10, 10, 0])

    def _test_sparse_mask_hybrid_fixed(self):
        i = self.IndexTensor([
            [1, 3, 0, 4],
            [2, 1, 2, 3],
        ])
        v = self.ValueTensor([[1, 2], [2, 3], [3, 4], [4, 5]])
        # TODO: This is also testing that, if coalesce is a no-op,
        # the indices don't get permuted. I don't know if we actually
        # want to give this invariant.
        x = self.SparseTensor(i, v, torch.Size([5, 4, 2])).coalesce()
        dense = self.ValueTensor([
            [[1, 3], [2, 2], [3, 3], [4, 2]],
            [[5, 7], [6, 7], [7, 9], [8, 9]],
            [[9, 2], [10, 4], [11, 1], [12, 3]],
            [[13, 5], [14, 1], [15, 1], [16, 6]],
            [[17, 7], [18, 2], [19, 7], [20, 1]],
        ])
        res = dense.sparse_mask(x)
        exp_v = self.ValueTensor([[7, 9], [14, 1], [3, 3], [20, 1]])
        expected = self.SparseTensor(i, exp_v, torch.Size([5, 4, 2]))
        self.assertEqual(res, expected)

        i = self.IndexTensor([
            [1, 3, 0, 4],
            [2, 1, 2, 3],
        ])
        v = self.ValueTensor(4, 2, 0)
        x = self.SparseTensor(i, v, torch.Size([5, 4, 2, 0])).coalesce()
        dense = self.ValueTensor(5, 4, 2, 0)
        res = dense.sparse_mask(x)
        exp_v = self.ValueTensor(4, 2, 0)
        expected = self.SparseTensor(i, exp_v, torch.Size([5, 4, 2, 0]))
        self.assertEqual(res, expected)

    @skipIfRocm
    def test_sparse_mask_hybrid(self):
        self._test_sparse_mask_hybrid_fixed()

        self._test_sparse_mask_shape(9, 12, [5, 6], [2, 3])
        self._test_sparse_mask_shape(9, 12, [10, 10, 10], [3])
        self._test_sparse_mask_shape(9, 12, [50, 30, 20], [2])
        self._test_sparse_mask_shape(9, 12, [5, 5, 5, 5, 5, 5], [2])
        self._test_sparse_mask_shape(0, 12, [10, 10, 10], [2])
        self._test_sparse_mask_shape(9, 0, [10, 10, 10], [2])
        self._test_sparse_mask_shape(0, 0, [10, 10, 10], [2])
        self._test_sparse_mask_shape(9, 12, [10, 10, 10], [2, 0])
        self._test_sparse_mask_shape(0, 12, [10, 10, 10], [2, 0])
        self._test_sparse_mask_shape(9, 0, [10, 10, 10], [2, 0])
        self._test_sparse_mask_shape(0, 0, [10, 10, 10], [2, 0])
        self._test_sparse_mask_shape(0, 0, [10, 10, 0], [2, 0])

    def _test_zeros(self, nnzs, shape, out_shape_i, out_shape_v=None):
        out_shape = out_shape_i + (out_shape_v or [])
        for nnz in nnzs:
            out, _, _ = self._gen_sparse(len(out_shape_i), nnz, out_shape)
            torch.zeros(*shape, out=out)
            self.assertEqual(tuple(out.size()), tuple(shape))
            self.assertTrue(out._indices().numel() == out._values().numel() == 0)
            self.assertEqual(out._nnz(), 0)
            self.assertEqual(out._sparseDims(), len(shape))
            self.assertEqual(out._denseDims(), 0)

    def test_zeros(self):
        def test_shape(i_shapes, v_shapes, shape, nnzs):
            for i_dim in range(1, len(i_shapes) + 1):
                for v_dim in range(len(v_shapes) + 1):
                    self._test_zeros(nnzs, shape, i_shapes[:i_dim], v_shapes[:v_dim])
        test_shape([2, 3, 4], [3, 4, 5, 6], [2, 3, 4], [9, 12])
        test_shape([0, 3, 4], [3, 4, 5, 6], [2, 3, 4], [0])
        test_shape([2, 3, 4], [0, 4, 5, 6], [2, 3, 4], [9, 12])
        test_shape([2, 3, 4], [3, 4, 5, 6], [2, 3, 0], [9, 12])
        test_shape([0, 3, 4], [3, 4, 5, 6], [2, 3, 0], [0])
        test_shape([2, 3, 4], [0, 4, 5, 6], [2, 3, 0], [9, 12])

    def _test_zeros_like(self, nnzs, template_shape_i, template_shape_v=None):
        template_shape_v = template_shape_v or []
        template_shape = template_shape_i + template_shape_v
        for nnz in nnzs:
            t, _, _ = self._gen_sparse(len(template_shape_i), nnz, template_shape)
            res = torch.zeros_like(t)
            self.assertEqual(tuple(res.size()), tuple(template_shape))
            self.assertTrue(res._indices().numel() == res._values().numel() == 0)
            self.assertEqual(res._nnz(), 0)
            self.assertEqual(res._sparseDims(), len(template_shape_i))
            self.assertEqual(res._denseDims(), len(template_shape_v))

    def test_zeros_like(self):
        def test_shape(i_shapes, v_shapes, nnzs):
            for i_dim in range(1, len(i_shapes) + 1):
                for v_dim in range(len(v_shapes) + 1):
                    self._test_zeros_like(nnzs, i_shapes[:i_dim], v_shapes[:v_dim])
        test_shape([2, 3, 4], [3, 4, 5, 6], [9, 12])
        test_shape([0, 3, 4], [3, 4, 5, 6], [0])
        test_shape([2, 3, 4], [0, 4, 5, 6], [9, 12])
        test_shape([2, 3, 4], [3, 4, 5, 6], [9, 12])
        test_shape([0, 3, 4], [3, 4, 5, 6], [0])
        test_shape([2, 3, 4], [0, 4, 5, 6], [9, 12])

    def _test_narrow(self, input, narrow_args):
        expected = input.to_dense().narrow(*narrow_args)
        self.assertEqual(expected, input.narrow_copy(*narrow_args).to_dense())

    def _all_narrow_combs(self, shape):
        for dim, dim_sz in enumerate(shape):
            for start in range(dim_sz):
                for length in range(dim_sz - start):
                    yield [dim, start, length]

    @skipIfRocm
    def test_narrow(self):
        shape = [3, 3, 4, 2]
        input, _, _ = self._gen_sparse(4, 19, shape)
        for narrow_args in self._all_narrow_combs(shape):
            self._test_narrow(input, narrow_args)

        self.assertRaises(RuntimeError, lambda: input.narrow_copy(-1, 0, 3))  # dim < 0
        self.assertRaises(RuntimeError, lambda: input.narrow_copy(10, 0, 3))  # dim > input.dim()
        self.assertRaises(RuntimeError, lambda: input.narrow_copy(0, shape[0] + 1, 3))  # start > size of dim
        self.assertRaises(RuntimeError, lambda: input.narrow_copy(0, 2, shape[0]))  # start+length > size of dim

        with_dense, _, _ = self._gen_sparse(2, 7, shape)
        for narrow_args in self._all_narrow_combs(shape):
            self._test_narrow(with_dense, narrow_args)

        self.assertRaises(RuntimeError, lambda: with_dense.narrow_copy(10, 0, 3))  # dim > sparseDim + denseDim

    def _test_log1p_tensor(self, input, dense_tensor):
        expected_output = torch.tensor(dense_tensor).log1p_()
        self.assertEqual(expected_output, input.log1p().to_dense())
        self.assertEqual(expected_output, input.coalesce().log1p_().to_dense())

        # test in-place op on uncoalesced input
        with self.assertRaisesRegex(RuntimeError, "in-place on uncoalesced tensors is not supported yet"):
            input.log1p_()

        input.requires_grad_()
        self.assertTrue(input.requires_grad)

        # test autograd
        x = input.clone()
        y = input.log1p()
        with self.assertRaisesRegex(RuntimeError, "log1p of a sparse tensor is made to be non-differentiable"):
            y.backward(x)

    @skipIfRocm
    def test_log1p(self):
        input = torch.sparse_coo_tensor(
            torch.LongTensor([[0], [1], [2]]).transpose(1, 0).clone().detach(),
            torch.FloatTensor([3, 4, 5]),
            torch.Size([3]),
            device=self.device)
        self._test_log1p_tensor(input, [3., 4., 5.])

        # test uncoalesced input
        input_uncoalesced = torch.sparse_coo_tensor(
            torch.LongTensor([[0], [1], [2], [0], [1], [2]]).transpose(1, 0).clone().detach(),
            torch.FloatTensor([2, 3, 4, 1, 1, 1]),
            torch.Size([3]),
            device=self.device)
        self._test_log1p_tensor(input_uncoalesced, [3., 4., 5.])

        input = torch.sparse_coo_tensor(
            torch.zeros([2, 0]),
            torch.zeros([0, 5, 5, 5, 5, 5, 5, 0]),
            torch.Size([0, 0, 5, 5, 5, 5, 5, 5, 0]),
            device=self.device)
        self._test_log1p_tensor(input, torch.zeros([0, 0, 5, 5, 5, 5, 5, 5, 0]))

        input = torch.sparse_coo_tensor(
            torch.zeros([1, 5]),
            torch.zeros([5, 6, 0]),
            torch.Size([5, 6, 0]),
            device=self.device)
        self._test_log1p_tensor(input, torch.zeros([5, 6, 0]))

    @skipIfRocm
    def test_sparse_add_coalesce(self):
        i = self.IndexTensor([[1, 2, 1]])
        v = self.ValueTensor([3, 4, 5])
        x = self.SparseTensor(i, v, torch.Size([3]))
        y = self.SparseTensor(i, v, torch.Size([3]))
        z = x + y

        self.assertFalse(z._indices().numel() != 2 and z.is_coalesced())

        i = self.IndexTensor([[1, 2, 1]])
        v = self.ValueTensor(3, 0)
        x = self.SparseTensor(i, v, torch.Size([3, 0]))
        y = self.SparseTensor(i, v, torch.Size([3, 0]))
        z = x + y

        self.assertFalse(z._indices().numel() != 2 and z.is_coalesced())

    @cuda_only
    def test_storage_not_null(self):
        x = torch.cuda.sparse.FloatTensor(2)
        self.assertNotEqual(x.get_device(), -1)

        x = torch.cuda.sparse.FloatTensor(2, 0)
        self.assertNotEqual(x.get_device(), -1)

    @cuda_only
    @unittest.skipIf(torch.cuda.device_count() < 2, "only one GPU detected")
    def test_same_gpu(self):
        def check_device(x, device_id):
            self.assertEqual(x.get_device(), device_id)
            self.assertEqual(x._values().get_device(), device_id)
            self.assertEqual(x._indices().get_device(), device_id)

        i = self.IndexTensor([[2]]).cuda(1)
        v = self.ValueTensor([5]).cuda(1)
        x = self.SparseTensor(i, v, torch.Size([3]), device=1)
        check_device(x, 1)

        i = self.IndexTensor([[2]]).cuda(1)
        v = self.ValueTensor(1, 0).cuda(1)
        x = self.SparseTensor(i, v, torch.Size([3, 0]), device=1)
        check_device(x, 1)

        x = self.SparseTensor(3, device=1)
        check_device(x, 1)

        x = self.SparseTensor(3, 0, device=1)
        check_device(x, 1)

        i = self.IndexTensor([[2]]).cuda(1)
        v = self.ValueTensor([5]).cuda(0)
        self.assertRaises(RuntimeError, lambda: self.SparseTensor(i, v, torch.Size([3])))

        i = self.IndexTensor([[2]]).cuda(1)
        v = self.ValueTensor(1, 0).cuda(0)
        self.assertRaises(RuntimeError, lambda: self.SparseTensor(i, v, torch.Size([3, 0])))

    def _test_new_device(self, size, device):
        with torch.cuda.device(device):
            x = torch.cuda.sparse.DoubleTensor(*size)
        self.assertEqual(x.get_device(), device)
        x1 = x.new()
        x2 = x.new(2, 3)
        self.assertEqual(x1.get_device(), device)
        self.assertEqual(x2.get_device(), device)

    @cuda_only
    def test_new_device_single_gpu(self):
        self._test_new_device((), 0)
        self._test_new_device((30, 20), 0)
        self._test_new_device((30, 20, 10), 0)
        self._test_new_device((30, 20, 10, 0), 0)

    @cuda_only
    @unittest.skipIf(torch.cuda.device_count() < 2, "only one GPU detected")
    def test_new_device_multi_gpu(self):
        self._test_new_device((), 1)
        self._test_new_device((30, 20), 1)
        self._test_new_device((30, 20, 10), 1)
        self._test_new_device((30, 20, 10, 0), 1)

    @skipIfRocm
    def test_new(self):
        def test_shape(sparse_dims, nnz, with_size):
            x, indices, values = self._gen_sparse(sparse_dims, nnz, with_size)
            if not x.is_cuda:
                # CUDA sparse tensors currently requires the size to be
                # specified if nDimV > 0
                self.assertEqual(x.new(indices, values), x)
            self.assertEqual(x.new(indices, values, x.size()), x)

        test_shape(3, 10, 100)
        test_shape(3, 0, [100, 100, 0])

    @cpu_only  # not really, but we only really want to run this once
    def test_factory(self):
        for test_empty_tensor in [True, False]:
            if test_empty_tensor:
                default_size = torch.Size([1, 3, 0])
                size = torch.Size([3, 3, 0])
            else:
                default_size = torch.Size([1, 3])
                size = torch.Size([3, 3])
            for include_size in [True, False]:
                for use_tensor_idx in [True, False]:
                    for use_tensor_val in [True, False]:
                        for use_cuda in ([False] if not torch.cuda.is_available() else [True, False]):
                            # have to include size with cuda sparse tensors
                            include_size = include_size or use_cuda
                            dtype = torch.float64
                            long_dtype = torch.int64
                            device = torch.device('cpu') if not use_cuda else \
                                torch.device(torch.cuda.device_count() - 1)
                            indices = torch.tensor(([0], [2]), dtype=long_dtype) if use_tensor_idx else ([0], [2])
                            if test_empty_tensor:
                                values = self.ValueTensor(1, 0)
                            else:
                                if use_tensor_val:
                                    values = torch.tensor([1.], dtype=dtype)
                                else:
                                    values = 1.
                            if include_size:
                                sparse_tensor = torch.sparse_coo_tensor(indices, values, size, dtype=dtype,
                                                                        device=device, requires_grad=True)
                            else:
                                sparse_tensor = torch.sparse_coo_tensor(indices, values, dtype=dtype,
                                                                        device=device, requires_grad=True)
                            self.assertEqual(indices, sparse_tensor._indices())
                            self.assertEqual(values, sparse_tensor._values())
                            self.assertEqual(size if include_size else default_size, sparse_tensor.size())
                            self.assertEqual(dtype, sparse_tensor.dtype)
                            if use_cuda:
                                self.assertEqual(device, sparse_tensor._values().device)
                            self.assertEqual(True, sparse_tensor.requires_grad)

    def test_factory_size_check(self):
        indices = self.IndexTensor([[1, 2],
                                    [0, 2]])
        values = self.ValueTensor([.5, .5])
        sizes = torch.Size([2, 3])
        with self.assertRaisesRegex(RuntimeError, "sizes is inconsistent with indices"):
            torch.sparse_coo_tensor(indices, values, sizes)

        indices.fill_(-1)
        with self.assertRaisesRegex(RuntimeError, "found negative index"):
            torch.sparse_coo_tensor(indices, values, sizes)

        indices = self.IndexTensor([[1, 2],
                                    [0, 2]])
        values = self.ValueTensor(2, 1, 0)
        sizes = torch.Size([2, 3, 1, 0])
        with self.assertRaisesRegex(RuntimeError, "sizes is inconsistent with indices"):
            torch.sparse_coo_tensor(indices, values, sizes)

        indices = self.IndexTensor([[1, 2],
                                    [0, 2]])
        values = self.ValueTensor(2, 2, 2)
        sizes = torch.Size([0, 0, 2, 2])
        with self.assertRaisesRegex(RuntimeError, "sizes is inconsistent with indices"):
            torch.sparse_coo_tensor(indices, values, sizes)

        indices = self.IndexTensor([[1, 2],
                                    [0, 2]])
        values = self.ValueTensor([[1, 1, 1], [1, 1, 1]])
        sizes = torch.Size([3, 3, 2])
        with self.assertRaisesRegex(RuntimeError, "values has incorrect size"):
            torch.sparse_coo_tensor(indices, values, sizes)

        indices = self.IndexTensor([[1, 2],
                                    [0, 2]])
        values = self.ValueTensor(2, 1, 0)
        sizes = torch.Size([3, 3, 2, 0])
        with self.assertRaisesRegex(RuntimeError, "values has incorrect size"):
            torch.sparse_coo_tensor(indices, values, sizes)

    def test_factory_default(self):
        tensor = self.SparseTensor()
        expected_indices = self.IndexTensor(1, 0)
        expected_size = torch.Size([0])
        self.assertEqual(tensor._indices(), expected_indices)
        self.assertEqual(tensor.shape, expected_size)

    def test_factory_empty_indices(self):
        device = 'cuda' if self.is_cuda else 'cpu'
        tensor = self.SparseTensor()
        expected_indices = torch.empty((1, 0), dtype=torch.long, device=device)
        self.assertEqual(tensor._indices(), expected_indices)

        tensor = torch.sparse_coo_tensor(torch.Size([2, 0]), device=device)
        expected_indices = torch.empty((2, 0), dtype=torch.long, device=device)
        self.assertEqual(tensor._indices(), expected_indices)

        tensor = torch.sparse_coo_tensor(torch.Size([2, 2, 0]), device=device)
        expected_indices = torch.empty((3, 0), dtype=torch.long, device=device)
        self.assertEqual(tensor._indices(), expected_indices)

        tensor = torch.sparse_coo_tensor(torch.Size([2, 2, 0, 0]), device=device)
        expected_indices = torch.empty((4, 0), dtype=torch.long, device=device)
        self.assertEqual(tensor._indices(), expected_indices)

    def test_factory_nnz(self):
        indices = self.IndexTensor([[0]])  # (sparseDims, nnz): (1, 1)
        values = self.ValueTensor([[1, 1], [1, 1]])  # (nnz, ...): (2, 2)
        sizes = torch.Size([2, 2])
        with self.assertRaisesRegex(RuntimeError, "indices and values must have same nnz"):
            torch.sparse_coo_tensor(indices, values, sizes)

        indices = self.IndexTensor([[0]])  # (sparseDims, nnz): (1, 1)
        values = self.ValueTensor(2, 0)  # (nnz, ...): (2, 0)
        sizes = torch.Size([2, 0])
        with self.assertRaisesRegex(RuntimeError, "indices and values must have same nnz"):
            torch.sparse_coo_tensor(indices, values, sizes)

    def test_factory_nnz_zero(self):
        def test_shape(i_shape, v_shape, size, expected_size):
            device = 'cuda' if self.is_cuda else 'cpu'
            if size:
                t = torch.sparse_coo_tensor(torch.empty(i_shape), torch.empty(v_shape), torch.Size(size), device=device)
            else:
                t = torch.sparse_coo_tensor(torch.empty(i_shape), torch.empty(v_shape), device=device)
            expected_indices = torch.empty(i_shape, device=device)
            expected_values = torch.empty(v_shape, device=device)
            expected_size = torch.Size(expected_size)
            self.assertEqual(t._indices(), expected_indices)
            self.assertEqual(t._values(), expected_values)
            self.assertEqual(t.size(), expected_size)

        test_shape([1, 0], [0, 2, 4, 0], None, [0, 2, 4, 0])
        test_shape([3, 0], [0, 2, 4, 0], None, [0, 0, 0, 2, 4, 0])
        test_shape([1, 0], [0, 2, 4, 0], [0, 2, 4, 0], [0, 2, 4, 0])
        test_shape([3, 0], [0, 2, 4, 0], [0, 0, 0, 2, 4, 0], [0, 0, 0, 2, 4, 0])
        test_shape([3, 0], [0, 2, 4, 0], [1, 2, 3, 2, 4, 0], [1, 2, 3, 2, 4, 0])

    def test_factory_dense_dims(self):
        indices = self.IndexTensor([[0]])
        values = self.ValueTensor([[[1, 1, 1], [1, 1, 1]]])
        sizes = torch.Size([1, 3, 4])
        with self.assertRaisesRegex(RuntimeError, "values has incorrect size"):
            torch.sparse_coo_tensor(indices, values, sizes)

        indices = self.IndexTensor([[0]])
        values = self.ValueTensor(1, 2, 3, 0)
        sizes = torch.Size([1, 3, 4, 0])
        with self.assertRaisesRegex(RuntimeError, "values has incorrect size"):
            torch.sparse_coo_tensor(indices, values, sizes)

    @cpu_only
    def test_factory_type_inference(self):
        t = torch.sparse_coo_tensor(torch.tensor(([0], [2])), torch.tensor([1.], dtype=torch.float32))
        self.assertEqual(torch.float32, t.dtype)
        t = torch.sparse_coo_tensor(torch.tensor(([0], [2])), torch.tensor([1.], dtype=torch.float64))
        self.assertEqual(torch.float64, t.dtype)
        t = torch.sparse_coo_tensor(torch.tensor(([0], [2])), torch.tensor([1]))
        self.assertEqual(torch.int64, t.dtype)

        t = torch.sparse_coo_tensor(torch.tensor(([0], [2])), torch.FloatTensor(1, 0))
        self.assertEqual(torch.float32, t.dtype)
        t = torch.sparse_coo_tensor(torch.tensor(([0], [2])), torch.DoubleTensor(1, 0))
        self.assertEqual(torch.float64, t.dtype)
        t = torch.sparse_coo_tensor(torch.tensor(([0], [2])), torch.LongTensor(1, 0))
        self.assertEqual(torch.int64, t.dtype)

    @cuda_only
    def test_factory_device_type_inference(self):
        # both indices/values are CUDA
        shape = (1, 3)
        for indices_device in ['cuda', 'cpu']:
            for values_device in ['cuda', 'cpu']:
                for sparse_device in ['cuda', 'cpu', None]:
                    for test_empty_tensor in [True, False]:
                        if test_empty_tensor:
                            t = torch.sparse_coo_tensor(torch.tensor(([0], [2]), device=indices_device),
                                                        self.ValueTensor(1, 0).to(values_device),
                                                        (1, 3, 0), device=sparse_device)
                        else:
                            t = torch.sparse_coo_tensor(torch.tensor(([0], [2]), device=indices_device),
                                                        torch.tensor([1.], device=values_device),
                                                        (1, 3), device=sparse_device)
                        should_be_cuda = sparse_device == 'cuda' or (sparse_device is None and values_device == 'cuda')
                        self.assertEqual(should_be_cuda, t.is_cuda)

    @cpu_only
    def test_factory_copy(self):
        def test_tensor(indices, values, indices_equal, values_equal):
            sparse_tensor = torch.sparse_coo_tensor(indices, values, dtype=torch.float64)
            if indices_equal:
                self.assertEqual(indices.data_ptr(), sparse_tensor._indices().data_ptr())
            else:
                self.assertNotEqual(indices.data_ptr(), sparse_tensor._indices().data_ptr())
            if values_equal:
                self.assertEqual(values.data_ptr(), sparse_tensor._values().data_ptr())
            else:
                self.assertNotEqual(values.data_ptr(), sparse_tensor._values().data_ptr())

        # both correct
        indices = torch.tensor(([0], [2]), dtype=torch.int64)
        values = torch.tensor([1.], dtype=torch.float64)
        test_tensor(indices, values, True, True)

        indices = torch.tensor(([0], [2]), dtype=torch.int64)
        values = torch.DoubleTensor(1, 0)
        test_tensor(indices, values, True, True)

        # only indices correct
        indices = torch.tensor(([0], [2]), dtype=torch.int64)
        values = torch.tensor([1.], dtype=torch.float32)
        test_tensor(indices, values, True, False)

        indices = torch.tensor(([0], [2]), dtype=torch.int64)
        values = torch.FloatTensor(1, 0)
        test_tensor(indices, values, True, True)  # An empty tensor's data_ptr is always equal to 0

        # only values correct
        indices = torch.tensor(([0], [2]), dtype=torch.int32)
        values = torch.tensor([1.], dtype=torch.float64)
        test_tensor(indices, values, False, True)

        indices = torch.tensor(([0], [2]), dtype=torch.int32)
        values = torch.DoubleTensor(1, 0)
        test_tensor(indices, values, False, True)

        # neither correct
        indices = torch.tensor(([0], [2]), dtype=torch.int32)
        values = torch.tensor([1.], dtype=torch.float32)
        test_tensor(indices, values, False, False)

        indices = torch.tensor(([0], [2]), dtype=torch.int32)
        values = torch.FloatTensor(1, 0)
        test_tensor(indices, values, False, True)  # An empty tensor's data_ptr is always equal to 0

    @cpu_only  # just run once, we test both cpu and cuda
    def test_constructor_device_legacy(self):
        i = torch.tensor([[0, 1, 1], [2, 0, 2]])
        v = torch.tensor([3., 4., 5.])
        size = torch.Size([2, 3])

        self.assertRaises(RuntimeError, lambda: torch.sparse.FloatTensor(device='cuda'))
        self.assertRaises(RuntimeError, lambda: torch.sparse.FloatTensor(i, v, device='cuda'))
        self.assertRaises(RuntimeError, lambda: torch.sparse.FloatTensor(i, v, size, device='cuda'))
        self.assertRaises(RuntimeError, lambda: torch.sparse.FloatTensor(torch.Size([2, 3, 4]), device='cuda'))

        x = torch.sparse_coo_tensor(i, v, size, device='cpu')
        self.assertRaises(RuntimeError, lambda: x.new(device='cuda'))
        self.assertRaises(RuntimeError, lambda: x.new(i, v, device='cuda'))
        self.assertRaises(RuntimeError, lambda: x.new(i, v, size, device='cuda'))
        self.assertRaises(RuntimeError, lambda: x.new(torch.Size([2, 3, 4]), device='cuda'))

        if torch.cuda.is_available():
            self.assertRaises(RuntimeError, lambda: torch.cuda.sparse.FloatTensor(device='cpu'))
            self.assertRaises(RuntimeError, lambda: torch.cuda.sparse.FloatTensor(i, v, device='cpu'))
            self.assertRaises(RuntimeError, lambda: torch.cuda.sparse.FloatTensor(i, v, size, device='cpu'))
            self.assertRaises(RuntimeError, lambda: torch.cuda.sparse.FloatTensor(torch.Size([2, 3, 4]), device='cpu'))

            x = torch.sparse_coo_tensor(i, v, size, device='cuda')
            self.assertRaises(RuntimeError, lambda: x.new(device='cpu'))
            self.assertRaises(RuntimeError, lambda: x.new(i, v, device='cpu'))
            self.assertRaises(RuntimeError, lambda: x.new(i, v, size, device='cpu'))
            self.assertRaises(RuntimeError, lambda: x.new(torch.Size([2, 3, 4]), device='cpu'))

    @cpu_only  # not really, but we only really want to run this once
    def test_dtypes(self):
        all_sparse_dtypes = [dtype for dtype in torch.testing.get_all_dtypes() if dtype != torch.float16]
        TestTorch._test_dtypes(self, all_sparse_dtypes, torch.sparse_coo, torch.device('cpu'))
        if torch.cuda.is_available():
            TestTorch._test_dtypes(self, all_sparse_dtypes, torch.sparse_coo, torch.device('cuda:0'))

    @cpu_only  # not really, but we only really want to run this once
    def test_empty_full(self):
        all_sparse_dtypes = [dtype for dtype in torch.testing.get_all_dtypes() if dtype != torch.float16]
        TestTorch._test_empty_full(self, all_sparse_dtypes, torch.sparse_coo, torch.device('cpu'))
        if torch.cuda.device_count() > 0:
            TestTorch._test_empty_full(self, all_sparse_dtypes, torch.sparse_coo, None)
            TestTorch._test_empty_full(self, all_sparse_dtypes, torch.sparse_coo, torch.device('cuda:0'))

    def test_is_sparse(self):
        x = torch.randn(3, 3)
        self.assertFalse(x.is_sparse)

        x = torch.randn(3, 3, 0)
        self.assertFalse(x.is_sparse)

        x = self.SparseTensor()
        self.assertTrue(x.is_sparse)

        x = self.SparseTensor(1, 0)
        self.assertTrue(x.is_sparse)

    @skipIfRocm
    def test_resize_as(self):
        def do_test(t):
            y = t.new().resize_as_(t).zero_()
            self.assertEqual(y.shape, t.shape)
            # Check that y can be added to t. Currently, this requires that
            # _sparseDims and _denseDims match.
            self.assertEqual(t, t + y)

        do_test(self.SparseTensor())
        do_test(self.SparseTensor(3, 0))
        do_test(self.SparseTensor(3, 3))

    def _test_resize_shape(self, x_i, x_v, x_size, y_i, y_v, y_size):
        x_v_numel = torch.zeros(x_v).numel()
        y_v_numel = torch.zeros(y_v).numel()
        x = torch.sparse_coo_tensor(torch.zeros(x_i),
                                    torch.arange(x_v_numel).resize_(x_v).to(torch.float),
                                    torch.Size(x_size))
        x_dense = x.to_dense()
        y = torch.sparse_coo_tensor(torch.zeros(y_i),
                                    torch.ones(y_v).to(torch.float),
                                    torch.Size(y_size))
        y_dense = y.to_dense()
        x.resize_as_(y)
        x_dense.resize_as_(y_dense)
        self.assertEqual(x.shape, y.shape)
        self.assertEqual(x._sparseDims(), y._sparseDims())
        self.assertEqual(x._denseDims(), y._denseDims())
        self.assertEqual(x.shape, x_dense.shape)
        self.assertEqual(y.shape, y_dense.shape)
        # Here we make sure that the original data are preserved after resizing
        self.assertEqual(x.to_dense().view(-1)[0:x_v_numel].view(x_v),
                         x_dense.view(-1)[0:x_v_numel].view(x_v))

    def test_resize(self):
        # 1. Expand the size of some dense dimensions [Supported]
        self._test_resize_shape([1, 1], [1, 2, 3], [2, 2, 3],
                                [1, 1], [1, 2, 4], [2, 2, 4])

        self._test_resize_shape([1, 1], [1, 2, 0], [2, 2, 0],
                                [1, 1], [1, 2, 4], [2, 2, 4])

        # 2. Expand the size of some sparse dimensions [Supported]
        self._test_resize_shape([1, 1], [1, 2, 3], [2, 2, 3],
                                [1, 1], [1, 2, 3], [4, 2, 3])

        # 3. Change the shapes of both sparse and dense dimensions when nnz is zero [Supported]
        self._test_resize_shape([1, 0], [0, 2, 3], [2, 2, 3],
                                [2, 0], [0, 2, 4, 5], [1, 1, 2, 4, 5])

        self._test_resize_shape([1, 0], [0, 2, 3], [2, 2, 3],
                                [2, 0], [0, 2, 4, 0], [1, 1, 2, 4, 0])

        # 4. Add dims to dense dimensions [Not Supported]
        with self.assertRaisesRegex(RuntimeError, "changing the number of dense dimensions"):
            self._test_resize_shape([1, 1], [1, 2, 3], [2, 2, 3],
                                    [1, 1], [1, 2, 3, 4], [2, 2, 3, 4])

        with self.assertRaisesRegex(RuntimeError, "changing the number of dense dimensions"):
            self._test_resize_shape([1, 1], [1, 2, 3], [2, 2, 3],
                                    [1, 1], [1, 2, 3, 0], [2, 2, 3, 0])

        # 5. Remove dims from dense dimensions [Not Supported]
        with self.assertRaisesRegex(RuntimeError, "changing the number of dense dimensions"):
            self._test_resize_shape([1, 1], [1, 2, 3], [2, 2, 3],
                                    [1, 1], [1, 2], [2, 2])

        # 6. Change the number of sparse dimensions on a non-empty sparse tensor [Not Supported]
        with self.assertRaisesRegex(RuntimeError, "changing the number of sparse dimensions"):
            self._test_resize_shape([1, 1], [1, 2, 3], [2, 2, 3],
                                    [2, 1], [1, 2, 3], [1, 2, 2, 3])

        # 7. Shrink the size of some sparse dimensions on a non-empty sparse tensor [Not Supported]
        with self.assertRaisesRegex(RuntimeError, "shrinking the size of sparse dimensions"):
            self._test_resize_shape([1, 1], [1, 2, 3], [2, 2, 3],
                                    [1, 1], [1, 2, 3], [1, 2, 3])

        # 8. Shrink the size of some dense dimensions on a non-empty sparse tensor [Not Supported]
        with self.assertRaisesRegex(RuntimeError, "shrinking the size of dense dimensions"):
            self._test_resize_shape([1, 1], [1, 2, 3], [2, 2, 3],
                                    [1, 1], [1, 2, 2], [2, 2, 2])

        with self.assertRaisesRegex(RuntimeError, "shrinking the size of dense dimensions"):
            self._test_resize_shape([1, 1], [1, 2, 3], [2, 2, 3],
                                    [1, 1], [1, 2, 0], [2, 2, 0])

    def test_is_nonzero(self):
        self.assertTrue(torch.sparse_coo_tensor(([0],), 1., (1,)).is_nonzero())
        self.assertFalse(torch.sparse_coo_tensor(([0],), 0., (1,)).is_nonzero())
        self.assertFalse(torch.sparse_coo_tensor(([0], [0]), 0., (1, 1)).is_nonzero())
        self.assertFalse(torch.sparse_coo_tensor(([0, 0],), (0., 0.), (1,)).is_nonzero())
        self.assertFalse(torch.sparse_coo_tensor(([0, 0],), (-1., 1.), (1,)).is_nonzero())
        self.assertTrue(torch.sparse_coo_tensor(torch.zeros(0, 1), 12.3, []).is_nonzero())  # scalar sparse tensor
        with self.assertRaisesRegex(RuntimeError, "bool value of Tensor with no values is ambiguous"):
            torch.sparse_coo_tensor(([0, 1],), self.ValueTensor(2, 0), (4, 0)).is_nonzero()


class TestUncoalescedSparse(TestSparse):
    def setUp(self):
        super(TestUncoalescedSparse, self).setUp()
        self.is_uncoalesced = True


@unittest.skipIf(not TEST_CUDA, 'CUDA not available')
class TestCudaSparse(TestSparse):
    def setUp(self):
        super(TestCudaSparse, self).setUp()
        self.is_cuda = True
        self.device = 'cuda'
        self.IndexTensor = torch.cuda.LongTensor
        self.ValueTensor = torch.cuda.DoubleTensor
        self.SparseTensor = torch.cuda.sparse.DoubleTensor


@unittest.skipIf(not TEST_CUDA, 'CUDA not available')
class TestCudaUncoalescedSparse(TestCudaSparse):
    def setUp(self):
        super(TestCudaUncoalescedSparse, self).setUp()
        self.is_uncoalesced = True


class TestSparseOneOff(TestCase):
    @unittest.skipIf(not TEST_CUDA, 'CUDA not available')
    def test_cuda_from_cpu(self):
        with self.assertRaisesRegex(
                RuntimeError,
                "backend of indices \\(CUDA\\) must match backend of values \\(CPU\\)"):
            torch.sparse.FloatTensor(torch.zeros(1, 4).long().cuda(),
                                     torch.randn(4, 4, 4),
                                     [3, 4, 4])

        with self.assertRaisesRegex(
                RuntimeError,
                "backend of indices \\(CUDA\\) must match backend of values \\(CPU\\)"):
            torch.sparse.FloatTensor(torch.zeros(1, 4).long().cuda(),
                                     torch.randn(4, 4, 4, 0),
                                     [3, 4, 4, 0])

        with self.assertRaisesRegex(
                RuntimeError,
                "backend of indices \\(CUDA\\) must match backend of values \\(CPU\\)"):
            torch.sparse.FloatTensor(torch.LongTensor(1, 0).cuda(),
                                     torch.randn(0, 4, 4, 0),
                                     [0, 4, 4, 0])

    @unittest.skipIf(not TEST_CUDA, 'CUDA not available')
    def test_cuda_sparse_cpu_dense_add(self):
        x = torch.zeros(3, 4, 4)
        sparse_y = torch.cuda.sparse.FloatTensor(torch.zeros(1, 4).long().cuda(),
                                                 torch.randn(4, 4, 4).cuda(),
                                                 [3, 4, 4])
        with self.assertRaisesRegex(RuntimeError, "add: expected 'other' to be a CPU tensor\\, but got a CUDA tensor"):
            x + sparse_y

        x = torch.zeros(3, 4, 4, 0)
        sparse_y = torch.cuda.sparse.FloatTensor(torch.zeros(1, 4).long().cuda(),
                                                 torch.randn(4, 4, 4, 0).cuda(),
                                                 [3, 4, 4, 0])
        with self.assertRaisesRegex(RuntimeError, "add: expected 'other' to be a CPU tensor\\, but got a CUDA tensor"):
            x + sparse_y

        x = torch.zeros(0, 4, 4, 0)
        sparse_y = torch.cuda.sparse.FloatTensor(torch.LongTensor(1, 0).cuda(),
                                                 torch.randn(0, 4, 4, 0).cuda(),
                                                 [0, 4, 4, 0])
        with self.assertRaisesRegex(RuntimeError, "add: expected 'other' to be a CPU tensor\\, but got a CUDA tensor"):
            x + sparse_y


if __name__ == '__main__':
    run_tests()
