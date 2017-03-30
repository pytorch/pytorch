import torch
from torch import sparse

import itertools
import random
import unittest
from common import TestCase, run_tests
from numbers import Number

# triplet := (index type, value type, sparse type)
cpu_triplet = (
    torch.LongTensor,
    torch.DoubleTensor,
    torch.sparse.DoubleTensor)
type_triplets = [cpu_triplet]
if torch.cuda.is_available():
    cuda_triplet = (
        torch.cuda.LongTensor,
        torch.cuda.DoubleTensor,
        torch.cuda.sparse.DoubleTensor)
    type_triplets.append(cuda_triplet)


class TestSparse(TestCase):

    @staticmethod
    def _gen_sparse(d, nnz, with_size, is_cuda=False):  # FIXME remove default is_cuda value to ensure coverage
        if isinstance(with_size, Number):
            v = torch.randn(nnz)
            i = (torch.rand(d, nnz) * with_size).type(torch.LongTensor)
            x = torch.sparse.DoubleTensor(i, v)
        else:
            v_size = [nnz] + list(with_size[d:])
            v = torch.randn(*v_size)
            i = torch.rand(d, nnz) * \
                torch.Tensor(with_size[:d]).repeat(nnz, 1).transpose(0, 1)
            i = i.type(torch.LongTensor)
            x = torch.sparse.DoubleTensor(i, v, torch.Size(with_size))

        if is_cuda:
            return x.cuda(), i.cuda(), v.cuda()
        else:
            return x, i, v

    def test_basic(self):
        for is_cuda in [False, True] if torch.cuda.is_available() else [False]:
            x, i, v = self._gen_sparse(3, 10, 100, is_cuda)

            self.assertEqual(i, x.indices())
            self.assertEqual(v, x.values())

            x, i, v = self._gen_sparse(3, 10, [100, 100, 100], is_cuda)
            self.assertEqual(i, x.indices())
            self.assertEqual(v, x.values())
            self.assertEqual(x.ndimension(), 3)
            self.assertEqual(x.nnz(), 10)
            for i in range(3):
                self.assertEqual(x.size(i), 100)

        for _, _, SparseTensor in type_triplets:
            # Make sure we can access empty indices / values
            x = SparseTensor()
            self.assertEqual(x.indices().numel(), 0)
            self.assertEqual(x.values().numel(), 0)

    def test_to_dense(self):
        for IndexTensor, ValueTensor, SparseTensor in type_triplets:
            i = IndexTensor([
                [0, 1, 2, 2],
                [0, 0, 0, 3],
                [0, 0, 1, 4],
            ])
            v = ValueTensor([2, 1, 3, 4])
            x = SparseTensor(i, v, torch.Size([3, 4, 5]))
            res = ValueTensor([
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

            x.to_dense()  # Tests double to_dense for memory corruption
            x.to_dense()
            x.to_dense()
            self.assertEqual(res, x.to_dense())

    def test_to_dense_hybrid(self):
        for IndexTensor, ValueTensor, SparseTensor in type_triplets:
            i = IndexTensor([
                [0, 1, 2, 2],
                [0, 0, 0, 3],
            ])
            v = ValueTensor([[2, 3], [1, 2], [3, 4], [4, 5]])
            x = SparseTensor(i, v, torch.Size([3, 4, 2]))
            res = ValueTensor([
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

            x.to_dense()  # Tests double to_dense for memory corruption
            x.to_dense()
            x.to_dense()
            self.assertEqual(res, x.to_dense())

    def test_contig(self):
        for IndexTensor, ValueTensor, SparseTensor in type_triplets:
            i = IndexTensor([
                [1, 0, 35, 14, 39, 6, 71, 66, 40, 27],
                [92, 31, 62, 50, 22, 65, 89, 74, 56, 34],
            ])
            v = ValueTensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
            x = SparseTensor(i, v, torch.Size([100, 100]))
            exp_i = IndexTensor([
                [0, 1, 6, 14, 27, 35, 39, 40, 66, 71],
                [31, 92, 65, 50, 34, 62, 22, 56, 74, 89],
            ])
            exp_v = ValueTensor([2, 1, 6, 4, 10, 3, 5, 9, 8, 7])
            x.contiguous()
            self.assertEqual(exp_i, x.indices())
            self.assertEqual(exp_v, x.values())

            i = IndexTensor([
                [2, 0, 2, 1],
                [0, 0, 3, 0],
                [1, 0, 4, 0],
            ])
            v = ValueTensor([3, 2, 4, 1])
            x = SparseTensor(i, v, torch.Size([3, 4, 5]))
            exp_i = IndexTensor([
                [0, 1, 2, 2],
                [0, 0, 0, 3],
                [0, 0, 1, 4],
            ])
            exp_v = ValueTensor([2, 1, 3, 4])

            x.contiguous()
            self.assertEqual(exp_i, x.indices())
            self.assertEqual(exp_v, x.values())

            # Duplicate indices
            i = IndexTensor([
                [0, 0, 2, 0],
                [0, 0, 3, 0],
                [0, 0, 4, 0],
            ])
            v = ValueTensor([3, 2, 4, 1])
            x = SparseTensor(i, v, torch.Size([3, 4, 5]))
            exp_i = IndexTensor([
                [0, 2],
                [0, 3],
                [0, 4],
            ])
            exp_v = ValueTensor([6, 4])

            x.contiguous()
            self.assertEqual(exp_i, x.indices())
            self.assertEqual(exp_v, x.values())

    def test_contig_hybrid(self):
        for IndexTensor, ValueTensor, SparseTensor in type_triplets:
            i = IndexTensor([
                [1, 0, 35, 14, 39, 6, 71, 66, 40, 27],
                [92, 31, 62, 50, 22, 65, 89, 74, 56, 34],
            ])
            v = ValueTensor([
                [1, 2], [2, 3], [3, 4], [4, 5], [5, 6],
                [6, 7], [7, 8], [8, 9], [9, 10], [10, 11],
            ])
            x = SparseTensor(i, v, torch.Size([100, 100, 2]))
            exp_i = IndexTensor([
                [0, 1, 6, 14, 27, 35, 39, 40, 66, 71],
                [31, 92, 65, 50, 34, 62, 22, 56, 74, 89],
            ])
            exp_v = ValueTensor([
                [2, 3], [1, 2], [6, 7], [4, 5], [10, 11],
                [3, 4], [5, 6], [9, 10], [8, 9], [7, 8],
            ])
            x.contiguous()
            self.assertEqual(exp_i, x.indices())
            self.assertEqual(exp_v, x.values())

            i = IndexTensor([
                [2, 0, 2, 1],
                [0, 0, 3, 0],
                [1, 0, 4, 0],
            ])
            v = ValueTensor([[3, 3, 3], [2, 2, 2], [4, 4, 4], [1, 1, 1]])
            x = SparseTensor(i, v, torch.Size([3, 4, 5, 3]))
            exp_i = IndexTensor([
                [0, 1, 2, 2],
                [0, 0, 0, 3],
                [0, 0, 1, 4],
            ])
            exp_v = ValueTensor([[2, 2, 2], [1, 1, 1], [3, 3, 3], [4, 4, 4]])

            x.contiguous()
            self.assertEqual(exp_i, x.indices())
            self.assertEqual(exp_v, x.values())

            # Duplicate indices
            i = IndexTensor([
                [0, 0, 2, 0],
                [0, 0, 3, 0],
                [0, 0, 4, 0],
            ])
            v = ValueTensor([[3, 2, 3], [2, 1, 1], [4, 3, 4], [1, 1, 1]])
            x = SparseTensor(i, v, torch.Size([3, 4, 5, 3]))
            exp_i = IndexTensor([
                [0, 2],
                [0, 3],
                [0, 4],
            ])
            exp_v = ValueTensor([[6, 4, 5], [4, 3, 4]])

            x.contiguous()
            self.assertEqual(exp_i, x.indices())
            self.assertEqual(exp_v, x.values())

    def test_transpose(self):
        for is_cuda in [False, True]:
            x = self._gen_sparse(4, 20, 5, is_cuda=is_cuda)[0]
            y = x.to_dense()

            for i, j in itertools.combinations(range(4), 2):
                x = x.transpose_(i, j)
                y = y.transpose(i, j)
                self.assertEqual(x.to_dense(), y)

                x = x.transpose(i, j)
                y = y.transpose(i, j)
                self.assertEqual(x.to_dense(), y)

    def test_mm(self):
        def test_shape(di, dj, dk):
            x, _, _ = self._gen_sparse(2, 20, [di, dj])
            t = torch.randn(di, dk)
            y = torch.randn(dj, dk)
            alpha = random.random()
            beta = random.random()

            res = torch.addmm(alpha, t, beta, x, y)
            expected = torch.addmm(alpha, t, beta, x.to_dense(), y)
            self.assertEqual(res, expected)

            res = torch.addmm(t, x, y)
            expected = torch.addmm(t, x.to_dense(), y)
            self.assertEqual(res, expected)

            res = torch.mm(x, y)
            expected = torch.mm(x.to_dense(), y)
            self.assertEqual(res, expected)

        test_shape(10, 100, 100)
        test_shape(100, 1000, 200)
        test_shape(64, 10000, 300)

    def test_saddmm(self):
        def test_shape(di, dj, dk):
            x = self._gen_sparse(2, 20, [di, dj])[0]
            t = self._gen_sparse(2, 20, [di, dk])[0]
            y = torch.randn(dj, dk)
            alpha = random.random()
            beta = random.random()

            res = torch.saddmm(alpha, t, beta, x, y)
            expected = torch.addmm(alpha, t.to_dense(), beta, x.to_dense(), y)
            self.assertEqual(res.to_dense(), expected)

            res = torch.saddmm(t, x, y)
            expected = torch.addmm(t.to_dense(), x.to_dense(), y)
            self.assertEqual(res.to_dense(), expected)

            res = torch.smm(x, y)
            expected = torch.mm(x.to_dense(), y)
            self.assertEqual(res.to_dense(), expected)

        test_shape(7, 5, 3)
        test_shape(1000, 100, 100)
        test_shape(3000, 64, 300)

    def test_dsmm(self):
        def test_shape(di, dj, dk):
            for is_cuda in [False, True]:
                x = self._gen_sparse(2, 20, [di, dj], is_cuda)[0]
                y = torch.randn(dj, dk)
                if is_cuda:
                    y = y.cuda()

                res = torch.dsmm(x, y)
                expected = torch.mm(x.to_dense(), y)
                self.assertEqual(res, expected)

        test_shape(7, 5, 3)
        test_shape(1000, 100, 100)
        test_shape(3000, 64, 300)

    def test_hsmm(self):
        def test_shape(di, dj, dk):
            for is_cuda in [False, True]:
                x = self._gen_sparse(2, 20, [di, dj], is_cuda)[0]
                y = torch.randn(dj, dk)
                if is_cuda:
                    y = y.cuda()

                res = torch.hsmm(x, y)
                expected = torch.mm(x.to_dense(), y)
                self.assertEqual(res.to_dense(), expected)

        test_shape(7, 5, 3)
        test_shape(1000, 100, 100)
        test_shape(3000, 64, 300)

    def _test_spadd_shape(self, shape_i, shape_v=None):
        for is_cuda in [False, True]:
            shape = shape_i + (shape_v or [])
            x, _, _ = self._gen_sparse(len(shape_i), 10, shape, is_cuda)
            y = torch.randn(*shape)
            if is_cuda:
                y = y.cuda()
            r = random.random()

            res = torch.add(y, r, x)
            expected = y + r * x.to_dense()

            self.assertEqual(res, expected)

            # Non contiguous dense tensor
            s = list(shape)
            s[0] = shape[-1]
            s[-1] = shape[0]
            y = torch.randn(*s)
            if is_cuda:
                y = y.cuda()
            y.transpose_(0, len(s) - 1)
            r = random.random()

            res = torch.add(y, r, x)
            expected = y + r * x.to_dense()

            self.assertEqual(res, expected)

    def test_spadd(self):
        self._test_spadd_shape([5, 6])
        self._test_spadd_shape([10, 10, 10])
        self._test_spadd_shape([50, 30, 20])
        self._test_spadd_shape([5, 5, 5, 5, 5, 5])

    def test_spadd_hybrid(self):
        self._test_spadd_shape([5, 6], [2, 3])
        self._test_spadd_shape([10, 10, 10], [3])
        self._test_spadd_shape([50, 30, 20], [2])
        self._test_spadd_shape([5, 5, 5, 5, 5, 5], [2])

    def _test_basic_ops_shape(self, shape_i, shape_v=None):
        for is_cuda in [False, True]:
            shape = shape_i + (shape_v or [])
            x1, _, _ = self._gen_sparse(len(shape_i), 9, shape, is_cuda)
            x2, _, _ = self._gen_sparse(len(shape_i), 12, shape, is_cuda)

            y1 = x1 + x2
            y2 = x1.clone()
            y2.add_(x2)
            expected = x1.to_dense() + x2.to_dense()
            self.assertEqual(y1.to_dense(), expected)
            self.assertEqual(y2.to_dense(), expected)

            y1 = x1 - x2
            y2 = x1.clone()
            y2.sub_(x2)
            expected = x1.to_dense() - x2.to_dense()
            self.assertEqual(y1.to_dense(), expected)
            self.assertEqual(y2.to_dense(), expected)

            y1 = x1 * x2
            y2 = x1.clone()
            y2.mul_(x2)
            expected = x1.to_dense() * x2.to_dense()
            self.assertEqual(y1.to_dense(), expected)
            self.assertEqual(y2.to_dense(), expected)

            y1 = x1 * 37.5
            y2 = x1.clone()
            y2.mul_(37.5)
            expected = x1.to_dense() * 37.5
            self.assertEqual(y1.to_dense(), expected)
            self.assertEqual(y2.to_dense(), expected)

            y1 = x1 / 37.5
            y2 = x1.clone()
            y2.div_(37.5)
            expected = x1.to_dense() / 37.5
            self.assertEqual(y1.to_dense(), expected)
            self.assertEqual(y2.to_dense(), expected)

            y = x1.clone()
            y.zero_()
            expected = torch.zeros(x1.size())
            self.assertEqual(y.to_dense(), expected)

    def test_basic_ops(self):
        self._test_basic_ops_shape([5, 6])
        self._test_basic_ops_shape([10, 10, 10])
        self._test_basic_ops_shape([50, 30, 20])
        self._test_basic_ops_shape([5, 5, 5, 5, 5, 5])

    def test_basic_ops_hybrid(self):
        self._test_basic_ops_shape([5, 6], [2, 3])
        self._test_basic_ops_shape([10, 10, 10], [3])
        self._test_basic_ops_shape([50, 30, 20], [2])
        self._test_basic_ops_shape([5, 5, 5, 5, 5, 5], [2])

    def _test_sparse_mask_shape(self, shape_i, shape_v=None):
        for is_cuda in [False, True]:
            shape = shape_i + (shape_v or [])
            x1, _, _ = self._gen_sparse(len(shape_i), 9, shape, is_cuda)
            x2, _, _ = self._gen_sparse(len(shape_i), 12, shape, is_cuda)

            y1 = x1 + x2
            y2 = x1.clone()
            y2.add_(x2)
            expected = x1.to_dense() + x2.to_dense()
            self.assertEqual(y1.to_dense(), expected)
            self.assertEqual(y2.to_dense(), expected)

    def test_sparse_mask(self):
        for IndexTensor, ValueTensor, SparseTensor in type_triplets:
            i = IndexTensor([
                [1, 3, 3, 0, 4],
                [2, 1, 1, 2, 3],
            ])
            v = ValueTensor([1, 2, 3, 4, 5])
            x = SparseTensor(i, v, torch.Size([5, 4]))
            dense = ValueTensor([
                [1, 2, 3, 4],
                [5, 6, 7, 8],
                [9, 10, 11, 12],
                [13, 14, 15, 16],
                [17, 18, 19, 20],
            ])
            exp_v = ValueTensor([7, 14, 14, 3, 20])
            res = dense.sparse_mask(x)
            expected = SparseTensor(i, exp_v, torch.Size([5, 4]))
            self.assertEqual(res, expected)

    def test_sparse_mask_hybrid(self):
        for IndexTensor, ValueTensor, SparseTensor in type_triplets:
            i = IndexTensor([
                [1, 3, 3, 0, 4],
                [2, 1, 1, 2, 3],
            ])
            v = ValueTensor([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]])
            x = SparseTensor(i, v, torch.Size([5, 4, 2]))
            dense = ValueTensor([
                [[1, 3], [2, 2], [3, 3], [4, 2]],
                [[5, 7], [6, 7], [7, 9], [8, 9]],
                [[9, 2], [10, 4], [11, 1], [12, 3]],
                [[13, 5], [14, 1], [15, 1], [16, 6]],
                [[17, 7], [18, 2], [19, 7], [20, 1]],
            ])
            res = dense.sparse_mask(x)
            exp_v = ValueTensor([[7, 9], [14, 1], [14, 1], [3, 3], [20, 1]])
            expected = SparseTensor(i, exp_v, torch.Size([5, 4, 2]))
            self.assertEqual(res, expected)


if __name__ == '__main__':
    run_tests()
