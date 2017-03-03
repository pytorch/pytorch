import torch
from torch import sparse

import itertools
import random
import unittest
from common import TestCase, run_tests
from numbers import Number

SparseTensor = sparse.DoubleTensor


class TestSparse(TestCase):

    @staticmethod
    def _gen_sparse(d, nnz, with_size):
        if isinstance(with_size, Number):
            v = torch.randn(nnz)
            i = (torch.rand(d, nnz) * with_size).type(torch.LongTensor)
            x = SparseTensor(i, v)
        else:
            v_size = [nnz] + list(with_size[d:])
            v = torch.randn(*v_size)
            i = torch.rand(d, nnz) * \
                torch.Tensor(with_size[:d]).repeat(nnz, 1).transpose(0, 1)
            i = i.type(torch.LongTensor)
            x = SparseTensor(i, v, torch.Size(with_size))

        return x, i, v

    def test_basic(self):
        x, i, v = self._gen_sparse(3, 10, 100)

        self.assertEqual(i, x.indices())
        self.assertEqual(v, x.values())

        x, i, v = self._gen_sparse(3, 10, [100, 100, 100])
        self.assertEqual(i, x.indices())
        self.assertEqual(v, x.values())
        self.assertEqual(x.ndimension(), 3)
        self.assertEqual(x.nnz(), 10)
        for i in range(3):
            self.assertEqual(x.size(i), 100)

        # Make sure we can access empty indices / values
        x = SparseTensor()
        self.assertEqual(x.indices().numel(), 0)
        self.assertEqual(x.values().numel(), 0)

    def test_to_dense(self):
        i = torch.LongTensor([
            [0, 1, 2, 2],
            [0, 0, 0, 3],
            [0, 0, 1, 4],
        ])
        v = torch.Tensor([2, 1, 3, 4])
        x = SparseTensor(i, v, torch.Size([3, 4, 5]))
        res = torch.Tensor([
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
        i = torch.LongTensor([
            [0, 1, 2, 2],
            [0, 0, 0, 3],
        ])
        v = torch.Tensor([[2, 3], [1, 2], [3, 4], [4, 5]])
        x = SparseTensor(i, v, torch.Size([3, 4, 2]))
        res = torch.Tensor([
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
        i = torch.LongTensor([
            [1, 0, 35, 14, 39, 6, 71, 66, 40, 27],
            [92, 31, 62, 50, 22, 65, 89, 74, 56, 34],
        ])
        v = torch.Tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        x = SparseTensor(i, v, torch.Size([100, 100]))
        exp_i = torch.LongTensor([
            [0, 1, 6, 14, 27, 35, 39, 40, 66, 71],
            [31, 92, 65, 50, 34, 62, 22, 56, 74, 89],
        ])
        exp_v = torch.Tensor([2, 1, 6, 4, 10, 3, 5, 9, 8, 7])
        x.contiguous()
        self.assertEqual(exp_i, x.indices())
        self.assertEqual(exp_v, x.values())

        i = torch.LongTensor([
            [2, 0, 2, 1],
            [0, 0, 3, 0],
            [1, 0, 4, 0],
        ])
        v = torch.Tensor([3, 2, 4, 1])
        x = SparseTensor(i, v, torch.Size([3, 4, 5]))
        exp_i = torch.LongTensor([
            [0, 1, 2, 2],
            [0, 0, 0, 3],
            [0, 0, 1, 4],
        ])
        exp_v = torch.Tensor([2, 1, 3, 4])

        x.contiguous()
        self.assertEqual(exp_i, x.indices())
        self.assertEqual(exp_v, x.values())

        # Duplicate indices
        i = torch.LongTensor([
            [0, 0, 2, 0],
            [0, 0, 3, 0],
            [0, 0, 4, 0],
        ])
        v = torch.Tensor([3, 2, 4, 1])
        x = SparseTensor(i, v, torch.Size([3, 4, 5]))
        exp_i = torch.LongTensor([
            [0, 2],
            [0, 3],
            [0, 4],
        ])
        exp_v = torch.Tensor([6, 4])

        x.contiguous()
        self.assertEqual(exp_i, x.indices())
        self.assertEqual(exp_v, x.values())

    def test_contig_hybrid(self):
        i = torch.LongTensor([
            [1, 0, 35, 14, 39, 6, 71, 66, 40, 27],
            [92, 31, 62, 50, 22, 65, 89, 74, 56, 34],
        ])
        v = torch.Tensor([
            [1, 2], [2, 3], [3, 4], [4, 5], [5, 6],
            [6, 7], [7, 8], [8, 9], [9, 10], [10, 11],
        ])
        x = SparseTensor(i, v, torch.Size([100, 100, 2]))
        exp_i = torch.LongTensor([
            [0, 1, 6, 14, 27, 35, 39, 40, 66, 71],
            [31, 92, 65, 50, 34, 62, 22, 56, 74, 89],
        ])
        exp_v = torch.Tensor([
            [2, 3], [1, 2], [6, 7], [4, 5], [10, 11],
            [3, 4], [5, 6], [9, 10], [8, 9], [7, 8],
        ])
        x.contiguous()
        self.assertEqual(exp_i, x.indices())
        self.assertEqual(exp_v, x.values())

        i = torch.LongTensor([
            [2, 0, 2, 1],
            [0, 0, 3, 0],
            [1, 0, 4, 0],
        ])
        v = torch.Tensor([[3, 3, 3], [2, 2, 2], [4, 4, 4], [1, 1, 1]])
        x = SparseTensor(i, v, torch.Size([3, 4, 5, 3]))
        exp_i = torch.LongTensor([
            [0, 1, 2, 2],
            [0, 0, 0, 3],
            [0, 0, 1, 4],
        ])
        exp_v = torch.Tensor([[2, 2, 2], [1, 1, 1], [3, 3, 3], [4, 4, 4]])

        x.contiguous()
        self.assertEqual(exp_i, x.indices())
        self.assertEqual(exp_v, x.values())

        # Duplicate indices
        i = torch.LongTensor([
            [0, 0, 2, 0],
            [0, 0, 3, 0],
            [0, 0, 4, 0],
        ])
        v = torch.Tensor([[3, 2, 3], [2, 1, 1], [4, 3, 4], [1, 1, 1]])
        x = SparseTensor(i, v, torch.Size([3, 4, 5, 3]))
        exp_i = torch.LongTensor([
            [0, 2],
            [0, 3],
            [0, 4],
        ])
        exp_v = torch.Tensor([[6, 4, 5], [4, 3, 4]])

        x.contiguous()
        self.assertEqual(exp_i, x.indices())
        self.assertEqual(exp_v, x.values())

    def test_transpose(self):
        x = self._gen_sparse(4, 20, 5)[0]
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

            expected = torch.addmm(alpha, t, beta, x.to_dense(), y)
            res = torch.addmm(alpha, t, beta, x, y)
            self.assertEqual(res, expected)

            expected = torch.addmm(t, x.to_dense(), y)
            res = torch.addmm(t, x, y)
            self.assertEqual(res, expected)

            expected = torch.mm(x.to_dense(), y)
            res = torch.mm(x, y)
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

            expected = torch.addmm(alpha, t.to_dense(), beta, x.to_dense(), y)
            res = torch.saddmm(alpha, t, beta, x, y)
            self.assertEqual(res.to_dense(), expected)

            expected = torch.addmm(t.to_dense(), x.to_dense(), y)
            res = torch.saddmm(t, x, y)
            self.assertEqual(res.to_dense(), expected)

            expected = torch.mm(x.to_dense(), y)
            res = torch.smm(x, y)
            self.assertEqual(res.to_dense(), expected)

        test_shape(7, 5, 3)
        test_shape(1000, 100, 100)
        test_shape(3000, 64, 300)

    def _test_spadd_shape(self, shape_i, shape_v=None):
        shape = shape_i + (shape_v or [])
        x, _, _ = self._gen_sparse(len(shape_i), 10, shape)
        y = torch.randn(*shape)
        r = random.random()

        expected = y + r * x.to_dense()
        res = torch.add(y, r, x)

        self.assertEqual(res, expected)

        # Non contiguous dense tensor
        s = list(shape)
        s[0] = shape[-1]
        s[-1] = shape[0]
        y = torch.randn(*s).transpose_(0, len(s) - 1)
        r = random.random()

        expected = y + r * x.to_dense()
        res = torch.add(y, r, x)

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
        shape = shape_i + (shape_v or [])
        x1, _, _ = self._gen_sparse(len(shape_i), 9, shape)
        x2, _, _ = self._gen_sparse(len(shape_i), 12, shape)

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


if __name__ == '__main__':
    run_tests()
