import sys
import math
import random
import torch
import tempfile
import unittest
from itertools import product, chain
from common import TestCase, iter_indices

SIZE = 100

class TestSparse(TestCase):
    @staticmethod
    def __gen_sparse(d, nnz, max_dim, with_size=None):
        i = (torch.rand(d, nnz) * max_dim).type(torch.LongTensor)
        v = torch.rand(nnz)
        if with_size is None:
            x = torch.SparseTensor(i, v)
        else:
            x = torch.SparseTensor(i, v, with_size)

        return x, i, v

    @staticmethod
    def __eq(x, y, prec=1e-5):
        return (x - y).abs().le(prec).all()

    def test_basic(self):
        x, i, v = self.__gen_sparse(3, 10, 100)

        self.assertTrue(i.eq(x.indicies()).all())
        self.assertTrue(self.__eq(v, x.values()))

        x, i, v = self.__gen_sparse(3, 10, 100,
                                    torch.LongTensor([100, 100, 100]))

        self.assertTrue(i.eq(x.indicies()).all())
        self.assertTrue(self.__eq(v, x.values()))
        self.assertEqual(x.ndimension(), 3)
        self.assertEqual(x.nnz(), 10)
        for i in range(3):
            self.assertEqual(x.size(i), 100)

    def test_to_dense(self):
        i = torch.LongTensor([
            [0, 1, 2, 2],
            [0, 0, 0, 3],
            [0, 0, 1, 4],
        ])
        v = torch.Tensor([2, 1, 3, 4])
        x = torch.SparseTensor(i, v, torch.LongTensor([3, 4, 5]))
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
        self.assertTrue(self.__eq(res, x.to_dense()))

    def test_contig(self):
        i = torch.LongTensor([
            [ 1,  0, 35, 14, 39,  6, 71, 66, 40, 27],
            [92, 31, 62, 50, 22, 65, 89, 74, 56, 34],
        ])
        v = torch.Tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        x = torch.SparseTensor(i, v, torch.LongTensor([100, 100]))
        exp_i = torch.LongTensor([
            [ 0,  1,  6, 14, 27, 35, 39, 40, 66, 71],
            [31, 92, 65, 50, 34, 62, 22, 56, 74, 89],
        ])
        exp_v = torch.Tensor([2, 1, 6, 4, 10, 3, 5, 9, 8, 7])
        x.contiguous()
        self.assertTrue(x.indicies().eq(exp_i).all())
        self.assertTrue(self.__eq(exp_v, x.values()))

        i = torch.LongTensor([
            [2, 0, 2, 1],
            [0, 0, 3, 0],
            [1, 0, 4, 0],
        ])
        v = torch.Tensor([3, 2, 4, 1])
        x = torch.SparseTensor(i, v, torch.LongTensor([3, 4, 5]))
        exp_i = torch.LongTensor([
            [0, 1, 2, 2],
            [0, 0, 0, 3],
            [0, 0, 1, 4],
        ])
        exp_v = torch.Tensor([2, 1, 3, 4])

        x.contiguous()
        self.assertTrue(x.indicies().eq(exp_i).all())
        self.assertTrue(self.__eq(exp_v, x.values()))

        # Duplicate indicies
        i = torch.LongTensor([
            [0, 0, 2, 0],
            [0, 0, 3, 0],
            [0, 0, 4, 0],
        ])
        v = torch.Tensor([3, 2, 4, 1])
        x = torch.SparseTensor(i, v, torch.LongTensor([3, 4, 5]))
        exp_i = torch.LongTensor([
            [0, 2],
            [0, 3],
            [0, 4],
        ])
        exp_v = torch.Tensor([6, 4])

        x.contiguous()
        self.assertTrue(x.indicies().eq(exp_i).all())
        self.assertTrue(self.__eq(exp_v, x.values()))


    def test_addSmm(self):
        def test_shape(di, dj, dk):
            x, _, _ = self.__gen_sparse(2, 20, di, torch.LongTensor([di, dj]))
            y = torch.randn(dj, dk)
            t = torch.zeros(di, dk)

            expected = torch.addmm(t, x.to_dense(), y)
            res = torch.SparseTensor._torch.addSmm(t, x, y)

            self.assertTrue(self.__eq(res, expected))

        test_shape(10, 100, 100)
        test_shape(100, 1000, 200)
        test_shape(1000, 1000, 500)



if __name__ == '__main__':
    unittest.main()

