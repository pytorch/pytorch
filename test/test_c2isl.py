# NB: tvm MUST be imported first, because of static initializer shenanigans.
import tvm

import torch
import torch.cuda
import torch.isl
from torch.autograd import Variable
from common import TestCase, run_tests

HAS_CUDA = True
if not torch.cuda.is_available():
    print('CUDA not available, skipping tests')
    TestCase = object  # noqa: F811
    HAS_CUDA = False


class TestC2isl(TestCase):
    def test_isl_mat_mul(self):
        M = 4
        N = 2
        K = 8

        # In test suite, this defaults to double
        a = Variable(torch.randn(M, K).cuda())
        b = Variable(torch.randn(K, N).cuda())

        x = torch.isl.IslMatMul()(a, b)
        y = a.matmul(b)
        self.assertEqual(x, y)

    def test_bad_size(self):
        a = Variable(torch.randn(2, 2).cuda())
        b = Variable(torch.randn(2, 2).cuda())
        c = Variable(torch.randn(4, 4).cuda())
        d = Variable(torch.randn(4, 4).cuda())

        op = torch.isl.IslMatMul()
        op(a, b)
        self.assertRaises(RuntimeError, lambda: op(c, d))

    def test_bad_type(self):
        a = Variable(torch.randn(2, 2).cuda())
        b = Variable(torch.randn(2, 2).cuda())
        op = torch.isl.IslMatMul('int32')
        self.assertRaises(RuntimeError, lambda: op(a, b))

    def test_bad_type_later(self):
        x = [[1, 1], [1, 1]]
        a = Variable(torch.FloatTensor(x).cuda())
        b = Variable(torch.FloatTensor(x).cuda())
        c = Variable(torch.DoubleTensor(x).cuda())
        d = Variable(torch.DoubleTensor(x).cuda())
        op = torch.isl.IslMatMul('float32')
        op(a, b)
        self.assertRaises(RuntimeError, lambda: op(c, d))

if __name__ == '__main__':
    run_tests()
