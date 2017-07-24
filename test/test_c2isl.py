import torch
import torch.cuda
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
        x = torch._C._functions.ISLMatMul()(a, b)
        y = a.matmul(b)
        self.assertEqual(x, y)

if __name__ == '__main__':
    run_tests()
