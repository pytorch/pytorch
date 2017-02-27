import sys
import os
import math
import random
import torch
import torch.cuda
import tempfile
import unittest
import warnings
from itertools import product, chain
from functools import wraps
from common import TestCase, iter_indices, TEST_NUMPY, run_tests, download_file

if TEST_NUMPY:
    import numpy as np

SIZE = 100


def skipIfNoLapack(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            fn(*args, **kwargs)
        except Exception as e:
            if 'Lapack library not found' in e.args[0]:
                raise unittest.SkipTest('Compiled without Lapack')
            raise
    return wrapper


class TestTorch(TestCase):

    def test_dot(self):
        types = {
            'torch.DoubleTensor': 1e-8,
            'torch.FloatTensor': 1e-4,
        }
        for tname, prec in types.items():
            v1 = torch.randn(100).type(tname)
            v2 = torch.randn(100).type(tname)
            res1 = torch.dot(v1, v2)
            res2 = 0
            for i, j in zip(v1, v2):
                res2 += i * j
            self.assertEqual(res1, res2)

    def _testMath(self, torchfn, mathfn):
        size = (10, 5)
        # contiguous
        m1 = torch.randn(*size)
        res1 = torchfn(m1[4])
        res2 = res1.clone().zero_()
        for i, v in enumerate(m1[4]):
            res2[i] = mathfn(v)
        self.assertEqual(res1, res2)

        # non-contiguous
        m1 = torch.randn(*size)
        res1 = torchfn(m1[:, 4])
        res2 = res1.clone().zero_()
        for i, v in enumerate(m1[:, 4]):
            res2[i] = mathfn(v)
        self.assertEqual(res1, res2)

    def _testMathByName(self, function_name):
        torchfn = getattr(torch, function_name)
        mathfn = getattr(math, function_name)
        self._testMath(torchfn, mathfn)

    def test_sin(self):
        self._testMathByName('sin')

    def test_sinh(self):
        self._testMathByName('sinh')

    def test_asin(self):
        self._testMath(torch.asin, lambda x: math.asin(x) if abs(x) <= 1 else float('nan'))

    def test_cos(self):
        self._testMathByName('cos')

    def test_cosh(self):
        self._testMathByName('cosh')

    def test_acos(self):
        self._testMath(torch.acos, lambda x: math.acos(x) if abs(x) <= 1 else float('nan'))

    def test_tan(self):
        self._testMathByName('tan')

    def test_tanh(self):
        self._testMathByName('tanh')

    def test_atan(self):
        self._testMathByName('atan')

    def test_log(self):
        self._testMath(torch.log, lambda x: math.log(x) if x > 0 else float('nan'))

    def test_sqrt(self):
        self._testMath(torch.sqrt, lambda x: math.sqrt(x) if x > 0 else float('nan'))

    def test_exp(self):
        self._testMathByName('exp')

    def test_floor(self):
        self._testMathByName('floor')

    def test_ceil(self):
        self._testMathByName('ceil')

    def test_rsqrt(self):
        self._testMath(torch.rsqrt, lambda x: 1 / math.sqrt(x) if x > 0 else float('nan'))

    def test_sigmoid(self):
        # TODO: why not simulate math.sigmoid like with rsqrt?
        inputValues = [-1000, -1, 0, 0.5, 1, 2, 1000]
        expectedOutput = [0.0000, 0.2689, 0.5, 0.6225, 0.7311, 0.8808, 1.000]
        precision_4dps = 0.0002

        def checkType(tensor):
            self.assertEqual(tensor(inputValues).sigmoid(), tensor(expectedOutput), precision_4dps)

        checkType(torch.FloatTensor)
        checkType(torch.DoubleTensor)

    def test_frac(self):
        self._testMath(torch.frac, lambda x: math.fmod(x, 1))

    def test_trunc(self):
        self._testMath(torch.trunc, lambda x: x - math.fmod(x, 1))

    def test_round(self):
        self._testMath(torch.round, round)

    def test_has_storage(self):
        self.assertIsNotNone(torch.Tensor().storage())
        self.assertIsNotNone(torch.Tensor(0).storage())
        self.assertIsNotNone(torch.Tensor([]).storage())
        self.assertIsNotNone(torch.Tensor().clone().storage())
        self.assertIsNotNone(torch.Tensor([0, 0, 0]).nonzero().storage())

    @unittest.skipIf(not TEST_NUMPY, "Numpy not found")
    def test_has_storage_numpy(self):
        arr = np.array([], dtype=np.float32)
        self.assertIsNotNone(torch.Tensor(arr).storage())

    def _testSelection(self, torchfn, mathfn):
        # contiguous
        m1 = torch.randn(100, 100)
        res1 = torchfn(m1)
        res2 = m1[0, 0]
        for i, j in iter_indices(m1):
            res2 = mathfn(res2, m1[i, j])
        self.assertEqual(res1, res2)

        # non-contiguous
        m1 = torch.randn(10, 10, 10)
        m2 = m1[:, 4]
        res1 = torchfn(m2)
        res2 = m2[0, 0]
        for i, j in iter_indices(m2):
            res2 = mathfn(res2, m2[i][j])
        self.assertEqual(res1, res2)

        # with indices
        m1 = torch.randn(100, 100)
        res1val, res1ind = torchfn(m1, 1)
        res2val = m1[:, 0:1].clone()
        res2ind = res1ind.clone().fill_(0)
        for i, j in iter_indices(m1):
            if mathfn(res2val[i, 0], m1[i, j]) != res2val[i, 0]:
                res2val[i, 0] = m1[i, j]
                res2ind[i, 0] = j

        maxerr = 0
        for i in range(res1val.size(0)):
            maxerr = max(maxerr, abs(res1val[i][0] - res2val[i][0]))
            self.assertEqual(res1ind[i][0], res2ind[i][0])
        self.assertLessEqual(abs(maxerr), 1e-5)

        # NaNs
        for index in (0, 4, 99):
            m1 = torch.randn(100)
            m1[index] = float('nan')
            res1val, res1ind = torch.max(m1, 0)
            self.assertNotEqual(res1val[0], res1val[0])
            self.assertEqual(res1ind[0], index)
            res1val = torchfn(m1)
            self.assertNotEqual(res1val, res1val)

    def test_max(self):
        self._testSelection(torch.max, max)

    def test_min(self):
        self._testSelection(torch.min, min)

    def _testCSelection(self, torchfn, mathfn):
        # Two tensors
        size = (100, 100)
        a = torch.rand(*size)
        b = torch.rand(*size)
        c = torchfn(a, b)
        expected_c = torch.zeros(*size)
        expected_c.map2_(a, b, lambda _, a, b: mathfn(a, b))
        self.assertEqual(expected_c, c, 0)

    def test_max_elementwise(self):
        self._testCSelection(torch.max, max)

    def test_min_elementwise(self):
        self._testCSelection(torch.min, min)

    def test_lerp(self):
        def TH_lerp(a, b, weight):
            return a + weight * (b - a)

        size = (100, 100)
        a = torch.rand(*size)
        b = torch.rand(*size)
        w = random.random()
        result = torch.lerp(a, b, w)
        expected = a.clone()
        expected.map2_(a, b, lambda _, a, b: TH_lerp(a, b, w))
        self.assertEqual(result, expected)

    def test_all_any(self):
        def test(size):
            x = torch.ones(*size).byte()
            self.assertTrue(x.all())
            self.assertTrue(x.any())

            x[3] = 0
            self.assertFalse(x.all())
            self.assertTrue(x.any())

            x.zero_()
            self.assertFalse(x.all())
            self.assertFalse(x.any())

            x.fill_(2)
            self.assertTrue(x.all())
            self.assertTrue(x.any())

        test((10,))
        test((5, 5))

    def test_mv(self):
        m1 = torch.randn(100, 100)
        v1 = torch.randn(100)

        res1 = torch.mv(m1, v1)
        res2 = res1.clone().zero_()
        for i, j in iter_indices(m1):
            res2[i] += m1[i][j] * v1[j]

        self.assertEqual(res1, res2)

    def test_add(self):
        # [res] torch.add([res,] tensor1, tensor2)
        m1 = torch.randn(100, 100)
        v1 = torch.randn(100)

        # contiguous
        res1 = torch.add(m1[4], v1)
        res2 = res1.clone().zero_()
        for i in range(m1.size(1)):
            res2[i] = m1[4, i] + v1[i]
        self.assertEqual(res1, res2)

        m1 = torch.randn(100, 100)
        v1 = torch.randn(100)

        # non-contiguous
        res1 = torch.add(m1[:, 4], v1)
        res2 = res1.clone().zero_()
        for i in range(m1.size(0)):
            res2[i] = m1[i, 4] + v1[i]
        self.assertEqual(res1, res2)

        # [res] torch.add([res,] tensor, value)
        m1 = torch.randn(10, 10)

        # contiguous
        res1 = m1.clone()
        res1[3].add_(2)
        res2 = m1.clone()
        for i in range(m1.size(1)):
            res2[3, i] = res2[3, i] + 2
        self.assertEqual(res1, res2)

        # non-contiguous
        m1 = torch.randn(10, 10)
        res1 = m1.clone()
        res1[:, 3].add_(2)
        res2 = m1.clone()
        for i in range(m1.size(0)):
            res2[i, 3] = res2[i, 3] + 2
        self.assertEqual(res1, res2)

        # [res] torch.add([res,] tensor1, value, tensor2)

    def test_csub(self):
        # with a tensor
        a = torch.randn(100, 90)
        b = a.clone().normal_()

        res_add = torch.add(a, -1, b)
        res_csub = a.clone()
        res_csub.sub_(b)
        self.assertEqual(res_add, res_csub)

        # with a scalar
        a = torch.randn(100, 100)

        scalar = 123.5
        res_add = torch.add(a, -scalar)
        res_csub = a.clone()
        res_csub.sub_(scalar)
        self.assertEqual(res_add, res_csub)

    def test_neg(self):
        a = torch.randn(100, 90)
        zeros = torch.Tensor().resize_as_(a).zero_()

        res_add = torch.add(zeros, -1, a)
        res_neg = a.clone()
        res_neg.neg_()
        self.assertEqual(res_neg, res_add)

    def test_reciprocal(self):
        a = torch.randn(100, 89)
        zeros = torch.Tensor().resize_as_(a).zero_()

        res_pow = torch.pow(a, -1)
        res_reciprocal = a.clone()
        res_reciprocal.reciprocal_()
        self.assertEqual(res_reciprocal, res_pow)

    def test_mul(self):
        m1 = torch.randn(10, 10)
        res1 = m1.clone()
        res1[:, 3].mul_(2)
        res2 = m1.clone()
        for i in range(res1.size(0)):
            res2[i, 3] = res2[i, 3] * 2
        self.assertEqual(res1, res2)

    def test_div(self):
        m1 = torch.randn(10, 10)
        res1 = m1.clone()
        res1[:, 3].div_(2)
        res2 = m1.clone()
        for i in range(m1.size(0)):
            res2[i, 3] = res2[i, 3] / 2
        self.assertEqual(res1, res2)

    def test_fmod(self):
        m1 = torch.Tensor(10, 10).uniform_(-10., 10.)
        res1 = m1.clone()
        q = 2.1
        res1[:, 3].fmod_(q)
        res2 = m1.clone()
        for i in range(m1.size(1)):
            res2[i, 3] = math.fmod(res2[i, 3], q)
        self.assertEqual(res1, res2)

    def test_remainder(self):
        m1 = torch.Tensor(10, 10).uniform_(-10., 10.)
        res1 = m1.clone()
        q = 2.1
        res1[:, 3].remainder_(q)
        res2 = m1.clone()
        for i in range(m1.size(0)):
            res2[i, 3] = res2[i, 3] % q
        self.assertEqual(res1, res2)

    def test_mm(self):
        # helper function
        def matrixmultiply(mat1, mat2):
            n = mat1.size(0)
            m = mat1.size(1)
            p = mat2.size(1)
            res = torch.zeros(n, p)
            for i, j in iter_indices(res):
                res[i, j] = sum(mat1[i, k] * mat2[k, j] for k in range(m))
            return res

        # contiguous case
        n, m, p = 10, 10, 5
        mat1 = torch.randn(n, m)
        mat2 = torch.randn(m, p)
        res = torch.mm(mat1, mat2)

        res2 = matrixmultiply(mat1, mat2)
        self.assertEqual(res, res2)

        # non contiguous case 1
        n, m, p = 10, 10, 5
        mat1 = torch.randn(n, m)
        mat2 = torch.randn(p, m).t()
        res = torch.mm(mat1, mat2)

        res2 = matrixmultiply(mat1, mat2)
        self.assertEqual(res, res2)

        # non contiguous case 2
        n, m, p = 10, 10, 5
        mat1 = torch.randn(m, n).t()
        mat2 = torch.randn(m, p)
        res = torch.mm(mat1, mat2)

        res2 = matrixmultiply(mat1, mat2)
        self.assertEqual(res, res2)

        # non contiguous case 3
        n, m, p = 10, 10, 5
        mat1 = torch.randn(m, n).t()
        mat2 = torch.randn(p, m).t()
        res = torch.mm(mat1, mat2)

        res2 = matrixmultiply(mat1, mat2)
        self.assertEqual(res, res2)

        # test with zero stride
        n, m, p = 10, 10, 5
        mat1 = torch.randn(n, m)
        mat2 = torch.randn(m, 1).expand(m, p)
        res = torch.mm(mat1, mat2)

        res2 = matrixmultiply(mat1, mat2)
        self.assertEqual(res, res2)

    def test_bmm(self):
        num_batches = 10
        M, N, O = 23, 8, 12
        b1 = torch.randn(num_batches, M, N)
        b2 = torch.randn(num_batches, N, O)
        res = torch.bmm(b1, b2)
        for i in range(num_batches):
            r = torch.mm(b1[i], b2[i])
            self.assertEqual(r, res[i])

    def test_addbmm(self):
        # num_batches = 10
        # M, N, O = 12, 8, 5
        num_batches = 2
        M, N, O = 2, 3, 4
        b1 = torch.randn(num_batches, M, N)
        b2 = torch.randn(num_batches, N, O)
        res = torch.bmm(b1, b2)
        res2 = torch.Tensor().resize_as_(res[0]).zero_()

        res2.addbmm_(b1, b2)
        self.assertEqual(res2, res.sum(0)[0])

        res2.addbmm_(1, b1, b2)
        self.assertEqual(res2, res.sum(0)[0] * 2)

        res2.addbmm_(1., .5, b1, b2)
        self.assertEqual(res2, res.sum(0)[0] * 2.5)

        res3 = torch.addbmm(1, res2, 0, b1, b2)
        self.assertEqual(res3, res2)

        res4 = torch.addbmm(1, res2, .5, b1, b2)
        self.assertEqual(res4, res.sum(0)[0] * 3)

        res5 = torch.addbmm(0, res2, 1, b1, b2)
        self.assertEqual(res5, res.sum(0)[0])

        res6 = torch.addbmm(.1, res2, .5, b1, b2)
        self.assertEqual(res6, res2 * .1 + res.sum(0) * .5)

    def test_baddbmm(self):
        num_batches = 10
        M, N, O = 12, 8, 5
        b1 = torch.randn(num_batches, M, N)
        b2 = torch.randn(num_batches, N, O)
        res = torch.bmm(b1, b2)
        res2 = torch.Tensor().resize_as_(res).zero_()

        res2.baddbmm_(b1, b2)
        self.assertEqual(res2, res)

        res2.baddbmm_(1, b1, b2)
        self.assertEqual(res2, res * 2)

        res2.baddbmm_(1, .5, b1, b2)
        self.assertEqual(res2, res * 2.5)

        res3 = torch.baddbmm(1, res2, 0, b1, b2)
        self.assertEqual(res3, res2)

        res4 = torch.baddbmm(1, res2, .5, b1, b2)
        self.assertEqual(res4, res * 3)

        res5 = torch.baddbmm(0, res2, 1, b1, b2)
        self.assertEqual(res5, res)

        res6 = torch.baddbmm(.1, res2, .5, b1, b2)
        self.assertEqual(res6, res2 * .1 + res * .5)

    def test_clamp(self):
        m1 = torch.rand(100).mul(5).add(-2.5)  # uniform in [-2.5, 2.5]
        # just in case we're extremely lucky.
        min_val = -1
        max_val = 1
        m1[1] = min_val
        m1[2] = max_val

        res1 = m1.clone()
        res1.clamp_(min_val, max_val)
        res2 = m1.clone()
        for i in iter_indices(res2):
            res2[i] = max(min_val, min(max_val, res2[i]))
        self.assertEqual(res1, res2)

        res1 = torch.clamp(m1, min=min_val)
        res2 = m1.clone()
        for i in iter_indices(res2):
            res2[i] = max(min_val, res2[i])
        self.assertEqual(res1, res2)

        res1 = torch.clamp(m1, max=max_val)
        res2 = m1.clone()
        for i in iter_indices(res2):
            res2[i] = min(max_val, res2[i])
        self.assertEqual(res1, res2)

    def test_pow(self):
        # [res] torch.pow([res,] x)

        # base - tensor, exponent - number
        # contiguous
        m1 = torch.randn(100, 100)
        res1 = torch.pow(m1[4], 3)
        res2 = res1.clone().zero_()
        for i in range(res2.size(0)):
            res2[i] = math.pow(m1[4][i], 3)
        self.assertEqual(res1, res2)

        # non-contiguous
        m1 = torch.randn(100, 100)
        res1 = torch.pow(m1[:, 4], 3)
        res2 = res1.clone().zero_()
        for i in range(res2.size(0)):
            res2[i] = math.pow(m1[i, 4], 3)
        self.assertEqual(res1, res2)

        # base - number, exponent - tensor
        # contiguous
        m1 = torch.randn(100, 100)
        res1 = torch.pow(3, m1[4])
        res2 = res1.clone().zero_()
        for i in range(res2.size(0)):
            res2[i] = math.pow(3, m1[4, i])
        self.assertEqual(res1, res2)

        # non-contiguous
        m1 = torch.randn(100, 100)
        res1 = torch.pow(3, m1[:, 4])
        res2 = res1.clone().zero_()
        for i in range(res2.size(0)):
            res2[i] = math.pow(3, m1[i][4])
        self.assertEqual(res1, res2)

    def _test_cop(self, torchfn, mathfn):
        def reference_implementation(res2):
            for i, j in iter_indices(sm1):
                idx1d = i * sm1.size(0) + j
                res2[i, j] = mathfn(sm1[i, j], sm2[idx1d])
            return res2

        # contiguous
        m1 = torch.randn(10, 10, 10)
        m2 = torch.randn(10, 10 * 10)
        sm1 = m1[4]
        sm2 = m2[4]
        res1 = torchfn(sm1, sm2)
        res2 = reference_implementation(res1.clone())
        self.assertEqual(res1, res2)

        # non-contiguous
        m1 = torch.randn(10, 10, 10)
        m2 = torch.randn(10 * 10, 10 * 10)
        sm1 = m1[:, 4]
        sm2 = m2[:, 4]
        res1 = torchfn(sm1, sm2)
        res2 = reference_implementation(res1.clone())
        self.assertEqual(res1, res2)

    def test_cdiv(self):
        self._test_cop(torch.div, lambda x, y: x / y)

    def test_cfmod(self):
        self._test_cop(torch.fmod, math.fmod)

    def test_cremainder(self):
        self._test_cop(torch.remainder, lambda x, y: x % y)

    def test_cmul(self):
        self._test_cop(torch.mul, lambda x, y: x * y)

    def test_cpow(self):
        self._test_cop(torch.pow, lambda x, y: float('nan') if x < 0 else math.pow(x, y))

    # TODO: these tests only check if it's possible to pass a return value
    # it'd be good to expand them
    def test_sum(self):
        x = torch.rand(100, 100)
        res1 = torch.sum(x, 1)
        res2 = torch.Tensor()
        torch.sum(x, 1, out=res2)
        self.assertEqual(res1, res2)

    def test_prod(self):
        x = torch.rand(100, 100)
        res1 = torch.prod(x, 1)
        res2 = torch.Tensor()
        torch.prod(x, 1, out=res2)
        self.assertEqual(res1, res2)

    def test_cumsum(self):
        x = torch.rand(100, 100)
        res1 = torch.cumsum(x, 1)
        res2 = torch.Tensor()
        torch.cumsum(x, 1, out=res2)
        self.assertEqual(res1, res2)

    def test_cumprod(self):
        x = torch.rand(100, 100)
        res1 = torch.cumprod(x, 1)
        res2 = torch.Tensor()
        torch.cumprod(x, 1, out=res2)
        self.assertEqual(res1, res2)

    def test_cross(self):
        x = torch.rand(100, 3, 100)
        y = torch.rand(100, 3, 100)
        res1 = torch.cross(x, y)
        res2 = torch.Tensor()
        torch.cross(x, y, out=res2)
        self.assertEqual(res1, res2)

    def test_zeros(self):
        res1 = torch.zeros(100, 100)
        res2 = torch.Tensor()
        torch.zeros(100, 100, out=res2)
        self.assertEqual(res1, res2)

    def test_histc(self):
        x = torch.Tensor((2, 4, 2, 2, 5, 4))
        y = torch.histc(x, 5, 1, 5)  # nbins,  min,  max
        z = torch.Tensor((0, 3, 0, 2, 1))
        self.assertEqual(y, z)

    def test_ones(self):
        res1 = torch.ones(100, 100)
        res2 = torch.Tensor()
        torch.ones(100, 100, out=res2)
        self.assertEqual(res1, res2)

    def test_diag(self):
        x = torch.rand(100, 100)
        res1 = torch.diag(x)
        res2 = torch.Tensor()
        torch.diag(x, out=res2)
        self.assertEqual(res1, res2)

    def test_eye(self):
        res1 = torch.eye(100, 100)
        res2 = torch.Tensor()
        torch.eye(100, 100, out=res2)
        self.assertEqual(res1, res2)

    def test_renorm(self):
        m1 = torch.randn(10, 5)
        res1 = torch.Tensor()

        def renorm(matrix, value, dim, max_norm):
            m1 = matrix.transpose(dim, 0).contiguous()
            # collapse non-dim dimensions.
            m2 = m1.clone().resize_(m1.size(0), int(math.floor(m1.nelement() / m1.size(0))))
            norms = m2.norm(value, 1)
            # clip
            new_norms = norms.clone()
            new_norms[torch.gt(norms, max_norm)] = max_norm
            new_norms.div_(norms.add_(1e-7))
            # renormalize
            m1.mul_(new_norms.expand_as(m1))
            return m1.transpose(dim, 0)

        # note that the axis fed to torch.renorm is different (2~=1)
        maxnorm = m1.norm(2, 1).mean()
        m2 = renorm(m1, 2, 1, maxnorm)
        m1.renorm_(2, 1, maxnorm)
        self.assertEqual(m1, m2, 1e-5)
        self.assertEqual(m1.norm(2, 0), m2.norm(2, 0), 1e-5)

        m1 = torch.randn(3, 4, 5)
        m2 = m1.transpose(1, 2).contiguous().clone().resize_(15, 4)
        maxnorm = m2.norm(2, 0).mean()
        m2 = renorm(m2, 2, 1, maxnorm)
        m1.renorm_(2, 1, maxnorm)
        m3 = m1.transpose(1, 2).contiguous().clone().resize_(15, 4)
        self.assertEqual(m3, m2)
        self.assertEqual(m3.norm(2, 0), m2.norm(2, 0))

    def test_multinomial(self):
        # with replacement
        n_row = 3
        for n_col in range(4, 5 + 1):
            prob_dist = torch.rand(n_row, n_col)
            prob_dist.select(1, n_col - 1).fill_(0)  # index n_col shouldn't be sampled
            n_sample = n_col
            sample_indices = torch.multinomial(prob_dist, n_sample, True)
            self.assertEqual(prob_dist.dim(), 2)
            self.assertEqual(sample_indices.size(1), n_sample)
            for index in product(range(n_row), range(n_sample)):
                self.assertNotEqual(sample_indices[index], n_col, "sampled an index with zero probability")

        # without replacement
        n_row = 3
        for n_col in range(4, 5 + 1):
            prob_dist = torch.rand(n_row, n_col)
            prob_dist.select(1, n_col - 1).fill_(0)  # index n_col shouldn't be sampled
            n_sample = 3
            sample_indices = torch.multinomial(prob_dist, n_sample, False)
            self.assertEqual(prob_dist.dim(), 2)
            self.assertEqual(sample_indices.size(1), n_sample)
            for i in range(n_row):
                row_samples = {}
                for j in range(n_sample):
                    sample_idx = sample_indices[i, j]
                    self.assertNotEqual(sample_idx, n_col - 1,
                                        "sampled an index with zero probability")
                    self.assertNotIn(sample_idx, row_samples, "sampled an index twice")
                    row_samples[sample_idx] = True

        # vector
        n_col = 4
        prob_dist = torch.rand(n_col)
        n_sample = n_col
        sample_indices = torch.multinomial(prob_dist, n_sample, True)
        s_dim = sample_indices.dim()
        self.assertEqual(sample_indices.dim(), 1, "wrong number of dimensions")
        self.assertEqual(prob_dist.dim(), 1, "wrong number of prob_dist dimensions")
        self.assertEqual(sample_indices.size(0), n_sample, "wrong number of samples")

    def test_range(self):
        res1 = torch.range(0, 1)
        res2 = torch.Tensor()
        torch.range(0, 1, out=res2)
        self.assertEqual(res1, res2, 0)

        # Check range for non-contiguous tensors.
        x = torch.zeros(2, 3)
        torch.range(0, 3, out=x.narrow(1, 1, 2))
        res2 = torch.Tensor(((0, 0, 1), (0, 2, 3)))
        self.assertEqual(x, res2, 1e-16)

        # Check negative
        res1 = torch.Tensor((1, 0))
        res2 = torch.Tensor()
        torch.range(1, 0, -1, out=res2)
        self.assertEqual(res1, res2, 0)

        # Equal bounds
        res1 = torch.ones(1)
        res2 = torch.Tensor()
        torch.range(1, 1, -1, out=res2)
        self.assertEqual(res1, res2, 0)
        torch.range(1, 1, 1, out=res2)
        self.assertEqual(res1, res2, 0)

        # FloatTensor
        res1 = torch.range(0.6, 0.9, 0.1, out=torch.FloatTensor())
        self.assertEqual(res1.size(0), 4)
        res1 = torch.range(1, 10, 0.3, out=torch.FloatTensor())
        self.assertEqual(res1.size(0), 31)

        # DoubleTensor
        res1 = torch.range(0.6, 0.9, 0.1, out=torch.DoubleTensor())
        self.assertEqual(res1.size(0), 4)
        res1 = torch.range(1, 10, 0.3, out=torch.DoubleTensor())
        self.assertEqual(res1.size(0), 31)

    def test_randperm(self):
        _RNGState = torch.get_rng_state()
        res1 = torch.randperm(100)
        res2 = torch.LongTensor()
        torch.set_rng_state(_RNGState)
        torch.randperm(100, out=res2)
        self.assertEqual(res1, res2, 0)

    def assertIsOrdered(self, order, x, mxx, ixx, task):
        SIZE = 4
        if order == 'descending':
            check_order = lambda a, b: a >= b
        elif order == 'ascending':
            check_order = lambda a, b: a <= b
        else:
            error('unknown order "{}", must be "ascending" or "descending"'.format(order))

        are_ordered = True
        for j, k in product(range(SIZE), range(1, SIZE)):
            self.assertTrue(check_order(mxx[j][k - 1], mxx[j][k]),
                            'torch.sort ({}) values unordered for {}'.format(order, task))

        seen = set()
        indicesCorrect = True
        size = x.size(x.dim() - 1)
        for k in range(size):
            seen.clear()
            for j in range(size):
                self.assertEqual(x[k][ixx[k][j]], mxx[k][j],
                                 'torch.sort ({}) indices wrong for {}'.format(order, task))
                seen.add(ixx[k][j])
            self.assertEqual(len(seen), size)

    def test_sort(self):
        SIZE = 4
        x = torch.rand(SIZE, SIZE)
        res1val, res1ind = torch.sort(x)

        # Test use of result tensor
        res2val = torch.Tensor()
        res2ind = torch.LongTensor()
        torch.sort(x, out=(res2val, res2ind))
        self.assertEqual(res1val, res2val, 0)
        self.assertEqual(res1ind, res2ind, 0)

        # Test sorting of random numbers
        self.assertIsOrdered('ascending', x, res2val, res2ind, 'random')

        # Test simple sort
        self.assertEqual(
            torch.sort(torch.Tensor((50, 40, 30, 20, 10)))[0],
            torch.Tensor((10, 20, 30, 40, 50)),
            0
        )

        # Test that we still have proper sorting with duplicate keys
        x = torch.floor(torch.rand(SIZE, SIZE) * 10)
        torch.sort(x, out=(res2val, res2ind))
        self.assertIsOrdered('ascending', x, res2val, res2ind, 'random with duplicate keys')

        # DESCENDING SORT
        x = torch.rand(SIZE, SIZE)
        res1val, res1ind = torch.sort(x, x.dim() - 1, True)

        # Test use of result tensor
        res2val = torch.Tensor()
        res2ind = torch.LongTensor()
        torch.sort(x, x.dim() - 1, True, out=(res2val, res2ind))
        self.assertEqual(res1val, res2val, 0)
        self.assertEqual(res1ind, res2ind, 0)

        # Test sorting of random numbers
        self.assertIsOrdered('descending', x, res2val, res2ind, 'random')

        # Test simple sort task
        self.assertEqual(
            torch.sort(torch.Tensor((10, 20, 30, 40, 50)), 0, True)[0],
            torch.Tensor((50, 40, 30, 20, 10)),
            0
        )

        # Test that we still have proper sorting with duplicate keys
        self.assertIsOrdered('descending', x, res2val, res2ind, 'random with duplicate keys')

    def test_topk(self):
        def topKViaSort(t, k, dim, dir):
            sorted, indices = t.sort(dim, dir)
            return sorted.narrow(dim, 0, k), indices.narrow(dim, 0, k)

        def compareTensors(t, res1, ind1, res2, ind2, dim):
            # Values should be exactly equivalent
            self.assertEqual(res1, res2, 0)

            # Indices might differ based on the implementation, since there is
            # no guarantee of the relative order of selection
            if not ind1.eq(ind2).all():
                # To verify that the indices represent equivalent elements,
                # gather from the input using the topk indices and compare against
                # the sort indices
                vals = t.gather(dim, ind2)
                self.assertEqual(res1, vals, 0)

        def compare(t, k, dim, dir):
            topKVal, topKInd = t.topk(k, dim, dir, True)
            sortKVal, sortKInd = topKViaSort(t, k, dim, dir)
            compareTensors(t, sortKVal, sortKInd, topKVal, topKInd, dim)

        t = torch.rand(random.randint(1, SIZE),
                       random.randint(1, SIZE),
                       random.randint(1, SIZE))

        for kTries in range(3):
            for dimTries in range(3):
                for transpose in (True, False):
                    for dir in (True, False):
                        testTensor = t
                        if transpose:
                            dim1 = random.randrange(t.ndimension())
                            dim2 = dim1
                            while dim1 == dim2:
                                dim2 = random.randrange(t.ndimension())

                            testTensor = t.transpose(dim1, dim2)

                        dim = random.randrange(testTensor.ndimension())
                        k = random.randint(1, testTensor.size(dim))
                        compare(testTensor, k, dim, dir)

    def test_topk_arguments(self):
        q = torch.randn(10, 2, 10)
        # Make sure True isn't mistakenly taken as the 2nd dimension (interpreted as 1)
        self.assertRaises(TypeError, lambda: q.topk(4, True))

    def test_kthvalue(self):
        SIZE = 50
        x = torch.rand(SIZE, SIZE, SIZE)
        x0 = x.clone()

        k = random.randint(1, SIZE)
        res1val, res1ind = torch.kthvalue(x, k)
        res2val, res2ind = torch.sort(x)

        self.assertEqual(res1val[:, :, 0], res2val[:, :, k - 1], 0)
        self.assertEqual(res1ind[:, :, 0], res2ind[:, :, k - 1], 0)
        # test use of result tensors
        k = random.randint(1, SIZE)
        res1val = torch.Tensor()
        res1ind = torch.LongTensor()
        torch.kthvalue(x, k, out=(res1val, res1ind))
        res2val, res2ind = torch.sort(x)
        self.assertEqual(res1val[:, :, 0], res2val[:, :, k - 1], 0)
        self.assertEqual(res1ind[:, :, 0], res2ind[:, :, k - 1], 0)

        # test non-default dim
        k = random.randint(1, SIZE)
        res1val, res1ind = torch.kthvalue(x, k, 0)
        res2val, res2ind = torch.sort(x, 0)
        self.assertEqual(res1val[0], res2val[k - 1], 0)
        self.assertEqual(res1ind[0], res2ind[k - 1], 0)

        # non-contiguous
        y = x.narrow(1, 0, 1)
        y0 = y.contiguous()
        k = random.randint(1, SIZE)
        res1val, res1ind = torch.kthvalue(y, k)
        res2val, res2ind = torch.kthvalue(y0, k)
        self.assertEqual(res1val, res2val, 0)
        self.assertEqual(res1ind, res2ind, 0)

        # check that the input wasn't modified
        self.assertEqual(x, x0, 0)

        # simple test case (with repetitions)
        y = torch.Tensor((3, 5, 4, 1, 1, 5))
        self.assertEqual(torch.kthvalue(y, 3)[0], torch.Tensor((3,)), 0)
        self.assertEqual(torch.kthvalue(y, 2)[0], torch.Tensor((1,)), 0)

    def test_median(self):
        for size in (155, 156):
            x = torch.rand(size, size)
            x0 = x.clone()

            res1val, res1ind = torch.median(x)
            res2val, res2ind = torch.sort(x)
            ind = int(math.floor((size + 1) / 2) - 1)

            self.assertEqual(res2val.select(1, ind), res1val.select(1, 0), 0)
            self.assertEqual(res2val.select(1, ind), res1val.select(1, 0), 0)

            # Test use of result tensor
            res2val = torch.Tensor()
            res2ind = torch.LongTensor()
            torch.median(x, out=(res2val, res2ind))
            self.assertEqual(res2val, res1val, 0)
            self.assertEqual(res2ind, res1ind, 0)

            # Test non-default dim
            res1val, res1ind = torch.median(x, 0)
            res2val, res2ind = torch.sort(x, 0)
            self.assertEqual(res1val[0], res2val[ind], 0)
            self.assertEqual(res1ind[0], res2ind[ind], 0)

            # input unchanged
            self.assertEqual(x, x0, 0)

    def test_mode(self):
        x = torch.range(1, SIZE * SIZE).clone().resize_(SIZE, SIZE)
        x[:2] = 1
        x[:, :2] = 1
        x0 = x.clone()

        # Pre-calculated results.
        res1val = torch.Tensor(SIZE, 1).fill_(1)
        # The indices are the position of the last appearance of the mode element.
        res1ind = torch.LongTensor(SIZE, 1).fill_(1)
        res1ind[0] = SIZE - 1
        res1ind[1] = SIZE - 1

        res2val, res2ind = torch.mode(x)

        self.assertEqual(res1val, res2val, 0)
        self.assertEqual(res1ind, res2ind, 0)

        # Test use of result tensor
        res2val = torch.Tensor()
        res2ind = torch.LongTensor()
        torch.mode(x, out=(res2val, res2ind))
        self.assertEqual(res1val, res2val, 0)
        self.assertEqual(res1ind, res2ind, 0)

        # Test non-default dim
        res2val, res2ind = torch.mode(x, 0)
        self.assertEqual(res1val.view(1, SIZE), res2val, 0)
        self.assertEqual(res1ind.view(1, SIZE), res2ind, 0)

        # input unchanged
        self.assertEqual(x, x0, 0)

    def test_tril(self):
        x = torch.rand(SIZE, SIZE)
        res1 = torch.tril(x)
        res2 = torch.Tensor()
        torch.tril(x, out=res2)
        self.assertEqual(res1, res2, 0)

    def test_triu(self):
        x = torch.rand(SIZE, SIZE)
        res1 = torch.triu(x)
        res2 = torch.Tensor()
        torch.triu(x, out=res2)
        self.assertEqual(res1, res2, 0)

    def test_cat(self):
        SIZE = 10
        for dim in range(3):
            x = torch.rand(13, SIZE, SIZE).transpose(0, dim)
            y = torch.rand(17, SIZE, SIZE).transpose(0, dim)
            z = torch.rand(19, SIZE, SIZE).transpose(0, dim)

            res1 = torch.cat((x, y, z), dim)
            self.assertEqual(res1.narrow(dim, 0, 13), x, 0)
            self.assertEqual(res1.narrow(dim, 13, 17), y, 0)
            self.assertEqual(res1.narrow(dim, 30, 19), z, 0)

        x = torch.randn(20, SIZE, SIZE)
        self.assertEqual(torch.cat(torch.split(x, 7)), x)
        self.assertEqual(torch.cat(torch.chunk(x, 7)), x)

        y = torch.randn(1, SIZE, SIZE)
        z = torch.cat([x, y])
        self.assertEqual(z.size(), (21, SIZE, SIZE))

        self.assertRaises(TypeError, lambda: torch.cat([]))

    def test_stack(self):
        x = torch.rand(2, 3, 4)
        y = torch.rand(2, 3, 4)
        z = torch.rand(2, 3, 4)
        for dim in range(4):
            res = torch.stack((x, y, z), dim)
            expected_size = x.size()[:dim] + (3,) + x.size()[dim:]
            self.assertEqual(res.size(), expected_size)
            self.assertEqual(res.select(dim, 0), x, 0)
            self.assertEqual(res.select(dim, 1), y, 0)
            self.assertEqual(res.select(dim, 2), z, 0)

    def test_unbind(self):
        x = torch.rand(2, 3, 4, 5)
        for dim in range(4):
            res = torch.unbind(x, dim)
            self.assertEqual(x.size(dim), len(res))
            for i in range(dim):
                self.assertEqual(x.select(dim, i), res[i])

    def test_linspace(self):
        _from = random.random()
        to = _from + random.random()
        res1 = torch.linspace(_from, to, 137)
        res2 = torch.Tensor()
        torch.linspace(_from, to, 137, out=res2)
        self.assertEqual(res1, res2, 0)
        self.assertRaises(RuntimeError, lambda: torch.linspace(0, 1, 1))
        self.assertEqual(torch.linspace(0, 0, 1), torch.zeros(1), 0)

        # Check linspace for generating with start > end.
        self.assertEqual(torch.linspace(2, 0, 3), torch.Tensor((2, 1, 0)), 0)

        # Check linspace for non-contiguous tensors.
        x = torch.zeros(2, 3)
        y = torch.linspace(0, 3, 4, out=x.narrow(1, 1, 2))
        self.assertEqual(x, torch.Tensor(((0, 0, 1), (0, 2, 3))), 0)

    def test_logspace(self):
        _from = random.random()
        to = _from + random.random()
        res1 = torch.logspace(_from, to, 137)
        res2 = torch.Tensor()
        torch.logspace(_from, to, 137, out=res2)
        self.assertEqual(res1, res2, 0)
        self.assertRaises(RuntimeError, lambda: torch.logspace(0, 1, 1))
        self.assertEqual(torch.logspace(0, 0, 1), torch.ones(1), 0)

        # Check logspace_ for generating with start > end.
        self.assertEqual(torch.logspace(1, 0, 2), torch.Tensor((10, 1)), 0)

        # Check logspace_ for non-contiguous tensors.
        x = torch.zeros(2, 3)
        y = torch.logspace(0, 3, 4, out=x.narrow(1, 1, 2))
        self.assertEqual(x, torch.Tensor(((0, 1, 10), (0, 100, 1000))), 0)

    def test_rand(self):
        torch.manual_seed(123456)
        res1 = torch.rand(SIZE, SIZE)
        res2 = torch.Tensor()
        torch.manual_seed(123456)
        torch.rand(SIZE, SIZE, out=res2)
        self.assertEqual(res1, res2)

    def test_randn(self):
        torch.manual_seed(123456)
        res1 = torch.randn(SIZE, SIZE)
        res2 = torch.Tensor()
        torch.manual_seed(123456)
        torch.randn(SIZE, SIZE, out=res2)
        self.assertEqual(res1, res2)

    @skipIfNoLapack
    def test_gesv(self):
        a = torch.Tensor(((6.80, -2.11, 5.66, 5.97, 8.23),
                          (-6.05, -3.30, 5.36, -4.44, 1.08),
                          (-0.45, 2.58, -2.70, 0.27, 9.04),
                          (8.32, 2.71, 4.35, -7.17, 2.14),
                          (-9.67, -5.14, -7.26, 6.08, -6.87))).t()
        b = torch.Tensor(((4.02, 6.19, -8.22, -7.57, -3.03),
                          (-1.56, 4.00, -8.67, 1.75, 2.86),
                          (9.81, -4.09, -4.57, -8.61, 8.99))).t()

        res1 = torch.gesv(b, a)[0]
        self.assertLessEqual(b.dist(torch.mm(a, res1)), 1e-12)
        ta = torch.Tensor()
        tb = torch.Tensor()
        res2 = torch.gesv(b, a, out=(tb, ta))[0]
        res3 = torch.gesv(b, a, out=(b, a))[0]
        self.assertEqual(res1, tb)
        self.assertEqual(res1, b)
        self.assertEqual(res1, res2)
        self.assertEqual(res1, res3)

        # test reuse
        res1 = torch.gesv(b, a)[0]
        ta = torch.Tensor()
        tb = torch.Tensor()
        torch.gesv(b, a, out=(tb, ta))[0]
        self.assertEqual(res1, tb)
        torch.gesv(b, a, out=(tb, ta))[0]
        self.assertEqual(res1, tb)

    @skipIfNoLapack
    def test_qr(self):

        # Since the QR decomposition is unique only up to the signs of the rows of
        # R, we must ensure these are positive before doing the comparison.
        def canonicalize(q, r):
            d = r.diag().sign().diag()
            return torch.mm(q, d), torch.mm(d, r)

        def canon_and_check(q, r, expected_q, expected_r):
            q_canon, r_canon = canonicalize(q, r)
            expected_q_canon, expected_r_canon = canonicalize(expected_q, expected_r)
            self.assertEqual(q_canon, expected_q_canon)
            self.assertEqual(r_canon, expected_r_canon)

        def check_qr(a, expected_q, expected_r):
            # standard invocation
            q, r = torch.qr(a)
            canon_and_check(q, r, expected_q, expected_r)

            # in-place
            q, r = torch.Tensor(), torch.Tensor()
            torch.qr(a, out=(q, r))
            canon_and_check(q, r, expected_q, expected_r)

            # manually calculate qr using geqrf and orgqr
            m = a.size(0)
            n = a.size(1)
            k = min(m, n)
            result, tau = torch.geqrf(a)
            self.assertEqual(result.size(0), m)
            self.assertEqual(result.size(1), n)
            self.assertEqual(tau.size(0), k)
            r = torch.triu(result.narrow(0, 0, k))
            q, _ = torch.orgqr(result, tau)
            q, r = q.narrow(1, 0, k), r
            canon_and_check(q, r, expected_q, expected_r)

        # check square case
        a = torch.Tensor(((1, 2, 3), (4, 5, 6), (7, 8, 10)))

        expected_q = torch.Tensor((
            (-1.230914909793328e-01, 9.045340337332914e-01, 4.082482904638621e-01),
            (-4.923659639173310e-01, 3.015113445777629e-01, -8.164965809277264e-01),
            (-8.616404368553292e-01, -3.015113445777631e-01, 4.082482904638634e-01)))
        expected_r = torch.Tensor((
            (-8.124038404635959e+00, -9.601136296387955e+00, -1.193987e+01),
            (0.000000000000000e+00, 9.045340337332926e-01, 1.507557e+00),
            (0.000000000000000e+00, 0.000000000000000e+00, 4.082483e-01)))

        check_qr(a, expected_q, expected_r)

        # check rectangular thin
        a = torch.Tensor((
            (1, 2, 3),
            (4, 5, 6),
            (7, 8, 9),
            (10, 11, 13),
        ))
        expected_q = torch.Tensor((
            (-0.0776150525706334, -0.833052161400748, 0.3651483716701106),
            (-0.3104602102825332, -0.4512365874254053, -0.1825741858350556),
            (-0.5433053679944331, -0.0694210134500621, -0.7302967433402217),
            (-0.7761505257063329, 0.3123945605252804, 0.5477225575051663)
        ))
        expected_r = torch.Tensor((
            (-12.8840987267251261, -14.5916298832790581, -17.0753115655393231),
            (0, -1.0413152017509357, -1.770235842976589),
            (0, 0, 0.5477225575051664)
        ))

        check_qr(a, expected_q, expected_r)

        # check rectangular fat
        a = torch.Tensor((
            (1, 2, 3, 4),
            (5, 6, 7, 8),
            (9, 10, 11, 13)
        ))
        expected_q = torch.Tensor((
            (-0.0966736489045663, 0.907737593658436, 0.4082482904638653),
            (-0.4833682445228317, 0.3157348151855452, -0.8164965809277254),
            (-0.870062840141097, -0.2762679632873518, 0.4082482904638621)
        ))
        expected_r = torch.Tensor((
            (-1.0344080432788603e+01, -1.1794185166357092e+01,
             -1.3244289899925587e+01, -1.5564457473635180e+01),
            (0.0000000000000000e+00, 9.4720444555662542e-01,
             1.8944088911132546e+00, 2.5653453733825331e+00),
            (0.0000000000000000e+00, 0.0000000000000000e+00,
             1.5543122344752192e-15, 4.0824829046386757e-01)
        ))
        check_qr(a, expected_q, expected_r)

    @skipIfNoLapack
    def test_ormqr(self):
        mat1 = torch.randn(10, 10)
        mat2 = torch.randn(10, 10)
        q, r = torch.qr(mat1)
        m, tau = torch.geqrf(mat1)

        res1 = torch.mm(q, mat2)
        res2, _ = torch.ormqr(m, tau, mat2)
        self.assertEqual(res1, res2)

        res1 = torch.mm(mat2, q)
        res2, _ = torch.ormqr(m, tau, mat2, False)
        self.assertEqual(res1, res2)

        res1 = torch.mm(q.t(), mat2)
        res2, _ = torch.ormqr(m, tau, mat2, True, True)
        self.assertEqual(res1, res2)

        res1 = torch.mm(mat2, q.t())
        res2, _ = torch.ormqr(m, tau, mat2, False, True)
        self.assertEqual(res1, res2)

    @skipIfNoLapack
    def test_trtrs(self):
        a = torch.Tensor(((6.80, -2.11, 5.66, 5.97, 8.23),
                          (-6.05, -3.30, 5.36, -4.44, 1.08),
                          (-0.45, 2.58, -2.70, 0.27, 9.04),
                          (8.32, 2.71, 4.35, -7.17, 2.14),
                          (-9.67, -5.14, -7.26, 6.08, -6.87))).t()
        b = torch.Tensor(((4.02, 6.19, -8.22, -7.57, -3.03),
                          (-1.56, 4.00, -8.67, 1.75, 2.86),
                          (9.81, -4.09, -4.57, -8.61, 8.99))).t()

        U = torch.triu(a)
        L = torch.tril(a)

        # solve Ux = b
        x = torch.trtrs(b, U)[0]
        self.assertLessEqual(b.dist(torch.mm(U, x)), 1e-12)
        x = torch.trtrs(b, U, True, False, False)[0]
        self.assertLessEqual(b.dist(torch.mm(U, x)), 1e-12)

        # solve Lx = b
        x = torch.trtrs(b, L, False)[0]
        self.assertLessEqual(b.dist(torch.mm(L, x)), 1e-12)
        x = torch.trtrs(b, L, False, False, False)[0]
        self.assertLessEqual(b.dist(torch.mm(L, x)), 1e-12)

        # solve U'x = b
        x = torch.trtrs(b, U, True, True)[0]
        self.assertLessEqual(b.dist(torch.mm(U.t(), x)), 1e-12)
        x = torch.trtrs(b, U, True, True, False)[0]
        self.assertLessEqual(b.dist(torch.mm(U.t(), x)), 1e-12)

        # solve U'x = b by manual transposition
        y = torch.trtrs(b, U.t(), False, False)[0]
        self.assertLessEqual(x.dist(y), 1e-12)

        # solve L'x = b
        x = torch.trtrs(b, L, False, True)[0]
        self.assertLessEqual(b.dist(torch.mm(L.t(), x)), 1e-12)
        x = torch.trtrs(b, L, False, True, False)[0]
        self.assertLessEqual(b.dist(torch.mm(L.t(), x)), 1e-12)

        # solve L'x = b by manual transposition
        y = torch.trtrs(b, L.t(), True, False)[0]
        self.assertLessEqual(x.dist(y), 1e-12)

        # test reuse
        res1 = torch.trtrs(b, a)[0]
        ta = torch.Tensor()
        tb = torch.Tensor()
        torch.trtrs(b, a, out=(tb, ta))
        self.assertEqual(res1, tb, 0)
        tb.zero_()
        torch.trtrs(b, a, out=(tb, ta))
        self.assertEqual(res1, tb, 0)

    @skipIfNoLapack
    def test_gels(self):
        def _test(a, b, expectedNorm):
            a_copy = a.clone()
            b_copy = b.clone()
            res1 = torch.gels(b, a)[0]
            self.assertEqual(a, a_copy, 0)
            self.assertEqual(b, b_copy, 0)
            self.assertEqual((torch.mm(a, res1) - b).norm(), expectedNorm, 1e-8)

            ta = torch.Tensor()
            tb = torch.Tensor()
            res2 = torch.gels(b, a, out=(tb, ta))[0]
            self.assertEqual(a, a_copy, 0)
            self.assertEqual(b, b_copy, 0)
            self.assertEqual((torch.mm(a, res1) - b).norm(), expectedNorm, 1e-8)

            res3 = torch.gels(b, a, out=(b, a))[0]
            self.assertEqual((torch.mm(a_copy, b) - b_copy).norm(), expectedNorm, 1e-8)
            self.assertEqual(res1, tb, 0)
            self.assertEqual(res1, b, 0)
            self.assertEqual(res1, res2, 0)
            self.assertEqual(res1, res3, 0)

        # basic test
        expectedNorm = 0
        a = torch.Tensor(((1.44, -9.96, -7.55, 8.34),
                          (-7.84, -0.28, 3.24, 8.09),
                          (-4.39, -3.24, 6.27, 5.28),
                          (4.53, 3.83, -6.64, 2.06))).t()
        b = torch.Tensor(((8.58, 8.26, 8.48, -5.28),
                          (9.35, -4.43, -0.70, -0.26))).t()
        _test(a, b, expectedNorm)

        # test overderemined
        expectedNorm = 17.390200628863
        a = torch.Tensor(((1.44, -9.96, -7.55, 8.34, 7.08, -5.45),
                          (-7.84, -0.28, 3.24, 8.09, 2.52, -5.70),
                          (-4.39, -3.24, 6.27, 5.28, 0.74, -1.19),
                          (4.53, 3.83, -6.64, 2.06, -2.47, 4.70))).t()
        b = torch.Tensor(((8.58, 8.26, 8.48, -5.28, 5.72, 8.93),
                          (9.35, -4.43, -0.70, -0.26, -7.36, -2.52))).t()
        _test(a, b, expectedNorm)

        # test underdetermined
        expectedNorm = 0
        a = torch.Tensor(((1.44, -9.96, -7.55),
                          (-7.84, -0.28, 3.24),
                          (-4.39, -3.24, 6.27),
                          (4.53, 3.83, -6.64))).t()
        b = torch.Tensor(((8.58, 8.26, 8.48),
                          (9.35, -4.43, -0.70))).t()
        _test(a, b, expectedNorm)

        # test reuse
        expectedNorm = 0
        a = torch.Tensor(((1.44, -9.96, -7.55, 8.34),
                          (-7.84, -0.28, 3.24, 8.09),
                          (-4.39, -3.24, 6.27, 5.28),
                          (4.53, 3.83, -6.64, 2.06))).t()
        b = torch.Tensor(((8.58, 8.26, 8.48, -5.28),
                          (9.35, -4.43, -0.70, -0.26))).t()
        ta = torch.Tensor()
        tb = torch.Tensor()
        torch.gels(b, a, out=(tb, ta))
        self.assertEqual((torch.mm(a, tb) - b).norm(), expectedNorm, 1e-8)
        torch.gels(b, a, out=(tb, ta))
        self.assertEqual((torch.mm(a, tb) - b).norm(), expectedNorm, 1e-8)
        torch.gels(b, a, out=(tb, ta))
        self.assertEqual((torch.mm(a, tb) - b).norm(), expectedNorm, 1e-8)

    @skipIfNoLapack
    def test_eig(self):
        a = torch.Tensor(((1.96, 0.00, 0.00, 0.00, 0.00),
                          (-6.49, 3.80, 0.00, 0.00, 0.00),
                          (-0.47, -6.39, 4.17, 0.00, 0.00),
                          (-7.20, 1.50, -1.51, 5.70, 0.00),
                          (-0.65, -6.34, 2.67, 1.80, -7.10))).t().contiguous()
        e = torch.eig(a)[0]
        ee, vv = torch.eig(a, True)
        te = torch.Tensor()
        tv = torch.Tensor()
        eee, vvv = torch.eig(a, True, out=(te, tv))
        self.assertEqual(e, ee, 1e-12)
        self.assertEqual(ee, eee, 1e-12)
        self.assertEqual(ee, te, 1e-12)
        self.assertEqual(vv, vvv, 1e-12)
        self.assertEqual(vv, tv, 1e-12)

        # test reuse
        X = torch.randn(4, 4)
        X = torch.mm(X.t(), X)
        e, v = torch.zeros(4, 2), torch.zeros(4, 4)
        torch.eig(X, True, out=(e, v))
        Xhat = torch.mm(torch.mm(v, torch.diag(e.select(1, 0))), v.t())
        self.assertEqual(X, Xhat, 1e-8, 'VeV\' wrong')
        self.assertFalse(v.is_contiguous(), 'V is contiguous')

        torch.eig(X, True, out=(e, v))
        Xhat = torch.mm(v, torch.mm(e.select(1, 0).diag(), v.t()))
        self.assertEqual(X, Xhat, 1e-8, 'VeV\' wrong')
        self.assertFalse(v.is_contiguous(), 'V is contiguous')

        # test non-contiguous
        X = torch.randn(4, 4)
        X = torch.mm(X.t(), X)
        e = torch.zeros(4, 2, 2)[:, 1]
        v = torch.zeros(4, 2, 4)[:, 1]
        self.assertFalse(v.is_contiguous(), 'V is contiguous')
        self.assertFalse(e.is_contiguous(), 'E is contiguous')
        torch.eig(X, True, out=(e, v))
        Xhat = torch.mm(torch.mm(v, torch.diag(e.select(1, 0))), v.t())
        self.assertEqual(X, Xhat, 1e-8, 'VeV\' wrong')

    @skipIfNoLapack
    def test_symeig(self):
        xval = torch.rand(100, 3)
        cov = torch.mm(xval.t(), xval)
        rese = torch.zeros(3)
        resv = torch.zeros(3, 3)

        # First call to symeig
        self.assertTrue(resv.is_contiguous(), 'resv is not contiguous')
        torch.symeig(cov.clone(), True, out=(rese, resv))
        ahat = torch.mm(torch.mm(resv, torch.diag(rese)), resv.t())
        self.assertEqual(cov, ahat, 1e-8, 'VeV\' wrong')

        # Second call to symeig
        self.assertFalse(resv.is_contiguous(), 'resv is contiguous')
        torch.symeig(cov.clone(), True, out=(rese, resv))
        ahat = torch.mm(torch.mm(resv, torch.diag(rese)), resv.t())
        self.assertEqual(cov, ahat, 1e-8, 'VeV\' wrong')

        # test non-contiguous
        X = torch.rand(5, 5)
        X = X.t() * X
        e = torch.zeros(4, 2).select(1, 1)
        v = torch.zeros(4, 2, 4)[:, 1]
        self.assertFalse(v.is_contiguous(), 'V is contiguous')
        self.assertFalse(e.is_contiguous(), 'E is contiguous')
        torch.symeig(X, True, out=(e, v))
        Xhat = torch.mm(torch.mm(v, torch.diag(e)), v.t())
        self.assertEqual(X, Xhat, 1e-8, 'VeV\' wrong')

    @skipIfNoLapack
    def test_svd(self):
        a = torch.Tensor(((8.79, 6.11, -9.15, 9.57, -3.49, 9.84),
                          (9.93, 6.91, -7.93, 1.64, 4.02, 0.15),
                          (9.83, 5.04, 4.86, 8.83, 9.80, -8.99),
                          (5.45, -0.27, 4.85, 0.74, 10.00, -6.02),
                          (3.16, 7.98, 3.01, 5.80, 4.27, -5.31))).t().clone()
        u, s, v = torch.svd(a)
        uu = torch.Tensor()
        ss = torch.Tensor()
        vv = torch.Tensor()
        uuu, sss, vvv = torch.svd(a, out=(uu, ss, vv))
        self.assertEqual(u, uu, 0, 'torch.svd')
        self.assertEqual(u, uuu, 0, 'torch.svd')
        self.assertEqual(s, ss, 0, 'torch.svd')
        self.assertEqual(s, sss, 0, 'torch.svd')
        self.assertEqual(v, vv, 0, 'torch.svd')
        self.assertEqual(v, vvv, 0, 'torch.svd')

        # test reuse
        X = torch.randn(4, 4)
        U, S, V = torch.svd(X)
        Xhat = torch.mm(U, torch.mm(S.diag(), V.t()))
        self.assertEqual(X, Xhat, 1e-8, 'USV\' wrong')

        self.assertFalse(U.is_contiguous(), 'U is contiguous')
        torch.svd(X, out=(U, S, V))
        Xhat = torch.mm(U, torch.mm(S.diag(), V.t()))
        self.assertEqual(X, Xhat, 1e-8, 'USV\' wrong')

        # test non-contiguous
        X = torch.randn(5, 5)
        U = torch.zeros(5, 2, 5)[:, 1]
        S = torch.zeros(5, 2)[:, 1]
        V = torch.zeros(5, 2, 5)[:, 1]

        self.assertFalse(U.is_contiguous(), 'U is contiguous')
        self.assertFalse(S.is_contiguous(), 'S is contiguous')
        self.assertFalse(V.is_contiguous(), 'V is contiguous')
        torch.svd(X, out=(U, S, V))
        Xhat = torch.mm(U, torch.mm(S.diag(), V.t()))
        self.assertEqual(X, Xhat, 1e-8, 'USV\' wrong')

    @skipIfNoLapack
    def test_inverse(self):
        M = torch.randn(5, 5)
        MI = torch.inverse(M)
        E = torch.eye(5)
        self.assertFalse(MI.is_contiguous(), 'MI is contiguous')
        self.assertEqual(E, torch.mm(M, MI), 1e-8, 'inverse value')
        self.assertEqual(E, torch.mm(MI, M), 1e-8, 'inverse value')

        MII = torch.Tensor(5, 5)
        torch.inverse(M, out=MII)
        self.assertFalse(MII.is_contiguous(), 'MII is contiguous')
        self.assertEqual(MII, MI, 0, 'inverse value in-place')
        # second call, now that MII is transposed
        torch.inverse(M, out=MII)
        self.assertFalse(MII.is_contiguous(), 'MII is contiguous')
        self.assertEqual(MII, MI, 0, 'inverse value in-place')

    @unittest.skip("Not implemented yet")
    def test_conv2(self):
        x = torch.rand(math.floor(torch.uniform(50, 100)), math.floor(torch.uniform(50, 100)))
        k = torch.rand(math.floor(torch.uniform(10, 20)), math.floor(torch.uniform(10, 20)))
        imvc = torch.conv2(x, k)
        imvc2 = torch.conv2(x, k, 'V')
        imfc = torch.conv2(x, k, 'F')

        ki = k.clone()
        ks = k.storage()
        kis = ki.storage()
        for i in range(ks.size() - 1, 0, -1):
            kis[ks.size() - i + 1] = ks[i]
        # for i=ks.size(), 1, -1 do kis[ks.size()-i+1]=ks[i] end
        imvx = torch.xcorr2(x, ki)
        imvx2 = torch.xcorr2(x, ki, 'V')
        imfx = torch.xcorr2(x, ki, 'F')

        self.assertEqual(imvc, imvc2, 0, 'torch.conv2')
        self.assertEqual(imvc, imvx, 0, 'torch.conv2')
        self.assertEqual(imvc, imvx2, 0, 'torch.conv2')
        self.assertEqual(imfc, imfx, 0, 'torch.conv2')
        self.assertLessEqual(math.abs(x.dot(x) - torch.xcorr2(x, x)[0][0]), 1e-10, 'torch.conv2')

        xx = torch.Tensor(2, x.size(1), x.size(2))
        xx[1].copy_(x)
        xx[2].copy_(x)
        kk = torch.Tensor(2, k.size(1), k.size(2))
        kk[1].copy_(k)
        kk[2].copy_(k)

        immvc = torch.conv2(xx, kk)
        immvc2 = torch.conv2(xx, kk, 'V')
        immfc = torch.conv2(xx, kk, 'F')

        self.assertEqual(immvc[0], immvc[1], 0, 'torch.conv2')
        self.assertEqual(immvc[0], imvc, 0, 'torch.conv2')
        self.assertEqual(immvc2[0], imvc2, 0, 'torch.conv2')
        self.assertEqual(immfc[0], immfc[1], 0, 'torch.conv2')
        self.assertEqual(immfc[0], imfc, 0, 'torch.conv2')

    @unittest.skip("Not implemented yet")
    def test_conv3(self):
        x = torch.rand(math.floor(torch.uniform(20, 40)),
                       math.floor(torch.uniform(20, 40)),
                       math.floor(torch.uniform(20, 40)))
        k = torch.rand(math.floor(torch.uniform(5, 10)),
                       math.floor(torch.uniform(5, 10)),
                       math.floor(torch.uniform(5, 10)))
        imvc = torch.conv3(x, k)
        imvc2 = torch.conv3(x, k, 'V')
        imfc = torch.conv3(x, k, 'F')

        ki = k.clone()
        ks = k.storage()
        kis = ki.storage()
        for i in range(ks.size() - 1, 0, -1):
            kis[ks.size() - i + 1] = ks[i]
        imvx = torch.xcorr3(x, ki)
        imvx2 = torch.xcorr3(x, ki, 'V')
        imfx = torch.xcorr3(x, ki, 'F')

        self.assertEqual(imvc, imvc2, 0, 'torch.conv3')
        self.assertEqual(imvc, imvx, 0, 'torch.conv3')
        self.assertEqual(imvc, imvx2, 0, 'torch.conv3')
        self.assertEqual(imfc, imfx, 0, 'torch.conv3')
        self.assertLessEqual(math.abs(x.dot(x) - torch.xcorr3(x, x)[0][0][0]), 4e-10, 'torch.conv3')

        xx = torch.Tensor(2, x.size(1), x.size(2), x.size(3))
        xx[1].copy_(x)
        xx[2].copy_(x)
        kk = torch.Tensor(2, k.size(1), k.size(2), k.size(3))
        kk[1].copy_(k)
        kk[2].copy_(k)

        immvc = torch.conv3(xx, kk)
        immvc2 = torch.conv3(xx, kk, 'V')
        immfc = torch.conv3(xx, kk, 'F')

        self.assertEqual(immvc[0], immvc[1], 0, 'torch.conv3')
        self.assertEqual(immvc[0], imvc, 0, 'torch.conv3')
        self.assertEqual(immvc2[0], imvc2, 0, 'torch.conv3')
        self.assertEqual(immfc[0], immfc[1], 0, 'torch.conv3')
        self.assertEqual(immfc[0], imfc, 0, 'torch.conv3')

    @unittest.skip("Not implemented yet")
    def _test_conv_corr_eq(self, fn, fn_2_to_3):
        ix = math.floor(random.randint(20, 40))
        iy = math.floor(random.randint(20, 40))
        iz = math.floor(random.randint(20, 40))
        kx = math.floor(random.randint(5, 10))
        ky = math.floor(random.randint(5, 10))
        kz = math.floor(random.randint(5, 10))

        x = torch.rand(ix, iy, iz)
        k = torch.rand(kx, ky, kz)

        o3 = fn(x, k)
        o32 = torch.zeros(o3.size())
        fn_2_to_3(x, k, o3, o32)
        self.assertEqual(o3, o32)

    @unittest.skip("Not implemented yet")
    def test_xcorr3_xcorr2_eq(self):
        def reference(x, k, o3, o32):
            for i in range(o3.size(1)):
                for j in range(k.size(1)):
                    o32[i].add(torch.xcorr2(x[i + j - 1], k[j]))
        self._test_conv_corr_eq(lambda x, k: torch.xcorr3(x, k), reference)

    @unittest.skip("Not implemented yet")
    def test_xcorr3_xcorr2_eq(self):
        def reference(x, k, o3, o32):
            for i in range(x.size(1)):
                for j in range(k.size(1)):
                    o32[i].add(torch.xcorr2(x[i], k[k.size(1) - j + 1], 'F'))
        self._test_conv_corr_eq(lambda x, k: torch.xcorr3(x, k, 'F'), reference)

    @unittest.skip("Not implemented yet")
    def test_conv3_conv2_eq(self):
        def reference(x, k, o3, o32):
            for i in range(o3.size(1)):
                for j in range(k.size(1)):
                    o32[i].add(torch.conv2(x[i + j - 1], k[k.size(1) - j + 1]))
        self._test_conv_corr_eq(lambda x, k: torch.conv3(x, k), reference)

    @unittest.skip("Not implemented yet")
    def test_fconv3_fconv2_eq(self):
        def reference(x, k, o3, o32):
            for i in range(o3.size(1)):
                for j in range(k.size(1)):
                    o32[i + j - 1].add(torch.conv2(x[i], k[j], 'F'))
        self._test_conv_corr_eq(lambda x, k: torch.conv3(x, k, 'F'), reference)

    def test_logical(self):
        x = torch.rand(100, 100) * 2 - 1
        xx = x.clone()

        xgt = torch.gt(x, 1)
        xlt = torch.lt(x, 1)

        xeq = torch.eq(x, 1)
        xne = torch.ne(x, 1)

        neqs = xgt + xlt
        all = neqs + xeq
        self.assertEqual(neqs.sum(), xne.sum(), 0)
        self.assertEqual(x.nelement(), all.sum())

    def test_RNGState(self):
        state = torch.get_rng_state()
        stateCloned = state.clone()
        before = torch.rand(1000)

        self.assertEqual(state.ne(stateCloned).long().sum(), 0, 0)

        torch.set_rng_state(state)
        after = torch.rand(1000)
        self.assertEqual(before, after, 0)

    def test_RNGStateAliasing(self):
        # Fork the random number stream at this point
        gen = torch.Generator()
        gen.set_state(torch.get_rng_state())
        self.assertEqual(gen.get_state(), torch.get_rng_state())

        target_value = torch.rand(1000)
        # Dramatically alter the internal state of the main generator
        _ = torch.rand(100000)
        forked_value = torch.rand(gen, 1000)
        self.assertEqual(target_value, forked_value, 0, "RNG has not forked correctly.")

    def test_boxMullerState(self):
        torch.manual_seed(123)
        odd_number = 101
        seeded = torch.randn(odd_number)
        state = torch.get_rng_state()
        midstream = torch.randn(odd_number)
        torch.set_rng_state(state)
        repeat_midstream = torch.randn(odd_number)
        torch.manual_seed(123)
        reseeded = torch.randn(odd_number)
        self.assertEqual(midstream, repeat_midstream, 0,
                         'get_rng_state/set_rng_state not generating same sequence of normally distributed numbers')
        self.assertEqual(seeded, reseeded, 0,
                         'repeated calls to manual_seed not generating same sequence of normally distributed numbers')

    def test_manual_seed(self):
        rng_state = torch.get_rng_state()
        torch.manual_seed(2)
        x = torch.randn(100)
        self.assertEqual(torch.initial_seed(), 2)
        torch.manual_seed(2)
        y = torch.randn(100)
        self.assertEqual(x, y)
        torch.set_rng_state(rng_state)

    @skipIfNoLapack
    def test_cholesky(self):
        x = torch.rand(10, 10) + 1e-1
        A = torch.mm(x, x.t())

        # default Case
        C = torch.potrf(A)
        B = torch.mm(C.t(), C)
        self.assertEqual(A, B, 1e-14)

        # test Upper Triangular
        U = torch.potrf(A, True)
        B = torch.mm(U.t(), U)
        self.assertEqual(A, B, 1e-14, 'potrf (upper) did not allow rebuilding the original matrix')

        # test Lower Triangular
        L = torch.potrf(A, False)
        B = torch.mm(L, L.t())
        self.assertEqual(A, B, 1e-14, 'potrf (lower) did not allow rebuilding the original matrix')

    @skipIfNoLapack
    def test_potrs(self):
        a = torch.Tensor(((6.80, -2.11, 5.66, 5.97, 8.23),
                          (-6.05, -3.30, 5.36, -4.44, 1.08),
                          (-0.45, 2.58, -2.70, 0.27, 9.04),
                          (8.32, 2.71, 4.35, -7.17, 2.14),
                          (-9.67, -5.14, -7.26, 6.08, -6.87))).t()
        b = torch.Tensor(((4.02, 6.19, -8.22, -7.57, -3.03),
                          (-1.56, 4.00, -8.67, 1.75, 2.86),
                          (9.81, -4.09, -4.57, -8.61, 8.99))).t()

        # make sure 'a' is symmetric PSD
        a = torch.mm(a, a.t())

        # upper Triangular Test
        U = torch.potrf(a)
        x = torch.potrs(b, U)
        self.assertLessEqual(b.dist(torch.mm(a, x)), 1e-12)

        # lower Triangular Test
        L = torch.potrf(a, False)
        x = torch.potrs(b, L, False)
        self.assertLessEqual(b.dist(torch.mm(a, x)), 1e-12)

    @skipIfNoLapack
    def tset_potri(self):
        a = torch.Tensor(((6.80, -2.11, 5.66, 5.97, 8.23),
                          (-6.05, -3.30, 5.36, -4.44, 1.08),
                          (-0.45, 2.58, -2.70, 0.27, 9.04),
                          (8.32, 2.71, 4.35, -7.17, 2.14),
                          (-9.67, -5.14, -7.26, 6.08, -6.87))).t()

        # make sure 'a' is symmetric PSD
        a = a * a.t()

        # compute inverse directly
        inv0 = torch.inverse(a)

        # default case
        chol = torch.potrf(a)
        inv1 = torch.potri(chol)
        self.assertLessEqual(inv0.dist(inv1), 1e-12)

        # upper Triangular Test
        chol = torch.potrf(a, 'U')
        inv1 = torch.potri(chol, 'U')
        self.assertLessEqual(inv0.dist(inv1), 1e-12)

        # lower Triangular Test
        chol = torch.potrf(a, 'L')
        inv1 = torch.potri(chol, 'L')
        self.assertLessEqual(inv0.dist(inv1), 1e-12)

    @skipIfNoLapack
    def test_pstrf(self):
        def checkPsdCholesky(a, uplo, inplace):
            if inplace:
                u = torch.Tensor(a.size())
                piv = torch.IntTensor(a.size(0))
                kwargs = {'out': (u, piv)}
            else:
                kwargs = {}
            args = [a]

            if uplo is not None:
                args += [uplo]

            u, piv = torch.pstrf(*args, **kwargs)

            if uplo is False:
                a_reconstructed = torch.mm(u, u.t())
            else:
                a_reconstructed = torch.mm(u.t(), u)

            piv = piv.long()
            a_permuted = a.index_select(0, piv).index_select(1, piv)
            self.assertEqual(a_permuted, a_reconstructed, 1e-14)

        dimensions = ((5, 1), (5, 3), (5, 5), (10, 10))
        for dim in dimensions:
            m = torch.Tensor(*dim).uniform_()
            a = torch.mm(m, m.t())
            # add a small number to the diagonal to make the matrix numerically positive semidefinite
            for i in range(m.size(0)):
                a[i][i] = a[i][i] + 1e-7
            for inplace in (True, False):
                for uplo in (None, True, False):
                    checkPsdCholesky(a, uplo, inplace)

    def test_numel(self):
        b = torch.ByteTensor(3, 100, 100)
        self.assertEqual(b.nelement(), 3 * 100 * 100)
        self.assertEqual(b.numel(), 3 * 100 * 100)

    def _consecutive(self, size, start=1):
        sequence = torch.ones(int(torch.Tensor(size).prod(0)[0])).cumsum(0)
        sequence.add_(start - 1)
        return sequence.resize_(*size)

    def test_index(self):
        reference = self._consecutive((3, 3, 3))
        self.assertEqual(reference[0], self._consecutive((3, 3)), 0)
        self.assertEqual(reference[1], self._consecutive((3, 3), 10), 0)
        self.assertEqual(reference[2], self._consecutive((3, 3), 19), 0)
        self.assertEqual(reference[0, 1], self._consecutive((3,), 4), 0)
        self.assertEqual(reference[0:2], self._consecutive((2, 3, 3)), 0)
        self.assertEqual(reference[2, 2, 2], 27, 0)
        self.assertEqual(reference[:], self._consecutive((3, 3, 3)), 0)

        # indexing with Ellipsis
        self.assertEqual(reference[..., 2], torch.Tensor([[3, 6, 9],
                                                          [12, 15, 18],
                                                          [21, 24, 27]]), 0)
        self.assertEqual(reference[0, ..., 2], torch.Tensor([3, 6, 9]), 0)
        self.assertEqual(reference[..., 2], reference[:, :, 2], 0)
        self.assertEqual(reference[0, ..., 2], reference[0, :, 2], 0)
        self.assertEqual(reference[0, 2, ...], reference[0, 2], 0)
        self.assertEqual(reference[..., 2, 2, 2], 27, 0)
        self.assertEqual(reference[2, ..., 2, 2], 27, 0)
        self.assertEqual(reference[2, 2, ..., 2], 27, 0)
        self.assertEqual(reference[2, 2, 2, ...], 27, 0)
        self.assertEqual(reference[...], reference, 0)

        reference_5d = self._consecutive((3, 3, 3, 3, 3))
        self.assertEqual(reference_5d[..., 1, 0], reference_5d[:, :, :, 1, 0], 0)
        self.assertEqual(reference_5d[2, ..., 1, 0], reference_5d[2, :, :, 1, 0], 0)
        self.assertEqual(reference_5d[2, 1, 0, ..., 1], reference_5d[2, 1, 0, :, 1], 0)
        self.assertEqual(reference_5d[...], reference_5d, 0)

        # LongTensor indexing
        reference = self._consecutive((5, 5, 5))
        idx = torch.LongTensor([2, 4])
        self.assertEqual(reference[idx], torch.stack([reference[2], reference[4]]))
        # TODO: enable one indexing is implemented like in numpy
        # self.assertEqual(reference[2, idx], torch.stack([reference[2, 2], reference[2, 4]]))
        # self.assertEqual(reference[3, idx, 1], torch.stack([reference[3, 2], reference[3, 4]])[:, 1])

        # None indexing
        self.assertEqual(reference[2, None], reference[2].unsqueeze(0))
        self.assertEqual(reference[2, None, None], reference[2].unsqueeze(0).unsqueeze(0))
        self.assertEqual(reference[2:4, None], reference[2:4].unsqueeze(1))
        self.assertEqual(reference[None, 2, None, None], reference.unsqueeze(0)[:, 2].unsqueeze(0).unsqueeze(0))
        self.assertEqual(reference[None, 2:5, None, None], reference.unsqueeze(0)[:, 2:5].unsqueeze(2).unsqueeze(2))

        # indexing with step
        reference = self._consecutive((10, 10, 10))
        self.assertEqual(reference[1:5:2], torch.stack([reference[1], reference[3]], 0))
        self.assertEqual(reference[1:6:2], torch.stack([reference[1], reference[3], reference[5]], 0))
        self.assertEqual(reference[1:9:4], torch.stack([reference[1], reference[5]], 0))
        self.assertEqual(reference[2:4, 1:5:2], torch.stack([reference[2:4, 1], reference[2:4, 3]], 1))
        self.assertEqual(reference[3, 1:6:2], torch.stack([reference[3, 1], reference[3, 3], reference[3, 5]], 0))
        self.assertEqual(reference[None, 2, 1:9:4], torch.stack([reference[2, 1], reference[2, 5]], 0).unsqueeze(0))
        self.assertEqual(reference[:, 2, 1:6:2],
                         torch.stack([reference[:, 2, 1], reference[:, 2, 3], reference[:, 2, 5]], 1))

        self.assertRaises(ValueError, lambda: reference[1:9:0])
        self.assertRaises(ValueError, lambda: reference[1:9:-1])

        self.assertRaises(IndexError, lambda: reference[1, 1, 1, 1])
        self.assertRaises(IndexError, lambda: reference[1, 1, 1, 1:1])
        self.assertRaises(IndexError, lambda: reference[3, 3, 3, 3, 3, 3, 3, 3])

        self.assertRaises(TypeError, lambda: reference[0.0])
        self.assertRaises(TypeError, lambda: reference[0.0:2.0])
        self.assertRaises(TypeError, lambda: reference[0.0, 0.0:2.0])
        self.assertRaises(TypeError, lambda: reference[0.0, :, 0.0:2.0])
        self.assertRaises(TypeError, lambda: reference[0.0, ..., 0.0:2.0])
        self.assertRaises(TypeError, lambda: reference[0.0, :, 0.0])

    def test_newindex(self):
        reference = self._consecutive((3, 3, 3))
        # This relies on __index__() being correct - but we have separate tests for that

        def checkPartialAssign(index):
            reference = torch.zeros(3, 3, 3)
            reference[index] = self._consecutive((3, 3, 3))[index]
            self.assertEqual(reference[index], self._consecutive((3, 3, 3))[index], 0)
            reference[index] = 0
            self.assertEqual(reference, torch.zeros(3, 3, 3), 0)

        checkPartialAssign(0)
        checkPartialAssign(1)
        checkPartialAssign(2)
        checkPartialAssign((0, 1))
        checkPartialAssign((1, 2))
        checkPartialAssign((0, 2))
        checkPartialAssign(torch.LongTensor((0, 2)))

        with self.assertRaises(IndexError):
            reference[1, 1, 1, 1] = 1
        with self.assertRaises(IndexError):
            reference[1, 1, 1, (1, 1)] = 1
        with self.assertRaises(IndexError):
            reference[3, 3, 3, 3, 3, 3, 3, 3] = 1
        with self.assertRaises(TypeError):
            reference[0.0] = 1
        with self.assertRaises(TypeError):
            reference[0.0:2.0] = 1
        with self.assertRaises(TypeError):
            reference[0.0, 0.0:2.0] = 1
        with self.assertRaises(TypeError):
            reference[0.0, :, 0.0:2.0] = 1
        with self.assertRaises(TypeError):
            reference[0.0, ..., 0.0:2.0] = 1
        with self.assertRaises(TypeError):
            reference[0.0, :, 0.0] = 1

        # LongTensor assignments are not fully supported yet
        with self.assertRaises(TypeError):
            reference[0, torch.LongTensor([2, 4])] = 1

    def test_index_copy(self):
        num_copy, num_dest = 3, 20
        dest = torch.randn(num_dest, 4, 5)
        src = torch.randn(num_copy, 4, 5)
        idx = torch.randperm(num_dest).narrow(0, 0, num_copy)
        dest2 = dest.clone()
        dest.index_copy_(0, idx, src)
        for i in range(idx.size(0)):
            dest2[idx[i]].copy_(src[i])
        self.assertEqual(dest, dest2, 0)

        dest = torch.randn(num_dest)
        src = torch.randn(num_copy)
        idx = torch.randperm(num_dest).narrow(0, 0, num_copy)
        dest2 = dest.clone()
        dest.index_copy_(0, idx, src)
        for i in range(idx.size(0)):
            dest2[idx[i]] = src[i]
        self.assertEqual(dest, dest2, 0)

    def test_index_add(self):
        num_copy, num_dest = 3, 3
        dest = torch.randn(num_dest, 4, 5)
        src = torch.randn(num_copy, 4, 5)
        idx = torch.randperm(num_dest).narrow(0, 0, num_copy)
        dest2 = dest.clone()
        dest.index_add_(0, idx, src)
        for i in range(idx.size(0)):
            dest2[idx[i]].add_(src[i])
        self.assertEqual(dest, dest2)

        dest = torch.randn(num_dest)
        src = torch.randn(num_copy)
        idx = torch.randperm(num_dest).narrow(0, 0, num_copy)
        dest2 = dest.clone()
        dest.index_add_(0, idx, src)
        for i in range(idx.size(0)):
            dest2[idx[i]] = dest2[idx[i]] + src[i]
        self.assertEqual(dest, dest2)

    # Fill idx with valid indices.
    def _fill_indices(self, idx, dim, dim_size, elems_per_row, m, n, o):
        for i in range(1 if dim == 0 else m):
            for j in range(1 if dim == 1 else n):
                for k in range(1 if dim == 2 else o):
                    ii = [i, j, k]
                    ii[dim] = slice(0, idx.size(dim) + 1)
                    idx[tuple(ii)] = torch.randperm(dim_size)[0:elems_per_row]

    def test_gather(self):
        m, n, o = random.randint(10, 20), random.randint(10, 20), random.randint(10, 20)
        elems_per_row = random.randint(1, 10)
        dim = random.randrange(3)

        src = torch.randn(m, n, o)
        idx_size = [m, n, o]
        idx_size[dim] = elems_per_row
        idx = torch.LongTensor().resize_(*idx_size)
        self._fill_indices(idx, dim, src.size(dim), elems_per_row, m, n, o)

        actual = torch.gather(src, dim, idx)
        expected = torch.Tensor().resize_(*idx_size)
        for i in range(idx_size[0]):
            for j in range(idx_size[1]):
                for k in range(idx_size[2]):
                    ii = [i, j, k]
                    ii[dim] = idx[i, j, k]
                    expected[i, j, k] = src[tuple(ii)]
        self.assertEqual(actual, expected, 0)

        idx[0][0][0] = 23
        self.assertRaises(RuntimeError, lambda: torch.gather(src, dim, idx))

        src = torch.randn(3, 4, 5)
        expected, idx = src.max(2)
        actual = torch.gather(src, 2, idx)
        self.assertEqual(actual, expected, 0)

    def test_scatter(self):
        m, n, o = random.randint(10, 20), random.randint(10, 20), random.randint(10, 20)
        elems_per_row = random.randint(1, 10)
        dim = random.randrange(3)

        idx_size = [m, n, o]
        idx_size[dim] = elems_per_row
        idx = torch.LongTensor().resize_(*idx_size)
        self._fill_indices(idx, dim, ([m, n, o])[dim], elems_per_row, m, n, o)
        src = torch.Tensor().resize_(*idx_size).normal_()

        actual = torch.zeros(m, n, o).scatter_(dim, idx, src)
        expected = torch.zeros(m, n, o)
        for i in range(idx_size[0]):
            for j in range(idx_size[1]):
                for k in range(idx_size[2]):
                    ii = [i, j, k]
                    ii[dim] = idx[i, j, k]
                    expected[tuple(ii)] = src[i, j, k]
        self.assertEqual(actual, expected, 0)

        idx[0][0][0] = 34
        self.assertRaises(RuntimeError, lambda: torch.zeros(m, n, o).scatter_(dim, idx, src))

    def test_scatterFill(self):
        m, n, o = random.randint(10, 20), random.randint(10, 20), random.randint(10, 20)
        elems_per_row = random.randint(1, 10)
        dim = random.randrange(3)

        val = random.random()
        idx_size = [m, n, o]
        idx_size[dim] = elems_per_row
        idx = torch.LongTensor().resize_(*idx_size)
        self._fill_indices(idx, dim, ([m, n, o])[dim], elems_per_row, m, n, o)

        actual = torch.zeros(m, n, o).scatter_(dim, idx, val)
        expected = torch.zeros(m, n, o)
        for i in range(idx_size[0]):
            for j in range(idx_size[1]):
                for k in range(idx_size[2]):
                    ii = [i, j, k]
                    ii[dim] = idx[i, j, k]
                    expected[tuple(ii)] = val
        self.assertEqual(actual, expected, 0)

        idx[0][0][0] = 28
        self.assertRaises(RuntimeError, lambda: torch.zeros(m, n, o).scatter_(dim, idx, val))

    def test_masked_copy(self):
        num_copy, num_dest = 3, 10
        dest = torch.randn(num_dest)
        src = torch.randn(num_copy)
        mask = torch.ByteTensor((0, 0, 0, 0, 1, 0, 1, 0, 1, 0))
        dest2 = dest.clone()
        dest.masked_copy_(mask, src)
        j = 0
        for i in range(num_dest):
            if mask[i]:
                dest2[i] = src[j]
                j += 1
        self.assertEqual(dest, dest2, 0)

        # make source bigger than number of 1s in mask
        src = torch.randn(num_dest)
        dest.masked_copy_(mask, src)

        # make src smaller. this should fail
        src = torch.randn(num_copy - 1)
        with self.assertRaises(RuntimeError):
            dest.masked_copy_(mask, src)

    def test_masked_select(self):
        num_src = 10
        src = torch.randn(num_src)
        mask = torch.rand(num_src).clamp(0, 1).mul(2).floor().byte()
        dst = src.masked_select(mask)
        dst2 = []
        for i in range(num_src):
            if mask[i]:
                dst2 += [src[i]]
        self.assertEqual(dst, torch.Tensor(dst2), 0)

    def test_masked_fill(self):
        num_dest = 10
        dst = torch.randn(num_dest)
        mask = torch.rand(num_dest).mul(2).floor().byte()
        val = random.random()
        dst2 = dst.clone()
        dst.masked_fill_(mask, val)
        for i in range(num_dest):
            if mask[i]:
                dst2[i] = val
        self.assertEqual(dst, dst2, 0)

    def test_abs(self):
        size = 1000
        max_val = 1000
        original = torch.rand(size).mul(max_val)
        # Tensor filled with values from {-1, 1}
        switch = torch.rand(size).mul(2).floor().mul(2).add(-1)

        types = ['torch.DoubleTensor', 'torch.FloatTensor', 'torch.LongTensor', 'torch.IntTensor']
        for t in types:
            data = original.type(t)
            switch = switch.type(t)
            res = torch.mul(data, switch)
            self.assertEqual(res.abs(), data, 1e-16)

        # Checking that the right abs function is called for LongTensor
        bignumber = 2 ^ 31 + 1
        res = torch.LongTensor((-bignumber,))
        self.assertGreater(res.abs()[0], 0)

    def test_view(self):
        tensor = torch.rand(15)
        template = torch.rand(3, 5)
        empty = torch.Tensor()
        target = template.size()
        self.assertEqual(tensor.view_as(template).size(), target)
        self.assertEqual(tensor.view(3, 5).size(), target)
        self.assertEqual(tensor.view(torch.Size([3, 5])).size(), target)
        self.assertEqual(tensor.view(-1, 5).size(), target)
        self.assertEqual(tensor.view(3, -1).size(), target)
        tensor_view = tensor.view(5, 3)
        tensor_view.fill_(random.uniform(0, 1))
        self.assertEqual((tensor_view - tensor).abs().max(), 0)
        self.assertEqual(empty.view_as(empty), empty)
        self.assertEqual(empty.view(0), empty)
        self.assertRaises(RuntimeError, lambda: tensor.view(15, 0))
        self.assertRaises(RuntimeError, lambda: tensor.view(7, -1))
        self.assertRaises(RuntimeError, lambda: tensor.view(15, -1, -1))

    def test_expand(self):
        tensor = torch.rand(1, 8, 1)
        tensor2 = torch.rand(5)
        template = torch.rand(4, 8, 5)
        target = template.size()
        self.assertEqual(tensor.expand_as(template).size(), target)
        self.assertEqual(tensor.expand(4, 8, 5).size(), target)
        self.assertEqual(tensor.expand(target).size(), target)
        self.assertEqual(tensor2.expand_as(template).size(), target)
        self.assertEqual(tensor2.expand(4, 8, 5).size(), target)
        self.assertEqual(tensor2.expand(target).size(), target)

        # test double expand
        self.assertEqual(tensor2.expand(1, 5).expand(2, 2, 5), tensor2.repeat(2, 2, 1))

        # test non-contiguous
        noncontig = torch.randn(5, 2, 1, 3)[:, 0]
        assert not noncontig.is_contiguous()
        self.assertEqual(noncontig.expand(2, 5, 4, 3), noncontig.contiguous().repeat(2, 1, 4, 1))

        # make sure it's compatible with unsqueeze
        expanded = tensor2.expand(1, 1, 5)
        unsqueezed = tensor2.unsqueeze(0).unsqueeze(1)
        self.assertEqual(expanded, unsqueezed)
        self.assertEqual(expanded.stride(), unsqueezed.stride())

    def test_repeat(self):
        result = torch.Tensor()
        tensor = torch.rand(8, 4)
        size = (3, 1, 1)
        torchSize = torch.Size(size)
        target = [3, 8, 4]
        self.assertEqual(tensor.repeat(*size).size(), target, 'Error in repeat')
        self.assertEqual(tensor.repeat(torchSize).size(), target, 'Error in repeat using LongStorage')
        result = tensor.repeat(*size)
        self.assertEqual(result.size(), target, 'Error in repeat using result')
        result = tensor.repeat(torchSize)
        self.assertEqual(result.size(), target, 'Error in repeat using result and LongStorage')
        self.assertEqual((result.mean(0).view(8, 4) - tensor).abs().max(), 0, 'Error in repeat (not equal)')

    def test_is_same_size(self):
        t1 = torch.Tensor(3, 4, 9, 10)
        t2 = torch.Tensor(3, 4)
        t3 = torch.Tensor(1, 9, 3, 3)
        t4 = torch.Tensor(3, 4, 9, 10)

        self.assertFalse(t1.is_same_size(t2))
        self.assertFalse(t1.is_same_size(t3))
        self.assertTrue(t1.is_same_size(t4))

    def test_is_set_to(self):
        t1 = torch.Tensor(3, 4, 9, 10)
        t2 = torch.Tensor(3, 4, 9, 10)
        t3 = torch.Tensor().set_(t1)
        t4 = t3.clone().resize_(12, 90)
        self.assertFalse(t1.is_set_to(t2))
        self.assertTrue(t1.is_set_to(t3))
        self.assertTrue(t3.is_set_to(t1), "is_set_to should be symmetric")
        self.assertFalse(t1.is_set_to(t4))
        self.assertFalse(torch.Tensor().is_set_to(torch.Tensor()),
                         "Tensors with no storages should not appear to be set "
                         "to each other")

    def test_tensor_set(self):
        t1 = torch.Tensor()
        t2 = torch.Tensor(3, 4, 9, 10).uniform_()
        t1.set_(t2)
        self.assertEqual(t1.storage()._cdata, t2.storage()._cdata)
        size = torch.Size([9, 3, 4, 10])
        t1.set_(t2.storage(), 0, size)
        self.assertEqual(t1.size(), size)
        t1.set_(t2.storage(), 0, tuple(size))
        self.assertEqual(t1.size(), size)
        self.assertEqual(t1.stride(), (120, 40, 10, 1))
        stride = (10, 360, 90, 1)
        t1.set_(t2.storage(), 0, size, stride)
        self.assertEqual(t1.stride(), stride)

    def test_equal(self):
        # Contiguous, 1D
        t1 = torch.Tensor((3, 4, 9, 10))
        t2 = t1.contiguous()
        t3 = torch.Tensor((1, 9, 3, 10))
        t4 = torch.Tensor((3, 4, 9))
        t5 = torch.Tensor()
        self.assertTrue(t1.equal(t2))
        self.assertFalse(t1.equal(t3))
        self.assertFalse(t1.equal(t4))
        self.assertFalse(t1.equal(t5))
        self.assertTrue(torch.equal(t1, t2))
        self.assertFalse(torch.equal(t1, t3))
        self.assertFalse(torch.equal(t1, t4))
        self.assertFalse(torch.equal(t1, t5))

        # Non contiguous, 2D
        s = torch.Tensor(((1, 2, 3, 4), (5, 6, 7, 8)))
        s1 = s[:, 1:3]
        s2 = s1.clone()
        s3 = torch.Tensor(((2, 3), (6, 7)))
        s4 = torch.Tensor(((0, 0), (0, 0)))

        self.assertFalse(s1.is_contiguous())
        self.assertTrue(s1.equal(s2))
        self.assertTrue(s1.equal(s3))
        self.assertFalse(s1.equal(s4))
        self.assertTrue(torch.equal(s1, s2))
        self.assertTrue(torch.equal(s1, s3))
        self.assertFalse(torch.equal(s1, s4))

    def test_element_size(self):
        byte = torch.ByteStorage().element_size()
        char = torch.CharStorage().element_size()
        short = torch.ShortStorage().element_size()
        int = torch.IntStorage().element_size()
        long = torch.LongStorage().element_size()
        float = torch.FloatStorage().element_size()
        double = torch.DoubleStorage().element_size()

        self.assertEqual(byte, torch.ByteTensor().element_size())
        self.assertEqual(char, torch.CharTensor().element_size())
        self.assertEqual(short, torch.ShortTensor().element_size())
        self.assertEqual(int, torch.IntTensor().element_size())
        self.assertEqual(long, torch.LongTensor().element_size())
        self.assertEqual(float, torch.FloatTensor().element_size())
        self.assertEqual(double, torch.DoubleTensor().element_size())

        self.assertGreater(byte, 0)
        self.assertGreater(char, 0)
        self.assertGreater(short, 0)
        self.assertGreater(int, 0)
        self.assertGreater(long, 0)
        self.assertGreater(float, 0)
        self.assertGreater(double, 0)

        # These tests are portable, not necessarily strict for your system.
        self.assertEqual(byte, 1)
        self.assertEqual(char, 1)
        self.assertGreaterEqual(short, 2)
        self.assertGreaterEqual(int, 2)
        self.assertGreaterEqual(int, short)
        self.assertGreaterEqual(long, 4)
        self.assertGreaterEqual(long, int)
        self.assertGreaterEqual(double, float)

    def test_split(self):
        tensor = torch.rand(7, 4)
        split_size = 3
        dim = 0
        target_sizes = ([3, 4], [3, 4], [1, 4])
        splits = tensor.split(split_size, dim)
        start = 0
        for target_size, split in zip(target_sizes, splits):
            self.assertEqual(split.size(), target_size)
            self.assertEqual(tensor.narrow(dim, start, target_size[dim]), split, 0)
            start = start + target_size[dim]

    def test_chunk(self):
        tensor = torch.rand(4, 7)
        num_chunks = 3
        dim = 1
        target_sizes = ([4, 3], [4, 3], [4, 1])
        splits = tensor.chunk(num_chunks, dim)
        start = 0
        for target_size, split in zip(target_sizes, splits):
            self.assertEqual(split.size(), target_size)
            self.assertEqual(tensor.narrow(dim, start, target_size[dim]), split, 0)
            start = start + target_size[dim]

    def test_tolist(self):
        list0D = []
        tensor0D = torch.Tensor(list0D)
        self.assertEqual(tensor0D.tolist(), list0D)

        table1D = [1, 2, 3]
        tensor1D = torch.Tensor(table1D)
        storage = torch.Storage(table1D)
        self.assertEqual(tensor1D.tolist(), table1D)
        self.assertEqual(storage.tolist(), table1D)
        self.assertEqual(tensor1D.tolist(), table1D)
        self.assertEqual(storage.tolist(), table1D)

        table2D = [[1, 2], [3, 4]]
        tensor2D = torch.Tensor(table2D)
        self.assertEqual(tensor2D.tolist(), table2D)

        tensor3D = torch.Tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
        tensorNonContig = tensor3D.select(1, 1)
        self.assertFalse(tensorNonContig.is_contiguous())
        self.assertEqual(tensorNonContig.tolist(), [[3, 4], [7, 8]])

    def test_permute(self):
        orig = [1, 2, 3, 4, 5, 6, 7]
        perm = list(torch.randperm(7))
        x = torch.Tensor(*orig).fill_(0)
        new = list(map(lambda x: x - 1, x.permute(*perm).size()))
        self.assertEqual(perm, new)
        self.assertEqual(x.size(), orig)

    def test_storageview(self):
        s1 = torch.LongStorage((3, 4, 5))
        s2 = torch.LongStorage(s1, 1)

        self.assertEqual(s2.size(), 2)
        self.assertEqual(s2[0], s1[1])
        self.assertEqual(s2[1], s1[2])

        s2[1] = 13
        self.assertEqual(13, s1[2])

    def test_nonzero(self):
        num_src = 12

        types = [
            'torch.ByteTensor',
            'torch.CharTensor',
            'torch.ShortTensor',
            'torch.IntTensor',
            'torch.FloatTensor',
            'torch.DoubleTensor',
            'torch.LongTensor',
        ]

        shapes = [
            torch.Size((12,)),
            torch.Size((12, 1)),
            torch.Size((1, 12)),
            torch.Size((6, 2)),
            torch.Size((3, 2, 2)),
        ]

        for t in types:
            while True:
                tensor = torch.rand(num_src).mul(2).floor().type(t)
                if tensor.sum() > 0:
                    break
            for shape in shapes:
                tensor = tensor.clone().resize_(shape)
                dst1 = torch.nonzero(tensor)
                dst2 = tensor.nonzero()
                dst3 = torch.LongTensor()
                torch.nonzero(tensor, out=dst3)
                if len(shape) == 1:
                    dst = []
                    for i in range(num_src):
                        if tensor[i] != 0:
                            dst += [i]

                    self.assertEqual(dst1.select(1, 0), torch.LongTensor(dst), 0)
                    self.assertEqual(dst2.select(1, 0), torch.LongTensor(dst), 0)
                    self.assertEqual(dst3.select(1, 0), torch.LongTensor(dst), 0)
                elif len(shape) == 2:
                    # This test will allow through some False positives. It only checks
                    # that the elements flagged positive are indeed non-zero.
                    for i in range(dst1.size(0)):
                        self.assertNotEqual(tensor[dst1[i, 0], dst1[i, 1]], 0)
                elif len(shape) == 3:
                    # This test will allow through some False positives. It only checks
                    # that the elements flagged positive are indeed non-zero.
                    for i in range(dst1.size(0)):
                        self.assertNotEqual(tensor[dst1[i, 0], dst1[i, 1], dst1[i, 2]], 0)

    def test_deepcopy(self):
        from copy import deepcopy
        a = torch.randn(5, 5)
        b = torch.randn(5, 5)
        c = a.view(25)
        q = [a, [a.storage(), b.storage()], b, c]
        w = deepcopy(q)
        self.assertEqual(w[0], q[0], 0)
        self.assertEqual(w[1][0], q[1][0], 0)
        self.assertEqual(w[1][1], q[1][1], 0)
        self.assertEqual(w[1], q[1], 0)
        self.assertEqual(w[2], q[2], 0)

        # Check that deepcopy preserves sharing
        w[0].add_(1)
        for i in range(a.numel()):
            self.assertEqual(w[1][0][i], q[1][0][i] + 1)
        self.assertEqual(w[3], c + 1)
        w[2].sub_(1)
        for i in range(a.numel()):
            self.assertEqual(w[1][1][i], q[1][1][i] - 1)

    def test_copy(self):
        from copy import copy
        a = torch.randn(5, 5)
        a_clone = a.clone()
        b = copy(a)
        b.fill_(1)
        # copy is a shallow copy, only copies the tensor view,
        # not the data
        self.assertEqual(a, b)

    def test_pickle(self):
        if sys.version_info[0] == 2:
            import cPickle as pickle
        else:
            import pickle
        a = torch.randn(5, 5)
        serialized = pickle.dumps(a)
        b = pickle.loads(serialized)
        self.assertEqual(a, b)

    def test_bernoulli(self):
        t = torch.ByteTensor(10, 10)

        def isBinary(t):
            return torch.ne(t, 0).mul_(torch.ne(t, 1)).sum() == 0

        p = 0.5
        t.bernoulli_(p)
        self.assertTrue(isBinary(t))

        p = torch.rand(SIZE)
        t.bernoulli_(p)
        self.assertTrue(isBinary(t))

        q = torch.rand(5, 5)
        self.assertTrue(isBinary(q.bernoulli()))

    def test_normal(self):
        q = torch.Tensor(100, 100)
        q.normal_()
        self.assertEqual(q.mean(), 0, 0.2)
        self.assertEqual(q.std(), 1, 0.2)

        q.normal_(2, 3)
        self.assertEqual(q.mean(), 2, 0.3)
        self.assertEqual(q.std(), 3, 0.3)

        mean = torch.Tensor(100, 100)
        std = torch.Tensor(100, 100)
        mean[:50] = 0
        mean[50:] = 1
        std[:, :50] = 4
        std[:, 50:] = 1

        r = torch.normal(mean)
        self.assertEqual(r[:50].mean(), 0, 0.2)
        self.assertEqual(r[50:].mean(), 1, 0.2)
        self.assertEqual(r.std(), 1, 0.2)

        r = torch.normal(mean, 3)
        self.assertEqual(r[:50].mean(), 0, 0.2)
        self.assertEqual(r[50:].mean(), 1, 0.2)
        self.assertEqual(r.std(), 3, 0.2)

        r = torch.normal(2, std)
        self.assertEqual(r.mean(), 2, 0.2)
        self.assertEqual(r[:, :50].std(), 4, 0.3)
        self.assertEqual(r[:, 50:].std(), 1, 0.2)

        r = torch.normal(mean, std)
        self.assertEqual(r[:50].mean(), 0, 0.2)
        self.assertEqual(r[50:].mean(), 1, 0.2)
        self.assertEqual(r[:, :50].std(), 4, 0.3)
        self.assertEqual(r[:, 50:].std(), 1, 0.2)

    def test_serialization(self):
        a = [torch.randn(5, 5).float() for i in range(2)]
        b = [a[i % 2] for i in range(4)]
        b += [a[0].storage()]
        b += [a[0].storage()[1:4]]
        b += [torch.range(1, 10).int()]
        t1 = torch.FloatTensor().set_(a[0].storage()[1:4], 0, (3,), (1,))
        t2 = torch.FloatTensor().set_(a[0].storage()[1:4], 0, (3,), (1,))
        b += [(t1.storage(), t1.storage(), t2.storage())]
        b += [a[0].storage()[0:2]]
        for use_name in (False, True):
            with tempfile.NamedTemporaryFile() as f:
                handle = f if not use_name else f.name
                torch.save(b, handle)
                f.seek(0)
                c = torch.load(handle)
            self.assertEqual(b, c, 0)
            self.assertTrue(isinstance(c[0], torch.FloatTensor))
            self.assertTrue(isinstance(c[1], torch.FloatTensor))
            self.assertTrue(isinstance(c[2], torch.FloatTensor))
            self.assertTrue(isinstance(c[3], torch.FloatTensor))
            self.assertTrue(isinstance(c[4], torch.FloatStorage))
            c[0].fill_(10)
            self.assertEqual(c[0], c[2], 0)
            self.assertEqual(c[4], torch.FloatStorage(25).fill_(10), 0)
            c[1].fill_(20)
            self.assertEqual(c[1], c[3], 0)
            self.assertEqual(c[4], c[5][1:4], 0)

            # check that serializing the same storage view object unpickles
            # it as one object not two (and vice versa)
            views = c[7]
            self.assertEqual(views[0]._cdata, views[1]._cdata)
            self.assertEqual(views[0], views[2])
            self.assertNotEqual(views[0]._cdata, views[2]._cdata)

            rootview = c[8]
            self.assertEqual(rootview.data_ptr(), c[0].data_ptr())

    @unittest.skipIf(not torch.cuda.is_available(), 'no CUDA')
    def test_serialization_cuda(self):
        device_count = torch.cuda.device_count()
        t0 = torch.cuda.FloatTensor(5).fill_(1)
        torch.cuda.set_device(device_count - 1)
        tn = torch.cuda.FloatTensor(3).fill_(2)
        torch.cuda.set_device(0)
        b = (t0, tn)
        with tempfile.NamedTemporaryFile() as f:
            torch.save(b, f)
            f.seek(0)
            c = torch.load(f)
            self.assertEqual(b, c, 0)
            u0, un = c
            self.assertEqual(u0.get_device(), 0)
            self.assertEqual(un.get_device(), device_count - 1)

    def test_serialization_backwards_compat(self):
        a = [torch.range(1 + i, 25 + i).view(5, 5).float() for i in range(2)]
        b = [a[i % 2] for i in range(4)]
        b += [a[0].storage()]
        b += [a[0].storage()[1:4]]
        DATA_URL = 'https://s3.amazonaws.com/pytorch/legacy_serialized.pt'
        data_dir = os.path.join(os.path.dirname(__file__), 'data')
        test_file_path = os.path.join(data_dir, 'legacy_serialized.pt')
        succ = download_file(DATA_URL, test_file_path)
        if not succ:
            warnings.warn(("Couldn't download the test file for backwards compatibility! "
                           "Tests will be incomplete!"), RuntimeWarning)
            return
        c = torch.load(test_file_path)
        self.assertEqual(b, c, 0)
        self.assertTrue(isinstance(c[0], torch.FloatTensor))
        self.assertTrue(isinstance(c[1], torch.FloatTensor))
        self.assertTrue(isinstance(c[2], torch.FloatTensor))
        self.assertTrue(isinstance(c[3], torch.FloatTensor))
        self.assertTrue(isinstance(c[4], torch.FloatStorage))
        c[0].fill_(10)
        self.assertEqual(c[0], c[2], 0)
        self.assertEqual(c[4], torch.FloatStorage(25).fill_(10), 0)
        c[1].fill_(20)
        self.assertEqual(c[1], c[3], 0)
        self.assertEqual(c[4], c[5][1:4], 0)

    def test_serialization_container(self):
        def import_module(name, filename):
            if sys.version_info >= (3, 5):
                import importlib.util
                spec = importlib.util.spec_from_file_location(name, filename)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
            else:
                import imp
                module = imp.load_source(name, filename)
            sys.modules[module.__name__] = module
            return module

        import os
        with tempfile.NamedTemporaryFile() as checkpoint:
            fname = os.path.join(os.path.dirname(__file__), 'data/network1.py')
            module = import_module('tmpmodule', fname)
            torch.save(module.Net(), checkpoint)

            # First check that the checkpoint can be loaded without warnings
            checkpoint.seek(0)
            with warnings.catch_warnings(record=True) as w:
                loaded = torch.load(checkpoint)
                self.assertTrue(isinstance(loaded, module.Net))
                self.assertEquals(len(w), 0)

            # Replace the module with different source
            fname = os.path.join(os.path.dirname(__file__), 'data/network2.py')
            module = import_module('tmpmodule', fname)
            checkpoint.seek(0)
            with warnings.catch_warnings(record=True) as w:
                loaded = torch.load(checkpoint)
                self.assertTrue(isinstance(loaded, module.Net))
                self.assertEquals(len(w), 1)
                self.assertTrue(w[0].category, 'SourceChangeWarning')

    def test_from_buffer(self):
        a = bytearray([1, 2, 3, 4])
        self.assertEqual(torch.ByteStorage.from_buffer(a).tolist(), [1, 2, 3, 4])
        shorts = torch.ShortStorage.from_buffer(a, 'big')
        self.assertEqual(shorts.size(), 2)
        self.assertEqual(shorts.tolist(), [258, 772])
        ints = torch.IntStorage.from_buffer(a, 'little')
        self.assertEqual(ints.size(), 1)
        self.assertEqual(ints[0], 67305985)
        f = bytearray([0x40, 0x10, 0x00, 0x00])
        floats = torch.FloatStorage.from_buffer(f, 'big')
        self.assertEqual(floats.size(), 1)
        self.assertEqual(floats[0], 2.25)

    def test_print(self):
        for t in torch._tensor_classes:
            if t in torch.sparse._sparse_tensor_classes:
                continue
            if t.is_cuda and not torch.cuda.is_available():
                continue
            obj = t(100, 100).fill_(1)
            obj.__repr__()
            str(obj)
        for t in torch._storage_classes:
            if t.is_cuda and not torch.cuda.is_available():
                continue
            obj = t(100).fill_(1)
            obj.__repr__()
            str(obj)

        x = torch.Tensor([4, float('inf'), 1.5, float('-inf'), 0, float('nan'), 1])
        x.__repr__()
        str(x)

    def test_unsqueeze(self):
        x = torch.randn(2, 3, 4)
        y = x.unsqueeze(1)
        self.assertEqual(y, x.view(2, 1, 3, 4))
        y = x.clone().unsqueeze_(2)
        self.assertEqual(y, x.view(2, 3, 1, 4))

        x = x[:, 1]
        self.assertFalse(x.is_contiguous())
        y = x.unsqueeze(1)
        self.assertEqual(y, x.contiguous().view(2, 1, 4))
        y = x.clone().unsqueeze_(2)
        self.assertEqual(y, x.contiguous().view(2, 4, 1))

        self.assertRaises(RuntimeError, lambda: torch.Tensor().unsqueeze(0))

    def test_iter(self):
        x = torch.randn(5, 5)
        for i, sub in enumerate(x):
            self.assertEqual(sub, x[i])

    def test_accreal_type(self):
        x = torch.randn(2, 3, 4) * 10
        self.assertIsInstance(x.double().sum(), float)
        self.assertIsInstance(x.float().sum(), float)
        self.assertIsInstance(x.long().sum(), int)
        self.assertIsInstance(x.int().sum(), int)
        self.assertIsInstance(x.short().sum(), int)
        self.assertIsInstance(x.char().sum(), int)
        self.assertIsInstance(x.byte().sum(), int)

    @unittest.skipIf(not torch.cuda.is_available(), 'no CUDA')
    def test_pin_memory(self):
        x = torch.randn(3, 5)
        self.assertFalse(x.is_pinned())
        pinned = x.pin_memory()
        self.assertTrue(pinned.is_pinned())
        self.assertEqual(pinned, x)
        self.assertNotEqual(pinned.data_ptr(), x.data_ptr())

    @unittest.skipIf(not TEST_NUMPY, "Numpy not found")
    def test_numpy_unresizable(self):
        x = np.zeros((2, 2))
        y = torch.from_numpy(x)
        with self.assertRaises(ValueError):
            x.resize((5, 5))

        z = torch.randn(5, 5)
        w = z.numpy()
        with self.assertRaises(RuntimeError):
            z.resize_(10, 10)
        with self.assertRaises(ValueError):
            w.resize((10, 10))

    @unittest.skipIf(not TEST_NUMPY, "Numpy not found")
    def test_toNumpy(self):
        types = [
            'torch.ByteTensor',
            'torch.IntTensor',
            'torch.FloatTensor',
            'torch.DoubleTensor',
            'torch.LongTensor',
        ]
        for tp in types:
            # 1D
            sz = 10
            x = torch.randn(sz).mul(255).type(tp)
            y = x.numpy()
            for i in range(sz):
                self.assertEqual(x[i], y[i])

            # 1D > 0 storage offset
            xm = torch.randn(sz * 2).mul(255).type(tp)
            x = xm.narrow(0, sz - 1, sz)
            self.assertTrue(x.storage_offset() > 0)
            y = x.numpy()
            for i in range(sz):
                self.assertEqual(x[i], y[i])

            def check2d(x, y):
                for i in range(sz1):
                    for j in range(sz2):
                        self.assertEqual(x[i][j], y[i][j])

            # empty
            x = torch.Tensor().type(tp)
            y = x.numpy()
            self.assertEqual(y.size, 0)

            # contiguous 2D
            sz1 = 3
            sz2 = 5
            x = torch.randn(sz1, sz2).mul(255).type(tp)
            y = x.numpy()
            check2d(x, y)

            # with storage offset
            xm = torch.randn(sz1 * 2, sz2).mul(255).type(tp)
            x = xm.narrow(0, sz1 - 1, sz1)
            y = x.numpy()
            self.assertTrue(x.storage_offset() > 0)
            check2d(x, y)

            # non-contiguous 2D
            x = torch.randn(sz2, sz1).t().mul(255).type(tp)
            y = x.numpy()
            check2d(x, y)

            # with storage offset
            xm = torch.randn(sz2 * 2, sz1).mul(255).type(tp)
            x = xm.narrow(0, sz2 - 1, sz2).t()
            y = x.numpy()
            self.assertTrue(x.storage_offset() > 0)
            check2d(x, y)

            # non-contiguous 2D with holes
            xm = torch.randn(sz2 * 2, sz1 * 2).mul(255).type(tp)
            x = xm.narrow(0, sz2 - 1, sz2).narrow(1, sz1 - 1, sz1).t()
            y = x.numpy()
            self.assertTrue(x.storage_offset() > 0)
            check2d(x, y)

            # check writeable
            x = torch.randn(3, 4).mul(255).type(tp)
            y = x.numpy()
            self.assertTrue(y.flags.writeable)
            y[0][1] = 3
            self.assertTrue(x[0][1] == 3)
            y = x.t().numpy()
            self.assertTrue(y.flags.writeable)
            y[0][1] = 3
            self.assertTrue(x[0][1] == 3)

    @unittest.skipIf(not TEST_NUMPY, "Numpy not found")
    def test_from_numpy(self):
        dtypes = [
            np.double,
            np.float,
            np.int64,
            np.int32,
            np.uint8
        ]
        for dtype in dtypes:
            array = np.array([1, 2, 3, 4], dtype=dtype)
            self.assertEqual(torch.from_numpy(array), torch.Tensor([1, 2, 3, 4]))

        # check storage offset
        x = np.linspace(1, 125, 125)
        x.shape = (5, 5, 5)
        x = x[1]
        expected = torch.range(1, 125).view(5, 5, 5)[1]
        self.assertEqual(torch.from_numpy(x), expected)

        # check noncontiguous
        x = np.linspace(1, 25, 25)
        x.shape = (5, 5)
        expected = torch.range(1, 25).view(5, 5).t()
        self.assertEqual(torch.from_numpy(x.T), expected)

        # check noncontiguous with holes
        x = np.linspace(1, 125, 125)
        x.shape = (5, 5, 5)
        x = x[:, 1]
        expected = torch.range(1, 125).view(5, 5, 5)[:, 1]
        self.assertEqual(torch.from_numpy(x), expected)

    @unittest.skipIf(not TEST_NUMPY, "Numpy not found")
    def test_numpy_index(self):
        i = np.int32([0, 1, 2])
        x = torch.randn(5, 5)
        for idx in i:
            self.assertFalse(isinstance(idx, int))
            self.assertEqual(x[idx], x[int(idx)])

    def test_comparison_ops(self):
        x = torch.randn(5, 5)
        y = torch.randn(5, 5)

        eq = x == y
        for idx in iter_indices(x):
            self.assertIs(x[idx] == y[idx], eq[idx] == 1)

        ne = x != y
        for idx in iter_indices(x):
            self.assertIs(x[idx] != y[idx], ne[idx] == 1)

        lt = x < y
        for idx in iter_indices(x):
            self.assertIs(x[idx] < y[idx], lt[idx] == 1)

        le = x <= y
        for idx in iter_indices(x):
            self.assertIs(x[idx] <= y[idx], le[idx] == 1)

        gt = x > y
        for idx in iter_indices(x):
            self.assertIs(x[idx] > y[idx], gt[idx] == 1)

        ge = x >= y
        for idx in iter_indices(x):
            self.assertIs(x[idx] >= y[idx], ge[idx] == 1)

    def test_logical_ops(self):
        x = torch.randn(5, 5).gt(0)
        y = torch.randn(5, 5).gt(0)

        and_result = x & y
        for idx in iter_indices(x):
            if and_result[idx]:
                self.assertTrue(x[idx] and y[idx])
            else:
                self.assertFalse(x[idx] and y[idx])

        or_result = x | y
        for idx in iter_indices(x):
            if or_result[idx]:
                self.assertTrue(x[idx] or y[idx])
            else:
                self.assertFalse(x[idx] or y[idx])

        xor_result = x ^ y
        for idx in iter_indices(x):
            if xor_result[idx]:
                self.assertTrue(x[idx] ^ y[idx])
            else:
                self.assertFalse(x[idx] ^ y[idx])

        x_clone = x.clone()
        x_clone &= y
        self.assertEqual(x_clone, and_result)

        x_clone = x.clone()
        x_clone |= y
        self.assertEqual(x_clone, or_result)

        x_clone = x.clone()
        x_clone ^= y
        self.assertEqual(x_clone, xor_result)

    def test_apply(self):
        x = torch.range(1, 5)
        res = x.clone().apply_(lambda k: k + k)
        self.assertEqual(res, x * 2)
        self.assertRaises(RuntimeError, lambda: x.apply_(lambda k: "str"))

    def test_Size(self):
        x = torch.Size([1, 2, 3])
        self.assertIsInstance(x, tuple)
        self.assertEqual(x[0], 1)
        self.assertEqual(x[1], 2)
        self.assertEqual(x[2], 3)
        self.assertEqual(len(x), 3)
        self.assertRaises(TypeError, lambda: torch.Size(torch.ones(3)))

        self.assertIsInstance(x * 2, torch.Size)
        self.assertIsInstance(x[:-1], torch.Size)
        self.assertIsInstance(x + x, torch.Size)


if __name__ == '__main__':
    run_tests()
