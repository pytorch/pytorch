import sys
import os
import math
import random
import copy
import torch
import torch.cuda
import tempfile
import unittest
import warnings
from itertools import product, combinations
from common import TestCase, iter_indices, TEST_NUMPY, run_tests, download_file, skipIfNoLapack, \
    suppress_warnings

if TEST_NUMPY:
    import numpy as np

SIZE = 100


class TestTorch(TestCase):

    def test_dot(self):
        types = {
            'torch.DoubleTensor': 1e-8,
            'torch.FloatTensor': 1e-4,
        }
        for tname, _prec in types.items():
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

    def test_lgamma(self):
        self._testMathByName('lgamma')

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
        res1val, res1ind = torchfn(m1, 1, False)
        res2val = m1[:, 0:1].clone().squeeze()
        res2ind = res1ind.clone().fill_(0)
        for i, j in iter_indices(m1):
            if mathfn(res2val[i], m1[i, j]) != res2val[i]:
                res2val[i] = m1[i, j]
                res2ind[i] = j

        maxerr = 0
        for i in range(res1val.size(0)):
            maxerr = max(maxerr, abs(res1val[i] - res2val[i]))
            self.assertEqual(res1ind[i], res2ind[i])
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

    def test_dim_reduction(self):
        dim_red_fns = [
            "mean", "median", "mode", "norm", "prod",
            "std", "sum", "var", "max", "min"]

        def normfn_attr(t, dim, keepdim=False):
            attr = getattr(torch, "norm")
            return attr(t, 2, dim, keepdim)

        for fn_name in dim_red_fns:
            x = torch.randn(3, 4, 5)
            fn_attr = getattr(torch, fn_name) if fn_name != "norm" else normfn_attr

            def fn(t, dim, keepdim=False):
                ans = fn_attr(x, dim, keepdim=keepdim)
                return ans if not isinstance(ans, tuple) else ans[0]

            dim = random.randint(0, 2)
            self.assertEqual(fn(x, dim).unsqueeze(dim), fn(x, dim, keepdim=True))
            self.assertEqual(x.ndimension() - 1, fn(x, dim).ndimension())
            self.assertEqual(x.ndimension(), fn(x, dim, keepdim=True).ndimension())

            # check 1-d behavior
            x = torch.randn(1)
            dim = 0
            self.assertEqual(fn(x, dim), fn(x, dim, keepdim=True))
            self.assertEqual(x.ndimension(), fn(x, dim).ndimension())
            self.assertEqual(x.ndimension(), fn(x, dim, keepdim=True).ndimension())

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
        # Check the Floating point case
        m1 = torch.Tensor(10, 10).uniform_(-10., 10.)
        res1 = m1.clone()
        res2 = m1.clone()
        qs = torch.arange(-5.1, 4.1)
        # Check the case where the divisor is a simple float
        for col_idx, q in enumerate(qs):
            # Reference
            for i in range(m1.size(0)):
                res2[i, col_idx] = res2[i, col_idx] % q
            # To test
            res1[:, col_idx].remainder_(q)
        self.assertEqual(res1, res2)
        # Check the case where the divisor is a tensor
        res1 = m1.clone()
        res1.remainder_(qs.unsqueeze(0).expand_as(res1))
        self.assertEqual(res1, res2)

        # Check the LongTensor case
        long_m1 = torch.LongTensor(10, 10).random_(-10, 10)
        long_res1 = long_m1.clone()
        long_res2 = long_m1.clone()
        long_qs = torch.arange(-5, 5).long()
        long_qs[5] = 5  # Can't handle the divisor=0 case
        for col_idx, long_q in enumerate(long_qs):
            # Reference
            for i in range(long_m1.size(0)):
                long_res2[i, col_idx] = long_res2[i, col_idx] % long_q
            # To test
            long_res1[:, col_idx].remainder_(long_q)
        self.assertEqual(long_res1, long_res2)
        # Divisor is a tensor case
        long_res1 = long_m1.clone()
        long_res1.remainder_(long_qs.unsqueeze(0).expand_as(long_res1))

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

    @staticmethod
    def _test_btrifact(self, cast):
        a = torch.FloatTensor((((1.3722, -0.9020),
                                (1.8849, 1.9169)),
                               ((0.7187, -1.1695),
                                (-0.0139, 1.3572)),
                               ((-1.6181, 0.7148),
                                (1.3728, 0.1319))))
        a = cast(a)
        info = cast(torch.IntTensor())
        a_LU = a.btrifact(info=info)
        self.assertEqual(info.abs().sum(), 0)
        P, a_L, a_U = torch.btriunpack(*a_LU)
        a_ = torch.bmm(P, torch.bmm(a_L, a_U))
        self.assertEqual(a_, a)

    @skipIfNoLapack
    def test_btrifact(self):
        self._test_btrifact(self, lambda t: t)

    @staticmethod
    def _test_btrisolve(self, cast):
        a = torch.FloatTensor((((1.3722, -0.9020),
                                (1.8849, 1.9169)),
                               ((0.7187, -1.1695),
                                (-0.0139, 1.3572)),
                               ((-1.6181, 0.7148),
                                (1.3728, 0.1319))))
        b = torch.FloatTensor(((4.02, 6.19),
                               (-1.56, 4.00),
                               (9.81, -4.09)))
        a, b = cast(a), cast(b)
        info = cast(torch.IntTensor())
        LU_data, pivots = a.btrifact(info=info)
        self.assertEqual(info.abs().sum(), 0)
        x = torch.btrisolve(b, LU_data, pivots)
        b_ = torch.bmm(a, x.unsqueeze(2)).squeeze()
        self.assertEqual(b_, b)

    @skipIfNoLapack
    def test_btrisolve(self):
        self._test_btrisolve(self, lambda t: t)

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
        self.assertEqual(res2, res.sum(0, False))

        res2.addbmm_(1, b1, b2)
        self.assertEqual(res2, res.sum(0, False) * 2)

        res2.addbmm_(1., .5, b1, b2)
        self.assertEqual(res2, res.sum(0, False) * 2.5)

        res3 = torch.addbmm(1, res2, 0, b1, b2)
        self.assertEqual(res3, res2)

        res4 = torch.addbmm(1, res2, .5, b1, b2)
        self.assertEqual(res4, res.sum(0, False) * 3)

        res5 = torch.addbmm(0, res2, 1, b1, b2)
        self.assertEqual(res5, res.sum(0, False))

        res6 = torch.addbmm(.1, res2, .5, b1, b2)
        self.assertEqual(res6, res2 * .1 + (res.sum(0) * .5))

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

        res1 = torchfn(sm1, sm2.view(10, 10))
        res2 = reference_implementation(res1.clone())
        self.assertEqual(res1, res2)

        # non-contiguous
        m1 = torch.randn(10, 10, 10)
        m2 = torch.randn(10 * 10, 10 * 10)
        sm1 = m1[:, 4]
        sm2 = m2[:, 4]
        # view as sm1.size()
        sm2.set_(sm2.storage(), sm2.storage_offset(), sm1.size(), (sm2.stride()[0] * 10, sm2.stride()[0]))
        res1 = torchfn(sm1, sm2)
        # reference_implementation assumes 1-d sm2
        sm2.set_(sm2.storage(), sm2.storage_offset(), m2[:, 4].size(), m2[:, 4].stride())
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
            norms = m2.norm(value, 1, True)
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

    @suppress_warnings
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

    def test_arange(self):
        res1 = torch.arange(0, 1)
        res2 = torch.Tensor()
        torch.arange(0, 1, out=res2)
        self.assertEqual(res1, res2, 0)

        # Check arange for non-contiguous tensors.
        x = torch.zeros(2, 3)
        torch.arange(0, 4, out=x.narrow(1, 1, 2))
        res2 = torch.Tensor(((0, 0, 1), (0, 2, 3)))
        self.assertEqual(x, res2, 1e-16)

        # Check negative
        res1 = torch.Tensor((1, 0))
        res2 = torch.Tensor()
        torch.arange(1, -1, -1, out=res2)
        self.assertEqual(res1, res2, 0)

        # Equal bounds
        res1 = torch.ones(1)
        res2 = torch.Tensor()
        torch.arange(1, 0, -1, out=res2)
        self.assertEqual(res1, res2, 0)
        torch.arange(1, 2, 1, out=res2)
        self.assertEqual(res1, res2, 0)

        # FloatTensor
        res1 = torch.arange(0.6, 0.89, 0.1, out=torch.FloatTensor())
        self.assertEqual(res1.size(0), 3)
        res1 = torch.arange(1, 10, 0.3, out=torch.FloatTensor())
        self.assertEqual(res1.size(0), 31)

        # DoubleTensor
        res1 = torch.arange(0.6, 0.89, 0.1, out=torch.DoubleTensor())
        self.assertEqual(res1.size(0), 3)
        res1 = torch.arange(1, 10, 0.3, out=torch.DoubleTensor())
        self.assertEqual(res1.size(0), 31)

        # Check that it's exclusive
        r = torch.arange(0, 5)
        self.assertEqual(r.min(), 0)
        self.assertEqual(r.max(), 4)
        self.assertEqual(r.numel(), 5)

        r = torch.arange(0, 5, 2)
        self.assertEqual(r.min(), 0)
        self.assertEqual(r.max(), 4)
        self.assertEqual(r.numel(), 3)

        r1 = torch.arange(0, 5 + 1e-6)
        r2 = torch.arange(0, 5)
        r3 = torch.arange(0, 5 - 1e-6)
        self.assertEqual(r1[:-1], r2, 0)
        self.assertEqual(r2, r3, 0)

        r1 = torch.arange(10, -1 + 1e-6, -1)
        r2 = torch.arange(10, -1, -1)
        r3 = torch.arange(10, -1 - 1e-6, -1)
        self.assertEqual(r1, r2, 0)
        self.assertEqual(r2, r3[:-1], 0)

    @staticmethod
    def _select_broadcastable_dims(dims_full=None):
        # select full dimensionality
        if dims_full is None:
            dims_full = []
            ndims = random.randint(1, 4)
            dims_full = [random.randint(1, 8) for _ in range(ndims)]
        else:
            ndims = len(dims_full)

        # select actual dimensions for ops:
        # larger: full ndims, individual sizes may be reduced
        # smaller: possibly reduced ndims, sizes may be reduced
        smaller_ndims = random.randint(1, ndims)
        dims_small = []
        dims_large = []
        for i in range(ndims - 1, -1, -1):
            j = random.randint(1, 3)
            if j == 1:  # no reduced singleton dimension
                ds = dims_full[i]
                dl = dims_full[i]
            elif j == 2:  # larger may have reduced singleton dimension
                ds = dims_full[i]
                dl = 1 if len(dims_small) < smaller_ndims else dims_full[i]
            elif j == 3:  # smaller may have reduced singleton dimension
                ds = 1
                dl = dims_full[i]
            dims_large = [dl] + dims_large
            if len(dims_small) < smaller_ndims:
                dims_small = [ds] + dims_small
        return (dims_small, dims_large, dims_full)

    @staticmethod
    def _test_broadcast(self, cast):

        # all functions
        fns = {
            "dist", "atan2", "pow", "lerp", "add",
            "sub", "mul", "div", "fmod", "remainder",
            "eq", "ge", "gt", "le", "lt", "max", "min", "ne",
            "addcdiv", "addcmul", "masked_scatter", "masked_select", "masked_fill",
            "map", "map2", "copy"
        }
        # functions with three tensor arguments
        fns_3_args = {"addcdiv", "addcmul", "map2"}

        for fn in fns:
            (dims_small, dims_large, dims_full) = self._select_broadcastable_dims()
            small = cast(torch.randn(*dims_small).float())
            large = cast(torch.randn(*dims_large).float())
            small_expanded = small.expand(*dims_full)
            large_expanded = large.expand(*dims_full)
            small2 = None
            small2_expanded = None
            if fn in fns_3_args:
                # create another smaller tensor
                (dims_small2, _, _) = self._select_broadcastable_dims(dims_full)
                small2 = cast(torch.randn(*dims_small2).float())
                small2_expanded = small2.expand(*dims_full)

            if hasattr(large_expanded, fn):
                # run through tensor versions of functions
                # and verify fully expanded inputs give same results
                expanded = {large: large_expanded, small: small_expanded, small2: small2_expanded}

                def tensorfn(myfn, t1, t2):
                    if fn == "lerp":
                        return myfn(t1, 0.5)
                    elif fn == "masked_select":
                        return myfn(t1 < 0)
                    elif fn in fns_3_args:
                        return myfn(1, t1, t2)
                    else:
                        return myfn(t1)

                # test various orders
                for first, second, third in [(large, small, small2), (small, large, small2),
                                             (small2, small, large), (small2, large, small)]:
                    if first is None:
                        break  # ignore last iter when small2 is None
                    method_expanded = getattr(expanded[first], fn)
                    method = getattr(first, fn)
                    r1 = tensorfn(method_expanded, expanded[second], expanded[third])
                    r2 = tensorfn(method, second, third)
                    self.assertEqual(r1, r2)

            # now for torch. versions of functions
            if hasattr(torch, fn):
                fntorch = getattr(torch, fn)
                expanded = {large: large_expanded, small: small_expanded, small2: small2_expanded}

                def torchfn(t1, t2, t3):
                    if fn == "lerp":
                        return fntorch(t1, t2, 0.5)
                    elif fn == "masked_select":
                        return fntorch(t1, t2 < 0)
                    elif fn in fns_3_args:
                        return fntorch(t1, 1.0, t2, t3)
                    else:
                        return fntorch(t1, t2)

                # test various orders
                for first, second, third in [(large, small, small2), (small, large, small2),
                                             (small2, small, large), (small2, large, small)]:
                    if first is None:
                        break  # ignore last iter when small2 is None
                    r1 = torchfn(expanded[first], expanded[second], expanded[third])
                    r2 = torchfn(first, second, third)
                    self.assertEqual(r1, r2)

            # now for in place functions
            # in-place tensor is not broadcastable; test only guaranteed
            # to work by broadcasting other argument(s)
            if not hasattr(large_expanded, fn + "_"):
                continue

            # need to clone largeExpanded so we can reuse, since functions are in-place
            large_expanded_clone = large_expanded.clone()

            def tensorfn_inplace(t0, t1, t2=None):
                t0_fn = getattr(t0, fn + "_")
                if fn == "lerp":
                    return t0_fn(t1, 0.5)
                elif fn == "masked_scatter":
                    return t0_fn(t1 < 0.5, cast(torch.arange(1, t0.nelement() + 1).float()))
                elif fn == "masked_fill":
                    return t0_fn(t1 < 0.5, 1.0)
                elif fn == "map":
                    return t0_fn(t1, lambda x, y: x + y)
                elif fn == "map2":
                    return t0_fn(t1, t2, lambda x, y, z: x + y + z)
                elif fn in fns_3_args:
                    return t0_fn(1.0, t1, t2)
                else:
                    return t0_fn(t1)
            r1 = tensorfn_inplace(large_expanded, small_expanded, small2_expanded)
            r2 = tensorfn_inplace(large_expanded_clone, small, small2)
            # in-place pointwise operations don't actually work if the in-place
            # tensor is 0-strided (numpy has the same issue)
            if (0 not in large_expanded.stride() and 0 not in large_expanded_clone.stride()):
                self.assertEqual(r1, r2)

            def broadcastable(t0, t1, t2=None):
                try:
                    t1.expand_as(t0)
                    if t2 is not None:
                        t2.expand_as(t0)
                except RuntimeError:
                    return False
                return True

            def _test_in_place_broadcastable(t0, t1, t2=None):
                if not broadcastable(t0, t1, t2):
                    same_size = t0.numel() == t1.numel() and (t0.numel() == t2.numel() if t2 is not None else True)
                    if not same_size:
                        self.assertRaises(RuntimeError, lambda: tensorfn_inplace(t0, t1, t2))
                else:
                    tensorfn_inplace(t0, t1, t2)

            if fn not in fns_3_args:
                _test_in_place_broadcastable(small, large_expanded)
                _test_in_place_broadcastable(small, large)
            else:
                _test_in_place_broadcastable(small2, small_expanded, large_expanded)
                _test_in_place_broadcastable(small2, small, large)

    def test_broadcast(self):
        self._test_broadcast(self, lambda t: t)

    @staticmethod
    def _test_broadcast_fallback(self, cast):
        # functions that should fallback to pointwise behavior
        fns_fallback = {"add", "sub", "div", "mul", "pow", "fmod", "remainder",
                        "eq", "ge", "gt", "le", "lt", "max", "min", "ne",
                        "addcdiv", "addcmul", "masked_scatter", "masked_select", "masked_fill",
                        "map", "map2", "copy", "dist", "atan2", "lerp"}
        # functions with three tensor arguments
        fns_3_args = {"addcdiv", "addcmul", "map2"}
        # functions that don't broadcast result size_ -- don't check result shape but
        # still run functions to verify that broadcastable arguments don't error out
        fns_no_result_broadcast = {"masked_select"}

        for fn in fns_fallback:
            # case 1: both broadcastable and nElems equal -- verify that we broadcast
            t0 = cast(torch.randn(1, 4).float())
            t1 = cast(torch.randn(4, 1).float())
            t2 = cast(torch.randn(4).float())
            broadcast_size = torch.Size([4, 4])
            if not hasattr(t0, fn):
                continue
            t0_fn = getattr(t0, fn)
            t1_fn = getattr(t1, fn)

            def tensorfn(myfn, t1, t2):
                if fn == "lerp":
                    return myfn(t1, 0.5)
                elif fn == "masked_scatter":
                    return myfn(t1 < 0.5, cast(torch.randn(4 * 4).float()))
                elif fn == "masked_fill":
                    return myfn(t1 < 0.5, 1.0)
                elif fn == "masked_select":
                    return myfn(t1 < 0.5)
                elif fn == "map":
                    return myfn(t1, lambda x, y: x + y)
                elif fn == "map2":
                    return myfn(t1, t2, lambda x, y, z: x + y + z)
                elif fn in fns_3_args:
                    return myfn(1.0, t1, t2)
                else:
                    return myfn(t1)
            r0 = tensorfn(t0_fn, t1, t2)
            r1 = tensorfn(t1_fn, t0, t2)
            if torch.is_tensor(r0) and fn not in fns_no_result_broadcast:
                self.assertEqual(broadcast_size, r0.size())
                self.assertEqual(broadcast_size, r1.size())

            # case 2: broadcastable and not nElemes equal -- tested by test_fallback

            # case 3: not broadcastable nElems equal -- verify we fallback
            for inplace in False, True:
                t0 = cast(torch.randn(1, 6).float())
                t1 = cast(torch.randn(2, 3).float())
                t2 = cast(torch.randn(3, 2).float())
                if not hasattr(t0, fn if not inplace else fn + "_"):
                    continue
                t0_fn = getattr(t0, fn if not inplace else fn + "_")
                t1_fn = getattr(t1, fn if not inplace else fn + "_")
                t2_fn = getattr(t2, fn if not inplace else fn + "_")

                def verify_fallback_warnings(w):
                    self.assertEqual(len(w), 1)
                    self.assertTrue(issubclass(w[0].category, UserWarning))
                    self.assertTrue("Falling back" in str(w[0].message))
                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter('always', UserWarning)
                    r0 = tensorfn(t0_fn, t1, t2)
                    verify_fallback_warnings(w)
                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter('always', UserWarning)
                    r1 = tensorfn(t1_fn, t0, t2)
                    verify_fallback_warnings(w)
                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter('always', UserWarning)
                    r2 = tensorfn(t2_fn, t0, t1)
                    verify_fallback_warnings(w)
                if torch.is_tensor(r0) and fn not in fns_no_result_broadcast:
                    self.assertEqual(t0.size(), r0.size())
                    self.assertEqual(t1.size(), r1.size())
                    self.assertEqual(t2.size(), r2.size())

            # case 4: not broadcastable and not nEleme equal -- tested by test_fallback

    def test_broadcast_fallback(self):
        self._test_broadcast_fallback(self, lambda t: t)

    @staticmethod
    def _test_broadcast_fused_matmul(self, cast):
        fns = ["baddbmm", "addbmm", "addmm", "addmv", "addr"]

        for fn in fns:
            batch_dim = random.randint(1, 8)
            n_dim = random.randint(1, 8)
            m_dim = random.randint(1, 8)
            p_dim = random.randint(1, 8)

            def dims_full_for_fn():
                if fn == "baddbmm":
                    return ([batch_dim, n_dim, p_dim], [batch_dim, n_dim, m_dim], [batch_dim, m_dim, p_dim])
                elif fn == "addbmm":
                    return ([n_dim, p_dim], [batch_dim, n_dim, m_dim], [batch_dim, m_dim, p_dim])
                elif fn == "addmm":
                    return ([n_dim, p_dim], [n_dim, m_dim], [m_dim, p_dim])
                elif fn == "addmv":
                    return ([n_dim], [n_dim, m_dim], [m_dim])
                elif fn == "addr":
                    return ([n_dim, m_dim], [n_dim], [m_dim])
                else:
                    raise AssertionError("unknown function")

            (t0_dims_full, t1_dims, t2_dims) = dims_full_for_fn()
            (t0_dims_small, _, _) = self._select_broadcastable_dims(t0_dims_full)

            t0_small = cast(torch.randn(*t0_dims_small).float())
            t1 = cast(torch.randn(*t1_dims).float())
            t2 = cast(torch.randn(*t2_dims).float())

            t0_full = cast(t0_small.expand(*t0_dims_full))

            fntorch = getattr(torch, fn)
            r0 = fntorch(t0_small, t1, t2)
            r1 = fntorch(t0_full, t1, t2)
            self.assertEqual(r0, r1)

    def test_broadcast_fused_matmul(self):
        self._test_broadcast_fused_matmul(self, lambda t: t)

    @staticmethod
    def _test_broadcast_batched_matmul(self, cast):
        n_dim = random.randint(1, 8)
        m_dim = random.randint(1, 8)
        p_dim = random.randint(1, 8)
        full_batch_dims = [random.randint(1, 3) for i in range(random.randint(1, 3))]
        (batch_dims_small, _, _) = self._select_broadcastable_dims(full_batch_dims)

        def verify_batched_matmul(full_lhs, one_dimensional):
            if not one_dimensional:
                lhs_dims = [n_dim, m_dim]
                rhs_dims = [m_dim, p_dim]
                result_dims = [n_dim, p_dim]
            else:
                lhs_dims = [n_dim, m_dim] if full_lhs else [m_dim]
                rhs_dims = [m_dim, p_dim] if not full_lhs else [m_dim]
                result_dims = [n_dim] if full_lhs else [p_dim]

            lhs_mat_dims = lhs_dims if len(lhs_dims) != 1 else [1, m_dim]
            rhs_mat_dims = rhs_dims if len(rhs_dims) != 1 else [m_dim, 1]
            full_mat_dims = lhs_mat_dims if full_lhs else rhs_mat_dims
            dim0_dims = rhs_dims if full_lhs else lhs_dims
            small_dims = batch_dims_small + (rhs_mat_dims if full_lhs else lhs_mat_dims)

            small = cast(torch.randn(*(small_dims)).float())
            dim0 = cast(torch.randn(*(dim0_dims)).float())
            full = cast(torch.randn(*(full_batch_dims + full_mat_dims)).float())
            if not one_dimensional:
                (lhsTensors, rhsTensors) = ((full,), (small, dim0)) if full_lhs else ((small, dim0), (full,))
            else:
                (lhsTensors, rhsTensors) = ((full,), (dim0,)) if full_lhs else ((dim0,), (full,))

            def maybe_squeeze_result(l, r, result):
                if len(lhs_dims) == 1 and l.dim() != 1:
                    return result.squeeze(-2)
                elif len(rhs_dims) == 1 and r.dim() != 1:
                    return result.squeeze(-1)
                else:
                    return result

            for lhs in lhsTensors:
                lhs_expanded = lhs.expand(*(torch.Size(full_batch_dims) + torch.Size(lhs_mat_dims)))
                lhs_expanded_matmul_fn = getattr(lhs_expanded, "matmul")
                for rhs in rhsTensors:
                    rhs_expanded = ((rhs if len(rhs_dims) != 1 else rhs.unsqueeze(-1)).
                                    expand(*(torch.Size(full_batch_dims) + torch.Size(rhs_mat_dims))))
                    truth = maybe_squeeze_result(lhs_expanded, rhs_expanded, lhs_expanded_matmul_fn(rhs_expanded))
                    for l in (lhs, lhs_expanded):
                        for r in (rhs, rhs_expanded):
                            l_matmul_fn = getattr(l, "matmul")
                            result = maybe_squeeze_result(l, r, l_matmul_fn(r))
                            self.assertEqual(truth, result)
                            # test torch.matmul function as well
                            torch_result = maybe_squeeze_result(l, r, torch.matmul(l, r))
                            self.assertEqual(truth, torch_result)

                # compare to bmm
                bmm_result = (torch.bmm(lhs_expanded.contiguous().view(-1, *lhs_mat_dims),
                                        rhs_expanded.contiguous().view(-1, *rhs_mat_dims)))
                self.assertEqual(truth.view(-1, *result_dims), bmm_result.view(-1, *result_dims))

        for indices in product((True, False), repeat=2):
            verify_batched_matmul(*indices)

    def test_broadcast_batched_matmul(self):
        self._test_broadcast_batched_matmul(self, lambda t: t)

    def test_matmul_out(self):

        def check_matmul(size1, size2):
            a = torch.randn(size1)
            b = torch.randn(size2)
            expected = torch.matmul(a, b)

            out = torch.Tensor(expected.size()).zero_()
            # make output non-contiguous
            out = out.transpose(-1, -2).contiguous().transpose(-1, -2)
            self.assertFalse(out.is_contiguous())

            torch.matmul(a, b, out=out)
            self.assertEqual(expected, out)

        check_matmul((2, 3, 4), (2, 4, 5))
        check_matmul((2, 3, 4), (4, 5))

    def test_broadcast_copy_fn(self):
        torch.zeros(5, 6).copy_(torch.zeros(6))

        def verify_fallback_warnings(w):
            self.assertEqual(len(w), 1)
            self.assertTrue(issubclass(w[0].category, UserWarning))
            self.assertTrue("Falling back" in str(w[0].message))

        # suppress broadcastable warning
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always', UserWarning)
            torch.zeros(5, 6).copy_(torch.zeros(30), broadcast=True)
            verify_fallback_warnings(w)

    def test_randperm(self):
        _RNGState = torch.get_rng_state()
        res1 = torch.randperm(100)
        res2 = torch.LongTensor()
        torch.set_rng_state(_RNGState)
        torch.randperm(100, out=res2)
        self.assertEqual(res1, res2, 0)

    def test_random(self):
        # This test is flaky with p<=(2/(ub-lb))^200=6e-36
        t = torch.FloatTensor(200)
        lb = 1
        ub = 4

        t.fill_(-1)
        t.random_(lb, ub)
        self.assertEqual(t.min(), lb)
        self.assertEqual(t.max(), ub - 1)

        t.fill_(-1)
        t.random_(ub)
        self.assertEqual(t.min(), 0)
        self.assertEqual(t.max(), ub - 1)

    def assertIsOrdered(self, order, x, mxx, ixx, task):
        SIZE = 4
        if order == 'descending':
            def check_order(a, b):
                return a >= b
        elif order == 'ascending':
            def check_order(a, b):
                return a <= b
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

        for _kTries in range(3):
            for _dimTries in range(3):
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
        res1val, res1ind = torch.kthvalue(x, k, False)
        res2val, res2ind = torch.sort(x)

        self.assertEqual(res1val[:, :], res2val[:, :, k - 1], 0)
        self.assertEqual(res1ind[:, :], res2ind[:, :, k - 1], 0)
        # test use of result tensors
        k = random.randint(1, SIZE)
        res1val = torch.Tensor()
        res1ind = torch.LongTensor()
        torch.kthvalue(x, k, False, out=(res1val, res1ind))
        res2val, res2ind = torch.sort(x)
        self.assertEqual(res1val[:, :], res2val[:, :, k - 1], 0)
        self.assertEqual(res1ind[:, :], res2ind[:, :, k - 1], 0)

        # test non-default dim
        k = random.randint(1, SIZE)
        res1val, res1ind = torch.kthvalue(x, k, 0, False)
        res2val, res2ind = torch.sort(x, 0)
        self.assertEqual(res1val, res2val[k - 1], 0)
        self.assertEqual(res1ind, res2ind[k - 1], 0)

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

            nelem = x.nelement()
            res1val = torch.median(x)
            res2val, _ = torch.sort(x.view(nelem))
            ind = int(math.floor((nelem + 1) / 2) - 1)

            self.assertEqual(res2val[ind], res1val, 0)

            res1val, res1ind = torch.median(x, dim=1, keepdim=False)
            res2val, res2ind = torch.sort(x)
            ind = int(math.floor((size + 1) / 2) - 1)

            self.assertEqual(res2val.select(1, ind), res1val, 0)
            self.assertEqual(res2val.select(1, ind), res1val, 0)

            # Test use of result tensor
            res2val = torch.Tensor()
            res2ind = torch.LongTensor()
            torch.median(x, keepdim=False, out=(res2val, res2ind))
            self.assertEqual(res2val, res1val, 0)
            self.assertEqual(res2ind, res1ind, 0)

            # Test non-default dim
            res1val, res1ind = torch.median(x, 0, keepdim=False)
            res2val, res2ind = torch.sort(x, 0)
            self.assertEqual(res1val, res2val[ind], 0)
            self.assertEqual(res1ind, res2ind[ind], 0)

            # input unchanged
            self.assertEqual(x, x0, 0)

    def test_mode(self):
        x = torch.arange(1, SIZE * SIZE + 1).clone().resize_(SIZE, SIZE)
        x[:2] = 1
        x[:, :2] = 1
        x0 = x.clone()

        # Pre-calculated results.
        res1val = torch.Tensor(SIZE).fill_(1)
        # The indices are the position of the last appearance of the mode element.
        res1ind = torch.LongTensor(SIZE).fill_(1)
        res1ind[0] = SIZE - 1
        res1ind[1] = SIZE - 1

        res2val, res2ind = torch.mode(x, keepdim=False)
        self.assertEqual(res1val, res2val, 0)
        self.assertEqual(res1ind, res2ind, 0)

        # Test use of result tensor
        res2val = torch.Tensor()
        res2ind = torch.LongTensor()
        torch.mode(x, keepdim=False, out=(res2val, res2ind))
        self.assertEqual(res1val, res2val, 0)
        self.assertEqual(res1ind, res2ind, 0)

        # Test non-default dim
        res2val, res2ind = torch.mode(x, 0, False)
        self.assertEqual(res1val, res2val, 0)
        self.assertEqual(res1ind, res2ind, 0)

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
        for dim in range(-3, 3):
            pos_dim = dim if dim >= 0 else 3 + dim
            x = torch.rand(13, SIZE, SIZE).transpose(0, pos_dim)
            y = torch.rand(17, SIZE, SIZE).transpose(0, pos_dim)
            z = torch.rand(19, SIZE, SIZE).transpose(0, pos_dim)

            res1 = torch.cat((x, y, z), dim)
            self.assertEqual(res1.narrow(pos_dim, 0, 13), x, 0)
            self.assertEqual(res1.narrow(pos_dim, 13, 17), y, 0)
            self.assertEqual(res1.narrow(pos_dim, 30, 19), z, 0)

        x = torch.randn(20, SIZE, SIZE)
        self.assertEqual(torch.cat(torch.split(x, 7)), x)
        self.assertEqual(torch.cat(torch.chunk(x, 7)), x)

        y = torch.randn(1, SIZE, SIZE)
        z = torch.cat([x, y])
        self.assertEqual(z.size(), (21, SIZE, SIZE))

        self.assertRaises(RuntimeError, lambda: torch.cat([]))

    def test_stack(self):
        x = torch.rand(2, 3, 4)
        y = torch.rand(2, 3, 4)
        z = torch.rand(2, 3, 4)
        for dim in range(4):
            res = torch.stack((x, y, z), dim)
            res_neg = torch.stack((x, y, z), dim - 4)
            expected_size = x.size()[:dim] + (3,) + x.size()[dim:]
            self.assertEqual(res, res_neg)
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

        # check big matrix
        a = torch.randn(1000, 1000)
        q, r = torch.qr(a)
        a_qr = torch.mm(q, r)
        self.assertEqual(a, a_qr, prec=1e-3)

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
    def test_xcorr3_xcorr2_eq_full(self):
        def reference(x, k, o3, o32):
            for i in range(x.size(1)):
                for j in range(k.size(1)):
                    o32[i].add(torch.xcorr2(x[i], k[k.size(1) - j + 1], 'F'))
        self._test_conv_corr_eq(lambda x, k: torch.xcorr3(x, k, 'F'), reference)

    @unittest.skip("Not implemented yet")
    def test_conv3_conv2_eq_valid(self):
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
        forked_value = torch.rand(1000, generator=gen)
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

        lst = [list(range(i, i + 10)) for i in range(0, 100, 10)]
        tensor = torch.DoubleTensor(lst)
        for _i in range(100):
            idx1_start = random.randrange(10)
            idx1_end = idx1_start + random.randrange(1, 10 - idx1_start + 1)
            idx1_step = random.randrange(1, 8)
            idx1 = slice(idx1_start, idx1_end, idx1_step)
            if random.randrange(2) == 0:
                idx2_start = random.randrange(10)
                idx2_end = idx2_start + random.randrange(1, 10 - idx2_start + 1)
                idx2_step = random.randrange(1, 8)
                idx2 = slice(idx2_start, idx2_end, idx2_step)
                lst_indexed = list(map(lambda l: l[idx2], lst[idx1]))
                tensor_indexed = tensor[idx1, idx2]
            else:
                lst_indexed = lst[idx1]
                tensor_indexed = tensor[idx1]
            self.assertEqual(torch.DoubleTensor(lst_indexed), tensor_indexed)

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

    @staticmethod
    def _test_advancedindex(self, conv_fn):
        # Tests for Integer Array Indexing, Part I - Purely integer array
        # indexing

        def consec(size, start=1):
            sequence = torch.ones(int(torch.Tensor(size).prod(0)[0])).cumsum(0)
            sequence.add_(start - 1)
            return sequence.view(*size)

        # pick a random valid indexer type
        def ri(indices):
            choice = random.randint(0, 2)
            if choice == 0:
                return conv_fn(torch.LongTensor(indices))
            elif choice == 1:
                return list(indices)
            else:
                return tuple(indices)

        # First, we will test indexing to generate return values

        # Case 1: Purely Integer Array Indexing
        reference = conv_fn(consec((10,)))
        self.assertEqual(reference[ri([0]), ], consec((1,)))
        self.assertEqual(reference[ri([3]), ], consec((1,), 4))
        self.assertEqual(reference[ri([2, 3, 4]), ], consec((3,), 3))
        self.assertEqual(reference[ri([0, 2, 4]), ], torch.Tensor([1, 3, 5]))

        # setting values
        reference[ri([0],), ] = -1
        self.assertEqual(reference[ri([0]), ], torch.Tensor([-1]))
        reference[ri([2, 3, 4]), ] = 3
        self.assertEqual(reference[ri([2, 3, 4]), ], torch.Tensor([3, 3, 3]))
        reference[ri([0, 2, 4]), ] = conv_fn(torch.Tensor([5, 4, 3]))
        self.assertEqual(reference[ri([0, 2, 4]), ], torch.Tensor([5, 4, 3]))

        # Tensor with stride != 1

        # strided is [1, 3, 5, 7]
        reference = conv_fn(consec((10,)))
        strided = conv_fn(torch.Tensor())
        strided.set_(reference.storage(), storage_offset=0,
                     size=torch.Size([4]), stride=[2])

        self.assertEqual(strided[ri([0]), ], torch.Tensor([1]))
        self.assertEqual(strided[ri([3]), ], torch.Tensor([7]))
        self.assertEqual(strided[ri([1, 2]), ], torch.Tensor([3, 5]))
        self.assertEqual(strided[ri([[2, 1], [0, 3]]), ],
                         torch.Tensor([[5, 3], [1, 7]]))

        # stride is [4, 8]
        strided = conv_fn(torch.Tensor())
        strided.set_(reference.storage(), storage_offset=4,
                     size=torch.Size([2]), stride=[4])
        self.assertEqual(strided[ri([0]), ], torch.Tensor([5]))
        self.assertEqual(strided[ri([1]), ], torch.Tensor([9]))
        self.assertEqual(strided[ri([0, 1]), ], torch.Tensor([5, 9]))
        self.assertEqual(strided[ri([[0, 1], [1, 0]]), ],
                         torch.Tensor([[5, 9], [9, 5]]))

        # reference is 1 2
        #              3 4
        #              5 6
        reference = conv_fn(consec((3, 2)))
        self.assertEqual(reference[ri([0, 1, 2]), ri([0])], torch.Tensor([1, 3, 5]))
        self.assertEqual(reference[ri([0, 1, 2]), ri([1])], torch.Tensor([2, 4, 6]))
        self.assertEqual(reference[ri([0]), ri([0])], consec((1,)))
        self.assertEqual(reference[ri([2]), ri([1])], consec((1,), 6))
        self.assertEqual(reference[[ri([0, 0]), ri([0, 1])]], torch.Tensor([1, 2]))
        self.assertEqual(reference[[ri([0, 1, 1, 0, 2]), ri([1])]],
                         torch.Tensor([2, 4, 4, 2, 6]))
        self.assertEqual(reference[[ri([0, 0, 1, 1]), ri([0, 1, 0, 0])]],
                         torch.Tensor([1, 2, 3, 3]))

        rows = ri([[0, 0],
                   [1, 2]])
        columns = [0],
        self.assertEqual(reference[rows, columns], torch.Tensor([[1, 1],
                                                                [3, 5]]))

        rows = ri([[0, 0],
                   [1, 2]])
        columns = ri([1, 0])
        self.assertEqual(reference[rows, columns], torch.Tensor([[2, 1],
                                                                [4, 5]]))
        rows = ri([[0, 0],
                   [1, 2]])
        columns = ri([[0, 1],
                      [1, 0]])
        self.assertEqual(reference[rows, columns], torch.Tensor([[1, 2],
                                                                [4, 5]]))

        # setting values
        reference[ri([0]), ri([1])] = -1
        self.assertEqual(reference[ri([0]), ri([1])], torch.Tensor([-1]))
        reference[ri([0, 1, 2]), ri([0])] = conv_fn(torch.Tensor([-1, 2, -4]))
        self.assertEqual(reference[ri([0, 1, 2]), ri([0])], torch.Tensor([-1,
                         2, -4]))
        reference[rows, columns] = conv_fn(torch.Tensor([[4, 6], [2, 3]]))
        self.assertEqual(reference[rows, columns],
                         torch.Tensor([[4, 6], [2, 3]]))

        # Verify still works with Tranposed (i.e. non-contiguous) Tensors

        reference = conv_fn(torch.Tensor([[0, 1, 2, 3],
                                          [4, 5, 6, 7],
                                          [8, 9, 10, 11]])).t_()

        # Tranposed: [[0, 4, 8],
        #             [1, 5, 9],
        #             [2, 6, 10],
        #             [3, 7, 11]]

        self.assertEqual(reference[ri([0, 1, 2]), ri([0])], torch.Tensor([0, 1,
                         2]))
        self.assertEqual(reference[ri([0, 1, 2]), ri([1])], torch.Tensor([4, 5,
                         6]))
        self.assertEqual(reference[ri([0]), ri([0])], torch.Tensor([0]))
        self.assertEqual(reference[ri([2]), ri([1])], torch.Tensor([6]))
        self.assertEqual(reference[[ri([0, 0]), ri([0, 1])]], torch.Tensor([0, 4]))
        self.assertEqual(reference[[ri([0, 1, 1, 0, 3]), ri([1])]],
                         torch.Tensor([4, 5, 5, 4, 7]))
        self.assertEqual(reference[[ri([0, 0, 1, 1]), ri([0, 1, 0, 0])]],
                         torch.Tensor([0, 4, 1, 1]))

        rows = ri([[0, 0],
                   [1, 2]])
        columns = [0],
        self.assertEqual(reference[rows, columns], torch.Tensor([[0, 0],
                                                                [1, 2]]))

        rows = ri([[0, 0],
                   [1, 2]])
        columns = ri([1, 0])
        self.assertEqual(reference[rows, columns], torch.Tensor([[4, 0],
                                                                [5, 2]]))
        rows = ri([[0, 0],
                   [1, 3]])
        columns = ri([[0, 1],
                      [1, 2]])
        self.assertEqual(reference[rows, columns], torch.Tensor([[0, 4],
                                                                [5, 11]]))

        # setting values
        reference[ri([0]), ri([1])] = -1
        self.assertEqual(reference[ri([0]), ri([1])], torch.Tensor([-1]))
        reference[ri([0, 1, 2]), ri([0])] = conv_fn(torch.Tensor([-1, 2, -4]))
        self.assertEqual(reference[ri([0, 1, 2]), ri([0])], torch.Tensor([-1,
                         2, -4]))
        reference[rows, columns] = conv_fn(torch.Tensor([[4, 6], [2, 3]]))
        self.assertEqual(reference[rows, columns],
                         torch.Tensor([[4, 6], [2, 3]]))

        # stride != 1

        # strided is [[1 3 5 7],
        #             [9 11 13 15]]

        reference = conv_fn(torch.arange(0, 24).view(3, 8))
        strided = conv_fn(torch.Tensor())
        strided.set_(reference.storage(), 1, size=torch.Size([2, 4]),
                     stride=[8, 2])

        self.assertEqual(strided[ri([0, 1]), ri([0])], torch.Tensor([1, 9]))
        self.assertEqual(strided[ri([0, 1]), ri([1])], torch.Tensor([3, 11]))
        self.assertEqual(strided[ri([0]), ri([0])], torch.Tensor([1]))
        self.assertEqual(strided[ri([1]), ri([3])], torch.Tensor([15]))
        self.assertEqual(strided[[ri([0, 0]), ri([0, 3])]], torch.Tensor([1, 7]))
        self.assertEqual(strided[[ri([1]), ri([0, 1, 1, 0, 3])]],
                         torch.Tensor([9, 11, 11, 9, 15]))
        self.assertEqual(strided[[ri([0, 0, 1, 1]), ri([0, 1, 0, 0])]],
                         torch.Tensor([1, 3, 9, 9]))

        rows = ri([[0, 0],
                   [1, 1]])
        columns = [0],
        self.assertEqual(strided[rows, columns], torch.Tensor([[1, 1],
                                                              [9, 9]]))

        rows = ri([[0, 1],
                   [1, 0]])
        columns = ri([1, 2])
        self.assertEqual(strided[rows, columns], torch.Tensor([[3, 13],
                                                              [11, 5]]))
        rows = ri([[0, 0],
                   [1, 1]])
        columns = ri([[0, 1],
                      [1, 2]])
        self.assertEqual(strided[rows, columns], torch.Tensor([[1, 3],
                                                              [11, 13]]))

        # setting values

        # strided is [[10, 11],
        #             [17, 18]]

        reference = conv_fn(torch.arange(0, 24).view(3, 8))
        strided = conv_fn(torch.Tensor())
        strided.set_(reference.storage(), 10, size=torch.Size([2, 2]),
                     stride=[7, 1])
        self.assertEqual(strided[ri([0]), ri([1])], torch.Tensor([11]))
        strided[ri([0]), ri([1])] = -1
        self.assertEqual(strided[ri([0]), ri([1])], torch.Tensor([-1]))

        reference = conv_fn(torch.arange(0, 24).view(3, 8))
        strided = conv_fn(torch.Tensor())
        strided.set_(reference.storage(), 10, size=torch.Size([2, 2]),
                     stride=[7, 1])
        self.assertEqual(strided[ri([0, 1]), ri([1, 0])], torch.Tensor([11,
                         17]))
        strided[ri([0, 1]), ri([1, 0])] = conv_fn(torch.Tensor([-1, 2]))
        self.assertEqual(strided[ri([0, 1]), ri([1, 0])], torch.Tensor([-1,
                         2]))

        reference = conv_fn(torch.arange(0, 24).view(3, 8))
        strided = conv_fn(torch.Tensor())
        strided.set_(reference.storage(), 10, size=torch.Size([2, 2]),
                     stride=[7, 1])

        rows = ri([[0],
                   [1]])
        columns = ri([[0, 1],
                      [0, 1]])
        self.assertEqual(strided[rows, columns],
                         torch.Tensor([[10, 11], [17, 18]]))
        strided[rows, columns] = conv_fn(torch.Tensor([[4, 6], [2, 3]]))
        self.assertEqual(strided[rows, columns],
                         torch.Tensor([[4, 6], [2, 3]]))

        # Tests using less than the number of dims, and ellipsis

        # reference is 1 2
        #              3 4
        #              5 6
        reference = conv_fn(consec((3, 2)))
        self.assertEqual(reference[ri([0, 2]), ], torch.Tensor([[1, 2], [5, 6]]))
        self.assertEqual(reference[ri([1]), ...], torch.Tensor([[3, 4]]))
        self.assertEqual(reference[..., ri([1])], torch.Tensor([[2], [4], [6]]))

        if TEST_NUMPY:
            # we use numpy to compare against, to verify that our advanced
            # indexing semantics are the same, and also for ease of test
            # writing

            def tensor_indices_to_np(tensor, indices):
                # convert the Torch Tensor to a numpy array
                if (tensor.is_cuda):
                    tensor = tensor.cpu()
                npt = tensor.numpy()

                # convert indices
                idxs = tuple(i.tolist() if isinstance(i, torch.LongTensor) else
                             i for i in indices)

                return npt, idxs

            def get_numpy(tensor, indices):
                npt, idxs = tensor_indices_to_np(tensor, indices)

                # index and return as a Torch Tensor
                return torch.Tensor(npt[idxs])

            def set_numpy(tensor, indices, value):
                if not isinstance(value, int):
                    if value.is_cuda:
                        value = value.cpu()
                    value = value.numpy()

                npt, idxs = tensor_indices_to_np(tensor, indices)
                npt[idxs] = value
                return npt

            def assert_get_eq(tensor, indexer):
                self.assertEqual(reference[indexer],
                                 conv_fn(get_numpy(reference, indexer)))

            def assert_set_eq(tensor, indexer, val):
                pyt = tensor.clone()
                numt = tensor.clone()
                pyt[indexer] = val
                numt = conv_fn(torch.Tensor(set_numpy(numt, indexer, val)))
                self.assertEqual(pyt, numt)

            def get_set_tensor(indexed, indexer):
                set_size = indexed[indexer].size()
                set_count = indexed[indexer].numel()
                set_tensor = conv_fn(torch.randperm(set_count).view(set_size).double())
                return set_tensor

            # Tensor is  0  1  2  3  4
            #            5  6  7  8  9
            #           10 11 12 13 14
            #           15 16 17 18 19
            reference = conv_fn(torch.arange(0, 20).view(4, 5))

            indices_to_test = [
                # grab the second, fourth columns
                [slice(None), [1, 3]],

                # first, third rows,
                [[0, 2], slice(None)],

                # weird shape
                [slice(None), [[0, 1],
                               [2, 3]]]
            ]

            # only test dupes on gets
            get_indices_to_test = indices_to_test + [[slice(None), [0, 1, 1, 2, 2]]]

            for indexer in get_indices_to_test:
                assert_get_eq(reference, indexer)

            for indexer in indices_to_test:
                assert_set_eq(reference, indexer, 44)
                assert_set_eq(reference,
                              indexer,
                              get_set_tensor(reference, indexer))

            reference = conv_fn(torch.arange(0, 160).view(4, 8, 5))

            indices_to_test = [
                [slice(None), slice(None), [0, 3, 4]],
                [slice(None), [2, 4, 5, 7], slice(None)],
                [[2, 3], slice(None), slice(None)],
                [slice(None), [0, 2, 3], [1, 3, 4]],
                [slice(None), [0], [1, 2, 4]],
                [slice(None), [0, 1, 3], [4]],
                [slice(None), [[0, 1], [1, 0]], [[2, 3], [3, 0]]],
                [slice(None), [[0, 1], [1, 0]], [[2, 3]]],
                [slice(None), [[0, 1], [2, 3]], [[0]]],
                [slice(None), [[5, 6]], [[0, 3], [4, 4]]],
                [slice(None), [[2]], [[0, 3], [4, 4]]],
                [[0, 2, 3], [1, 3, 4], slice(None)],
                [[0], [1, 2, 4], slice(None)],
                [[0, 1, 3], [4], slice(None)],
                [[[0, 1], [1, 0]], [[2, 1], [3, 5]], slice(None)],
                [[[0, 1], [1, 0]], [[2, 3]], slice(None)],
                [[[0, 1], [2, 3]], [[0]], slice(None)],
                [[[2, 1]], [[0, 3], [4, 4]], slice(None)],
                [[[2]], [[0, 3], [4, 1]], slice(None)],

                # less dim, ellipsis
                [[0, 2], ],
                [[0, 2], slice(None)],
                [[0, 2], Ellipsis],
                [[0, 2], slice(None), Ellipsis],
                [[0, 2], Ellipsis, slice(None)],
                [[0, 2], [1, 3]],
                [[0, 2], [1, 3], Ellipsis],
                [Ellipsis, [1, 3], [2, 3]],
                [Ellipsis, [2, 3, 4]],
                [Ellipsis, slice(None), [2, 3, 4]],
                [slice(None), Ellipsis, [2, 3, 4]],
            ]

            for indexer in indices_to_test:
                assert_get_eq(reference, indexer)
                assert_set_eq(reference, indexer, 212)
                assert_set_eq(reference,
                              indexer,
                              get_set_tensor(reference, indexer))

            reference = conv_fn(torch.arange(0, 1296).view(3, 9, 8, 6))

            indices_to_test = [
                [slice(None), slice(None), slice(None), [0, 3, 4]],
                [slice(None), slice(None), [2, 4, 5, 7], slice(None)],
                [slice(None), [2, 3], slice(None), slice(None)],
                [[1, 2], slice(None), slice(None), slice(None)],
                [slice(None), slice(None), [0, 2, 3], [1, 3, 4]],
                [slice(None), slice(None), [0], [1, 2, 4]],
                [slice(None), slice(None), [0, 1, 3], [4]],
                [slice(None), slice(None), [[0, 1], [1, 0]], [[2, 3]]],
                [slice(None), slice(None), [[0, 1], [2, 3]], [[0]]],
                [slice(None), slice(None), [[5, 6]], [[0, 3], [4, 4]]],
                [slice(None), [0, 2, 3], [1, 3, 4], slice(None)],
                [slice(None), [0], [1, 2, 4], slice(None)],
                [slice(None), [0, 1, 3], [4], slice(None)],
                [slice(None), [[0, 1], [3, 4]], [[2, 3], [0, 1]], slice(None)],
                [slice(None), [[0, 1], [3, 4]], [[2, 3]], slice(None)],
                [slice(None), [[0, 1], [3, 2]], [[0]], slice(None)],
                [slice(None), [[2, 1]], [[0, 3], [6, 4]], slice(None)],
                [slice(None), [[2]], [[0, 3], [4, 2]], slice(None)],
                [[0, 1, 2], [1, 3, 4], slice(None), slice(None)],
                [[0], [1, 2, 4], slice(None), slice(None)],
                [[0, 1, 2], [4], slice(None), slice(None)],
                [[[0, 1], [0, 2]], [[2, 4], [1, 5]], slice(None), slice(None)],
                [[[0, 1], [1, 2]], [[2, 0]], slice(None), slice(None)],
                [[[2, 2]], [[0, 3], [4, 5]], slice(None), slice(None)],
                [[[2]], [[0, 3], [4, 5]], slice(None), slice(None)],
                [slice(None), [3, 4, 6], [0, 2, 3], [1, 3, 4]],
                [slice(None), [2, 3, 4], [1, 3, 4], [4]],
                [slice(None), [0, 1, 3], [4], [1, 3, 4]],
                [slice(None), [6], [0, 2, 3], [1, 3, 4]],
                [slice(None), [2, 3, 5], [3], [4]],
                [slice(None), [0], [4], [1, 3, 4]],
                [slice(None), [6], [0, 2, 3], [1]],
                [slice(None), [[0, 3], [3, 6]], [[0, 1], [1, 3]], [[5, 3], [1, 2]]],
                [[2, 2, 1], [0, 2, 3], [1, 3, 4], slice(None)],
                [[2, 0, 1], [1, 2, 3], [4], slice(None)],
                [[0, 1, 2], [4], [1, 3, 4], slice(None)],
                [[0], [0, 2, 3], [1, 3, 4], slice(None)],
                [[0, 2, 1], [3], [4], slice(None)],
                [[0], [4], [1, 3, 4], slice(None)],
                [[1], [0, 2, 3], [1], slice(None)],
                [[[1, 2], [1, 2]], [[0, 1], [2, 3]], [[2, 3], [3, 5]], slice(None)],

                # less dim, ellipsis
                [Ellipsis, [0, 3, 4]],
                [Ellipsis, slice(None), [0, 3, 4]],
                [Ellipsis, slice(None), slice(None), [0, 3, 4]],
                [slice(None), Ellipsis, [0, 3, 4]],
                [slice(None), slice(None), Ellipsis, [0, 3, 4]],
                [slice(None), [0, 2, 3], [1, 3, 4]],
                [slice(None), [0, 2, 3], [1, 3, 4], Ellipsis],
                [Ellipsis, [0, 2, 3], [1, 3, 4], slice(None)],
                [[0], [1, 2, 4]],
                [[0], [1, 2, 4], slice(None)],
                [[0], [1, 2, 4], Ellipsis],
                [[0], [1, 2, 4], Ellipsis, slice(None)],
                [[1], ],
                [[0, 2, 1], [3], [4]],
                [[0, 2, 1], [3], [4], slice(None)],
                [[0, 2, 1], [3], [4], Ellipsis],
                [Ellipsis, [0, 2, 1], [3], [4]],
            ]

            for indexer in indices_to_test:
                assert_get_eq(reference, indexer)
                assert_set_eq(reference, indexer, 1333)
                assert_set_eq(reference,
                              indexer,
                              get_set_tensor(reference, indexer))
            indices_to_test += [
                [slice(None), slice(None), [[0, 1], [1, 0]], [[2, 3], [3, 0]]],
                [slice(None), slice(None), [[2]], [[0, 3], [4, 4]]],
            ]
            for indexer in indices_to_test:
                assert_get_eq(reference, indexer)
                assert_set_eq(reference, indexer, 1333)

    def test_advancedindex(self):
        self._test_advancedindex(self, lambda x: x)

    @staticmethod
    def _test_advancedindex_big(self, conv_fn):
        reference = conv_fn(torch.arange(0, 123344).int())

        self.assertEqual(reference[[0, 123, 44488, 68807, 123343], ],
                         torch.LongTensor([0, 123, 44488, 68807, 123343]))

    def test_advancedindex_big(self):
        self._test_advancedindex_big(self, lambda x: x)

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
    @staticmethod
    def _fill_indices(self, idx, dim, dim_size, elems_per_row, m, n, o):
        for i in range(1 if dim == 0 else m):
            for j in range(1 if dim == 1 else n):
                for k in range(1 if dim == 2 else o):
                    ii = [i, j, k]
                    ii[dim] = slice(0, idx.size(dim) + 1)
                    idx[tuple(ii)] = torch.randperm(dim_size)[0:elems_per_row]

    @staticmethod
    def _test_gather(self, cast, test_bounds=True):
        m, n, o = random.randint(10, 20), random.randint(10, 20), random.randint(10, 20)
        elems_per_row = random.randint(1, 10)
        dim = random.randrange(3)

        src = torch.randn(m, n, o)
        idx_size = [m, n, o]
        idx_size[dim] = elems_per_row
        idx = torch.LongTensor().resize_(*idx_size)
        TestTorch._fill_indices(self, idx, dim, src.size(dim), elems_per_row, m, n, o)

        src = cast(src)
        idx = cast(idx)

        actual = torch.gather(src, dim, idx)
        expected = cast(torch.Tensor().resize_(*idx_size))
        for i in range(idx_size[0]):
            for j in range(idx_size[1]):
                for k in range(idx_size[2]):
                    ii = [i, j, k]
                    ii[dim] = idx[i, j, k]
                    expected[i, j, k] = src[tuple(ii)]
        self.assertEqual(actual, expected, 0)

        if test_bounds:
            idx[0][0][0] = 23
            self.assertRaises(RuntimeError, lambda: torch.gather(src, dim, idx))

        src = cast(torch.randn(3, 4, 5))
        expected, idx = src.max(2, True)
        expected = cast(expected)
        idx = cast(idx)
        actual = torch.gather(src, 2, idx)
        self.assertEqual(actual, expected, 0)

    def test_gather(self):
        self._test_gather(self, lambda t: t)

    @staticmethod
    def _test_scatter_base(self, cast, method, is_scalar=False, test_bounds=True):
        m, n, o = random.randint(10, 20), random.randint(10, 20), random.randint(10, 20)
        elems_per_row = random.randint(1, 10)
        dim = random.randrange(3)

        idx_size = [m, n, o]
        idx_size[dim] = elems_per_row
        idx = cast(torch.LongTensor().resize_(*idx_size))
        TestTorch._fill_indices(self, idx, dim, ([m, n, o])[dim], elems_per_row, m, n, o)

        if is_scalar:
            src = random.random()
        else:
            src = cast(torch.Tensor(*idx_size).normal_())

        base = cast(torch.randn(m, n, o))
        actual = getattr(base.clone(), method)(dim, idx, src)
        expected = base.clone()
        for i in range(idx_size[0]):
            for j in range(idx_size[1]):
                for k in range(idx_size[2]):
                    ii = [i, j, k]
                    ii[dim] = idx[i, j, k]
                    if method == 'scatter_' and not is_scalar:
                        expected[tuple(ii)] = src[i, j, k]
                    elif method == 'scatter_add_':
                        expected[tuple(ii)] += src[i, j, k]
                    else:
                        expected[tuple(ii)] = src
        self.assertEqual(actual, expected, 0)

        if test_bounds:
            idx[0][0][0] = 34
            with self.assertRaises(RuntimeError):
                getattr(base.clone(), method)(dim, idx, src)

    def test_scatter(self):
        self._test_scatter_base(self, lambda t: t, 'scatter_')

    def test_scatterAdd(self):
        self._test_scatter_base(self, lambda t: t, 'scatter_add_')

    def test_scatterFill(self):
        self._test_scatter_base(self, lambda t: t, 'scatter_', True)

    def test_masked_scatter(self):
        num_copy, num_dest = 3, 10
        dest = torch.randn(num_dest)
        src = torch.randn(num_copy)
        mask = torch.ByteTensor((0, 0, 0, 0, 1, 0, 1, 0, 1, 0))
        dest2 = dest.clone()
        dest.masked_scatter_(mask, src)
        j = 0
        for i in range(num_dest):
            if mask[i]:
                dest2[i] = src[j]
                j += 1
        self.assertEqual(dest, dest2, 0)

        # make source bigger than number of 1s in mask
        src = torch.randn(num_dest)
        dest.masked_scatter_(mask, src)

        # make src smaller. this should fail
        src = torch.randn(num_copy - 1)
        with self.assertRaises(RuntimeError):
            dest.masked_scatter_(mask, src)

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
            # abs is used in assertEqual so we use the slow version instead
            self.assertTensorsSlowEqual(res.abs(), data, 1e-16)

        # Checking that the right abs function is called for LongTensor
        bignumber = 2 ^ 31 + 1
        res = torch.LongTensor((-bignumber,))
        self.assertGreater(res.abs()[0], 0)

    def test_unbiased(self):
        tensor = torch.randn(100)
        self.assertEqual(tensor.var(0), tensor.var(0, unbiased=True))
        self.assertEqual(tensor.var(), tensor.var(unbiased=True))
        self.assertEqual(tensor.var(unbiased=False), tensor.var(0, unbiased=False)[0])

        tensor = torch.FloatTensor([1.0, 2.0])
        self.assertEqual(tensor.var(unbiased=True), 0.5)
        self.assertEqual(tensor.var(unbiased=False), 0.25)

        tensor = torch.randn(100)
        self.assertEqual(tensor.std(0), tensor.std(0, unbiased=True))
        self.assertEqual(tensor.std(), tensor.std(unbiased=True))
        self.assertEqual(tensor.std(unbiased=False), tensor.std(0, unbiased=False)[0])

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
        # suppress broadcastable warning
        with warnings.catch_warnings(record=True):
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
        t1.set_(t2.storage(), 0, size=size, stride=stride)
        self.assertEqual(t1.size(), size)
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
        b += [torch.arange(1, 11).int()]
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
            self.assertEqual(c[4][1:4], c[5], 0)

            # check that serializing the same storage view object unpickles
            # it as one object not two (and vice versa)
            views = c[7]
            self.assertEqual(views[0]._cdata, views[1]._cdata)
            self.assertEqual(views[0], views[2])
            self.assertNotEqual(views[0]._cdata, views[2]._cdata)

            rootview = c[8]
            self.assertEqual(rootview.data_ptr(), c[0].data_ptr())

    def test_half_tensor(self):
        x = torch.randn(5, 5).float()
        y = torch.randn(5, 5).float()
        xh, yh = x.half(), y.half()

        self.assertEqual(x.half().float(), x, 1e-3)

        z = torch.Tensor(5, 5)
        self.assertEqual(z.copy_(xh), x, 1e-3)

        with tempfile.NamedTemporaryFile() as f:
            torch.save(xh, f)
            f.seek(0)
            xh2 = torch.load(f)
            self.assertEqual(xh.float(), xh2.float())

    @unittest.skipIf(not torch.cuda.is_available(), 'no CUDA')
    def test_half_tensor_cuda(self):
        x = torch.randn(5, 5).half()
        self.assertEqual(x.cuda(), x)

        xc = x.cuda()
        with tempfile.NamedTemporaryFile() as f:
            torch.save(xc, f)
            f.seek(0)
            xc2 = torch.load(f)
            self.assertIsInstance(xc2, type(xc))
            self.assertEqual(xc.float(), xc2.float())

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
        a = [torch.arange(1 + i, 26 + i).view(5, 5).float() for i in range(2)]
        b = [a[i % 2] for i in range(4)]
        b += [a[0].storage()]
        b += [a[0].storage()[1:4]]
        path = download_file('https://download.pytorch.org/test_data/legacy_serialized.pt')
        c = torch.load(path)
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
        self.assertEqual(c[4][1:4], c[5], 0)

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

    def test_serialization_map_location(self):
        test_file_path = download_file('https://download.pytorch.org/test_data/gpu_tensors.pt')

        def map_location(storage, loc):
            return storage

        tensor = torch.load(test_file_path, map_location=map_location)
        self.assertEqual(type(tensor), torch.FloatTensor)
        self.assertEqual(tensor, torch.FloatTensor([[1.0, 2.0], [3.0, 4.0]]))

        tensor = torch.load(test_file_path, map_location={'cuda:0': 'cpu'})
        self.assertEqual(type(tensor), torch.FloatTensor)
        self.assertEqual(tensor, torch.FloatTensor([[1.0, 2.0], [3.0, 4.0]]))

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

    def test_from_file(self):
        size = 10000
        with tempfile.NamedTemporaryFile() as f:
            s1 = torch.FloatStorage.from_file(f.name, True, size)
            t1 = torch.FloatTensor(s1).copy_(torch.randn(size))

            # check mapping
            s2 = torch.FloatStorage.from_file(f.name, True, size)
            t2 = torch.FloatTensor(s2)
            self.assertEqual(t1, t2, 0)

            # check changes to t1 from t2
            rnum = random.uniform(-1, 1)
            t1.fill_(rnum)
            self.assertEqual(t1, t2, 0)

            # check changes to t2 from t1
            rnum = random.uniform(-1, 1)
            t2.fill_(rnum)
            self.assertEqual(t1, t2, 0)

    def test_print(self):
        for t in torch._tensor_classes:
            if t == torch.HalfTensor:
                continue  # HalfTensor does not support fill
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

        x = torch.Tensor()
        self.assertEqual(list(x), [])

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
            np.int16,
            np.uint8
        ]
        for dtype in dtypes:
            array = np.array([1, 2, 3, 4], dtype=dtype)
            self.assertEqual(torch.from_numpy(array), torch.Tensor([1, 2, 3, 4]))

        # check storage offset
        x = np.linspace(1, 125, 125)
        x.shape = (5, 5, 5)
        x = x[1]
        expected = torch.arange(1, 126).view(5, 5, 5)[1]
        self.assertEqual(torch.from_numpy(x), expected)

        # check noncontiguous
        x = np.linspace(1, 25, 25)
        x.shape = (5, 5)
        expected = torch.arange(1, 26).view(5, 5).t()
        self.assertEqual(torch.from_numpy(x.T), expected)

        # check noncontiguous with holes
        x = np.linspace(1, 125, 125)
        x.shape = (5, 5, 5)
        x = x[:, 1]
        expected = torch.arange(1, 126).view(5, 5, 5)[:, 1]
        self.assertEqual(torch.from_numpy(x), expected)

        # check zero dimensional
        x = np.zeros((0, 2))
        self.assertRaises(RuntimeError, lambda: torch.from_numpy(x))

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

        invert_result = ~x
        for idx in iter_indices(x):
            self.assertEqual(1 - x[idx], invert_result[idx])

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
        x = torch.arange(1, 6)
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

    # unit test for THTensor_(copyTranspose)
    @unittest.skipIf(not TEST_NUMPY, "Numpy not found")
    def test_big_transpose(self):
        t = torch.rand(456, 789)
        t1 = t.t().contiguous()
        t2 = torch.from_numpy(t.numpy().transpose())
        self.assertEqual(t1, t2)

    def test_inplace_division(self):
        t = torch.rand(5, 5)
        id_before = id(t)
        t /= 2
        id_after = id(t)
        self.assertEqual(id_before, id_after)

# Functions to test negative dimension wrapping
METHOD = 1
INPLACE_METHOD = 2
FUNCTIONAL = 4
DIM_ARG = None


def make_neg_dim_test(name, tensor_arg, arg_constr, types, extra_dim=0):
    def neg_dim_test(self):
        if isinstance(tensor_arg, list):
            assert METHOD not in types and INPLACE_METHOD not in types
            x = [torch.randn(arg) for arg in tensor_arg]
            ndim = len(tensor_arg[-1])
        else:
            x = torch.randn(*tensor_arg)
            ndim = len(tensor_arg)
        ndim += extra_dim

        n_dim_to_test = sum(map(lambda e: e is DIM_ARG, arg_constr()))

        for dims_val in combinations(range(ndim), n_dim_to_test):
            arg = arg_constr()
            arg_neg = copy.deepcopy(arg)
            idx = 0
            for i, v in enumerate(arg):
                if v is DIM_ARG:
                    arg[i] = dims_val[idx]
                    arg_neg[i] = dims_val[idx] - ndim
                    idx += 1

            if METHOD in types:
                a = getattr(x, name)(*arg)
                b = getattr(x, name)(*arg_neg)
                self.assertEqual(a, b)

            if INPLACE_METHOD in types:
                a = x.clone()
                getattr(a, name + '_')(*arg)
                b = x.clone()
                getattr(b, name + '_')(*arg_neg)
                self.assertEqual(a, b)

            if FUNCTIONAL in types:
                a = getattr(torch, name)(x, *arg)
                b = getattr(torch, name)(x, *arg_neg)
                self.assertEqual(a, b)

    return neg_dim_test


def idx_tensor(size, max_val):
    return torch.LongTensor(*size).random_(0, max_val - 1)

neg_dim_tests = [
    ('narrow', (10, 20, 30), lambda: [DIM_ARG, 0, 5], [METHOD]),
    ('transpose', (10, 20, 30), lambda: [DIM_ARG, DIM_ARG], [METHOD, INPLACE_METHOD, FUNCTIONAL]),
    ('size', (10, 20, 30), lambda: [DIM_ARG], [METHOD]),
    ('cat', [(2, 3, 4), (2, 3, 4)], lambda: [DIM_ARG], [FUNCTIONAL]),
    ('chunk', (10, 20, 30), lambda: [5, DIM_ARG], [METHOD, FUNCTIONAL]),
    ('gather', (10, 20), lambda: [DIM_ARG, idx_tensor((10, 20), 10)], [METHOD, FUNCTIONAL]),
    ('index_select', (10, 10), lambda: [DIM_ARG, idx_tensor((10,), 10)], [METHOD, FUNCTIONAL]),
    ('split', (10, 20), lambda: [5, DIM_ARG], [METHOD, FUNCTIONAL]),
    ('squeeze', (10, 1, 20, 1), lambda: [DIM_ARG], [METHOD, INPLACE_METHOD, FUNCTIONAL]),
    ('unbind', (2, 3, 4), lambda: [DIM_ARG], [FUNCTIONAL]),
    ('unsqueeze', (10, 20), lambda: [DIM_ARG], [METHOD, INPLACE_METHOD, FUNCTIONAL], 1),
    ('cumprod', (10, 20), lambda: [DIM_ARG], [METHOD, FUNCTIONAL]),
    ('cumsum', (10, 20), lambda: [DIM_ARG], [METHOD, FUNCTIONAL]),
    ('mean', (10, 20), lambda: [DIM_ARG], [METHOD, FUNCTIONAL]),
    ('median', (10, 20), lambda: [DIM_ARG], [METHOD, FUNCTIONAL]),
    ('mode', (10, 20), lambda: [DIM_ARG], [METHOD, FUNCTIONAL]),
    ('norm', (10, 20), lambda: [2, DIM_ARG], [METHOD, FUNCTIONAL]),
    ('prod', (10, 20), lambda: [DIM_ARG], [METHOD, FUNCTIONAL]),
    ('std', (10, 20), lambda: [DIM_ARG], [METHOD, FUNCTIONAL]),
    ('sum', (10, 20), lambda: [DIM_ARG], [METHOD, FUNCTIONAL]),
    ('var', (10, 20), lambda: [DIM_ARG], [METHOD, FUNCTIONAL]),
    ('kthvalue', (10, 20), lambda: [3, DIM_ARG], [METHOD, FUNCTIONAL]),
    ('max', (10, 20), lambda: [DIM_ARG], [METHOD, FUNCTIONAL]),
    ('min', (10, 20), lambda: [DIM_ARG], [METHOD, FUNCTIONAL]),
    ('sort', (10, 20), lambda: [DIM_ARG], [METHOD, FUNCTIONAL]),
    ('topk', (10, 20), lambda: [5, DIM_ARG], [METHOD, FUNCTIONAL]),
    ('renorm', (10, 20), lambda: [2, DIM_ARG, 1], [METHOD, INPLACE_METHOD, FUNCTIONAL]),
    ('index_add', (10, 10), lambda: [DIM_ARG, idx_tensor((10,), 10), torch.randn(10, 10)], [INPLACE_METHOD]),
    ('index_copy', (10, 10), lambda: [DIM_ARG, idx_tensor((10,), 10), torch.randn(10, 10)], [INPLACE_METHOD]),
    ('index_fill', (10, 10), lambda: [DIM_ARG, idx_tensor((10,), 10), 12], [INPLACE_METHOD]),
    ('scatter', (10, 10), lambda: [DIM_ARG, idx_tensor((10, 10), 10), torch.randn(10, 10)], [INPLACE_METHOD]),
    ('select', (10, 20), lambda: [DIM_ARG, 3], [METHOD]),
    ('unfold', (10, 20), lambda: [DIM_ARG, 5, 2], [METHOD]),
]

for decl in neg_dim_tests:
    if len(decl) == 4:
        name, tensor_arg, arg_constr, types = decl
        extra_dim = 0
    elif len(decl) == 5:
        name, tensor_arg, arg_constr, types, extra_dim = decl

    test_name = 'test_' + name + '_neg_dim'

    assert not hasattr(TestTorch, test_name), "Duplicated test name: " + test_name
    setattr(TestTorch, test_name, make_neg_dim_test(name, tensor_arg, arg_constr, types, extra_dim))

if __name__ == '__main__':
    run_tests()
