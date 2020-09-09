import warnings
import unittest
from itertools import product
import random

import torch

from torch.testing._internal.common_utils import \
    (TestCase, run_tests, do_test_empty_full, TEST_NUMPY, suppress_warnings,
     IS_WINDOWS, torch_to_numpy_dtype_dict, slowTest)
from torch.testing._internal.common_device_type import \
    (instantiate_device_type_tests, deviceCountAtLeast, onlyOnCPUAndCUDA,
     onlyCPU, skipCUDAIfNotRocm, largeCUDATensorTest, precisionOverride, dtypes,
     onlyCUDA, skipCPUIf, dtypesIfCUDA)

if TEST_NUMPY:
    import numpy as np

# Test suite for tensor creation ops
#
# Includes creation functions like torch.eye, random creation functions like
#   torch.rand, and *like functions like torch.ones_like.
# DOES NOT INCLUDE view ops, which are tested in TestViewOps (currently in
#   test_torch.py) OR numpy interop (which is also still tested in test_torch.py)
#
# See https://pytorch.org/docs/master/torch.html#creation-ops

class TestTensorCreation(TestCase):
    exact_dtype = True

    # TODO: this test should be updated
    @onlyOnCPUAndCUDA
    def test_empty_full(self, device):
        torch_device = torch.device(device)
        device_type = torch_device.type

        if device_type == 'cpu':
            do_test_empty_full(self, torch.testing.get_all_math_dtypes('cpu'), torch.strided, torch_device)
        if device_type == 'cuda':
            do_test_empty_full(self, torch.testing.get_all_math_dtypes('cpu'), torch.strided, None)
            do_test_empty_full(self, torch.testing.get_all_math_dtypes('cpu'), torch.strided, torch_device)

    # TODO: this test should be updated
    @suppress_warnings
    @onlyOnCPUAndCUDA
    @deviceCountAtLeast(1)
    def test_tensor_device(self, devices):
        device_type = torch.device(devices[0]).type
        if device_type == 'cpu':
            self.assertEqual('cpu', torch.tensor(5).device.type)
            self.assertEqual('cpu',
                             torch.ones((2, 3), dtype=torch.float32, device='cpu').device.type)
            self.assertEqual('cpu',
                             torch.ones((2, 3), dtype=torch.float32, device='cpu:0').device.type)
            self.assertEqual('cpu',
                             torch.tensor(torch.ones((2, 3), dtype=torch.float32), device='cpu:0').device.type)

            if TEST_NUMPY:
                self.assertEqual('cpu', torch.tensor(np.random.randn(2, 3), device='cpu').device.type)
        if device_type == 'cuda':
            self.assertEqual('cuda:0', str(torch.tensor(5).cuda(0).device))
            self.assertEqual('cuda:0', str(torch.tensor(5).cuda('cuda:0').device))
            self.assertEqual('cuda:0',
                             str(torch.tensor(5, dtype=torch.int64, device=0).device))
            self.assertEqual('cuda:0',
                             str(torch.tensor(5, dtype=torch.int64, device='cuda:0').device))
            self.assertEqual('cuda:0',
                             str(torch.tensor(torch.ones((2, 3), dtype=torch.float32), device='cuda:0').device))

            if TEST_NUMPY:
                self.assertEqual('cuda:0', str(torch.tensor(np.random.randn(2, 3), device='cuda:0').device))

            for device in devices:
                with torch.cuda.device(device):
                    device_string = 'cuda:' + str(torch.cuda.current_device())
                    self.assertEqual(device_string,
                                     str(torch.tensor(5, dtype=torch.int64, device='cuda').device))

            with self.assertRaises(RuntimeError):
                torch.tensor(5).cuda('cpu')
            with self.assertRaises(RuntimeError):
                torch.tensor(5).cuda('cpu:0')

            if len(devices) > 1:
                self.assertEqual('cuda:1', str(torch.tensor(5).cuda(1).device))
                self.assertEqual('cuda:1', str(torch.tensor(5).cuda('cuda:1').device))
                self.assertEqual('cuda:1',
                                 str(torch.tensor(5, dtype=torch.int64, device=1).device))
                self.assertEqual('cuda:1',
                                 str(torch.tensor(5, dtype=torch.int64, device='cuda:1').device))
                self.assertEqual('cuda:1',
                                 str(torch.tensor(torch.ones((2, 3), dtype=torch.float32),
                                     device='cuda:1').device))

                if TEST_NUMPY:
                    self.assertEqual('cuda:1',
                                     str(torch.tensor(np.random.randn(2, 3), device='cuda:1').device))

    # TODO: this test should be updated
    @onlyOnCPUAndCUDA
    def test_as_strided_neg(self, device):
        error = r'as_strided: Negative strides are not supported at the ' \
                r'moment, got strides: \[-?[0-9]+(, -?[0-9]+)*\]'
        with self.assertRaisesRegex(RuntimeError, error):
            torch.as_strided(torch.ones(3, 3, device=device), (1, 1), (2, -1))
        with self.assertRaisesRegex(RuntimeError, error):
            torch.as_strided(torch.ones(14, device=device), (2,), (-11,))

    # TODO: this test should be updated
    def test_zeros(self, device):
        res1 = torch.zeros(100, 100, device=device)
        res2 = torch.tensor((), device=device)
        torch.zeros(100, 100, device=device, out=res2)

        self.assertEqual(res1, res2)

        boolTensor = torch.zeros(2, 2, device=device, dtype=torch.bool)
        expected = torch.tensor([[False, False], [False, False]],
                                device=device, dtype=torch.bool)
        self.assertEqual(boolTensor, expected)

        halfTensor = torch.zeros(1, 1, device=device, dtype=torch.half)
        expected = torch.tensor([[0.]], device=device, dtype=torch.float16)
        self.assertEqual(halfTensor, expected)

        bfloat16Tensor = torch.zeros(1, 1, device=device, dtype=torch.bfloat16)
        expected = torch.tensor([[0.]], device=device, dtype=torch.bfloat16)
        self.assertEqual(bfloat16Tensor, expected)

        complexTensor = torch.zeros(2, 2, device=device, dtype=torch.complex64)
        expected = torch.tensor([[0., 0.], [0., 0.]], device=device, dtype=torch.complex64)
        self.assertEqual(complexTensor, expected)

    # TODO: this test should be updated
    def test_zeros_out(self, device):
        shape = (3, 4)
        out = torch.zeros(shape, device=device)
        torch.zeros(shape, device=device, out=out)

        # change the dtype, layout, device
        with self.assertRaises(RuntimeError):
            torch.zeros(shape, device=device, dtype=torch.int64, out=out)
        with self.assertRaises(RuntimeError):
            torch.zeros(shape, device=device, layout=torch.sparse_coo, out=out)

        # leave them the same
        self.assertEqual(torch.zeros(shape, device=device),
                         torch.zeros(shape, device=device, dtype=out.dtype, out=out))
        self.assertEqual(torch.zeros(shape, device=device),
                         torch.zeros(shape, device=device, layout=torch.strided, out=out))
        self.assertEqual(torch.zeros(shape, device=device),
                         torch.zeros(shape, device=device, out=out))

    # TODO: this test should be updated
    def test_ones(self, device):
        res1 = torch.ones(100, 100, device=device)
        res2 = torch.tensor((), device=device)
        torch.ones(100, 100, device=device, out=res2)
        self.assertEqual(res1, res2)

        # test boolean tensor
        res1 = torch.ones(1, 2, device=device, dtype=torch.bool)
        expected = torch.tensor([[True, True]], device=device, dtype=torch.bool)
        self.assertEqual(res1, expected)

    # TODO: this test should be updated
    @onlyCPU
    def test_constructor_dtypes(self, device):
        default_type = torch.Tensor().type()
        self.assertIs(torch.Tensor().dtype, torch.get_default_dtype())

        self.assertIs(torch.uint8, torch.ByteTensor.dtype)
        self.assertIs(torch.float32, torch.FloatTensor.dtype)
        self.assertIs(torch.float64, torch.DoubleTensor.dtype)

        torch.set_default_tensor_type('torch.FloatTensor')
        self.assertIs(torch.float32, torch.get_default_dtype())
        self.assertIs(torch.FloatStorage, torch.Storage)

        torch.set_default_dtype(torch.float64)
        self.assertIs(torch.float64, torch.get_default_dtype())
        self.assertIs(torch.DoubleStorage, torch.Storage)

        torch.set_default_tensor_type(torch.FloatTensor)
        self.assertIs(torch.float32, torch.get_default_dtype())
        self.assertIs(torch.FloatStorage, torch.Storage)

        if torch.cuda.is_available():
            torch.set_default_tensor_type(torch.cuda.FloatTensor)
            self.assertIs(torch.float32, torch.get_default_dtype())
            self.assertIs(torch.float32, torch.cuda.FloatTensor.dtype)
            self.assertIs(torch.cuda.FloatStorage, torch.Storage)

            torch.set_default_dtype(torch.float64)
            self.assertIs(torch.float64, torch.get_default_dtype())
            self.assertIs(torch.cuda.DoubleStorage, torch.Storage)

        # don't support integral or sparse default types.
        self.assertRaises(TypeError, lambda: torch.set_default_tensor_type('torch.IntTensor'))
        self.assertRaises(TypeError, lambda: torch.set_default_dtype(torch.int64))

        # don't allow passing dtype to set_default_tensor_type
        self.assertRaises(TypeError, lambda: torch.set_default_tensor_type(torch.float32))

        torch.set_default_tensor_type(default_type)

    # TODO: this test should be updated
    @onlyCPU
    def test_constructor_device_legacy(self, device):
        self.assertRaises(RuntimeError, lambda: torch.FloatTensor(device='cuda'))
        self.assertRaises(RuntimeError, lambda: torch.FloatTensor(torch.Size([2, 3, 4]), device='cuda'))
        self.assertRaises(RuntimeError, lambda: torch.FloatTensor((2.0, 3.0), device='cuda'))

        self.assertRaises(RuntimeError, lambda: torch.Tensor(device='cuda'))
        self.assertRaises(RuntimeError, lambda: torch.Tensor(torch.Size([2, 3, 4]), device='cuda'))
        self.assertRaises(RuntimeError, lambda: torch.Tensor((2.0, 3.0), device='cuda'))

        x = torch.randn((3,), device='cpu')
        self.assertRaises(RuntimeError, lambda: x.new(device='cuda'))
        self.assertRaises(RuntimeError, lambda: x.new(torch.Size([2, 3, 4]), device='cuda'))
        self.assertRaises(RuntimeError, lambda: x.new((2.0, 3.0), device='cuda'))

        if torch.cuda.is_available():
            self.assertRaises(RuntimeError, lambda: torch.cuda.FloatTensor(device='cpu'))
            self.assertRaises(RuntimeError, lambda: torch.cuda.FloatTensor(torch.Size([2, 3, 4]), device='cpu'))
            self.assertRaises(RuntimeError, lambda: torch.cuda.FloatTensor((2.0, 3.0), device='cpu'))

            default_type = torch.Tensor().type()
            torch.set_default_tensor_type(torch.cuda.FloatTensor)
            self.assertRaises(RuntimeError, lambda: torch.Tensor(device='cpu'))
            self.assertRaises(RuntimeError, lambda: torch.Tensor(torch.Size([2, 3, 4]), device='cpu'))
            self.assertRaises(RuntimeError, lambda: torch.Tensor((2.0, 3.0), device='cpu'))
            torch.set_default_tensor_type(torch.cuda.FloatTensor)
            torch.set_default_tensor_type(default_type)

            x = torch.randn((3,), device='cuda')
            self.assertRaises(RuntimeError, lambda: x.new(device='cpu'))
            self.assertRaises(RuntimeError, lambda: x.new(torch.Size([2, 3, 4]), device='cpu'))
            self.assertRaises(RuntimeError, lambda: x.new((2.0, 3.0), device='cpu'))

    # TODO: this test should be updated
    @suppress_warnings
    @onlyCPU
    def test_tensor_factory(self, device):
        # TODO: This test probably doesn't make too much sense now that
        # torch.tensor has been established for a while; it makes more
        # sense to test the legacy behavior in terms of the new behavior
        expected = torch.Tensor([1, 1])
        # test data
        res1 = torch.tensor([1, 1])
        self.assertEqual(res1, expected, exact_dtype=False)

        res1 = torch.tensor([1, 1], dtype=torch.int)
        self.assertEqual(res1, expected, exact_dtype=False)
        self.assertIs(torch.int, res1.dtype)

        # test copy
        res2 = torch.tensor(expected)
        self.assertEqual(res2, expected)
        res2[1] = 2
        self.assertEqual(expected, torch.ones_like(expected))

        res2 = torch.tensor(expected, dtype=torch.int)
        self.assertEqual(res1, expected, exact_dtype=False)
        self.assertIs(torch.int, res1.dtype)

        # test copy with numpy
        if TEST_NUMPY:
            for dtype in [np.float64, np.int64, np.int8, np.uint8]:
                a = np.array([5.]).astype(dtype)
                res1 = torch.tensor(a)
                self.assertEqual(5., res1[0].item())
                a[0] = 7.
                self.assertEqual(5., res1[0].item())

        # test boolean tensor
        a = torch.tensor([True, True, False, True, True], dtype=torch.bool)
        b = torch.tensor([-1, -1.1, 0, 1, 1.1], dtype=torch.bool)
        self.assertEqual(a, b)
        c = torch.tensor([-0.1, -1.1, 0, 1, 0.1], dtype=torch.bool)
        self.assertEqual(a, c)
        d = torch.tensor((-.3, 0, .3, 1, 3 / 7), dtype=torch.bool)
        e = torch.tensor((True, False, True, True, True), dtype=torch.bool)
        self.assertEqual(e, d)
        f = torch.tensor((-1, 0, -1.1, 1, 1.1), dtype=torch.bool)
        self.assertEqual(e, f)

        int64_max = torch.iinfo(torch.int64).max
        int64_min = torch.iinfo(torch.int64).min
        float64_max = torch.finfo(torch.float64).max
        float64_min = torch.finfo(torch.float64).min
        g_1 = torch.tensor((float('nan'), 0, int64_min, int64_max, int64_min - 1), dtype=torch.bool)
        self.assertEqual(e, g_1)
        g_2 = torch.tensor((int64_max + 1, 0, (int64_max + 1) * 2, (int64_max + 1) * 2 + 1, float64_min), dtype=torch.bool)
        self.assertEqual(e, g_2)
        g_3 = torch.tensor((float64_max, 0, float64_max + 1, float64_min - 1, float64_max + 1e291), dtype=torch.bool)
        self.assertEqual(e, g_3)

        h = torch.tensor([True, False, False, True, False, True, True], dtype=torch.bool)
        i = torch.tensor([1e-323, 1e-324, 0j, 1e-323j, 1e-324j, 1 + 2j, -1j], dtype=torch.bool)
        self.assertEqual(h, i)
        j = torch.tensor((True, True, True, True), dtype=torch.bool)
        k = torch.tensor((1e323, -1e323, float('inf'), -float('inf')), dtype=torch.bool)
        self.assertEqual(j, k)

    # TODO: this test should be updated
    @suppress_warnings
    @onlyCPU
    def test_tensor_factory_copy_var(self, device):
        def check_copy(copy, is_leaf, requires_grad, data_ptr=None):
            if data_ptr is None:
                data_ptr = copy.data_ptr
            self.assertEqual(copy, source, exact_dtype=False)
            self.assertTrue(copy.is_leaf == is_leaf)
            self.assertTrue(copy.requires_grad == requires_grad)
            self.assertTrue(copy.data_ptr == data_ptr)

        source = torch.randn(5, 5, dtype=torch.double, requires_grad=True)
        # test torch.tensor()
        check_copy(torch.tensor(source), True, False)
        check_copy(torch.tensor(source, requires_grad=False), True, False)
        check_copy(torch.tensor(source, requires_grad=True), True, True)

        # test tensor.new_tensor()
        copy = torch.randn(1)
        check_copy(copy.new_tensor(source), True, False)
        check_copy(copy.new_tensor(source, requires_grad=False), True, False)
        check_copy(copy.new_tensor(source, requires_grad=True), True, True)

        # test torch.as_tensor()
        check_copy(torch.as_tensor(source), source.is_leaf, source.requires_grad, source.data_ptr)  # not copy
        check_copy(torch.as_tensor(source, dtype=torch.float), False, True)  # copy and keep the graph

    # TODO: this test should be updated
    @onlyCPU
    def test_tensor_factory_type_inference(self, device):
        def test_inference(default_dtype):
            saved_dtype = torch.get_default_dtype()
            torch.set_default_dtype(default_dtype)
            default_complex_dtype = torch.complex64 if default_dtype == torch.float32 else torch.complex128
            self.assertIs(default_dtype, torch.tensor(()).dtype)
            self.assertIs(default_dtype, torch.tensor(5.).dtype)
            self.assertIs(torch.int64, torch.tensor(5).dtype)
            self.assertIs(torch.bool, torch.tensor(True).dtype)
            self.assertIs(torch.int32, torch.tensor(5, dtype=torch.int32).dtype)
            self.assertIs(default_dtype, torch.tensor(((7, 5), (9, 5.))).dtype)
            self.assertIs(default_dtype, torch.tensor(((5., 5), (3, 5))).dtype)
            self.assertIs(torch.int64, torch.tensor(((5, 3), (3, 5))).dtype)
            self.assertIs(default_complex_dtype, torch.tensor(((5, 3 + 2j), (3, 5 + 4j))).dtype)

            if TEST_NUMPY:
                self.assertIs(torch.float64, torch.tensor(np.array(())).dtype)
                self.assertIs(torch.float64, torch.tensor(np.array(5.)).dtype)
                if np.array(5).dtype == np.int64:  # np long, which can be 4 bytes (e.g. on windows)
                    self.assertIs(torch.int64, torch.tensor(np.array(5)).dtype)
                else:
                    self.assertIs(torch.int32, torch.tensor(np.array(5)).dtype)
                self.assertIs(torch.uint8, torch.tensor(np.array(3, dtype=np.uint8)).dtype)
                self.assertIs(default_dtype, torch.tensor(((7, np.array(5)), (np.array(9), 5.))).dtype)
                self.assertIs(torch.float64, torch.tensor(((7, 5), (9, np.array(5.)))).dtype)
                self.assertIs(torch.int64, torch.tensor(((5, np.array(3)), (np.array(3), 5))).dtype)
            torch.set_default_dtype(saved_dtype)

        test_inference(torch.float64)
        test_inference(torch.float32)

    # TODO: this test should be updated
    @suppress_warnings
    @onlyCPU
    def test_new_tensor(self, device):
        expected = torch.autograd.Variable(torch.ByteTensor([1, 1]))
        # test data
        res1 = expected.new_tensor([1, 1])
        self.assertEqual(res1, expected)
        res1 = expected.new_tensor([1, 1], dtype=torch.int)
        self.assertEqual(res1, expected, exact_dtype=False)
        self.assertIs(torch.int, res1.dtype)

        # test copy
        res2 = expected.new_tensor(expected)
        self.assertEqual(res2, expected)
        res2[1] = 2
        self.assertEqual(expected, torch.ones_like(expected))
        res2 = expected.new_tensor(expected, dtype=torch.int)
        self.assertEqual(res2, expected, exact_dtype=False)
        self.assertIs(torch.int, res2.dtype)

        # test copy with numpy
        if TEST_NUMPY:
            a = np.array([5.])
            res1 = torch.tensor(a)
            res1 = res1.new_tensor(a)
            self.assertEqual(5., res1[0].item())
            a[0] = 7.
            self.assertEqual(5., res1[0].item())

        if torch.cuda.device_count() >= 2:
            expected = expected.cuda(1)
            res1 = expected.new_tensor([1, 1])
            self.assertEqual(res1.get_device(), expected.get_device())
            res1 = expected.new_tensor([1, 1], dtype=torch.int)
            self.assertIs(torch.int, res1.dtype)
            self.assertEqual(res1.get_device(), expected.get_device())

            res2 = expected.new_tensor(expected)
            self.assertEqual(res2.get_device(), expected.get_device())
            res2 = expected.new_tensor(expected, dtype=torch.int)
            self.assertIs(torch.int, res1.dtype)
            self.assertEqual(res2.get_device(), expected.get_device())
            res2 = expected.new_tensor(expected, dtype=torch.int, device=0)
            self.assertIs(torch.int, res1.dtype)
            self.assertEqual(res2.get_device(), 0)

            res1 = expected.new_tensor(1)
            self.assertEqual(res1.get_device(), expected.get_device())
            res1 = expected.new_tensor(1, dtype=torch.int)
            self.assertIs(torch.int, res1.dtype)
            self.assertEqual(res1.get_device(), expected.get_device())

    # TODO: this test should be updated
    @onlyCPU
    def test_as_tensor(self, device):
        # from python data
        x = [[0, 1], [2, 3]]
        self.assertEqual(torch.tensor(x), torch.as_tensor(x))
        self.assertEqual(torch.tensor(x, dtype=torch.float32), torch.as_tensor(x, dtype=torch.float32))

        # python data with heterogeneous types
        z = [0, 'torch']
        with self.assertRaisesRegex(TypeError, "invalid data type"):
            torch.tensor(z)
            torch.as_tensor(z)

        # python data with self-referential lists
        z = [0]
        z += [z]
        with self.assertRaisesRegex(TypeError, "self-referential lists are incompatible"):
            torch.tensor(z)
            torch.as_tensor(z)

        z = [[1, 2], z]
        with self.assertRaisesRegex(TypeError, "self-referential lists are incompatible"):
            torch.tensor(z)
            torch.as_tensor(z)

        # from tensor (doesn't copy unless type is different)
        y = torch.tensor(x)
        self.assertIs(y, torch.as_tensor(y))
        self.assertIsNot(y, torch.as_tensor(y, dtype=torch.float32))
        if torch.cuda.is_available():
            self.assertIsNot(y, torch.as_tensor(y, device='cuda'))
            y_cuda = y.to('cuda')
            self.assertIs(y_cuda, torch.as_tensor(y_cuda))
            self.assertIs(y_cuda, torch.as_tensor(y_cuda, device='cuda'))

        if TEST_NUMPY:
            # doesn't copy
            for dtype in [np.float64, np.int64, np.int8, np.uint8]:
                n = np.random.rand(5, 6).astype(dtype)
                n_astensor = torch.as_tensor(n)
                self.assertEqual(torch.tensor(n), n_astensor)
                n_astensor[0][0] = 25.7
                self.assertEqual(torch.tensor(n), n_astensor)

            # changing dtype causes copy
            n = np.random.rand(5, 6).astype(np.float32)
            n_astensor = torch.as_tensor(n, dtype=torch.float64)
            self.assertEqual(torch.tensor(n, dtype=torch.float64), n_astensor)
            n_astensor[0][1] = 250.8
            self.assertNotEqual(torch.tensor(n, dtype=torch.float64), n_astensor)

            # changing device causes copy
            if torch.cuda.is_available():
                n = np.random.randn(5, 6)
                n_astensor = torch.as_tensor(n, device='cuda')
                self.assertEqual(torch.tensor(n, device='cuda'), n_astensor)
                n_astensor[0][2] = 250.9
                self.assertNotEqual(torch.tensor(n, device='cuda'), n_astensor)

    # TODO: this test should be updated
    @suppress_warnings
    def test_range(self, device):
        res1 = torch.range(0, 1, device=device)
        res2 = torch.tensor((), device=device)
        torch.range(0, 1, device=device, out=res2)
        self.assertEqual(res1, res2, atol=0, rtol=0)

        # Check range for non-contiguous tensors.
        x = torch.zeros(2, 3, device=device)
        torch.range(0, 3, device=device, out=x.narrow(1, 1, 2))
        res2 = torch.tensor(((0, 0, 1), (0, 2, 3)), device=device, dtype=torch.float32)
        self.assertEqual(x, res2, atol=1e-16, rtol=0)

        # Check negative
        res1 = torch.tensor((1, 0), device=device, dtype=torch.float32)
        res2 = torch.tensor((), device=device)
        torch.range(1, 0, -1, device=device, out=res2)
        self.assertEqual(res1, res2, atol=0, rtol=0)

        # Equal bounds
        res1 = torch.ones(1, device=device)
        res2 = torch.tensor((), device=device)
        torch.range(1, 1, -1, device=device, out=res2)
        self.assertEqual(res1, res2, atol=0, rtol=0)
        torch.range(1, 1, 1, device=device, out=res2)
        self.assertEqual(res1, res2, atol=0, rtol=0)

    # TODO: this test should be updated
    def test_range_warning(self, device):
        with warnings.catch_warnings(record=True) as w:
            torch.range(0, 10, device=device)
            self.assertEqual(len(w), 1)

    # TODO: this test should be updated
    @onlyCPU
    def test_arange(self, device):
        res = torch.tensor(range(10000))
        res1 = torch.arange(0, 10000)  # Use a larger number so vectorized code can be triggered
        res2 = torch.tensor([], dtype=torch.int64)
        torch.arange(0, 10000, out=res2)
        self.assertEqual(res, res1, atol=0, rtol=0)
        self.assertEqual(res, res2, atol=0, rtol=0)

        # Vectorization on non-contiguous tensors
        res = torch.rand(3, 3, 300000).to(torch.int64)
        res = res.permute(2, 0, 1)
        torch.arange(0, 300000 * 3 * 3, out=res)
        self.assertEqual(res.flatten(), torch.arange(0, 300000 * 3 * 3))

        # Check arange with only one argument
        res1 = torch.arange(10)
        res2 = torch.arange(0, 10)
        self.assertEqual(res1, res2, atol=0, rtol=0)

        # Check arange for non-contiguous tensors.
        x = torch.zeros(2, 3)
        torch.arange(0, 4, out=x.narrow(1, 1, 2))
        res2 = torch.Tensor(((0, 0, 1), (0, 2, 3)))
        self.assertEqual(x, res2, atol=1e-16, rtol=0)

        # Check negative
        res1 = torch.Tensor((1, 0))
        res2 = torch.Tensor()
        torch.arange(1, -1, -1, out=res2)
        self.assertEqual(res1, res2, atol=0, rtol=0)

        # Equal bounds
        res1 = torch.ones(1)
        res2 = torch.Tensor()
        torch.arange(1, 0, -1, out=res2)
        self.assertEqual(res1, res2, atol=0, rtol=0)
        torch.arange(1, 2, 1, out=res2)
        self.assertEqual(res1, res2, atol=0, rtol=0)

        # FloatTensor
        res1 = torch.arange(0.6, 0.89, 0.1, out=torch.FloatTensor())
        self.assertEqual(res1, [0.6, 0.7, 0.8])
        res1 = torch.arange(1, 10, 0.3, out=torch.FloatTensor())
        self.assertEqual(res1.size(0), 30)
        self.assertEqual(res1[0], 1)
        self.assertEqual(res1[29], 9.7)

        # DoubleTensor
        res1 = torch.arange(0.6, 0.89, 0.1, out=torch.DoubleTensor())
        self.assertEqual(res1, [0.6, 0.7, 0.8])
        res1 = torch.arange(1, 10, 0.3, out=torch.DoubleTensor())
        self.assertEqual(res1.size(0), 30)
        self.assertEqual(res1[0], 1)
        self.assertEqual(res1[29], 9.7)

        # Bool Input matching numpy semantics
        r = torch.arange(True)
        self.assertEqual(r[0], 0)
        r2 = torch.arange(False)
        self.assertEqual(len(r2), 0)
        self.assertEqual(r.dtype, torch.int64)
        self.assertEqual(r2.dtype, torch.int64)

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
        # NB: without the dtype, we'll infer output type to be int64
        r2 = torch.arange(0, 5, dtype=torch.float32)
        r3 = torch.arange(0, 5 - 1e-6)
        self.assertEqual(r1[:-1], r2, atol=0, rtol=0)
        self.assertEqual(r2, r3, atol=0, rtol=0)

        r1 = torch.arange(10, -1 + 1e-6, -1)
        # NB: without the dtype, we'll infer output type to be int64
        r2 = torch.arange(10, -1, -1, dtype=torch.float32)
        r3 = torch.arange(10, -1 - 1e-6, -1)
        self.assertEqual(r1, r2, atol=0, rtol=0)
        self.assertEqual(r2, r3[:-1], atol=0, rtol=0)

        # Test Rounding Errors
        line = torch.zeros(size=(1, 49))
        self.assertWarnsRegex(UserWarning, 'The out tensor will be resized',
                              lambda: torch.arange(-1, 1, 2. / 49, dtype=torch.float32, out=line))
        self.assertEqual(line.shape, [50])

        x = torch.empty(1).expand(10)
        self.assertRaises(RuntimeError, lambda: torch.arange(10, out=x))
        msg = "unsupported range"
        self.assertRaisesRegex(RuntimeError, msg, lambda: torch.arange(0, float('inf')))
        self.assertRaisesRegex(RuntimeError, msg, lambda: torch.arange(float('inf')))

        for device in torch.testing.get_all_device_types():
            self.assertRaisesRegex(RuntimeError, msg, lambda: torch.arange(-5, float('nan'), device=device))
            # check with step size
            self.assertRaisesRegex(RuntimeError, msg, lambda: torch.arange(0, float('-inf'), -1, device=device))
            self.assertRaisesRegex(RuntimeError, msg, lambda: torch.arange(0, float('inf'), device=device))
            self.assertRaisesRegex(RuntimeError, msg, lambda: torch.arange(float('-inf'), 10, device=device))
            self.assertRaisesRegex(RuntimeError, msg, lambda: torch.arange(float('nan'), 10, device=device))
            self.assertRaisesRegex(RuntimeError, msg, lambda: torch.arange(float('inf'), device=device))
            self.assertRaisesRegex(RuntimeError, msg, lambda: torch.arange(float('nan'), device=device))

            self.assertRaisesRegex(
                RuntimeError, "overflow",
                lambda: torch.arange(1.175494351e-38, 3.402823466e+38, device=device))

            # check that it holds a consistent output shape on precision-cornered step sizes
            d = torch.arange(-4.0, 4.0, 0.01, dtype=torch.float32, device=device)
            self.assertEqual(d.shape[0], 800)

    # TODO: this test should be updated
    @onlyCPU
    def test_arange_inference(self, device):
        saved_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.float32)
        # end only
        self.assertIs(torch.float32, torch.arange(1.).dtype)
        self.assertIs(torch.float32, torch.arange(torch.tensor(1.)).dtype)
        self.assertIs(torch.float32, torch.arange(torch.tensor(1., dtype=torch.float64)).dtype)

        self.assertIs(torch.int64, torch.arange(1).dtype)
        self.assertIs(torch.int64, torch.arange(torch.tensor(1)).dtype)
        self.assertIs(torch.int64, torch.arange(torch.tensor(1, dtype=torch.int16)).dtype)

        # start, end, [step]
        self.assertIs(torch.float32, torch.arange(1., 3).dtype)
        self.assertIs(torch.float32, torch.arange(torch.tensor(1., dtype=torch.float64), 3).dtype)
        self.assertIs(torch.float32, torch.arange(1, 3.).dtype)
        self.assertIs(torch.float32, torch.arange(torch.tensor(1, dtype=torch.int16), torch.tensor(3.)).dtype)
        self.assertIs(torch.float32, torch.arange(1, 3, 1.).dtype)
        self.assertIs(torch.float32,
                      torch.arange(torch.tensor(1),
                                   torch.tensor(3, dtype=torch.int16),
                                   torch.tensor(1., dtype=torch.float64)).dtype)

        self.assertIs(torch.int64, torch.arange(1, 3).dtype)
        self.assertIs(torch.int64, torch.arange(torch.tensor(1), 3).dtype)
        self.assertIs(torch.int64, torch.arange(torch.tensor(1), torch.tensor(3, dtype=torch.int16)).dtype)
        self.assertIs(torch.int64, torch.arange(1, 3, 1).dtype)
        self.assertIs(torch.int64,
                      torch.arange(torch.tensor(1),
                                   torch.tensor(3),
                                   torch.tensor(1, dtype=torch.int16)).dtype)
        torch.set_default_dtype(saved_dtype)

    def test_empty_strided(self, device):
        for shape in [(2, 3, 4), (0, 2, 0)]:
            # some of these cases are pretty strange, just verifying that if as_strided
            # allows them then empty_strided can as well.
            for strides in [(12, 4, 1), (2, 4, 6), (0, 0, 0)]:
                empty_strided = torch.empty_strided(shape, strides, device=device)
                # as_strided checks the storage size is big enough to support such a strided tensor;
                # instead of repeating this calculation, we just use empty_strided which does the same
                # calculation when setting the storage size.
                as_strided = torch.empty(empty_strided.storage().size(),
                                         device=device).as_strided(shape, strides)
                self.assertEqual(empty_strided.shape, as_strided.shape)
                self.assertEqual(empty_strided.stride(), as_strided.stride())

    def test_strided_mismatched_stride_shape(self, device):
        for shape, strides in [((1, ), ()), ((1, 2), (1, ))]:
            with self.assertRaisesRegex(RuntimeError, "mismatch in length of strides and shape"):
                torch.tensor(0.42, device=device).as_strided(shape, strides)

            with self.assertRaisesRegex(RuntimeError, "mismatch in length of strides and shape"):
                torch.tensor(0.42, device=device).as_strided_(shape, strides)

    def test_empty_tensor_props(self, device):
        sizes = [(0,), (0, 3), (5, 0), (5, 0, 3, 0, 2), (0, 3, 0, 2), (0, 5, 0, 2, 0)]
        for size in sizes:
            x = torch.empty(tuple(size), device=device)
            self.assertEqual(size, x.shape)
            self.assertTrue(x.is_contiguous())
            size_ones_instead_of_zeros = (x if x != 0 else 1 for x in size)
            y = torch.empty(tuple(size_ones_instead_of_zeros), device=device)
            self.assertEqual(x.stride(), y.stride())

    def test_eye(self, device):
        for dtype in torch.testing.get_all_dtypes():
            if dtype == torch.bfloat16:
                continue
            for n, m in product([3, 5, 7], repeat=2):
                # Construct identity using diagonal and fill
                res1 = torch.eye(n, m, device=device, dtype=dtype)
                naive_eye = torch.zeros(n, m, dtype=dtype, device=device)
                naive_eye.diagonal(dim1=-2, dim2=-1).fill_(1)
                self.assertEqual(naive_eye, res1)

                # Check eye_out outputs
                res2 = torch.empty(0, device=device, dtype=dtype)
                torch.eye(n, m, out=res2)
                self.assertEqual(res1, res2)

    @unittest.skipIf(not TEST_NUMPY, "NumPy not found")
    @precisionOverride({torch.float: 1e-8, torch.double: 1e-10})
    @dtypes(*(torch.testing.get_all_fp_dtypes(include_half=False, include_bfloat16=False) +
              torch.testing.get_all_complex_dtypes()))
    def test_linspace_vs_numpy(self, device, dtype):
        start = -0.0316082797944545745849609375 + (0.8888888888j if dtype.is_complex else 0)
        end = .0315315723419189453125 + (0.444444444444j if dtype.is_complex else 0)

        for steps in [1, 2, 3, 5, 11, 256, 257, 2**22]:
            t = torch.linspace(start, end, steps, device=device, dtype=dtype)
            a = np.linspace(start, end, steps, dtype=torch_to_numpy_dtype_dict[dtype])
            t = t.cpu()
            self.assertEqual(t, torch.from_numpy(a))
            self.assertTrue(t[0].item() == a[0])
            self.assertTrue(t[steps - 1].item() == a[steps - 1])

    @unittest.skipIf(not TEST_NUMPY, "NumPy not found")
    @precisionOverride({torch.float: 1e-6, torch.double: 1e-10})
    @dtypes(torch.float, torch.double)
    def test_logspace_vs_numpy(self, device, dtype):
        start = -0.0316082797944545745849609375
        end = .0315315723419189453125

        for steps in [1, 2, 3, 5, 11, 256, 257, 2**22]:
            t = torch.logspace(start, end, steps, device=device, dtype=dtype)
            a = np.logspace(start, end, steps, dtype=torch_to_numpy_dtype_dict[dtype])
            t = t.cpu()
            self.assertEqual(t, torch.from_numpy(a))
            self.assertEqual(t[0], a[0])
            self.assertEqual(t[steps - 1], a[steps - 1])

    @largeCUDATensorTest('16GB')
    def test_range_factories_64bit_indexing(self, device):
        bigint = 2 ** 31 + 1
        t = torch.arange(bigint, dtype=torch.long, device=device)
        self.assertEqual(t[-1].item(), bigint - 1)
        del t
        t = torch.linspace(0, 1, bigint, dtype=torch.float, device=device)
        self.assertEqual(t[-1].item(), 1)
        del t
        t = torch.logspace(0, 1, bigint, 2, dtype=torch.float, device=device)
        self.assertEqual(t[-1].item(), 2)
        del t

    @onlyOnCPUAndCUDA
    def test_tensor_ctor_device_inference(self, device):
        torch_device = torch.device(device)
        values = torch.tensor((1, 2, 3), device=device)

        # Tests tensor and as_tensor
        # Note: warnings are suppressed (suppresses warnings)
        for op in (torch.tensor, torch.as_tensor):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.assertEqual(op(values).device, torch_device)
                self.assertEqual(op(values, dtype=torch.float64).device, torch_device)

                if self.device_type == 'cuda':
                    with torch.cuda.device(device):
                        self.assertEqual(op(values.cpu()).device, torch.device('cpu'))

        # Tests sparse ctor
        indices = torch.tensor([[0, 1, 1],
                                [2, 0, 1],
                                [2, 1, 0]], device=device)
        sparse_size = (3, 3, 3)

        sparse_default = torch.sparse_coo_tensor(indices, values, sparse_size)
        self.assertEqual(sparse_default.device, torch_device)

        sparse_with_dtype = torch.sparse_coo_tensor(indices, values, sparse_size, dtype=torch.float64)
        self.assertEqual(sparse_with_dtype.device, torch_device)

        if self.device_type == 'cuda':
            with torch.cuda.device(device):
                sparse_with_dtype = torch.sparse_coo_tensor(indices.cpu(), values.cpu(),
                                                            sparse_size, dtype=torch.float64)
                self.assertEqual(sparse_with_dtype.device, torch.device('cpu'))

    def test_tensor_factories_empty(self, device):
        # ensure we can create empty tensors from each factory function
        shapes = [(5, 0, 1), (0,), (0, 0, 1, 0, 2, 0, 0)]

        for shape in shapes:
            for dt in torch.testing.get_all_dtypes():

                self.assertEqual(shape, torch.zeros(shape, device=device, dtype=dt).shape)
                self.assertEqual(shape, torch.zeros_like(torch.zeros(shape, device=device, dtype=dt)).shape)
                self.assertEqual(shape, torch.full(shape, 3, device=device, dtype=dt).shape)
                self.assertEqual(shape, torch.full_like(torch.zeros(shape, device=device, dtype=dt), 3).shape)
                self.assertEqual(shape, torch.ones(shape, device=device, dtype=dt).shape)
                self.assertEqual(shape, torch.ones_like(torch.zeros(shape, device=device, dtype=dt)).shape)
                self.assertEqual(shape, torch.empty(shape, device=device, dtype=dt).shape)
                self.assertEqual(shape, torch.empty_like(torch.zeros(shape, device=device, dtype=dt)).shape)
                self.assertEqual(shape, torch.empty_strided(shape, (0,) * len(shape), device=device, dtype=dt).shape)

                if dt == torch.bfloat16 and device.startswith('cuda') and IS_WINDOWS:
                    # TODO: https://github.com/pytorch/pytorch/issues/33793
                    self.assertRaises(RuntimeError, lambda: torch.randint(6, shape, device=device, dtype=dt).shape)
                elif dt == torch.bool:
                    self.assertEqual(shape, torch.randint(2, shape, device=device, dtype=dt).shape)
                    self.assertEqual(shape, torch.randint_like(torch.zeros(shape, device=device, dtype=dt), 2).shape)
                elif dt.is_complex:
                    self.assertRaises(RuntimeError, lambda: torch.randint(6, shape, device=device, dtype=dt).shape)
                else:
                    self.assertEqual(shape, torch.randint(6, shape, device=device, dtype=dt).shape)
                    self.assertEqual(shape, torch.randint_like(torch.zeros(shape, device=device, dtype=dt), 6).shape)

                if dt not in {torch.double, torch.float, torch.half, torch.bfloat16, torch.complex64, torch.complex128}:
                    self.assertRaises(RuntimeError, lambda: torch.rand(shape, device=device, dtype=dt).shape)

                if dt == torch.double or dt == torch.float or dt.is_complex:
                    self.assertEqual(shape, torch.randn(shape, device=device, dtype=dt).shape)
                    self.assertEqual(shape, torch.randn_like(torch.zeros(shape, device=device, dtype=dt)).shape)

        self.assertEqual((0,), torch.arange(0, device=device).shape)
        self.assertEqual((0, 0), torch.eye(0, device=device).shape)
        self.assertEqual((0, 0), torch.eye(0, 0, device=device).shape)
        self.assertEqual((5, 0), torch.eye(5, 0, device=device).shape)
        self.assertEqual((0, 5), torch.eye(0, 5, device=device).shape)
        self.assertEqual((0,), torch.linspace(1, 1, 0, device=device).shape)
        self.assertEqual((0,), torch.logspace(1, 1, 0, device=device).shape)
        self.assertEqual((0,), torch.randperm(0, device=device).shape)
        self.assertEqual((0,), torch.bartlett_window(0, device=device).shape)
        self.assertEqual((0,), torch.bartlett_window(0, periodic=False, device=device).shape)
        self.assertEqual((0,), torch.hamming_window(0, device=device).shape)
        self.assertEqual((0,), torch.hann_window(0, device=device).shape)
        self.assertEqual((1, 1, 0), torch.tensor([[[]]], device=device).shape)
        self.assertEqual((1, 1, 0), torch.as_tensor([[[]]], device=device).shape)

    @onlyCUDA
    def test_tensor_factory_gpu_type_inference(self, device):
        saved_type = torch.Tensor().type()
        torch.set_default_tensor_type(torch.cuda.DoubleTensor)
        torch.set_default_dtype(torch.float32)
        self.assertIs(torch.float32, torch.tensor(0.).dtype)
        self.assertEqual(torch.device(device), torch.tensor(0.).device)
        torch.set_default_dtype(torch.float64)
        self.assertIs(torch.float64, torch.tensor(0.).dtype)
        self.assertEqual(torch.device(device), torch.tensor(0.).device)
        torch.set_default_tensor_type(saved_type)

    @onlyCUDA
    def test_tensor_factory_gpu_type(self, device):
        saved_type = torch.Tensor().type()
        torch.set_default_tensor_type(torch.cuda.FloatTensor)
        x = torch.zeros((5, 5))
        self.assertIs(torch.float32, x.dtype)
        self.assertTrue(x.is_cuda)
        torch.set_default_tensor_type(torch.cuda.DoubleTensor)
        x = torch.zeros((5, 5))
        self.assertIs(torch.float64, x.dtype)
        self.assertTrue(x.is_cuda)
        torch.set_default_tensor_type(saved_type)

    @skipCPUIf(True, 'compares device with cpu')
    @dtypes(torch.int, torch.long, torch.float, torch.double)
    def test_arange_device_vs_cpu(self, device, dtype):
        cpu_tensor = torch.arange(0, 10, dtype=dtype, device='cpu')
        device_tensor = torch.arange(0, 10, dtype=dtype, device=device)
        self.assertEqual(cpu_tensor, device_tensor)

    @onlyCUDA
    @skipCUDAIfNotRocm
    def test_arange_bfloat16(self, device):
        ref_tensor = torch.tensor([0, 1, 2, 3], dtype=torch.bfloat16, device=device)
        bfloat16_tensor = torch.arange(0, 4, dtype=torch.bfloat16, device=device)
        self.assertEqual(ref_tensor, bfloat16_tensor)

        # step=2
        ref_tensor = torch.tensor([0, 2, 4], dtype=torch.bfloat16, device=device)
        bfloat16_tensor = torch.arange(0, 6, step=2, dtype=torch.bfloat16, device=device)
        self.assertEqual(ref_tensor, bfloat16_tensor)

    @dtypes(*torch.testing.get_all_dtypes(include_bool=False, include_half=False))
    @dtypesIfCUDA(*torch.testing.get_all_dtypes(include_bool=False, include_half=True))
    def test_linspace(self, device, dtype):
        _from = random.random()
        to = _from + random.random()
        res1 = torch.linspace(_from, to, 137, device=device, dtype=dtype)
        res2 = torch.tensor((), device=device, dtype=dtype)
        torch.linspace(_from, to, 137, dtype=dtype, out=res2)
        self.assertEqual(res1, res2, atol=0, rtol=0)

        # small tensor
        self.assertEqual(torch.linspace(10, 20, 11, device=device, dtype=dtype),
                         torch.tensor(list(range(10, 21)), device=device, dtype=dtype))
        # large tensor
        if dtype not in (torch.int8, torch.uint8):
            self.assertEqual(torch.linspace(10, 2000, 1991, device=device, dtype=dtype),
                             torch.tensor(list(range(10, 2001)), device=device, dtype=dtype))

        # Vectorization on non-contiguous tensors
        if dtype not in (torch.int8, torch.uint8):  # int8 and uint8 are too small for this test
            res = torch.rand(3, 3, 1000, device=device).to(dtype)
            res = res.permute(2, 0, 1)
            torch.linspace(0, 1000 * 3 * 3, 1000 * 3 * 3, out=res)
            self.assertEqual(res.flatten(), torch.linspace(0, 1000 * 3 * 3, 1000 * 3 * 3, device=device, dtype=dtype))

        self.assertRaises(RuntimeError, lambda: torch.linspace(0, 1, -1, device=device, dtype=dtype))
        # steps = 1
        self.assertEqual(torch.linspace(0, 1, 1, device=device, dtype=dtype),
                         torch.zeros(1, device=device, dtype=dtype), atol=0, rtol=0)
        # steps = 0
        self.assertEqual(torch.linspace(0, 1, 0, device=device, dtype=dtype).numel(), 0, atol=0, rtol=0)

        # Check linspace for generating the correct output for each dtype.
        start = 0 if dtype == torch.uint8 else -100
        expected_lin = torch.tensor([start + .5 * i for i in range(401)], device=device, dtype=torch.double)
        actual_lin = torch.linspace(start, start + 200, 401, device=device, dtype=dtype)
        # If on GPU, allow for minor error depending on dtype.
        tol = 0.
        if device != 'cpu':
            if dtype == torch.half:
                tol = 1e-1
            elif dtype == torch.float:
                tol = 1e-5
            elif dtype == torch.double:
                tol = 1e-10

        self.assertEqual(expected_lin.to(dtype), actual_lin, atol=tol, rtol=0)

        # Check linspace for generating with start > end.
        self.assertEqual(torch.linspace(2, 0, 3, device=device, dtype=dtype),
                         torch.tensor((2, 1, 0), device=device, dtype=dtype),
                         atol=0, rtol=0)

        # Create non-complex tensor from complex numbers
        if not dtype.is_complex:
            self.assertRaises(RuntimeError, lambda: torch.linspace(1j, 2j, 3, device=device, dtype=dtype))

        # Check for race condition (correctness when applied on a large tensor).
        if dtype not in (torch.int8, torch.uint8, torch.int16, torch.half, torch.bfloat16):
            y = torch.linspace(0, 999999 + (999999j if dtype.is_complex else 0),
                               1000000, device=device, dtype=dtype)
            if dtype.is_complex:
                cond = torch.logical_and(y[:-1].real < y[1:].real, y[:-1].imag < y[1:].imag)
            else:
                cond = y[:-1] < y[1:]
            correct = all(cond)
            self.assertTrue(correct)

        # Check linspace for non-contiguous tensors.
        x = torch.zeros(2, 3, device=device, dtype=dtype)
        y = torch.linspace(0, 3, 4, out=x.narrow(1, 1, 2), dtype=dtype)
        self.assertEqual(x, torch.tensor(((0, 0, 1), (0, 2, 3)), device=device, dtype=dtype), atol=0, rtol=0)

    def test_linspace_deduction(self, device):
        # Test deduction from input parameters.
        self.assertEqual(torch.linspace(1, 2, device=device).dtype, torch.float32)
        self.assertEqual(torch.linspace(1., 2, device=device).dtype, torch.float32)
        self.assertEqual(torch.linspace(1., -2., device=device).dtype, torch.float32)
        # TODO: Need fix
        with self.assertRaises(RuntimeError):
            torch.linspace(1j, -2j, device=device)

    # The implementation of linspace+logspace goes through a different path
    # when the steps arg is equal to 0 or 1. For other values of `steps`
    # they call specialized linspace (or logspace) kernels.
    LINSPACE_LOGSPACE_SPECIAL_STEPS = [0, 1]

    # NOTE [Linspace+Logspace precision override]
    # Our Linspace and logspace torch.half CUDA kernels are not very precise.
    # Since linspace/logspace are deterministic, we can compute an expected
    # amount of error (by testing without a precision override), adding a tiny
    # amount (EPS) to that, and using that value as the override.
    LINSPACE_LOGSPACE_EXTRA_EPS = 1e-5

    # Compares linspace device vs. cpu
    def _test_linspace(self, device, dtype, steps):
        a = torch.linspace(0, 10, steps=steps, dtype=dtype, device=device)
        b = torch.linspace(0, 10, steps=steps)
        self.assertEqual(a, b, exact_dtype=False)

    # See NOTE [Linspace+Logspace precision override]
    @skipCPUIf(True, "compares with CPU")
    @precisionOverride({torch.half: 0.0039 + LINSPACE_LOGSPACE_EXTRA_EPS})
    @dtypesIfCUDA(*(torch.testing.get_all_fp_dtypes() + torch.testing.get_all_complex_dtypes()))
    def test_linspace_device_vs_cpu(self, device, dtype):
        self._test_linspace(device, dtype, steps=10)

    @skipCPUIf(True, "compares with CPU")
    @dtypesIfCUDA(*(torch.testing.get_all_fp_dtypes() + torch.testing.get_all_complex_dtypes()))
    def test_linspace_special_steps(self, device, dtype):
        for steps in self.LINSPACE_LOGSPACE_SPECIAL_STEPS:
            self._test_linspace(device, dtype, steps=steps)

    # Compares logspace device vs cpu
    def _test_logspace(self, device, dtype, steps):
        a = torch.logspace(1, 1.1, steps=steps, dtype=dtype, device=device)
        b = torch.logspace(1, 1.1, steps=steps)
        self.assertEqual(a, b, exact_dtype=False)

    # Compares logspace device vs cpu
    def _test_logspace_base2(self, device, dtype, steps):
        a = torch.logspace(1, 1.1, steps=steps, base=2, dtype=dtype, device=device)
        b = torch.logspace(1, 1.1, steps=steps, base=2)
        self.assertEqual(a, b, exact_dtype=False)

    # See NOTE [Linspace+Logspace precision override]
    @skipCPUIf(True, "compares with CPU")
    @precisionOverride({torch.half: 0.025 + LINSPACE_LOGSPACE_EXTRA_EPS})
    @dtypesIfCUDA(torch.half, torch.float, torch.double)
    @dtypes(torch.float, torch.double)
    def test_logspace_device_vs_cpu(self, device, dtype):
        self._test_logspace(device, dtype, steps=10)

    # See NOTE [Linspace+Logspace precision override]
    @skipCPUIf(True, "compares with CPU")
    @precisionOverride({torch.half: 0.0201 + LINSPACE_LOGSPACE_EXTRA_EPS})
    @dtypesIfCUDA(torch.half, torch.float, torch.double)
    @dtypes(torch.float, torch.double)
    def test_logspace_base2(self, device, dtype):
        self._test_logspace_base2(device, dtype, steps=10)

    @skipCPUIf(True, "compares with CPU")
    @dtypesIfCUDA(torch.half, torch.float, torch.double)
    @dtypes(torch.float, torch.double)
    def test_logspace_special_steps(self, device, dtype):
        for steps in self.LINSPACE_LOGSPACE_SPECIAL_STEPS:
            self._test_logspace(device, dtype, steps=steps)
            self._test_logspace_base2(device, dtype, steps=steps)

    @precisionOverride({torch.half: 1e-1, torch.float: 1e-5, torch.double: 1e-10})
    @dtypes(torch.uint8, torch.int8, torch.short, torch.int, torch.long, torch.float, torch.double)
    @dtypesIfCUDA(torch.uint8, torch.int8, torch.short, torch.int, torch.long, torch.half, torch.float, torch.double)
    def test_logspace(self, device, dtype):
        _from = random.random()
        to = _from + random.random()
        res1 = torch.logspace(_from, to, 137, device=device, dtype=dtype)
        res2 = torch.tensor((), device=device, dtype=dtype)
        torch.logspace(_from, to, 137, device=device, dtype=dtype, out=res2)
        self.assertEqual(res1, res2, atol=0, rtol=0)
        self.assertRaises(RuntimeError, lambda: torch.logspace(0, 1, -1, device=device, dtype=dtype))
        self.assertEqual(torch.logspace(0, 1, 1, device=device, dtype=dtype),
                         torch.ones(1, device=device, dtype=dtype), atol=0, rtol=0)

        # Check precision - start, stop and base are chosen to avoid overflow
        # steps is chosen so that step size is not subject to rounding error
        # a tolerance is needed for gpu tests due to differences in computation
        atol = None
        rtol = None
        if self.device_type == 'cpu':
            atol = 0
            rtol = 0
        self.assertEqual(torch.tensor([2. ** (i / 8.) for i in range(49)], device=device, dtype=dtype),
                         torch.logspace(0, 6, steps=49, base=2, device=device, dtype=dtype),
                         atol=atol, rtol=rtol)

        # Check non-default base=2
        self.assertEqual(torch.logspace(1, 1, 1, 2, device=device, dtype=dtype),
                         torch.ones(1, device=device, dtype=dtype) * 2)
        self.assertEqual(torch.logspace(0, 2, 3, 2, device=device, dtype=dtype),
                         torch.tensor((1, 2, 4), device=device, dtype=dtype))

        # Check logspace_ for generating with start > end.
        self.assertEqual(torch.logspace(1, 0, 2, device=device, dtype=dtype),
                         torch.tensor((10, 1), device=device, dtype=dtype), atol=0, rtol=0)

        # Check logspace_ for non-contiguous tensors.
        x = torch.zeros(2, 3, device=device, dtype=dtype)
        y = torch.logspace(0, 3, 4, base=2, device=device, dtype=dtype, out=x.narrow(1, 1, 2))
        self.assertEqual(x, torch.tensor(((0, 1, 2), (0, 4, 8)), device=device, dtype=dtype), atol=0, rtol=0)

    @onlyOnCPUAndCUDA
    @dtypes(torch.half, torch.float, torch.double)
    def test_full_inference(self, device, dtype):
        size = (2, 2)

        prev_default = torch.get_default_dtype()
        torch.set_default_dtype(dtype)

        # Tests bool fill value inference
        t = torch.full(size, True)
        self.assertEqual(t.dtype, torch.bool)

        # Tests integer fill value inference
        t = torch.full(size, 1)
        self.assertEqual(t.dtype, torch.long)

        # Tests float fill value inference
        t = torch.full(size, 1.)
        self.assertEqual(t.dtype, dtype)

        # Tests complex inference
        t = torch.full(size, (1 + 1j))
        ctype = torch.complex128 if dtype is torch.double else torch.complex64
        self.assertEqual(t.dtype, ctype)

        torch.set_default_dtype(prev_default)

    def test_full_out(self, device):
        size = (5,)
        o = torch.empty(size, device=device, dtype=torch.long)

        # verifies dtype/out conflict throws a RuntimeError
        with self.assertRaises(RuntimeError):
            torch.full(o.shape, 1., dtype=torch.float, out=o)

        # verifies out dtype overrides inference
        self.assertEqual(torch.full(o.shape, 1., out=o).dtype, o.dtype)
        self.assertEqual(torch.full(size, 1, out=o).dtype, o.dtype)



# Class for testing random tensor creation ops, like torch.randint
class TestRandomTensorCreation(TestCase):
    exact_dtype = True

    # TODO: this test should be updated
    @onlyCPU
    def test_randint_inference(self, device):
        size = (2, 1)
        for args in [(3,), (1, 3)]:  # (low,) and (low, high)
            self.assertIs(torch.int64, torch.randint(*args, size=size).dtype)
            self.assertIs(torch.int64, torch.randint(*args, size=size, layout=torch.strided).dtype)
            self.assertIs(torch.int64, torch.randint(*args, size=size, generator=torch.default_generator).dtype)
            self.assertIs(torch.float32, torch.randint(*args, size=size, dtype=torch.float32).dtype)
            out = torch.empty(size, dtype=torch.float32)
            self.assertIs(torch.float32, torch.randint(*args, size=size, out=out).dtype)
            self.assertIs(torch.float32, torch.randint(*args, size=size, out=out, dtype=torch.float32).dtype)
            out = torch.empty(size, dtype=torch.int64)
            self.assertIs(torch.int64, torch.randint(*args, size=size, out=out).dtype)
            self.assertIs(torch.int64, torch.randint(*args, size=size, out=out, dtype=torch.int64).dtype)

    # TODO: this test should be updated
    @onlyCPU
    def test_randint(self, device):
        SIZE = 100

        def seed(generator):
            if generator is None:
                torch.manual_seed(123456)
            else:
                generator.manual_seed(123456)
            return generator

        for generator in (None, torch.Generator()):
            generator = seed(generator)
            res1 = torch.randint(0, 6, (SIZE, SIZE), generator=generator)
            res2 = torch.empty((), dtype=torch.int64)
            generator = seed(generator)
            torch.randint(0, 6, (SIZE, SIZE), generator=generator, out=res2)
            generator = seed(generator)
            res3 = torch.randint(6, (SIZE, SIZE), generator=generator)
            res4 = torch.empty((), dtype=torch.int64)
            generator = seed(generator)
            torch.randint(6, (SIZE, SIZE), out=res4, generator=generator)
            self.assertEqual(res1, res2)
            self.assertEqual(res1, res3)
            self.assertEqual(res1, res4)
            self.assertEqual(res2, res3)
            self.assertEqual(res2, res4)
            self.assertEqual(res3, res4)
            self.assertTrue((res1 < 6).all().item())
            self.assertTrue((res1 >= 0).all().item())

    @dtypes(torch.half, torch.float, torch.double,
            torch.complex32, torch.complex64, torch.complex128)
    def test_randn(self, device, dtype):
        SIZE = 100
        for size in [0, SIZE]:
            torch.manual_seed(123456)
            res1 = torch.randn(size, size, dtype=dtype, device=device)
            res2 = torch.tensor([], dtype=dtype, device=device)
            torch.manual_seed(123456)
            torch.randn(size, size, out=res2)
            self.assertEqual(res1, res2)

    @dtypes(torch.float, torch.double, torch.complex64, torch.complex128)
    def test_rand(self, device, dtype):
        SIZE = 100
        for size in [0, SIZE]:
            torch.manual_seed(123456)
            res1 = torch.rand(size, size, dtype=dtype, device=device)
            res2 = torch.tensor([], dtype=dtype, device=device)
            torch.manual_seed(123456)
            torch.rand(size, size, out=res2)
            self.assertEqual(res1, res2)

    @slowTest
    def test_randperm(self, device):
        if device == 'cpu':
            rng_device = None
        else:
            rng_device = [device]

        # Test core functionality. On CUDA, for small n, randperm is offloaded to CPU instead. For large n, randperm is
        # executed on GPU.
        for n in (100, 50000, 100000):
            # Ensure both integer and floating-point numbers are tested. Half follows an execution path that is
            # different from others on CUDA.
            for dtype in (torch.long, torch.half, torch.float):
                if n > 2049 and dtype == torch.half:  # Large n for torch.half will raise an exception, do not test here.
                    continue
                with torch.random.fork_rng(devices=rng_device):
                    res1 = torch.randperm(n, dtype=dtype, device=device)
                res2 = torch.empty(0, dtype=dtype, device=device)
                torch.randperm(n, out=res2, dtype=dtype, device=device)
                self.assertEqual(res1, res2, atol=0, rtol=0)

        # Default type is long
        for n in (100, 10000):
            self.assertEqual(torch.randperm(n, device=device).dtype, torch.long)

        # randperm of 0 elements is an empty tensor
        res1 = torch.randperm(0)
        res2 = torch.tensor(5, dtype=dtype, device=device)
        torch.randperm(0, out=res2)
        self.assertEqual(res1.numel(), 0)
        self.assertEqual(res2.numel(), 0)

        # Test exceptions when n is too large for a floating point type
        for dtype, small_n, large_n in ((torch.half, 2**11 + 1, 2**11 + 2),
                                        (torch.float, 2**24 + 1, 2**24 + 2),
                                        (torch.double, 2**25,  # 2**53 + 1 is too large to run
                                         2**53 + 2)):
            res = torch.empty(0, dtype=dtype, device=device)
            torch.randperm(small_n, out=res)  # No exception expected
            self.assertRaises(RuntimeError, lambda: torch.randperm(large_n, out=res, device=device))

        # Test non-contiguous tensors
        for n in (4, 5, 6, 10, 20):
            non_contiguous_tensor = torch.zeros((2, 3), dtype=torch.long, device=device).t()
            self.assertFalse(non_contiguous_tensor.is_contiguous())
            with torch.random.fork_rng(devices=rng_device):
                res = torch.randperm(n, dtype=torch.long, device=device)
            torch.randperm(n, out=non_contiguous_tensor)
            self.assertEqual(non_contiguous_tensor, res)


# Class for testing *like ops, like torch.ones_like
class TestLikeTensorCreation(TestCase):
    exact_dtype = True

    # TODO: this test should be updated
    def test_ones_like(self, device):
        expected = torch.ones(100, 100, device=device)

        res1 = torch.ones_like(expected)
        self.assertEqual(res1, expected)

        # test boolean tensor
        expected = torch.tensor([True, True], device=device, dtype=torch.bool)
        res1 = torch.ones_like(expected)
        self.assertEqual(res1, expected)

    # TODO: this test should be updated
    @onlyCPU
    def test_empty_like(self, device):
        x = torch.autograd.Variable(torch.Tensor())
        y = torch.autograd.Variable(torch.randn(4, 4))
        z = torch.autograd.Variable(torch.IntTensor([1, 2, 3]))
        for a in (x, y, z):
            self.assertEqual(torch.empty_like(a).shape, a.shape)
            self.assertEqualTypeString(torch.empty_like(a), a)

    def test_zeros_like(self, device):
        expected = torch.zeros((100, 100,), device=device)

        res1 = torch.zeros_like(expected)
        self.assertEqual(res1, expected)

    @deviceCountAtLeast(2)
    def test_zeros_like_multiple_device(self, devices):
        expected = torch.zeros(100, 100, device=devices[0])
        x = torch.randn(100, 100, device=devices[1], dtype=torch.float32)
        output = torch.zeros_like(x)
        self.assertEqual(output, expected)

    @deviceCountAtLeast(2)
    def test_ones_like_multiple_device(self, devices):
        expected = torch.ones(100, 100, device=devices[0])
        x = torch.randn(100, 100, device=devices[1], dtype=torch.float32)
        output = torch.ones_like(x)
        self.assertEqual(output, expected)

    # Full-like precedence is the explicit dtype then the dtype of the "like"
    # tensor.
    @onlyOnCPUAndCUDA
    def test_full_like_inference(self, device):
        size = (2, 2)
        like = torch.empty((5,), device=device, dtype=torch.long)

        self.assertEqual(torch.full_like(like, 1.).dtype, torch.long)
        self.assertEqual(torch.full_like(like, 1., dtype=torch.complex64).dtype,
                         torch.complex64)


instantiate_device_type_tests(TestTensorCreation, globals())
instantiate_device_type_tests(TestRandomTensorCreation, globals())
instantiate_device_type_tests(TestLikeTensorCreation, globals())

if __name__ == '__main__':
    run_tests()
