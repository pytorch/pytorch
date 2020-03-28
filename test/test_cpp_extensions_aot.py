import os
import unittest

import torch.testing._internal.common_utils as common
from torch.testing._internal.common_utils import IS_WINDOWS
from torch.testing._internal.common_cuda import TEST_CUDA
import torch
import torch.backends.cudnn
import torch.utils.cpp_extension

try:
    import torch_test_cpp_extension.cpp as cpp_extension
    import torch_test_cpp_extension.msnpu as msnpu_extension
    import torch_test_cpp_extension.rng as rng_extension
except ImportError:
    raise RuntimeError(
        "test_cpp_extensions_aot.py cannot be invoked directly. Run "
        "`python run_test.py -i test_cpp_extensions_aot_ninja` instead."
    )


class TestCppExtensionAOT(common.TestCase):
    """Tests ahead-of-time cpp extensions

    NOTE: run_test.py's test_cpp_extensions_aot_ninja target
    also runs this test case, but with ninja enabled. If you are debugging
    a test failure here from the CI, check the logs for which target
    (test_cpp_extensions_aot_no_ninja vs test_cpp_extensions_aot_ninja)
    failed.
    """

    def test_extension_function(self):
        x = torch.randn(4, 4)
        y = torch.randn(4, 4)
        z = cpp_extension.sigmoid_add(x, y)
        self.assertEqual(z, x.sigmoid() + y.sigmoid())

    def test_extension_module(self):
        mm = cpp_extension.MatrixMultiplier(4, 8)
        weights = torch.rand(8, 4, dtype=torch.double)
        expected = mm.get().mm(weights)
        result = mm.forward(weights)
        self.assertEqual(expected, result)

    def test_backward(self):
        mm = cpp_extension.MatrixMultiplier(4, 8)
        weights = torch.rand(8, 4, dtype=torch.double, requires_grad=True)
        result = mm.forward(weights)
        result.sum().backward()
        tensor = mm.get()

        expected_weights_grad = tensor.t().mm(torch.ones([4, 4], dtype=torch.double))
        self.assertEqual(weights.grad, expected_weights_grad)

        expected_tensor_grad = torch.ones([4, 4], dtype=torch.double).mm(weights.t())
        self.assertEqual(tensor.grad, expected_tensor_grad)

    @unittest.skipIf(not TEST_CUDA, "CUDA not found")
    def test_cuda_extension(self):
        import torch_test_cpp_extension.cuda as cuda_extension

        x = torch.zeros(100, device="cuda", dtype=torch.float32)
        y = torch.zeros(100, device="cuda", dtype=torch.float32)

        z = cuda_extension.sigmoid_add(x, y).cpu()

        # 2 * sigmoid(0) = 2 * 0.5 = 1
        self.assertEqual(z, torch.ones_like(z))

    @unittest.skipIf(IS_WINDOWS, "Not available on Windows")
    def test_no_python_abi_suffix_sets_the_correct_library_name(self):
        # For this test, run_test.py will call `python setup.py install` in the
        # cpp_extensions/no_python_abi_suffix_test folder, where the
        # `BuildExtension` class has a `no_python_abi_suffix` option set to
        # `True`. This *should* mean that on Python 3, the produced shared
        # library does not have an ABI suffix like
        # "cpython-37m-x86_64-linux-gnu" before the library suffix, e.g. "so".
        # On Python 2 there is no ABI suffix anyway.
        root = os.path.join("cpp_extensions", "no_python_abi_suffix_test", "build")
        matches = [f for _, _, fs in os.walk(root) for f in fs if f.endswith("so")]
        self.assertEqual(len(matches), 1, str(matches))
        self.assertEqual(matches[0], "no_python_abi_suffix_test.so", str(matches))

    def test_optional(self):
        has_value = cpp_extension.function_taking_optional(torch.ones(5))
        self.assertTrue(has_value)
        has_value = cpp_extension.function_taking_optional(None)
        self.assertFalse(has_value)


class TestMSNPUTensor(common.TestCase):
    @classmethod
    def setUpClass(cls):
        msnpu_extension.init_msnpu_extension()

    def test_unregistered(self):
        a = torch.arange(0, 10, device='cpu')
        with self.assertRaisesRegex(RuntimeError, "Could not run"):
            b = torch.arange(0, 10, device='msnpu')

    def test_zeros(self):
        a = torch.empty(5, 5, device='cpu')
        self.assertEqual(a.device, torch.device('cpu'))

        b = torch.empty(5, 5, device='msnpu')
        self.assertEqual(b.device, torch.device('msnpu', 0))
        self.assertEqual(msnpu_extension.get_test_int(), 0)
        self.assertEqual(torch.get_default_dtype(), b.dtype)

        c = torch.empty((5, 5), dtype=torch.int64, device='msnpu')
        self.assertEqual(msnpu_extension.get_test_int(), 0)
        self.assertEqual(torch.int64, c.dtype)

    def test_add(self):
        a = torch.empty(5, 5, device='msnpu', requires_grad=True)
        self.assertEqual(msnpu_extension.get_test_int(), 0)

        b = torch.empty(5, 5, device='msnpu')
        self.assertEqual(msnpu_extension.get_test_int(), 0)

        c = a + b
        self.assertEqual(msnpu_extension.get_test_int(), 1)

    def test_conv_backend_override(self):
        # To simplify tests, we use 4d input here to avoid doing view4d( which
        # needs more overrides) in _convolution.
        input = torch.empty(2, 4, 10, 2, device='msnpu', requires_grad=True)
        weight = torch.empty(6, 4, 2, 2, device='msnpu', requires_grad=True)
        bias = torch.empty(6, device='msnpu')

        # Make sure forward is overriden
        out = torch.nn.functional.conv1d(input, weight, bias, 2, 0, 1, 1)
        self.assertEqual(msnpu_extension.get_test_int(), 2)
        self.assertEqual(out.shape[0], input.shape[0])
        self.assertEqual(out.shape[1], weight.shape[0])

        # Make sure backward is overriden
        # Double backward is dispatched to _convolution_double_backward.
        # It is not tested here as it involves more computation/overrides.
        grad = torch.autograd.grad(out, input, out, create_graph=True)
        self.assertEqual(msnpu_extension.get_test_int(), 3)
        self.assertEqual(grad[0].shape, input.shape)


class TestRNGExtension(common.TestCase):

    def test_rng(self):
        fourty_two = torch.full((10,), 42, dtype=torch.int64)

        t = torch.empty(10, dtype=torch.int64).random_()
        self.assertNotEqual(t, fourty_two)

        gen = torch.Generator(device='cpu')
        t = torch.empty(10, dtype=torch.int64).random_(generator=gen)
        self.assertNotEqual(t, fourty_two)

        self.assertEqual(rng_extension.getInstanceCount(), 0)
        gen = rng_extension.createTestCPUGenerator(42)
        self.assertEqual(rng_extension.getInstanceCount(), 1)
        copy = gen
        self.assertEqual(rng_extension.getInstanceCount(), 1)
        self.assertEqual(gen, copy)
        copy2 = rng_extension.identity(copy)
        self.assertEqual(rng_extension.getInstanceCount(), 1)
        self.assertEqual(gen, copy2)
        t = torch.empty(10, dtype=torch.int64).random_(generator=gen)
        self.assertEqual(rng_extension.getInstanceCount(), 1)
        self.assertEqual(t, fourty_two)
        del gen
        self.assertEqual(rng_extension.getInstanceCount(), 1)
        del copy
        self.assertEqual(rng_extension.getInstanceCount(), 1)
        del copy2
        self.assertEqual(rng_extension.getInstanceCount(), 0)

if __name__ == "__main__":
    common.run_tests()
