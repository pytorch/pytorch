import os
import unittest

import torch.testing._internal.common_utils as common
from torch.testing._internal.common_utils import IS_WINDOWS, skipIfRocm
from torch.testing._internal.common_cuda import TEST_CUDA
import torch
import torch.backends.cudnn
import torch.utils.cpp_extension
from scipy import stats

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

    def setUp(self):
        super(TestRNGExtension, self).setUp()
        rng_extension.registerOps()

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

class TestCUDA_CSPRNG_Generator(common.TestCase):

    def setUp(self):
        import torch_test_cpp_extension.csprng as csprng_extension
        super(TestCUDA_CSPRNG_Generator, self).setUp()
        csprng_extension.registerOps()

    def distribution(self, t):
        results = {
            'uniform'               : stats.kstest(t.cpu(), 'uniform'),
            'normal(0.0, 1.0)'      : stats.kstest(t.cpu(), 'norm'),
            'exponential(1.0)'      : stats.kstest(t.cpu(), 'expon'),
            'cauchy(0.0, 1.0)'      : stats.kstest(t.cpu(), 'cauchy'),
            'geometric(0.5)'        : stats.kstest(t.cpu(), 'geom', args=(0.5,)),
            'lognormal(0.0, 0.25)'  : stats.kstest(t.cpu(), 'lognorm', args=(0.25, 0.0)),
        }
        return min(results.items(), key=lambda res: res[1].statistic)

    @skipIfRocm
    @unittest.skipIf(not TEST_CUDA, "CUDA not found")
    def test_random(self):
        import torch_test_cpp_extension.csprng as csprng_extension
        gen = csprng_extension.create_CUDA_CSPRNG_Generator()
        for dtype in [torch.bool, torch.uint8, torch.int8, torch.int16, 
                      torch.int32, torch.int64, torch.float, torch.double]:
            t = torch.empty(100, dtype=dtype, device='cuda').random_(generator=gen)
            # print(t)

    @skipIfRocm
    @unittest.skipIf(not TEST_CUDA, "CUDA not found")
    def test_random2(self):
        import torch_test_cpp_extension.csprng as csprng_extension
        gen = csprng_extension.create_CUDA_CSPRNG_Generator()
        s = torch.zeros(20, 20, dtype=torch.uint8, device='cuda')
        t = s[:, 7]
        self.assertFalse(t.is_contiguous())
        t.random_(generator=gen)
        t = s[7, :]
        self.assertTrue(t.is_contiguous())
        t.random_(generator=gen)
        # print(s)

    @skipIfRocm
    @unittest.skipIf(not TEST_CUDA, "CUDA not found")
    def test_bool(self):
        import torch_test_cpp_extension.csprng as csprng_extension
        gen = csprng_extension.create_CUDA_CSPRNG_Generator()
        size = 10000
        for i in range(100):
            t = torch.empty(size, dtype=torch.bool, device='cuda').random_(generator=gen)
            percentage = (t.eq(True)).to(torch.int).sum().item() / size
            self.assertTrue(0.48 < percentage < 0.52)

    @skipIfRocm
    @unittest.skipIf(not TEST_CUDA, "CUDA not found")
    def test_ints(self):
        import torch_test_cpp_extension.csprng as csprng_extension
        gen = csprng_extension.create_CUDA_CSPRNG_Generator()
        for (dtype, size, prec) in [(torch.uint8, 10000, 1), (torch.int8, 10000, 1), (torch.int16, 1000000, 100)]:
            t = torch.empty(size, dtype=dtype, device='cuda').random_(generator=gen)
            avg = t.sum().item() / size
            # print(avg)
            # print(torch.iinfo(dtype).max / 2)
            self.assertEqual(avg, torch.iinfo(dtype).max / 2, prec)
        for (dtype, size, prec) in [(torch.int32, 1000000, 1e7), (torch.int64, 1000000, 1e16)]:
            t = torch.empty(size, dtype=dtype, device='cuda').random_(generator=gen)
            avg = (t / size).sum().item()
            # print(avg)
            # print(torch.iinfo(dtype).max / 2)
            self.assertEqual(avg, torch.iinfo(dtype).max / 2, prec)

    @skipIfRocm
    @unittest.skipIf(not TEST_CUDA, "CUDA not found")
    def test_uniform1(self):
        import torch_test_cpp_extension.csprng as csprng_extension
        gen = csprng_extension.create_CUDA_CSPRNG_Generator()
        size = 1000
        alpha = 0.1
        for dtype in [torch.float, torch.double]:
            for from_ in [-100, 0, 1000]:
                for to_ in [-42, 0, 4242]:
                    if to_ > from_:
                        range_ = to_ - from_
                        t = torch.empty(size, dtype=dtype, device='cuda').uniform_(from_, to_, generator=gen)
                        self.assertTrue(from_  <= t.min() < from_ + alpha * range_)
                        self.assertTrue(to_ - alpha * range_  <= t.max() < to_)

    @skipIfRocm
    @unittest.skipIf(not TEST_CUDA, "CUDA not found")
    def test_normal1(self):
        import torch_test_cpp_extension.csprng as csprng_extension
        gen = csprng_extension.create_CUDA_CSPRNG_Generator()
        size = 1000
        for dtype in [torch.float, torch.double]:
            for mean in [-42.42, 0.0, 4242]:
                for std in [1.0, 2.0, 3.0]:
                    t = torch.empty(size, dtype=dtype, device='cuda').normal_(mean, std, generator=gen)
                    self.assertEqual(t.mean().item(), mean, 1)
                    self.assertEqual(t.std().item(), std, 1)

    @skipIfRocm
    @unittest.skipIf(not TEST_CUDA, "CUDA not found")
    def test_normal2(self):
        import torch_test_cpp_extension.csprng as csprng_extension

        def helper(self, device, dtype, ptype, t_transform, std_transform):
            q = torch.empty(100, 100, dtype=dtype, device=device)

            q.normal_()
            self.assertEqual(t_transform(q).mean(), 0, 0.2)
            self.assertEqual(t_transform(q).std(), std_transform(1), 0.2)

            q.normal_(2, 3)
            self.assertEqual(t_transform(q).mean(), 2, 0.3)
            self.assertEqual(t_transform(q).std(), std_transform(3), 0.3)

            q = torch.empty(100, 100, dtype=dtype, device=device)
            q_row1 = q[0:1].clone()
            q[99:100].normal_()
            self.assertEqual(t_transform(q[99:100]).mean(), 0, 0.2)
            self.assertEqual(t_transform(q[99:100]).std(), std_transform(1), 0.2)
            self.assertEqual(t_transform(q[0:1]).clone(), t_transform(q_row1))

            mean = torch.empty(100, 100, dtype=dtype, device=device)
            mean[:50].fill_(ptype(0))
            mean[50:].fill_(ptype(1))

            std = torch.empty(100, 100, dtype=torch.float, device=device)
            std[:, :50] = 4
            std[:, 50:] = 1

            r = torch.normal(mean)
            self.assertEqual(r.dtype, dtype)
            self.assertEqual(str(r.device), device)
            self.assertEqual(t_transform(r[:50]).mean(), 0, 0.2)
            self.assertEqual(t_transform(r[50:]).mean(), 1, 0.2)
            self.assertEqual(t_transform(r).std(), std_transform(1), 0.2)

            r.fill_(42)
            r = torch.normal(mean, 3)
            self.assertEqual(r.dtype, dtype)
            self.assertEqual(str(r.device), device)
            self.assertEqual(t_transform(r[:50]).mean(), 0, 0.2)
            self.assertEqual(t_transform(r[50:]).mean(), 1, 0.2)
            self.assertEqual(t_transform(r).std(), std_transform(3), 0.2)

            r.fill_(42)
            torch.normal(mean, 3, out=r)
            self.assertEqual(r.dtype, dtype)
            self.assertEqual(str(r.device), device)
            self.assertEqual(t_transform(r[:50]).mean(), 0, 0.2)
            self.assertEqual(t_transform(r[50:]).mean(), 1, 0.2)
            self.assertEqual(t_transform(r).std(), std_transform(3), 0.2)

            r.fill_(42)
            r = torch.normal(2, std)
            self.assertFalse(r.dtype.is_complex)
            self.assertEqual(str(r.device), device)
            self.assertEqual(r.mean(), 2, 0.2)
            self.assertEqual(r[:, :50].std(), 4, 0.3)
            self.assertEqual(r[:, 50:].std(), 1, 0.2)

            r.fill_(42)
            torch.normal(2, std, out=r)
            self.assertFalse(r.dtype.is_complex)
            self.assertEqual(str(r.device), device)
            self.assertEqual(r.mean(), 2, 0.2)
            self.assertEqual(r[:, :50].std(), 4, 0.3)
            self.assertEqual(r[:, 50:].std(), 1, 0.2)

            r.fill_(42)
            r = torch.normal(mean, std)
            self.assertEqual(r.dtype, dtype)
            self.assertEqual(str(r.device), device)
            self.assertEqual(t_transform(r[:50]).mean(), 0, 0.2)
            self.assertEqual(t_transform(r[50:]).mean(), 1, 0.2)
            self.assertEqual(t_transform(r[:, :50]).std(), std_transform(4), 0.3)
            self.assertEqual(t_transform(r[:, 50:]).std(), std_transform(1), 0.2)

            r.fill_(42)
            torch.normal(mean, std, out=r)
            self.assertEqual(r.dtype, dtype)
            self.assertEqual(str(r.device), device)
            self.assertEqual(t_transform(r[:50]).mean(), 0, 0.2)
            self.assertEqual(t_transform(r[50:]).mean(), 1, 0.2)
            self.assertEqual(t_transform(r[:, :50]).std(), std_transform(4), 0.3)
            self.assertEqual(t_transform(r[:, 50:]).std(), std_transform(1), 0.2)

            r.fill_(42)
            r = torch.normal(2, 3, (100, 100), dtype=dtype, device=device)
            self.assertEqual(r.dtype, dtype)
            self.assertEqual(str(r.device), device)
            self.assertEqual(t_transform(r).mean(), 2, 0.3)
            self.assertEqual(t_transform(r).std(), std_transform(3), 0.3)

            r.fill_(42)
            torch.normal(2, 3, (100, 100), dtype=dtype, device=device, out=r)
            self.assertEqual(r.dtype, dtype)
            self.assertEqual(str(r.device), device)
            self.assertEqual(t_transform(r).mean(), 2, 0.3)
            self.assertEqual(t_transform(r).std(), std_transform(3), 0.3)

        device = 'cuda:0'
        for dtype in [torch.float, torch.double] :
            if dtype.is_complex:
                helper(self, device, dtype, lambda x: complex(x, x),
                    lambda t: torch.real(t).to(torch.float), lambda mean: mean / math.sqrt(2))
                helper(self, device, dtype, lambda x: complex(x, x),
                    lambda t: torch.imag(t).to(torch.float), lambda mean: mean / math.sqrt(2))
                self.assertRaisesRegex(
                    RuntimeError, "normal expects standard deviation to be non-complex",
                    lambda: torch.normal(0, torch.empty(100, 100, dtype=dtype, device=device)))
                out = torch.empty(100, 100, dtype=dtype, device=device)
                self.assertRaisesRegex(
                    RuntimeError, "normal expects standard deviation to be non-complex",
                    lambda: torch.normal(0, torch.empty(100, 100, dtype=dtype, device=device), out=out))
            else:
                helper(self, device, dtype, lambda x: x, lambda t: t, lambda mean: mean)

    @skipIfRocm
    @unittest.skipIf(not TEST_CUDA, "CUDA not found")
    def test_uniform(self):
        import torch_test_cpp_extension.csprng as csprng_extension
        gen = csprng_extension.create_CUDA_CSPRNG_Generator()
        size = 1000
        for dtype in [torch.float, torch.double]:
            t = torch.empty(size, dtype=dtype, device='cuda').uniform_(generator=gen)
            actual_distribution = self.distribution(t)
            self.assertEqual(actual_distribution[0], "uniform")
            self.assertTrue(actual_distribution[1].statistic < 0.1)

    @skipIfRocm
    @unittest.skipIf(not TEST_CUDA, "CUDA not found")
    def test_normal(self):
        import torch_test_cpp_extension.csprng as csprng_extension
        gen = csprng_extension.create_CUDA_CSPRNG_Generator()
        size = 1000
        mean = 0.0
        std = 1.0
        for dtype in [torch.float, torch.double]:
            t = torch.empty(size, dtype=dtype, device='cuda').normal_(generator=gen)
            actual_distribution = self.distribution(t)
            self.assertEqual(actual_distribution[0], "normal(" + str(mean) + ", " + str(std) + ")")
            self.assertTrue(actual_distribution[1].statistic < 0.1)

    @skipIfRocm
    @unittest.skipIf(not TEST_CUDA, "CUDA not found")
    def test_cauchy(self):
        import torch_test_cpp_extension.csprng as csprng_extension
        gen = csprng_extension.create_CUDA_CSPRNG_Generator()
        size = 1000
        median = 0.0
        sigma = 1.0
        for dtype in [torch.float, torch.double]:
            t = torch.empty(size, dtype=dtype, device='cuda').cauchy_(median=median, sigma=sigma, generator=gen)
            actual_distribution = self.distribution(t)
            self.assertEqual(actual_distribution[0], "cauchy(" + str(median) + ", " + str(sigma) + ")")
            self.assertTrue(actual_distribution[1].statistic < 0.1)

    @skipIfRocm
    @unittest.skipIf(not TEST_CUDA, "CUDA not found")
    def test_log_normal(self):
        import torch_test_cpp_extension.csprng as csprng_extension
        gen = csprng_extension.create_CUDA_CSPRNG_Generator()
        size = 1000
        mean = 0.0
        std = 0.25
        for dtype in [torch.float, torch.double]:
            t = torch.empty(size, dtype=dtype, device='cuda').log_normal_(mean=mean, std=std, generator=gen)
            actual_distribution = self.distribution(t)
            self.assertEqual(actual_distribution[0], "lognormal(" + str(mean) + ", " + str(std) + ")")
            self.assertTrue(actual_distribution[1].statistic < 0.1)

    @skipIfRocm
    @unittest.skipIf(not TEST_CUDA, "CUDA not found")
    def test_geometric(self):
        import torch_test_cpp_extension.csprng as csprng_extension
        gen = csprng_extension.create_CUDA_CSPRNG_Generator()
        size = 1000
        p = 0.5
        for dtype in [torch.float, torch.double]:
            t = torch.empty(size, dtype=dtype, device='cuda').geometric_(p=p, generator=gen)
            actual_distribution = self.distribution(t)
            self.assertEqual(actual_distribution[0], "geometric(" + str(p) + ")")
            self.assertTrue(actual_distribution[1].statistic < 0.6)

    @skipIfRocm
    @unittest.skipIf(not TEST_CUDA, "CUDA not found")
    def test_exponential(self):
        import torch_test_cpp_extension.csprng as csprng_extension
        gen = csprng_extension.create_CUDA_CSPRNG_Generator()
        size = 1000
        lambd = 1.0
        for dtype in [torch.float, torch.double]:
            t = torch.empty(size, dtype=dtype, device='cuda').exponential_(lambd=lambd, generator=gen)
            actual_distribution = self.distribution(t)
            self.assertEqual(actual_distribution[0], "exponential(" + str(lambd) + ")")
            self.assertTrue(actual_distribution[1].statistic < 0.1)

if __name__ == "__main__":
    common.run_tests()
