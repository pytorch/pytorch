import unittest

import torch
import torch.utils.cpp_extension
import torch_test_cpp_extension as cpp_extension

import common

TEST_CUDA = torch.cuda.is_available()


class TestCppExtension(common.TestCase):
    def test_extension_function(self):
        x = torch.randn(4, 4)
        y = torch.randn(4, 4)
        z = cpp_extension.sigmoid_add(x, y)
        self.assertEqual(z, x.sigmoid() + y.sigmoid())

    def test_extension_module(self):
        mm = cpp_extension.MatrixMultiplier(4, 8)
        weights = torch.rand(8, 4)
        expected = mm.get().mm(weights)
        result = mm.forward(weights)
        self.assertEqual(expected, result)

    def test_jit_compile_extension(self):
        module = torch.utils.cpp_extension.load(
            name='jit_extension',
            sources=[
                'cpp_extensions/jit_extension.cpp',
                'cpp_extensions/jit_extension2.cpp'
            ],
            extra_include_paths=['cpp_extensions'],
            extra_cflags=['-g'],
            verbose=True)
        x = torch.randn(4, 4)
        y = torch.randn(4, 4)

        z = module.tanh_add(x, y)
        self.assertEqual(z, x.tanh() + y.tanh())

        # Checking we can call a method defined not in the main C++ file.
        z = module.exp_add(x, y)
        self.assertEqual(z, x.exp() + y.exp())

        # Checking we can use this JIT-compiled class.
        doubler = module.Doubler(2, 2)
        self.assertEqual(doubler.get().sum(), 4)
        self.assertEqual(doubler.forward().sum(), 8)

    @unittest.skipIf(not TEST_CUDA, "CUDA not found")
    def test_cuda_extension(self):
        import torch_test_cuda_extension as cuda_extension

        x = torch.FloatTensor(100).zero_().cuda()
        y = torch.FloatTensor(100).zero_().cuda()

        z = cuda_extension.sigmoid_add(x, y).cpu()

        # 2 * sigmoid(0) = 2 * 0.5 = 1
        self.assertEqual(z, torch.ones_like(z))

    @unittest.skipIf(not TEST_CUDA, "CUDA not found")
    def test_jit_cuda_extension(self):
        # NOTE: The name of the extension must equal the name of the module.
        module = torch.utils.cpp_extension.load(
            name='torch_test_cuda_extension',
            sources=[
                'cpp_extensions/cuda_extension.cpp',
                'cpp_extensions/cuda_extension.cu'
            ],
            extra_cuda_cflags=['-O2'],
            verbose=True)

        x = torch.FloatTensor(100).zero_().cuda()
        y = torch.FloatTensor(100).zero_().cuda()

        z = module.sigmoid_add(x, y).cpu()

        # 2 * sigmoid(0) = 2 * 0.5 = 1
        self.assertEqual(z, torch.ones_like(z))


if __name__ == '__main__':
    common.run_tests()
