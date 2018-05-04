import unittest

import torch
import torch.utils.cpp_extension
try:
    import torch_test_cpp_extension.cpp as cpp_extension
except ImportError:
    print("\'test_cpp_extensions.py\' cannot be invoked directly. " +
          "Run \'python run_test.py -i cpp_extensions\' for the \'test_cpp_extensions.py\' tests.")
    raise

import common

from torch.utils.cpp_extension import CUDA_HOME
TEST_CUDA = torch.cuda.is_available() and CUDA_HOME is not None


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

    def test_backward(self):
        mm = cpp_extension.MatrixMultiplier(4, 8)
        weights = torch.rand(8, 4, requires_grad=True)
        result = mm.forward(weights)
        result.sum().backward()
        tensor = mm.get()

        expected_weights_grad = tensor.t().mm(torch.ones([4, 4]))
        self.assertEqual(weights.grad, expected_weights_grad)

        expected_tensor_grad = torch.ones([4, 4]).mm(weights.t())
        self.assertEqual(tensor.grad, expected_tensor_grad)

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
        self.assertIsNone(doubler.get().grad)
        self.assertEqual(doubler.get().sum(), 4)
        self.assertEqual(doubler.forward().sum(), 8)

    @unittest.skipIf(not TEST_CUDA, "CUDA not found")
    def test_cuda_extension(self):
        import torch_test_cpp_extension.cuda as cuda_extension

        x = torch.zeros(100, device='cuda', dtype=torch.float32)
        y = torch.zeros(100, device='cuda', dtype=torch.float32)

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

        x = torch.zeros(100, device='cuda', dtype=torch.float32)
        y = torch.zeros(100, device='cuda', dtype=torch.float32)

        z = module.sigmoid_add(x, y).cpu()

        # 2 * sigmoid(0) = 2 * 0.5 = 1
        self.assertEqual(z, torch.ones_like(z))

    def test_optional(self):
        has_value = cpp_extension.function_taking_optional(torch.ones(5))
        self.assertTrue(has_value)
        has_value = cpp_extension.function_taking_optional(None)
        self.assertFalse(has_value)

    def test_inline_jit_compile_extension_with_functions_as_list(self):
        cpp_source = '''
        at::Tensor tanh_add(at::Tensor x, at::Tensor y) {
          return x.tanh() + y.tanh();
        }
        '''

        module = torch.utils.cpp_extension.load_inline(
            name='inline_jit_extension_with_functions_list',
            cpp_sources=cpp_source,
            functions='tanh_add',
            verbose=True)

        self.assertEqual(module.tanh_add.__doc__.split('\n')[2], 'tanh_add')

        x = torch.randn(4, 4)
        y = torch.randn(4, 4)

        z = module.tanh_add(x, y)
        self.assertEqual(z, x.tanh() + y.tanh())

    def test_inline_jit_compile_extension_with_functions_as_dict(self):
        cpp_source = '''
        at::Tensor tanh_add(at::Tensor x, at::Tensor y) {
          return x.tanh() + y.tanh();
        }
        '''

        module = torch.utils.cpp_extension.load_inline(
            name='inline_jit_extension_with_functions_dict',
            cpp_sources=cpp_source,
            functions={'tanh_add': 'Tanh and then sum :D'},
            verbose=True)

        self.assertEqual(
            module.tanh_add.__doc__.split('\n')[2], 'Tanh and then sum :D')

    def test_inline_jit_compile_extension_multiple_sources_and_no_functions(self):
        cpp_source1 = '''
        at::Tensor sin_add(at::Tensor x, at::Tensor y) {
          return x.sin() + y.sin();
        }
        '''

        cpp_source2 = '''
        #include <torch/torch.h>
        at::Tensor sin_add(at::Tensor x, at::Tensor y);
        PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
          m.def("sin_add", &sin_add, "sin(x) + sin(y)");
        }
        '''

        module = torch.utils.cpp_extension.load_inline(
            name='inline_jit_extension',
            cpp_sources=[cpp_source1, cpp_source2],
            verbose=True)

        x = torch.randn(4, 4)
        y = torch.randn(4, 4)

        z = module.sin_add(x, y)
        self.assertEqual(z, x.sin() + y.sin())

    @unittest.skipIf(not TEST_CUDA, "CUDA not found")
    def test_inline_jit_compile_extension_cuda(self):
        cuda_source = '''
        __global__ void cos_add_kernel(
            const float* __restrict__ x,
            const float* __restrict__ y,
            float* __restrict__ output,
            const int size) {
          const auto index = blockIdx.x * blockDim.x + threadIdx.x;
          if (index < size) {
            output[index] = __cosf(x[index]) + __cosf(y[index]);
          }
        }

        at::Tensor cos_add(at::Tensor x, at::Tensor y) {
          auto output = at::zeros_like(x);
          const int threads = 1024;
          const int blocks = (output.numel() + threads - 1) / threads;
          cos_add_kernel<<<blocks, threads>>>(x.data<float>(), y.data<float>(), output.data<float>(), output.numel());
          return output;
        }
        '''

        # Here, the C++ source need only declare the function signature.
        cpp_source = 'at::Tensor cos_add(at::Tensor x, at::Tensor y);'

        module = torch.utils.cpp_extension.load_inline(
            name='inline_jit_extension_cuda',
            cpp_sources=cpp_source,
            cuda_sources=cuda_source,
            functions=['cos_add'],
            verbose=True)

        self.assertEqual(module.cos_add.__doc__.split('\n')[2], 'cos_add')

        x = torch.randn(4, 4, device='cuda', dtype=torch.float32)
        y = torch.randn(4, 4, device='cuda', dtype=torch.float32)

        z = module.cos_add(x, y)
        self.assertEqual(z, x.cos() + y.cos())

    def test_inline_jit_compile_extension_throws_when_functions_is_bad(self):
        with self.assertRaises(ValueError):
            torch.utils.cpp_extension.load_inline(
                name='invalid_jit_extension', cpp_sources='', functions=5)
