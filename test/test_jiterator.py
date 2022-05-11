# Owner(s): ["module: cuda"]

import torch
from torch.cuda.jiterator import _create_jit_fn as create_jit_fn
import sys
from itertools import product
from torch.testing._internal.common_utils import TestCase, parametrize, run_tests, TEST_CUDA
from torch.testing._internal.common_dtype import all_types_and_complex_and
from torch.testing._internal.common_device_type import (
    skipCUDAIfRocm, skipCUDAIf, instantiate_device_type_tests, dtypes, toleranceOverride, tol)
from torch.testing._internal.common_cuda import _get_torch_cuda_version

if not TEST_CUDA:
    print('CUDA not available, skipping tests', file=sys.stderr)
    TestCase = object  # noqa: F811


code_string = "template <typename T> T my_fused_kernel(T x, T y, T alpha, T beta) { return alpha * x + beta * y; }"
jitted_fn = create_jit_fn(code_string, alpha=1, beta=1)

def ref_fn(x, y, alpha=1, beta=1):
    return alpha * x + beta * y

class TestPythonJiterator(TestCase):
    @skipCUDAIfRocm
    @parametrize("shape_strides", [
        (([3, 3], [3, 1]), ([3, 3], [3, 1])),  # contiguous
    ])
    @dtypes(*product(all_types_and_complex_and(torch.half, torch.bfloat16),
                     all_types_and_complex_and(torch.half, torch.bfloat16)))
    def test_all_dtype_contiguous(self, device, dtypes, shape_strides):
        a_buffer = torch.rand(9, device=device).mul(10).type(dtypes[0])
        b_buffer = torch.rand(9, device=device).mul(10).type(dtypes[1])

        a = a_buffer.as_strided(*shape_strides[0])
        b = b_buffer.as_strided(*shape_strides[1])

        expected = ref_fn(a, b)
        result = jitted_fn(a, b)

        self.assertEqual(expected, result)

    @skipCUDAIfRocm
    # See https://github.com/pytorch/pytorch/pull/76394#issuecomment-1118018287 for details
    @skipCUDAIf(_get_torch_cuda_version() < (11, 6), "On cuda 11.3, nvrtcCompileProgram is taking too long to "
                "compile jiterator generated kernels for non-contiguous input that requires dynamic-casting.")
    @parametrize("shape_strides", [
        (([3, 3], [1, 3]), ([3, 1], [1, 3])),  # non-contiguous
    ])
    @dtypes(*product(all_types_and_complex_and(torch.half, torch.bfloat16),
                     all_types_and_complex_and(torch.half, torch.bfloat16)))
    def test_all_dtype_noncontiguous(self, device, dtypes, shape_strides):
        a_buffer = torch.rand(9, device=device).mul(10).type(dtypes[0])
        b_buffer = torch.rand(9, device=device).mul(10).type(dtypes[1])

        a = a_buffer.as_strided(*shape_strides[0])
        b = b_buffer.as_strided(*shape_strides[1])

        expected = ref_fn(a, b)
        result = jitted_fn(a, b)

        self.assertEqual(expected, result)

    @skipCUDAIfRocm
    @dtypes(torch.float, torch.double, torch.float16, torch.bfloat16)
    @parametrize("alpha", [-1, 2.0, None])
    @parametrize("beta", [3, -4.2, None])
    @toleranceOverride({torch.float16 : tol(atol=1e-2, rtol=1e-3)})
    def test_extra_args(self, device, dtype, alpha, beta):
        a = torch.rand(3, device=device).mul(10).type(dtype)
        b = torch.rand(3, device=device).mul(10).type(dtype)

        extra_args = {}
        if alpha is not None:
            extra_args["alpha"] = alpha
        if beta is not None:
            extra_args["beta"] = beta

        expected = ref_fn(a, b, **extra_args)
        result = jitted_fn(a, b, **extra_args)

        self.assertEqual(expected, result)

    @skipCUDAIfRocm
    def test_bool_extra_args(self, device):
        code_string = "template <typename T> T conditional(T x, T mask, bool is_train) { return is_train ? x * mask : x; }"
        jitted_fn = create_jit_fn(code_string, is_train=False)

        def ref_fn(x, mask, is_train):
            return x * mask if is_train else x

        a = torch.rand(3, device=device)
        b = torch.rand(3, device=device)

        expected = ref_fn(a, b, is_train=True)
        result = jitted_fn(a, b, is_train=True)
        self.assertEqual(expected, result)

    @skipCUDAIfRocm
    @parametrize("num_inputs", list(range(1, 9)))
    def test_various_num_inputs(self, num_inputs):
        inputs = []
        for i in range(num_inputs):
            inputs.append(torch.rand(3, device='cuda').mul(10))

        input_string = ",".join([f"T i{i}" for i in range(num_inputs)])
        function_body = "+".join([f"i{i}" for i in range(num_inputs)])
        code_string = f"template <typename T> T my_kernel({input_string}) {{ return {function_body}; }}"
        jitted_fn = create_jit_fn(code_string)

        def ref_fn(*inputs):
            return torch.sum(torch.stack(inputs), dim=0)

        expected = ref_fn(*inputs)
        result = jitted_fn(*inputs)

        self.assertEqual(expected, result)

    @skipCUDAIfRocm
    @parametrize("code_string", [
        "template <typename T> T my _kernel(T x) { return x; }",
        "template <typename T> Tmy_kernel(T x) { return x; }",
    ])
    def test_invalid_function_name(self, code_string):
        with self.assertRaises(Exception):
            jitted_fn = create_jit_fn(code_string)


instantiate_device_type_tests(TestPythonJiterator, globals(), only_for="cuda")

if __name__ == '__main__':
    run_tests()
