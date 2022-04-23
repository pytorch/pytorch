import torch
from torch.cuda.jiterator import create_jit_fn
from torch.testing._internal.common_utils import TestCase, parametrize, run_tests, instantiate_parametrized_tests

code_string = "template <typename T> T python_jitted(T x, T y) { return  -x * y + x - y; }"
jitted_fn = create_jit_fn(code_string, "python_jitted", alpha=1, beta=2)

# c = fn(a, b, beta=3, alpha=-1)

def ref_fn(x, y):
    return -x * y + x - y

class TestPythonJiterator(TestCase):
    @parametrize("dtype", [
                           torch.float, torch.double, torch.half,
                        #    torch.bfloat16,  failing due to numerical difference
                           torch.uint8, torch.int8, torch.int16, torch.int, torch.long,
                           torch.complex64, torch.complex128,
                            #    torch.bool,
                           ])
    def test_all_types(self, dtype):
        a = torch.rand(3, device='cuda').mul(10).type(dtype)
        b = torch.rand(3, device='cuda').mul(10).type(dtype)

        expected = ref_fn(a, b)
        result = jitted_fn(a, b)

        rtol =  0.00001
        if dtype is torch.half:
            rtol = 1e-2
        assert torch.allclose(expected, result, rtol=rtol)

    shape_stride_cases = [
        # shape: [1]
        ([1], [1]),     # contiguous
        # shape: [3]
        ([3], [1]),     # contiguous
        ([3], [3]),     # non-contiguous
        # shape: [3, 1]
        ([3,1], [1,1]), # contiguous
        ([3,1], [1,3]), # non-contiguous
        # shape: [1, 3]
        ([1,3], [3,1]), # contiguous
        ([1,3], [1,2]), # non-contiguous
        # shape: [3, 3]
        ([3,3], [3,1]), # contiguous
        ([3,3], [1,3]), # non-contiguous
        ([3,3], [0,2]), # non-contiguous
        ([3,3], [2,1]), # non-contiguous
    ]
    @parametrize("a_shape_stride", shape_stride_cases)
    @parametrize("b_shape_stride", shape_stride_cases)
    def test_all_shape_stride(self, a_shape_stride, b_shape_stride, dtype=torch.float):
        a_buffer = torch.rand(9, device='cuda').mul(10).type(dtype)
        b_buffer = torch.rand(9, device='cuda').mul(10).type(dtype)

        a = a_buffer.as_strided(*a_shape_stride)
        b = b_buffer.as_strided(*b_shape_stride)

        expected = ref_fn(a, b)
        result = jitted_fn(a, b)

        rtol =  0.00001
        if dtype is torch.half:
            rtol = 1e-2
        assert torch.allclose(expected, result, rtol=rtol)

    shape_stride_cases_sm= [
        # shape: [3, 1]
        ([3,1], [1,3]), # non-contiguous
        # shape: [3, 3]
        ([3,3], [3,1]), # contiguous
        ([3,3], [1,3]), # non-contiguous
    ]
    dtypes = [
        torch.float, torch.double, torch.half,
    #    torch.bfloat16,  failing due to numerical difference
        # torch.uint8,   failing
        torch.int8, torch.int16, torch.int, torch.long,
        torch.complex64, torch.complex128,
    ]
    @parametrize("a_dtype", dtypes)
    @parametrize("b_dtype", dtypes)
    @parametrize("a_shape_stride", shape_stride_cases_sm)
    @parametrize("b_shape_stride", shape_stride_cases_sm)
    def test_all_dynamic_casts(self, a_dtype, b_dtype, a_shape_stride, b_shape_stride):
        a_buffer = torch.rand(9, device='cuda').mul(10).type(a_dtype)
        b_buffer = torch.rand(9, device='cuda').mul(10).type(b_dtype)

        a = a_buffer.as_strided(*a_shape_stride)
        b = b_buffer.as_strided(*b_shape_stride)

        expected = ref_fn(a, b)
        result = jitted_fn(a, b)

        rtol =  0.00001
        if a_dtype is torch.half or b_dtype is torch.half:
            rtol = 1e-2
        assert torch.allclose(expected, result, rtol=rtol)

instantiate_parametrized_tests(TestPythonJiterator)

if __name__ == '__main__':
    run_tests()