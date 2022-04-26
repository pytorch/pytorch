import torch
from torch.cuda.jiterator import create_jit_fn
from torch.testing._internal.common_utils import TestCase, parametrize, run_tests, instantiate_parametrized_tests

code_string = "template <typename T> T python_jitted(T x, T y, T alpha, T beta) { return  -x * y + x - y + alpha - beta; }"
jitted_fn = create_jit_fn(code_string, "python_jitted", alpha=0, beta=0)

def ref_fn(x, y, alpha=0, beta=0):
    return -x * y + x - y + alpha - beta

class TestPythonJiterator(TestCase):
    @parametrize("dtype", [
                           torch.float, torch.double, torch.half,
                        #    torch.bfloat16,  failing due to numerical difference
                           torch.uint8, torch.int8, torch.int16, torch.int, torch.long,
                           torch.complex64, torch.complex128,
                            #    torch.bool,
                           ])
    def test_all_dtypes(self, dtype):
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

        rtol = 0.00001
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

        rtol = 0.00001
        if a_dtype is torch.half or b_dtype is torch.half:
            rtol = 1e-2
        assert torch.allclose(expected, result, rtol=rtol)

    @parametrize("dtype", [
                           torch.float, torch.double, torch.half,
                        #    output type is mismatching for following cases
                        #    torch.uint8, torch.int8, torch.int16, torch.int, torch.long,
                            #    torch.bfloat16,  failing due to numerical difference
                           ])
    @parametrize("alpha", [-1, 2.0, None])
    @parametrize("beta", [3, -4.2, None])
    def test_extra_args(self, dtype, alpha, beta):
        a = torch.rand(3, device='cuda').mul(10).type(dtype)
        b = torch.rand(3, device='cuda').mul(10).type(dtype)

        extra_args = {}
        if alpha is not None:
            extra_args["alpha"] = alpha
        if beta is not None:
            extra_args["beta"] = beta

        expected = ref_fn(a, b, **extra_args)
        result = jitted_fn(a, b, **extra_args)

        rtol = 0.00001
        if dtype is torch.half:
            rtol = 1e-2
        assert torch.allclose(expected, result, rtol=rtol)

    @parametrize("num_inputs", [1, 2, 3])
    def test_various_num_inputs(self, num_inputs):
        inputs = []
        for i in range(num_inputs):
            inputs.append(torch.rand(3, device='cuda').mul(10))

        input_string = ",".join([f"T i{i}" for i in range(num_inputs)])
        function_body = "+".join([f"i{i}" for i in range(num_inputs)])
        code_string = f"template <typename T> T python_jitted({input_string}) {{ return {function_body}; }}"
        jitted_fn = create_jit_fn(code_string, "python_jitted")

        def ref_fn(*inputs):
            return torch.sum(torch.stack(inputs), dim=0)

        expected = ref_fn(*inputs)
        result = jitted_fn(*inputs)

        assert torch.allclose(expected, result)

instantiate_parametrized_tests(TestPythonJiterator)

if __name__ == '__main__':
    run_tests()