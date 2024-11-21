# Owner(s): ["module: inductor"]

import os
import unittest

import torch
from torch._dynamo.utils import same
from torch._inductor import config as inductor_config, metrics
from torch._inductor.codegen.triton import TritonScheduling
from torch._inductor.test_case import run_tests, TestCase
from torch.testing._internal.common_cuda import PLATFORM_SUPPORTS_FP8
from torch.testing._internal.inductor_utils import GPU_TYPE, HAS_GPU
from torch.utils._pytree import tree_map


DO_PERF_TEST = os.environ.get("DO_PERF_TEST") == "1"


if HAS_GPU:
    torch.set_default_device(GPU_TYPE)


class MockScheduler:
    available_buffer_names = ()

    @staticmethod
    def get_backend(cls, *args):
        return TritonScheduling(cls)


@inductor_config.patch(
    {
        "benchmark_kernel": True,
        "loop_ordering_after_fusion": True,
        "triton.unique_kernel_names": True,
        "defer_reduction_split": True,
    }
)
class DeferReductionSplitTest(TestCase):
    device = GPU_TYPE

    def do_acc_test(self, f, *args, cast_fp8=True):
        expect = f(*args)
        actual = torch.compile(f)(*args)

        if cast_fp8:

            def _cast(x):
                if isinstance(x, torch.Tensor) and x.dtype in (
                    torch.float8_e5m2,
                    torch.float8_e4m3fn,
                ):
                    return x.to(torch.float32)
                return x

            # Wordaround the issue that call allclose on fp8 tensor triggers error
            #   RuntimeError: "mul_cuda" not implemented for 'Float8_e4m3fn'
            expect = tree_map(_cast, expect)
            actual = tree_map(_cast, actual)
        self.assertTrue(same(expect, actual, tol=1e-3))

    def setUp(self):
        super().setUp()
        metrics.reset()

    def test_patter_1(self):
        def f(x):
            y = x.abs().max(dim=-1)
            z = x.abs().max()
            return y[0], z

        A, B = 1024, 1536
        x = torch.randn(A, B, device=GPU_TYPE)

        self.do_acc_test(f, x)
        self.assertEqual(2, metrics.generated_kernel_count)
        expected_num_bytes = A * B + 3 * A + 1
        expected_num_bytes *= x.itemsize
        self.assertEqual(expected_num_bytes, metrics.num_bytes_accessed)

    def test_patter_2(self):
        def f(x):
            y = x.abs().max()
            z = x / 10.0
            z_t = z.t().contiguous().t()
            return y, z, z_t

        A, B = 1024, 1536
        x = torch.randn(A, B, device=GPU_TYPE)

        self.do_acc_test(f, x)

        self.assertEqual(2, metrics.generated_kernel_count)

        expected_num_bytes = A * B * 3 + A * 2 + 1
        expected_num_bytes *= x.itemsize
        self.assertEqual(expected_num_bytes, metrics.num_bytes_accessed)

    @unittest.skipIf(not PLATFORM_SUPPORTS_FP8, "FP8 requires H100+ and MI300+")
    def test_patter_fp8(self):
        ref_dtype = torch.bfloat16
        M, K = 4096, 2048

        input_tensor = torch.randn(
            M, K, device="cuda", dtype=ref_dtype, requires_grad=False
        )
        scale = torch.Tensor([10.0]).to("cuda")

        E4M3_MAX_POS = torch.finfo(torch.float8_e4m3fn).max

        def f(tensor_x_inp, scale_x):
            tensor_x = tensor_x_inp * scale_x
            tensor_x = tensor_x.clamp(min=-1 * E4M3_MAX_POS, max=E4M3_MAX_POS)
            tensor_fp8 = tensor_x.to(torch.float8_e4m3fn)

            tensor_x_t = (tensor_x_inp * scale_x).t()
            tensor_x_t = tensor_x_t.clamp(min=-1 * E4M3_MAX_POS, max=E4M3_MAX_POS)
            tensor_fp8_t = tensor_x_t.to(torch.float8_e4m3fn)

            tensor_fp8_t = tensor_fp8_t.contiguous().t()

            new_scale = tensor_x_inp.abs().max()

            return (tensor_fp8, tensor_fp8_t, new_scale)

        test_pattern = torch.compile(f)
        tensor_fp8, tensor_fp8_t, new_scale = test_pattern(input_tensor, scale)

        # fused_cast_transpose_amax, second_level_amax_reduction
        self.assertEqual(2, metrics.generated_kernel_count)

        expected_numbytes = scale.nbytes  # scalar
        expected_numbytes += input_tensor.nbytes  # input
        expected_numbytes += tensor_fp8.nbytes + tensor_fp8_t.nbytes  # output
        expected_numbytes += new_scale.nbytes  # new scaler output
        expected_numbytes += 2 * M * scale.itemsize  # second-level reduction in/output
        self.assertEqual(expected_numbytes, metrics.num_bytes_accessed)

        self.do_acc_test(f, input_tensor, scale)

    def test_patter_cannot_fuse(self):
        def f(x):
            y = x.abs().max()
            z = x / 10.0
            z_t = z.t().contiguous().t()
            return y, z, z_t

        A, B = 8192, 128
        x = torch.randn(A, B, device=GPU_TYPE)

        self.do_acc_test(f, x)

        # The reduction tiling should not be modified. 128 is too small compared with the original split_size
        self.assertEqual(3, metrics.generated_kernel_count)


if __name__ == "__main__":
    if HAS_GPU:
        run_tests()
