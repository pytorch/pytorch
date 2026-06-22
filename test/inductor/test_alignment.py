# Owner(s): ["module: inductor"]
import contextlib
import sys
import unittest

import torch
from torch._inductor import config
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    MACOS_VERSION,
    parametrize,
)
from torch.testing._internal.inductor_utils import GPU_TYPE, RUN_CPU, RUN_GPU


try:
    try:
        from . import test_torchinductor
    except ImportError:
        import test_torchinductor  # @manual=fbcode//caffe2/test/inductor:test_inductor-library
except unittest.SkipTest:
    if __name__ == "__main__":
        sys.exit(0)
    raise

TestCase = test_torchinductor.TestCase
check_model = test_torchinductor.check_model
check_model_gpu = test_torchinductor.check_model_gpu
skip_if_cpp_wrapper = test_torchinductor.skip_if_cpp_wrapper
copy_tests = test_torchinductor.copy_tests
define_custom_op_for_test = test_torchinductor.define_custom_op_for_test


@instantiate_parametrized_tests
class CommonTemplate:
    def test_unaligned_input(self):
        def fn(x):
            return torch.nn.functional.relu(x)

        x = torch.randn(1024 + 16, device=self.device)[1:-15]
        # TODO (malfet): Investigate failures on MacOS-14
        with (
            contextlib.nullcontext()
            if self.device != "mps" or MACOS_VERSION >= 15.0
            else self.assertRaises(AssertionError)
        ):
            self.common(fn, (x,), check_lowp=False)

    def test_unaligned_input_2d(self):
        def fn(x):
            return torch.nn.functional.relu(x)

        x = torch.randn(1024, 1024 + 16, device=self.device)[:, 1:-15]
        self.common(fn, (x,), check_lowp=False)

    def test_alignment_without_custom_op(self):
        def fn(x):
            a = torch.nn.functional.relu(x)
            b = (3 * a)[1:-15]
            c = torch.cos(b)
            return c

        x = torch.randn(1024 + 16, device=self.device)
        self.common(fn, (x,), check_lowp=False)

    @config.patch(implicit_fallbacks=True)
    def test_no_align_for_custom_op(self):
        def slice1d(x):
            return (3 * x)[1:-15]

        def slice1d_meta(x):
            return torch.empty_like(x)[1:-15]

        define_custom_op_for_test("slice1d", slice1d, slice1d_meta)

        def fn(x):
            a = torch.nn.functional.relu(x)
            b = torch.ops.test.slice1d(a)
            c = torch.cos(b)
            return c

        x = torch.randn(1024 + 16, device=self.device)
        self.common(fn, (x,), check_lowp=False)

    @config.patch(implicit_fallbacks=True)
    def test_no_align_for_custom_op_2d(self):
        def slice2d(x):
            return (3 * x)[..., 1:-15]

        def slice2d_meta(x):
            return torch.empty_like(x)[..., 1:-15]

        define_custom_op_for_test("slice2d", slice2d, slice2d_meta)

        def fn(x):
            a = torch.nn.functional.relu(x)
            b = torch.ops.test.slice2d(a)
            c = torch.cos(b)
            return c

        x = torch.randn(1024, 1024 + 16, device=self.device)
        self.common(fn, (x,), check_lowp=False)

    @config.patch(implicit_fallbacks=True, alignment_asserts=True)
    @skip_if_cpp_wrapper(
        "Inductor does not generate alignment assertion for cpp_wrapper right now"
    )
    def test_incorrect_meta_for_custom_op_2d(self):
        def slice2d(x):
            return (3 * x)[..., 1:-15]

        def slice2d_meta(x):
            return torch.empty_like(x)[..., 0:-16]

        define_custom_op_for_test("slice2d_incorrect_meta", slice2d, slice2d_meta)

        def fn(x):
            a = torch.nn.functional.relu(x)
            b = torch.ops.test.slice2d_incorrect_meta(a)
            c = torch.cos(b)
            return c

        x = torch.randn(1024, 1024 + 16, device=self.device)

        expected_error = "Expect the tensor to be 16 bytes aligned. Fail due to storage_offset=1 itemsize=4"
        with self.assertRaisesRegex(AssertionError, expected_error):
            self.common(fn, (x,), check_lowp=False)

    def test_slice(self):
        def f(x):
            return x[1:] + 1

        x = torch.randn(1025, device=self.device)
        self.common(f, (x,))

    def test_view_dtype_slice(self):
        def f(x):
            return x.view(dtype=torch.float32)[1:] + 1

        x = torch.randn(1025 * 2, dtype=torch.bfloat16, device=self.device)
        self.common(f, (x,), reference_in_float=False)

    @parametrize(
        "size",
        (
            # wrapper for size = 128: https://gist.github.com/shunting314/88f1e72957b9fc5e9826aaa346a0e652
            # ptx: https://gist.github.com/shunting314/eb657ee8821eef9f0685b7b91e2ad5c2
            # the ptx file uses ld.global.b32 to load input buffer
            128,
            # wrapper for size = 1024: https://gist.github.com/shunting314/d7f64e1f52f6b1e2ec25e1a51052ce43
            # ptx: https://gist.github.com/shunting314/a24ff7563bb6b04523d11b119ab0f2b2
            # the ptx file uses ld.global.v2.b32 to load input buffer
            1024,
            # wrapper for size = 1024 * 1024: https://gist.github.com/shunting314/016b95cf0b6e9a75c25f5c9d5ed0a2ba
            # ptx: https://gist.github.com/shunting314/360112a4893c759b114c12fc99958297
            # the ptx file uses ld.global.v4.b32 to load input buffer
            1024 * 1024,
        ),
    )
    def test_slice_view_dtype(self, size):
        offset = 1

        def f(x):
            return x[2:].view(dtype=torch.float32) + 1

        x = torch.randn((size + offset) * 2, dtype=torch.bfloat16, device=self.device)
        self.common(f, (x,), reference_in_float=False)

    def test_Q4_K_dequantization(self):
        """
        Test the alignment issue for Q4_K dequantization.
        """

        QK_K = 256
        K_SCALE_SIZE = 12

        def get_scale_min(scales):
            n_blocks = scales.shape[0]
            scales = scales.view(torch.uint8)
            scales = scales.reshape((n_blocks, 3, 4))

            d, m, m_d = torch.split(scales, scales.shape[-2] // 3, dim=-2)

            sc = torch.cat([d & 0x3F, (m_d & 0x0F) | ((d >> 2) & 0x30)], dim=-1)
            min = torch.cat([m & 0x3F, (m_d >> 4) | ((m >> 2) & 0x30)], dim=-1)

            return (sc.reshape((n_blocks, 8)), min.reshape((n_blocks, 8)))

        def split_block_dims(blocks, *args):
            n_max = blocks.shape[1]
            dims = list(args) + [n_max - sum(args)]
            return torch.split(blocks, dims, dim=1)

        def dequantize_blocks_Q4_K(blocks, block_size, type_size):
            n_blocks = blocks.shape[0]

            d, dmin, scales, qs = split_block_dims(blocks, 2, 2, K_SCALE_SIZE)
            d = d.view(torch.float16)
            dmin = dmin.view(torch.float16)

            sc, m = get_scale_min(scales)

            d = (d * sc).reshape((n_blocks, -1, 1))
            dm = (dmin * m).reshape((n_blocks, -1, 1))

            qs = qs.reshape((n_blocks, -1, 1, 32)) >> torch.tensor(
                [0, 4], device=d.device, dtype=torch.uint8
            ).reshape((1, 1, 2, 1))
            qs = (qs & 0x0F).reshape((n_blocks, -1, 32))

            return (d * qs - dm).reshape((n_blocks, QK_K))

        data = torch.randint(
            0, 16, (18432, 1728), device=self.device, dtype=torch.uint8
        )

        def dequantize(data):
            block_size, type_size = 256, 144
            rows = data.reshape((-1, data.shape[-1])).view(torch.uint8)
            n_blocks = rows.numel() // type_size
            blocks = rows.reshape((n_blocks, type_size))
            blocks = dequantize_blocks_Q4_K(blocks, block_size, type_size)
            return blocks.reshape(18432, 3072)

        self.common(dequantize, (data,), check_lowp=False, atol=1e-3, rtol=1e-3)


if RUN_CPU:

    class CpuTests(TestCase):
        common = check_model
        device = "cpu"

    copy_tests(CommonTemplate, CpuTests, "cpu")

if RUN_GPU:

    class GPUTests(TestCase):
        common = check_model_gpu
        device = GPU_TYPE

    copy_tests(CommonTemplate, GPUTests, GPU_TYPE)

if __name__ == "__main__":
    from torch._inductor.test_case import run_tests

    if RUN_CPU or RUN_GPU:
        run_tests()
