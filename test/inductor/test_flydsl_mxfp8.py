# Owner(s): ["module: inductor"]

from types import SimpleNamespace
from unittest import mock

import torch
import torch.nn.functional as F
from torch._inductor import config
from torch._inductor.codegen.flydsl import flydsl_utils
from torch._inductor.kernel import mm
from torch._inductor.runtime.flydsl_cache import run_cached_flydsl
from torch._inductor.utils import run_and_get_code
from torch.nn.functional import ScalingType, SwizzleType  # type: ignore[attr-defined]
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import parametrize, run_tests, TestCase


class _FakeNode:
    def __init__(self, shape, stride, dtype, *, offset=0, device=None):
        self._shape = list(shape)
        self._stride = list(stride)
        self._dtype = dtype
        self._layout = SimpleNamespace(offset=offset)
        self._device = device or torch.device("cuda", 0)

    def get_device(self):
        return self._device

    def get_size(self):
        return self._shape

    def get_stride(self):
        return self._stride

    def get_dtype(self):
        return self._dtype

    def get_layout(self):
        return self._layout


class _CacheParam:
    def __cache_signature__(self):
        return ("mxfp8_gfx950_v1", 64, 96, 256, "bfloat16")


class TestFlyDSLMXFP8Metadata(TestCase):
    def _candidate_args(self):
        a = _FakeNode((64, 256), (256, 1), torch.float8_e4m3fn)
        b = _FakeNode((256, 96), (1, 256), torch.float8_e4m3fn)
        scale_a = _FakeNode((64, 8), (8, 1), torch.float8_e8m0fnu)
        scale_b = _FakeNode((96, 8), (8, 1), torch.float8_e8m0fnu)
        return (
            a,
            b,
            [scale_a],
            [ScalingType.BlockWise1x32.value],
            [SwizzleType.NO_SWIZZLE.value],
            [scale_b],
            [ScalingType.BlockWise1x32.value],
            [SwizzleType.NO_SWIZZLE.value],
            None,
            torch.bfloat16,
            [],
            False,
        )

    def test_exact_v2_signature(self):
        args = self._candidate_args()
        with mock.patch.object(torch.version, "hip", "test"):
            self.assertTrue(mm._is_rocm_mxfp8_v2_candidate(*args))

            bad_fast_accum = (*args[:-1], True)
            self.assertFalse(mm._is_rocm_mxfp8_v2_candidate(*bad_fast_accum))

            bad_swizzle = list(args)
            bad_swizzle[4] = [SwizzleType.SWIZZLE_32_4_4.value]
            self.assertFalse(mm._is_rocm_mxfp8_v2_candidate(*bad_swizzle))

    def test_strict_baseline_layout(self):
        args = self._candidate_args()
        a, b, scale_a, _, _, scale_b, *_ = args
        layout = SimpleNamespace(
            size=[64, 96],
            stride=[96, 1],
            dtype=torch.bfloat16,
            device=torch.device("cuda", 0),
            offset=0,
        )

        with mock.patch.object(mm, "use_flydsl_mxfp8_template", return_value=True):
            self.assertEqual(
                mm.get_flydsl_mxfp8_template_kwargs(
                    layout, a, b, scale_a[0], scale_b[0]
                ),
                {
                    "GEMM_M": 64,
                    "GEMM_N": 96,
                    "GEMM_K": 256,
                    "OUT_DTYPE": "bfloat16",
                },
            )

            bad_b = _FakeNode((256, 96), (96, 1), torch.float8_e4m3fn)
            self.assertIsNone(
                mm.get_flydsl_mxfp8_template_kwargs(
                    layout, a, bad_b, scale_a[0], scale_b[0]
                )
            )

            bad_scale = _FakeNode((64, 8), (8, 1), torch.float8_e8m0fnu, offset=1)
            self.assertIsNone(
                mm.get_flydsl_mxfp8_template_kwargs(layout, a, b, bad_scale, scale_b[0])
            )

            bad_device_scale = _FakeNode(
                (64, 8),
                (8, 1),
                torch.float8_e8m0fnu,
                device=torch.device("cpu"),
            )
            self.assertIsNone(
                mm.get_flydsl_mxfp8_template_kwargs(
                    layout, a, b, bad_device_scale, scale_b[0]
                )
            )

            with mock.patch.object(
                mm,
                "is_unaligned",
                side_effect=lambda node: node is a,
            ):
                self.assertIsNone(
                    mm.get_flydsl_mxfp8_template_kwargs(
                        layout, a, b, scale_a[0], scale_b[0]
                    )
                )

    def test_runtime_cache_reuses_specialization(self):
        jit_func = SimpleNamespace()
        compiled = mock.Mock()
        compiler = mock.Mock(return_value=compiled)

        first = run_cached_flydsl(
            jit_func,
            "compile-args",
            constexpr_param=_CacheParam(),
            compiler=compiler,
            dispatch_args=("first-dispatch",),
        )
        second = run_cached_flydsl(
            jit_func,
            "different-compile-args",
            constexpr_param=_CacheParam(),
            compiler=compiler,
            dispatch_args=("second-dispatch",),
        )

        self.assertIs(first, compiled)
        self.assertIs(second, compiled)
        compiler.assert_called_once_with(jit_func, "compile-args")
        compiled.assert_called_once_with("second-dispatch")

    def test_precompile_uses_flydsl_compile_only_contract(self):
        source = mm.flydsl_mxfp8_scaled_mm_template.source

        self.assertIn('{"COMPILE_ONLY": "1"}', source)
        self.assertNotIn("FLYDSL_COMPILE_ONLY", source)


class TestFlyDSLMXFP8Device(TestCase):
    def _make_inputs(self, m, n, k, device):
        a_values = ((torch.arange(m * k, device=device) % 9) - 4) / 2
        b_values = ((torch.arange(n * k, device=device) % 11) - 5) / 2
        a = a_values.reshape(m, k).to(torch.float8_e4m3fn)
        b = b_values.reshape(n, k).to(torch.float8_e4m3fn)

        k32 = k // 32
        a_exp = (
            torch.arange(m, device=device)[:, None]
            + torch.arange(k32, device=device)[None, :]
        ) % 5 - 2
        b_exp = (
            2 * torch.arange(n, device=device)[:, None]
            + torch.arange(k32, device=device)[None, :]
        ) % 5 - 2
        scale_a = torch.pow(2.0, a_exp.float()).to(torch.float8_e8m0fnu)
        scale_b = torch.pow(2.0, b_exp.float()).to(torch.float8_e8m0fnu)
        return a, b, scale_a, scale_b

    @parametrize("out_dtype", [torch.bfloat16, torch.float16])
    def test_scaled_mm_v2_flydsl_baseline(self, device, out_dtype):
        if torch.version.hip is None:
            self.skipTest("requires ROCm")
        arch = torch.cuda.get_device_properties(device).gcnArchName.split(":", 1)[0]
        if arch != "gfx950":
            self.skipTest("requires gfx950")
        if not flydsl_utils.runtime_available():
            self.skipTest("FlyDSL runtime unavailable")

        m, n, k = 64, 96, 256
        a, b, scale_a, scale_b = self._make_inputs(m, n, k, device)

        def fn(a, b, scale_a, scale_b):
            return F.scaled_mm(
                a,
                b.t(),
                scale_a,
                ScalingType.BlockWise1x32,
                scale_b,
                ScalingType.BlockWise1x32,
                swizzle_a=SwizzleType.NO_SWIZZLE,
                swizzle_b=SwizzleType.NO_SWIZZLE,
                output_dtype=out_dtype,
            )

        expected = fn(a, b, scale_a, scale_b)
        a_dequant = a.float() * scale_a.float().repeat_interleave(32, 1)
        b_dequant = b.float() * scale_b.float().repeat_interleave(32, 1)
        reference = (a_dequant @ b_dequant.t()).to(out_dtype)
        self.assertEqual(expected, reference, rtol=2e-2, atol=5e-1)

        torch._dynamo.reset()
        with config.patch(
            max_autotune_gemm=True,
            max_autotune_gemm_backends="FLYDSL",
        ):
            compiled = torch.compile(fn, backend="inductor", fullgraph=True)
            actual, (code,) = run_and_get_code(compiled, a, b, scale_a, scale_b)

        self.assertEqual(actual, expected, rtol=2e-2, atol=5e-1)
        self.assertIn("async_compile.flydsl", code)
        self.assertIn("make_mxfp8_scaled_mm_gfx950", code)
        self.assertIn("mat2.transpose(0, 1)", code)
        self.assertNotIn("extern_kernels._scaled_mm_v2(", code)


instantiate_device_type_tests(TestFlyDSLMXFP8Device, globals(), only_for="cuda")


if __name__ == "__main__":
    run_tests()
