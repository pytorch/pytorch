# Owner(s): ["module: inductor"]

from types import SimpleNamespace

import torch
from torch._inductor import config
from torch._inductor.codegen.flydsl import flydsl_utils
from torch._inductor.kernel import mm
from torch._inductor.utils import run_and_get_code
from torch.nn.functional import ScalingType, SwizzleType  # type: ignore[attr-defined]
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TestCase,
)


# E2M1 has eight magnitudes and no inf/nan; code = sign << 3 | magnitude index.
E2M1_MAGNITUDES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


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


def make_mxfp4_operand(rows, k, device, generator=None):
    """Return (packed fp4x2 [rows, k // 2], e8m0 scale [rows, k // 32], fp32).

    Values are drawn on the E2M1 grid and the scales are exact powers of two, so
    the fp32 tensor is the operand's exact value rather than an approximation of
    it. That keeps a numerics check measuring the kernel instead of the
    quantizer.
    """
    codes = torch.randint(
        0, 16, (rows, k), device=device, dtype=torch.uint8, generator=generator
    )
    lut = torch.tensor(E2M1_MAGNITUDES, device=device, dtype=torch.float32)
    magnitude = lut[(codes & 7).long()]
    values = torch.where(codes >= 8, -magnitude, magnitude)

    exponents = torch.randint(
        124, 131, (rows, k // 32), device=device, dtype=torch.uint8, generator=generator
    )
    scales = torch.pow(2.0, exponents.float() - 127.0)

    packed = (
        codes[:, 0::2].to(torch.int16) | (codes[:, 1::2].to(torch.int16) << 4)
    ).to(torch.uint8)
    return (
        packed.contiguous().view(torch.float4_e2m1fn_x2),
        exponents.contiguous().view(torch.float8_e8m0fnu),
        values * scales.repeat_interleave(32, dim=1),
    )


def scaled_mm_mxfp4(a, b_t, scale_a, scale_b, out_dtype):
    """A [M, K // 2] x B [K // 2, N] under the ROCm MXFP4 contract."""
    return torch._scaled_mm_v2(
        a,
        b_t,
        [scale_a],
        [ScalingType.BlockWise1x32.value],
        [SwizzleType.NO_SWIZZLE.value],
        [scale_b],
        [ScalingType.BlockWise1x32.value],
        [SwizzleType.NO_SWIZZLE.value],
        None,
        out_dtype,
    )


class TestFlyDSLMXFP4Metadata(TestCase):
    def _candidate_args(self, **overrides):
        m, n, k = 64, 96, 256
        a = _FakeNode((m, k // 2), (k // 2, 1), torch.float4_e2m1fn_x2)
        b = _FakeNode((k // 2, n), (1, k // 2), torch.float4_e2m1fn_x2)
        scale_a = _FakeNode((m, k // 32), (k // 32, 1), torch.float8_e8m0fnu)
        scale_b = _FakeNode((n, k // 32), (k // 32, 1), torch.float8_e8m0fnu)
        args = {
            "mat_a": a,
            "mat_b": b,
            "scale_a": [scale_a],
            "recipe_a": [ScalingType.BlockWise1x32.value],
            "swizzle_a": [SwizzleType.NO_SWIZZLE.value],
            "scale_b": [scale_b],
            "recipe_b": [ScalingType.BlockWise1x32.value],
            "swizzle_b": [SwizzleType.NO_SWIZZLE.value],
            "bias": None,
            "out_dtype": torch.bfloat16,
            "contraction_dim": None,
            "use_fast_accum": False,
        }
        args.update(overrides)
        return args

    def test_exact_v2_signature(self):
        if torch.version.hip is None:
            self.skipTest("ROCm-only gate")
        self.assertEqual(
            mm._get_rocm_mxfp_v2_format(**self._candidate_args()), "mxfp4"
        )

    @parametrize(
        "override",
        [
            {"swizzle_a": [SwizzleType.SWIZZLE_32_4_4.value]},
            {"recipe_b": [ScalingType.BlockWise1x16.value]},
            {"out_dtype": torch.float32},
            {"use_fast_accum": True},
            {"contraction_dim": [1]},
        ],
    )
    def test_rejects_out_of_contract_signature(self, override):
        if torch.version.hip is None:
            self.skipTest("ROCm-only gate")
        self.assertIsNone(
            mm._get_rocm_mxfp_v2_format(**self._candidate_args(**override))
        )

    def test_detects_fp8_operands(self):
        if torch.version.hip is None:
            self.skipTest("ROCm-only gate")
        args = self._candidate_args()
        args["mat_a"] = _FakeNode((64, 256), (256, 1), torch.float8_e4m3fn)
        args["mat_b"] = _FakeNode((256, 96), (1, 256), torch.float8_e4m3fn)
        self.assertEqual(mm._get_rocm_mxfp_v2_format(**args), "mxfp8")

    def test_tile_config_units_are_elements(self):
        from torch._inductor.kernel.vendored_templates.flydsl.kernels import (
            mxfp_gemm_derived,
        )

        mxfp4 = mxfp_gemm_derived(
            "mxfp4",
            block_m=64, block_n=64, block_k=256, stages=2, m_waves=1, n_waves=1
        )
        mxfp8 = mxfp_gemm_derived(
            "mxfp8",
            block_m=64,
            block_n=64,
            block_k=256,
            stages=2,
            m_waves=1,
            n_waves=1,
        )
        mxfp8_lds_scale = mxfp_gemm_derived(
            "mxfp8",
            block_m=64,
            block_n=64,
            block_k=256,
            stages=2,
            m_waves=1,
            n_waves=1,
            lds_scale_req=1,
        )
        # TILE_K is logical elements for both formats; only storage width changes.
        self.assertEqual(mxfp4.block_k_bytes, 128)
        self.assertEqual(mxfp8.block_k_bytes, 256)
        self.assertEqual(mxfp4.a_stage_bytes, 64 * 128)
        self.assertEqual(mxfp8.a_stage_bytes, 64 * 256)
        self.assertEqual(mxfp4.k_halves, mxfp8.k_halves)
        self.assertTrue(mxfp8_lds_scale.lds_scale)

    def test_cache_signature_includes_lds_scale(self):
        from torch._inductor.kernel.vendored_templates.flydsl.kernels import (
            MXFPGemmParams,
        )

        global_scale = MXFPGemmParams(
            mxfp_format="mxfp4",
            m=256,
            n=256,
            k=512,
            out_dtype="bfloat16",
            lds_scale=0,
        )
        lds_scale = MXFPGemmParams(
            mxfp_format="mxfp4",
            m=256,
            n=256,
            k=512,
            out_dtype="bfloat16",
            lds_scale=1,
        )
        mxfp8 = MXFPGemmParams(
            mxfp_format="mxfp8",
            m=256,
            n=256,
            k=512,
            out_dtype="bfloat16",
            lds_scale=0,
        )
        self.assertNotEqual(
            global_scale.__cache_signature__(), lds_scale.__cache_signature__()
        )
        self.assertNotEqual(
            global_scale.__cache_signature__(), mxfp8.__cache_signature__()
        )

    def test_unsupported_tile_k_is_rejected(self):
        from torch._inductor.kernel.vendored_templates.flydsl.kernels import (
            mxfp_gemm_derived,
        )

        with self.assertRaises(ValueError):
            mxfp_gemm_derived(
                "mxfp4",
                block_m=128, block_n=128, block_k=64, stages=2, m_waves=1, n_waves=1
            )


class TestFlyDSLMXFP4Device(TestCase):
    def _skip_unless_supported(self, device):
        if torch.version.hip is None:
            self.skipTest("FlyDSL MXFP4 template is ROCm-only")
        arch = torch.cuda.get_device_properties(device).gcnArchName.split(":", 1)[0]
        if arch != "gfx950":
            self.skipTest(f"MXFP4 scaled MFMA requires gfx950, got {arch}")
        if not flydsl_utils.runtime_available():
            self.skipTest("FlyDSL runtime unavailable")

    @parametrize(
        "shape,tile,out_dtype",
        [
            (
                (32, 32, 256),
                (16, 16, 128, 2, 1, 1, 0, 0),
                torch.bfloat16,
            ),  # scalar scale fallback
            (
                (32, 64, 1024),
                (32, 64, 512, 2, 1, 2, 0, 0),
                torch.bfloat16,
            ),  # packed-unit global scale path
            (
                (32, 64, 1024),
                (32, 64, 512, 2, 1, 2, 0, 1),
                torch.bfloat16,
            ),  # shared LDS-staged scale path
            (
                (128, 128, 512),
                (128, 128, 256, 2, 2, 2, 0, 1),
                torch.bfloat16,
            ),  # multi-wave LDS-staged scale path
            (
                (128, 128, 1024),
                (64, 64, 128, 4, 2, 2, 0, 0),
                torch.bfloat16,
            ),  # deep pipeline with FP4 DMA wait counts
            (
                (256, 256, 512),
                (256, 256, 256, 2, 4, 2, 0, 0),
                torch.float16,
            ),  # asymmetric waves and fp16 output
        ],
    )
    def test_mxfp4_tile_configs_match_reference(
        self, device, shape, tile, out_dtype
    ):
        self._skip_unless_supported(device)

        import flydsl.compiler as flyc

        from torch._inductor.kernel.vendored_templates.flydsl.kernels import (
            make_mxfp_scaled_mm_gfx950,
        )

        m, n, k = shape
        (
            block_m,
            block_n,
            block_k,
            stages,
            m_waves,
            n_waves,
            group_m,
            lds_scale,
        ) = tile
        a, scale_a, a_ref = make_mxfp4_operand(m, k, device)
        b, scale_b, b_ref = make_mxfp4_operand(n, k, device)
        out = torch.zeros(m, n, device=device, dtype=out_dtype)
        a_u8 = a.view(torch.uint8)
        b_u8 = b.view(torch.uint8)
        scale_a_u8 = scale_a.view(torch.uint8)
        scale_b_u8 = scale_b.view(torch.uint8)

        launcher = make_mxfp_scaled_mm_gfx950(
            mxfp_format="mxfp4",
            m=m,
            n=n,
            k=k,
            out_dtype="bfloat16" if out_dtype == torch.bfloat16 else "float16",
            block_m=block_m,
            block_n=block_n,
            block_k=block_k,
            stages=stages,
            m_waves=m_waves,
            n_waves=n_waves,
            group_m=group_m,
            lds_scale=lds_scale,
        )
        runtime_args = (a_u8, b_u8, scale_a_u8, scale_b_u8, out, 0)
        compiled = flyc.compile(
            launcher,
            *[
                flyc.from_torch_tensor(t).mark_layout_dynamic()
                for t in (a_u8, b_u8, scale_a_u8, scale_b_u8, out)
            ],
            0,
        )
        compiled(*runtime_args)
        torch.cuda.synchronize()

        reference = (a_ref @ b_ref.t()).to(out_dtype)
        rel_l2 = ((out.float() - reference.float()).norm() / reference.norm()).item()
        self.assertLess(rel_l2, 5e-3)

    @parametrize("out_dtype", [torch.bfloat16, torch.float16])
    def test_scaled_mm_v2_flydsl_matches_reference(self, device, out_dtype):
        self._skip_unless_supported(device)
        m, n, k = 256, 256, 512
        a, scale_a, a_ref = make_mxfp4_operand(m, k, device)
        b, scale_b, b_ref = make_mxfp4_operand(n, k, device)
        b_t = b.view(torch.uint8).t().view(torch.float4_e2m1fn_x2)

        with config.patch(
            {
                "max_autotune": True,
                "max_autotune_gemm_backends": "FLYDSL",
            }
        ):
            compiled = torch.compile(scaled_mm_mxfp4, dynamic=False)
            out, code = run_and_get_code(compiled, a, b_t, scale_a, scale_b, out_dtype)

        self.assertIn("gemm_mxfp_gfx950", "\n".join(code))
        # The tolerance covers FP32 accumulation-order and output-rounding
        # differences relative to the reference matmul.
        reference = a_ref @ b_ref.t()
        rel_l2 = ((out.float() - reference).norm() / reference.norm()).item()
        self.assertLess(rel_l2, 5e-3)

    def test_unsupported_signature_falls_back(self, device):
        self._skip_unless_supported(device)
        m, n, k = 64, 64, 128
        a = torch.randn(m, k, device=device).to(torch.float8_e4m3fn)
        b = torch.randn(n, k, device=device).to(torch.float8_e4m3fn).t()
        scale_a = torch.ones((), device=device)
        scale_b = torch.ones((), device=device)

        def tensorwise(a, b, scale_a, scale_b):
            return torch._scaled_mm_v2(
                a,
                b,
                [scale_a],
                [ScalingType.TensorWise.value],
                [SwizzleType.NO_SWIZZLE.value],
                [scale_b],
                [ScalingType.TensorWise.value],
                [SwizzleType.NO_SWIZZLE.value],
                None,
                torch.bfloat16,
            )

        # ATen has to stay in the backend list: this signature has no FlyDSL
        # choice by construction, and with FLYDSL alone there would be nothing
        # left to select rather than a fallback to exercise.
        with config.patch(
            {"max_autotune": True, "max_autotune_gemm_backends": "ATEN,FLYDSL"}
        ):
            _, code = run_and_get_code(
                torch.compile(tensorwise, dynamic=False), a, b, scale_a, scale_b
            )
        self.assertNotIn("gemm_mxfp_gfx950", "\n".join(code))


instantiate_parametrized_tests(TestFlyDSLMXFP4Metadata)
instantiate_device_type_tests(TestFlyDSLMXFP4Device, globals(), only_for="cuda")


if __name__ == "__main__":
    run_tests()
