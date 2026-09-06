# Owner(s): ["module: inductor"]

from types import SimpleNamespace
from unittest import mock

import torch
import torch.nn.functional as F
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


def _candidate_args(mxfp_format, **overrides):
    m, n, k = 64, 96, 256
    if mxfp_format == "mxfp4":
        dtype = torch.float4_e2m1fn_x2
        storage_k = k // 2
        contraction_dim = None
    elif mxfp_format == "mxfp8":
        dtype = torch.float8_e4m3fn
        storage_k = k
        contraction_dim = []
    else:
        raise AssertionError(f"unsupported MXFP format: {mxfp_format}")
    args = {
        "mat_a": _FakeNode((m, storage_k), (storage_k, 1), dtype),
        "mat_b": _FakeNode((storage_k, n), (1, storage_k), dtype),
        "scale_a": [
            _FakeNode((m, k // 32), (k // 32, 1), torch.float8_e8m0fnu)
        ],
        "recipe_a": [ScalingType.BlockWise1x32.value],
        "swizzle_a": [SwizzleType.NO_SWIZZLE.value],
        "scale_b": [
            _FakeNode((n, k // 32), (k // 32, 1), torch.float8_e8m0fnu)
        ],
        "recipe_b": [ScalingType.BlockWise1x32.value],
        "swizzle_b": [SwizzleType.NO_SWIZZLE.value],
        "bias": None,
        "out_dtype": torch.bfloat16,
        "contraction_dim": contraction_dim,
        "use_fast_accum": False,
    }
    args.update(overrides)
    return args


def _run_mxfp_tile(mxfp_format, shape, tile, out_dtype, a, b, scale_a, scale_b):
    import flydsl.compiler as flyc

    from torch._inductor.kernel.vendored_templates.flydsl.kernels import (
        make_mxfp_scaled_mm_gfx950,
    )

    m, n, k = shape
    block_m, block_n, block_k, stages, m_waves, n_waves, group_m = tile[:7]
    lds_scale = tile[7] if len(tile) == 8 else 0
    out = torch.zeros(m, n, device=a.device, dtype=out_dtype)
    a_u8 = a.view(torch.uint8)
    b_u8 = b.view(torch.uint8)
    scale_a_u8 = scale_a.view(torch.uint8)
    scale_b_u8 = scale_b.view(torch.uint8)
    launcher = make_mxfp_scaled_mm_gfx950(
        mxfp_format=mxfp_format,
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
    return out


def _mxfp8_reference(a, b, scale_a, scale_b, out_dtype):
    a_dequant = a.float() * scale_a.float().repeat_interleave(32, 1)
    b_dequant = b.float() * scale_b.float().repeat_interleave(32, 1)
    return (a_dequant @ b_dequant.t()).to(out_dtype)


class TestFlyDSLMXFPMetadata(TestCase):
    @parametrize(
        "mxfp_format,contraction_dim",
        [("mxfp4", None), ("mxfp8", None), ("mxfp8", [])],
    )
    def test_exact_v2_signature(self, mxfp_format, contraction_dim):
        with mock.patch.object(torch.version, "hip", "test"):
            self.assertEqual(
                mm._get_rocm_mxfp_v2_format(
                    **_candidate_args(
                        mxfp_format, contraction_dim=contraction_dim
                    )
                ),
                mxfp_format,
            )

    @parametrize(
        "mxfp_format,override",
        [
            ("mxfp4", {"swizzle_a": [SwizzleType.SWIZZLE_32_4_4.value]}),
            ("mxfp4", {"recipe_b": [ScalingType.BlockWise1x16.value]}),
            ("mxfp4", {"out_dtype": torch.float32}),
            ("mxfp4", {"use_fast_accum": True}),
            ("mxfp4", {"contraction_dim": [1]}),
            ("mxfp8", {"swizzle_a": [SwizzleType.SWIZZLE_32_4_4.value]}),
            ("mxfp8", {"use_fast_accum": True}),
        ],
    )
    def test_rejects_out_of_contract_signature(self, mxfp_format, override):
        with mock.patch.object(torch.version, "hip", "test"):
            self.assertIsNone(
                mm._get_rocm_mxfp_v2_format(
                    **_candidate_args(mxfp_format, **override)
                )
            )

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


class TestFlyDSLMXFP8Metadata(TestCase):
    def test_strict_baseline_layout(self):
        args = _candidate_args("mxfp8")
        a = args["mat_a"]
        b = args["mat_b"]
        scale_a = args["scale_a"]
        scale_b = args["scale_b"]
        layout = SimpleNamespace(
            size=[64, 96],
            stride=[96, 1],
            dtype=torch.bfloat16,
            device=torch.device("cuda", 0),
            offset=0,
        )

        with mock.patch.object(mm, "use_flydsl_gemm_template", return_value=True):
            configs = mm.get_flydsl_mxfp_template_kwargs(
                "mxfp8", layout, a, b, scale_a[0], scale_b[0]
            )
            # Autotuning is off by default, so the selector returns exactly one
            # config -- the tile it picks must divide 64x96x256 exactly, since
            # the kernel has no boundary predication.
            self.assertEqual(len(configs), 1)
            for gemm_config in configs:
                self.assertEqual(gemm_config["GEMM_M"], 64)
                self.assertEqual(gemm_config["GEMM_N"], 96)
                self.assertEqual(gemm_config["GEMM_K"], 256)
                self.assertEqual(gemm_config["OUT_DTYPE"], "bfloat16")
                self.assertEqual(64 % gemm_config["TILE_M"], 0)
                self.assertEqual(96 % gemm_config["TILE_N"], 0)
                self.assertEqual(256 % gemm_config["TILE_K"], 0)

            bad_b = _FakeNode((256, 96), (96, 1), torch.float8_e4m3fn)
            self.assertEqual(
                mm.get_flydsl_mxfp_template_kwargs(
                    "mxfp8", layout, a, bad_b, scale_a[0], scale_b[0]
                ),
                [],
            )

            bad_scale = _FakeNode((64, 8), (8, 1), torch.float8_e8m0fnu, offset=1)
            self.assertEqual(
                mm.get_flydsl_mxfp_template_kwargs(
                    "mxfp8", layout, a, b, bad_scale, scale_b[0]
                ),
                [],
            )

            bad_device_scale = _FakeNode(
                (64, 8),
                (8, 1),
                torch.float8_e8m0fnu,
                device=torch.device("cpu"),
            )
            self.assertEqual(
                mm.get_flydsl_mxfp_template_kwargs(
                    "mxfp8", layout, a, b, bad_device_scale, scale_b[0]
                ),
                [],
            )

            with mock.patch.object(
                mm,
                "is_unaligned",
                side_effect=lambda node: node is a,
            ):
                self.assertEqual(
                    mm.get_flydsl_mxfp_template_kwargs(
                        "mxfp8", layout, a, b, scale_a[0], scale_b[0]
                    ),
                    [],
                )

    def test_precompile_uses_flydsl_compile_only_contract(self):
        source = mm.flydsl_mxfp_scaled_mm_template.source

        self.assertIn('{"COMPILE_ONLY": "1"}', source)
        self.assertNotIn("FLYDSL_COMPILE_ONLY", source)


class _MXFPDeviceTest(TestCase):
    def _skip_unless_supported(self, device):
        if torch.version.hip is None:
            self.skipTest("requires ROCm")
        arch = torch.cuda.get_device_properties(device).gcnArchName.split(":", 1)[0]
        if arch != "gfx950":
            self.skipTest(f"requires gfx950, got {arch}")
        if not flydsl_utils.runtime_available():
            self.skipTest("FlyDSL runtime unavailable")


class TestFlyDSLMXFP8Device(_MXFPDeviceTest):
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
        self._skip_unless_supported(device)

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
        reference = _mxfp8_reference(a, b, scale_a, scale_b, out_dtype)
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
        self.assertIn("make_mxfp_scaled_mm_gfx950", code)
        self.assertIn("mat2.transpose(0, 1)", code)
        self.assertNotIn("extern_kernels._scaled_mm_v2(", code)

    # One entry per tiling feature the parameterized kernel exposes, so a
    # regression in LDS staging, register blocking or the staged pipeline shows
    # up as a numerical failure on the specific config that broke.
    @parametrize(
        "shape,tile,out_dtype",
        [
            # (m, n, k), (TILE_M, TILE_N, TILE_K, STAGES, M_WARPS, N_WARPS,
            # GROUP_M[, LDS_SCALE])
            (
                (64, 96, 256),
                (16, 16, 128, 2, 1, 1, 0),
                torch.bfloat16,
            ),  # minimal, 1 wave
            (
                (64, 64, 512),
                (64, 64, 128, 2, 1, 1, 0),
                torch.bfloat16,
            ),  # 4x4 register blocking
            (
                (128, 128, 512),
                (64, 64, 128, 2, 2, 2, 0),
                torch.bfloat16,
            ),  # 2x2 waves over LDS
            (
                (128, 128, 512),
                (64, 64, 256, 2, 2, 2, 0),
                torch.bfloat16,
            ),  # two MFMA steps / tile
            (
                (64, 64, 1024),
                (64, 64, 512, 2, 1, 1, 0),
                torch.bfloat16,
            ),  # four MFMA steps / tile
            (
                (128, 128, 512),
                (64, 64, 128, 3, 2, 2, 0),
                torch.bfloat16,
            ),  # odd stage count
            (
                (128, 128, 1024),
                (64, 64, 128, 4, 2, 2, 0),
                torch.bfloat16,
            ),  # 4-stage pipeline
            (
                (128, 128, 512),
                (128, 128, 128, 2, 4, 4, 0),
                torch.bfloat16,
            ),  # 1024 threads
            (
                (256, 256, 512),
                (128, 128, 128, 2, 2, 4, 0),
                torch.bfloat16,
            ),  # asymmetric waves
            (
                (1024, 256, 512),
                (128, 128, 128, 2, 2, 2, 4),
                torch.bfloat16,
            ),  # GROUP_M swizzle
            (
                (256, 256, 512),
                (256, 256, 128, 2, 2, 2, 0),
                torch.bfloat16,
            ),  # 8x8 deep blocking
            (
                (256, 256, 512),
                (256, 256, 128, 2, 2, 2, 0),
                torch.float16,
            ),  # fp16 direct store
            (
                (256, 128, 512),
                (256, 128, 128, 2, 2, 1, 0),
                torch.bfloat16,
            ),  # rectangular 8x8
            (
                (64, 64, 512),
                (64, 64, 256, 2, 1, 1, 0, 1),
                torch.bfloat16,
            ),  # shared LDS-staged scale path
            (
                (128, 128, 512),
                (128, 128, 256, 2, 2, 2, 0, 1),
                torch.bfloat16,
            ),  # multi-wave LDS-staged scale path
        ],
    )
    def test_mxfp8_tile_configs_match_reference(self, device, shape, tile, out_dtype):
        self._skip_unless_supported(device)
        m, n, k = shape
        a, b, scale_a, scale_b = self._make_inputs(m, n, k, device)
        out = _run_mxfp_tile(
            "mxfp8", shape, tile, out_dtype, a, b, scale_a, scale_b
        )
        reference = _mxfp8_reference(a, b, scale_a, scale_b, out_dtype)
        self.assertEqual(out, reference, rtol=2e-2, atol=5e-1)

    def test_scaled_mm_v2_flydsl_autotunes_multiple_configs(self, device):
        self._skip_unless_supported(device)

        from torch._inductor.heuristics.template import flydsl as flydsl_heuristics

        m, n, k = 128, 128, 512
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
                output_dtype=torch.bfloat16,
            )

        expected = fn(a, b, scale_a, scale_b)

        with config.patch(
            max_autotune_gemm=True,
            max_autotune_gemm_backends="FLYDSL",
            flydsl_enable_autotuning=True,
        ):
            # More than one tile divides this shape, so autotuning has a real
            # choice to make here.
            candidates = flydsl_heuristics.get_mxfp_gemm_configs_for_shape(
                "mxfp8", m, n, k, "bfloat16"
            )
            self.assertGreater(len(candidates), 1)

            torch._dynamo.reset()
            compiled = torch.compile(fn, backend="inductor", fullgraph=True)
            actual, (code,) = run_and_get_code(compiled, a, b, scale_a, scale_b)

        self.assertEqual(actual, expected, rtol=2e-2, atol=5e-1)
        self.assertIn("async_compile.flydsl", code)


class TestFlyDSLMXFP4Device(_MXFPDeviceTest):
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
        m, n, k = shape
        a, scale_a, a_ref = make_mxfp4_operand(m, k, device)
        b, scale_b, b_ref = make_mxfp4_operand(n, k, device)
        out = _run_mxfp_tile(
            "mxfp4", shape, tile, out_dtype, a, b, scale_a, scale_b
        )
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

        with config.patch(
            {"max_autotune": True, "max_autotune_gemm_backends": "ATEN,FLYDSL"}
        ):
            _, code = run_and_get_code(
                torch.compile(tensorwise, dynamic=False), a, b, scale_a, scale_b
            )
        self.assertNotIn("gemm_mxfp_gfx950", "\n".join(code))


instantiate_parametrized_tests(TestFlyDSLMXFPMetadata)
instantiate_device_type_tests(TestFlyDSLMXFP8Device, globals(), only_for="cuda")
instantiate_device_type_tests(TestFlyDSLMXFP4Device, globals(), only_for="cuda")


if __name__ == "__main__":
    run_tests()
