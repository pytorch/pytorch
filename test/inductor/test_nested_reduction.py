# Owner(s): ["module: inductor"]

"""End-to-end nested-reduction behavior and kernel-form tests."""

import re
from unittest.mock import patch

import torch
import torch._inductor.config as inductor_config
import torch.nn.functional as F
from torch._higher_order_ops.inline_asm_elementwise import inline_asm_elementwise
from torch._inductor import metrics
from torch._inductor.choices import InductorChoices
from torch._inductor.test_case import run_tests, TestCase
from torch._inductor.utils import fresh_inductor_cache, run_and_get_code
from torch._inductor.virtualized import V
from torch.testing import FileCheck
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    skipIfRocm,
    skipIfXpu,
)
from torch.testing._internal.inductor_utils import (
    get_func_call,
    get_kernel_launch,
    GPU_TYPE,
    HAS_GPU,
)


MXFP4_RECIP_UE8M0_ASM = (
    "{.reg .pred p_zero; .reg .s32 neg_exp; .reg .f32 neg_exp_f, result; "
    "setp.eq.u32 p_zero, $1, 0; sub.s32 neg_exp, 127, $1; "
    "cvt.rn.f32.s32 neg_exp_f, neg_exp; ex2.approx.f32 result, neg_exp_f; "
    "selp.f32 $0, 0f00000000, result, p_zero;}"
)

E2M1X2_PACK_ASM = (
    "{.reg .b8 t; cvt.rn.satfinite.e2m1x2.f32 t, $2, $1; cvt.u32.u8 $0, t;}"
)


def _choices_context(force_persistent: bool | None):
    import contextlib

    if force_persistent is None:
        return contextlib.nullcontext()

    class _Choices(InductorChoices):
        @staticmethod
        def should_use_cooperative_reduction(*args, **kwargs):
            return False

        @staticmethod
        def should_use_persistent_reduction(*args, **kwargs):
            return force_persistent

    return V.set_choices_handler(_Choices())


class TestBase(TestCase):
    force_persistent_outer_reduction: bool | None = None

    def setUp(self):
        super().setUp()
        metrics.reset()
        torch._dynamo.utils.clear_compilation_metrics()
        self._nested_reduction_ctx = inductor_config.patch(
            {
                "split_reductions": False,
                "triton.nested_reduction": True,
                "loop_ordering_after_fusion": True,
            }
        )
        self._nested_reduction_ctx.__enter__()
        self._choices_ctx = _choices_context(self.force_persistent_outer_reduction)
        self._choices_ctx.__enter__()

    def tearDown(self):
        self._choices_ctx.__exit__(None, None, None)
        self._nested_reduction_ctx.__exit__(None, None, None)
        super().tearDown()

    def check_numeric(self, f, args, tol=1e-2):
        ref = f(*args)
        act = torch.compile(f)(*args)
        self.assertEqual(act, ref, atol=tol, rtol=tol)

    def get_unnested_reference(self, f, args, **compile_kwargs):
        with inductor_config.patch("triton.nested_reduction", False):
            ref = torch.compile(f, **compile_kwargs)(*args)
        metrics.reset()
        torch._dynamo.reset()
        return ref

    def check_nested_matches_unnested(self, f, args, tol=1e-2):
        ref = self.get_unnested_reference(f, args)
        act = torch.compile(f)(*args)
        self.assertEqual(act, ref, atol=tol, rtol=tol)

    def _check_looped_internal_source(self, f, shape, expected_passes):
        if self.force_persistent_outer_reduction is not False:
            self.skipTest("requires a looped reduction")

        x = torch.randn(*shape, device=GPU_TYPE, dtype=torch.bfloat16)
        expected = self.get_unnested_reference(f, (x,))
        actual, sources = run_and_get_code(torch.compile(f), x)
        self.assertEqual(actual, expected, atol=1e-2, rtol=1e-2)
        self.check_fusion()
        FileCheck().check_count(
            "for r0_offset in tl.range", expected_passes, exactly=True
        ).run("\n".join(sources))

    def check_fusion(self, expected_kernels=1):
        self.assertEqual(metrics.codegen_nested_reduction, 1)
        if expected_kernels is not None:
            self.assertEqual(metrics.generated_kernel_count, expected_kernels)

    def check_no_fusion(self):
        self.assertEqual(metrics.codegen_nested_reduction, 0)

    def check_non_leaf_epilogue_fallback(self):
        self.assertGreater(metrics.generated_kernel_count, 1)
        self.assertGreater(
            metrics.generated_kernel_count,
            metrics.codegen_nested_reduction,
        )


def _rmsnorm(x_flat):
    return x_flat / torch.sqrt(torch.mean(x_flat * x_flat, dim=-1, keepdim=True) + 1e-6)


def _layernorm(x_flat):
    mean = x_flat.mean(dim=-1, keepdim=True)
    var = x_flat.var(dim=-1, keepdim=True, correction=0)
    return (x_flat - mean) / torch.sqrt(var + 1e-6)


def _mxfp6_pack_four_to_three(values, realize=True):
    if realize:
        values = torch.ops._inductor_test.realize(values)
    values = values.view(*values.shape[:-1], values.shape[-1] // 4, 4)
    low = values[..., 0] | ((values[..., 1] & 0x03) << 6)
    middle = ((values[..., 1] >> 2) & 0x0F) | ((values[..., 2] & 0x0F) << 4)
    high = ((values[..., 2] >> 4) & 0x03) | (values[..., 3] << 2)
    if realize:
        low = torch.ops._inductor_test.realize(low)
        middle = torch.ops._inductor_test.realize(middle)
        high = torch.ops._inductor_test.realize(high)
    return torch.stack((low, middle, high), dim=-1).to(torch.uint8)


def _mxfp6_four_to_three_quantize(x, group_size=32):
    B, D = x.shape
    x = torch.nn.functional.silu(x) * 1.125
    xg = x.view(B, D // group_size, group_size).float()
    scale = xg.abs().amax(dim=-1).clamp(min=1e-12) / 7.5
    values = (xg / scale.unsqueeze(-1)).round().to(torch.int32) & 0x3F
    return _mxfp6_pack_four_to_three(values).view(B, D // 4, 3), scale


def _rmsnorm_factor4_three_output_epilogue(x, weight, group_size=32):
    B, D = x.shape
    normalized = torch.nn.functional.rms_norm(x, (D,), weight)
    groups = normalized.view(B, D // group_size, group_size)
    scale = groups.abs().amax(dim=-1)
    lanes = groups.view(B, D // group_size, group_size // 4, 4)
    outputs = tuple(
        torch.ops._inductor_test.realize(
            (lane + 1) * lanes[..., lane] / scale.unsqueeze(-1)
        )
        for lane in range(3)
    )
    return torch.stack(outputs, dim=-1), scale


def _mxfp6_internal_source_full_resolution_fork(x, group_size=32):
    B, D = x.shape
    xg = x.view(B, D // group_size, group_size).float()
    scale = xg.abs().amax(dim=-1).clamp_min(1e-6) / 7.5
    scaled = torch.ops._inductor_test.realize(xg / scale.unsqueeze(-1))
    values = torch.ops._inductor_test.realize(scaled.round().to(torch.int32) & 0x3F)
    sibling = torch.ops._inductor_test.realize(scaled + 1)
    return _mxfp6_pack_four_to_three(values), scale, sibling


def _mxfp6_preshuffled_quantize(x, shifted=False):
    B, D = x.shape
    G = 32
    scale_group = 4
    subs = 3
    row_tiles = B // (scale_group * 32)
    k_tiles = D // (2 * G * subs)

    x = torch.nn.functional.silu(x) * 1.125
    blocks = x.view(row_tiles, scale_group, 32, k_tiles, subs, 2, G)
    blocks = blocks.permute(0, 3, 5, 2, 4, 1, 6).reshape(
        row_tiles, k_tiles, 64, subs, scale_group, G
    )
    blocks = blocks.float()
    max_abs = blocks.abs().amax(dim=-1)
    scale_exponent = torch.ceil(torch.log2((max_abs / 7.5).clamp(min=2.0**-127)))
    scale_exponent = torch.where(
        max_abs == 0, torch.zeros_like(scale_exponent), scale_exponent
    ).clamp(min=-127.0, max=127.0)
    values = _float_to_mxfp6_e2m3(blocks / torch.pow(2.0, scale_exponent).unsqueeze(-1))
    packed = _mxfp6_pack_four_to_three(values.to(torch.int32) & 0x3F, realize=False)
    packed = packed.reshape(row_tiles, k_tiles, 2, 32, subs, scale_group, G // 4, 3)
    if shifted:
        packed = torch.roll(packed, 1, -2)
    packed = packed.permute(0, 5, 3, 1, 4, 2, 6, 7)
    return packed.reshape(B, D * 3 // 4), scale_exponent.reshape(-1)


def _float_to_mxfp6_e2m3(x):
    sign = (x < 0).to(torch.int32)
    absolute = torch.clamp(torch.abs(x), max=7.5)
    subnormal_mantissa = torch.round(absolute * 8.0).to(torch.int32)
    subnormal_bits = (sign << 5) | torch.where(
        subnormal_mantissa >= 8,
        torch.full_like(subnormal_mantissa, 1 << 3),
        torch.clamp(subnormal_mantissa, min=0),
    )
    exponent = torch.where(
        absolute < 2.0,
        torch.ones_like(absolute),
        torch.where(
            absolute < 4.0,
            torch.full_like(absolute, 2.0),
            torch.full_like(absolute, 3.0),
        ),
    )
    fraction = absolute / torch.pow(2.0, exponent - 1.0) - 1.0
    mantissa = torch.round(fraction * 8.0).to(torch.int32)
    exponent = exponent.to(torch.int32)
    carry = mantissa >= 8
    exponent = torch.where(carry, exponent + 1, exponent)
    mantissa = torch.where(carry, torch.zeros_like(mantissa), mantissa)
    overflow = exponent > 3
    exponent = torch.where(overflow, torch.full_like(exponent, 3), exponent)
    mantissa = torch.where(overflow, torch.full_like(mantissa, 7), mantissa)
    normal_bits = (sign << 5) | (exponent << 3) | mantissa
    bits = torch.where(absolute < 1.0, subnormal_bits, normal_bits)
    return torch.where(absolute == 0, torch.zeros_like(bits), bits).to(torch.uint8)


def _swizzle_scale(scale):
    rows, cols = scale.shape
    blocks = scale.view(rows // 128, 128, cols // 4, 4).permute(0, 2, 1, 3)
    return blocks.reshape(-1, 4, 32, 4).transpose(1, 2).reshape(rows, cols)


def _mxfp6_pack_scale_swizzle(x, group_size=32, shifted=False):
    packed, scale = _mxfp6_four_to_three_quantize(x, group_size)
    if shifted:
        scale = torch.roll(scale, 1, -1)
    return packed, _swizzle_scale(scale)


def _rmsnorm_block_scale_swizzle(x, weight, G):
    B, D = x.shape
    x = F.rms_norm(x, (D,), weight)
    x_groups = x.view(B, D // G, G)
    amax = x_groups.abs().amax(dim=-1)
    scale = (amax / 448.0).clamp(min=1e-12)
    payload = (x_groups / scale.unsqueeze(-1)).to(torch.float16)
    return payload.view(B, D).float(), _swizzle_scale(scale)


def _rmsnorm_mxfp8_scale_swizzle(x, weight, G):
    B, D = x.shape
    x = F.rms_norm(x, (D,), weight)
    x_groups = x.view(B, D // G, G)
    amax = x_groups.abs().float().amax(dim=-1)
    scale = (amax / 448.0).clamp_min(torch.finfo(torch.float32).tiny)
    if scale.device.type == "cuda":
        scale_u8 = inline_asm_elementwise(
            scale,
            asm_str="cvt.rp.satfinite.ue8m0x2.f32 $0, 0.0, $1;",
            constraints="=h,r",
            dtype=torch.uint16,
        ).to(torch.uint8)
    else:
        scale_bits = scale.view(torch.int32)
        biased_exp = (scale_bits >> 23) & 0xFF
        mantissa = scale_bits & 0x7FFFFF
        scale_u8 = torch.clamp(
            biased_exp + (mantissa != 0).to(torch.int32), max=254
        ).to(torch.uint8)
    scale_f32 = torch.ldexp(
        torch.ones_like(scale, dtype=torch.float32),
        scale_u8.to(torch.int32) - 127,
    )
    payload = (
        (x_groups.float() / scale_f32.unsqueeze(-1))
        .clamp(min=-448.0, max=448.0)
        .to(torch.float8_e4m3fn)
    )
    return payload.view(B, D), _swizzle_scale(scale_u8)


@instantiate_parametrized_tests
class _NestedReductionBase:
    """Tests for fusing dependent cross-axis reductions into a single kernel."""

    # ---- Small dim in X falls back ----

    def _weighted_norm_reduce_k(self, norm, reduce_fn, B, K, D):
        rfn = {
            "sum": torch.Tensor.sum,
            "amax": torch.Tensor.amax,
            "amin": torch.Tensor.amin,
            "prod": torch.Tensor.prod,
        }[reduce_fn]

        def f(x, w):
            x_normed = norm(x.reshape(x.shape[0] * K, D)).reshape(x.shape)
            return rfn(w[:, :, None] * x_normed, dim=1)

        x = torch.randn(B, K, D, device=GPU_TYPE)
        w = torch.randn(B, K, device=GPU_TYPE)
        self.check_numeric(f, (x, w))
        self.check_no_fusion()

    @parametrize("B", [32, 256])
    @parametrize("K", [16, 32])
    def test_rmsnorm_weighted_sum(self, B, K):
        self._weighted_norm_reduce_k(_rmsnorm, "sum", B, K, 4096)

    @parametrize("K", [16, 32])
    def test_rmsnorm_weighted_max(self, K):
        self._weighted_norm_reduce_k(_rmsnorm, "amax", 64, K, 4096)

    @parametrize("reduce_fn", ["sum", "amax", "amin"])
    def test_rmsnorm_weighted_reduce_B1(self, reduce_fn):
        """B=1 flattened small_dim_in_x still falls back."""
        self._weighted_norm_reduce_k(_rmsnorm, reduce_fn, 1, 16, 1024)

    def test_layernorm_weighted_sum(self):
        self._weighted_norm_reduce_k(_layernorm, "sum", 64, 16, 4096)

    def test_layernorm_weighted_sum_B1(self):
        self._weighted_norm_reduce_k(_layernorm, "sum", 1, 16, 1024)

    def test_fullres_prologue_small_dim_in_x_loop_order(self):
        """Remap full-res prologue from physical [B*K, D] to logical [B, K, D]."""

        B, K, D = 16, 16, 1024

        def f(x, w, bias):
            x_flat = x.reshape(B * K, D)
            rms = torch.sqrt(torch.mean(x_flat * x_flat, dim=-1, keepdim=True) + 1e-6)
            y = torch.ops._inductor_test.realize(
                torch.relu((x_flat / rms).reshape(B, K, D) + bias[:, None, :])
            )
            return y, (w[:, :, None] * y).sum(dim=1)

        x = torch.randn(B, K, D, device=GPU_TYPE)
        w = torch.randn(B, K, device=GPU_TYPE)
        bias = torch.randn(B, D, device=GPU_TYPE)
        self.check_numeric(f, (x, w, bias))
        self.check_no_fusion()

    # ---- Small dim in R: norm + block reduce ----

    def _norm_block_reduce(self, norm, reduce_fn, B, D, G):
        rfn = {
            "sum": torch.Tensor.sum,
            "amax": torch.Tensor.amax,
            "amin": torch.Tensor.amin,
            "prod": torch.Tensor.prod,
        }[reduce_fn]

        def f(x):
            x_normed = norm(x)
            grouped = x_normed.reshape(x.shape[0], x.shape[1] // G, G)
            if reduce_fn == "amax":
                # Block scale amax is max(abs(x)); min/max tests cover signed variants.
                return grouped.abs().amax(dim=-1)
            return rfn(grouped, dim=-1)

        x = torch.randn(B, D, device=GPU_TYPE)
        self.check_numeric(f, (x,))
        self.check_fusion()

    @parametrize(
        "B,D,G",
        [
            (32, 4096, 16),
            (256, 4096, 32),
            (4, 384, 128),
        ],
    )
    def test_layernorm_block_amax(self, B, D, G):
        self._norm_block_reduce(_layernorm, "amax", B, D, G)

    def test_nested_reduction_skips_benchmark_fusion(self):
        B, D, G = 32, 4096, 16

        def f(x):
            x = _layernorm(x)
            return x.reshape(B, D // G, G).abs().amax(dim=-1)

        x = torch.randn(B, D, device=GPU_TYPE)
        ref = f(x)
        with inductor_config.patch("benchmark_fusion", True):
            act = torch.compile(f)(x)
        self.assertEqual(act, ref)
        self.check_fusion(expected_kernels=None)

    @parametrize("G", [8, 16])
    def test_rmsnorm_block_amax(self, G):
        self._norm_block_reduce(_rmsnorm, "amax", 128, 8192, G)

    def test_multiple_parent_reductions_block_amax(self):
        B, D, G = 32, 4096, 16

        def f(x):
            row_sum = x.sum(dim=-1, keepdim=True)
            row_square_sum = (x * x).sum(dim=-1, keepdim=True)
            normalized = (x - row_sum / D) * torch.rsqrt(row_square_sum / D + 1e-6)
            block_amax = normalized.reshape(B, D // G, G).abs().amax(dim=-1)
            return block_amax, row_sum, row_square_sum

        x = torch.randn(B, D, device=GPU_TYPE)
        self.check_numeric(f, (x,))
        self.check_fusion()

    @parametrize("reduce_fn", ["sum", "amin"])
    def test_layernorm_block_reduce(self, reduce_fn):
        self._norm_block_reduce(_layernorm, reduce_fn, 64, 4096, 16)

    def test_layernorm_block_prod(self):
        B, D, G = 64, 4096, 8

        def f(x):
            x_normed = torch.tanh(_layernorm(x))
            return x_normed.reshape(B, D // G, G).prod(dim=-1)

        x = torch.randn(B, D, device=GPU_TYPE)
        ref = f(x.cpu()).to(GPU_TYPE)
        act = torch.compile(f)(x)
        self.assertEqual(act, ref, atol=1e-3, rtol=1e-3)
        self.check_fusion()

    def test_layernorm_block_amax_group_size_512(self):
        self._norm_block_reduce(_layernorm, "amax", 32, 4096, 512)

    def test_layernorm_block_amax_non_power_of_2_groups(self):
        """D/G need not be a power of 2."""
        self._norm_block_reduce(_layernorm, "amax", 16, 6144, 128)

    # ---- Epilogue dtype conversion ----

    def test_weighted_rmsnorm_reduce_k_bf16_epilogue(self):
        def f(x, w):
            x_normed = _rmsnorm(x.reshape(x.shape[0] * 16, 4096)).reshape(x.shape)
            return (w[:, :, None] * x_normed).sum(dim=1).to(torch.bfloat16)

        x = torch.randn(64, 16, 4096, device=GPU_TYPE)
        w = torch.randn(64, 16, device=GPU_TYPE)
        self.check_numeric(f, (x, w))
        self.check_no_fusion()

    def test_layernorm_block_amax_bf16_epilogue(self):
        def f(x):
            return (
                _layernorm(x)
                .reshape(x.shape[0], -1, 16)
                .abs()
                .amax(dim=-1)
                .to(torch.bfloat16)
            )

        x = torch.randn(64, 4096, device=GPU_TYPE)
        self.check_numeric(f, (x,))
        self.check_fusion()

    # ---- Downstream pointwise fusion ----

    def test_weighted_rmsnorm_reduce_k_pointwise_epilogue(self):
        """Weighted small-dim-in-X reduction with a pointwise epilogue falls back."""

        def f(x, w, scale, bias):
            x_normed = _rmsnorm(x.reshape(x.shape[0] * 16, 4096)).reshape(x.shape)
            out = (w[:, :, None] * x_normed).sum(dim=1)
            return out * scale + bias

        x = torch.randn(64, 16, 4096, device=GPU_TYPE)
        w = torch.randn(64, 16, device=GPU_TYPE)
        scale = torch.randn(64, 4096, device=GPU_TYPE)
        bias = torch.randn(64, 4096, device=GPU_TYPE)
        self.check_numeric(f, (x, w, scale, bias))
        self.check_no_fusion()

    def test_layernorm_block_amax_reduced_pointwise_epilogue(self):
        """Fuse out * scale + bias after reduced-output block amax."""

        def f(x, scale, bias):
            out = (
                _layernorm(x)
                .reshape(x.shape[0], x.shape[1] // 16, 16)
                .abs()
                .amax(dim=-1)
            )
            return out * scale + bias

        x = torch.randn(64, 4096, device=GPU_TYPE)
        scale = torch.randn(64, 256, device=GPU_TYPE)
        bias = torch.randn(64, 256, device=GPU_TYPE)
        self.check_numeric(f, (x, scale, bias))
        self.check_fusion()

    def test_rmsnorm_block_scale_swizzle(self):
        B, D, G = 128, 4096, 32
        x = torch.randn(B, D, device=GPU_TYPE)
        weight = torch.randn(D, device=GPU_TYPE)

        def f(x, weight):
            return _rmsnorm_block_scale_swizzle(x, weight, G)

        ref_payload, ref_scale = f(x, weight)
        payload, scale = torch.compile(f)(x, weight)
        self.assertEqual(payload, ref_payload, atol=1e-2, rtol=1e-2)
        self.assertEqual(scale, ref_scale, atol=1e-6, rtol=1e-6)
        self.check_fusion()

    # ---- Edge cases ----

    @parametrize(
        "B,D,G",
        [(256, 4096, 16), (128, 4096, 32), (256, 8192, 32)],
    )
    def test_edge_B_equals_D_over_G(self, B, D, G):
        """When B == D/G, size-based matching is ambiguous."""
        self._norm_block_reduce(_layernorm, "amax", B, D, G)

    @parametrize("BK", [16, 32])
    def test_edge_B_equals_K(self, BK):
        """When B == K, size-based matching is ambiguous."""
        self._weighted_norm_reduce_k(_rmsnorm, "sum", BK, BK, 4096)

    # ---- Dynamic shapes ----

    @parametrize("dynamic", [False, True])
    def test_shapes_weighted_rmsnorm_reduce_k(self, dynamic):
        """Dynamic small-dim-in-x falls back."""
        K = 16

        def f(x, w):
            B, D = x.shape[0], x.shape[2]
            x_flat = x.reshape(B * K, D)
            rms = torch.sqrt(torch.mean(x_flat * x_flat, dim=-1, keepdim=True) + 1e-6)
            x_normed = (x_flat / rms).reshape(B, K, D)
            return (w[:, :, None] * x_normed).sum(dim=1)

        compiled = torch.compile(f, dynamic=dynamic)
        for B, D in [(32, 1024), (64, 2048), (128, 4096)] if dynamic else [(32, 4096)]:
            x = torch.randn(B, K, D, device=GPU_TYPE)
            w = torch.randn(B, K, device=GPU_TYPE)
            if dynamic:
                torch._dynamo.mark_static(x, 1)
                torch._dynamo.mark_static(w, 1)
            ref = f(x, w)
            act = compiled(x, w)
            self.assertEqual(act, ref, atol=1e-2, rtol=1e-2)
        self.check_no_fusion()

    @parametrize("dynamic", [False, True])
    def test_shapes_layernorm_block_amax(self, dynamic):
        def f(x):
            return _layernorm(x).reshape(x.shape[0], -1, 16).abs().amax(dim=-1)

        compiled = torch.compile(f, dynamic=dynamic)
        for B in [32, 64, 256] if dynamic else [32]:
            x = torch.randn(B, 4096, device=GPU_TYPE)
            self.assertEqual(compiled(x), f(x), atol=1e-2, rtol=1e-2)
        self.check_fusion()

    def test_dynamic_shapes_varying_batch_and_dim(self):
        """Dynamic shapes: vary both B and D at runtime."""

        def f(x, weight):
            x = F.rms_norm(x, (x.shape[-1],), weight)
            B, D = x.shape
            return x.view(B, D // 128, 128).abs().amax(dim=-1)

        compiled = torch.compile(f, dynamic=True)
        for B, D in [(4, 512), (8, 1024), (16, 2048)]:
            x = torch.randn(B, D, device=GPU_TYPE)
            w = torch.randn(D, device=GPU_TYPE)
            ref = f(x, w)
            act = compiled(x, w)
            self.assertEqual(act, ref, atol=1e-2, rtol=1e-2)
        self.check_fusion()

    @parametrize("dynamic", ["mark", "automatic"])
    def test_dynamic_materialized_parent_output(self, dynamic):
        D, G = 1024, 32

        def f(x):
            normalized = _rmsnorm(x)
            groups = normalized.view(x.shape[0], D // G, G)
            return normalized, groups.abs().amax(dim=-1)

        compiled = torch.compile(f, fullgraph=True)
        first = torch.randn(32, D, device=GPU_TYPE)
        if dynamic == "mark":
            torch._dynamo.mark_dynamic(first, 0)
        self.assertEqual(compiled(first), f(first), atol=1e-3, rtol=1e-3)
        second = torch.randn(48, D, device=GPU_TYPE)
        self.assertEqual(compiled(second), f(second), atol=1e-3, rtol=1e-3)
        expected_kernels = 1 if dynamic == "mark" else 2
        self.assertEqual(metrics.codegen_nested_reduction, expected_kernels)
        self.assertEqual(metrics.generated_kernel_count, expected_kernels)

    # ---- Producer-consumer: node2 reads node1's materialized output ----
    # Instead of node1 and node2 sharing a common input, node2 reads
    # node1's output. This triggers the producer-consumer path in
    # NestedReduction.can_fuse.

    @parametrize("B", [1, 128])
    def test_producer_consumer_rmsnorm_amax(self, B):
        """RMS norm materializes output, amax reads it."""
        D, G = 4096, 16

        def f(x, weight):
            x = F.rms_norm(x, (D,), weight)
            return x.view(B, D // G, G).abs().amax(dim=-1)

        x = torch.randn(B, D, device=GPU_TYPE)
        w = torch.randn(D, device=GPU_TYPE)
        self.check_numeric(f, (x, w))
        self.check_fusion()

    def test_grouped_reduction_input_broadcast_parent_axis(self):
        B, D, G = 16, 1024, 16

        def f(x):
            s = x.sum(dim=-1, keepdim=True)
            y = torch.ops._inductor_test.realize(s.expand_as(x))
            return y.reshape(B, D // G, G).amax(dim=-1)

        x = torch.randn(B, D, device=GPU_TYPE)
        self.check_numeric(f, (x,))
        self.check_fusion()

    @parametrize("pointwise_kind", ["full", "row_broadcast", "col_broadcast"])
    @parametrize("epilogue_resolution", ["reduced", "full"])
    def test_reduction_fusion_pointwise_prologue_epilogue(
        self,
        pointwise_kind,
        epilogue_resolution,
    ):
        B, D, G = 128, 4096, 128

        def f(x, weight, prologue_extra, epilogue_extra):
            x = F.rms_norm(x, (D,), weight)
            x = x.view(B, D // G, G)
            if pointwise_kind == "full":
                prologue_extra = prologue_extra.view(B, D // G, G)
            elif pointwise_kind == "row_broadcast":
                prologue_extra = prologue_extra[:, :, None]
            else:
                prologue_extra = prologue_extra.view(D // G, G)
            x = torch.ops._inductor_test.realize(x + prologue_extra)
            out = x.abs().amax(dim=-1)
            out = out + epilogue_extra
            if epilogue_resolution == "reduced":
                return out
            return (x / (out.abs() + 1e-6)[:, :, None]).view(B, D)

        x = torch.randn(B, D, device=GPU_TYPE)
        w = torch.randn(D, device=GPU_TYPE)
        prologue_extra_shape = {
            "full": (B, D),
            "row_broadcast": (B, 1),
            "col_broadcast": (D,),
        }[pointwise_kind]
        epilogue_extra_shape = {
            "full": (B, D // G),
            "row_broadcast": (B, 1),
            "col_broadcast": (D // G,),
        }[pointwise_kind]
        prologue_extra = torch.randn(prologue_extra_shape, device=GPU_TYPE)
        epilogue_extra = torch.randn(epilogue_extra_shape, device=GPU_TYPE)
        self.check_numeric(f, (x, w, prologue_extra, epilogue_extra))
        self.check_fusion()

    def test_reduced_resolution_pointwise_prologue(self):
        from torch._inductor.scheduler import FusedNestedReductions

        B, D, G = 128, 4096, 128

        def f(x, group_extra, epilogue_extra):
            sums = (x * x).sum(dim=-1, keepdim=True)
            inv = torch.rsqrt(sums / D + 1e-6)
            group_extra = torch.ops._inductor_test.realize(group_extra + sums)
            x = (x * inv).view(B, D // G, G)
            out = (x + group_extra[:, :, None]).abs().amax(dim=-1)
            return out + epilogue_extra

        x = torch.randn(B, D, device=GPU_TYPE)
        group_extra = torch.randn(B, D // G, device=GPU_TYPE)
        epilogue_extra = torch.randn(B, D // G, device=GPU_TYPE)
        saw_reduced_prologue = False

        def check_reduction_fusion(nodes):
            nonlocal saw_reduced_prologue
            fused_nodes = [n for n in nodes if isinstance(n, FusedNestedReductions)]
            self.assertEqual(len(fused_nodes), 1)
            node2_nodes = list(fused_nodes[0].node2.get_nodes())
            reductions = [sn for sn in node2_nodes if sn.is_reduction()]
            self.assertEqual(len(reductions), 1)
            reduction = reductions[0]
            reduction_names = reduction.get_operation_names()
            _, (reduced_numel, _) = reduction.group
            for sn in node2_nodes:
                if sn.is_reduction():
                    continue
                is_prologue = bool(sn.get_operation_names() & reduction.ancestors)
                is_epilogue = bool(reduction_names & sn.ancestors)
                self.assertTrue(is_prologue or is_epilogue)
                if is_prologue:
                    _, (sn_numel, _) = sn.group
                    saw_reduced_prologue |= sn_numel == reduced_numel
            return nodes

        with inductor_config.patch(
            _post_fusion_custom_pass=check_reduction_fusion,
            fx_graph_cache=False,
        ):
            self.check_numeric(f, (x, group_extra, epilogue_extra))
        self.assertTrue(saw_reduced_prologue)
        self.check_fusion()

    # ---- Exotic indexing ----

    def test_transposed_input(self):
        """Non-contiguous (transposed) input - numerics must be correct."""

        def f(x):
            x = x.t()
            rms = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + 1e-6)
            x_norm = x / rms
            return x_norm.reshape(x.shape[0], -1, 16).abs().amax(dim=-1)

        x = torch.randn(4096, 64, device=GPU_TYPE)
        self.check_numeric(f, (x,))

    def test_strided_slice_input(self):
        """Stride-2 slice input - numerics must be correct."""

        def f(x):
            x = x[:, ::2]
            rms = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + 1e-6)
            x_norm = x / rms
            return x_norm.reshape(x.shape[0], -1, 16).abs().amax(dim=-1)

        x = torch.randn(32, 4096, device=GPU_TYPE)
        self.check_numeric(f, (x,))

    def test_multi_op_prologue_and_epilogue(self):
        """Prologue does mul+add+relu, epilogue does log1p+clamp."""
        B, D, G = 64, 4096, 128

        def f(x, weight, bias, scale):
            x = F.rms_norm(x, (D,), weight)
            x_scaled = torch.ops._inductor_test.realize(torch.relu(x * scale + bias))
            amax = x_scaled.view(B, D // G, G).abs().amax(dim=-1)
            return torch.clamp(torch.log1p(amax), min=0.0, max=10.0)

        x = torch.randn(B, D, device=GPU_TYPE)
        w = torch.randn(D, device=GPU_TYPE)
        bias = torch.randn(D, device=GPU_TYPE)
        scale = torch.randn(D, device=GPU_TYPE)
        self.check_numeric(f, (x, w, bias, scale))
        self.check_fusion()

    @inductor_config.patch(emulate_precision_casts=True)
    def test_fullres_epilogue_with_multiple_outputs(self):
        """Full-res epilogue producing both converted output and scale."""
        B, D, G = 64, 4096, 128
        qmax = 448.0

        def f(x, weight):
            x = F.rms_norm(x, (D,), weight)
            x_groups = x.view(B, D // G, G)
            amax = x_groups.abs().amax(dim=-1)
            scale = (amax / qmax).clamp(min=1e-12)
            x_quant = (x_groups / scale.unsqueeze(-1)).to(torch.float16)
            return x_quant.view(B, D).float(), scale

        x = torch.randn(B, D, device=GPU_TYPE)
        w = torch.randn(D, device=GPU_TYPE)
        self.check_nested_matches_unnested(f, (x, w))
        self.check_fusion()

    def test_grouped_reduction_with_weight_mul(self):
        """Grouped reduction input involves element-wise weight multiply."""
        B, D, G = 128, 4096, 32

        def f(x, weight, group_weight):
            x = F.rms_norm(x, (D,), weight)
            # Weight multiply before grouped reduction
            weighted = x * group_weight
            return weighted.view(B, D // G, G).abs().amax(dim=-1)

        x = torch.randn(B, D, device=GPU_TYPE)
        w = torch.randn(D, device=GPU_TYPE)
        gw = torch.randn(D, device=GPU_TYPE)
        self.check_numeric(f, (x, w, gw))
        self.check_fusion()

    # ---- Producer-consumer ----

    @inductor_config.patch(emulate_precision_casts=True)
    def test_producer_consumer_rmsnorm_scale(self):
        """RMS norm + amax + converted scale epilogue."""
        B, D, G = 128, 4096, 16

        def f(x, weight):
            x = F.rms_norm(x, (D,), weight)
            x = x.view(B, D // G, G)
            amax = x.abs().amax(dim=-1)
            scale = (amax / 448.0).clamp(min=1e-12).to(torch.float16)
            return scale.float()

        x = torch.randn(B, D, device=GPU_TYPE)
        w = torch.randn(D, device=GPU_TYPE)
        self.check_numeric(f, (x, w), tol=0.01)
        self.check_fusion()

    @inductor_config.patch(emulate_precision_casts=True)
    @parametrize("B", [128, 1])
    def test_producer_consumer_rmsnorm_quant(self, B):
        """RMS norm + amax + scale + full-res convert epilogue."""
        D, G = 4096, 128
        qmax = 448.0

        def f(x, weight):
            x = F.rms_norm(x, (D,), weight)
            x_groups = x.view(B, D // G, G)
            amax = x_groups.abs().amax(dim=-1)
            scale = (amax / qmax).clamp(min=1e-12)
            x_quant = (x_groups / scale.unsqueeze(-1)).to(torch.float16)
            return x_quant.view(B, D).float(), scale

        x = torch.randn(B, D, device=GPU_TYPE)
        w = torch.randn(D, device=GPU_TYPE)
        self.check_nested_matches_unnested(f, (x, w))
        self.check_fusion()

    @inductor_config.patch(emulate_precision_casts=True)
    @parametrize(
        "weight_layout",
        [
            "scalar",
            "scalar_1d",
            "feature",
            "singleton",
            "singleton_batch",
            "batch",
            "full",
            "leading_singleton",
            "leading_singleton_feature",
            "leading_singleton_batch",
            "leading_singleton_full",
        ],
    )
    def test_producer_consumer_residual_rmsnorm_quant(self, weight_layout):
        B, D, G = 128, 2048, 128
        qmax = 448.0
        min_scale = 1.0 / (qmax * 512.0)

        def f(x, residual, weight):
            h = x.float() + residual.float()
            variance = h.pow(2).mean(dim=-1, keepdim=True)
            normed = h * torch.rsqrt(variance + 1e-6)
            normed_bf16 = normed.to(torch.bfloat16) * weight
            grouped = normed_bf16.view(B, D // G, G)
            absmax = grouped.abs().amax(dim=-1, keepdim=True).float()
            scales = (absmax / qmax).clamp(min=min_scale)
            x_scaled = (grouped / scales).clamp(-qmax, qmax)
            x_quant = x_scaled.to(torch.float16).view(B, D)
            return x_quant.float(), scales.squeeze(-1)

        x = torch.randn(B, D, device=GPU_TYPE, dtype=torch.bfloat16)
        residual = torch.randn(B, D, device=GPU_TYPE, dtype=torch.bfloat16)
        weight_shapes = {
            "scalar": (),
            "scalar_1d": (1,),
            "feature": (D,),
            "singleton": (1, 1),
            "singleton_batch": (1, D),
            "batch": (B, 1),
            "full": (B, D),
            "leading_singleton": (1, 1, 1),
            "leading_singleton_feature": (1, 1, D),
            "leading_singleton_batch": (1, B, 1),
            "leading_singleton_full": (1, B, D),
        }
        w = torch.randn(
            weight_shapes[weight_layout], device=GPU_TYPE, dtype=torch.bfloat16
        )

        weighted = x.to(torch.bfloat16) * w
        self.assertEqual(weighted.dtype, torch.bfloat16)
        self.assertEqual(weighted.numel(), x.numel())
        self.check_nested_matches_unnested(f, (x, residual, w))
        self.check_fusion()

    @parametrize(
        "B,K,D",
        [(64, 16, 4096), (1, 16, 1024)],
    )
    def test_fullres_epilogue_small_dim_in_x(self, B, K, D):
        """Small-dim-in-X full-res consumer falls back."""

        def f(x, w):
            x_normed = _rmsnorm(x.reshape(B * K, D)).reshape(B, K, D)
            s = (w[:, :, None] * x_normed).sum(dim=1)
            return x_normed + s[:, None, :]

        x = torch.randn(B, K, D, device=GPU_TYPE)
        w = torch.randn(B, K, device=GPU_TYPE)
        self.check_numeric(f, (x, w))
        self.check_no_fusion()

    def test_nested_reduction_rejects_shifted_parent_output(self):
        B, D, G = 4, 1024, 16

        def f(x):
            y = F.rms_norm(x, (D,))
            shifted = torch.roll(y, 1, -1)
            scale = shifted.view(B, D // G, G).abs().amax(dim=-1)
            return y, scale

        x = torch.randn(B, D, device=GPU_TYPE)
        self.check_numeric(f, (x,))
        self.check_no_fusion()

    # G=2 makes the REDUCED and SUB_PARENT domains share a numel
    # (outer_rnumel // G == outer_rnumel // 2), so a pair consumer is only
    # classified correctly if the domain check disambiguates them rather than
    # matching on numel alone. The values differ -- element 0 of each pair is
    # not the amax over that pair -- so a misclassification shows up as a
    # numeric mismatch, not just a lost fusion.
    @parametrize("D,G", [(1024, 2), (1024, 16), (4608, 16)])
    @parametrize("benchmark_fusion", [False, True])
    def test_producer_consumer_rmsnorm_interleaved_pair_epilogue(
        self, D, G, benchmark_fusion
    ):
        B = 32

        def f(x, weight):
            y = F.rms_norm(x, (D,), weight)
            yg = y.view(B, D // G, G)
            scale = (yg.abs().amax(dim=-1) / 6.0).clamp(min=1e-12, max=448.0)
            pairs = yg.view(B, D // G, G // 2, 2)
            packed_surrogate = (pairs[..., 0] + 2 * pairs[..., 1]) / scale.unsqueeze(-1)
            return packed_surrogate, scale

        x = torch.randn(B, D, device=GPU_TYPE, dtype=torch.bfloat16)
        weight = torch.randn(D, device=GPU_TYPE, dtype=torch.bfloat16)
        with inductor_config.patch("benchmark_fusion", benchmark_fusion):
            self.check_nested_matches_unnested(f, (x, weight))
        self.check_fusion(expected_kernels=None if benchmark_fusion else 1)

    def test_nested_reduction_reduced_only_consumer_group_size_two(self):
        B, D, G = 32, 1024, 2

        def f(x, weight):
            y = F.rms_norm(x, (D,), weight)
            scale = y.view(B, D // G, G).abs().amax(dim=-1)
            return scale * 2

        x = torch.randn(B, D, device=GPU_TYPE, dtype=torch.bfloat16)
        weight = torch.randn(D, device=GPU_TYPE, dtype=torch.bfloat16)
        self.check_nested_matches_unnested(f, (x, weight))
        self.check_fusion()

    def _check_sub_parent_indirect_index_mask(self, f, args):
        ref = f(*args)
        act, source_codes = run_and_get_code(torch.compile(f, fullgraph=True), *args)
        self.assertEqual(act, ref, atol=1e-3, rtol=1e-3)
        self.check_fusion()
        FileCheck().check("tl.device_assert").check(
            "lane2_r0_index_mask & xmask"
        ).check("tl.load").check("lane2_r0_index_mask & xmask").run(
            "\n\n".join(source_codes)
        )

    def test_standalone_sub_parent_preserves_indirect_index_mask(self):
        B, D = 64, 4608

        def f(x, table):
            scale = torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + 1e-6)
            pairs = x.view(B, D // 2, 2)
            index = (0.01 / (pairs[..., 0].abs() + 1e-11)).long().clamp(min=0)
            index = torch.where(
                pairs[..., 0].abs() > 1e-6, torch.zeros_like(index), index
            )
            return table[index] * scale + pairs[..., 1] * scale, scale

        x = torch.ones(B, D, device=GPU_TYPE)
        table = torch.randn(4, device=GPU_TYPE)
        self._check_sub_parent_indirect_index_mask(f, (x, table))

    @parametrize("G", [2, 16])
    def test_nested_sub_parent_preserves_indirect_index_mask(self, G):
        B, D = 64, 4608

        def f(x, table):
            y = _rmsnorm(x)
            groups = y.view(B, D // G, G)
            scale = (groups.abs().amax(dim=-1) / 6.0).clamp(min=1e-12)
            pairs = groups.view(B, D // G, G // 2, 2)
            index = (1e-11 / scale).long().unsqueeze(-1).expand_as(pairs[..., 0])
            return table[index] + pairs[..., 1] / scale.unsqueeze(-1), scale

        x = torch.ones(B, D, device=GPU_TYPE)
        table = torch.randn(4, device=GPU_TYPE)
        self._check_sub_parent_indirect_index_mask(f, (x, table))

    @parametrize("scale_first", (False, True))
    def test_sub_parent_fusion_is_independent_of_nested_append_order(self, scale_first):
        B, D, G = 32, 1024, 16

        def f(x, weight):
            y = F.rms_norm(x, (D,), weight)
            groups = y.view(B, D // G, G)
            scale = groups.abs().amax(dim=-1)
            pairs = groups.view(B, D // G, G // 2, 2)
            packed = (pairs[..., 0] + 2 * pairs[..., 1]) / scale.unsqueeze(-1)
            return (scale, packed) if scale_first else (packed, scale)

        x = torch.randn(B, D, device=GPU_TYPE, dtype=torch.bfloat16)
        weight = torch.randn(D, device=GPU_TYPE, dtype=torch.bfloat16)
        self.check_nested_matches_unnested(f, (x, weight))
        self.check_fusion()

    @parametrize("dynamic_axis", ["batch", "reduction"])
    def test_dynamic_sub_parent_epilogue(self, dynamic_axis):
        G = 16

        def f(x, weight):
            batch, dim = x.shape
            y = F.rms_norm(x, (dim,), weight)
            yg = y.view(batch, dim // G, G)
            scale = (yg.abs().amax(dim=-1) / 6.0).clamp(min=1e-12, max=448.0)
            pairs = yg.view(batch, dim // G, G // 2, 2)
            return (pairs[..., 0] + 2 * pairs[..., 1]) / scale.unsqueeze(-1)

        shapes = (
            [(4, 4096), (7, 4096)]
            if dynamic_axis == "batch"
            else [(4, 4096), (4, 4608)]
        )
        inputs = [
            (
                torch.randn(B, D, device=GPU_TYPE),
                torch.randn(D, device=GPU_TYPE),
            )
            for B, D in shapes
        ]
        x, weight = inputs[0]
        if dynamic_axis == "batch":
            torch._dynamo.mark_dynamic(x, 0)
        else:
            torch._dynamo.mark_dynamic(x, 1)
            torch._dynamo.mark_dynamic(weight, 0)
        compiled = torch.compile(f, fullgraph=True)
        for x, weight in inputs:
            self.assertEqual(compiled(x, weight), f(x, weight), atol=1e-2, rtol=1e-2)
        self.assertEqual(metrics.codegen_nested_reduction, 1)
        self.assertEqual(metrics.generated_kernel_count, 1)

    @parametrize("gate", ["max_fusion_size", "no_fuse_buffer"])
    def test_sub_parent_append_respects_fusion_gate(self, gate):
        B, D, G = 32, 1024, 16

        def f(x, weight):
            y = F.rms_norm(x, (D,), weight)
            yg = y.view(B, D // G, G)
            scale = (yg.abs().amax(dim=-1) / 6.0).clamp(min=1e-12, max=448.0)
            pairs = yg.view(B, D // G, G // 2, 2)
            return (pairs[..., 0] + 2 * pairs[..., 1]) / scale.unsqueeze(-1)

        def mark_pointwise_outputs_no_fuse(nodes):
            # comm_lowering uses this barrier to preserve compute/collective overlap.
            for node in nodes:
                if not node.is_reduction():
                    V.graph.no_fuse_buffer_names.update(node.get_buffer_names())
            return nodes

        patches = {
            "max_fusion_size": {"max_fusion_size": 2},
            "no_fuse_buffer": {
                "_pre_fusion_custom_pass": mark_pointwise_outputs_no_fuse
            },
        }
        patch = patches[gate]
        patch["fx_graph_cache"] = False
        x = torch.randn(B, D, device=GPU_TYPE, dtype=torch.bfloat16)
        weight = torch.randn(D, device=GPU_TYPE, dtype=torch.bfloat16)
        with inductor_config.patch(patch):
            self.check_nested_matches_unnested(f, (x, weight))
        self.assertEqual(metrics.codegen_nested_reduction, 1)
        self.assertEqual(metrics.generated_kernel_count, 2)

    @parametrize(
        "B,D,swizzled",
        ((1, 4096, False), (128, 4096, False), (128, 4608, True)),
    )
    @skipIfRocm
    @skipIfXpu(msg="NVFP4 inline asm requires CUDA")
    @inductor_config.patch(emulate_precision_casts=True)
    def test_producer_consumer_rmsnorm_nvfp4_inline_asm(self, B, D, swizzled):
        if torch.cuda.get_device_capability()[0] < 10:
            self.skipTest("NVFP4 inline asm requires SM100+")

        def f(x, weight):
            packed, scale = _rmsnorm_nvfp4(x, weight)
            return packed, _swizzle_scale(scale) if swizzled else scale

        x = torch.randn(B, D, device=GPU_TYPE, dtype=torch.bfloat16)
        w = torch.randn(D, device=GPU_TYPE, dtype=torch.bfloat16)

        # inline_asm_elementwise has no eager implementation.
        ref = self.get_unnested_reference(f, (x, w), fullgraph=True)
        act = torch.compile(f, fullgraph=True)(x, w)
        self.assertEqual(act[0], ref[0])
        self.assertEqual(act[1].float(), ref[1].float(), atol=1e-2, rtol=1e-2)
        self.check_fusion()

    @parametrize("B", [1, 128])
    @skipIfRocm
    @skipIfXpu(msg="MXFP4 inline asm requires CUDA")
    def test_producer_consumer_rmsnorm_mxfp4_inline_asm(self, B):
        if torch.cuda.get_device_capability()[0] < 10:
            self.skipTest("MXFP4 inline asm requires SM100+")

        D, G = 4096, 32

        def f(x, weight):
            return _rmsnorm_mxfp4(x, weight, G)

        x = torch.randn(B, D, device=GPU_TYPE, dtype=torch.bfloat16)
        weight = torch.randn(D, device=GPU_TYPE, dtype=torch.bfloat16)

        # inline_asm_elementwise has no eager implementation.
        expected = self.get_unnested_reference(f, (x, weight), fullgraph=True)
        actual = torch.compile(f, fullgraph=True)(x, weight)
        self.assertEqual(actual, expected)
        self.check_fusion()

    def test_producer_consumer_rejects_broadcast_parent_source(self):
        B, D, G = 32, 1024, 16

        def f(x, weight):
            y = F.rms_norm(x, (D,), weight)
            yg = y.view(B, D // G, G)
            amax = yg.abs().amax(dim=-1)
            scale = (amax / 6.0).clamp(min=1e-12, max=448.0)
            shifted = yg[..., 2].unsqueeze(-1).expand(B, D // G, G // 2)
            return shifted / scale.unsqueeze(-1), scale

        x = torch.randn(B, D, device=GPU_TYPE)
        weight = torch.randn(D, device=GPU_TYPE)
        self.check_nested_matches_unnested(f, (x, weight))
        self.assertEqual(metrics.codegen_nested_reduction, 1)
        self.assertGreater(metrics.generated_kernel_count, 1)

    def test_producer_consumer_rejects_conflicting_parent_source_index(self):
        B, D, G = 32, 1024, 16

        def f(x, weight):
            y = F.rms_norm(x, (D,), weight)
            yg = y.view(B, D // G, G)
            amax = yg.abs().amax(dim=-1)
            scale = (amax / 6.0).clamp(min=1e-12, max=448.0)
            side = yg[..., 2] + scale
            yg = yg.view(B, D // G, G // 2, 2)
            even = yg[..., 0] / scale.unsqueeze(-1)
            odd = yg[..., 1] / scale.unsqueeze(-1)
            return even, odd, scale, side

        x = torch.randn(B, D, device=GPU_TYPE)
        weight = torch.randn(D, device=GPU_TYPE)
        self.check_nested_matches_unnested(f, (x, weight))
        self.assertEqual(metrics.codegen_nested_reduction, 1)
        self.assertGreater(metrics.generated_kernel_count, 1)

    def test_producer_consumer_rejects_sub_parent_mutation(self):
        B, D, G = 32, 1024, 16

        def f(x, weight, out):
            y = F.rms_norm(x, (D,), weight)
            yg = y.view(B, D // G, G)
            scale = (yg.abs().amax(dim=-1) / 6.0).clamp(min=1e-12, max=448.0)
            pairs = yg.view(B, D // G, G // 2, 2)
            out.copy_((pairs[..., 0] + 2 * pairs[..., 1]) / scale.unsqueeze(-1))
            return scale

        x = torch.randn(B, D, device=GPU_TYPE)
        weight = torch.randn(D, device=GPU_TYPE)
        out = torch.empty(B, D // G, G // 2, device=GPU_TYPE)
        ref_out = torch.empty_like(out)
        expected = f(x, weight, ref_out)
        actual = torch.compile(f, fullgraph=True)(x, weight, out)
        self.assertEqual(actual, expected)
        self.assertEqual(out, ref_out)
        self.assertEqual(metrics.codegen_nested_reduction, 1)
        self.assertEqual(metrics.generated_kernel_count, 2)

    def test_producer_consumer_sub_parent_source_mutated_later(self):
        B, D, G = 8, 1024, 16

        def f(x):
            norm = (x.square().mean(-1) + 1e-5).rsqrt()
            xg = x.view(B, D // G, G)
            scale = (xg.abs() * norm[:, None, None]).amax(-1).clamp_min(1e-12)
            pairs = xg.view(B, D // G, G // 2, 2)
            packed = (pairs[..., 0] + 2 * pairs[..., 1]) / scale.unsqueeze(-1)
            x.add_(3.0)
            return packed, scale, x

        x = torch.randn(B, D, device=GPU_TYPE)
        ref_x = x.clone()
        expected = f(ref_x)
        actual = torch.compile(f, fullgraph=True)(x)
        self.assertEqual(actual, expected)
        self.assertEqual(x, ref_x)
        self.assertEqual(metrics.codegen_nested_reduction, 1)

    def test_producer_consumer_rejects_sub_parent_grouped_axis_x(self):
        B, K, D = 8, 16, 1024

        def f(x, weight):
            y = F.rms_norm(x, (D,), weight)
            scale = (y.abs().amax(dim=1) / 6.0).clamp(min=1e-12, max=448.0)
            pairs = y.view(B, K // 2, 2, D)
            packed = pairs[:, :, 0] + 2 * pairs[:, :, 1]
            return packed / scale.unsqueeze(1), scale

        x = torch.randn(B, K, D, device=GPU_TYPE)
        weight = torch.randn(D, device=GPU_TYPE)
        self.check_nested_matches_unnested(f, (x, weight))
        self.check_no_fusion()

    def test_producer_consumer_rejects_sub_parent_output_reader(self):
        B, D, G = 32, 1024, 16

        def f(x, weight):
            y = F.rms_norm(x, (D,), weight)
            yg = y.view(B, D // G, G)
            scale = (yg.abs().amax(dim=-1) / 6.0).clamp(min=1e-12, max=448.0)
            pairs = yg.view(B, D // G, G // 2, 2)
            even = pairs[..., 0] / scale.unsqueeze(-1)
            full = even.repeat_interleave(2, dim=-1).view(B, D)
            return even, full, scale

        x = torch.randn(B, D, device=GPU_TYPE)
        weight = torch.randn(D, device=GPU_TYPE)
        self.check_numeric(f, (x, weight))
        self.assertEqual(metrics.codegen_nested_reduction, 1)
        self.check_non_leaf_epilogue_fallback()

    def test_producer_consumer_rejects_shifted_sub_parent_intermediate(self):
        B, D, G = 32, 1024, 16

        def f(x, weight):
            y = F.rms_norm(x, (D,), weight)
            yg = y.view(B, D // G, G)
            scale = (yg.abs().amax(dim=-1) / 6.0).clamp(min=1e-12, max=448.0)
            pairs = yg.view(B, D // G, G // 2, 2)
            even = torch.ops._inductor_test.realize(pairs[..., 0] / scale.unsqueeze(-1))
            return even, torch.roll(even, 1, dims=-1), scale

        x = torch.randn(B, D, device=GPU_TYPE)
        weight = torch.randn(D, device=GPU_TYPE)
        self.check_nested_matches_unnested(f, (x, weight))
        self.assertEqual(metrics.codegen_nested_reduction, 1)
        self.check_non_leaf_epilogue_fallback()

    def test_producer_consumer_rejects_transposed_sub_parent_frame(self):
        B, D, G = 8, 512, 16

        def f(x):
            mean = x.float().mean(dim=-1, keepdim=True)
            var = x.float().var(dim=-1, keepdim=True, correction=0)
            y = (x.float() - mean) / torch.sqrt(var + 1e-6)
            yg = y.view(B, D // G, G)
            scale = (yg.abs().amax(dim=-1) / 6.0).clamp_min(1e-12)
            pairs = yg.view(B, D // G, G // 2, 2)
            even = pairs[..., 0] / scale.unsqueeze(-1)
            transposed = (
                y.view(B, D // 2, 2)[..., 0].transpose(0, 1).reshape(B, D // G, G // 2)
            )
            return even, transposed / scale.unsqueeze(-1), scale

        x = torch.randn(B, D, device=GPU_TYPE, dtype=torch.bfloat16)
        self.check_nested_matches_unnested(f, (x,))
        self.assertEqual(metrics.codegen_nested_reduction, 1)
        self.check_non_leaf_epilogue_fallback()

    def test_producer_consumer_rejects_shifted_reduced_source(self):
        B, D, G = 4, 1024, 16

        def f(x):
            y = F.rms_norm(x, (D,))
            yg = y.view(B, D // G, G)
            scale = (yg.abs().amax(dim=-1) / 6.0).clamp(min=1e-12, max=448.0)
            pairs = yg.view(B, D // G, G // 2, 2)
            shifted_scale = torch.roll(scale, 1, dims=-1).unsqueeze(-1)
            return (
                pairs[..., 0] / shifted_scale,
                pairs[..., 1] / shifted_scale,
                scale,
            )

        x = torch.randn(B, D, device=GPU_TYPE)
        self.check_nested_matches_unnested(f, (x,))
        self.assertEqual(metrics.codegen_nested_reduction, 1)
        self.assertGreater(metrics.generated_kernel_count, 1)

    def test_producer_consumer_sub_parent_intermediate(self):
        B, D, G = 32, 1024, 16

        def f(x, weight):
            y = F.rms_norm(x, (D,), weight)
            yg = y.view(B, D // G, G)
            scale = (yg.abs().amax(dim=-1) / 6.0).clamp(min=1e-12, max=448.0)
            pairs = yg.view(B, D // G, G // 2, 2)
            even = torch.ops._inductor_test.realize(pairs[..., 0] / scale.unsqueeze(-1))
            return even + 1, scale

        x = torch.randn(B, D, device=GPU_TYPE)
        weight = torch.randn(D, device=GPU_TYPE)
        self.check_nested_matches_unnested(f, (x, weight))
        self.check_fusion()

    def test_producer_consumer_broadcasts_outer_reduction_output(self):
        B, D, G = 32, 1024, 16

        def f(x):
            row_sum = (x.float() * x.float()).sum(dim=-1)
            rstd = torch.rsqrt(row_sum[:, None] / D + 1e-6)
            xg = (x.float() * rstd).view(B, D // G, G)
            scale = xg.abs().amax(dim=-1).clamp(min=1e-12, max=448.0)
            pairs = xg.view(B, D // G, G // 2, 2)
            packed = (
                pairs[..., 0].float()
                + 2 * pairs[..., 1].float()
                + row_sum[:, None, None]
            ) / scale.unsqueeze(-1)
            return packed, scale, row_sum

        x = torch.randn(B, D, device=GPU_TYPE, dtype=torch.bfloat16)
        self.check_nested_matches_unnested(f, (x,))
        self.check_fusion()

    @parametrize("shared_external_source", [False, True])
    def test_producer_consumer_inlined_parent_full_source(self, shared_external_source):
        B, D, G = 32, 1024, 16

        def f(x, weight):
            y = F.rms_norm(x, (D,), weight)
            if shared_external_source:
                z = torch.ops._inductor_test.realize(y + x)
                scale = z.view(B, D // G, G).abs().amax(dim=-1).clamp_min(1e-12)
                pairs = x.view(B, D // G, G // 2, 2)
                packed = (pairs[..., 0] + 2 * pairs[..., 1]) / scale.unsqueeze(-1)
            else:
                yg = y.view(B, D // G, G)
                scale = (yg.abs().amax(dim=-1) / 6.0).clamp(min=1e-12, max=448.0)
                source = yg + scale.unsqueeze(-1)
                pairs = source.view(B, D // G, G // 2, 2)
                packed = pairs[..., 0] + 2 * pairs[..., 1]
            return packed, scale

        x = torch.randn(B, D, device=GPU_TYPE)
        weight = torch.randn(D, device=GPU_TYPE)
        expected = f(x, weight)
        actual, sources = run_and_get_code(torch.compile(f), x, weight)
        self.assertEqual(actual, expected, atol=1e-2, rtol=1e-2)
        self.check_fusion()
        expected_splits = 1 if shared_external_source else 2
        FileCheck().check_count("tl.split(", expected_splits, exactly=True).run(
            "\n".join(sources)
        )

    def test_producer_consumer_independent_sub_parent_source(self):
        B, D, G = 32, 1024, 16

        def f(x, weight, z):
            y = F.rms_norm(x, (D,), weight)
            yg = y.view(B, D // G, G)
            scale = (yg.abs().amax(dim=-1) / 6.0).clamp(min=1e-12, max=448.0)
            pairs = yg.view(B, D // G, G // 2, 2)
            return (pairs[..., 0] + 2 * pairs[..., 1] + z) / scale.unsqueeze(-1)

        x = torch.randn(B, D, device=GPU_TYPE)
        weight = torch.randn(D, device=GPU_TYPE)
        z = torch.randn(B, D // G, G // 2, device=GPU_TYPE)
        self.check_nested_matches_unnested(f, (x, weight, z))
        self.check_fusion()

    def test_producer_consumer_rejects_shifted_inlined_parent_full_source(self):
        B, D, G = 32, 1024, 16

        def f(x, weight):
            y = F.rms_norm(x, (D,), weight)
            yg = y.view(B, D // G, G)
            scale = (yg.abs().amax(dim=-1) / 6.0).clamp(min=1e-12, max=448.0)
            source = yg + scale.unsqueeze(-1)
            shifted = source[..., 2].unsqueeze(-1).expand(B, D // G, G // 2)
            return shifted / scale.unsqueeze(-1), scale

        x = torch.randn(B, D, device=GPU_TYPE)
        weight = torch.randn(D, device=GPU_TYPE)
        self.check_nested_matches_unnested(f, (x, weight))
        self.assertEqual(metrics.codegen_nested_reduction, 1)
        self.assertGreater(metrics.generated_kernel_count, 1)

    def test_producer_consumer_rejects_sub_parent_output_fullres_reader(self):
        B, D, G = 32, 1024, 16

        def f(x, weight):
            y = F.rms_norm(x, (D,), weight)
            yg = y.view(B, D // G, G)
            scale = (yg.abs().amax(dim=-1) / 6.0).clamp(min=1e-12, max=448.0)
            yg = yg.view(B, D // G, G // 2, 2)
            even = yg[..., 0] / scale.unsqueeze(-1)
            odd = yg[..., 1] / scale.unsqueeze(-1)
            full = torch.stack([even, odd], dim=-1).view(B, D)
            return even, odd, full, scale

        x = torch.randn(B, D, device=GPU_TYPE)
        weight = torch.randn(D, device=GPU_TYPE)
        self.check_nested_matches_unnested(f, (x, weight))
        self.assertEqual(metrics.codegen_nested_reduction, 1)
        self.check_non_leaf_epilogue_fallback()

    def test_producer_consumer_rejects_sub_parent_output_reduction_reader(self):
        B, D, G = 32, 1024, 16

        def f(x, weight):
            y = F.rms_norm(x, (D,), weight)
            yg = y.view(B, D // G, G)
            scale = (yg.abs().amax(dim=-1) / 6.0).clamp(min=1e-12, max=448.0)
            yg = yg.view(B, D // G, G // 2, 2)
            even = yg[..., 0] / scale.unsqueeze(-1)
            odd = yg[..., 1] / scale.unsqueeze(-1)
            reduced = even.sum(dim=-1) + odd.sum(dim=-1)
            return even, odd, reduced, scale

        x = torch.randn(B, D, device=GPU_TYPE)
        weight = torch.randn(D, device=GPU_TYPE)
        self.check_nested_matches_unnested(f, (x, weight))
        self.assertEqual(metrics.codegen_nested_reduction, 1)
        self.check_non_leaf_epilogue_fallback()

    @skipIfRocm
    @skipIfXpu(msg="NVFP4 inline asm requires CUDA")
    def test_standalone_nvfp4_inline_asm(self):
        if torch.cuda.get_device_capability()[0] < 10:
            self.skipTest("NVFP4 inline asm requires SM100+")

        B, D, G = 32, 4096, 16

        def f(x):
            xg = x.view(B, D // G, G)
            amax = xg.float().abs().amax(dim=-1)
            scale = (amax / 6.0).clamp(min=1e-12, max=448.0).to(torch.float8_e4m3fn)
            xg = xg.view(B, D // G, G // 2, 2)
            scale_f = scale.float().unsqueeze(-1)
            even = xg[..., 0].float() / scale_f
            odd = xg[..., 1].float() / scale_f
            packed = inline_asm_elementwise(
                even,
                odd,
                asm_str=E2M1X2_PACK_ASM,
                constraints="=r,f,f",
                dtype=torch.int32,
                is_pure=True,
                pack=1,
            )
            return packed.to(torch.uint8).view(B, D // 2), scale.view(B, D // G)

        x = torch.randn(B, D, device=GPU_TYPE, dtype=torch.bfloat16)

        ref = self.get_unnested_reference(f, (x,), fullgraph=True)
        act = torch.compile(f, fullgraph=True)(x)
        self.assertEqual(act[0], ref[0])
        self.assertEqual(act[1].float(), ref[1].float(), atol=1e-2, rtol=1e-2)
        self.check_fusion()

    @skipIfRocm
    @skipIfXpu(msg="NVFP4 inline asm requires CUDA")
    def test_standalone_nvfp4_inline_asm_rejects_fullres_reader(self):
        if torch.cuda.get_device_capability()[0] < 10:
            self.skipTest("NVFP4 inline asm requires SM100+")

        B, D, G = 32, 4096, 16

        def f(x):
            xg = x.view(B, D // G, G)
            scale = (
                (xg.float().abs().amax(dim=-1) / 6.0)
                .clamp(min=1e-12, max=448.0)
                .to(torch.float8_e4m3fn)
            )
            xg = xg.view(B, D // G, G // 2, 2)
            scale_f = scale.float().unsqueeze(-1)
            even = xg[..., 0].float() / scale_f
            odd = xg[..., 1].float() / scale_f
            packed = inline_asm_elementwise(
                even,
                odd,
                asm_str=E2M1X2_PACK_ASM,
                constraints="=r,f,f",
                dtype=torch.int32,
                is_pure=True,
                pack=1,
            )
            extra = (even.repeat_interleave(2, dim=-1) + 1.0).view(B, D)
            return packed.to(torch.uint8).view(B, D // 2), scale.view(B, D // G), extra

        x = torch.randn(B, D, device=GPU_TYPE, dtype=torch.bfloat16)
        ref = self.get_unnested_reference(f, (x,), fullgraph=True)
        act = torch.compile(f, fullgraph=True)(x)
        self.assertEqual(act[0], ref[0])
        self.assertEqual(act[1].float(), ref[1].float(), atol=1e-2, rtol=1e-2)
        self.assertEqual(act[2], ref[2], atol=1e-2, rtol=1e-2)
        self.check_non_leaf_epilogue_fallback()

    # Cover non-power-of-two X and R extents.
    @parametrize("B,D,G", [(32, 1024, 16), (1, 16, 16), (3, 16, 16), (3, 72, 24)])
    def test_standalone_sub_parent_epilogue(self, B, D, G):
        def f(x):
            xg = x.view(B, D // G, G)
            amax = xg.float().abs().amax(dim=-1)
            scale = (amax / 6.0).clamp(min=1e-12, max=448.0)
            xg = xg.view(B, D // G, G // 2, 2)
            scale_f = scale.unsqueeze(-1)
            even = ((xg[..., 0].float() / scale_f).to(torch.float16) + 1.0).float()
            odd = ((xg[..., 1].float() / scale_f).to(torch.float16) - 1.0).float()
            return even, odd, scale

        x = torch.randn(B, D, device=GPU_TYPE, dtype=torch.bfloat16)
        self.check_nested_matches_unnested(f, (x,))
        self.check_fusion()

    @parametrize("dynamic_axis", ["batch", "reduction"])
    def test_dynamic_standalone_sub_parent_epilogue(self, dynamic_axis):
        B, D = 4, 512

        def f(x):
            batch, dim = x.shape
            scale = (x.float().abs().amax(dim=-1) / 6.0).clamp(min=1e-12, max=448.0)
            pairs = x.view(batch, dim // 2, 2)
            scale_f = scale.unsqueeze(-1)
            even = pairs[..., 0].float() / scale_f
            odd = pairs[..., 1].float() / scale_f
            return even, odd, scale

        shapes = (
            [(batch, D) for batch in (4, 8, 16)]
            if dynamic_axis == "batch"
            else [(B, dim) for dim in (510, 768, 1022)]
        )
        inputs = [
            torch.randn(shape, device=GPU_TYPE, dtype=torch.bfloat16)
            for shape in shapes
        ]
        torch._dynamo.mark_dynamic(inputs[0], 0 if dynamic_axis == "batch" else 1)
        compiled = torch.compile(f, fullgraph=True)
        for x in inputs:
            self.assertEqual(compiled(x), f(x), atol=1e-2, rtol=1e-2)
        self.check_fusion()

    @parametrize("dynamic_axis", [None, "batch", "feature"])
    def test_pointwise_producer_standalone_sub_parent_epilogue(self, dynamic_axis):
        B, D, G = 4, 512, 16

        def f(x):
            y = torch.nn.functional.gelu(x)
            batch, dim = y.shape
            groups = y.view(batch, dim // G, G)
            scale = (groups.abs().amax(dim=-1) / 6.0).clamp(min=1e-12, max=448.0)
            pairs = groups.view(batch, dim // G, G // 2, 2)
            scale = scale.unsqueeze(-1)
            return pairs[..., 0] / scale, pairs[..., 1] / scale

        shapes = {
            None: [(B, D)],
            "batch": [(batch, D) for batch in (4, 8, 16)],
            "feature": [(B, dim) for dim in (512, 768, 1024)],
        }[dynamic_axis]
        inputs = [
            torch.randn(shape, device=GPU_TYPE, dtype=torch.bfloat16)
            for shape in shapes
        ]
        if dynamic_axis is not None:
            torch._dynamo.mark_dynamic(inputs[0], 0 if dynamic_axis == "batch" else 1)

        compiled = torch.compile(f, fullgraph=True)
        for x in inputs:
            self.assertEqual(compiled(x), f(x), atol=1e-2, rtol=1e-2)
        self.check_fusion()

    def test_independent_sub_parent_source(self):
        B, D = 4, 512

        def f(x, z):
            y = torch.sin(z)
            scale = (x.float().abs().amax(dim=-1) / 6.0).clamp(min=1e-12, max=448.0)
            x_pairs = x.view(B, D // 2, 2)
            y_pairs = y.view(B, D // 2, 2)
            scale = scale.unsqueeze(-1)
            even = (x_pairs[..., 0] + y_pairs[..., 0]).float() / scale
            odd = (x_pairs[..., 1] + y_pairs[..., 1]).float() / scale
            return even, odd, scale

        x = torch.randn(B, D, device=GPU_TYPE, dtype=torch.bfloat16)
        z = torch.randn_like(x)
        self.check_nested_matches_unnested(f, (x, z))
        self.check_fusion()

    def test_cat_producer_standalone_sub_parent_epilogue(self):
        B, D = 4, 512

        def f(x, y):
            joined = torch.cat((x, y), dim=-1)
            scale = (joined.float().abs().amax(dim=-1) / 6.0).clamp(
                min=1e-12, max=448.0
            )
            pairs = joined.view(B, D // 2, 2)
            return (
                pairs[..., 0].float() / scale.unsqueeze(-1),
                pairs[..., 1].float() / scale.unsqueeze(-1),
                scale,
            )

        x = torch.randn(B, D // 2, device=GPU_TYPE, dtype=torch.bfloat16)
        y = torch.randn_like(x)
        self.check_nested_matches_unnested(f, (x, y))
        self.check_fusion()

    def _standalone_sub_parent_graph(self, B=32, D=1024, G=16):
        def f(x):
            xg = x.view(B, D // G, G)
            amax = xg.float().abs().amax(dim=-1)
            scale = (amax / 6.0).clamp(min=1e-12, max=448.0)
            xg2 = xg.view(B, D // G, G // 2, 2)
            sf = scale.unsqueeze(-1)
            even = ((xg2[..., 0].float() / sf).to(torch.float16) + 1.0).float()
            odd = ((xg2[..., 1].float() / sf).to(torch.float16) - 1.0).float()
            return even, odd, scale

        return f, torch.randn(B, D, device=GPU_TYPE, dtype=torch.bfloat16)

    def test_standalone_sub_parent_has_staged_identity(self):
        from torch._inductor.scheduler import FusedStagedReduction

        saw_staged_reduction = False

        def check_fusion(nodes):
            nonlocal saw_staged_reduction
            staged = [node for node in nodes if isinstance(node, FusedStagedReduction)]
            self.assertEqual(len(staged), 1)
            self.assertIs(type(staged[0]), FusedStagedReduction)
            saw_staged_reduction = True
            return nodes

        f, x = self._standalone_sub_parent_graph()
        with inductor_config.patch(
            _post_fusion_custom_pass=check_fusion,
            fx_graph_cache=False,
        ):
            self.check_numeric(f, (x,))
        self.assertTrue(saw_staged_reduction)
        self.check_fusion()

    def test_standalone_sub_parent_rejects_incompatible_reduction(self):
        from torch._inductor.scheduler import FusedStagedReduction

        f, x = self._standalone_sub_parent_graph()
        z = torch.randn(32 * 1024 // 16, 16, device=GPU_TYPE)
        saw_staged_reduction = False

        def g(x, z):
            return f(x), (z + 1).sum(dim=-1)

        def check_fusion(nodes):
            nonlocal saw_staged_reduction
            staged = [node for node in nodes if type(node) is FusedStagedReduction]
            self.assertEqual(len(staged), 1)
            num_reductions = sum(node.is_reduction() for node in staged[0].get_nodes())
            self.assertEqual(num_reductions, 1)
            saw_staged_reduction = True
            return nodes

        with inductor_config.patch(
            aggressive_fusion=True,
            _post_fusion_custom_pass=check_fusion,
            fx_graph_cache=False,
        ):
            self.check_numeric(g, (x, z))
        self.assertTrue(saw_staged_reduction)

    def test_looped_standalone_sub_parent_large_group(self):
        if self.force_persistent_outer_reduction is not False:
            self.skipTest("requires a looped reduction")

        f, x = self._standalone_sub_parent_graph(B=8, D=16384, G=16384)
        self.check_nested_matches_unnested(f, (x,))
        self.check_fusion()

    def test_standalone_sub_parent_declines_benchmark_fusion(self):
        # benchmark_fusion re-expands a group through generic scheduling, which
        # cannot represent the epilogue's derived group. The group must decline
        # benchmarking and still emit the staged kernel, not abort the compile.
        # Deliberately does not assert codegen_nested_reduction: benchmarking
        # times real kernels, so an unrelated pair winning or losing on noise
        # can change whether the group forms. Not aborting is the contract.
        f, x = self._standalone_sub_parent_graph()
        ref = f(x)
        with inductor_config.patch({"benchmark_fusion": True}):
            act = torch.compile(f, fullgraph=True)(x)
        self.assertEqual(act, ref, atol=1e-2, rtol=1e-2)

    def test_standalone_sub_parent_declines_combo_kernel(self):
        # Same for combo kernels. An independent same-shaped reduction gives
        # combo grouping something to try to combine the sub-parent node with.
        f, x = self._standalone_sub_parent_graph()
        z = torch.randn(32, 1024, device=GPU_TYPE, dtype=torch.bfloat16)

        def g(x, z):
            return f(x), (z.float() ** 2).sum(dim=-1)

        ref = g(x, z)
        with inductor_config.patch(
            {"combo_kernels": True, "combo_kernels_pointwise_only": False}
        ):
            act = torch.compile(g, fullgraph=True)(x, z)
        self.assertEqual(act, ref, atol=1e-2, rtol=1e-2)
        self.assertEqual(metrics.codegen_nested_reduction, 1)

    def test_producer_consumer_mxfp6_four_to_three_pack(self):
        B, D, G = (
            (8, 16384, 16384)
            if self.force_persistent_outer_reduction is False
            else (32, 1024, 32)
        )
        values = torch.tensor(
            [-1.5, -0.75, 0.25, 1.5], device=GPU_TYPE, dtype=torch.bfloat16
        )
        x = values.repeat(B, D // values.numel())

        def f(x):
            return _mxfp6_four_to_three_quantize(x, G)

        self.check_nested_matches_unnested(f, (x,))
        self.check_fusion()

    @inductor_config.patch(
        {
            "fx_graph_cache": False,
            "loop_ordering_after_fusion": False,
            "triton.coalesce_tiling_analysis": False,
        }
    )
    def test_rmsnorm_factor4_three_output_epilogue(self):
        B, D, G = 8, 4096, 32
        x = torch.randn(B, D, device=GPU_TYPE, dtype=torch.bfloat16)
        weight = torch.randn(D, device=GPU_TYPE, dtype=torch.bfloat16)
        self.check_nested_matches_unnested(
            _rmsnorm_factor4_three_output_epilogue, (x, weight, G)
        )
        self.check_fusion()

    def test_dynamic_batch_mxfp6_four_to_three_pack(self):
        D = 1024

        def f(x):
            return _mxfp6_four_to_three_quantize(x, 32)

        inputs = [
            torch.randn(B, D, device=GPU_TYPE, dtype=torch.bfloat16) for B in (8, 13)
        ]
        for x in inputs:
            torch._dynamo.mark_static(x, 1)
        with inductor_config.patch("triton.nested_reduction", False):
            ref_compiled = torch.compile(f, fullgraph=True, dynamic=True)
            refs = [ref_compiled(x) for x in inputs]

        metrics.reset()
        torch._dynamo.reset()
        compiled = torch.compile(f, fullgraph=True, dynamic=True)
        for x, ref in zip(inputs, refs):
            self.assertEqual(compiled(x), ref, atol=1e-2, rtol=1e-2)
        self.check_fusion()

    def test_producer_consumer_mxfp6_four_to_three_pack_exact(self):
        """Use an exact scale of one to compare eager packing bit-for-bit."""
        B, D, G = 32, 1024, 32

        def f(x):
            xg = x.view(B, D // G, G)
            scale = xg.abs().amax(dim=-1).clamp(min=1e-12) / 8.0
            values = (xg / scale.unsqueeze(-1)).round().to(torch.int32) & 0x3F
            return _mxfp6_pack_four_to_three(values).view(B, D // 4, 3), scale

        group = torch.arange(G, device=GPU_TYPE, dtype=torch.float32) % 17 - 8
        x = group.repeat(B, D // G).view(B, D).contiguous()
        x[:, ::G] = 8.0  # pin the group max so the scale is exactly 1.0

        expected = f(x)
        self.assertTrue((expected[1] == 1.0).all())
        actual = torch.compile(f, fullgraph=True)(x)
        self.assertEqual(actual[0], expected[0], atol=0, rtol=0)
        self.assertEqual(actual[1], expected[1], atol=0, rtol=0)
        self.check_fusion()

    def test_looped_internal_source_uses_second_pass(self):
        B, D = 8, 16384

        def f(x):
            source = torch.ops._inductor_test.realize(torch.nn.functional.silu(x))
            scale = x.float().abs().amax(dim=-1)
            pairs = source.view(B, D // 2, 2)
            scale = scale.unsqueeze(-1)
            return pairs[..., 0] / scale, pairs[..., 1] / scale

        self._check_looped_internal_source(f, (B, D), expected_passes=2)

    def test_looped_internal_source_uses_reduced_output(self):
        B, D = 8, 16384

        def f(x):
            source_input = torch.ops._inductor_test.realize(torch.nn.functional.silu(x))
            scale = x.float().abs().amax(dim=-1)
            source = torch.ops._inductor_test.realize(
                source_input + scale.unsqueeze(-1)
            )
            pairs = source.view(B, D // 2, 2)
            return pairs[..., 0] + 1, pairs[..., 1] + 2

        self._check_looped_internal_source(f, (B, D), expected_passes=2)

    def test_looped_internal_source_reuses_final_reduction_pass(self):
        B, D = 8, 16384

        def f(x):
            source = torch.ops._inductor_test.realize(torch.nn.functional.silu(x))
            scale = x.float().abs().amax(dim=-1)
            post_reduction = torch.ops._inductor_test.realize(
                x.float() + scale.unsqueeze(-1)
            )
            pairs = source.view(B, D // 2, 2)
            return pairs[..., 0] + 2, pairs[..., 1] + 3, post_reduction

        self._check_looped_internal_source(f, (B, D), expected_passes=2)

    def test_looped_internal_source_closes_final_reduction_pass(self):
        B, D = 8, 16384

        def f(x):
            source = torch.ops._inductor_test.realize(torch.nn.functional.silu(x))
            first_scale = x.float().abs().amax(dim=-1)
            shifted = torch.ops._inductor_test.realize(
                x.float() + first_scale.unsqueeze(-1)
            )
            scale = shifted.abs().amax(dim=-1).unsqueeze(-1)
            pairs = source.view(B, D // 2, 2)
            return pairs[..., 0] / scale, pairs[..., 1] / scale

        self._check_looped_internal_source(f, (B, D), expected_passes=3)

    @parametrize("shifted", [False, True])
    def test_producer_consumer_mxfp6_preshuffled_four_to_three_pack(self, shifted):
        B, D = 128, 384

        def f(x):
            return _mxfp6_preshuffled_quantize(x, shifted)

        x = (torch.arange(B * D, device=GPU_TYPE) % 29 - 14).to(torch.bfloat16).view(
            B, D
        ) / 4
        expected = self.get_unnested_reference(f, (x,))
        actual = torch.compile(f)(x)
        self.assertEqual(actual[0], expected[0], atol=0, rtol=0)
        self.assertEqual(actual[1], expected[1], atol=1e-2, rtol=1e-2)
        if shifted:
            unshifted, _scale = _mxfp6_preshuffled_quantize(x, shifted=False)
            self.assertFalse(torch.equal(expected[0], unshifted))
            self.check_non_leaf_epilogue_fallback()
        else:
            self.check_fusion()

    @parametrize("shifted", [False, True])
    def test_producer_consumer_mxfp6_pack_scale_swizzle(self, shifted):
        B, D = 128, 384

        def f(x):
            return _mxfp6_pack_scale_swizzle(x, shifted=shifted)

        x = torch.randn(B, D, device=GPU_TYPE, dtype=torch.float16)
        expected = self.get_unnested_reference(f, (x,))
        actual = torch.compile(f, fullgraph=True)(x)
        self.assertEqual(actual, expected)
        if shifted:
            self.check_non_leaf_epilogue_fallback()
            return
        self.check_fusion()

    def test_producer_consumer_mxfp6_rejects_shifted_intermediate(self):
        B, D, G = 32, 1024, 32

        def f(x):
            xg = torch.nn.functional.silu(x).view(B, D // G, G).float()
            scale = xg.abs().amax(dim=-1).clamp(min=1e-12) / 7.5
            base = torch.arange(D, device=x.device).view(1, D // G, G)
            values = torch.ops._inductor_test.realize(
                (base + (scale.unsqueeze(-1) > 0).to(torch.int32)) & 0x3F
            ).view(B, D // 4, 4)
            low = torch.ops._inductor_test.realize(
                values[..., 0] | ((values[..., 1] & 0x03) << 6)
            )
            middle = torch.ops._inductor_test.realize(
                ((values[..., 1] >> 2) & 0x0F) | ((values[..., 2] & 0x0F) << 4)
            )
            high = torch.ops._inductor_test.realize(
                ((values[..., 2] >> 4) & 0x03) | (values[..., 3] << 2)
            )
            low = torch.roll(low, 1, -1)
            return torch.stack((low, middle, high), dim=-1).to(torch.uint8), scale

        x = torch.randn(B, D, device=GPU_TYPE, dtype=torch.bfloat16)
        self.check_nested_matches_unnested(f, (x,))
        self.check_non_leaf_epilogue_fallback()

    def test_producer_consumer_mxfp6_rejects_nontrailing_output_lane(self):
        B, D, G = 32, 1024, 32

        def f(x):
            groups = x.view(B, D // G, G).float()
            scale = groups.abs().amax(dim=-1).clamp_min(1e-6)
            values = ((groups / scale.unsqueeze(-1)) * 7).round().to(torch.int32)
            values = (values & 0x3F).view(B, D // 4, 4)
            low = values[..., 0] | ((values[..., 1] & 0x03) << 6)
            middle = ((values[..., 1] >> 2) & 0x0F) | ((values[..., 2] & 0x0F) << 4)
            high = ((values[..., 2] >> 4) & 0x03) | (values[..., 3] << 2)
            return torch.stack((low, middle, high), dim=-2).to(torch.uint8), scale

        x = torch.randn(B, D, device=GPU_TYPE, dtype=torch.bfloat16)
        self.check_nested_matches_unnested(f, (x,))
        self.check_non_leaf_epilogue_fallback()

    def test_producer_consumer_mxfp6_rejects_source_read_by_reduction(self):
        B, D, G = 32, 1024, 32

        def f(x):
            xg = x.view(B, D // G, G).float()
            scale = xg.abs().amax(dim=-1).clamp_min(1e-6)
            values = torch.ops._inductor_test.realize(
                ((xg / scale.unsqueeze(-1)) * 7).round().to(torch.int32) & 0x3F
            )
            reduced = values.float().abs().amax(dim=-1) + 1
            return _mxfp6_pack_four_to_three(values), reduced

        values = torch.tensor(
            [-1.5, -0.75, 0.25, 1.5], device=GPU_TYPE, dtype=torch.bfloat16
        )
        x = values.repeat(B, D // values.numel())
        self.check_nested_matches_unnested(f, (x,))
        self.check_no_fusion()

    @parametrize("prologue_kind", ["output", "realized"])
    def test_nested_sub_parent_rejects_parent_prologue_source(self, prologue_kind):
        B, G = 32, 16
        D = 4096 if self.force_persistent_outer_reduction is False else 512

        def f(x, weight):
            prologue = x * weight + 1
            if prologue_kind == "realized":
                prologue = torch.ops._inductor_test.realize(prologue)
            normalized = _rmsnorm(prologue)
            groups = normalized.view(B, D // G, G)
            scale = (groups.abs().amax(dim=-1) / 6).clamp(1e-12, 448)
            pairs = groups.view(B, D // G, G // 2, 2)
            even = (pairs[..., 0] / scale.unsqueeze(-1)).to(torch.float16)
            odd = (pairs[..., 1] / scale.unsqueeze(-1)).to(torch.float16)
            outputs = even, odd, scale
            return (*outputs, prologue) if prologue_kind == "output" else outputs

        args = (
            torch.randn(B, D, dtype=torch.bfloat16, device=GPU_TYPE),
            torch.randn(D, dtype=torch.bfloat16, device=GPU_TYPE),
        )
        self.check_nested_matches_unnested(f, args)
        self.check_fusion(expected_kernels=2)

    def test_nested_sub_parent_parent_stage_sibling_source(self):
        # The sub-parent plan declines this topology, but so does the staged
        # fusion gate, so the kernel count below does not isolate the planner
        # guard; test_inductor_scheduler.py covers that directly.
        B, G = 32, 16
        D = 4096 if self.force_persistent_outer_reduction is False else 512
        realize = torch.ops._inductor_test.realize

        def f(x, weight):
            prologue = x * weight + 1
            # Consumes the prologue but never feeds the parent reduction, so it
            # is emitted in the parent stage without being a reduction ancestor.
            gates = realize(torch.sigmoid(prologue)).view(B, D // G, G // 2, 2)
            normalized = realize(_rmsnorm(prologue))
            groups = normalized.view(B, D // G, G)
            scale = (groups.abs().amax(dim=-1) / 6).clamp(1e-12, 448)
            pairs = groups.view(B, D // G, G // 2, 2)
            inverse = scale.unsqueeze(-1)
            even = (pairs[..., 0] / inverse * gates[..., 0]).to(torch.float16)
            odd = (pairs[..., 1] / inverse * gates[..., 1]).to(torch.float16)
            return even, odd, scale, prologue

        args = (
            torch.randn(B, D, dtype=torch.bfloat16, device=GPU_TYPE),
            torch.randn(D, dtype=torch.bfloat16, device=GPU_TYPE),
        )
        self.check_nested_matches_unnested(f, args)
        self.assertGreaterEqual(metrics.generated_kernel_count, 2)

    def test_mxfp6_internal_source_full_resolution_fork(self):
        x = torch.randn(32, 1024, device=GPU_TYPE, dtype=torch.bfloat16)
        self.check_nested_matches_unnested(
            _mxfp6_internal_source_full_resolution_fork, (x,)
        )
        self.check_fusion()

    def test_looped_mxfp6_rejects_source_before_reduction(self):
        if self.force_persistent_outer_reduction is not False:
            self.skipTest("requires a looped reduction")

        B, D = 8, 16384

        def f(x):
            source = torch.ops._inductor_test.realize(
                (torch.nn.functional.silu(x) * 7).round().to(torch.int32) & 0x3F
            )
            reduction_input = torch.ops._inductor_test.realize(source.float() * 2)
            scale = reduction_input.abs().amax(dim=-1)
            return _mxfp6_pack_four_to_three(source), scale

        x = torch.randn(B, D, device=GPU_TYPE, dtype=torch.bfloat16)
        self.check_nested_matches_unnested(f, (x,))
        self.check_no_fusion()

    def test_looped_mxfp6_rejects_source_chain_used_by_reduction(self):
        if self.force_persistent_outer_reduction is not False:
            self.skipTest("requires a looped reduction")

        B, D = 8, 16384

        def f(x):
            base = torch.ops._inductor_test.realize(torch.nn.functional.silu(x))
            source = torch.ops._inductor_test.realize(base + 1)
            scale = x.float().abs().amax(dim=-1)
            sibling = base.float().abs().amax(dim=-1)
            pairs = source.view(B, D // 2, 2)
            return pairs[..., 0], pairs[..., 1], scale, sibling

        x = torch.randn(B, D, device=GPU_TYPE, dtype=torch.bfloat16)
        self.check_nested_matches_unnested(f, (x,))
        self.check_no_fusion()

    def test_standalone_sub_parent_rejects_output_fullres_reader(self):
        B, D, G = 32, 1024, 16

        def f(x):
            xg = x.view(B, D // G, G)
            scale = (xg.float().abs().amax(dim=-1) / 6.0).clamp(min=1e-12, max=448.0)
            xg = xg.view(B, D // G, G // 2, 2)
            even = xg[..., 0].float() / scale.unsqueeze(-1)
            odd = xg[..., 1].float() / scale.unsqueeze(-1)
            full = torch.stack([even, odd], dim=-1).view(B, D)
            return even, odd, full, scale

        x = torch.randn(B, D, device=GPU_TYPE, dtype=torch.bfloat16)
        self.check_nested_matches_unnested(f, (x,))
        self.check_non_leaf_epilogue_fallback()

    def test_standalone_sub_parent_rejects_output_reduction_reader(self):
        B, D, G = 32, 1024, 16

        def f(x):
            xg = x.view(B, D // G, G)
            scale = (xg.float().abs().amax(dim=-1) / 6.0).clamp(min=1e-12, max=448.0)
            xg = xg.view(B, D // G, G // 2, 2)
            even = xg[..., 0].float() / scale.unsqueeze(-1)
            odd = xg[..., 1].float() / scale.unsqueeze(-1)
            reduced = even.sum(dim=-1) + odd.sum(dim=-1)
            return even, odd, reduced, scale

        x = torch.randn(B, D, device=GPU_TYPE, dtype=torch.bfloat16)
        self.check_nested_matches_unnested(f, (x,))
        self.check_non_leaf_epilogue_fallback()

    def test_standalone_sub_parent_leaves_incompatible_reader_unfused(self):
        B, D = 4, 512

        def f(x):
            scale = (x.float().abs().amax(dim=-1) / 6.0).clamp(min=1e-12, max=448.0)
            pairs = x.view(B, D // 2, 2)
            even = pairs[..., 0].float() / scale.unsqueeze(-1)
            odd = pairs[..., 1].float() / scale.unsqueeze(-1)
            side = torch.sin(even[:, ::2])
            return even, odd, side, scale

        x = torch.randn(B, D, device=GPU_TYPE, dtype=torch.bfloat16)
        self.check_numeric(f, (x,))
        self.assertEqual(metrics.codegen_nested_reduction, 1)
        self.assertEqual(metrics.generated_kernel_count, 2)

    def test_standalone_sub_parent_multidimensional_reduction(self):
        B, C, D = 4, 4, 16

        def f(x):
            scale = (x.float().abs().amax(dim=(-2, -1)) / 6.0).clamp(
                min=1e-12, max=448.0
            )
            pairs = x.view(B, C, D // 2, 2)
            scale_f = scale[:, None, None]
            even = pairs[..., 0].float() / scale_f
            odd = pairs[..., 1].float() / scale_f
            return even, odd, scale

        x = torch.randn(B, C, D, device=GPU_TYPE, dtype=torch.bfloat16)
        self.check_nested_matches_unnested(f, (x,))
        self.check_fusion()

    def test_standalone_sub_parent_rejects_mismatched_parent_coordinates(self):
        def f(x):
            parent = torch.as_strided(x, (2, 6, 8), (100, 10, 1))
            scale = (parent.float().abs().amax(dim=-1) / 6.0).clamp(
                min=1e-12, max=448.0
            )
            even = torch.as_strided(x, (3, 4, 4), (100, 10, 2))
            odd = torch.as_strided(x, (3, 4, 4), (100, 10, 2), storage_offset=1)
            scale_f = scale.reshape(3, 4, 1)
            return even.float() / scale_f, odd.float() / scale_f, scale

        x = torch.randn(256, device=GPU_TYPE, dtype=torch.bfloat16)
        self.check_numeric(f, (x,))
        self.check_no_fusion()

    def test_standalone_sub_parent_rejects_offset_source(self):
        B, D = 4, 16

        def f(storage):
            x = torch.as_strided(storage, (B, D), (D + 1, 1), storage_offset=1)
            scale = (x.float().abs().amax(dim=-1) / 6.0).clamp(min=1e-12, max=448.0)
            pairs = x.view(B, D // 2, 2)
            return (
                pairs[..., 0].float() / scale.unsqueeze(-1),
                pairs[..., 1].float() / scale.unsqueeze(-1),
                scale,
            )

        storage = torch.randn(B * (D + 1), device=GPU_TYPE, dtype=torch.bfloat16)
        self.check_numeric(f, (storage,))
        self.check_no_fusion()

    def test_standalone_sub_parent_mismatched_masked_source(self):
        B, D = 4, 16

        def f(x):
            parent = torch.nn.functional.pad(x, (2, 0), value=0.0)
            child = torch.nn.functional.pad(x, (2, 0), value=1.0)
            scale = (parent.float().abs().amax(dim=-1) / 6.0).clamp(
                min=1e-12, max=448.0
            )
            pairs = child.view(B, D // 2, 2)
            return (
                pairs[..., 0].float() / scale.unsqueeze(-1),
                pairs[..., 1].float() / scale.unsqueeze(-1),
                scale,
            )

        x = torch.randn(B, D - 2, device=GPU_TYPE, dtype=torch.bfloat16)
        self.check_numeric(f, (x,))
        self.check_fusion()

    def test_standalone_sub_parent_masked_group_source_falls_back(self):
        B, D, G = 2, 48, 16

        def f(x):
            groups = x.view(B, D // G, G)
            scale = (groups.float().abs().amax(dim=-1) / 6.0).clamp(min=1e-6)
            padded_scale = torch.nn.functional.pad(scale[:, :-1], (0, 1), value=7.0)
            pairs = groups.view(B, D // G, G // 2, 2)
            return pairs[..., 0].float() / padded_scale.unsqueeze(-1), scale

        x = torch.randn(B, D, device=GPU_TYPE, dtype=torch.bfloat16)
        self.check_numeric(f, (x,))
        self.check_no_fusion()
        self.assertEqual(metrics.generated_kernel_count, 2)

    def test_standalone_sub_parent_rejects_same_buffer_scalar(self):
        B, D, G = 32, 1024, 16

        def f(x):
            xg = x.view(B, D // G, G)
            amax = xg.float().abs().amax(dim=-1)
            scale = (amax / 6.0).clamp(min=1e-12, max=448.0)
            xg = xg.view(B, D // G, G // 2, 2)
            scale_f = scale.unsqueeze(-1)
            scalar = x[0, 0].float()
            even = xg[..., 0].float() / scale_f + scalar
            odd = xg[..., 1].float() / scale_f - scalar
            return even, odd, scale

        x = torch.randn(B, D, device=GPU_TYPE, dtype=torch.bfloat16)
        self.check_numeric(f, (x,))
        self.check_no_fusion()
        self.assertGreater(metrics.generated_kernel_count, 1)

    def test_standalone_sub_parent_rejects_shared_broadcast_source(self):
        B, D = 4, 16

        # A normalized dep cannot recover which nontrivial domain was broadcast.
        def f(x, row):
            scale = ((x + row[:, None]).float().abs().amax(dim=-1) / 6.0).clamp(
                min=1e-12, max=448.0
            )
            pairs = x.view(B, D // 2, 2)
            row = row[:, None]
            even = pairs[..., 0].float() / scale[:, None] + row
            odd = pairs[..., 1].float() / scale[:, None] - row
            return even, odd, scale

        x = torch.randn(B, D, device=GPU_TYPE, dtype=torch.bfloat16)
        row = torch.randn(B, device=GPU_TYPE, dtype=torch.bfloat16)
        self.check_numeric(f, (x, row))
        self.check_no_fusion()

    def test_standalone_sub_parent_parent_only_broadcast_source(self):
        B, D = 4, 16

        def f(x, row):
            scale = ((x + row[:, None]).float().abs().amax(dim=-1) / 6.0).clamp(
                min=1e-12, max=448.0
            )
            pairs = x.view(B, D // 2, 2)
            even = pairs[..., 0].float() / scale[:, None]
            odd = pairs[..., 1].float() / scale[:, None]
            return even, odd, scale

        x = torch.randn(B, D, device=GPU_TYPE, dtype=torch.bfloat16)
        row = torch.randn(B, device=GPU_TYPE, dtype=torch.bfloat16)
        self.check_nested_matches_unnested(f, (x, row))
        self.check_fusion()

    def test_standalone_sub_parent_shared_scalar_source(self):
        B, D = 4, 16

        def f(x, scalar):
            scale = ((x * scalar).float().abs().amax(dim=-1) / 6.0).clamp(
                min=1e-12, max=448.0
            )
            pairs = x.view(B, D // 2, 2)
            even = pairs[..., 0].float() / scale[:, None] + scalar
            odd = pairs[..., 1].float() / scale[:, None] - scalar
            return even, odd, scale

        x = torch.randn(B, D, device=GPU_TYPE, dtype=torch.bfloat16)
        scalar = torch.tensor(2.0, device=GPU_TYPE, dtype=torch.bfloat16)
        self.check_nested_matches_unnested(f, (x, scalar))
        self.check_fusion()

    def test_standalone_sub_parent_rejects_ambiguous_source_load(self):
        B, D, G = 32, 1024, 16

        def f(x):
            xg = x.view(B, D // G, G)
            amax = xg.float().abs().amax(dim=-1)
            scale = (amax / 6.0).clamp(min=1e-12, max=448.0)
            side = xg[..., 2].float() + scale
            xg = xg.view(B, D // G, G // 2, 2)
            scale_f = scale.unsqueeze(-1)
            even = xg[..., 0].float() / scale_f
            odd = xg[..., 1].float() / scale_f
            return even, odd, scale, side

        x = torch.randn(B, D, device=GPU_TYPE, dtype=torch.bfloat16)
        self.check_numeric(f, (x,))
        self.check_no_fusion()
        self.assertGreater(metrics.generated_kernel_count, 1)

    def test_standalone_sub_parent_rejects_shifted_reduction_output(self):
        B, D = 4, 512

        def f(x):
            scale = (x.float().abs().amax(dim=-1) / 6.0).clamp(min=1e-12, max=448.0)
            shifted_scale = torch.roll(scale, 1).unsqueeze(-1)
            pairs = x.view(B, D // 2, 2)
            even = pairs[..., 0].float() / shifted_scale
            odd = pairs[..., 1].float() / shifted_scale
            return even, odd, scale

        x = torch.randn(B, D, device=GPU_TYPE, dtype=torch.bfloat16)
        self.check_numeric(f, (x,))
        self.check_no_fusion()
        self.assertGreater(metrics.generated_kernel_count, 1)

    def test_standalone_sub_parent_rejects_transposed_sibling_frame(self):
        B, D, G = 8, 512, 16

        def f(x):
            xg = x.view(B, D // G, G)
            scale = (xg.float().abs().amax(dim=-1) / 6.0).clamp(min=1e-12, max=448.0)
            pairs = xg.view(B, D // G, G // 2, 2)
            even = pairs[..., 0].float() / scale.unsqueeze(-1)
            transposed = (
                x.view(B, D // 2, 2)[..., 0].transpose(0, 1).reshape(B, D // G, G // 2)
            )
            return even, transposed.float() / scale.unsqueeze(-1), scale

        x = torch.randn(B, D, device=GPU_TYPE, dtype=torch.bfloat16)
        self.check_nested_matches_unnested(f, (x,))
        self.assertEqual(metrics.codegen_nested_reduction, 1)
        self.check_non_leaf_epilogue_fallback()

    def test_standalone_sub_parent_allows_reduced_sibling_source(self):
        B, D, G = 32, 1024, 16

        def f(x, y):
            xg = x.view(B, D // G, G)
            yg = y.view(B, D // G, G)
            amax = (xg.float().abs() + yg.float().abs()).amax(dim=-1)
            scale = (amax / 6.0).clamp(min=1e-12, max=448.0)
            pairs = xg.view(B, D // G, G // 2, 2)
            even = pairs[..., 0].float() / scale.unsqueeze(-1)
            odd = pairs[..., 1].float() / scale.unsqueeze(-1)
            side = yg[..., 0].float() + scale
            return even, odd, side, scale

        x = torch.randn(B, D, device=GPU_TYPE, dtype=torch.bfloat16)
        y = torch.randn_like(x)
        self.check_nested_matches_unnested(f, (x, y))
        self.check_fusion()

    # TODO: Reduce the looped full-resolution plus sub-parent form from three
    # passes to two.
    def test_standalone_sub_parent_allows_fullres_sibling(self):
        B, D, G = 32, 1024, 16

        def f(x):
            xg = x.view(B, D // G, G)
            scale = (xg.float().abs().amax(dim=-1) / 6.0).clamp(min=1e-12, max=448.0)
            full = xg.float() / scale.unsqueeze(-1)
            pairs = xg.view(B, D // G, G // 2, 2)
            even = pairs[..., 0].float() / scale.unsqueeze(-1)
            odd = pairs[..., 1].float() / scale.unsqueeze(-1)
            return even, odd, full, scale

        x = torch.randn(B, D, device=GPU_TYPE, dtype=torch.bfloat16)
        self.check_nested_matches_unnested(f, (x,))
        self.check_fusion()

    def test_standalone_sub_parent_rejects_mutation(self):
        B, D, G = 32, 1024, 16

        def f(x, out):
            xg = x.view(B, D // G, G)
            scale = (xg.float().abs().amax(dim=-1) / 6.0).clamp(min=1e-12, max=448.0)
            pairs = xg.view(B, D // G, G // 2, 2)
            out.copy_(pairs[..., 0].float() / scale.unsqueeze(-1))
            return scale

        x = torch.randn(B, D, device=GPU_TYPE, dtype=torch.bfloat16)
        out = torch.empty(B, D // G, G // 2, device=GPU_TYPE)
        ref_out = torch.empty_like(out)
        ref_scale = f(x, ref_out)
        act_scale = torch.compile(f, fullgraph=True)(x, out)
        self.assertEqual(act_scale, ref_scale, atol=1e-2, rtol=1e-2)
        self.assertEqual(out, ref_out, atol=1e-2, rtol=1e-2)
        self.check_no_fusion()
        self.assertGreater(metrics.generated_kernel_count, 1)

    def test_fullres_x_epilogue_rejects_intermediate_dependency(self):
        """Do not fuse a full-res consumer before its extra producer."""
        B, K, D = 16, 16, 1024

        def f(x, w):
            x_normed = _rmsnorm(x.reshape(B * K, D)).reshape(B, K, D)
            s = (w[:, :, None] * x_normed).sum(dim=1)
            row_sum = torch.ops._inductor_test.realize(s.sum(dim=-1, keepdim=True))
            return x_normed + s[:, None, :] + row_sum[:, None, :]

        x = torch.randn(B, K, D, device=GPU_TYPE)
        w = torch.randn(B, K, device=GPU_TYPE)
        self.check_numeric(f, (x, w))
        self.check_no_fusion()

    def test_epilogue_rejects_intermediate_dependency(self):
        """Do not fuse a pointwise epilogue before another dependent node."""
        from torch._inductor.scheduler import FusedNestedReductions

        B, D, G = 64, 4096, 128

        def f(x, weight):
            x = F.rms_norm(x, (D,), weight)
            amax = x.view(B, D // G, G).abs().amax(dim=-1)
            row_sum = torch.ops._inductor_test.realize(amax.sum(dim=-1, keepdim=True))
            return amax + row_sum

        saw_nested_reduction = False

        def check_reduction_fusion(nodes):
            nonlocal saw_nested_reduction
            fused_nodes = [n for n in nodes if isinstance(n, FusedNestedReductions)]
            self.assertEqual(len(fused_nodes), 1)
            saw_nested_reduction = True
            node2_pointwise = [
                sn for sn in fused_nodes[0].node2.get_nodes() if not sn.is_reduction()
            ]
            self.assertEqual(node2_pointwise, [])
            return nodes

        x = torch.randn(B, D, device=GPU_TYPE)
        w = torch.randn(D, device=GPU_TYPE)
        with inductor_config.patch(
            _post_fusion_custom_pass=check_reduction_fusion,
            fx_graph_cache=False,
        ):
            self.check_numeric(f, (x, w))
        self.assertTrue(saw_nested_reduction)
        self.check_fusion(expected_kernels=None)

    # ---- Fusion rejection: patterns that must NOT use nested reduction ----

    def _check_rejected(self, f, args):
        """Verify numerics are correct but nested reduction did not fire."""
        self.check_numeric(f, args)
        self.assertEqual(metrics.codegen_nested_reduction, 0)

    @parametrize("G", [17, 2048])
    def test_reject_bad_group_size(self, G):
        """Non-power-of-2 or too-large group_size must not fuse."""
        D = G * 4

        def f(x):
            return _rmsnorm(x).reshape(4, -1, G).abs().amax(dim=-1)

        self._check_rejected(f, (torch.randn(4, D, device=GPU_TYPE),))

    def test_two_grouped_stages_differing_group_size(self):
        """Planning a second grouped stage asks a min-block tiling question about
        an outer node that is already staged. Its [X, R/G] body has no split in
        the parent's (numel, rnumel) frame, so the planner must skip coalescing
        analysis rather than ask for one that cannot be built."""
        B, D, G1, G2 = 64, 512, 8, 4

        def f(x, w):
            n = _rmsnorm(x * w)
            g1 = n.reshape(B, D // G1, G1).abs().amax(dim=-1)
            g2 = n.reshape(B, D // G2, G2).abs().amax(dim=-1)
            return n.reshape(B, D // G1, G1) / g1.unsqueeze(-1), g1, g2

        args = (torch.randn(B, D, device=GPU_TYPE), torch.randn(D, device=GPU_TYPE))
        self.check_nested_matches_unnested(f, args)

    def test_small_outer_reduction_fuses(self):
        self._norm_block_reduce(_rmsnorm, "amax", 4, 128, 16)

    @parametrize("reduce_fn,G", [("argmax", 128), ("var", 128)])
    def test_reject_unsupported_reduction_type(self, reduce_fn, G):
        """argmax/var need special accumulator handling."""
        rfn = getattr(torch.Tensor, reduce_fn)
        kw = {"correction": 0} if reduce_fn == "var" else {}

        def f(x):
            return rfn(_rmsnorm(x).reshape(4, -1, G), dim=-1, **kw)

        self._check_rejected(f, (torch.randn(4, 4096, device=GPU_TYPE),))

    def test_reject_three_iter_dims(self):
        """[B, H, groups, G] needs explicit 3D mapping."""

        def f(x):
            return _rmsnorm(x.reshape(8, 1024)).reshape(4, 2, 8, 128).abs().amax(dim=-1)

        self._check_rejected(f, (torch.randn(4, 2, 1024, device=GPU_TYPE),))

    def test_reject_x_grouped_reduction_with_three_iter_dims(self):
        """[B, H, K, D].sum(dim=2) needs explicit 3D X-axis mapping."""
        B, H, K, D = 2, 3, 16, 512

        def f(x, w):
            x_normed = _rmsnorm(x.reshape(B * H * K, D)).reshape(B, H, K, D)
            return (w[:, :, :, None] * x_normed).sum(dim=2)

        x = torch.randn(B, H, K, D, device=GPU_TYPE)
        w = torch.randn(B, H, K, device=GPU_TYPE)
        self._check_rejected(f, (x, w))

    def test_reject_transposed_parent_reduction_broadcast(self):
        B, D, G = 64, 64, 16

        def f(x):
            column_sum = x.sum(dim=0)
            expanded = column_sum[None, :].expand(B, D)
            return expanded.reshape(B, D // G, G).sum(dim=-1)

        self._check_rejected(f, (torch.randn(B, D, device=GPU_TYPE),))

    def test_reject_transposed_grouped_pointwise_producer(self):
        B, D, G = 64, 64, 16

        def f(x, bias):
            column_sum = x.sum(dim=0)
            grouped = torch.ops._inductor_test.realize(
                column_sum[None, :].expand(B, D).reshape(B, D // G, G) + bias
            )
            return grouped.sum(dim=-1)

        args = (
            torch.randn(B, D, device=GPU_TYPE),
            torch.randn(B, D // G, G, device=GPU_TYPE),
        )
        self._check_rejected(f, args)

    def test_reject_transposed_grouped_internal_read(self):
        B = groups = G = 16
        D = groups * G

        def f(x, weight):
            normalized = F.rms_norm(x, (D,), weight)
            grouped = normalized.view(B, groups, G)
            scale = grouped.abs().amax(dim=-1)
            output = grouped / (scale.T.abs() + 1e-3).unsqueeze(-1)
            return output.view(B, D), scale

        args = (
            torch.randn(B, D, device=GPU_TYPE),
            torch.randn(D, device=GPU_TYPE),
        )
        self.check_numeric(f, args)
        self.assertEqual(metrics.codegen_nested_reduction, 1)
        self.assertEqual(metrics.generated_kernel_count, 2)

    def test_reject_parent_read_from_local_reduction_input(self):
        B, D, G = 64, 512, 32

        def f(x):
            row_max = x.amax(dim=-1, keepdim=True)
            centered = (x - row_max).exp()
            row_sum = (centered * x).sum(dim=-1)
            block_max = centered.view(B, D // G, G).amax(dim=-1)
            return centered, row_sum, block_max

        compiled = torch.compile(f)
        compiled(torch.randn(B, D, device=GPU_TYPE))
        x = torch.randn(B, D, device=GPU_TYPE) * 3.0
        self.assertEqual(compiled(x), f(x), atol=1e-4, rtol=1e-4)
        self.assertGreater(metrics.generated_kernel_count, 1)

    # The group must stay a real reduction for the ambiguity to exist, and the
    # default threshold unrolls a group this small into pointwise ops.
    @inductor_config.patch("unroll_reductions_threshold", 4)
    def test_reject_ambiguous_reduced_sub_parent_domain(self):
        """G equal to a sub-parent factor makes one lane match the reduced output."""
        B, D, G = 64, 512, 4

        def f(x, weight):
            normalized = F.rms_norm(x, (D,), weight)
            groups = normalized.view(B, D // G, G)
            amax = groups.abs().amax(dim=-1)
            return amax, groups[..., 0] / (amax + 1e-6)

        args = (torch.randn(B, D, device=GPU_TYPE), torch.randn(D, device=GPU_TYPE))
        self._check_rejected(f, args)
        self.assertEqual(metrics.generated_kernel_count, 2)

    def test_reject_multiple_reduce_dims(self):
        """[B, groups, G1, G2] needs one local reduce axis."""

        def f(x):
            return _rmsnorm(x).reshape(4, 32, 16, 8).abs().amax(dim=(-1, -2))

        self._check_rejected(f, (torch.randn(4, 4096, device=GPU_TYPE),))

    def test_reject_split_reduction(self):
        """True split reduction changes total numel."""

        def f(x):
            return x.reshape(4, 4, 512).sum(dim=-1).sum(dim=-1)

        self._check_rejected(f, (torch.randn(4, 2048, device=GPU_TYPE),))


@inductor_config.patch("force_disable_caches", True)
class NestedReductionTest(_NestedReductionBase, TestBase):
    force_persistent_outer_reduction = True


@inductor_config.patch("force_disable_caches", True)
class NestedReductionNonPersistentTest(_NestedReductionBase, TestBase):
    force_persistent_outer_reduction = False


TRITON_KERNEL_RE = re.compile(
    r"(?ms)^@triton_heuristics.*?(?=^@triton_heuristics|^async_compile\.wait|\Z)"
)


def _kernel_name(kernel_code: str) -> str:
    match = re.search(r"^def (triton_[^(]+)\(", kernel_code, re.MULTILINE)
    if match is None:
        raise AssertionError("could not find Triton kernel name")
    return match.group(1)


def _nested_kernel_signature(
    force_persistent_outer_reduction: bool | None,
) -> str | tuple[str, ...]:
    if force_persistent_outer_reduction is False:
        return "triton_red_fused"
    if force_persistent_outer_reduction is True:
        return "triton_per_fused"
    return ("triton_red_fused", "triton_per_fused")


def _is_wrapper_launched_kernel(wrapper_code: str, kernel_code: str) -> bool:
    return (
        re.search(rf"\b{re.escape(_kernel_name(kernel_code))}\b", wrapper_code)
        is not None
    )


def _run_and_capture_source_bundle(
    f,
    args,
    kernel_signature: str | tuple[str, ...],
    *,
    dynamic: bool = False,
    force_persistent_outer_reduction: bool | None = None,
) -> tuple[str, list[str]]:
    def capture():
        with (
            inductor_config.patch("triton.nested_reduction", True),
            _choices_context(force_persistent_outer_reduction),
        ):
            compiled = torch.compile(f, dynamic=dynamic)
            return compiled(*args)

    with fresh_inductor_cache():
        _, source_codes = run_and_get_code(capture)
    metrics.reset()
    torch._dynamo.reset()

    combined_code = "\n\n".join(source_codes)
    wrapper_code = next(code for code in source_codes if get_func_call() in code)
    kernel_signatures = (
        (kernel_signature,) if isinstance(kernel_signature, str) else kernel_signature
    )
    kernel_codes = [
        kernel_code
        for kernel_code in TRITON_KERNEL_RE.findall(combined_code)
        if any(signature in kernel_code for signature in kernel_signatures)
        and _is_wrapper_launched_kernel(wrapper_code, kernel_code)
    ]
    return wrapper_code, kernel_codes


def _run_and_capture_sources(
    f,
    args,
    kernel_signature: str | tuple[str, ...],
    *,
    dynamic: bool = False,
    force_persistent_outer_reduction: bool | None = None,
) -> tuple[str, str]:
    wrapper_code, kernel_codes = _run_and_capture_source_bundle(
        f,
        args,
        kernel_signature,
        dynamic=dynamic,
        force_persistent_outer_reduction=force_persistent_outer_reduction,
    )
    if len(kernel_codes) != 1:
        nested_kernel_codes = [
            code
            for code in kernel_codes
            if "'min_xblock':" in code or "'min_rblock':" in code
        ]
        if len(nested_kernel_codes) == 1:
            kernel_codes = nested_kernel_codes
    if len(kernel_codes) != 1:
        raise AssertionError(
            f"expected exactly one fused kernel matching {kernel_signature!r}, "
            f"got {len(kernel_codes)}: "
            f"{[_kernel_name(kernel_code) for kernel_code in kernel_codes]}"
        )
    return wrapper_code, kernel_codes[0]


def _capture_layernorm_block_amax_kernel_sources(
    batch_size: int,
    D: int,
    G: int,
    *,
    norm_kind: str = "layernorm",
    reduction: str = "amax",
    force_persistent_outer_reduction: bool | None = None,
) -> tuple[str, str]:
    def f(x, G):
        if norm_kind == "layernorm":
            mean = x.mean(dim=-1, keepdim=True)
            var = x.var(dim=-1, keepdim=True, correction=0)
            x_normed = (x - mean) / torch.sqrt(var + 1e-6)
        else:
            rms = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + 1e-6)
            x_normed = x / rms

        grouped = x_normed.reshape(x.shape[0], x.shape[1] // G, G)
        if reduction == "amax":
            return grouped.abs().amax(dim=-1)
        if reduction == "sum":
            return grouped.sum(dim=-1)
        if reduction == "amin":
            return grouped.amin(dim=-1)
        raise AssertionError(f"unsupported reduction: {reduction}")

    x = torch.randn(batch_size, D, device=GPU_TYPE)
    return _run_and_capture_sources(
        f,
        (x, G),
        _nested_kernel_signature(force_persistent_outer_reduction),
        force_persistent_outer_reduction=force_persistent_outer_reduction,
    )


def _capture_dynamic_layernorm_block_amax_kernel_sources(
    batch_size: int, *, force_persistent_outer_reduction: bool | None = None
) -> tuple[str, str]:
    def f(x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, correction=0)
        x_normed = (x - mean) / torch.sqrt(var + 1e-6)
        return x_normed.reshape(x.shape[0], x.shape[1] // 16, 16).abs().amax(dim=-1)

    x = torch.randn(batch_size, 4096, device=GPU_TYPE)
    return _run_and_capture_sources(
        f,
        (x,),
        "triton_red_fused",
        dynamic=True,
        force_persistent_outer_reduction=force_persistent_outer_reduction,
    )


def _capture_amax_kernel_sources(
    batch_size: int, *, force_persistent_outer_reduction: bool | None = None
) -> tuple[str, str]:
    B, D, G = batch_size, 4096, 16

    def f(x, weight):
        x = F.rms_norm(x, (D,), weight)
        return x.view(B, D // G, G).abs().amax(dim=-1)

    x = torch.randn(B, D, device=GPU_TYPE)
    w = torch.randn(D, device=GPU_TYPE)
    return _run_and_capture_sources(
        f,
        (x, w),
        _nested_kernel_signature(force_persistent_outer_reduction),
        force_persistent_outer_reduction=force_persistent_outer_reduction,
    )


def _capture_producer_scale_kernel_sources(
    batch_size: int, *, force_persistent_outer_reduction: bool | None = None
) -> tuple[str, str]:
    B, D, G = batch_size, 4096, 16

    def f(x, weight):
        x = F.rms_norm(x, (D,), weight)
        x = x.view(B, D // G, G)
        amax = x.abs().amax(dim=-1)
        scale = (amax / 448.0).clamp(min=1e-12).to(torch.float16)
        return scale.float()

    x = torch.randn(B, D, device=GPU_TYPE)
    w = torch.randn(D, device=GPU_TYPE)
    return _run_and_capture_sources(
        f,
        (x, w),
        _nested_kernel_signature(force_persistent_outer_reduction),
        force_persistent_outer_reduction=force_persistent_outer_reduction,
    )


def _capture_fullres_kernel_sources(
    batch_size: int, *, force_persistent_outer_reduction: bool | None = None
) -> tuple[str, str]:
    B, D, G = batch_size, 4096, 128
    qmax = 448.0

    def f(x, weight):
        x = F.rms_norm(x, (D,), weight)
        x_groups = x.view(B, D // G, G)
        amax = x_groups.abs().amax(dim=-1)
        scale = (amax / qmax).clamp(min=1e-12)
        x_quant = (x_groups / scale.unsqueeze(-1)).to(torch.float16)
        return x_quant.view(B, D).float(), scale

    x = torch.randn(B, D, device=GPU_TYPE)
    w = torch.randn(D, device=GPU_TYPE)
    return _run_and_capture_sources(
        f,
        (x, w),
        _nested_kernel_signature(force_persistent_outer_reduction),
        force_persistent_outer_reduction=force_persistent_outer_reduction,
    )


def _capture_rmsnorm_block_scale_swizzle_sources(
    batch_size: int, *, force_persistent_outer_reduction: bool | None = None
) -> tuple[str, str]:
    B, D, G = batch_size, 4096, 32

    def f(x, weight):
        return _rmsnorm_block_scale_swizzle(x, weight, G)

    x = torch.randn(B, D, device=GPU_TYPE)
    weight = torch.randn(D, device=GPU_TYPE)
    return _run_and_capture_sources(
        f,
        (x, weight),
        _nested_kernel_signature(force_persistent_outer_reduction),
        force_persistent_outer_reduction=force_persistent_outer_reduction,
    )


def _capture_rmsnorm_mxfp8_scale_swizzle_sources(
    batch_size: int, *, force_persistent_outer_reduction: bool | None = None
) -> tuple[str, str]:
    B, D, G = batch_size, 4096, 32

    def f(x, weight):
        return _rmsnorm_mxfp8_scale_swizzle(x, weight, G)

    x = torch.randn(B, D, device=GPU_TYPE, dtype=torch.bfloat16)
    weight = torch.ones(D, device=GPU_TYPE, dtype=torch.bfloat16)
    return _run_and_capture_sources(
        f,
        (x, weight),
        _nested_kernel_signature(force_persistent_outer_reduction),
        force_persistent_outer_reduction=force_persistent_outer_reduction,
    )


def _rmsnorm_nvfp4(x, weight):
    B, D = x.shape
    G = 16
    x = F.rms_norm(x, (D,), weight)
    x = x.view(B, D // G, G)
    amax = x.abs().amax(dim=-1)
    scale = (amax / 6.0).clamp(min=1e-12, max=448.0).to(torch.float8_e4m3fn)
    xg = x.view(B, D // G, G // 2, 2)
    inv_scale = 1.0 / scale.float().unsqueeze(-1)
    even = xg[..., 0].float() * inv_scale
    odd = xg[..., 1].float() * inv_scale
    packed = inline_asm_elementwise(
        even,
        odd,
        asm_str=E2M1X2_PACK_ASM,
        constraints="=r,f,f",
        dtype=torch.int32,
        is_pure=True,
        pack=1,
    )
    return packed.to(torch.uint8).view(B, D // 2), scale.view(B, D // G)


def _capture_nvfp4_kernel_sources(
    batch_size: int, *, force_persistent_outer_reduction: bool | None = None
) -> tuple[str, str]:
    B, D = batch_size, 4096

    x = torch.randn(B, D, device=GPU_TYPE, dtype=torch.bfloat16)
    w = torch.randn(D, device=GPU_TYPE, dtype=torch.bfloat16)
    with inductor_config.patch(emulate_precision_casts=True):
        return _run_and_capture_sources(
            _rmsnorm_nvfp4,
            (x, w),
            _nested_kernel_signature(force_persistent_outer_reduction),
            force_persistent_outer_reduction=force_persistent_outer_reduction,
        )


def _capture_nvfp4_scale_swizzle_kernel_sources(
    batch_size: int,
    hidden_size: int,
    *,
    force_persistent_outer_reduction: bool | None = None,
) -> tuple[str, str]:
    B, D = batch_size, hidden_size

    def f(x, weight):
        packed, scale = _rmsnorm_nvfp4(x, weight)
        return packed, _swizzle_scale(scale)

    x = torch.randn(B, D, device=GPU_TYPE, dtype=torch.bfloat16)
    w = torch.randn(D, device=GPU_TYPE, dtype=torch.bfloat16)
    with inductor_config.patch(emulate_precision_casts=True):
        return _run_and_capture_sources(
            f,
            (x, w),
            _nested_kernel_signature(force_persistent_outer_reduction),
            force_persistent_outer_reduction=force_persistent_outer_reduction,
        )


def _rmsnorm_mxfp4(x, weight, G):
    from torch._inductor import inductor_prims

    B, D = x.shape
    x = F.rms_norm(x, (D,), weight)
    x = x.view(B, D // G, G)
    amax = x.abs().amax(dim=-1)
    scale = inductor_prims.cvt_e8m0_rceil((amax / 6.0).clamp(min=1e-12)).view(B, D // G)
    xg = x.view(B, D // G, G // 2, 2)
    inv_scale = inline_asm_elementwise(
        scale.to(torch.int32),
        asm_str=MXFP4_RECIP_UE8M0_ASM,
        constraints="=f,r",
        dtype=torch.float32,
        is_pure=True,
        pack=1,
    ).unsqueeze(-1)
    even = xg[..., 0].float() * inv_scale
    odd = xg[..., 1].float() * inv_scale
    packed = inline_asm_elementwise(
        even,
        odd,
        asm_str=E2M1X2_PACK_ASM,
        constraints="=r,f,f",
        dtype=torch.int32,
        is_pure=True,
        pack=1,
    )
    return packed.to(torch.uint8).view(B, D // 2), scale


def _capture_mxfp4_kernel_sources(
    batch_size: int, *, force_persistent_outer_reduction: bool | None = None
) -> tuple[str, str]:
    B, D, G = batch_size, 4096, 32

    def f(x, weight):
        return _rmsnorm_mxfp4(x, weight, G)

    x = torch.randn(B, D, device=GPU_TYPE, dtype=torch.bfloat16)
    w = torch.randn(D, device=GPU_TYPE, dtype=torch.bfloat16)
    return _run_and_capture_sources(
        f,
        (x, w),
        _nested_kernel_signature(force_persistent_outer_reduction),
        force_persistent_outer_reduction=force_persistent_outer_reduction,
    )


def _capture_standalone_nvfp4_kernel_sources(
    batch_size: int, *, force_persistent_outer_reduction: bool | None = None
) -> tuple[str, str]:
    B, D, G = batch_size, 4096, 16

    def f(x):
        xg = x.view(B, D // G, G)
        amax = xg.float().abs().amax(dim=-1)
        scale = (amax / 6.0).clamp(min=1e-12, max=448.0).to(torch.float8_e4m3fn)
        xg = xg.view(B, D // G, G // 2, 2)
        scale_f = scale.float().unsqueeze(-1)
        even = xg[..., 0].float() / scale_f
        odd = xg[..., 1].float() / scale_f
        packed = inline_asm_elementwise(
            even,
            odd,
            asm_str=E2M1X2_PACK_ASM,
            constraints="=r,f,f",
            dtype=torch.int32,
            is_pure=True,
            pack=1,
        )
        return packed.to(torch.uint8).view(B, D // 2), scale.view(B, D // G)

    x = torch.randn(B, D, device=GPU_TYPE, dtype=torch.bfloat16)
    return _run_and_capture_sources(
        f,
        (x,),
        _nested_kernel_signature(force_persistent_outer_reduction),
        force_persistent_outer_reduction=force_persistent_outer_reduction,
    )


def _capture_mxfp6_four_to_three_pack_sources(
    batch_size: int, *, force_persistent_outer_reduction: bool | None = None
) -> tuple[str, str]:
    B, D = batch_size, 1024
    x = torch.randn(B, D, device=GPU_TYPE, dtype=torch.bfloat16)
    return _run_and_capture_sources(
        _mxfp6_four_to_three_quantize,
        (x,),
        _nested_kernel_signature(force_persistent_outer_reduction),
        force_persistent_outer_reduction=force_persistent_outer_reduction,
    )


def _capture_mxfp6_internal_source_sources(
    batch_size: int, *, force_persistent_outer_reduction: bool | None = None
) -> tuple[str, str]:
    x = torch.randn(batch_size, 1024, device=GPU_TYPE, dtype=torch.bfloat16)
    return _run_and_capture_sources(
        _mxfp6_internal_source_full_resolution_fork,
        (x,),
        _nested_kernel_signature(force_persistent_outer_reduction),
        force_persistent_outer_reduction=force_persistent_outer_reduction,
    )


def _capture_standalone_sub_parent_epilogue_sources(
    batch_size: int,
    variant: str = "plain",
    *,
    force_persistent_outer_reduction: bool | None = None,
) -> tuple[str, str]:
    B = batch_size
    D = 510 if variant == "dynamic_r" else 1024
    G = 16

    def f(x):
        if variant == "dynamic_r":
            batch, dim = x.shape
            scale = (x.float().abs().amax(dim=-1) / 6.0).clamp(min=1e-12, max=448.0)
            pairs = x.view(batch, dim // 2, 2)
            scale_f = scale.unsqueeze(-1)
            return (
                pairs[..., 0].float() / scale_f,
                pairs[..., 1].float() / scale_f,
                scale,
            )

        if variant == "pointwise_producer":
            x = torch.nn.functional.gelu(x)
        xg = x.view(B, D // G, G)
        amax = xg.float().abs().amax(dim=-1)
        scale = (amax / 6.0).clamp(min=1e-12, max=448.0)
        xg = xg.view(B, D // G, G // 2, 2)
        scale_f = scale.unsqueeze(-1)
        if variant == "pointwise_producer":
            return xg[..., 0] / scale_f, xg[..., 1] / scale_f
        even = ((xg[..., 0].float() / scale_f).to(torch.float16) + 1.0).float()
        odd = ((xg[..., 1].float() / scale_f).to(torch.float16) - 1.0).float()
        return even, odd, scale

    x = torch.randn(B, D, device=GPU_TYPE, dtype=torch.bfloat16)
    if variant == "dynamic_r":
        torch._dynamo.mark_dynamic(x, 1)
    return _run_and_capture_sources(
        f,
        (x,),
        _nested_kernel_signature(
            None if variant == "dynamic_r" else force_persistent_outer_reduction
        ),
        force_persistent_outer_reduction=force_persistent_outer_reduction,
    )


def _capture_bf16_layernorm_block_amax_epilogue_sources(
    batch_size: int, *, force_persistent_outer_reduction: bool | None = None
) -> tuple[str, str]:
    def f(x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, correction=0)
        x_normed = (x - mean) / torch.sqrt(var + 1e-6)
        return (
            x_normed.reshape(x.shape[0], x.shape[1] // 16, 16)
            .abs()
            .amax(dim=-1)
            .to(torch.bfloat16)
        )

    x = torch.randn(batch_size, 4096, device=GPU_TYPE)
    return _run_and_capture_sources(
        f,
        (x,),
        _nested_kernel_signature(force_persistent_outer_reduction),
        force_persistent_outer_reduction=force_persistent_outer_reduction,
    )


def _capture_layernorm_block_amax_pointwise_epilogue_sources(
    batch_size: int, *, force_persistent_outer_reduction: bool | None = None
) -> tuple[str, str]:
    def f(x, scale, bias):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, correction=0)
        x_normed = (x - mean) / torch.sqrt(var + 1e-6)
        out = x_normed.reshape(x.shape[0], x.shape[1] // 16, 16).abs().amax(dim=-1)
        return out * scale + bias

    x = torch.randn(batch_size, 4096, device=GPU_TYPE)
    scale = torch.randn(batch_size, 256, device=GPU_TYPE)
    bias = torch.randn(batch_size, 256, device=GPU_TYPE)
    return _run_and_capture_sources(
        f,
        (x, scale, bias),
        _nested_kernel_signature(force_persistent_outer_reduction),
        force_persistent_outer_reduction=force_persistent_outer_reduction,
    )


class _InternalsBase:
    force_persistent_outer_reduction: bool | None = None

    def setUp(self):
        super().setUp()
        metrics.reset()
        torch._dynamo.utils.clear_compilation_metrics()

    def looped_or_persistent(self, looped, persistent):
        return looped if self.force_persistent_outer_reduction is False else persistent

    def check_code(
        self,
        code_str,
        num_kernels,
        num_allocs: int | None = None,
        num_deallocs: int | None = None,
    ):
        FileCheck().check(get_func_call()).check_count(
            get_kernel_launch(),
            num_kernels,
            exactly=True,
        ).run(code_str)
        if num_allocs is not None:
            FileCheck().check(get_func_call()).check_count(
                "empty_strided", num_allocs, exactly=True
            ).run(code_str)
        if num_deallocs is not None and not inductor_config.cpp_wrapper:
            FileCheck().check(get_func_call()).check_count(
                "del ", num_deallocs, exactly=True
            ).run(code_str)

    def check_kernel_io_counts(
        self,
        kernel_code: str,
        *,
        input_counts: dict[int, int],
        num_outputs: int,
        num_store_instructions: int | None = None,
    ) -> None:
        load_ids = [int(i) for i in re.findall(r"tl\.load\(in_ptr(\d+)\b", kernel_code)]
        output_load_ids = re.findall(
            r"tl\.load\(((?:out|in_out)_ptr\d+)\b", kernel_code
        )
        store_ids = re.findall(r"tl\.store\(((?:out|in_out)_ptr\d+)\b", kernel_code)
        actual_input_counts = {
            idx: load_ids.count(idx) for idx in sorted(set(load_ids))
        }
        self.assertEqual(actual_input_counts, input_counts)
        self.assertEqual(len(output_load_ids), 0)
        if num_store_instructions is None:
            num_store_instructions = num_outputs
        self.assertEqual(len(store_ids), num_store_instructions)
        self.assertEqual(len(set(store_ids)), num_outputs)

    def check_kernel_meta(
        self, kernel_code: str, *, num_inputs: int, num_stores: int
    ) -> None:
        FileCheck().check_count(
            f"'num_load': {num_inputs}", 1, exactly=True
        ).check_count(f"'num_store': {num_stores}", 1, exactly=True).run(kernel_code)

    def check_axis_classification_contract(
        self,
        kernel_code: str,
        *,
        min_xblock: int | None = None,
        min_rblock: int | None = None,
    ) -> None:
        if min_xblock is None:
            FileCheck().check_not("'min_xblock':").run(kernel_code)
        else:
            FileCheck().check_count(f"'min_xblock': {min_xblock}", 1, exactly=True).run(
                kernel_code
            )
        if min_rblock is None:
            FileCheck().check_not("'min_rblock':").run(kernel_code)
        else:
            FileCheck().check_count(f"'min_rblock': {min_rblock}", 1, exactly=True).run(
                kernel_code
            )

    def assert_single_kernel_form(
        self,
        capture,
        *capture_args,
        input_counts: dict[int, int],
        num_outputs: int,
        num_store_instructions: int | None = None,
        meta_num_load: int | None = None,
        num_allocs: int | None = None,
        num_deallocs: int | None = None,
        min_xblock: int | None = None,
        min_rblock: int | None = None,
        extra_checks: FileCheck | None = None,
    ) -> str:
        wrapper_code, kernel_code = capture(
            *capture_args,
            force_persistent_outer_reduction=self.force_persistent_outer_reduction,
        )
        if num_deallocs is None:
            num_deallocs = len(input_counts)
        self.check_kernel_io_counts(
            kernel_code,
            input_counts=input_counts,
            num_outputs=num_outputs,
            num_store_instructions=num_store_instructions,
        )
        meta_load = (
            meta_num_load if meta_num_load is not None else sum(input_counts.values())
        )
        self.check_kernel_meta(
            kernel_code,
            num_inputs=meta_load,
            num_stores=(
                num_store_instructions
                if num_store_instructions is not None
                else num_outputs
            ),
        )
        if num_allocs is None:
            num_allocs = num_outputs
        self.check_code(
            wrapper_code,
            num_kernels=1,
            num_allocs=num_allocs,
            num_deallocs=num_deallocs,
        )
        self.check_axis_classification_contract(
            kernel_code,
            min_xblock=min_xblock,
            min_rblock=min_rblock,
        )
        if extra_checks is not None:
            extra_checks.run(kernel_code)
        return kernel_code

    def test_layernorm_block_amax_kernel_form(self):
        self.assert_single_kernel_form(
            _capture_layernorm_block_amax_kernel_sources,
            32,
            4096,
            16,
            input_counts=self.looped_or_persistent({0: 2}, {0: 1}),
            num_outputs=1,
            meta_num_load=self.looped_or_persistent(2, 1),
            min_rblock=16,
            extra_checks=(
                FileCheck().check("\n    nested_R0_LOCAL_REDUCTION_SIZE")
                if self.force_persistent_outer_reduction is False
                else None
            ),
        )

    def test_dynamic_layernorm_block_amax_kernel_form(self):
        self.assert_single_kernel_form(
            _capture_dynamic_layernorm_block_amax_kernel_sources,
            32,
            input_counts={0: 2},
            num_outputs=1,
            min_rblock=16,
        )

    def test_nested_kernel_disables_cooperative_reduction(self):
        if self.force_persistent_outer_reduction is False:
            self.skipTest("cooperative reduction only applies to persistent kernels")

        class _CooperativeChoices(InductorChoices):
            @staticmethod
            def should_use_cooperative_reduction(*args, **kwargs):
                return True

            @staticmethod
            def should_use_persistent_reduction(*args, **kwargs):
                return True

        with V.set_choices_handler(_CooperativeChoices()):
            _wrapper_code, kernel_code = _capture_layernorm_block_amax_kernel_sources(
                32,
                4096,
                16,
                force_persistent_outer_reduction=None,
            )

        FileCheck().check_not("rsplit").check_not("RSPLIT").run(kernel_code)
        self.check_axis_classification_contract(kernel_code, min_rblock=16)

    def test_producer_consumer_amax_kernel_form(self):
        self.assert_single_kernel_form(
            _capture_amax_kernel_sources,
            128,
            input_counts=self.looped_or_persistent({0: 2, 1: 1}, {0: 1, 1: 1}),
            num_outputs=1,
            meta_num_load=self.looped_or_persistent(3, 2),
            min_rblock=16,
            extra_checks=FileCheck().check_not("tl.split("),
        )

    @inductor_config.patch("triton.multi_kernel", True)
    def test_producer_consumer_amax_multi_kernel_form(self):
        self.assert_single_kernel_form(
            _capture_amax_kernel_sources,
            128,
            input_counts=self.looped_or_persistent({0: 2, 1: 1}, {0: 1, 1: 1}),
            num_outputs=1,
            meta_num_load=self.looped_or_persistent(3, 2),
            min_rblock=16,
            extra_checks=FileCheck().check_not("tl.split("),
        )

    def test_producer_consumer_scale_kernel_form(self):
        self.assert_single_kernel_form(
            _capture_producer_scale_kernel_sources,
            128,
            input_counts=self.looped_or_persistent({0: 2, 1: 1}, {0: 1, 1: 1}),
            num_outputs=1,
            meta_num_load=self.looped_or_persistent(3, 2),
            min_rblock=16,
        )

    def test_fullres_kernel_form(self):
        with patch(
            "torch._inductor.codegen.simd._SubParentValueResolver",
            side_effect=AssertionError("unexpected sub-parent resolver"),
        ):
            self.assert_single_kernel_form(
                _capture_fullres_kernel_sources,
                128,
                input_counts=self.looped_or_persistent({0: 2, 1: 1}, {0: 1, 1: 1}),
                num_outputs=2,
                meta_num_load=self.looped_or_persistent(3, 2),
                min_rblock=128,
                extra_checks=FileCheck()
                .check_not("tl.split(")
                .check("tl.broadcast_to"),
            )

    def test_rmsnorm_block_scale_swizzle_kernel_form(self):
        self.assert_single_kernel_form(
            _capture_rmsnorm_block_scale_swizzle_sources,
            128,
            input_counts=self.looped_or_persistent({0: 2, 1: 1}, {0: 1, 1: 1}),
            num_outputs=2,
            meta_num_load=self.looped_or_persistent(3, 2),
            min_rblock=32,
            extra_checks=FileCheck().check_regex(
                r"tl\.store\(out_ptr[0-9]+ \+ \(4\*\(x0 // 32\) "
                r"\+ 16\*\(\(x0 % 32\)\)"
            ),
        )

    @skipIfRocm
    @skipIfXpu(msg="NVFP4 inline asm requires CUDA")
    def test_nvfp4_inline_asm_kernel_form(self):
        if torch.cuda.get_device_capability()[0] < 10:
            self.skipTest("NVFP4 inline asm requires SM100+")

        kernel_code = self.assert_single_kernel_form(
            _capture_nvfp4_kernel_sources,
            128,
            input_counts=self.looped_or_persistent({0: 2, 1: 1}, {0: 1, 1: 1}),
            num_outputs=2,
            meta_num_load=self.looped_or_persistent(3, 2),
            min_rblock=16,
            extra_checks=FileCheck()
            .check("tl.split(")
            .check("tl.broadcast_to")
            .check("tl.inline_asm_elementwise")
            .check("cvt.rn.satfinite.e2m1x2.f32"),
        )
        self.assertEqual(kernel_code.count(".to(tl.float8e4nv)"), 1)
        self.assertNotIn(".to(tl.uint8, bitcast=True)", kernel_code)
        self.assertNotIn(").to(tl.float8e4nv, bitcast=True)", kernel_code)

    @skipIfRocm
    @skipIfXpu(msg="NVFP4 inline asm requires CUDA")
    def test_nvfp4_scale_swizzle_reuses_group_scale(self):
        if torch.cuda.get_device_capability()[0] < 10:
            self.skipTest("NVFP4 inline asm requires SM100+")

        kernel_code = self.assert_single_kernel_form(
            _capture_nvfp4_scale_swizzle_kernel_sources,
            128,
            4608,
            input_counts=self.looped_or_persistent({0: 2, 1: 1}, {0: 1, 1: 1}),
            num_outputs=2,
            meta_num_load=self.looped_or_persistent(3, 2),
            min_rblock=16,
            extra_checks=FileCheck().check("4*(x0 // 32) + 16*((x0 % 32))"),
        )
        self.assertEqual(kernel_code.count(".to(tl.float8e4nv)"), 1)
        self.assertRegex(
            kernel_code[kernel_code.index(".to(tl.float8e4nv)") :],
            r"tmp[0-9]+ = \(tmp[0-9]+ / tmp[0-9]+\)\n"
            r"\s+tmp[0-9]+ = tl\.reshape\(tl\.broadcast_to",
        )

    @skipIfRocm
    @skipIfXpu(msg="MXFP4 inline asm requires CUDA")
    def test_mxfp4_inline_asm_kernel_form(self):
        if torch.cuda.get_device_capability()[0] < 10:
            self.skipTest("MXFP4 inline asm requires SM100+")

        self.assert_single_kernel_form(
            _capture_mxfp4_kernel_sources,
            128,
            input_counts=self.looped_or_persistent({0: 2, 1: 1}, {0: 1, 1: 1}),
            num_outputs=2,
            meta_num_load=self.looped_or_persistent(3, 2),
            min_rblock=32,
            extra_checks=FileCheck()
            .check_count("cvt.rp.satfinite.ue8m0x2.f32", 1, exactly=True)
            .check_count("ex2.approx.f32", 1, exactly=True)
            .check_not("libdevice.ldexp")
            .check_count("cvt.rn.satfinite.e2m1x2.f32", 1, exactly=True),
        )

    @skipIfRocm
    def test_rmsnorm_mxfp8_scale_swizzle_kernel_form(self):
        if GPU_TYPE == "cuda":
            if torch.cuda.get_device_capability() < (10, 0):
                self.skipTest("E8M0 inline PTX requires SM100+")
            extra_checks = FileCheck().check_count(
                "cvt.rp.satfinite.ue8m0x2.f32", 1, exactly=True
            )
        else:
            extra_checks = (
                FileCheck()
                .check_not("cvt.rp.satfinite")
                .check("8388607")
                .check_not("cvt.rp.satfinite")
            )

        self.assert_single_kernel_form(
            _capture_rmsnorm_mxfp8_scale_swizzle_sources,
            128,
            input_counts=self.looped_or_persistent({0: 2, 1: 1}, {0: 1, 1: 1}),
            num_outputs=2,
            meta_num_load=self.looped_or_persistent(3, 2),
            min_rblock=32,
            extra_checks=extra_checks,
        )

    def assert_standalone_nvfp4_inline_asm_kernel_form(
        self, force_persistent_outer_reduction: bool | None
    ) -> None:
        def looped_or_persistent(looped, persistent):
            return looped if force_persistent_outer_reduction is False else persistent

        wrapper_code, kernel_code = _capture_standalone_nvfp4_kernel_sources(
            128,
            force_persistent_outer_reduction=force_persistent_outer_reduction,
        )
        self.check_kernel_io_counts(
            kernel_code,
            input_counts=looped_or_persistent({0: 3}, {0: 1}),
            num_outputs=2,
        )
        self.check_kernel_meta(
            kernel_code,
            num_inputs=looped_or_persistent(3, 1),
            num_stores=2,
        )
        self.check_code(wrapper_code, num_kernels=1, num_allocs=2, num_deallocs=1)
        self.check_axis_classification_contract(
            kernel_code,
            min_xblock=None,
            min_rblock=2,
        )
        extra_checks = (
            FileCheck().check_count("tl.split(", 0, exactly=True)
            if force_persistent_outer_reduction is False
            else FileCheck().check_count("tl.split(", 1, exactly=True)
        )
        extra_checks.check("tl.inline_asm_elementwise").check(
            "cvt.rn.satfinite.e2m1x2.f32"
        ).run(kernel_code)

    @skipIfRocm
    @skipIfXpu(msg="NVFP4 inline asm requires CUDA")
    def test_standalone_nvfp4_inline_asm_kernel_form(self):
        if torch.cuda.get_device_capability()[0] < 10:
            self.skipTest("NVFP4 inline asm requires SM100+")

        self.assert_standalone_nvfp4_inline_asm_kernel_form(
            self.force_persistent_outer_reduction
        )

    @skipIfRocm
    @skipIfXpu(msg="NVFP4 inline asm requires CUDA")
    def test_standalone_nvfp4_inline_asm_default_kernel_form(self):
        if torch.cuda.get_device_capability()[0] < 10:
            self.skipTest("NVFP4 inline asm requires SM100+")

        self.assert_standalone_nvfp4_inline_asm_kernel_form(None)

    def test_mxfp6_four_to_three_pack_kernel_form(self):
        self.assert_single_kernel_form(
            _capture_mxfp6_four_to_three_pack_sources,
            32,
            input_counts=self.looped_or_persistent({0: 2}, {0: 1}),
            num_outputs=2,
            num_store_instructions=4,
            num_deallocs=2,
            meta_num_load=self.looped_or_persistent(2, 1),
            min_xblock=None,
            min_rblock=4,
            extra_checks=FileCheck().check_count("tl.split(", 3, exactly=True),
        )

    def test_mxfp6_internal_source_kernel_form(self):
        self.assert_single_kernel_form(
            _capture_mxfp6_internal_source_sources,
            32,
            input_counts=self.looped_or_persistent({0: 2}, {0: 1}),
            num_outputs=3,
            num_store_instructions=5,
            num_deallocs=3,
            meta_num_load=self.looped_or_persistent(2, 1),
            min_xblock=None,
            min_rblock=4,
            extra_checks=FileCheck().check_count("tl.split(", 3, exactly=True),
        )

    def test_standalone_sub_parent_epilogue_kernel_form(self):
        self.assert_single_kernel_form(
            _capture_standalone_sub_parent_epilogue_sources,
            128,
            input_counts=self.looped_or_persistent({0: 3}, {0: 1}),
            num_outputs=3,
            num_deallocs=2,
            meta_num_load=self.looped_or_persistent(3, 1),
            min_xblock=None,
            min_rblock=2,
            extra_checks=(
                FileCheck().check_count("tl.split(", 0, exactly=True)
                if self.force_persistent_outer_reduction is False
                else FileCheck().check_count("tl.split(", 1, exactly=True)
            ),
        )

    @inductor_config.patch("triton.multi_kernel", True)
    def test_standalone_sub_parent_epilogue_multi_kernel_form(self):
        self.assert_single_kernel_form(
            _capture_standalone_sub_parent_epilogue_sources,
            128,
            input_counts=self.looped_or_persistent({0: 3}, {0: 1}),
            num_outputs=3,
            num_deallocs=2,
            meta_num_load=self.looped_or_persistent(3, 1),
            min_xblock=None,
            min_rblock=2,
            extra_checks=(
                FileCheck().check_count("tl.split(", 0, exactly=True)
                if self.force_persistent_outer_reduction is False
                else FileCheck().check_count("tl.split(", 1, exactly=True)
            ),
        )

    @inductor_config.patch(benchmark_kernel=True)
    def test_standalone_sub_parent_epilogue_kernel_num_gb(self):
        B, D, G = 128, 1024, 16
        _wrapper_code, kernel_code = _capture_standalone_sub_parent_epilogue_sources(
            B,
            force_persistent_outer_reduction=self.force_persistent_outer_reduction,
        )
        match = re.search(r"'kernel_num_gb': ([\d.eE+-]+)", kernel_code)
        if match is None:
            raise AssertionError("staged kernel metadata is missing kernel_num_gb")
        minimum_bytes = B * D * 2 + 2 * B * (D // 2) * 4 + B * (D // G) * 4
        self.assertGreaterEqual(float(match.group(1)), minimum_bytes / 1e9)

    def test_pointwise_producer_sub_parent_kernel_form(self):
        self.assert_single_kernel_form(
            _capture_standalone_sub_parent_epilogue_sources,
            128,
            "pointwise_producer",
            input_counts=self.looped_or_persistent({0: 3}, {0: 1}),
            num_outputs=2,
            meta_num_load=self.looped_or_persistent(3, 1),
            min_xblock=None,
            min_rblock=2,
            extra_checks=(
                FileCheck().check_count("tl.split(", 0, exactly=True)
                if self.force_persistent_outer_reduction is False
                else FileCheck().check_count("tl.split(", 1, exactly=True)
            ),
        )

    def test_dynamic_r_sub_parent_kernel_form(self):
        self.assert_single_kernel_form(
            _capture_standalone_sub_parent_epilogue_sources,
            4,
            "dynamic_r",
            input_counts={0: 3},
            num_outputs=3,
            meta_num_load=3,
            num_deallocs=2,
            min_xblock=None,
            min_rblock=2,
            extra_checks=FileCheck().check_count("tl.split(", 0, exactly=True),
        )

    def test_bf16_layernorm_block_amax_epilogue_kernel_form(self):
        self.assert_single_kernel_form(
            _capture_bf16_layernorm_block_amax_epilogue_sources,
            64,
            input_counts=self.looped_or_persistent({0: 2}, {0: 1}),
            num_outputs=1,
            meta_num_load=self.looped_or_persistent(2, 1),
            min_rblock=16,
        )

    def test_layernorm_block_amax_pointwise_epilogue_kernel_form(self):
        self.assert_single_kernel_form(
            _capture_layernorm_block_amax_pointwise_epilogue_sources,
            64,
            input_counts=self.looped_or_persistent(
                {0: 2, 1: 1, 2: 1},
                {0: 1, 1: 1, 2: 1},
            ),
            num_outputs=1,
            meta_num_load=self.looped_or_persistent(4, 3),
            min_rblock=16,
        )


class NestedReductionInternalsPersistentTest(_InternalsBase, TestCase):
    __unittest_skip__ = not HAS_GPU
    force_persistent_outer_reduction = True


class NestedReductionInternalsNonPersistentTest(_InternalsBase, TestCase):
    __unittest_skip__ = not HAS_GPU
    force_persistent_outer_reduction = False


class NestedReductionAOTITest(TestCase):
    __unittest_skip__ = not HAS_GPU

    def test_rmsnorm_block_amax(self):
        B, D, G = 8, 1024, 32

        class Model(torch.nn.Module):
            def forward(self, x):
                normalized = _rmsnorm(x)
                block_amax = normalized.reshape(B, D // G, G).abs().amax(dim=-1)
                return normalized, block_amax

        model = Model()
        x = torch.randn(B, D, device=GPU_TYPE)
        expected = model(x)
        metrics.reset()
        with fresh_inductor_cache():
            exported = torch.export.export(model, (x,))
            package_path = torch._inductor.aoti_compile_and_package(
                exported,
                inductor_configs={
                    "loop_ordering_after_fusion": True,
                    "triton.nested_reduction": True,
                },
            )
            compiled = torch._inductor.aoti_load_package(package_path)
            actual = compiled(x)

        self.assertEqual(actual, expected, atol=1e-2, rtol=1e-2)
        self.assertEqual(metrics.codegen_nested_reduction, 1)


if __name__ == "__main__":
    if HAS_GPU:
        run_tests()
