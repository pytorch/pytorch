# Owner(s): ["module: inductor"]

"""End-to-end nested-reduction behavior and kernel-form tests."""

import re

import torch
import torch._inductor.config as inductor_config
from torch._inductor import metrics
from torch._inductor.choices import InductorChoices
from torch._inductor.test_case import run_tests, TestCase
from torch._inductor.utils import fresh_inductor_cache, run_and_get_code
from torch._inductor.virtualized import V
from torch.testing import FileCheck
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
)
from torch.testing._internal.inductor_utils import (
    get_func_call,
    get_kernel_launch,
    GPU_TYPE,
    HAS_GPU,
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
            "triton.nested_reduction", True
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

    def check_fusion(self, expected_kernels=1):
        self.assertEqual(metrics.codegen_nested_reduction, 1)
        if expected_kernels is not None:
            self.assertEqual(metrics.generated_kernel_count, expected_kernels)

    def check_no_fusion(self):
        self.assertEqual(metrics.codegen_nested_reduction, 0)


def _rmsnorm(x_flat):
    return x_flat / torch.sqrt(torch.mean(x_flat * x_flat, dim=-1, keepdim=True) + 1e-6)


def _layernorm(x_flat):
    mean = x_flat.mean(dim=-1, keepdim=True)
    var = x_flat.var(dim=-1, keepdim=True, correction=0)
    return (x_flat - mean) / torch.sqrt(var + 1e-6)


@instantiate_parametrized_tests
class _NestedReductionBase:
    """Tests for fusing dependent cross-axis reductions into a single kernel."""

    # ---- Small dim in X: falls back to existing fusion ----

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
        """B=1 flattened small_dim_in_x still falls back cleanly."""
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

    @parametrize("G", [8, 16])
    def test_rmsnorm_block_amax(self, G):
        self._norm_block_reduce(_rmsnorm, "amax", 128, 8192, G)

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
        """Pointwise after weighted small-dim-in-X reduction falls back."""

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
        """Dynamic small-dim-in-x falls back cleanly."""
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
        import torch.nn.functional as F

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

    # ---- Producer-consumer: node2 reads node1's materialized output ----
    # Instead of node1 and node2 sharing a common input, node2 reads
    # node1's output. This triggers the producer-consumer path in
    # NestedReduction.can_fuse.

    @parametrize("B", [1, 128])
    def test_producer_consumer_rmsnorm_amax(self, B):
        """RMS norm materializes output, amax reads it."""
        import torch.nn.functional as F

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
        import torch.nn.functional as F

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
        import torch.nn.functional as F

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
        """Full-res epilogue producing both FP8 output and a second derived output."""
        import torch.nn.functional as F

        B, D, G = 64, 4096, 128
        fp8_max = torch.finfo(torch.float8_e4m3fn).max

        def f(x, weight):
            x = F.rms_norm(x, (D,), weight)
            x_groups = x.view(B, D // G, G)
            amax = x_groups.abs().amax(dim=-1)
            scale = (amax / fp8_max).clamp(min=1e-12)
            x_fp8 = (x_groups / scale.unsqueeze(-1)).to(torch.float8_e4m3fn)
            return x_fp8.view(B, D).float(), scale

        x = torch.randn(B, D, device=GPU_TYPE)
        w = torch.randn(D, device=GPU_TYPE)
        self.check_numeric(f, (x, w))
        self.check_fusion()

    def test_grouped_reduction_with_weight_mul(self):
        """Grouped reduction input involves element-wise weight multiply."""
        import torch.nn.functional as F

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
        """RMS norm + amax + scale epilogue (clamp + to_fp8)."""
        import torch.nn.functional as F

        B, D, G = 128, 4096, 16

        def f(x, weight):
            x = F.rms_norm(x, (D,), weight)
            x = x.view(B, D // G, G)
            amax = x.abs().amax(dim=-1)
            scale = (amax / 448.0).clamp(min=1e-12).to(torch.float8_e4m3fn)
            return scale.float()

        x = torch.randn(B, D, device=GPU_TYPE)
        w = torch.randn(D, device=GPU_TYPE)
        self.check_numeric(f, (x, w), tol=0.01)
        self.check_fusion()

    @inductor_config.patch(emulate_precision_casts=True)
    @parametrize("B", [128, 1])
    def test_producer_consumer_rmsnorm_fp8_quant(self, B):
        """RMS norm + amax + scale + full-res quantize epilogue."""
        import torch.nn.functional as F

        D, G = 4096, 128
        fp8_max = torch.finfo(torch.float8_e4m3fn).max

        def f(x, weight):
            x = F.rms_norm(x, (D,), weight)
            x_groups = x.view(B, D // G, G)
            amax = x_groups.abs().amax(dim=-1)
            scale = (amax / fp8_max).clamp(min=1e-12)
            x_fp8 = (x_groups / scale.unsqueeze(-1)).to(torch.float8_e4m3fn)
            return x_fp8.view(B, D).float(), scale

        x = torch.randn(B, D, device=GPU_TYPE)
        w = torch.randn(D, device=GPU_TYPE)
        self.check_numeric(f, (x, w))
        self.check_fusion()

    @inductor_config.patch(emulate_precision_casts=True)
    def test_producer_consumer_residual_rmsnorm_fp8_quant(self):
        B, D, G = 128, 2048, 128
        fp8_max = torch.finfo(torch.float8_e4m3fn).max
        fp8_min_scale = 1.0 / (fp8_max * 512.0)

        def f(x, residual, weight):
            h = x.float() + residual.float()
            variance = h.pow(2).mean(dim=-1, keepdim=True)
            normed = h * torch.rsqrt(variance + 1e-6)
            normed_bf16 = normed.to(torch.bfloat16) * weight
            grouped = normed_bf16.view(B, D // G, G)
            absmax = grouped.abs().amax(dim=-1, keepdim=True).float()
            scales = (absmax / fp8_max).clamp(min=fp8_min_scale)
            x_scaled = (grouped / scales).clamp(-fp8_max, fp8_max)
            x_fp8 = x_scaled.to(torch.float8_e4m3fn).view(B, D)
            return x_fp8.float(), scales.squeeze(-1)

        x = torch.randn(B, D, device=GPU_TYPE, dtype=torch.bfloat16)
        residual = torch.randn(B, D, device=GPU_TYPE, dtype=torch.bfloat16)
        w = torch.randn(D, device=GPU_TYPE, dtype=torch.bfloat16)
        self.check_numeric(f, (x, residual, w))
        self.check_fusion()

    @parametrize("B,K,D", [(64, 16, 4096), (1, 16, 1024)])
    def test_no_fullres_epilogue_small_dim_in_x(self, B, K, D):
        """Small-dim-in-X with full-res consumer falls back."""

        def f(x, w):
            x_normed = _rmsnorm(x.reshape(B * K, D)).reshape(B, K, D)
            s = (w[:, :, None] * x_normed).sum(dim=1)
            return x_normed + s[:, None, :]

        x = torch.randn(B, K, D, device=GPU_TYPE)
        w = torch.randn(B, K, device=GPU_TYPE)
        self.check_numeric(f, (x, w))
        self.check_no_fusion()

    def test_epilogue_rejects_intermediate_dependency(self):
        """Do not fuse a pointwise epilogue before another dependent node."""
        import torch.nn.functional as F
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


class NestedReductionTest(_NestedReductionBase, TestBase):
    force_persistent_outer_reduction = True


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


def _nested_kernel_signature(force_persistent_outer_reduction: bool | None) -> str:
    return (
        "triton_red_fused"
        if force_persistent_outer_reduction is False
        else "triton_per_fused"
    )


def _is_wrapper_launched_kernel(wrapper_code: str, kernel_code: str) -> bool:
    return (
        re.search(rf"\b{re.escape(_kernel_name(kernel_code))}\b", wrapper_code)
        is not None
    )


def _run_and_capture_source_bundle(
    f,
    args,
    kernel_signature: str,
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
    kernel_codes = [
        kernel_code
        for kernel_code in TRITON_KERNEL_RE.findall(combined_code)
        if kernel_signature in kernel_code
        and _is_wrapper_launched_kernel(wrapper_code, kernel_code)
    ]
    return wrapper_code, kernel_codes


def _run_and_capture_sources(
    f,
    args,
    kernel_signature: str,
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
    import torch.nn.functional as F

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
    import torch.nn.functional as F

    def f(x, weight):
        x = F.rms_norm(x, (D,), weight)
        x = x.view(B, D // G, G)
        amax = x.abs().amax(dim=-1)
        scale = (amax / 448.0).clamp(min=1e-12).to(torch.float8_e4m3fn)
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
    fp8_max = torch.finfo(torch.float8_e4m3fn).max
    import torch.nn.functional as F

    def f(x, weight):
        x = F.rms_norm(x, (D,), weight)
        x_groups = x.view(B, D // G, G)
        amax = x_groups.abs().amax(dim=-1)
        scale = (amax / fp8_max).clamp(min=1e-12)
        x_fp8 = (x_groups / scale.unsqueeze(-1)).to(torch.float8_e4m3fn)
        return x_fp8.view(B, D).float(), scale

    x = torch.randn(B, D, device=GPU_TYPE)
    w = torch.randn(D, device=GPU_TYPE)
    return _run_and_capture_sources(
        f,
        (x, w),
        _nested_kernel_signature(force_persistent_outer_reduction),
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
    ) -> None:
        load_ids = [int(i) for i in re.findall(r"tl\.load\(in_ptr(\d+)\b", kernel_code)]
        output_load_ids = re.findall(r"tl\.load\(out_ptr(\d+)\b", kernel_code)
        store_ids = re.findall(r"tl\.store\(out_ptr(\d+)\b", kernel_code)
        actual_input_counts = {
            idx: load_ids.count(idx) for idx in sorted(set(load_ids))
        }
        self.assertEqual(actual_input_counts, input_counts)
        self.assertEqual(len(output_load_ids), 0)
        self.assertEqual(len(store_ids), num_outputs)
        self.assertEqual(len(set(store_ids)), num_outputs)

    def check_kernel_meta(
        self, kernel_code: str, *, num_inputs: int, num_outputs: int
    ) -> None:
        FileCheck().check_count(
            f"'num_load': {num_inputs}", 1, exactly=True
        ).check_count(f"'num_store': {num_outputs}", 1, exactly=True).run(kernel_code)

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
        meta_num_load: int | None = None,
        num_allocs: int | None = None,
        num_deallocs: int | None = None,
        min_xblock: int | None = None,
        min_rblock: int | None = None,
        extra_checks: FileCheck | None = None,
    ) -> None:
        wrapper_code, kernel_code = capture(
            *capture_args,
            force_persistent_outer_reduction=self.force_persistent_outer_reduction,
        )
        if num_deallocs is None:
            num_deallocs = len(input_counts)
        self.check_kernel_io_counts(
            kernel_code, input_counts=input_counts, num_outputs=num_outputs
        )
        meta_load = (
            meta_num_load if meta_num_load is not None else sum(input_counts.values())
        )
        self.check_kernel_meta(
            kernel_code,
            num_inputs=meta_load,
            num_outputs=num_outputs,
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
            return

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
        self.assert_single_kernel_form(
            _capture_fullres_kernel_sources,
            128,
            input_counts=self.looped_or_persistent({0: 2, 1: 1}, {0: 1, 1: 1}),
            num_outputs=2,
            meta_num_load=self.looped_or_persistent(3, 2),
            min_rblock=128,
            extra_checks=FileCheck().check_not("tl.split(").check("tl.broadcast_to"),
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


if __name__ == "__main__":
    if HAS_GPU:
        run_tests()
