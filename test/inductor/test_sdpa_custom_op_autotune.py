# Owner(s): ["module: inductor"]
"""
Tests for custom op autotuning with SDPA decompositions.

Validates that custom ops can be autotuned when decompositions involve:
- CompositeImplicitAutograd kernels (e.g. ATen SDPA math backend) that create
  intermediate real tensors during FakeTensorMode tracing
- Multiple input shapes in a single sweep (fallback choice reuse)
- triton_op-based kernels (inlining skip)
"""

import torch
import torch.nn.functional as F
from torch._inductor import config
from torch._inductor.kernel.custom_op import (
    CustomOpConfig,
    register_custom_op_autotuning,
)
from torch._inductor.test_case import run_tests, TestCase
from torch.testing._internal.common_utils import skipIfXpu
from torch.testing._internal.inductor_utils import HAS_GPU


torch.set_float32_matmul_precision("high")


class TestCustomOpAutotuneSDPA(TestCase):
    """Test custom op autotuning with SDPA decompositions."""

    def setUp(self) -> None:
        super().setUp()
        if not HAS_GPU:
            self.skipTest("GPU not available")
        self.device = "cuda"
        self.dtype = torch.bfloat16

    def _run_sdpa_autotune_test(self, op_fn, inputs, expected, test_name):
        """Run a compiled autotune test and verify numerical correctness."""

        @torch.compile(
            options={
                "max_autotune": True,
                "benchmark_kernel": True,
            }
        )
        def test_model(*args):
            return op_fn(*args)

        torch._dynamo.reset()

        with config.patch(fx_graph_cache=False):
            compiled_result = test_model(*inputs)

        self.assertEqual(
            compiled_result.shape, expected.shape, f"{test_name} shape mismatch"
        )
        torch.testing.assert_close(
            compiled_result,
            expected,
            rtol=2e-1,
            atol=5e-1,
            msg=f"{test_name} numerical mismatch",
        )

    @skipIfXpu
    def test_sdpa_math_backend_decomposition(self):
        """Test autotuning with ATen SDPA math backend decomposition.

        This is the key regression test for the FakeTensor fix. When flash/cudnn/
        mem_efficient backends are disabled, ATen SDPA falls back to the math backend
        (_scaled_dot_product_attention_math), which is a C++ CompositeImplicitAutograd
        kernel. During make_fx tracing, this kernel creates intermediate real tensors
        (e.g. scalar_tensor(-inf) for causal masks) that violate FakeTensorMode.

        The fix temporarily patches FakeTensorMode.allow_non_fake_inputs = True
        during make_fx tracing so these intermediate real tensors are tolerated.
        """
        # Disable optimized SDPA backends to force math backend
        orig_flash = torch.backends.cuda.flash_sdp_enabled()
        orig_mem = torch.backends.cuda.mem_efficient_sdp_enabled()
        orig_cudnn = torch.backends.cuda.cudnn_sdp_enabled()

        try:
            torch.backends.cuda.enable_flash_sdp(False)
            torch.backends.cuda.enable_mem_efficient_sdp(False)
            torch.backends.cuda.enable_cudnn_sdp(False)

            test_op_name = f"test_lib::sdpa_math_{id(self)}"

            def decomp_aten_sdpa(
                query: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor,
                is_causal: bool = True,
            ) -> torch.Tensor:
                return F.scaled_dot_product_attention(
                    query, key, value, is_causal=is_causal
                )

            def decomp_manual_sdpa(
                query: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor,
                is_causal: bool = True,
            ) -> torch.Tensor:
                """Manual SDPA implementation using basic ops."""
                scale = query.shape[-1] ** -0.5
                attn = torch.matmul(query * scale, key.transpose(-2, -1))
                if is_causal:
                    L, S = attn.shape[-2], attn.shape[-1]
                    mask = torch.ones(L, S, dtype=torch.bool, device=query.device).tril()
                    attn = attn.masked_fill(~mask, float("-inf"))
                attn = torch.softmax(attn, dim=-1)
                return torch.matmul(attn, value)

            @torch.library.custom_op(test_op_name, mutates_args=())
            def sdpa_op(
                query: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor,
                is_causal: bool = True,
            ) -> torch.Tensor:
                return F.scaled_dot_product_attention(
                    query, key, value, is_causal=is_causal
                )

            @sdpa_op.register_fake
            def _(query, key, value, is_causal=True):
                return torch.empty_like(query)

            register_custom_op_autotuning(
                sdpa_op,
                configs=[
                    CustomOpConfig(decomp_aten_sdpa),
                    CustomOpConfig(decomp_manual_sdpa),
                ],
                name=f"test_sdpa_math_{id(self)}",
                input_gen_fns={
                    "query": lambda t: torch.randn_like(t),
                    "key": lambda t: torch.randn_like(t),
                    "value": lambda t: torch.randn_like(t),
                },
            )

            B, H, L, D = 1, 4, 64, 64
            query = torch.randn(B, H, L, D, device=self.device, dtype=self.dtype)
            key = torch.randn(B, H, L, D, device=self.device, dtype=self.dtype)
            value = torch.randn(B, H, L, D, device=self.device, dtype=self.dtype)

            expected = F.scaled_dot_product_attention(
                query, key, value, is_causal=True
            )

            self._run_sdpa_autotune_test(
                sdpa_op, (query, key, value), expected, "SDPA_math_backend"
            )

        finally:
            torch.backends.cuda.enable_flash_sdp(orig_flash)
            torch.backends.cuda.enable_mem_efficient_sdp(orig_mem)
            torch.backends.cuda.enable_cudnn_sdp(orig_cudnn)
            torch._dynamo.reset()

    @skipIfXpu
    def test_sdpa_multiple_shapes(self):
        """Test autotuning across multiple shapes.

        Validates the _ReusedExternKernelChoice fix: fallback choice must be added
        for every shape, not just the first. Previously, when the extern kernel was
        already registered for shape 1, shape 2 would skip adding the fallback,
        leading to missing choices.
        """
        orig_flash = torch.backends.cuda.flash_sdp_enabled()
        orig_mem = torch.backends.cuda.mem_efficient_sdp_enabled()
        orig_cudnn = torch.backends.cuda.cudnn_sdp_enabled()

        try:
            torch.backends.cuda.enable_flash_sdp(False)
            torch.backends.cuda.enable_mem_efficient_sdp(False)
            torch.backends.cuda.enable_cudnn_sdp(False)

            test_op_name = f"test_lib::sdpa_multi_{id(self)}"

            def decomp_sdpa(query, key, value, is_causal=True):
                return F.scaled_dot_product_attention(
                    query, key, value, is_causal=is_causal
                )

            @torch.library.custom_op(test_op_name, mutates_args=())
            def sdpa_op(
                query: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor,
                is_causal: bool = True,
            ) -> torch.Tensor:
                return F.scaled_dot_product_attention(
                    query, key, value, is_causal=is_causal
                )

            @sdpa_op.register_fake
            def _(query, key, value, is_causal=True):
                return torch.empty_like(query)

            register_custom_op_autotuning(
                sdpa_op,
                configs=[CustomOpConfig(decomp_sdpa)],
                name=f"test_sdpa_multi_{id(self)}",
                input_gen_fns={
                    "query": lambda t: torch.randn_like(t),
                    "key": lambda t: torch.randn_like(t),
                    "value": lambda t: torch.randn_like(t),
                },
            )

            # Test two different shapes
            shapes = [(1, 4, 32, 64), (2, 4, 64, 64)]
            for B, H, L, D in shapes:
                query = torch.randn(B, H, L, D, device=self.device, dtype=self.dtype)
                key = torch.randn(B, H, L, D, device=self.device, dtype=self.dtype)
                value = torch.randn(B, H, L, D, device=self.device, dtype=self.dtype)

                expected = F.scaled_dot_product_attention(
                    query, key, value, is_causal=True
                )

                self._run_sdpa_autotune_test(
                    sdpa_op,
                    (query, key, value),
                    expected,
                    f"SDPA_multi_shape_{B}x{H}x{L}x{D}",
                )
                torch._dynamo.reset()

        finally:
            torch.backends.cuda.enable_flash_sdp(orig_flash)
            torch.backends.cuda.enable_mem_efficient_sdp(orig_mem)
            torch.backends.cuda.enable_cudnn_sdp(orig_cudnn)
            torch._dynamo.reset()


if __name__ == "__main__":
    run_tests()
