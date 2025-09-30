# Owner(s): ["module: inductor"]
import unittest

import torch
from torch._inductor import config
from torch._inductor.kernel.custom_op import (
    autotune_custom_op,
    register_custom_op_lowering,
)
from torch._inductor.lowering import lowerings
from torch._inductor.test_case import run_tests, TestCase
from torch._inductor.utils import run_and_get_code
from torch.testing._internal.common_utils import skipIfXpu
from torch.testing._internal.inductor_utils import GPU_TYPE, HAS_GPU


torch.set_float32_matmul_precision("high")


def rmsnorm_standard(
    input_tensor: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    """
    Standard RMSNorm implementation using variance computation.

    Args:
        input_tensor: Input tensor of shape (..., hidden_dim)
        weight: Weight tensor of shape (hidden_dim,)
        eps: Small constant for numerical stability

    Returns:
        Normalized tensor of same shape as input_tensor
    """
    variance = input_tensor.pow(2).mean(-1, keepdim=True)
    normalized = input_tensor * torch.rsqrt(variance + eps)
    return normalized * weight


def rmsnorm_explicit_rms(
    input_tensor: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    """
    Alternative RMSNorm implementation using explicit RMS computation.

    This variant computes the sum of squares explicitly and then
    calculates the RMS, which may have different numerical properties.

    Args:
        input_tensor: Input tensor of shape (..., hidden_dim)
        weight: Weight tensor of shape (hidden_dim,)
        eps: Small constant for numerical stability

    Returns:
        Normalized tensor of same shape as input_tensor
    """
    sum_sq = (input_tensor * input_tensor).sum(-1, keepdim=True)
    mean_sq = sum_sq / input_tensor.shape[-1]
    rms = torch.sqrt(mean_sq + eps)
    normalized = input_tensor / rms
    return normalized * weight


def rmsnorm_fused(
    input_tensor: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    """
    Fused RMSNorm implementation with combined operations.

    This variant fuses the variance computation and normalization
    into a single expression, potentially enabling better compiler
    optimizations.

    Args:
        input_tensor: Input tensor of shape (..., hidden_dim)
        weight: Weight tensor of shape (hidden_dim,)
        eps: Small constant for numerical stability

    Returns:
        Normalized tensor of same shape as input_tensor
    """
    return (
        input_tensor
        * torch.rsqrt(input_tensor.pow(2).mean(-1, keepdim=True) + eps)
        * weight
    )


def gelu_standard(x: torch.Tensor) -> torch.Tensor:
    """
    Standard GELU implementation using erf.

    GELU(x) = 0.5 * x * (1 + erf(x / sqrt(2)))

    Args:
        x: Input tensor

    Returns:
        GELU activated tensor
    """
    return 0.5 * x * (1.0 + torch.erf(x / (2**0.5)))


def gelu_tanh_approximation(x: torch.Tensor) -> torch.Tensor:
    """
    GELU implementation using tanh approximation.

    GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))

    Args:
        x: Input tensor

    Returns:
        GELU activated tensor
    """
    sqrt_2_over_pi = (2.0 / 3.14159265359) ** 0.5
    return 0.5 * x * (1.0 + torch.tanh(sqrt_2_over_pi * (x + 0.044715 * x.pow(3))))


def gelu_sigmoid_approximation(x: torch.Tensor) -> torch.Tensor:
    """
    GELU implementation using sigmoid approximation.

    GELU(x) ≈ x * sigmoid(1.702 * x)

    Args:
        x: Input tensor

    Returns:
        GELU activated tensor
    """
    return x * torch.sigmoid(1.702 * x)


def sdpa_standard(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale: float = None,
) -> torch.Tensor:
    """Standard Scaled Dot Product Attention implementation."""
    if scale is None:
        scale = query.shape[-1] ** -0.5

    attn_scores = torch.matmul(query, key.transpose(-2, -1)) * scale
    attn_weights = torch.softmax(attn_scores, dim=-1)
    output = torch.matmul(attn_weights, value)
    return output


def sdpa_reordered(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale: float = None,
) -> torch.Tensor:
    """Reordered computation SDPA (pre-scale query)."""
    if scale is None:
        scale = query.shape[-1] ** -0.5

    scaled_query = query * scale
    attn_scores = torch.matmul(scaled_query, key.transpose(-2, -1))
    attn_weights = torch.softmax(attn_scores, dim=-1)
    output = torch.matmul(attn_weights, value)
    return output


def sdpa_explicit_transpose(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale: float = None,
) -> torch.Tensor:
    """Explicit transpose SDPA (compute key transpose first)."""
    if scale is None:
        scale = query.shape[-1] ** -0.5

    key_t = key.transpose(-2, -1)
    attn_scores = torch.matmul(query, key_t) * scale
    attn_weights = torch.softmax(attn_scores, dim=-1)
    output = torch.matmul(attn_weights, value)
    return output


class TestCustomOpAutoTune(TestCase):
    """Test suite for custom op autotuning integration with PyTorch Inductor."""

    def setUp(self):
        """Set up test environment."""
        super().setUp()
        self.device = GPU_TYPE if HAS_GPU else "cpu"
        self.dtype = torch.float16 if HAS_GPU else torch.float32

    def _create_test_inputs(self, batch_size=2, seq_len=64, hidden_dim=256):
        """Create test inputs for RMSNorm operations."""
        input_tensor = torch.randn(
            batch_size,
            seq_len,
            hidden_dim,
            device=self.device,
            dtype=self.dtype,
            requires_grad=False,
        )
        weight = torch.randn(
            hidden_dim, device=self.device, dtype=self.dtype, requires_grad=False
        )
        return input_tensor, weight

    @skipIfXpu
    def test_rmsnorm_custom_op_with_proper_lowering(self):
        """
        Test RMSNorm autotuning using the CORRECT approach with custom op registration.

        This test demonstrates the proper way to use autotune_custom_op within
        an inductor lowering context, not directly in user code.
        """

        test_op_name = f"test_lib::rmsnorm_{id(self)}"

        @torch.library.custom_op(test_op_name, mutates_args=())
        def test_rmsnorm_op(
            input_tensor: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6
        ) -> torch.Tensor:
            """Test custom RMSNorm operation for proper autotuning."""
            return rmsnorm_standard(input_tensor, weight, eps)

        @test_rmsnorm_op.register_fake
        def _(input_tensor: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6):
            return torch.empty_like(input_tensor)

        lib_name, op_name = test_op_name.split("::")
        op_object = getattr(getattr(torch.ops, lib_name), op_name)

        @register_custom_op_lowering(op_object)
        def test_rmsnorm_lowering(input_tensor, weight, eps: float = 1e-6):
            """This is called during inductor lowering - CORRECT CONTEXT for autotune_custom_op"""
            return autotune_custom_op(
                name="test_rmsnorm_autotuned",
                decompositions=[rmsnorm_standard, rmsnorm_explicit_rms, rmsnorm_fused],
                inputs=[input_tensor, weight],
                kwargs={"eps": eps},
            )

        input_tensor, weight = self._create_test_inputs()
        eps = 1e-6

        # Test eager mode
        expected = rmsnorm_standard(input_tensor, weight, eps)

        # Step 4: Test compiled mode with autotuning (inference mode only)
        @torch.compile
        def test_model(x, w):
            return op_object(x, w, eps)

        # Clear cache to ensure fresh compilation
        torch._dynamo.reset()

        with config.patch(
            max_autotune=True,
            max_autotune_gemm_backends="TRITON",
            fx_graph_cache=False,
            benchmark_kernel=True,
        ):
            compiled_result = test_model(input_tensor, weight)

        # Verify correctness
        self.assertEqual(compiled_result.shape, expected.shape)
        torch.testing.assert_close(compiled_result, expected, rtol=1e-2, atol=1e-2)

    def _create_attention_inputs(
        self, batch_size=2, num_heads=8, seq_len=32, head_dim=64
    ):
        """Create test inputs for attention operations."""
        query = torch.randn(
            batch_size,
            num_heads,
            seq_len,
            head_dim,
            device=self.device,
            dtype=self.dtype,
            requires_grad=False,
        )
        key = torch.randn(
            batch_size,
            num_heads,
            seq_len,
            head_dim,
            device=self.device,
            dtype=self.dtype,
            requires_grad=False,
        )
        value = torch.randn(
            batch_size,
            num_heads,
            seq_len,
            head_dim,
            device=self.device,
            dtype=self.dtype,
            requires_grad=False,
        )
        return query, key, value

    # @skipIfXpu
    # def test_sdpa_custom_op_with_proper_lowering(self):
    #     """
    #     Test Scaled Dot Product Attention autotuning with custom op registration.

    #     This test demonstrates autotune_custom_op with a different operation (SDPA)
    #     using three different implementation strategies that can be benchmarked.
    #     """

    #     test_op_name = f"test_lib::sdpa_{id(self)}"

    #     @torch.library.custom_op(test_op_name, mutates_args=())
    #     def test_sdpa_op(
    #         query: torch.Tensor,
    #         key: torch.Tensor,
    #         value: torch.Tensor,
    #         scale: float = None,
    #     ) -> torch.Tensor:
    #         """Test custom SDPA operation for proper autotuning."""
    #         return sdpa_standard(query, key, value, scale)

    #     @test_sdpa_op.register_fake
    #     def _(
    #         query: torch.Tensor,
    #         key: torch.Tensor,
    #         value: torch.Tensor,
    #         scale: float = None,
    #     ):
    #         return torch.empty_like(query)

    #     lib_name, op_name = test_op_name.split("::")
    #     op_object = getattr(getattr(torch.ops, lib_name), op_name)

    #     @register_custom_op_lowering(op_object)
    #     def test_sdpa_lowering(query, key, value, scale: float = None):
    #         """SDPA inductor lowering with autotune_custom_op"""
    #         return autotune_custom_op(
    #             name="test_sdpa_autotuned",
    #             decompositions=[sdpa_standard, sdpa_reordered, sdpa_explicit_transpose],
    #             inputs=[query, key, value],
    #             kwargs={"scale": scale},
    #         )

    #     # Create test inputs
    #     query, key, value = self._create_attention_inputs()

    #     # Test reference implementation
    #     expected = sdpa_standard(query, key, value)

    #     # Test compiled mode with autotuning
    #     @torch.compile
    #     def test_model(q, k, v):
    #         return op_object(q, k, v)

    #     # Clear cache to ensure fresh compilation
    #     torch._dynamo.reset()

    #     with config.patch(
    #         max_autotune=True,
    #         max_autotune_gemm_backends="TRITON",
    #         fx_graph_cache=False,
    #         benchmark_kernel=True,
    #     ):
    #         compiled_result = test_model(query, key, value)

    #     # Verify correctness
    #     self.assertEqual(compiled_result.shape, expected.shape)
    #     torch.testing.assert_close(compiled_result, expected, rtol=1e-2, atol=1e-2)

    # def test_sdpa_implementations_numerical_equivalence(self):
    #     """
    #     Test that all SDPA implementations are numerically equivalent.

    #     This validates the correctness of different SDPA decomposition
    #     implementations without involving autotuning.
    #     """
    #     # Test configurations - keep small to avoid memory issues
    #     test_configs = [
    #         {"batch_size": 1, "num_heads": 4, "seq_len": 16, "head_dim": 32},
    #         {"batch_size": 2, "num_heads": 8, "seq_len": 32, "head_dim": 64},
    #     ]

    #     sdpa_implementations = [
    #         ("Standard", sdpa_standard),
    #         ("Reordered", sdpa_reordered),
    #         ("Explicit Transpose", sdpa_explicit_transpose),
    #     ]

    #     for config_idx, test_config in enumerate(test_configs):
    #         with self.subTest(config=config_idx, **test_config):
    #             query, key, value = self._create_attention_inputs(**test_config)

    #             # Test all implementations
    #             results = {}
    #             for name, impl in sdpa_implementations:
    #                 result = impl(query, key, value)
    #                 results[name] = result

    #                 # Verify shape and finiteness
    #                 self.assertEqual(result.shape, query.shape)
    #                 self.assertTrue(
    #                     torch.isfinite(result).all(),
    #                     f"{name} produced non-finite values",
    #                 )

    #             # Verify all results are numerically equivalent
    #             reference_result = results["Standard"]
    #             for name, result in results.items():
    #                 if name != "Standard":
    #                     torch.testing.assert_close(
    #                         result,
    #                         reference_result,
    #                         rtol=1e-2,
    #                         atol=1e-2,
    #                         msg=f"{name} differs from Standard",
    #                     )


if __name__ == "__main__":
    from torch._inductor.utils import is_big_gpu

    # Set env to make it work in CI
    if HAS_GPU and is_big_gpu():
        run_tests()
    else:
        # Run basic tests even on CPU
        run_tests()
