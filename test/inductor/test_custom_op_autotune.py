# Owner(s): ["module: inductor"]
"""
Test suite for custom operation autotuning integration with PyTorch Inductor.

This module tests the custom op autotuning system, which allows users to provide
multiple decomposition implementations of custom operations and automatically
select the best performing one through Inductor's autotuning system.

The tests cover:
1. Custom op registration and autotuning for different operations (RMSNorm, MLP)
2. Numerical equivalence between different decomposition implementations
3. End-to-end compilation and performance validation
"""

import unittest

import torch
from torch._inductor import config
from torch._inductor.kernel.custom_op import (
    autotune_custom_op,
    register_custom_op_lowering,
)
from torch._inductor.test_case import run_tests, TestCase
from torch.testing._internal.common_utils import skipIfXpu
from torch.testing._internal.inductor_utils import GPU_TYPE, HAS_GPU

torch.set_float32_matmul_precision("high")


def rmsnorm_standard(
    x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-8
) -> torch.Tensor:
    """Standard RMSNorm implementation: sqrt(mean(x^2) + eps)."""
    variance = x.pow(2).mean(dim=-1, keepdim=True)
    rstd = torch.rsqrt(variance + eps)
    return x * rstd * weight


def rmsnorm_explicit_rms(
    x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-8
) -> torch.Tensor:
    """RMSNorm using explicit RMS computation without torch.norm to avoid conflicts."""
    # Compute RMS manually: sqrt(sum(x^2) / n)
    x_squared_sum = (x * x).sum(dim=-1, keepdim=True)
    rms = torch.sqrt(x_squared_sum / x.shape[-1])
    return x / (rms + eps) * weight


def rmsnorm_fused(
    x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-8
) -> torch.Tensor:
    """RMSNorm with fused operations for potentially better performance."""
    x_squared = x * x
    mean_x_squared = x_squared.mean(dim=-1, keepdim=True)
    rstd = (mean_x_squared + eps).rsqrt()
    return x * rstd * weight


def gelu_standard(x: torch.Tensor) -> torch.Tensor:
    """Standard GELU: 0.5 * x * (1 + erf(x / sqrt(2)))."""
    return 0.5 * x * (1.0 + torch.erf(x / (2**0.5)))


def gelu_tanh_approximation(x: torch.Tensor) -> torch.Tensor:
    """GELU using tanh approximation for faster computation."""
    sqrt_2_over_pi = (2.0 / 3.14159265359) ** 0.5
    return 0.5 * x * (1.0 + torch.tanh(sqrt_2_over_pi * (x + 0.044715 * x.pow(3))))


def gelu_sigmoid_approximation(x: torch.Tensor) -> torch.Tensor:
    """GELU using sigmoid approximation: x * sigmoid(1.702 * x)."""
    return x * torch.sigmoid(1.702 * x)


def mlp_standard(
    input_tensor: torch.Tensor,
    gate_weight: torch.Tensor,
    up_weight: torch.Tensor,
    down_weight: torch.Tensor,
) -> torch.Tensor:
    """
    Standard MLP with gated activation (SwiGLU-style):
    1. gate_proj = input @ gate_weight
    2. up_proj = input @ up_weight
    3. gated = gelu(gate_proj) * up_proj
    4. output = gated @ down_weight
    """
    gate_proj = torch.matmul(input_tensor, gate_weight)
    up_proj = torch.matmul(input_tensor, up_weight)
    gated = gelu_standard(gate_proj) * up_proj
    output = torch.matmul(gated, down_weight)
    return output


def mlp_fused_projections(
    input_tensor: torch.Tensor,
    gate_weight: torch.Tensor,
    up_weight: torch.Tensor,
    down_weight: torch.Tensor,
) -> torch.Tensor:
    """
    MLP with fused gate and up projections using concatenated weights.
    Computes both projections in a single matmul, then splits the result.
    """
    # Concatenate weights for fused matmul
    fused_weight = torch.cat([gate_weight, up_weight], dim=1)
    fused_proj = torch.matmul(input_tensor, fused_weight)

    # Split into gate and up projections
    intermediate_dim = gate_weight.shape[1]
    gate_proj = fused_proj[..., :intermediate_dim]
    up_proj = fused_proj[..., intermediate_dim:]

    gated = gelu_standard(gate_proj) * up_proj
    output = torch.matmul(gated, down_weight)
    return output


def mlp_approximated_gelu(
    input_tensor: torch.Tensor,
    gate_weight: torch.Tensor,
    up_weight: torch.Tensor,
    down_weight: torch.Tensor,
) -> torch.Tensor:
    """
    MLP using tanh-approximated GELU for potentially faster activation.
    Uses the tanh approximation which may be faster on some hardware.
    """
    gate_proj = torch.matmul(input_tensor, gate_weight)
    up_proj = torch.matmul(input_tensor, up_weight)
    gated = gelu_tanh_approximation(gate_proj) * up_proj
    output = torch.matmul(gated, down_weight)
    return output


class TestCustomOpAutoTune(TestCase):
    """Test custom operation autotuning functionality."""

    def setUp(self):
        """Set up test environment with appropriate device and dtype."""
        super().setUp()
        self.device = "cuda" if HAS_GPU else "cpu"
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32

    def create_rmsnorm_test_inputs(self, batch_size=2, seq_len=32, hidden_dim=256):
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
    def test_rmsnorm_custom_op_autotune(self):
        """
        Test RMSNorm autotuning with multiple decomposition variants.

        This tests the custom op autotuning system with RMSNorm, which has
        three different implementations that can be benchmarked for performance.
        """
        test_op_name = f"test_lib::rmsnorm_{id(self)}"

        @torch.library.custom_op(test_op_name, mutates_args=())
        def test_rmsnorm_op(
            input_tensor: torch.Tensor, weight: torch.Tensor, eps: float = 1e-8
        ) -> torch.Tensor:
            return rmsnorm_standard(input_tensor, weight, eps)

        @test_rmsnorm_op.register_fake
        def _(input_tensor: torch.Tensor, weight: torch.Tensor, eps: float = 1e-8):
            return torch.empty_like(input_tensor)

        lib_name, op_name = test_op_name.split("::")
        op_object = getattr(getattr(torch.ops, lib_name), op_name)

        @register_custom_op_lowering(op_object)
        def test_rmsnorm_lowering(input_tensor, weight, eps: float = 1e-8):
            """RMSNorm inductor lowering with multiple decomposition choices for autotuning."""
            return autotune_custom_op(
                name="test_rmsnorm_autotuned",
                decompositions=[
                    rmsnorm_standard,
                    rmsnorm_explicit_rms,
                    rmsnorm_fused,
                ],
                inputs=[input_tensor, weight],
                kwargs={"eps": eps},
            )

        # Create test inputs
        input_tensor, weight = self.create_rmsnorm_test_inputs()

        # Test eager mode reference
        expected = rmsnorm_standard(input_tensor, weight)

        # Test compiled mode with autotuning
        @torch.compile
        def test_model(inp, w):
            return op_object(inp, w)

        # Clear cache to ensure fresh compilation
        torch._dynamo.reset()

        with config.patch(
            max_autotune=True,
            max_autotune_gemm_backends="TRITON",
            fx_graph_cache=False,
            benchmark_kernel=True,
        ):
            compiled_result = test_model(input_tensor, weight)

        # Verify correctness with relaxed tolerances for autotuning
        self.assertEqual(compiled_result.shape, expected.shape)
        torch.testing.assert_close(compiled_result, expected, rtol=2e-1, atol=5e-1)

    def test_rmsnorm_implementations_numerical_equivalence(self):
        """
        Test that all RMSNorm implementations are numerically equivalent.

        This validates that the different RMSNorm decomposition variants produce
        the same results (within numerical precision) without autotuning.
        """
        test_configs = [
            {"batch_size": 1, "seq_len": 32, "hidden_dim": 128},
            {"batch_size": 2, "seq_len": 64, "hidden_dim": 256},
            {"batch_size": 4, "seq_len": 128, "hidden_dim": 512},
        ]

        rmsnorm_implementations = [
            ("Standard", rmsnorm_standard),
            ("Explicit RMS", rmsnorm_explicit_rms),
            ("Fused", rmsnorm_fused),
        ]

        eps = 1e-6

        for config_idx, test_config in enumerate(test_configs):
            with self.subTest(config=config_idx, **test_config):
                input_tensor, weight = self.create_rmsnorm_test_inputs(**test_config)

                # Test all implementations
                results = {}
                for name, impl in rmsnorm_implementations:
                    result = impl(input_tensor, weight, eps)
                    results[name] = result

                    # Verify shape and finiteness
                    self.assertEqual(result.shape, input_tensor.shape)
                    self.assertTrue(
                        torch.isfinite(result).all(),
                        f"{name} produced non-finite values",
                    )

                # Verify all results are numerically equivalent
                reference_result = results["Standard"]
                for name, result in results.items():
                    if name != "Standard":
                        torch.testing.assert_close(
                            result,
                            reference_result,
                            rtol=1e-2,
                            atol=1e-2,
                            msg=f"{name} differs from Standard",
                        )

    def create_mlp_test_inputs(
        self,
        batch_size=2,
        seq_len=32,
        hidden_dim=512,
        intermediate_dim=1024,
        output_dim=256,
    ):
        """Create test inputs for MLP operations."""
        input_tensor = torch.randn(
            batch_size,
            seq_len,
            hidden_dim,
            device=self.device,
            dtype=self.dtype,
            requires_grad=False,
        )
        gate_weight = torch.randn(
            hidden_dim,
            intermediate_dim,
            device=self.device,
            dtype=self.dtype,
            requires_grad=False,
        )
        up_weight = torch.randn(
            hidden_dim,
            intermediate_dim,
            device=self.device,
            dtype=self.dtype,
            requires_grad=False,
        )
        down_weight = torch.randn(
            intermediate_dim,
            output_dim,
            device=self.device,
            dtype=self.dtype,
            requires_grad=False,
        )
        return input_tensor, gate_weight, up_weight, down_weight

    @skipIfXpu
    def test_mlp_custom_op_autotune(self):
        """
        Test MLP autotuning with multiple decomposition variants.

        This tests a more complex operation (SwiGLU MLP) with multiple implementations:
        1. Standard: Separate matmuls for gate and up projections
        2. Fused: Single matmul with concatenated weights then split
        3. Approximated: Uses tanh-approximated GELU instead of exact GELU
        """
        test_op_name = f"test_lib::mlp_{id(self)}"

        @torch.library.custom_op(test_op_name, mutates_args=())
        def test_mlp_op(
            input_tensor: torch.Tensor,
            gate_weight: torch.Tensor,
            up_weight: torch.Tensor,
            down_weight: torch.Tensor,
        ) -> torch.Tensor:
            return mlp_standard(input_tensor, gate_weight, up_weight, down_weight)

        @test_mlp_op.register_fake
        def _(
            input_tensor: torch.Tensor,
            gate_weight: torch.Tensor,
            up_weight: torch.Tensor,
            down_weight: torch.Tensor,
        ):
            # Output shape: (..., output_dim) where output_dim = down_weight.shape[1]
            batch_shape = input_tensor.shape[:-1]
            output_dim = down_weight.shape[1]
            return torch.empty(
                *batch_shape,
                output_dim,
                dtype=input_tensor.dtype,
                device=input_tensor.device,
            )

        lib_name, op_name = test_op_name.split("::")
        op_object = getattr(getattr(torch.ops, lib_name), op_name)

        @register_custom_op_lowering(op_object)
        def test_mlp_lowering(input_tensor, gate_weight, up_weight, down_weight):
            """MLP inductor lowering with multiple decomposition choices for autotuning."""
            return autotune_custom_op(
                name="test_mlp_autotuned",
                decompositions=[
                    mlp_standard,
                    mlp_fused_projections,
                    mlp_approximated_gelu,
                ],
                inputs=[input_tensor, gate_weight, up_weight, down_weight],
                kwargs={},
            )

        # Create test inputs
        input_tensor, gate_weight, up_weight, down_weight = (
            self.create_mlp_test_inputs()
        )

        # Test eager mode reference
        expected = mlp_standard(input_tensor, gate_weight, up_weight, down_weight)

        # Test compiled mode with autotuning
        @torch.compile
        def test_model(inp, gate_w, up_w, down_w):
            return op_object(inp, gate_w, up_w, down_w)

        # Clear cache to ensure fresh compilation
        torch._dynamo.reset()

        with config.patch(
            max_autotune=True,
            max_autotune_gemm_backends="TRITON",
            fx_graph_cache=False,
            benchmark_kernel=True,
        ):
            compiled_result = test_model(
                input_tensor, gate_weight, up_weight, down_weight
            )

        # Verify correctness with relaxed tolerances for autotuning
        self.assertEqual(compiled_result.shape, expected.shape)
        torch.testing.assert_close(compiled_result, expected, rtol=2e-1, atol=5e-1)

    def test_mlp_implementations_numerical_equivalence(self):
        """
        Test that all MLP implementations are numerically equivalent.

        This validates that the different MLP decomposition variants produce
        the same results (within numerical precision) without autotuning.
        """
        test_configs = [
            {
                "batch_size": 1,
                "seq_len": 16,
                "hidden_dim": 128,
                "intermediate_dim": 256,
                "output_dim": 64,
            },
            {
                "batch_size": 2,
                "seq_len": 32,
                "hidden_dim": 256,
                "intermediate_dim": 512,
                "output_dim": 128,
            },
        ]

        mlp_implementations = [
            ("Standard", mlp_standard),
            ("Fused Projections", mlp_fused_projections),
            ("Approximated GELU", mlp_approximated_gelu),
        ]

        for config_idx, test_config in enumerate(test_configs):
            with self.subTest(config=config_idx, **test_config):
                input_tensor, gate_weight, up_weight, down_weight = (
                    self.create_mlp_test_inputs(**test_config)
                )

                # Test all implementations
                results = {}
                for name, impl in mlp_implementations:
                    result = impl(input_tensor, gate_weight, up_weight, down_weight)
                    results[name] = result

                    # Verify shape and finiteness
                    expected_shape = (*input_tensor.shape[:-1], down_weight.shape[1])
                    self.assertEqual(result.shape, expected_shape)
                    self.assertTrue(
                        torch.isfinite(result).all(),
                        f"{name} produced non-finite values",
                    )

                # Verify numerical equivalence (with relaxed tolerance for approximated GELU)
                reference_result = results["Standard"]
                for name, result in results.items():
                    if name != "Standard":
                        # Use more relaxed tolerance for approximated GELU since it's an approximation
                        rtol = 1e-1 if "Approximated" in name else 1e-2
                        atol = 1e-1 if "Approximated" in name else 1e-2

                        torch.testing.assert_close(
                            result,
                            reference_result,
                            rtol=rtol,
                            atol=atol,
                            msg=f"{name} differs from Standard",
                        )


if __name__ == "__main__":
    run_tests()
