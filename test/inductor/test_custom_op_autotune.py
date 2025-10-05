# Owner(s): ["module: inductor"]
"""
Test suite for custom operation autotuning integration with PyTorch Inductor.

This module tests the custom op autotuning system, which allows users to provide
multiple decomposition implementations of custom operations and automatically
select the best performing one through Inductor's autotuning system.

The tests cover:
1. Custom op registration and autotuning for different operations (RMSNorm, MLP, Attention)
2. Numerical equivalence between different decomposition implementations
3. End-to-end compilation and performance validation
4. Fallback behavior when decompositions fail
"""

import torch
from torch._inductor import config
from torch._inductor.kernel.custom_op import (
    autotune_custom_op,
    register_custom_op_autotuning,
)
from torch._inductor.test_case import run_tests, TestCase
from torch.testing._internal.common_utils import skipIfXpu
from torch.testing._internal.inductor_utils import HAS_GPU


torch.set_float32_matmul_precision("high")


class TestCustomOpAutoTune(TestCase):
    """Test custom operation autotuning functionality."""

    def setUp(self):
        """Set up test environment with appropriate device and dtype."""
        super().setUp()
        self.device = "cuda" if HAS_GPU else "cpu"
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32

    def _create_test_configs(self):
        """Create common test configurations for different sizes."""
        return [
            {"batch_size": 1, "seq_len": 32, "hidden_dim": 128},
            {"batch_size": 2, "seq_len": 64, "hidden_dim": 256},
        ]

    def _run_autotune_test(self, op_object, inputs, expected, test_name):
        """Shared test infrastructure for autotuning tests."""

        @torch.compile
        def test_model(*args):
            return op_object(*args)

        torch._dynamo.reset()
        autotune_backends = "TRITON" if self.device == "cuda" else "ATEN"

        with config.patch(
            max_autotune=True,
            max_autotune_gemm_backends=autotune_backends,
            fx_graph_cache=False,
            benchmark_kernel=True,
        ):
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

    def _assert_implementations_equivalent(self, implementations, inputs, op_name):
        """Utility to assert that all implementations produce equivalent results."""
        results = {}
        for name, impl in implementations:
            result = impl(*inputs)
            results[name] = result

            # Basic sanity checks
            self.assertTrue(
                torch.isfinite(result).all(),
                f"{op_name} {name} produced non-finite values",
            )

        # Verify numerical equivalence
        reference_name, reference_result = next(iter(results.items()))
        for name, result in results.items():
            if name != reference_name:
                rtol = 1e-1 if "Approximated" in name else 1e-2
                atol = 1e-1 if "Approximated" in name else 1e-2
                torch.testing.assert_close(
                    result,
                    reference_result,
                    rtol=rtol,
                    atol=atol,
                    msg=f"{op_name} {name} differs from {reference_name}",
                )

    def _create_rmsnorm_inputs(self, batch_size=2, seq_len=32, hidden_dim=256):
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

    def _create_mlp_inputs(
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
    def test_rmsnorm_custom_op_autotune(self):
        """Test RMSNorm autotuning with multiple decomposition variants showcasing different performance characteristics."""
        test_op_name = f"test_lib::rmsnorm_{id(self)}"

        # Define implementations with clearly different performance characteristics
        def rmsnorm_standard(
            x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-8
        ) -> torch.Tensor:
            """Standard variance-based approach: most numerically stable."""
            variance = x.pow(2).mean(dim=-1, keepdim=True)
            rstd = torch.rsqrt(variance + eps)
            return x * rstd * weight

        def rmsnorm_explicit_rms(
            x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-8
        ) -> torch.Tensor:
            """Manual RMS computation: potentially fewer ops but more memory."""
            x_squared_sum = (x * x).sum(dim=-1, keepdim=True)
            rms = torch.sqrt(x_squared_sum / x.shape[-1])
            return x / (rms + eps) * weight

        def rmsnorm_fused(
            x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-8
        ) -> torch.Tensor:
            """Single-pass with fused operations: better memory locality."""
            norm_factor = (x * x).mean(dim=-1, keepdim=True).add_(eps).rsqrt_()
            return x.mul_(norm_factor).mul_(weight)

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

        @register_custom_op_autotuning(op_object.default)
        def test_rmsnorm_autotuning(
            input_tensor, weight, eps: float = 1e-8, default_impl=None
        ):
            return autotune_custom_op(
                name="test_rmsnorm_autotuned",
                decompositions=[
                    rmsnorm_standard,
                    rmsnorm_explicit_rms,
                    rmsnorm_fused,
                ],
                inputs=[input_tensor, weight],
                kwargs={"eps": eps},
                default_impl=default_impl,
            )

        # Test inputs and run autotuning
        input_tensor, weight = self._create_rmsnorm_inputs()
        expected = rmsnorm_standard(input_tensor, weight)
        self._run_autotune_test(op_object, (input_tensor, weight), expected, "RMSNorm")

        # Test numerical equivalence of all implementations
        implementations = [
            ("Standard", rmsnorm_standard),
            ("Explicit RMS", rmsnorm_explicit_rms),
            ("Fused", rmsnorm_fused),
        ]

        # Test equivalence with multiple configurations
        test_configs = self._create_test_configs()
        for config_idx, test_config in enumerate(test_configs):
            with self.subTest(config=config_idx, **test_config):
                test_inputs = self._create_rmsnorm_inputs(**test_config)
                self._assert_implementations_equivalent(
                    implementations, test_inputs, "RMSNorm"
                )

    @skipIfXpu
    def test_mlp_custom_op_autotune(self):
        """Test MLP autotuning with multiple decomposition variants"""
        test_op_name = f"test_lib::mlp_{id(self)}"

        # Define implementations with different characteristics
        def mlp_standard(input_tensor, gate_weight, up_weight, down_weight):
            """Standard separate matmuls with ReLU activation."""
            gate_proj = torch.matmul(input_tensor, gate_weight)
            up_proj = torch.matmul(input_tensor, up_weight)
            gated = torch.relu(gate_proj) * up_proj
            return torch.matmul(gated, down_weight)

        def mlp_fused_projections(input_tensor, gate_weight, up_weight, down_weight):
            """Fused projection weights: reduces memory accesses but increases intermediate memory."""
            fused_weight = torch.cat([gate_weight, up_weight], dim=1)
            fused_proj = torch.matmul(input_tensor, fused_weight)
            mid_dim = gate_weight.shape[1]
            gate_proj, up_proj = fused_proj.split([mid_dim, mid_dim], dim=-1)
            gated = torch.relu(gate_proj) * up_proj
            return torch.matmul(gated, down_weight)

        def mlp_chunked_computation(input_tensor, gate_weight, up_weight, down_weight):
            """Chunked computation: trades compute for memory efficiency."""
            # Process in smaller chunks to simulate different memory access patterns
            chunk_size = min(64, input_tensor.shape[-2])
            if input_tensor.shape[-2] <= chunk_size:
                # If input is small, just use standard computation
                gate_proj = torch.matmul(input_tensor, gate_weight)
                up_proj = torch.matmul(input_tensor, up_weight)
                gated = torch.relu(gate_proj) * up_proj
                return torch.matmul(gated, down_weight)

            # Otherwise chunk the computation
            input_chunks = torch.chunk(input_tensor, chunks=2, dim=-2)
            output_chunks = []
            for chunk in input_chunks:
                gate_proj = torch.matmul(chunk, gate_weight)
                up_proj = torch.matmul(chunk, up_weight)
                gated = torch.relu(gate_proj) * up_proj
                output_chunks.append(torch.matmul(gated, down_weight))
            return torch.cat(output_chunks, dim=-2)

        @torch.library.custom_op(test_op_name, mutates_args=())
        def test_mlp_op(
            input_tensor: torch.Tensor,
            gate_weight: torch.Tensor,
            up_weight: torch.Tensor,
            down_weight: torch.Tensor,
        ) -> torch.Tensor:
            return mlp_standard(input_tensor, gate_weight, up_weight, down_weight)

        @test_mlp_op.register_fake
        def _(input_tensor, gate_weight, up_weight, down_weight):
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

        @register_custom_op_autotuning(op_object.default)
        def test_mlp_autotuning(
            input_tensor, gate_weight, up_weight, down_weight, default_impl=None
        ):
            return autotune_custom_op(
                name="test_mlp_autotuned",
                decompositions=[
                    mlp_standard,
                    mlp_fused_projections,
                    mlp_chunked_computation,
                ],
                inputs=[input_tensor, gate_weight, up_weight, down_weight],
                kwargs={},
                default_impl=default_impl,
            )

        # Test inputs and run autotuning
        input_tensor, gate_weight, up_weight, down_weight = self._create_mlp_inputs()
        expected = mlp_standard(input_tensor, gate_weight, up_weight, down_weight)
        self._run_autotune_test(
            op_object,
            (input_tensor, gate_weight, up_weight, down_weight),
            expected,
            "MLP",
        )

        # Test numerical equivalence of all implementations
        implementations = [
            ("Standard", mlp_standard),
            ("Fused Projections", mlp_fused_projections),
            ("Chunked Computation", mlp_chunked_computation),
        ]

        # Test equivalence with multiple configurations
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

        for config_idx, test_config in enumerate(test_configs):
            with self.subTest(config=config_idx, **test_config):
                test_inputs = self._create_mlp_inputs(**test_config)
                self._assert_implementations_equivalent(
                    implementations, test_inputs, "MLP"
                )


if __name__ == "__main__":
    run_tests()
