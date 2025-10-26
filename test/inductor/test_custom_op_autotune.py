# Owner(s): ["module: inductor"]
"""
Tests for custom operation autotuning with PyTorch Inductor.

Users can register custom ops with multiple decomposition implementations and let
Inductor automatically select the best performing variant. Key features tested:

- Name-based input generators (use argument names instead of indices)
- Dynamic shape handling across multiple compilations
- Parametric tuning with tuning_knob for combinatorial parameter exploration
- Numerical correctness and performance validation
"""

import torch
from torch._inductor import config
from torch._inductor.kernel.custom_op import (
    CustomOpConfig,
    register_custom_op_autotuning,
)
from torch._inductor.test_case import run_tests, TestCase
from torch.testing._internal.common_utils import skipIfXpu
from torch.testing._internal.inductor_utils import HAS_GPU


torch.set_float32_matmul_precision("high")


class TestCustomOpAutoTune(TestCase):
    """Test custom operation autotuning functionality."""

    def setUp(self) -> None:
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

    def _assert_implementations_equivalent(self, decompositions, inputs, op_name):
        """Utility to assert that all implementations produce equivalent results."""
        implementations = [(func.__name__, func) for func in decompositions]
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

    def _create_rmsnorm_inputs(self, batch_size=32, seq_len=2048, hidden_dim=512):
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
    def test_rmsnorm_custom_op_autotune_with_dynamic_shape(self):
        """Test RMSNorm autotuning decomposition variants compared to fallback default with dynamic shapes."""
        test_op_name = f"test_lib::rmsnorm_{id(self)}"

        def rmsnorm_decomposition1(
            x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-8
        ) -> torch.Tensor:
            """Variance-based approach: compute variance then rsqrt."""
            variance = x.pow(2).mean(dim=-1, keepdim=True)
            rstd = torch.rsqrt(variance + eps)
            return x * rstd * weight

        def rmsnorm_decomposition2(
            x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-8
        ) -> torch.Tensor:
            x_var = x

            variance = x_var.pow(2).mean(dim=-1, keepdim=True)

            x = x * torch.rsqrt(variance + eps)

            if weight is not None:
                x = x * weight
            return x

        def rmsnorm_decomposition3(
            x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-8
        ) -> torch.Tensor:
            x_squared = x.pow(2)
            variance = x_squared.mean(dim=-1, keepdim=True)

            rstd = torch.rsqrt(variance + eps)
            normalized = x * rstd

            if weight is not None:
                normalized = normalized * weight
            return normalized

        @torch.library.custom_op(test_op_name, mutates_args=())
        def test_rmsnorm_op(
            input_tensor: torch.Tensor, weight: torch.Tensor, eps: float = 1e-8
        ) -> torch.Tensor:
            return torch.nn.functional.rms_norm(
                input_tensor, input_tensor.shape[-1:], weight, eps=eps
            )

        @test_rmsnorm_op.register_fake
        def _(input_tensor: torch.Tensor, weight: torch.Tensor, eps: float = 1e-8):
            return torch.empty_like(input_tensor)

        lib_name, op_name = test_op_name.split("::")
        op_object = getattr(getattr(torch.ops, lib_name), op_name)

        decompositions = [
            rmsnorm_decomposition1,
            rmsnorm_decomposition2,
            rmsnorm_decomposition3,
        ]

        register_custom_op_autotuning(
            op_object.default,
            configs=[CustomOpConfig(decomp) for decomp in decompositions],
            name="test_rmsnorm_autotuned",
            input_gen_fns={
                "x": lambda x: torch.randn_like(x, device=self.device) * 0.02,
                "weight": lambda weight: torch.ones_like(weight, device=self.device),
            },
        )

        # Test multiple shapes to verify dynamic shape handling
        test_shapes = [(2, 16, 128), (8, 32, 256)]

        for i, (batch_size, seq_len, hidden_dim) in enumerate(test_shapes):
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

            # Test numerical equivalence for all decompositions
            self._assert_implementations_equivalent(
                decompositions, (input_tensor, weight), f"RMSNorm_{i}"
            )

            # Test autotuning
            expected = rmsnorm_decomposition1(input_tensor, weight)
            self._run_autotune_test(
                op_object, (input_tensor, weight), expected, f"RMSNorm_{i}"
            )

    @skipIfXpu
    def test_mlp_custom_op_autotune(self):
        """Test MLP autotuning with method parameter controlling different decomposition variants"""
        test_op_name = f"test_lib::mlp_{id(self)}"

        def mlp_variants(
            input_tensor: torch.Tensor,
            gate_weight: torch.Tensor,
            up_weight: torch.Tensor,
            down_weight: torch.Tensor,
            method: int = 0,
        ) -> torch.Tensor:
            """MLP implementation with different computational approaches controlled by method parameter."""

            if method == 0:
                # Separate matmuls: standard implementation with torch.matmul
                gate_proj = torch.matmul(input_tensor, gate_weight)
                up_proj = torch.matmul(input_tensor, up_weight)
                gated = torch.relu(gate_proj) * up_proj
                return torch.matmul(gated, down_weight)

            elif method == 1:
                # Batched approach: uses torch.mm with reshaped tensors
                batch_shape = input_tensor.shape[:-1]
                hidden_dim = input_tensor.shape[-1]
                output_dim = down_weight.shape[-1]

                input_2d = input_tensor.view(-1, hidden_dim)

                gate_proj = torch.mm(input_2d, gate_weight)
                up_proj = torch.mm(input_2d, up_weight)

                gated = torch.relu(gate_proj) * up_proj
                output_2d = torch.mm(gated, down_weight)

                return output_2d.view(*batch_shape, output_dim)

            elif method == 2:
                # Fused weights approach: concatenate then split weights
                # Concatenate gate and up weights for one matrix multiply
                fused_weight = torch.cat([gate_weight, up_weight], dim=1)
                fused_proj = torch.matmul(input_tensor, fused_weight)

                intermediate_dim = gate_weight.shape[1]
                gate_proj, up_proj = fused_proj.split(
                    [intermediate_dim, intermediate_dim], dim=-1
                )

                gated = torch.relu(gate_proj) * up_proj

                return torch.matmul(gated, down_weight)

        @torch.library.custom_op(test_op_name, mutates_args=())
        def test_mlp_op(
            input_tensor: torch.Tensor,
            gate_weight: torch.Tensor,
            up_weight: torch.Tensor,
            down_weight: torch.Tensor,
            method: int = 0,
        ) -> torch.Tensor:
            return mlp_variants(
                input_tensor, gate_weight, up_weight, down_weight, method=method
            )

        @test_mlp_op.register_fake
        def _(
            input_tensor: torch.Tensor,
            gate_weight: torch.Tensor,
            up_weight: torch.Tensor,
            down_weight: torch.Tensor,
            method: int = 0,
        ):
            return torch.empty(
                input_tensor.shape[:-1] + (down_weight.shape[-1],),
                device=input_tensor.device,
                dtype=input_tensor.dtype,
            )

        lib_name, op_name = test_op_name.split("::")
        op_object = getattr(getattr(torch.ops, lib_name), op_name)

        # Use explicit configs with method parameter as tuning knob
        register_custom_op_autotuning(
            op_object.default,
            configs=[
                CustomOpConfig(method=1),  # Batched approach
                CustomOpConfig(method=2),  # Fused weights
            ],
            name="test_mlp_autotuned",
            input_gen_fns={
                "input_tensor": lambda fake_tensor: torch.randn_like(
                    fake_tensor, device=self.device
                )
                * 0.1,
                "gate_weight": lambda fake_tensor: torch.randn_like(
                    fake_tensor, device=self.device
                )
                * 0.05,
                "up_weight": lambda fake_tensor: torch.randn_like(
                    fake_tensor, device=self.device
                )
                * 0.05,
                "down_weight": lambda fake_tensor: torch.randn_like(
                    fake_tensor, device=self.device
                )
                * 0.05,
            },
        )

        # Create test inputs using the original helper method
        input_tensor, gate_weight, up_weight, down_weight = self._create_mlp_inputs()

        # Test that all method variants produce numerically equivalent results
        expected = mlp_variants(
            input_tensor, gate_weight, up_weight, down_weight, method=0
        )

        for method in [1, 2]:
            result = mlp_variants(
                input_tensor, gate_weight, up_weight, down_weight, method=method
            )
            torch.testing.assert_close(
                result,
                expected,
                rtol=1e-5,
                atol=1e-5,
                msg=f"Method {method} not equivalent to method 0",
            )

        # Test autotuning - all should be mathematically equivalent
        self._run_autotune_test(
            op_object,
            (input_tensor, gate_weight, up_weight, down_weight),
            expected,
            "MLP",
        )

    def _create_decompose_k_inputs(self, m=256, k=65536, n=1024):
        """Create test inputs for decompose_k matrix multiplication - divisible by all k_splits values."""
        # Ensure k is divisible by all k_splits values: [2, 32, 64, 128, 256]
        k = ((k + 255) // 256) * 256  # Round up to nearest multiple of 256
        a = torch.randn(m, k, device=self.device, dtype=self.dtype, requires_grad=False)
        b = torch.randn(k, n, device=self.device, dtype=self.dtype, requires_grad=False)
        return a, b

    @skipIfXpu
    def test_decompose_k_custom_op_autotune(self):
        """Test decompose_k autotuning with parameter tuning for k_splits values using decomposition functions."""
        test_op_name = f"test_lib::decompose_k_{id(self)}"

        def decompose_k_implementation(
            a: torch.Tensor, b: torch.Tensor, k_splits: int = 4
        ) -> torch.Tensor:
            """Matrix multiply with k-way decomposition - Python implementation."""
            m = a.shape[0]
            n = b.shape[1]
            k = a.shape[1]

            k_parts = k // k_splits
            B = k_splits

            a_reshaped = torch.permute(
                a.reshape(m, B, k_parts), (1, 0, 2)
            )  # [B, m, k_parts]
            b_reshaped = b.reshape(B, k_parts, n)  # [B, k_parts, n]

            result = torch.bmm(a_reshaped, b_reshaped)  # [B, m, n]

            return torch.sum(result, dim=0)  # [m, n]

        @torch.library.custom_op(test_op_name, mutates_args=())
        def test_decompose_k_op(
            a: torch.Tensor, b: torch.Tensor, k_splits: int = 4
        ) -> torch.Tensor:
            """Matrix multiply with k-way decomposition - custom op using the decomposition."""
            return decompose_k_implementation(a, b, k_splits)

        @test_decompose_k_op.register_fake
        def _(a: torch.Tensor, b: torch.Tensor, k_splits: int = 4):
            return torch.empty(a.shape[0], b.shape[1], device=a.device, dtype=a.dtype)

        lib_name, op_name = test_op_name.split("::")
        op_object = getattr(getattr(torch.ops, lib_name), op_name)

        # Register autotuning with different k_splits values using decomposition function
        register_custom_op_autotuning(
            op_object.default,
            configs=[
                CustomOpConfig(k_splits=32),
                CustomOpConfig(k_splits=64),
                CustomOpConfig(k_splits=128),
                CustomOpConfig(k_splits=256),
            ],
            name="test_decompose_k_autotuned",
            input_gen_fns={
                "a": lambda fake_tensor: torch.randn_like(
                    fake_tensor, device=self.device
                )
                * 0.1,
                "b": lambda fake_tensor: torch.randn_like(
                    fake_tensor, device=self.device
                )
                * 0.1,
            },
        )

        a, b = self._create_decompose_k_inputs()
        expected = a @ b
        self._run_autotune_test(op_object, (a, b), expected, "DecomposeK")

    @skipIfXpu
    def test_multi_parameter_tuning(self):
        """Test autotuning with multiple parameters using scale_mode and chunk_size."""
        op_name = f"test_lib::multi_param_{id(self)}"

        def multi_param_scaling(
            x: torch.Tensor,
            factor: torch.Tensor,
            scale_mode: int = 1,
            chunk_size: int = 16,
        ) -> torch.Tensor:
            """Different scaling approaches controlled by scale_mode parameter."""
            if scale_mode == 1:
                # Simple broadcasting
                return x * factor
            elif scale_mode == 2:
                # Process in chunks
                batch_size, seq_len = x.shape[:2]
                chunks = []
                for start in range(0, seq_len, chunk_size):
                    end = min(start + chunk_size, seq_len)
                    chunk = x[:, start:end]
                    chunks.append(chunk * factor)
                return torch.cat(chunks, dim=1)
            elif scale_mode == 3:
                # Using einsum for scaling
                return torch.einsum("...i,i->...i", x, factor)

        @torch.library.custom_op(op_name, mutates_args=())
        def multi_param_op(
            x: torch.Tensor,
            factor: torch.Tensor,
            scale_mode: int = 1,
            chunk_size: int = 16,
        ) -> torch.Tensor:
            return multi_param_scaling(x, factor, scale_mode, chunk_size)

        @multi_param_op.register_fake
        def _(
            x: torch.Tensor,
            factor: torch.Tensor,
            scale_mode: int = 1,
            chunk_size: int = 16,
        ):
            return torch.empty_like(x)

        lib_name, op_suffix = op_name.split("::")
        op_object = getattr(getattr(torch.ops, lib_name), op_suffix)

        # Use explicit configs with scale_mode and chunk_size parameters as tuning knobs
        register_custom_op_autotuning(
            op_object.default,
            configs=[
                CustomOpConfig(scale_mode=1),  # Broadcast
                CustomOpConfig(scale_mode=2, chunk_size=16),  # Chunked 16
                CustomOpConfig(scale_mode=2, chunk_size=32),  # Chunked 32
                CustomOpConfig(scale_mode=3),  # Einsum
            ],
            name="multi_param_autotuned",
            input_gen_fns={
                "x": lambda t: torch.randn_like(t, device=self.device) * 0.1,
                "factor": lambda t: torch.ones(
                    t.shape[-1], device=self.device, dtype=t.dtype
                ),
            },
        )

        # Create test inputs
        test_x = torch.randn(4, 64, 128, device=self.device, dtype=self.dtype)
        test_factor = torch.ones(128, device=self.device, dtype=self.dtype) * 2.0

        # Verify numerical equivalence across all approaches
        expected_result = test_x * test_factor

        # Test each scale_mode variant
        configs = [
            (1, 16),  # broadcast, chunk_size ignored
            (2, 16),  # chunked with size 16
            (2, 32),  # chunked with size 32
            (3, 16),  # einsum, chunk_size ignored
        ]

        for i, (scale_mode, chunk_size) in enumerate(configs):
            result = multi_param_scaling(
                test_x, test_factor, scale_mode=scale_mode, chunk_size=chunk_size
            )
            torch.testing.assert_close(
                result,
                expected_result,
                rtol=1e-5,
                atol=1e-5,
                msg=f"scale_mode {scale_mode} with chunk_size {chunk_size} not equivalent to expected",
            )

        # Test autotuning
        self._run_autotune_test(
            op_object, (test_x, test_factor), expected_result, "MultiParam"
        )


if __name__ == "__main__":
    run_tests()
