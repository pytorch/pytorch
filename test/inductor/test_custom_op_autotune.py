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
    register_custom_op_autotuning,
    register_parametric_op_autotuning,
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

    def _create_rmsnorm_inputs(self, batch_size=8, seq_len=1024, hidden_dim=512):
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
            """vLLM-style RMSNorm implementation - variance computation first approach."""
            hidden_size = x.shape[-1]
            x_var = x  # In vLLM, this could be sliced for variance_size_override

            variance = x_var.pow(2).mean(dim=-1, keepdim=True)

            x = x * torch.rsqrt(variance + eps)

            if weight is not None:
                x = x * weight
            return x

        def rmsnorm_decomposition3(
            x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-8
        ) -> torch.Tensor:
            """vLLM-style RMSNorm with extended variance computation pattern."""
            # Get hidden size (following vLLM pattern)
            hidden_size = x.shape[-1]

            x_squared = x.pow(2)
            variance = x_squared.mean(dim=-1, keepdim=True)

            rstd = torch.rsqrt(variance + eps)
            normalized = x * rstd

            # Apply weight scaling
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

        # Example of user-friendly input generation functions
        register_custom_op_autotuning(
            op_object.default,
            decompositions=decompositions,
            name="test_rmsnorm_autotuned",
            input_gen_fns={
                0: lambda fake_tensor: torch.randn_like(fake_tensor, device="cuda")
                * 0.02,  # Small values for input
                1: lambda fake_tensor: torch.ones_like(
                    fake_tensor, device="cuda"
                ),  # Ones for weight
            },
        )

        # Test inputs
        input_tensor, weight = self._create_rmsnorm_inputs()

        # Test numerical equivalence for all decompositions
        self._assert_implementations_equivalent(
            decompositions, (input_tensor, weight), "RMSNorm"
        )

        # Test autotuning
        expected = rmsnorm_decomposition1(input_tensor, weight)
        self._run_autotune_test(op_object, (input_tensor, weight), expected, "RMSNorm")

    @skipIfXpu
    def test_mlp_custom_op_autotune(self):
        """Test MLP autotuning with multiple decomposition variants"""
        test_op_name = f"test_lib::mlp_{id(self)}"

        # Define implementations with different approaches but no intentional inefficiencies
        def mlp_decomposition1(input_tensor, gate_weight, up_weight, down_weight):
            """Separate matmuls: standard implementation with torch.matmul."""
            gate_proj = torch.matmul(input_tensor, gate_weight)
            up_proj = torch.matmul(input_tensor, up_weight)
            gated = torch.relu(gate_proj) * up_proj
            return torch.matmul(gated, down_weight)

        def mlp_decomposition2(input_tensor, gate_weight, up_weight, down_weight):
            """Batched approach: uses torch.mm with reshaped tensors."""
            batch_shape = input_tensor.shape[:-1]
            hidden_dim = input_tensor.shape[-1]
            output_dim = down_weight.shape[-1]

            # Reshape for batched operations
            input_2d = input_tensor.view(-1, hidden_dim)

            # Use torch.mm for potentially better performance
            gate_proj = torch.mm(input_2d, gate_weight)
            up_proj = torch.mm(input_2d, up_weight)

            # Activation and gating
            gated = torch.relu(gate_proj) * up_proj
            output_2d = torch.mm(gated, down_weight)

            # Reshape back
            return output_2d.view(*batch_shape, output_dim)

        def mlp_decomposition3(input_tensor, gate_weight, up_weight, down_weight):
            """Fused weights approach: concatenate then split weights."""
            # Concatenate gate and up weights for one matrix multiply
            fused_weight = torch.cat([gate_weight, up_weight], dim=1)
            fused_proj = torch.matmul(input_tensor, fused_weight)

            # Split the result
            intermediate_dim = gate_weight.shape[1]
            gate_proj, up_proj = fused_proj.split(
                [intermediate_dim, intermediate_dim], dim=-1
            )

            # Apply activation and final projection
            gated = torch.relu(gate_proj) * up_proj
            return torch.matmul(gated, down_weight)

        @torch.library.custom_op(test_op_name, mutates_args=())
        def test_mlp_op(
            input_tensor: torch.Tensor,
            gate_weight: torch.Tensor,
            up_weight: torch.Tensor,
            down_weight: torch.Tensor,
        ) -> torch.Tensor:
            return mlp_decomposition1(
                input_tensor, gate_weight, up_weight, down_weight
            )  # Use one of the defined implementations

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

        decompositions = [
            mlp_decomposition1,
            mlp_decomposition2,
            mlp_decomposition3,
        ]

        register_custom_op_autotuning(
            op_object.default,
            decompositions=decompositions,
            name="test_mlp_autotuned",
            input_gen_fns={
                0: lambda fake_tensor: torch.randn_like(fake_tensor, device="cuda")
                * 0.1,  # Input tensor
                1: lambda fake_tensor: torch.randn_like(fake_tensor, device="cuda")
                * 0.05,  # Gate weight
                2: lambda fake_tensor: torch.randn_like(fake_tensor, device="cuda")
                * 0.05,  # Up weight
                3: lambda fake_tensor: torch.randn_like(fake_tensor, device="cuda")
                * 0.05,  # Down weight
            },
        )

        # Test inputs
        input_tensor, gate_weight, up_weight, down_weight = self._create_mlp_inputs()

        # Test numerical equivalence for all decompositions
        self._assert_implementations_equivalent(
            decompositions, (input_tensor, gate_weight, up_weight, down_weight), "MLP"
        )

        # Test autotuning
        expected = mlp_decomposition1(input_tensor, gate_weight, up_weight, down_weight)
        self._run_autotune_test(
            op_object,
            (input_tensor, gate_weight, up_weight, down_weight),
            expected,
            "MLP",
        )

    def _create_decompose_k_inputs(self, m=256, k=65536, n=1024):
        """Create test inputs for decompose_k matrix multiplication - matching real test sizes."""
        a = torch.randn(m, k, device=self.device, dtype=self.dtype, requires_grad=False)
        b = torch.randn(k, n, device=self.device, dtype=self.dtype, requires_grad=False)
        return a, b

    @skipIfXpu
    def test_decompose_k_custom_op_autotune(self):
        """Test decompose_k autotuning with different k_splits values for matrix multiplication."""
        test_op_name = f"test_lib::decompose_k_{id(self)}"

        def decompose_k_base(
            a: torch.Tensor, b: torch.Tensor, k_splits: int
        ) -> torch.Tensor:
            """Matrix multiply with k-way decomposition - exactly like PyTorch's mm.py implementation."""
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

        import functools

        decompose_k_decomposition1 = functools.partial(
            decompose_k_base, k_splits=2
        )  # split2
        decompose_k_decomposition2 = functools.partial(
            decompose_k_base, k_splits=32
        )  # split32
        decompose_k_decomposition3 = functools.partial(
            decompose_k_base, k_splits=64
        )  # split64
        decompose_k_decomposition4 = functools.partial(
            decompose_k_base, k_splits=128
        )  # split128
        decompose_k_decomposition5 = functools.partial(
            decompose_k_base, k_splits=256
        )  # split256

        # Set names for better debugging
        decompose_k_decomposition1.__name__ = "decompose_k_split2"
        decompose_k_decomposition2.__name__ = "decompose_k_split32"
        decompose_k_decomposition3.__name__ = "decompose_k_split64"
        decompose_k_decomposition4.__name__ = "decompose_k_split128"
        decompose_k_decomposition5.__name__ = "decompose_k_split256"

        @torch.library.custom_op(test_op_name, mutates_args=())
        def test_decompose_k_op(
            a: torch.Tensor, b: torch.Tensor, k_splits: int = 4
        ) -> torch.Tensor:
            return decompose_k_base(a, b, k_splits)

        @test_decompose_k_op.register_fake
        def _(a: torch.Tensor, b: torch.Tensor, k_splits: int = 4):
            return torch.empty(a.shape[0], b.shape[1], device=a.device, dtype=a.dtype)

        lib_name, op_name = test_op_name.split("::")
        op_object = getattr(getattr(torch.ops, lib_name), op_name)

        # Define decompositions with different k_splits values - same as real test
        decompositions = [
            decompose_k_decomposition1,  # k_splits=2
            decompose_k_decomposition2,  # k_splits=32
            decompose_k_decomposition3,  # k_splits=64
            decompose_k_decomposition4,  # k_splits=128
            decompose_k_decomposition5,  # k_splits=256
        ]

        register_custom_op_autotuning(
            op_object.default,
            decompositions=decompositions,
            name="test_decompose_k_autotuned",
            input_gen_fns={
                0: lambda fake_tensor: torch.randn_like(fake_tensor, device="cuda")
                * 0.1,  # Matrix A
                1: lambda fake_tensor: torch.randn_like(fake_tensor, device="cuda")
                * 0.1,  # Matrix B
            },
        )

        # Test inputs
        a, b = self._create_decompose_k_inputs()

        # Test autotuning - autotune will benchmark all decompositions and choose the fastest
        expected = a @ b
        self._run_autotune_test(op_object, (a, b), expected, "DecomposeK")

    @skipIfXpu
    def test_parametric_op_autotune_basic(self):
        """Test parametric autotuning API with equivalent implementation variants."""
        op_name = f"test_lib::parametric_basic_{id(self)}"

        def elementwise_multiply_variants(
            x: torch.Tensor, weight: torch.Tensor, method: int = 0
        ) -> torch.Tensor:
            """Elementwise multiplication with different implementation strategies."""
            if method == 0:
                return x * weight
            elif method == 1:
                return torch.mul(x, weight)
            elif method == 2:
                return x * weight.expand_as(x)
            else:
                raise ValueError(f"Invalid method: {method}")

        @torch.library.custom_op(op_name, mutates_args=())
        def parametric_basic_op(
            x: torch.Tensor, weight: torch.Tensor, method: int = 0
        ) -> torch.Tensor:
            return elementwise_multiply_variants(x, weight, method)

        @parametric_basic_op.register_fake
        def _(x: torch.Tensor, weight: torch.Tensor, method: int = 0):
            return torch.empty_like(x)

        lib_name, op_suffix = op_name.split("::")
        op_object = getattr(getattr(torch.ops, lib_name), op_suffix)

        register_parametric_op_autotuning(
            custom_op=op_object.default,
            implementation_fn=elementwise_multiply_variants,
            parameter_name="method",
            parameter_values=[0, 1, 2],
            name="parametric_basic_autotuned",
            input_gen_fns={
                0: lambda t: torch.randn_like(t, device=self.device) * 0.1,
                1: lambda t: torch.ones_like(t, device=self.device),
            },
        )

        # Validate numerical equivalence across methods - using larger sizes for meaningful perf differences
        test_input = torch.randn(32, 2048, 4096, device=self.device, dtype=self.dtype)
        test_weight = torch.ones(4096, device=self.device, dtype=self.dtype)

        baseline_result = elementwise_multiply_variants(
            test_input, test_weight, method=0
        )
        for method in [1, 2]:
            result = elementwise_multiply_variants(test_input, test_weight, method)
            torch.testing.assert_close(
                result,
                baseline_result,
                rtol=1e-6,
                atol=1e-6,
                msg=f"Method {method} not equivalent to baseline",
            )

        self._run_autotune_test(
            op_object, (test_input, test_weight), baseline_result, "ParametricBasic"
        )

    @skipIfXpu
    def test_parametric_op_autotune_normalization(self):
        """Test parametric autotuning with different normalization algorithms."""
        op_name = f"test_lib::parametric_norm_{id(self)}"
        eps = 1e-5

        def normalization_variants(
            x: torch.Tensor, weight: torch.Tensor, method: int = 0
        ) -> torch.Tensor:
            """Weighted normalization with different mathematical approaches."""
            if method == 0:
                # Layer normalization
                mean = x.mean(dim=-1, keepdim=True)
                var = x.var(dim=-1, keepdim=True, unbiased=False)
                return (x - mean) / torch.sqrt(var + eps) * weight

            elif method == 1:
                # RMS normalization
                rms = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + eps)
                return x / rms * weight

            elif method == 2:
                # Reshaped layer normalization for different memory patterns
                batch_size, seq_len, hidden_dim = x.shape
                x_flat = x.reshape(batch_size * seq_len, hidden_dim)
                weight_flat = weight.expand(batch_size * seq_len, -1)

                mean = x_flat.mean(dim=-1, keepdim=True)
                var = x_flat.var(dim=-1, keepdim=True, unbiased=False)
                normalized = (x_flat - mean) / torch.sqrt(var + eps) * weight_flat

                return normalized.reshape(batch_size, seq_len, hidden_dim)

            elif method == 3:
                # Einstein summation approach
                mean = x.mean(dim=-1, keepdim=True)
                centered = x - mean
                var = torch.mean(centered * centered, dim=-1, keepdim=True)
                normalized = centered / torch.sqrt(var + eps)
                return torch.einsum("bsh,h->bsh", normalized, weight)

            elif method == 4:
                # Chunked processing
                batch_size, seq_len, hidden_dim = x.shape
                chunk_size = hidden_dim // 4

                mean = x.mean(dim=-1, keepdim=True)
                var = x.var(dim=-1, keepdim=True, unbiased=False)
                normalized = (x - mean) / torch.sqrt(var + eps)

                chunks = []
                for start in range(0, hidden_dim, chunk_size):
                    end = min(start + chunk_size, hidden_dim)
                    chunk = normalized[:, :, start:end] * weight[start:end]
                    chunks.append(chunk)

                return torch.cat(chunks, dim=-1)
            else:
                raise ValueError(f"Invalid method: {method}")

        @torch.library.custom_op(op_name, mutates_args=())
        def parametric_norm_op(
            x: torch.Tensor, weight: torch.Tensor, method: int = 0
        ) -> torch.Tensor:
            return normalization_variants(x, weight, method)

        @parametric_norm_op.register_fake
        def _(x: torch.Tensor, weight: torch.Tensor, method: int = 0):
            return torch.empty_like(x)

        lib_name, op_suffix = op_name.split("::")
        op_object = getattr(getattr(torch.ops, lib_name), op_suffix)

        register_parametric_op_autotuning(
            custom_op=op_object.default,
            implementation_fn=normalization_variants,
            parameter_name="method",
            parameter_values=[0, 1, 2, 3, 4],
            name="parametric_norm_autotuned",
            input_gen_fns={
                0: lambda t: torch.randn_like(t, device=self.device) * 0.1,
                1: lambda t: torch.ones_like(t, device=self.device),
            },
        )

        # Validate all methods produce finite results with correct shapes - large workload for perf differences
        test_input = torch.randn(16, 4096, 8192, device=self.device, dtype=self.dtype)
        test_weight = torch.ones(8192, device=self.device, dtype=self.dtype)

        for method in range(5):
            result = normalization_variants(test_input, test_weight, method)
            self.assertTrue(
                torch.isfinite(result).all(),
                f"Method {method} produced non-finite values",
            )
            self.assertEqual(
                result.shape, test_input.shape, f"Method {method} changed tensor shape"
            )

        baseline_result = normalization_variants(test_input, test_weight, method=0)
        self._run_autotune_test(
            op_object, (test_input, test_weight), baseline_result, "ParametricNorm"
        )


if __name__ == "__main__":
    run_tests()
