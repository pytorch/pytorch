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
from torch._inductor.kernel.custom_op import register_custom_op_autotuning
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
            """vLLM-style RMSNorm implementation - variance computation first approach."""
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

        register_custom_op_autotuning(
            op_object.default,
            decompositions=decompositions,
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
        """Create test inputs for decompose_k matrix multiplication - divisible by all k_splits values."""
        # Ensure k is divisible by all k_splits values: [2, 32, 64, 128, 256]
        k = ((k + 255) // 256) * 256  # Round up to nearest multiple of 256
        a = torch.randn(m, k, device=self.device, dtype=self.dtype, requires_grad=False)
        b = torch.randn(k, n, device=self.device, dtype=self.dtype, requires_grad=False)
        return a, b

    @skipIfXpu
    def test_decompose_k_custom_op_autotune(self):
        """Test decompose_k autotuning with parameter tuning for k_splits values."""
        test_op_name = f"test_lib::decompose_k_{id(self)}"

        def decompose_k_implementation(
            a: torch.Tensor, b: torch.Tensor, k_splits: int = 4
        ) -> torch.Tensor:
            """Matrix multiply with k-way decomposition - parameter-tuned implementation."""
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
            return decompose_k_implementation(a, b, k_splits)

        @test_decompose_k_op.register_fake
        def _(a: torch.Tensor, b: torch.Tensor, k_splits: int = 4):
            return torch.empty(a.shape[0], b.shape[1], device=a.device, dtype=a.dtype)

        lib_name, op_name = test_op_name.split("::")
        op_object = getattr(getattr(torch.ops, lib_name), op_name)

        register_custom_op_autotuning(
            custom_op=op_object.default,
            decompositions=[decompose_k_implementation],
            tuning_knob={"k_splits": [32, 64]},
            max_autotune_configs={"k_splits": [8, 16, 128, 256, 512]},
            name="test_decompose_k_autotuned",
            input_gen_fns={
                "a": lambda fake_tensor: torch.randn_like(
                    fake_tensor, device=self.device
                )
                * 0.1,  # Matrix A
                "b": lambda fake_tensor: torch.randn_like(
                    fake_tensor, device=self.device
                )
                * 0.1,  # Matrix B
            },
        )

        a, b = self._create_decompose_k_inputs()
        expected = a @ b
        self._run_autotune_test(op_object, (a, b), expected, "DecomposeK")

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

        register_custom_op_autotuning(
            custom_op=op_object.default,
            decompositions=[normalization_variants],
            tuning_knob={"method": [0, 1, 2, 3, 4]},
            name="parametric_norm_autotuned",
            input_gen_fns={
                "x": lambda t: torch.randn_like(t, device=self.device) * 0.1,
                "weight": lambda t: torch.ones_like(t, device=self.device),
            },
        )

        test_input = torch.randn(8, 512, 1024, device=self.device, dtype=self.dtype)
        test_weight = torch.ones(1024, device=self.device, dtype=self.dtype)

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

    @skipIfXpu
    def test_multi_parameter_tuning(self):
        """Test autotuning with multiple parameters - demonstrates Cartesian product."""
        op_name = f"test_lib::multi_param_{id(self)}"

        def multi_param_scaling(
            x: torch.Tensor,
            factor: torch.Tensor,
            scale_mode: int = 1,
            chunk_size: int = 16,
        ) -> torch.Tensor:
            """Simple scaling with two parameters - always mathematically equivalent."""
            batch_size, seq_len = x.shape[:2]

            # All modes produce the same result, just with different computational patterns
            if scale_mode == 1:
                # Simple broadcasting
                return x * factor
            elif scale_mode == 2:
                # Process in chunks (chunk_size doesn't affect correctness)
                chunks = []
                for start in range(0, seq_len, chunk_size):
                    end = min(start + chunk_size, seq_len)
                    chunk = x[:, start:end]
                    chunks.append(chunk * factor)
                return torch.cat(chunks, dim=1)
            else:  # scale_mode == 3
                # Using einsum (chunk_size parameter ignored here)
                return torch.einsum("...i,i->...i", x, factor)

        @torch.library.custom_op(op_name, mutates_args=())
        def multi_param_op(
            x: torch.Tensor,
            factor: torch.Tensor,
            scale_mode: int = 1,
            chunk_size: int = 16,
        ) -> torch.Tensor:
            return x * factor  # Simple fallback

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

        # Test multi-parameter tuning: 1 algorithm × 3 scale_modes × 2 chunk_sizes = 6 variants
        register_custom_op_autotuning(
            custom_op=op_object.default,
            decompositions=[multi_param_scaling],
            tuning_knob={"scale_mode": [1, 2, 3], "chunk_size": [16, 32]},
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

        # Verify numerical equivalence across all parameter combinations
        expected_result = test_x * test_factor

        for scale_mode in [1, 2, 3]:
            for chunk_size in [16, 32]:
                result = multi_param_scaling(
                    test_x, test_factor, scale_mode=scale_mode, chunk_size=chunk_size
                )
                torch.testing.assert_close(
                    result,
                    expected_result,
                    rtol=1e-5,
                    atol=1e-5,
                    msg=f"scale_mode={scale_mode}, chunk_size={chunk_size} not equivalent to expected",
                )

        # Test autotuning - benchmarks all 6 parameter combinations
        self._run_autotune_test(
            op_object, (test_x, test_factor), expected_result, "MultiParameter"
        )

    @skipIfXpu
    def test_max_autotune_knob_integration(self):
        """Test that max_autotune config flag enables additional tuning configs for custom ops."""
        test_op_name = f"test_lib::max_autotune_test_{id(self)}"

        def mm_split_k_implementation(
            a: torch.Tensor,
            b: torch.Tensor,
            k_splits: int = 4,
            tile_size: int = 64,
            use_chunked_compute: bool = False,
        ) -> torch.Tensor:
            """Matrix multiply with split-k and algorithmic variations that actually affect computation."""
            m = a.shape[0]
            n = b.shape[1]
            k = a.shape[1]

            # tile_size affects how we split the computation - this actually matters
            if use_chunked_compute and k >= tile_size:
                # Process in smaller chunks for better cache locality
                effective_k_splits = min(k_splits, k // tile_size)
            else:
                # Standard split-k approach
                effective_k_splits = k_splits

            effective_k_splits = max(1, effective_k_splits)
            k_parts = k // effective_k_splits
            B = effective_k_splits

            if k_parts == 0:
                return a @ b

            # The actual computation pattern that can be tuned
            if use_chunked_compute:
                # Process in smaller tiles for better memory access
                results = []
                for i in range(B):
                    start_k = i * k_parts
                    end_k = (i + 1) * k_parts if i < B - 1 else k
                    partial = a[:, start_k:end_k] @ b[start_k:end_k, :]
                    results.append(partial)
                result = sum(results)
            else:
                # Standard batched approach
                a_reshaped = torch.permute(
                    a[:, : B * k_parts].reshape(m, B, k_parts), (1, 0, 2)
                )
                b_reshaped = b[: B * k_parts].reshape(B, k_parts, n)
                result = torch.sum(torch.bmm(a_reshaped, b_reshaped), dim=0)

                # Handle remainder
                if B * k_parts < k:
                    remainder_a = a[:, B * k_parts :]
                    remainder_b = b[B * k_parts :]
                    result += remainder_a @ remainder_b

            return result

        @torch.library.custom_op(test_op_name, mutates_args=())
        def test_mm_split_k_op(
            a: torch.Tensor,
            b: torch.Tensor,
            k_splits: int = 4,
            tile_size: int = 64,
            use_chunked_compute: bool = False,
        ) -> torch.Tensor:
            return mm_split_k_implementation(
                a, b, k_splits, tile_size, use_chunked_compute
            )

        @test_mm_split_k_op.register_fake
        def _(
            a: torch.Tensor,
            b: torch.Tensor,
            k_splits: int = 4,
            tile_size: int = 64,
            use_chunked_compute: bool = False,
        ):
            return torch.empty(a.shape[0], b.shape[1], device=a.device, dtype=a.dtype)

        lib_name, op_name = test_op_name.split("::")
        op_object = getattr(getattr(torch.ops, lib_name), op_name)

        register_custom_op_autotuning(
            custom_op=op_object.default,
            decompositions=[mm_split_k_implementation],
            tuning_knob={"k_splits": [32, 64]},
            max_autotune_configs={
                "tile_size": [32, 64, 128],
                "use_chunked_compute": [False, True],
            },
            name="test_mm_split_k_max_autotune",
            input_gen_fns={
                0: lambda fake_tensor: torch.randn_like(fake_tensor, device=self.device)
                * 0.1,
                1: lambda fake_tensor: torch.randn_like(fake_tensor, device=self.device)
                * 0.1,
            },
        )

        # Create inputs with k divisible by all potential k_splits and block_size values
        m, k, n = 256, 8192, 1024  # k=8192 is divisible by [8, 16, 32, 64, 128, 256]
        a = torch.randn(m, k, device=self.device, dtype=self.dtype, requires_grad=False)
        b = torch.randn(k, n, device=self.device, dtype=self.dtype, requires_grad=False)
        expected = a @ b

        # Test with max_autotune=True to verify extended configs are used
        torch._dynamo.reset()
        with config.patch(
            max_autotune=True,
            max_autotune_gemm_backends="TRITON" if self.device == "cuda" else "ATEN",
            fx_graph_cache=False,
            benchmark_kernel=True,
        ):

            @torch.compile
            def test_model_max_autotune(a, b):
                return op_object(a, b)

            result_max_autotune = test_model_max_autotune(a, b)

        # Test with max_autotune=False - should use only basic configs
        torch._dynamo.reset()
        with config.patch(
            max_autotune=False,
            fx_graph_cache=False,
            benchmark_kernel=True,
        ):

            @torch.compile
            def test_model_basic(a, b):
                return op_object(a, b)

            result_basic = test_model_basic(a, b)

        # Both should produce correct results
        torch.testing.assert_close(
            result_max_autotune,
            expected,
            rtol=2e-1,
            atol=5e-1,
            msg="max_autotune=True result incorrect",
        )

        torch.testing.assert_close(
            result_basic,
            expected,
            rtol=2e-1,
            atol=5e-1,
            msg="max_autotune=False result incorrect",
        )

    @skipIfXpu
    def test_triton_kernel_style_max_autotune_configs(self):
        """Test max_autotune_configs with Triton-style kernel configurations similar to tuned_mm."""
        test_op_name = f"test_lib::triton_kernel_test_{id(self)}"

        def triton_style_mm_implementation(
            a: torch.Tensor,
            b: torch.Tensor,
            BLOCK_M: int = 64,
            BLOCK_N: int = 64,
            BLOCK_K: int = 32,
            num_stages: int = 2,
            num_warps: int = 4,
            group_m: int = 8,
        ) -> torch.Tensor:
            """Matrix multiply implementation with Triton-style kernel parameters.

            These parameters mirror the ones used in tuned_mm's GemmConfig:
            - BLOCK_M, BLOCK_N, BLOCK_K: Tile sizes for matrix blocking
            - num_stages: Number of pipeline stages
            - num_warps: Number of warps per thread block
            - group_m: Grouping factor for better memory coalescing
            """
            # For demonstration, we use these parameters to influence computation pattern
            # In a real Triton kernel, these would directly control the kernel configuration

            m, k = a.shape
            k_b, n = b.shape
            assert k == k_b, f"Matrix dimensions don't match: {k} != {k_b}"

            # Use block sizes to determine computation strategy
            if BLOCK_M >= 128 and BLOCK_N >= 128:
                # Large block strategy - process in larger chunks
                return torch.mm(a, b)
            elif BLOCK_K >= 64:
                # Large K block strategy - use split-k if beneficial
                k_splits = min(4, k // BLOCK_K)
                if k_splits > 1:
                    k_per_split = k // k_splits
                    results = []
                    for i in range(k_splits):
                        start_k = i * k_per_split
                        end_k = (i + 1) * k_per_split if i < k_splits - 1 else k
                        partial_result = torch.mm(
                            a[:, start_k:end_k], b[start_k:end_k, :]
                        )
                        results.append(partial_result)
                    return sum(results)
                else:
                    return torch.mm(a, b)
            else:
                # Small block strategy - standard implementation
                return torch.mm(a, b)

        @torch.library.custom_op(test_op_name, mutates_args=())
        def test_triton_kernel_op(
            a: torch.Tensor,
            b: torch.Tensor,
            BLOCK_M: int = 64,
            BLOCK_N: int = 64,
            BLOCK_K: int = 32,
            num_stages: int = 2,
            num_warps: int = 4,
            group_m: int = 8,
        ) -> torch.Tensor:
            return triton_style_mm_implementation(
                a, b, BLOCK_M, BLOCK_N, BLOCK_K, num_stages, num_warps, group_m
            )

        @test_triton_kernel_op.register_fake
        def _(
            a: torch.Tensor,
            b: torch.Tensor,
            BLOCK_M: int = 64,
            BLOCK_N: int = 64,
            BLOCK_K: int = 32,
            num_stages: int = 2,
            num_warps: int = 4,
            group_m: int = 8,
        ):
            return torch.empty(a.shape[0], b.shape[1], device=a.device, dtype=a.dtype)

        lib_name, op_name = test_op_name.split("::")
        op_object = getattr(getattr(torch.ops, lib_name), op_name)

        register_custom_op_autotuning(
            custom_op=op_object.default,
            decompositions=[triton_style_mm_implementation],
            tuning_knob={
                "BLOCK_M": [64, 128],
                "BLOCK_N": [64, 128],
                "BLOCK_K": [32, 64],
            },
            max_autotune_configs={
                "num_stages": [2, 4],
                "num_warps": [4, 8],
                "group_m": [8],
            },
            name="test_triton_kernel_max_autotune",
            input_gen_fns={
                0: lambda fake_tensor: torch.randn_like(fake_tensor, device=self.device)
                * 0.1,
                1: lambda fake_tensor: torch.randn_like(fake_tensor, device=self.device)
                * 0.1,
            },
        )

        # Create test inputs
        m, k, n = 512, 1024, 768
        a = torch.randn(m, k, device=self.device, dtype=self.dtype, requires_grad=False)
        b = torch.randn(k, n, device=self.device, dtype=self.dtype, requires_grad=False)
        expected = a @ b

        # Test with max_autotune to demonstrate Triton-style kernel parameter autotuning
        @torch.compile
        def test_model(a, b):
            return op_object(a, b)

        torch._dynamo.reset()

        with config.patch(
            max_autotune=True,
            max_autotune_gemm_backends="TRITON" if self.device == "cuda" else "ATEN",
            fx_graph_cache=False,
            benchmark_kernel=True,
        ):
            result = test_model(a, b)

        # Verify correctness
        torch.testing.assert_close(
            result,
            expected,
            rtol=2e-1,
            atol=5e-1,
            msg="Triton-style kernel config result incorrect",
        )


if __name__ == "__main__":
    run_tests()
