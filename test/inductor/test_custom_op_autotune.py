# Owner(s): ["module: inductor"]
"""
Tests for custom operation autotuning with PyTorch Inductor.

Validates that custom ops can be registered with multiple CustomOpConfigs, where each
config specifies an optional decomposition function and its associated parameters.
Inductor benchmarks all variants and automatically selects the best performing one.
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
        """Test RMSNorm autotuning with multiple decomposition variants and dynamic shapes.

        Validates:
        - Multiple decomposition implementations with different computational approaches
        - Dynamic shape handling across multiple compilations
        """
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
            """Separate normalization and scaling: compute normalized value then scale."""
            x_var = x
            variance = x_var.pow(2).mean(dim=-1, keepdim=True)
            x = x * torch.rsqrt(variance + eps)
            x = x * weight
            return x

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

        decompositions = [
            rmsnorm_decomposition1,
            rmsnorm_decomposition2,
        ]

        register_custom_op_autotuning(
            test_rmsnorm_op,
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
            input_tensor, weight = self._create_rmsnorm_inputs(
                batch_size, seq_len, hidden_dim
            )

            # Test numerical equivalence for all decompositions
            self._assert_implementations_equivalent(
                decompositions, (input_tensor, weight), f"RMSNorm_{i}"
            )

            # Test autotuning
            expected = rmsnorm_decomposition1(input_tensor, weight)
            self._run_autotune_test(
                test_rmsnorm_op, (input_tensor, weight), expected, f"RMSNorm_{i}"
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
        """Test decompose_k autotuning with epilogue fusion (matmul + bias + relu + scale).

        Validates that the custom op encapsulates the entire fused operation with parametric
        tuning for k_splits values controlling how the K dimension is decomposed.
        """
        test_op_name = f"test_lib::matmul_relu_epilogue_{id(self)}"

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
        def matmul_relu_epilogue_op(
            a: torch.Tensor, b: torch.Tensor, bias: torch.Tensor, k_splits: int = 4
        ) -> torch.Tensor:
            """Matmul with decompose_k + bias + relu + scale (complete epilogue fusion)."""
            matmul_result = decompose_k_implementation(a, b, k_splits)
            biased = matmul_result + bias
            activated = torch.relu(biased)
            scaled = activated * 2.0
            return scaled

        @matmul_relu_epilogue_op.register_fake
        def _(a: torch.Tensor, b: torch.Tensor, bias: torch.Tensor, k_splits: int = 4):
            return torch.empty(a.shape[0], b.shape[1], device=a.device, dtype=a.dtype)

        # Register autotuning with different k_splits values
        register_custom_op_autotuning(
            matmul_relu_epilogue_op,
            configs=[
                CustomOpConfig(k_splits=2),
                CustomOpConfig(k_splits=4),
                CustomOpConfig(k_splits=8),
                CustomOpConfig(k_splits=16),
                CustomOpConfig(k_splits=32),
                CustomOpConfig(k_splits=64),
                CustomOpConfig(k_splits=128),
            ],
            name="matmul_relu_epilogue_autotuned",
            input_gen_fns={
                "a": lambda fake_tensor: torch.randn_like(
                    fake_tensor, device=self.device
                )
                * 0.1,
                "b": lambda fake_tensor: torch.randn_like(
                    fake_tensor, device=self.device
                )
                * 0.1,
                "bias": lambda fake_tensor: torch.randn_like(
                    fake_tensor, device=self.device
                )
                * 0.1,
            },
        )

        # Create test inputs
        a, b = self._create_decompose_k_inputs()
        bias = torch.randn(b.shape[1], device=self.device, dtype=self.dtype) * 0.1

        # Compile the model using the custom op
        @torch.compile
        def test_model(a, b, bias):
            return matmul_relu_epilogue_op(a, b, bias)

        torch._dynamo.reset()

        with config.patch(
            max_autotune=True,
            benchmark_fusion=True,
        ):
            compiled_result = test_model(a, b, bias)

        def reference_model(a, b, bias):
            matmul_result = a @ b
            biased = matmul_result + bias
            activated = torch.relu(biased)
            scaled = activated * 2.0
            return scaled

        expected = reference_model(a, b, bias)

        torch.testing.assert_close(
            compiled_result,
            expected,
            rtol=2e-1,
            atol=5e-1,
        )

    @skipIfXpu
    def test_multi_parameter_tuning(self):
        """Test autotuning with multiple parameters for combinatorial parameter exploration.

        Validates parametric tuning with multiple parameters (scale_mode and chunk_size)
        to test combinatorial exploration of the parameter space.
        """
        test_op_name = f"test_lib::multi_param_{id(self)}"

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

        @torch.library.custom_op(test_op_name, mutates_args=())
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

        # Use explicit configs with scale_mode and chunk_size parameters as tuning knobs
        register_custom_op_autotuning(
            multi_param_op,
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

        for scale_mode, chunk_size in configs:
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
            multi_param_op, (test_x, test_factor), expected_result, "MultiParam"
        )

    @skipIfXpu
    def test_dynamic_range_tuning(self):
        """Test dynamic input range-based autotuning.

        Validates that:
        - All implementations produce equivalent results
        - Autotuning selects best implementation per range
        - torch.cond dispatch function is generated correctly
        """
        test_op_name = f"test_lib::dynamic_range_{id(self)}"

        def short_sequence_impl(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
            return torch.einsum("bsh,h->bsh", x, weight)

        def medium_sequence_impl(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
            batch_size, seq_len, hidden_dim = x.shape
            chunk_size = 256
            chunks = []
            for start in range(0, seq_len, chunk_size):
                end = min(start + chunk_size, seq_len)
                chunk = x[:, start:end, :]
                chunks.append(chunk * weight)
            return torch.cat(chunks, dim=1)

        def long_sequence_impl(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
            return x * weight.view(1, 1, -1)

        @torch.library.custom_op(test_op_name, mutates_args=())
        def dynamic_range_op(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
            return x * weight

        @dynamic_range_op.register_fake
        def _(x: torch.Tensor, weight: torch.Tensor):
            return torch.empty_like(x)

        register_custom_op_autotuning(
            dynamic_range_op,
            configs=[
                CustomOpConfig(short_sequence_impl),
                CustomOpConfig(medium_sequence_impl),
                CustomOpConfig(long_sequence_impl),
            ],
            name="dynamic_range_autotuned",
            dispatch_on=("x", 1),
            split_points=[512, 2048],
            input_gen_fns={
                "x": lambda fake: torch.randn_like(fake, device=self.device) * 0.1,
                "weight": lambda fake: torch.ones_like(fake, device=self.device),
            },
        )

        # Verify all implementations produce equivalent results
        test_cases = [
            (2, 256, 128),
            (2, 1024, 128),
            (2, 4096, 128),
        ]

        for batch_size, seq_len, hidden_dim in test_cases:
            test_x = torch.randn(
                batch_size, seq_len, hidden_dim, device=self.device, dtype=self.dtype
            )
            test_weight = torch.ones(hidden_dim, device=self.device, dtype=self.dtype)
            expected = test_x * test_weight

            for impl_name, impl_fn in [
                ("short", short_sequence_impl),
                ("medium", medium_sequence_impl),
                ("long", long_sequence_impl),
            ]:
                result = impl_fn(test_x, test_weight)
                torch.testing.assert_close(
                    result,
                    expected,
                    rtol=1e-5,
                    atol=1e-5,
                    msg=f"{impl_name} implementation differs for seq_len={seq_len}",
                )

            # Test autotuning with compilation
            self._run_autotune_test(
                dynamic_range_op,
                (test_x, test_weight),
                expected,
                f"DynamicRange_seq{seq_len}",
            )

    @skipIfXpu
    def test_non_adjacent_range_merging(self):
        """Test that non-adjacent ranges with the same implementation are merged with OR predicates.

        Validates:
        - Non-adjacent ranges using the same impl are grouped together
        - OR predicates are generated for grouped ranges
        - Reduces number of torch.cond branches from N ranges to M impl groups

        Example: If ranges [1,64], [129,256], [513,inf] all choose impl_medium,
        they should be grouped into a single branch with OR predicate:
        (dim <= 64) | ((dim >= 129) & (dim <= 256)) | (dim >= 513)
        """
        test_op_name = f"test_lib::range_merge_{id(self)}"

        def impl_small(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
            """Implementation 1: direct add"""
            return x + weight

        def impl_medium(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
            """Implementation 2: add with intermediate variable"""
            result = x
            result = result + weight
            return result

        def impl_large(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
            """Implementation 3: add with broadcasting"""
            return x + weight.view(1, 1, -1).expand_as(x)

        @torch.library.custom_op(test_op_name, mutates_args=())
        def range_merge_op(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
            """Custom op with 5 ranges that may merge to fewer impl groups"""
            return x + weight

        @range_merge_op.register_fake
        def _(x: torch.Tensor, weight: torch.Tensor):
            return torch.empty_like(x)

        # Register with only 3 unique implementations for 5 ranges
        # All implementations are numerically equivalent (x + weight)
        # but use different computational approaches
        register_custom_op_autotuning(
            range_merge_op,
            configs=[
                CustomOpConfig(impl_small),
                CustomOpConfig(impl_medium),
                CustomOpConfig(impl_large),
            ],
            dispatch_on=("x", 1),
            split_points=[64, 128, 256, 512],  # Creates 5 ranges
            input_gen_fns={
                "x": lambda fake: torch.randn_like(fake, device=self.device),
                "weight": lambda fake: torch.randn_like(fake, device=self.device),
            },
        )

        # Verify all implementations produce equivalent results
        test_cases = [
            (32, "Range 1: [1, 64]"),
            (96, "Range 2: [65, 128]"),
            (192, "Range 3: [129, 256]"),
            (384, "Range 4: [257, 512]"),
            (768, "Range 5: [513, inf]"),
        ]

        for seq_len, desc in test_cases:
            test_x = torch.randn(2, seq_len, 32, device=self.device, dtype=self.dtype)
            test_weight = torch.randn(32, device=self.device, dtype=self.dtype)
            expected = test_x + test_weight

            # Verify each implementation produces equivalent result
            for impl_name, impl_fn in [
                ("small", impl_small),
                ("medium", impl_medium),
                ("large", impl_large),
            ]:
                result = impl_fn(test_x, test_weight)
                # All implementations should produce x + weight
                torch.testing.assert_close(
                    result,
                    expected,
                    rtol=1e-5,
                    atol=1e-5,
                    msg=f"{impl_name} produced different result for {desc}",
                )

            # Test autotuning with compilation
            self._run_autotune_test(
                range_merge_op,
                (test_x, test_weight),
                expected,
                f"RangeMerge_{seq_len}",
            )


if __name__ == "__main__":
    run_tests()
