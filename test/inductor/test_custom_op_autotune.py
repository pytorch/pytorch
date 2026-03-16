# Owner(s): ["module: inductor"]
"""
Tests for custom operation autotuning with PyTorch Inductor.

Validates that custom ops can be registered with multiple CustomOpConfigs, where each
config specifies an optional decomposition function and its associated parameters.
Inductor benchmarks all variants and automatically selects the best performing one.
"""

import unittest

import torch
import torch._inductor.runtime.benchmarking
from torch._dynamo.utils import counters
from torch._inductor import config
from torch._inductor.kernel.custom_op import (
    CustomOpConfig,
    register_custom_op_autotuning,
)
from torch._inductor.test_case import run_tests, TestCase
from torch.testing import FileCheck
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    IS_MACOS,
    parametrize,
    skipIfXpu,
)
from torch.testing._internal.inductor_utils import (
    HAS_CPU,
    HAS_GPU,
    HAS_TRITON,
    IS_BIG_GPU,
)


torch.set_float32_matmul_precision("high")


@unittest.skipIf(IS_MACOS, "TODO: mac")
@unittest.skipUnless(HAS_GPU and HAS_TRITON, "requires GPU and Triton")
class TestCustomOpAutoTune(TestCase):
    """Test custom operation autotuning functionality."""

    def setUp(self) -> None:
        """Set up test environment with appropriate device and dtype."""
        super().setUp()
        torch._dynamo.reset()
        self.device = "cuda" if HAS_GPU else "cpu"
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32
        # Clear any previous lowering registrations to ensure test isolation
        from torch._inductor.lowering import user_lowerings

        user_lowerings.clear()

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
                "input_tensor": lambda x: torch.randn_like(x, device=self.device)
                * 0.02,
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
        """Create test inputs for decompose_k matrix multiplication.
        Tensor a: Input matrix of shape (m, k)
        Tensor b: Weight matrix of shape (k, n)
        Tensor bias: Bias vector of shape (n,)
        """
        # Ensure k is divisible by all k_splits values: [2, 32, 64, 128, 256]
        k = ((k + 255) // 256) * 256  # Round up to nearest multiple of 256
        sd = k**0.25
        a = (
            torch.randn(m, k, device=self.device, dtype=self.dtype, requires_grad=False)
            / sd
        )
        b = (
            torch.randn(k, n, device=self.device, dtype=self.dtype, requires_grad=False)
            / sd
        )
        bias = (
            torch.randn(n, device=self.device, dtype=self.dtype, requires_grad=False)
            * 0.1
        )
        return a, b, bias

    @skipIfXpu
    def test_decompose_k_custom_op_autotune_dynamic_config_for_input_shape(self):
        """Test decompose_k autotuning with with epilogue fusion(matmul+bias+relu+scale) and
        dynamic config generation based on matmul input shapes.

        Validates that the custom op encapsulates the entire fused operation (matmul + bias
        + relu + scale) with parametric tuning for k_splits values controlling how the K
        dimension is decomposed. The config generator receives correct parameter names and
        shapes, dynamically generates different k_split configs using get_k_splits for
        different input shapes, and produces correct results matching the reference implementation.
        """
        test_op_name = f"test_lib::matmul_relu_epilogue_dynamic_{id(self)}"

        def decompose_k_implementation(
            a: torch.Tensor, b: torch.Tensor, k_splits: int = 4
        ) -> torch.Tensor:
            """Matrix multiply with k-way decomposition."""
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
        def matmul_relu_epilogue_dynamic_op(
            a: torch.Tensor, b: torch.Tensor, bias: torch.Tensor, k_splits: int = 4
        ) -> torch.Tensor:
            """Matmul with decompose_k + bias + relu + scale (complete epilogue fusion)."""
            matmul_result = decompose_k_implementation(a, b, k_splits)
            biased = matmul_result + bias
            activated = torch.relu(biased)
            scaled = activated * 2.0
            return scaled

        @matmul_relu_epilogue_dynamic_op.register_fake
        def _(a: torch.Tensor, b: torch.Tensor, bias: torch.Tensor, k_splits: int = 4):
            return torch.empty(a.shape[0], b.shape[1], device=a.device, dtype=a.dtype)

        # Define dynamic config generator using get_k_splits
        def generate_k_split_configs(
            fake_tensors: dict[str, torch.Tensor],
        ) -> list[CustomOpConfig]:
            """Generate k_split configs based on input matrix dimensions."""
            from torch._inductor.utils import get_k_splits

            m, k = fake_tensors["a"].shape[-2:]
            _, n = fake_tensors["b"].shape[-2:]

            k_splits_list = get_k_splits(m, n, k)

            return [CustomOpConfig(k_splits=k) for k in k_splits_list]

        register_custom_op_autotuning(
            matmul_relu_epilogue_dynamic_op,
            config_generator=generate_k_split_configs,
            name="matmul_relu_epilogue_dynamic_autotuned",
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

        # Test multiple shapes to verify dynamic config generation
        test_shapes = [
            (256, 16384, 1024),
            (256, 65536, 1024),
        ]

        for m, k, n in test_shapes:
            # Use helper function to create test inputs
            a, b, bias = self._create_decompose_k_inputs(m, k, n)

            @torch.compile
            def test_model(a, b, bias):
                return matmul_relu_epilogue_dynamic_op(a, b, bias)

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
                rtol=2e-3,
                atol=5e-3,
                # msg=f"Failed for shape ({m}, {k}, {n})",
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
    def test_range_based_static_shape_no_cond_dispatch(self):
        """Test dispatch code generation for static vs dynamic shapes.

        Static shapes (dynamic=False): No dispatch logic, best impl is inlined.
        Dynamic shapes (dynamic=True): Dispatch logic generated for runtime selection.
        """
        import re

        from torch._inductor.utils import run_and_get_code

        test_op_name = f"test_lib::static_no_cond_{id(self)}"

        def impl_a(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
            return x + weight

        def impl_b(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
            return x + weight.view(1, 1, -1).expand_as(x)

        def impl_c(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
            return torch.add(x, weight)

        @torch.library.custom_op(test_op_name, mutates_args=())
        def static_no_cond_op(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
            return x + weight

        @static_no_cond_op.register_fake
        def _(x: torch.Tensor, weight: torch.Tensor):
            return torch.empty_like(x)

        register_custom_op_autotuning(
            static_no_cond_op,
            configs=[
                CustomOpConfig(impl_a),
                CustomOpConfig(impl_b),
                CustomOpConfig(impl_c),
            ],
            dispatch_on={"tensor_name": "x", "dim": 1},
            split_points=[64, 128, 256, 512],
            input_gen_fns={
                "x": lambda fake: torch.randn_like(fake, device=self.device),
                "weight": lambda fake: torch.randn_like(fake, device=self.device),
            },
        )

        autotune_backends = "TRITON" if self.device == "cuda" else "ATEN"
        test_x = torch.randn(2, 96, 32, device=self.device, dtype=self.dtype)
        test_weight = torch.randn(32, device=self.device, dtype=self.dtype)

        def find_shape_dispatch(code_list):
            pattern = re.compile(r"if\s+s\d+\s*[<>=]")
            return [
                line.strip()
                for code in code_list
                for line in code.split("\n")
                if pattern.search(line)
            ]

        def test_model(x, weight):
            return static_no_cond_op(x, weight)

        # Static shape (dynamic=False) - should NOT have shape dispatch
        torch._dynamo.reset()
        with config.patch(
            max_autotune=True,
            max_autotune_gemm_backends=autotune_backends,
            fx_graph_cache=False,
            benchmark_kernel=True,
        ):
            result_static, code_list_static = run_and_get_code(
                torch.compile(test_model, dynamic=False), test_x, test_weight
            )
            self.assertEqual(result_static, test_x + test_weight)

        dispatch_static = find_shape_dispatch(code_list_static)
        if dispatch_static:
            print(f"[Static] Found dispatch logic: {dispatch_static}")
        else:
            print("[Static] No dispatch logic found (expected for static shapes)")
        self.assertFalse(
            dispatch_static, "Static shapes should not have dispatch logic"
        )

        # Dynamic shape (dynamic=True) - SHOULD have shape dispatch
        torch._dynamo.reset()
        with config.patch(
            max_autotune=True,
            max_autotune_gemm_backends=autotune_backends,
            fx_graph_cache=False,
            benchmark_kernel=True,
        ):
            result_dynamic, code_list_dynamic = run_and_get_code(
                torch.compile(test_model, dynamic=True), test_x, test_weight
            )
            self.assertEqual(result_dynamic, test_x + test_weight)

        dispatch_dynamic = find_shape_dispatch(code_list_dynamic)
        if dispatch_dynamic:
            print(f"[Dynamic] Found dispatch logic: {dispatch_dynamic}")
        else:
            print("[Dynamic] No dispatch logic found (unexpected for dynamic shapes)")
        self.assertTrue(dispatch_dynamic, "Dynamic shapes should have dispatch logic")

    @skipIfXpu
    def test_benchmark_with_cudagraphs_uses_cuda_graph_benchmarking(self):
        """Test that benchmark_with_cudagraphs flag causes CUDA graph benchmarking to be used."""
        if self.device != "cuda":
            self.skipTest("CUDA graph test requires CUDA device")

        from unittest.mock import patch

        test_op_name = f"test_lib::cudagraph_patch_{id(self)}"

        def fast_decomposition(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
            return x @ weight

        @torch.library.custom_op(test_op_name, mutates_args=())
        def cudagraph_patch_op(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
            return x @ weight

        @cudagraph_patch_op.register_fake
        def _(x: torch.Tensor, weight: torch.Tensor):
            return torch.empty(
                x.shape[0], weight.shape[1], device=x.device, dtype=x.dtype
            )

        register_custom_op_autotuning(
            cudagraph_patch_op,
            configs=[CustomOpConfig(fast_decomposition)],
            name="cudagraph_patch_autotuned",
            benchmark_with_cudagraphs=True,
            input_gen_fns={
                "x": lambda t: torch.randn_like(t, device=self.device),
                "weight": lambda t: torch.randn_like(t, device=self.device),
            },
        )

        test_x = torch.randn(64, 256, device=self.device, dtype=self.dtype)
        test_weight = torch.randn(256, 128, device=self.device, dtype=self.dtype)

        @torch.compile
        def test_model(x, weight):
            return cudagraph_patch_op(x, weight)

        cuda_graph_benchmark_called = False
        original_benchmark_gpu_with_cuda_graph = torch._inductor.runtime.benchmarking.Benchmarker.benchmark_gpu_with_cuda_graph

        def patched_benchmark_gpu_with_cuda_graph(self, fn):
            nonlocal cuda_graph_benchmark_called
            cuda_graph_benchmark_called = True
            return original_benchmark_gpu_with_cuda_graph(self, fn)

        torch._dynamo.reset()
        with config.patch(max_autotune=True, fx_graph_cache=False):
            with patch.object(
                torch._inductor.runtime.benchmarking.Benchmarker,
                "benchmark_gpu_with_cuda_graph",
                patched_benchmark_gpu_with_cuda_graph,
            ):
                result = test_model(test_x, test_weight)

        self.assertTrue(
            cuda_graph_benchmark_called,
            "benchmark_gpu_with_cuda_graph should have been called",
        )
        torch.testing.assert_close(result, test_x @ test_weight, rtol=1e-1, atol=1e-1)

    @skipIfXpu
    def test_min_speedup_threshold_api(self):
        """Test that min_speedup_threshold parameter is accepted and compilation works."""
        test_op_name = f"test_lib::min_speedup_{id(self)}"

        def decomposition(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
            return x @ weight

        @torch.library.custom_op(test_op_name, mutates_args=())
        def min_speedup_op(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
            return x @ weight

        @min_speedup_op.register_fake
        def _(x: torch.Tensor, weight: torch.Tensor):
            return torch.empty(
                x.shape[0], weight.shape[1], device=x.device, dtype=x.dtype
            )

        # Test that API accepts min_speedup_threshold parameter
        register_custom_op_autotuning(
            min_speedup_op,
            configs=[CustomOpConfig(decomposition)],
            name="min_speedup_autotuned",
            min_speedup_threshold=1.5,
            input_gen_fns={
                "x": lambda t: torch.randn_like(t, device=self.device),
                "weight": lambda t: torch.randn_like(t, device=self.device),
            },
        )

        test_x = torch.randn(64, 256, device=self.device, dtype=self.dtype)
        test_weight = torch.randn(256, 128, device=self.device, dtype=self.dtype)

        @torch.compile
        def test_model(x, weight):
            return min_speedup_op(x, weight)

        torch._dynamo.reset()
        with config.patch(max_autotune=True, fx_graph_cache=False):
            result = test_model(test_x, test_weight)

        torch.testing.assert_close(result, test_x @ test_weight, rtol=1e-1, atol=1e-1)

    @skipIfXpu
    def test_config_patching_in_generated_code(self):
        """Test that coordinate_descent_tuning config_patches flows through to generated code."""
        if self.device != "cuda":
            self.skipTest(
                "coordinate_descent_tuning test requires CUDA for Triton codegen"
            )

        test_op_name = f"test_lib::coord_descent_{id(self)}"

        def decomposition(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
            return torch.ones_like(x)

        @torch.library.custom_op(test_op_name, mutates_args=())
        def coord_op(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
            return x @ weight

        @coord_op.register_fake
        def _(x: torch.Tensor, weight: torch.Tensor):
            return torch.empty(
                x.shape[0], weight.shape[1], device=x.device, dtype=x.dtype
            )

        # Register with config_patches containing coordinate_descent_tuning
        register_custom_op_autotuning(
            coord_op,
            configs=[
                CustomOpConfig(
                    decomposition, config_patches={"coordinate_descent_tuning": True}
                )
            ],
            name="coord_descent_autotuned",
            input_gen_fns={
                "x": lambda t: torch.randn_like(t, device=self.device),
                "weight": lambda t: torch.randn_like(t, device=self.device),
            },
        )

        test_x = torch.randn(1024, 1024, device=self.device, dtype=self.dtype)
        test_weight = torch.randn(1024, 1024, device=self.device, dtype=self.dtype)

        @torch.compile
        def test_model(x, weight):
            return coord_op(x, weight)

        torch._dynamo.reset()

        # Compile and capture generated code
        with config.patch(max_autotune=True, fx_graph_cache=False):
            result, code = torch._inductor.utils.run_and_get_code(
                test_model, test_x, test_weight
            )

        # Check that coordinate_descent_tuning is enabled in the generated code's inductor_meta
        FileCheck().check("'coordinate_descent_tuning': True").run("\n".join(code))

    @skipIfXpu
    @config.patch(
        {
            "test_configs.force_custom_op_decomposition": True,
            "test_configs.force_no_impl_grouping": True,
        }
    )
    def test_split_config_patching_in_generated_code(self):
        """Test that coordinate_descent_tuning config_patches flows through to generated code."""
        if self.device != "cuda":
            self.skipTest(
                "coordinate_descent_tuning test requires CUDA for Triton codegen"
            )

        test_op_name = f"test_lib::coord_descent_{id(self)}"

        def decomposition(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
            return torch.full(
                (x.shape[0], weight.shape[1]),
                x.shape[0],
                dtype=weight.dtype,
                device=weight.device,
            )

        @torch.library.custom_op(test_op_name, mutates_args=())
        def coord_op(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
            if x.shape[0] == 128:
                return torch.empty(
                    x.shape[0],
                    weight.shape[1],
                    dtype=weight.dtype,
                    device=weight.device,
                )
            return x @ weight

        @coord_op.register_fake
        def _(x: torch.Tensor, weight: torch.Tensor):
            return torch.empty(
                x.shape[0], weight.shape[1], device=x.device, dtype=x.dtype
            )

        # Register with config_patches containing coordinate_descent_tuning
        register_custom_op_autotuning(
            coord_op,
            configs=[
                CustomOpConfig(
                    decomposition, config_patches={"coordinate_descent_tuning": True}
                )
            ],
            name="coord_descent_autotuned",
            input_gen_fns={
                "x": lambda t: torch.randn_like(t, device=self.device),
                "weight": lambda t: torch.randn_like(t, device=self.device),
            },
            dispatch_on={"tensor_name": "x", "dim": 0},
            split_points=[128, 512],
        )

        test_x = torch.randn(1024, 1024, device=self.device, dtype=self.dtype)
        test_weight = torch.randn(1024, 1024, device=self.device, dtype=self.dtype)

        @torch.compile
        def test_model(x, weight):
            return coord_op(x, weight)

        torch._dynamo.mark_dynamic(test_x, 0)

        result, code = torch._inductor.utils.run_and_get_code(
            test_model, test_x, test_weight
        )

        #
        with torch._dynamo.config.patch(error_on_recompile=True):
            for i in range(10):
                x = torch.randn(256 + i, 1024, device=self.device, dtype=self.dtype)
                torch._dynamo.mark_dynamic(x, 0)
                self.assertEqual(
                    test_model(x, test_weight), decomposition(x, test_weight)
                )

        # Check that coordinate_descent_tuning is enabled in the generated code's inductor_meta.
        # config_patches should flow to both benchmark graphs and final dispatch code.
        code_with_coord = [c for c in code if "'coordinate_descent_tuning': True" in c]
        self.assertEqual(
            len(code_with_coord),
            len(code),
            f"Expected all {len(code)} code modules to have coordinate_descent_tuning, "
            f"but only {len(code_with_coord)} have it",
        )

    @skipIfXpu
    def test_benchmark_real_trace_symbolic(self):
        """Verify benchmarking uses real values but tracing uses symbolic shapes."""
        if self.device != "cuda":
            self.skipTest("Test requires CUDA")

        # Track shapes seen by the real op implementation
        shapes_seen = []

        test_op_name = f"test_lib::shape_tracker_{id(self)}"

        def decomposition(x, weight):
            return x @ weight

        @torch.library.custom_op(test_op_name, mutates_args=())
        def shape_tracker_op(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
            # This runs during benchmarking with REAL values
            shapes_seen.append(x.shape[0])
            return x @ weight

        @shape_tracker_op.register_fake
        def _(x, weight):
            return torch.empty(
                x.shape[0], weight.shape[1], device=x.device, dtype=x.dtype
            )

        register_custom_op_autotuning(
            shape_tracker_op,
            configs=[CustomOpConfig(decomposition)],
            name="shape_tracker_autotuned",
            input_gen_fns={
                "x": lambda t: torch.randn_like(t, device=self.device),
                "weight": lambda t: torch.randn_like(t, device=self.device),
            },
            dispatch_on={"tensor_name": "x", "dim": 0},
            split_points=[128, 512],
        )

        test_x = torch.randn(1024, 64, device=self.device, dtype=self.dtype)
        test_weight = torch.randn(64, 32, device=self.device, dtype=self.dtype)

        @torch.compile(dynamic=True)
        def test_model(x, weight):
            return shape_tracker_op(x, weight)

        torch._dynamo.mark_dynamic(test_x, 0)
        torch._dynamo.reset()
        shapes_seen.clear()

        with config.patch(max_autotune=True, fx_graph_cache=False):
            result, code = torch._inductor.utils.run_and_get_code(
                test_model, test_x, test_weight
            )

        # Verify we got concrete integers during benchmarking (not symbolic)
        unique_shapes = sorted(set(shapes_seen))
        for shape in unique_shapes:
            self.assertIsInstance(shape, int, f"Expected int, got {type(shape)}")

        # Verify we hit all 3 ranges during autotuning
        ranges_hit = set()
        for shape in shapes_seen:
            if 1 <= shape <= 128:
                ranges_hit.add("range_1_128")
            elif 129 <= shape <= 512:
                ranges_hit.add("range_129_512")
            elif shape > 512:
                ranges_hit.add("range_513_inf")

        self.assertEqual(
            len(ranges_hit),
            3,
            f"Expected 3 ranges hit during benchmarking, got {ranges_hit}",
        )

        # Verify tracing uses SYMBOLIC shapes in generated code
        import re

        has_symbolic = any(re.search(r"\bs\d+\b", c) for c in code)
        self.assertTrue(has_symbolic, "Expected symbolic shapes in generated code")

    @skipIfXpu
    def test_torch_cond_with_shape_accessing_implementations(self):
        """Test torch.cond dispatch with implementations that access tensor shapes.

        Validates that implementations like decompose_k that access tensor shapes
        (e.g., `m, k = mat1.shape`) work correctly with torch.cond dispatch.
        The fix uses _build_cond_dispatch_graph to pre-trace each implementation.
        """
        test_op_name = f"test_lib::shape_access_cond_{id(self)}"

        def shape_accessing_impl(
            mat1: torch.Tensor, mat2: torch.Tensor
        ) -> torch.Tensor:
            m, k = mat1.shape  # Shape access that would break naive make_fx
            n = mat2.shape[1]
            k_splits = 4
            if k % k_splits == 0:
                k_parts = k // k_splits
                a = torch.permute(mat1.reshape(m, k_splits, k_parts), (1, 0, 2))
                b = mat2.reshape(k_splits, k_parts, n)
                return torch.sum(torch.bmm(a, b), dim=0)
            return mat1 @ mat2

        def simple_impl(mat1: torch.Tensor, mat2: torch.Tensor) -> torch.Tensor:
            return mat1 @ mat2

        @torch.library.custom_op(test_op_name, mutates_args=())
        def shape_access_op(mat1: torch.Tensor, mat2: torch.Tensor) -> torch.Tensor:
            return mat1 @ mat2

        @shape_access_op.register_fake
        def _(mat1: torch.Tensor, mat2: torch.Tensor):
            return torch.empty(
                mat1.shape[0], mat2.shape[1], device=mat1.device, dtype=mat1.dtype
            )

        register_custom_op_autotuning(
            shape_access_op,
            configs=[CustomOpConfig(simple_impl), CustomOpConfig(shape_accessing_impl)],
            name="shape_access_autotuned",
            dispatch_on={"tensor_name": "mat1", "dim": 0},
            split_points=[4, 16],
            input_gen_fns={
                "mat1": lambda t: torch.randn_like(t, device=self.device) * 0.1,
                "mat2": lambda t: torch.randn_like(t, device=self.device) * 0.1,
            },
        )

        test_mat1 = torch.randn(8, 64, device=self.device, dtype=self.dtype)
        test_mat2 = torch.randn(64, 32, device=self.device, dtype=self.dtype)

        @torch.compile(dynamic=True)
        def test_model(mat1, mat2):
            return shape_access_op(mat1, mat2)

        torch._dynamo.reset()
        with config.patch(max_autotune=True, fx_graph_cache=False):
            result = test_model(test_mat1, test_mat2)

        torch.testing.assert_close(result, test_mat1 @ test_mat2, rtol=1e-1, atol=1e-1)

    @skipIfXpu
    @unittest.skipIf(not IS_BIG_GPU, "Test requires large GPU memory")
    def test_empty_config_generator_falls_back_to_triton(self):
        """Test that empty config_generator falls back to normal mm autotuning.

        When config_generator returns empty list, the user_lowering returns None
        and graph.py falls back to the normal lowering (triton mm autotuning).
        """
        from torch._inductor.lowering import user_lowerings

        # Config generator that returns empty - should trigger fallback
        def empty_config_gen(fake_tensors):
            return []

        register_custom_op_autotuning(
            torch.ops.aten.mm.default,
            config_generator=empty_config_gen,
            name="test_empty_fallback",
        )

        self.assertIn(torch.ops.aten.mm.default, user_lowerings)

        # Use shapes that will trigger triton autotuning
        test_a = torch.randn(64, 128, device=self.device, dtype=self.dtype)
        test_b = torch.randn(128, 64, device=self.device, dtype=self.dtype)

        @torch.compile
        def test_model(a, b):
            return torch.mm(a, b)

        # Enable max_autotune with TRITON backend
        with config.patch(
            max_autotune=True,
            max_autotune_gemm_backends="TRITON",
            fx_graph_cache=False,
        ):
            result = test_model(test_a, test_b)

        # Verify correctness
        torch.testing.assert_close(
            result, torch.mm(test_a, test_b), rtol=1e-1, atol=1e-1
        )

    @skipIfXpu
    @config.patch({"test_configs.force_custom_op_decomposition": True})
    def test_guard_safety_drops_unsafe_decomposition(self):
        """Test that decompositions adding guards are replaced with fallback.

        Compiles with m=8 (divisible by 4, satisfies the unsafe impl's guard),
        then calls with m=7 (not divisible by 4). If the Mod guard leaked,
        the m=7 call would either crash or produce incorrect results because
        the compiled code assumes m is divisible by 4.
        """
        test_op_name = f"test_lib::guard_safety_{id(self)}"

        def unsafe_impl(mat1: torch.Tensor, mat2: torch.Tensor) -> torch.Tensor:
            m = mat1.shape[0]
            if m % 4 == 0:
                n = mat2.shape[1]
                k = mat1.shape[1]
                m_parts = m // 4
                a = mat1.reshape(4, m_parts, k)
                result = torch.bmm(a, mat2.unsqueeze(0).expand(4, -1, -1))
                return result.reshape(m, n)
            return mat1 @ mat2

        def safe_impl(mat1: torch.Tensor, mat2: torch.Tensor) -> torch.Tensor:
            return mat1 @ mat2

        @torch.library.custom_op(test_op_name, mutates_args=())
        def guard_safety_op(mat1: torch.Tensor, mat2: torch.Tensor) -> torch.Tensor:
            return mat1 @ mat2

        @guard_safety_op.register_fake
        def _(mat1: torch.Tensor, mat2: torch.Tensor):
            return torch.empty(
                mat1.shape[0], mat2.shape[1], device=mat1.device, dtype=mat1.dtype
            )

        register_custom_op_autotuning(
            guard_safety_op,
            configs=[CustomOpConfig(unsafe_impl)],
            name="guard_safety_autotuned",
            dispatch_on={"tensor_name": "mat1", "dim": 0},
            split_points=[4, 16],
            input_gen_fns={
                "mat1": lambda t: torch.randn_like(t, device=self.device) * 0.1,
                "mat2": lambda t: torch.randn_like(t, device=self.device) * 0.1,
            },
        )

        @torch.compile(dynamic=True)
        def test_model(mat1, mat2):
            return guard_safety_op(mat1, mat2)

        torch._dynamo.reset()
        counters.clear()

        # First call: m=8 (divisible by 4) triggers compilation
        mat1 = torch.randn(8, 64, device=self.device, dtype=self.dtype)
        mat2 = torch.randn(64, 32, device=self.device, dtype=self.dtype)
        with config.patch(max_autotune=True, fx_graph_cache=False):
            result = test_model(mat1, mat2)
        torch.testing.assert_close(result, mat1 @ mat2, rtol=1e-1, atol=1e-1)

        # Verify the guard safety counter was incremented
        self.assertGreater(
            counters["inductor"]["custom_op_decomp_guard_skips"],
            0,
            "Expected custom_op_decomp_guard_skips counter to be incremented",
        )

        # Second call: m=7 (NOT divisible by 4). If the unsafe impl was kept
        # and its Mod(m,4)==0 guard leaked, this would crash on the reshape
        # (can't reshape [7, 64] into [4, m_parts, 64] when 7 isn't divisible by 4).
        mat1_odd = torch.randn(7, 64, device=self.device, dtype=self.dtype)
        mat2_odd = torch.randn(64, 32, device=self.device, dtype=self.dtype)
        with config.patch(max_autotune=True, fx_graph_cache=False):
            result_odd = test_model(mat1_odd, mat2_odd)
        torch.testing.assert_close(
            result_odd, mat1_odd @ mat2_odd, rtol=1e-1, atol=1e-1
        )

    def test_fallback_choice_reuse(self):
        """Test that _create_fallback_choice reuses the same choice for the same op.

        Since kwargs are passed at bind time rather than baked into the kernel,
        the same ExternKernelChoice is reused across compilations.
        """
        from torch._inductor.kernel.custom_op import _create_fallback_choice

        test_op_name = f"test_lib::fallback_reuse_{id(self)}"

        @torch.library.custom_op(test_op_name, mutates_args=())
        def reuse_test_op(x: torch.Tensor) -> torch.Tensor:
            return x.clone()

        @reuse_test_op.register_fake
        def _(x: torch.Tensor):
            return torch.empty_like(x)

        op_overload = reuse_test_op._opoverload

        choice1 = _create_fallback_choice(op_overload)
        choice2 = _create_fallback_choice(op_overload)

        self.assertIs(choice1, choice2)

    @skipIfXpu
    @config.patch({"test_configs.force_custom_op_decomposition": False})
    def test_fallback_different_kwargs(self):
        """Test that the same fallback ExternKernelChoice works with different kwargs.

        Previously, kwargs were baked into the fallback wrapper at choice creation
        time, so a second call with different kwargs would either fail (duplicate
        extern kernel) or use stale kwargs. Now kwargs are passed at bind time,
        so both calls in the same compiled function get their own correct kwargs.
        """
        from torch._inductor.utils import run_and_get_code

        test_op_name = "test_lib::fallback_kwargs"

        def scale_impl(x: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
            return x * scale

        @torch.library.custom_op(test_op_name, mutates_args=())
        def kwargs_op(x: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
            return x * scale

        @kwargs_op.register_fake
        def _(x: torch.Tensor, scale: float = 1.0):
            return torch.empty_like(x)

        register_custom_op_autotuning(
            kwargs_op,
            configs=[CustomOpConfig(scale_impl)],
            name="fallback_kwargs_test",
        )

        x = torch.ones(4, 4, device=self.device, dtype=self.dtype)

        @torch.compile
        def test_model(x):
            a = kwargs_op(x, scale=2.0)
            b = kwargs_op(x, scale=5.0)
            return a, b

        torch._dynamo.reset()
        (result_a, result_b), codes = run_and_get_code(test_model, x)

        torch.testing.assert_close(result_a, x * 2.0, rtol=1e-3, atol=1e-3)
        torch.testing.assert_close(result_b, x * 5.0, rtol=1e-3, atol=1e-3)

        # Both calls should use the same fallback kernel with different kwargs in codegen
        code = "\n".join(codes)
        self.assertIn("scale=2.0", code)
        self.assertIn("scale=5.0", code)

    @skipIfXpu
    @config.patch({"test_configs.force_custom_op_decomposition": True})
    def test_config_overrides_runtime_kwargs(self):
        """Test that CustomOpConfig params override runtime default kwargs.

        The op has scale=1.0 as default, but the config specifies scale=7.0.
        With force_custom_op_decomposition=True, the decomposition should be
        called with scale=7.0 and this should be reflected in the output code.
        """
        from torch._inductor.utils import run_and_get_code

        test_op_name = f"test_lib::config_override_{id(self)}"

        def scale_impl(x: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
            return x * scale

        @torch.library.custom_op(test_op_name, mutates_args=())
        def config_override_op(x: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
            return x * scale

        @config_override_op.register_fake
        def _(x: torch.Tensor, scale: float = 1.0):
            return torch.empty_like(x)

        register_custom_op_autotuning(
            config_override_op,
            configs=[CustomOpConfig(scale_impl, scale=7.0)],
            name=f"config_override_test_{id(self)}",
        )

        x = torch.ones(4, 4, device=self.device, dtype=self.dtype)

        @torch.compile
        def test_model(x):
            return config_override_op(x)

        torch._dynamo.reset()
        result, codes = run_and_get_code(test_model, x)

        # The config's scale=7.0 should override the runtime default scale=1.0
        torch.testing.assert_close(result, x * 7.0, rtol=1e-3, atol=1e-3)
        code = "\n".join(codes)
        self.assertIn("7.0", code)

    @skipIfXpu
    def test_input_gen_fns_invoked(self):
        """Test that input_gen_fns are actually called during benchmarking."""
        test_op_name = f"test_lib::input_gen_test_{id(self)}"
        gen_calls = []

        def decomp(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
            return x * weight

        @torch.library.custom_op(test_op_name, mutates_args=())
        def input_gen_op(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
            return x * weight

        @input_gen_op.register_fake
        def _(x: torch.Tensor, weight: torch.Tensor):
            return torch.empty_like(x)

        register_custom_op_autotuning(
            input_gen_op,
            configs=[CustomOpConfig(decomp)],
            name=f"input_gen_test_{id(self)}",
            input_gen_fns={
                "x": lambda t: (gen_calls.append("x"), torch.ones_like(t))[1],
                "weight": lambda t: (gen_calls.append("weight"), torch.ones_like(t))[1],
            },
        )

        x = torch.randn(4, 4, device=self.device, dtype=self.dtype)
        w = torch.randn(4, 4, device=self.device, dtype=self.dtype)

        @torch.compile
        def test_model(x, w):
            return input_gen_op(x, w)

        torch._dynamo.reset()
        test_model(x, w)

        self.assertIn("x", gen_calls)
        self.assertIn("weight", gen_calls)

    @skipIfXpu
    @parametrize("force_choice", [None, True, False])
    def test_kwargs_codegen(self, force_choice):
        """Test that kwargs are correctly passed through codegen for both fallback and decomposition.

        This validates that when a custom op is called with non-default kwargs:
        1. Fallback path: kwargs flow through ExternKernelCaller to FallbackKernel.create
        2. Decomposition path: kwargs are passed via functools.partial to the decomposition

        Tests all paths via force_custom_op_decomposition:
        - None: normal autotuning
        - True: force decomposition
        - False: force fallback
        """
        test_op_name = f"test_lib::kwargs_codegen_{id(self)}_{force_choice}"

        def scale_impl(x: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
            return x * scale

        @torch.library.custom_op(test_op_name, mutates_args=())
        def kwargs_op(x: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
            return x * scale

        @kwargs_op.register_fake
        def _(x: torch.Tensor, scale: float = 1.0):
            return torch.empty_like(x)

        register_custom_op_autotuning(
            kwargs_op,
            configs=[CustomOpConfig(scale_impl)],
            name=f"kwargs_codegen_test_{force_choice}",
        )

        x = torch.ones(4, 4, device=self.device, dtype=self.dtype)
        scale_value = 3.0
        expected = x * scale_value

        @torch.compile
        def test_model(x):
            return kwargs_op(x, scale=scale_value)

        torch._dynamo.reset()
        with config.patch({"test_configs.force_custom_op_decomposition": force_choice}):
            result = test_model(x)

        path_name = {None: "autotuned", True: "decomposition", False: "fallback"}[
            force_choice
        ]
        torch.testing.assert_close(
            result, expected, rtol=1e-3, atol=1e-3, msg=f"{path_name} path failed"
        )

    @skipIfXpu
    @config.patch(
        {
            "test_configs.force_custom_op_decomposition": True,
        }
    )
    def test_shape_dependent_computation(self):
        """Test that decompositions using shape in computation (e.g., x * x.shape[0]) work correctly.

        This validates that make_fx tracing uses symbolic inputs so that shape-dependent
        computations produce symbolic results (e.g., x * s0) rather than concrete values
        (e.g., x * 512). The symbolic tracing is essential for correct codegen with
        dynamic shapes.

        Key validation: compile ONCE, then run with MULTIPLE sizes. If tracing used
        concrete values, the result would be wrong for different sizes.

        NOTE: We use a single-input op to avoid shape compatibility guards between
        two different dynamic tensors (which would trigger guard safety filtering).
        """
        test_op_name = f"test_lib::shape_compute_{id(self)}"

        def shape_compute_impl(x: torch.Tensor) -> torch.Tensor:
            """Scale by first dimension size - captures x * s0 with symbolic tracing."""
            return x * x.shape[0]

        @torch.library.custom_op(test_op_name, mutates_args=())
        def shape_compute_op(x: torch.Tensor) -> torch.Tensor:
            # Fallback: just returns x (no shape dependency)
            return x.clone()

        @shape_compute_op.register_fake
        def _(x: torch.Tensor):
            return torch.empty_like(x)

        register_custom_op_autotuning(
            shape_compute_op,
            configs=[CustomOpConfig(shape_compute_impl)],
            name="shape_compute_autotuned",
            input_gen_fns={
                "x": lambda t: torch.randn_like(t, device=self.device) * 0.1,
            },
        )

        @torch.compile(dynamic=True)
        def test_model(x):
            return shape_compute_op(x)

        # Compile once with initial size
        test_x = torch.randn(8, 32, 64, device=self.device, dtype=self.dtype)
        torch._dynamo.mark_dynamic(test_x, 0)

        with config.patch(max_autotune=True, fx_graph_cache=False):
            result = test_model(test_x)

        # Expected: decomposition result (x * x.shape[0]), NOT fallback (x.clone())
        expected = test_x * test_x.shape[0]
        torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-5)

        # Now test with DIFFERENT sizes using the SAME compiled model
        # If tracing used concrete values (e.g., x * 8), these would fail
        for first_dim in [16, 32, 64, 128]:
            test_x = torch.randn(
                first_dim, 32, 64, device=self.device, dtype=self.dtype
            )

            result = test_model(test_x)
            expected = test_x * first_dim  # x * x.shape[0] should use actual size

            torch.testing.assert_close(
                result,
                expected,
                rtol=1e-5,
                atol=1e-5,
                msg=f"Failed for shape[0]={first_dim}: symbolic tracing may have captured concrete value",
            )

    @skipIfXpu
    def test_partial_input_gen_fns(self):
        """Test autotuning when input_gen_fns covers only some inputs.

        The uncovered inputs should fall back to ir_node_to_tensor with concrete hints.
        """
        test_op_name = f"test_lib::partial_gen_{id(self)}"

        def decomposition(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
            return x @ weight

        @torch.library.custom_op(test_op_name, mutates_args=())
        def partial_gen_op(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
            return x @ weight

        @partial_gen_op.register_fake
        def _(x: torch.Tensor, weight: torch.Tensor):
            return torch.empty(
                x.shape[0], weight.shape[1], device=x.device, dtype=x.dtype
            )

        # Only provide input_gen_fn for "x", not "weight"
        register_custom_op_autotuning(
            partial_gen_op,
            configs=[CustomOpConfig(decomposition)],
            name="partial_gen_autotuned",
            input_gen_fns={
                "x": lambda t: torch.randn_like(t, device=self.device),
            },
        )

        test_x = torch.randn(64, 256, device=self.device, dtype=self.dtype)
        test_weight = torch.randn(256, 128, device=self.device, dtype=self.dtype)

        @torch.compile
        def test_model(x, weight):
            return partial_gen_op(x, weight)

        torch._dynamo.reset()
        with config.patch(max_autotune=True, fx_graph_cache=False):
            result = test_model(test_x, test_weight)

        torch.testing.assert_close(result, test_x @ test_weight, rtol=1e-1, atol=1e-1)

    def test_cudagraph_memory_cleanup(self):
        """Test that CUDA graph destruction automatically cleans up cuBLAS workspaces."""
        if self.device != "cuda":
            self.skipTest("CUDA graph test requires CUDA device")

        # Clear everything first
        torch.cuda.synchronize()
        torch._C._cuda_clearCublasWorkspaces()

        # Create test tensors and establish baseline with some mm activity
        a = torch.randn(256, 256, device=self.device, dtype=self.dtype)
        b = torch.randn(256, 256, device=self.device, dtype=self.dtype)
        _ = torch.mm(a, b)  # This creates cublas workspace on default stream
        torch.cuda.synchronize()

        baseline_memory = torch.cuda.memory_allocated()

        # Warmup on the stream
        _ = torch.mm(a, b)
        torch.cuda.synchronize()

        # Capture into CUDA graph
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            c = torch.mm(a, b)
        torch.cuda.synchronize()

        memory_after_capture = torch.cuda.memory_allocated()
        self.assertGreater(
            memory_after_capture, baseline_memory, "Capture should allocate memory"
        )

        # Deleting the graph should automatically clean up cuBLAS workspaces
        del graph, c
        torch.cuda.synchronize()

        memory_after_cleanup = torch.cuda.memory_allocated()

        self.assertEqual(
            memory_after_cleanup,
            baseline_memory,
            f"Memory leak detected: baseline={baseline_memory}, after_cleanup={memory_after_cleanup}",
        )

    def test_cudagraph_memory_cleanup_benchmarker(self):
        """Test that CUDA graph benchmarking cleans up memory without leaking."""
        if self.device != "cuda":
            self.skipTest("CUDA graph test requires CUDA device")

        # Clear everything first
        torch.cuda.synchronize()
        torch._C._cuda_clearCublasWorkspaces()

        # Create test tensors
        a = torch.randn(256, 256, device=self.device, dtype=self.dtype)
        b = torch.randn(256, 256, device=self.device, dtype=self.dtype)

        # Use the actual benchmarking infrastructure with CUDA graph capture
        benchmarker = torch._inductor.runtime.benchmarking.benchmarker

        def mm_callable():
            return torch.mm(a, b)

        # This should capture into CUDA graph, benchmark, and clean up properly
        _ = benchmarker.benchmark_gpu_with_cuda_graph(mm_callable)
        torch.cuda.synchronize()

        memory_after_first = torch.cuda.memory_allocated()

        # Run benchmarking again - memory should not grow
        for _ in range(3):
            _ = benchmarker.benchmark_gpu_with_cuda_graph(mm_callable)

        memory_after_many = torch.cuda.memory_allocated()

        # Memory should not grow significantly across multiple benchmark runs
        self.assertEqual(
            memory_after_many,
            memory_after_first,
            f"Memory leak detected: after_first={memory_after_first}, after_many={memory_after_many}",
        )


instantiate_parametrized_tests(TestCustomOpAutoTune)


if __name__ == "__main__":
    from torch._inductor.utils import is_big_gpu

    if HAS_GPU and HAS_CPU and is_big_gpu():
        run_tests()
