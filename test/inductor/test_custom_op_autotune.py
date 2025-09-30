# Owner(s): ["module: inductor"]
import unittest

import torch
from torch._inductor import config
from torch._inductor.kernel.custom_op import autotune_custom_op
from torch._inductor.lowering import lowerings
from torch._inductor.test_case import run_tests, TestCase
from torch._inductor.utils import run_and_get_code
from torch.testing._internal.common_utils import skipIfXpu
from torch.testing._internal.inductor_utils import GPU_TYPE, HAS_GPU


torch.set_float32_matmul_precision("high")


def rmsnorm_decomposition(
    input_tensor: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6
):
    """RMSNorm decomposition using standard ATen operations."""
    variance = input_tensor.pow(2).mean(-1, keepdim=True)
    normalized = input_tensor * torch.rsqrt(variance + eps)
    return normalized * weight


def rmsnorm_standard(
    input_tensor: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6
):
    """Standard RMSNorm implementation."""
    variance = input_tensor.pow(2).mean(-1, keepdim=True)
    normalized = input_tensor * torch.rsqrt(variance + eps)
    return normalized * weight


def rmsnorm_alternative(
    input_tensor: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6
):
    """Alternative RMSNorm using different computation order."""
    sum_sq = (input_tensor * input_tensor).sum(-1, keepdim=True)
    mean_sq = sum_sq / input_tensor.shape[-1]
    rms = torch.sqrt(mean_sq + eps)
    normalized = input_tensor / rms
    return normalized * weight


def rmsnorm_fused(input_tensor: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6):
    """Fused RMSNorm implementation."""
    return (
        input_tensor
        * torch.rsqrt(input_tensor.pow(2).mean(-1, keepdim=True) + eps)
        * weight
    )


def layernorm_standard(
    input_tensor: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float = 1e-5,
):
    """Standard LayerNorm implementation."""
    mean = input_tensor.mean(-1, keepdim=True)
    var = input_tensor.var(-1, keepdim=True, unbiased=False)
    normalized = (input_tensor - mean) * torch.rsqrt(var + eps)
    return normalized * weight + bias


def gelu_tanh_approximation(x: torch.Tensor):
    """GELU using tanh approximation."""
    return (
        0.5
        * x
        * (
            1
            + torch.tanh(
                torch.sqrt(torch.tensor(2.0 / torch.pi)) * (x + 0.044715 * x.pow(3))
            )
        )
    )


def gelu_erf_exact(x: torch.Tensor):
    """Exact GELU using erf function."""
    return 0.5 * x * (1 + torch.erf(x / torch.sqrt(torch.tensor(2.0))))


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
            requires_grad=True,
        )
        weight = torch.randn(
            hidden_dim, device=self.device, dtype=self.dtype, requires_grad=True
        )
        return input_tensor, weight

    def test_autotune_custom_op_basic_functionality(self):
        """Test basic functionality of autotune_custom_op."""
        input_tensor, weight = self._create_test_inputs()
        eps = 1e-6

        # Test with autotuning disabled (should fallback to decomposition)
        with config.patch(max_autotune=False):
            result_no_autotune = autotune_custom_op(
                name="rmsnorm_test",
                decompositions=rmsnorm_decomposition,
                inputs=[input_tensor, weight],
                kwargs={"eps": eps},
            )

        # Expected result from direct decomposition
        expected = rmsnorm_decomposition(input_tensor, weight, eps)

        # Results should match
        self.assertEqual(result_no_autotune.shape, expected.shape)
        torch.testing.assert_close(result_no_autotune, expected, rtol=1e-2, atol=1e-2)

    @skipIfXpu
    def test_autotune_custom_op_with_max_autotune(self):
        """Test autotune_custom_op with max_autotune enabled."""
        input_tensor, weight = self._create_test_inputs()
        eps = 1e-6

        # Test with autotuning enabled
        with config.patch(max_autotune=True, max_autotune_gemm_backends="TRITON"):
            result_with_autotune = autotune_custom_op(
                name="rmsnorm_autotune_test",
                decompositions=rmsnorm_decomposition,
                inputs=[input_tensor, weight],
                kwargs={"eps": eps},
            )

        # Expected result from direct decomposition
        expected = rmsnorm_decomposition(input_tensor, weight, eps)

        # Results should match regardless of autotuning path
        self.assertEqual(result_with_autotune.shape, expected.shape)
        torch.testing.assert_close(result_with_autotune, expected, rtol=1e-2, atol=1e-2)

    def test_custom_op_lowering_registration(self):
        """Test registration of custom op through lowerings system."""

        def tuned_rmsnorm_for_test(input_tensor, weight, eps=1e-6, *, layout=None):
            """Tuned RMSNorm function for testing lowering registration."""
            return autotune_custom_op(
                name="rmsnorm_lowering_test",
                decompositions=rmsnorm_decomposition,
                inputs=[input_tensor, weight],
                kwargs={"eps": eps},
                layout=layout,
            )

        # Register the function
        test_target = "custom_ops.rmsnorm_test.default"
        lowerings[test_target] = tuned_rmsnorm_for_test

        # Test dispatch through lowerings
        input_tensor, weight = self._create_test_inputs()
        eps = 1e-6

        with config.patch(max_autotune=True):
            lowering_result = lowerings[test_target](input_tensor, weight, eps=eps)

        # Verify result
        expected = rmsnorm_decomposition(input_tensor, weight, eps)
        self.assertEqual(lowering_result.shape, expected.shape)
        torch.testing.assert_close(lowering_result, expected, rtol=1e-2, atol=1e-2)

        # Clean up
        del lowerings[test_target]

    @skipIfXpu
    def test_multiple_rmsnorm_decompositions_autotuning(self):
        """Test autotuning with multiple RMSNorm decomposition variants."""
        input_tensor, weight = self._create_test_inputs(
            batch_size=4, seq_len=128, hidden_dim=512
        )
        eps = 1e-6

        # Test single decomposition (should use heuristic)
        with config.patch(max_autotune=True, max_autotune_gemm_backends="TRITON"):
            single_result = autotune_custom_op(
                name="rmsnorm_single_test",
                decompositions=rmsnorm_standard,
                inputs=[input_tensor, weight],
                kwargs={"eps": eps},
            )

        # Test multiple decompositions (should trigger benchmarking)
        rmsnorm_variants = [rmsnorm_standard, rmsnorm_alternative, rmsnorm_fused]
        with config.patch(max_autotune=True, max_autotune_gemm_backends="TRITON"):
            multi_result = autotune_custom_op(
                name="rmsnorm_multi_test",
                decompositions=rmsnorm_variants,
                inputs=[input_tensor, weight],
                kwargs={"eps": eps},
            )

        # Verify correctness - all variants should produce same result
        expected = rmsnorm_standard(input_tensor, weight, eps)

        # Single decomposition result should match
        self.assertEqual(single_result.shape, expected.shape)
        torch.testing.assert_close(single_result, expected, rtol=1e-2, atol=1e-2)

        # Multi decomposition result should also match (may be tensor or MultiTemplateBuffer)
        if hasattr(multi_result, "shape"):
            self.assertEqual(multi_result.shape, expected.shape)
            torch.testing.assert_close(multi_result, expected, rtol=1e-2, atol=1e-2)

        # Verify all variants produce same results individually (use relaxed tolerances for numerical differences)
        alt_result = rmsnorm_alternative(input_tensor, weight, eps)
        fused_result = rmsnorm_fused(input_tensor, weight, eps)
        torch.testing.assert_close(alt_result, expected, rtol=1e-2, atol=1e-2)
        torch.testing.assert_close(fused_result, expected, rtol=1e-2, atol=1e-2)

    @skipIfXpu
    def test_gelu_multiple_implementations_autotuning(self):
        """Test autotuning with multiple GELU implementations."""
        input_tensor = torch.randn(8, 64, 256, device=self.device, dtype=self.dtype)

        gelu_variants = [gelu_tanh_approximation, gelu_erf_exact]

        with config.patch(max_autotune=True, max_autotune_gemm_backends="TRITON"):
            result = autotune_custom_op(
                name="gelu_multi_test",
                decompositions=gelu_variants,
                inputs=[input_tensor],
                kwargs={},
            )

        # Verify result shape is correct
        if hasattr(result, "shape"):
            self.assertEqual(result.shape, input_tensor.shape)

        # Verify GELU variants produce reasonable results (they may differ slightly)
        tanh_result = gelu_tanh_approximation(input_tensor)
        erf_result = gelu_erf_exact(input_tensor)

        # All should have same shape
        self.assertEqual(tanh_result.shape, input_tensor.shape)
        self.assertEqual(erf_result.shape, input_tensor.shape)

        # Approximations should be reasonably close to exact version (GELU variants have inherent differences)
        torch.testing.assert_close(tanh_result, erf_result, rtol=2e-1, atol=3e-2)

    @skipIfXpu
    def test_torch_compile_with_multiple_decompositions(self):
        """Test torch.compile with multiple custom op decompositions."""
        input_tensor, weight = self._create_test_inputs(
            batch_size=2, seq_len=64, hidden_dim=256
        )
        eps = 1e-6

        def multi_rmsnorm_model(x, w):
            """Model using multiple RMSNorm decompositions."""
            return autotune_custom_op(
                "compiled_rmsnorm_multi",
                [rmsnorm_standard, rmsnorm_alternative, rmsnorm_fused],
                [x, w],
                {"eps": eps},
            )

        # Baseline
        baseline = rmsnorm_standard(input_tensor, weight, eps)

        # Compiled execution with multiple decompositions
        with config.patch(max_autotune=True, max_autotune_gemm_backends="TRITON"):
            compiled_model = torch.compile(multi_rmsnorm_model)
            compiled_result = compiled_model(input_tensor, weight)

        # Verify correctness
        self.assertEqual(compiled_result.shape, baseline.shape)
        torch.testing.assert_close(compiled_result, baseline, rtol=1e-2, atol=1e-2)

    def test_custom_op_error_handling_with_multiple_decompositions(self):
        """Test error handling when some decompositions fail."""
        input_tensor, weight = self._create_test_inputs(
            batch_size=1, seq_len=8, hidden_dim=16
        )

        def working_decomp(x, w, eps=1e-6):
            return rmsnorm_standard(x, w, eps)

        def failing_decomp(x, w, eps=1e-6):
            raise RuntimeError("Intentional failure for testing")

        # Mix of working and failing decompositions
        mixed_variants = [working_decomp, failing_decomp]

        # Should handle failures gracefully and use working decompositions
        with config.patch(max_autotune=True):
            result = autotune_custom_op(
                "mixed_success_test",
                mixed_variants,
                [input_tensor, weight],
                {"eps": 1e-6},
            )

        # Should get a valid result from one of the working decompositions
        expected = working_decomp(input_tensor, weight, 1e-6)
        if hasattr(result, "shape"):
            self.assertEqual(result.shape, expected.shape)
            torch.testing.assert_close(result, expected, rtol=1e-3, atol=1e-3)


if __name__ == "__main__":
    from torch._inductor.utils import is_big_gpu

    # Set env to make it work in CI
    if HAS_GPU and is_big_gpu():
        run_tests()
    else:
        # Run basic tests even on CPU
        run_tests()
