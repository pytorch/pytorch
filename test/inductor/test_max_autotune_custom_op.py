# Owner(s): ["module: inductor"]

"""
Test custom operation lowering in inductor.

Focus on testing how custom operations get lowered to subgraphs vs fallback.
"""

import unittest

import torch
from torch._decomp import register_decomposition
from torch._inductor.lowering import lowerings
from torch.library import custom_op, register_fake
from torch.testing._internal.common_utils import TestCase


# Define a simple RMSNorm custom operation
@custom_op("testlib::rms_norm", mutates_args=())
def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Root Mean Square Layer Normalization"""
    variance = x.pow(2).mean(-1, keepdim=True)
    x = x * torch.rsqrt(variance + eps)
    return x * weight


@register_fake("testlib::rms_norm")
def _(x, weight, eps=1e-6):
    assert (
        x.shape[-1] == weight.shape[-1]
    ), f"Last dim mismatch: {x.shape[-1]} vs {weight.shape[-1]}"
    return torch.empty_like(x)


@register_decomposition(torch.ops.testlib.rms_norm.default)
def rms_norm_decomposition(x, weight, eps=1e-6):
    # Use exact aten operations to work with TensorBox IR objects
    variance = torch.ops.aten.pow.Tensor_Scalar(x, 2.0)
    variance = torch.ops.aten.mean.dim(variance, [-1], True)
    variance_plus_eps = torch.ops.aten.add.Tensor(variance, eps)
    rsqrt_var = torch.ops.aten.rsqrt.default(variance_plus_eps)
    x_normalized = torch.ops.aten.mul.Tensor(x, rsqrt_var)
    return torch.ops.aten.mul.Tensor(x_normalized, weight)


def register_subgraph_lowering_for_rms_norm():
    """Register a subgraph lowering instead of using fallback"""

    def rms_norm_subgraph_lowering(x, weight, eps=1e-6):
        # Force decomposition into aten ops (subgraph lowering)
        variance = torch.ops.aten.pow.Tensor_Scalar(x, 2)
        variance = torch.ops.aten.mean.dim(variance, [-1], True)
        variance_plus_eps = torch.ops.aten.add.Tensor(variance, eps)
        rsqrt_var = torch.ops.aten.rsqrt.default(variance_plus_eps)
        x_normalized = torch.ops.aten.mul.Tensor(x, rsqrt_var)
        result = torch.ops.aten.mul.Tensor(x_normalized, weight)
        return result

    lowerings[torch.ops.testlib.rms_norm.default] = rms_norm_subgraph_lowering


class TestMaxAutotuneCustomOp(TestCase):

    def setUp(self):
        super().setUp()
        # Clean up any existing lowerings
        if torch.ops.testlib.rms_norm.default in lowerings:
            del lowerings[torch.ops.testlib.rms_norm.default]

    def tearDown(self):
        super().tearDown()
        # Clean up registered lowerings
        if torch.ops.testlib.rms_norm.default in lowerings:
            del lowerings[torch.ops.testlib.rms_norm.default]

    def test_rms_norm_fallback_behavior(self):
        """Test that custom ops use fallback by default"""
        # Simple test data
        x = torch.randn(4, 16, device="cpu", dtype=torch.float32)
        weight = torch.randn(16, device="cpu", dtype=torch.float32)

        # Test uncompiled
        result_uncompiled = torch.ops.testlib.rms_norm(x, weight)
        self.assertEqual(result_uncompiled.shape, x.shape)

        # Test compiled (should use fallback)
        @torch.compile(backend="inductor")
        def model_fallback(x, weight):
            return torch.ops.testlib.rms_norm(x, weight)

        result_fallback = model_fallback(x, weight)
        self.assertEqual(result_fallback.shape, x.shape)
        self.assertTrue(
            torch.allclose(result_uncompiled, result_fallback, atol=1e-4, rtol=1e-4)
        )

    def test_rms_norm_subgraph_lowering(self):
        """Test custom op with forced subgraph lowering"""
        # Register subgraph lowering
        register_subgraph_lowering_for_rms_norm()

        # Test data
        x = torch.randn(4, 16, device="cpu", dtype=torch.float32)
        weight = torch.randn(16, device="cpu", dtype=torch.float32)

        @torch.compile(backend="inductor")
        def model_subgraph(x, weight):
            return torch.ops.testlib.rms_norm(x, weight)

        result_subgraph = model_subgraph(x, weight)
        self.assertEqual(result_subgraph.shape, x.shape)

        # Compare with manual decomposition
        expected = rms_norm_decomposition(x, weight)
        self.assertTrue(torch.allclose(result_subgraph, expected, atol=1e-5, rtol=1e-5))

    def test_lowering_registry_modification(self):
        """Test that we can modify the lowering registry"""
        # Initially should not be in registry
        self.assertNotIn(torch.ops.testlib.rms_norm.default, lowerings)

        # Register subgraph lowering
        register_subgraph_lowering_for_rms_norm()

        # Should now be in registry
        self.assertIn(torch.ops.testlib.rms_norm.default, lowerings)

        # Test that it works
        x = torch.randn(2, 8, device="cpu")
        weight = torch.randn(8, device="cpu")

        @torch.compile(backend="inductor")
        def model(x, weight):
            return torch.ops.testlib.rms_norm(x, weight)

        result = model(x, weight)
        self.assertEqual(result.shape, x.shape)

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA required for GPU test")
    def test_rms_norm_cuda_basic(self):
        """Basic CUDA test for RMS norm"""
        register_subgraph_lowering_for_rms_norm()

        x = torch.randn(4, 32, device="cuda", dtype=torch.float16)
        weight = torch.randn(32, device="cuda", dtype=torch.float16)

        @torch.compile(backend="inductor")
        def model(x, weight):
            return torch.ops.testlib.rms_norm(x, weight)

        result = model(x, weight)
        self.assertEqual(result.shape, x.shape)
        self.assertEqual(result.device.type, "cuda")


if __name__ == "__main__":
    unittest.main()
