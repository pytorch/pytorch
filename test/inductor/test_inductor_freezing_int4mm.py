# Owner(s): ["module: inductor"]
"""
Tests for INT4 GPU matmul fusion in freezing_patterns.py

This tests the pattern:
    (int4_mm(x, w1, gs, sz1), int4_mm(x, w2, gs, sz2), int4_mm(x, w3, gs, sz3))
    -> split(int4_mm(x, cat(w1,w2,w3), gs, cat(sz1,sz2,sz3)), sizes)

Requires: torchao, CUDA or MPS
"""

import unittest

import torch
from torch import nn
from torch._dynamo.utils import counters
from torch._inductor import config
from torch._inductor.test_case import TestCase as InductorTestCase
from torch._inductor.utils import run_and_get_code
from torch.testing import FileCheck
from torch.testing._internal.common_utils import TEST_WITH_ROCM


HAS_CUDA = torch.cuda.is_available()
HAS_MPS = torch.backends.mps.is_available()
HAS_CUDA_OR_MPS = HAS_CUDA or HAS_MPS
DEVICE = "cuda" if HAS_CUDA else "mps" if HAS_MPS else "cpu"


@unittest.skipIf(not HAS_CUDA_OR_MPS, "requires CUDA or MPS")
@unittest.skipIf(TEST_WITH_ROCM, "INT4 packed format not supported on ROCm")
class Int4MMFusionTests(InductorTestCase):
    """Tests for INT4 GPU matmul fusion during freezing."""

    device = DEVICE

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        # Import torchao here (after skip check) to avoid import errors on CPU CI
        from torchao.quantization import Int4WeightOnlyConfig, quantize_
        from torchao.quantization.quantize_.workflows.int4.int4_packing_format import (
            Int4PackingFormat,
        )

        cls.Int4WeightOnlyConfig = Int4WeightOnlyConfig
        cls.Int4PackingFormat = Int4PackingFormat
        # Use staticmethod to prevent Python from binding as instance method
        cls.quantize_model = staticmethod(quantize_)

    def setUp(self):
        super().setUp()
        torch._dynamo.reset()
        counters.clear()

    def tearDown(self):
        super().tearDown()
        torch._dynamo.reset()

    def test_int4_mm_fusion_three_way(self):
        """Test 3-way INT4 matmul fusion (like Q/K/V projections)."""

        class QKVProjection(nn.Module):
            def __init__(self, dim=512):
                super().__init__()
                self.q_proj = nn.Linear(dim, dim, bias=False)
                self.k_proj = nn.Linear(dim, dim, bias=False)
                self.v_proj = nn.Linear(dim, dim, bias=False)

            def forward(self, x):
                return self.q_proj(x), self.k_proj(x), self.v_proj(x)

        model = QKVProjection(dim=512).eval().to(self.device).to(torch.bfloat16)

        # Quantize to INT4
        int4_config = self.Int4WeightOnlyConfig(
            group_size=128,
            int4_packing_format=self.Int4PackingFormat.TILE_PACKED_TO_4D,
        )
        self.quantize_model(model, int4_config)

        x = torch.randn(2, 128, 512, device=self.device, dtype=torch.bfloat16)

        # Get reference output
        with torch.no_grad():
            ref = model(x)

        # Compile with freezing
        with config.patch({"freezing": True}):
            compiled_model = torch.compile(model)

            with torch.no_grad():
                out, code = run_and_get_code(compiled_model, x)

        # Verify output correctness
        for i in range(3):
            self.assertEqual(ref[i], out[i])

        # Verify fusion: should be 1 fused INT4 matmul, not 3 separate ones
        FileCheck().check_count(
            "= torch.ops.aten._weight_int4pack_mm.default(", 1, exactly=True
        ).run(code[0])

    def test_int4_mm_fusion_two_way(self):
        """Test 2-way INT4 matmul fusion (like K/V in cross-attention)."""

        class KVProjection(nn.Module):
            def __init__(self, dim=512):
                super().__init__()
                self.k_proj = nn.Linear(dim, dim, bias=False)
                self.v_proj = nn.Linear(dim, dim, bias=False)

            def forward(self, x):
                return self.k_proj(x), self.v_proj(x)

        model = KVProjection(dim=512).eval().to(self.device).to(torch.bfloat16)

        # Quantize to INT4
        int4_config = self.Int4WeightOnlyConfig(
            group_size=128,
            int4_packing_format=self.Int4PackingFormat.TILE_PACKED_TO_4D,
        )
        self.quantize_model(model, int4_config)

        x = torch.randn(2, 128, 512, device=self.device, dtype=torch.bfloat16)

        # Get reference output
        with torch.no_grad():
            ref = model(x)

        # Compile with freezing
        with config.patch({"freezing": True}):
            compiled_model = torch.compile(model)

            with torch.no_grad():
                out, code = run_and_get_code(compiled_model, x)

        # Verify output correctness
        for i in range(2):
            self.assertEqual(ref[i], out[i])

        # Verify fusion: should be 1 fused INT4 matmul, not 2 separate ones
        FileCheck().check_count(
            "= torch.ops.aten._weight_int4pack_mm.default(", 1, exactly=True
        ).run(code[0])

    def test_int4_mm_fusion_different_group_sizes(self):
        """Test INT4 fusion works with different group_size values."""

        class TwoLinear(nn.Module):
            def __init__(self, dim=512):
                super().__init__()
                self.linear1 = nn.Linear(dim, dim, bias=False)
                self.linear2 = nn.Linear(dim, dim, bias=False)

            def forward(self, x):
                return self.linear1(x), self.linear2(x)

        # Test all supported group sizes
        for group_size in [32, 64, 128, 256]:
            model = TwoLinear(dim=512).eval().to(self.device).to(torch.bfloat16)

            int4_config = self.Int4WeightOnlyConfig(
                group_size=group_size,
                int4_packing_format=self.Int4PackingFormat.TILE_PACKED_TO_4D,
            )
            self.quantize_model(model, int4_config)

            x = torch.randn(2, 64, 512, device=self.device, dtype=torch.bfloat16)

            with torch.no_grad():
                ref = model(x)

            torch._dynamo.reset()

            with config.patch(
                {
                    "freezing": True,
                    "freezing_discard_parameters": False,
                    "implicit_fallbacks": True,
                }
            ):
                compiled_model = torch.compile(model)

                with torch.no_grad():
                    out, code = run_and_get_code(compiled_model, x)

            # Verify correctness
            for i in range(2):
                self.assertEqual(ref[i], out[i])

            # Verify fusion: should be 1 fused INT4 matmul
            FileCheck().check_count(
                "= torch.ops.aten._weight_int4pack_mm.default(", 1, exactly=True
            ).run(code[0])

    def test_int4_mm_no_fusion_different_inputs(self):
        """Test that INT4 matmuls with different inputs are NOT fused."""

        class DifferentInputs(nn.Module):
            def __init__(self, dim=512):
                super().__init__()
                self.linear1 = nn.Linear(dim, dim, bias=False)
                self.linear2 = nn.Linear(dim, dim, bias=False)

            def forward(self, x, y):
                # Different inputs - should NOT be fused
                return self.linear1(x), self.linear2(y)

        model = DifferentInputs(dim=512).eval().to(self.device).to(torch.bfloat16)

        int4_config = self.Int4WeightOnlyConfig(
            group_size=128,
            int4_packing_format=self.Int4PackingFormat.TILE_PACKED_TO_4D,
        )
        self.quantize_model(model, int4_config)

        x = torch.randn(2, 64, 512, device=self.device, dtype=torch.bfloat16)
        y = torch.randn(2, 64, 512, device=self.device, dtype=torch.bfloat16)

        with torch.no_grad():
            ref = model(x, y)

        with config.patch({"freezing": True}):
            compiled_model = torch.compile(model)

            with torch.no_grad():
                out, code = run_and_get_code(compiled_model, x, y)

        # Verify correctness
        for i in range(2):
            self.assertEqual(ref[i], out[i])

        # Should NOT be fused - expect 2 separate calls (different inputs)
        FileCheck().check_count(
            "= torch.ops.aten._weight_int4pack_mm.default(", 2, exactly=True
        ).run(code[0])

    def test_int4_mm_no_fusion_different_k_dims(self):
        """Test that INT4 matmuls with different K dimensions are NOT fused."""

        class DifferentKDims(nn.Module):
            def __init__(self):
                super().__init__()
                # Different input dimensions (K) - should NOT be fused
                self.linear1 = nn.Linear(512, 256, bias=False)
                self.linear2 = nn.Linear(256, 256, bias=False)

            def forward(self, x):
                # Apply linear1, then use its output for linear2
                # Both share same output dim but different input dims
                out1 = self.linear1(x)
                out2 = self.linear2(out1)
                return out1, out2

        model = DifferentKDims().eval().to(self.device).to(torch.bfloat16)

        int4_config = self.Int4WeightOnlyConfig(
            group_size=128,
            int4_packing_format=self.Int4PackingFormat.TILE_PACKED_TO_4D,
        )
        self.quantize_model(model, int4_config)

        x = torch.randn(2, 64, 512, device=self.device, dtype=torch.bfloat16)

        with torch.no_grad():
            ref = model(x)

        torch._dynamo.reset()

        with config.patch({"freezing": True}):
            compiled_model = torch.compile(model)

            with torch.no_grad():
                out, code = run_and_get_code(compiled_model, x)

        # Verify correctness
        for i in range(2):
            self.assertEqual(ref[i], out[i])

        # Should NOT be fused due to different K dimensions - expect 2 separate calls
        FileCheck().check_count(
            "= torch.ops.aten._weight_int4pack_mm.default(", 2, exactly=True
        ).run(code[0])

    def test_int4_mm_fusion_different_output_dims(self):
        """Test INT4 fusion with different output dimensions (asymmetric projections)."""

        class AsymmetricProjection(nn.Module):
            def __init__(self):
                super().__init__()
                # Different output dimensions - should still fuse
                self.proj1 = nn.Linear(512, 256, bias=False)
                self.proj2 = nn.Linear(512, 256, bias=False)
                self.proj3 = nn.Linear(512, 256, bias=False)

            def forward(self, x):
                return self.proj1(x), self.proj2(x), self.proj3(x)

        model = AsymmetricProjection().eval().to(self.device).to(torch.bfloat16)

        int4_config = self.Int4WeightOnlyConfig(
            group_size=128,
            int4_packing_format=self.Int4PackingFormat.TILE_PACKED_TO_4D,
        )
        self.quantize_model(model, int4_config)

        x = torch.randn(2, 64, 512, device=self.device, dtype=torch.bfloat16)

        with torch.no_grad():
            ref = model(x)

        torch._dynamo.reset()

        with config.patch({"freezing": True}):
            compiled_model = torch.compile(model)

            with torch.no_grad():
                out, code = run_and_get_code(compiled_model, x)

        # Verify correctness
        for i in range(3):
            self.assertEqual(ref[i], out[i])

        # Should be fused into 1 call (different output dims is OK)
        FileCheck().check_count(
            "= torch.ops.aten._weight_int4pack_mm.default(", 1, exactly=True
        ).run(code[0])

    # Note: Testing float16 or mixed dtypes is not possible with torchao's
    # Int4TilePackedTo4dTensor which only supports bfloat16. The dtype validation
    # in check_int4_gpu_concat_weights is defensive for future format support.
    #
    # Note: Testing non-constant weights is not possible because freezing always
    # converts weights to constants (get_attr). The get_attr check is still important
    # to guard against edge cases or future changes to the freezing behavior.


if __name__ == "__main__":
    from torch._inductor.test_case import run_tests

    if HAS_CUDA_OR_MPS:
        run_tests(needs="filelock")
