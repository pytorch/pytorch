# Owner(s): ["oncall: pt2"]
"""
End-to-end test: Transformer with regional inductor.

This test verifies TODO 12: End-to-end test with regional inductor on Transformer.
Tests the full beautiful.py flow with a Transformer model.
Uses fx_traceback.annotate() to mark attention and MLP regions.
Verifies output matches eager execution.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.fx import traceback as fx_traceback
from torch.fx.passes.regional_inductor import regional_inductor
from torch.testing._internal.common_utils import run_tests, TestCase


class SimpleAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        with fx_traceback.annotate({"compile_with_inductor": "attention"}):
            out = F.softmax(self.proj(x), dim=-1)
        return out


class SimpleTransformer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.attn = SimpleAttention(dim)
        self.mlp = nn.Linear(dim, dim)

    def forward(self, x):
        x = self.attn(x)
        with fx_traceback.annotate({"compile_with_inductor": "mlp"}):
            x = self.mlp(x)
        return x


class TestE2ETransformerRegionalInductor(TestCase):
    def test_simple_transformer_with_regional_inductor(self):
        """Test end-to-end Transformer with regional inductor compilation."""
        torch.manual_seed(42)
        model = SimpleTransformer(32)
        example_input = torch.randn(2, 16, 32)

        # Trace with Precompile.dynamo()
        gm, _, guards, example_inputs = torch.Precompile.dynamo(model, example_input)

        # Compile with Precompile.aot_autograd() using regional_inductor
        compiled_fn, _ = torch.Precompile.aot_autograd(
            gm, guards, compiler=regional_inductor
        )

        # Run inference and compare
        with torch.no_grad():
            compiled_out = compiled_fn(example_inputs)
            eager_out = model(example_input)

        self.assertTrue(
            torch.allclose(compiled_out, eager_out, atol=1e-4),
            f"Transformer output should match. Max diff: {(compiled_out - eager_out).abs().max()}",
        )

    def test_transformer_attention_only_annotated(self):
        """Test Transformer with only attention region annotated."""

        class AttentionOnlyTransformer(nn.Module):
            def __init__(self, dim):
                super().__init__()
                self.proj = nn.Linear(dim, dim)
                self.mlp = nn.Linear(dim, dim)

            def forward(self, x):
                with fx_traceback.annotate({"compile_with_inductor": "attention"}):
                    x = F.softmax(self.proj(x), dim=-1)
                x = self.mlp(x)
                return x

        torch.manual_seed(123)
        model = AttentionOnlyTransformer(16)
        example_input = torch.randn(2, 8, 16)

        gm, _, guards, example_inputs = torch.Precompile.dynamo(model, example_input)
        compiled_fn, _ = torch.Precompile.aot_autograd(
            gm, guards, compiler=regional_inductor
        )

        with torch.no_grad():
            compiled_out = compiled_fn(example_inputs)
            eager_out = model(example_input)

        self.assertTrue(
            torch.allclose(compiled_out, eager_out, atol=1e-4),
            "Attention-only annotated Transformer should match eager",
        )

    def test_transformer_mlp_only_annotated(self):
        """Test Transformer with only MLP region annotated."""

        class MLPOnlyTransformer(nn.Module):
            def __init__(self, dim):
                super().__init__()
                self.attn_proj = nn.Linear(dim, dim)
                self.mlp = nn.Linear(dim, dim)

            def forward(self, x):
                x = F.softmax(self.attn_proj(x), dim=-1)
                with fx_traceback.annotate({"compile_with_inductor": "mlp"}):
                    x = self.mlp(x)
                return x

        torch.manual_seed(456)
        model = MLPOnlyTransformer(16)
        example_input = torch.randn(2, 8, 16)

        gm, _, guards, example_inputs = torch.Precompile.dynamo(model, example_input)
        compiled_fn, _ = torch.Precompile.aot_autograd(
            gm, guards, compiler=regional_inductor
        )

        with torch.no_grad():
            compiled_out = compiled_fn(example_inputs)
            eager_out = model(example_input)

        self.assertTrue(
            torch.allclose(compiled_out, eager_out, atol=1e-4),
            "MLP-only annotated Transformer should match eager",
        )

    def test_transformer_multiple_layers(self):
        """Test Transformer with multiple attention and MLP layers."""

        class MultiLayerTransformer(nn.Module):
            def __init__(self, dim, num_layers=2):
                super().__init__()
                self.layers = nn.ModuleList()
                for _ in range(num_layers):
                    self.layers.append(
                        nn.ModuleDict(
                            {
                                "attn": nn.Linear(dim, dim),
                                "mlp": nn.Linear(dim, dim),
                            }
                        )
                    )

            def forward(self, x):
                for i, layer in enumerate(self.layers):
                    with fx_traceback.annotate(
                        {"compile_with_inductor": f"attention_{i}"}
                    ):
                        x = F.softmax(layer["attn"](x), dim=-1)
                    with fx_traceback.annotate({"compile_with_inductor": f"mlp_{i}"}):
                        x = layer["mlp"](x)
                return x

        torch.manual_seed(789)
        model = MultiLayerTransformer(16, num_layers=2)
        example_input = torch.randn(2, 4, 16)

        gm, _, guards, example_inputs = torch.Precompile.dynamo(model, example_input)
        compiled_fn, _ = torch.Precompile.aot_autograd(
            gm, guards, compiler=regional_inductor
        )

        with torch.no_grad():
            compiled_out = compiled_fn(example_inputs)
            eager_out = model(example_input)

        self.assertTrue(
            torch.allclose(compiled_out, eager_out, atol=1e-4),
            "Multi-layer Transformer should match eager",
        )

    def test_transformer_no_annotations(self):
        """Test that regional_inductor still works with no annotated regions."""

        class UnannotatedTransformer(nn.Module):
            def __init__(self, dim):
                super().__init__()
                self.attn_proj = nn.Linear(dim, dim)
                self.mlp = nn.Linear(dim, dim)

            def forward(self, x):
                x = F.softmax(self.attn_proj(x), dim=-1)
                x = self.mlp(x)
                return x

        torch.manual_seed(1000)
        model = UnannotatedTransformer(16)
        example_input = torch.randn(2, 4, 16)

        gm, _, guards, example_inputs = torch.Precompile.dynamo(model, example_input)
        compiled_fn, _ = torch.Precompile.aot_autograd(
            gm, guards, compiler=regional_inductor
        )

        with torch.no_grad():
            compiled_out = compiled_fn(example_inputs)
            eager_out = model(example_input)

        self.assertTrue(
            torch.allclose(compiled_out, eager_out, atol=1e-4),
            "Unannotated Transformer should match eager (no regions compiled)",
        )


if __name__ == "__main__":
    run_tests()
