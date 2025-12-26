# Owner(s): ["oncall: pt2"]
"""
Deprecation test for old AOT compile API.

This test verifies TODO 14: Deprecate old `aot_compile` / `save_compiled_function` /
`load_compiled_function` pattern.

Tests that:
1. The old API functions emit FutureWarning deprecation warnings
2. The new torch.Precompile API works as a replacement
"""

import io
import warnings

import torch
import torch.nn as nn

from torch.testing._internal.common_utils import run_tests, TestCase


class SimpleModel(nn.Module):
    def forward(self, x):
        return x * 2


class TestDeprecationWarnings(TestCase):
    def test_load_compiled_function_deprecation_warning(self):
        """Test that torch.compiler.load_compiled_function emits a deprecation warning."""
        # Create a dummy file-like object (we expect it to fail during deserialization,
        # but the warning should be emitted before that)
        dummy_file = io.BytesIO(b"dummy data")

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            try:
                torch.compiler.load_compiled_function(dummy_file)
            except Exception:
                # We expect this to fail due to invalid data, but we only care about the warning
                pass

            # Check that a FutureWarning was emitted
            deprecation_warnings = [
                warning
                for warning in w
                if issubclass(warning.category, FutureWarning)
                and "load_compiled_function" in str(warning.message)
            ]
            self.assertTrue(
                len(deprecation_warnings) > 0,
                "load_compiled_function should emit a FutureWarning deprecation warning",
            )
            self.assertIn(
                "torch.Precompile",
                str(deprecation_warnings[0].message),
                "Deprecation warning should mention torch.Precompile as replacement",
            )

    def test_new_precompile_api_works_as_replacement(self):
        """Test that the new Precompile API works as a replacement for the old pattern."""
        model = SimpleModel()
        example_input = torch.randn(2, 3)

        # New API should work
        gm, _, guards, example_inputs = torch.Precompile.dynamo(model, example_input)
        compiled_fn, _ = torch.Precompile.aot_autograd(gm, guards)

        # Verify the compiled function produces correct output
        with torch.no_grad():
            output = compiled_fn(example_inputs)
            expected = model(example_input)

        self.assertTrue(
            torch.allclose(output, expected),
            "New Precompile API should produce correct output",
        )

    def test_precompile_api_available_in_torch_namespace(self):
        """Test that torch.Precompile is accessible and has required methods."""
        self.assertTrue(
            hasattr(torch, "Precompile"), "torch.Precompile should be accessible"
        )
        self.assertTrue(
            hasattr(torch.Precompile, "dynamo"),
            "torch.Precompile.dynamo should be accessible",
        )
        self.assertTrue(
            hasattr(torch.Precompile, "aot_autograd"),
            "torch.Precompile.aot_autograd should be accessible",
        )


if __name__ == "__main__":
    run_tests()
