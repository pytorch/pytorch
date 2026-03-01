# Owner(s): ["module: dynamo"]
"""
Test for unbacked_bindings KeyError fix in slice/cat operations.

Regression tests for PyTorch issue #174340 where slicing tensors
with inherited unbacked symbols (from cat/repeat_interleave) caused
a KeyError during Inductor lowering.

The bug occurred because compute_unbacked_bindings() only tracks "fresh"
unbacked symbols. When operations like cat/slice work with existing unbacked
symbols inherited from inputs, no unbacked_bindings metadata was created,
causing KeyError when accessing node.meta["unbacked_bindings"] in lowering.

The fix adds fallback logic in ShapeProp and FakeTensorProp that checks
if result contains ANY unbacked symbols (not just fresh ones) and creates
bindings for them using _free_unbacked_symbols_with_path.
"""

import unittest

import torch
import torch._dynamo as dynamo
from torch._dynamo.testing import CompileCounter


class TestUnbackedBindings(unittest.TestCase):
    """Test suite for unbacked_bindings metadata propagation fix."""

    def setUp(self):
        """Reset dynamo state before each test."""
        dynamo.reset()

    def _cat_slice_fn(self, x, repeats, use_select=False):
        """
        Helper function that triggers the unbacked_bindings bug.

        Creates a pattern: repeat_interleave (creates u0) -> cat (u0+1) -> slice
        This pattern previously caused KeyError: 'unbacked_bindings'.
        """
        reps_t = torch.tensor(repeats, device=x.device, dtype=torch.long)
        vals = torch.repeat_interleave(x, reps_t)
        vals = torch.cat([torch.zeros(1, device=vals.device, dtype=vals.dtype), vals])
        if use_select:
            vals = vals.unsqueeze(1)
            vals = torch.cat([vals, vals], dim=1)
            return vals[:, 0]
        return vals[1:]

    def test_cat_slice_unbacked_bindings(self):
        """
        Test basic repeat_interleave -> cat -> slice pattern.

        This is the exact reproduction case from issue #174340.
        Before fix: KeyError: 'unbacked_bindings' in slice lowering.
        After fix: Compiles and runs successfully.
        """
        x = torch.tensor([10, 20, 30], dtype=torch.int32)
        repeats = [2, 3, 1]
        expected = torch.tensor([10, 10, 20, 20, 20, 30], dtype=torch.int32)

        # Test eager execution
        eager_result = self._cat_slice_fn(x, repeats)
        self.assertTrue(torch.equal(eager_result, expected))

        # Test compiled execution - should not raise KeyError
        compiled_fn = torch.compile(
            lambda x, r: self._cat_slice_fn(x, r), fullgraph=True, backend="inductor"
        )
        compiled_result = compiled_fn(x, repeats)

        # Verify results match
        self.assertTrue(torch.equal(compiled_result, expected))
        self.assertTrue(torch.equal(compiled_result, eager_result))

    def test_cat_select_unbacked_bindings(self):
        """
        Test repeat_interleave -> cat -> select pattern.

        The lowering code also accesses unbacked_bindings in select operations,
        so this tests that code path as well.
        """
        x = torch.tensor([1.0, 2.0, 3.0])
        repeats = [1, 2, 1]
        expected = torch.tensor([0.0, 1.0, 2.0, 2.0, 3.0])

        # Test eager execution
        eager_result = self._cat_slice_fn(x, repeats, use_select=True)
        self.assertTrue(torch.allclose(eager_result, expected))

        # Test compiled execution - should not raise KeyError
        compiled_fn = torch.compile(
            lambda x, r: self._cat_slice_fn(x, r, use_select=True),
            fullgraph=True,
            backend="inductor",
        )
        compiled_result = compiled_fn(x, repeats)

        # Verify results match
        self.assertTrue(torch.allclose(compiled_result, expected))
        self.assertTrue(torch.allclose(compiled_result, eager_result))

    def test_multiple_patterns(self):
        """
        Test multiple input patterns to ensure robustness.

        Verifies the fix works for different dtypes and repeat patterns.
        """
        patterns = [
            {
                "x": torch.tensor([10, 20, 30], dtype=torch.int32),
                "repeats": [2, 3, 1],
                "expected": torch.tensor([10, 10, 20, 20, 20, 30], dtype=torch.int32),
                "use_select": False,
            },
            {
                "x": torch.tensor([5.0, 10.0]),
                "repeats": [2, 3],
                "expected": torch.tensor([5.0, 5.0, 10.0, 10.0, 10.0]),
                "use_select": False,
            },
        ]

        for i, pattern in enumerate(patterns):
            with self.subTest(pattern=i):
                eager_result = self._cat_slice_fn(
                    pattern["x"], pattern["repeats"], pattern["use_select"]
                )
                self.assertTrue(
                    torch.allclose(eager_result.float(), pattern["expected"].float())
                )

                compiled_fn = torch.compile(
                    lambda x, r: self._cat_slice_fn(x, r, pattern["use_select"]),
                    fullgraph=True,
                    backend="inductor",
                )
                compiled_result = compiled_fn(pattern["x"], pattern["repeats"])

                self.assertTrue(
                    torch.allclose(
                        compiled_result.float(), pattern["expected"].float()
                    )
                )
                self.assertTrue(
                    torch.allclose(compiled_result.float(), eager_result.float())
                )

    def test_fullgraph_compilation(self):
        """
        Verify that compilation occurs with fullgraph=True.

        This ensures no graph breaks occur due to the unbacked symbols.
        """
        x = torch.tensor([10, 20, 30], dtype=torch.int32)
        repeats = [2, 3, 1]

        counter = CompileCounter()
        compiled_fn = torch.compile(
            lambda x, r: self._cat_slice_fn(x, r), fullgraph=True, backend=counter
        )
        result = compiled_fn(x, repeats)

        # Should compile exactly once (fullgraph)
        self.assertEqual(counter.frame_count, 1)
        # Should have operations (exact count varies by decomposition)
        self.assertGreater(counter.op_count, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
