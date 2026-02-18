import torch
import unittest


class TestSortStableFix(unittest.TestCase):
    """
    Test for PyTorch issue #174459: Inconsistency between inductor and eager
    backends for aten.sort(stable=None) on CUDA.
    """

    def setUp(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")

    def test_sort_stable_parameter_consistency(self):
        """Test that stable parameter works consistently between backends."""
        def model_func(self, stable, dim, descending):
            return torch.ops.aten.sort(self, stable=stable, dim=dim, descending=descending)

        # Test case with all equal elements to highlight stability differences
        test_input = torch.tensor([[2, 2, 2], [1, 1, 1]], dtype=torch.int64, device='cuda')

        compiled_eager = torch.compile(model_func, backend="eager")
        compiled_inductor = torch.compile(model_func, backend="inductor")

        # Test stable=None (should default to False)
        eager_none = compiled_eager(test_input, stable=None, dim=-1, descending=True)
        inductor_none = compiled_inductor(test_input, stable=None, dim=-1, descending=True)

        # Test stable=False explicitly
        eager_false = compiled_eager(test_input, stable=False, dim=-1, descending=True)
        inductor_false = compiled_inductor(test_input, stable=False, dim=-1, descending=True)

        # Test stable=True
        eager_true = compiled_eager(test_input, stable=True, dim=-1, descending=True)
        inductor_true = compiled_inductor(test_input, stable=True, dim=-1, descending=True)

        # Values should always be the same
        torch.testing.assert_close(eager_none[0], inductor_none[0])
        torch.testing.assert_close(eager_false[0], inductor_false[0])
        torch.testing.assert_close(eager_true[0], inductor_true[0])

        # stable=None should behave same as stable=False
        torch.testing.assert_close(eager_none, eager_false)
        torch.testing.assert_close(inductor_none, inductor_false)

        # stable=True should work consistently between backends
        torch.testing.assert_close(eager_true, inductor_true)

        # Most importantly: inductor should now differentiate between stable=False and stable=True
        # (This was the main bug - inductor was treating both the same)
        with self.assertRaises(AssertionError):
            torch.testing.assert_close(inductor_false[1], inductor_true[1])


if __name__ == "__main__":
    unittest.main()