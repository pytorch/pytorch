"""
Test for torch.cuda.synchronize() triggering graph breaks during compilation.

With graph breaks, synchronize operations cause dynamo to break the graph at that point,
ensuring eager and compiled modes behave identically.

Expected behavior:
- Code before synchronize runs in compiled graph
- Synchronize triggers a graph break
- Synchronize and code after run in eager mode
- Assertions before synchronize are properly observed
"""

import unittest
import torch


class TestSynchronizeGraphBreak(unittest.TestCase):

    def setUp(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
        torch._dynamo.reset()

    def test_eager_mode_baseline(self):
        """Test that eager mode correctly catches the assertion."""
        def func():
            a = torch.tensor([1.0, -2.0], device="cuda")
            result = torch.all(a > 0)
            assert result, "should throw"
            torch.cuda.synchronize()
            print("should not run")

        # In eager mode, the assertion should be raised and print should not execute
        with self.assertRaises(AssertionError):
            func()

    def test_compiled_mode_matches_eager(self):
        """Test that compiled mode behaves like eager mode with graph breaks."""
        def func():
            a = torch.tensor([1.0, -2.0], device="cuda")
            result = torch.all(a > 0)
            assert result, "should throw"
            torch.cuda.synchronize()  # Graph break happens here
            print("should not run")

        # Compile with aot_eager backend
        f_c = torch.compile(func, backend="aot_eager")

        # Should raise AssertionError just like eager mode
        # Graph breaks ensure synchronize and assertions work correctly
        with self.assertRaises(AssertionError):
            f_c()

    def test_synchronize_with_device_arg(self):
        """Test that synchronize with device argument also works correctly."""
        def func(x):
            y = x + 1
            torch.cuda.synchronize(device='cuda:0')  # Graph break
            return y + 1

        compiled_func = torch.compile(func, backend="eager")
        x = torch.tensor([1.0], device="cuda")
        result = compiled_func(x)

        expected = torch.tensor([3.0], device="cuda")
        self.assertTrue(torch.allclose(result, expected))

    def test_synchronize_preserves_assertions(self):
        """Test that assertions before synchronize are properly observed."""
        import io
        from contextlib import redirect_stdout

        def func():
            a = torch.tensor([1.0, -2.0], device="cuda")
            result = torch.all(a > 0)
            assert result, "should throw"
            torch.cuda.synchronize()
            print("should not run")

        torch._dynamo.reset()
        f_c = torch.compile(func, backend="aot_eager")

        # Capture output to verify print doesn't execute
        captured_output = io.StringIO()
        exception_raised = False

        try:
            with redirect_stdout(captured_output):
                f_c()
        except AssertionError:
            exception_raised = True

        output = captured_output.getvalue()

        # With graph breaks:
        # 1. AssertionError is raised (assertion properly observed)
        # 2. Print doesn't execute (code after assertion doesn't run)
        self.assertTrue(exception_raised, "AssertionError should be raised in compiled mode")
        self.assertNotIn("should not run", output, "Print should not execute if assertion fails")


if __name__ == "__main__":
    unittest.main()
