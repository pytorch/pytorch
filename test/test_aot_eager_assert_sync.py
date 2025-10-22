"""
Test for AOT eager bug where torch.cuda.synchronize() is dropped during graph optimization,
causing _assert_async failures to not be observed like in eager mode.

Bug description:
- In eager mode: assertion is caught and print is not executed
- In aot_eager mode: synchronize() is dropped; print executes; later CUDA kernel assert surfaces

Expected behavior: aot_eager should match eager mode behavior.
"""

import unittest
import torch
import sys
import os


class TestAOTEagerAssertSync(unittest.TestCase):
    
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
    
    def test_aot_eager_should_match_eager(self):
        """Test that aot_eager mode should behave like eager mode (currently fails)."""
        def func():
            a = torch.tensor([1.0, -2.0], device="cuda")
            result = torch.all(a > 0)
            assert result, "should throw"  
            torch.cuda.synchronize()
            print("should not run")
        
        # Compile with aot_eager backend
        f_c = torch.compile(func, backend="aot_eager")
        
        # This should raise AssertionError like eager mode, but currently doesn't
        # due to synchronize() being dropped from the graph
        with self.assertRaises(AssertionError):
            f_c()
    
    def test_minimal_repro_with_logs(self):
        """Minimal repro case that can be used to examine graph differences."""
        def func():
            a = torch.tensor([1.0, -2.0], device="cuda")
            result = torch.all(a > 0)
            assert result, "should throw"
            torch.cuda.synchronize()
            print("should not run")

        # Reset dynamo state
        torch._dynamo.reset()
        
        # Compile with aot_eager
        f_c = torch.compile(func, backend="aot_eager")
        
        # Capture output to check if print executes (it shouldn't)
        import io
        from contextlib import redirect_stdout
        
        captured_output = io.StringIO()
        exception_raised = False
        
        try:
            with redirect_stdout(captured_output):
                f_c()
        except Exception as e:
            exception_raised = True
            
        output = captured_output.getvalue()
        
        # The test currently fails because:
        # 1. No AssertionError is raised (synchronize() dropped, assertion not observed)
        # 2. "should not run" is printed (should not happen)
        
        print(f"Output captured: '{output}'")
        print(f"Exception raised: {exception_raised}")
        
        # After the fix, this should pass: 
        # - AssertionError should be raised (sync preserved, assertion observed)
        # - Print should not execute ("should not run" not in output)
        self.assertTrue(exception_raised, "AssertionError should be raised in aot_eager mode")
        self.assertNotIn("should not run", output, "Print should not execute if assertion fails")
        
    def test_fix_validation(self):
        """Test that the fix properly preserves synchronize operations."""
        # Import the effects module to check our fix
        from torch._higher_order_ops.effects import get_effect_key, _EffectType
        
        # Test that synchronize operations are detected as side-effectful
        import torch.cuda
        
        # Check if torch.cuda.synchronize is properly detected
        effect = get_effect_key(torch.cuda.synchronize, (), {})
        self.assertEqual(effect, _EffectType.ORDERED, 
                        "torch.cuda.synchronize should be detected as having ordered side effects")
        
        print("✓ Fix validation: torch.cuda.synchronize is properly registered as side-effectful")


def run_with_graph_logging():
    """Helper function to run the test with TORCH_LOGS enabled to show graph differences."""
    print("=== Running with TORCH_LOGS to show graph differences ===")
    
    def func():
        a = torch.tensor([1.0, -2.0], device="cuda")
        result = torch.all(a > 0)
        assert result, "should throw"
        torch.cuda.synchronize()
        print("should not run")

    torch._dynamo.reset()
    
    # Enable detailed logging
    os.environ["TORCH_LOGS"] = "graph_code,aot_graphs"
    
    try:
        f_c = torch.compile(func, backend="aot_eager")
        f_c()
    except Exception as e:
        print(f"Exception: {e}")
    finally:
        # Clean up environment
        if "TORCH_LOGS" in os.environ:
            del os.environ["TORCH_LOGS"]


if __name__ == "__main__":
    # First run with logging to show the issue
    run_with_graph_logging()
    
    # Then run the actual tests
    unittest.main()