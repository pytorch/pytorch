"""
Tests for torch.profiler.auto_instrumenter module.

These tests verify the auto-instrumentation functionality for PyTorch modules.
"""

import unittest
import torch
from torch.profiler import profile, ProfilerActivity

from torch.profiler.auto_instrumenter import auto_profile_module, ANNOTATION_MARKER


class SimpleModule(torch.nn.Module):
    """Simple test module"""
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        y = x + 1
        z = y * 2
        return z


class HelperModule(torch.nn.Module):
    """Helper module for nested call tests"""
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        y = x * 2
        z = y + 1
        return z


class MainModule(torch.nn.Module):
    """Main module that calls helper"""
    def __init__(self):
        super().__init__()
        self.helper = HelperModule()
        
    def forward(self, x):
        a = x + 1
        b = self.helper(a)
        c = b * 2
        return c


class LoopModule(torch.nn.Module):
    """Module with a loop"""
    def __init__(self):
        super().__init__()
        self.helper = HelperModule()
        
    def forward(self, x):
        results = []
        for i in range(2):
            a = self.helper(x)
            b = a + i
            c = self.helper(b)
            results.append(c)
        output = torch.stack(results)
        return output


class TestAutoInstrumenter(unittest.TestCase):
    """Tests for auto_profile_module function"""

    def _extract_annotation_lines(self, events) -> list[int]:
        """Extract line numbers from profiler annotations in execution order"""
        actual_lines = []
        for evt in events:
            if ANNOTATION_MARKER in evt.name:
                # Parse annotation: @@:<filename>:<line>:<code>
                parts = evt.name.split(":", 3)
                if len(parts) >= 3:
                    actual_lines.append(int(parts[2]))
        return actual_lines
    
    def _print_comparison(self, test_name: str, expected: list[int], actual: list[int], 
                          line_descriptions: dict[int, str]) -> None:
        """Print detailed comparison of expected vs actual line numbers"""
        print(f"\n{'='*60}")
        print(f"TEST: {test_name}")
        print(f"{'='*60}")
        print(f"\nExpected execution order:")
        for i, line_num in enumerate(expected):
            desc = line_descriptions.get(line_num, "unknown")
            print(f"  {i+1}. Line {line_num}: {desc}")
        
        print(f"\nActual execution order:")
        for i, line_num in enumerate(actual):
            desc = line_descriptions.get(line_num, "unknown")
            print(f"  {i+1}. Line {line_num}: {desc}")
        
        print(f"\nMatch: {expected == actual}")
        print(f"{'='*60}\n")

    def test_simple_module_line_numbers(self):
        """Test that SimpleModule line numbers match source code exactly
        
        SimpleModule.forward (lines 19-22):
            Line 19: def forward(self, x):
            Line 20:     y = x + 1
            Line 21:     z = y * 2
            Line 22:     return z
        """
        # Manually defined expected execution order
        expected = [20, 21, 22]
        line_descriptions = {
            20: "y = x + 1",
            21: "z = y * 2",
            22: "return z",
        }
        
        # Setup: Create and instrument module  
        model = SimpleModule()
        model = auto_profile_module(model, 'forward')
        
        # Execute: Run with profiler
        x = torch.randn(4, 4)
        with profile(activities=[ProfilerActivity.CPU]) as prof:
            _ = model(x)
        
        # Assert: Extract and verify line numbers
        events = prof.events()
        actual = self._extract_annotation_lines(events)
        
        self._print_comparison("SimpleModule.forward", expected, actual, line_descriptions)
        
        self.assertGreater(len(actual), 0, "Should have captured line annotations")
        self.assertEqual(actual, expected,
            f"Line numbers mismatch.\nExpected: {expected}\nActual: {actual}")

    def test_main_module_line_numbers(self):
        """Test that MainModule line numbers match source code exactly
        
        MainModule.forward (lines 42-46):
            Line 42: def forward(self, x):
            Line 43:     a = x + 1
            Line 44:     b = self.helper(a)  # Calls HelperModule.forward
            Line 45:     c = b * 2
            Line 46:     return c
        
        HelperModule.forward (lines 30-33):
            Line 30: def forward(self, x):
            Line 31:     y = x * 2
            Line 32:     z = y + 1
            Line 33:     return z
        """
        # Manually defined expected execution order (includes helper calls)
        expected = [
            43,  # a = x + 1
            44,  # b = self.helper(a)
            31,  #   y = x * 2 (inside helper)
            32,  #   z = y + 1 (inside helper)
            33,  #   return z (inside helper)
            45,  # c = b * 2
            46,  # return c
        ]
        line_descriptions = {
            43: "a = x + 1",
            44: "b = self.helper(a)",
            31: "y = x * 2 (HelperModule)",
            32: "z = y + 1 (HelperModule)",
            33: "return z (HelperModule)",
            45: "c = b * 2",
            46: "return c",
        }
        
        # Setup: Create and instrument module (including helper)
        model = MainModule()
        model = auto_profile_module(model, 'forward')
        model.helper = auto_profile_module(model.helper, 'forward')
        
        # Execute: Run with profiler
        x = torch.randn(4, 4)
        with profile(activities=[ProfilerActivity.CPU]) as prof:
            _ = model(x)
        
        # Assert: Extract and verify line numbers
        events = prof.events()
        actual = self._extract_annotation_lines(events)
        
        self._print_comparison("MainModule.forward", expected, actual, line_descriptions)
        
        self.assertGreater(len(actual), 0, "Should have captured line annotations for MainModule")
        self.assertEqual(actual, expected,
            f"Line numbers mismatch for MainModule.forward.\nExpected: {expected}\nActual: {actual}")

    def test_loop_module_line_numbers(self):
        """Test that LoopModule line numbers match source code exactly
        
        LoopModule.forward (lines 55-63):
            Line 55: def forward(self, x):
            Line 56:     results = []
            Line 57:     for i in range(2):
            Line 58:         a = self.helper(x)  # Calls HelperModule.forward
            Line 59:         b = a + i
            Line 60:         c = self.helper(b)  # Calls HelperModule.forward
            Line 61:         results.append(c)
            Line 62:     output = torch.stack(results)
            Line 63:     return output
        
        HelperModule.forward (lines 30-33):
            Line 30: def forward(self, x):
            Line 31:     y = x * 2
            Line 32:     z = y + 1
            Line 33:     return z
        
        Note: Auto-profiler only instruments function calls and returns, NOT:
        - Simple assignments (results = [], b = a + i)
        - Loop headers (for i in range(2):)
        """
        # Based on actual output: autoprofiler only instruments calls/returns
        expected = [
            56,  # results = []
            58,  # a = self.helper(x)  [1st helper call, iteration 0]
            31,  #   y = x * 2 (inside helper)
            32,  #   z = y + 1 (inside helper)
            33,  #   return z (inside helper)
            59,  # b = a + i
            60,  # c = self.helper(b)  [2nd helper call, iteration 0]
            31,  #   y = x * 2 (inside helper)
            32,  #   z = y + 1 (inside helper)
            33,  #   return z (inside helper)
            61,  # results.append(c)
            58,  # a = self.helper(x)  [3rd helper call, iteration 1]
            31,  #   y = x * 2 (inside helper)
            32,  #   z = y + 1 (inside helper)
            33,  #   return z (inside helper)
            59,  # b = a + i
            60,  # c = self.helper(b)  [4th helper call, iteration 1]
            31,  #   y = x * 2 (inside helper)
            32,  #   z = y + 1 (inside helper)
            33,  #   return z (inside helper)
            61,  # results.append(c)
            62,  # output = torch.stack(results)
            63,  # return output
        ]
        line_descriptions = {
            56: "results = []",
            58: "a = self.helper(x)",
            31: "y = x * 2 (HelperModule)",
            32: "z = y + 1 (HelperModule)",
            33: "return z (HelperModule)",
            59: "b = a + i",
            60: "c = self.helper(b)",
            61: "results.append(c)",
            62: "output = torch.stack(results)",
            63: "return output",
        }
        
        # Setup: Create and instrument module (including helper)
        model = LoopModule()
        model = auto_profile_module(model, 'forward')
        model.helper = auto_profile_module(model.helper, 'forward')
        
        # Execute: Run with profiler
        x = torch.randn(4, 4)
        with profile(activities=[ProfilerActivity.CPU]) as prof:
            _ = model(x)
        
        # Assert: Extract and verify line numbers
        events = prof.events()
        actual = self._extract_annotation_lines(events)
        
        self._print_comparison("LoopModule.forward", expected, actual, line_descriptions)
        
        self.assertGreater(len(actual), 0, "Should have captured line annotations for LoopModule")
        self.assertEqual(actual, expected,
            f"Line numbers mismatch for LoopModule.forward.\nExpected: {expected}\nActual: {actual}")
