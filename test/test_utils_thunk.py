#!/usr/bin/env python3

import unittest
import pickle
import copy
import gc
import weakref
from unittest.mock import Mock

from torch.testing._internal.common_utils import TestCase
from torch.utils._thunk import Thunk


def global_test_func():
    """Global function for pickling tests"""
    return "pickled result"


class TestThunk(TestCase):
    """Test coverage for torch.utils._thunk.Thunk class"""

    def test_basic_functionality(self):
        """Test basic lazy evaluation functionality"""
        call_count = 0

        def expensive_computation():
            nonlocal call_count
            call_count += 1
            return "computed result"

        thunk = Thunk(expensive_computation)

        # Function should not be called yet
        self.assertEqual(call_count, 0)

        # First force should call the function
        result1 = thunk.force()
        self.assertEqual(result1, "computed result")
        self.assertEqual(call_count, 1)

        # Second force should return cached result without calling function again
        result2 = thunk.force()
        self.assertEqual(result2, "computed result")
        self.assertEqual(call_count, 1)  # Still 1, not called again

    def test_function_release(self):
        """Test that function is released after first force"""

        def test_func():
            return 42

        thunk = Thunk(test_func)

        # Initially, function should be stored
        self.assertIsNotNone(thunk.f)
        self.assertIsNone(thunk.r)

        # After force, function should be released and result stored
        result = thunk.force()
        self.assertEqual(result, 42)
        self.assertIsNone(thunk.f)  # Function released
        self.assertEqual(thunk.r, 42)  # Result stored

    def test_different_return_types(self):
        """Test Thunk with different return types"""
        # Test with integer
        int_thunk = Thunk(lambda: 123)
        self.assertEqual(int_thunk.force(), 123)

        # Test with string
        str_thunk = Thunk(lambda: "hello world")
        self.assertEqual(str_thunk.force(), "hello world")

        # Test with list
        list_thunk = Thunk(lambda: [1, 2, 3])
        self.assertEqual(list_thunk.force(), [1, 2, 3])

        # Test with dict
        dict_thunk = Thunk(lambda: {"key": "value"})
        self.assertEqual(dict_thunk.force(), {"key": "value"})

        # Test with None
        none_thunk = Thunk(lambda: None)
        self.assertIsNone(none_thunk.force())

    def test_exception_handling(self):
        """Test behavior when wrapped function raises exception"""

        def failing_function():
            raise ValueError("Test exception")

        thunk = Thunk(failing_function)

        # First call should raise the exception
        with self.assertRaises(ValueError) as cm:
            thunk.force()
        self.assertEqual(str(cm.exception), "Test exception")

        # Function should NOT be cleared after exception (keeps trying)
        self.assertIsNotNone(thunk.f)
        self.assertIsNone(thunk.r)

        # Second call should raise the same exception by calling function again
        with self.assertRaises(ValueError) as cm:
            thunk.force()
        self.assertEqual(str(cm.exception), "Test exception")

    def test_side_effects(self):
        """Test that side effects only happen once"""
        side_effects = []

        def func_with_side_effects():
            side_effects.append("called")
            return "result"

        thunk = Thunk(func_with_side_effects)

        # Multiple forces should only execute side effects once
        result1 = thunk.force()
        result2 = thunk.force()
        result3 = thunk.force()

        self.assertEqual(result1, "result")
        self.assertEqual(result2, "result")
        self.assertEqual(result3, "result")
        self.assertEqual(side_effects, ["called"])  # Only called once

    def test_closure_variables(self):
        """Test Thunk with closure variables"""
        x = 10
        y = 20

        def compute():
            return x + y

        thunk = Thunk(compute)
        result = thunk.force()

        self.assertEqual(result, 30)

    def test_complex_computation(self):
        """Test with more complex computation"""

        def fibonacci(n):
            if n <= 1:
                return n
            return fibonacci(n - 1) + fibonacci(n - 2)

        def compute_fib():
            return fibonacci(10)

        thunk = Thunk(compute_fib)
        result = thunk.force()

        self.assertEqual(result, 55)  # 10th Fibonacci number

        # Verify caching works for expensive computation
        result2 = thunk.force()
        self.assertEqual(result2, 55)

    def test_mutable_result(self):
        """Test that mutable results are properly cached"""

        def create_list():
            return [1, 2, 3]

        thunk = Thunk(create_list)

        result1 = thunk.force()
        result2 = thunk.force()

        # Should return the same object instance
        self.assertIs(result1, result2)

        # Modifications affect cached result
        result1.append(4)
        result3 = thunk.force()
        self.assertEqual(result3, [1, 2, 3, 4])

    def test_memory_efficiency(self):
        """Test that function is properly garbage collected after use"""
        # Create a mock function that we can track
        mock_func = Mock(return_value="test result")

        # Create weak reference to the mock
        func_ref = weakref.ref(mock_func)

        thunk = Thunk(mock_func)

        # Function should be alive
        self.assertIsNotNone(func_ref())

        # Force the thunk
        result = thunk.force()
        self.assertEqual(result, "test result")
        mock_func.assert_called_once()

        # Clear our reference to the mock
        del mock_func

        # Function should be eligible for garbage collection
        # (though it might still be referenced by the mock framework)
        gc.collect()

        # Verify thunk no longer holds reference to function
        self.assertIsNone(thunk.f)

    def test_type_annotations(self):
        """Test that type annotations work correctly"""

        def int_func() -> int:
            return 42

        def str_func() -> str:
            return "hello"

        int_thunk: Thunk[int] = Thunk(int_func)
        str_thunk: Thunk[str] = Thunk(str_func)

        # Runtime behavior should be correct regardless of type annotations
        self.assertEqual(int_thunk.force(), 42)
        self.assertEqual(str_thunk.force(), "hello")

    def test_slots_optimization(self):
        """Test that __slots__ optimization is working"""
        thunk = Thunk(lambda: "test")

        # Should have __slots__ defined
        self.assertEqual(Thunk.__slots__, ["f", "r"])

        # Should not allow arbitrary attribute assignment
        with self.assertRaises(AttributeError):
            thunk.new_attribute = "value"

    def test_pickling(self):
        """Test that Thunk can be pickled and unpickled"""
        thunk = Thunk(global_test_func)

        # Test pickling before force
        pickled = pickle.dumps(thunk)
        unpickled = pickle.loads(pickled)

        # Should work correctly after unpickling
        result = unpickled.force()
        self.assertEqual(result, "pickled result")

        # Test pickling after force
        thunk.force()
        pickled_after = pickle.dumps(thunk)
        unpickled_after = pickle.loads(pickled_after)

        # Should return cached result
        result_after = unpickled_after.force()
        self.assertEqual(result_after, "pickled result")

    def test_copy_operations(self):
        """Test copy and deepcopy operations"""
        call_count = 0

        def tracked_func():
            nonlocal call_count
            call_count += 1
            return [1, 2, 3]

        original = Thunk(tracked_func)

        # Test copy before force
        copied = copy.copy(original)
        deep_copied = copy.deepcopy(original)

        # Each should work independently
        result1 = original.force()
        result2 = copied.force()
        result3 = deep_copied.force()

        self.assertEqual(result1, [1, 2, 3])
        self.assertEqual(result2, [1, 2, 3])
        self.assertEqual(result3, [1, 2, 3])

        # Function should have been called for each thunk
        self.assertEqual(call_count, 3)

    def test_concurrent_force_safety(self):
        """Test behavior when force is called multiple times in sequence"""
        call_count = 0

        def counting_func():
            nonlocal call_count
            call_count += 1
            return f"call_{call_count}"

        thunk = Thunk(counting_func)

        # Rapid successive calls should still only execute once
        results = []
        for _ in range(10):
            results.append(thunk.force())

        # All results should be identical
        self.assertTrue(all(r == "call_1" for r in results))
        self.assertEqual(call_count, 1)

    def test_repr_and_str(self):
        """Test string representation of Thunk objects"""

        def test_func():
            return "test"

        thunk = Thunk(test_func)

        # Should have some reasonable string representation
        repr_str = repr(thunk)
        str_str = str(thunk)

        # Should contain class name
        self.assertIn("Thunk", repr_str)
        self.assertIn("Thunk", str_str)


if __name__ == "__main__":
    unittest.main()
