#!/usr/bin/env python3

import gc
import unittest
import weakref
from unittest.mock import Mock

from torch.testing._internal.common_utils import TestCase
from torch.utils._functools import cache_method


class TestCacheMethod(TestCase):
    """Test coverage for torch.utils._functools.cache_method decorator"""

    def test_basic_caching(self):
        """Test that cache_method properly caches method results"""
        call_count = 0

        class TestClass:
            @cache_method
            def cached_method(self, x, y):
                nonlocal call_count
                call_count += 1
                return x + y

        obj = TestClass()

        # First call should execute the method
        result1 = obj.cached_method(1, 2)
        self.assertEqual(result1, 3)
        self.assertEqual(call_count, 1)

        # Second call with same args should return cached result
        result2 = obj.cached_method(1, 2)
        self.assertEqual(result2, 3)
        self.assertEqual(call_count, 1)  # Should not increment

        # Call with different args should execute again
        result3 = obj.cached_method(2, 3)
        self.assertEqual(result3, 5)
        self.assertEqual(call_count, 2)

        # Original args should still be cached
        result4 = obj.cached_method(1, 2)
        self.assertEqual(result4, 3)
        self.assertEqual(call_count, 2)  # Should not increment

    def test_per_instance_caching(self):
        """Test that different instances have separate caches"""
        call_count = 0

        class TestClass:
            @cache_method
            def cached_method(self, x):
                nonlocal call_count
                call_count += 1
                return x * 2

        obj1 = TestClass()
        obj2 = TestClass()

        # Call on first instance
        result1 = obj1.cached_method(5)
        self.assertEqual(result1, 10)
        self.assertEqual(call_count, 1)

        # Call on second instance with same args should execute (separate cache)
        result2 = obj2.cached_method(5)
        self.assertEqual(result2, 10)
        self.assertEqual(call_count, 2)

        # Verify both instances maintain separate caches
        result3 = obj1.cached_method(5)  # Should be cached
        result4 = obj2.cached_method(5)  # Should be cached
        self.assertEqual(result3, 10)
        self.assertEqual(result4, 10)
        self.assertEqual(call_count, 2)  # No additional calls

    def test_cache_storage_attribute(self):
        """Test that cache is stored in dynamically created attribute"""

        class TestClass:
            @cache_method
            def my_method(self, x):
                return x**2

        obj = TestClass()

        # Before calling, cache attribute shouldn't exist
        self.assertFalse(hasattr(obj, "_cache_method_my_method"))

        # After calling, cache attribute should exist
        result = obj.my_method(3)
        self.assertEqual(result, 9)
        self.assertTrue(hasattr(obj, "_cache_method_my_method"))

        # Cache should contain the result
        cache = getattr(obj, "_cache_method_my_method")
        self.assertIsInstance(cache, dict)
        self.assertEqual(cache[(3,)], 9)

    def test_multiple_methods_separate_caches(self):
        """Test that different methods on same instance have separate caches"""

        class TestClass:
            @cache_method
            def method_a(self, x):
                return x + 1

            @cache_method
            def method_b(self, x):
                return x * 2

        obj = TestClass()

        # Call both methods
        result_a = obj.method_a(5)
        result_b = obj.method_b(5)

        self.assertEqual(result_a, 6)
        self.assertEqual(result_b, 10)

        # Should have separate cache attributes
        self.assertTrue(hasattr(obj, "_cache_method_method_a"))
        self.assertTrue(hasattr(obj, "_cache_method_method_b"))

        cache_a = getattr(obj, "_cache_method_method_a")
        cache_b = getattr(obj, "_cache_method_method_b")

        # Caches should be separate
        self.assertIsNot(cache_a, cache_b)
        self.assertEqual(cache_a[(5,)], 6)
        self.assertEqual(cache_b[(5,)], 10)

    def test_kwargs_assertion(self):
        """Test that cache_method asserts when kwargs are used"""

        class TestClass:
            @cache_method
            def cached_method(self, x, y=None):
                return x + (y or 0)

        obj = TestClass()

        # Positional args should work
        result = obj.cached_method(1, 2)
        self.assertEqual(result, 3)

        # Keyword args should raise AssertionError
        with self.assertRaises(AssertionError):
            obj.cached_method(1, y=2)

        # Mixed args should also raise AssertionError
        with self.assertRaises(AssertionError):
            obj.cached_method(1, 2, y=3)

    def test_empty_arguments(self):
        """Test caching with no arguments (besides self)"""
        call_count = 0

        class TestClass:
            @cache_method
            def no_args_method(self):
                nonlocal call_count
                call_count += 1
                return "constant_result"

        obj = TestClass()

        # First call
        result1 = obj.no_args_method()
        self.assertEqual(result1, "constant_result")
        self.assertEqual(call_count, 1)

        # Second call should be cached
        result2 = obj.no_args_method()
        self.assertEqual(result2, "constant_result")
        self.assertEqual(call_count, 1)

        # Cache should store empty tuple as key
        cache = getattr(obj, "_cache_method_no_args_method")
        self.assertEqual(cache[()], "constant_result")

    def test_exception_handling(self):
        """Test that exceptions are not cached"""
        call_count = 0

        class TestClass:
            @cache_method
            def failing_method(self, should_fail):
                nonlocal call_count
                call_count += 1
                if should_fail:
                    raise ValueError("Test exception")
                return "success"

        obj = TestClass()

        # First call that fails
        with self.assertRaises(ValueError):
            obj.failing_method(True)
        self.assertEqual(call_count, 1)

        # Second call with same args should execute again (exception not cached)
        with self.assertRaises(ValueError):
            obj.failing_method(True)
        self.assertEqual(call_count, 2)

        # Successful call should be cached
        result = obj.failing_method(False)
        self.assertEqual(result, "success")
        self.assertEqual(call_count, 3)

        # Successful call should be cached on repeat
        result2 = obj.failing_method(False)
        self.assertEqual(result2, "success")
        self.assertEqual(call_count, 3)  # No additional call

    def test_hashable_arguments_only(self):
        """Test that cache_method only works with hashable arguments"""

        class TestClass:
            @cache_method
            def method_with_list(self, lst):
                return sum(lst)

            @cache_method
            def method_with_hashable(self, tup, num, string):
                return sum(tup) + num + len(string)

        obj = TestClass()

        # Unhashable arguments (list) should raise TypeError
        with self.assertRaises(TypeError):
            obj.method_with_list([1, 2, 3])

        # Hashable arguments should work fine
        call_count = 0

        class TestClassHashable:
            @cache_method
            def complex_method(self, tup, num, string):
                nonlocal call_count
                call_count += 1
                return sum(tup) + num + len(string)

        obj2 = TestClassHashable()

        # Test with tuple, number, and string (all hashable)
        result1 = obj2.complex_method((1, 2), 3, "hello")
        self.assertEqual(result1, 11)  # 1+2+3+5
        self.assertEqual(call_count, 1)

        # Same args should be cached
        result2 = obj2.complex_method((1, 2), 3, "hello")
        self.assertEqual(result2, 11)
        self.assertEqual(call_count, 1)

        # Different args should execute again
        result3 = obj2.complex_method((1, 2), 4, "hello")
        self.assertEqual(result3, 12)  # 1+2+4+5
        self.assertEqual(call_count, 2)

    def test_metadata_preservation(self):
        """Test that @functools.wraps preserves function metadata"""

        class TestClass:
            @cache_method
            def documented_method(self, x):
                """This is a test method with documentation."""
                return x * 2

        obj = TestClass()

        # Function metadata should be preserved
        self.assertEqual(obj.documented_method.__name__, "documented_method")
        self.assertEqual(
            obj.documented_method.__doc__, "This is a test method with documentation."
        )

        # Method should still work
        result = obj.documented_method(5)
        self.assertEqual(result, 10)

    def test_memory_management(self):
        """Test that cache_method doesn't keep self alive inappropriately"""
        call_count = 0
        weak_refs = []

        class TestClass:
            def __init__(self, value):
                self.value = value

            @cache_method
            def cached_method(self, x):
                nonlocal call_count
                call_count += 1
                return self.value + x

        # Create object and get weak reference
        obj = TestClass(10)
        weak_refs.append(weakref.ref(obj))

        # Use the cached method
        result = obj.cached_method(5)
        self.assertEqual(result, 15)
        self.assertEqual(call_count, 1)

        # Delete object reference
        del obj
        gc.collect()

        # Object should be collectible (weak reference should be None)
        # Note: This test might be flaky depending on GC behavior, but demonstrates intent
        self.assertIsNone(weak_refs[0]())

    def test_inheritance(self):
        """Test cache_method works correctly with inheritance"""
        call_count_base = 0
        call_count_derived = 0

        class BaseClass:
            @cache_method
            def base_method(self, x):
                nonlocal call_count_base
                call_count_base += 1
                return x * 2

        class DerivedClass(BaseClass):
            @cache_method
            def derived_method(self, x):
                nonlocal call_count_derived
                call_count_derived += 1
                return x * 3

        obj = DerivedClass()

        # Test base method
        result1 = obj.base_method(5)
        self.assertEqual(result1, 10)
        self.assertEqual(call_count_base, 1)

        # Test derived method
        result2 = obj.derived_method(5)
        self.assertEqual(result2, 15)
        self.assertEqual(call_count_derived, 1)

        # Both should be cached independently
        result3 = obj.base_method(5)
        result4 = obj.derived_method(5)
        self.assertEqual(result3, 10)
        self.assertEqual(result4, 15)
        self.assertEqual(call_count_base, 1)
        self.assertEqual(call_count_derived, 1)

    def test_large_cache(self):
        """Test cache behavior with many different arguments"""
        call_count = 0

        class TestClass:
            @cache_method
            def cached_method(self, x):
                nonlocal call_count
                call_count += 1
                return x**2

        obj = TestClass()

        # Call with many different arguments
        for i in range(100):
            result = obj.cached_method(i)
            self.assertEqual(result, i**2)

        self.assertEqual(call_count, 100)

        # All should be cached now
        for i in range(100):
            result = obj.cached_method(i)
            self.assertEqual(result, i**2)

        # Call count shouldn't increase
        self.assertEqual(call_count, 100)

        # Cache should contain all 100 entries
        cache = getattr(obj, "_cache_method_cached_method")
        self.assertEqual(len(cache), 100)

    def test_return_none(self):
        """Test that None return values are properly cached"""
        call_count = 0

        class TestClass:
            @cache_method
            def returns_none(self, x):
                nonlocal call_count
                call_count += 1
                return None if x > 0 else "not none"

        obj = TestClass()

        # First call returning None
        result1 = obj.returns_none(1)
        self.assertIsNone(result1)
        self.assertEqual(call_count, 1)

        # Second call with same args should return cached None
        result2 = obj.returns_none(1)
        self.assertIsNone(result2)
        self.assertEqual(call_count, 1)  # Should not increment

        # Call with different args
        result3 = obj.returns_none(0)
        self.assertEqual(result3, "not none")
        self.assertEqual(call_count, 2)

    def test_cache_sentinel(self):
        """Test that the cache sentinel object works correctly"""
        from torch.utils._functools import _cache_sentinel

        class TestClass:
            @cache_method
            def method_returning_sentinel(self, return_sentinel):
                # Edge case: what if method returns the sentinel object?
                if return_sentinel:
                    return _cache_sentinel
                return "normal_value"

        obj = TestClass()

        # This is an edge case that would break the cache logic
        # The current implementation would incorrectly treat the sentinel as "not cached"
        result1 = obj.method_returning_sentinel(True)
        self.assertIs(result1, _cache_sentinel)

        # Due to implementation details, this might call the method again
        # This documents current behavior - ideally this would be cached too
        result2 = obj.method_returning_sentinel(True)
        self.assertIs(result2, _cache_sentinel)


if __name__ == "__main__":
    unittest.main()
