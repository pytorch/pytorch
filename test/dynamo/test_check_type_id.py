# Owner(s): ["module: dynamo"]
"""
Test for TYPE_MATCH guard and ___check_type_id function.

This test demonstrates how the TYPE_MATCH guard works in PyTorch Dynamo.
When a function is compiled, Dynamo installs guards to ensure the compiled
code remains valid. TYPE_MATCH guards ensure that values maintain their
exact type (using type identity, not just type equality).
"""

import re

import torch
import torch._dynamo
import torch._dynamo.test_case
from torch._dynamo.eval_frame import _debug_get_cache_entry_list
from torch.testing._internal.common_utils import munge_exc


class TestCheckTypeId(torch._dynamo.test_case.TestCase):
    @staticmethod
    def _find_guard_lines(guard_manager_str: str, keyword: str) -> list[str]:
        # Normalize and anonymize type IDs, then return lines containing the keyword
        normalized = re.sub(
            r"\d{7,}", "<type_id>", munge_exc(guard_manager_str), flags=re.MULTILINE
        )
        pattern = re.compile(rf"^.*{re.escape(keyword)}.*$", re.MULTILINE)
        return pattern.findall(normalized)

    def test_type_match_with_different_values(self):
        """
        Test that TYPE_MATCH guard correctly identifies type mismatches.

        This test compiles a function that uses a global variable and verifies:
        1. The compiled function works with values of the same type
        2. The function recompiles when the type changes
        3. The ___check_type_id/check_obj_id guard is present in the generated code
        4. The check_type_id should present the user-friendly code that specify the type
        """

        # Define a global variable that we'll guard on
        class Config:
            multiplier = 2  # int type

        def fn(x):
            # This will trigger a TYPE_MATCH guard on Config.multiplier
            return x * Config.multiplier

        # Compile the function
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)

        # First call - should compile and install guards
        x = torch.randn(4)
        result1 = opt_fn(x)
        expected1 = x * 2
        self.assertTrue(torch.allclose(result1, expected1))

        # Get the cache entry to inspect guards
        cache_entries = _debug_get_cache_entry_list(fn.__code__)
        self.assertEqual(len(cache_entries), 1)

        # Check that the guard string contains check_type_id
        guard_str = str(cache_entries[0].guard_manager)
        matches = self._find_guard_lines(guard_str, "ID_MATCH")
        self.assertIn("___check_obj_id", matches[0])
        self.assertIn(
            ".TestCheckTypeId.test_type_match_with_different_values.<locals>.Config'>",
            matches[0],
        )
        # Match the first part (everything before "type=")
        first_part = matches[0].split("type=")[0]
        expected_first_part = (
            "| | +- ID_MATCH: ___check_obj_id(L['Config'], <type_id>), "
        )
        self.assertEqual(first_part, expected_first_part)

        # Match the second part (the type string)
        second_part = matches[0].split("type=")[1].rstrip()
        expected_second_part = (
            "TestCheckTypeId.test_type_match_with_different_values.<locals>.Config'>"
        )
        self.assertIn(expected_second_part, second_part)

    def test_type_match_with_custom_classes(self):
        """
        Test TYPE_MATCH guard with custom class instances.

        Demonstrates that the guard checks type identity, not structural equality.
        """

        class Point:
            def __init__(self, x, y):
                self.x = x
                self.y = y

        class Point2D:
            def __init__(self, x, y):
                self.x = x
                self.y = y

        point = Point(1, 2)

        def fn(tensor):
            # Access point's attributes, triggering TYPE_MATCH guard on point
            return tensor + point.x + point.y

        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)

        # First call with Point instance
        x = torch.ones(4)
        result1 = opt_fn(x)
        expected1 = x + 1 + 2
        self.assertTrue(torch.allclose(result1, expected1))

        # Verify guard contains check_type_id
        cache_entries = _debug_get_cache_entry_list(fn.__code__)
        self.assertEqual(len(cache_entries), 1)

        guard_str = str(cache_entries[0].guard_manager)
        matches = self._find_guard_lines(guard_str, "TYPE_MATCH")
        # Match the first part (everything before "type=")
        first_part = matches[0].split("type=")[0]
        expected_first_part = (
            "| | +- TYPE_MATCH: ___check_type_id(L['point'], <type_id>), "
        )
        self.assertEqual(first_part, expected_first_part)

        # Match the second part (the type string)
        second_part = matches[0].split("type=")[1].rstrip()
        expected_second_part = (
            "TestCheckTypeId.test_type_match_with_custom_classes.<locals>.Point'>"
        )
        self.assertIn(expected_second_part, second_part)


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
