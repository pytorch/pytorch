# Owner(s): ["module: onnx"]
"""Unit tests for the internal registration wrapper module."""

from typing import Sequence

from torch.onnx._internal import registration
from torch.testing._internal import common_utils


@common_utils.instantiate_parametrized_tests
class TestGlobalHelpers(common_utils.TestCase):
    @common_utils.parametrize(
        "available_opsets, target, expected",
        [
            ((7, 8, 9, 10, 11), 16, 11),
            ((7, 8, 9, 10, 11), 11, 11),
            ((7, 8, 9, 10, 11), 10, 10),
            ((7, 8, 9, 10, 11), 9, 9),
            ((7, 8, 9, 10, 11), 8, 8),
            ((7, 8, 9, 10, 11), 7, 7),
            ((9, 10, 16), 16, 16),
            ((9, 10, 16), 15, 10),
            ((9, 10, 16), 10, 10),
            ((9, 10, 16), 9, 9),
            ((9, 10, 16), 8, 9),
            ((9, 10, 16), 7, 9),
            ((7, 9, 10, 16), 16, 16),
            ((7, 9, 10, 16), 10, 10),
            ((7, 9, 10, 16), 9, 9),
            ((7, 9, 10, 16), 8, 9),
            ((7, 9, 10, 16), 7, 7),
            ([17], 16, None),  # New op added in 17
            ([9], 9, 9),
            ([9], 8, 9),
            ([], 16, None),
            ([], 9, None),
            ([], 8, None),
            ([8], 16, None),  # Ops lower than 9 are not supported by versions >= 9
        ],
    )
    def test_dispatch_opset_version_returns_correct_version(
        self, available_opsets: Sequence[int], target: int, expected: int
    ):
        actual = registration._dispatch_opset_version(target, available_opsets)
        self.assertEqual(actual, expected)


class TestOverrideDict(common_utils.TestCase):
    def setUp(self):
        self.override_dict: registration.OverrideDict[
            str, int
        ] = registration.OverrideDict()

    def test_get_item_returns_base_value_when_no_override(self):
        self.override_dict["a"] = 42
        self.override_dict["b"] = 0

        self.assertEqual(self.override_dict["a"], 42)
        self.assertEqual(self.override_dict["b"], 0)
        self.assertEqual(len(self.override_dict), 2)

    def test_get_item_returns_overridden_value_when_override(self):
        self.override_dict["a"] = 42
        self.override_dict["b"] = 0
        self.override_dict.override("a", 100)
        self.override_dict.override("c", 1)

        self.assertEqual(self.override_dict["a"], 100)
        self.assertEqual(self.override_dict["b"], 0)
        self.assertEqual(self.override_dict["c"], 1)
        self.assertEqual(len(self.override_dict), 3)

    def test_get_item_raises_key_error_when_not_found(self):
        self.override_dict["a"] = 42

        with self.assertRaises(KeyError):
            self.override_dict["nonexistent_key"]

    def test_get_returns_overridden_value_when_override(self):
        self.override_dict["a"] = 42
        self.override_dict["b"] = 0
        self.override_dict.override("a", 100)
        self.override_dict.override("c", 1)

        self.assertEqual(self.override_dict.get("a"), 100)
        self.assertEqual(self.override_dict.get("b"), 0)
        self.assertEqual(self.override_dict.get("c"), 1)
        self.assertEqual(len(self.override_dict), 3)

    def test_get_returns_none_when_not_found(self):
        self.override_dict["a"] = 42

        self.assertEqual(self.override_dict.get("nonexistent_key"), None)

    def test_in_base_returns_true_for_base_value(self):
        self.override_dict["a"] = 42
        self.override_dict["b"] = 0
        self.override_dict.override("a", 100)
        self.override_dict.override("c", 1)

        self.assertIn("a", self.override_dict)
        self.assertIn("b", self.override_dict)
        self.assertIn("c", self.override_dict)

        self.assertTrue(self.override_dict.in_base("a"))
        self.assertTrue(self.override_dict.in_base("b"))
        self.assertFalse(self.override_dict.in_base("c"))
        self.assertFalse(self.override_dict.in_base("nonexistent_key"))

    def test_overridden_returns_true_for_overridden_value(self):
        self.override_dict["a"] = 42
        self.override_dict["b"] = 0
        self.override_dict.override("a", 100)
        self.override_dict.override("c", 1)

        self.assertTrue(self.override_dict.overridden("a"))
        self.assertFalse(self.override_dict.overridden("b"))
        self.assertTrue(self.override_dict.overridden("c"))
        self.assertFalse(self.override_dict.overridden("nonexistent_key"))

    def test_overrides_returns_all_keys_overridden(self):
        self.override_dict["a"] = 42
        self.override_dict["b"] = 0
        self.override_dict.override("a", 100)
        self.override_dict.override("c", 1)

        self.assertEqual(set(self.override_dict.overrides()), {"a", "c"})

    def test_remove_override_removes_overridden_value(self):
        self.override_dict["a"] = 42
        self.override_dict["b"] = 0
        self.override_dict.override("a", 100)
        self.override_dict.override("c", 1)

        self.assertEqual(self.override_dict["a"], 100)
        self.assertEqual(self.override_dict["c"], 1)

        self.override_dict.remove_override("a")
        self.override_dict.remove_override("c")
        self.assertEqual(self.override_dict["a"], 42)
        self.assertEqual(self.override_dict.get("c"), None)
        self.assertEqual(set(self.override_dict.overrides()), set())
        self.assertFalse(self.override_dict.overridden("a"))
        self.assertFalse(self.override_dict.overridden("c"))

    def test_remove_override_removes_overridden_key(self):
        self.override_dict.override("a", 100)
        self.assertEqual(self.override_dict["a"], 100)
        self.assertEqual(len(self.override_dict), 1)
        self.override_dict.remove_override("a")
        self.assertEqual(len(self.override_dict), 0)
        self.assertNotIn("a", self.override_dict)

    def test_overriden_key_precededs_base_key_regardless_of_insert_order(self):
        self.override_dict["a"] = 42
        self.override_dict.override("a", 100)
        self.override_dict["a"] = 0

        self.assertEqual(self.override_dict["a"], 100)
        self.assertEqual(len(self.override_dict), 1)

    def test_bool_is_true_when_not_empty(self):
        if self.override_dict:
            self.fail("OverrideDict should be false when empty")
        self.override_dict.override("a", 1)
        if not self.override_dict:
            self.fail("OverrideDict should be true when not empty")
        self.override_dict["a"] = 42
        if not self.override_dict:
            self.fail("OverrideDict should be true when not empty")
        self.override_dict.remove_override("a")
        if not self.override_dict:
            self.fail("OverrideDict should be true when not empty")


if __name__ == "__main__":
    common_utils.run_tests()
