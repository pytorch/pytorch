import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(REPO_ROOT))

from tools.linter.adapters.stable_shim_usage_linter import (
    check_file,
    get_shim_functions,
    write_shim_function_versions,
)


class TestStableShimUsageLinter(unittest.TestCase):
    """Test stable shim usage linter functionality."""

    def test_get_shim_functions(self):
        """
        Test parsing a comprehensive sample shim.h that covers all edge cases:
        - Simple versioned functions
        - Multiple functions with different versions
        - Typedef function pointers
        - Unversioned functions (should be ignored)
        - Nested version blocks with platform ifdefs
        - Functions in #else branches (should NOT be versioned)
        - Commented out functions (should be ignored)
        - Complex nested conditionals
        - Functions after #elif (should be versioned based on elif condition)
        """
        test_dir = Path(__file__).parent / "stable_shim_usage_linter_data"
        sample_shim = test_dir / "sample_shim.h"

        self.assertTrue(
            sample_shim.exists(),
            f"Sample shim file not found at {sample_shim}",
        )

        result = get_shim_functions([sample_shim])

        expected = {
            # Simple versioned function (2.10)
            "simple_versioned_func": (2, 10),
            # Multiple functions with version 2.9
            "old_function_1": (2, 9),
            "old_function_2": (2, 9),
            # Typedef function pointer (2.10)
            "callback_function_ptr": (2, 10),
            # Nested version blocks (2.11)
            "platform_specific_func": (2, 11),
            "always_available_func": (2, 11),
            # Function in #if branch (2.10), NOT legacy_fallback which is in #else
            "modern_implementation": (2, 10),
            # Actual function (2.10), NOT commented_out_func
            "actual_function": (2, 10),
            # Complex nested (2.12)
            "deeply_nested_func": (2, 12),
            "outer_block_func": (2, 12),
            # Multiple typedefs
            "legacy_callback": (2, 9),
            "modern_callback": (2, 10),
            # Using declarations and struct/class (2.10)
            "OpaqueHandle": (2, 10),
            "HandleType": (2, 10),
            # Using declarations and struct/class (2.11)
            "NewOpaqueStruct": (2, 11),
            "NewOpaqueClass": (2, 11),
            "NewHandleType": (2, 11),
            # Primary path (2.10) and secondary path (2.9) from #if/#elif
            "primary_path": (2, 10),
            "secondary_path": (2, 9),
        }

        self.assertEqual(result, expected)

    def test_write_shim_function_versions(self):
        """Test that write_shim_function_versions creates the expected output file."""
        functions = {
            "func_a": (2, 9),
            "func_b": (2, 10),
            "func_c": (2, 9),
            "func_d": (2, 11),
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt") as tmp:
            tmp_path = tmp.name

            write_shim_function_versions(functions, tmp_path)

            with open(tmp_path) as f:
                content = f.read()

            self.assertIn("Auto-generated file", content)
            self.assertIn("DO NOT EDIT MANUALLY", content)

            lines = [
                line
                for line in content.split("\n")
                if line and not line.startswith("#")
            ]
            expected_lines = [
                "func_a: TORCH_VERSION_2_9_0",
                "func_c: TORCH_VERSION_2_9_0",
                "func_b: TORCH_VERSION_2_10_0",
                "func_d: TORCH_VERSION_2_11_0",
            ]

            self.assertEqual(lines, expected_lines)

    def test_check_file(self):
        """
        Test checking a file for proper usage of versioned shim functions.
        This tests various scenarios:
        - Correct usage with proper version guards
        - Unversioned calls (no guard)
        - Insufficient version guards
        - Higher version guards (acceptable)
        - Nested blocks
        - #else branches (no protection)
        - #elif branches with version guards
        """
        test_dir = Path(__file__).parent / "stable_shim_usage_linter_data"
        sample_shim = test_dir / "sample_shim.h"
        sample_usage = test_dir / "sample_usage.h"
        self.assertTrue(sample_shim.exists(), f"Sample shim not found at {sample_shim}")
        self.assertTrue(
            sample_usage.exists(), f"Sample usage not found at {sample_usage}"
        )

        shim_functions = get_shim_functions([sample_shim])
        lint_messages = check_file(str(sample_usage), shim_functions)

        # Expected errors based on sample_usage.h:
        # Line 15: unversioned call to simple_versioned_func
        # Line 21: insufficient version (2.9) for simple_versioned_func (needs 2.10)
        # Line 38: insufficient version (2.9) for simple_versioned_func (needs 2.10)
        # Line 39: insufficient version (2.9) for callback_function_ptr (needs 2.10)
        # Line 63: unversioned call in #else branch to simple_versioned_func
        # Line 83: insufficient version (2.10) for always_available_func (needs 2.11)
        # Line 89: unversioned call to old_function_1
        # Line 90: unversioned call to old_function_2
        # Line 103: insufficient version (2.9) for HandleType (needs 2.10)
        # Line 109: unversioned call to OpaqueHandle (needs 2.10)
        # Line 110: unversioned call to NewOpaqueStruct (needs 2.11)
        # Line 125: insufficient version (2.10) for NewOpaqueStruct (needs 2.11)
        # Line 126: insufficient version (2.10) for NewOpaqueClass (needs 2.11)

        expected_errors = [
            (15, "unversioned-shim-call", "simple_versioned_func"),
            (21, "insufficient-version-for-shim-call", "simple_versioned_func"),
            (38, "insufficient-version-for-shim-call", "simple_versioned_func"),
            (39, "insufficient-version-for-shim-call", "callback_function_ptr"),
            (63, "unversioned-shim-call", "simple_versioned_func"),
            (83, "insufficient-version-for-shim-call", "always_available_func"),
            (89, "unversioned-shim-call", "old_function_1"),
            (90, "unversioned-shim-call", "old_function_2"),
            (103, "insufficient-version-for-shim-call", "HandleType"),
            (109, "unversioned-shim-call", "OpaqueHandle"),
            (110, "unversioned-shim-call", "NewOpaqueStruct"),
            (125, "insufficient-version-for-shim-call", "NewOpaqueStruct"),
            (126, "insufficient-version-for-shim-call", "NewOpaqueClass"),
        ]

        self.assertEqual(
            len(lint_messages),
            len(expected_errors),
            f"Expected {len(expected_errors)} errors, got {len(lint_messages)}. "
            f"Errors: {[(msg.line, msg.name) for msg in lint_messages]}",
        )

        errors_by_line = {msg.line: msg for msg in lint_messages}
        for line, error_name, func_name in expected_errors:
            self.assertIn(
                line,
                errors_by_line,
                f"Expected error on line {line} for {func_name}, but not found",
            )
            msg = errors_by_line[line]
            self.assertEqual(msg.name, error_name)
            self.assertIsNotNone(msg.description)
            self.assertTrue(func_name in msg.description)


if __name__ == "__main__":
    unittest.main()
