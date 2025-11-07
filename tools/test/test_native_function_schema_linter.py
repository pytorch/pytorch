import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tools.linter.adapters.native_function_schema_linter import (
    check_file,
    extract_native_ops_from_ops_h,
    get_current_torch_version,
    get_registered_adapters,
    is_only_default_arg_value_change,
    LintSeverity,
    read_native_ops_txt,
    write_native_ops_txt,
)


class TestNativeFunctionSchemaLinter(unittest.TestCase):
    """Test native function schema linter functionality."""

    def test_extract_native_ops_from_ops_h(self):
        """
        Test extracting ops from a sample ops.h file.
        Should extract ops with and without overloads.
        """
        test_dir = Path(__file__).parent / "native_function_schema_linter_data"
        sample_ops_h = test_dir / "sample_ops.h"

        self.assertTrue(
            sample_ops_h.exists(),
            f"Sample ops.h file not found at {sample_ops_h}",
        )

        result = extract_native_ops_from_ops_h(sample_ops_h)

        expected = {
            "aten::empty_like",
            "aten::transpose.int",
            "aten::clone",
            "aten::zero_",
            "aten::copy_",
        }

        self.assertEqual(result, expected)

    def test_write_and_read_native_ops_txt(self):
        """
        Test writing and reading native_ops.txt file.
        """
        ops = {
            "aten::empty_like",
            "aten::transpose.int",
            "aten::clone",
            "aten::zero_",
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt") as tmp:
            tmp_path = Path(tmp.name)

            write_native_ops_txt(ops, tmp_path)

            with open(tmp_path) as f:
                content = f.read()

            self.assertIn("Auto-generated file", content)
            self.assertIn("DO NOT EDIT MANUALLY", content)
            self.assertIn("stable_native_function_schema_linter", content)

            read_ops = read_native_ops_txt(tmp_path)
            self.assertEqual(read_ops, ops)

            lines = [
                line
                for line in content.split("\n")
                if line and not line.startswith("#")
            ]
            expected_lines = [
                "aten::clone",
                "aten::empty_like",
                "aten::transpose.int",
                "aten::zero_",
            ]

            self.assertEqual(lines, expected_lines)

    def test_get_registered_adapters(self):
        """
        Test extracting registered schema adapters from shim_common.cpp.
        Should return dict mapping op name to (major, minor) version tuple.
        """
        test_dir = Path(__file__).parent / "native_function_schema_linter_data"
        sample_shim_common = test_dir / "sample_shim_common.cpp"

        self.assertTrue(
            sample_shim_common.exists(),
            f"Sample shim_common.cpp file not found at {sample_shim_common}",
        )

        result = get_registered_adapters(sample_shim_common)

        expected = {
            "aten::empty_like": (2, 11),
            "aten::transpose": (2, 10),
            "aten::clone": (2, 9),
        }

        self.assertEqual(result, expected)

    def test_is_only_default_arg_value_change(self):
        """
        Test detection of default-only schema changes.
        """
        # Only default value changed
        old_schema = "empty_like(Tensor self, int arg=0) -> Tensor"
        new_schema = "empty_like(Tensor self, int arg=1) -> Tensor"
        is_default_only, changed_args = is_only_default_arg_value_change(
            old_schema, new_schema
        )
        self.assertTrue(is_default_only)
        self.assertEqual(changed_args, ["arg"])

        # New parameter added
        old_schema = "empty_like(Tensor self) -> Tensor"
        new_schema = "empty_like(Tensor self, *, int new_arg=0) -> Tensor"
        is_default_only, changed_args = is_only_default_arg_value_change(
            old_schema, new_schema
        )
        self.assertFalse(is_default_only)
        self.assertEqual(changed_args, [])

        # Type changed
        old_schema = "empty_like(Tensor self, *, int arg=0) -> Tensor"
        new_schema = "empty_like(Tensor self, *, float arg=0.0) -> Tensor"
        is_default_only, changed_args = is_only_default_arg_value_change(
            old_schema, new_schema
        )
        self.assertFalse(is_default_only)

        # Parameter name changed - should be OK
        old_schema = "empty_like(Tensor self, *, int old_name=0) -> Tensor"
        new_schema = "empty_like(Tensor self, *, int new_name=0) -> Tensor"
        is_default_only, changed_args = is_only_default_arg_value_change(
            old_schema, new_schema
        )
        self.assertTrue(is_default_only)
        self.assertEqual(changed_args, [])

    @patch("tools.linter.adapters.native_function_schema_linter.subprocess.run")
    def test_check_file_with_schema_changes(self, mock_run):
        """
        Test check_file with schema changes for tracked ops.
        Tests scenarios: structural change (ERROR), default-only change (WARNING), correct adapter, incorrect version.
        """
        current_version = get_current_torch_version()
        wrong_version = (current_version[0], current_version[1] - 1)

        def check_with_adapter(adapter_registration: str, diff_output: str):
            """Helper to check file with a given adapter registration and diff."""
            mock_run.return_value.stdout = diff_output

            with tempfile.NamedTemporaryFile(mode="w", suffix=".cpp") as shim_tmp:
                shim_path = Path(shim_tmp.name)
                shim_tmp.write(adapter_registration)
                shim_tmp.flush()

                return check_file(
                    "native_functions.yaml",
                    Path("native_functions.yaml"),
                    shim_path,
                )

        with self.subTest("structural change without adapter - ERROR"):
            # New parameter added (structural change)
            diff_output = """
--- a/aten/src/ATen/native/native_functions.yaml
+++ b/aten/src/ATen/native/native_functions.yaml
@@ -100,7 +100,7 @@
-- func: empty_like(Tensor self) -> Tensor
+- func: empty_like(Tensor self, *, int new_arg=0) -> Tensor
"""
            result = check_with_adapter(
                'register_schema_adapter("aten::clone", TORCH_VERSION_2_9_0, adapt_clone);\n',
                diff_output,
            )

            self.assertEqual(len(result), 1)
            self.assertEqual(result[0].severity, LintSeverity.ERROR)
            self.assertEqual(result[0].name, "missing-schema-adapter")
            self.assertIn("Structural schema change", result[0].description)
            self.assertIn("empty_like", result[0].description)

        with self.subTest("default value change without adapter - WARNING"):
            # Only default value changed
            diff_output = """
--- a/aten/src/ATen/native/native_functions.yaml
+++ b/aten/src/ATen/native/native_functions.yaml
@@ -100,7 +100,7 @@
-- func: empty_like(Tensor self, *, int arg=0) -> Tensor
+- func: empty_like(Tensor self, *, int arg=1) -> Tensor
"""
            result = check_with_adapter(
                'register_schema_adapter("aten::clone", TORCH_VERSION_2_9_0, adapt_clone);\n',
                diff_output,
            )

            self.assertEqual(len(result), 1)
            self.assertEqual(result[0].severity, LintSeverity.WARNING)
            self.assertEqual(result[0].name, "default-value-change")
            self.assertIn(
                "Default value changed for arg in 'empty_like'", result[0].description
            )
            self.assertIn("TORCH_FEATURE_VERSION", result[0].description)

        with self.subTest("correct adapter version"):
            diff_output = """
--- a/aten/src/ATen/native/native_functions.yaml
+++ b/aten/src/ATen/native/native_functions.yaml
@@ -100,7 +100,7 @@
-- func: empty_like(Tensor self) -> Tensor
+- func: empty_like(Tensor self, *, int new_arg=0) -> Tensor
"""
            version_macro = f"TORCH_VERSION_{current_version[0]}_{current_version[1]}_0"
            result = check_with_adapter(
                f'register_schema_adapter("aten::empty_like", {version_macro}, adapt_empty_like);\n',
                diff_output,
            )

            self.assertEqual(len(result), 0)

        with self.subTest("incorrect adapter version"):
            diff_output = """
--- a/aten/src/ATen/native/native_functions.yaml
+++ b/aten/src/ATen/native/native_functions.yaml
@@ -100,7 +100,7 @@
-- func: empty_like(Tensor self) -> Tensor
+- func: empty_like(Tensor self, *, int new_arg=0) -> Tensor
"""
            version_macro = f"TORCH_VERSION_{wrong_version[0]}_{wrong_version[1]}_0"
            result = check_with_adapter(
                f'register_schema_adapter("aten::empty_like", {version_macro}, adapt_empty_like);\n',
                diff_output,
            )

            self.assertEqual(len(result), 1)
            self.assertEqual(result[0].name, "incorrect-adapter-version")
            self.assertIn(
                f"Adapter registered with: TORCH_VERSION_{wrong_version[0]}_{wrong_version[1]}_0\n"
                f"Current version: TORCH_VERSION_{current_version[0]}_{current_version[1]}_0",
                result[0].description,
            )

    @patch("tools.linter.adapters.native_function_schema_linter.subprocess.run")
    def test_check_file_no_changes(self, mock_run):
        """
        Test check_file passes when there are no schema changes.
        """
        mock_run.return_value.stdout = ""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".cpp") as shim_tmp:
            shim_path = Path(shim_tmp.name)
            shim_tmp.write("")
            shim_tmp.flush()

            result = check_file(
                "native_functions.yaml",
                Path("native_functions.yaml"),
                shim_path,
            )

            self.assertEqual(len(result), 0)


if __name__ == "__main__":
    unittest.main()
