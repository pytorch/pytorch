import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(REPO_ROOT))

from tools.linter.adapters.stable_shim_version_linter import (
    check_file,
    get_added_lines,
    get_current_version,
)


class TestStableShimVersionLinter(unittest.TestCase):
    """Test the overall stable shim version linter functionality."""

    def test_get_added_lines_simple_addition(self):
        """Test parsing a simple git diff with added lines."""
        simulated_diff = """diff --git a/torch/csrc/stable/c/shim.h b/torch/csrc/stable/c/shim.h
index 365c954dbe7..18ca6525f73 100644
--- a/torch/csrc/stable/c/shim.h
+++ b/torch/csrc/stable/c/shim.h
@@ -15,6 +15,13 @@
 extern "C" {
 #endif

+AOTI_TORCH_EXPORT AOTITorchError torch_call_dispatcher()
+    const char* opName,
+    const char* overloadName,
+    StableIValue* stack,
+    uint64_t extension_build_version);
+
+
 #if TORCH_FEATURE_VERSION >= TORCH_VERSION_2_10_0
 using StableIValue = uint64_t;
"""
        with patch("subprocess.run") as mock_run:
            # Mock both git diff calls to return our simulated diff
            mock_result = MagicMock()
            mock_result.returncode = 0
            mock_result.stdout = simulated_diff
            mock_run.return_value = mock_result

            result = get_added_lines("torch/csrc/stable/c/shim.h")

            # Lines 18-24 should be marked as added
            # Hunk starts at line 15, then 3 context lines (15-17),
            # then 7 added lines (18-24): 5 lines of function declaration + 2 empty lines
            self.assertEqual(result, {18, 19, 20, 21, 22, 23, 24})

    def test_get_added_lines_parse_multiple_hunks(self):
        """Test parsing git diff with multiple hunks."""
        simulated_diff = """diff --git a/torch/csrc/stable/c/shim.h b/torch/csrc/stable/c/shim.h
index 365c954dbe7..c5fcf7a09cf 100644
--- a/torch/csrc/stable/c/shim.h
+++ b/torch/csrc/stable/c/shim.h
@@ -15,6 +15,13 @@
 extern "C" {
 #endif

+AOTI_TORCH_EXPORT AOTITorchError torch_call_dispatcher(
+    const char* opName,
+    const char* overloadName,
+    StableIValue* stack,
+    uint64_t extension_build_version);
+
+
 #if TORCH_FEATURE_VERSION >= TORCH_VERSION_2_10_0
 using StableIValue = uint64_t;

@@ -39,6 +46,12 @@ AOTI_TORCH_EXPORT AOTITorchError torch_library_impl(

 #endif // TORCH_FEATURE_VERSION >= TORCH_VERSION_2_10_0

+AOTI_TORCH_EXPORT AOTITorchError torch_library_impl(
+    TorchLibraryHandle self,
+    const char* name,
+    void (*fn)(StableIValue*, uint64_t, uint64_t),
+    uint64_t extension_build_version);
+
 #ifdef __cplusplus
 } // extern "C"
 #endif
"""
        with patch("subprocess.run") as mock_run:
            mock_result = MagicMock()
            mock_result.returncode = 0
            mock_result.stdout = simulated_diff
            mock_run.return_value = mock_result

            result = get_added_lines("test.h")

            # Should find added lines in both hunks
            self.assertEqual(
                result, {18, 19, 20, 21, 22, 23, 24, 49, 50, 51, 52, 53, 54}
            )

    def test_get_current_version(self):
        """Test that we can get the current version from version.txt."""
        version = get_current_version()
        self.assertIsInstance(version, tuple)
        self.assertEqual(len(version), 2)
        # We can't check torch.__version__ here so this is the best we can do :(
        self.assertIsInstance(version[0], int)
        self.assertIsInstance(version[1], int)

    def test_check_file_with_simulated_content(self):
        """Test checking a file with simulated content and git diffs."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".h", delete=False) as f:
            f.write("""
#ifndef TEST_H
#define TEST_H

extern "C" {

#if TORCH_FEATURE_VERSION >= TORCH_VERSION_2_10_0
AOTI_TORCH_EXPORT int new_function_correct_version();
#endif // TORCH_FEATURE_VERSION >= TORCH_VERSION_2_10_0

#if TORCH_FEATURE_VERSION >= TORCH_VERSION_2_9_0
AOTI_TORCH_EXPORT int new_function_wrong_version();
#endif // TORCH_FEATURE_VERSION >= TORCH_VERSION_2_9_0

AOTI_TORCH_EXPORT int function_without_version();

} // extern "C"

#endif
""")
            f.flush()
            temp_file = f.name

            # Mock git diff to say lines 8, 12, and 15 are new
            simulated_diff = """@@ -1,0 +8,1 @@
+AOTI_TORCH_EXPORT int new_function_correct_version();
@@ -1,0 +12,1 @@
+AOTI_TORCH_EXPORT int new_function_wrong_version();
@@ -1,0 +15,1 @@
+AOTI_TORCH_EXPORT int function_without_version();
"""

            with patch("subprocess.run") as mock_run:
                mock_result = MagicMock()
                mock_result.returncode = 0
                mock_result.stdout = simulated_diff
                mock_run.return_value = mock_result

                # Mock version to be 2.10
                with patch(
                    "tools.linter.adapters.stable_shim_version_linter.get_current_version",
                    return_value=(2, 10),
                ):
                    lint_messages = check_file(temp_file)

                    # Should have 2 errors:
                    # 1. Line 12: wrong version (2.9 instead of 2.10)
                    # 2. Line 15: no version block
                    self.assertEqual(len(lint_messages), 2)

                    errors_by_name = {msg.name: msg for msg in lint_messages}

                    # Check error 1: wrong-version-for-new-function
                    self.assertIn("wrong-version-for-new-function", errors_by_name)
                    wrong_version_msg = errors_by_name["wrong-version-for-new-function"]
                    self.assertEqual(wrong_version_msg.line, 12)
                    self.assertIsNotNone(wrong_version_msg.description)
                    self.assertTrue(
                        "should use TORCH_VERSION_2_10_0, but is wrapped in TORCH_VERSION_2_9_0"
                        in wrong_version_msg.description
                    )

                    # Check error 2: unversioned-function-declaration
                    self.assertIn("unversioned-function-declaration", errors_by_name)
                    unversioned_msg = errors_by_name["unversioned-function-declaration"]
                    self.assertEqual(unversioned_msg.line, 15)
                    self.assertIsNotNone(unversioned_msg.description)
                    self.assertTrue(
                        "outside of TORCH_FEATURE_VERSION block"
                        in unversioned_msg.description
                    )
                    self.assertTrue(
                        "TORCH_FEATURE_VERSION >= TORCH_VERSION_2_10_0"
                        in unversioned_msg.description
                    )


if __name__ == "__main__":
    unittest.main()
