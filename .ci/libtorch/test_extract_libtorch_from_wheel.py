#!/usr/bin/env python3
"""Tests for extract_libtorch_from_wheel.py, focusing on fix_rpath."""

import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from extract_libtorch_from_wheel import fix_rpath


class TestFixRpath(unittest.TestCase):
    def test_skips_non_linux(self):
        """fix_rpath should be a no-op on non-linux platforms."""
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "libfoo.so"
            p.write_bytes(b"fake")
            with patch("extract_libtorch_from_wheel.subprocess.run") as mock_run:
                fix_rpath(Path(d), "macos")
                mock_run.assert_not_called()

    def test_calls_patchelf_on_so_files(self):
        """fix_rpath should call patchelf on .so files."""
        with tempfile.TemporaryDirectory() as d:
            so = Path(d) / "libtorch.so"
            so.write_bytes(b"fake")
            txt = Path(d) / "readme.txt"
            txt.write_text("not a library")

            with (
                patch("extract_libtorch_from_wheel.shutil.which", return_value="/usr/local/bin/patchelf"),
                patch("extract_libtorch_from_wheel.subprocess.run") as mock_run,
            ):
                mock_run.return_value = subprocess.CompletedProcess(args=[], returncode=0)
                fix_rpath(Path(d), "linux")

                mock_run.assert_called_once()
                args = mock_run.call_args[0][0]
                self.assertEqual(args[0], "/usr/local/bin/patchelf")
                self.assertEqual(args[1], "--set-rpath")
                self.assertEqual(args[2], "$ORIGIN")
                self.assertEqual(args[3], "--force-rpath")
                self.assertIn("libtorch.so", args[4])

    def test_skips_non_so_files(self):
        """fix_rpath should not touch files without .so in the name."""
        with tempfile.TemporaryDirectory() as d:
            lib = Path(d) / "libfoo.a"
            lib.write_bytes(b"fake")

            with (
                patch("extract_libtorch_from_wheel.shutil.which", return_value="/usr/local/bin/patchelf"),
                patch("extract_libtorch_from_wheel.subprocess.run") as mock_run,
            ):
                fix_rpath(Path(d), "linux")
                mock_run.assert_not_called()

    def test_raises_if_patchelf_not_found(self):
        """fix_rpath should raise if patchelf is not available."""
        with tempfile.TemporaryDirectory() as d:
            so = Path(d) / "libfoo.so"
            so.write_bytes(b"fake")

            with patch("extract_libtorch_from_wheel.shutil.which", return_value=None):
                with self.assertRaises(FileNotFoundError):
                    fix_rpath(Path(d), "linux")

    @unittest.skipUnless(shutil.which("patchelf"), "patchelf not installed")
    def test_integration_with_real_patchelf(self):
        """Integration test: actually run patchelf if available."""
        with tempfile.TemporaryDirectory() as d:
            so = Path(d) / "libtest.so"
            # Create a minimal shared library with gcc if available
            gcc = shutil.which("gcc") or shutil.which("cc")
            if not gcc:
                self.skipTest("no C compiler available")

            src = Path(d) / "test.c"
            src.write_text("void foo(void) {}")
            result = subprocess.run(
                [gcc, "-shared", "-o", str(so), str(src)],
                capture_output=True,
            )
            if result.returncode != 0:
                self.skipTest("failed to compile test shared library")

            fix_rpath(Path(d), "linux")

            # Verify the rpath was set
            result = subprocess.run(
                ["patchelf", "--print-rpath", str(so)],
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.stdout.strip(), "$ORIGIN")


if __name__ == "__main__":
    unittest.main()
