#!/usr/bin/env python3
"""Tests for the libtorch extraction pipeline."""

import shutil
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from extract_libtorch_from_wheel import (
    compute_zip_prefix,
    copy_bin,
    copy_cmake,
    copy_includes,
    copy_libraries,
    create_libtorch_zip,
    extract_wheel,
    find_wheel,
    fix_rpath,
    get_git_hash,
    parse_version_from_wheel,
    write_metadata,
)


def _make_fake_wheel(wheel_dir: Path, version: str = "2.6.0") -> Path:
    name = f"torch-{version}-cp310-cp310-linux_x86_64.whl"
    wheel_path = wheel_dir / name
    with zipfile.ZipFile(wheel_path, "w") as zf:
        zf.writestr("torch/lib/libtorch.so", "fake")
        zf.writestr("torch/lib/libtorch_cpu.so", "fake")
        zf.writestr("torch/lib/libtorch_python.so", "excluded")
        zf.writestr("torch/lib/_C.cpython-310-x86_64-linux-gnu.so", "excluded")
        zf.writestr("torch/include/torch/torch.h", "// header")
        zf.writestr("torch/share/cmake/Torch/TorchConfig.cmake", "# cmake")
        zf.writestr("torch/version.py", "git_version = 'abc123'\n")
    return wheel_path


class TestSmokeExtraction(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.mkdtemp()
        self.wheel_dir = Path(self._tmp) / "wheels"
        self.wheel_dir.mkdir()
        self.output_dir = Path(self._tmp) / "output"
        self.output_dir.mkdir()
        _make_fake_wheel(self.wheel_dir)

    def tearDown(self):
        shutil.rmtree(self._tmp)

    def _extract(self, platform="linux"):
        wheel_path = find_wheel(str(self.wheel_dir))
        version = parse_version_from_wheel(wheel_path)
        extract_dir = self.wheel_dir / "_tmp"
        extract_dir.mkdir()
        torch_dir = extract_wheel(wheel_path, extract_dir)

        libtorch_dir = extract_dir / "libtorch"
        libtorch_dir.mkdir()
        for sub in ["lib", "bin", "include", "share"]:
            (libtorch_dir / sub).mkdir()

        copy_libraries(torch_dir, libtorch_dir / "lib", platform)
        copy_includes(torch_dir, libtorch_dir / "include")
        copy_cmake(torch_dir, libtorch_dir / "share")
        copy_bin(torch_dir, libtorch_dir / "bin", platform)
        write_metadata(libtorch_dir, version, get_git_hash(torch_dir))

        prefix = compute_zip_prefix(platform, "cu126", "shared-with-deps")
        return create_libtorch_zip(libtorch_dir, self.output_dir, prefix, version)

    def test_output_zip_has_expected_files(self):
        zip_path = self._extract()
        with zipfile.ZipFile(zip_path) as zf:
            names = set(zf.namelist())
        self.assertIn("libtorch/lib/libtorch.so", names)
        self.assertIn("libtorch/include/torch/torch.h", names)
        self.assertTrue(any("TorchConfig.cmake" in n for n in names))
        self.assertIn("libtorch/build-version", names)
        self.assertIn("libtorch/build-hash", names)

    def test_excluded_libs_not_in_zip(self):
        zip_path = self._extract()
        with zipfile.ZipFile(zip_path) as zf:
            names = set(zf.namelist())
        self.assertNotIn("libtorch/lib/libtorch_python.so", names)
        self.assertFalse(any("_C.cpython" in n for n in names))

    def test_metadata_content(self):
        zip_path = self._extract()
        with zipfile.ZipFile(zip_path) as zf:
            self.assertEqual(zf.read("libtorch/build-version").decode().strip(), "2.6.0")  # noqa: B950
            self.assertEqual(zf.read("libtorch/build-hash").decode().strip(), "abc123")

    def test_latest_symlink_created(self):
        self._extract()
        symlinks = list(self.output_dir.glob("*-latest.zip"))
        self.assertEqual(len(symlinks), 1)
        self.assertTrue(symlinks[0].is_symlink())

    def test_zip_prefix_naming(self):
        self.assertEqual(compute_zip_prefix("linux", "cu126", "shared-with-deps"), "libtorch-shared-with-deps")  # noqa: B950
        self.assertEqual(compute_zip_prefix("macos", "cpu", "shared-with-deps"), "libtorch-macos-arm64")  # noqa: B950
        self.assertEqual(compute_zip_prefix("windows", "cu126", "shared-with-deps"), "libtorch-win-shared-with-deps")  # noqa: B950

    def test_find_wheel_errors(self):
        empty = Path(self._tmp) / "empty"
        empty.mkdir()
        with self.assertRaises(FileNotFoundError):
            find_wheel(str(empty))
        _make_fake_wheel(self.wheel_dir, "2.7.0")  # now two wheels
        with self.assertRaises(RuntimeError):
            find_wheel(str(self.wheel_dir))


class TestFixRpath(unittest.TestCase):
    def test_raises_on_non_linux(self):
        with tempfile.TemporaryDirectory() as d:
            Path(d, "libfoo.so").write_bytes(b"fake")
            with patch("sys.platform", "darwin"):
                with self.assertRaises(RuntimeError):
                    fix_rpath(Path(d))

    def test_calls_patchelf_on_so_files(self):
        with tempfile.TemporaryDirectory() as d:
            Path(d, "libtorch.so").write_bytes(b"fake")
            Path(d, "readme.txt").write_text("not a library")

            with (
                patch("sys.platform", "linux"),
                patch("extract_libtorch_from_wheel.shutil.which", return_value="/usr/local/bin/patchelf"),
                patch("extract_libtorch_from_wheel.subprocess.run") as mock_run,
            ):
                mock_run.return_value = subprocess.CompletedProcess(args=[], returncode=0)
                fix_rpath(Path(d))

                mock_run.assert_called_once()
                args = mock_run.call_args[0][0]
                self.assertEqual(args[0], "/usr/local/bin/patchelf")
                self.assertEqual(args[1], "--set-rpath")
                self.assertEqual(args[2], "$ORIGIN")
                self.assertEqual(args[3], "--force-rpath")
                self.assertIn("libtorch.so", args[4])

    def test_calls_patchelf_on_versioned_so(self):
        with tempfile.TemporaryDirectory() as d:
            Path(d, "libfoo.so.1").write_bytes(b"fake")

            with (
                patch("sys.platform", "linux"),
                patch("extract_libtorch_from_wheel.shutil.which", return_value="/usr/local/bin/patchelf"),
                patch("extract_libtorch_from_wheel.subprocess.run") as mock_run,
            ):
                mock_run.return_value = subprocess.CompletedProcess(args=[], returncode=0)
                fix_rpath(Path(d))
                mock_run.assert_called_once()

    def test_skips_non_so_files(self):
        with tempfile.TemporaryDirectory() as d:
            Path(d, "libfoo.a").write_bytes(b"fake")

            with (
                patch("sys.platform", "linux"),
                patch("extract_libtorch_from_wheel.shutil.which", return_value="/usr/local/bin/patchelf"),
                patch("extract_libtorch_from_wheel.subprocess.run") as mock_run,
            ):
                fix_rpath(Path(d))
                mock_run.assert_not_called()

    def test_raises_if_patchelf_not_found(self):
        with tempfile.TemporaryDirectory() as d:
            Path(d, "libfoo.so").write_bytes(b"fake")

            with (
                patch("sys.platform", "linux"),
                patch("extract_libtorch_from_wheel.shutil.which", return_value=None),
            ):
                with self.assertRaises(FileNotFoundError):
                    fix_rpath(Path(d))

    def test_raises_on_patchelf_failure(self):
        with tempfile.TemporaryDirectory() as d:
            Path(d, "libfoo.so").write_bytes(b"fake")

            with (
                patch("sys.platform", "linux"),
                patch("extract_libtorch_from_wheel.shutil.which", return_value="/usr/local/bin/patchelf"),
                patch("extract_libtorch_from_wheel.subprocess.run") as mock_run,
            ):
                mock_run.return_value = subprocess.CompletedProcess(args=[], returncode=1, stderr="cannot find section")
                with self.assertRaises(RuntimeError):
                    fix_rpath(Path(d))

    @unittest.skipUnless(sys.platform == "linux", "patchelf only supported on Linux")
    @unittest.skipUnless(shutil.which("patchelf"), "patchelf not installed")
    def test_integration_with_real_patchelf(self):
        with tempfile.TemporaryDirectory() as d:
            so = Path(d) / "libtest.so"
            gcc = shutil.which("gcc") or shutil.which("cc")
            if not gcc:
                self.skipTest("no C compiler available")

            src = Path(d) / "test.c"
            src.write_text("void foo(void) {}")
            result = subprocess.run([gcc, "-shared", "-o", str(so), str(src)], capture_output=True)
            if result.returncode != 0:
                self.skipTest("failed to compile test shared library")

            fix_rpath(Path(d))

            result = subprocess.run(["patchelf", "--print-rpath", str(so)], capture_output=True, text=True)
            self.assertEqual(result.stdout.strip(), "$ORIGIN")


if __name__ == "__main__":
    unittest.main()
