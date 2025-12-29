import os
import tempfile
from unittest import mock

from torch._tempdir import get_pytorch_tmpdir, get_temp_path
from torch.testing._internal.common_utils import run_tests, TestCase


class TestTempDir(TestCase):
    def test_default_uses_system_temp(self):
        """Verify fallback to tempfile.gettempdir() when PYTORCH_TMPDIR unset."""
        # Remove only PYTORCH_TMPDIR, preserve system temp env vars (TMPDIR, TEMP, TMP)
        env_without_pytorch_tmpdir = {
            k: v for k, v in os.environ.items() if k != "PYTORCH_TMPDIR"
        }
        with mock.patch.dict(os.environ, env_without_pytorch_tmpdir, clear=True):
            result = get_pytorch_tmpdir()
            # Should match system temp resolution which respects TMPDIR/TEMP/TMP
            self.assertEqual(result, tempfile.gettempdir())

    def test_respects_pytorch_tmpdir_env(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch.dict(os.environ, {"PYTORCH_TMPDIR": tmpdir}):
                result = get_pytorch_tmpdir()
                self.assertEqual(result, tmpdir)

    def test_raises_if_pytorch_tmpdir_does_not_exist(self):
        # Use a path that's unlikely to exist on any platform
        nonexistent = os.path.join(tempfile.gettempdir(), "nonexistent_12345_xyz")
        with mock.patch.dict(os.environ, {"PYTORCH_TMPDIR": nonexistent}):
            with self.assertRaises(RuntimeError) as ctx:
                get_pytorch_tmpdir()
            self.assertIn("does not exist", str(ctx.exception))

    def test_raises_if_pytorch_tmpdir_is_not_directory(self):
        with tempfile.NamedTemporaryFile(delete=False) as f:
            temp_file = f.name
        try:
            with mock.patch.dict(os.environ, {"PYTORCH_TMPDIR": temp_file}):
                with self.assertRaises(RuntimeError) as ctx:
                    get_pytorch_tmpdir()
                self.assertIn("not a directory", str(ctx.exception))
        finally:
            os.unlink(temp_file)

    def test_get_temp_path_creates_subdirectory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch.dict(os.environ, {"PYTORCH_TMPDIR": tmpdir}):
                subdir_path = get_temp_path(subdirectory="test_subdir")
                self.assertTrue(os.path.isdir(subdir_path))

    def test_get_temp_path_with_subdirectory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch.dict(os.environ, {"PYTORCH_TMPDIR": tmpdir}):
                result = get_temp_path(subdirectory="subdir")
                self.assertEqual(result, os.path.join(tmpdir, "subdir"))

    def test_get_temp_path_with_filename(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch.dict(os.environ, {"PYTORCH_TMPDIR": tmpdir}):
                result = get_temp_path(filename="file.txt")
                self.assertEqual(result, os.path.join(tmpdir, "file.txt"))

    def test_get_temp_path_with_subdirectory_and_filename(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch.dict(os.environ, {"PYTORCH_TMPDIR": tmpdir}):
                result = get_temp_path(subdirectory="subdir", filename="file.txt")
                self.assertEqual(result, os.path.join(tmpdir, "subdir", "file.txt"))
                # Verify subdirectory was created
                self.assertTrue(os.path.isdir(os.path.join(tmpdir, "subdir")))

    def test_get_temp_path_nested_subdirectory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch.dict(os.environ, {"PYTORCH_TMPDIR": tmpdir}):
                # Use os.path.join for cross-platform nested paths
                nested = os.path.join("level1", "level2", "level3")
                result = get_temp_path(subdirectory=nested)
                expected = os.path.join(tmpdir, "level1", "level2", "level3")
                self.assertEqual(result, expected)
                self.assertTrue(os.path.isdir(expected))

    def test_get_temp_path_idempotent_directory_creation(self):
        """Verify calling get_temp_path multiple times doesn't raise errors."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch.dict(os.environ, {"PYTORCH_TMPDIR": tmpdir}):
                # Call twice with same subdirectory
                result1 = get_temp_path(subdirectory="repeated_subdir")
                result2 = get_temp_path(subdirectory="repeated_subdir")
                self.assertEqual(result1, result2)
                self.assertTrue(os.path.isdir(result1))


if __name__ == "__main__":
    run_tests()
