import importlib
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock


def import_oss_utils():
    os.environ.setdefault("HOME", str(Path.home()))
    return importlib.import_module("tools.code_coverage.package.oss.utils")


class TestCodeCoverageUtils(unittest.TestCase):
    def test_get_gcda_files_recursively_discovers_and_sorts_files(self) -> None:
        oss_utils = import_oss_utils()
        with tempfile.TemporaryDirectory() as tmpdir:
            build_dir = Path(tmpdir) / "build"
            nested_dir = build_dir / "aten" / "nested"
            nested_dir.mkdir(parents=True)

            top_level = build_dir / "top.gcda"
            deep_file = nested_dir / "deep.gcda"
            top_level.touch()
            deep_file.touch()
            (build_dir / "ignore.txt").touch()

            with mock.patch.object(
                oss_utils, "get_pytorch_folder", return_value=tmpdir
            ):
                self.assertEqual(
                    oss_utils.get_gcda_files(),
                    sorted([str(top_level), str(deep_file)]),
                )

    def test_get_gcda_files_returns_empty_when_build_folder_missing(self) -> None:
        oss_utils = import_oss_utils()
        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch.object(
                oss_utils, "get_pytorch_folder", return_value=tmpdir
            ):
                self.assertEqual(oss_utils.get_gcda_files(), [])


if __name__ == "__main__":
    unittest.main()
