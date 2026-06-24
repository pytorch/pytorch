#!/usr/bin/env python3
"""Tests for the libtorch extraction pipeline."""

import shutil
import tempfile
import unittest
import zipfile
from pathlib import Path

from extract_libtorch_from_wheel import (
    compute_zip_prefix,
    copy_bin,
    copy_cmake,
    copy_includes,
    copy_libraries,
    create_libtorch_zip,
    extract_wheel,
    find_wheel,
    get_git_hash,
    parse_version_from_wheel,
    write_metadata,
)


def _make_fake_wheel(wheel_dir: Path) -> Path:
    wheel_path = wheel_dir / "torch-0.0-cp310-cp310-linux_x86_64.whl"
    with zipfile.ZipFile(wheel_path, "w") as zf:
        zf.writestr("torch/lib/libtorch.so", "fake")
        zf.writestr("torch/include/torch/torch.h", "// header")
        zf.writestr("torch/share/cmake/Torch/TorchConfig.cmake", "# cmake")
        zf.writestr("torch/version.py", "git_version = 'abc123'\n")
    return wheel_path


class TestExtraction(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.mkdtemp()
        self.wheel_dir = Path(self._tmp) / "wheels"
        self.wheel_dir.mkdir()
        self.output_dir = Path(self._tmp) / "output"
        self.output_dir.mkdir()
        _make_fake_wheel(self.wheel_dir)

    def tearDown(self):
        shutil.rmtree(self._tmp)

    def test_extraction(self):
        wheel_path = find_wheel(str(self.wheel_dir))
        version = parse_version_from_wheel(wheel_path)
        extract_dir = self.wheel_dir / "_tmp"
        extract_dir.mkdir()
        torch_dir = extract_wheel(wheel_path, extract_dir)

        libtorch_dir = extract_dir / "libtorch"
        libtorch_dir.mkdir()
        for sub in ["lib", "bin", "include", "share"]:
            (libtorch_dir / sub).mkdir()

        copy_libraries(torch_dir, libtorch_dir / "lib", "linux")
        copy_includes(torch_dir, libtorch_dir / "include")
        copy_cmake(torch_dir, libtorch_dir / "share")
        copy_bin(torch_dir, libtorch_dir / "bin", "linux")
        write_metadata(libtorch_dir, version, get_git_hash(torch_dir))

        prefix = compute_zip_prefix("linux", "cu126", "shared-with-deps", "x86_64")
        zip_path = create_libtorch_zip(libtorch_dir, self.output_dir, prefix, version)

        with zipfile.ZipFile(zip_path) as zf:
            names = set(zf.namelist())

        self.assertIn("libtorch/lib/libtorch.so", names)
        self.assertIn("libtorch/include/torch/torch.h", names)
        self.assertTrue(any("TorchConfig.cmake" in n for n in names))


if __name__ == "__main__":
    unittest.main()
