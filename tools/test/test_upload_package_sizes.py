from __future__ import annotations

import os
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from typing import Any
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from tools.stats import upload_package_sizes


sys.path.remove(str(REPO_ROOT))


class TestUploadPackageSizes(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dist = Path(tmp.name)

    def write_wheel(self, name: str, files: dict[str, bytes]) -> Path:
        wheel = self.dist / name
        with zipfile.ZipFile(wheel, "w", zipfile.ZIP_DEFLATED) as archive:
            for path, data in files.items():
                archive.writestr(path, data)
        return wheel

    def run_main(self) -> list[dict[str, Any]]:
        """Run the script on `self.dist` and return the `package_size` metrics."""
        with (
            mock.patch.object(upload_package_sizes, "emit_metric") as emit_metric,
            mock.patch.dict("os.environ", {"HEAD_SHA": "abc123"}),
            mock.patch("sys.argv", ["upload_package_sizes", str(self.dist)]),
        ):
            upload_package_sizes.main()
        for call in emit_metric.call_args_list:
            self.assertEqual(call.args[0], "package_size")
        return [call.args[1] for call in emit_metric.call_args_list]

    def test_sizes_cover_the_wheel_and_its_largest_files(self) -> None:
        # Random bytes do not compress, zeros do.
        cuda = os.urandom(5000)
        wheel = self.write_wheel(
            "torch-2.16.0a0+gitabc-cp312-cp312-linux_x86_64.whl",
            {
                "torch/lib/libtorch_cuda.so": cuda,
                "torch/lib/libtorch_cpu.so": bytes(3000),
                "torch/__init__.py": b"x" * 10,
            },
        )

        sizes = upload_package_sizes.package_sizes(wheel)

        self.assertEqual(sizes["wheel_bytes"], wheel.stat().st_size)
        self.assertEqual(sizes["uncompressed_bytes"], 8010)
        self.assertEqual(
            list(sizes["largest_files"]),
            [
                "torch/lib/libtorch_cuda.so",
                "torch/lib/libtorch_cpu.so",
                "torch/__init__.py",
            ],
        )
        cpu = sizes["largest_files"]["torch/lib/libtorch_cpu.so"]
        self.assertEqual(cpu["uncompressed_bytes"], 3000)
        self.assertLess(cpu["compressed_bytes"], 100)
        self.assertGreaterEqual(
            sizes["largest_files"]["torch/lib/libtorch_cuda.so"]["compressed_bytes"],
            5000,
        )

    def test_only_the_largest_files_are_listed(self) -> None:
        # File i holds i + 1 bytes.
        wheel = self.write_wheel(
            "many.whl", {f"pkg/file{i}": b"x" * (i + 1) for i in range(5)}
        )

        sizes = upload_package_sizes.package_sizes(wheel, largest_files=2)

        self.assertEqual(list(sizes["largest_files"]), ["pkg/file4", "pkg/file3"])
        self.assertEqual(sizes["uncompressed_bytes"], 1 + 2 + 3 + 4 + 5)

    def test_only_the_torch_wheel_is_recorded_with_its_commit(self) -> None:
        self.write_wheel("torch-2.16.0-cp312-linux.whl", {"torch/lib/a.so": b"a" * 40})
        self.write_wheel("torchvision-0.30.0-cp312-linux.whl", {"tv/b.so": b"b" * 7})
        self.write_wheel("rocm_composable_kernel-10.0-py3.whl", {"ck/c.so": b"c" * 9})
        (self.dist / ".ninja_log").write_text("not a wheel")

        metrics = self.run_main()

        self.assertEqual(
            [(metric["wheel"], metric["uncompressed_bytes"]) for metric in metrics],
            [("torch-2.16.0-cp312-linux.whl", 40)],
        )
        self.assertEqual(metrics[0]["head_sha"], "abc123")

    def test_directory_without_a_torch_wheel_uploads_nothing(self) -> None:
        self.write_wheel("torchvision-0.30.0-cp312-linux.whl", {"tv/b.so": b"b" * 7})

        self.assertEqual(self.run_main(), [])


if __name__ == "__main__":
    unittest.main()
