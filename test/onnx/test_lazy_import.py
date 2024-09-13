# Owner(s): ["module: onnx"]

import subprocess
import sys
import tempfile

import pytorch_test_common

from torch.testing._internal import common_utils


class TestLazyONNXPackages(pytorch_test_common.ExportTestCase):
    def _test_package_is_lazily_imported(self, pkg, torch_pkg="torch.onnx"):
        with tempfile.TemporaryDirectory() as wd:
            r = subprocess.run(
                [sys.executable, "-Ximporttime", "-c", "import torch.onnx"],
                capture_output=True,
                text=True,
                cwd=wd,
                check=True,
            )

        # The extra space makes sure we're checking the package, not any package containing its name.
        self.assertTrue(
            f" {pkg}" not in r.stderr,
            f"`{pkg}` should not be imported, full importtime: {r.stderr}",
        )

    def test_onnxruntime_is_lazily_imported(self):
        self._test_package_is_lazily_imported("onnxruntime")

    def test_onnxscript_is_lazily_imported(self):
        self._test_package_is_lazily_imported("onnxscript")


if __name__ == "__main__":
    common_utils.run_tests()
