# Owner(s): ["module: onnx"]

import subprocess
import sys
import tempfile

import pytorch_test_common

from torch.testing._internal import common_utils


class TestLazyONNXRuntime(pytorch_test_common.ExportTestCase):

    def test_onnxruntime_is_lazily_imported(self):

        with tempfile.TemporaryDirectory() as wd:
            r = subprocess.run([sys.executable, '-Ximporttime', '-c', 'import sys, torch.onnx'], capture_output=True, text=True, cwd=wd, check=True)

        self.assertTrue(' onnxruntime' not in r.stderr, f'`onnxruntime` should not be imported, full importtime: {r.stderr}')


if __name__ == "__main__":
    common_utils.run_tests()
