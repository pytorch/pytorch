# Owner(s): ["module: cpp"]

import os
import shutil
import subprocess
import sys
import unittest
from pathlib import Path

from torch.testing._internal.common_device_type import (
    instantiate_device_type_tests,
    onlyCUDA,
)
from torch.testing._internal.common_utils import IS_LINUX, run_tests, shell, TestCase


class TestPythonAgnostic(TestCase):
    @classmethod
    def setUpClass(cls):
        # Wipe the dist dir if it exists
        cls.extension_root = Path(__file__).parent.parent
        cls.dist_dir = os.path.join(cls.extension_root, "dist")
        if os.path.exists(cls.dist_dir):
            shutil.rmtree(cls.dist_dir)

        # Build the wheel
        wheel_cmd = [sys.executable, "-m", "build", "--wheel", "--no-isolation"]
        return_code = shell(wheel_cmd, cwd=cls.extension_root, env=os.environ)
        if return_code != 0:
            raise RuntimeError("python_agnostic bdist_wheel failed to build")

    @onlyCUDA
    @unittest.skipIf(not IS_LINUX, "test requires linux tools ldd and nm")
    def test_extension_is_python_agnostic(self, device):
        # For this test, run_test.py will call `python -m build --wheel --no-isolation` in the
        # cpp_extensions/python_agnostic_extension folder, where the extension and
        # setup calls specify py_limited_api to `True`. To approximate that the
        # extension is indeed python agnostic, we test
        #   a. The extension wheel name contains "cp39-abi3", meaning the wheel
        # should be runnable for any Python 3 version after and including 3.9
        #   b. The produced shared library does not have libtorch_python.so as a
        # dependency from the output of "ldd _C.so"
        #   c. The .so does not need any python related symbols. We approximate
        # this by running "nm -u _C.so" and grepping that nothing starts with "Py"

        matches = list(Path(self.dist_dir).glob("*.whl"))
        self.assertEqual(len(matches), 1, msg=str(matches))
        whl_file = matches[0]
        self.assertRegex(str(whl_file), r".*python_agnostic-0\.0-cp39-abi3-.*\.whl")

        build_dir = os.path.join(self.extension_root, "build")
        matches = list(Path(build_dir).glob("**/*.so"))
        self.assertEqual(len(matches), 1, msg=str(matches))
        so_file = matches[0]
        lddtree = subprocess.check_output(["ldd", so_file]).decode("utf-8")
        self.assertFalse("torch_python" in lddtree)

        missing_symbols = subprocess.check_output(["nm", "-u", so_file]).decode("utf-8")
        self.assertFalse("Py" in missing_symbols)


instantiate_device_type_tests(TestPythonAgnostic, globals(), only_for="cuda")

if __name__ == "__main__":
    run_tests()
