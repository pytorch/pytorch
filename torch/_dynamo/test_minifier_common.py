import os
import re
import shutil
import subprocess
import tempfile
import unittest

import torch
import torch._dynamo
import torch._dynamo.test_case
from torch._dynamo.debug_utils import TEST_REPLACEABLE_COMMENT

class MinifierTestBase(torch._dynamo.test_case.TestCase):
    DEBUG_DIR = os.path.join(tempfile.gettempdir(), "_torchdynamo_debug_")

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls._exit_stack.enter_context(
            unittest.mock.patch.object(
                torch._dynamo.config,
                "debug_dir_root",
                cls.DEBUG_DIR,
            )
        )
        os.makedirs(cls.DEBUG_DIR, exist_ok=True)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(torch._dynamo.config.debug_dir_root, ignore_errors=True)
        cls._exit_stack.close()

    def setUp(self):
        super().setUp()

    def tearDown(self):
        super().tearDown()

    # Search for the name of the first function defined in a code string.
    def _get_fn_name(self, code):
        fn_name_match = re.search(r"def (\w+)\(", code)
        if fn_name_match is not None:
            return fn_name_match.group(1)
        return None

    # Run `code` in a separate python process.
    # Returns the completed process state and the directory containing the
    # minifier launcher script, if `code` outputted it.
    def _run_test_code(self, code):
        proc = subprocess.run(
            ["python3", "-c", code], capture_output=True, cwd=self.DEBUG_DIR
        )

        print(proc.stderr.decode("utf-8"))

        repro_dir_match = re.search(
            r"(\S+)minifier_launcher.py", proc.stderr.decode("utf-8")
        )
        if repro_dir_match is not None:
            # Print repro directory for debugging generated code.
            # Make sure to comment out `shutil.rmtree...` above as well.
            print("repro dir:", repro_dir_match.group(1))
            return proc, repro_dir_match.group(1)
        return proc, None

    # Patch generated files with testing patches
    def _inject_code(self, patch_code, filename):
        patch_code = f"""\
{patch_code}
torch._dynamo.config.debug_dir_root = "{self.DEBUG_DIR}"
"""
        with open(filename, "r") as f:
            code = f.read()
        code = code.replace(TEST_REPLACEABLE_COMMENT, patch_code)
        with open(filename, "w") as f:
            f.write(code)
        return code

    # Runs the minifier launcher script in `repro_dir`, patched with `patch_code`.
    def _run_minifier_launcher(self, patch_code, repro_dir):
        self.assertIsNotNone(repro_dir)
        launch_file = os.path.join(repro_dir, "minifier_launcher.py")
        self.assertTrue(os.path.exists(launch_file))
        launch_code = self._inject_code(patch_code, launch_file)

        launch_proc = subprocess.run(
            ["python3", launch_file],
            capture_output=True,
            cwd=repro_dir,
        )
        print(launch_proc.stderr.decode("utf-8"))

        return launch_proc, launch_code

    # Runs the repro script in `repro_dir`, patched with `patch_code`
    def _run_repro(self, patch_code, repro_dir):
        self.assertIsNotNone(repro_dir)
        repro_file = os.path.join(repro_dir, "repro.py")
        self.assertTrue(os.path.exists(repro_file))
        repro_code = self._inject_code(patch_code, repro_file)

        repro_proc = subprocess.run(
            ["python3", repro_file], capture_output=True, cwd=repro_dir
        )
        print(repro_proc.stderr.decode("utf-8"))

        return repro_proc, repro_code

    # Template for testing code.
    # `run_code` is the code to run for the test case.
    # `patch_code` is the code to be patched in every generated file.
    def _gen_test_code(self, run_code, repro_after, repro_level, patch_code):
        return f"""\
import torch
import torch._dynamo
{patch_code}
torch._dynamo.config.repro_after = "{repro_after}"
torch._dynamo.config.repro_level = {repro_level}
torch._dynamo.config.debug_dir_root = "{self.DEBUG_DIR}"
{run_code}
"""

    # Runs a full minifier test.
    # Minifier tests generally consist of 3 stages:
    # 1. Run the problematic code (in a separate process since it could segfault)
    # 2. Run the generated minifier launcher script
    # 3. Run the generated repro script
    def _run_full_test(self, run_code, repro_after, repro_level, patch_code):
        test_code = self._gen_test_code(run_code, repro_after, repro_level, patch_code)
        test_proc, repro_dir = self._run_test_code(test_code)
        self.assertIsNotNone(repro_dir)
        launch_proc, launch_code = self._run_minifier_launcher(patch_code, repro_dir)
        repro_proc, repro_code = self._run_repro(patch_code, repro_dir)
        return ((test_proc, launch_proc, repro_proc), (launch_code, repro_code))
