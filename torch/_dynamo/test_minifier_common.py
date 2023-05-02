import os
import re
import shutil
import subprocess
import tempfile

import torch
import torch._dynamo
import torch._dynamo.test_case


class MinifierTestBase(torch._dynamo.test_case.TestCase):
    DEBUG_DIR = tempfile.mkdtemp()

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls._exit_stack.enter_context(
            torch._dynamo.config.patch(debug_dir_root=cls.DEBUG_DIR)
        )
        # These configurations make new process startup slower.  Disable them
        # for the minification tests to speed them up.
        cls._exit_stack.enter_context(
            torch._inductor.config.patch(
                {
                    # https://github.com/pytorch/pytorch/issues/100376
                    "pattern_matcher": False,
                    # multiprocess compilation takes a long time to warmup
                    "compile_threads": 1,
                    # https://github.com/pytorch/pytorch/issues/100378
                    "cpp.vec_isa_ok": False,
                }
            )
        )

    @classmethod
    def tearDownClass(cls):
        if os.getenv("PYTORCH_KEEP_TMPDIR", "0") != "1":
            shutil.rmtree(cls.DEBUG_DIR)
        else:
            print(f"test_minifier_common tmpdir kept at: {cls.DEBUG_DIR}")
        cls._exit_stack.close()

    # Run `code` in a separate python process.
    # Returns the completed process state and the directory containing the
    # minifier launcher script, if `code` outputted it.
    def _run_test_code(self, code):
        proc = subprocess.run(
            ["python3", "-c", code], capture_output=True, cwd=self.DEBUG_DIR
        )
        print("stdout:", proc.stdout.decode("utf-8"))
        print("stderr:", proc.stderr.decode("utf-8"))
        repro_dir_match = re.search(
            r"(\S+)minifier_launcher.py", proc.stderr.decode("utf-8")
        )
        if repro_dir_match is not None:
            return proc, repro_dir_match.group(1)
        return proc, None

    # Runs the minifier launcher script in `repro_dir`
    def _run_minifier_launcher(self, repro_dir):
        self.assertIsNotNone(repro_dir)
        launch_file = os.path.join(repro_dir, "minifier_launcher.py")
        with open(launch_file, "r") as f:
            launch_code = f.read()
        self.assertTrue(os.path.exists(launch_file))

        launch_proc = subprocess.run(
            ["python3", launch_file],
            capture_output=True,
            cwd=repro_dir,
        )
        print("minifier stdout:", launch_proc.stdout.decode("utf-8"))
        print("minifier stderr:", launch_proc.stderr.decode("utf-8"))

        return launch_proc, launch_code

    # Runs the repro script in `repro_dir`
    def _run_repro(self, repro_dir):
        self.assertIsNotNone(repro_dir)
        repro_file = os.path.join(repro_dir, "repro.py")
        with open(repro_file, "r") as f:
            repro_code = f.read()
        self.assertTrue(os.path.exists(repro_file))

        repro_proc = subprocess.run(
            ["python3", repro_file], capture_output=True, cwd=repro_dir
        )
        return repro_proc, repro_code

    # Template for testing code.
    # `run_code` is the code to run for the test case.
    # `patch_code` is the code to be patched in every generated file; usually
    # just use this to turn on bugs via the config
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
        print("running minifier")
        launch_proc, launch_code = self._run_minifier_launcher(repro_dir)
        print("running repro")
        repro_proc, repro_code = self._run_repro(repro_dir)
        return (test_proc, launch_proc, repro_proc), (launch_code, repro_code)

    def _run_full_test_nocode(self, run_code, repro_after, repro_level, patch_code):
        tbs, _ = self._run_full_test(run_code, repro_after, repro_level, patch_code)
        return tbs
