# mypy: allow-untyped-defs
import dataclasses
import io
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
import traceback
from typing import Optional
from unittest.mock import patch

import torch
import torch._dynamo
import torch._dynamo.test_case
from torch._dynamo.trace_rules import _as_posix_path
from torch.utils._traceback import report_compile_source_on_error


@dataclasses.dataclass
class MinifierTestResult:
    minifier_code: str
    repro_code: str

    def _get_module(self, t):
        match = re.search(r"class Repro\(torch\.nn\.Module\):\s+([ ].*\n| *\n)+", t)
        assert match is not None, "failed to find module"
        r = match.group(0)
        r = re.sub(r"\s+$", "\n", r, flags=re.MULTILINE)
        r = re.sub(r"\n{3,}", "\n\n", r)
        return r.strip()

    def get_exported_program_path(self):
        # Extract the exported program file path from AOTI minifier's repro.py
        # Regular expression pattern to match the file path
        pattern = r'torch\.export\.load\(\s*["\'](.*?)["\']\s*\)'
        # Search for the pattern in the text
        match = re.search(pattern, self.repro_code)
        # Extract and print the file path if a match is found
        if match:
            file_path = match.group(1)
            return file_path
        return None

    def minifier_module(self):
        return self._get_module(self.minifier_code)

    def repro_module(self):
        return self._get_module(self.repro_code)


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

    def _gen_codegen_fn_patch_code(self, device, bug_type):
        assert bug_type in ("compile_error", "runtime_error", "accuracy")
        return f"""\
{torch._dynamo.config.codegen_config()}
{torch._inductor.config.codegen_config()}
torch._inductor.config.{"cpp" if device == "cpu" else "triton"}.inject_relu_bug_TESTING_ONLY = {bug_type!r}
"""

    def _maybe_subprocess_run(self, args, *, isolate, cwd=None):
        if not isolate:
            assert len(args) >= 2, args
            assert args[0] == "python3", args
            if args[1] == "-c":
                assert len(args) == 3, args
                code = args[2]
                args = ["-c"]
            else:
                assert len(args) >= 2, args
                with open(args[1]) as f:
                    code = f.read()
                args = args[1:]

            # WARNING: This is not a perfect simulation of running
            # the program out of tree.  We only interpose on things we KNOW we
            # need to handle for tests.  If you need more stuff, you will
            # need to augment this appropriately.

            # NB: Can't use save_config because that will omit some fields,
            # but we must save and reset ALL fields
            dynamo_config = torch._dynamo.config.get_config_copy()
            inductor_config = torch._inductor.config.get_config_copy()
            try:
                stderr = io.StringIO()
                log_handler = logging.StreamHandler(stderr)
                log = logging.getLogger("torch._dynamo")
                log.addHandler(log_handler)
                try:
                    prev_cwd = _as_posix_path(os.getcwd())
                    if cwd is not None:
                        cwd = _as_posix_path(cwd)
                        os.chdir(cwd)
                    with patch("sys.argv", args), report_compile_source_on_error():
                        exec(code, {"__name__": "__main__", "__compile_source__": code})
                    rc = 0
                except Exception:
                    rc = 1
                    traceback.print_exc(file=stderr)
                finally:
                    log.removeHandler(log_handler)
                    if cwd is not None:
                        os.chdir(prev_cwd)  # type: ignore[possibly-undefined]
                    # Make sure we don't leave buggy compiled frames lying
                    # around
                    torch._dynamo.reset()
            finally:
                torch._dynamo.config.load_config(dynamo_config)
                torch._inductor.config.load_config(inductor_config)

            # TODO: return a more appropriate data structure here
            return subprocess.CompletedProcess(
                args,
                rc,
                b"",
                stderr.getvalue().encode("utf-8"),
            )
        else:
            if cwd is not None:
                cwd = _as_posix_path(cwd)
            return subprocess.run(args, capture_output=True, cwd=cwd, check=False)

    # Run `code` in a separate python process.
    # Returns the completed process state and the directory containing the
    # minifier launcher script, if `code` outputted it.
    def _run_test_code(self, code, *, isolate):
        proc = self._maybe_subprocess_run(
            ["python3", "-c", code], isolate=isolate, cwd=self.DEBUG_DIR
        )

        print("test stdout:", proc.stdout.decode("utf-8"))
        print("test stderr:", proc.stderr.decode("utf-8"))
        repro_dir_match = re.search(
            r"(\S+)minifier_launcher.py", proc.stderr.decode("utf-8")
        )
        if repro_dir_match is not None:
            return proc, repro_dir_match.group(1)
        return proc, None

    # Runs the minifier launcher script in `repro_dir`
    def _run_minifier_launcher(self, repro_dir, isolate, *, minifier_args=()):
        self.assertIsNotNone(repro_dir)
        launch_file = _as_posix_path(os.path.join(repro_dir, "minifier_launcher.py"))
        with open(launch_file) as f:
            launch_code = f.read()
        self.assertTrue(os.path.exists(launch_file))

        args = ["python3", launch_file, "minify", *minifier_args]
        if not isolate:
            args.append("--no-isolate")
        launch_proc = self._maybe_subprocess_run(args, isolate=isolate, cwd=repro_dir)
        print("minifier stdout:", launch_proc.stdout.decode("utf-8"))
        stderr = launch_proc.stderr.decode("utf-8")
        print("minifier stderr:", stderr)
        self.assertNotIn("Input graph did not fail the tester", stderr)

        return launch_proc, launch_code

    # Runs the repro script in `repro_dir`
    def _run_repro(self, repro_dir, *, isolate=True):
        self.assertIsNotNone(repro_dir)
        repro_file = _as_posix_path(os.path.join(repro_dir, "repro.py"))
        with open(repro_file) as f:
            repro_code = f.read()
        self.assertTrue(os.path.exists(repro_file))

        repro_proc = self._maybe_subprocess_run(
            ["python3", repro_file], isolate=isolate, cwd=repro_dir
        )
        print("repro stdout:", repro_proc.stdout.decode("utf-8"))
        print("repro stderr:", repro_proc.stderr.decode("utf-8"))
        return repro_proc, repro_code

    # Template for testing code.
    # `run_code` is the code to run for the test case.
    # `patch_code` is the code to be patched in every generated file; usually
    # just use this to turn on bugs via the config
    def _gen_test_code(self, run_code, repro_after, repro_level):
        repro_after_line = (
            f"""\
torch._dynamo.config.repro_after = "{repro_after}"
"""
            if repro_after
            else ""
        )
        return f"""\
import torch
import torch._dynamo
{_as_posix_path(torch._dynamo.config.codegen_config())}
{_as_posix_path(torch._inductor.config.codegen_config())}
{repro_after_line}
torch._dynamo.config.repro_level = {repro_level}
torch._dynamo.config.debug_dir_root = "{_as_posix_path(self.DEBUG_DIR)}"
{run_code}
"""

    # Runs a full minifier test.
    # Minifier tests generally consist of 3 stages:
    # 1. Run the problematic code
    # 2. Run the generated minifier launcher script
    # 3. Run the generated repro script
    #
    # If possible, you should run the test with isolate=False; use
    # isolate=True only if the bug you're testing would otherwise
    # crash the process
    def _run_full_test(
        self, run_code, repro_after, expected_error, *, isolate, minifier_args=()
    ) -> Optional[MinifierTestResult]:
        if isolate:
            repro_level = 3
        elif expected_error is None or expected_error == "AccuracyError":
            repro_level = 4
        else:
            repro_level = 2
        test_code = self._gen_test_code(run_code, repro_after, repro_level)
        print("running test", file=sys.stderr)
        test_proc, repro_dir = self._run_test_code(test_code, isolate=isolate)
        if expected_error is None:
            # Just check that there was no error
            self.assertEqual(test_proc.returncode, 0)
            self.assertIsNone(repro_dir)
            return None
        # NB: Intentionally do not test return code; we only care about
        # actually generating the repro, we don't have to crash
        self.assertIn(expected_error, test_proc.stderr.decode("utf-8"))
        self.assertIsNotNone(repro_dir)
        print("running minifier", file=sys.stderr)
        _minifier_proc, minifier_code = self._run_minifier_launcher(
            repro_dir, isolate=isolate, minifier_args=minifier_args
        )
        print("running repro", file=sys.stderr)
        repro_proc, repro_code = self._run_repro(repro_dir, isolate=isolate)
        self.assertIn(expected_error, repro_proc.stderr.decode("utf-8"))
        self.assertNotEqual(repro_proc.returncode, 0)
        return MinifierTestResult(minifier_code=minifier_code, repro_code=repro_code)
