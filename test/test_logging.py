# Owner(s): ["module: unknown"]

import glob
import gzip
import logging
import os
import subprocess
import tempfile
import textwrap
import unittest
import warnings
from pathlib import Path

import torch
import torch._logging._internal as log_internal
from torch._logging._internal import _init_logs, trace_log, trace_structured
from torch.testing._internal.common_utils import run_tests, TestCase


class LoggingTest(TestCase):
    def testApiUsage(self):
        """
        This test verifies that api usage logging is not triggered via static
        initialization. Since it's triggered at first invocation only - we just
        subprocess
        """
        s = TestCase.runWithPytorchAPIUsageStderr("import torch")
        self.assertRegex(s, "PYTORCH_API_USAGE.*import")
        # import the shared library directly - it triggers static init but doesn't call anything
        s = TestCase.runWithPytorchAPIUsageStderr(
            f"from ctypes import CDLL; CDLL('{torch._C.__file__}')"
        )
        self.assertNotRegex(s, "PYTORCH_API_USAGE")

    def test_trace_structured_with_logging_disable(self):
        """trace_structured should work even when logging.disable(DEBUG) is active."""
        old_disable = logging.root.manager.disable
        old_env = os.environ.get("TORCH_TRACE")
        old_handler = log_internal.LOG_TRACE_HANDLER
        try:
            logging.disable(logging.DEBUG)

            with tempfile.TemporaryDirectory() as tmpdir:
                os.environ["TORCH_TRACE"] = tmpdir
                log_internal.LOG_TRACE_HANDLER = None
                _init_logs()

                self.assertTrue(trace_log.handlers)
                self.assertTrue(trace_log.isEnabledFor(logging.DEBUG))

                trace_structured(
                    "test_event",
                    metadata_fn=lambda: {"key": "value"},
                    expect_trace_id=False,
                    record_logging_overhead=False,
                )

                log_files = glob.glob(os.path.join(tmpdir, "*.log"))
                self.assertTrue(log_files, "Expected a trace log file to be created")
                with open(log_files[0]) as f:
                    content = f.read()
                self.assertIn("test_event", content)

                # Close the handler before the TemporaryDirectory is cleaned up,
                # otherwise Windows fails with a PermissionError on the open file.
                log_internal.LOG_TRACE_HANDLER.close()
        finally:
            logging.disable(old_disable)
            if old_env is None:
                os.environ.pop("TORCH_TRACE", None)
            else:
                os.environ["TORCH_TRACE"] = old_env
            log_internal.LOG_TRACE_HANDLER = old_handler
            _init_logs()

    @unittest.skipIf(not hasattr(os, "fork"), "requires os.fork")
    def test_trace_handler_reopens_after_fork(self):
        old_env = os.environ.get("TORCH_TRACE")
        old_handler = log_internal.LOG_TRACE_HANDLER
        child_pid = None
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                os.environ["TORCH_TRACE"] = tmpdir
                log_internal.LOG_TRACE_HANDLER = None
                _init_logs()

                trace_structured(
                    "pre_fork",
                    metadata_fn=lambda: {"pid": os.getpid()},
                    expect_trace_id=False,
                    record_logging_overhead=False,
                )
                log_internal.LOG_TRACE_HANDLER.flush()

                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore",
                        message="This process .* is multi-threaded, use of fork\\(\\) may lead to deadlocks in the child.",
                        category=DeprecationWarning,
                    )
                    child_pid = os.fork()
                if child_pid == 0:
                    try:
                        trace_structured(
                            "child_event",
                            metadata_fn=lambda: {"pid": os.getpid()},
                            expect_trace_id=False,
                            record_logging_overhead=False,
                        )
                        log_internal.LOG_TRACE_HANDLER.flush()
                        log_internal.LOG_TRACE_HANDLER.close()
                    except BaseException:
                        os._exit(1)
                    os._exit(0)

                _, status = os.waitpid(child_pid, 0)
                child_pid = None
                self.assertTrue(os.WIFEXITED(status))
                self.assertEqual(os.WEXITSTATUS(status), 0)

                trace_structured(
                    "parent_event",
                    metadata_fn=lambda: {"pid": os.getpid()},
                    expect_trace_id=False,
                    record_logging_overhead=False,
                )
                log_internal.LOG_TRACE_HANDLER.flush()

                log_files = sorted(glob.glob(os.path.join(tmpdir, "*.log")))
                self.assertEqual(len(log_files), 2)

                contents = [Path(path).read_text() for path in log_files]
                child_logs = [
                    content for content in contents if "child_event" in content
                ]
                parent_logs = [
                    content for content in contents if "parent_event" in content
                ]

                self.assertEqual(len(child_logs), 1)
                self.assertEqual(len(parent_logs), 1)
                self.assertNotIn("pre_fork", child_logs[0])
                self.assertNotIn("parent_event", child_logs[0])
                self.assertIn("pre_fork", parent_logs[0])
                self.assertNotIn("child_event", parent_logs[0])

                # Close the handler before the TemporaryDirectory is cleaned up,
                # otherwise Windows fails with a PermissionError on the open file.
                log_internal.LOG_TRACE_HANDLER.close()
        finally:
            if child_pid is not None:
                try:
                    os.waitpid(child_pid, 0)
                except ChildProcessError:
                    pass
            if old_env is None:
                os.environ.pop("TORCH_TRACE", None)
            else:
                os.environ["TORCH_TRACE"] = old_env
            log_internal.LOG_TRACE_HANDLER = old_handler
            _init_logs()

    def test_collect_tlparse_output_preserves_multiple_trace_logs(self):
        repo_root = Path(__file__).resolve().parent.parent

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            runner_temp = tmp_path / "runner"
            trace_dir = runner_temp / "torch_traces"
            trace_dir.mkdir(parents=True)
            (trace_dir / "first.log").write_text("first\n")
            (trace_dir / "second.log").write_text("second\n")

            workdir = tmp_path / "work"
            (workdir / "test").mkdir(parents=True)

            fake_bin = tmp_path / "bin"
            fake_bin.mkdir()
            args_log = tmp_path / "tlparse_args.txt"
            fake_tlparse = fake_bin / "tlparse"
            fake_tlparse.write_text(
                textwrap.dedent(
                    """\
                    #!/usr/bin/env bash
                    set -euo pipefail
                    {
                      echo CALL
                      for arg in "$@"; do
                        printf '%s\\n' "$arg"
                      done
                    } >> "$TLPARSE_ARGS_LOG"

                    out=""
                    source_arg=""
                    while [[ $# -gt 0 ]]; do
                      case "$1" in
                        -o)
                          out="$2"
                          shift 2
                          ;;
                        --overwrite|--no-browser)
                          shift
                          ;;
                        *)
                          source_arg="$1"
                          shift
                          ;;
                      esac
                    done

                    if [[ -z "$out" || -z "$source_arg" ]]; then
                      exit 2
                    fi

                    rm -rf "$out"
                    mkdir -p "$out"
                    printf '%s\\n' "$source_arg" > "$out/source.txt"
                    """
                )
            )
            fake_tlparse.chmod(0o755)

            env = os.environ.copy()
            env.update(
                {
                    "ENABLE_TORCH_TRACE": "1",
                    "RUNNER_TEMP": str(runner_temp),
                    "PATH": f"{fake_bin}{os.pathsep}{env['PATH']}",
                    "TLPARSE_ARGS_LOG": str(args_log),
                    "TORCH_TRACE_HELPER": str(repo_root / ".ci/pytorch/torch_trace.sh"),
                }
            )

            completed = subprocess.run(
                [
                    "bash",
                    "-c",
                    'set -euo pipefail; source "$TORCH_TRACE_HELPER"; collect_tlparse_output',
                ],
                cwd=workdir,
                env=env,
                text=True,
                capture_output=True,
            )
            self.assertEqual(
                completed.returncode, 0, completed.stdout + completed.stderr
            )

            output_dir = workdir / "test/test-reports/tlparse_output"
            raw_dir = output_dir / "raw"
            self.assertTrue((raw_dir / "first.log.gz").is_file())
            self.assertTrue((raw_dir / "second.log.gz").is_file())

            with gzip.open(raw_dir / "first.log.gz", "rt") as f:
                self.assertEqual(f.read(), "first\n")
            with gzip.open(raw_dir / "second.log.gz", "rt") as f:
                self.assertEqual(f.read(), "second\n")

            self.assertEqual(
                (output_dir / "parsed/source.txt").read_text().strip(),
                str(trace_dir),
            )

            tlparse_args = args_log.read_text().splitlines()
            self.assertEqual(tlparse_args.count("CALL"), 1)
            self.assertIn(str(trace_dir), tlparse_args)
            self.assertNotIn(str(trace_dir / "first.log"), tlparse_args)
            self.assertNotIn(str(trace_dir / "second.log"), tlparse_args)


if __name__ == "__main__":
    run_tests()
