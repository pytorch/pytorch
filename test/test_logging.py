# Owner(s): ["module: unknown"]

import glob
import gzip
import logging
import os
import subprocess
import sys
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
    def test_backend_autoload_registers_torch_logs_before_env_parse(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            package_dir = os.path.join(tmpdir, "torch_issue173759_backend")
            dist_info_dir = os.path.join(
                tmpdir, "torch_issue173759_backend-0.0.0.dist-info"
            )
            os.mkdir(package_dir)
            os.mkdir(dist_info_dir)

            with open(os.path.join(package_dir, "__init__.py"), "w") as f:
                f.write(
                    "import logging\n"
                    "from torch._logging._internal import "
                    "getArtifactLogger, register_artifact, register_log\n"
                    "register_log('issue173759_component', "
                    "'torch_issue173759_backend')\n"
                    "register_artifact('issue173759_artifact', "
                    "'test artifact')\n"
                    "def _autoload():\n"
                    "    pass\n"
                    "def emit_logs():\n"
                    "    logging.getLogger(__name__).info("
                    "'issue173759 component log')\n"
                    "    getArtifactLogger("
                    "__name__, 'issue173759_artifact').info("
                    "'issue173759 artifact log')\n"
                )

            with open(os.path.join(dist_info_dir, "METADATA"), "w") as f:
                f.write(
                    "Metadata-Version: 2.1\n"
                    "Name: torch-issue173759-backend\n"
                    "Version: 0.0.0\n"
                )

            with open(os.path.join(dist_info_dir, "entry_points.txt"), "w") as f:
                f.write(
                    "[torch.backends]\n"
                    "issue173759 = torch_issue173759_backend:_autoload\n"
                )

            env = os.environ.copy()
            env["PYTHONPATH"] = os.pathsep.join(
                filter(None, [tmpdir, env.get("PYTHONPATH", "")])
            )
            env["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "1"
            env["TORCH_LOGS"] = "+issue173759_component,+issue173759_artifact"
            env["TORCH_LOGS_FORMAT"] = "%(message)s"

            pytorch_root = os.path.dirname(os.path.dirname(torch.__file__))
            result = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    "import torch; "
                    "from torch_issue173759_backend import emit_logs; "
                    "emit_logs()",
                ],
                cwd=pytorch_root,
                env=env,
                capture_output=True,
                text=True,
            )

            self.assertEqual(
                result.returncode,
                0,
                lambda msg: f"{msg}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}",
            )
            self.assertIn("issue173759 component log", result.stderr)
            self.assertIn("issue173759 artifact log", result.stderr)

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

    def test_init_logs_preserves_active_trace_handler_stream(self):
        old_env = os.environ.get("TORCH_TRACE")
        old_handler = log_internal.LOG_TRACE_HANDLER
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                os.environ["TORCH_TRACE"] = tmpdir
                log_internal.LOG_TRACE_HANDLER = None
                _init_logs()

                trace_structured(
                    "before_reinit",
                    metadata_fn=dict,
                    expect_trace_id=False,
                    record_logging_overhead=False,
                )
                handler = log_internal.LOG_TRACE_HANDLER
                handler.flush()
                stream_path = handler._stream_path
                self.assertIsNotNone(stream_path)

                _init_logs()
                self.assertIs(log_internal.LOG_TRACE_HANDLER, handler)

                trace_structured(
                    "after_reinit",
                    metadata_fn=dict,
                    expect_trace_id=False,
                    record_logging_overhead=False,
                )
                handler.flush()
                self.assertEqual(handler._stream_path, stream_path)

                log_files = glob.glob(os.path.join(tmpdir, "*.log"))
                self.assertEqual(len(log_files), 1)
                content = Path(stream_path).read_text()
                self.assertIn("before_reinit", content)
                self.assertIn("after_reinit", content)

                handler.close()
        finally:
            if old_env is None:
                os.environ.pop("TORCH_TRACE", None)
            else:
                os.environ["TORCH_TRACE"] = old_env
            log_internal.LOG_TRACE_HANDLER = old_handler
            _init_logs()

    def test_init_logs_removes_marked_handlers_if_weak_registry_is_reset(self):
        old_logs = os.environ.get(log_internal.LOG_ENV_VAR)
        old_logs_out = os.environ.get(log_internal.LOG_OUT_ENV_VAR)
        old_log_state = log_internal._get_log_state()
        created_handlers = set()

        def close_created_handlers():
            for handler in created_handlers:
                handler.close()
            created_handlers.clear()

        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                try:
                    log_path = os.path.join(tmpdir, "torch.log")
                    os.environ[log_internal.LOG_ENV_VAR] = "+dynamic"
                    os.environ[log_internal.LOG_OUT_ENV_VAR] = log_path
                    log_internal._set_log_state(log_internal.LogState())

                    _init_logs()
                    for log_qname in log_internal.log_registry.get_log_qnames():
                        created_handlers.update(logging.getLogger(log_qname).handlers)

                    log_internal.handlers.clear()
                    _init_logs()
                    for log_qname in log_internal.log_registry.get_log_qnames():
                        logger = logging.getLogger(log_qname)
                        created_handlers.update(logger.handlers)
                        self.assertEqual(
                            len(logger.handlers),
                            2,
                            f"{log_qname} should only have stream and file handlers",
                        )
                finally:
                    close_created_handlers()
        finally:
            if old_logs is None:
                os.environ.pop(log_internal.LOG_ENV_VAR, None)
            else:
                os.environ[log_internal.LOG_ENV_VAR] = old_logs
            if old_logs_out is None:
                os.environ.pop(log_internal.LOG_OUT_ENV_VAR, None)
            else:
                os.environ[log_internal.LOG_OUT_ENV_VAR] = old_logs_out
            log_internal._set_log_state(old_log_state)
            _init_logs()
            close_created_handlers()

    def test_trace_handler_close_stream_can_skip_flush(self):
        paths: list[Path] = []

        def make_handler() -> tuple[log_internal.LazyTraceHandler, Path]:
            fd, path = tempfile.mkstemp()
            os.close(fd)
            paths.append(Path(path))

            handler = log_internal.LazyTraceHandler(None)
            handler.stream = open(path, "w+")  # noqa: SIM115
            handler._pid = os.getpid()
            handler._pending_log_version = True
            return handler, Path(path)

        try:
            handler, path = make_handler()
            handler.stream.write("discarded parent buffer")
            handler._close_stream(flush=False)

            self.assertEqual(path.read_text(), "")
            self.assertIsNone(handler.stream)
            self.assertIsNone(handler._pid)
            self.assertFalse(handler._pending_log_version)

            handler, path = make_handler()
            handler.stream.write("flushed parent buffer")
            handler._close_stream()

            self.assertEqual(path.read_text(), "flushed parent buffer")
            self.assertIsNone(handler.stream)
        finally:
            for path in paths:
                path.unlink(missing_ok=True)

    @unittest.skipIf(not hasattr(os, "fork"), "requires os.fork")
    def test_trace_handler_close_after_fork_skips_flush(self):
        fd, path_str = tempfile.mkstemp()
        path = Path(path_str)
        child_pid = None

        handler = log_internal.LazyTraceHandler(None)
        handler.stream = os.fdopen(fd, "w+")
        handler._pid = os.getpid()
        handler._pending_log_version = True
        try:
            handler.stream.write("discarded parent buffer")

            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="This process .* is multi-threaded, use of fork\\(\\) may lead to deadlocks in the child.",
                    category=DeprecationWarning,
                )
                child_pid = os.fork()
            if child_pid == 0:
                try:
                    handler.close()
                except BaseException:
                    os._exit(1)
                os._exit(0)

            _, status = os.waitpid(child_pid, 0)
            child_pid = None
            self.assertTrue(os.WIFEXITED(status))
            self.assertEqual(os.WEXITSTATUS(status), 0)

            self.assertEqual(path.read_text(), "")
        finally:
            if child_pid is not None:
                try:
                    os.waitpid(child_pid, 0)
                except ChildProcessError:
                    pass
            handler._close_stream(flush=False)
            path.unlink(missing_ok=True)

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

    @unittest.skipIf(
        sys.platform == "win32", "tests a bash CI helper with POSIX path semantics"
    )
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
