# Owner(s): ["module: unknown"]

import glob
import logging
import os
import subprocess
import sys
import tempfile

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
                f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}",
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


if __name__ == "__main__":
    run_tests()
