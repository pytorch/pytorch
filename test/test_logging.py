# Owner(s): ["module: unknown"]

import glob
import logging
import os
import tempfile

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


if __name__ == "__main__":
    run_tests()
