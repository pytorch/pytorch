# Owner(s): ["oncall: export"]

import os
import subprocess
import sys
import tempfile
import textwrap

import torch
from torch.testing._internal.common_utils import run_tests, TestCase


class TestExportLogging(TestCase):
    def _check_non_strict_trace_has_compile_context(self, export_call):
        script = textwrap.dedent(
            f"""
            import glob
            import os
            import torch

            class M(torch.nn.Module):
                def forward(self, x):
                    return torch.sin(x) + 1

            {export_call}

            log_files = glob.glob(os.path.join(os.environ["TORCH_TRACE"], "*.log"))
            if len(log_files) != 1:
                raise AssertionError(f"Expected one trace log, got {{log_files}}")

            with open(log_files[0]) as f:
                describe_tensor_lines = [
                    line for line in f if '"describe_tensor"' in line
                ]

            if not describe_tensor_lines:
                raise AssertionError("Expected describe_tensor records in trace log")
            if not any('"frame_id"' in line for line in describe_tensor_lines):
                raise AssertionError(
                    "Expected non-strict export trace records to have a frame_id"
                )
            if any('"stack"' in line for line in describe_tensor_lines):
                raise AssertionError(
                    "Non-strict export trace records should use compile context, "
                    "not fallback stack metadata"
                )
            """
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            env = os.environ.copy()
            env["TORCH_TRACE"] = tmpdir
            pytorch_root = os.path.dirname(os.path.dirname(torch.__file__))
            result = subprocess.run(
                [sys.executable, "-c", script],
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

    def test_public_non_strict_export_trace_has_compile_context(self):
        self._check_non_strict_trace_has_compile_context(
            "torch.export.export(M(), (torch.randn(2, 3),), strict=False)"
        )

    def test_private_non_strict_export_trace_has_compile_context(self):
        self._check_non_strict_trace_has_compile_context(
            "import torch.export._trace as export_trace; "
            "export_trace._export(M(), (torch.randn(2, 3),), "
            "strict=False, pre_dispatch=False)"
        )


if __name__ == "__main__":
    run_tests()
