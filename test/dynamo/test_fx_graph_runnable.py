# Owner(s): ["module: dynamo"]
import io
import logging
import subprocess
import sys
import tempfile

import torch
import torch._logging.structured
from torch._inductor.test_case import TestCase
from torch.testing._internal.common_utils import IS_FBCODE


class FxGraphRunnableArtifactFilter(logging.Filter):
    def filter(self, record):
        return (
            "artifact" in record.metadata
            and record.metadata["artifact"]["name"] == "fx_graph_runnable"
        )


class StructuredTracePayloadFormatter(logging.Formatter):
    def format(self, record):
        return record.payload.strip()


trace_log = logging.getLogger("torch.__trace")


class FxGraphRunnableTest(TestCase):
    def setUp(self):
        super().setUp()
        torch._dynamo.reset()
        torch._logging.structured.INTERN_TABLE.clear()
        self.old_level = trace_log.level
        trace_log.setLevel(logging.DEBUG)

        # Create a custom filter specifically for fx_graph_runnable entries
        self.filter = FxGraphRunnableArtifactFilter()

        # Create a separate buffer and handler for capturing fx_graph_runnable entries
        self.buffer = io.StringIO()
        self.handler = logging.StreamHandler(self.buffer)
        self.handler.setFormatter(StructuredTracePayloadFormatter())
        self.handler.addFilter(self.filter)
        trace_log.addHandler(self.handler)

    def tearDown(self):
        trace_log.removeHandler(self.handler)
        trace_log.setLevel(self.old_level)

    def _exec_and_verify_payload(self):
        # Write captured payload & run it in a fresh Python process
        payload = self.buffer.getvalue().strip()
        self.assertTrue(payload, "Expected fx_graph_runnable payload but got nothing")
        self.assertIn("def forward", payload)  # sanity-check for actual FX code

        with tempfile.NamedTemporaryFile("w", suffix=".py") as tmp:
            tmp.write(payload)
            tmp.flush()
            res = subprocess.run(
                [sys.executable, tmp.name], capture_output=True, text=True, timeout=30
            )
            self.assertEqual(
                res.returncode,
                0,
                f"Standalone fx_graph_runnable failed:\nSTDERR:\n{res.stderr}",
            )

    # basic tests
    def test_basic_tensor_add(self):
        def f(x):
            return x + 1

        torch.compile(f)(torch.randn(4))
        self._exec_and_verify_payload()

    def test_two_inputs_matmul(self):
        def f(a, b):
            return (a @ b).relu()

        a, b = torch.randn(2, 3), torch.randn(3, 4)
        torch.compile(f)(a, b)
        self._exec_and_verify_payload()

    def test_scalar_multiply(self):
        def f(x):
            return x * 2

        torch.compile(f)(torch.randn(5))
        self._exec_and_verify_payload()


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    if not IS_FBCODE:
        # fbcode complains about not being able to find torch in subprocess
        run_tests()
