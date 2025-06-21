# Owner(s): ["module: dynamo"]
import io
import logging
import subprocess
import sys
import tempfile

import torch
import torch._logging.structured
from torch._inductor.test_case import TestCase


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

    def test_basic(self):
        # Compile and run a simple function to generate fx_graph_runnable entries
        def simple_fn(x):
            return x.add(torch.ones_like(x))

        fn_opt = torch.compile(simple_fn)
        fn_opt(torch.ones(10, 10))

        # Extract the payload from fx_graph_runnable entries
        fx_graph_runnable_payload = self.buffer.getvalue().strip()

        # Verify that we captured fx_graph_runnable payload
        self.assertTrue(
            len(fx_graph_runnable_payload) > 0,
            "Should have captured fx_graph_runnable payload",
        )

        # The payload should contain FX graph code
        self.assertIn(
            "def forward",
            fx_graph_runnable_payload,
            "Payload should contain FX graph forward method",
        )

        # Run the fx_graph_runnable_payload directly in a subprocess since it's self-contained
        # Write the payload directly to a temporary file and execute it
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py") as temp_file:
            temp_file.write(fx_graph_runnable_payload)
            temp_file.flush()  # Ensure content is written to disk

            # Run the payload directly in a subprocess
            result = subprocess.run(
                [sys.executable, temp_file.name],
                capture_output=True,
                text=True,
                timeout=30,
            )

            # Verify the subprocess executed successfully (payload is self-contained)
            self.assertEqual(
                result.returncode, 0, f"Subprocess failed with error: {result.stderr}"
            )


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
