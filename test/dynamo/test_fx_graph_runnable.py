# Owner(s): ["module: dynamo"]
import io
import logging
import subprocess
import sys
import tempfile
import unittest

import torch
import torch._logging.structured
import torch.distributed as dist
from torch._inductor.test_case import TestCase
from torch.testing._internal.common_utils import IS_FBCODE, IS_SANDCASTLE


if torch.distributed.is_available():
    from torch.distributed._tensor import DeviceMesh, DTensor, Replicate, Shard
    from torch.testing._internal.distributed.fake_pg import FakeStore


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


class ToyModel(torch.nn.Module):
    def __init__(self, input_size=10, hidden_size=20, output_size=5):
        super().__init__()
        self.linear1 = torch.nn.Linear(input_size, hidden_size)
        self.linear2 = torch.nn.Linear(hidden_size, output_size)
        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(0.1)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


@unittest.skipIf(IS_FBCODE or IS_SANDCASTLE, "Skip in fbcode/sandcastle")
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

    # testing dynamic shapes
    def test_dynamic_shapes_run(self):
        def f(x):
            return (x @ x.transpose(0, 1)).relu()

        a = torch.randn(10, 12)
        torch._dynamo.mark_dynamic(a, 0)
        torch._dynamo.mark_dynamic(a, 1)

        torch.compile(f)(a)
        self._exec_and_verify_payload()

    def test_broadcast_add_dynamic(self):
        def f(x, y):
            return x + y * 2

        x = torch.randn(5, 1)
        y = torch.randn(1, 8)
        torch._dynamo.mark_dynamic(x, 0)
        torch._dynamo.mark_dynamic(y, 1)

        torch.compile(f)(x, y)
        self._exec_and_verify_payload()

    def test_toy_model_basic(self):
        model = ToyModel(input_size=8, hidden_size=16, output_size=4)
        model.eval()  # Set to eval mode to avoid dropout randomness

        x = torch.randn(3, 8)
        torch.compile(model)(x)
        self._exec_and_verify_payload()

    def test_toy_model_batch_processing(self):
        model = ToyModel(input_size=12, hidden_size=24, output_size=6)
        model.eval()

        x = torch.randn(16, 12)
        torch.compile(model)(x)
        self._exec_and_verify_payload()

    def test_toy_model_dynamic_batch(self):
        model = ToyModel(input_size=10, hidden_size=20, output_size=5)
        model.eval()

        x = torch.randn(7, 10)
        torch._dynamo.mark_dynamic(x, 0)

        torch.compile(model)(x)
        self._exec_and_verify_payload()

    # Distributed collectives tests with FakeProcessGroup
    @unittest.skipIf(
        not torch.distributed.is_available(), "Torch distributed not available."
    )
    @unittest.skipIf(IS_FBCODE or IS_SANDCASTLE, "Skip in fbcode/sandcastle")
    def test_all_reduce_collective(self):
        store = FakeStore()
        dist.init_process_group(backend="fake", rank=0, world_size=2, store=store)

        def f(x):
            dist.all_reduce(x)
            return x * 2

        try:
            x = torch.randn(4, 4)
            torch.compile(f)(x)
        finally:
            dist.destroy_process_group()

        self._exec_and_verify_payload()

    @unittest.skipIf(
        not torch.distributed.is_available(), "Torch distributed not available."
    )
    @unittest.skipIf(IS_FBCODE or IS_SANDCASTLE, "Skip in fbcode/sandcastle")
    def test_all_gather_collective(self):
        store = FakeStore()
        dist.init_process_group(backend="fake", rank=0, world_size=2, store=store)

        def f(x):
            output_tensors = [torch.empty_like(x) for _ in range(2)]
            dist.all_gather(output_tensors, x)
            return output_tensors[0] + output_tensors[1]

        try:
            x = torch.randn(3, 3)
            torch.compile(f)(x)
        finally:
            dist.destroy_process_group()

        self._exec_and_verify_payload()

    @unittest.skipIf(
        not torch.distributed.is_available(), "Torch distributed not available."
    )
    @unittest.skipIf(IS_FBCODE or IS_SANDCASTLE, "Skip in fbcode/sandcastle")
    def test_broadcast_collective(self):
        store = FakeStore()
        dist.init_process_group(backend="fake", rank=0, world_size=2, store=store)

        def f(x):
            dist.broadcast(x, src=0)
            return x.sum()

        try:
            x = torch.randn(5, 5)
            torch.compile(f)(x)
        finally:
            dist.destroy_process_group()

        self._exec_and_verify_payload()

    @unittest.skipIf(
        not torch.distributed.is_available(), "Torch distributed not available."
    )
    @unittest.skipIf(IS_FBCODE or IS_SANDCASTLE, "Skip in fbcode/sandcastle")
    def test_reduce_scatter_collective(self):
        store = FakeStore()
        dist.init_process_group(backend="fake", rank=0, world_size=2, store=store)

        def f(x):
            input_list = [x, x.clone()]
            output = torch.empty_like(x)
            dist.reduce_scatter(output, input_list)
            return output

        try:
            x = torch.randn(4, 4)
            torch.compile(f)(x)
        finally:
            dist.destroy_process_group()

        self._exec_and_verify_payload()

    @unittest.skipIf(
        not torch.distributed.is_available(), "Torch distributed not available"
    )
    @unittest.skipIf(IS_FBCODE or IS_SANDCASTLE, "Skip in fbcode/sandcastle")
    def test_dtensor_compile_redistribute(self):
        store = FakeStore()
        dist.init_process_group(backend="fake", rank=0, world_size=2, store=store)

        mesh = DeviceMesh("cpu", list(range(2)))

        def f(x, y):
            dt = DTensor.from_local(x.reshape(2, 4), mesh, [Shard(0)], run_check=False)
            dt2 = DTensor.from_local(y.reshape(4, 2), mesh, [Shard(1)], run_check=False)
            dt_out = torch.matmul(dt, dt2)
            dt_out_redistribute = dt_out.redistribute(mesh, [Replicate()])
            return dt_out_redistribute.to_local()

        try:
            x = torch.arange(8, dtype=torch.float32)
            y = torch.arange(8, dtype=torch.float32)
            torch.compile(f)(x, y)
        finally:
            dist.destroy_process_group()

        self._exec_and_verify_payload()


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    if not (IS_FBCODE or IS_SANDCASTLE):
        # fbcode complains about not being able to find torch in subprocess
        run_tests()
