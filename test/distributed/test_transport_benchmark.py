# Owner(s): ["oncall: distributed"]

import contextlib
import io
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import benchmarks.distributed.transport.benchmark as benchmark

from torch.testing._internal.common_utils import run_tests, TestCase


class TestTransportBenchmark(TestCase):
    def test_device(self):
        with patch.dict(os.environ, {"LOCAL_RANK": "3"}):
            self.assertEqual(str(benchmark._device("cuda")), "cuda:3")
            self.assertEqual(str(benchmark._device("cpu")), "cpu")

    def test_counter_source(self):
        with (
            patch.object(benchmark, "_rdma_wire_bytes", return_value=(1, 2)),
            patch.object(benchmark, "_netdev_wire_bytes", return_value=(3, 4)),
        ):
            self.assertEqual(
                benchmark._counter_source("eth0", "ibverbs", False), "rdma"
            )
            self.assertEqual(benchmark._counter_source("eth0", "ucxx", False), "netdev")
            self.assertEqual(benchmark._counter_source("eth0", "ucxx", True), "rdma")
            self.assertEqual(benchmark._wire_bytes("eth0", "rdma"), (1, 2))
            self.assertEqual(benchmark._wire_bytes("eth0", "netdev"), (3, 4))

        with patch.object(benchmark, "_rdma_wire_bytes", return_value=None):
            with self.assertRaisesRegex(RuntimeError, "RDMA counters are unavailable"):
                benchmark._counter_source("eth0", "ucxx", True)

        benchmark._validate_counter_sources("rdma", "rdma")
        with self.assertRaisesRegex(RuntimeError, "counter sources differ"):
            benchmark._validate_counter_sources("rdma", "netdev")

    def test_rdma_counter_port(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            interface = root / "eth0"
            counters = (
                interface
                / "device"
                / "infiniband"
                / "mlx5_0"
                / "ports"
                / "2"
                / "counters"
            )
            counters.mkdir(parents=True)
            (interface / "dev_port").write_text("1")
            (counters / "port_xmit_data").write_text("11")
            (counters / "port_rcv_data").write_text("13")
            with patch.object(benchmark, "_SYS_CLASS_NET", root):
                self.assertEqual(benchmark._rdma_wire_bytes("eth0"), (44, 52))

    def test_rank_decisions(self):
        source, destination, read_target = benchmark._buffers(
            4, 2, benchmark.torch.device("cpu")
        )
        expected = benchmark.torch.full((4,), 3, dtype=benchmark.torch.uint8)
        self.assertEqual(source, expected)
        self.assertEqual(
            destination, benchmark.torch.zeros(4, dtype=benchmark.torch.uint8)
        )
        self.assertEqual(
            read_target, benchmark.torch.zeros(4, dtype=benchmark.torch.uint8)
        )
        self.assertTrue(benchmark._connects(0, True))
        self.assertFalse(benchmark._connects(1, True))
        self.assertTrue(benchmark._connects(1, False))

    def test_parse_args(self):
        args = benchmark.parse_args(
            [
                "--backend",
                "UCXX",
                "--device",
                "cuda",
                "--tensor-device",
                "cpu",
                "--one-way-connect",
                "--rdma-counters",
            ]
        )
        self.assertEqual(args.tensor_device, "cpu")
        self.assertTrue(args.one_way_connect)
        self.assertTrue(args.rdma_counters)
        self.assertEqual(
            args.sizes,
            [
                8,
                64,
                256,
                1024,
                4096,
                16384,
                65536,
                262144,
                1048576,
                4194304,
                16777216,
                67108864,
            ],
        )

        with (
            contextlib.redirect_stderr(io.StringIO()),
            self.assertRaises(SystemExit),
        ):
            benchmark.parse_args(["--backend", "ibverbs", "--one-way-connect"])

    def test_output_metadata(self):
        args = benchmark.parse_args(
            [
                "--backend",
                "tcp",
                "--device",
                "cpu",
                "--interfaces",
                "eth0",
                "--minimum-line-rate",
                "0.5",
            ]
        )
        with (
            patch.dict(os.environ, {"RANK": "0"}),
            patch.object(benchmark, "_line_rate_gbps", return_value=400.0),
        ):
            output = benchmark._output(args, [], "netdev")
        self.assertEqual(output["device"], "cpu")
        self.assertEqual(output["tensor_device"], "cpu")
        self.assertEqual(output["counter_source"], "netdev")
        self.assertEqual(output["minimum_line_rate"], 0.5)


if __name__ == "__main__":
    run_tests()
