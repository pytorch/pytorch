# Owner(s): ["oncall: distributed"]

import contextlib
import io
import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
import benchmarks.distributed.transport.benchmark as benchmark


sys.path.remove(str(REPO_ROOT))

from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TestCase,
)


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
        source, destination, read_target = benchmark._buffers(4, 2, torch.device("cpu"))
        expected = torch.full((4,), 3, dtype=torch.uint8)
        self.assertEqual(source, expected)
        self.assertEqual(destination, torch.zeros(4, dtype=torch.uint8))
        self.assertEqual(read_target, torch.zeros(4, dtype=torch.uint8))
        self.assertTrue(benchmark._connects(0, True))
        self.assertFalse(benchmark._connects(1, True))
        self.assertTrue(benchmark._connects(1, False))
        self.assertTrue(benchmark._connects(2, True))
        self.assertFalse(benchmark._connects(3, True))

    @parametrize("world_size", [2, 4, 8])
    def test_rank_options(self, world_size):
        options = [{"device_name": f"mlx5_{rank}"} for rank in range(world_size)]
        self.assertEqual(
            benchmark._rank_options(json.dumps(options), world_size), options
        )
        self.assertEqual(
            benchmark._rank_options('{"plugin":"UCX"}', world_size),
            [{"plugin": "UCX"}] * world_size,
        )
        with self.assertRaisesRegex(ValueError, "one object per rank"):
            benchmark._rank_options(json.dumps(options[:-1]), world_size)

    def test_pair_exchange(self):
        def gather(values, value):
            values[:] = ["rank0", "rank1", value, "rank3"]

        with (
            patch.object(benchmark.dist, "get_world_size", return_value=4),
            patch.object(benchmark.dist, "get_rank", return_value=2),
            patch.object(benchmark.dist, "all_gather_object", side_effect=gather),
        ):
            self.assertEqual(benchmark._exchange("rank2"), "rank3")

    def test_aggregate_uses_elapsed_time(self):
        ranks = [
            {
                "seconds": seconds,
                "latency_us": 1 if rank % 2 == 0 else None,
                "before": (0, 0),
                "after": (1_000_000_000, 500_000_000),
            }
            for rank, seconds in enumerate([1.0, 1.1, 1.8, 2.0])
        ]
        result = benchmark._summarize(1_000_000, 1000, ranks)
        self.assertEqual(result["seconds"], 2)
        self.assertEqual(result["aggregate_bandwidth_gbps"], 8)
        self.assertEqual(result["pairs"][1]["ranks"], [2, 3])
        self.assertEqual(result["pairs"][1]["local_wire"], {"tx_gbps": 4, "rx_gbps": 2})

    def test_line_rate_checks_every_pair(self):
        pair = {
            "bandwidth_gbps": 390,
            "local_wire": {"tx_gbps": 390, "rx_gbps": 390},
            "peer_wire": {"tx_gbps": 390, "rx_gbps": 390},
        }
        measurement = {
            "devices": [{"line_rate_gbps": 400}] * 4,
            "write": {"aggregate_bandwidth_gbps": 780, "pairs": [pair, pair]},
            "read": {"aggregate_bandwidth_gbps": 780, "pairs": [pair, pair]},
        }
        benchmark._check_line_rate([measurement], 0.8)
        slow = dict(pair, peer_wire={"tx_gbps": 10, "rx_gbps": 10})
        measurement["write"]["pairs"] = [pair, slow]
        with self.assertRaisesRegex(RuntimeError, "below 80%"):
            benchmark._check_line_rate([measurement], 0.8)

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
                "--async-op",
            ]
        )
        self.assertEqual(args.tensor_device, "cpu")
        self.assertTrue(args.one_way_connect)
        self.assertTrue(args.rdma_counters)
        self.assertTrue(args.async_op)
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

        with (
            contextlib.redirect_stderr(io.StringIO()),
            self.assertRaises(SystemExit),
        ):
            benchmark.parse_args(["--backend", "ibverbs", "--async-op", "--cuda-graph"])

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


instantiate_parametrized_tests(TestTransportBenchmark)


if __name__ == "__main__":
    run_tests()
