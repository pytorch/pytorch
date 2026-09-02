"""Measure a transport between two torchrun ranks."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from pathlib import Path
from typing import Any, TYPE_CHECKING

import torch
import torch.distributed as dist
from torch.distributed._transport import new_transport


if TYPE_CHECKING:
    from collections.abc import Callable


def _parse_sizes(value: str) -> list[int]:
    sizes = [int(size) for size in value.split(",")]
    if not sizes or any(size <= 0 for size in sizes):
        raise argparse.ArgumentTypeError("sizes must be positive")
    return sizes


def _device(args: argparse.Namespace) -> torch.device:
    if args.device == "cuda":
        return torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    return torch.device(args.device)


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _line_rate_gbps(interface: str | None) -> float | None:
    if interface is None:
        return None
    try:
        return int(Path(f"/sys/class/net/{interface}/speed").read_text()) / 1000
    except (OSError, ValueError):
        return None


def _interface(args: argparse.Namespace) -> str | None:
    if args.interfaces is None:
        return None
    interfaces = args.interfaces.split(",")
    if len(interfaces) not in (1, 2):
        raise ValueError("interfaces must name one interface or one per rank")
    rank = int(os.environ["RANK"])
    return interfaces[min(rank, len(interfaces) - 1)]


def _rdma_wire_bytes(interface: str) -> tuple[int, int] | None:
    devices = list(Path(f"/sys/class/net/{interface}/device/infiniband").glob("*"))
    if len(devices) != 1:
        return None
    counters = devices[0] / "ports/1/counters"
    try:
        return (
            int((counters / "port_xmit_data").read_text()) * 4,
            int((counters / "port_rcv_data").read_text()) * 4,
        )
    except (OSError, ValueError):
        return None


def _wire_bytes(interface: str | None, backend: str) -> tuple[int, int] | None:
    if interface is None:
        return None
    if (
        backend in {"ibverbs", "torchcomms"}
        and (counters := _rdma_wire_bytes(interface)) is not None
    ):
        return counters
    root = Path(f"/sys/class/net/{interface}/statistics")
    try:
        return int((root / "tx_bytes").read_text()), int(
            (root / "rx_bytes").read_text()
        )
    except (OSError, ValueError):
        return None


def _wire_rate(
    before: tuple[int, int] | None,
    after: tuple[int, int] | None,
    seconds: float,
) -> dict[str, float] | None:
    if before is None or after is None:
        return None
    return {
        "tx_gbps": (after[0] - before[0]) * 8 / seconds / 1e9,
        "rx_gbps": (after[1] - before[1]) * 8 / seconds / 1e9,
    }


def _exchange(value: Any) -> Any:
    values = [None, None]
    dist.all_gather_object(values, value)
    return values[1 - dist.get_rank()]


def run(args: argparse.Namespace) -> list[dict[str, Any]]:
    dist.init_process_group(
        "gloo",
        init_method=args.init_method,
        rank=int(os.environ["RANK"]),
        world_size=int(os.environ["WORLD_SIZE"]),
    )
    if dist.get_world_size() != 2:
        raise RuntimeError("transport benchmark requires exactly two ranks")
    rank = dist.get_rank()
    device = _device(args)
    if args.cuda_graph and device.type != "cuda":
        raise ValueError("--cuda-graph requires --device cuda")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    options = json.loads(args.options)
    if isinstance(options, list):
        if len(options) != 2:
            raise ValueError("rank-specific options must contain two objects")
        options = options[rank]
    if not isinstance(options, dict):
        raise ValueError("options must be an object or a two-element list")
    interface = _interface(args)
    transport = new_transport(args.backend, device, **options)
    results = []
    try:
        peer_url = _exchange(transport.bind())
        if transport.connect(peer_url) != 0:
            raise RuntimeError("transport connection failed")
        for size in args.sizes:
            source = torch.full((size,), rank + 1, dtype=torch.uint8, device=device)
            destination = torch.zeros_like(source)
            read_target = torch.zeros_like(source)
            source_memory = transport.register_memory(source)
            destination_memory = transport.register_memory(destination)
            read_memory = transport.register_memory(read_target)
            peer_source = _exchange(source_memory.to_remote_buffer())
            peer_destination = _exchange(destination_memory.to_remote_buffer())
            _sync(device)
            source_view = source_memory.to_view()
            read_view = read_memory.to_mutable_view()

            def write() -> None:
                if transport.write(source_view, peer_destination) != 0:
                    raise RuntimeError("write failed")

            def read() -> None:
                if transport.read(read_view, peer_source) != 0:
                    raise RuntimeError("read failed")

            for _ in range(args.warmup):
                if rank == 0:
                    write()
                    read()
            _sync(device)
            write_op: Callable[[], None] = write
            read_op: Callable[[], None] = read
            if rank == 0 and args.cuda_graph:
                write_graph = torch.cuda.CUDAGraph()
                read_graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(write_graph):
                    write()
                with torch.cuda.graph(read_graph):
                    read()
                write_op = write_graph.replay
                read_op = read_graph.replay
                _sync(device)
            source.fill_(rank + 3)
            destination.zero_()
            read_target.zero_()
            _sync(device)
            dist.barrier()
            write_wire_before = _wire_bytes(interface, args.backend)
            write_start = time.perf_counter_ns()
            write_samples = []
            read_samples = []
            if rank == 0:
                for _ in range(args.iterations):
                    start = time.perf_counter_ns()
                    write_op()
                    _sync(device)
                    write_samples.append(time.perf_counter_ns() - start)
            dist.barrier()
            write_seconds = (time.perf_counter_ns() - write_start) / 1e9
            write_wire = _wire_rate(
                write_wire_before,
                _wire_bytes(interface, args.backend),
                write_seconds,
            )
            peer_write_wire = _exchange(write_wire)

            read_wire_before = _wire_bytes(interface, args.backend)
            read_start = time.perf_counter_ns()
            if rank == 0:
                for _ in range(args.iterations):
                    start = time.perf_counter_ns()
                    read_op()
                    _sync(device)
                    read_samples.append(time.perf_counter_ns() - start)
            dist.barrier()
            read_seconds = (time.perf_counter_ns() - read_start) / 1e9
            read_wire = _wire_rate(
                read_wire_before,
                _wire_bytes(interface, args.backend),
                read_seconds,
            )
            peer_read_wire = _exchange(read_wire)
            _sync(device)
            if rank == 1:
                torch.testing.assert_close(destination, torch.full_like(destination, 3))
            if rank == 0:
                torch.testing.assert_close(read_target, torch.full_like(read_target, 4))
                write_seconds = statistics.median(write_samples) / 1e9
                read_seconds = statistics.median(read_samples) / 1e9
                results.append(
                    {
                        "size_bytes": size,
                        "write_latency_us": write_seconds * 1e6,
                        "write_bandwidth_gbps": size * 8 / write_seconds / 1e9,
                        "read_latency_us": read_seconds * 1e6,
                        "read_bandwidth_gbps": size * 8 / read_seconds / 1e9,
                        "write_local_wire": write_wire,
                        "write_peer_wire": peer_write_wire,
                        "read_local_wire": read_wire,
                        "read_peer_wire": peer_read_wire,
                    }
                )
    finally:
        transport.close()
        dist.destroy_process_group()
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--init-method", default="env://")
    parser.add_argument(
        "--interfaces", help="network interface, or a comma-separated pair"
    )
    parser.add_argument(
        "--options",
        default="{}",
        help="backend options as JSON, optionally one object per rank",
    )
    parser.add_argument(
        "--sizes",
        type=_parse_sizes,
        default=_parse_sizes("8,4096,1048576,67108864"),
    )
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--cuda-graph", action="store_true")
    parser.add_argument("--minimum-line-rate", type=float, default=0.8)
    args = parser.parse_args()
    if args.warmup < 0 or args.iterations <= 0:
        parser.error("warmup must be nonnegative and iterations must be positive")
    if not 0 <= args.minimum_line_rate <= 1:
        parser.error("minimum-line-rate must be between zero and one")
    return args


if __name__ == "__main__":
    parsed = parse_args()
    measurements = run(parsed)
    if int(os.environ["RANK"]) == 0:
        output = {
            "backend": parsed.backend,
            "interfaces": parsed.interfaces,
            "line_rate_gbps": _line_rate_gbps(_interface(parsed)),
            "results": measurements,
        }
        print(json.dumps(output, indent=2))
        if output["line_rate_gbps"] is not None and parsed.minimum_line_rate:
            rates = []
            for result in measurements:
                write_local = result["write_local_wire"]
                write_peer = result["write_peer_wire"]
                read_local = result["read_local_wire"]
                read_peer = result["read_peer_wire"]
                if None in (write_local, write_peer, read_local, read_peer):
                    raise RuntimeError("physical NIC counters are unavailable")
                rates.append(
                    min(
                        result["write_bandwidth_gbps"],
                        result["read_bandwidth_gbps"],
                        write_local["tx_gbps"],
                        write_peer["rx_gbps"],
                        read_local["rx_gbps"],
                        read_peer["tx_gbps"],
                    )
                )
            achieved = max(rates)
            if achieved < parsed.minimum_line_rate * output["line_rate_gbps"]:
                raise RuntimeError(
                    f"{achieved:.1f} Gb/s is below {parsed.minimum_line_rate:.0%} of line rate"
                )
