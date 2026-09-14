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


_SYS_CLASS_NET = Path("/sys/class/net")


def _parse_sizes(value: str) -> list[int]:
    sizes = [int(size) for size in value.split(",")]
    if not sizes or any(size <= 0 for size in sizes):
        raise argparse.ArgumentTypeError("sizes must be positive")
    return sizes


def _device(value: str) -> torch.device:
    if value == "cuda":
        return torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    return torch.device(value)


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _line_rate_gbps(interface: str | None) -> float | None:
    if interface is None:
        return None
    try:
        return int((_SYS_CLASS_NET / interface / "speed").read_text()) / 1000
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
    root = _SYS_CLASS_NET / interface
    devices = list((root / "device" / "infiniband").glob("*"))
    if len(devices) != 1:
        return None
    try:
        port = int((root / "dev_port").read_text()) + 1
        counters = devices[0] / f"ports/{port}/counters"
        return (
            int((counters / "port_xmit_data").read_text()) * 4,
            int((counters / "port_rcv_data").read_text()) * 4,
        )
    except (OSError, ValueError):
        return None


def _netdev_wire_bytes(interface: str) -> tuple[int, int] | None:
    root = _SYS_CLASS_NET / interface / "statistics"
    try:
        return int((root / "tx_bytes").read_text()), int(
            (root / "rx_bytes").read_text()
        )
    except (OSError, ValueError):
        return None


def _counter_source(
    interface: str | None, backend: str, force_rdma: bool
) -> str | None:
    if interface is None:
        return None
    if force_rdma:
        if _rdma_wire_bytes(interface) is None:
            raise RuntimeError(f"RDMA counters are unavailable for {interface}")
        return "rdma"
    if (
        backend.lower() in {"ibverbs", "torchcomms"}
        and _rdma_wire_bytes(interface) is not None
    ):
        return "rdma"
    if _netdev_wire_bytes(interface) is None:
        return None
    return "netdev"


def _wire_bytes(interface: str | None, source: str | None) -> tuple[int, int] | None:
    if interface is None or source is None:
        return None
    if source == "rdma":
        return _rdma_wire_bytes(interface)
    return _netdev_wire_bytes(interface)


def _validate_counter_sources(local: str | None, peer: str | None) -> None:
    if local != peer:
        raise RuntimeError(f"counter sources differ between ranks: {local}, {peer}")


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


def _buffers(
    size: int, rank: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    source = torch.full((size,), rank + 1, dtype=torch.uint8, device=device)
    return source, torch.zeros_like(source), torch.zeros_like(source)


def _connects(rank: int, one_way: bool) -> bool:
    return not one_way or rank == 0


def run(args: argparse.Namespace) -> tuple[list[dict[str, Any]], str | None]:
    dist.init_process_group(
        "gloo",
        init_method=args.init_method,
        rank=int(os.environ["RANK"]),
        world_size=int(os.environ["WORLD_SIZE"]),
    )
    if dist.get_world_size() != 2:
        raise RuntimeError("transport benchmark requires exactly two ranks")
    rank = dist.get_rank()
    device = _device(args.device)
    tensor_device = _device(args.tensor_device or args.device)
    if args.cuda_graph and (device.type != "cuda" or tensor_device.type != "cuda"):
        raise ValueError("--cuda-graph requires CUDA transport and tensor devices")
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
    counter_source = _counter_source(interface, args.backend, args.rdma_counters)
    _validate_counter_sources(counter_source, _exchange(counter_source))
    transport = new_transport(args.backend, device, **options)
    results = []
    try:
        peer_url = _exchange(transport.bind())
        if _connects(rank, args.one_way_connect) and transport.connect(peer_url) != 0:
            raise RuntimeError("transport connection failed")
        dist.barrier()
        for size in args.sizes:
            source, destination, read_target = _buffers(size, rank, tensor_device)
            source_memory = transport.register_memory(source)
            destination_memory = transport.register_memory(destination)
            read_memory = transport.register_memory(read_target)
            peer_source = _exchange(source_memory.to_remote_buffer())
            peer_destination = _exchange(destination_memory.to_remote_buffer())
            _sync(tensor_device)
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
            _sync(tensor_device)
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
                _sync(tensor_device)
            source.fill_(rank + 3)
            destination.zero_()
            read_target.zero_()
            _sync(tensor_device)
            dist.barrier()
            write_wire_before = _wire_bytes(interface, counter_source)
            write_start = time.perf_counter_ns()
            write_samples = []
            read_samples = []
            if rank == 0:
                for _ in range(args.iterations):
                    start = time.perf_counter_ns()
                    write_op()
                    _sync(tensor_device)
                    write_samples.append(time.perf_counter_ns() - start)
            dist.barrier()
            write_seconds = (time.perf_counter_ns() - write_start) / 1e9
            write_wire = _wire_rate(
                write_wire_before,
                _wire_bytes(interface, counter_source),
                write_seconds,
            )
            peer_write_wire = _exchange(write_wire)

            read_wire_before = _wire_bytes(interface, counter_source)
            read_start = time.perf_counter_ns()
            if rank == 0:
                for _ in range(args.iterations):
                    start = time.perf_counter_ns()
                    read_op()
                    _sync(tensor_device)
                    read_samples.append(time.perf_counter_ns() - start)
            dist.barrier()
            read_seconds = (time.perf_counter_ns() - read_start) / 1e9
            read_wire = _wire_rate(
                read_wire_before,
                _wire_bytes(interface, counter_source),
                read_seconds,
            )
            peer_read_wire = _exchange(read_wire)
            _sync(tensor_device)
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
    return results, counter_source


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--tensor-device", help="tensor device when it differs from the transport"
    )
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
        default=_parse_sizes(
            "8,64,256,1024,4096,16384,65536,262144,1048576,4194304,16777216,67108864"
        ),
    )
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--cuda-graph", action="store_true")
    parser.add_argument(
        "--rdma-counters",
        action="store_true",
        help="read RDMA device counters instead of network-interface counters",
    )
    parser.add_argument(
        "--one-way-connect",
        action="store_true",
        help="connect rank 0 only for same-host UCXX validation",
    )
    parser.add_argument("--minimum-line-rate", type=float, default=0.8)
    parsed = parser.parse_args(args)
    if parsed.warmup < 0 or parsed.iterations <= 0:
        parser.error("warmup must be nonnegative and iterations must be positive")
    if not 0 <= parsed.minimum_line_rate <= 1:
        parser.error("minimum-line-rate must be between zero and one")
    if parsed.one_way_connect and parsed.backend.lower() != "ucxx":
        parser.error("one-way-connect is supported only by UCXX")
    return parsed


def _output(
    args: argparse.Namespace,
    measurements: list[dict[str, Any]],
    counter_source: str | None,
) -> dict[str, Any]:
    return {
        "backend": args.backend,
        "device": str(_device(args.device)),
        "tensor_device": str(_device(args.tensor_device or args.device)),
        "cuda_graph": args.cuda_graph,
        "one_way_connect": args.one_way_connect,
        "counter_source": counter_source,
        "options": json.loads(args.options),
        "warmup": args.warmup,
        "iterations": args.iterations,
        "interfaces": args.interfaces,
        "line_rate_gbps": _line_rate_gbps(_interface(args)),
        "minimum_line_rate": args.minimum_line_rate,
        "results": measurements,
    }


if __name__ == "__main__":
    parsed = parse_args()
    measurements, selected_counter_source = run(parsed)
    if int(os.environ["RANK"]) == 0:
        output = _output(parsed, measurements, selected_counter_source)
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
