"""Measure concurrent transport pairs between adjacent torchrun ranks."""

from __future__ import annotations

import argparse
import json
import os
import socket
import statistics
import time
from pathlib import Path
from typing import Any, TYPE_CHECKING

import torch
import torch.distributed as dist
from torch.distributed._transport import new_transport, Work


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
    if len(interfaces) not in (1, int(os.environ.get("WORLD_SIZE", "2"))):
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
    values = [None] * dist.get_world_size()
    dist.all_gather_object(values, value)
    return values[dist.get_rank() ^ 1]


def _rank_options(value: str, world_size: int) -> list[dict[str, Any]]:
    options = json.loads(value)
    if isinstance(options, dict):
        return [options] * world_size
    if (
        not isinstance(options, list)
        or len(options) != world_size
        or not all(isinstance(option, dict) for option in options)
    ):
        raise ValueError("options must be an object or one object per rank")
    return options


def _buffers(
    size: int, rank: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    source = torch.full((size,), rank + 1, dtype=torch.uint8, device=device)
    return source, torch.zeros_like(source), torch.zeros_like(source)


def _connects(rank: int, one_way: bool) -> bool:
    return not one_way or rank % 2 == 0


def _complete(result: int | Work) -> None:
    if isinstance(result, Work):
        result.wait()
    elif result != 0:
        raise RuntimeError(f"transport operation failed with status {result}")


def _measure(
    operation: Callable[[], None],
    args: argparse.Namespace,
    device: torch.device,
    interface: str | None,
    counter_source: str | None,
) -> list[dict[str, Any]]:
    samples = []
    before = _wire_bytes(interface, counter_source)
    dist.barrier()
    phase_start = time.perf_counter_ns()
    if dist.get_rank() % 2 == 0:
        for _ in range(args.iterations):
            start = time.perf_counter_ns()
            operation()
            _sync(device)
            samples.append(time.perf_counter_ns() - start)
    dist.barrier()
    seconds = (time.perf_counter_ns() - phase_start) / 1e9
    after = _wire_bytes(interface, counter_source)
    local = {
        "seconds": seconds,
        "latency_us": statistics.median(samples) / 1e3 if samples else None,
        "before": before,
        "after": after,
    }
    values: list[Any] = [None] * dist.get_world_size()
    dist.all_gather_object(values, local)
    return values


def _summarize(
    size: int, iterations: int, ranks: list[dict[str, Any]]
) -> dict[str, Any]:
    seconds = max(rank["seconds"] for rank in ranks)
    pairs = []
    for rank in range(0, len(ranks), 2):
        local, peer = ranks[rank : rank + 2]
        latency = local["latency_us"]
        pairs.append(
            {
                "ranks": [rank, rank + 1],
                "latency_us": latency,
                "bandwidth_gbps": size * 8 / latency / 1e3,
                "local_wire": _wire_rate(local["before"], local["after"], seconds),
                "peer_wire": _wire_rate(peer["before"], peer["after"], seconds),
            }
        )
    return {
        "seconds": seconds,
        "aggregate_bandwidth_gbps": size * 8 * iterations * len(pairs) / seconds / 1e9,
        "pairs": pairs,
    }


def run(args: argparse.Namespace) -> tuple[list[dict[str, Any]], str | None]:
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size < 2 or world_size % 2:
        raise ValueError("transport benchmark requires an even number of ranks")
    rank = int(os.environ["RANK"])
    options = _rank_options(args.options, world_size)[rank]
    environments = _rank_options(args.rank_env, world_size)
    if any(
        not isinstance(value, str)
        for environment in environments
        for value in environment.values()
    ):
        raise ValueError("rank-env values must be strings")
    os.environ.update(environments[rank])
    dist.init_process_group(
        "gloo",
        init_method=args.init_method,
        rank=rank,
        world_size=world_size,
    )
    device = _device(args.device)
    tensor_device = _device(args.tensor_device or args.device)
    if args.cuda_graph and (device.type != "cuda" or tensor_device.type != "cuda"):
        raise ValueError("--cuda-graph requires CUDA transport and tensor devices")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    interface = _interface(args)
    counter_source = _counter_source(interface, args.backend, args.rdma_counters)
    _validate_counter_sources(counter_source, _exchange(counter_source))
    devices: list[Any] = [None] * world_size
    dist.all_gather_object(
        devices,
        {
            "host": socket.gethostname(),
            "device": str(tensor_device),
            "interface": interface,
            "line_rate_gbps": _line_rate_gbps(interface),
        },
    )
    if world_size > 2 and interface is not None:
        nics = [(entry["host"], entry["interface"]) for entry in devices]
        if len(set(nics)) != world_size:
            raise ValueError("concurrent pairs require a distinct NIC per rank")
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
            _sync(tensor_device)
            source_descriptor = source_memory.to_remote_buffer()
            peer_source = type(source_descriptor).deserialize(
                _exchange(source_descriptor.serialize())
            )
            target_descriptor = destination_memory.to_remote_buffer()
            peer_destination = type(target_descriptor).deserialize(
                _exchange(target_descriptor.serialize())
            )
            source_view = source_memory.to_view()
            read_view = read_memory.to_mutable_view()

            def write() -> None:
                _complete(
                    transport.write(
                        source_view, peer_destination, async_op=args.async_op
                    )
                )

            def read() -> None:
                _complete(
                    transport.read(read_view, peer_source, async_op=args.async_op)
                )

            for _ in range(args.warmup):
                if rank % 2 == 0:
                    write()
                    read()
            _sync(tensor_device)
            write_op: Callable[[], None] = write
            read_op: Callable[[], None] = read
            if rank % 2 == 0 and args.cuda_graph:
                write_graph = torch.cuda.CUDAGraph()
                read_graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(write_graph):
                    write()
                with torch.cuda.graph(read_graph):
                    read()
                write_op = write_graph.replay
                read_op = read_graph.replay
                _sync(tensor_device)
            dist.barrier()
            source.fill_(rank + 3)
            destination.zero_()
            read_target.zero_()
            _sync(tensor_device)
            writes = _measure(write_op, args, tensor_device, interface, counter_source)
            reads = _measure(read_op, args, tensor_device, interface, counter_source)
            _sync(tensor_device)
            if rank % 2:
                torch.testing.assert_close(
                    destination, torch.full_like(destination, (rank ^ 1) + 3)
                )
            else:
                torch.testing.assert_close(
                    read_target, torch.full_like(read_target, (rank ^ 1) + 3)
                )
            if rank == 0:
                write_result = _summarize(size, args.iterations, writes)
                read_result = _summarize(size, args.iterations, reads)
                write_pair = write_result["pairs"][0]
                read_pair = read_result["pairs"][0]
                results.append(
                    {
                        "size_bytes": size,
                        "write_latency_us": write_pair["latency_us"],
                        "write_bandwidth_gbps": write_pair["bandwidth_gbps"],
                        "read_latency_us": read_pair["latency_us"],
                        "read_bandwidth_gbps": read_pair["bandwidth_gbps"],
                        "write_local_wire": write_pair["local_wire"],
                        "write_peer_wire": write_pair["peer_wire"],
                        "read_local_wire": read_pair["local_wire"],
                        "read_peer_wire": read_pair["peer_wire"],
                        "write": write_result,
                        "read": read_result,
                        "devices": devices,
                    }
                )
            dist.barrier()
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
        "--interfaces", help="network interface, or a comma-separated list per rank"
    )
    parser.add_argument(
        "--rank-env",
        default="{}",
        help="environment variables as JSON, optionally one object per rank",
    )
    parser.add_argument("--output", type=Path, help="also write results to a JSON file")
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
    execution = parser.add_mutually_exclusive_group()
    execution.add_argument("--cuda-graph", action="store_true")
    execution.add_argument(
        "--async-op", action="store_true", help="submit and wait on a Work per transfer"
    )
    parser.add_argument(
        "--rdma-counters",
        action="store_true",
        help="read RDMA device counters instead of network-interface counters",
    )
    parser.add_argument(
        "--one-way-connect",
        action="store_true",
        help="connect only even ranks for one-sided validation",
    )
    parser.add_argument("--minimum-line-rate", type=float, default=0.8)
    parsed = parser.parse_args(args)
    if parsed.warmup < 0 or parsed.iterations <= 0:
        parser.error("warmup must be nonnegative and iterations must be positive")
    if not 0 <= parsed.minimum_line_rate <= 1:
        parser.error("minimum-line-rate must be between zero and one")
    if parsed.one_way_connect and parsed.backend.lower() not in (
        "mooncake",
        "nixl",
        "ucxx",
    ):
        parser.error("one-way-connect is supported only by Mooncake, NIXL, and UCXX")
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
        "async_op": args.async_op,
        "one_way_connect": args.one_way_connect,
        "counter_source": counter_source,
        "options": json.loads(args.options),
        "rank_env": json.loads(args.rank_env),
        "world_size": int(os.environ.get("WORLD_SIZE", "2")),
        "warmup": args.warmup,
        "iterations": args.iterations,
        "interfaces": args.interfaces,
        "line_rate_gbps": _line_rate_gbps(_interface(args)),
        "minimum_line_rate": args.minimum_line_rate,
        "results": measurements,
    }


def _check_line_rate(measurements: list[dict[str, Any]], minimum: float) -> None:
    if not minimum:
        return
    fractions = []
    for result in measurements:
        rates = [device["line_rate_gbps"] for device in result["devices"]]
        if any(rate is None or rate <= 0 for rate in rates):
            return
        capacities = [
            min(rates[rank], rates[rank + 1]) for rank in range(0, len(rates), 2)
        ]
        fractions_per_size = []
        for operation, local_direction, peer_direction in (
            ("write", "tx_gbps", "rx_gbps"),
            ("read", "rx_gbps", "tx_gbps"),
        ):
            measured = result[operation]
            fractions_per_size.append(
                measured["aggregate_bandwidth_gbps"] / sum(capacities)
            )
            for pair, capacity in zip(measured["pairs"], capacities):
                local, peer = pair["local_wire"], pair["peer_wire"]
                if local is None or peer is None:
                    raise RuntimeError("physical NIC counters are unavailable")
                fractions_per_size.append(
                    min(
                        pair["bandwidth_gbps"],
                        local[local_direction],
                        peer[peer_direction],
                    )
                    / capacity
                )
        fractions.append(min(fractions_per_size))
    achieved = max(fractions)
    if achieved < minimum:
        raise RuntimeError(f"{achieved:.1%} is below {minimum:.0%} of line rate")


if __name__ == "__main__":
    parsed = parse_args()
    measurements, selected_counter_source = run(parsed)
    if int(os.environ["RANK"]) == 0:
        output = _output(parsed, measurements, selected_counter_source)
        print(json.dumps(output, indent=2))
        if parsed.output is not None:
            parsed.output.write_text(json.dumps(output, indent=2) + "\n")
        _check_line_rate(measurements, parsed.minimum_line_rate)
