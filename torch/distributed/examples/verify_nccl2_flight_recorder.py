#!/usr/bin/env python3
"""Exercise NCCL2 Flight Recorder dumps through the distributed debug server.

NCCL2 is the default process group and is used for every collective. After the
deliberate hang, the process-group store coordinates cleanup without issuing
another collective on the failed communicator. The local abort-hook dump is
disabled so the per-rank pickle files prove that the debug server observed the
unhealthy rank and requested an NCCL2 Flight Recorder dump from every worker.

Run with at least two local GPUs:

    torchrun --nproc-per-node=2 \
        torch/distributed/examples/verify_nccl2_flight_recorder.py

The script prints the output directory and analyzer command on success.
"""

import glob
import os
import pickle
import time
from datetime import timedelta


# Keep the failed NCCL2 process group alive long enough for the debug server to
# poll it. Disable the local timeout dump so only the server can create the
# per-rank files checked below. These must be set before importing torch because
# NCCL environment variables are cached.
os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = "0"
os.environ["TORCH_FR_DUMP_ON_TIMEOUT"] = "0"
os.environ.setdefault("TORCH_FR_BUFFER_SIZE", "100")

import torch
import torch.distributed as dist
from torch.distributed.debug import start_debug_server, stop_debug_server


def _shared_dump_dir(rank: int, device: torch.device) -> str:
    configured = os.environ.get("FR_DUMP_DIR")
    value: list[object] = [configured]
    if configured is None and rank == 0:
        value[0] = f"/tmp/nccl2_fr_debug_{int(time.time())}"
    dist.broadcast_object_list(value, src=0, device=device)
    dump_dir = value[0]
    if not isinstance(dump_dir, str):
        raise RuntimeError(f"invalid dump directory: {dump_dir!r}")
    return dump_dir


def _wait_for_rank_dumps(prefix: str, world_size: int, timeout: float) -> list[str]:
    expected = [f"{prefix}{rank}" for rank in range(world_size)]
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if all(os.path.isfile(path) and os.path.getsize(path) > 0 for path in expected):
            return expected
        time.sleep(0.2)
    missing = [path for path in expected if not os.path.isfile(path)]
    raise RuntimeError(f"timed out waiting for Flight Recorder dumps: {missing}")


def _validate_dumps(paths: list[str]) -> None:
    found_active_all_reduce = False
    for path in paths:
        deadline = time.monotonic() + 10
        while True:
            try:
                with open(path, "rb") as trace_file:
                    trace = pickle.load(trace_file)
                break
            except (EOFError, OSError, pickle.UnpicklingError):
                if time.monotonic() >= deadline:
                    raise
                time.sleep(0.2)
        if "entries" not in trace or "pg_config" not in trace:
            raise RuntimeError(f"invalid Flight Recorder dump: {path}")
        for entry in trace["entries"]:
            if entry["profiling_name"] == "nccl2:all_reduce" and not entry["retired"]:
                found_active_all_reduce = True
    if not found_active_all_reduce:
        raise RuntimeError("no active NCCL2 all_reduce found in the dumps")


def main() -> None:
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        raise RuntimeError("this example requires at least two CUDA devices")

    dump_interval = float(os.environ.get("FR_DUMP_INTERVAL", "2"))
    timeout_seconds = int(os.environ.get("COMM_TIMEOUT", "10"))
    hanging_rank = int(os.environ.get("HANGING_RANK", "-1"))
    debug_port = int(os.environ.get("DEBUG_SERVER_PORT", "25999"))

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
    device = torch.device("cuda", local_rank % torch.cuda.device_count())
    torch.cuda.set_device(device)
    dist.init_process_group("nccl2", timeout=timedelta(seconds=timeout_seconds))
    world_size = dist.get_world_size()
    if world_size < 2:
        raise RuntimeError("this example requires at least two ranks")

    if hanging_rank < 0:
        hanging_rank = world_size - 1
    if not 0 <= hanging_rank < world_size:
        raise ValueError(f"HANGING_RANK must be in [0, {world_size})")

    dump_dir = _shared_dump_dir(rank, device)
    per_rank_dir = os.path.join(dump_dir, "per_rank")
    if rank == 0:
        os.makedirs(per_rank_dir, exist_ok=False)
    dist.barrier()

    dump_prefix = os.path.join(per_rank_dir, "rank_")
    os.environ["TORCH_FR_DUMP_TEMP_FILE"] = dump_prefix

    start_debug_server(
        port=debug_port,
        start_method="spawn",
        dump_dir=dump_dir,
        dump_interval=dump_interval,
        enabled_dumps={"c10d_health_check", "stacks"},
    )

    tensor = torch.full((1024,), float(rank + 1), device=device)
    dist.all_reduce(tensor)
    torch.cuda.synchronize(device)
    dist.barrier()

    if rank == 0:
        print(f"Debug server: http://localhost:{debug_port}", flush=True)
        print(f"Dump directory: {dump_dir}", flush=True)
        print(
            "Local timeout dumps are disabled; per-rank files must come from "
            "the c10d health handler observing NCCL2.",
            flush=True,
        )

    pending_work = None
    if rank == hanging_rank:
        print(f"[rank {rank}] skipping the failing all_reduce", flush=True)
    else:
        print(f"[rank {rank}] issuing the failing all_reduce", flush=True)
        pending_work = dist.all_reduce(tensor, async_op=True)

    # The watchdog checks once per second. Leave enough time for the collective
    # timeout, the next health poll, and the asynchronous file writes.
    time.sleep(timeout_seconds + 3 * dump_interval + 5)

    world_pg = dist.distributed_c10d._get_default_group()
    store = dist.distributed_c10d._get_process_group_store(world_pg)
    done_key = f"verify_nccl2_fr/{os.path.basename(dump_dir)}"
    error: str | None = None
    if rank == 0:
        try:
            paths = _wait_for_rank_dumps(
                dump_prefix, world_size, timeout=2 * dump_interval + 10
            )
            _validate_dumps(paths)
            health_dumps = glob.glob(os.path.join(dump_dir, "c10d_health_check_*.txt"))
            if not health_dumps:
                raise RuntimeError("no periodic c10d health-check dump found")
            print("NCCL2 debug-server Flight Recorder test passed.", flush=True)
            print(
                "Analyze with:\n"
                "  python -m torch.distributed.flight_recorder.fr_trace "
                f"{per_rank_dir} -p rank_",
                flush=True,
            )
        except Exception as exc:
            error = str(exc)
        finally:
            store.set(done_key, "1")
    else:
        store.wait([done_key], timedelta(seconds=2 * timeout_seconds + 60))

    if rank == 0:
        stop_debug_server()
    dist.distributed_c10d._abort_process_group()

    if error is not None:
        raise RuntimeError(error)
    del pending_work


if __name__ == "__main__":
    main()
