"""RCCL-in-the-loop repro for the FSDP2 chunked-loss race on MI350X.

This mimics the inner loop of ``foreach_reduce`` in FSDP2 without any of
FSDP's machinery: no FSDPParamGroup, no autograd, no modules. Just:

  per chunk:
    1. alloc ``rs_input`` on default stream, fill with known values
       (rank-specific, so reduce-sum across ranks is predictable)
    2. RS stream waits on default stream
    3. alloc ``rs_output`` on RS stream
    4. dist.reduce_scatter_tensor(rs_output, rs_input)
    5. accumulator._local += rs_output           # accumulation on RS stream
       (first chunk: accumulator._local = rs_output; subsequent: +=)
    6. optionally also do an all_gather (matches reshard_after_forward=True)
    7. drop python refs to rs_input / rs_output
    8. go to next chunk (no CPU sync)

Then torch.cuda.synchronize() and check accumulator. Over many iterations
of (n_chunks × these steps), any value drift on rank 1's shard is a race.

If this reproduces the FSDP symptom (rank 1 under-accumulates), the bug
is at the (HIP allocator × RCCL × stream) intersection — not in FSDP.

Run (2 GPUs required):
    python /home/weif/rccl_chunked_loss_repro.py
"""

from __future__ import annotations

import argparse
import os
import sys
import traceback

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def _init(rank: int, world_size: int) -> torch.device:
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29501")  # parent may override
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    torch.cuda.set_device(rank)
    # NCCL on CUDA builds; RCCL on ROCm builds (same backend name).
    dist.init_process_group(backend="nccl", world_size=world_size, rank=rank)
    return torch.device(f"cuda:{rank}")


SYNC_MODES = (
    "none",                   # no per-chunk sync (baseline race)
    "device",                 # torch.cuda.synchronize() — full CPU drain
    "rs_stream",              # rs_stream.synchronize() — CPU wait on RS
    "rs_event_cpu",           # event after RS comm, event.synchronize() CPU-wait
    "rs_event_stream",        # event after RS comm, default.wait_event() — stream-level only
    "post_accum_event_cpu",   # event after accumulate, event.synchronize() CPU-wait
    "post_accum_event_stream",  # event after accumulate, default.wait_event() — stream-level only
)


def _chunk_pattern(
    rank: int,
    world_size: int,
    device: torch.device,
    iterations: int,
    n_chunks: int,
    shard_numel: int,
    do_ag: bool,
    high_priority_rs: bool,
    sync_mode: str,
    use_rccl: bool = True,
) -> int:
    if sync_mode not in SYNC_MODES:
        raise ValueError(f"sync_mode must be in {SYNC_MODES}, got {sync_mode!r}")
    """Each iteration runs ``n_chunks`` RS-then-accumulate steps, no sync
    between chunks. Mirrors FSDP chunked-loss."""
    dtype = torch.float32
    full_numel = shard_numel * world_size

    # RS stream (high-priority like FSDP's reduce_scatter_stream)
    rs_stream = torch.cuda.Stream(
        device=device, priority=-1 if high_priority_rs else 0
    )
    # AG stream (for the optional AG between iterations, matching
    # reshard_after_forward=True)
    ag_stream = torch.cuda.Stream(device=device, priority=0)

    # The persistent accumulator — plays the role of
    # ``fsdp_param.sharded_param.grad._local_tensor``.
    # Set up by chunk 0's assignment; here we pre-allocate to have a stable
    # buffer between iterations.
    accumulator = torch.zeros(shard_numel, device=device, dtype=dtype)

    # For AG: an "unsharded" buffer that we'd AG into each iter.
    if do_ag:
        ag_output = torch.zeros(full_numel, device=device, dtype=dtype)
        ag_input = torch.ones(shard_numel, device=device, dtype=dtype) * (rank + 1)

    races = 0
    for it in range(iterations):
        accumulator.zero_()
        # sync the zero_() with the RS stream before any RS-stream reads it
        zero_ev = torch.cuda.current_stream(device).record_event()
        rs_stream.wait_event(zero_ev)

        # Use a fresh "local_is_set" flag for chunk 0; subsequent chunks +=.
        local_is_set = False

        for chunk in range(n_chunks):
            # 1. alloc rs_input on default stream, fill with known values.
            #    value = (rank + 1) * (chunk + 1) so the reduce-sum output
            #    is predictable.
            rs_input = torch.empty(full_numel, device=device, dtype=dtype)
            rs_input.fill_(float((rank + 1) * (chunk + 1)))
            # 2. RS stream waits on default stream so copy-in kernel is done
            rs_stream.wait_stream(torch.cuda.current_stream(device))

            with torch.cuda.stream(rs_stream):
                # 3. alloc rs_output on RS stream
                rs_output = torch.empty(shard_numel, device=device, dtype=dtype)
                # 4. reduce-scatter (or local stand-in when --no-rccl)
                if use_rccl:
                    dist.reduce_scatter_tensor(rs_output, rs_input)
                else:
                    # local op that produces the same value on each rank's
                    # shard — keeps the stream + allocator shape intact
                    # without RCCL. Equivalent to summing (rank+1)*(c+1)
                    # over ranks for a uniform-value rs_input.
                    rs_output.fill_(float((chunk + 1) * world_size * (world_size + 1) // 2))
                # Event right after the collective, before accumulate.
                post_rs_event = rs_stream.record_event() if sync_mode in (
                    "rs_event_cpu", "rs_event_stream"
                ) else None
                # 5. accumulation (+= view of rs_output)
                if not local_is_set:
                    # chunk 0: assign; downstream this mirrors
                    # ``sharded_param.grad = to_sharded_dtensor(new_sharded_grad)``
                    accumulator.copy_(rs_output)
                    local_is_set = True
                else:
                    accumulator.add_(rs_output)
                post_accum_event = rs_stream.record_event() if sync_mode in (
                    "post_accum_event_cpu", "post_accum_event_stream"
                ) else None
            # drop refs — rs_input & rs_output go to allocator free lists
            del rs_input, rs_output

            if do_ag:
                with torch.cuda.stream(ag_stream):
                    dist.all_gather_into_tensor(ag_output, ag_input)

            # === per-chunk sync bisect ===
            if sync_mode == "device":
                torch.cuda.synchronize()
            elif sync_mode == "rs_stream":
                rs_stream.synchronize()
            elif sync_mode == "rs_event_cpu":
                assert post_rs_event is not None
                post_rs_event.synchronize()
            elif sync_mode == "rs_event_stream":
                assert post_rs_event is not None
                torch.cuda.current_stream(device).wait_event(post_rs_event)
            elif sync_mode == "post_accum_event_cpu":
                assert post_accum_event is not None
                post_accum_event.synchronize()
            elif sync_mode == "post_accum_event_stream":
                assert post_accum_event is not None
                torch.cuda.current_stream(device).wait_event(post_accum_event)
            # sync_mode == "none" => no sync
        # end chunk loop

        # make default stream wait for RS stream's last accumulation
        # before we read accumulator
        end_ev = rs_stream.record_event()
        torch.cuda.current_stream(device).wait_event(end_ev)

        torch.cuda.synchronize()

        # Expected accumulator value for this rank:
        # Each chunk c contributes reduce-sum of (rank+1)*(c+1) across ranks
        # but reduce-scatter: each rank gets one slice of the reduced tensor.
        # rs_input is flat of size full_numel; reduce-scatter gives each
        # rank shard_numel of the reduce-sum. Since rs_input is all
        # (rank+1)*(c+1) (uniform), the reduced value is
        # sum_r (r+1)*(c+1) = (c+1) * T(world_size), where
        # T(W) = 1+2+..+W = W*(W+1)/2.
        # Each chunk's shard is thus (c+1)*T(W), uniform. Accumulator is
        # chunk 0 copy + chunk 1..n_chunks-1 adds =
        # 1*T(W) + 2*T(W) + ... + n_chunks*T(W) = S(n_chunks)*T(W)
        # where S(K) = K*(K+1)/2.
        T = world_size * (world_size + 1) // 2
        S = n_chunks * (n_chunks + 1) // 2
        expected = float(S * T)
        minv = accumulator.min().item()
        maxv = accumulator.max().item()
        uniform = abs(maxv - minv) < 1e-3
        correct = abs(minv - expected) < 1e-3 and uniform
        if not correct:
            races += 1
            if rank == 1 or races <= 5:
                print(
                    f"[rank {rank}] iter={it:4d} RACE "
                    f"min={minv:.3f} max={maxv:.3f} expected={expected:.3f}"
                )
        elif it % max(iterations // 10, 1) == 0:
            print(
                f"[rank {rank}] iter={it:4d} OK   "
                f"min={minv:.3f} max={maxv:.3f} expected={expected:.3f}"
            )

    print(
        f"[rank {rank}] DONE. races: {races}/{iterations} "
        f"(n_chunks={n_chunks}, shard_numel={shard_numel}, "
        f"do_ag={do_ag}, hipri_rs={high_priority_rs}, "
        f"sync_mode={sync_mode})"
    )
    return races


def _worker(rank: int, world_size: int, args: argparse.Namespace) -> None:
    try:
        device = _init(rank, world_size)
        if args.record_memory:
            torch.cuda.memory._record_memory_history(
                enabled="all", max_entries=500_000
            )
        _chunk_pattern(
            rank=rank,
            world_size=world_size,
            device=device,
            iterations=args.iterations,
            n_chunks=args.n_chunks,
            shard_numel=args.shard_numel,
            do_ag=args.ag,
            high_priority_rs=args.hipri,
            sync_mode=args.sync_mode,
            use_rccl=not args.no_rccl,
        )
        if args.record_memory:
            path = f"{args.record_memory}_rank{rank}.pickle"
            torch.cuda.memory._dump_snapshot(path)
            torch.cuda.memory._record_memory_history(enabled=None)
            print(f"[rank {rank}] memory snapshot → {path}")
    except Exception:
        traceback.print_exc()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--n-chunks", type=int, default=4)
    # head.weight is [vocab=128, dim=32] fp32; shard_numel on 2 ranks = 2048
    parser.add_argument("--shard-numel", type=int, default=2048)
    parser.add_argument("--ag", action="store_true", default=True,
                        help="include an all_gather per chunk (matches "
                             "reshard_after_forward=True)")
    parser.add_argument("--no-ag", dest="ag", action="store_false")
    parser.add_argument("--hipri", action="store_true", default=True,
                        help="RS stream at high priority like FSDP")
    parser.add_argument("--no-hipri", dest="hipri", action="store_false")
    parser.add_argument("--sync-mode", choices=SYNC_MODES, default="none",
                        help="per-chunk sync variant to bisect which layer "
                             "owns the race; see SYNC_MODES")
    parser.add_argument("--no-rccl", action="store_true",
                        help="replace dist.reduce_scatter_tensor with a "
                             "local sharded copy (keeps the streams + allocator "
                             "pattern but removes RCCL from the loop)")
    parser.add_argument("--record-memory", type=str, default="",
                        help="path prefix for torch.cuda.memory snapshot; "
                             "per-rank suffix _rank{R}.pickle appended")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA/HIP not available.")
        return 2
    world_size = min(torch.cuda.device_count(), 2)
    if world_size < 2:
        print("Need ≥2 GPUs.")
        return 2

    # pick a random port per run so back-to-back invocations don't EADDRINUSE
    import random
    os.environ["MASTER_PORT"] = str(29500 + random.randint(1, 9999))

    print(
        f"iterations={args.iterations}, n_chunks={args.n_chunks}, "
        f"shard_numel={args.shard_numel}, ag={args.ag}, "
        f"hipri={args.hipri}, sync_mode={args.sync_mode}"
    )
    print(f"hip={torch.version.hip}, cuda={torch.version.cuda}")

    mp.spawn(_worker, args=(world_size, args), nprocs=world_size, join=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
