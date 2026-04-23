"""`_sleep` variant of /home/weif/rccl_chunked_loss_repro.py.

Purpose: discriminate **pure stream-ordering bug** from **caching-allocator
event-gating bug** as the root cause of the FSDP2 chunked-loss race.

The parent repro races 49-50/50 on ROCm, 0/N on CUDA. The doc claims CUDA is
safe because NCCL + CUDA's allocator record_stream/event gating serializes
cross-stream reuse; ROCm's event gating is broken.

Counter-hypothesis (user): if the fix is
``current_stream().wait_event(post_reduce_event)`` at post_backward exit,
then this is a stream-ordering race that should fire on CUDA too once we
widen the race window artificially. ``torch.cuda._sleep(cycles)`` enqueues
a busy-wait kernel on the current stream that reliably widens the window.

Where the sleep is injected:
    * ``--sleep-where rs_after_accum`` (default): on RS stream AFTER the
      accumulate kernel. Keeps RS stream busy long past the point where a
      correct allocator should consider rs_output "quiesced".
    * ``--sleep-where rs_before_accum``: on RS stream BEFORE accumulate. This
      widens the window *while* the accumulate is still reading rs_output,
      which is the stricter condition for a pure cross-stream data race.

Discrimination:
    * If CUDA + a large sleep still shows 0 races → allocator event gating
      really does hide it. The fix on ROCm works around the RCCL/allocator
      gating gap; there is no "pure FSDP2 stream-ordering bug" in the sense
      of being platform-agnostic.
    * If CUDA + large sleep races → the doc's framing is right that FSDP2
      has a latent stream-ordering gap, and CUDA's allocator gating was just
      masking it by timing (not by correctness guarantee).

Run on 2 GPUs:
    python /home/weif/rccl_chunked_loss_repro_sleep.py \\
        --iterations 50 --sleep-cycles 100_000_000
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
    os.environ.setdefault("MASTER_PORT", "29501")
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    torch.cuda.set_device(rank)
    dist.init_process_group(backend="nccl", world_size=world_size, rank=rank)
    return torch.device(f"cuda:{rank}")


SYNC_MODES = (
    "none",
    "device",
    "rs_stream",
    "rs_event_cpu",
    "rs_event_stream",
    "post_accum_event_cpu",
    "post_accum_event_stream",
)

SLEEP_WHERE = (
    "none",               # no sleep (baseline)
    "rs_after_accum",     # on RS stream AFTER accumulate
    "rs_before_accum",    # on RS stream BEFORE accumulate (while rs_output is live-read)
    "rs_between_rs_and_accum",  # on RS stream between reduce_scatter and accumulate
)

HOLD_REFS = (
    "off",    # del rs_input, rs_output at chunk end (standalone default)
    "input",  # hold rs_input across chunks; rs_output dropped — mimics FSDP
    "both",   # hold rs_input + rs_output across chunks
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
    use_rccl: bool,
    sleep_cycles: int,
    sleep_where: str,
    hold_refs: str,
) -> int:
    if sync_mode not in SYNC_MODES:
        raise ValueError(f"sync_mode must be in {SYNC_MODES}, got {sync_mode!r}")
    if sleep_where not in SLEEP_WHERE:
        raise ValueError(f"sleep_where must be in {SLEEP_WHERE}, got {sleep_where!r}")
    if hold_refs not in HOLD_REFS:
        raise ValueError(f"hold_refs must be in {HOLD_REFS}, got {hold_refs!r}")
    dtype = torch.float32
    full_numel = shard_numel * world_size

    rs_stream = torch.cuda.Stream(
        device=device, priority=-1 if high_priority_rs else 0
    )
    ag_stream = torch.cuda.Stream(device=device, priority=0)

    accumulator = torch.zeros(shard_numel, device=device, dtype=dtype)

    if do_ag:
        ag_output = torch.zeros(full_numel, device=device, dtype=dtype)
        ag_input = torch.ones(shard_numel, device=device, dtype=dtype) * (rank + 1)

    def maybe_sleep(where: str) -> None:
        if sleep_cycles > 0 and sleep_where == where:
            torch.cuda._sleep(sleep_cycles)

    races = 0
    for it in range(iterations):
        accumulator.zero_()
        zero_ev = torch.cuda.current_stream(device).record_event()
        rs_stream.wait_event(zero_ev)

        local_is_set = False
        # Per-iteration lists that retain Python refs to rs_input / rs_output
        # across all chunks when --hold-refs is set. Mimics FSDP's
        # reduce_scatter_states.append(...) behavior, which pins
        # reduce_scatter_input (and, via sharded_param.grad, chunk 0's
        # reduce_output) for the duration of backward.
        held_rs_inputs: list[torch.Tensor] = []
        held_rs_outputs: list[torch.Tensor] = []

        for chunk in range(n_chunks):
            rs_input = torch.empty(full_numel, device=device, dtype=dtype)
            rs_input.fill_(float((rank + 1) * (chunk + 1)))
            rs_stream.wait_stream(torch.cuda.current_stream(device))

            with torch.cuda.stream(rs_stream):
                rs_output = torch.empty(shard_numel, device=device, dtype=dtype)
                if use_rccl:
                    dist.reduce_scatter_tensor(rs_output, rs_input)
                else:
                    rs_output.fill_(float((chunk + 1) * world_size * (world_size + 1) // 2))
                post_rs_event = rs_stream.record_event() if sync_mode in (
                    "rs_event_cpu", "rs_event_stream"
                ) else None
                # Optional sleep between RS and accumulate: keeps rs_output
                # "in flight" on RS stream without the accumulate having read
                # it yet. If the default stream's next-chunk rs_input gets the
                # same block during this window and writes to it, the
                # subsequent accumulate reads corrupted data.
                maybe_sleep("rs_between_rs_and_accum")
                # Optional sleep BEFORE accumulate: same effect as the above
                # but placed at the top of the accumulate phase; kept for
                # legacy flag name clarity.
                maybe_sleep("rs_before_accum")
                if not local_is_set:
                    accumulator.copy_(rs_output)
                    local_is_set = True
                else:
                    accumulator.add_(rs_output)
                # Optional sleep AFTER accumulate: rs_output reads are done,
                # but RS stream stays busy, so its free-list event fires late.
                # If the allocator reuses rs_output's block before that event,
                # we'd corrupt accumulator via RS stream's in-progress add_
                # — but since add_ has completed, this only widens the window
                # for subsequent chunks' cross-stream interactions.
                maybe_sleep("rs_after_accum")
                post_accum_event = rs_stream.record_event() if sync_mode in (
                    "post_accum_event_cpu", "post_accum_event_stream"
                ) else None
            if hold_refs in ("input", "both"):
                held_rs_inputs.append(rs_input)
            if hold_refs == "both":
                held_rs_outputs.append(rs_output)
            del rs_input, rs_output

            if do_ag:
                with torch.cuda.stream(ag_stream):
                    dist.all_gather_into_tensor(ag_output, ag_input)

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

        end_ev = rs_stream.record_event()
        torch.cuda.current_stream(device).wait_event(end_ev)

        torch.cuda.synchronize()
        # Release per-iteration held refs AFTER sync so the allocator can
        # recycle the blocks for the next iteration safely.
        del held_rs_inputs
        del held_rs_outputs

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
        f"sync_mode={sync_mode}, sleep_cycles={sleep_cycles}, "
        f"sleep_where={sleep_where}, hold_refs={hold_refs})"
    )
    return races


def _worker(rank: int, world_size: int, args: argparse.Namespace) -> None:
    try:
        device = _init(rank, world_size)
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
            sleep_cycles=args.sleep_cycles,
            sleep_where=args.sleep_where,
            hold_refs=args.hold_refs,
        )
    except Exception:
        traceback.print_exc()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--n-chunks", type=int, default=4)
    parser.add_argument("--shard-numel", type=int, default=2048)
    parser.add_argument("--ag", action="store_true", default=True)
    parser.add_argument("--no-ag", dest="ag", action="store_false")
    parser.add_argument("--hipri", action="store_true", default=True)
    parser.add_argument("--no-hipri", dest="hipri", action="store_false")
    parser.add_argument("--sync-mode", choices=SYNC_MODES, default="none")
    parser.add_argument("--no-rccl", action="store_true")
    parser.add_argument("--sleep-cycles", type=int, default=0,
                        help="cycles for torch.cuda._sleep on RS stream; "
                             "~1e9 cycles ≈ many ms on modern GPUs")
    parser.add_argument("--sleep-where", choices=SLEEP_WHERE, default="rs_after_accum",
                        help="where on the RS stream to inject _sleep")
    parser.add_argument("--hold-refs", choices=HOLD_REFS, default="off",
                        help="hold Python refs to rs_input (or rs_input+rs_output) "
                             "across chunks; mimics FSDP's reduce_scatter_states.append")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA/HIP not available.")
        return 2
    world_size = min(torch.cuda.device_count(), 2)
    if world_size < 2:
        print("Need ≥2 GPUs.")
        return 2

    import random
    os.environ["MASTER_PORT"] = str(29500 + random.randint(1, 9999))

    print(
        f"iterations={args.iterations}, n_chunks={args.n_chunks}, "
        f"shard_numel={args.shard_numel}, ag={args.ag}, "
        f"hipri={args.hipri}, sync_mode={args.sync_mode}, "
        f"sleep_cycles={args.sleep_cycles}, sleep_where={args.sleep_where}, "
        f"hold_refs={args.hold_refs}"
    )
    print(f"hip={torch.version.hip}, cuda={torch.version.cuda}")

    mp.spawn(_worker, args=(world_size, args), nprocs=world_size, join=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
