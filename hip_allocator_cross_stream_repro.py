"""Minimal repro attempts for suspected HIP caching-allocator stream-tracking
races, motivated by the FSDP2 chunked-loss failure on MI350X (see
fsdp2_chunked_loss_rocm_race.md).

STATUS: all three scenarios below currently PASS on MI350X once ``gold`` is
properly event-synced between default-stream zero and other-stream use. The
earlier "failures" were noise from an unsynced ``gold.zero_()``, not from
the caching allocator. This is a *negative* result for the hypothesis that
the FSDP race is a generic cross-stream allocator reuse bug reachable from
pure PyTorch. The FSDP symptom must involve something more specific (RCCL
workspace, internal allocator-segment splitting, collective-initiated free,
or allocator state that only arises after a RS comm). Use this script as a
starting point / scaffold rather than a working repro.

The script runs three scenarios, each a candidate pattern for the FSDP2
chunked-loss failure:

Scenario A: pure cross-stream reuse.
  Alloc A on S1, queue slow reads of A on S1. Drop A. Alloc B same size on
  S2; fill B with zeros on S2. If the allocator reuses A's block for B
  without a proper cross-stream wait, S2's zeros corrupt S1's still-pending
  reads.

Scenario B: alloc on S1, USE on S2, free, realloc on S1 (no record_stream).
  Models FSDP's reduce_scatter_input: allocated on default stream, then
  read from RS stream during the collective. If the block is freed (after
  the stream-wait on the RS event on default stream) and immediately
  reused, sequential-on-S1 is safe BUT any still-pending S2 kernel that
  read A's storage will race with the new S1 write.

Scenario C: FSDP accumulation pattern.
  Alloc A on S2. Queue: ``gold.add_(A)`` many times on S2 (gold is
  allocated on default/S1). This exactly mirrors
  ``fsdp_param.sharded_param.grad._local_tensor += new_sharded_grad``
  which runs on the RS stream, with ``_local_tensor`` from a prior chunk
  (allocated on RS stream earlier) and ``new_sharded_grad`` a view of the
  current chunk's RS output. Drop A. Alloc B on S1, fill with zeros on
  S1. If A's block is reused, and S2's pending adds haven't completed,
  gold under-accumulates — matching the FSDP symptom.

Run: python /home/weif/hip_allocator_cross_stream_repro.py
"""

import sys
import torch


def same_ptr(a: torch.Tensor, b: torch.Tensor) -> bool:
    return a.data_ptr() == b.data_ptr()


def _prep_trial(gold: torch.Tensor, streams: tuple[torch.cuda.Stream, ...]) -> None:
    """Zero gold on default stream then make every test stream wait on it.

    Without this, ``gold.zero_()`` from the current trial can race with
    the pending ``gold.add_(A)`` kernels on s1/s2 and cause spurious
    'race' reports that are just lack of cross-stream sync on ``gold``.
    """
    gold.zero_()
    ev = torch.cuda.current_stream().record_event()
    for s in streams:
        s.wait_event(ev)


def scenario_a(
    n_elements: int = 2048,
    inner_reads: int = 400,
    trials: int = 500,
    verbose_every: int = 100,
) -> int:
    """Pure cross-stream reuse: alloc A on S1, read on S1, drop, alloc B on S2."""
    dev = torch.device("cuda:0")
    dtype = torch.float32

    s1 = torch.cuda.Stream(device=dev)
    s2 = torch.cuda.Stream(device=dev)

    gold = torch.zeros(n_elements, device=dev, dtype=dtype)

    reuse_hits = 0
    failures = 0
    min_seen = float("inf")
    ptrs_seen: set[int] = set()

    torch.cuda.synchronize()
    print("--- Scenario A: pure cross-stream reuse ---")
    print(
        f"device={dev}, n_elements={n_elements} "
        f"(= {n_elements * 4 / 1024:.2f} KiB), inner_reads={inner_reads}, "
        f"trials={trials}"
    )
    print(f"hip={torch.version.hip}, cuda={torch.version.cuda}")

    # Warm up the allocator so the free-list has the right bucket shape.
    # We want a small pool of same-size blocks to maximize the odds that
    # B's alloc on S2 lands on A's freshly-freed block on S1.
    warm: list[torch.Tensor] = []
    for _ in range(8):
        warm.append(torch.empty(n_elements, device=dev, dtype=dtype))
    warm.clear()
    torch.cuda.synchronize()

    for trial in range(trials):
        _prep_trial(gold, (s1, s2))

        # 1) Alloc A on S1, fill with ones, queue many reads of A into gold.
        with torch.cuda.stream(s1):
            A = torch.empty(n_elements, device=dev, dtype=dtype)
            A.fill_(1.0)
            a_ptr = A.data_ptr()
            for _ in range(inner_reads):
                gold.add_(A)

        # 2) Drop A. Block goes to allocator's S1 free queue with an event.
        del A

        # 3) Alloc B on S2 with the same size. If the allocator pulls A's
        # block off S1's pending-free list for S2, it must insert a
        # cross-stream wait on S1's free-time event. If not, B's writes on
        # S2 race with s1's still-pending reads of A's storage.
        with torch.cuda.stream(s2):
            B = torch.empty(n_elements, device=dev, dtype=dtype)
            b_ptr = B.data_ptr()
            # 4) Overwrite the block on S2 with values != 1.0 so a missed
            # wait shows up in gold.
            B.fill_(0.0)
        del B

        if a_ptr == b_ptr:
            reuse_hits += 1
        ptrs_seen.add(a_ptr)
        ptrs_seen.add(b_ptr)

        # 5) Drain and inspect gold. Expected: every element == inner_reads.
        torch.cuda.synchronize()
        minv = gold.min().item()
        maxv = gold.max().item()
        if minv < min_seen:
            min_seen = minv

        ok = abs(minv - inner_reads) < 1e-3 and abs(maxv - inner_reads) < 1e-3
        if not ok:
            failures += 1

        if verbose_every and (trial % verbose_every == 0 or not ok):
            reused = "REUSE" if a_ptr == b_ptr else "new  "
            status = "OK " if ok else "RACE"
            print(
                f"  trial={trial:4d} {status} {reused} "
                f"a_ptr={a_ptr:#x} b_ptr={b_ptr:#x} "
                f"gold min={minv:.3f} max={maxv:.3f} "
                f"(expected {float(inner_reads):.3f})"
            )

    print()
    print(f"allocator reused A's block on {reuse_hits}/{trials} trials")
    print(f"distinct pointers observed: {len(ptrs_seen)}")
    print(f"under-accumulation (race) on     {failures}/{trials} trials")
    print(f"worst gold min observed: {min_seen:.3f} (expected {float(inner_reads):.3f})")
    print()
    if failures == 0:
        print("PASS: no cross-stream reuse race observed.")
        return 0
    else:
        print("FAIL: cross-stream allocator reuse races with pending kernel.")
        return 1


def scenario_b(
    n_elements: int = 2048,
    inner_reads: int = 400,
    trials: int = 500,
    verbose_every: int = 100,
) -> int:
    """Alloc on S1, use on S2, free, realloc on S1.

    Models FSDP's reduce_scatter_input: allocated on default stream, read
    by the RS collective on RS stream, then freed. If allocator reuses
    the block on S1 (same stream = sequential for S1 but still pending
    on S2), S1's new write races with S2's still-pending read of the old
    contents.
    """
    dev = torch.device("cuda:0")
    dtype = torch.float32

    s1 = torch.cuda.Stream(device=dev)
    s2 = torch.cuda.Stream(device=dev)

    gold = torch.zeros(n_elements, device=dev, dtype=dtype)

    reuse_hits = 0
    failures = 0
    min_seen = float("inf")

    print()
    print("--- Scenario B: alloc S1, use S2 (no record_stream), realloc S1 ---")
    print(
        f"n_elements={n_elements} inner_reads={inner_reads} trials={trials}"
    )

    # Warm up
    warm: list[torch.Tensor] = []
    for _ in range(8):
        warm.append(torch.empty(n_elements, device=dev, dtype=dtype))
    warm.clear()
    torch.cuda.synchronize()

    for trial in range(trials):
        _prep_trial(gold, (s1, s2))

        with torch.cuda.stream(s1):
            A = torch.empty(n_elements, device=dev, dtype=dtype)
            A.fill_(1.0)
            a_ptr = A.data_ptr()

        # s2 waits for s1's fill, then queues many reads on s2.
        s2.wait_stream(s1)
        with torch.cuda.stream(s2):
            for _ in range(inner_reads):
                gold.add_(A)

        # Drop A from Python. Block goes to allocator; allocator recorded
        # A's alloc stream (s1) but the last use is on s2 (no record_stream).
        del A

        # Realloc same size on s1. Likely reuses the block.
        with torch.cuda.stream(s1):
            B = torch.empty(n_elements, device=dev, dtype=dtype)
            b_ptr = B.data_ptr()
            B.fill_(0.0)
        del B

        if a_ptr == b_ptr:
            reuse_hits += 1

        torch.cuda.synchronize()
        minv = gold.min().item()
        maxv = gold.max().item()
        if minv < min_seen:
            min_seen = minv

        ok = abs(minv - inner_reads) < 1e-3 and abs(maxv - inner_reads) < 1e-3
        if not ok:
            failures += 1

        if verbose_every and (trial % verbose_every == 0 or not ok):
            reused = "REUSE" if a_ptr == b_ptr else "new  "
            status = "OK " if ok else "RACE"
            print(
                f"  trial={trial:4d} {status} {reused} "
                f"gold min={minv:.3f} max={maxv:.3f} "
                f"(expected {float(inner_reads):.3f})"
            )

    print()
    print(f"Scenario B: reused A's block on {reuse_hits}/{trials}, races: {failures}/{trials}")
    print(f"  worst gold min: {min_seen:.3f} (expected {float(inner_reads):.3f})")
    return 0 if failures == 0 else 1


def scenario_c(
    n_elements: int = 2048,
    inner_reads: int = 400,
    trials: int = 2000,
    verbose_every: int = 500,
    s2_priority: int = -1,   # FSDP's RS stream is high-priority
) -> int:
    """FSDP accumulation pattern.

    gold (persistent, default stream) is accumulated into via A (held by
    s2, gets freed across iterations). Each trial: alloc A on s2, add
    A into gold on s2 many times, drop A, alloc B same-size on s1, fill
    B with zeros on s1.

    Key subtlety: gold.zero_() runs on the default stream but gold.add_(A)
    runs on s2 without any explicit wait — mirrors FSDP where the
    ``_local_tensor`` is on default-stream storage but accumulation runs
    on the high-priority RS stream. We also make s2 high-priority like
    FSDP's reduce_scatter_stream.
    """
    dev = torch.device("cuda:0")
    dtype = torch.float32

    s1 = torch.cuda.Stream(device=dev)
    s2 = torch.cuda.Stream(device=dev, priority=s2_priority)

    gold = torch.zeros(n_elements, device=dev, dtype=dtype)

    reuse_hits = 0
    failures = 0
    min_seen = float("inf")
    race_trials: list[tuple[int, float, float, bool]] = []

    print()
    print("--- Scenario C: FSDP accumulation pattern ---")
    print(
        f"n_elements={n_elements} inner_reads={inner_reads} trials={trials} "
        f"s2_priority={s2_priority} (negative = higher)"
    )

    # Warm
    warm: list[torch.Tensor] = []
    for _ in range(8):
        warm.append(torch.empty(n_elements, device=dev, dtype=dtype))
    warm.clear()
    torch.cuda.synchronize()

    for trial in range(trials):
        _prep_trial(gold, (s1, s2))   # default stream

        with torch.cuda.stream(s2):
            A = torch.empty(n_elements, device=dev, dtype=dtype)
            A.fill_(1.0)
            a_ptr = A.data_ptr()
            for _ in range(inner_reads):
                gold.add_(A)    # s2 reads gold (from default) + A, writes gold
        del A

        with torch.cuda.stream(s1):
            B = torch.empty(n_elements, device=dev, dtype=dtype)
            b_ptr = B.data_ptr()
            B.fill_(0.0)
        del B

        if a_ptr == b_ptr:
            reuse_hits += 1

        torch.cuda.synchronize()
        minv = gold.min().item()
        maxv = gold.max().item()
        if minv < min_seen:
            min_seen = minv

        ok = abs(minv - inner_reads) < 1e-3 and abs(maxv - inner_reads) < 1e-3
        if not ok:
            failures += 1
            race_trials.append((trial, minv, maxv, a_ptr == b_ptr))

        if verbose_every and (trial % verbose_every == 0):
            reused = "REUSE" if a_ptr == b_ptr else "new  "
            print(
                f"  trial={trial:4d} OK  {reused} "
                f"gold min={minv:.3f} max={maxv:.3f} "
                f"(expected {float(inner_reads):.3f})"
            )

    print()
    print(f"Scenario C: reused A's block on {reuse_hits}/{trials}, races: {failures}/{trials}")
    print(f"  worst gold min: {min_seen:.3f} (expected {float(inner_reads):.3f})")
    if race_trials:
        print("  racing trials:")
        for t, mn, mx, reused in race_trials[:20]:
            tag = "REUSE" if reused else "new"
            print(f"    trial={t:4d} min={mn:.3f} max={mx:.3f} ({tag})")
        if len(race_trials) > 20:
            print(f"    ... ({len(race_trials) - 20} more)")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA/HIP not available.")
        sys.exit(2)
    rc_a = scenario_a()
    rc_b = scenario_b()
    rc_c = scenario_c()
    print()
    print("=== SUMMARY ===")
    print(f"Scenario A (pure cross-stream reuse):        {'FAIL' if rc_a else 'PASS'}")
    print(f"Scenario B (alloc S1 / use S2 / realloc S1): {'FAIL' if rc_b else 'PASS'}")
    print(f"Scenario C (FSDP accumulation shape):        {'FAIL' if rc_c else 'PASS'}")
    sys.exit(rc_a | rc_b | rc_c)
