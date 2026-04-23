"""Quick analyzer for the torch.cuda.memory snapshot from the repro.

We care about:
  - How many distinct addresses got used for ``rs_output`` (shard_numel=2048,
    fp32 = 8 KiB). Over-reuse of the same small set of blocks is a
    prerequisite for the race.
  - Which streams those allocations were tagged to.
  - Whether RCCL-side allocations show up and what stream/lifetime they
    have.
"""

from __future__ import annotations

import pickle
import sys
from collections import Counter


SHARD_BYTES = 2048 * 4           # rs_output
FULL_BYTES = 2048 * 2 * 4        # rs_input (world_size=2)


def main(path: str) -> None:
    with open(path, "rb") as f:
        snap = pickle.load(f)

    segments = snap.get("segments", [])
    print(f"segments: {len(segments)}")

    # Walk every block in every segment; size is the requested size.
    by_size: Counter = Counter()
    by_stream_size: Counter = Counter()
    for seg in segments:
        seg_stream = seg.get("stream")
        for blk in seg.get("blocks", []):
            sz = blk.get("requested_size") or blk.get("size")
            if sz is None:
                continue
            by_size[sz] += 1
            by_stream_size[(seg_stream, sz)] += 1

    print("top 15 block sizes across segments (bytes):")
    for sz, n in by_size.most_common(15):
        tag = ""
        if sz == SHARD_BYTES:
            tag = "  <-- rs_output shard"
        elif sz == FULL_BYTES:
            tag = "  <-- rs_input full"
        print(f"  {sz:>10d}  count={n}{tag}")

    print("\nper-stream breakdown for suspect sizes:")
    for (stream, sz), n in sorted(by_stream_size.items()):
        if sz in (SHARD_BYTES, FULL_BYTES):
            print(f"  stream={stream}  size={sz}  count={n}")

    # Allocation events: history of events if recorded.
    events = snap.get("device_traces", [])
    print(f"\ndevice_traces (per-device): {len(events)}")
    for dev_idx, trace in enumerate(events):
        alloc_events = [e for e in trace if e.get("action") in ("alloc", "free")]
        alloc_size_counter: Counter = Counter(e.get("size") for e in alloc_events)
        print(f"  device {dev_idx}: {len(alloc_events)} alloc/free events total")
        for sz, n in alloc_size_counter.most_common(6):
            tag = ""
            if sz == SHARD_BYTES:
                tag = "  <-- rs_output"
            elif sz == FULL_BYTES:
                tag = "  <-- rs_input"
            print(f"    size={sz} count={n}{tag}")

    # Reuse analysis: for SHARD_BYTES, how many distinct addrs were used?
    shard_addrs: Counter = Counter()
    for trace in events:
        for e in trace:
            if e.get("action") == "alloc" and e.get("size") == SHARD_BYTES:
                shard_addrs[e.get("addr")] += 1
    if shard_addrs:
        print(
            f"\nrs_output ({SHARD_BYTES} B): "
            f"{len(shard_addrs)} distinct addrs across "
            f"{sum(shard_addrs.values())} allocs"
        )
        top = shard_addrs.most_common(5)
        for addr, n in top:
            print(f"  addr={addr:#x} reused {n} times")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "/tmp/rccl_race_snap_rank1.pickle")
