"""Roofline microbenchmark for the CUPTI monitor's columnar decode.

Measures how fast ``CuptiMonitorBuffer.decode()`` turns a raw CUPTI
user-defined-record buffer into demuxed ``{kind: {field_id: column}}`` columns --
the per-buffer work the monitor's worker thread does for every completed buffer.
This is the ceiling on how many records/s the monitor can stream off CUPTI before
the decode (not CUPTI, not the GPU) becomes the bottleneck.

It is pure CPU + numpy: it synthesizes the byte buffer and the captured record
layout by hand and calls the real ``decode()``, so it needs only a torch build
with ``torch.profiler.cupti`` -- no GPU, no CUPTI, no libcupti preload.

Three things it reports:

  1. roofline -- records/s vs records-per-buffer (small buffers are dominated by
     the fixed per-buffer numpy/setup cost; large buffers reach steady state).
  2. decode strategy -- single kind (homogeneous stride) vs uniform record size
     (stride + KIND dispatch) vs variable size (the per-record Python walk).
  3. field selection -- a minimal timer selection (KIND/START/END) vs the full
     ProfilerObserver selection, numeric-only vs including the const char* NAME
     field (whose per-record pointer deref is the dominant cost).

Run:  python benchmarks/profiler_benchmark/bench_decode.py
      python benchmarks/profiler_benchmark/bench_decode.py --json out.json
"""

from __future__ import annotations

import argparse
import ctypes
import json
import time
from typing import Any

from torch.profiler.cupti.cupti_python import ActivityKind
from torch.profiler.cupti.monitor import CuptiMonitorBuffer
from torch.profiler.cupti.records import Kernel


_KERNEL = int(ActivityKind.CONCURRENT_KERNEL)
_MEMCPY = int(ActivityKind.MEMCPY)
_KIND_SIZE = 4  # *_FIELD_KIND (id 0) is the 4-byte kind header at offset 0
_NAME_ID = int(Kernel.NAME)  # const char* field -> per-record deref path


def _layout(field_ids: list[int]) -> tuple[int, list[tuple[int, int, int]]]:
    """A captured record layout for ``field_ids``: KIND (id 0, 4B) at offset 0,
    then every other field 8B on an 8-byte stride. Returns (record_size,
    [(field_id, offset, size), ...])."""
    fields: list[tuple[int, int, int]] = [(0, 0, _KIND_SIZE)]
    offset = 8  # pad past the 4-byte KIND to the first 8-byte field
    for fid in field_ids:
        if fid == 0:
            continue
        fields.append((fid, offset, 8))
        offset += 8
    return offset, fields


def _build_buffer(
    layouts: dict[int, tuple[int, list[tuple[int, int, int]]]],
    kinds_cycle: list[int],
    n_records: int,
) -> tuple[list[Any], int, int]:
    """Synthesize a buffer of ``n_records`` records cycling through ``kinds_cycle``.
    String fields are pointed at one shared, valid C string (so the deref path is
    exercised). Returns (keepalive, address, valid_size)."""
    shared_str = ctypes.create_string_buffer(b"some_mangled_kernel_symbol_v2")
    str_ptr = ctypes.addressof(shared_str)
    keep: list[Any] = [shared_str]

    blob = bytearray()
    for i in range(n_records):
        kind = kinds_cycle[i % len(kinds_cycle)]
        rsz, fields = layouts[kind]
        rec = bytearray(rsz)
        rec[0:_KIND_SIZE] = int(kind).to_bytes(_KIND_SIZE, "little")
        for fid, off, sz in fields:
            if fid == 0:
                continue
            val = str_ptr if fid == _NAME_ID else (i & 0xFFFFFFFF)
            rec[off : off + sz] = int(val).to_bytes(sz, "little")
        blob += rec

    buf = (ctypes.c_uint8 * max(len(blob), 1)).from_buffer_copy(bytes(blob))
    keep.append(buf)
    return keep, ctypes.addressof(buf), len(blob)


def _decode(addr: int, valid_size: int, layouts: dict) -> dict:
    record_layouts = [(k, rsz, fields) for k, (rsz, fields) in layouts.items()]
    buf = CuptiMonitorBuffer((addr, valid_size, 0, 0, record_layouts))
    buf._returned = True  # synthetic (non-pool) pointer: skip the RAII return
    return buf.decode()


def _time_decode(
    addr: int, valid_size: int, layouts: dict, n_records: int, *, iters: int, reps: int
) -> dict[str, float]:
    """Time ``decode()`` over ``reps`` reps of ``iters`` calls each; report the
    best (least noisy) rep as records/s, MB/s, ns/record."""
    # Correctness/warm-up: total decoded records must equal what we wrote.
    out = _decode(addr, valid_size, layouts)
    decoded = sum(len(next(iter(cols.values()))) for cols in out.values())
    if decoded != n_records:
        raise RuntimeError(f"decoded {decoded} != {n_records} records")

    best_per_call = float("inf")
    for _ in range(reps):
        t0 = time.perf_counter()
        for _ in range(iters):
            _decode(addr, valid_size, layouts)
        best_per_call = min(best_per_call, (time.perf_counter() - t0) / iters)

    return {
        "records_per_s": n_records / best_per_call,
        "mb_per_s": (valid_size / 1e6) / best_per_call,
        "ns_per_record": best_per_call / n_records * 1e9,
        "us_per_buffer": best_per_call * 1e6,
        "record_size": valid_size // n_records,
    }


# Field selections -> the kernel field-id lists that drive the layout width.
_PROFILER_NUMERIC = [
    int(Kernel.START),
    int(Kernel.END),
    int(Kernel.DEVICE_ID),
    int(Kernel.CONTEXT_ID),
    int(Kernel.STREAM_ID),
    int(Kernel.CORRELATION_ID),
    int(Kernel.GRAPH_NODE_ID),
    int(Kernel.GRAPH_ID),
]
_SELECTIONS = {
    "timer (KIND/START/END)": [int(Kernel.START), int(Kernel.END)],
    "profiler numeric (9 fields)": _PROFILER_NUMERIC,
    "profiler + NAME str (10 fields)": _PROFILER_NUMERIC + [_NAME_ID],
}


def _fmt(m: dict[str, float]) -> str:
    return (
        f"{m['records_per_s'] / 1e6:7.2f} M rec/s  "
        f"{m['mb_per_s'] / 1e3:6.2f} GB/s  "
        f"{m['ns_per_record']:7.1f} ns/rec  "
        f"{m['us_per_buffer']:8.1f} us/buf  "
        f"({m['record_size']}B/rec)"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--iters", type=int, default=50, help="decode calls per rep")
    ap.add_argument("--reps", type=int, default=5, help="reps (best is reported)")
    ap.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=[256, 1024, 4096, 16384, 65536, 262144],
        help="records-per-buffer points for the roofline sweep",
    )
    ap.add_argument("--json", type=str, default=None, help="write results as JSON")
    args = ap.parse_args()

    results: dict[str, Any] = {"roofline": {}, "strategy": {}, "selection": {}}

    # --- 1. Roofline: records/s vs records-per-buffer (single kind). Two curves:
    # the vectorized numeric path (shows the small-buffer overhead -> steady-state
    # rise) and the NAME-deref path (the per-record floor the profiler actually pays).
    print("\n=== Roofline: throughput vs records-per-buffer (single kind) ===")
    for label, fids in (
        ("profiler numeric", _PROFILER_NUMERIC),
        ("profiler + NAME", _PROFILER_NUMERIC + [_NAME_ID]),
    ):
        print(f"  -- {label} --")
        layouts = {_KERNEL: _layout(fids)}
        results["roofline"][label] = []
        for n in args.sizes:
            keep, addr, vs = _build_buffer(layouts, [_KERNEL], n)
            m = _time_decode(addr, vs, layouts, n, iters=args.iters, reps=args.reps)
            del keep
            print(f"    {n:>8} rec/buf : {_fmt(m)}")
            results["roofline"][label].append({"records": n, **m})

    # --- 2. Decode strategy (at a fixed steady-state size), profiler numeric fields.
    n = 65536
    print(f"\n=== Decode strategy ({n} rec/buf, profiler numeric fields) ===")
    # Hold the field selection identical across all three so only the demux
    # strategy differs: the second kind has the SAME fields, and for the variable
    # case only its record_size differs (16B of trailing pad) to force the walk.
    ker = _layout(_PROFILER_NUMERIC)
    mc_same = (ker[0], ker[1])  # same size -> KIND-dispatch path
    mc_padded = (ker[0] + 16, ker[1])  # different size -> per-record walk
    strategies = {
        "single kind (stride)": ({_KERNEL: ker}, [_KERNEL]),
        "uniform size (KIND dispatch)": (
            {_KERNEL: ker, _MEMCPY: mc_same},
            [_KERNEL, _MEMCPY],
        ),
        "variable size (per-record walk)": (
            {_KERNEL: ker, _MEMCPY: mc_padded},
            [_KERNEL, _MEMCPY],
        ),
    }
    for name, (layouts, cycle) in strategies.items():
        keep, addr, vs = _build_buffer(layouts, cycle, n)
        m = _time_decode(addr, vs, layouts, n, iters=args.iters, reps=args.reps)
        del keep
        print(f"  {name:<34}: {_fmt(m)}")
        results["strategy"][name] = m

    # --- 3. Field selection (single kind, steady-state size).
    print(f"\n=== Field selection ({n} rec/buf, single kind) ===")
    for name, fids in _SELECTIONS.items():
        layouts = {_KERNEL: _layout(fids)}
        keep, addr, vs = _build_buffer(layouts, [_KERNEL], n)
        m = _time_decode(addr, vs, layouts, n, iters=args.iters, reps=args.reps)
        del keep
        print(f"  {name:<34}: {_fmt(m)}")
        results["selection"][name] = m

    # --- Roofline summary: the sustained ceiling for each path.
    numeric_peak = max(
        r["records_per_s"] for r in results["roofline"]["profiler numeric"]
    )
    name_peak = max(r["records_per_s"] for r in results["roofline"]["profiler + NAME"])
    print(
        f"\nSustained decode ceiling: {numeric_peak / 1e6:.1f} M records/s numeric, "
        f"{name_peak / 1e6:.1f} M records/s with the NAME deref (the profiler's floor)."
    )

    if args.json:
        with open(args.json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
