# Copyright (c) 2025-2026, Tri Dao.
"""Intra-kernel trace profiler for CuTe-DSL kernels.

Emits Chrome Trace JSON (compatible with Perfetto / chrome://tracing) from
per-warp instrumentation inserted directly into CuTe-DSL kernels.

Toggle with QUACK_TRACE=1 env var.  When disabled (the default) every trace
call is a compile-time no-op — the JIT never emits any profiling PTX.

Design decisions
----------------
**Two-timer approach (inspired by Triton Proton).**
NVIDIA GPUs expose two timers accessible from PTX:
  - %globaltimer — device-wide, ~1 GHz, synchronized across all SMs.
  - %clock64    — per-SM cycle counter, ~2.1 GHz on H100, *not* synchronized
                  across SMs (confirmed empirically: cross-SM spread > 400M ticks
                  vs ~200 ticks for globaltimer on the same launch).
We read %globaltimer once at init and once at flush (per warp) to anchor each
warp's timeline to a device-wide epoch, then read %clock64 for every event.
This gives us low-overhead per-event timestamps (local SM register read) while
retaining cross-SM comparability.  During post-processing the per-slot pair
    (init_globaltimer, init_clock64) and (final_globaltimer, final_clock64)
auto-calibrates the clock64-to-nanosecond conversion:
    ratio = (final_gt - init_gt) / (final_clk - init_clk)
    event_ns = init_gt + (event_clk - init_clk) * ratio

**Compact events (inspired by ThunderKittens).**
Each event is 8 bytes: a raw 32-bit %clock value and a packed (region_id,
event_type) tag, stored with a single v2.u32 streaming store.  The device
writes the raw clock — no subtraction needed.  The host computes deltas
during post-processing using init_clock from metadata with proper u32
wraparound.  Block and warp identity (constant per slot) are stored once
in per-slot metadata instead of per event.

**Minimal live registers.**
The TraceContext dataclass carries only 3 DSL values across loop iterations:
  - slot_ptr (64-bit)  — base of this warp's interleaved [metadata|events]
  - cnt     (32-bit)  — circular buffer write index
  - is_active (1-bit) — predicate for stores (warp leader AND warp sampling)
init_clk is NOT stored — the device writes raw clock values and the host
subtracts init_clk during post-processing.  This saves one register vs
computing deltas on device.

**Interleaved per-slot layout.**
Each warp's metadata and events are contiguous in memory:
    [meta₀ events₀ | meta₁ events₁ | ...]
This means the device needs only ONE pointer (slot_ptr) instead of separate
metadata and event pointers, saving another register.

**Warp sampling.**
An optional warp_ids parameter restricts profiling to specific warps.
Non-selected warps execute predicated stores that the GPU evaluates to
hardware no-ops — zero store bandwidth and no branch divergence.

**Auto-interned region names.**
ctx.b("mma") / ctx.e("mma") auto-assign integer IDs via a module-level
registry at JIT time.  The host reads the same registry at write_trace time.
No region_names parameter needed on either side.

Usage
-----
Host:
    with TraceSession("trace.json", grid_size=G, block_size=B) as sess:
        my_kernel[grid, block](..., sess.ptr)

Device (safe to call from all lanes):
    ctx = TraceContext.create(trace_ptr)
    ctx.b("load"); ctx.e("load")
    ctx.flush()
"""

from __future__ import annotations

import json
import math
import os
import struct
from typing import Optional
from collections import defaultdict
from dataclasses import dataclass

import torch

import cutlass
import cutlass.cute as cute
from cutlass import Int32, Int64, const_expr
from cutlass.base_dsl.arch import Arch
from cutlass._mlir.dialects import llvm, nvvm
from cutlass.cutlass_dsl import T

from .copy_utils import store, store_v2
from .cute_dsl_utils import ParamsBase

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
QUACK_TRACE_ENV = "QUACK_TRACE"

EVENT_BEGIN = 0
EVENT_END = 1
EVENT_MARK = 2

# Per-event record: (u32 raw_clock, u16 region_id, u16 event_type)
# raw_clock is the 32-bit %clock value at event time; the host converts to
# nanoseconds by subtracting init_clock and applying the calibration ratio.
EVENT_SIZE = 8
EVENT_STRUCT = struct.Struct("<IHH")

# Per-slot metadata written at init and flush:
#   u64 init_globaltimer       (offset  0)
#   u64 init_clock64           (offset  8)
#   u64 final_globaltimer      (offset 16)
#   u64 final_clock64          (offset 24)
#   u32 info                   (offset 32)  — block(lo16) | packed_warp_smid(hi16)
#   u32 cnt                    (offset 36)  — number of events recorded
METADATA_SIZE = 40
METADATA_STRUCT = struct.Struct("<QQQQII")

# Chrome trace color names (conservative set that works in all viewers)
_CNAME_LIST = [
    "thread_state_uninterruptible",
    "thread_state_iowait",
    "thread_state_running",
    "thread_state_runnable",
    "thread_state_sleeping",
    "thread_state_unknown",
    "background_memory_dump",
    "light_memory_dump",
    "detailed_memory_dump",
    "vsync_highlight_color",
    "generic_work",
    "good",
    "bad",
    "terrible",
    "black",
    "grey",
    "white",
    "yellow",
    "olive",
    "rail_response",
    "rail_animation",
    "rail_idle",
    "rail_load",
    "startup",
    "heap_dump_stack_frame",
    "heap_dump_object_type",
    "heap_dump_child_node_arrow",
    "cq_build_running",
    "cq_build_passed",
    "cq_build_failed",
    "cq_build_abandoned",
    "cq_build_attempt_runnig",
    "cq_build_attempt_passed",
    "cq_build_attempt_failed",
    "rail_animate",
    "cq_build_attempt_running",
]


def enabled() -> bool:
    """Check QUACK_TRACE=1.  Evaluated at JIT time so disabled = no codegen."""
    return os.environ.get(QUACK_TRACE_ENV, "") == "1"


# Module-level registry: auto-populated by TraceContext at JIT time,
# read by TraceSession at write_trace time.  No need for the user to
# pass region_names to both sides.
_REGION_REGISTRY: dict[int, str] = {}


def _intern_region(name: str) -> int:
    """Assign a stable integer ID to a region name.  JIT-time only."""
    for rid, n in _REGION_REGISTRY.items():
        if n == name:
            return rid
    rid = len(_REGION_REGISTRY)
    _REGION_REGISTRY[rid] = name
    return rid


def _reset_region_registry():
    """Clear the registry.  Called by TraceContext.create so each kernel starts fresh."""
    _REGION_REGISTRY.clear()


# ---------------------------------------------------------------------------
# Contiguous buffer layout (shared between host and device)
# ---------------------------------------------------------------------------
# Per-slot data is interleaved: metadata followed by events for each slot.
# This means the device only needs ONE pointer per warp (the slot base).
#
#   ┌──────────────────────────────────────┐  slot 0
#   │ metadata (40B) │ events (8B × cap)   │
#   ├──────────────────────────────────────┤  slot 1
#   │ metadata (40B) │ events (8B × cap)   │
#   ├──────────────────────────────────────┤  ...
#   │ ...                                  │
#   └──────────────────────────────────────┘
#
# In u32 elements: slot_size = META_ELEMS + per_warp_cap * EVENT_ELEMS

META_ELEMS = METADATA_SIZE // 4  # 10 u32 elements per slot's metadata
EVENT_ELEMS = EVENT_SIZE // 4  # 2 u32 elements per event


def _slot_size(per_warp_cap: int) -> int:
    """Per-slot size in bytes (metadata + events)."""
    return METADATA_SIZE + per_warp_cap * EVENT_SIZE


def _buf_total_bytes(total_slots: int, per_warp_cap: int) -> int:
    return total_slots * _slot_size(per_warp_cap)


# ---------------------------------------------------------------------------
# Device-side helpers
# ---------------------------------------------------------------------------
# Timer reads use side-effecting inline asm instead of nvvm.read_ptx_sreg_* so
# MLIR CSE cannot merge adjacent reads and erase elapsed time.
# Unpredicated stores use cute.arch.store (which wraps nvvm.store_ext with nice
# pointer/Numeric handling).  Predicated stores still need inline asm since the
# NVVM store op doesn't support PTX predication.


def _read_globaltimer():
    return Int64(
        llvm.inline_asm(
            T.i64(),
            [],
            "mov.u64 $0, %globaltimer;",
            "=l",
            has_side_effects=True,
            is_align_stack=False,
        )
    )


def _read_clock64():
    return Int64(
        llvm.inline_asm(
            T.i64(),
            [],
            "mov.u64 $0, %clock64;",
            "=l",
            has_side_effects=True,
            is_align_stack=False,
        )
    )


def _read_clock():
    return cutlass.Int32(
        llvm.inline_asm(
            T.i32(),
            [],
            "mov.u32 $0, %clock;",
            "=r",
            has_side_effects=True,
            is_align_stack=False,
        )
    )


def _read_smid():
    return cutlass.Int32(nvvm.read_ptx_sreg_smid(T.i32()))


def _gmem_ptr(dtype, addr):
    """Create a cute global-memory pointer from an Int64 address."""
    return cute.make_ptr(dtype, Int64(addr), cute.AddressSpace.gmem)


def _is_warp_leader():
    """Return a DSL predicate for the warp leader thread.

    Uses nvvm.elect_sync() on SM90+ (hardware single-thread election),
    falls back to lane_idx() == 0 on older architectures.
    """
    if cutlass.base_dsl.BaseDSL._get_dsl().get_arch_enum() >= Arch.sm_90:
        if cutlass.const_expr(cutlass.CUDA_VERSION.major) == 12:
            return cutlass.Boolean(nvvm.elect_sync(T.bool()))
        elif cutlass.const_expr(cutlass.CUDA_VERSION.major) == 13:
            return cutlass.Boolean(nvvm.elect_sync())
        else:
            raise ValueError(f"CUDA_VERSION.major must be >= 12, got {cutlass.CUDA_VERSION.major}")
    return cute.arch.lane_idx() == 0


# ---------------------------------------------------------------------------
# Device-side: TraceContext
# ---------------------------------------------------------------------------


@dataclass
class TraceContext(ParamsBase):
    """Per-warp trace recorder for use inside CuTe-DSL kernels.

    Use the ``create`` classmethod (not ``__init__``) to construct.  Named
    regions (ctx.b("mma") / ctx.e("mma")) are resolved to integer IDs at JIT
    time.  Optional warp_ids restricts profiling to specific warps.

    Usage::

        ctx = TraceContext.create(trace_ptr)
        ctx.b("load"); ctx.e("load")
        ctx.flush()
    """

    # Compile-time constants (auto-detected as static by ParamsBase)
    per_warp_cap: int = 0
    warp_ids: tuple | None = None

    # DSL values (auto-serialized by ParamsBase across cutlass.range loops).
    # slot_ptr points to this warp's interleaved [metadata | events] region.
    # Metadata at slot_ptr+0, events at slot_ptr+META_ELEMS.
    slot_ptr: cute.Pointer = None
    cnt: cutlass.Int32 = None
    is_active: cutlass.Boolean = None

    # ── Public factory ──────────────────────────────────────────────────────

    @classmethod
    def create(
        cls,
        buf_ptr: Optional[Int64],
        per_warp_cap: int = 4096,
        warp_ids: tuple[int, ...] | list[int] | None = None,
    ):
        """Create and initialize a TraceContext.  Safe to call from all lanes.

        Only lane 0 (warp leader) performs stores; all other lanes execute
        the arithmetic but skip the writes via predication.  The caller does
        NOT need an ``if is_warp_leader():`` guard.

        Region names are auto-interned by ctx.b("name") / ctx.e("name") via a
        module-level registry — no explicit region_names list needed.
        """
        assert (per_warp_cap & (per_warp_cap - 1)) == 0, "per_warp_cap must be power of 2"
        _reset_region_registry()
        warp_ids = tuple(warp_ids) if warp_ids is not None else None

        if not enabled() or const_expr(buf_ptr is None):
            return cls(
                per_warp_cap=per_warp_cap,
                warp_ids=warp_ids,
                slot_ptr=None,
                cnt=None,
                is_active=None,
            )

        SLOT_ELEMS = META_ELEMS + per_warp_cap * EVENT_ELEMS  # u32 elements per slot

        bdx, bdy, bdz = cute.arch.block_dim()
        warps_per_block = (bdx * bdy * bdz + cute.arch.WARP_SIZE - 1) // cute.arch.WARP_SIZE
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        smid = _read_smid()

        # Linearize blockIdx across all grid dimensions.
        bidx, bidy, bidz = cute.arch.block_idx()
        gdx, gdy, gdz = cute.arch.grid_dim()
        linear_block = bidx + bidy * gdx + bidz * gdx * gdy
        slot = linear_block * warps_per_block + warp_idx

        # Single pointer to this warp's interleaved [metadata | events] region.
        buf = _gmem_ptr(Int32, Int64(buf_ptr))
        slot_ptr = buf + slot * SLOT_ELEMS

        # is_active gates all stores: warp leader only, AND warp sampling if set.
        is_leader = _is_warp_leader()
        if warp_ids is not None:
            is_active = cutlass.Boolean(False)
            for wid in warp_ids:
                is_active = is_active | (warp_idx == wid)
            is_active = is_active & is_leader
        else:
            is_active = is_leader

        # Pack warp + smid into 16 bits: warp[5:0] | smid[15:6]
        packed = (warp_idx & 0x3F) | ((smid & 0x3FF) << 6)
        info = linear_block | (packed << 16)

        # Read timers for metadata (host-side calibration).
        gt = _read_globaltimer()
        clk64 = _read_clock64()

        # Write init metadata at slot_ptr. cnt is written by flush().
        store(slot_ptr, gt, is_active, cop="cs")  # offset 0: init_gt
        store(slot_ptr + 2, clk64, is_active, cop="cs")  # offset 8: init_clk64
        store(slot_ptr + 8, info, is_active, cop="cs")  # offset 32: info

        return cls(
            per_warp_cap=per_warp_cap,
            warp_ids=warp_ids,
            slot_ptr=slot_ptr,
            cnt=Int32(0),
            is_active=is_active,
        )

    def flush(self):
        """Write final timer pair and event count.  Safe to call from all lanes."""
        if self.slot_ptr is None:
            return
        gt = _read_globaltimer()
        clk = _read_clock64()
        store(self.slot_ptr + 4, gt, self.is_active, cop="cs")  # final_gt
        store(self.slot_ptr + 6, clk, self.is_active, cop="cs")  # final_clk
        store(self.slot_ptr + 9, self.cnt, self.is_active, cop="cs")  # cnt

    # ── Recording ───────────────────────────────────────────────────────────

    def _record(self, region_id: int, event_type: int):
        if self.slot_ptr is None:
            return
        clk = _read_clock()  # raw 32-bit clock; host subtracts init_clk
        evt_off = META_ELEMS + (self.cnt & (self.per_warp_cap - 1)) * EVENT_ELEMS
        tag = Int32(region_id) | (Int32(event_type) << 16)
        store_v2(self.slot_ptr + evt_off, clk, tag, self.is_active, cop="cs")
        self.cnt += 1

    # Integer-ID API
    def record_b(self, region_id: int):
        self._record(region_id, EVENT_BEGIN)

    def record_e(self, region_id: int):
        self._record(region_id, EVENT_END)

    def record_m(self, region_id: int):
        self._record(region_id, EVENT_MARK)

    # Named-region API (string → int resolved at JIT time via module registry)
    def b(self, name: str):
        self._record(_intern_region(name), EVENT_BEGIN)

    def e(self, name: str):
        self._record(_intern_region(name), EVENT_END)

    def m(self, name: str):
        self._record(_intern_region(name), EVENT_MARK)


# ═══════════════════════════════════════════════════════════════════════════
# Host-side
# ═══════════════════════════════════════════════════════════════════════════


def _unpack_warp(packed: int) -> int:
    return packed & 0x3F


def _unpack_smid(packed: int) -> int:
    return packed >> 6


@dataclass
class _Event:
    """Reconstructed event with absolute timestamp (nanoseconds)."""

    ts: int
    id: int
    type: int
    block: int
    warp_smid: int

    @property
    def warp(self) -> int:
        return _unpack_warp(self.warp_smid)

    @property
    def smid(self) -> int:
        return _unpack_smid(self.warp_smid)


@dataclass
class _SlotMeta:
    """Per-slot metadata read back from device."""

    init_gt: int
    init_clk: int
    final_gt: int
    final_clk: int
    info: int
    cnt: int

    @property
    def block(self) -> int:
        return self.info & 0xFFFF

    @property
    def warp_smid(self) -> int:
        return (self.info >> 16) & 0xFFFF

    @property
    def init_clk32(self) -> int:
        """Low 32 bits of init_clock64 (%clock at init time)."""
        return self.init_clk & 0xFFFFFFFF

    @property
    def ratio(self) -> float:
        """clock64 ticks → nanoseconds conversion factor for this slot."""
        dclk = self.final_clk - self.init_clk
        return (self.final_gt - self.init_gt) / dclk if dclk > 0 else 1.0

    def clock_to_ns(self, raw_clock32: int) -> float:
        """Convert a raw 32-bit clock value to absolute nanoseconds."""
        delta = (raw_clock32 - self.init_clk32) & 0xFFFFFFFF  # u32 wraparound
        return self.init_gt + delta * self.ratio


@dataclass
class TraceWriteOptions:
    scale: float = 1e-3  # globaltimer is ns; Chrome trace displayTimeUnit is also "ns"
    emit_complete_events: bool = True  # pair B/E into ph:"X" (more robust in viewers)
    group_by_smid: bool = False  # pid = block id (ordered); True = pid = SM id
    emit_summary_json: bool = True
    summary_hist_bins: int = 128


class TraceSession:
    """Host-side profiling session.

    Allocates a single contiguous device buffer, provides one pointer (sess.ptr)
    to pass to the kernel, and writes Chrome Trace JSON on exit.

    Can be used as a context manager for automatic sync + write:

        with TraceSession("trace.json", grid_size=G, block_size=B) as sess:
            my_kernel[grid, block](..., sess.ptr)
        # trace.json written here
    """

    def __init__(
        self,
        path: str | None = None,
        *,
        per_warp_cap: int = 4096,
        grid_size: int = 1,
        block_size: int = 128,
        warp_ids: list[int] | tuple[int, ...] | None = None,
        device: str | torch.device = "cuda",
    ):
        assert (per_warp_cap & (per_warp_cap - 1)) == 0, "per_warp_cap must be power of 2"
        self.path = path
        self.per_warp_cap = per_warp_cap
        self.total_blocks = grid_size
        self.warps_per_block = (block_size + 31) // 32
        self.warp_ids = tuple(warp_ids) if warp_ids is not None else None
        self.device = device

        if not enabled():
            self.d_buf = None
            return

        total_slots = self.total_blocks * self.warps_per_block
        self.d_buf = torch.zeros(
            _buf_total_bytes(total_slots, per_warp_cap),
            dtype=torch.uint8,
            device=device,
        )

    @property
    def ptr(self):
        """Device pointer as Int64, or None when tracing is disabled.
        Pass directly as an Optional[Int64] kernel argument."""
        from cutlass.cutlass_dsl import Int64

        return Int64(self.d_buf.data_ptr()) if self.d_buf is not None else None

    def reset(self):
        if self.d_buf is not None:
            self.d_buf.zero_()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.path and enabled():
            torch.cuda.synchronize()
            self.write_trace(self.path)
        return False

    # ── Read helpers ────────────────────────────────────────────────────────

    def _raw_bytes(self):
        return self.d_buf.cpu().numpy()

    def _read_metadata(self, raw) -> list[_SlotMeta]:
        total_slots = self.total_blocks * self.warps_per_block
        slot_bytes = _slot_size(self.per_warp_cap)
        return [
            _SlotMeta(*METADATA_STRUCT.unpack_from(raw, s * slot_bytes)) for s in range(total_slots)
        ]

    def _read_events(self, raw, metas) -> list[_Event]:
        total_slots = self.total_blocks * self.warps_per_block
        slot_bytes = _slot_size(self.per_warp_cap)
        events = []
        for s in range(total_slots):
            cnt = metas[s].cnt
            n = min(cnt, self.per_warp_cap)
            start = (cnt & (self.per_warp_cap - 1)) if cnt > self.per_warp_cap else 0
            # Events start after metadata within this slot.
            slot_events_off = s * slot_bytes + METADATA_SIZE
            meta = metas[s]
            for i in range(n):
                idx = (start + i) & (self.per_warp_cap - 1)
                raw_clk, eid, etype = EVENT_STRUCT.unpack_from(
                    raw,
                    slot_events_off + idx * EVENT_SIZE,
                )
                events.append(
                    _Event(
                        ts=int(meta.clock_to_ns(raw_clk)),
                        id=eid,
                        type=etype,
                        block=meta.block,
                        warp_smid=meta.warp_smid,
                    )
                )
        events.sort(
            key=lambda ev: (
                ev.ts,
                _unpack_smid(ev.warp_smid),
                ev.block,
                _unpack_warp(ev.warp_smid),
                0 if ev.type == 0 else (1 if ev.type == 2 else 2),
                ev.id,
            )
        )
        return events

    def _region_name(self, rid: int) -> str:
        return _REGION_REGISTRY.get(rid, str(rid))

    # ── Chrome Trace JSON output ────────────────────────────────────────────

    def write_trace(self, path: str, opt: TraceWriteOptions | None = None):
        if not enabled():
            return
        opt = opt or TraceWriteOptions()

        raw = self._raw_bytes()
        metas = self._read_metadata(raw)
        events = self._read_events(raw, metas)
        if not events:
            print("intra_kernel_profiler::trace: 0 events")
            return

        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        min_ts = events[0].ts
        trace_events: list[dict] = []

        # Collect unique pids/tids and build name metadata entries.
        used_pids: set[int] = set()
        used_threads: set[tuple[int, int]] = set()
        block_to_smid: dict[int, int] = {}
        for e in events:
            sm, b, w = e.smid, e.block, e.warp
            block_to_smid.setdefault(b, sm)
            pid = sm if opt.group_by_smid else b
            tid = ((b << 6) | w) if opt.group_by_smid else (w * 32)
            used_pids.add(pid)
            used_threads.add((pid, tid))

        for pid in sorted(used_pids):
            pname = (
                f"SM {pid:03d}"
                if opt.group_by_smid
                else f"SM {block_to_smid.get(pid, 0):03d} Block {pid:04d}"
            )
            trace_events.append(
                {"ph": "M", "name": "process_name", "pid": pid, "tid": 0, "args": {"name": pname}}
            )
            trace_events.append(
                {
                    "ph": "M",
                    "name": "process_sort_index",
                    "pid": pid,
                    "tid": 0,
                    "args": {"sort_index": pid},
                }
            )
        for pid, tid in sorted(used_threads):
            if opt.group_by_smid:
                tname = f"Block {tid >> 6:04d} Warp {tid & 0x3F:02d}"
            else:
                tname = f"Warp {tid // 32:02d}"
            trace_events.append(
                {"ph": "M", "name": "thread_name", "pid": pid, "tid": tid, "args": {"name": tname}}
            )
            trace_events.append(
                {
                    "ph": "M",
                    "name": "thread_sort_index",
                    "pid": pid,
                    "tid": tid,
                    "args": {"sort_index": tid},
                }
            )

        # Convert events to Chrome Trace format.
        if opt.emit_complete_events:
            out_events = self._pair_begin_end(events, opt)
            for ts, dur, pid, tid, rid, kind, b, w, sm in out_events:
                ev = {
                    "name": self._region_name(rid),
                    "pid": pid,
                    "tid": tid,
                    "cname": _CNAME_LIST[rid % len(_CNAME_LIST)],
                    "args": {"sm": sm, "block": b, "warp": w},
                }
                if kind == 0:
                    ev.update(ph="X", ts=(ts - min_ts) * opt.scale, dur=dur * opt.scale)
                else:
                    ev.update(ph="i", s="t", ts=(ts - min_ts) * opt.scale)
                trace_events.append(ev)
        else:
            out_events = []
            for e in events:
                sm, b, w = e.smid, e.block, e.warp
                pid = sm if opt.group_by_smid else b
                tid = ((b << 6) | w) if opt.group_by_smid else (w * 32)
                ph = "B" if e.type == 0 else ("E" if e.type == 1 else "i")
                ev = {
                    "name": self._region_name(e.id),
                    "ph": ph,
                    "ts": (e.ts - min_ts) * opt.scale,
                    "pid": pid,
                    "tid": tid,
                    "cname": _CNAME_LIST[e.id % len(_CNAME_LIST)],
                    "args": {"sm": sm, "block": b, "warp": w},
                }
                if e.type == EVENT_MARK:
                    ev["s"] = "t"
                trace_events.append(ev)

        with open(path, "w") as f:
            json.dump({"displayTimeUnit": "ns", "traceEvents": trace_events}, f)
        print(f"intra_kernel_profiler::trace: {len(events)} events -> {path}")

        if opt.emit_complete_events and opt.emit_summary_json:
            self._write_summary(path, out_events, opt)

    @staticmethod
    def _pair_begin_end(events, opt):
        """Match B/E events into (ts, dur, pid, tid, rid, kind, block, warp, sm) tuples."""
        thread_states: dict[tuple, dict[int, list[int]]] = defaultdict(lambda: defaultdict(list))
        out = []
        for e in events:
            sm, b, w = e.smid, e.block, e.warp
            pid = sm if opt.group_by_smid else b
            tid = ((b << 6) | w) if opt.group_by_smid else (w * 32)
            key = (pid, tid)
            if e.type == EVENT_BEGIN:
                thread_states[key][e.id].append(e.ts)
            elif e.type == EVENT_END:
                stack = thread_states[key][e.id]
                if stack:
                    t0 = stack.pop()
                    if e.ts >= t0:
                        out.append((t0, e.ts - t0, pid, tid, e.id, 0, b, w, sm))
            else:
                out.append((e.ts, 0, pid, tid, e.id, 1, b, w, sm))
        return out

    # ── Summary JSON ────────────────────────────────────────────────────────

    def _write_summary(self, trace_path, out_events, opt):
        base = trace_path.rsplit(".json", 1)[0] if trace_path.endswith(".json") else trace_path
        summary_path = base + "_summary.json"

        region_stats: dict[int, list[float]] = defaultdict(list)
        for ts, dur, pid, tid, rid, kind, b, w, sm in out_events:
            if kind == 0:
                region_stats[rid].append(dur * opt.scale)

        regions = []
        for rid in sorted(region_stats):
            durs = region_stats[rid]
            n = len(durs)
            if n == 0:
                continue
            mean = sum(durs) / n
            min_d, max_d = min(durs), max(durs)
            var_pop = sum((d - mean) ** 2 for d in durs) / n
            var_sample = sum((d - mean) ** 2 for d in durs) / (n - 1) if n > 1 else 0
            cv = math.sqrt(var_sample) / abs(mean) if abs(mean) > 0 and n > 1 else None

            bins = opt.summary_hist_bins or 128
            hist = [0] * bins
            if max_d > min_d:
                for d in durs:
                    hist[
                        min(int(max(0.0, min(1.0, (d - min_d) / (max_d - min_d))) * bins), bins - 1)
                    ] += 1
            else:
                hist[0] = n

            w_bin = (max_d - min_d) / bins if max_d > min_d else 0
            pcts = {}
            for p in (5, 10, 25, 50, 75, 90, 95, 99):
                q, cum, val = p / 100.0, 0.0, min_d
                for i, c in enumerate(hist):
                    prev = cum
                    cum += c / n
                    if cum >= q:
                        prob = c / n
                        frac = max(0.0, min(1.0, (q - prev) / prob)) if prob > 0 else 0
                        val = min_d + w_bin * i + frac * w_bin
                        break
                pcts[f"p{p}"] = val

            regions.append(
                {
                    "region": rid,
                    "name": self._region_name(rid),
                    "count": n,
                    "mean_dur": mean,
                    "cv_dur": cv,
                    "min_dur": min_d,
                    "max_dur": max_d,
                    "var_dur_pop": var_pop,
                    "var_dur_sample": var_sample,
                    "percentiles": pcts,
                    "hist": {
                        "bins": bins,
                        "min": min_d,
                        "max": max_d,
                        "prob": [c / n for c in hist],
                    },
                }
            )

        with open(summary_path, "w") as f:
            json.dump(
                {
                    "trace": trace_path,
                    "displayTimeUnit": "ns",
                    "scale": opt.scale,
                    "blocks": self.total_blocks,
                    "warps_per_block": self.warps_per_block,
                    "per_warp_cap": self.per_warp_cap,
                    "regions": regions,
                },
                f,
                indent=2,
            )
        print(f"intra_kernel_profiler::trace: summary -> {summary_path}")
