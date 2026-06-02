# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.


"""In-process CUPTI activity multiplexer (v2 user-defined records).

CUPTI allows a single subscriber per process, and the v2 subscriber-scope
user-defined-records API lets that subscriber collect a *selected set of
fields* per activity kind (a compact record, vs. the ~200 B full record).
This module owns that one subscriber and lets multiple in-process
**observers** share it:

  * Each observer registers the fields it needs per activity kind
    (`{kind: {field_ids}}`, or ``"all"`` for the kind's full field set).
  * The mux configures the underlying user-defined record as the **union**
    of all observers' fields per kind. So if observer A wants ``{f1}`` and
    observer B wants ``{f1, f2}`` of kind ``k``, CUPTI collects ``{f1, f2}``
    once.
  * On drain the mux parses each buffer **once** (driven by CUPTI's
    per-buffer record layout) and **demuxes** column views back to each
    observer -- A gets ``f1``, B gets ``f1, f2``.

Registration is dynamic: a transient observer (e.g. a profiler enabling
itself for one step) can register a large field set, then unregister; the
union shrinks back and the per-record byte cost drops to the always-on
level. (Validated: ``experiments/probe_dynamic_fields.py``.)

Requires libcupti >= 13.2 (130200) with the v2 subscriber-scope APIs;
there is no v1 fallback -- the mux is v2-only under the hood, and "full
record" is just an observer selecting every field of a kind.

This is deliberately framework-agnostic (no llama4x imports) so it can be
lifted into a torch-native CUPTI monitor: the same union/demux layer is
what makes such a monitor cheap enough for always-on timing.
"""

from __future__ import annotations

import ctypes
import logging
import os
import threading
import time
from typing import Any, Callable

# Activity kind / field-id constants live in `types` (cupti.types.*).
from torch.profiler.cupti.types import FIELD_REGISTRY, MAX_FIELD_ID, STRING_FIELDS

logger: logging.Logger = logging.getLogger(__name__)

_CUPTI_SUCCESS: int = 0
_MIN_VERSION: int = 130200


# --- v2 user-defined-records ctypes structs --------------------------------
class _UDFieldLayoutEntry(ctypes.Structure):
    _fields_ = [
        ("structSize", ctypes.c_size_t),
        ("fieldId", ctypes.c_int),
        ("offset", ctypes.c_size_t),
        ("size", ctypes.c_size_t),
        ("alignment", ctypes.c_size_t),
    ]


class _UDRecordLayout(ctypes.Structure):
    _fields_ = [
        ("structSize", ctypes.c_size_t),
        ("pEntries", ctypes.POINTER(_UDFieldLayoutEntry)),
        ("numFields", ctypes.c_size_t),
        ("recordSize", ctypes.c_size_t),
    ]


class _UDCompleteInfo(ctypes.Structure):
    _fields_ = [
        ("structSize", ctypes.c_size_t),
        ("threadId", ctypes.c_uint64),
        ("ppRecordLayouts", ctypes.POINTER(ctypes.POINTER(_UDRecordLayout))),
        ("numRecordLayouts", ctypes.c_size_t),
    ]


class _UDFieldSelection(ctypes.Structure):
    _fields_ = [
        ("structSize", ctypes.c_size_t),
        ("numFields", ctypes.c_size_t),
        ("pFieldIds", ctypes.POINTER(ctypes.c_int)),
    ]


class _UDActivityConfig(ctypes.Structure):
    _fields_ = [
        ("structSize", ctypes.c_size_t),
        ("fieldSelection", _UDFieldSelection),
    ]


_UD_REQUEST_FN = ctypes.CFUNCTYPE(
    None,
    ctypes.POINTER(ctypes.POINTER(ctypes.c_uint8)),
    ctypes.POINTER(ctypes.c_size_t),
    ctypes.POINTER(ctypes.c_size_t),
    ctypes.c_void_p,
)
_UD_COMPLETE_FN = ctypes.CFUNCTYPE(
    None,
    ctypes.POINTER(ctypes.c_uint8),
    ctypes.c_size_t,
    ctypes.c_size_t,
    ctypes.POINTER(_UDCompleteInfo),
)

# Buffer size handed to CUPTI (bytes). 8 MB matches upstream cupti samples.
_ACTIVITY_BUFFER_BYTES: int = 8 * 1024 * 1024


def _resolve_v2_libcupti() -> "ctypes.CDLL | None":
    """Return a CDLL for a loaded libcupti that has the v2 user-defined API
    and version >= 13.2. Scans ALL mapped libcupti (the process may have
    more than one -- e.g. torch's 13.1 and the wheel's 13.3 share the
    soname; load order decides which is primary) and picks a v2-capable
    one rather than blindly taking the first map entry."""
    paths: list[str] = []
    try:
        with open("/proc/self/maps") as f:
            for line in f:
                if "libcupti.so" in line:
                    pth = line.split()[-1]
                    if pth not in paths:
                        paths.append(pth)
    except OSError:
        return None
    req = (
        "cuptiActivitySetAttribute_v2",
        "cuptiActivityRegisterCallbacks_v2",
        "cuptiActivityEnable_v2",
        "cuptiActivityDisable_v2",
    )
    for pth in paths:
        try:
            lib = ctypes.CDLL(pth)
        except OSError:
            continue
        if not all(hasattr(lib, s) for s in req):
            continue
        ver = ctypes.c_uint32()
        if lib.cuptiGetVersion(ctypes.byref(ver)) != _CUPTI_SUCCESS:
            continue
        if ver.value < _MIN_VERSION:
            continue
        return lib
    return None


def _deref_cstr(ptr: int) -> str:
    """Read a NUL-terminated C string at ``ptr`` (a ``const char*`` from a CUPTI
    record, e.g. the kernel name). Must run while the owning buffer is still
    alive -- i.e. inside the buffer-completed callback -- since CUPTI frees the
    string afterwards. Returns "" for a null pointer."""
    if not ptr:
        return ""
    return ctypes.string_at(ptr).decode("utf-8", "replace")


def _subset_sum_indices(weights: "list[int]", target: int) -> "list[int] | None":
    """Indices of a subset of ``weights`` summing exactly to ``target`` (or
    None). Inputs are a handful of discovered filler-field widths, so a small
    DP over reachable sums is plenty."""
    if target == 0:
        return []
    reach: dict[int, list[int]] = {0: []}
    for i, w in enumerate(weights):
        for s in list(reach.keys()):
            ns = s + w
            if 0 < ns <= target and ns not in reach:
                reach[ns] = reach[s] + [i]
    return reach.get(target)


# CUPTI user-defined records are 8-byte aligned (verified: a 28 B field sum
# yields a 32 B record, 36 B -> 40 B), so a record's size is its field-width sum
# rounded up to a multiple of this.
_RECORD_ALIGN = 8


def _plan_padding(
    union: "dict[int, frozenset[int]]",
    fsize_seen: "dict[int, dict[int, int]]",
    candidates: "dict[int, tuple[int, ...]]",
) -> "dict[int, frozenset[int]]":
    """Compute the field selection to enable so every kind in ``union`` reaches
    the same record size, sizing fillers by a SUBSET-SUM over field widths
    DISCOVERED from CUPTI's layout (``fsize_seen``: kind -> {field id: byte
    size}). This lets a multi-kind buffer take the vectorized stride+dispatch
    decode instead of the per-record walk. Returns one of:

      * ``union`` unchanged -- observer widths not all discovered yet, already
        uniform, or no subset of available fillers closes a kind's gap;
      * a *discovery* selection (``union`` + every candidate filler id for the
        narrow kinds) when candidate widths aren't known yet -- enabling it
        makes the next buffer reveal them (CUPTI reports size 0 for non-fields);
      * a *padded* selection (``union`` + the subset-sum of fillers) once widths
        are known, so records are uniform.

    A kind's base size is computed from the widths of its OWN selected fields
    (KIND + ``union[kind]``), NOT the live recordSize -- which would be polluted
    while a kind is transiently widened for discovery. ``gap`` is then a
    difference of two 8-aligned sizes (a multiple of 8); adding fillers whose
    widths sum to exactly ``gap`` shifts the aligned size by exactly ``gap``."""
    if len(union) <= 1:
        return union

    def base_size(kind: int) -> "int | None":
        # KIND (id 0) is always selected first; include it.
        widths = [fsize_seen.get(kind, {}).get(0)]
        widths += [fsize_seen.get(kind, {}).get(f) for f in union[kind]]
        if any(w is None or w == 0 for w in widths):
            return None  # not all selected widths discovered yet
        s = sum(widths)
        return ((s + _RECORD_ALIGN - 1) // _RECORD_ALIGN) * _RECORD_ALIGN

    base: dict[int, int] = {}
    for kind in union:
        b = base_size(kind)
        if b is None:
            return union
        base[kind] = b
    target = max(base.values())
    if all(b == target for b in base.values()):
        return union

    discovery: dict[int, frozenset[int]] = {}
    padded: dict[int, frozenset[int]] = {}
    need_discovery = False
    for kind, fids in union.items():
        gap = target - base[kind]
        if gap == 0:
            # Already at target -- never widen this kind, even during discovery.
            discovery[kind] = padded[kind] = frozenset(fids)
            continue
        cands = [c for c in candidates.get(kind, ()) if c not in fids]
        seen = fsize_seen.get(kind, {})
        # Candidates whose width we've discovered and that are real (size > 0).
        known = [(c, seen[c]) for c in cands if seen.get(c, 0) > 0]
        idx = _subset_sum_indices([w for _, w in known], gap)
        if idx is not None:
            sel = frozenset(set(fids) | {known[i][0] for i in idx})
            discovery[kind] = padded[kind] = sel  # solvable; no widening needed
        elif any(c not in seen for c in cands):
            need_discovery = True  # widen ONLY this kind to learn its widths
            discovery[kind] = frozenset(set(fids) | set(cands))
            padded[kind] = frozenset(fids)
        else:
            return union  # no subset of available fillers hits the gap
    return discovery if need_discovery else padded


class Observer:
    """A registered consumer of demuxed activity records.

    The mux's poll thread invokes ``callback(kind, {field_id: ndarray})``
    for each registered kind, passing the demuxed columns for the records
    in the just-drained buffer -- the same shape a raw CUPTI
    buffer-completed handler would parse, minus the buffer plumbing. The
    callback runs ON THE MUX POLL THREAD, so keep it cheap and
    thread-safe; hand the arrays to your own queue/aggregator for heavier
    work. The arrays are private copies, safe to retain.
    """

    def __init__(
        self,
        wants: dict[int, "frozenset[int]"],
        callback: "Callable[[int, dict[int, Any]], None]",
    ) -> None:
        # Resolved (no "all") {kind: frozenset(field_ids)}.
        self.wants: dict[int, frozenset[int]] = dict(wants)
        self._callback = callback


class CuptiActivityMux:
    """Owns the single CUPTI subscriber and multiplexes v2 user-defined
    activity records to multiple :class:`Observer` s by field union + demux.

    Lifecycle: construct -> ``register(wants, callback)`` (starts the poll
    thread; the callback fires from it each drain) -> ``close()``.
    ``register``/``unregister`` reconfigure the field union on the fly.
    """

    def __init__(self, enable_hes: bool = True, poll_interval_ms: int = 50) -> None:
        self._poll_interval_ms: int = poll_interval_ms
        self._lib: Any = None
        self._cupti: Any = None
        self._sub: Any = None
        self._armed = False
        self._observers: list[Observer] = []
        self._lock = threading.Lock()
        # Current enabled field union per kind (what CUPTI is configured for).
        self._enabled: dict[int, frozenset[int]] = {}
        # When set, the enabled selection is padded with filler fields so all
        # kinds share one recordSize -> multi-kind buffers vectorize. Opt-in
        # (enable_uniform_padding) since it trades buffer bandwidth for speed.
        self._pad_to_uniform: bool = False
        # Per-field byte widths discovered from CUPTI layouts (kind -> {field
        # id: size}); drive padding without hardcoded widths.
        self._fsize_seen: dict[int, dict[int, int]] = {}
        # Buffers handed to CUPTI; kept alive until close.
        self._buffers: list[Any] = []
        self._req_cb = _UD_REQUEST_FN(self._on_request)
        self._comp_cb = _UD_COMPLETE_FN(self._on_complete)
        self._poller: "_Poller | None" = None
        self._time_converter: Any = None

        try:
            # pyrefly: ignore [missing-import]
            from cupti import cupti
        except ImportError as e:
            logger.warning("CuptiActivityMux: cupti-python unavailable (%s).", e)
            return
        self._cupti = cupti
        if enable_hes:
            _enable_hes(cupti)
        try:
            self._sub = cupti.subscribe(_noop_cb, 0)
        except Exception as e:
            logger.warning("CuptiActivityMux: cuptiSubscribe failed (%s).", e)
            self._cupti = None
            return
        lib = _resolve_v2_libcupti()
        if lib is None:
            logger.warning(
                "CuptiActivityMux: no v2-capable libcupti (>= %d) loaded; "
                "disabling.",
                _MIN_VERSION,
            )
            try:
                cupti.unsubscribe(self._sub)
            except Exception:
                pass
            self._sub = None
            self._cupti = None
            return
        self._lib = lib
        self._sub_h = ctypes.c_void_p(int(self._sub))
        for sym in (
            "cuptiActivitySetAttribute_v2",
            "cuptiActivityRegisterCallbacks_v2",
            "cuptiActivityEnable_v2",
            "cuptiActivityDisable_v2",
        ):
            getattr(lib, sym).restype = ctypes.c_int
        if not self._arm(int(cupti.ActivityAttribute.ATTR_USER_DEFINED_RECORDS)):
            self.close()
            return
        self._armed = True
        self._enable_clock_alignment()

    @property
    def available(self) -> bool:
        return self._armed

    # --- registration ------------------------------------------------------
    def register(
        self,
        wants: dict[int, "set[int] | str"],
        callback: "Callable[[int, dict[int, Any]], None]",
    ) -> Observer:
        """Register an observer wanting ``{kind: {field_ids} | "all"}``.

        ``callback(kind, {field_id: ndarray})`` is invoked from the mux poll
        thread each drain with the demuxed columns. Recomputes the field
        union, reconfigures CUPTI as needed, and starts the poll thread on
        first registration."""
        resolved: dict[int, frozenset[int]] = {}
        for kind, fields in wants.items():
            registry = FIELD_REGISTRY.get(kind)
            if registry is None:
                raise ValueError(f"CuptiActivityMux: unsupported activity kind {kind}")
            if fields == "all":
                resolved[kind] = frozenset(registry)
            else:
                bad = set(fields) - registry
                if bad:
                    raise ValueError(
                        f"CuptiActivityMux: unknown fields {bad} for kind {kind}"
                    )
                resolved[kind] = frozenset(fields)
        obs = Observer(resolved, callback)
        with self._lock:
            prospective = [*self._observers, obs]
        # _reconfigure commits the new observer set only after the (disable-
        # flush-)enable, so obs receives no stale pre-registration records.
        self._reconfigure(prospective)
        self.start_poller(self._poll_interval_ms)
        return obs

    def unregister(self, obs: Observer) -> None:
        with self._lock:
            if obs not in self._observers:
                return
            prospective = [o for o in self._observers if o is not obs]
        # obs stays in the live set during _reconfigure's flush, so it still
        # gets its tail records before being dropped.
        self._reconfigure(prospective)

    def _effective_union(
        self, union: "dict[int, frozenset[int]]"
    ) -> "dict[int, frozenset[int]]":
        """The field selection to actually enable: ``union`` padded to a uniform
        recordSize when ``pad_to_uniform`` is on and padding is achievable,
        else ``union`` unchanged (the returned object is ``union`` itself when
        no padding was applied, which callers use to detect that)."""
        if self._pad_to_uniform:
            # Filler candidates per kind = every field id below the scan ceiling
            # the observer isn't already requesting (KIND id 0 excluded); the
            # planner discovers which are real + their widths.
            candidates = {k: range(1, MAX_FIELD_ID) for k in union}
            return _plan_padding(union, self._fsize_seen, candidates)
        return union

    def _union_of(self, observers: "list[Observer]") -> dict[int, frozenset[int]]:
        union: dict[int, set[int]] = {}
        for obs in observers:
            for kind, fields in obs.wants.items():
                union.setdefault(kind, set()).update(fields)
        return {k: frozenset(v) for k, v in union.items()}

    def _reconfigure(self, observers: "list[Observer]") -> None:
        """Bring CUPTI's enabled selection in line with ``observers``' effective
        union, following CUPTI's MANDATORY disable-flush-enable order: a kind
        that is removed or whose field selection changes is disabled, then ALL
        pending records are flushed (so records collected under the old
        selection are drained with their old layout), then the changed/new
        kinds are enabled. The flush runs without _lock held -- it delivers
        through _parse_and_demux, which takes _lock.

        The new observer set is committed only after the enable, so a freshly
        added observer never sees stale pre-reconfigure records and a leaving
        one still gets its tail during the flush. The effective union may be
        padded for uniform recordSize (see _effective_union) when
        pad_to_uniform is on."""
        if not self._lib:
            with self._lock:
                self._observers = list(observers)
            return
        with self._lock:
            target = self._effective_union(self._union_of(observers))
            # Disable kinds that are removed or whose selection changed; their
            # records must be flushed (below) before any re-enable.
            to_disable = [
                k
                for k in list(self._enabled)
                if k not in target or self._enabled[k] != target[k]
            ]
            for kind in to_disable:
                self._lib.cuptiActivityDisable_v2(self._sub_h, ctypes.c_int(kind))
                del self._enabled[kind]
            to_enable = {k: f for k, f in target.items() if self._enabled.get(k) != f}
        # MANDATORY flush between disable and enable (CUPTI requirement) --
        # only when something was disabled; new-kind-only enables need none.
        # Outside _lock to avoid the flush deadlock.
        if to_disable:
            self.poll(force=True)
        with self._lock:
            for kind, fields in to_enable.items():
                rc = self._enable_kind_locked(kind, tuple(sorted(fields)))
                if rc == _CUPTI_SUCCESS:
                    self._enabled[kind] = fields
                else:
                    logger.warning(
                        "CuptiActivityMux: Enable_v2(kind=%d) -> %d", kind, rc
                    )
            self._observers = list(observers)

    def _enable_kind_locked(self, kind: int, field_ids: "tuple[int, ...]") -> int:
        # CUPTI requires *_FIELD_KIND (id 0) to be the first selected field.
        ordered = (0,) + tuple(sorted(f for f in field_ids if f != 0))
        arr = (ctypes.c_int * len(ordered))(*ordered)
        sel = _UDFieldSelection(
            structSize=ctypes.sizeof(_UDFieldSelection),
            numFields=len(ordered),
            pFieldIds=ctypes.cast(arr, ctypes.POINTER(ctypes.c_int)),
        )
        cfg = _UDActivityConfig(
            structSize=ctypes.sizeof(_UDActivityConfig), fieldSelection=sel
        )
        return self._lib.cuptiActivityEnable_v2(
            self._sub_h, ctypes.c_int(kind), ctypes.byref(cfg)
        )

    # --- arm / buffers / parse --------------------------------------------
    def _arm(self, attr_userdef: int) -> bool:
        lib = self._lib
        try:
            val = ctypes.c_uint8(1)
            sz = ctypes.c_size_t(1)
            if (
                lib.cuptiActivitySetAttribute_v2(
                    self._sub_h,
                    ctypes.c_int(attr_userdef),
                    ctypes.byref(sz),
                    ctypes.byref(val),
                )
                != _CUPTI_SUCCESS
            ):
                return False
            if (
                lib.cuptiActivityRegisterCallbacks_v2(
                    self._sub_h, self._req_cb, self._comp_cb
                )
                != _CUPTI_SUCCESS
            ):
                return False
        except Exception as e:
            logger.warning("CuptiActivityMux: arm failed (%s).", e)
            return False
        return True

    def _enable_clock_alignment(self) -> None:
        """Align CUPTI record timestamps to torch's approximate clock so
        observers share a time base with the rest of torch profiling.
        Best-effort: requires the torch profiler C bindings."""
        try:
            import torch

            prof = torch._C._profiler
            addr = prof._cupti_approximate_time_callback_address()
            fn = self._lib.cuptiActivityRegisterTimestampCallback
            fn.argtypes = [ctypes.c_void_p]
            fn.restype = ctypes.c_int
            if fn(ctypes.c_void_p(addr)) == _CUPTI_SUCCESS:
                self._time_converter = prof._ApproximateClockToUnixTimeConverter()
        except Exception as e:
            logger.info("CuptiActivityMux: clock alignment unavailable (%s).", e)

    def convert_time(self, value: int) -> int:
        """Convert a record timestamp (approximate-clock ns) to unix-epoch ns;
        identity if clock alignment wasn't established."""
        if value == 0 or self._time_converter is None:
            return value
        return int(self._time_converter.to_unix_ns(int(value)))

    def now_ns(self) -> int:
        """Current CUPTI timestamp (ns), in the same clock domain as the
        record START/END fields delivered to observers -- for stamping
        wall-clock boundaries against record times. Falls back to
        ``time.time_ns()`` if the CUPTI timestamp API is unavailable (CUPTI's
        clock is CLOCK_REALTIME epoch-ns here, which matches)."""
        lib = self._lib
        if lib is not None:
            ts = ctypes.c_uint64(0)
            try:
                fn = getattr(lib, "cuptiGetTimestamp_v2", None)
                if fn is not None and self._sub_h is not None:
                    if fn(self._sub_h, ctypes.byref(ts)) == _CUPTI_SUCCESS:
                        return int(ts.value)
                fn = getattr(lib, "cuptiGetTimestamp", None)
                if fn is not None and fn(ctypes.byref(ts)) == _CUPTI_SUCCESS:
                    return int(ts.value)
            except Exception:
                pass
        return time.time_ns()

    def _on_request(self, pp_buffer, p_size, p_max, _info) -> None:
        buf = (ctypes.c_uint8 * _ACTIVITY_BUFFER_BYTES)()
        with self._lock:
            self._buffers.append(buf)
        pp_buffer[0] = ctypes.cast(buf, ctypes.POINTER(ctypes.c_uint8))
        p_size[0] = _ACTIVITY_BUFFER_BYTES
        p_max[0] = 0

    def _on_complete(self, buffer, _size, valid_size, p_info) -> None:
        """CUPTI worker thread: parse the buffer per its layout and demux
        column views to interested observers. Only the requested columns
        are copied; the buffer itself is read zero-copy."""
        try:
            if valid_size == 0:
                return
            self._parse_and_demux(buffer, int(valid_size), p_info[0])
        except Exception as e:
            logger.warning("CuptiActivityMux: parse failed (%s).", e)

    def _parse_and_demux(self, buffer, valid_size: int, info: Any) -> None:
        import numpy as np

        with self._lock:
            observers = list(self._observers)
        if not observers:
            return

        # Per-kind layout spec (present for every enabled kind, not just the
        # ones in this buffer): kind -> (recordSize, {field_id: (offset, size)}).
        # size is the authoritative field width; fields are read unsigned.
        layouts: dict[int, "tuple[int, dict[int, tuple[int, int]]]"] = {}
        for kind in range(int(info.numRecordLayouts)):
            if kind not in FIELD_REGISTRY or not bool(info.ppRecordLayouts[kind]):
                continue
            lay = info.ppRecordLayouts[kind][0]
            rsz = int(lay.recordSize)
            if rsz <= 0:
                continue
            fields = {
                lay.pEntries[i].fieldId: (
                    int(lay.pEntries[i].offset),
                    int(lay.pEntries[i].size),
                )
                for i in range(int(lay.numFields))
            }
            layouts[kind] = (rsz, fields)
        if not layouts:
            return

        # Record per-field widths discovered from CUPTI's layout, so uniform-
        # padding can size fillers without hardcoded widths (see _plan_padding).
        # Monotonic; lockless write is fine.
        for _k, (_rsz, _fl) in layouts.items():
            d = self._fsize_seen.setdefault(_k, {})
            for _fid, (_off, _sz) in _fl.items():
                d[_fid] = _sz

        addr = ctypes.cast(buffer, ctypes.c_void_p).value
        raw = np.ctypeslib.as_array((ctypes.c_uint8 * valid_size).from_address(addr))

        # Record start offsets per kind. A buffer holds records of (possibly)
        # multiple kinds, each beginning with *_FIELD_KIND (id 0) at offset 0
        # and sized by its kind's recordSize.
        #
        # Two vectorizable cases avoid the per-record Python walk:
        #   * One enabled kind => homogeneous buffer; starts are a simple
        #     stride. (register()/unregister() flush before entering a
        #     single-kind era, so the buffer is guaranteed contiguous.)
        #   * All enabled kinds share one recordSize => records still tile at
        #     a fixed stride even though kinds vary; we stride to the starts,
        #     gather the 4-byte KIND column at each, and group by kind with
        #     boolean masks. Exact (no resync guess) -- the size is uniform,
        #     so a record's kind never shifts the next record's offset.
        # Only genuinely variable-size, multi-kind buffers fall back to the
        # sequential walk (CUPTI records aren't self-synchronizing, so the
        # boundaries can't be found without reading each record's KIND).
        rszs = {rsz for rsz, _ in layouts.values()}
        positions: dict[int, Any] = {}
        if len(layouts) == 1:
            ((kind, (rsz, _)),) = layouts.items()
            n = valid_size // rsz
            if n:
                positions[kind] = np.arange(n, dtype=np.int64) * rsz
        elif len(rszs) == 1:
            rsz = next(iter(rszs))
            n = valid_size // rsz
            if n:
                starts = np.arange(n, dtype=np.int64) * rsz
                kinds_col = (
                    raw[starts[:, None] + np.arange(4)].copy().view("<u4").ravel()
                )
                for kind in layouts:
                    sel = starts[kinds_col == kind]
                    if sel.size:
                        positions[kind] = sel
        else:
            pos_lists: dict[int, list[int]] = {k: [] for k in layouts}
            pos = 0
            while pos + 4 <= valid_size:
                kind = int(raw[pos : pos + 4].view("<u4")[0])
                ent = layouts.get(kind)
                if ent is None:
                    break  # unknown kind: can't determine its size, stop
                pos_lists[kind].append(pos)
                pos += ent[0]
            positions = {
                k: np.array(v, dtype=np.int64) for k, v in pos_lists.items() if v
            }

        for kind, pos_arr in positions.items():
            interested = [o for o in observers if kind in o.wants]
            if not interested:
                continue
            fields = layouts[kind][1]
            str_fields = STRING_FIELDS.get(kind, frozenset())
            wanted: set[int] = set()
            for o in interested:
                wanted |= set(o.wants[kind])
            cols: dict[int, Any] = {}
            for fid in wanted:
                ent = fields.get(fid)
                if ent is None:
                    continue
                off, size = ent
                if fid in str_fields and size == 8:
                    # const char* field: deref each pointer to a Python str NOW,
                    # while the buffer (and the strings it points to) is alive.
                    ptrs = (
                        raw[pos_arr[:, None] + np.arange(off, off + 8)]
                        .copy()
                        .view("<u8")
                        .ravel()
                    )
                    cols[fid] = np.array(
                        [_deref_cstr(int(p)) for p in ptrs], dtype=object
                    )
                    continue
                if size not in (1, 2, 4, 8):
                    continue
                idx = pos_arr[:, None] + np.arange(off, off + size)
                cols[fid] = raw[idx].copy().view(f"<u{size}").ravel()
            for o in interested:
                chunk = {fid: cols[fid] for fid in o.wants[kind] if fid in cols}
                if not chunk:
                    continue
                try:
                    o._callback(kind, chunk)
                except Exception as e:
                    logger.warning(
                        "CuptiActivityMux: observer callback failed (%s).", e
                    )

    # --- draining ----------------------------------------------------------
    def poll(self, force: bool = False) -> None:
        """Flush CUPTI so completed buffers are delivered + demuxed. Use
        ``force=True`` (FLUSH_FORCED) at a step boundary / on shutdown;
        the default best-effort flush avoids draining the partial buffer."""
        if not self._armed or self._cupti is None:
            return
        self._cupti.activity_flush_all(1 if force else 0)

    def force_drain(self) -> None:
        """Force-flush CUPTI now so all completed records are parsed and
        delivered to observers' callbacks before the caller reads them. An
        observer should call this at the start of its ``drain()`` -- BEFORE
        taking its own lock, since the flush delivers synchronously through
        the callback (taking the lock first would deadlock)."""
        self.poll(force=True)

    def enable_uniform_padding(self, enabled: bool = True) -> None:
        """Opt into (or out of) padding the enabled selection so all kinds
        share one recordSize, letting multi-kind buffers take the vectorized
        stride+dispatch decode instead of the per-record walk -- at the cost of
        extra buffer bandwidth (the filler fields). Best-effort: padding only
        applies when achievable (some kind may lack spare fields), and the
        decoder still verifies actual recordSize uniformity before striding.
        Reconfigures via the disable-flush-enable path so no buffer straddles
        the selection change."""
        with self._lock:
            if self._pad_to_uniform == enabled:
                return
            self._pad_to_uniform = enabled
            armed = self._armed
            observers = list(self._observers)
        if not armed:
            return
        self._reconfigure(observers)

    def _maybe_recalibrate(self) -> None:
        """Driven by the poll thread: once uniform padding is on, re-evaluate
        the effective selection as discovered field sizes accrue (union ->
        discovery-widened -> minimally padded) and reconfigure whenever it
        changes, until it stabilizes. No-op when padding is off or nothing
        changed."""
        if not self._pad_to_uniform or not self._armed:
            return
        with self._lock:
            observers = list(self._observers)
            changed = self._effective_union(self._union_of(observers)) != self._enabled
        if changed:
            self._reconfigure(observers)

    def start_poller(self, interval_ms: int) -> None:
        if self._poller is None and self._armed:
            self._poller = _Poller(self, interval_ms)
            self._poller.start()

    def close(self) -> None:
        if self._poller is not None:
            self._poller.stop()
            self._poller.join()
            self._poller = None
        if self._armed and self._lib is not None:
            try:
                self.poll(force=True)
            except Exception:
                pass
            with self._lock:
                for kind in list(self._enabled):
                    try:
                        self._lib.cuptiActivityDisable_v2(
                            self._sub_h, ctypes.c_int(kind)
                        )
                    except Exception:
                        pass
                self._enabled.clear()
        self._armed = False
        if self._sub is not None and self._cupti is not None:
            try:
                self._cupti.unsubscribe(self._sub)
            except (AttributeError, TypeError):
                pass
            except Exception as e:
                logger.warning("CuptiActivityMux.close: unsubscribe failed: %s", e)
            self._sub = None
        self._buffers.clear()


class _Poller(threading.Thread):
    """Daemon thread driving ``mux.poll()`` every ``interval_ms``."""

    def __init__(self, mux: CuptiActivityMux, interval_ms: int) -> None:
        super().__init__(name="cupti-mux-poller", daemon=True)
        self._mux = mux
        self._interval_s = max(1, int(interval_ms)) / 1000.0
        self._stop_event = threading.Event()

    def run(self) -> None:
        while not self._stop_event.wait(self._interval_s):
            try:
                self._mux.poll(force=False)
                # Advance uniform-padding calibration as discovered field
                # widths accrue (union -> discovery -> padded), off the
                # training thread.
                self._mux._maybe_recalibrate()
            except Exception as e:
                logger.warning("CuptiActivityMux poller: %s", e)

    def stop(self) -> None:
        self._stop_event.set()


def _noop_cb(_userdata: Any, _domain: int, _cbid: int, _cbdata: Any) -> None:
    return


# Set True once HES hardware kernel-timestamp collection has been turned on
# (process-wide, one-way). HES persists for the session and can't be
# disabled, so this is for observability only.
_HES_ENABLED: bool = False


def _enable_hes(cupti: Any) -> None:
    """Best-effort HES (hardware kernel-timestamp) enable on a THROWAWAY
    subscription, before the mux's real subscribe. HES must not share a
    subscription with user-defined records (NOT_COMPATIBLE) but persists
    process-wide after unsubscribe. Kill switch: TORCH_CUPTI_MUX_DISABLE_HES=1
    (kept for compatibility with the timer's env)."""
    global _HES_ENABLED
    if _HES_ENABLED or os.getenv("TORCH_CUPTI_MUX_DISABLE_HES", "0") == "1":
        return
    enable_hw_trace = getattr(cupti, "activity_enable_hw_trace", None)
    if enable_hw_trace is None:
        return
    sub = None
    try:
        sub = cupti.subscribe(_noop_cb, 0)
        enable_hw_trace(1)
        _HES_ENABLED = True
    except Exception as e:
        logger.info("CuptiActivityMux: HES enable failed (%s).", e)
    finally:
        if sub is not None:
            try:
                cupti.unsubscribe(sub)
            except Exception:
                pass


def hes_enabled() -> bool:
    return _HES_ENABLED


def enable_hes_early() -> None:
    """Arm HES (hardware kernel timestamps) before any CUDA context exists.

    HES only engages if armed before CUDA driver init, so this raises if a
    context already exists -- surfacing the misuse instead of silently
    degrading to software timestamps. Idempotent; best-effort otherwise."""
    import torch

    if torch.cuda.is_initialized():
        raise RuntimeError(
            "enable_hes_early() must be called before CUDA context creation"
        )
    try:
        # pyrefly: ignore [missing-import]
        from cupti import cupti
    except ImportError:
        return
    _enable_hes(cupti)


_INSTANCE: "CuptiActivityMux | None" = None
_INSTANCE_LOCK = threading.Lock()


def instance() -> CuptiActivityMux:
    """Return the process-wide mux (the single CUPTI owner), creating it on
    first use. All observers register on this shared instance."""
    global _INSTANCE
    with _INSTANCE_LOCK:
        if _INSTANCE is None:
            _INSTANCE = CuptiActivityMux()
        return _INSTANCE
