"""Explicitly bracket a raw kernel launch so the CUPTI monitor's ``CommsObserver``
records it as a collective.

The :class:`CommMonitorHook` auto-tags torchcomms collectives, but kernels that bypass
the torch dispatcher -- e.g. a raw ``@triton.jit`` symmetric-memory kernel launched via
``kernel[grid](...)`` (``triton_symm_mem_barrier``, all-gather) -- are invisible to both
the dispatcher and Triton's launch hook (which sees only the name). For those, wrap the
launch in :meth:`CollectiveMeasurer.measure`; it does exactly what the hook's eager /
graph paths do:

* eager: push an external-correlation id, record the collective's metadata under it
  (this is what marks the id a collective for the observer), record a start CUDA event
  (the observer's ``on_start`` signal), and pop on exit. CUPTI tags the wrapped launch's
  kernel with the id, so the observer joins it.
* graph capture: delegate to the :class:`_GraphCommAnchor` (start event off the
  collective's critical path via :class:`GraphSideStream` + a capture node-walk for the
  kernel's ``graph_node_id``s -- the same :class:`_CollectiveKernelScope` the hook uses);
  no external id is pushed (graph collectives are keyed by ``graph_node_id``).

The observer keeps a kernel only when its external id / graph node is marked a collective
(see ``CommsObserver`` -- there is no kernel-name heuristic), which is exactly what this
does; metadata is whatever the caller supplies (there is no NCCL plugin descriptor).
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any

from torch.utils._python_dispatch import TorchDispatchMode


class CollectiveMeasurer:
    """Brackets raw kernel launches as collectives for a ``CommsObserver``. Hold one per
    process (it owns a reusable start event); pass the same ``_GraphCommAnchor`` the
    comms setup wired into the observer's ``event_resolver`` so graph-captured launches
    are counted, or leave ``anchor=None`` for eager-only use."""

    def __init__(self, monitor: Any, *, anchor: Any = None) -> None:
        self._monitor = monitor
        self._anchor = anchor
        self._start_event: Any = None

    @contextmanager
    def measure(self, name: str, **metadata: Any):
        """Bracket the wrapped launch as collective ``name`` with ``metadata`` (e.g.
        ``dtype=..., numel=...``). Eager or graph-capture is detected automatically."""
        import torch

        monitor = self._monitor
        if monitor is None:
            yield
            return
        if self._anchor is not None and torch.cuda.is_current_stream_capturing():
            with self._anchor.collective(
                torch.cuda.current_stream(), {"name": name, **metadata}
            ):
                yield
            return
        eid = monitor.push_external_correlation_id()
        if eid is None:
            # Monitor not collecting (no subscriber / not started): don't tag, so a
            # process-wide SymmMemDispatchMode is a no-op outside a session and no
            # metadata accumulates. Nothing was pushed, so nothing to pop.
            yield
            return
        monitor.add_collective_metadata(name=name, **metadata)
        self._record_start()
        try:
            yield
        finally:
            monitor.pop_external_correlation_id()

    def _record_start(self) -> None:
        # A reusable CUDA start event on the current stream under the pushed id, so CUPTI
        # emits a CUDA_EVENT the observer maps (by correlation) to fire on_start. No-op
        # without CUDA; mirrors CommMonitorHook._record_eager_start.
        import torch

        if not torch.cuda.is_available():
            return
        if self._start_event is None:
            self._start_event = torch.cuda.Event()
        self._start_event.record()


def _first_symm_mem_tensor(args: Any, kwargs: dict[str, Any]) -> Any:
    """The first symmetric-memory tensor among ``args``/``kwargs`` (incl. list/tuple
    elements), or None. ``is_symm_mem_tensor`` is an O(1), non-collective check."""
    import torch
    from torch.distributed._symmetric_memory import is_symm_mem_tensor

    def scan(vals: Any) -> Any:
        for v in vals:
            if isinstance(v, torch.Tensor):
                if is_symm_mem_tensor(v):
                    return v
            elif isinstance(v, (list, tuple)):
                hit = scan(v)
                if hit is not None:
                    return hit
        return None

    return scan(args) if scan(args) is not None else scan(tuple(kwargs.values()))


def _op_name(func: Any) -> str:
    schema = getattr(func, "_schema", None)
    return schema.name if schema is not None else str(func)


class SymmMemDispatchMode(TorchDispatchMode):
    """Auto-measures *dispatcher* ops that operate on a symmetric-memory tensor (e.g.
    ``torch.ops.symm_mem.*`` collectives, or any aten op reading a symm-mem tensor) as
    collectives, so the ``CommsObserver`` records them with no manual wrapping. Enter it
    on the workload (main) thread while comms monitoring is active::

        with SymmMemDispatchMode(monitor, anchor=anchor):
            ...  # symm_mem ops here are measured automatically

    Raw kernels that bypass the dispatcher (a ``@triton.jit`` symm-mem kernel) are not
    visible here -- wrap those with :meth:`CollectiveMeasurer.measure`."""

    def __init__(self, monitor: Any, *, anchor: Any = None) -> None:
        super().__init__()
        self._measurer = CollectiveMeasurer(monitor, anchor=anchor)

    def __torch_dispatch__(
        self, func: Any, types: Any, args: Any = (), kwargs: Any = None
    ):
        kwargs = kwargs or {}
        sm = _first_symm_mem_tensor(args, kwargs)
        if sm is None:
            return func(*args, **kwargs)
        with self._measurer.measure(
            _op_name(func), dtype=str(sm.dtype), numel=int(sm.numel())
        ):
            return func(*args, **kwargs)


_active_dispatch_mode: SymmMemDispatchMode | None = None


def enable_symm_mem_dispatch(
    monitor: Any, *, anchor: Any = None
) -> SymmMemDispatchMode:
    """Enable :class:`SymmMemDispatchMode` for the rest of the process (this thread), so
    symm-mem dispatcher ops are auto-measured with no per-region wrapping -- call once at
    startup instead of wrapping each step. Idempotent; pair with
    :func:`disable_symm_mem_dispatch`. The mode no-ops while the monitor isn't collecting,
    so it's cheap outside a session, but it does intercept every op on this thread while
    enabled (a per-op dispatch cost) -- only enable it for symm-mem profiling runs."""
    global _active_dispatch_mode
    if _active_dispatch_mode is None:
        mode = SymmMemDispatchMode(monitor, anchor=anchor)
        mode.__enter__()
        _active_dispatch_mode = mode
    return _active_dispatch_mode


def disable_symm_mem_dispatch() -> None:
    """Disable a process-wide mode enabled via :func:`enable_symm_mem_dispatch`. Idempotent."""
    global _active_dispatch_mode
    if _active_dispatch_mode is not None:
        _active_dispatch_mode.__exit__(None, None, None)
        _active_dispatch_mode = None
