"""Async workspace compilation protocol for torch.compile.

This module implements the core data structures for an experimental
compilation architecture where independent workspaces communicate
through a shared, append-only delta-encoded whiteboard instead of
triggering full recompilations on guard failures.

The design has three parts:
  1. DeltaEvent   -- a compact description of what changed (shape, guard, etc.)
  2. CompilerWhiteboard -- thread-safe pub/sub log that workspaces post to
  3. WorkspaceContext   -- state machine for one isolated compilation region
"""

from __future__ import annotations

import enum
import logging
import struct
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from collections.abc import Callable

    from torch import fx

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Delta event protocol
# ---------------------------------------------------------------------------

class DeltaEventType(enum.Enum):
    SHAPE_CHANGE = "SHAPE_CHANGE"
    GUARD_FAIL = "GUARD_FAIL"
    FUSION_HINT = "FUSION_HINT"
    WORKSPACE_READY = "WORKSPACE_READY"
    WORKSPACE_FAILED = "WORKSPACE_FAILED"


@dataclass(frozen=True, slots=True)
class DeltaEvent:
    """A single, immutable change record posted to the whiteboard."""

    workspace_id: int
    event_type: DeltaEventType
    # Structured metadata: e.g. {"dim": 0, "old": 16, "new": 32}
    symbolic_constraints: dict[str, Any]
    # Compact binary payload for bulk data (strides, offsets, etc.)
    binary_payload: bytes = b""
    timestamp: float = field(default_factory=time.monotonic)

    def encode_shape_delta(self) -> bytes:
        """Encode symbolic_constraints as a compact binary delta.

        Format: repeated (dim_index: uint16, old_size: int64, new_size: int64)
        """
        parts: list[bytes] = []
        if "dim" in self.symbolic_constraints:
            dim = self.symbolic_constraints["dim"]
            old = self.symbolic_constraints.get("old", 0)
            new = self.symbolic_constraints.get("new", 0)
            parts.append(struct.pack("<Hqq", dim, old, new))
        return b"".join(parts)

    @staticmethod
    def decode_shape_delta(data: bytes) -> list[dict[str, int]]:
        """Decode binary shape delta back into a list of dim changes."""
        entry_size = struct.calcsize("<Hqq")
        results: list[dict[str, int]] = []
        for offset in range(0, len(data), entry_size):
            dim, old, new = struct.unpack_from("<Hqq", data, offset)
            results.append({"dim": dim, "old": old, "new": new})
        return results


# ---------------------------------------------------------------------------
# Compiler whiteboard (pub/sub event log)
# ---------------------------------------------------------------------------

# Callback type: (event) -> None
_SubscriberCallback = "Callable[[DeltaEvent], None]"


class CompilerWhiteboard:
    """Thread-safe, append-only event log with pub/sub.

    Workspaces publish DeltaEvents here; subscribers get notified
    asynchronously. The log is bounded to prevent unbounded memory
    growth in long-running sessions.
    """

    def __init__(self, max_log_size: int = 4096) -> None:
        self._log: deque[DeltaEvent] = deque(maxlen=max_log_size)
        self._lock = threading.Lock()
        self._subscribers: dict[DeltaEventType, list[Callable[[DeltaEvent], None]]] = (
            defaultdict(list)
        )
        self._global_subscribers: list[Callable[[DeltaEvent], None]] = []

    def publish(self, event: DeltaEvent) -> None:
        """Append event and notify subscribers. Non-blocking for publisher."""
        with self._lock:
            self._log.append(event)
            callbacks = list(self._subscribers.get(event.event_type, []))
            global_cbs = list(self._global_subscribers)

        log.debug(
            "Whiteboard: ws=%d type=%s constraints=%s",
            event.workspace_id,
            event.event_type.value,
            event.symbolic_constraints,
        )

        for cb in callbacks:
            try:
                cb(event)
            except Exception:
                log.exception("Subscriber callback failed for %s", event.event_type)
        for cb in global_cbs:
            try:
                cb(event)
            except Exception:
                log.exception("Global subscriber callback failed")

    def subscribe(
        self,
        event_type: DeltaEventType,
        callback: Callable[[DeltaEvent], None],
    ) -> None:
        """Register a callback for a specific event type."""
        with self._lock:
            self._subscribers[event_type].append(callback)

    def subscribe_all(self, callback: Callable[[DeltaEvent], None]) -> None:
        """Register a callback for all event types."""
        with self._lock:
            self._global_subscribers.append(callback)

    def get_events_since(self, timestamp: float) -> list[DeltaEvent]:
        """Return all events with timestamp > given value."""
        with self._lock:
            return [e for e in self._log if e.timestamp > timestamp]

    def get_all_events(self) -> list[DeltaEvent]:
        """Return a snapshot of the full log."""
        with self._lock:
            return list(self._log)

    def clear(self) -> None:
        """Reset the whiteboard (for testing / torch._dynamo.reset)."""
        with self._lock:
            self._log.clear()
            self._subscribers.clear()
            self._global_subscribers.clear()

    def event_count(self, event_type: DeltaEventType | None = None) -> int:
        """Count events, optionally filtered by type."""
        with self._lock:
            if event_type is None:
                return len(self._log)
            return sum(1 for e in self._log if e.event_type == event_type)


# Singleton whiteboard -- created lazily, cleared on torch._dynamo.reset()
_global_whiteboard: CompilerWhiteboard | None = None
_whiteboard_lock = threading.Lock()


def get_whiteboard() -> CompilerWhiteboard:
    """Return the global CompilerWhiteboard singleton."""
    global _global_whiteboard
    with _whiteboard_lock:
        if _global_whiteboard is None:
            _global_whiteboard = CompilerWhiteboard()
        return _global_whiteboard


def reset_whiteboard() -> None:
    """Clear and reset the global whiteboard."""
    global _global_whiteboard
    with _whiteboard_lock:
        if _global_whiteboard is not None:
            _global_whiteboard.clear()
        _global_whiteboard = None


# ---------------------------------------------------------------------------
# Workspace context (state machine for one compilation region)
# ---------------------------------------------------------------------------

class WorkspaceState(enum.Enum):
    CREATED = "CREATED"
    TRACING = "TRACING"
    COMPILING = "COMPILING"
    COMPILED = "COMPILED"
    PATCHING = "PATCHING"
    FAILED = "FAILED"


_workspace_id_counter = 0
_workspace_id_lock = threading.Lock()


def _next_workspace_id() -> int:
    global _workspace_id_counter
    with _workspace_id_lock:
        _workspace_id_counter += 1
        return _workspace_id_counter


class WorkspaceContext:
    """State machine for an isolated compilation region.

    Each workspace owns a subgraph, compiles it independently, and can
    patch its compiled output when the whiteboard signals a shape change
    that affects its inputs.
    """

    def __init__(
        self,
        subgraph: fx.GraphModule,
        example_inputs: list[torch.Tensor],
        whiteboard: CompilerWhiteboard,
        workspace_name: str = "",
    ) -> None:
        self.workspace_id = _next_workspace_id()
        self.workspace_name = workspace_name or f"ws_{self.workspace_id}"
        self.state = WorkspaceState.CREATED
        self.subgraph = subgraph
        self.example_inputs = example_inputs
        self.whiteboard = whiteboard

        # Compiled artifacts
        self._compiled_fn: Callable[..., Any] | None = None
        self._compile_thread: threading.Thread | None = None
        self._compile_error: Exception | None = None

        # Shape tracking for patching
        self._input_shapes: list[list[int]] = [
            list(t.shape) for t in example_inputs if isinstance(t, torch.Tensor)
        ]
        self._patched_shapes: list[list[int]] | None = None

        # Lock for state transitions
        self._lock = threading.Lock()

    @property
    def is_ready(self) -> bool:
        return self.state == WorkspaceState.COMPILED

    @property
    def is_compiling(self) -> bool:
        return self.state in (WorkspaceState.TRACING, WorkspaceState.COMPILING)

    def compile_sync(self, compiler_fn: Callable[..., Any]) -> None:
        """Compile the subgraph synchronously."""
        with self._lock:
            self.state = WorkspaceState.COMPILING
        try:
            self._compiled_fn = compiler_fn(self.subgraph, self.example_inputs)
            with self._lock:
                self.state = WorkspaceState.COMPILED
            self.whiteboard.publish(DeltaEvent(
                workspace_id=self.workspace_id,
                event_type=DeltaEventType.WORKSPACE_READY,
                symbolic_constraints={"name": self.workspace_name},
            ))
        except Exception as e:
            with self._lock:
                self.state = WorkspaceState.FAILED
                self._compile_error = e
            self.whiteboard.publish(DeltaEvent(
                workspace_id=self.workspace_id,
                event_type=DeltaEventType.WORKSPACE_FAILED,
                symbolic_constraints={"error": str(e)},
            ))
            raise

    def compile_async(self, compiler_fn: Callable[..., Any]) -> None:
        """Compile the subgraph in a background thread."""
        def _worker() -> None:
            try:
                self.compile_sync(compiler_fn)
            except Exception:
                log.exception(
                    "Background compilation failed for workspace %s",
                    self.workspace_name,
                )

        self._compile_thread = threading.Thread(
            target=_worker,
            name=f"async_compile_{self.workspace_name}",
            daemon=True,
        )
        with self._lock:
            self.state = WorkspaceState.COMPILING
        self._compile_thread.start()

    def wait_for_compilation(self, timeout: float | None = None) -> bool:
        """Block until background compilation completes. Returns success."""
        if self._compile_thread is not None:
            self._compile_thread.join(timeout=timeout)
        return self.state == WorkspaceState.COMPILED

    def try_patch_shapes(self, new_inputs: list[torch.Tensor]) -> bool:
        """Attempt to patch compiled kernel for new input shapes.

        Only batch-dimension (dim 0) changes are patchable in this PoC.
        Returns True if patching succeeded, False if full recompile needed.
        """
        if self._compiled_fn is None:
            return False

        new_shapes = [list(t.shape) for t in new_inputs if isinstance(t, torch.Tensor)]
        if len(new_shapes) != len(self._input_shapes):
            return False

        # Check that only dim 0 differs
        for old_shape, new_shape in zip(self._input_shapes, new_shapes):
            if len(old_shape) != len(new_shape):
                return False
            for dim_idx in range(len(old_shape)):
                if old_shape[dim_idx] != new_shape[dim_idx] and dim_idx != 0:
                    return False

        with self._lock:
            self.state = WorkspaceState.PATCHING

        # Post the shape change to the whiteboard
        for i, (old_shape, new_shape) in enumerate(
            zip(self._input_shapes, new_shapes)
        ):
            if old_shape[0] != new_shape[0]:
                event = DeltaEvent(
                    workspace_id=self.workspace_id,
                    event_type=DeltaEventType.SHAPE_CHANGE,
                    symbolic_constraints={
                        "input_idx": i,
                        "dim": 0,
                        "old": old_shape[0],
                        "new": new_shape[0],
                    },
                )
                self.whiteboard.publish(event)

        # Update tracked shapes -- the compiled_fn with dynamic=True
        # already handles varying batch dims via symbolic shapes
        self._patched_shapes = new_shapes

        with self._lock:
            self.state = WorkspaceState.COMPILED

        return True

    def execute(self, *args: Any) -> Any:
        """Run the compiled subgraph (or fall back to eager)."""
        if self._compiled_fn is not None and self.state == WorkspaceState.COMPILED:
            return self._compiled_fn(*args)
        # Eager fallback
        return self.subgraph(*args)

    def invalidate(self) -> None:
        """Mark workspace for full recompile on next execution."""
        with self._lock:
            self._compiled_fn = None
            self.state = WorkspaceState.CREATED

