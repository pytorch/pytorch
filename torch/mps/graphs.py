# torch/mps/graphs.py
#
# User-facing API for MPS command-buffer capture/replay, analogous to
# torch.cuda.graphs. Backed by MTLIndirectCommandBuffer via the C++ binding
# torch._C._MPSStreamGraph.
#
# Usage:
#     g = torch.mps.MPSGraph()
#
#     # Warmup: ensure kernel pipelines are compiled before capture.
#     for _ in range(3):
#         y = model(x)
#
#     # Capture once.
#     with torch.mps.graph(g):
#         y = model(x)
#
#     # Replay many times — bypasses per-call CPU encoding overhead.
#     for _ in range(1000):
#         g.replay()
#         use(y)
#
# Constraints (v1):
#   - Buffers are bound by pointer at capture; reuse requires `.copy_()`
#     into the same storage between replays.
#   - Capture cannot be nested.
#   - Tensor shapes must not change between capture and replays.
#   - Allocation inside the capture region is currently rejected.

import contextlib
from typing import Any

import torch


class MPSGraph:
    """Capture/replay primitive for MPS streams.

    Mirrors :class:`torch.cuda.CUDAGraph`. Use :func:`torch.mps.graph` as a
    context manager rather than calling ``capture_begin``/``capture_end``
    directly.
    """

    def __init__(self, max_commands: int = 4096) -> None:
        self._inner = torch._C._MPSStreamGraph(max_commands)  # type: ignore[attr-defined]

    def capture_begin(self) -> None:
        """Begin recording subsequent MPS dispatches into this graph."""
        self._inner.capture_begin()

    def capture_end(self) -> None:
        """End recording. Graph is now ready to replay."""
        self._inner.capture_end()

    def replay(self) -> None:
        """Re-execute the captured sequence on the same stream.

        Blocks until the replay completes (v1 is synchronous).
        """
        self._inner.replay()

    def num_commands(self) -> int:
        """Number of compute dispatches recorded."""
        return self._inner.num_commands()

    def is_capturing(self) -> bool:
        return self._inner.is_capturing()

    def is_ready(self) -> bool:
        return self._inner.is_ready()


@contextlib.contextmanager
def graph(g: MPSGraph):
    """Context manager for capturing into an :class:`MPSGraph`.

    Example::

        g = torch.mps.MPSGraph()
        x = torch.randn(64, device="mps")
        with torch.mps.graph(g):
            y = x.relu()
        g.replay()

    All MPS dispatches inside the ``with`` block are recorded; non-MPS work
    (CPU compute, host allocations) executes normally and is not captured.
    """
    if not torch.backends.mps.is_available():
        raise RuntimeError("torch.mps.graph: MPS backend not available")
    g.capture_begin()
    try:
        yield g
    finally:
        g.capture_end()


def make_graphed_callables(
    callable_: Any,
    sample_args: tuple,
    *,
    num_warmup_iters: int = 3,
) -> Any:
    """Wrap ``callable_`` so that subsequent calls replay a captured graph.

    Pattern from :func:`torch.cuda.make_graphed_callables`. Captures with
    ``sample_args`` after ``num_warmup_iters`` of eager warmup, then returns
    a wrapper that copies new inputs into the captured input buffers and
    replays.

    .. warning::
        v1 limitation: ``callable_`` must accept exactly the same tensor
        shapes/dtypes/devices on every call. Inputs are bound by pointer
        during capture; the wrapper rebinds them via in-place ``.copy_()``.
    """
    # Warmup so kernel pipelines compile before capture.
    for _ in range(num_warmup_iters):
        callable_(*sample_args)

    g = MPSGraph()
    with graph(g):
        captured_out = callable_(*sample_args)

    def wrapper(*new_args):
        if len(new_args) != len(sample_args):
            raise ValueError(
                f"make_graphed_callables: expected {len(sample_args)} args, "
                f"got {len(new_args)}"
            )
        for sample, new in zip(sample_args, new_args):
            if isinstance(sample, torch.Tensor) and sample is not new:
                sample.copy_(new)
        g.replay()
        return captured_out

    return wrapper


__all__ = ["MPSGraph", "graph", "make_graphed_callables"]
