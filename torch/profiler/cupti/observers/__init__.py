# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

"""Observers built on the CUPTI activity mux.

Each observer registers a per-kind field selection with the shared mux
(``torch.profiler.cupti.instance()``) and consumes demuxed records via a
callback that fires on the mux poll thread.
"""

from torch.profiler.cupti.observers.base import MuxObserver
from torch.profiler.cupti.observers.node_timer import NodeTimerObserver
from torch.profiler.cupti.observers.profiler import ProfilerObserver


__all__ = ["MuxObserver", "NodeTimerObserver", "ProfilerObserver"]
