
"""In-process CUPTI activity multiplexing for torch.profiler.

``mux`` owns the single CUPTI subscriber and v2 user-defined-records
stream; observers (see ``observers/``) register a field selection per
activity kind and receive demuxed records via a callback driven by the
mux's poll thread. Activity kind / field-id constants are in ``types``
(e.g. ``torch.profiler.cupti.types.KIND_MEMCPY``).
"""

from torch.profiler.cupti import types
from torch.profiler.cupti.mux import (
    CuptiActivityMux,
    enable_hes_early,
    hes_enabled,
    instance,
    Observer,
)
from torch.profiler.cupti.observers.profiler import CuptiProfiler, ProfilerSession


__all__ = [
    "types",
    "CuptiActivityMux",
    "Observer",
    "ProfilerSession",
    "CuptiProfiler",
    "instance",
    "hes_enabled",
    "enable_hes_early",
]
