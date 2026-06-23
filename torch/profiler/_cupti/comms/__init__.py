"""Collective (comms) consumers for the in-process CUPTI activity monitor.

The producer -- ``CommsObserver`` and its ``CommRecord`` -- lives in
``torch.profiler._cupti.observers.comms``. This package holds the consumers built on
top of it: the ``CommRecordPlugin`` lifecycle extension point (:mod:`plugin`), the
torchcomms tagging hook (:class:`CommMonitorHook`), the hang detector
(:class:`HangDetectorPlugin`, which consumes the observer's stall heartbeat), and the
``FlightRecorderPlugin`` (:mod:`flight_recorder`), which serializes to the OSS
FlightRecorder schema ``fr_trace`` reads.

Service-specific serializers (e.g. the torchcomms ``clog`` format, the ncclx analyzer
``comm_dump`` format) live with their consumers rather than in core: implement them as
:class:`CommRecordPlugin` subclasses where the format + its tests live.
"""

from torch.profiler._cupti.comms.flight_recorder import FlightRecorderPlugin
from torch.profiler._cupti.comms.hang_detector import HangDetectorPlugin
from torch.profiler._cupti.comms.hook import _GraphCommAnchor, CommMonitorHook
from torch.profiler._cupti.comms.measure import (
    CollectiveMeasurer,
    disable_symm_mem_dispatch,
    enable_symm_mem_dispatch,
    SymmMemDispatchMode,
)
from torch.profiler._cupti.comms.plugin import CommRecordPlugin


__all__ = [
    "CommRecordPlugin",
    "CommMonitorHook",
    "CollectiveMeasurer",
    "SymmMemDispatchMode",
    "enable_symm_mem_dispatch",
    "disable_symm_mem_dispatch",
    "HangDetectorPlugin",
    "FlightRecorderPlugin",
]
