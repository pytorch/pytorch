"""
Data types and dataclasses for profile analysis.
"""

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from torch._inductor.analysis.device_info import DeviceSpec
from torch.utils._ordered_set import OrderedSet


@dataclass
class ProfileEvent:
    category: str
    key: str
    self_device_time_ms: float
    # the benchmark is run multiple times and we average the count across all the
    # runs. It should be an integer but define a float just in case.
    count: float


@dataclass(frozen=True)
class KernelStats:
    flops: int
    bw: float
    latency: float  # us
    achieved_flops: float
    achieved_bandwidth: float
    bound_type: str = "unknown"  # "compute", "memory", or "unknown"


@dataclass(frozen=False)
class Device:
    name: str
    index: int
    info: Optional[DeviceSpec]
    stats: "KernelNameMap"

    def __repr__(self) -> str:
        return f"Device({self.name}, {self.index}): {self.info}"


@dataclass(frozen=True)
class _IdxEvt:
    name: str
    cat: str
    ts: int
    end_ts: int
    tid: int
    parent: Optional[int]  # index into per-thread array
    idx: int  # global index into self.events


# Type aliases
KernelNameMap = defaultdict[str, OrderedSet[KernelStats]]
DeviceMap = Dict[int, Device]
Table = Tuple[List[str], Dict[str, List[str]]]
