from __future__ import annotations

import os
from enum import Enum


class TimingAggregation(str, Enum):
    MEAN = "MEAN"
    MEDIAN = "MEDIAN"


timing_aggregation: TimingAggregation = TimingAggregation(
    os.environ.get("TORCHINDUCTOR_INCREMENTAL_TIMING_AGGREGATION", "MEDIAN").upper()
)


# The event resolver daemon exits after this many seconds with no events.
_IDLE_TIMEOUT_S: int = 600
