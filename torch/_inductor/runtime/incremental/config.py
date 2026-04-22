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


min_samples_before_filter: int = int(
    os.environ.get("TORCHINDUCTOR_INCREMENTAL_MIN_SAMPLES_BEFORE_FILTER", "3")
)

max_samples_per_launcher: int = int(
    os.environ.get("TORCHINDUCTOR_INCREMENTAL_MAX_SAMPLES_PER_LAUNCHER", "25")
)

forced_timing_rounds: int = int(
    os.environ.get("TORCHINDUCTOR_INCREMENTAL_FORCED_TIMING_ROUNDS", "3")
)

sampling_rate: int = int(os.environ.get("TORCHINDUCTOR_INCREMENTAL_SAMPLING_RATE", "5"))

initial_threshold: float = float(
    os.environ.get("TORCHINDUCTOR_INCREMENTAL_INITIAL_THRESHOLD", "2.5")
)

threshold_decay_exp: float = float(
    os.environ.get("TORCHINDUCTOR_INCREMENTAL_THRESHOLD_DECAY_EXP", "0.1")
)
