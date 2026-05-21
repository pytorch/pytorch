from __future__ import annotations

import os
from enum import Enum
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from torch._inductor.runtime.hints import HeuristicType


# Bump this to enable incremental autotuning for users who have
# config.incremental_autotune = True (or the env var set to "1"). The JK
# "pytorch/inductor:incremental_autotune_version" must be <= this value
# for the feature to activate.
_INCREMENTAL_AUTOTUNE_VERSION: int = 1

# How long the timing-resolution background thread waits for new work
# before exiting (it restarts on demand). Higher = stays alive longer
# between sparse autotuning workloads; lower = releases its
# background-thread slot sooner.
_RESOLVER_IDLE_TIMEOUT_S: int = 600


class LauncherTimingAggregation(str, Enum):
    """How a launcher reduces its sample list to a single
    representative timing."""

    MEAN = "MEAN"
    MEDIAN = "MEDIAN"


# How a launcher's accumulated timing samples are reduced to a single
# representative value used for all comparisons. MEDIAN is robust
# against single-iteration outliers; MEAN is simpler and slightly
# cheaper.
launcher_timing_aggregation: LauncherTimingAggregation = LauncherTimingAggregation(
    os.environ.get("TORCHINDUCTOR_INCREMENTAL_TIMING_AGGREGATION", "MEDIAN").upper()
)


# Minimum number of timings a launcher must accumulate before it's
# eligible to be filtered out as too-slow. Too low risks discarding
# launchers whose first few samples were unrepresentatively slow;
# too high delays filtering and wastes dispatches on obvious losers.
min_timings_before_filter: int = int(
    os.environ.get("TORCHINDUCTOR_INCREMENTAL_MIN_SAMPLES_BEFORE_FILTER", "3")
)

# Maximum number of timings collected per launcher before it's
# considered fully measured and eligible for promotion to the
# certified winner. Bigger = more confident timing at the cost of
# more dispatches before convergence.
max_timings_per_launcher: int = int(
    os.environ.get("TORCHINDUCTOR_INCREMENTAL_MAX_SAMPLES_PER_LAUNCHER", "25")
)

# Force a timed dispatch on every call until the launcher has at
# least this many timings; afterwards only ``timed_sampling_rate``
# of dispatches are timed. Ensures enough early samples to make
# filter decisions before falling back to the lower sampling rate.
force_timing_if_lt_n_timings: int = int(
    os.environ.get("TORCHINDUCTOR_INCREMENTAL_FORCED_TIMING_ROUNDS", "3")
)

# Once past the forced-timing window, time only one in N dispatches.
# Higher = less benchmarking overhead in steady state, slower
# convergence to the per-launcher cap.
timed_sampling_rate: int = int(
    os.environ.get("TORCHINDUCTOR_INCREMENTAL_SAMPLING_RATE", "5")
)

# Initial multiplier for the filter check: a candidate launcher is
# discarded when its timing exceeds this factor times the
# best-so-far timing. 1.0 = exact comparison (any slower → discard);
# higher = more tolerance for noise early on.
initial_filter_threshold: float = float(
    os.environ.get("TORCHINDUCTOR_INCREMENTAL_INITIAL_THRESHOLD", "2.5")
)

# Controls how fast the per-launcher filter tolerance tightens as
# more samples accumulate. Lower (closer to 0) = stays relaxed
# longer; higher = tightens faster.
filter_threshold_decay_exp: float = float(
    os.environ.get("TORCHINDUCTOR_INCREMENTAL_THRESHOLD_DECAY_EXP", "0.1")
)

# Per-config async-compile timeout. If a single config doesn't
# compile within this many seconds it's recorded as a failed
# compilation; the underlying compile may still finish in the
# background but its result is discarded.
compile_timeout_s: float = float(
    os.environ.get("TORCHINDUCTOR_INCREMENTAL_COMPILE_TIMEOUT_S", "60")
)


# Kernel heuristic types to include in incremental autotuning. Default is all
# HeuristicType members. Override via the
# ``TORCHINDUCTOR_INCREMENTAL_AUTOTUNE_INCLUDE`` env var as a comma-separated
# list of ``HeuristicType`` names (case-insensitive), e.g.
# ``"pointwise,template"``. Empty string = no types enabled (incremental
# autotune effectively off); unset = default (all).
def _compute_included_heuristic_types() -> frozenset[HeuristicType]:
    from torch._inductor.runtime.hints import HeuristicType

    raw = os.environ.get("TORCHINDUCTOR_INCREMENTAL_AUTOTUNE_INCLUDE")
    if raw is None:
        return frozenset(HeuristicType)
    if not raw:
        return frozenset()
    return frozenset(
        HeuristicType[name.strip().upper()] for name in raw.split(",") if name.strip()
    )


_INCLUDED_HEURISTIC_TYPES: frozenset[HeuristicType] = (
    _compute_included_heuristic_types()
)


def should_include(heuristic_type: HeuristicType) -> bool:
    """Return True if ``heuristic_type`` participates in incremental autotuning."""
    return heuristic_type in _INCLUDED_HEURISTIC_TYPES
