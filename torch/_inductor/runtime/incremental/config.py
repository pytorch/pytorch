from __future__ import annotations

_MIN_SAMPLES_BEFORE_FILTER: int = 5

_INITIAL_THRESHOLD: float = 2.5
_THRESHOLD_DECAY_EXP: float = 0.1

_MAX_SAMPLES_PER_LAUNCHER: int = 15

_FORCED_TIMING_ROUNDS: int = 3

_SAMPLING_RATE: int = 5

# The event resolver daemon exits after this many seconds with no events.
_IDLE_TIMEOUT_S: int = 600

# Precomputed threshold scale factors for sample counts 1.._MAX_SAMPLES_PER_LAUNCHER.
# Index i corresponds to sample_count == i+1.  Avoids repeated pow() calls.
_THRESHOLD_MULTIPLIERS: tuple[float, ...] = tuple(
    1.0 - ((n - 1) / (_MAX_SAMPLES_PER_LAUNCHER - 1)) ** _THRESHOLD_DECAY_EXP
    for n in range(1, _MAX_SAMPLES_PER_LAUNCHER + 1)
)
