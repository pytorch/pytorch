"""
Public common utilities for FSDP.
"""

from enum import Enum, auto


class BackwardPrefetch(Enum):
    """
    This configures explicit backward prefetching. For NCCL backend, any
    collectives, even if issued in different streams, contend for the same
    per-device NCCL stream, which is why the relative order in which the
    collectives are issued matters for overlapping. The different backward
    prefetching settings correspond to different orderings.

    - ``BACKWARD_PRE``: This prefetches the next set of parameters before the
    current set of parameter's gradient computation. This improves backward
    pass throughput by overlapping communication (next all-gather) and
    computation (current gradient computation) but may increase the peak memory
    usage since two sets of parameters are in memory at once.
    - ``BACKWARD_POST``: This prefetches the next set of parameters after the
    current set of parameter's gradient computation. This may improve backward
    pass throughput by overlapping communication (current reduce-scatter) and
    computation (next gradient computation) and does not increase the peak
    memory usage since the current set of parameters are freed before
    prefetching the next set. Specifically, the next all-gather is reordered to
    be before the current reduce-scatter.

    For both modes, the ordering that defines "current" and "next" is not
    always correct in the current implementation, so this may cause some
    performance regression for some models.
    """

    BACKWARD_PRE = auto()
    BACKWARD_POST = auto()
