"""
This file includes public utilities for FSDP such as classes for the FSDP
constructor arguments. We keep this file private and only for organization, and
we make the contents public via other files.
"""

from enum import auto, Enum


class BackwardPrefetch(Enum):
    """
    This configures explicit backward prefetching, which can improve throughput
    but may slightly increase peak memory usage.

    For NCCL backend, any collectives, even if issued in different streams,
    contend for the same per-device NCCL stream, which is why the relative
    order in which the collectives are issued matters for overlapping. The
    different backward prefetching settings correspond to different orderings.

    - ``BACKWARD_PRE``: This prefetches the next set of parameters before the
    current set of parameter's gradient computation. This improves backward
    pass throughput by overlapping communication (next all-gather) and
    computation (current gradient computation).
    - ``BACKWARD_POST``: This prefetches the next set of parameters after the
    current set of parameter's gradient computation. This may improve backward
    pass throughput by overlapping communication (current reduce-scatter) and
    computation (next gradient computation). Specifically, the next all-gather
    is reordered to be before the current reduce-scatter.
    """

    # NOTE: For both modes, the ordering that defines "current" and "next" is
    # not always correct in the current implementation, so this may cause some
    # performance regression for some models.
    BACKWARD_PRE = auto()
    BACKWARD_POST = auto()
