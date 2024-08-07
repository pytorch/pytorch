from typing import Any

import torch
from torch.distributed import GradBucket


__all__ = ["noop_hook"]


def noop_hook(_: Any, bucket: GradBucket) -> torch.futures.Future[torch.Tensor]:
    """
    Return a future that wraps the input, so it is a no-op that does not incur any communication overheads.

    This hook should **only** be used for headroom analysis of allreduce optimization,
    instead of the normal gradient synchronization.
    For example, if only less than 10% speedup of training time can be observed after this hook is registered,
    it usually implies that allreduce is not a performance bottleneck for this case.
    Such instrumentation can be particularly useful
    if GPU traces cannot be easily retrieved or the trace analysis is complicated
    some factors such as the overlap between allreduce and computation or the desynchronization across ranks.

    Example::
        >>> # xdoctest: +SKIP
        >>> ddp_model.register_comm_hook(None, noop_hook)
    """
    fut: torch.futures.Future[torch.Tensor] = torch.futures.Future()
    fut.set_result(bucket.buffer())

    return fut
