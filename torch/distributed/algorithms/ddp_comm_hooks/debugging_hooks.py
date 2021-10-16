from typing import Any

import torch
import torch.distributed as dist


def noop_hook(_: Any, bucket: dist.GradBucket) -> torch.futures.Future[torch.Tensor]:
    """
    This DDP communication hook returns the a future that wraps the input,
    so it is a noop that does not incur any communication overheads.

    This hook should **only** be used for headroom analysis of allreduce optimization,
    instead of the normal gradient synchronization.
    For example, if only less than 10% speedup of training time can be observed after this hook is registered,
    it usually implies that allreduce is not a performance bottleneck for this case.
    Such instrumentation can be particularly useful
    if GPU traces cannot be easily retrieved or the trace analysis is complicated
    some factors such as the overlap between allreduce and computation or the desynchronization across ranks.

    Example::
        >>> ddp_model.register_comm_hook(None, noop_hook)
    """
    fut: torch.futures.Future[torch.Tensor] = torch.futures.Future()
    fut.set_result(bucket.buffer())

    return fut
