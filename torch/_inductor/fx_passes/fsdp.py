import logging
from typing import Callable

import torch
from torch._inductor.fx_passes.bucketing import (
    bucket_all_gather_by_mb,
    filter_fsdp_all_gather_wait,
    merge_all_gather,
)


logger: logging.Logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def bucket_fsdp_all_gather(
    gm: torch.fx.GraphModule, all_gather_bucket_cap_mb_callback: Callable[[int], float]
) -> None:
    """
    Bucketing pass for SimpleFSDP all_gather ops.

    Attributes:
        gm (torch.fx.GraphModule): Graph module of the graph.
        all_gather_bucket_cap_mb_callback (Callable[[int], float]): callback function that
            takes in bucket id and returns size of a bucket in megabytes.

    Usage:
    ```
    from torch._inductor.fx_passes.bucketing import (
        bucket_all_gather,
        bucket_size_determinator,
    )


    def _bucket_all_gather(graph):
        return bucket_all_gather(graph.owning_module, bucket_size_determinator)


    torch._inductor.config.post_grad_custom_post_pass = _bucket_all_gather
    ```
    """

    ag_buckets = bucket_all_gather_by_mb(
        gm,
        all_gather_bucket_cap_mb_callback,
        filter_wait_node=filter_fsdp_all_gather_wait,
    )
    if len(ag_buckets) == 0:
        return
    merge_all_gather(gm, ag_buckets)
