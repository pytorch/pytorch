import logging
from typing import Callable

import torch
from torch._inductor.fx_passes.bucketing import (
    bucket_all_gather_by_mb,
    merge_all_gather,
)


logger: logging.Logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def is_graph_input(node: torch.fx.Node) -> bool:
    return node.op == "placeholder"


def is_fsdp_all_gather_wait(wait: torch.fx.Node) -> bool:
    # Assume all_gather_into_tensor input is either graph input
    # or dtype conversion of graph input
    ag_node = wait.args[0]  # type: ignore[arg-type, union-attr]
    return (
        is_graph_input(ag_node.args[0])  # type: ignore[arg-type, union-attr]
        or (  # type: ignore[arg-type, union-attr]
            ag_node.args[0].op == "call_function"  # type: ignore[arg-type, union-attr]
            and ag_node.args[0].target  # type: ignore[arg-type, union-attr]
            == torch.ops.prims.convert_element_type.default  # type: ignore[arg-type, union-attr]
            and is_graph_input(ag_node.args[0].args[0])  # type: ignore[arg-type, union-attr]
        )
    )


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
        filter_wait_node=is_fsdp_all_gather_wait,
    )
    if len(ag_buckets) == 0:
        return
    merge_all_gather(gm, ag_buckets)
