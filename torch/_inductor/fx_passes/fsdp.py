import logging
from collections.abc import Callable

import torch
from torch._inductor.fx_passes.bucketing import (
    bucket_all_gather_by_mb,
    bucket_reduce_scatter_by_mb,
    BucketMode,
    merge_all_gather,
    merge_reduce_scatter,
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


def is_graph_output(node: torch.fx.Node) -> bool:
    return all(user.op == "output" for user in node.users)


def is_fsdp_reduce_scatter_wait(wait: torch.fx.Node) -> bool:
    if is_graph_output(wait):
        return True

    if len(wait.users) == 1:
        user = next(iter(wait.users))
        assert user is not None
        return (
            is_graph_output(user)
            and user.op == "call_function"
            and user.target is torch.ops.prims.convert_element_type.default
        )

    return False


def bucket_fsdp_all_gather(
    gm: torch.fx.GraphModule,
    bucket_cap_mb_by_bucket_idx: Callable[[int], float] | None = None,
    mode: BucketMode = "default",
) -> None:
    """
    Bucketing pass for SimpleFSDP all_gather ops.

    Attributes:
        gm (torch.fx.GraphModule): Graph module of the graph.
        bucket_cap_mb_by_bucket_idx (Callable[[int], float] | None): callback function that
            takes in bucket id and returns size of a bucket in megabytes.
    """
    if bucket_cap_mb_by_bucket_idx is None:
        from torch._inductor.fx_passes.bucketing import (
            bucket_cap_mb_by_bucket_idx_default,
        )

        bucket_cap_mb_by_bucket_idx = bucket_cap_mb_by_bucket_idx_default
    assert bucket_cap_mb_by_bucket_idx is not None
    ag_buckets = bucket_all_gather_by_mb(
        gm,
        bucket_cap_mb_by_bucket_idx,
        filter_wait_node=is_fsdp_all_gather_wait,
    )
    if len(ag_buckets) == 0:
        return
    merge_all_gather(gm, ag_buckets, mode)


def bucket_fsdp_reduce_scatter(
    gm: torch.fx.GraphModule,
    bucket_cap_mb_by_bucket_idx: Callable[[int], float] | None = None,
    mode: BucketMode = "default",
) -> None:
    """
    Bucketing pass for SimpleFSDP reduce_scatter ops.

    Attributes:
        gm (torch.fx.GraphModule): Graph module of the graph.
        bucket_cap_mb_by_bucket_idx (Callable[[int], float] | None): callback function that
            takes in bucket idx and returns size of a bucket in megabytes. By default
            torch._inductor.fx_passes.bucketing.bucket_cap_mb_by_bucket_idx_default is used.

    """
    if bucket_cap_mb_by_bucket_idx is None:
        from torch._inductor.fx_passes.bucketing import (
            bucket_cap_mb_by_bucket_idx_default,
        )

        bucket_cap_mb_by_bucket_idx = bucket_cap_mb_by_bucket_idx_default
    rs_buckets = bucket_reduce_scatter_by_mb(
        gm,
        bucket_cap_mb_by_bucket_idx,
        filter_wait_node=is_fsdp_reduce_scatter_wait,
    )
    if len(rs_buckets) == 0:
        return
    merge_reduce_scatter(gm, rs_buckets, mode)
