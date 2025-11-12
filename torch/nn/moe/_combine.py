from typing import Optional

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem


def dedup_combine(
    inp: torch.Tensor,
    topk_indices_intranode: torch.Tensor,
    occurrences: torch.Tensor,
    topk_node_idx: torch.Tensor,
    dispatch_intra_plan: symm_mem.ExchangePlan,
    dispatch_inter_plan: symm_mem.ExchangePlan,
    intra_node_group: dist.ProcessGroup,
    inter_node_group: dist.ProcessGroup,
    seqlen: int,
    topk_weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    hid_dim = inp.shape[1]
    dtype = inp.dtype
    device = inp.device

    nnodes = inter_node_group.size()
    topk_nodes = topk_node_idx.shape[1]

    # Number of CUDA blocks, 8 block per inter-node rank
    n_blocks = nnodes * 8
    # Start offset of each CUDA block
    b_start = torch.full((n_blocks,), -1, dtype=torch.int64, device=device)
    # Number of tokens for each CUDA block
    b_len = torch.full((n_blocks,), -1, dtype=torch.int64, device=device)
    # Ready signal for each CUDA block. In this test we set all tokens as ready in one shot
    b_head = torch.full((n_blocks,), -1, dtype=torch.int64, device=device)

    # Output of a2av_2d_reduce
    # worst case: every token chooses me as 1 of the topk nodes
    max_intra_combine_len = seqlen * nnodes
    combine_intra_out = torch.empty(
        (max_intra_combine_len, hid_dim), dtype=dtype, device=device
    )
    combine_inter_out = torch.empty(
        (seqlen * topk_nodes, hid_dim), dtype=dtype, device=device
    )

    torch.ops.symm_mem._all_to_all_v_2d_index_reduce(
        inp,
        combine_intra_out,
        topk_indices_intranode,
        occurrences,
        dispatch_intra_plan.dst_offsets,
        dispatch_inter_plan.dst_offsets,
        dispatch_inter_plan.out_splits,
        intra_node_group.group_name,
        b_start,
        b_len,
        b_head,
    )

    # Combine the tokens inter node
    torch.ops.symm_mem._all_to_all_put_signal_in(
        combine_intra_out,
        combine_inter_out,
        dispatch_inter_plan.dst_offsets,
        dispatch_inter_plan.src_offsets,
        inter_node_group.group_name,
        b_start,
        b_len,
        b_head,
    )

    # Sort tokens from expert-grouped form (`combine_inter_out`) to
    # token-grouped form (i.e. copies of one token processed by different
    # experts are put together)
    sorted_indices = torch.argsort(topk_node_idx.view(-1))
    inverse_sorted = torch.empty_like(combine_inter_out)
    inverse_sorted[sorted_indices] = combine_inter_out

    # Sum up the processed result from topk experts, for each token
    condensed_out = inverse_sorted.view(*topk_node_idx.shape, -1).sum(dim=1)

    # Alternatively, we can use scatter_reduce_, but
    # "This operation may behave nondeterministically"
    # per its documentation.
    # reduce_indices = sorted_indices // topk_nodes
    # condensed_out = torch.empty((seqlen, hid_dim), dtype=dtype, device=device)
    # condensed_out.scatter_reduce_(
    #     dim=0,
    #     index=reduce_indices.unsqueeze(1).expand(-1, hid_dim),
    #     src=combine_inter_out, reduce="sum",
    #     include_self=False,
    # )

    return condensed_out
