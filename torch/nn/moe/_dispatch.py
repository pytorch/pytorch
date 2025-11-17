from typing import Optional

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem


# Helper function to determine the position of a token in an expert chunk from a
# source rank to a destination rank
def _get_occurrence_numbers(topk_indices: torch.Tensor, n_experts: int) -> torch.Tensor:
    """
    Transform topk_indices to show which occurrence each element is, occurrence number starts from 0.

    Example: tensor([1, 2, 1, 3, 1, 2]) -> tensor([0, 0, 1, 0, 2, 1])
    """
    device = topk_indices.device
    # Get unique values and their inverse mapping
    unique_vals, inverse = torch.unique(topk_indices, return_inverse=True)

    # Create a tensor to count occurrences for each unique value
    n_unique = n_experts
    n_elements = len(topk_indices)

    # Create a matrix where each row corresponds to a unique value
    # and columns correspond to positions in the original topk_indices
    indicator_matrix = torch.zeros(
        n_unique, n_elements, dtype=torch.float, device=device
    )
    indicator_matrix[inverse, torch.arange(n_elements)] = 1.0

    # Cumulative sum along columns gives us occurrence numbers
    occurrence_counts = torch.cumsum(indicator_matrix, dim=1) - indicator_matrix

    # Extract the occurrence number for each position
    result = occurrence_counts[inverse, torch.arange(n_elements, device=device)]

    return result.long()


def dedup_dispatch(
    inp: torch.Tensor,
    topk_node_idx: torch.Tensor,
    topk_expert_idx: torch.Tensor,
    n_experts: int,
    inter_node_group: dist.ProcessGroup,
    intra_node_group: dist.ProcessGroup,
    out_len_ratio: int,
    align: int,
    topk_weights: Optional[torch.Tensor] = None,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    symm_mem.ExchangePlan,
    symm_mem.ExchangePlan,
    Optional[torch.Tensor],
    torch.Tensor,  # for debug
    torch.Tensor,  # for debug
]:
    r"""
    This is a collective operation that dispatches tokens to experts in a
    deduplicated fashion. Tokens are first shuffled to preferred nodes per
    `topk_node_idx`, then shuffled to preferred experts per `topk_expert_idx`
    within each node.

    Args:
        inp (class:`torch.Tensor`): the input tensor containing the tokens to dispatch.
        topk_node_idx (class:`torch.Tensor`): the topk node indices for each token.
        topk_expert_idx (class:`torch.Tensor`): the topk expert indices for each token.
        n_experts (int): the total number of experts.
        inter_node_group (class:`dist.ProcessGroup`): the process group for inter-node communication.
        intra_node_group (class:`dist.ProcessGroup`): the process group for intra-node communication.
        out_len_ratio (int): the ratio of the output length to the input length.
            This is used to determine the maximum output length.
        align (int): the alignment of the first token, in the output tensor, for
            the chunk each expert receives. This can be used to satisfy the
            alignment requirement of group GEMM. If not specified, the alignment
            is set to 1.
        topk_weights (class:`torch.Tensor`, optional): the topk weights for each
            token. The weights are exchanged inter-node only. When combining the
            tokens from intra-node group, you can apply the weights to the
            combined result.

    Returns:
        A tuple containing the following tensors:
            - intra_out (class:`torch.Tensor`): the output tensor containing the dispatched tokens.
            - topk_indices_intranode_out (class:`torch.Tensor`): the topk expert
              indices for each token after intra-node communication.
            - inter_plan (class:`symm_mem.ExchangePlan`): the exchange plan used for inter-node communication.
            - intra_plan (class:`symm_mem.ExchangePlan`): the exchange plan used for intra-node communication.
            - topk_weights_out (class:`torch.Tensor`, optional): the topk
              weights after inter-node exchange. If `topk_weights` is not
              provided as input to this API, `None` is returned for this field.
            - inter_out (class:`torch.Tensor`): the output tensor containing the
              dispatched tokens after inter-node communication. (for debug)
            - recv_intra_inp_splits (class:`torch.Tensor`): the input splits for the intra-node exchange plan. (for debug)

    .. note::
        This function must be run under a :class:`torch.cuda.MemPool` context backed by a Symmetric Memory allocator. For example:
        ```
        import torch
        import torch.distributed._symmetric_memory as symm_mem

        allocator = symm_mem.get_mempool_allocator(device)
        mempool = torch.cuda.MemPool(allocator)

        with torch.cuda.use_mem_pool(mempool):
            (
                intra_out,
                topk_indices_intranode_out,
                inter_plan,
                intra_plan,
                topk_weights_out,
                *_,
            ) = dedup_dispatch(
                inp,
                topk_node_idx,
                topk_expert_idx,
                n_experts,
                inter_group,
                intra_group,
                out_len_ratio,
                align,
                topk_weights,
            )
        ```
    """

    seqlen = inp.shape[0]
    if topk_node_idx.shape[0] != seqlen:
        raise ValueError(
            f"topk_node_idx must have the same length as inp, {topk_node_idx.shape[0]} != {seqlen}"
        )
    if topk_expert_idx.shape[0] != seqlen:
        raise ValueError(
            f"topk_expert_idx must have the same length as inp, {topk_expert_idx.shape[0]} != {seqlen}"
        )
    hid_dim = inp.shape[1]
    dtype = inp.dtype
    device = inp.device

    nnodes = inter_node_group.size()
    node_id = inter_node_group.rank()

    # Number of experts per node
    experts_per_node = n_experts // nnodes
    # Number of topk nodes
    topk_nodes = topk_node_idx.shape[1]
    # Number of topk experts
    topk_experts = topk_expert_idx.shape[1]

    # Convert indices to splits
    in_splits = torch.histc(topk_node_idx, bins=nnodes, min=0, max=nnodes - 1)
    sorted_indices = torch.argsort(topk_node_idx.view(-1))
    expanded_inp = inp[sorted_indices // topk_nodes]
    expanded_topk_idx = topk_expert_idx[sorted_indices // topk_nodes]

    # worst case: every token chooses me as 1 of the topk nodes
    max_out_len = seqlen * nnodes

    # Create symm_mem tensors
    inter_out = torch.empty((max_out_len, hid_dim), dtype=dtype, device=device)
    src_offsets = torch.empty(nnodes, dtype=torch.int64, device=device)
    out_splits = torch.empty(nnodes, dtype=torch.int64, device=device)
    dst_offsets = torch.empty(nnodes, dtype=torch.int64, device=device)

    # Create intra-node topk indices, (expanded_seqlen, topk_experts)
    # Filler to indicate unused slots -- no expert to send this token to
    UNUSED = -1
    topk_indices_intranode_out = torch.full(
        (max_out_len, topk_experts), UNUSED, dtype=torch.int64, device=device
    )

    # Create inter-node exchange plan
    inter_plan = symm_mem.make_a2a_exchange_plan(
        in_splits, src_offsets, out_splits, dst_offsets, inter_node_group.group_name
    )

    # Number of CUDA blocks, 8 block per inter-node rank
    n_blocks = nnodes * 8
    # Start offset of each CUDA block
    b_start = torch.full((n_blocks,), -1, dtype=torch.int64, device=device)
    # Number of tokens for each CUDA block
    b_len = torch.full((n_blocks,), -1, dtype=torch.int64, device=device)
    # Ready signal for each CUDA block. In this test we set all tokens as ready in one shot
    b_head = torch.full((n_blocks,), -1, dtype=torch.int64, device=device)

    # Create side stream for inter-node data exchange
    inter_node_stream = torch.Stream(device=device)
    # Inter-node token exchange can start as soon as inter_plan is ready and signals are created
    current_stream = torch.accelerator.current_stream(device)
    inter_node_stream.wait_stream(current_stream)

    with inter_node_stream:
        # Fire up inter-node token exchange
        # inp: (expanded_seqlen, hid_dim)
        symm_mem.all_to_all_v(
            expanded_inp,
            inter_out,
            inter_plan,
            inter_node_group.group_name,
            b_start,
            b_len,
            b_head,
        )

    # Exchange topk weights inter-node if provided, no need to exchange it intra-node
    topk_weights_out = None
    if topk_weights is not None:
        expanded_topk_weights = topk_weights[sorted_indices // topk_nodes]
        topk_weights_out = torch.empty(
            max_out_len,
            *topk_weights.shape[1:],
            dtype=topk_weights.dtype,
            device=device,
        )
        inter_node_stream.wait_stream(current_stream)
        with inter_node_stream:
            symm_mem.all_to_all_v(
                expanded_topk_weights,
                topk_weights_out,
                inter_plan,
                inter_node_group.group_name,
            )

    # On default stream, we continue to prepare for intra-node exchange

    # Inter-node topk indices exchange, (expanded_seqlen, topk)
    symm_mem.all_to_all_v(
        expanded_topk_idx,
        topk_indices_intranode_out,
        inter_plan,
        inter_node_group.group_name,
    )

    # Rebase expert indices
    topk_indices_intranode_out -= node_id * experts_per_node

    # Now, some planning about intra-node exchange

    # Convert intra-node indices to splits
    intra_in_splits = torch.histc(
        topk_indices_intranode_out,
        bins=experts_per_node,
        min=0,
        max=experts_per_node - 1,
    )

    # Create symm_mem tensors
    intra_src_offsets = torch.empty(experts_per_node, dtype=torch.int64, device=device)
    intra_out_splits = torch.empty(experts_per_node, dtype=torch.int64, device=device)
    intra_dst_offsets = torch.empty(experts_per_node, dtype=torch.int64, device=device)
    max_out_len_intra = seqlen * out_len_ratio
    intra_out = torch.empty(max_out_len_intra, hid_dim, dtype=dtype, device=device)

    # Create intra-node exchange plan on a side stream, to overlap with the
    # occurrence calculation
    intra_plan_stream = torch.Stream(device=device)
    intra_plan_stream.wait_stream(current_stream)
    with intra_plan_stream:
        intra_plan = symm_mem.make_a2a_2d_exchange_plan(
            intra_in_splits,
            intra_src_offsets,
            intra_out_splits,
            intra_dst_offsets,
            intra_node_group.group_name,
            align,
        )

    # Figure out rank of each token in its expert chunk
    occurrences = _get_occurrence_numbers(
        topk_indices_intranode_out.view(-1), n_experts
    )

    # Wait for completion of intra-node exchange plan
    current_stream.wait_stream(intra_plan_stream)

    # Now let's fire up intra-node exchange
    torch.ops.symm_mem._all_to_all_v_2d_index_push(
        inter_out,
        intra_out,
        topk_indices_intranode_out,
        occurrences,
        intra_plan.dst_offsets,
        intra_node_group.group_name,
        b_start,
        b_len,
        b_head,
    )

    # Join inter-node exchange stream
    # The completion of intra-node exchange would also indicate the completion
    # of inter-node exchange, here we still join inter-node stream for
    # completion of the stream graph
    current_stream.wait_stream(inter_node_stream)

    return (
        intra_out,
        topk_indices_intranode_out,
        inter_plan,
        intra_plan,  # necessary
        topk_weights_out,  # if topk_weights is provided
        inter_out,  # for debug
        intra_in_splits,  # for debug
    )
