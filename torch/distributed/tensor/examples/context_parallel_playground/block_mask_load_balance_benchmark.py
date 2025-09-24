"""
To run the example, use the following command:
python block_mask_load_balance_benchmark.py

Pre-requisite:
pip install jsonargparse
"""

import os
import random

from typing import Union

import torch
import torch.distributed as dist
import torch.distributed.tensor.experimental._attention as cp_attn

from torch import Tensor
from torch._prims_common import DeviceLikeType
from torch.autograd.grad_mode import no_grad
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor.experimental._attention import (
    _cp_options,
    _DispatchMode,
    _set_cp_global_var,
    context_parallel,
    create_cp_block_mask,
    PerDocumentHeadTailLoadBalancer,
)
from torch.nn.attention.flex_attention import (
    _mask_mod_signature,
    create_block_mask,
    flex_attention,
)

# Compile the flex_attention function
compiled_create_block_mask = torch.compile(
    create_block_mask, dynamic=False, fullgraph=True
)
torch._dynamo.config.cache_size_limit = 100


# utils
def get_device_type() -> str:
    return "cuda"


# for Document Masking:
# copied from https://github.com/meta-pytorch/attention-gym/blob/main/attn_gym/masks/document_mask.py
def generate_random_lengths(total_length, num_documents) -> list[int]:
    # Initialize all lengths to 1 to ensure each document has at least one token
    lengths = [1] * num_documents
    remaining_length = total_length - num_documents

    # Randomly distribute the remaining length
    for _ in range(remaining_length):
        index = random.randint(0, num_documents - 1)
        lengths[index] += 1

    return lengths


# TODO: add generate_random_lengths_in_range(..., min, max)


# generate random document lengths
def generate_random_lengths_in_chunks(
    total_length, num_documents, chunk_size
) -> list[int]:
    # Generate a list of random document lengths so that each document contains
    # some number of chunks of size `chunk_size`. This means each document's length
    # must be a multiple of `chunk_size`. Besides, the lengths of all the documents
    # sum up to `total_length`.
    num_chunks = total_length // chunk_size
    assert total_length % chunk_size == 0 and num_chunks >= num_documents

    num_chunks_per_document = [1] * num_documents
    remaining_chunks = num_chunks - num_documents
    # Randomly distribute the remaining chunks
    for _ in range(remaining_chunks):
        index = random.randint(0, num_documents - 1)  # document_id
        num_chunks_per_document[index] += 1

    return [num_chunks * chunk_size for num_chunks in num_chunks_per_document]


def length_to_offsets(
    lengths: list[list[int]], device: Union[str, torch.device]
) -> Tensor:
    """Converts a list of lengths to a list of offsets.

    Args:
        lengths: A list of lengths.

    """
    offsets = [[0] + lengths_in_batch for lengths_in_batch in lengths]
    offsets = torch.tensor(offsets, device=device, dtype=torch.int32)
    offsets = torch.cumsum(offsets, dim=-1)
    return offsets


def _offsets_to_doc_ids_tensor(offsets):
    doc_ids = []
    device = offsets.device
    for batch_idx in range(offsets.size(0)):
        counts = offsets[batch_idx][1:] - offsets[batch_idx][:-1]
        doc_id = torch.repeat_interleave(
            torch.arange(len(counts), device=device, dtype=torch.int32), counts
        )
        doc_ids.append(doc_id)

    return torch.stack(doc_ids)


def generate_doc_mask_mod(
    mask_mod: _mask_mod_signature, offsets: Tensor
) -> _mask_mod_signature:
    """Generates mask mods that apply to inputs to flex attention in the sequence stacked
    format.

    Args:
        mask_mod: The mask mod to apply to the documents
        offsets: This tensor should be of shape(num_documents + 1)
            this should contain the cumulative counts of document tokens.
            e.g. if you have 3 documents of length 2, 4, 3 then
            offsets = [0, 2, 6, 9]

    Note:
        What is the sequence stacked format? When assembling batches of inputs, we
        take multiple sequences and stack them together to form 1 large sequence. We then
        use masking to ensure that the attention scores are only applied to tokens within
        the same document.
    """
    document_id = _offsets_to_doc_ids_tensor(offsets)

    def doc_mask_mod(b, h, q_idx, kv_idx):
        same_doc = document_id[b][q_idx] == document_id[b][kv_idx]
        q_logical = q_idx - offsets[b, document_id[b, q_idx]]
        kv_logical = kv_idx - offsets[b, document_id[b, kv_idx]]
        inner_mask = mask_mod(b, h, q_logical, kv_logical)
        return same_doc & inner_mask

    return doc_mask_mod


# mask function
def causal_mask(b, h, q_idx, kv_idx):
    return q_idx >= kv_idx


def compute_block_mask_sparsity(block_mask: BlockMask) -> float:
    """util function for computing the sparsity within `block_mask`"""
    total_size = block_mask.numel()
    computed_blocks = block_mask.kv_num_blocks.sum()
    if block_mask.full_kv_num_blocks is not None:
        computed_blocks += block_mask.full_kv_num_blocks.sum()

    computed_size = (
        computed_blocks.item() * block_mask.BLOCK_SIZE[0] * block_mask.BLOCK_SIZE[1]
    )
    dense_ratio = 1.0 * computed_size / total_size
    sparsity = 1 - dense_ratio

    return sparsity


# same logic as in torch.distributed.tensor.experimental._attention.create_cp_block_mask
# but doesn't rely on DeviceMesh
def _create_cp_block_mask(
    mask_mod: _mask_mod_signature,
    B: int,
    H: int,
    Q_LEN: int,
    KV_LEN: int,
    qkv_shuffle_indices: Optional[Tensor] = None,
    device: DeviceLikeType = "cuda",
) -> BlockMask:
    from torch.nn.attention.flex_attention import _DEFAULT_SPARSE_BLOCK_SIZE

    # q_idx, kv_idx: index after shuffling
    # qkv_shuffle_indices[q_idx]: q_idx before shuffling
    # qkv_shuffle_indices[kv_idx]: kv_idx before shuffling
    # new_mask_mod[local_q_idx][global_kv_idx] = mask_mod[qkv_shuffle_indices[global_q_idx]][qkv_shuffle_indices[global_kv_idx]]
    def _rewrite_mask_mod(
        mask_mod: _mask_mod_signature,
        rank: int,
        world_size: int,
        block_size: int,
        local_q_size: int,
        qkv_shuffle_indices: Optional[torch.Tensor] = None,
    ) -> _mask_mod_signature:
        # local_q_idx -> global_q_idx
        def qkv_idx_restore(
            b: torch.Tensor, idx_post_shuffle: torch.Tensor
        ) -> torch.Tensor:
            if qkv_shuffle_indices is not None:
                if qkv_shuffle_indices.ndim == 1:  # identical across batches
                    idx_pre_shuffle = qkv_shuffle_indices[idx_post_shuffle]
                else:
                    idx_pre_shuffle = qkv_shuffle_indices[b][idx_post_shuffle]
            else:
                idx_pre_shuffle = idx_post_shuffle

            return idx_pre_shuffle

        def local_q_idx_to_q_idx(local_q_idx: torch.Tensor) -> torch.Tensor:
            # calculate local block_idx and block_offset
            local_blk_idx, local_blk_offset = (
                local_q_idx // block_size,
                local_q_idx % block_size,
            )
            # NOTE: load balancing is not used
            local_num_blocks = local_q_size // block_size
            blk_idx = local_num_blocks * rank + local_blk_idx
            return blk_idx * block_size + local_blk_offset

        return lambda b, h, q_idx, kv_idx: mask_mod(
            b,
            h,
            qkv_idx_restore(b, local_q_idx_to_q_idx(q_idx)),
            qkv_idx_restore(b, kv_idx),
        )

    Q_SHARD_LEN = Q_LEN // world_size
    block_size = _DEFAULT_SPARSE_BLOCK_SIZE
    block_mask = compiled_create_block_mask(
        _rewrite_mask_mod(
            mask_mod,
            rank,
            world_size,
            block_size,
            Q_SHARD_LEN,
            qkv_shuffle_indices=qkv_shuffle_indices,
        ),
        B,
        H,
        Q_SHARD_LEN,
        KV_LEN,
        device=device,
        BLOCK_SIZE=(block_size, block_size),
    )


# benchmark:
#   Head-tail load balance vs. Sparsity-based load balance on Document Mask
def benchmark_load_balance_document_mask(
    world_size: int,
    B: int,
    H: int,
    S: int,
    D: int,
    document_count: int,  # parameters for document generation
) -> None:
    device_type = get_device_type()

    # random init
    random.seed(10)

    # initialize document mask
    lengths = [(generate_random_lengths(S, document_count)) for _ in range(B)]
    offsets = length_to_offsets(lengths, device_type)
    document_causal_mask = generate_doc_mask_mod(causal_mask, offsets)

    # full BlockMask
    block_mask = compiled_create_block_mask(
        document_causal_mask,
        B=B,
        H=1,
        Q_LEN=S,
        KV_LEN=S,
        device=device_type,
    )

    print(f"Full BlockMask sparsity={compute_block_mask_sparsity(block_mask)}")

    # simulate context parallel block_mask sharding:
    for rank in range(world_size):
        print(f"rank: {rank} / {world_size}")
        load_balancer = HeadTailLoadBalancer

        print("\n")

    cp_block_mask = _create_cp_block_mask(
        document_causal_mask,
        B=B,
        H=H,
        Q_LEN=S,
        KV_LEN=S,
        qkv_shuffle_indices=,
        device=device_type,
    )

    # prepare input buffer
    cp_q = q.detach().clone()
    cp_k = k.detach().clone()
    cp_v = v.detach().clone()

    with no_grad():
        with context_parallel(
            device_mesh,
            buffers=[cp_q, cp_k, cp_v],
            buffer_seq_dims=[seq_dim] * 3,
            load_balancer=load_balancer,
        ):
            # TODO: compiled flex_attention doesn't work with reuse of block_mask
            import torch.nn.attention.flex_attention as fa

            fa._FLEX_ATTENTION_DISABLE_COMPILE_DEBUG = True
            forward_compiled_time = benchmark_torch_function_in_microseconds(
                # compiled_flex_attention,
                flex_attention,
                cp_q,
                cp_k,
                cp_v,
                block_mask=cp_block_mask,
                enable_gqa=True,
            )
            backward_compiled_time = None

            cp_q.requires_grad = False
            cp_k.requires_grad = False
            cp_v.requires_grad = False

    # compute sparsity for cp block_mask
    total_size = cp_block_mask.numel() * world_size
    computed_blocks = cp_block_mask.kv_num_blocks.sum()
    if cp_block_mask.full_kv_num_blocks is not None:
        computed_blocks += cp_block_mask.full_kv_num_blocks.sum()

    computed_size = (
        computed_blocks.item()
        * cp_block_mask.BLOCK_SIZE[0]
        * cp_block_mask.BLOCK_SIZE[1]
    )
    dense_ratio = computed_size / total_size
    sparsity = 1 - dense_ratio

    exp_result = ExperimentResults(
        fwd_time=forward_compiled_time,
        bwd_time=backward_compiled_time,
        sparsity=sparsity,
    )
    result = add_metrics_to_result(cp_exp_config, exp_result)

    print(f"rank: {rank} / {world_size}, sparsity={sparsity}")
    print_results(
        [Experiment(cp_exp_config, {"cp_flex_attn": result})],
        save_path=None,
        show_speedups=False,
    )
    print("\n\n")


if __name__ == "__main__":
    # this script is launched via torchrun which automatically manages ProcessGroup
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    # assert world_size == 4  # our example uses 4 worker ranks

    try:
        # flex_attn_causal_masking(world_size, rank)
        flex_attn_document_causal_masking(world_size, rank)
    finally:
        dist.barrier()
        dist.destroy_process_group()
