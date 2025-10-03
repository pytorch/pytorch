"""
To run the example, use the following command:
python block_mask_load_balance_benchmark.py
"""

import random
from typing import Optional, Union

import pandas as pd

import torch
from torch import Tensor
from torch._prims_common import DeviceLikeType
from torch.distributed.tensor.experimental._attention import (
    HeadTailLoadBalancer,
    PTRRLoadBalancer,
)
from torch.nn.attention.flex_attention import (
    _mask_mod_signature,
    BlockMask,
    create_block_mask,
)


# Compile the BlockMask creation function
"""
compiled_create_block_mask = torch.compile(
    create_block_mask, dynamic=False, fullgraph=True
)
"""
compiled_create_block_mask = create_block_mask  # turn-off compilation
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


def generate_random_lengths_in_range(total_length, min_len, max_len) -> list[int]:
    res = []

    generated_length = 0
    while generated_length < total_length:
        length = random.randint(min_len, max_len)
        if generated_length + length > total_length:
            length = total_length - generated_length

        res.append(length)
        generated_length += length

    print(f"document lengths = {res}")
    return res


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


def report_stats(records: list[float]) -> dict[str, float]:
    """
    util function for reporting statistics:
        size, min, max, mean, variance
    """
    mean = sum(records) / len(records)

    return {
        "size": len(records),
        "min": min(records),
        "max": max(records),
        "mean": mean,
        "variance": sum((x - mean) ** 2 for x in records),
    }


# same logic as in torch.distributed.tensor.experimental._attention.create_cp_block_mask
# but doesn't rely on DeviceMesh
def _create_cp_block_mask(
    mask_mod: _mask_mod_signature,
    B: int,
    H: int,
    Q_LEN: int,
    KV_LEN: int,
    rank: int,
    world_size: int,
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

    # unlike the original create_cp_block_mask, we don't need to modify seq_lengths
    return block_mask


# benchmark:
#   Head-tail load balance vs. Sparsity-based load balance on Document Mask
def benchmark_load_balance_document_mask(
    world_size: int,
    B: int,
    H: int,
    S: int,
    D: int,
    document_count: int,  # parameters for document generation
    filename: Optional[str] = "./document_mask_load_balance_benchmark.csv",
) -> None:
    device_type = get_device_type()

    # random init
    random.seed(10)

    exp_records = []  # (bm_sparsity, max_sparsity_base, max_sparsity_auto)
    num_experiments = 1000

    for _ in range(num_experiments):
        # initialize document mask
        # lengths = [(generate_random_lengths(S, document_count)) for _ in range(B)]
        lengths = [(generate_random_lengths_in_range(S, 10, 4000)) for _ in range(B)]
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

        full_block_mask_sparsity = compute_block_mask_sparsity(block_mask)

        load_balancer = HeadTailLoadBalancer
        load_balance_indices = load_balancer._generate_indices(
            S, world_size, device_type
        )
        # simulate context parallel block_mask sharding:
        base_line_context_parallel_block_mask_sparsity = []
        for rank in range(world_size):
            cp_block_mask = _create_cp_block_mask(
                document_causal_mask,
                B=B,
                H=H,
                Q_LEN=S,
                KV_LEN=S,
                rank=rank,
                world_size=world_size,
                qkv_shuffle_indices=load_balance_indices,
                device=device_type,
            )
            block_mask_sparsity_on_rank = compute_block_mask_sparsity(cp_block_mask)
            base_line_context_parallel_block_mask_sparsity.append(
                block_mask_sparsity_on_rank
            )

        base_line_context_parallel_block_mask_sparsity_min = report_stats(
            base_line_context_parallel_block_mask_sparsity
        )["min"]
        base_line_context_parallel_block_mask_sparsity_max = report_stats(
            base_line_context_parallel_block_mask_sparsity
        )["max"]

        load_balancer = PTRRLoadBalancer
        load_balance_indices = load_balancer._generate_indices(
            block_mask, world_size, device_type
        )
        # simulate context parallel block_mask sharding:
        auto_load_balance_context_parallel_block_mask_sparsity = []
        for rank in range(world_size):
            cp_block_mask = _create_cp_block_mask(
                document_causal_mask,
                B=B,
                H=H,
                Q_LEN=S,
                KV_LEN=S,
                rank=rank,
                world_size=world_size,
                qkv_shuffle_indices=load_balance_indices,
                device=device_type,
            )
            auto_load_balance_context_parallel_block_mask_sparsity.append(
                compute_block_mask_sparsity(cp_block_mask)
            )

        auto_load_balance_context_parallel_block_mask_sparsity_min = report_stats(
            auto_load_balance_context_parallel_block_mask_sparsity
        )["min"]
        auto_load_balance_context_parallel_block_mask_sparsity_max = report_stats(
            auto_load_balance_context_parallel_block_mask_sparsity
        )["max"]

        exp_records.append(
            (
                full_block_mask_sparsity,
                base_line_context_parallel_block_mask_sparsity_min,
                base_line_context_parallel_block_mask_sparsity_max,
                auto_load_balance_context_parallel_block_mask_sparsity_min,
                auto_load_balance_context_parallel_block_mask_sparsity_max,
            )
        )

    # write to file
    df = pd.DataFrame(
        exp_records,
        columns=[
            "attn_sparsity",
            "min_sparsity_base",
            "max_sparsity_base",
            "min_sparsity_auto",
            "max_sparsity_auto",
        ],
    )
    df.to_csv(filename)


if __name__ == "__main__":
    benchmark_load_balance_document_mask(
        world_size=8,
        B=1,
        H=1,
        S=8192,
        D=32,
        document_count=3,
    )
