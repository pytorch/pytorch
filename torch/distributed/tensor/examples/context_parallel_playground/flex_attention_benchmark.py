import os
import random

from typing import Union

import pandas as pd

import torch
from torch import Tensor

from torch.distributed.tensor.examples.context_parallel_playground.flex_perf import (
    # add_metrics_to_result,
    # benchmark_torch_function_in_microseconds,
    Experiment,
    ExperimentConfig,
    # ExperimentResults,
    generate_inputs,
    print_results,
    run_flex_attention,
)

from torch.nn.attention.flex_attention import (
    _mask_mod_signature,
    BlockMask,
    create_block_mask,
    flex_attention,
)


# --- util functions ---
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


# --- benchmarking functions ---
cfa = torch.compile(flex_attention, dynamic=False, fullgraph=True)
cbm = torch.compile(create_block_mask, dynamic=False, fullgraph=True)


def flex_attention_benchmark(filename: str) -> None:
    # random init
    random.seed(10)
    torch.manual_seed(42)

    # attention size: B=8, H=8, S=1024, D=64
    B, H, S, D = 8, 8, 8192, 64
    device = torch.device("cuda")
    dtype = torch.float32
    batch_size, q_heads, q_seq_len, kv_heads, kv_seq_len, head_dim = (
        B,
        H,
        S,
        H,
        S,
        D,
    )

    query, key, value = generate_inputs(
        batch_size,
        q_heads,
        q_seq_len,
        kv_heads,
        kv_seq_len,
        head_dim,
        dtype,
        device,
        requires_grad=True,
        nested_tensors=False,
    )

    # generate block_mask
    num_experiments = 400
    exp_records = []

    for i in range(num_experiments):
        num_documents = 1 + i
        lengths = [
            (generate_random_lengths(q_seq_len, num_documents))
            for _ in range(batch_size)
        ]
        offsets = length_to_offsets(lengths, device)
        document_causal_mask = generate_doc_mask_mod(causal_mask, offsets)

        # create block_mask
        block_mask = create_block_mask(
            document_causal_mask,
            B=batch_size,
            H=q_heads,
            Q_LEN=q_seq_len,
            KV_LEN=kv_seq_len,
            device=device,
        )

        block_mask_sparsity = compute_block_mask_sparsity(block_mask)

        exp_config = ExperimentConfig(
            shape=(batch_size, q_heads, q_seq_len, kv_heads, kv_seq_len, head_dim),
            attn_type="document_causal",
            dtype=dtype,
            calculate_bwd_time=True,
            cal_bandwidth=True,
            backends=["efficient"],
        )

        exp_result = run_flex_attention(exp_config, query, key, value, None, block_mask)

        # print results
        print(
            f"experiment {i}: sparsity={block_mask_sparsity}, document_lens={lengths}"
        )
        print_results(
            [Experiment(exp_config, {"flex_attn": exp_result})],
            save_path=None,
            show_speedups=False,
        )
        exp_records.append(
            (
                1.0 - block_mask_sparsity,
                exp_result.fwd_time / 1000,
                exp_result.bwd_time / 1000,
            )
        )
        print("\n")

    # write to file
    df = pd.DataFrame(
        exp_records,
        columns=[
            "mask density",
            "fwd_time",
            "bwd_time",
        ],
    )
    df.to_csv(filename)


if __name__ == "__main__":
    flex_attention_benchmark("flex_attention_density_benchmark.csv")
