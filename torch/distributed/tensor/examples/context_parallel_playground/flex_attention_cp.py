"""
To run the example, use the following command:
torchrun --standalone --nnodes=1 --nproc-per-node=4 flex_attention_cp.py
"""

import os
import random

from typing import Union

import torch
import torch.distributed as dist
import torch.distributed.tensor.experimental._attention as cp_attn

from torch import Tensor
from torch.autograd.grad_mode import no_grad
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor.examples.context_parallel_playground.flex_perf import (
    add_metrics_to_result,
    benchmark_torch_function_in_microseconds,
    Experiment,
    ExperimentConfig,
    ExperimentResults,
    print_results,
    run_flex_attention,
)
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
compiled_flex_attention = torch.compile(flex_attention, dynamic=False, fullgraph=True)
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


# examples
def flex_attn_causal_masking(world_size: int, rank: int) -> None:
    # init device mesh
    device_type = get_device_type()
    device_mesh = init_device_mesh(
        device_type=get_device_type(),
        mesh_shape=(world_size,),
        mesh_dim_names=("cp",),
    )

    # config
    dtype = torch.float32
    B = 8  # batch
    H = 8  # n_heads
    S = 256 * world_size  # seq_len
    D = 64  # head_dim
    exp_config = ExperimentConfig(
        shape=(B, H, S, H, S, D),
        attn_type="causal",
        dtype=dtype,
        calculate_bwd_time=False,
        cal_bandwidth=False,
        backends=["efficient"],
    )
    enable_load_balance = True

    # init input
    torch.manual_seed(10)
    qkv = [
        torch.rand(
            (B, H, S, D),
            device=device_type,
            dtype=dtype,
            requires_grad=True,
        )
        for _ in range(3)
    ]

    # local forward pass
    # we first test the case where mask is the same across batches
    block_mask = compiled_create_block_mask(
        causal_mask,
        B=B,
        H=1,
        Q_LEN=S,
        KV_LEN=S,
        device=device_type,
    )

    # compute sparsity for cp block_mask
    total_size = block_mask.numel()
    computed_blocks = block_mask.kv_num_blocks.sum()
    if block_mask.full_kv_num_blocks is not None:
        computed_blocks += block_mask.full_kv_num_blocks.sum()

    computed_size = (
        computed_blocks.item() * block_mask.BLOCK_SIZE[0] * block_mask.BLOCK_SIZE[1]
    )
    dense_ratio = computed_size / total_size
    sparsity = 1 - dense_ratio

    if rank == 0:
        print(f"Full BlockMask sparsity={sparsity}")

    q, k, v = qkv
    # ignore backward pass for now
    with no_grad():
        exp_result = run_flex_attention(exp_config, q, k, v, None, block_mask)

    print(f"rank: {rank} / {world_size}")
    print_results(
        [Experiment(exp_config, {"flex_attn": exp_result})],
        save_path=None,
        show_speedups=False,
    )
    print("\n\n")

    # context parallel exp config
    cp_exp_config = ExperimentConfig(
        shape=(B, H, S // world_size, H, S, D),
        attn_type="causal",
        dtype=dtype,
        calculate_bwd_time=False,
        cal_bandwidth=False,
        backends=["efficient"],
    )

    # context parallel forward pass
    seq_dim = 2
    _cp_options.enable_load_balance = enable_load_balance
    _set_cp_global_var("cp_shard_dim", seq_dim)  # shard on sequence dim
    # set CP context dispatch mode to use TORCH_FUNCTION for flex_attention
    cp_attn._dispatch_mode = _DispatchMode.TORCH_FUNCTION

    cp_block_mask = create_cp_block_mask(
        causal_mask,
        B=B,
        H=1,
        Q_LEN=S,
        KV_LEN=S,
        device_mesh=device_mesh,
        load_balancer=None,  # default load-balance
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
            load_balancer=None,
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


def flex_attn_document_causal_masking(world_size: int, rank: int) -> None:
    # init device mesh
    device_type = get_device_type()
    device_mesh = init_device_mesh(
        device_type=get_device_type(),
        mesh_shape=(world_size,),
        mesh_dim_names=("cp",),
    )

    # random init
    random.seed(10)
    torch.manual_seed(10)

    # parameters for document generation
    doc_count = 3

    # config
    dtype = torch.float32
    B = 8  # batch
    H = 1  # n_heads
    # S = 256 * world_size  # seq_len
    S = 8192
    D = 32  # head_dim
    exp_config = ExperimentConfig(
        shape=(B, H, S, H, S, D),
        attn_type="document_mask",
        dtype=dtype,
        calculate_bwd_time=False,
        cal_bandwidth=False,
        backends=["efficient"],
    )
    enable_load_balance = True

    # initialize document mask
    lengths = [
        (
            generate_random_lengths_in_chunks(S, doc_count, chunk_size=2 * world_size)
            if enable_load_balance
            else generate_random_lengths(S, doc_count)
        )
        for _ in range(B)
    ]
    offsets = length_to_offsets(lengths, device_type)
    document_causal_mask = generate_doc_mask_mod(causal_mask, offsets)

    # init input
    qkv = [
        torch.rand(
            (B, H, S, D),
            device=device_type,
            dtype=dtype,
            requires_grad=True,
        )
        for _ in range(3)
    ]

    # local forward pass
    # we first test the case where mask is the same across batches
    block_mask = compiled_create_block_mask(
        document_causal_mask,
        B=B,
        H=1,
        Q_LEN=S,
        KV_LEN=S,
        device=device_type,
    )

    # compute sparsity for cp block_mask
    total_size = block_mask.numel()
    computed_blocks = block_mask.kv_num_blocks.sum()
    if block_mask.full_kv_num_blocks is not None:
        computed_blocks += block_mask.full_kv_num_blocks.sum()

    computed_size = (
        computed_blocks.item() * block_mask.BLOCK_SIZE[0] * block_mask.BLOCK_SIZE[1]
    )
    dense_ratio = computed_size / total_size
    sparsity = 1 - dense_ratio

    if rank == 0:
        print(f"Full BlockMask sparsity={sparsity}")

    q, k, v = qkv
    # ignore backward pass for now
    with no_grad():
        exp_result = run_flex_attention(exp_config, q, k, v, None, block_mask)

    print(f"rank: {rank} / {world_size}")
    print_results(
        [Experiment(exp_config, {"flex_attn": exp_result})],
        save_path=None,
        show_speedups=False,
    )
    print("\n\n")

    # context parallel exp config
    cp_exp_config = ExperimentConfig(
        shape=(B, H, S // world_size, H, S, D),
        attn_type="document_mask",
        dtype=dtype,
        calculate_bwd_time=False,
        cal_bandwidth=False,
        backends=["efficient"],
    )

    # context parallel forward pass
    seq_dim = 2
    _cp_options.enable_load_balance = enable_load_balance
    _set_cp_global_var("cp_shard_dim", seq_dim)  # shard on sequence dim
    # set CP context dispatch mode to use TORCH_FUNCTION for flex_attention
    cp_attn._dispatch_mode = _DispatchMode.TORCH_FUNCTION

    # generate load balancer
    load_balancer = (
        PerDocumentHeadTailLoadBalancer(lengths, world_size, torch.device(device_type))
        if enable_load_balance
        else None
    )

    cp_block_mask = create_cp_block_mask(
        document_causal_mask,
        B=B,
        H=1,
        Q_LEN=S,
        KV_LEN=S,
        device_mesh=device_mesh,
        load_balancer=load_balancer,  # default load-balance or per-document load-balance
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
