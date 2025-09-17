"""
To run the example, use the following command:
torchrun --standalone --nnodes=1 --nproc-per-node=4 flex_attention_cp.py
"""

import os

import torch
import torch.distributed as dist
from torch.autograd.grad_mode import no_grad
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor.examples.flex_perf import (
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
    context_parallel,
)
from torch.nn.attention.flex_attention import create_block_mask, flex_attention

# Compile the flex_attention function
compiled_flex_attention = torch.compile(flex_attention, dynamic=False, fullgraph=True)
compiled_create_block_mask = torch.compile(
    create_block_mask, dynamic=False, fullgraph=True
)


def get_device_type() -> str:
    return "cuda"


def flex_attn_example(world_size: int, rank: int) -> None:
    device_type = get_device_type()
    device_handle = getattr(torch, device_type, None)
    assert device_handle is not None, f"Unsupported device type: {device_type}"
    num_devices_per_host = device_handle.device_count()
    device_handle.set_device(rank % num_devices_per_host)
    torch._dynamo.config.cache_size_limit = 100

    # numeric params
    atol = 1e-6
    rtol = 1e-2

    # init device mesh
    device_mesh = init_device_mesh(
        device_type=device_type,
        mesh_shape=(world_size,),
        mesh_dim_names=("cp",),
    )

    def causal_mask(b, h, q_idx, kv_idx):
        return q_idx >= kv_idx

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
    from torch.distributed.tensor.experimental._attention import (
        _set_cp_global_var,
        create_cp_block_mask,
    )

    _cp_options.enable_load_balance = True

    _set_cp_global_var("cp_shard_dim", seq_dim)  # shard on sequence dim

    # set CP context dispatch mode to use TORCH_FUNCTION for flex_attention
    torch.distributed.tensor.experimental._attention._dispatch_mode = (
        _DispatchMode.TORCH_FUNCTION
    )

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

    print(f"rank: {rank} / {world_size}, cp_flex_attn completes")

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
    print(f"computed_size = {computed_size}")
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
        flex_attn_example(world_size, rank)
    finally:
        dist.barrier()
        dist.destroy_process_group()
