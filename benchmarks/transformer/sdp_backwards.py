import random

import numpy as np
import torch
import torch.utils.benchmark as benchmark
from torch.profiler import profile, ProfilerActivity, record_function


class CompositeMHA(torch.nn.Module):
    def __init__(self, num_heads, in_proj_weight, in_proj_bias, out_proj):
        super().__init__()
        self.in_proj_weight = in_proj_weight
        self.in_proj_bias = in_proj_bias
        self.out_proj = out_proj
        self.num_heads = num_heads

    def forward(self, query, key, value, mask):
        if not (query is key and key is value):
            raise NotImplementedError(
                "query, key and value must be the same Tensor for now."
            )
        if mask is not None:
            raise NotImplementedError("mask is currently not supported.")

        query_projected = torch.nn.functional.linear(
            query, self.in_proj_weight, self.in_proj_bias
        )

        batch_size = query_projected.size(0)
        embed_dim = query_projected.size(2)
        head_dim = embed_dim // (self.num_heads * 3)

        query, key, value = query_projected.chunk(3, -1)

        query = query.view(batch_size, -1, self.num_heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, self.num_heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, self.num_heads, head_dim).transpose(1, 2)

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        attn = torch.nn.functional.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=False,
        )

        attn = attn.transpose(1, 2).reshape(batch_size, -1, self.num_heads * head_dim)
        # Match return signature of nn.MHA
        return self.out_proj(attn)


def build_composite_mha_from_nn_mha(pt):
    assert pt._qkv_same_embed_dim
    in_proj_weight = pt.in_proj_weight
    assert in_proj_weight is not None
    assert pt.batch_first
    return CompositeMHA(pt.num_heads, pt.in_proj_weight, pt.in_proj_bias, pt.out_proj)


def forw_back(model, input, upward):
    output = model(*input)
    output.backward(upward)


# Context manger not working in timer


def forw_back_fused(model, input, upward):
    with torch.backends.cuda.sdp_kernel(enable_math=False, enable_mem_efficient=True):
        output = model(*input)
        output.backward(upward)


def forw_back_eager(model, input, upward):
    with torch.backends.cuda.sdp_kernel(enable_math=True, enable_mem_efficient=False):
        output = model(*input)
        output.backward(upward)


def run_timing(
    min_run_time, batch_size, embed_dimension, num_heads, max_sequence_len, dtype
):
    dropout_p = 0.0
    mask = None

    pt = torch.nn.MultiheadAttention(
        embed_dim=embed_dimension,
        num_heads=num_heads,
        batch_first=True,
        dropout=dropout_p,
    )
    npt = pt.cuda().to(dtype)
    cpt = build_composite_mha_from_nn_mha(npt)
    x = torch.randn(
        batch_size,
        max_sequence_len,
        embed_dimension,
        dtype=dtype,
        device="cuda",
        requires_grad=True,
    )

    with torch.backends.cuda.sdp_kernel(enable_math=False, enable_mem_efficient=True):
        rand_fused_upward = cpt(x, x, x, mask).clone().detach()

    with torch.backends.cuda.sdp_kernel(enable_math=True, enable_mem_efficient=False):
        rand_eager_upward = cpt(x, x, x, mask).clone().detach()

    t0 = benchmark.Timer(
        stmt="forw_back_fused(cpt, (x,x,x,mask), rand_fused_upward)",
        globals={
            "forw_back_fused": forw_back_fused,
            "cpt": cpt,
            "x": x,
            "rand_fused_upward": rand_fused_upward,
            "mask": mask,
        },
        label=f"Fused SDP forward and backward batch_size={batch_size} max_sequence_len={max_sequence_len} "
        f"num_heads={num_heads} embed_dimension={embed_dimension} dtype={dtype}",
        num_threads=torch.get_num_threads(),
    )

    t1 = benchmark.Timer(
        stmt="forw_back_eager(cpt, (x,x,x,mask), rand_eager_upward)",
        globals={
            "forw_back_eager": forw_back_eager,
            "cpt": cpt,
            "x": x,
            "rand_eager_upward": rand_eager_upward,
            "mask": mask,
        },
        label=f"Eager SDP forward and backward batch_size={batch_size} max_sequence_len={max_sequence_len} "
        f"num_heads={num_heads} embed_dimension={embed_dimension} dtype={dtype}",
        num_threads=torch.get_num_threads(),
    )

    m0 = t0.blocked_autorange(min_run_time=min_run_time)
    m1 = t1.blocked_autorange(min_run_time=min_run_time)

    print(m0)
    print(m1)

    activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]

    print("Profile for Fused".center(200, "-"))
    with torch.backends.cuda.sdp_kernel(enable_math=False, enable_mem_efficient=True):
        with profile(
            activities=activities, record_shapes=False, with_stack=True
        ) as prof:
            with record_function("Fused SDP forward and backward"):
                for _ in range(20):
                    forw_back(cpt, (x, x, x, mask), rand_fused_upward)
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))

    print("Profile for eager".center(200, "-"))
    with torch.backends.cuda.sdp_kernel(enable_math=True, enable_mem_efficient=False):
        with profile(
            activities=activities, record_shapes=False, with_stack=True
        ) as prof:
            with record_function("Fused SDP forward and backward"):
                for _ in range(20):
                    forw_back(cpt, (x, x, x, mask), rand_eager_upward)
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))


def main():
    seed = 123
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

    min_run_time = 10
    batch_size = 64
    num_heads = 32
    max_seq_len = 256
    embed_dim = 1024
    dtype = torch.bfloat16

    print(
        f"Running timing for batch_size={batch_size} max_sequence_len={max_seq_len} "
        f"num_heads={num_heads} embed_dimension={embed_dim} dtype={dtype}"
    )
    run_timing(min_run_time, batch_size, embed_dim, num_heads, max_seq_len, dtype)


if __name__ == "__main__":
    main()
