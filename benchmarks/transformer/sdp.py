import torch
import itertools
import numpy as np
import sys
import csv


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

        batch_size, seq_len, embed_dim = query_projected.size()
        head_dim = embed_dim // (self.num_heads * 3)

        # Transpose seq_len and num_heads dim
        query_projected = query_projected.view(
            batch_size, seq_len, 3 * self.num_heads, head_dim
        ).transpose(1, 2)
        query, key, value = query_projected.chunk(3, 1)

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        attn, _ = torch.nn.functional._scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=None,
            dropout_p=0.0,
            need_attn_weights=False,
            is_causal=False,
        )

        attn = attn.transpose(1, 2).reshape(
            batch_size, seq_len, self.num_heads * head_dim
        )
        # Match return signature of nn.MHA
        return self.out_proj(attn), None


def build_composite_mha_from_nn_mha(pt):
    assert pt._qkv_same_embed_dim
    in_proj_weight = pt.in_proj_weight
    assert in_proj_weight is not None
    assert pt.batch_first
    return CompositeMHA(pt.num_heads, pt.in_proj_weight, pt.in_proj_bias, pt.out_proj)


def benchmark_torch_function(iters, f, *args, **kwargs):
    if f is None:
        return None
    f(*args, **kwargs)
    torch.cuda.synchronize()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    for _ in range(iters):
        f(*args, **kwargs)
    end_event.record()
    torch.cuda.synchronize()
    return (start_event.elapsed_time(end_event) * 1.0e-3) / iters


def run_timing(batch_size, D, H, L, writer):
    dropout_p = 0.0
    mask = None

    pt = torch.nn.MultiheadAttention(
        embed_dim=D, num_heads=H, batch_first=True, dropout=dropout_p
    )
    npt = pt.eval().half().cuda()
    cpt = build_composite_mha_from_nn_mha(npt)

    x = torch.randn(batch_size, L, D)
    x = x.half().cuda()

    pt_output, _ = pt(x, x, x, mask)
    cp_output, _ = cpt(x, x, x, mask)

    # First order sanity check. Not a replacement for rigorous tests.
    assert torch.allclose(pt_output, cp_output, atol=1e-3, rtol=1e-3)

    with torch.backends.cuda.sdp_kernel(enable_math=True, enable_flash=True):
        with torch.inference_mode():
            pt_time = benchmark_torch_function(iters, npt, x, x, x, mask) * 1e3
            cp_time = benchmark_torch_function(iters, cpt, x, x, x, mask) * 1e3
            results = {}
            results["L"] = L
            results["H"] = H
            results["D"] = D
            results["pt_time"] = pt_time
            results["cp_time"] = cp_time
            results["speedup"] = pt_time / cp_time
            results["dtype"] = str(x.dtype)
            writer.writerow(results)


if __name__ == "__main__":
    iters = 100
    seed = 123
    np.random.seed(seed)
    torch.manual_seed(seed)

    headers = ["L", "H", "D", "pt_time", "cp_time", "speedup", "dtype"]
    writer = csv.DictWriter(sys.stdout, headers)
    writer.writeheader()

    batch_size = 64
    for H, L in itertools.product([1, 2, 4, 8, 16, 32], [64, 128, 256]):
        run_timing(batch_size, 1024, H, L, writer)
