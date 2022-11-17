import torch
import itertools
import numpy as np
import sys
import csv
import random

import warnings
warnings.filterwarnings("ignore")


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
            batch_size, -1, self.num_heads * head_dim
        )
        # Match return signature of nn.MHA
        return self.out_proj(attn), None


def build_composite_mha_from_nn_mha(pt):
    assert pt._qkv_same_embed_dim
    in_proj_weight = pt.in_proj_weight
    assert in_proj_weight is not None
    assert pt.batch_first
    return CompositeMHA(pt.num_heads, pt.in_proj_weight, pt.in_proj_bias, pt.out_proj)


def generate_rand_batch(batch_size, max_sequence_len, embed_dimension, pad_percentage=None, dtype=torch.float16, device="cuda"):
    if not pad_percentage:
        return torch.randn(batch_size, max_sequence_len, embed_dimension, dtype=dtype, device=device), None
    # Really slow but should work
    seq_len_list = [int(max_sequence_len * (1 - random.gauss(pad_percentage, 0.01))) for _ in range(batch_size)]
    # Make random ele max length
    seq_len_list[random.randint(0, batch_size - 1)] = max_sequence_len
    # print(f"Theoretical padding: {pad_percentage} actual: {1 - (sum(seq_len_list) / (batch_size * max_sequence_len))}")
    return torch.nested.nested_tensor([
        torch.randn(seq_len, embed_dimension, dtype=dtype, device=device) for seq_len in seq_len_list]), seq_len_list


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


def run_timing(iters, batch_size, embed_dimension, num_heads, max_sequence_len, pad_percentage, enable_math, enable_flash, writer):
    with torch.backends.cuda.sdp_kernel(enable_math=enable_math, enable_flash=enable_flash):
        with torch.inference_mode():
            dropout_p = 0.0
            mask = None

            pt = torch.nn.MultiheadAttention(
                embed_dim=embed_dimension, num_heads=num_heads, batch_first=True, dropout=dropout_p
            )
            npt = pt.eval().half().cuda()
            cpt = build_composite_mha_from_nn_mha(npt)
            x, lengths = generate_rand_batch(batch_size, max_sequence_len, embed_dimension, pad_percentage)
            pt_output, _ = pt(x, x, x, mask)
            cpt_output, _ = cpt(x, x, x, mask)

            # First order sanity check. Not a replacement for rigorous tests.
            if pt_output.is_nested and cpt_output.is_nested:
                for a, b in zip(pt_output.unbind(), cpt_output.unbind()):
                    assert torch.allclose(a, b, atol=1e-3, rtol=1e-3)
            else:
                assert torch.allclose(pt_output, cpt_output, atol=1e-3, rtol=1e-3)

            pt_time = benchmark_torch_function(iters, npt, x, x, x, mask) * 1e3
            cp_time = benchmark_torch_function(iters, cpt, x, x, x, mask) * 1e3
            results = {}
            results["max_sequence_len"] = max_sequence_len
            results["num_heads"] = num_heads
            results["embed_dimension"] = embed_dimension
            results["pt_time"] = pt_time
            results["cp_time"] = cp_time
            results["speedup"] = pt_time / cp_time
            results["dtype"] = str(x.dtype)
            results["enable_math"] = str(enable_math)
            results["enable_flash"] = str(enable_flash)
            writer.writerow(results)


def main():
    iters = 100
    seed = 123
    np.random.seed(seed)
    torch.manual_seed(seed)

    headers = ["max_sequence_len", "num_heads", "embed_dimension", "pt_time",
               "cp_time", "speedup", "dtype", "enable_math", "enable_flash"]
    writer = csv.DictWriter(sys.stdout, headers)
    writer.writeheader()

    batch_size = 64
    pad_percentage = 0.5

    for (enable_math, enable_flash) in [(False, True), (True, False), (True, True)]:
        for num_heads, max_seq_len in itertools.product([2, 4, 8, 16, 32], [64, 128, 256]):
            run_timing(iters, batch_size, 1024, num_heads, max_seq_len,
                       pad_percentage, enable_math, enable_flash, writer)
            run_timing(iters, batch_size, 1024, num_heads, max_seq_len,
                       pad_percentage, enable_math, enable_flash, writer)
            run_timing(iters, batch_size, 1024, num_heads, max_seq_len,
                       pad_percentage, enable_math, enable_flash, writer)


if __name__ == "__main__":
    main()
