"""Run as follows (with fish shell, slightly adapt syntax for other shells):

Fast path without chunking:

> for s in 8 16 32 64 128 256 512 1024; echo -n "s="(math "$s * 1024")" " ; env CUDA_VISIBLE_DEVICES=5 python3 batched_vmap_slow.py -s (math "$s * 1024") 2>/dev/null ; end

Output on my machine:

s=8192 1103.6ms 0.2ms 0.1ms 0.0ms 0.0ms 0.0ms 0.0ms 0.0ms 0.0ms 0.0ms
s=16384 1239.7ms 0.2ms 0.1ms 0.1ms 0.1ms 0.0ms 0.0ms 0.0ms 0.0ms 0.0ms
s=32768 1506.8ms 0.2ms 0.1ms 0.1ms 0.1ms 0.0ms 0.0ms 0.0ms 0.0ms 0.0ms
s=65536 1563.4ms 0.2ms 0.1ms 0.1ms 0.1ms 0.0ms 0.0ms 0.0ms 0.0ms 0.0ms
s=131072 2554.0ms 0.2ms 0.1ms 0.0ms 0.0ms 0.0ms 0.0ms 0.0ms 0.0ms 0.0ms
s=262144 s=524288 s=1048576 (All 3 OOM)

Slow path with chunking (just add `-c 1024`):

> for s in 8 16 32 64 128 256 512 1024; echo -n "s="(math "$s * 1024")" " ; env CUDA_VISIBLE_DEVICES=5 python3 batched_vmap_slow.py -s (math "$s * 1024") -c 1024 2>/dev/null ; end

Output on my machine:

s=8192 2644.7ms 0.1ms 0.1ms 0.1ms 0.0ms 0.0ms 0.0ms 0.0ms 0.0ms 0.0ms
s=16384 6848.3ms 3.5ms 2.5ms 2.4ms 2.4ms 2.4ms 2.4ms 2.3ms 2.3ms 2.3ms
s=32768 47551.9ms 14.9ms 12.8ms 8.9ms 9.2ms 8.9ms 8.7ms 8.4ms 8.1ms 8.5ms
s=65536 184588.4ms 59.8ms 54.5ms 43.3ms 38.9ms 35.9ms 36.2ms 36.3ms 36.0ms 35.9ms
s=131072 779158.3ms 213.1ms 201.0ms 199.3ms 199.3ms 203.6ms 206.8ms 202.8ms 206.1ms 203.9ms
(killed longer ones)
"""
import argparse
import time

import torch


device = "cpu"


def main(seqlen, chunksize=None):
    # cbm = torch.compile(create_block_mask)
    # cbm = torch.compile(create_mask, backend="eager", fullgraph=True)
    cbm = create_mask

    def get_mask():
        def mask_mod(b, h, q_idx, kv_idx):
            return q_idx >= kv_idx
        return cbm(mask_mod, Q_LEN=seqlen, KV_LEN=seqlen, device=device, vmap_chunk_size=chunksize)

    ts = []
    repeat = 1
    for _ in range(repeat):
        t0 = time.monotonic_ns()
        _ = get_mask()
        ts.append(time.monotonic_ns() - t0)
        print(f"{ts[-1] / 1_000_000:.1f}ms", end=" ", flush=True)
    print("", flush=True)



####  The below is copied from pytorch only adding chunk_size to vmap.
# https://github.com/pytorch/pytorch/blob/master/torch/nn/attention/flex_attention.py#L996
def create_mask(mod_fn, Q_LEN, KV_LEN, device="cuda", vmap_chunk_size=None):
    b = torch.arange(0, 1, device=device)
    h = torch.arange(0, 1, device=device)
    m = torch.arange(0, Q_LEN, device=device)
    n = torch.arange(0, KV_LEN, device=device)

    from torch._dynamo._trace_wrapped_higher_order_op import TransformGetItemToIndex

    with TransformGetItemToIndex():
        mask_mod = mod_fn
        mask_mod = _vmap_for_bhqkv(mask_mod, vmap_chunk_size)
        mask = mask_mod(b, h, m, n)
        return mask


# Need to define it here so that Dynamo doesn't skip it
def _vmap_for_bhqkv(fn, chunk_size=None):
    # We vamp a function 4 times, broadcasting the [b, h, q_idx, kv_idx] dimensions
    dimensions = [
        (None, None, None, 0),
        (None, None, 0, None),
        (None, 0, None, None),
        (0, None, None, None),
    ]

    for dims in dimensions:
        fn = torch.vmap(fn, in_dims=dims, out_dims=0, chunk_size=chunk_size)
    return fn


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="iykyk")
    parser.add_argument("-s", "--seqlen", type=int, default=32_768,
                        help="Sequence length in tokens.")
    parser.add_argument("-c", "--chunksz", type=int, default=None,
                        help="Chunk size for vmap.")
    args = parser.parse_args()

    main(seqlen=args.seqlen, chunksize=args.chunksz)
