import torch
import argparse
import time


device = "cpu"


def fn(x, y):
    return x * 100 + y


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


def _vmap_for_bhqkv(fn, chunk_size=None):
    # We vamp a function 4 times, broadcasting the [b, h, q_idx, kv_idx] dimensions
    dimensions = [
        (None, None, None, 0),
        # (None, None, 0, None),
        # (None, 0, None, None),
        # (0, None, None, None),
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
