import functools

import torch
from torch.distributed.tensor.experimental._attention import PTRRLoadBalancer

from torch.nn.attention.flex_attention import create_block_mask


compiled_create_block_mask = torch.compile(
    create_block_mask, dynamic=False, fullgraph=True
)


def causal_mask(rank, b, h, q_idx, kv_idx):
    return q_idx + rank * 128 >= kv_idx


for rank in range(4):
    mask_func = functools.partial(causal_mask, rank)

    bm = create_block_mask(
        mask_func,
        B=1,
        H=1,
        Q_LEN=128,
        KV_LEN=512,
        device="cuda",
    )

    print(f"{bm}, {bm.__repr__}, {bm.kv_num_blocks}, {bm.q_num_blocks}")

block_mask = create_block_mask(
    functools.partial(causal_mask, 0),
    B=1,
    H=1,
    Q_LEN=512,
    KV_LEN=512,
    device="cuda",
)
lb = PTRRLoadBalancer(block_mask, 2, "cuda")
idx = lb.generate_indices()
restore_idx = lb.generate_indices(restore=True)
x = torch.arange(512, device="cuda")
y = x[idx]
print(f"y={y}")
print(f"restore_idx={restore_idx}")
print(f"z={y[:, restore_idx[0]]}")
# z = y[restore_idx]
# print(f"z={z}")
# print(lb.generate_indices())
# print(lb.generate_indices(restore=True))
