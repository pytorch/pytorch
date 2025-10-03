import torch

from torch.nn.attention.flex_attention import create_block_mask, flex_attention

flex_attention = torch.compile(
    torch.nn.attention.flex_attention.flex_attention, dynamic=False, fullgraph=True
)


def causal_mask(b, h, q_idx, kv_idx):
    return q_idx >= kv_idx


qkv = [torch.rand((1, 1, 128, 32), device="cuda") for _ in range(3)]
out = flex_attention(
    *qkv,
    block_mask=create_block_mask(causal_mask, 1, 1, 128, 128, device="cuda"),
    return_aux=torch.nn.attention.flex_attention.AuxRequest(lse=False)
)

torch.set_printoptions(threshold=100000)
print(out)
