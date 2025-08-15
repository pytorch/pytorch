import torch
from torch.nn.attention.flex_attention import flex_attention

query = torch.randn(1, 12, 1, 16, device="cuda")
key = torch.randn(1, 2, 4096, 16, device="cuda")
value = torch.randn(1, 2, 4096, 16, device="cuda")


flex_compiled = torch.compile(flex_attention, fullgraph=True)

out = flex_compiled(query, key, value, enable_gqa=True)
