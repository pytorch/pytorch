import torch

device = 'mps'

query = torch.rand(32, 8, 128, 64, dtype=torch.float16, device=device)
key = torch.rand(32, 8, 128, 64, dtype=torch.float16, device=device)
value = torch.rand(32, 8, 128, 64, dtype=torch.float16, device=device)
multihead_attn = torch.nn.MultiheadAttention(8 * 128, 8)
with torch.backends.cuda.sdp_kernel(enable_math=False):
    print(multihead_attn(query, key, value)[0])
