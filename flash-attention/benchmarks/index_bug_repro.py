import torch

from flash_attn.flash_attn_interface import flash_attn_unpadded_func


torch.manual_seed(0)
batch_size = 32
seqlen_q = 97
seqlen_k = 573
nheads = 16
d = 64
dropout_p = 0.1
causal = True
dtype = torch.float16
device = 'cuda'

q = torch.randn(batch_size * seqlen_q, nheads, d, dtype=dtype, device=device, requires_grad=True)
k = torch.randn(batch_size * seqlen_k, nheads, d, dtype=dtype, device=device, requires_grad=True)
v = torch.randn(batch_size * seqlen_k, nheads, d, dtype=dtype, device=device, requires_grad=True)

# cu_seqlens_q = torch.arange(0, seqlen_q * (batch_size + 1), step=seqlen_q, device=device, dtype=torch.int32)
# cu_seqlens_k = torch.arange(0, seqlen_k * (batch_size + 1), step=seqlen_k, device=device, dtype=torch.int32)
cu_seqlens_q = torch.tensor([0] + [seqlen_q] * batch_size, dtype=torch.int32, device=device).cumsum(dim=-1).to(torch.int32)
cu_seqlens_k = torch.tensor([0] + [seqlen_k] * batch_size, dtype=torch.int32, device=device).cumsum(dim=-1).to(torch.int32)

out = flash_attn_unpadded_func(q, k, v, cu_seqlens_q, cu_seqlens_k, seqlen_q, seqlen_k,
                               dropout_p, causal=causal)
print(out.mean().item())  # To synchronize CUDA
g = torch.randn_like(out)
out.backward(g)
print(q.grad.mean().item(), k.grad.mean().item(), v.grad.mean().item())  # To synchronize CUDA
