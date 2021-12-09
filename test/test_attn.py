import torch
import torch.nn as nn
torch.manual_seed(0)
embed_dims = 3
num_heads = 3

# q = (L, N, Eq)
# k = (S, N, Ek)
# v = (S, N, Ev)
# key_padding_mask = (N, S)
# attn_mask = (L, S) or (N * num_heads, L, S)

# Batch First = True

multihead_attn = nn.MultiheadAttention(embed_dims, num_heads, batch_first=True)

q = torch.randn(5, 3, 3)
k = torch.randn(5, 3, 3)
v = torch.randn(5, 3, 3)

output = multihead_attn(q, k, v)

for i in range(5):
    q_s = q[i]
    k_s = k[i]
    v_s = v[i]
    o_s = multihead_attn(q_s, k_s, v_s)
    torch.testing.assert_allclose(output[0][i], o_s[0])

key_padding_mask = torch.randn(5, 3).bernoulli_().bool()
output = multihead_attn(q, k, v, key_padding_mask)

for i in range(5):
    q_s = q[i]
    k_s = k[i]
    v_s = v[i]
    o_s = multihead_attn(q_s, k_s, v_s, key_padding_mask[i])
    torch.testing.assert_allclose(output[0][i], o_s[0])

attn_mask = torch.randn(15, 3, 3)
output = multihead_attn(q, k, v, key_padding_mask, attn_mask=attn_mask)

for i in range(5):
    q_s = q[i]
    k_s = k[i]
    v_s = v[i]
    o_s = multihead_attn(q_s, k_s, v_s, key_padding_mask[i],
                         attn_mask=attn_mask[i * num_heads:i * num_heads + num_heads, :, :])
    torch.testing.assert_allclose(output[0][i], o_s[0])

# Batch First = False

multihead_attn = nn.MultiheadAttention(embed_dims, num_heads, batch_first=False, bias=True, add_bias_kv=True, add_zero_attn=True)

q = torch.randn(3, 5, 3)
k = torch.randn(3, 5, 3)
v = torch.randn(3, 5, 3)

output = multihead_attn(q, k, v)

for i in range(5):
    q_s = q[:, i, :]
    k_s = k[:, i, :]
    v_s = v[:, i, :]
    o_s = multihead_attn(q_s, k_s, v_s)
    torch.testing.assert_allclose(output[0][:, i, :], o_s[0])

key_padding_mask = torch.randn(5, 3).bernoulli_().bool()
output = multihead_attn(q, k, v, key_padding_mask)

for i in range(5):
    q_s = q[:, i, :]
    k_s = k[:, i, :]
    v_s = v[:, i, :]
    o_s = multihead_attn(q_s, k_s, v_s, key_padding_mask[i])
    torch.testing.assert_allclose(output[0][:, i, :], o_s[0])

attn_mask = torch.randn(15, 3, 3)
output = multihead_attn(q, k, v, key_padding_mask, attn_mask=attn_mask)

for i in range(5):
    q_s = q[:, i, :]
    k_s = k[:, i, :]
    v_s = v[:, i, :]
    o_s = multihead_attn(q_s, k_s, v_s, key_padding_mask[i],
                         attn_mask=attn_mask[i * num_heads:i * num_heads + num_heads, :, :])
    torch.testing.assert_allclose(output[0][:, i, :], o_s[0])
