import torch
from flash_attn.flash_attention import FlashAttention


def flash_attn(qkv, bs, sq, num_heads, sm_scale, mask=None):
    attn = FlashAttention(softmax_scale = sm_scale)
    out = attn(qkv, key_padding_mask=mask)[0]
    return out.view(bs, sq, -1)

def std_attn_forward(qkv, B, N, num_heads, scale, mask=None):
    qkv = qkv.permute(2, 0, 3, 1, 4) # (3, bs, nhead, sq, hdim)
    q, k, v = qkv[0], qkv[1], qkv[2]   # (bs, nhead, sq, hdim))
    q = q * scale
    attn = (q @ k.transpose(-2, -1)) # (bs, nheads, sq-q, sq-k)
    if mask is not None:
        mask = mask.bool()
        attn = attn.masked_fill(~mask[:, None, None, :], float("-inf"))
    attn = attn.softmax(dim=-1).type_as(q)
    x = (attn @ v)
    x = x.transpose(1, 2).reshape(B, N, -1)
    return x

if __name__ == '__main__':


    # for nheads in [8, 16]:
    for nheads in [8]:

        bs, sq, fdim, scale = 8, 256, 1024, 0.1
        qkv = torch.randn(bs, sq, fdim*3).half().cuda()
        qkv = qkv.view(bs, sq, 3, nheads, -1)

        mask = None
        out1 = flash_attn(qkv, bs, sq, nheads, scale, mask=mask)
        out2 = std_attn_forward(qkv, bs, sq, nheads, scale, mask=mask)
        print('nheads', nheads, 'same with mask off:', torch.allclose(out1, out2, atol=1e-3))

        mask = (torch.arange(sq) < sq//2).unsqueeze(0).repeat(bs, 1).cuda()
        out1 = flash_attn(qkv, bs, sq, nheads, scale, mask=mask)
        out2 = std_attn_forward(qkv, bs, sq, nheads, scale, mask=mask)

        fmask = mask.float().unsqueeze(2)
        out1 = out1 * fmask
        out2 = out2 * fmask
        print('nheads', nheads, 'same with mask on:', torch.allclose(out1, out2, atol=1e-3))
        breakpoint()
