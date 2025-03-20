import torch

def func(q, k, v):
    output = torch._scaled_dot_product_int8(
        q,
        k,
        v,
        attn_mask=None,
        scale=0.125,  # scale
        dropout_p=0.0,  # dropout
        is_causal=False,  # is_causal
        q_zp=0,
        q_scale=1,
        k_zp=0,
        k_scale=1,
        v_zp=0,
        v_scale=1,
        a_zp=0,
        a_scale=1,
        o_zp=0,
        o_scale=1,
    )
    return output

q = torch.randn((16, 64, 120, 64), dtype=torch.float, device='cpu') * 100
k = torch.randn((16, 64, 120, 64), dtype=torch.float, device='cpu') * 100
v = torch.randn((16, 64, 120, 64), dtype=torch.float, device='cpu') * 100
q = q.to(torch.uint8)
k = k.to(torch.uint8)
v = v.to(torch.uint8)
compiled_func = torch.compile(func, backend="inductor")

print(compiled_func(q, k, v))

