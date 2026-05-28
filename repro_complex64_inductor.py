import torch

def target_function(
    l_tok_embeddings_weight_, l_tokens_, l_attn_norm_0_weight_,
    l_wq_0_weight_, l_wk_0_weight_, l_wv_0_weight_, l_wo_0_weight_
):
    arange = torch.arange(0, 8, 2, device='cuda', dtype=torch.float32)
    angles = torch.true_divide(arange, 8)
    pow_1 = torch.pow(10000.0, angles)
    freqs = torch.true_divide(1.0, pow_1)
    t = torch.arange(0, 4, device='cuda', dtype=torch.float32)
    outer = torch.outer(t, freqs)
    freqs_1 = outer.float()
    ones_like = torch.ones_like(freqs_1)
    freqs_cis = torch.polar(ones_like, freqs_1)

    h = torch.nn.functional.embedding(l_tokens_, l_tok_embeddings_weight_)
    x_normed = h.float()
    pow_2 = x_normed.pow(2)
    mean = pow_2.mean(-1, keepdim=True)
    add = torch.add(mean, 1e-05)
    rsqrt = torch.rsqrt(add)
    x_normed_1 = torch.mul(x_normed, rsqrt)
    type_as = x_normed_1.type_as(h)
    normed_x = torch.mul(type_as, l_attn_norm_0_weight_)

    q = torch.nn.functional.linear(normed_x, l_wq_0_weight_)
    k = torch.nn.functional.linear(normed_x, l_wk_0_weight_)
    v = torch.nn.functional.linear(normed_x, l_wv_0_weight_)

    view = q.view(2, 4, 8, 8)
    q_1 = view.transpose(1, 2)
    view_1 = k.view(2, 4, 8, 8)
    k_1 = view_1.transpose(1, 2)
    view_2 = v.view(2, 4, 8, 8)
    v_1 = view_2.transpose(1, 2)

    float_3 = q_1.float()
    reshape = float_3.reshape(2, 8, 4, -1, 2)
    xq_ = torch.view_as_complex(reshape)

    float_4 = k_1.float()
    reshape_1 = float_4.reshape(2, 8, 4, -1, 2)
    xk_ = torch.view_as_complex(reshape_1)

    freqs_cis_1 = freqs_cis.reshape(1, 1, 4, -1)
    mul_2 = torch.mul(xq_, freqs_cis_1)
    view_as_real = torch.view_as_real(mul_2)
    xq_out = view_as_real.flatten(3)

    mul_3 = torch.mul(xk_, freqs_cis_1)
    view_as_real_1 = torch.view_as_real(mul_3)
    xk_out = view_as_real_1.flatten(3)

    q_2 = xq_out.type_as(q_1)
    k_2 = xk_out.type_as(k_1)

    mask = torch.full([1, 1, 4, 4], float('-inf'), device='cuda', dtype=torch.float32)
    mask_1 = torch.triu(mask, diagonal=1)
    transpose_3 = k_2.transpose(-2, -1)
    matmul = torch.matmul(q_2, transpose_3)

    scores = torch.mul(matmul, 0.35355339059327373)
    scores_1 = torch.add(scores, mask_1)
    attn_weights = torch.nn.functional.softmax(scores_1, dim=-1)
    attn_output = torch.matmul(attn_weights, v_1)

    transpose_4 = attn_output.transpose(1, 2)
    contiguous = transpose_4.contiguous()
    attn_output_1 = contiguous.view(2, 4, 64)
    linear_3 = torch.nn.functional.linear(attn_output_1, l_wo_0_weight_)

    return (arange, t, mask, h, angles, mask_1, x_normed, pow_1, pow_2, freqs, mean, outer, add, freqs_1, rsqrt, ones_like, x_normed_1, freqs_cis, type_as, freqs_cis_1, normed_x, q, k, v, view, view_1, view_2, q_1, k_1, v_1, float_3, float_4, reshape, reshape_1, xq_, xk_, mul_2, mul_3, view_as_real, view_as_real_1, xq_out, xk_out, q_2, k_2, transpose_3, matmul, scores, scores_1, attn_weights, attn_output, transpose_4, contiguous, attn_output_1, linear_3)


def get_inputs():
    torch.manual_seed(2077)
    return (
        (torch.randn([512, 64], dtype=torch.float32) * 0.1).cuda(),
        torch.zeros([2, 4], dtype=torch.int64).cuda(),
        (torch.randn([64], dtype=torch.float32) * 0.1).cuda(),
        (torch.randn([64, 64], dtype=torch.float32) * 0.1).cuda(),
        (torch.randn([64, 64], dtype=torch.float32) * 0.1).cuda(),
        (torch.randn([64, 64], dtype=torch.float32) * 0.1).cuda(),
        (torch.randn([64, 64], dtype=torch.float32) * 0.1).cuda(),
    )

if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA is not available. Please run this script on a GPU node.")
        exit(1)

    inputs = get_inputs()

    print("Running torch.compile(mode='autotune') on GPU...")
    opt_fn = torch.compile(target_function, mode="max-autotune")

    try:
        opt_fn(*inputs)
        print("Compilation Passed (Bug not triggered).")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"\n[BINGO!] Crashed with: {type(e).__name__}: {e}")