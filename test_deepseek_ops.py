import torch

def main():

    # input
    bsz = 4
    seqlen = 4096

    # embedding table
    dim = 2048

    # attention
    # wq
    n_heads = 16
    qk_head_dim = 192
    qk_nope_head_dim = 128
    qk_rope_head_dim = 64


    # real = torch.randn(4096, 32, dtype=torch.float32, device="cuda")
    # imag = torch.randn(4096, 32, dtype=torch.float32, device="cuda")
    # freqs_cis = torch.complex(real, imag)

    # from torchtitan.models.attention import build_attention, init_attention_mask
    # use_flex_attn=True
    # attn_mask_type = 'block_causal'
    # eos_id = 100001
    # sdpa = build_attention(use_flex_attn, attn_mask_type)
    # input = torch.randint(0, 10, (4, 4096), device="cuda")
    # init_attention_mask(input, eos_id)

    #  [call] op=[aten::embedding], key=[AutogradCUDA]
    #   [redispatch] op=[aten::embedding], key=[CUDA]
    #    [call] op=[aten::reshape], key=[CUDA]
    #     [call] op=[aten::view], key=[CUDA]
    #    [call] op=[aten::index_select], key=[CUDA]
    #     [call] op=[aten::empty.memory_format], key=[BackendSelect]
    #      [redispatch] op=[aten::empty.memory_format], key=[CUDA]
    #     [call] op=[aten::resize_], key=[CUDA]
    #     [call] op=[aten::view], key=[CUDA]
    #     [call] op=[aten::expand], key=[CUDA]
    #      [call] op=[aten::as_strided], key=[CUDA]
    #     [call] op=[aten::gather.out], key=[CUDA]
    #      [call] op=[aten::as_strided], key=[CUDA]
    #      [call] op=[aten::as_strided], key=[CUDA]
    #    [call] op=[aten::view], key=[CUDA]
    # model = torch.nn.Embedding(10, 10, device="cuda")
    # input = torch.randint(0, 10, (4, 4096), device="cuda")
    # model(input)

    # [call] op=[aten::rms_norm], key=[AutogradCUDA]
    #   [call] op=[aten::_fused_rms_norm], key=[AutogradCUDA]
    #    [redispatch] op=[aten::_fused_rms_norm], key=[CUDA]
    #     [call] op=[aten::empty.memory_format], key=[BackendSelect]
    #      [redispatch] op=[aten::empty.memory_format], key=[CUDA]
    #     [call] op=[aten::empty.memory_format], key=[BackendSelect]
    #      [redispatch] op=[aten::empty.memory_format], key=[CUDA]
    #     [call] op=[aten::view], key=[CUDA]
    # model = torch.nn.RMSNorm(2048, eps=0.1, device="cuda")
    # input = torch.randn(4, 4096, 2048, device="cuda")
    # model(input)

    #  [call] op=[aten::linear], key=[AutogradCUDA]
    #   [call] op=[aten::t], key=[AutogradCUDA]
    #    [redispatch] op=[aten::t], key=[ADInplaceOrView]
    #     [redispatch] op=[aten::t], key=[CUDA]
    #      [call] op=[aten::transpose.int], key=[CUDA]
    #       [call] op=[aten::as_strided], key=[CUDA]
    #   [call] op=[aten::matmul], key=[AutogradCUDA]
    #    [call] op=[aten::reshape], key=[AutogradCUDA]
    #     [call] op=[aten::view], key=[AutogradCUDA]
    #      [redispatch] op=[aten::view], key=[ADInplaceOrView]
    #       [redispatch] op=[aten::view], key=[CUDA]
    #    [call] op=[aten::mm], key=[AutogradCUDA]
    #     [redispatch] op=[aten::mm], key=[CUDA]
    #    [call] op=[aten::_unsafe_view], key=[AutogradCUDA]
    #     [redispatch] op=[aten::_unsafe_view], key=[CUDA]
    model = torch.nn.Linear(dim, 3072, bias=False, device="cuda")
    input = torch.randn(bsz, seqlen, dim, device="cuda")
    q = model(input)
    return


    #  [call] op=[aten::view], key=[AutogradCUDA]
    #   [redispatch] op=[aten::view], key=[ADInplaceOrView]
    #    [redispatch] op=[aten::view], key=[CUDA]
    q = q.view(bsz, seqlen, -1, qk_head_dim)


    #  [call] op=[aten::split_with_sizes], key=[AutogradCUDA]
    #   [redispatch] op=[aten::split_with_sizes], key=[ADInplaceOrView]
    #    [redispatch] op=[aten::split_with_sizes], key=[CUDA]
    #     [call] op=[aten::as_strided], key=[CUDA]
    #     [call] op=[aten::as_strided], key=[CUDA]
    q_nope, q_pe = torch.split(q, [qk_nope_head_dim, qk_rope_head_dim], dim=-1)


    def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
    #  [call] op=[aten::to.dtype], key=[AutogradCUDA]
    #  [call] op=[aten::view], key=[AutogradCUDA]
    #   [redispatch] op=[aten::view], key=[ADInplaceOrView]
    #    [redispatch] op=[aten::view], key=[CUDA]
        x = x.float().view(*x.shape[:-1], -1, 2)
    # [call] op=[aten::view_as_complex], key=[AutogradCUDA]
    #   [redispatch] op=[aten::view_as_complex], key=[ADInplaceOrView]
    #    [redispatch] op=[aten::view_as_complex], key=[CUDA]
        x = torch.view_as_complex(x)
        freqs_cis = freqs_cis.view(1, x.size(1), 1, x.size(-1))
    #  [call] op=[aten::view_as_real], key=[AutogradCUDA]
    #   [redispatch] op=[aten::view_as_real], key=[ADInplaceOrView]
    #    [redispatch] op=[aten::view_as_real], key=[CUDA]
    #  [call] op=[aten::flatten.using_ints], key=[AutogradCUDA]
    #   [call] op=[aten::view], key=[AutogradCUDA]
    #    [redispatch] op=[aten::view], key=[ADInplaceOrView]
    #     [redispatch] op=[aten::view], key=[CUDA]
        y = torch.view_as_real(x * freqs_cis).flatten(3)
    #  [call] op=[aten::to.dtype], key=[AutogradCUDA]
        return y.to(dtype)

    q_pe = apply_rotary_emb(q_pe, freqs_cis)
    # [call] op=[aten::cat], key=[AutogradCUDA]
    #   [redispatch] op=[aten::cat], key=[CUDA]
    q = torch.cat([q_nope, q_pe], dim=-1)  # (bsz, seqlen, n_heads, qk_head_dim)
    #  [call] op=[aten::transpose.int], key=[AutogradCUDA]
    #   [redispatch] op=[aten::transpose.int], key=[ADInplaceOrView]
    #    [redispatch] op=[aten::transpose.int], key=[CUDA]
    #     [call] op=[aten::as_strided], key=[CUDA]
    q = q.transpose(1, 2)


    # k = q.clone()
    # v = q.clone()


    # # output = sdpa(q, k, v, scale=0.07)


if __name__ == "__main__":
    main()
