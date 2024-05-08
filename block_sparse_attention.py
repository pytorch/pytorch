"""
Fused Attention
===============

This is a Triton implementation of the Flash Attention v2 algorithm from Tri Dao (https://tridao.me/publications/flash2/flash2.pdf)
Credits: OpenAI kernel team

Extra Credits:
- Original flash attention paper (https://arxiv.org/abs/2205.14135)
- Rabe and Staats (https://arxiv.org/pdf/2112.05682v2.pdf)

"""

import pytest
import torch
import torch.nn.functional as F
torch.set_default_device('cuda')

import triton
import triton.language as tl
from triton.testing import do_bench

def broadcast_to_dim(x, dim):
    while x.dim() < dim:
        x = x.unsqueeze(0)
    return x

DEFAULT_BLOCK=1024
def convert_mask_to_block_mask(mask, BLOCK=DEFAULT_BLOCK):
    assert mask.dtype == torch.bool
    mask = broadcast_to_dim(mask, 4)
    # mask: [B, H, Q, KV]
    B, H, Q, KV = mask.shape
    mask = mask.view(B, H, Q//BLOCK, BLOCK, KV//BLOCK, BLOCK) # [B, H, Q//BLOCK, BLOCK, KV//BLOCK, BLOCK]
    mask = mask.permute(0, 1, 2, 4, 3, 5) # [B, H, Q//BLOCK, KV//BLOCK, BLOCK, BLOCK]
    mask = mask.sum(dim=[-2, -1]) > 0 # [B, H, Q//BLOCK, KV//BLOCK]
    return mask

def convert_block_mask_to_mask(block_mask, BLOCK=DEFAULT_BLOCK):
    assert block_mask.dim() == 4
    B, H, Q, KV = block_mask.shape
    block_mask = block_mask.expand(BLOCK, BLOCK, *block_mask.shape)
    block_mask = block_mask.permute(2, 3, 4, 0, 5, 1).reshape(B, H, Q * BLOCK, KV * BLOCK)
    return block_mask

def convert_block_mask_to_blocksparse_info(block_mask):
    # block_mask: [B, H, Q//BLOCK, KV//BLOCK]
    assert block_mask.dtype == torch.bool
    assert block_mask.dim() == 4
    block_mask = block_mask.to(dtype=torch.int8)
    num_blocks_per_row = block_mask.sum(dim=3) # [B, H, Q//BLOCK]
    query_indices = torch.argsort(block_mask, dim=3, descending=True, stable=True)
    return query_indices.to(dtype=torch.int32), num_blocks_per_row.to(dtype=torch.int32)

def create_mask_from_score_mod(score_mod, B, H, M, N):
    device = 'cuda'
    b = torch.arange(0, B, device=device)
    h = torch.arange(0, H, device=device)
    m = torch.arange(0, M, device=device)
    n = torch.arange(0, N, device=device)
    score_mod = torch.vmap(score_mod, in_dims=(0, None, None, None, 0))
    score_mod = torch.vmap(score_mod, in_dims=(0, None, None, 0, None))
    score_mod = torch.vmap(score_mod, in_dims=(0, None, 0, None, None))
    score_mod = torch.vmap(score_mod, in_dims=(0, 0, None, None, None))
    out =  score_mod(torch.zeros(B, H, M, N, device=device), b, h, m, n).to(dtype=torch.bool)
    return out


# x_block_mask = convert_mask_to_block_mask(x > 0)
# query_indices, num_blocks = convert_block_mask_to_blocksparse_info(x_block_mask)


@triton.jit
def _attn_fwd_inner(acc, l_i, m_i, q,  #
                    K_block_ptr, V_block_ptr,  #
                    start_m, qk_scale,  #
                    kv_indices, num_blocks,
                    BLOCK_M: tl.constexpr, BLOCK_DMODEL: tl.constexpr, BLOCK_N: tl.constexpr,  #
                    offs_m: tl.constexpr, offs_n: tl.constexpr,  #
                    N_CTX: tl.constexpr,
                    BLOCKSPARSE_KV: tl.constexpr):
    # range of values handled by this stage
    # lo, hi = 0, (start_m + 1) * BLOCK_M # causal
    lo = 0
    # blocksparsity block size relative to BLOCK_N
    BLOCKSPARSE_KV_MULTIPLE = (BLOCKSPARSE_KV // BLOCK_N)
    hi = num_blocks * BLOCKSPARSE_KV_MULTIPLE
    # print("hi", hi)
    # lo, hi = 0, N_CTX # noncausal

    K_block_ptr = tl.advance(K_block_ptr, (0, lo))
    # loop over k, v and update accumulator
    # print(N_CTX // BLOCK_N, num_blocks)
    for start_n in range(0, hi):
        # start_n = tl.multiple_of(start_n, BLOCK_N)
        # -- compute qk ----
        k = tl.load(K_block_ptr)
        qk = tl.dot(q, k)
        # causal
        # mask = offs_m[:, None] >= (start_n + offs_n[None, :])
        # qk = qk * qk_scale + tl.where(mask, 0, -1.0e6)
        # m_ij = tl.maximum(m_i, tl.max(qk, 1))
        # qk -= m_ij[:, None]
        # noncausal
        m_ij = tl.maximum(m_i, tl.max(qk, 1) * qk_scale)
        qk = qk * qk_scale - m_ij[:, None]
        p = tl.math.exp2(qk)
        l_ij = tl.sum(p, 1)
        # -- update m_i and l_i
        alpha = tl.math.exp2(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        # -- update output accumulator --
        acc = acc * alpha[:, None]
        # update acc
        v = tl.load(V_block_ptr)
        p = p.to(tl.float16)
        acc = tl.dot(p, v, acc)
        # update m_i and l_i
        m_i = m_ij


        # TODO: This can probably be optimized...
        indices_idx = start_n // BLOCKSPARSE_KV_MULTIPLE
        cur_block = tl.load(kv_indices + indices_idx)
        next_block = tl.load(kv_indices + indices_idx + 1)
        needs_jump = (start_n + 1) % BLOCKSPARSE_KV_MULTIPLE == 0
        jump_to_block = (next_block - cur_block ) * BLOCKSPARSE_KV - (BLOCKSPARSE_KV_MULTIPLE - 1) * BLOCK_N
        offset = jump_to_block * needs_jump + (1 - needs_jump) * BLOCK_N

        V_block_ptr = tl.advance(V_block_ptr, (offset, 0))
        K_block_ptr = tl.advance(K_block_ptr, (0, offset))
    return acc, l_i, m_i



@triton.jit
def _attn_fwd(Q, K, V, sm_scale, M, Out,  #
              stride_qz, stride_qh, stride_qm, stride_qk,  #
              stride_kz, stride_kh, stride_kn, stride_kk,  #
              stride_vz, stride_vh, stride_vk, stride_vn,  #
              stride_oz, stride_oh, stride_om, stride_on,  #
              Z, H,  #
              kv_indices, num_blocks,
              stride_iz, stride_ih, stride_iq, stride_ik,
              N_CTX: tl.constexpr,  #
              BLOCK_M: tl.constexpr,  #
              BLOCK_DMODEL: tl.constexpr,  #
              BLOCK_N: tl.constexpr,  #
              BLOCKSPARSE_Q: tl.constexpr,
              BLOCKSPARSE_KV: tl.constexpr,
):
    tl.static_assert(BLOCKSPARSE_Q >= BLOCK_M and BLOCKSPARSE_Q % BLOCK_M == 0)
    tl.static_assert(BLOCKSPARSE_KV >= BLOCK_N and BLOCKSPARSE_KV % BLOCK_N == 0)
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H
    qvk_offset = off_z.to(tl.int64) * stride_qz + off_h.to(tl.int64) * stride_qh
    BLOCKSPARSE_Q_MULTIPLE = (BLOCKSPARSE_Q // BLOCK_M)
    indices_offset = off_z.to(tl.int64) * stride_iz + off_h.to(tl.int64) * stride_ih + (start_m // BLOCKSPARSE_Q_MULTIPLE) * stride_iq
    block_q_index = start_m // BLOCKSPARSE_Q_MULTIPLE
    kv_indices = kv_indices + indices_offset

    # block pointers
    Q_block_ptr = tl.make_block_ptr(
        base=Q + qvk_offset,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(stride_qm, stride_qk),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0),
    )
    v_order: tl.constexpr = (0, 1) if V.dtype.element_ty == tl.float8e5 else (1, 0)
    kv_start = tl.load(kv_indices) * BLOCKSPARSE_Q # first kv block we're loading

    V_block_ptr = tl.make_block_ptr(
        base=V + qvk_offset,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(stride_vk, stride_vn),
        offsets=(kv_start, 0),
        block_shape=(BLOCK_N, BLOCK_DMODEL),
        order=v_order,
    )
    K_block_ptr = tl.make_block_ptr(
        base=K + qvk_offset,
        shape=(BLOCK_DMODEL, N_CTX),
        strides=(stride_kk, stride_kn),
        offsets=(0, kv_start),
        block_shape=(BLOCK_DMODEL, BLOCK_N),
        order=(0, 1),
    )
    O_block_ptr = tl.make_block_ptr(
        base=Out + qvk_offset,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(stride_om, stride_on),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0),
    )
    # initialize offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    # initialize pointer to m and l
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    # load scales
    qk_scale = sm_scale
    qk_scale *= 1.44269504  # 1/log(2)
    # load q: it will stay in SRAM throughout
    q = tl.load(Q_block_ptr)
    acc, l_i, m_i = _attn_fwd_inner(acc, l_i, m_i, q, K_block_ptr, V_block_ptr,  #
                                    start_m, qk_scale,  #
                                    kv_indices, tl.load(num_blocks + block_q_index),
                                    BLOCK_M, BLOCK_DMODEL, BLOCK_N,  #
                                    offs_m, offs_n, N_CTX,
                                    BLOCKSPARSE_KV=BLOCKSPARSE_KV,
    )
    # epilogue
    m_i += tl.math.log2(l_i)
    acc = acc / l_i[:, None]
    m_ptrs = M + off_hz * N_CTX + offs_m
    tl.store(m_ptrs, m_i)
    tl.store(O_block_ptr, acc.to(Out.type.element_ty))


class _attention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, sm_scale, kv_indices=None, num_blocks=None):
        # shape constraints
        Q_LEN = q.shape[-2]
        KV_LEN = k.shape[-2]
        B, H = q.shape[0], q.shape[1]
        Lq, Lk, Lv = q.shape[-1], k.shape[-1], v.shape[-1]
        # when v is in float8_e5m2 it is transposed.
        assert Lq == Lk and (Lk == Lv or v.dtype == torch.float8_e5m2)
        assert Lk in {16, 32, 64, 128, 256}
        o = torch.empty_like(q)
        extra_kern_args = {}
        # Tuning for AMD target
        BLOCK_M = 128
        BLOCK_N = 64 if Lk <= 64 else 32
        num_stages = 4 if Lk <= 64 else 3
        num_warps = 4

        grid = (triton.cdiv(q.shape[2], BLOCK_M), q.shape[0] * q.shape[1], 1)
        M = torch.empty((q.shape[0], q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32)
        BLOCKSPARSE_Q = DEFAULT_BLOCK
        if kv_indices is None:
            assert num_blocks is None
            kv_indices = torch.arange(Q_LEN//BLOCKSPARSE_Q, dtype=torch.int).expand(Q_LEN//BLOCKSPARSE_Q, KV_LEN//BLOCKSPARSE_Q).contiguous()
            num_blocks = torch.zeros((1, 1, Q_LEN//BLOCKSPARSE_Q + 1), dtype=torch.int).fill_(kv_indices.shape[-1])

        kv_indices = kv_indices.expand(B, H, kv_indices.shape[-2], kv_indices.shape[-1])
        num_blocks = num_blocks.expand(B, H, num_blocks.shape[-1])
        kv_indices = kv_indices.contiguous()
        num_blocks = num_blocks.contiguous()

        _attn_fwd[grid](
            q, k, v, sm_scale, M, o,  #
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),  #
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),  #
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),  #
            o.stride(0), o.stride(1), o.stride(2), o.stride(3),  #
            q.shape[0], q.shape[1],  #
            kv_indices, num_blocks,
            kv_indices.stride(0), kv_indices.stride(1), kv_indices.stride(2), kv_indices.stride(3),
            N_CTX=q.shape[2],  #
            BLOCK_M=BLOCK_M,  #
            BLOCK_N=BLOCK_N,  #
            BLOCK_DMODEL=Lk,  #
            BLOCKSPARSE_Q=BLOCKSPARSE_Q, #
            BLOCKSPARSE_KV=BLOCKSPARSE_Q, # making it the same for now
            num_warps=num_warps,  #
            num_stages=num_stages,  #
            **extra_kern_args)

        ctx.save_for_backward(q, k, v, o, M)
        ctx.grid = grid
        ctx.sm_scale = sm_scale
        ctx.BLOCK_DMODEL = Lk
        return o


attention = _attention.apply

@pytest.mark.parametrize("Z, H, N_CTX, D_HEAD", [(8, 16, 8192, 64)])
# @pytest.mark.parametrize("Z, H, N_CTX, D_HEAD", [(1, 1, 1024, 64)])
def test_blocksparse_attention(Z, H, N_CTX, D_HEAD, dtype=torch.float16):
    q = torch.randn((Z, H, N_CTX, D_HEAD), dtype=dtype)
    k = torch.randn((Z, H, N_CTX, D_HEAD), dtype=dtype)
    v = torch.randn((Z, H, N_CTX, D_HEAD), dtype=dtype)
    sm_scale = 0.5
    full_mask = torch.rand(N_CTX, N_CTX) > 0
    causal_mask = torch.rand(N_CTX, N_CTX).tril() > 0
    normal_sparse_mask = torch.rand(N_CTX, N_CTX) > 0.1
    def sliding_window(_, b, h, m, n):
        return (m - n).abs() < 1024
    def sliding_window_causal(_, b, h, m, n):
        return ((m - n).abs() < 1024) & (m > n)

    sliding_window_mask = create_mask_from_score_mod(sliding_window, 1, 1, N_CTX, N_CTX)
    sliding_window_mask_causal = create_mask_from_score_mod(sliding_window_causal, 1, 1, N_CTX, N_CTX)
    for name, mask in [
        ("full", full_mask),
        ("causal", causal_mask),
        ("sparse", normal_sparse_mask),
        ("sliding window", sliding_window_mask),
        ("sliding window + causal", sliding_window_mask_causal),
    ]:
        block_mask = convert_mask_to_block_mask(mask)
        sparsity = block_mask.sum()/block_mask.numel()
        query_indices, num_blocks_per_query_block = convert_block_mask_to_blocksparse_info(block_mask)
        blocksparse_f = lambda: attention(q, k, v, sm_scale, query_indices, num_blocks_per_query_block)
        out = blocksparse_f()
        mask_with_blocks = convert_block_mask_to_mask(block_mask)
        eager_f = lambda: F.scaled_dot_product_attention(q, k, v, scale=sm_scale, attn_mask=mask_with_blocks)
        out_ref = eager_f()
        assert (out - out_ref).abs().max() < 1e-2
        eager_time = do_bench(lambda: eager_f())
        sparse_time = do_bench(lambda: blocksparse_f())
        print(f"{name}: eager: {eager_time}, blocksparse: {sparse_time}, sparsity: {sparsity}")




@pytest.mark.parametrize("Z, H, N_CTX, D_HEAD", [(1, 2, 1024, 64)])
@pytest.mark.parametrize("causal", [False])
def test_op(Z, H, N_CTX, D_HEAD, causal, dtype=torch.float16):
    torch.manual_seed(20)
    q = (torch.empty((Z, H, N_CTX, D_HEAD), dtype=dtype, device="cuda").normal_(mean=0.0, std=0.5))
    k = (torch.empty((Z, H, N_CTX, D_HEAD), dtype=dtype, device="cuda").normal_(mean=0.0, std=0.5))
    v = (torch.empty((Z, H, N_CTX, D_HEAD), dtype=dtype, device="cuda").normal_(mean=0.0, std=0.5))
    sm_scale = 0.5
    dout = torch.randn_like(q)
    # reference implementation
    M = torch.tril(torch.ones((N_CTX, N_CTX), device="cuda"))
    p = torch.matmul(q, k.transpose(2, 3)) * sm_scale
    if causal:
        p[:, :, M == 0] = float("-inf")
    p = torch.softmax(p.float(), dim=-1).half()
    # p = torch.exp(p)
    ref_out = torch.matmul(p, v)
    # ref_out.backward(dout)
    # ref_dv, v.grad = v.grad.clone(), None
    # ref_dk, k.grad = k.grad.clone(), None
    # ref_dq, q.grad = q.grad.clone(), None
    # triton implementation
    tri_out = attention(q, k, v, sm_scale).half()
    # tri_out.backward(dout)
    # tri_dv, v.grad = v.grad.clone(), None
    # tri_dk, k.grad = k.grad.clone(), None
    # tri_dq, q.grad = q.grad.clone(), None
    # compare
    assert torch.allclose(ref_out, tri_out, atol=1e-2, rtol=0)
    return
    rtol = 0.0
    # Relative tolerance workaround for known hardware limitation of MI200 GPU.
    # For detailss see https://pytorch.org/docs/stable/notes/numerical_accuracy.html#reduced-precision-fp16-and-bf16-gemms-and-convolutions-on-amd-instinct-mi200-devices
    if torch.version.hip is not None and triton.runtime.driver.active.get_current_target().arch == "gfx90a":
        rtol = 1e-2
    assert torch.allclose(ref_dv, tri_dv, atol=1e-2, rtol=rtol)
    assert torch.allclose(ref_dk, tri_dk, atol=1e-2, rtol=rtol)
    assert torch.allclose(ref_dq, tri_dq, atol=1e-2, rtol=rtol)


try:
    from flash_attn.flash_attn_interface import \
        flash_attn_qkvpacked_func as flash_attn_func
    HAS_FLASH = True
except BaseException:
    HAS_FLASH = False

TORCH_HAS_FP8 = hasattr(torch, 'float8_e5m2')
BATCH, N_HEADS, N_CTX, D_HEAD = 4, 48, 4096, 64
# vary seq length for fixed head and batch=4
configs = []
for mode in ["fwd"]:
    for causal in [False]:
        for fp8_inputs in [False]:
            configs.append(
                triton.testing.Benchmark(
                    x_names=["N_CTX"],
                    x_vals=[2**i for i in range(10, 15)],
                    line_arg="provider",
                    line_vals=["triton"] + (["flash"] if HAS_FLASH else []),
                    line_names=["Triton"] + (["Flash-2"] if HAS_FLASH else []),
                    styles=[("red", "-"), ("blue", "-")],
                    ylabel="ms",
                    plot_name=
                    f"fused-attention-batch{BATCH}-head{N_HEADS}-d{D_HEAD}-{mode}-causal={causal}-fp8={fp8_inputs}",
                    args={
                        "H": N_HEADS,
                        "BATCH": BATCH,
                        "D_HEAD": D_HEAD,
                        "dtype": torch.float16,
                        "mode": mode,
                        "causal": causal,
                        "fp8_inputs": fp8_inputs,
                    },
                ))


@triton.testing.perf_report(configs)
def bench_flash_attention(BATCH, H, N_CTX, D_HEAD, causal, mode, provider, fp8_inputs, dtype=torch.float16,
                          device="cuda"):
    assert mode in ["fwd", "bwd"]
    warmup = 25
    rep = 100
    if provider == "triton":
        q = torch.randn((BATCH, H, N_CTX, D_HEAD), dtype=dtype, device="cuda", requires_grad=True)
        k = torch.randn((BATCH, H, N_CTX, D_HEAD), dtype=dtype, device="cuda", requires_grad=True)
        v = torch.randn((BATCH, H, N_CTX, D_HEAD), dtype=dtype, device="cuda", requires_grad=True)
        if mode == "fwd" and TORCH_HAS_FP8 and fp8_inputs:
            q = q.to(torch.float8_e5m2)
            k = k.to(torch.float8_e5m2)
            v = v.permute(0, 1, 3, 2)
            v = v.to(torch.float8_e5m2)
        sm_scale = 1.3
        fn = lambda: attention(q, k, v, sm_scale)
        if mode == "bwd":
            o = fn()
            do = torch.randn_like(o)
            fn = lambda: o.backward(do, retain_graph=True)
        ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)
    if provider == "flash":
        qkv = torch.randn((BATCH, N_CTX, 3, H, D_HEAD), dtype=dtype, device=device, requires_grad=True)
        fn = lambda: flash_attn_func(qkv, causal=causal)
        if mode == "bwd":
            o = fn()
            do = torch.randn_like(o)
            fn = lambda: o.backward(do, retain_graph=True)
        ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)
    flops_per_matmul = 2.0 * BATCH * H * N_CTX * N_CTX * D_HEAD
    total_flops = 2 * flops_per_matmul
    if causal:
        total_flops *= 0.5
    if mode == "bwd":
        total_flops *= 2.5  # 2.0(bwd) + 0.5(recompute)
    return total_flops / ms * 1e-9


if __name__ == "__main__":
    # only works on post-Ampere GPUs right now
    bench_flash_attention.run(save_path=".", print_data=True)
