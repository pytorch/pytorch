""" Triton Implementation of the flex_attention Kernel for short query length (FlexDecoding)"""
import torch
from .. import config
from ..lowering import empty_strided, lowerings, register_lowering
from ..select_algorithm import autotune_select_algorithm, TritonTemplate
from torch._prims_common import make_contiguous_strides_for
from ..ir import (
    ComputedBuffer,
    FixedLayout,
    FlexibleLayout,
    InputBuffer,
    StorageBox,
    TensorBox,
)

def sdpa_decoding_grid(batch_size, num_heads, num_queries, d_model, meta):
    """How is this kernel parallelized?
    We create a grid of (batch_size * num_heads, TILE_KV, 1)
    Each block is responsible for iterating over blocks of keys and values calculating
    the local output for their tile of keys and values over all full length of query.
    """
    import triton

    # TODO: implementation split key & value. Use 1 for now.

    return (batch_size * num_heads, 1, 1)


sdpa_decoding_template = TritonTemplate(
    name="sdpa_decoding",
    grid=sdpa_decoding_grid,
    source=r"""
{{def_kernel("Q", "K", "V", "LSE")}}
    # Sub notation for this kernel:
    # Q: Query, K: Key, V: Value
    # M: Number of queries, N: Number of keys/values, D: Model dimension
    # z: Batch size, h: Number of heads, m: Number of queries per head, k: Number of keys per head
    # (Modifiable) Config options:
    # BLOCK_KV
    # BLOCK_N
    # SCORE_MOD_IS_LINEAR: Is the score modifier linear? If so, we can lift the
    # change of base out of the loop
    # ROWS_GUARANTEED_SAFE: Is it guaranteed that at least one value in each row
    # is not masked out? If so, we can skip an extra safety check
    # OUTPUT_LOGSUMEXP: We only need to store the logsumexp if we require grad

    # Define Q Strides
    stride_qz = {{stride("Q", 0)}}
    stride_qh = {{stride("Q", 1)}}
    stride_qm = {{stride("Q", 2)}}
    stride_qk = {{stride("Q", 3)}}
    # Define K Strides
    stride_kz = {{stride("K", 0)}}
    stride_kh = {{stride("K", 1)}}
    stride_kn = {{stride("K", 2)}}
    stride_kk = {{stride("K", 3)}}
    # Define V Strides
    stride_vz = {{stride("V", 0)}}
    stride_vh = {{stride("V", 1)}}
    stride_vk = {{stride("V", 2)}}
    stride_vn = {{stride("V", 3)}}

    Z = {{size("Q", 0)}}
    H = {{size("Q", 1)}}
    Q_CTX = {{size("Q", 2)}}
    N_CTX = {{size("K", 2)}}

    qk_scale = 1.0
    MATMUL_PRECISION = Q.dtype.element_ty if Q.dtype.element_ty == tl.float64 else tl.float32

    off_hz = tl.program_id(0)           
    off_n = tl.program_id(1)          # TODO: calculate start_n from bidx

    q_offset = off_hz * stride_qh
    Q_block_ptr = tl.make_block_ptr(
        base=Q + q_offset,
        shape=(Q_CTX, BLOCK_DMODEL),            # (Q, d) = (2, 64)
        strides=(stride_qm, stride_qk),
        offsets=(0, 0),                     # No offset: one CTA per query
        block_shape=(BLOCK_MMODEL, BLOCK_DMODEL),
        order=(1, 0)
    )

    kv_offset = off_hz * stride_kh
    K_block_ptr = tl.make_block_ptr(
        base=K + kv_offset,
        shape=(BLOCK_DMODEL, N_CTX),       # (d, N) = (64, 2048)
        strides=(stride_kk, stride_kn),
        offsets=(0, 0),                    # TODO: Add offset here for spliting K among CTAs
        block_shape=(BLOCK_DMODEL, BLOCK_N),
        order=(0, 1)
    )
    V_block_ptr = tl.make_block_ptr(
        base=V + kv_offset,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(stride_vk, stride_vn),
        offsets=(0, 0),
        block_shape=(BLOCK_N, BLOCK_DMODEL),
        order=(1, 0)
    )

    # initialize offsets
    offs_m = tl.arange(0, BLOCK_MMODEL)      
    offs_n = tl.arange(0, BLOCK_N)           
    offs_d = tl.arange(0, BLOCK_DMODEL)

    # initialize pointer to m and l
    m_i = tl.zeros([BLOCK_MMODEL], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_MMODEL], dtype=tl.float32)
    acc = tl.zeros([BLOCK_MMODEL, BLOCK_DMODEL], dtype=tl.float32)

    q = tl.load(Q_block_ptr)
    if SCORE_MOD_IS_LINEAR:
        qk_scale *= 1.44269504
    q = (q * qk_scale).to(MATMUL_PRECISION)
    # loop over k, v and update accumulator
    lo = off_n
    hi = N_CTX
    for start_n in range(lo, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        # -- load k, v --
        k = tl.load(K_block_ptr)
        v = tl.load(V_block_ptr)
        # -- compute qk ---
        qk = tl.zeros([BLOCK_MMODEL, BLOCK_N], dtype=tl.float32)
        qk = tl.sum(q[:, :, None]*k.to(MATMUL_PRECISION)[None, :, :], axis=-2)
        # ~~~~~~~~~~~~~~~~~~~ Apply score modification  ~~~~~~~~~~~~~~~~~~~
        m = offs_m[:, None]
        n = start_n + offs_n[None, :]
        {{ modification(
            score="qk",
            b="off_hz // H",
            h="off_hz % H",
            m="m",
            n="n",
            out="qk"
        ) | indent_except_first(2) }}
        # TODO: In the case that score_mod is linear, this can be LICMed
        if not SCORE_MOD_IS_LINEAR:
            qk *= 1.44269504
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        # -- compute scaling constant ---
        row_max = tl.max(qk, 1)
        m_i_new = tl.maximum(m_i, row_max)

        alpha = tl.math.exp2(m_i - m_i_new)
        p = tl.math.exp2(qk - m_i_new[:, None])
        if not ROWS_GUARANTEED_SAFE:
            masked_out_rows = (m_i_new == float("-inf"))
            alpha = tl.where(masked_out_rows, 0, alpha)
            p = tl.where(masked_out_rows[:, None], 0, p)

        # -- scale and update acc --
        acc *= alpha[:, None]
        p_ = p.to(MATMUL_PRECISION)[:, :, None] # dependent on this triton fix: https://github.com/htyu/triton/commit/c36c24c3cd5e872cb113f1cc56a46fb962ac4e27
        delta_acc = tl.sum(p_ * v.to(MATMUL_PRECISION), axis=-2)
        acc += delta_acc 

        # -- update m_i and l_i --
        l_i = l_i * alpha + tl.sum(p, 1)
        m_i = m_i_new
        # update pointers
        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))

    # Store output and logsumexp
    acc = acc / l_i[:, None]
    idx_z = off_hz // H
    idx_h = off_hz % H
    idx_m = offs_m[:, None]
    idx_d = tl.arange(0, BLOCK_DMODEL)[None, :]


    # TODO generalize and add proper mask support
    mask = (idx_m != -1) & (idx_d != -1)
    {{store_output(("idx_z", "idx_h", "idx_m", "idx_d"), "acc")}}
    

    # TODO dont want to write this if we dont require grad
    if OUTPUT_LOGSUMEXP:
        l_ptrs = LSE + off_hz * Q_CTX + offs_m
        lse = m_i + tl.math.log2(l_i)
        tl.store(l_ptrs, lse)
 """,
)

# Config: (BLOCK_N, TILE_KV, num_warp, num_stages)

_h100_default_config = {
    (torch.float32, 64): (128, 4, 4, 3),
    (torch.float32, 128): (32, 4, 4, 3),
    (torch.float32, 256): (32, 4, 4, 3),
    (torch.bfloat16, 64): (128, 4, 4, 3),
    (torch.bfloat16, 128): (64, 4, 4, 3),
    (torch.bfloat16, 256): (64, 4, 4, 3),
}

_a100_default_config = {
    (torch.float32, 64): (128, 4, 4, 3),
    (torch.float32, 128): (128, 4, 4, 3),
    (torch.float32, 256): (64, 4, 4, 3),
    (torch.bfloat16, 64): (128, 4, 4, 3),
    (torch.bfloat16, 128): (128, 4, 4, 3),
    (torch.bfloat16, 256): (32, 4, 4, 3),
}


def _get_decoding_default_config(query):
    dtype = query.get_dtype()
    head_dim = query.get_size()[-1]
    query_len = query.get_size()[-2]
    default_config = None

    if head_dim <= 256 and torch.cuda.get_device_capability() >= (9, 0):  # H100
        if dtype == torch.float32:
            default_config = (64, 4, 4, 3)
        else:
            default_config = (128, 4, 4, 3)
        default_config = _h100_default_config.get((dtype, head_dim), default_config)
    elif head_dim <= 256 and torch.cuda.get_device_capability() >= (8, 0):  # A100
        if dtype == torch.float32:
            default_config = (64, 4, 4, 3)
        else:
            default_config = (128, 4, 4, 3)
        default_config = _a100_default_config.get((dtype, head_dim), default_config)
    else:  # modest hardware or extremely large head_dim
        if dtype == torch.float32:
            default_config = (32, 4, 4, 3)
        else:
            default_config = (64, 4, 4, 3)

    return default_config


def create_flex_decoding_kernel(query, key, value, other_buffers, node, env):
    output_buffer = env[node.args[0]]
    assert isinstance(output_buffer.data, StorageBox), (
        "The output node for the flex attention subgraph must be a StorageBox, but got: ",
        type(output_buffer),
    )
    # Create the ComputedBuffer directly that will be inlined into the modification block
    subgraph_buffer = ComputedBuffer(
        name=None,
        layout=FlexibleLayout(
            device=output_buffer.data.get_device(),
            dtype=output_buffer.data.get_dtype(),
            size=output_buffer.data.get_size(),
        ),
        data=output_buffer.data.data,  # type: ignore[arg-type]
    )

    layout = FixedLayout(
        output_buffer.get_device(),
        query.get_dtype(),
        query.get_size(),
        make_contiguous_strides_for(query.get_size()),
    )
    # see NOTE:[TritonTemplates with multiple outputs]
    logsumexp_shape = query.get_size()[:-1]  # [B, H, M]
    logsumexp = empty_strided(
        logsumexp_shape,
        None,
        dtype=torch.float32,  # The logsumexp is always stored in fp32 regardless of the input dtype
        device=output_buffer.get_device(),
    )
    choices: List[Any] = []
    configs: List[Any] = []
    configs.append(_get_decoding_default_config(query))
    if config.max_autotune:
        configs += [
            (128, 4, 4, 3),
            (128, 2, 4, 3),
            (128, 4, 8, 2),
            (64, 4, 4, 3),
            (64, 2, 4, 3),
            (32, 4, 4, 3),
            (32, 2, 4, 3),
            (16, 4, 4, 3),
            (16, 2, 4, 3),
        ]

    # Note, we don't need to pass in the captured buffers explicitly
    # because they're implicitly added by the score_mod function
    # We do need to explicitly pass it in for autotuning though.
    for BLOCK_N, TILE_KV, num_warps, num_stages in configs:
        sdpa_decoding_template.maybe_append_choice(
            choices=choices,
            input_nodes=[query, key, value, logsumexp],
            layout=layout,
            subgraphs=subgraph_buffer,
            mutated_inputs=[
                logsumexp,
            ],
            num_stages=num_stages,
            num_warps=num_warps,
            BLOCK_N=BLOCK_N,
            TILE_KV=TILE_KV,
            BLOCK_MMODEL=query.get_size()[-2],
            BLOCK_DMODEL=query.get_size()[-1],
            # For now, we always assume the "sound" option
            SCORE_MOD_IS_LINEAR=False,
            ROWS_GUARANTEED_SAFE=False,
            OUTPUT_LOGSUMEXP=True,
        )
    inputs_for_autotuning = [query, key, value, logsumexp] + list(other_buffers)
    return (
        autotune_select_algorithm(
            "sdpa_decoding", choices, inputs_for_autotuning, layout
        ),
        logsumexp,
    )
