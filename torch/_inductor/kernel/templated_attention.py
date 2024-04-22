""" Triton Implementation of the Templated SDPA Kernel"""
import logging
from typing import Any, List

import torch
from .. import config
from ..lowering import empty_strided, lowerings, register_lowering
from ..select_algorithm import autotune_select_algorithm, TritonTemplate

log = logging.getLogger(__name__)
aten = torch.ops.aten


def sdpa_grid(batch_size, num_heads, num_queries, d_model, meta):
    """How is this kernel parallelized?
    We create a grid of (batch_size * num_heads, ceil_div(n_queries, query_block_size), 1)
    Each block is responsible for iterating over blocks of keys and values calculating
    the final attention output.
    """
    import triton

    return (triton.cdiv(num_queries, meta["BLOCK_M"]), batch_size * num_heads, 1)


sdpa_template = TritonTemplate(
    name="sdpa",
    grid=sdpa_grid,
    source=r"""
{{def_kernel("Q", "K", "V", "LSE")}}
    # Sub notation for this kernel:
    # Q: Query, K: Key, V: Value
    # M: Number of queries, N: Number of keys/values, D: Model dimension
    # z: Batch size, h: Number of heads, m: Number of queries per head, k: Number of keys per head
    # (Modifiable) Config options:
    # BLOCK_M
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
    N_CTX = {{size("Q", 2)}}

    qk_scale = 1.0
    MATMUL_PRECISION = Q.dtype.element_ty

    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)

    qkv_offset = off_hz * stride_qh
    Q_block_ptr = tl.make_block_ptr(
        base=Q + qkv_offset,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(stride_qm, stride_qk),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0)
    )
    K_block_ptr = tl.make_block_ptr(
        base=K + qkv_offset,
        shape=(BLOCK_DMODEL, N_CTX),
        strides=(stride_kk, stride_kn),
        offsets=(0, 0),
        block_shape=(BLOCK_DMODEL, BLOCK_N),
        order=(0, 1)
    )
    V_block_ptr = tl.make_block_ptr(
        base=V + qkv_offset,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(stride_vk, stride_vn),
        offsets=(0, 0),
        block_shape=(BLOCK_N, BLOCK_DMODEL),
        order=(1, 0)
    )
    # initialize offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    # initialize pointer to m and l
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)

    q = tl.load(Q_block_ptr)
    if SCORE_MOD_IS_LINEAR:
        qk_scale *= 1.44269504
    q = (q * qk_scale).to(MATMUL_PRECISION)
    # loop over k, v and update accumulator
    lo = 0
    hi = N_CTX
    for start_n in range(lo, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        # -- load k, v --
        k = tl.load(K_block_ptr)
        v = tl.load(V_block_ptr)
        # -- compute qk ---
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk = tl.dot(q, k.to(MATMUL_PRECISION), acc=qk)
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
        acc_scale = l_i * 0 + alpha  # workaround some compiler bug
        acc *= acc_scale[:, None]
        acc = tl.dot(p.to(MATMUL_PRECISION), v.to(MATMUL_PRECISION), acc)

        # -- update m_i and l_i --
        l_i = l_i * alpha + tl.sum(p, 1)
        m_i = m_i_new
        # update pointers
        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))

    # Store output and logsumexp
    acc = acc / l_i[:, None]
    idx_z = tl.program_id(1) // H
    idx_h = tl.program_id(1) % H
    idx_m = offs_m[:, None]
    idx_d = tl.arange(0, BLOCK_DMODEL)[None, :]

    # TODO generalize and add proper mask support
    mask = (idx_m != -1) & (idx_d != -1)
    {{store_output(("idx_z", "idx_h", "idx_m", "idx_d"), "acc")}}

    # TODO dont want to write this if we dont require grad
    if OUTPUT_LOGSUMEXP:
        l_ptrs = LSE + off_hz * N_CTX + offs_m
        lse = m_i + tl.math.log2(l_i)
        tl.store(l_ptrs, lse)
 """,
)


# TODO: We probably also need a layout constraint?
@register_lowering(torch.ops.higher_order.templated_attention, type_promotion_kind=None)
def templated_attention(*args, **kwargs):
    from torch._prims_common import make_contiguous_strides_for
    from ..ir import (
        ComputedBuffer,
        FixedLayout,
        FlexibleLayout,
        InputBuffer,
        StorageBox,
        TensorBox,
    )

    query, key, value, subgraph, *other_buffers = args

    def create_placeholder(name: str, dtype: torch.dtype) -> InputBuffer:
        return TensorBox.create(
            InputBuffer(
                name,
                FixedLayout(
                    query.get_device(),
                    dtype,
                    [
                        1,
                    ],
                    [
                        1,
                    ],
                ),
            )
        )

    scalar_inps = ["score", "b", "h", "m", "n"]
    env = {}
    cnt = 0
    placeholder_inps = [
        create_placeholder(name, dtype)
        for name, dtype in [
            ("score", query.get_dtype()),
            ("b", torch.int32),
            ("h", torch.int32),
            ("m", torch.int32),
            ("n", torch.int32),
        ]
    ]
    for node in subgraph.graph_module.graph.nodes:
        # There are two classes of placeholder inpts that we need
        # to handle differently. For the first n_scalar_inps inputs
        # we expect that these placeholders were generated by the make_fx call
        # in the templated Attention HOP. So we need to create a new placeholder
        # TensorBox for each of these inputs. For the rest of the inputs we
        # expect that these are lifted inputs that fill up the '*other_buffers'
        # tuple and already have corresponding TensorBoxes passed in as args.
        if node.op == "placeholder":
            is_lifted_input = cnt >= len(scalar_inps)
            env[node] = args[cnt - 1] if is_lifted_input else placeholder_inps[cnt]
            cnt += 1
        elif node.op == "call_function":
            # For call_function we use the defulat lowerings and pass in the
            # already created TensorBoxes as args
            from torch.utils._pytree import tree_map

            env[node] = lowerings[node.target](
                *tree_map(lambda x: env[x] if x in env else x, node.args)
            )
        elif node.op == "output":
            # For the output node we need to create a ComputedBuffer
            # which represents the actual score modification

            output_buffer = env[node.args[0]]
            assert isinstance(output_buffer.data, StorageBox), (
                "The output node for the templated attention subgraph must be a StorageBox, but got: ",
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
            if query.get_dtype() == torch.float32:
                configs.append((64, 64, 4, 3))
            else:
                configs.append((128, 64, 4, 3))
            if config.max_autotune:
                configs += [
                    (128, 64, 4, 3),
                    (128, 128, 4, 3),
                    (128, 128, 8, 2),
                    (64, 128, 4, 3),
                ]
            # Note, we don't need to pass in the captured buffers explicitly
            # because they're implicitly added by the score_mod function
            # We do need to explicitly pass it in for autotuning though.
            for BLOCK_M, BLOCK_N, num_warps, num_stages in configs:
                sdpa_template.maybe_append_choice(
                    choices=choices,
                    input_nodes=[query, key, value, logsumexp],
                    layout=layout,
                    subgraphs=subgraph_buffer,
                    mutated_inputs=[
                        logsumexp,
                    ],
                    num_stages=num_stages,
                    num_warps=num_warps,
                    BLOCK_M=BLOCK_M,
                    BLOCK_N=BLOCK_N,
                    BLOCK_DMODEL=query.get_size()[-1],
                    # For now, we always assume the "sound" option
                    SCORE_MOD_IS_LINEAR=False,
                    ROWS_GUARANTEED_SAFE=False,
                    OUTPUT_LOGSUMEXP=True,
                )
            inputs_for_autotuning = [query, key, value, logsumexp] + list(other_buffers)
            return (
                autotune_select_algorithm(
                    "sdpa", choices, inputs_for_autotuning, layout
                ),
                logsumexp,
            )
    raise ValueError("TemplatedAttention was passed a subgraph with no output node!")
