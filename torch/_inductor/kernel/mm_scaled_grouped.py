import logging
from typing import Any, Optional

import torch
from torch._dynamo.utils import counters
from torch._inductor.virtualized import V
from torch.utils._triton import has_triton_tma_device

from ..ir import ChoiceCaller, get_device_type, Layout, TensorBox
from ..lowering import register_lowering
from ..runtime.runtime_utils import next_power_of_2
from ..select_algorithm import (
    autotune_select_algorithm,
    ExternKernelChoice,
    realize_inputs,
    TritonTemplate,
)
from ..utils import get_num_sms, get_tma_workspace_arg, use_aten_gemm_kernels
from .mm_common import _is_static_problem, check_supported_striding, persistent_mm_grid


log = logging.getLogger(__name__)
aten = torch.ops.aten


# Copied from fbgemm grouped_gemm.py
triton_scaled_grouped_mm_source = r"""
{{def_kernel("a_ptr", "b_ptr", "a_scale_ptr", "b_scale_ptr", "m_sizes")}}
    tidx = tl.program_id(0)

    dtype = tl.float8e4nv
    TMA_SIZE: tl.constexpr = tl.constexpr(128)
    if USE_TMA_STORE:
        workspace_base = ws_ptr + tidx * 3 * TMA_SIZE
        c_desc_ptr = worspace_base + 2 * TMA_SIZE
    else:
        workspace_base = ws_ptr + tidx * 2 * TMA_SIZE
        c_desc_ptr = None

    a_desc_ptr = workspace_base
    b_desc_ptr = workspace_base + TMA_SIZE

    triton.language.extra.cuda.experimental_device_tensormap_create2d(
        desc_ptr=a_desc_ptr,
        global_address=a_ptr,
        load_size=[BLOCK_M, BLOCK_K],
        global_size=[M, K],
        element_ty=a_ptr.dtype.element_ty,
    )
    triton.language.extra.cuda.experimental_device_tensormap_create2d(
        desc_ptr=b_desc_ptr,
        global_address=b_ptr,
        load_size=[BLOCK_N, BLOCK_K],
        global_size=[N * G, K],
        element_ty=b_ptr.dtype.element_ty,
    )
    tl.extra.cuda.experimental_tensormap_fenceproxy_acquire(a_desc_ptr)
    tl.extra.cuda.experimental_tensormap_fenceproxy_acquire(b_desc_ptr)

    M_end_offset = 0
    iterated_tiles = 0
    for g in tl.range(G):
        # Move across groups
        M_start_offset = M_end_offset
        M_end_offset = tl.load(m_sizes + g)
        m_size = M_end_offset - M_start_offset

        if m_size > 0:
            N_start_offset = g.to(tl.int64) * N
            n_size = N
            num_m_tiles = tl.cdiv(m_size, BLOCK_M)
            num_n_tiles = tl.cdiv(n_size, BLOCK_N)
            num_tiles = num_m_tiles * num_n_tiles

            if USE_TMA_STORE:
                # pyre-ignore
                tl.extra.cuda.experimental_device_tensormap_create2d(
                    desc_ptr=c_desc_ptr,
                    global_address=c_ptr + M_start_offset * N,
                    load_size=[BLOCK_M, BLOCK_N],
                    global_size=[m_size, n_size],
                    element_ty=c_ptr.dtype.element_ty,
                )
                # pyre-ignore
                tl.extra.cuda.experimental_tensormap_fenceproxy_acquire(c_desc_ptr)

            # Move across tiles
            while tidx >= iterated_tiles and tidx < iterated_tiles + num_tiles:
                gidx = tidx - iterated_tiles
                # Split M first and N second.
                tile_m_idx = gidx % num_m_tiles
                tile_n_idx = gidx // num_m_tiles

                accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
                tl.static_assert(K % BLOCK_K == 0)
                if USE_TMA_LOAD:
                    m_offset = (M_start_offset + tile_m_idx * BLOCK_M).to(tl.int32)
                    n_offset = (N_start_offset + tile_n_idx * BLOCK_N).to(tl.int32)
                    for k_offset in range(0, K, BLOCK_K):
                        a = tl._experimental_descriptor_load(
                            a_desc_ptr,
                            [m_offset, k_offset],
                            [BLOCK_M, BLOCK_K],
                            dtype,
                        )
                        b = tl._experimental_descriptor_load(
                            b_desc_ptr,
                            [n_offset, k_offset],
                            [BLOCK_N, BLOCK_K],
                            dtype,
                        )
                        if USE_FAST_ACCUM:
                            accumulator = tl.dot(a, b.T, accumulator)
                        else:
                            accumulator += tl.dot(a, b.T)
                else:
                    offs_am = tile_m_idx * BLOCK_M + tl.arange(0, BLOCK_M)
                    offs_bn = tile_n_idx * BLOCK_N + tl.arange(0, BLOCK_N)
                    offs_k = tl.arange(0, BLOCK_K)
                    a_ptrs = (
                        a_desc_ptr
                        + (M_start_offset + offs_am[:, None]) * K
                        + offs_k[None, :]
                    )
                    b_ptrs = (
                        b_desc_ptr
                        + (N_start_offset + offs_bn[:, None]) * K
                        + offs_k[None, :]
                    )
                    for k_offset in range(0, K, BLOCK_K):
                        a = tl.load(a_ptrs, mask=offs_am[:, None] < m_size)
                        b = tl.load(b_ptrs, mask=offs_bn[:, None] < n_size)
                        accumulator += tl.dot(a, b.T)
                        a_ptrs += BLOCK_K
                        b_ptrs += BLOCK_K

                offs_am = tile_m_idx * BLOCK_M + tl.arange(0, BLOCK_M)
                offs_bn = tile_n_idx * BLOCK_N + tl.arange(0, BLOCK_N)
                a_scale = tl.load(
                    a_scale_ptr + M_start_offset + offs_am[:, None],
                    mask=offs_am[:, None] < m_size,
                )
                b_scale = tl.load(
                    b_scale_ptr + N_start_offset + offs_bn[None, :],
                    mask=offs_bn[None, :] < n_size,
                )
                c = accumulator.to(tl.float32) * a_scale * b_scale

                if USE_TMA_STORE:
                    m_offset = (tile_m_idx * BLOCK_M).to(tl.int32)
                    n_offset = (tile_n_idx * BLOCK_N).to(tl.int32)
                    tl._experimental_descriptor_store(
                        c_desc_ptr,
                        c.to(c_ptr.dtype.element_ty),
                        [m_offset, n_offset],
                    )
                else:
                    idx_m = (M_start_offset + offs_am[:, None])
                    idx_n = offs_bn[None, :]
                    mask = offs_am[:, None] < m_size and offs_bn[None, :] < n_size
                    {{store_output(("idx_m", "idx_n"), "c", "mask", indent_width=20)}}
                tidx += NUM_SMS

            iterated_tiles += num_tiles
"""

triton_scaled_grouped_mm_template = TritonTemplate(
    name="scaled_grouped_mm",
    grid=persistent_mm_grid,
    source=triton_scaled_grouped_mm_source,
)


def grouped_mm_args(
    mat1,
    mat2,
    layout=None,
    out_dtype=None,
):
    mat1, mat2 = realize_inputs(mat1, mat2)
    m, k1 = mat1.get_size()
    g, k2, n = mat2.get_size()
    k = V.graph.sizevars.guard_equals(k1, k2)
    if layout is None:
        from torch._inductor.ir import FixedLayout

        if out_dtype is None:
            out_dtype = mat1.get_dtype()

        layout = FixedLayout(
            mat1.get_device(),
            out_dtype,
            [m, n],
        )
    else:
        assert out_dtype is None, "out_dtype is ignored if layout is specified."

    return (g, m, n, k, layout, mat1, mat2)


aten__scaled_grouped_mm = ExternKernelChoice(
    torch._scaled_grouped_mm,
    "at::_scaled_grouped_mm",
    op_overload=aten._scaled_grouped_mm,
    has_out_variant=False,
)


@register_lowering(aten._scaled_grouped_mm.default, type_promotion_kind=None)
def tuned_scaled_grouped_mm(
    mat_a: TensorBox,
    mat_b: TensorBox,
    scale_a: TensorBox,
    scale_b: TensorBox,
    offs: Optional[TensorBox] = None,
    bias: Optional[TensorBox] = None,
    scale_result: Optional[TensorBox] = None,
    out_dtype: Optional[torch.dtype] = None,
    use_fast_accum: bool = False,
    layout: Optional[Layout] = None,
) -> TensorBox:
    g, m, n, k, layout, mat_a, mat_b = grouped_mm_args(
        mat_a, mat_b, layout=layout, out_dtype=out_dtype
    )

    counters["aten_mm_info"][f"aten._scaled_grouped_mm.default_{g}_{m}_{n}_{k}"] += 1
    log.info(
        "Tuned aten._scaled_grouped_mm.default: g=%s m=%s, n=%s, k=%s, mat1_dtype=%s, mat2_dtype=%s, output_layout=%s",
        g,
        m,
        n,
        k,
        mat_a.get_dtype(),
        mat_b.get_dtype(),
        layout,
    )

    device_type = get_device_type(mat_a)
    check_supported_striding(mat_a, mat_b)

    scale_a, scale_b = realize_inputs(scale_a, scale_b)

    # workaround for Inductor not supporting optional tensor input arguments
    input_nodes: list[Any, ...] = [mat_a, mat_b, scale_a, scale_b]
    if offs is not None:
        input_nodes.append(realize_inputs(offs))
    if bias is not None:
        input_nodes.append(realize_inputs(bias))

    aten_choice = aten__scaled_grouped_mm.bind(
        input_nodes,
        layout,
        out_dtype=out_dtype,
        use_fast_accum=use_fast_accum,
    )

    choices: list[ChoiceCaller] = []
    if use_aten_gemm_kernels():
        choices.append(aten_choice)

    _, is_nonzero = _is_static_problem(layout)

    scaled_grouped_mm_configs = V.choices.get_scaled_grouped_mm_configs(device_type)

    if is_nonzero and offs is not None and bias is None and has_triton_tma_device():
        for config in scaled_grouped_mm_configs(m, n, k):
            kwargs = {
                "G": g,
                "M": m,
                "M_BUCKET": next_power_of_2(m),
                "N": n,
                "K": k,
                "NUM_SMS": get_num_sms(),
                "USE_TMA_LOAD": True,
                "USE_TMA_STORE": False,
                "USE_FAST_ACCUM": use_fast_accum,
                "num_stages": config.num_stages,
                "num_warps": config.num_warps,
                **config.kwargs,
            }
            triton_scaled_grouped_mm_template.maybe_append_choice(
                choices,
                input_nodes=input_nodes,
                layout=layout,
                workspace_arg=get_tma_workspace_arg(
                    num_tma_descriptors=2,
                    device=mat_a.get_device(),
                ),
                **kwargs,
            )

    return autotune_select_algorithm("scaled_grouped_mm", choices, input_nodes, layout)
