# mypy: allow-untyped-defs
import logging
from dataclasses import dataclass
from typing import Any, Optional

import torch
from torch._dynamo.utils import counters
from torch._inductor.runtime.triton_compat import tl
from torch._inductor.virtualized import V
from torch.utils._triton import has_triton

from ..ir import ChoiceCaller, Layout, TensorBox
from ..lowering import register_lowering
from ..select_algorithm import (
    autotune_select_algorithm,
    ExternKernelChoice,
    realize_inputs,
    TritonTemplate,
)
from ..utils import (
    get_gpu_shared_memory,
    get_num_sms,
    has_free_symbols,
    use_aten_gemm_kernels,
)
from .mm_common import (
    _is_static_problem,
    check_supported_striding,
    persistent_grouped_mm_grid,
)


log = logging.getLogger(__name__)
aten = torch.ops.aten


@dataclass
class Config:
    kwargs: dict[str, int]
    num_stages: int
    num_warps: int


_NV_CONFIGS = [
    Config(
        {
            "BLOCK_M": block_size_m,
            "BLOCK_N": block_size_n,
            "BLOCK_K": block_size_k,
            "NUM_CONSUMER_GROUPS": 1,
        },
        num_stages=num_stages,
        num_warps=num_warps,
    )
    for block_size_m in [16, 32, 64, 128]
    for block_size_n in [64, 128, 256]
    for block_size_k in [64, 128, 256]
    for num_stages in [3, 4]
    for num_warps in [4, 8]
]


def grouped_mm_configs():
    return _NV_CONFIGS


def early_config_prune(g, m, configs, named_args):
    dtsize = 1
    pruned_configs = []
    for config in configs:
        kw = config.kwargs
        BLOCK_M, BLOCK_N, BLOCK_K, num_stages, num_warps, num_consumer_groups = (
            kw["BLOCK_M"],
            kw["BLOCK_N"],
            kw["BLOCK_K"],
            config.num_stages,
            config.num_warps,
            getattr(config, "num_consumer_groups", 0),
        )

        # 1. Prune NV configs depending on g and m.
        if not has_free_symbols((g, m)):
            a_is_2d, b_is_2d = named_args["A_IS_2D"], named_args["B_IS_2D"]
            m_avg = m // g if a_is_2d and not b_is_2d else m
            if m_avg <= 16:
                if BLOCK_M > 32:
                    continue
            elif m_avg <= 32:
                if BLOCK_M > 64:
                    continue
            elif m_avg <= 64:
                if BLOCK_M <= 16:
                    continue
            else:
                if BLOCK_M <= 32:
                    continue

        # 2. make sure we have enough smem
        max_shared_memory = get_gpu_shared_memory()

        required_shared_memory = (BLOCK_M + BLOCK_N) * BLOCK_K * num_stages * dtsize
        if required_shared_memory > max_shared_memory:
            continue

        use_warp_specialization = num_consumer_groups >= 1

        # 3. make sure we can partition for ws
        if use_warp_specialization:
            if num_warps != 4:
                continue

            # "tritongpu-warp-spec-data-partition"
            m_slice = BLOCK_M // num_consumer_groups
            n_slice = BLOCK_N // num_consumer_groups
            if m_slice < 64 and n_slice < 256:
                continue

        pruned_configs.append(config)

    return pruned_configs


triton_grouped_mm_source = r"""
{%- if SCALED %}
{%- if A_IS_2D or B_IS_2D %}
{{def_kernel("a_ptr", "b_ptr", "scale_a_ptr", "scale_b_ptr", "offsets_ptr")}}
{%- else %}
{{def_kernel("a_ptr", "b_ptr", "scale_a_ptr", "scale_b_ptr")}}
{%- endif %}
{%- else %}
{%- if A_IS_2D or B_IS_2D %}
{{def_kernel("a_ptr", "b_ptr", "offsets_ptr")}}
{%- else %}
{{def_kernel("a_ptr", "b_ptr")}}
{%- endif %}
{%- endif %}
    tidx = tl.program_id(0)

{%- set M_IS_VARYING = A_IS_2D and not B_IS_2D %}
{%- set N_IS_VARYING = not A_IS_2D and B_IS_2D %}
{%- set K_IS_VARYING = A_IS_2D and B_IS_2D %}

{%- if A_IS_2D %}
{%- if B_IS_2D %}
    G = {{size("offsets_ptr", 0)}}
{%- else %}
    G = {{size("b_ptr", 0)}}
{%- endif %}
{%- else %}
{%- if B_IS_2D %}
    G = {{size("a_ptr", 0)}}
{%- else %}
    G = {{size("a_ptr", 0)}}
{%- endif %}
{%- endif %}

    # the b_ptr tensor is given with its last two dims transposed, revert here

    M = {{size("a_ptr", -2)}}
    N = {{size("b_ptr", -1)}}
    K = {{size("a_ptr", -1)}}

    A_STRIDE_M = {{stride("a_ptr", -2)}}
    A_STRIDE_K = {{stride("a_ptr", -1)}}
{%- if not A_IS_2D %}
    A_STRIDE_G = {{stride("a_ptr", 0)}}
{%- if SCALED %}
    SCALE_A_STRIDE_G = {{stride("scale_a_ptr", 0)}}
{%- endif %}
{%- endif %}
    B_STRIDE_N = {{stride("b_ptr", -1)}}
    B_STRIDE_K = {{stride("b_ptr", -2)}}
{%- if not B_IS_2D %}
    B_STRIDE_G = {{stride("b_ptr", 0)}}
{%- if SCALED %}
    SCALE_B_STRIDE_G = {{stride("scale_b_ptr", 0)}}
{%- endif %}
{%- endif %}

{%- if USE_TMA_LOAD %}
{%- if USE_EXPERIMENTAL_MAKE_TENSOR_DESCRIPTOR %}
    a_desc = tl._experimental_make_tensor_descriptor(
{%- else %}
    a_desc = tl.make_tensor_descriptor(
{%- endif %}
        a_ptr,
{%- if A_IS_2D %}
        shape=[M, K],
        # fixme: strides=[A_STRIDE_M, A_STRIDE_K],
        strides=[{{stride("a_ptr", -2)}}, {{stride("a_ptr", -1)}}],
        block_shape=[BLOCK_M, BLOCK_K],
{%- else %}
        shape=[G, M, K],
        # fixme: strides=[A_STRIDE_G, A_STRIDE_M, A_STRIDE_K],
        strides=[{{stride("a_ptr", 0)}}, {{stride("a_ptr", -2)}}, {{stride("a_ptr", -1)}}],
        block_shape=[1, BLOCK_M, BLOCK_K],
{%- endif %}
    )

{%- if USE_EXPERIMENTAL_MAKE_TENSOR_DESCRIPTOR %}
    b_desc = tl._experimental_make_tensor_descriptor(
{%- else %}
    b_desc = tl.make_tensor_descriptor(
{%- endif %}
        b_ptr,
{%- if B_IS_2D %}
        shape=[N, K],
        # fixme: strides=[B_STRIDE_N, B_STRIDE_K],
        strides=[{{stride("b_ptr", -1)}}, {{stride("b_ptr", -2)}}],
        block_shape=[BLOCK_N, BLOCK_K],
{%- else %}
        shape=[G, N, K],
        # fixme: strides=[B_STRIDE_G, B_STRIDE_N, B_STRIDE_K],
        strides=[{{stride("b_ptr", 0)}}, {{stride("b_ptr", -1)}}, {{stride("b_ptr", -2)}}],
        block_shape=[1, BLOCK_N, BLOCK_K],
{%- endif %}
    )
{%- endif %}

{%- if M_IS_VARYING %}
    m_end_offset = 0
{%- endif %}
{%- if N_IS_VARYING %}
    n_end_offset = 0
{%- endif %}
{%- if K_IS_VARYING %}
    k_end_offset = 0
{%- endif %}
    iterated_tiles = 0
    for g in tl.range(G):
{%- if M_IS_VARYING %}
        # Move across groups
        m_start_offset = m_end_offset
        m_end_offset = tl.load(offsets_ptr + g)
        m_size = m_end_offset - m_start_offset
{%- if SCALED %}
        m_scale_start_offset = m_start_offset
{%- endif %}
{%- else %}
        m_start_offset = 0
        m_size = M
{%- if SCALED %}
        m_scale_start_offset = g * M
{%- endif %}
{%- endif %}

{%- if N_IS_VARYING %}
        # Move across groups
        n_start_offset = n_end_offset
        n_end_offset = tl.load(offsets_ptr + g)
        n_size = n_end_offset - n_start_offset
{%- if SCALED %}
        n_scale_start_offset = n_start_offset
{%- endif %}
{%- else %}
        n_start_offset = 0
        n_size = N
{%- if SCALED %}
        n_scale_start_offset = g * N
{%- endif %}
{%- endif %}

        if m_size > 0 and n_size > 0:
{%- if K_IS_VARYING %}
            # Move across groups
            k_start_offset = k_end_offset
            k_end_offset = tl.load(offsets_ptr + g)
            k_size = k_end_offset - k_start_offset
{%- else %}
            k_start_offset = 0
            k_size = K
{%- endif %}

            num_m_tiles = tl.cdiv(m_size, BLOCK_M)
            num_n_tiles = tl.cdiv(n_size, BLOCK_N)
            num_tiles = num_m_tiles * num_n_tiles

            # Move across tiles
            while tidx >= iterated_tiles and tidx < iterated_tiles + num_tiles:
                gidx = tidx - iterated_tiles
                # Split M first and N second.
                tile_m_idx = gidx % num_m_tiles
                tile_n_idx = gidx // num_m_tiles

                accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

{%- if USE_TMA_LOAD %}
                m_offset = (m_start_offset + tile_m_idx * BLOCK_M).to(tl.int32)
                n_offset = (n_start_offset + tile_n_idx * BLOCK_N).to(tl.int32)

                for k_offset in range(0, k_size, BLOCK_K):
{%- if A_IS_2D %}
                    a = a_desc.load([m_offset, k_start_offset + k_offset])
{%- else %}
                    a = a_desc.load([g, m_offset, k_start_offset + k_offset]).reshape(BLOCK_M, BLOCK_K)
{%- endif %}
{%- if B_IS_2D %}
                    b = b_desc.load([n_offset, k_start_offset + k_offset])
{%- else %}
                    b = b_desc.load([g, n_offset, k_start_offset + k_offset]).reshape(BLOCK_N, BLOCK_K)
{%- endif %}

{%- if K_IS_VARYING %}
                    if k_offset + BLOCK_K > k_size:
                        group_offs_k = k_offset + tl.arange(0, BLOCK_K)
                        a = tl.where(group_offs_k < k_size, a, 0)
                        b = tl.where(group_offs_k < k_size, b, 0)
{%- endif %}

{%- if USE_FAST_ACCUM %}
                    accumulator = tl.dot(a, b.T, accumulator)
{%- else %}
                    accumulator += tl.dot(a, b.T)
{%- endif %}
{%- else %}
                offs_am = tile_m_idx * BLOCK_M + tl.arange(0, BLOCK_M)
                offs_bn = tile_n_idx * BLOCK_N + tl.arange(0, BLOCK_N)
                offs_k = k_start_offset + tl.arange(0, BLOCK_K)
                a_ptrs = (
                    a_ptr
{%- if not A_IS_2D %}
                    + g * A_STRIDE_G
{%- endif %}
                    + (m_start_offset + offs_am[:, None]) * A_STRIDE_M
                    + offs_k[None, :] * A_STRIDE_K
                )
                b_ptrs = (
                    b_ptr
{%- if not B_IS_2D %}
                    + g * B_STRIDE_G
{%- endif %}
                    + (n_start_offset + offs_bn[:, None]) * B_STRIDE_N
                    + offs_k[None, :] * B_STRIDE_K
                )
                for k_offset in range(0, k_size, BLOCK_K):
                    a = tl.load(a_ptrs, mask=offs_am[:, None] < m_size)
                    b = tl.load(b_ptrs, mask=offs_bn[:, None] < n_size)
                    if k_offset + BLOCK_K > k_size:
                        group_offs_k = k_offset + tl.arange(0, BLOCK_K)
                        a = tl.where(group_offs_k < k_size, a, 0)
                        b = tl.where(group_offs_k < k_size, b, 0)
{%- if USE_FAST_ACCUM %}
                    accumulator = tl.dot(a, b.T, accumulator)
{%- else %}
                    accumulator += tl.dot(a, b.T)
{%- endif %}
                    a_ptrs += BLOCK_K
                    b_ptrs += BLOCK_K
{%- endif %}

                offs_am = tile_m_idx * BLOCK_M + tl.arange(0, BLOCK_M)
                offs_bn = tile_n_idx * BLOCK_N + tl.arange(0, BLOCK_N)
{%- if SCALED %}
                scale_a = tl.load(
                    scale_a_ptr
{%- if A_IS_2D %}
                    + m_scale_start_offset
{%- else %}
                    + g * SCALE_A_STRIDE_G
{%- endif %}
                    + offs_am[:, None],
                    mask=offs_am[:, None] < m_size,
                )
                scale_b = tl.load(
                    scale_b_ptr
{%- if B_IS_2D %}
                    + n_scale_start_offset
{%- else %}
                    + g * SCALE_B_STRIDE_G
{%- endif %}
                    + offs_bn[None, :],
                    mask=offs_bn[None, :] < n_size,
                )
                c = accumulator.to(tl.float32) * scale_a * scale_b
{%- else %}
                c = accumulator.to(tl.float32)
{%- endif %}

{%- if M_IS_VARYING %}
                idx_m = (m_start_offset + offs_am[:, None])
{%- else %}
                idx_m = offs_am[:, None]
{%- endif %}
{%- if N_IS_VARYING %}
                idx_n = (n_start_offset + offs_bn[None, :])
{%- else %}
                idx_n = offs_bn[None, :]
{%- endif %}
                mask = offs_am[:, None] < m_size and offs_bn[None, :] < n_size
{%- if M_IS_VARYING or N_IS_VARYING %}
                {{store_output(("idx_m", "idx_n"), "c", "mask", indent_width=16)}}
{%- else %}
                {{store_output(("g", "idx_m", "idx_n"), "c", "mask", indent_width=16)}}
{%- endif %}
                tidx += NUM_SMS

            iterated_tiles += num_tiles
"""


triton_grouped_mm_template = TritonTemplate(
    name="grouped_mm",
    grid=persistent_grouped_mm_grid,
    source=triton_grouped_mm_source,
)

triton_scaled_grouped_mm_template = TritonTemplate(
    name="scaled_grouped_mm",
    grid=persistent_grouped_mm_grid,
    source=triton_grouped_mm_source,
)


def grouped_mm_args(
    mat1: TensorBox,
    mat2: TensorBox,
    offs: Optional[TensorBox],
    layout=None,
    out_dtype=None,
):
    mat1, mat2 = realize_inputs(mat1, mat2)
    if offs is not None:
        realize_inputs(offs)
    mat1_size = mat1.get_size()
    mat2_size = mat2.get_size()

    m1dim, m2dim = len(mat1_size), len(mat2_size)

    assert m1dim == 2 or m1dim == 3
    assert m2dim == 2 or m2dim == 3

    if layout is None:
        from torch._inductor.ir import FixedLayout

        if out_dtype is None:
            out_dtype = mat1.get_dtype()

        dims = []
        if m1dim == 2:
            if m2dim == 2:
                assert offs is not None
                dims = [offs.get_size()[0], mat1_size[0], mat2_size[1]]
            else:
                dims = [mat1_size[0], mat2_size[-1]]
        else:
            if m2dim == 2:
                dims = [mat1_size[1], mat2_size[1]]
            else:
                dims = [mat1_size[0], mat1_size[1], mat2_size[-1]]
        layout = FixedLayout(
            mat1.get_device(),
            out_dtype,
            dims,
        )
    else:
        assert out_dtype is None, "out_dtype is ignored if layout is specified."

    return (mat1_size, mat2_size, layout, mat1, mat2, offs)


aten__grouped_mm = ExternKernelChoice(
    torch._grouped_mm,
    "at::_grouped_mm",
    op_overload=aten._grouped_mm,
    has_out_variant=False,
)


aten__scaled_grouped_mm = ExternKernelChoice(
    torch._scaled_grouped_mm,
    "at::_scaled_grouped_mm",
    op_overload=aten._scaled_grouped_mm,
    has_out_variant=False,
)


def can_use_triton_kernel(
    mat_a: TensorBox,
    mat_b: TensorBox,
    offs: Optional[TensorBox],
    bias: Optional[TensorBox],
    scale_result: Optional[TensorBox],
) -> bool:
    if not (
        torch.cuda.is_available()
        and torch.cuda.get_device_capability() >= (9, 0)
        and not torch.version.hip
    ):
        return False
    if not has_triton():
        return False

    # The _grouped_mm()/_scaled_grouped_mm() operator do not support
    # bias nor scale_result yet.
    if bias is not None:
        return False
    if scale_result is not None:
        return False

    if len(mat_a.get_size()) == 2 or len(mat_b.get_size()) == 2:
        return offs is not None
    else:
        return offs is None


def create_offsets(x, m1_size, m2_size, offs_size):
    m1_is_2d = len(m1_size) == 2
    m2_is_2d = len(m2_size) == 2
    if m1_is_2d:
        if m2_is_2d:
            k = V.graph.sizevars.size_hint(m1_size[1])
            noffs = V.graph.sizevars.size_hint(offs_size[0])
            step = k / noffs
            return torch.linspace(
                step, k, noffs, dtype=x.get_dtype(), device=x.get_device()
            )

        else:
            m = V.graph.sizevars.size_hint(m1_size[0])
            noffs = V.graph.sizevars.size_hint(offs_size[0])
            step = m / noffs
            return torch.linspace(
                step, m, noffs, dtype=x.get_dtype(), device=x.get_device()
            )
    else:
        if m2_is_2d:
            n = V.graph.sizevars.size_hint(m2_size[0])
            noffs = V.graph.sizevars.size_hint(offs_size[0])
            step = n / noffs
            return torch.linspace(
                step, n, noffs, dtype=x.get_dtype(), device=x.get_device()
            )
        else:
            return None


def _tuned_grouped_mm_common(
    operator_name: str,
    algorithm_name: str,
    extern_kernel_choice: ExternKernelChoice,
    kernel_template: TritonTemplate,
    mat_a: TensorBox,
    mat_b: TensorBox,
    scale_a: Optional[TensorBox] = None,
    scale_b: Optional[TensorBox] = None,
    offs: Optional[TensorBox] = None,
    bias: Optional[TensorBox] = None,
    scale_result: Optional[TensorBox] = None,
    out_dtype: Optional[torch.dtype] = None,
    use_fast_accum: Optional[bool] = None,
    layout: Optional[Layout] = None,
) -> TensorBox:
    assert (scale_a is None) == (scale_b is None)
    assert scale_result is None or scale_a is not None

    m1_size, m2_size, layout, mat_a, mat_b, offs = grouped_mm_args(
        mat_a, mat_b, offs, layout=layout, out_dtype=out_dtype
    )
    counters["aten_mm_info"][operator_name] += 1
    log_message = f"Tuned {operator_name}: mat1_shape=%s, mat2_shape=%s, mat1_dtype=%s, mat2_dtype=%s, output_layout=%s"
    log.info(
        log_message,
        m1_size,
        m2_size,
        mat_a.get_dtype(),
        mat_b.get_dtype(),
        layout,
    )

    if scale_a is not None and scale_b is not None:
        check_supported_striding(mat_a, mat_b)

    # workaround for Inductor not supporting optional tensor input arguments
    input_nodes: list[Any] = [mat_a, mat_b]
    if scale_a is not None:
        input_nodes.append(realize_inputs(scale_a))
    if scale_b is not None:
        input_nodes.append(realize_inputs(scale_b))
    if offs is not None:
        input_nodes.append(realize_inputs(offs))

    if use_fast_accum is None:
        aten_choice = extern_kernel_choice.bind(
            input_nodes,
            layout,
            out_dtype=out_dtype,
        )
    else:
        aten_choice = extern_kernel_choice.bind(
            input_nodes,
            layout,
            out_dtype=out_dtype,
            use_fast_accum=use_fast_accum,
        )
    if use_fast_accum is None:
        use_fast_accum = False

    choices: list[ChoiceCaller] = []
    if use_aten_gemm_kernels():
        choices.append(aten_choice)

    _, is_nonzero = _is_static_problem(layout)

    # Checking only for the equality of corresponding dims of
    # multiplicands here, relying on meta function checks for
    # everything else.
    if is_nonzero and can_use_triton_kernel(mat_a, mat_b, offs, bias, scale_result):
        scaled = scale_a is not None
        if len(m1_size) == 2:
            if len(m2_size) == 2:
                m, k1 = m1_size
                k2, _ = m2_size
                g = offs.get_size()[0]
                V.graph.sizevars.guard_equals(k1, k2)
                a_is_2d, b_is_2d = True, True
            else:
                g1 = offs.layout.size[0]
                m, k1 = m1_size
                g2, k2, _ = m2_size
                g = V.graph.sizevars.guard_equals(g1, g2)
                V.graph.sizevars.guard_equals(k1, k2)
                a_is_2d, b_is_2d = True, False
        else:
            if len(m2_size) == 2:
                g1 = offs.layout.size[0]
                g2, m, k1 = m1_size
                k2, _ = m2_size
                g = V.graph.sizevars.guard_equals(g1, g2)
                V.graph.sizevars.guard_equals(k1, k2)
                a_is_2d, b_is_2d = False, True
            else:
                g1, m, k1 = m1_size
                g2, k2, _ = m2_size
                g = V.graph.sizevars.guard_equals(g1, g2)
                V.graph.sizevars.guard_equals(k1, k2)
                a_is_2d, b_is_2d = False, False

        triton_has_make_tensor_descriptor = hasattr(tl, "make_tensor_descriptor")
        triton_has_experimental_make_tensor_descriptor = hasattr(
            tl, "_experimental_make_tensor_descriptor"
        )
        use_tma_load = (
            triton_has_make_tensor_descriptor
            or triton_has_experimental_make_tensor_descriptor
        )
        # The make_tensor_descriptor imposes this additional limitation.
        use_tma_load = use_tma_load and (
            mat_a.get_stride()[-1] == 1 and mat_b.get_stride()[-2] == 1
        )

        kwargs = {
            "SCALED": scaled,
            "A_IS_2D": a_is_2d,
            "B_IS_2D": b_is_2d,
            "USE_FAST_ACCUM": use_fast_accum,
            "NUM_SMS": get_num_sms(),
            "USE_TMA_LOAD": use_tma_load,
            "USE_EXPERIMENTAL_MAKE_TENSOR_DESCRIPTOR": triton_has_experimental_make_tensor_descriptor,
        }

        for config in early_config_prune(g, m, grouped_mm_configs(), kwargs):
            kernel_template.maybe_append_choice(
                choices,
                input_nodes=input_nodes,
                layout=layout,
                num_stages=config.num_stages,
                num_warps=config.num_warps,
                **kwargs,
                **config.kwargs,
            )

    input_gen_fns = {
        4: lambda x: create_offsets(
            x, m1_size, m2_size, offs.get_size() if offs is not None else None
        ),
    }
    return autotune_select_algorithm(
        algorithm_name, choices, input_nodes, layout, input_gen_fns=input_gen_fns
    )


@register_lowering(aten._grouped_mm.default, type_promotion_kind=None)
def tuned_grouped_mm(
    mat_a: TensorBox,
    mat_b: TensorBox,
    offs: Optional[TensorBox] = None,
    bias: Optional[TensorBox] = None,
    out_dtype: Optional[torch.dtype] = None,
    layout: Optional[Layout] = None,
) -> TensorBox:
    """Auto-tuning for _grouped_mm() operator."""

    return _tuned_grouped_mm_common(
        "aten._grouped_mm.default",
        "grouped_mm",
        aten__grouped_mm,
        triton_grouped_mm_template,
        mat_a,
        mat_b,
        None,
        None,
        offs,
        bias,
        None,
        out_dtype,
        None,
        layout,
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
    """Auto-tuning for _scaled_grouped_mm() operator."""

    return _tuned_grouped_mm_common(
        "aten._scaled_grouped_mm.default",
        "scaled_grouped_mm",
        aten__scaled_grouped_mm,
        triton_scaled_grouped_mm_template,
        mat_a,
        mat_b,
        scale_a,
        scale_b,
        offs,
        bias,
        scale_result,
        out_dtype,
        use_fast_accum,
        layout,
    )
