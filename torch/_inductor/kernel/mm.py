# mypy: allow-untyped-defs
import functools
import logging
from typing import Any, Dict, List, Optional

import torch
from torch._inductor.autoheuristic.autoheuristic import AutoHeuristicSelectAlgorithm
from torch._inductor.autoheuristic.autoheuristic_utils import (
    AHContext,
    context_add_strides,
    context_add_using_tf32,
    get_mixedmm_precondition,
    mixed_mm_operations,
    mm_operations,
)
from torch._inductor.codegen.cpp_gemm_template import CppGemmTemplate
from torch._inductor.virtualized import V

from .. import config as inductor_config, ir
from ..codegen.common import BackendFeature
from ..codegen.cuda.gemm_template import CUTLASS2xGemmTemplate, CUTLASS3xGemmTemplate
from ..codegen.rocm.ck_universal_gemm_template import CKGemmTemplate
from ..codegen.wrapper import PythonWrapperCodegen
from ..ir import FlexibleLayout, is_triton
from ..lowering import register_lowering
from ..select_algorithm import (
    autotune_select_algorithm,
    ExternKernelChoice,
    NoValidChoicesError,
    TritonTemplate,
)
from ..utils import (
    get_gpu_shared_memory,
    use_aten_gemm_kernels,
    use_ck_gemm_template,
    use_cpp_gemm_template,
    use_cutlass_template,
    use_max_autotune,
    use_triton_template,
)
from .mm_common import (
    _is_static_problem,
    addmm_epilogue,
    extra_mm_configs,
    int8_mm_configs,
    mixed_mm_configs,
    mm_args,
    mm_configs,
    mm_grid,
    mm_options,
    triton_config,
)


log = logging.getLogger(__name__)
aten = torch.ops.aten

mm_template = TritonTemplate(
    name="mm",
    grid=mm_grid,
    source=r"""
{{def_kernel("A", "B")}}
    M = {{size("A", 0)}}
    N = {{size("B", 1)}}
    K = {{size("A", 1)}}
    if M * N == 0:
        # early exit due to zero-size input(s)
        return
    stride_am = {{stride("A", 0)}}
    stride_ak = {{stride("A", 1)}}
    stride_bk = {{stride("B", 0)}}
    stride_bn = {{stride("B", 1)}}

    # based on triton.ops.matmul
    pid = tl.program_id(0)
    grid_m = (M + BLOCK_M - 1) // BLOCK_M
    grid_n = (N + BLOCK_N - 1) // BLOCK_N

    # re-order program ID for better L2 performance
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // (group_size)

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    if (stride_am == 1 and stride_ak == M) or (stride_am == K and stride_ak == 1):
        offs_a_m = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)
    else:
        offs_a_m = rm % M
    if (stride_bk == 1 and stride_bn == K) or (stride_bk == N and stride_bn == 1):
        offs_b_n = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)
    else:
        offs_b_n = rn % N
    offs_k = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)

    for k_idx in range(0, tl.cdiv(K, BLOCK_K)):
        {% if not EVEN_K %}
        a_mask = offs_k[None, :] < (K - k_idx * BLOCK_K)
        b_mask = offs_k[:, None] < (K - k_idx * BLOCK_K)
        {% endif %}
        a_k_idx_vals = offs_k[None, :] + (k_idx * BLOCK_K)
        b_k_idx_vals = offs_k[:, None] + (k_idx * BLOCK_K)

        idx_m = offs_a_m[:, None]
        idx_n = a_k_idx_vals
        {{load_input("A", "a", ("idx_m", "idx_n"), mask=None if EVEN_K else "a_mask", indent_width=8)}}

        idx_m = b_k_idx_vals
        idx_n = offs_b_n[None, :]
        {{load_input("B", "b", ("idx_m", "idx_n"), mask=None if EVEN_K else "b_mask", indent_width=8)}}
        {% if B_PROLOGUE_CAST_TYPE %} # TODO - replace with prologue fusion
        b = b.to(B_PROLOGUE_CAST_TYPE)
        {% endif %}
        acc += tl.dot(a, b, allow_tf32=ALLOW_TF32)

    # rematerialize rm and rn to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    idx_m = rm[:, None]
    idx_n = rn[None, :]
    mask = (idx_m < M) & (idx_n < N)

    # inductor generates a suffix
    {{store_output(("idx_m", "idx_n"), "acc", "mask")}}
""",
)


# prevent duplication registration of extern functions
@functools.lru_cache(None)
def lazy_register_extern_choice(fn):
    return ExternKernelChoice(fn)


aten_mm = ExternKernelChoice(torch.mm, "at::mm_out")

aten_addmm = ExternKernelChoice(
    torch.addmm, "at::addmm_out", op_overload=aten.addmm.default
)

aten__int_mm = ExternKernelChoice(torch._int_mm, "at::_int_mm")

aten__sparse_semi_structured_mm = ExternKernelChoice(
    torch._sparse_semi_structured_mm,
    "at::_sparse_semi_structured_mm",
    has_out_variant=False,
)


def _is_int8_mat(mat):
    return mat.get_dtype() in (torch.int8, torch.uint8)


def _is_large_block_for_cpu(m, n, k):
    # Thresholds are experimentally determined to reduce Triton CPU compile times
    return m * n > 2**13


def mm_config_kwargs(device):
    if device == "cpu":
        return {
            "scale": 0.5,
            "exclude": _is_large_block_for_cpu,
        }
    return {}


def bias_addmm(inp, mat1, mat2, *, out=None, alpha=1, beta=1):
    """
    Giving torch.addmm a 1D tensor calls a different (faster) cublasLt
    kernel under the hood.  There are a few shapes where this is slower,
    but they are rare.
    """
    if inp.stride(0) == 0 or inp.size(0) == 1:
        return torch.addmm(inp[0], mat1, mat2, out=out, alpha=alpha, beta=beta)
    return torch.addmm(inp, mat1, mat2, out=out, alpha=alpha, beta=beta)


aten_bias_addmm = ExternKernelChoice(bias_addmm, None)


@register_lowering(aten.mm, type_promotion_kind=None)
def tuned_mm(mat1, mat2, *, layout=None):
    m, n, k, layout, mat1, mat2 = mm_args(mat1, mat2, layout=layout)
    name = "mm"

    aten_layout = layout
    if not use_max_autotune():
        aten_layout = FlexibleLayout(
            device=layout.device, dtype=layout.dtype, size=layout.size
        )

    # options to tune from
    choices = (
        [aten_mm.bind((mat1, mat2), aten_layout)] if use_aten_gemm_kernels() else []
    )
    static_shape, is_nonzero = _is_static_problem(layout)
    if is_nonzero and use_triton_template(layout):
        for config in mm_configs(m, n, k, **mm_config_kwargs(ir.get_device_type(mat1))):
            mm_template.maybe_append_choice(
                choices,
                input_nodes=(mat1, mat2),
                layout=layout,
                **mm_options(config, m, n, k, layout),
            )
    if is_nonzero and use_cutlass_template(layout, m, n, k):
        CUTLASS3xGemmTemplate.add_cutlass_gemm_choices(choices, layout, [mat1, mat2])

    if is_nonzero and use_ck_gemm_template(layout, m, n, k):
        CKGemmTemplate.add_ck_gemm_choices(choices, layout, [mat1, mat2])

    if use_cpp_gemm_template(layout, mat1, mat2):
        CppGemmTemplate.add_choices(
            choices,
            layout,
            [mat1, mat2],
        )

    input_nodes = [mat1, mat2]
    if (
        is_nonzero
        and use_triton_template(layout)
        and torch._inductor.config.run_autoheuristic(name)
        and is_triton(mat1)
    ):
        always_included = []
        if use_aten_gemm_kernels():
            always_included.append("extern_mm")
        num_choices_before_extra_configs = len(choices)
        for config in extra_mm_configs(
            m, n, k, **mm_config_kwargs(ir.get_device_type(mat1))
        ):
            mm_template.maybe_append_choice(
                choices,
                input_nodes=(mat1, mat2),
                layout=layout,
                **mm_options(config, m, n, k, layout),
            )

        # using AutoHeuristic for ranking
        ah_choices = mm_autoheuristic(
            mat1,
            mat2,
            m,
            n,
            k,
            choices,
            name,
            input_nodes,
            mm_operations(),
            None,
            top_k=10,
            always_included=always_included,
        )
        if not torch._inductor.config.collect_autoheuristic(name):
            # if we are collecting data, we do not want to modify choices
            if ah_choices is not None and len(ah_choices) > 0:
                # the order in which autoheuristic returns choices is not the same as
                # as the order of choices, which affects things like epilogue fusion.
                # once epilogue fusion benchmarks choices in sorted order, I think we can
                # just use the order returned by autoheuristic
                choices = [choice for choice in choices if choice in ah_choices]
            else:
                choices = choices[:num_choices_before_extra_configs]

    if (
        len(choices) == 0
        and not use_aten_gemm_kernels()
        and inductor_config.autotune_fallback_to_aten
    ):
        log.warning("No choices for GEMM, using ATen backend as fallback")
        return aten_mm.bind((mat1, mat2), aten_layout).output_node()

    for k in inductor_config.external_matmul:
        choices.append(lazy_register_extern_choice(k).bind((mat1, mat2), layout))

    try:
        return autotune_select_algorithm(name, choices, [mat1, mat2], layout)
    except NoValidChoicesError:
        if not inductor_config.autotune_fallback_to_aten:
            raise
        log.warning("All choices for GEMM were invalid, using ATen backend as fallback")
        return aten_mm.bind((mat1, mat2), aten_layout).output_node()


@register_lowering(aten._int_mm, type_promotion_kind=None)
def tuned_int_mm(mat1, mat2, *, layout=None):
    m, n, k, layout, mat1, mat2 = mm_args(
        mat1, mat2, layout=layout, out_dtype=torch.int32
    )
    static_shape, is_nonzero = _is_static_problem(layout)
    use_cutlass = static_shape and is_nonzero and use_cutlass_template(layout, m, n, k)

    choices = (
        [aten__int_mm.bind((mat1, mat2), layout)] if use_aten_gemm_kernels() else []
    )

    # TODO: Re-enable eager mode implementation once cuBLAS is fixed
    if use_cutlass or use_triton_template(layout, enable_int32=True):
        choices = []

    if use_cutlass:
        CUTLASS3xGemmTemplate.add_cutlass_gemm_choices(
            choices, layout, [mat1, mat2], fuseable=True, non_fuseable=True
        )
    if is_nonzero and use_triton_template(layout, enable_int32=True):
        for config in int8_mm_configs(
            m, n, k, **mm_config_kwargs(ir.get_device_type(mat1))
        ):
            mm_template.maybe_append_choice(
                choices,
                input_nodes=(mat1, mat2),
                layout=layout,
                **mm_options(config, m, n, k, layout),
            )
    if len(choices) == 0:
        log.warning(
            "No choices for integer GEMM avaialbe using configured backends, using ATen backend as fallback"
        )
        choices = [aten__int_mm.bind((mat1, mat2), layout)]

    try:
        return autotune_select_algorithm("int_mm", choices, [mat1, mat2], layout)
    except NoValidChoicesError:
        if not inductor_config.autotune_fallback_to_aten:
            raise
        log.warning("All choices for GEMM were invalid, using ATen backend as fallback")
        choices = [aten__int_mm.bind((mat1, mat2), layout)]
        return autotune_select_algorithm("int_mm", choices, [mat1, mat2], layout)


@register_lowering(aten.addmm, type_promotion_kind=None)
def tuned_addmm(inp, mat1, mat2, *, alpha=1, beta=1, layout=None):
    ordered_kwargs_for_cpp_kernel = ("beta", "alpha")
    m, n, k, layout, mat1, mat2, inp_expanded = mm_args(mat1, mat2, inp, layout=layout)
    static_shape, is_nonzero = _is_static_problem(layout)
    if (not is_nonzero) or (not use_max_autotune()):
        # Use a FlexibleLayout if we are not autotuning.
        # This allows padding strides for the output.
        from torch._inductor.ir import FixedLayout, FlexibleLayout

        if isinstance(layout, FixedLayout):
            layout = FlexibleLayout(
                device=layout.device, dtype=layout.dtype, size=layout.size
            )
        choices = (
            [
                aten_addmm.bind(
                    (inp, mat1, mat2),
                    layout,
                    alpha=alpha,
                    beta=beta,
                )
            ]
            if use_aten_gemm_kernels()
            else []
        )
        return autotune_select_algorithm("addmm", choices, [inp, mat1, mat2], layout)

    choices = (
        [
            aten_addmm.bind(
                (inp_expanded, mat1, mat2),
                layout,
                alpha=alpha,
                beta=beta,
            )
        ]
        if use_aten_gemm_kernels()
        else []
    )

    if (
        use_aten_gemm_kernels()
        and inp_expanded.get_stride()[0] == 0
        and inp_expanded.get_device().type == "cuda"
        and inductor_config.triton.autotune_cublasLt
    ):
        # unexpand inp to make sure fused addmm from cublasLt is used
        choices.insert(
            0,
            aten_bias_addmm.bind(
                (inp_expanded, mat1, mat2), layout, alpha=alpha, beta=beta
            ),
        )

    if is_nonzero and use_triton_template(layout):
        for config in mm_configs(m, n, k, **mm_config_kwargs(ir.get_device_type(mat1))):
            mm_template.maybe_append_choice(
                choices,
                input_nodes=(inp_expanded, mat1, mat2),
                layout=layout,
                **mm_options(config, m, n, k, layout),
                prefix_args=1,
                epilogue_fn=addmm_epilogue(layout.dtype, alpha, beta),
            )

    if static_shape and is_nonzero and use_cutlass_template(layout, m, n, k):
        # Filter out a known cause of CUDA illegal memory access errors
        # broadcasting on the last dim of the bias term seems not to be working
        # in the linear GEMM epilogue used by addmm.
        if (
            PythonWrapperCodegen.statically_known_int_or_none(
                inp_expanded.layout.stride[-1]
            )
            != 0
        ):
            CUTLASS3xGemmTemplate.add_cutlass_gemm_choices(
                choices,
                layout,
                [mat1, mat2, inp_expanded],
                alpha=alpha,
                beta=beta,
            )

    if is_nonzero and use_ck_gemm_template(layout, m, n, k):
        CKGemmTemplate.add_ck_gemm_choices(
            choices,
            layout,
            [mat1, mat2, inp_expanded],
            alpha=alpha,
            beta=beta,
        )

    if use_cpp_gemm_template(layout, mat1, mat2):
        CppGemmTemplate.add_choices(
            choices,
            layout,
            [inp_expanded, mat1, mat2],
            alpha=alpha,
            beta=beta,
            has_bias=True,
        )

    add_aten_fallback = False
    if len(choices) == 0:
        log.warning("No choices for GEMM, using ATen backend as fallback")
        add_aten_fallback = True

    if add_aten_fallback:
        choices.append(
            aten_addmm.bind(
                (inp_expanded, mat1, mat2),
                layout,
                ordered_kwargs_for_cpp_kernel,
                alpha=alpha,
                beta=beta,
            )
        )

        if (
            inp_expanded.get_stride()[0] == 0
            and inp_expanded.get_device().type == "cuda"
            and inductor_config.triton.autotune_cublasLt
        ):
            # unexpand inp to make sure fused addmm from cublasLt is used
            choices.insert(
                0,
                aten_bias_addmm.bind(
                    (inp_expanded, mat1, mat2), layout, alpha=alpha, beta=beta
                ),
            )
    try:
        return autotune_select_algorithm(
            "addmm", choices, [inp_expanded, mat1, mat2], layout
        )
    except NoValidChoicesError:
        if not inductor_config.autotune_fallback_to_aten:
            raise
        log.warning("All choices for GEMM were invalid, using ATen backend as fallback")
        fallback_choice = aten_addmm.bind(
            (inp, mat1, mat2),
            layout,
            ordered_kwargs_for_cpp_kernel,
            alpha=alpha,
            beta=beta,
        )
        return fallback_choice.output_node()


@register_lowering(aten._sparse_semi_structured_mm, type_promotion_kind=None)
def tuned_sparse_semi_structured_mm(
    mat1, mat1_meta, mat2, *, out_dtype=None, layout=None
):
    from torch._inductor.select_algorithm import realize_inputs

    mat1, mat1_meta, mat2 = realize_inputs(mat1, mat1_meta, mat2)
    m1, k1 = mat1.get_size()
    m2, _ = mat1_meta.get_size()
    k2, n = mat2.get_size()
    m = V.graph.sizevars.guard_equals(m1, m2)
    k = V.graph.sizevars.guard_equals(2 * k1, k2)

    if layout is None:
        from torch._inductor.ir import FixedLayout

        layout = FixedLayout(
            mat2.get_device(),
            out_dtype if out_dtype else mat2.get_dtype(),
            [m, n],
            [n, 1],
        )
    else:
        assert out_dtype is None, "out_dtype is ignored if layout is specified."

    choices = (
        [
            aten__sparse_semi_structured_mm.bind(
                (mat1, mat1_meta, mat2), layout, out_dtype=out_dtype
            )
        ]
        if use_aten_gemm_kernels()
        else []
    )

    if m * n != 0 and use_cutlass_template(layout, m, n, k):
        CUTLASS2xGemmTemplate.add_cutlass_gemm_choices(
            choices, layout, [mat1, mat2, mat1_meta], fuseable=True, non_fuseable=True
        )

    return autotune_select_algorithm(
        "sparse_semi_structured_mm", choices, [mat1, mat1_meta, mat2], layout
    )


def fallback_mixed_mm(mat1, mat2, *, out):
    return torch.mm(mat1, mat2.to(mat1.dtype), out=out)


aten_fallback_mixed_mm = ExternKernelChoice(fallback_mixed_mm, None)


@functools.lru_cache(None)
def _is_sm7x_or_older_gpu(index: Optional[int]) -> bool:
    props = torch.cuda.get_device_properties(index or 0)
    return props.major <= 7


def dims_are_int(dims):
    return all(isinstance(dim, int) for dim in dims)


def try_heuristic(m, n, k, choices, mat1, mat2, mat2_dtype, layout):
    m, n, k = get_size_hints(mat1, mat2, m, n, k)
    if not dims_are_int([m, n, k]):
        return None

    if mat1.dtype != torch.float16:
        return None

    # only use heuristic if we are running on an A100
    # torch.cuda.get_device_capability() >= (8, 0) returns true for A10G
    # which does not have enough shared memory for one of the configs
    if (
        not torch.cuda.get_device_capability() >= (8, 0)
    ) or get_gpu_shared_memory() != 166912:
        return None

    if m == 1 and (n % 16 != 0 or k % 16 != 0):
        return None

    if m <= 16 and n >= 4096 and k >= 4096:
        return triton_config(
            BLOCK_M=16,
            BLOCK_N=64,
            BLOCK_K=128,
            num_stages=5,
            num_warps=4,
        )
    elif m > 16 and m <= 32 and n >= 4096 and k >= 4096:
        return triton_config(
            BLOCK_M=32,
            BLOCK_N=32,
            BLOCK_K=128,
            num_stages=5,
            num_warps=4,
        )
    elif m > 32 and m <= 64 and n >= 4096 and k >= 4096:
        return triton_config(
            BLOCK_M=64,
            BLOCK_N=32,
            BLOCK_K=128,
            num_stages=5,
            num_warps=4,
        )
    return None


def mm_autoheuristic(
    mat1,
    mat2,
    m,
    n,
    k,
    choices,
    name,
    input_nodes,
    ops,
    precondition,
    top_k: Optional[int] = None,
    always_included=None,
):
    m, n, k = get_size_hints(mat1, mat2, m, n, k)
    if not dims_are_int([m, n, k]):
        return None
    mat1_stride, mat2_stride = get_size_hints_strides(mat1, mat2)

    def get_context(m, k, n, mat1, mat2, mat1_stride, mat2_stride):
        context = AHContext()
        context.add_feature("m", m)
        context.add_feature("k", k)
        context.add_feature("n", n)
        context.add_feature("mat1_dtype", mat1.layout.dtype, is_categorical=True)
        context.add_feature("mat2_dtype", mat2.layout.dtype, is_categorical=True)
        context_add_strides(context, "mat1", mat1_stride)
        context_add_strides(context, "mat2", mat2_stride)
        context.add_feature(
            "mat1_iscontig", mat1.layout.is_contiguous(), is_categorical=True
        )
        context.add_feature(
            "mat2_iscontig", mat2.layout.is_contiguous(), is_categorical=True
        )
        if name == "mm":
            # for mixed_mm, we only consider fp16
            context_add_using_tf32(context, mat1.layout.dtype)
        return context

    def fallback():
        return None

    context = get_context(m, k, n, mat1, mat2, mat1_stride, mat2_stride)
    autoheuristic = AutoHeuristicSelectAlgorithm(
        fallback=fallback,
        choices=choices,
        input_nodes=input_nodes,
        context=context,
        name=name,
        augment_context=ops,
        precondition=precondition,
    )

    if top_k is not None:
        # TODO: is there a cleaner way to ensure aten.mm is always included?
        return autoheuristic.get_top_k_choices_caller(
            top_k, always_included=always_included
        )

    return autoheuristic.get_choice_caller()


def get_size_hints(mat1, mat2, m, n, k):
    if not isinstance(m, int) or not isinstance(k, int):
        (m, k) = V.graph.sizevars.size_hints(
            mat1.get_size(),
            fallback=torch._inductor.config.unbacked_symint_fallback,
        )

    if not isinstance(n, int) or not isinstance(k, int):
        (k, n) = V.graph.sizevars.size_hints(
            mat2.get_size(),
            fallback=torch._inductor.config.unbacked_symint_fallback,
        )
    return m, n, k


def get_size_hints_strides(mat1, mat2):
    mat1_stride = mat1.layout.stride
    mat2_stride = mat2.layout.stride
    strides = [mat1_stride, mat2_stride]
    strides_hints = []
    for stride in strides:
        if not isinstance(stride, int):
            stride = V.graph.sizevars.size_hints(
                stride,
                fallback=torch._inductor.config.unbacked_symint_fallback,
            )
        strides_hints.append(stride)
    return strides_hints[0], strides_hints[1]


def tuned_mixed_mm(mat1, mat2, mat2_dtype):
    m, n, k, layout, mat1, mat2 = mm_args(mat1, mat2, layout=None)
    static_shape, is_nonzero = _is_static_problem(layout)

    fallback = aten_fallback_mixed_mm.bind((mat1, mat2), layout)

    choices = [fallback]

    # can't use triton kernel unless one of these is true or if running on v100 (numerical issues)
    skip_triton = (
        (
            mat1.layout.dtype != torch.float32
            and not (mat2.layout.is_contiguous() or mat2.layout.is_transposed())
        )
        or _is_sm7x_or_older_gpu(layout.device.index)
        or inductor_config.mixed_mm_choice == "aten"
        or not V.graph.has_feature(layout.device, BackendFeature.TRITON_TEMPLATES)
        or (
            mat1.layout.dtype == torch.float32 and torch.backends.cuda.matmul.allow_tf32
        )
        or (mat1.layout.dtype == torch.bfloat16 and mat2.layout.dtype == torch.uint8)
    )

    if inductor_config.mixed_mm_choice == "triton":
        choices = []

    if not skip_triton:
        b_prologue_cast_type = f"tl.{mat2_dtype}".replace("torch.", "")
        if static_shape and inductor_config.mixed_mm_choice == "heuristic":
            choices = []
            config = try_heuristic(m, n, k, choices, mat1, mat2, mat2_dtype, layout)
            if config is not None:
                mm_template.maybe_append_choice(
                    choices,
                    input_nodes=(mat1, mat2),
                    layout=layout,
                    **mm_options(config, m, n, k, layout, b_prologue_cast_type),
                )
            choices.append(fallback)

        has_int8_tensor = _is_int8_mat(mat1) or _is_int8_mat(mat2)
        for config in mixed_mm_configs(
            m,
            n,
            k,
            has_int8_tensor=has_int8_tensor,
            **mm_config_kwargs(ir.get_device_type(mat1)),
        ):
            mm_template.maybe_append_choice(
                choices,
                input_nodes=(mat1, mat2),
                layout=layout,
                **mm_options(config, m, n, k, layout, b_prologue_cast_type),
            )

    if static_shape and is_nonzero and use_cutlass_template(layout, m, n, k):
        CUTLASS3xGemmTemplate.add_cutlass_gemm_choices(
            choices, layout, [mat1, mat2], fuseable=True, non_fuseable=True
        )
        CUTLASS2xGemmTemplate.add_cutlass_gemm_choices(
            choices, layout, [mat1, mat2], fuseable=True, non_fuseable=True
        )

    if skip_triton and not choices:
        choices = [fallback]

    name = "mixed_mm"
    input_nodes = [mat1, mat2]
    if torch._inductor.config.run_autoheuristic(name):
        choice = mm_autoheuristic(
            mat1,
            mat2,
            m,
            n,
            k,
            choices,
            name,
            input_nodes,
            mixed_mm_operations(),
            get_mixedmm_precondition,
        )
        if (
            not skip_triton
            and inductor_config.mixed_mm_choice == "heuristic"
            and choice is not None
        ):
            choices.insert(0, choice)
    return autotune_select_algorithm(name, choices, input_nodes, layout)


# This op is a special case of the int_mm op which we use based on the pattern
# _int_mm -> mul (defined in ../fx_passes/post_grad.py) in order to prevent
# realization of the int32 _int_mm output by forcing fusion with the mul op.
# This is only used when config.force_fuse_int_mm_with_mul = True
def tuned_fused_int_mm_mul(mat1, mat2, mat3, out_dtype, *, layout=None):
    out_dtype = (
        torch.promote_types(mat3.get_dtype(), torch.int32)
        if out_dtype is None
        else out_dtype
    )
    m, n, k, layout, mat1, mat2, mat3 = mm_args(
        mat1, mat2, mat3, layout=layout, out_dtype=out_dtype
    )
    choices: List[Dict[Any, Any]] = []
    for config in int8_mm_configs(
        m, n, k, **mm_config_kwargs(ir.get_device_type(mat1))
    ):
        mm_template.maybe_append_choice(
            choices,
            input_nodes=(mat1, mat2, mat3),
            layout=layout,
            **dict(mm_options(config, m, n, k, layout), ACC_TYPE="tl.int32"),
            suffix_args=1,
            epilogue_fn=V.ops.mul,
        )
    return autotune_select_algorithm("int_mm", choices, [mat1, mat2, mat3], layout)
