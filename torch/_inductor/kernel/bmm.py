# mypy: allow-untyped-defs
import logging
from typing import TYPE_CHECKING, Union

import torch
from torch._dynamo.utils import counters
from torch._inductor.codegen.rocm.ck_universal_gemm_template import CKGemmTemplate

from .. import ir, lowering as L
from ..kernel_inputs import MMKernelInputs
from ..select_algorithm import (
    autotune_select_algorithm,
    ExternKernelChoice,
    SymbolicGridFn,
    TritonTemplate,
)
from ..utils import (
    _use_cutlass_for_op,
    use_aten_gemm_kernels,
    use_ck_gemm_template,
    use_cpp_bmm_template,
    use_cutlass_template,
    use_triton_template,
)
from ..virtualized import V
from .mm_common import _is_static_problem, is_batch_stride_largest_or_zero, mm_args


if TYPE_CHECKING:
    from ..ir import ChoiceCaller
    from ..select_algorithm import KernelTemplate

log = logging.getLogger(__name__)
aten = torch.ops.aten


@SymbolicGridFn
def bmm_grid(b, m, n, meta, *, cdiv):
    return (cdiv(m, meta["BLOCK_M"]) * cdiv(n, meta["BLOCK_N"]), b, 1)


bmm_template = TritonTemplate(
    name="bmm",
    grid=bmm_grid,
    source=r"""
{{def_kernel("A", "B")}}
    M = {{size("A", -2)}}
    N = {{size("B", -1)}}
    K = {{size("A", -1)}}

    stride_aq = {{stride("A", 0)}}
    stride_am = {{stride("A", 1)}}
    stride_ak = {{stride("A", 2)}}

    stride_bq = {{stride("B", 0)}}
    stride_bk = {{stride("B", 1)}}
    stride_bn = {{stride("B", 2)}}

    # based on triton.ops.matmul
    pid = tl.program_id(0).to(INDEX_DTYPE)
    grid_m = (M + BLOCK_M - 1) // BLOCK_M
    grid_n = (N + BLOCK_N - 1) // BLOCK_N

    # re-order program ID for better L2 performance
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // (group_size)
    tl.assume(pid_m >= 0)
    tl.assume(pid_n >= 0)

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    if (stride_am == 1 and stride_ak == M) or (stride_am == K and stride_ak == 1):
        ram = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)
    else:
        ram = rm % M
    if (stride_bk == 1 and stride_bn == K) or (stride_bk == N and stride_bn == 1):
        rbn = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)
    else:
        rbn = rn % N

    rk = tl.arange(0, BLOCK_K)

    idx_q = tl.program_id(1).to(INDEX_DTYPE)  # batch dimension for BMM
    A = A + (ram[:, None] * stride_am + rk[None, :] * stride_ak + idx_q*stride_aq)
    B = B + (rk[:, None] * stride_bk + rbn[None, :] * stride_bn + idx_q*stride_bq)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)
    for k in range(K, 0, -BLOCK_K):
        if EVEN_K:
            a = tl.load(A)
            b = tl.load(B)
        else:
            a = tl.load(A, mask=rk[None, :] < k, other=0.)
            b = tl.load(B, mask=rk[:, None] < k, other=0.)
        acc += tl.dot(a, b, allow_tf32=ALLOW_TF32)
        A += BLOCK_K * stride_ak
        B += BLOCK_K * stride_bk

    # rematerialize rm and rn to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    idx_q = tl.program_id(1).to(INDEX_DTYPE)  # batch dimension for BMM
    idx_m = rm[:, None]
    idx_n = rn[None, :]
    mask = (idx_m < M) & (idx_n < N)

    # inductor generates a suffix
    {{store_output(("idx_q", "idx_m", "idx_n"), "acc", "mask", val_shape=("BLOCK_M", "BLOCK_N"))}}
""",
    cache_codegen_enabled_for_template=True,
)

aten_bmm = ExternKernelChoice(torch.bmm, "at::bmm_out")
aten_bmm_dtype = ExternKernelChoice(
    torch.bmm,
    "at::_bmm_out_dtype_cuda",
    name="bmm_dtype",
    op_overload=aten.bmm.dtype_out,
)
aten_baddbmm = ExternKernelChoice(
    torch.baddbmm, "at::baddbmm_out", op_overload=aten.baddbmm.out
)


@L.register_lowering(aten.bmm)
def tuned_bmm(mat1, mat2, out_dtype=None, *, layout=None):
    """
    Lowering for autotuning aten.bmm with different backends (Aten, Triton, CUTLASS, etc.)
    """
    if all(x.get_device().type == "cpu" for x in [mat1, mat2]):
        # decompose to small ops when memory bound
        if mat1.get_size()[1] == 1 or mat2.get_size()[2] == 1:
            mat1 = L.unsqueeze(mat1, -1)
            mat2 = L.unsqueeze(mat2, 1)
            return L.sum_(L.mul(mat1, mat2), axis=2)

        def is_valid_to_require_contiguous(t):
            if not ir.is_storage_and_layout(t):
                return True
            _, layout = ir.as_storage_and_layout(t, freeze=False)
            return isinstance(layout, ir.FlexibleLayout)

        def is_preferred_layout_as_bmm_input(sizes, strides):
            # contiguous on one of the last two dims
            return (
                strides[-1] == 1 and (sizes[-2] == 1 or strides[-2] >= sizes[-1])
            ) or (strides[-2] == 1 and (sizes[-1] == 1 or strides[-1] >= sizes[-2]))

        # Make the input of bmm contiguous
        # if it is not contiguous on either of the last two dims,
        # because bmm cpu implementation would do contiguous() if not.
        # This is to avoid additional copies in bmm.
        def may_require_contiguous(t, meta_t):
            sizes = meta_t.meta["val"].size()
            strides = meta_t.meta["val"].stride()
            if not is_preferred_layout_as_bmm_input(sizes, strides):
                t = ir.ExternKernel.require_contiguous(t)
            return t

        if is_valid_to_require_contiguous(mat1):
            meta_mat1 = V.graph.current_node.args[0]
            mat1 = may_require_contiguous(mat1, meta_mat1)
        if is_valid_to_require_contiguous(mat2):
            meta_mat2 = V.graph.current_node.args[1]
            mat2 = may_require_contiguous(mat2, meta_mat2)

    # TODO(coconutruben): integrate into MMKernelInputs when all callsites use that
    m, n, k, layout, mat1, mat2 = mm_args(
        mat1, mat2, layout=layout, out_dtype=out_dtype
    )
    name = "bmm"

    # Create MMKernelInputs for BMM at the top
    kernel_inputs = MMKernelInputs([mat1, mat2], out_dtype=out_dtype)

    # below is for getting an overview logging info of inductor mms
    batch_size = mat1.get_size()[0]  # Extract batch dimension
    counters["aten_mm_info"][f"aten.bmm_{batch_size}_{m}_{n}_{k}"] += 1
    log.info(
        "Tuned aten.bmm: batch=%s, m=%s, n=%s, k=%s, mat1_dtype=%s, mat2_dtype=%s, output_layout=%s",
        batch_size,
        m,
        n,
        k,
        mat1.get_dtype(),
        mat2.get_dtype(),
        layout,
    )

    aten_handler: ExternKernelChoice = aten_bmm
    aten_extra_kwargs = {}
    if out_dtype:
        assert mat1.get_device().type == "cuda", "out_dtype is only supported for CUDA"
        aten_handler = aten_bmm_dtype
        aten_extra_kwargs = {"out_dtype": out_dtype}

    choices: list[ChoiceCaller] = []

    # Collect all templates for unified call
    templates_to_use: list[Union[ExternKernelChoice, KernelTemplate]] = []
    kwarg_overrides = {}

    if use_aten_gemm_kernels():
        templates_to_use.append(aten_handler)
        kwarg_overrides[aten_handler.uid] = aten_extra_kwargs

    if use_triton_template(layout, check_max_autotune=False):
        # TODO: add out_dtype support for Triton Template
        assert out_dtype is None, "out_dtype is not supported for Triton"
        templates_to_use.append(bmm_template)

    # Single unified call for all templates
    choices.extend(
        V.choices.get_template_configs(
            kernel_inputs,
            templates_to_use,
            name,
            kwarg_overrides=kwarg_overrides,
        )
    )
    _, is_nonzero = _is_static_problem(layout)
    batch_stride_largest_or_zero = is_batch_stride_largest_or_zero(mat1, mat2, layout)
    if (
        batch_stride_largest_or_zero
        and is_nonzero
        and use_cutlass_template(layout, m, n, k)
        and _use_cutlass_for_op(name)
    ):
        from ..codegen.cuda.gemm_template import CUTLASS3xGemmTemplate

        CUTLASS3xGemmTemplate.add_cutlass_gemm_choices(
            choices, layout, kernel_inputs.nodes()
        )  # type: ignore[arg-type]

    if use_cpp_bmm_template(layout, mat1, mat2):
        from ..codegen.cpp_bmm_template import CppBmmTemplate

        CppBmmTemplate.add_choices(
            choices,
            layout,
            kernel_inputs.nodes(),
        )

    if use_ck_gemm_template(layout, m, n, k):
        CKGemmTemplate.add_ck_gemm_choices(choices, layout, kernel_inputs.nodes())

    return autotune_select_algorithm(name, choices, kernel_inputs.nodes(), layout)


@L.register_lowering(aten.baddbmm)
def tuned_baddbmm(inp, mat1, mat2, *, alpha=1, beta=1, layout=None):
    """
    Lowering for autotuning aten.mm with different backends (Aten, Triton, CUTLASS, etc.)
    """
    # TODO(coconutruben): integrate into MMKernelInputs when all callsites use that
    m, n, k, layout, mat1, mat2, inp = mm_args(mat1, mat2, inp, layout=layout)

    # Create MMKernelInputs for BadDBMM at the top
    kernel_inputs = MMKernelInputs(
        [inp, mat1, mat2], scalars=dict(alpha=alpha, beta=beta)
    )

    # below is for getting an overview logging info of inductor mms
    batch_size = mat1.get_size()[0]
    counters["aten_mm_info"][f"aten.baddbmm_{batch_size}_{m}_{n}_{k}"] += 1
    log.info(
        "Tuned aten.baddbmm: batch_size=%s, m=%s, n=%s, k=%s, mat1_dtype=%s, mat2_dtype=%s, inp=%s, output_layout=%s",
        batch_size,
        m,
        n,
        k,
        mat1.get_dtype(),
        mat2.get_dtype(),
        inp.get_dtype(),
        layout,
    )
    name = "baddbmm"
    # options to tune from
    choices: list[ChoiceCaller] = []

    # Collect all templates for unified call
    templates_to_use: list[Union[ExternKernelChoice, KernelTemplate]] = []
    if use_aten_gemm_kernels():
        templates_to_use.append(aten_baddbmm)

    if use_triton_template(layout, check_max_autotune=False):
        templates_to_use.append(bmm_template)

    # Single unified call for all templates
    choices.extend(
        V.choices.get_template_configs(kernel_inputs, templates_to_use, name)
    )

    return autotune_select_algorithm(name, choices, kernel_inputs.nodes(), layout)
