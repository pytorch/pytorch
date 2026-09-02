# mypy: allow-untyped-defs
import dataclasses
import logging
import math
from typing import TYPE_CHECKING

import torch
from torch._dynamo.utils import counters
from torch._inductor.codegen.rocm.ck_universal_gemm_template import CKGemmTemplate
from torch._inductor.kernel.mm_common import load_kernel_template

from .. import config as inductor_config, ir, lowering as L
from ..kernel_inputs import MMKernelInputs
from ..lowering import lowerings, make_pointwise, make_reduction, transform_args
from ..runtime.runtime_utils import get_max_y_grid
from ..select_algorithm import (
    autotune_select_algorithm,
    ExternKernelChoice,
    SymbolicGridFn,
    TritonTemplate,
)
from ..utils import (
    _use_cutlass_for_op,
    get_num_sms,
    use_aten_gemm_kernels,
    use_ck_gemm_template,
    use_cpp_bmm_template,
    use_cutlass_template,
    use_nv_universal_gemm_template,
    use_triton_template,
)
from torch.utils._triton import has_triton_stable_tma_api
from ..virtualized import ops, V
from .mm_common import (
    _is_static_problem,
    is_batch_stride_largest_or_zero,
    mm_args,
    use_native_matmul,
)


if TYPE_CHECKING:
    from ..ir import ChoiceCaller
    from ..select_algorithm import KernelTemplate

log = logging.getLogger(__name__)
aten = torch.ops.aten


@SymbolicGridFn
def bmm_grid(b, m, n, meta, *, cdiv, max):
    tiles = cdiv(m, meta["BLOCK_M"]) * cdiv(n, meta["BLOCK_N"])
    # Split batch across grid_y and grid_z to avoid exceeding CUDA grid_y limit.
    # When b <= max_y_grid, grid_z = 1 and behavior is identical to the original.
    max_y_grid = get_max_y_grid()
    grid_z = max(cdiv(b, max_y_grid), 1)
    grid_y = cdiv(b, grid_z)
    return (tiles, grid_y, grid_z)


# We define each template kernel in a separate file which is the name of the input to load_kernel_template
# (e.g. triton_bmm for templates/triton_bmm.py.jinja).
# If you are adding a new template, please follow that pattern and add a new file with your implementation in the templates folder.
bmm_template = TritonTemplate(
    name="bmm",
    grid=bmm_grid,
    source=load_kernel_template("triton_bmm"),
    cache_codegen_enabled_for_template=True,
)


@SymbolicGridFn
def blackwell_bmm_grid(b, m, n, meta, *, cdiv, max, min):
    grid_m = cdiv(m, meta["BLOCK_M"])
    if meta["TWO_CTAS"]:
        grid_m = cdiv(grid_m, 2) * 2
    tiles = grid_m * cdiv(n, meta["BLOCK_N"])
    grid_x = min(meta["NUM_SMS"], tiles)
    if meta["TWO_CTAS"]:
        grid_x = grid_x // 2 * 2
    max_y_grid = get_max_y_grid()
    grid_z = max(cdiv(b, max_y_grid), 1)
    return (grid_x, cdiv(b, grid_z), grid_z)


blackwell_bmm_template = TritonTemplate(
    name="blackwell_bmm",
    grid=blackwell_bmm_grid,
    source=load_kernel_template("triton_blackwell_ws_persistent_device_tma_bmm"),
    cache_codegen_enabled_for_template=True,
    prologue_loads_all_inputs=True,
)


@dataclasses.dataclass(frozen=True)
class BlackwellBMMConfig:
    block_m: int
    block_n: int
    block_k: int
    num_stages: int
    epilogue_subtile: int = 1
    two_ctas: bool = False


def append_blackwell_bmm_choice(
    choices,
    input_nodes,
    layout,
    *,
    config: BlackwellBMMConfig,
):
    mat1, mat2 = input_nodes
    if len(mat1.get_size()) != 3 or len(mat2.get_size()) != 3:
        raise NotImplementedError("Blackwell BMM requires rank-3 operands")
    batch, m, k = map(int, mat1.get_size())
    batch_b, k_b, n = map(int, mat2.get_size())
    if batch != batch_b or k != k_b:
        raise NotImplementedError("Blackwell BMM does not broadcast logical batches")
    output_size = list(map(int, layout.size))
    flattened_output = output_size == [batch * m, n]
    if output_size != [batch, m, n] and not flattened_output:
        raise NotImplementedError("unexpected Blackwell BMM output layout")
    a_row_major = mat1.get_stride()[2] == 1
    a_col_major = mat1.get_stride()[1] == 1
    b_row_major = mat2.get_stride()[2] == 1
    b_col_major = mat2.get_stride()[1] == 1
    if not (a_row_major or a_col_major) or not (b_row_major or b_col_major):
        raise NotImplementedError(
            "Blackwell BMM requires one contiguous matrix dimension"
        )
    m_tiles = math.ceil(m / config.block_m)
    if config.two_ctas:
        if not flattened_output:
            raise NotImplementedError(
                "2CTA Blackwell BMM requires a flattened rank-2 output descriptor"
            )
        if m_tiles % 2:
            raise NotImplementedError("2CTA Blackwell BMM requires two useful M peers")
    kwargs = {
        "BLOCK_M": config.block_m,
        "BLOCK_N": config.block_n,
        "BLOCK_K": config.block_k,
        "GROUP_M": 8,
        "NUM_SMS": get_num_sms(),
        "A_ROW_MAJOR": a_row_major,
        "B_ROW_MAJOR": b_row_major,
        "ALLOW_TF32": False,
        "USE_META_WS": True,
        "WARP_SPECIALIZE": True,
        "FLATTEN": False,
        "DATA_PARTITION_FACTOR": 1,
        "SEPARATE_EPILOGUE_STORE": True,
        "EPILOGUE_SUBTILE": config.epilogue_subtile,
        "TWO_CTAS": config.two_ctas,
        "FLATTEN_OUTPUT": flattened_output,
        "TMA_EXPERIMENTAL_API": not has_triton_stable_tma_api(),
        # Keep the output a normal logical rank-3 tensor.  Until 2CTA output
        # transformation supports rank-3 descriptors, use the generic pointer
        # store emitted by store_output.
        "tma_store": flattened_output,
        "transpose_discontiguous_tensor_descriptors_override": True,
    }
    if config.two_ctas:
        kwargs["ctas_per_cga"] = (2, 1, 1)
    error = blackwell_bmm_template.maybe_append_choice(
        choices,
        input_nodes=input_nodes,
        layout=layout,
        call_sizes=[batch, m, n],
        num_stages=config.num_stages,
        num_warps=8,
        **kwargs,
    )
    if error is not None:
        raise error

aten_bmm = ExternKernelChoice(torch.bmm, "at::bmm_out", op_overload=aten.bmm.out)
aten_bmm_dtype = ExternKernelChoice(
    torch.bmm,
    "at::_bmm_out_dtype_xpu" if torch.xpu._is_compiled() else "at::_bmm_out_dtype_cuda",
    name="bmm_dtype",
    op_overload=aten.bmm.dtype_out,
)
aten_baddbmm = ExternKernelChoice(
    torch.baddbmm, "at::baddbmm_out", op_overload=aten.baddbmm.out
)

# This path targets vmapped dot products and similar tiny vector contractions,
# where extern bmm launch overhead and lost fusion dominate. Keep the threshold
# conservative so reductions proven to be larger than the threshold continue
# through the normal bmm machinery.
_BMM_DOT_K_DECOMPOSE_THRESHOLD = 32
_BMM_DOT_DECOMPOSE_DTYPES = (
    torch.float16,
    torch.bfloat16,
    torch.float32,
    torch.float64,
)


@L.register_lowering(aten.bmm)
def tuned_bmm(mat1, mat2, out_dtype=None, *, layout=None):
    """
    Lowering for autotuning aten.bmm with different backends (Aten, Triton, CUTLASS, etc.)
    """
    sizevars = V.graph.sizevars
    dtype = mat1.get_dtype()
    device_type = mat1.get_device().type

    def dim_is_one_or_hint(dim):
        # The mul+sum decomposition is valid for any M/N. The size-1 hint is
        # only a profitability signal, so avoid specializing dynamic dims here.
        return sizevars.optimization_hint(dim, fallback=2) == 1

    def dim_is_not_known_gt(dim, threshold):
        # Do not use optimization_hint() for K: the same symbolic FX graph can
        # be code-cache reused across different concrete K values.
        return not sizevars.statically_known_gt(dim, threshold)

    if (
        out_dtype is None
        and device_type in ("cuda", "xpu")
        and device_type == mat2.get_device().type
        and dtype == mat2.get_dtype()
        and dtype in _BMM_DOT_DECOMPOSE_DTYPES
        and dim_is_one_or_hint(mat1.get_size()[1])
        and dim_is_one_or_hint(mat2.get_size()[2])
        and dim_is_not_known_gt(mat1.get_size()[2], _BMM_DOT_K_DECOMPOSE_THRESHOLD)
    ):
        # Preserve dot-shaped bmm as pointwise/reduction IR so surrounding
        # operations can fuse instead of dispatching a tiny extern bmm.
        mat1 = L.unsqueeze(mat1, -1)
        mat2 = L.unsqueeze(mat2, 1)
        return L.sum_(L.mul(mat1, mat2), axis=2)

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

    if use_native_matmul(mat1, mat2):
        mat1 = lowerings[aten.unsqueeze](mat1, -1)
        mat2 = lowerings[aten.unsqueeze](mat2, 1)
        args, kwargs = transform_args(
            args=[mat1, mat2],
            kwargs={},
            broadcast=True,
            type_promotion_kind=None,
            convert_input_to_bool=False,
        )  # Handles broadcasting the arguments

        if inductor_config.triton.codegen_upcast_to_fp32 and mat1.dtype in [
            torch.float16,
            torch.bfloat16,
        ]:

            def _to_dtype(x):
                return ops.to_dtype(x, mat1.dtype, use_compute_types=False)

            args = [make_pointwise(_to_dtype)(x) for x in args]

        mul_pointwise = make_pointwise(ops.dot)(*args)
        dot_reduction = make_reduction("dot")(mul_pointwise, 2)

        return dot_reduction

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
        if mat1.get_device().type not in ("cuda", "xpu"):
            raise AssertionError("out_dtype is only supported for CUDA or XPU")
        aten_handler = aten_bmm_dtype
        aten_extra_kwargs = {"out_dtype": out_dtype}

    choices: list[ChoiceCaller] = []

    # Collect all templates for unified call
    templates_to_use: list[ExternKernelChoice | KernelTemplate] = []
    kwarg_overrides = {}

    if use_aten_gemm_kernels():
        templates_to_use.append(aten_handler)
        kwarg_overrides[aten_handler.uid] = aten_extra_kwargs

    if use_triton_template(layout, check_max_autotune=False):
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
        from ..codegen.cutlass.gemm_template import CUTLASS3xGemmTemplate

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

    if is_nonzero and use_nv_universal_gemm_template(layout, m, n, k, mat1, mat2):
        from ..codegen.nv_universal_gemm import add_nv_universal_gemm_choices

        add_nv_universal_gemm_choices(choices, layout, kernel_inputs)

    node, _ = autotune_select_algorithm(name, choices, kernel_inputs.nodes(), layout)
    return node


@L.register_lowering(aten.baddbmm)
def tuned_baddbmm(inp, mat1, mat2, *, alpha=1, beta=1, layout=None):
    """
    Lowering for autotuning aten.mm with different backends (Aten, Triton, CUTLASS, etc.)
    """
    if use_native_matmul(mat1, mat2):
        if beta == 0:
            arg1 = 0
        else:
            arg1 = lowerings[aten.mul](beta, inp)

        if alpha == 0:
            arg2 = 0
        else:
            arg2 = lowerings[aten.mul](alpha, lowerings[aten.bmm](mat1, mat2))

        return lowerings[aten.add](arg1, arg2)

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
    templates_to_use: list[ExternKernelChoice | KernelTemplate] = []
    if use_aten_gemm_kernels():
        templates_to_use.append(aten_baddbmm)

    if use_triton_template(layout, check_max_autotune=False):
        templates_to_use.append(bmm_template)

    # Single unified call for all templates
    choices.extend(
        V.choices.get_template_configs(kernel_inputs, templates_to_use, name)
    )

    node, _ = autotune_select_algorithm(name, choices, kernel_inputs.nodes(), layout)
    return node
