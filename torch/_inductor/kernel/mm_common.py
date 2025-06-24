# mypy: allow-untyped-defs
import logging
from collections.abc import Sequence
from typing import Any

import sympy

import torch
from torch._inductor.select_algorithm import realize_inputs, SymbolicGridFn
from torch._inductor.utils import sympy_product
from torch._inductor.virtualized import V

from .. import config as inductor_config
from ..codegen.wrapper import PythonWrapperCodegen
from ..ir import _IntLike, Layout, TensorBox
from ..utils import get_num_sms, TMA_DESCRIPTOR_SIZE


log = logging.getLogger(__name__)


@SymbolicGridFn
def mm_grid(m, n, meta, *, cdiv):
    """
    The CUDA grid size for matmul triton templates.
    """
    return (cdiv(m, meta["BLOCK_M"]) * cdiv(n, meta["BLOCK_N"]), 1, 1)


@SymbolicGridFn
def persistent_mm_grid(M: int, N: int, meta: dict[str, Any], *, cdiv, min):
    """Defines the grid for persistent kernels."""
    return (
        min(meta["NUM_SMS"], cdiv(M, meta["BLOCK_M"]) * cdiv(N, meta["BLOCK_N"])),
        1,
        1,
    )


@SymbolicGridFn
def persistent_grouped_mm_grid(*args):
    meta = args[-1]
    return (meta["NUM_SMS"], 1, 1)


def acc_type(dtype):
    if dtype in (torch.float16, torch.bfloat16):
        return "tl.float32"
    return f"tl.{dtype}".replace("torch.", "")


def mm_options(config, sym_m, sym_n, sym_k, layout):
    """
    Common options to matmul triton templates.
    """
    even_k_symbolic = (
        # it isn't worth guarding on this
        sympy.gcd(sym_k, config.kwargs["BLOCK_K"]) == config.kwargs["BLOCK_K"]
    )
    allow_tf32 = torch.backends.cuda.matmul.allow_tf32 and (
        not inductor_config.force_same_precision
        or ((sym_m % 16) == 0 and (sym_n % 16) == 0 and (sym_k % 8) == 0)
    )
    options_dict = dict(
        EVEN_K=even_k_symbolic,
        ALLOW_TF32=allow_tf32,
        USE_FAST_ACCUM=False,  # Option for _scaled_mm
        ACC_TYPE=acc_type(layout.dtype),
        num_stages=config.num_stages,
        num_warps=config.num_warps,
        **config.kwargs,
    )

    # If GROUP_M not specified then default to 8
    if "GROUP_M" not in config.kwargs:
        group_m = config.kwargs.get("GROUP_M", 8)
        options_dict["GROUP_M"] = group_m

    return options_dict


def tma_options() -> dict[str, Any]:
    from torch.utils._triton import has_triton_stable_tma_api

    return {"TMA_EXPERIMENTAL_API": not has_triton_stable_tma_api()}


def persistent_mm_options(mat1, mat2):
    res = dict(
        A_ROW_MAJOR=not mat1.layout.is_transposed(),
        B_ROW_MAJOR=not mat2.layout.is_transposed(),
        NUM_SMS=get_num_sms(),
        TMA_SIZE=TMA_DESCRIPTOR_SIZE,
    )
    res.update(tma_options())
    return res


def scaled_mm_options(  # type: ignore[no-untyped-def]
    config,  # triton.Config
    sym_m: sympy.core.numbers.Integer,
    sym_n: sympy.core.numbers.Integer,
    sym_k: sympy.core.numbers.Integer,
    layout: Layout,
    scale_a,
    scale_b,
    use_fast_accum: bool,
    device_tma: bool = False,
) -> dict[str, Any]:
    def are_compatible_scales(size_a, size_b) -> bool:
        # Same sized scales are compatible
        if len(size_a) == len(size_b):
            return True

        # Both need to be scalars or len(1) tensors
        if len(size_a) <= 1 and len(size_b) <= 1:
            return True

        return False

    size_a, size_b = scale_a.get_size(), scale_b.get_size()
    assert are_compatible_scales(size_a, size_b), (
        "Expect scale_a and scale_b to be either both scalars (including single-element tensors) "
        f"or 1-dimensional tensors with the same size. Got scale_a: {len(size_a)} and scale_b: {len(size_b)}."
    )

    mm_template_options = mm_options(config, sym_m, sym_n, sym_k, layout)

    mm_template_options["ACC_TYPE"] = "tl.float32"
    mm_template_options["USE_FAST_ACCUM"] = use_fast_accum
    mm_template_options["SCALING_ROWWISE"] = len(size_a) == 2

    if device_tma:
        mm_template_options["TMA_SIZE"] = TMA_DESCRIPTOR_SIZE
        mm_template_options["NUM_SMS"] = get_num_sms()

    mm_template_options.update(tma_options())

    return mm_template_options


def mm_args(
    mat1,
    mat2,
    *others,
    layout=None,
    out_dtype=None,
    use_4x2_dim=False,
    mat2_transposed=False,
):
    """
    Common arg processing for mm,bmm,addmm,etc
    """
    mat1, mat2 = realize_inputs(mat1, mat2)
    *b1, m, k1 = mat1.get_size()
    if mat2_transposed:
        *b2, n, k2 = mat2.get_size()
    else:
        *b2, k2, n = mat2.get_size()
    b = [V.graph.sizevars.guard_equals(a, b) for a, b in zip(b1, b2)]
    if use_4x2_dim:
        k2 = k2 * 2
    k = V.graph.sizevars.guard_equals(k1, k2)
    if layout is None:
        from torch._inductor.ir import FixedLayout

        if out_dtype is None:
            out_dtype = mat1.get_dtype()

        layout = FixedLayout(
            mat1.get_device(),
            out_dtype,
            [*b, m, n],
        )
    else:
        assert out_dtype is None, "out_dtype is ignored if layout is specified."
    from ..lowering import expand

    others = [realize_inputs(expand(x, layout.size)) for x in others]

    return [m, n, k, layout, mat1, mat2, *others]


def mm_config_kwargs(device, exclude_condition, dtype_size=None):
    if device == "cpu":
        return {
            "scale": 0.5,
            "exclude": exclude_condition,
        }

    if dtype_size and inductor_config.max_autotune_gemm_search_space == "EXHAUSTIVE":
        return {
            "dtype_size": dtype_size,
        }
    return {}


def addmm_epilogue(dtype, alpha, beta):
    def epilogue(acc, bias):
        if alpha != 1:
            acc = V.ops.mul(acc, V.ops.constant(alpha, dtype))
        if beta != 1:
            bias = V.ops.mul(bias, V.ops.constant(beta, dtype))
        return V.ops.add(acc, bias)

    return epilogue


def scale_mm_epilogue():
    """
    Create an epilogue function that applies scaling to matrix multiplication result
    using the given scale factors.

    Args:
        dtype: The data type of the output
        scale_a: Scale factor for matrix A
        scale_b: Scale factor for matrix B

    Returns:
        Epilogue function that takes the accumulator and applies scaling
    """

    def epilogue(acc, inv_a_scale, inv_b_scale, bias=None):
        # The epilogue function receives the accumulator (result of mat1 @ mat2)
        # and applies the scaling factors
        # In the original scaled_mm, we use inverse scales, so we multiply by them
        mul_scales = V.ops.mul(inv_a_scale, inv_b_scale)
        mul_acc = V.ops.mul(acc, mul_scales)
        if bias is not None:
            return V.ops.add(mul_acc, bias)
        else:
            return mul_acc

    return epilogue


def _is_static_problem(layout: Layout) -> tuple[bool, bool]:
    """
    Check if input tensors and output layout have static shapes and non-zero sizes.

    Args:
        layout: Output layout object with a 'size' attribute.

    Returns:
        Tuple[bool, bool]: (is_static, is_nonzero)
            is_static: True if all shapes are statically known
            is_nonzero: True if all dimensions are non-zero
    """
    static_shape = True
    static_size = PythonWrapperCodegen.statically_known_list_of_ints_or_none(
        layout.size
    )
    if static_size is None:
        nonzero = True
        for s in layout.size:
            sz = PythonWrapperCodegen.statically_known_int_or_none(s)
            if sz is not None and sz == 0:
                nonzero = False
                break
        return False, nonzero
    numel = 1
    for dim in static_size:
        numel *= dim
    nonzero = numel > 0
    return static_shape, nonzero


def check_supported_striding(mat_a: TensorBox, mat_b: TensorBox) -> None:
    def is_row_major(stride: Sequence[_IntLike]) -> bool:
        return stride[-1] == 1

    def is_col_major(stride: Sequence[_IntLike]) -> bool:
        return stride[-2] == 1

    def has_zero_dim(size: Sequence[_IntLike]) -> bool:
        return bool(size[0] == 0 or size[1] == 0)

    # Check mat_a (self) stride requirements
    torch._check(
        is_row_major(mat_a.get_stride()) or has_zero_dim(mat_a.get_size()),
        lambda: f"mat_a must be row_major, got stride {mat_a.get_stride()}",
    )

    # Check mat_b stride requirements
    torch._check(
        is_col_major(mat_b.get_stride()) or has_zero_dim(mat_b.get_size()),
        lambda: f"mat_b must be col_major, got stride {mat_b.get_stride()}",
    )


def is_batch_stride_largest(mat1, mat2, layout) -> bool:
    """
    Checking if the batch stride is the largest in the stride.
    """
    sizes = [mat1.get_size(), mat2.get_size(), layout.size]
    strides = [mat1.get_stride(), mat2.get_stride(), layout.stride]
    for size, stride in zip(sizes, strides):
        assert len(size) == len(stride) == 3, "Expect 3D tensors"
        if stride[0] != sympy_product(size[1:]):
            return False

    return True
