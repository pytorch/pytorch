# mypy: allow-untyped-defs
import contextlib
import dataclasses
import functools
import itertools
import math
import operator
import re
import sys
import warnings
from collections.abc import Callable, Sequence
from enum import Enum
from typing import Any, cast, Optional, Union

import sympy

import torch
import torch.fx
from torch._inductor import dependencies
from torch._prims_common import is_float_dtype, is_integer_dtype
from torch.utils._ordered_set import OrderedSet
from torch.utils._sympy.functions import CeilDiv, FloorDiv, ModularIndexing
from torch.utils._sympy.symbol import free_symbol_is_type, symbol_is_type, SymT

from ..._dynamo.utils import counters
from .. import config, cpp_builder, cpu_vec_isa, ir, metrics
from ..debug import set_kernel_post_grad_provenance_tracing
from ..loop_body import LoopBody
from ..scheduler import (
    BaseSchedulerNode,
    BaseScheduling,
    ExternKernelSchedulerNode,
    ForeachKernelSchedulerNode,
    FusedSchedulerNode,
    Scheduler,
    SchedulerNode,
)
from ..utils import (
    cache_on_self,
    get_bounds_index_expr,
    get_fused_kernel_name,
    has_free_symbols,
    is_multi_outputs_template,
    is_welford_reduction,
    parallel_num_threads,
    Placeholder,
    sympy_index_symbol,
    sympy_index_symbol_with_prefix,
    sympy_product,
    sympy_subs,
)
from ..virtualized import NullKernelHandler, ops, OpsValue, V
from .common import (
    BackendFeature,
    BracesBuffer,
    CSE,
    CSEVariable,
    DataTypePropagation,
    DeferredLine,
    DTYPE_TO_COMPUTATION_DTYPE,
    IndentedBuffer,
    Kernel,
    KernelArgs,
    OpOverrides,
    OptimizationContext,
)
from .cpp_utils import (
    _get_dtype_from_loopbodies,
    _get_loop_body,
    cexpr,
    cexpr_index,
    codegen_rand,
    CppCSEVariable,
    DTYPE_TO_CPP,
    get_promote_dtype,
    INDEX_TYPE,
    LocalBufferContext,
    may_unify_binary_op_mask_type,
    promote_args,
    template_fusion_with_epilogues_supported,
    unify_mask_base_type,
    value_to_cpp,
)


_IS_WINDOWS = sys.platform == "win32"


@functools.cache
def get_export_declaration():
    return "__declspec(dllexport)" if _IS_WINDOWS else ""


schedule_log = torch._logging.getArtifactLogger(__name__, "schedule")

NATIVE_OMP_RTYPES = OrderedSet(["+", "*", "^", "||", "min", "max"])
RTYPE_TO_CPP = {
    "sum": "+",
    "prod": "*",
    "xor_sum": "^",
    "min": "min",
    "max": "max",
    "argmin": "argmin",
    "argmax": "argmax",
    "any": "||",
    "welford_reduce": "welford",
    "welford_combine": "welford",
}
VECTORIZABLE_RTYPES = OrderedSet(
    [
        "max",
        "min",
        "sum",
        "prod",
        "xor_sum",
        "welford_reduce",
        "welford_combine",
        "argmin",
        "argmax",
        "any",
    ]
)

PYTHON_TO_CPP = {
    "Tensor": "at::Tensor",
    "int": "long",
    "float": "double",
    "bool": "bool",
    "str": "std::string",
    "ScalarType": "c10::ScalarType",
    "MemoryFormat": "at::MemoryFormat",
    "Layout": "at::Layout",
    "Device": "at::Device",
    "number": "at::Scalar",
}

CONTAINER_PYTHON_TO_CPP = {
    "List": "std::vector",
    "Optional": "std::optional",
}

DTYPE_LOWP_FP = [
    torch.bfloat16,
    torch.float16,
]

VECTORIZABLE_DTYPES: list[torch.dtype] = [
    torch.float64,
    torch.float,
    torch.bfloat16,
    torch.float16,
    torch.bool,
    torch.uint8,
    torch.int8,
    torch.int32,
    torch.int64,
    torch.float8_e4m3fn,
    torch.float8_e5m2,
]


def reduction_init(reduction_type, dtype):
    if dtype in DTYPE_LOWP_FP:
        # Since load promotes all half-precision inputs to float, the initial
        # constant for reduction must be promoted as well
        dtype = torch.float32
    if reduction_type in ("xor_sum", "sum", "any"):
        return 0
    if reduction_type == "prod":
        return 1
    if reduction_type in ("max", "argmax", "min", "argmin"):
        cdtype = DTYPE_TO_CPP[dtype]
        if dtype == torch.bool and reduction_type in ("argmin", "argmax"):
            cdtype = DTYPE_TO_CPP[torch.float]
        min_var = (
            f"-std::numeric_limits<{cdtype}>::infinity()"
            if is_float_dtype(dtype)
            else f"std::numeric_limits<{cdtype}>::min()"
        )
        max_var = (
            f"std::numeric_limits<{cdtype}>::infinity()"
            if is_float_dtype(dtype)
            else f"std::numeric_limits<{cdtype}>::max()"
        )
        init_var = min_var if reduction_type in ("max", "argmax") else max_var
        return (
            init_var
            if reduction_type in ("max", "min")
            else f"IndexValue<{cdtype}>{{0, {init_var}}}"
        )
    if is_welford_reduction(reduction_type):
        return f"Welford<{DTYPE_TO_CPP[dtype]}>()"
    raise AssertionError(reduction_type)


def reduction_acc_type(reduction_type, dtype):
    scalar_type = DTYPE_TO_CPP[DTYPE_TO_COMPUTATION_DTYPE[dtype]]
    if is_welford_reduction(reduction_type):
        return f"Welford<{scalar_type}>"
    if reduction_type in ("argmin", "argmax"):
        if dtype == torch.bool:
            scalar_type = DTYPE_TO_CPP[torch.float]
        return f"IndexValue<{scalar_type}>"
    return scalar_type


def reduction_combine(
    reduction_type,
    var,
    next_value,
    helper_val=None,
    index: Optional[sympy.Symbol] = None,
    src_dtype=None,
):
    is_bool = src_dtype == torch.bool
    if reduction_type == "sum":
        if helper_val:
            return f"cascade_sum_combine({next_value}, &{helper_val})"
        else:
            conjunction = "|" if is_bool else "+"
            return f"{var} {conjunction} {next_value}"
    if reduction_type == "prod":
        return f"{var} * {next_value}"
    if reduction_type == "xor_sum":
        return f"{var} ^ {next_value}"
    if reduction_type == "any":
        return f"{var} || {next_value}"
    if reduction_type in ("min", "max"):
        return f"{reduction_type}_propagate_nan({var}, {next_value})"
    if reduction_type == "welford_reduce":
        if helper_val:
            return f"welford_combine({var}, {next_value}, &{helper_val})"
        else:
            return f"welford_combine({var}, {next_value})"
    if reduction_type == "welford_combine":
        if isinstance(next_value, tuple):
            mean, m2, weight = next_value
        else:
            mean, m2, weight = reduction_project(reduction_type, next_value)
        return f"welford_combine({var}, {{{mean}, {m2}, {weight}}})"
    if reduction_type in ("argmin", "argmax"):
        if (
            hasattr(next_value, "dtype")
            and next_value.dtype == torch.bool
            and not next_value.is_vec
        ):
            if index is not None:
                return f"{reduction_type}_combine({var}, static_cast<float>({next_value}), {index})"
            else:
                return (
                    f"{reduction_type}_combine({var}, static_cast<float>({next_value}))"
                )
        if index is not None:
            return f"{reduction_type}_combine({var}, {next_value}, {index})"
        else:
            return f"{reduction_type}_combine({var}, {next_value})"
    raise AssertionError(reduction_type)


def reduction_project(reduction_type, acc):
    if is_welford_reduction(reduction_type):
        return f"{acc}.mean", f"{acc}.m2", f"{acc}.weight"
    elif reduction_type in ("argmin", "argmax"):
        return f"{acc}.index"
    return acc


def move_code_under_inner_loop(
    code: IndentedBuffer,
    iter_var: sympy.Expr,
    new_iter_var: str,
    loop_start: sympy.Expr,
    loop_end: sympy.Expr,
) -> BracesBuffer:
    r"""
    f(iter_var) is transformed to f(new_iter_var) under the inner loop
      \/
    for (new_iter_var = loop_start; new_iter_var < loop_end; new_iter_var++) {
        f(new_iter_var)
    }
    Please be careful while using this function,
    as the variable defined in f(iter_var) will be invalid outside the for loop.
    For example:
    auto tmp0 = in_ptr[x0]; ->
    for (new_x0 = start; new_x0 < end; new_x0++){
        auto tmp0 = in_ptr[new_x0];
    }
    The tmp0 is invalid outside the loop.
    """
    transformed_code = BracesBuffer()
    with contextlib.ExitStack() as stack:
        transformed_code.writeline(
            f"for ({INDEX_TYPE} {new_iter_var} = {cexpr_index(loop_start)};"
            + f"{new_iter_var} < {cexpr_index(loop_end)}; {new_iter_var}++)"
        )
        stack.enter_context(transformed_code.indent())
        for _, line in enumerate(code._lines):
            assert isinstance(
                line,
                (
                    str,
                    DeferredLine,
                ),
            )
            deferred_name = None
            if isinstance(line, DeferredLine):
                deferred_name = line.name
                line = line.line
            new_line = re.sub(r"\b" + f"{iter_var}" + r"\b", f"{new_iter_var}", line)
            if deferred_name:
                new_line = DeferredLine(deferred_name, new_line)  # type: ignore[assignment]
            transformed_code.writeline(new_line)
    return transformed_code


def reduction_prefix_array(
    acc_var: Union[str, CSEVariable],
    acc_type: str,
    reduction_type: str,
    dtype: torch.dtype,
    len: Union[str, int],
    init_fn,
):
    """
    MSVC don't support dynamic array(VLA). So we use std::unique_ptr here.
    Ref: https://stackoverflow.com/questions/56555406/creating-dynamic-sized-array-using-msvc-c-compiler
    MSVC is the only one compiler without VLA. support. Since MSVC can't get good performance here.
    We just use unique_ptr make it works on MSVC.
    For other compilers, we continue to use VLA to get best performance.
    """
    code_buffer = IndentedBuffer()
    acc_decl = (
        f"auto {acc_var}_arr = std::make_unique<{acc_type}[]>({len});"
        if cpp_builder.is_msvc_cl()
        else f"{acc_type} {acc_var}_arr[{len}];"
    )
    code_buffer.writeline(f"{acc_decl}")
    code_buffer.writelines(
        [
            f"for (int i = 0; i < {len}; i++)",
            "{",
            f"    {acc_var}_arr[i] = {init_fn(reduction_type, dtype)};",
            "}",
        ],
    )
    return code_buffer


def replace_acc_name(buffer: IndentedBuffer, name: str, new_name: str):
    for i, line in enumerate(buffer._lines):
        assert isinstance(
            line,
            (
                str,
                DeferredLine,
            ),
        )
        if isinstance(line, DeferredLine):
            line.line = re.sub(r"\b" + f"{name}" + r"\b", f"{new_name}", line.line)
        else:
            buffer._lines[i] = re.sub(r"\b" + f"{name}" + r"\b", f"{new_name}", line)


def replace_cascade_sum_with_add(buffer: IndentedBuffer):
    """
    Replaces `acc = cascade_sum_combine(value, ...)` with `acc = acc + value;`
    """

    pattern = r"(.*?)\s*=\s*cascade_sum_combine\(([^,]+),.*?\);"
    for i, line in enumerate(buffer._lines):
        assert isinstance(
            line,
            (
                str,
                DeferredLine,
            ),
        )
        content = line.line if isinstance(line, DeferredLine) else line
        match = re.search(pattern, content)
        if match:
            acc, value = match.groups()
            new_content = re.sub(pattern, f"{acc} = {acc} + {value};", content)
            if isinstance(line, DeferredLine):
                line.line = new_content
            else:
                buffer._lines[i] = new_content


@functools.lru_cache
def stride_at(index: sympy.Expr, var: sympy.Symbol):
    if not index.has(var):
        # see test_torchinductor_dynamic_shapes.py::test_full_boolean_dynamic_shapes_cpu
        # which has tmp0 = ops.index_expr(s0 >= 1024, torch.bool) and fails below calculation.
        # in this case, there is no dependencies between index and var.
        return sympy.S.Zero
    replacement = {var: var + 1}
    new_index = sympy_subs(index, replacement)  # type: ignore[arg-type]
    return sympy.simplify(new_index - index)


@functools.lru_cache
def simplify_index_in_vec_range(index: sympy.Expr, var: sympy.Expr, vec_length: int):
    """
    Simplifies the index expression within the range of a vectorized loop.
    Given a vectorized loop variable `var` in the range of a loop with `vec_length`,
    this function transforms the `index` into an equivalent form. It handles
    simplifications for cases where `var` can be expressed as `vec_length * a + b`,
    where `b` ranges from 0 to `vec_length - 1`. The function reduces occurrences
    of `FloorDiv` and `ModularIndexing` in the `index` with best-effort optimizations.

    NOTE:
    The simplified index expression is intended for analysis purposes only, not
    for code generation. It replaces `FloorDiv` and `ModularIndexing` with free variables
    which are not dependent on the loop variable `var` in the vectorized range. Check
    https://github.com/pytorch/pytorch/pull/117221#discussion_r1449746217 for more details.

    Examples:
    1. If `var` is `x3` and `vec_length` is 16, and `x3 = 16*a + b`, then
       `FloorDiv(x3, div)` or `ModularIndexing(x3, div, mod)` becomes a free variable
       when `div` is divisible by 16.
    2. `ModularIndexing(x3, 1, mod)` can be simplified to `x3 + c` where `c` is a free
       variable when `mod` is divisible by 16.
    """

    div_freevar_id = 0
    mod_freevar_id = 0

    def visit_indexing_div(divisor):
        nonlocal div_freevar_id
        result = FloorDiv(var, divisor)
        if sympy.gcd(divisor, vec_length) == vec_length:
            result = sympy.Symbol(f"{var}_div_c{div_freevar_id}")
            div_freevar_id += 1
        return result

    def visit_modular_indexing(divisor, modulus):
        nonlocal mod_freevar_id
        result = ModularIndexing(var, divisor, modulus)
        if sympy.gcd(divisor, vec_length) == vec_length:
            result = sympy.Symbol(f"{var}_mod_c{mod_freevar_id}")
            mod_freevar_id += 1
        elif divisor == 1 and sympy.gcd(modulus, vec_length) == vec_length:
            result = var + sympy.Symbol(f"{var}_mod_c{mod_freevar_id}")
            mod_freevar_id += 1
        return result

    original_index = index

    div = sympy.Wild("divisor", integer=True)
    if index.has(FloorDiv):
        index = index.replace(FloorDiv(var, div), visit_indexing_div)

    mod = sympy.Wild("modulus", integer=True)
    if index.has(ModularIndexing):
        index = index.replace(ModularIndexing(var, div, mod), visit_modular_indexing)

    index = sympy.simplify(index)
    if index != original_index:
        return simplify_index_in_vec_range(index, var, vec_length)

    return index


@functools.lru_cache
def stride_at_vec_range(
    index: sympy.Expr, var: sympy.Symbol, vec_length: Optional[int] = None
):
    if vec_length:
        index = simplify_index_in_vec_range(index, var, vec_length)
    return stride_at(index, var)


@dataclasses.dataclass
class ParallelDepth:
    """
    A class representing parallel depth.
    Includes the starting depth of parallelism and the depth of parallelism.
    """

    parallel_depth: int
    start_depth: int


class OuterLoopFusedSchedulerNode(FusedSchedulerNode):
    @classmethod
    def fuse(  # type: ignore[override]
        cls, node1: BaseSchedulerNode, node2: BaseSchedulerNode, outer_loop_fusion_depth
    ):
        assert node1.scheduler is node2.scheduler
        assert all(
            type(node)
            in (
                OuterLoopFusedSchedulerNode,
                SchedulerNode,
                FusedSchedulerNode,
            )
            for node in (node1, node2)
        )
        if any(type(node) is OuterLoopFusedSchedulerNode for node in (node1, node2)):
            return cls(
                node1.scheduler,
                # pyrefly: ignore [bad-argument-type]
                (
                    list(node1.get_outer_nodes())
                    if type(node1) is OuterLoopFusedSchedulerNode
                    else [
                        node1,
                    ]
                )
                + (
                    list(node2.get_outer_nodes())
                    if type(node2) is OuterLoopFusedSchedulerNode
                    else [
                        node2,
                    ]
                ),
                outer_loop_fusion_depth,
            )
        else:
            return cls(node1.scheduler, [node1, node2], outer_loop_fusion_depth)  # type: ignore[list-item]

    def __init__(
        self,
        scheduler: "Scheduler",
        outer_fused_nodes: list[Union[FusedSchedulerNode, SchedulerNode]],
        outer_loop_fusion_depth,
    ):
        self.outer_fused_nodes: list[Union[FusedSchedulerNode, SchedulerNode]] = (
            outer_fused_nodes
        )
        self.outer_loop_fusion_depth = outer_loop_fusion_depth
        flatten_snodes = []
        for _node in self.outer_fused_nodes:
            assert isinstance(_node, (SchedulerNode, FusedSchedulerNode))
            flatten_snodes.extend(list(_node.get_nodes()))
        super().__init__(scheduler, flatten_snodes)  # type: ignore[arg-type]

    def get_outer_nodes(self):
        return self.outer_fused_nodes

    def check_outer_fusion_loop_level_attr(
        self, cpp_kernel_proxy_list, outer_loop_fusion_depth
    ):
        # This function ensures that the same tiling split is applied at each loop level within the outer loop fusion depth.
        # In the fusion stage, we only examine nodes with same vars and reduce.
        # However, for nodes with same vars and reduce, the loops may still have different tile splits.
        # For example (test_expr_vec_non_contiguous in test_cpu_repro.py):
        #   * buf0 tiling along the 2nd loop level, buf1 tiling along the 3rd loop level.
        # If the check failed, we should fall back to standard loop codegen.
        def _inner(
            left_loop_nest: LoopNest,
            right_loop_nest: LoopNest,
            loop_fusion_depth: int,
            current_checking_depth: int,
        ) -> bool:
            assert left_loop_nest.loops
            assert right_loop_nest.loops
            left_loop_level = left_loop_nest.loops[current_checking_depth]
            right_loop_level = right_loop_nest.loops[current_checking_depth]
            # Check if same loop level attr
            outer_loops_attr_compare_list = [
                "var",
                "size",
                "offset",
                "steps",
            ]
            if not (
                all(
                    getattr(left_loop_level, attr_compare)
                    == getattr(right_loop_level, attr_compare)
                    for attr_compare in outer_loops_attr_compare_list
                )
            ):
                return False

            assert loop_fusion_depth >= 1
            if (loop_fusion_depth := loop_fusion_depth - 1) > 0:
                # Check next loop level attr
                current_checking_depth = current_checking_depth + 1
                assert current_checking_depth < len(left_loop_nest.loops)
                assert current_checking_depth < len(right_loop_nest.loops)
                if not _inner(
                    left_loop_nest,
                    right_loop_nest,
                    loop_fusion_depth,
                    current_checking_depth,
                ):
                    return False

            return True

        for idx in range(len(cpp_kernel_proxy_list) - 1):
            left_loop_nest = cpp_kernel_proxy_list[idx].loop_nest
            right_loop_nest = cpp_kernel_proxy_list[idx + 1].loop_nest
            if not _inner(
                left_loop_nest,
                right_loop_nest,
                outer_loop_fusion_depth,
                0,
            ):
                return False

        for cpp_kernel_proxy in cpp_kernel_proxy_list:
            outer_ranges = functools.reduce(
                operator.mul,
                cpp_kernel_proxy.ranges[:outer_loop_fusion_depth],
            )
            # When the range of the first inner loop is much larger than the range of
            # all outer loops, do not fuse outer loop and fallback to standard loop codegen,
            # so that the inner loops with larger range have a chance to be parallelized.
            # We set a conservative threshold here:
            # First inner loop range / all outer loops range > 300.
            if (
                len(cpp_kernel_proxy.ranges) > outer_loop_fusion_depth
                and isinstance(outer_ranges, sympy.Integer)
                and isinstance(
                    cpp_kernel_proxy.ranges[outer_loop_fusion_depth],
                    sympy.Integer,
                )
                and outer_ranges * 300
                < cpp_kernel_proxy.ranges[outer_loop_fusion_depth]
            ):
                return False

        return True

    def merge_outer_fusion_kernels(
        self,
        cpp_kernel_proxy_list,
    ):
        kernel_group = cpp_kernel_proxy_list[0].kernel_group
        outer_loop_fused_kernel = OuterLoopFusedKernel(kernel_group)
        outer_loop_fused_kernel.inner = [
            proxy.loop_nest.from_loop_level(self.outer_loop_fusion_depth)
            for proxy in cpp_kernel_proxy_list
        ]
        outer_fused_proxy = cpp_kernel_proxy_list[0]
        outer_fused_proxy.loop_nest.kernel = outer_loop_fused_kernel
        outer_fused_proxy.loop_nest.loops = outer_fused_proxy.loop_nest.loops[
            : self.outer_loop_fusion_depth
        ]
        return outer_fused_proxy


class RecordOptimizationContext:
    def __init__(self, func_name: str = ""):
        self.func_name = func_name
        self.current_node: Optional[torch.fx.Node] = None
        self.opt_ctx: Optional[OptimizationContext] = None

    def __enter__(self):
        assert V.interpreter
        assert V.interpreter.current_node

        self.current_node = V.interpreter.current_node
        assert self.current_node is not None
        if OptimizationContext.key in self.current_node.meta:
            self.opt_ctx = self.current_node.meta[OptimizationContext.key]
        else:
            self.opt_ctx = OptimizationContext()
        assert self.opt_ctx is not None
        self.opt_ctx.ops_name = self.func_name
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        assert self.current_node
        assert self.opt_ctx
        self.current_node.meta[OptimizationContext.key] = self.opt_ctx

    def get_opt_ctx(self):
        return self.opt_ctx

    def get_fx_node(self):
        assert self.current_node
        return self.current_node


def decltype_promoted(*args):
    assert not any(isinstance(arg, CppCSEVariable) and arg.is_vec for arg in args), (
        "Promotion of vector types is not supported"
    )

    if (dt := get_promote_dtype(args)) is not None:
        return DTYPE_TO_CPP[dt]
    else:
        return f"decltype({args[0]})"


class CppOverrides(OpOverrides):
    """Map element-wise ops to C++"""

    @staticmethod
    def add(a, b):
        return f"{decltype_promoted(a, b)}({a} + {b})"

    @staticmethod
    def sub(a, b):
        return f"{decltype_promoted(a, b)}({a} - {b})"

    @staticmethod
    def mul(a, b):
        return f"{decltype_promoted(a, b)}({a} * {b})"

    @staticmethod
    def to_dtype(x, dtype, src_dtype=None, use_compute_types=True):
        assert isinstance(x, CppCSEVariable)
        if src_dtype is None:
            src_dtype = x.dtype
        expr = V.kernel.get_to_dtype_expr(x, dtype, src_dtype)
        csevar = V.kernel.cse.generate(V.kernel.compute, expr)
        csevar.update_on_args("to_dtype", (x, dtype), {"src_dtype": src_dtype})
        if dtype in DTYPE_LOWP_FP and src_dtype == torch.float:
            """
            https://github.com/pytorch/pytorch/issues/115260
            For FusedSchedulerNode[node1, node2], the node2 loads what node1 stores and the buffer is
            in low-precision floating point data type. When the output of node1 also serves as the output of the
            kernel, the result of nodes would be different from the case when output of node1 is not the output
            of the kernel (where we don't need to insert `to_dtype` for legalization). To address the problem, on
            storing the lowp node1 output, we also add the inverse dtype conversion to high precision data type
            to the cse cache.

            Example (pseudo code):
                node1_output = ...
                node1_output_lowp = to_dtype(node1_output, dtype=torch.bfloat16)
                store(buf, node1_output_lowp)
                node2_input_lowp = load(buf)
                node2_input = to_dtype(node2_input_lowp, dtype=torch.float)

            Without cse cache trick:
                node1_output = ...
                node1_output_lowp = to_dtype(node1_output, dtype=torch.bfloat16)
                store(buf, node1_output_lowp)
                node2_input_lowp = node_output_lowp # hit store cache
                node2_input = to_dtype(node2_input_lowp, dtype=torch.float)

            With cse cache trick:
                node1_output = ...
                node1_output_lowp = to_dtype(node1_output, dtype=torch.bfloat16)
                # also add `to_dtype(node1_input_lowp, dtype=torch.float)` -> `node1_output` to cse cache
                store(buf, node1_output_lowp)
                node2_input_lowp = node_output_lowp # hit store cache
                node2_input = node1_output # hit cse cache
            """
            V.kernel.cache_dtype_convert(x, src_dtype, csevar, dtype)
        return csevar

    @staticmethod
    def to_dtype_bitcast(x, dtype, src_dtype):
        assert dtype in DTYPE_TO_CPP, f"{dtype} missing from {__name__}.DTYPE_TO_CPP"
        return f"c10::bit_cast<{DTYPE_TO_CPP[dtype]}>({x})"

    @staticmethod
    def abs(x):
        return f"std::abs({x})"

    @staticmethod
    def sin(x):
        return f"std::sin({x})"

    @staticmethod
    def cos(x):
        return f"std::cos({x})"

    @staticmethod
    def neg(x):
        return f"decltype({x})(-{x})"

    @staticmethod
    def exp(x):
        # return f"Sleef_expf_u10({x})"
        return f"std::exp({x})"

    @staticmethod
    def exp2(x):
        return f"std::exp2({x})"

    @staticmethod
    def expm1(x):
        return f"std::expm1({x})"

    @staticmethod
    def erf(x):
        return f"std::erf({x})"

    @staticmethod
    def erfc(x):
        return f"std::erfc({x})"

    @staticmethod
    def erfinv(x):
        return f"calc_erfinv({x})"

    @staticmethod
    def sqrt(x):
        return f"std::sqrt({x})"

    @staticmethod
    def rsqrt(x):
        return f"1 / std::sqrt({x})"

    @staticmethod
    def log1p(x):
        bug = config.cpp.inject_log1p_bug_TESTING_ONLY
        if bug == "accuracy":
            return f"{x} + decltype({x})(1)"
        elif bug is None:
            return f"std::log1p({x})"
        else:
            raise AssertionError(
                f"unrecognized config cpp.inject_log1p_bug_TESTING_ONLY = {bug!r}"
            )

    @staticmethod
    def tan(x):
        return f"std::tan({x})"

    @staticmethod
    def tanh(x):
        return f"std::tanh({x})"

    @staticmethod
    def signbit(x):
        """
        On windows std::signbit only support float type.
        Ref: https://learn.microsoft.com/en-us/cpp/c-runtime-library/reference/signbit?view=msvc-170
        """
        return (
            f"std::signbit(static_cast<float>({x}))"
            if _IS_WINDOWS
            else f"std::signbit({x})"
        )

    @staticmethod
    def pow(a, b):
        return f"std::pow({a}, {b})"

    @staticmethod
    def log(x):
        return f"std::log({x})"

    @staticmethod
    def round(x):
        return f"std::nearbyint({x})"

    @staticmethod
    def floor(x):
        return f"std::floor({x})"

    @staticmethod
    def floordiv(a, b):
        # a and b are integer type
        quot = f"{a} / {b}"
        rem = f"{a} % {b}"
        return f"(({a} < 0) != ({b} < 0) ? ({rem} != 0 ? {quot} - 1 : {quot}) : {quot})"

    @staticmethod
    def ceil(x):
        return f"std::ceil({x})"

    @staticmethod
    def trunc(x):
        return f"std::trunc({x})"

    @staticmethod
    def truncdiv(a, b):
        # a and b are integer type
        return f"{a} / {b}"

    @staticmethod
    def fmod(a, b):
        return f"std::fmod({a}, {b})"

    @staticmethod
    def isinf(x):
        return f"std::isinf({x})"

    @staticmethod
    def isnan(x):
        return f"std::isnan({x})"

    @staticmethod
    def lgamma(x):
        return f"std::lgamma({x})"

    @staticmethod
    def acos(x):
        return f"std::acos({x})"

    @staticmethod
    def acosh(x):
        return f"std::acosh({x})"

    @staticmethod
    def cosh(x):
        return f"std::cosh({x})"

    @staticmethod
    def sinh(x):
        return f"std::sinh({x})"

    @staticmethod
    def asin(x):
        return f"std::asin({x})"

    @staticmethod
    def asinh(x):
        return f"std::asinh({x})"

    @staticmethod
    def atan2(x, y):
        return f"std::atan2({x}, {y})"

    @staticmethod
    def atan(x):
        return f"std::atan({x})"

    @staticmethod
    def atanh(x):
        return f"std::atanh({x})"

    @staticmethod
    def copysign(x, y):
        return f"std::copysign({x}, {y})"

    @staticmethod
    def frexp(x):
        cache_keys = f"frexp({x})[0]", f"frexp({x})[1]"
        if all(V.kernel.cse.try_get(cache_key) is not None for cache_key in cache_keys):
            return tuple(V.kernel.cse.try_get(cache_key) for cache_key in cache_keys)

        code = BracesBuffer()
        exponent = V.kernel.cse.newvar(dtype=torch.int32, shape=x.shape)
        mantissa = V.kernel.cse.newvar(dtype=x.dtype, shape=x.shape)
        code.writeline(f"int32_t {exponent};")
        code.writeline(f"auto {mantissa} = std::frexp({x}, &{exponent});")
        V.kernel.compute.splice(code)
        cse_vars = (mantissa, exponent)
        for cache_key, cse_var in zip(cache_keys, cse_vars):
            V.kernel.cse.put(cache_key, cse_var)
        return mantissa, exponent

    @staticmethod
    def hypot(x, y):
        return f"std::hypot({x}, {y})"

    @staticmethod
    def log10(x):
        return f"std::log10({x})"

    @staticmethod
    def log2(x):
        return f"std::log2({x})"

    @staticmethod
    def ldexp(x, n):
        return f"std::ldexp({x}, {n})"

    @staticmethod
    def nextafter(x, y):
        return f"std::nextafter({x}, {y})"

    @staticmethod
    def relu(x):
        bug = config.cpp.inject_relu_bug_TESTING_ONLY
        if bug == "compile_error":
            return "compile error!"
        elif bug == "runtime_error":
            return f"{x}; throw 1"
        elif bug == "accuracy":
            return f"{x} + decltype({x})(1)"
        elif bug is None:
            return f"std::max({x}, decltype({x})(0))"
        else:
            raise AssertionError(
                f"unrecognized config cpp.inject_relu_bug_TESTING_ONLY = {bug!r}"
            )

    @staticmethod
    def minimum(a, b):
        return f"min_propagate_nan({a}, {b})"

    @staticmethod
    def maximum(a, b):
        return f"max_propagate_nan({a}, {b})"

    @staticmethod
    def where(a, b, c):
        return f"{a} ? {b} : {c}"

    @staticmethod
    def mod(a, b):
        return f"mod({a}, {b})"

    @staticmethod
    def constant(val, dtype):
        return value_to_cpp(val, DTYPE_TO_CPP[dtype])

    @staticmethod
    def index_expr(expr, dtype):
        idx_str = cexpr(V.kernel.rename_indexing(expr))
        var = V.kernel.cse.generate(
            V.kernel.compute, idx_str, bounds=get_bounds_index_expr(expr)
        )
        return ops.to_dtype(var, dtype)

    @staticmethod
    def masked(mask, body, other):
        code = BracesBuffer()

        # Write masked operation into a lambda
        body_var = V.kernel.cse.newvar()
        code.writeline(f"auto {body_var} = [&]")
        with V.kernel.swap_buffers(code), code.indent():
            result = body()
            code.writeline(f"return {result};")
        code.writeline(";")
        V.kernel.compute.splice(code)

        # Use the lambda's return type as the type of other
        other_code = value_to_cpp(other, f"decltype({body_var}())")
        return f"{mask} ? {body_var}() : {other_code}"

    @staticmethod
    def logical_and(a, b):
        return f"{a} && {b}"

    @staticmethod
    def logical_not(a):
        return f"!{a}"

    @staticmethod
    def logical_or(a, b):
        return f"{a} || {b}"

    @staticmethod
    def logical_xor(a, b):
        return f"{a} != {b}"

    @staticmethod
    def bitwise_and(a, b):
        return f"decltype({a})({a} & {b})"

    @staticmethod
    def bitwise_not(a):
        return f"decltype({a})(~{a})"

    @staticmethod
    def bitwise_or(a, b):
        return f"decltype({a})({a} | {b})"

    @staticmethod
    def bitwise_xor(a, b):
        return f"decltype({a})({a} ^ {b})"

    @staticmethod
    def bitwise_left_shift(a, b):
        code = BracesBuffer()
        code.writeline("[&]()")
        with code.indent():
            scalar_t = DTYPE_TO_CPP[a.dtype]
            code.writeline(
                f"constexpr decltype({b}) max_shift = sizeof({scalar_t}) * CHAR_BIT;"
            )
            code.writeline(
                f"if ((static_cast<std::make_signed_t<{scalar_t}>>({b}) < 0) || ({b} >= max_shift))"
            )
            with code.indent():
                code.writeline(f"return decltype({a})(0);")
            code.writeline(
                f"return decltype({a})(static_cast<std::make_unsigned_t<{scalar_t}>>({a}) << {b});"
            )
        code.writeline("()")
        return code

    @staticmethod
    def bitwise_right_shift(a, b):
        code = BracesBuffer()
        code.writeline("[&]()")
        with code.indent():
            scalar_t = DTYPE_TO_CPP[a.dtype]
            code.writeline(
                f"constexpr decltype({b}) max_shift = sizeof({scalar_t}) * CHAR_BIT - std::is_signed_v<{scalar_t}>;"
            )
            code.writeline(
                f"if ((static_cast<std::make_signed_t<{scalar_t}>>({b}) < 0) || ({b} >= max_shift))"
            )
            with code.indent():
                code.writeline(f"return decltype({a})({a} >> max_shift);")
            code.writeline(f"return decltype({a})({a} >> {b});")
        code.writeline("()")
        return code

    @staticmethod
    def rand(seed: sympy.Expr, offset: sympy.Expr):
        return f"normalized_rand_cpu({seed}, {offset})"

    @staticmethod
    def randn(seed: sympy.Expr, offset: sympy.Expr):
        return f"randn_cpu({seed}, {offset})"

    @staticmethod
    def randint64(seed: sympy.Expr, offset: sympy.Expr, low, high):
        return f"randint64_cpu({seed}, {offset}, {low}, {high})"

    @staticmethod
    def sigmoid(x):
        return f"decltype({x})(1) / (decltype({x})(1) + std::exp(-{x}))"

    @staticmethod
    def sign(x):
        code = BracesBuffer()
        scalar_zero = f"decltype({x})(0)"
        scalar_one = f"decltype({x})(1)"
        code.writeline("[&]()")
        with code.indent():
            code.writeline(f"auto left = {x} > 0 ? {scalar_one} : {scalar_zero};")
            code.writeline(f"auto right = {x} < 0 ? {scalar_one} : {scalar_zero};")
            code.writeline("return left - right;")
        code.writeline("()")
        return code

    def partial_accumulate(
        self,
        name: str,
        reduction_type: str,
        value: CSEVariable,
        extra_meta: dict[str, Any],
    ) -> None:
        raise NotImplementedError


CppOverrides._initialize_pointwise_overrides("cpp")


class CppVecOverrides(CppOverrides):
    """Map element-wise ops to aten vectorization C++"""

    def __new__(cls, *args, **kargs):
        self = super().__new__(cls)

        def wrap(func):
            # `CppVecKernel` generates both scalar ops and vector ops according to
            # whether the inputs are scalars or vectors while all ops in `CppVecOverrides`
            # (except for some ops explained below) assume the inputs are vectors. We wrap the ops in
            # `CppVecOverrides` to broadcast scalar inputs to vectors if needed or fallback to
            # `CppOverrides` when all inputs are scalars.
            #
            # Notes on ops handled separately in their own functions:
            # `ops.masked`:
            #     needs recursive handling of masked body.
            # `ops.index_expr`:
            #     needs to further analyze the dependency of the index expression on
            #     the tiling itervar.
            def wrapper(*args, **kwargs):
                scalars = [
                    arg
                    for arg in args
                    if isinstance(arg, (int, sympy.Expr))
                    or (isinstance(arg, CppCSEVariable) and not arg.is_vec)
                ]
                vectors = [
                    arg
                    for arg in args
                    if isinstance(arg, CppCSEVariable) and arg.is_vec
                ]
                new_args = list(args)
                if scalars and vectors:
                    new_args = []
                    for arg in args:
                        if isinstance(arg, (int, sympy.Expr)):
                            if isinstance(arg, sympy.Expr) and not arg.is_number:
                                arg = ops.index_expr(arg, torch.int64)
                            else:
                                arg = ops.constant(arg, torch.int64)
                            arg = arg.value if isinstance(arg, OpsValue) else arg
                        new_args.append(arg)

                # DType Promotion
                if vectors:
                    # We have saw several data type mismatch issues related with index_expr in
                    # the lowering phase of torch.int8. torch.int32, torch.int64.
                    # 1. int32 and int64 in test_torchinductor.py::test_max_pool2d_with_indices_backward3_cpu
                    # 2. int8 and int32 in test_torchinductor.py::test_max_pool2d5_cpu
                    # 3. int32 and fp32 in test_torchinductor_dynamic_shapes.py::test_avg_pool2d8_dynamic_shapes_cpu
                    if len(new_args) == 2:
                        new_args = promote_args(new_args)
                    elif func is CppVecOverrides.where:
                        new_args[1:] = promote_args(new_args[1:])

                # Broadcast scalar args to vector
                if scalars and vectors:
                    assert isinstance(V.kernel, CppVecKernel)
                    new_args = [
                        (
                            V.kernel.broadcast(new_arg)
                            if (
                                isinstance(new_arg, CppCSEVariable)
                                and not new_arg.is_vec
                                and func
                                not in [
                                    CppVecOverrides.rand,
                                    CppVecOverrides.randn,
                                    CppVecOverrides.randint64,
                                ]
                            )
                            else new_arg
                        )
                        for new_arg in new_args
                    ]

                if vectors:
                    return func(*new_args, **kwargs)
                else:
                    # fallback to scalar ops
                    scalar_ops = super(CppVecOverrides, self)
                    scalar_func = getattr(scalar_ops, func.__name__)
                    assert scalar_func is not None
                    return scalar_func(*args, **kwargs)

            return wrapper

        for name, method in vars(CppVecOverrides).items():
            if getattr(method, "__class__", None) is staticmethod and name not in [
                "masked",
                "index_expr",
            ]:
                setattr(self, name, wrap(method.__func__))

        return self

    @staticmethod
    def add(a, b):
        return f"{a} + {b}"

    @staticmethod
    def sub(a, b):
        return f"{a} - {b}"

    @staticmethod
    def mul(a, b):
        return f"{a} * {b}"

    @staticmethod
    def truediv(a, b):
        return f"{a} / {b}"

    @staticmethod
    def abs(x):
        return f"{x}.abs()"

    @staticmethod
    def sin(x):
        return f"{x}.sin()"

    @staticmethod
    def cos(x):
        return f"{x}.cos()"

    @staticmethod
    def exp(x):
        return f"{x}.exp()"

    @staticmethod
    def exp2(x):
        return f"{x}.exp2()"

    @staticmethod
    def expm1(x):
        # decompose for a better performance
        vec_one = f"decltype({x})(1)"
        return f"{x}.exp() - {vec_one}"

    @staticmethod
    def erf(x):
        return f"{x}.erf()"

    @staticmethod
    def erfc(x):
        return f"{x}.erfc()"

    @staticmethod
    def erfinv(x):
        return f"{x}.erfinv()"

    @staticmethod
    def sqrt(x):
        return f"{x}.sqrt()"

    @staticmethod
    def eq(x, y):
        assert isinstance(V.kernel, CppVecKernel)
        assert isinstance(x, CppCSEVariable)
        assert x.dtype is not None
        return f"{V.kernel._get_mask_type(x.dtype)}({x} == {y})"

    @staticmethod
    def ne(x, y):
        assert isinstance(V.kernel, CppVecKernel)
        assert isinstance(x, CppCSEVariable)
        if x.dtype == torch.bool:
            assert y.dtype == torch.bool
            x_cast, y_cast = unify_mask_base_type(V.kernel.compute, (x, y))
            return f"{x_cast} != {y_cast}"
        else:
            assert x.dtype is not None
            return f"{V.kernel._get_mask_type(x.dtype)}({x} != {y})"

    @staticmethod
    def lt(x, y):
        assert isinstance(V.kernel, CppVecKernel)
        assert isinstance(x, CppCSEVariable)
        assert x.dtype is not None
        return f"{V.kernel._get_mask_type(x.dtype)}({x} < {y})"

    @staticmethod
    def gt(x, y):
        assert isinstance(V.kernel, CppVecKernel)
        assert isinstance(x, CppCSEVariable)
        assert x.dtype is not None
        return f"{V.kernel._get_mask_type(x.dtype)}({x} > {y})"

    @staticmethod
    def le(x, y):
        assert isinstance(V.kernel, CppVecKernel)
        assert isinstance(x, CppCSEVariable)
        assert x.dtype is not None
        return f"{V.kernel._get_mask_type(x.dtype)}({x} <= {y})"

    @staticmethod
    def ge(x, y):
        assert isinstance(V.kernel, CppVecKernel)
        assert isinstance(x, CppCSEVariable)
        assert x.dtype is not None
        return f"{V.kernel._get_mask_type(x.dtype)}({x} >= {y})"

    @staticmethod
    def and_(x, y):
        return f"{x} & {y}"

    @staticmethod
    def rsqrt(x):
        return f"{x}.rsqrt()"

    @staticmethod
    def pow(a, b):
        return f"{a}.pow({b})"

    @staticmethod
    def log(x):
        return f"{x}.log()"

    @staticmethod
    def round(x):
        return f"{x}.round()"

    @staticmethod
    def floor(x):
        return f"{x}.floor()"

    @staticmethod
    def ceil(x):
        return f"{x}.ceil()"

    @staticmethod
    def trunc(x):
        return f"{x}.trunc()"

    @staticmethod
    def fmod(a, b):
        return f"{a}.fmod({b})"

    @staticmethod
    def lgamma(x):
        return f"{x}.lgamma()"

    @staticmethod
    def logical_and(a, b):
        a, b = may_unify_binary_op_mask_type(a, b)
        return f"{a} & {b}"

    @staticmethod
    def logical_not(a):
        return f"~{a}"

    @staticmethod
    def logical_or(a, b):
        a, b = may_unify_binary_op_mask_type(a, b)
        return f"{a} | {b}"

    @staticmethod
    def logical_xor(a, b):
        a, b = may_unify_binary_op_mask_type(a, b)
        return f"{a} ^ {b}"

    @staticmethod
    def bitwise_and(a, b):
        a, b = may_unify_binary_op_mask_type(a, b)
        return f"{a} & {b}"

    @staticmethod
    def bitwise_not(a):
        return f"~{a}"

    @staticmethod
    def bitwise_or(a, b):
        a, b = may_unify_binary_op_mask_type(a, b)
        return f"{a} | {b}"

    @staticmethod
    def bitwise_xor(a, b):
        a, b = may_unify_binary_op_mask_type(a, b)
        return f"{a} ^ {b}"

    @staticmethod
    def bitwise_left_shift(a, b):
        return f"{a} << {b}"

    @staticmethod
    def bitwise_right_shift(a, b):
        return f"{a} >> {b}"

    @staticmethod
    def load_seed(name, offset):
        assert isinstance(V.kernel, CppVecKernel)
        return f"{V.kernel.load(name, offset)}"

    @staticmethod
    def rand(seed, offset):
        assert isinstance(V.kernel, CppVecKernel)
        code = BracesBuffer()
        rand_function = (
            f"result[offset_idx] = normalized_rand_cpu({seed}, offset[offset_idx]);"
        )
        return codegen_rand(offset, code, rand_function)

    @staticmethod
    def randn(seed, offset):
        assert isinstance(V.kernel, CppVecKernel)
        code = BracesBuffer()
        rand_function = f"result[offset_idx] = randn_cpu({seed}, offset[offset_idx]);"
        return codegen_rand(offset, code, rand_function)

    @staticmethod
    def randint64(seed, offset, low, high):
        assert isinstance(V.kernel, CppVecKernel)
        code = BracesBuffer()
        rand_function = f"result[offset_idx] = randint64_cpu({seed}, offset[offset_idx], {low}, {high});"
        return codegen_rand(offset, code, rand_function, torch.int64)

    @staticmethod
    def remainder(a, b):
        assert a.dtype == b.dtype, (
            "remainder vec implementation expect the same inputs' dtype."
        )
        return f"{a} - ({CppVecOverrides.floordiv(a, b)}) * {b}"

    @staticmethod
    def tan(a):
        return f"{a}.tan()"

    @staticmethod
    def tanh(a):
        if config.cpp.use_decompose_tanh:
            vec_one = f"decltype({a})(1)"
            vec_two = f"decltype({a})(2)"
            vec_minus_two = f"decltype({a})(-2)"
            return (
                f"{vec_two} / ({vec_one} + ({vec_minus_two} * {a}).exp()) - {vec_one}"
            )
        else:
            return f"{a}.tanh()"

    @staticmethod
    def reciprocal(a):
        return f"{a}.reciprocal()"

    @staticmethod
    def atan(x):
        return f"{x}.atan()"

    @staticmethod
    def acos(x):
        return f"{x}.acos()"

    @staticmethod
    def asin(x):
        return f"{x}.asin()"

    @staticmethod
    def cosh(x):
        return f"{x}.cosh()"

    @staticmethod
    def sinh(x):
        return f"{x}.sinh()"

    @staticmethod
    def log10(x):
        return f"{x}.log10()"

    @staticmethod
    def log2(x):
        return f"{x}.log2()"

    @staticmethod
    def nextafter(x, y):
        return f"{x}.nextafter({y})"

    @staticmethod
    def copysign(a, b):
        return f"{a}.copysign({b})"

    @staticmethod
    def atan2(a, b):
        return f"{a}.atan2({b})"

    @staticmethod
    def hypot(a, b):
        return f"{a}.hypot({b})"

    @staticmethod
    def atanh(x):
        # For real x, atanh(x) = 1/2 * log((1+x)/(1-x))
        vec_one = f"decltype({x})(1)"
        vec_one_half = f"decltype({x})(0.5)"
        return f"{vec_one_half} * (({vec_one} + {x})/({vec_one} - {x})).log()"

    @staticmethod
    def asinh(x):
        return f"{x}.asinh()"

    @staticmethod
    def acosh(x):
        return f"{x}.acosh()"

    @staticmethod
    def relu(x):
        bug = config.cpp.inject_relu_bug_TESTING_ONLY
        if bug == "compile_error":
            return "compile error!"
        elif bug == "runtime_error":
            return f"{x}; throw 1"
        elif bug == "accuracy":
            return f"{x} + decltype({x})(1)"
        elif bug is None:
            return f"at::vec::clamp_min({x}, decltype({x})(0))"
        else:
            raise AssertionError(
                f"unrecognized config cpp.inject_relu_bug_TESTING_ONLY = {bug!r}"
            )

    # TODO: this seems to be dead
    @staticmethod
    def sigmoid(x):
        return f"decltype({x})(1)/(decltype({x})(1) + {x}.neg().exp())"

    @staticmethod
    def neg(x):
        return f"{x}.neg()"

    @staticmethod
    def floordiv(a, b):
        if is_float_dtype(a.dtype):
            assert a.dtype == b.dtype, (
                "div_floor_floating_vec implementation expect the same inputs' dtype."
            )
            return f"div_floor_floating_vec({a}, {b})"
        else:
            assert all(is_integer_dtype(item.dtype) for item in [a, b])
            # a and b are integer type
            _t = f"decltype({a})"
            if V.kernel._get_raw_num_vectors(b.dtype) < 1:
                # Doing blend to set the remaining bits of b to non-zero
                b = f"{_t}::blend<{(1 << V.kernel.tiling_factor) - 1}>({_t}(1), {b})"
            quot = f"{a} / {b}"
            has_rem = f"({a} % {b} != {_t}(0))"
            is_neg = f"(({a} < {_t}(0)) != ({b} < {_t}(0)))"
            return f"{_t}::blendv({quot}, {quot} - {_t}(1), {has_rem} & {is_neg})"

    @staticmethod
    def truncdiv(a, b):
        # a and b are integer type
        if V.kernel._get_raw_num_vectors(b.dtype) < 1:
            # Doing blend to set the remaining bits of b to non-zero
            _t = f"decltype({b})"
            b = f"{_t}::blend<{(1 << V.kernel.tiling_factor) - 1}>({_t}(1), {b})"
        return f"{a} / {b}"

    @staticmethod
    def minimum(a, b):
        if a.dtype == torch.bool:
            assert b.dtype == torch.bool
            a_cast, b_cast = unify_mask_base_type(V.kernel.compute, (a, b))
            return f"{a_cast} & {b_cast}"
        else:
            return f"at::vec::minimum({a}, {b})"

    @staticmethod
    def maximum(a, b):
        if a.dtype == torch.bool:
            assert b.dtype == torch.bool
            a_cast, b_cast = unify_mask_base_type(V.kernel.compute, (a, b))
            return f"{a_cast} | {b_cast}"
        else:
            return f"at::vec::maximum({a}, {b})"

    @staticmethod
    def square(a):
        return f"{a} * {a}"

    @staticmethod
    def where(a, b, c):
        assert isinstance(V.kernel, CppVecKernel)
        if b.dtype == torch.bool:
            assert c.dtype == torch.bool
            blendv_a, blendv_b, blendv_c = unify_mask_base_type(
                V.kernel.compute, (a, b, c)
            )
            return f"decltype({blendv_b})::blendv({blendv_c}, {blendv_b}, {blendv_a})"
        else:
            return f"decltype({b})::blendv({c}, {b}, {V.kernel._get_mask_cast(a, b.dtype)})"

    @staticmethod
    def sign(x):
        code = BracesBuffer()
        vec_zero = f"decltype({x})(0)"
        vec_one = f"decltype({x})(1)"
        blendv_l = f"decltype({x})::blendv({vec_zero}, {vec_one}, {vec_zero} < {x})"
        blendv_r = f"decltype({x})::blendv({vec_zero}, {vec_one}, {x} < {vec_zero})"
        code.writeline("[&]()")
        with code.indent():
            code.writeline(f"auto left = {blendv_l};")
            code.writeline(f"auto right = {blendv_r};")
            code.writeline("return left - right;")
        code.writeline("()")
        return code

    @staticmethod
    def to_dtype(x, dtype, src_dtype=None, use_compute_types=True):
        assert dtype in [
            torch.bool,
            torch.float64,
            torch.float,
            torch.bfloat16,
            torch.float16,
            torch.uint8,
            torch.int8,
            torch.int32,
            torch.int64,
            torch.float8_e4m3fn,
            torch.float8_e5m2,
        ], f"{__name__} does not support {dtype}"
        assert isinstance(x, CppCSEVariable)
        src_dtype = x.dtype
        expr = V.kernel.get_to_dtype_expr(x, dtype, src_dtype)
        csevar = V.kernel.cse.generate(V.kernel.compute, expr)
        csevar.update_on_args("to_dtype", (x, dtype), {"src_dtype": src_dtype})
        if dtype in DTYPE_LOWP_FP and src_dtype == torch.float:
            V.kernel.cache_dtype_convert(x, src_dtype, csevar, dtype)
        return csevar

    @staticmethod
    def log1p(x):
        bug = config.cpp.inject_log1p_bug_TESTING_ONLY
        if bug == "accuracy":
            return f"{x} + decltype({x})(1)"
        elif bug is None:
            return f"{x}.log1p()"
        else:
            raise AssertionError(
                f"unrecognized config cpp.inject_log1p_bug_TESTING_ONLY = {bug!r}"
            )

    @staticmethod
    def masked(mask, body, other):
        assert isinstance(V.kernel, CppVecKernel)
        code = BracesBuffer()
        var = V.kernel.cse.newvar()
        with V.kernel.masked(mask) as new_mask:
            code.writeline(f"auto {var} = [&]")
            with V.kernel.swap_buffers(code), code.indent():
                result = body()
                code.writeline(f"return {result};")
        code.writeline(";")
        V.kernel.compute.splice(code)

        dtype = result.dtype
        body_code = f"{var}()"

        def maskify_or_vecify(code):
            return (
                f"{V.kernel._get_mask_type()}::from({code})"
                if dtype == torch.bool
                else f"{V.kernel._get_vec_type(dtype)}({code})"
            )

        if result.is_vec:
            body_code_vec = body_code
        else:
            body_code_vec = maskify_or_vecify(body_code)
        other_code = value_to_cpp(other, DTYPE_TO_CPP[dtype])
        # loading bool as VecMask<float, N>
        other_code_vec = maskify_or_vecify(other_code)
        assert isinstance(new_mask, CppCSEVariable), new_mask
        if new_mask.is_vec:
            code = BracesBuffer()
            code.writeline("[&]")
            with V.kernel.swap_buffers(code), code.indent():
                code.writeline(f"if ({new_mask}.all_zero())")
                with code.indent():
                    code.writeline(f"return {other_code_vec};")
                code.writeline("else")
                with code.indent():
                    # Create cse variable to reuse kernel.overrides.where
                    body_vec_var = V.kernel.cse.generate(
                        V.kernel.compute,
                        body_code_vec,
                    )
                    other_vec_var = V.kernel.cse.generate(
                        V.kernel.compute,
                        other_code_vec,
                    )
                    assert isinstance(body_vec_var, CppCSEVariable), body_vec_var
                    assert isinstance(other_vec_var, CppCSEVariable), other_vec_var
                    body_vec_var.dtype = dtype
                    other_vec_var.dtype = dtype
                    overrides: type[Union[CppOverrides, CppVecOverrides]] = (
                        # pyrefly: ignore [bad-assignment]
                        V.kernel.overrides
                    )  # type: ignore[has-type]
                    code.writeline(
                        f"return {overrides.where(new_mask, body_vec_var, other_vec_var)};"
                    )
            code.writeline("()")
            csevar = V.kernel.cse.generate(
                V.kernel.compute,
                code,
            )
            result.is_vec = True
        elif result.is_vec:
            csevar = V.kernel.cse.generate(
                V.kernel.compute, f"{mask} ? {body_code_vec} : {other_code_vec}"
            )
        else:
            csevar = V.kernel.cse.generate(
                V.kernel.compute, f"{mask} ? {body_code} : {other_code}"
            )
        # `result` is explicitly added to the args for correct propagation
        # of relevant itervars and vectorization status.
        csevar.update_on_args("masked", (mask, body, other, result), {})
        return csevar

    @staticmethod
    def index_expr(expr, dtype):
        assert isinstance(V.kernel, CppVecKernel)
        index = V.kernel.rename_indexing(expr)
        tiling_var = V.kernel.itervars[V.kernel.tiling_idx]
        stride = V.kernel._try_get_const_stride(index, tiling_var)
        if stride == 0:
            return CppOverrides.index_expr(expr, dtype)
        elif stride is not None:
            idx = V.kernel.cse.generate(
                V.kernel.compute, cexpr(index), bounds=get_bounds_index_expr(expr)
            )
            value = ops.to_dtype(idx, dtype)
            if isinstance(value, OpsValue):
                value = value.value
            csevar = V.kernel.arange(value, stride)
        else:
            csevar = V.kernel._load_or_store_non_contiguous(  # type: ignore[assignment]
                None, index, dtype, V.kernel.compute
            )
        # pyrefly: ignore [missing-attribute]
        csevar.update_on_args("index_expr", (expr, dtype), {})
        return csevar

    @staticmethod
    def frexp(x):
        cache_keys = f"frexp({x})[0]", f"frexp({x})[1]"
        if all(V.kernel.cse.try_get(cache_key) is not None for cache_key in cache_keys):
            return tuple(V.kernel.cse.try_get(cache_key) for cache_key in cache_keys)

        cdtype = DTYPE_TO_CPP[x.dtype]
        size = V.kernel.tail_size if V.kernel.tail_size else V.kernel.tiling_factor
        code = BracesBuffer()
        exponent = V.kernel.cse.newvar(dtype=torch.int32)
        mantissa = V.kernel.cse.newvar(dtype=x.dtype)
        exponent.update_on_args("frexp", (x,), kwargs={})
        mantissa.update_on_args("frexp", (x,), kwargs={})
        n_vec = V.kernel._get_num_vectors(x.dtype)
        mantissa_t = (
            f"at::vec::Vectorized<{cdtype}>"
            if n_vec == 1
            else f"at::vec::VectorizedN<{cdtype}, {n_vec}>"
        )
        code.writeline(
            f"at::vec::Vectorized<int32_t> {exponent};"
            if n_vec == 1
            else f"at::vec::VectorizedN<int32_t, {n_vec}> {exponent};"
        )
        code.writeline(f"{mantissa_t} {mantissa};")
        code.writeline("[&]()")
        with code.indent():
            code.writeline(
                f"__at_align__ std::array<{cdtype}, {V.kernel.tiling_factor}> tmpbuf;"
            )
            code.writeline(f"{x}.store(tmpbuf.data(), {cexpr_index(size)});")
            code.writeline(
                f"__at_align__ std::array<int32_t, {V.kernel.tiling_factor}> tmpbuf_exponent;"
            )
            code.writeline(
                f"__at_align__ std::array<{cdtype}, {V.kernel.tiling_factor}> tmpbuf_mantissa;"
            )
            code.writeline(f"for (int i = 0; i < {cexpr_index(size)}; i++)")
            with code.indent():
                code.writeline(
                    "tmpbuf_mantissa[i] = std::frexp(tmpbuf[i], &tmpbuf_exponent[i]);"
                )
            code.writeline(
                f"{exponent} = at::vec::Vectorized<int32_t>::loadu(tmpbuf_exponent.data(), {cexpr_index(size)});"
                if n_vec == 1
                else f"{exponent} = at::vec::VectorizedN<int32_t, {n_vec}>::loadu(tmpbuf_exponent.data(), {cexpr_index(size)});"
            )
            code.writeline(
                f"{mantissa} = {mantissa_t}::loadu(tmpbuf_mantissa.data(), {cexpr_index(size)});"
            )
        code.writeline("();")
        V.kernel.compute.splice(code)
        cse_vars = (mantissa, exponent)
        for cache_key, cse_var in zip(cache_keys, cse_vars):
            V.kernel.cse.put(cache_key, cse_var)
        return mantissa, exponent

    @classmethod
    def _scalarize(cls, scalar_func):
        def inner(*args, **kwargs):
            assert not kwargs
            kernel = V.kernel
            assert isinstance(kernel, CppVecKernel)
            code = BracesBuffer()
            code.writeline("[&]()")
            vec_dtype = args[0].dtype
            n_vec = kernel._get_num_vectors(vec_dtype)
            size = kernel.tail_size if kernel.tail_size else kernel.tiling_factor
            scalar_args = []
            cdtype = DTYPE_TO_CPP[vec_dtype]
            output_mask = scalar_func.__name__ in (
                "isinf",
                "isnan",
                "signbit",
            )
            octype = "bool" if output_mask else cdtype
            octype = (
                DTYPE_TO_CPP[args[-2]]
                if (scalar_func.__name__ == "to_dtype_bitcast")
                else octype
            )
            with code.indent():
                for argidx, arg in enumerate(args):
                    if isinstance(arg, CppCSEVariable):
                        assert arg.is_vec
                        assert arg.dtype == vec_dtype
                        code.writeline(
                            f"__at_align__ std::array<{cdtype}, {kernel.tiling_factor}> tmpbuf{argidx};"
                        )
                        code.writeline(
                            f"{arg}.store(tmpbuf{argidx}.data(), {cexpr_index(size)});"
                        )
                        scalar_args.append(f"tmpbuf{argidx}[i]")
                    else:
                        scalar_args.append(arg)
                code.writeline(
                    f"__at_align__ std::array<{octype}, {kernel.tiling_factor}> tmpbuf_out;"
                )
                res = scalar_func(*scalar_args)
                code.writeline(f"for (int i = 0; i < {cexpr_index(size)}; i++)")
                with code.indent():
                    code.writeline(f"tmpbuf_out[i] = {res};")
                load_args = f"tmpbuf_out.data(), {cexpr_index(size)}"
                if output_mask:
                    load_fn = f"at::vec::VecMask<{cdtype},{n_vec}>::from"
                elif n_vec == 1:
                    load_fn = f"at::vec::Vectorized<{octype}>::loadu"
                else:
                    load_fn = f" at::vec::VectorizedN<{octype}, {n_vec}>::loadu"
                code.writeline(f"return {load_fn}({load_args});")
            code.writeline("()")
            return code

        return inner

    @classmethod
    def _initialize_scalarize(cls):
        vec_vars = vars(CppVecOverrides)
        for name, method in vars(CppOverrides).items():
            if isinstance(method, staticmethod) and name not in vec_vars:
                func = cls._scalarize(method.__func__)
                func.__name__ = name
                setattr(cls, name, staticmethod(func))


CppVecOverrides._initialize_pointwise_overrides("cppvec")
CppVecOverrides._initialize_scalarize()


class CppTile2DOverrides(CppVecOverrides):
    @staticmethod
    def index_expr(expr, dtype):
        assert isinstance(V.kernel, CppTile2DKernel)
        expr = V.kernel.transform_indexing(expr)
        return CppVecOverrides.index_expr(expr, dtype)


class CppKernel(Kernel):
    """
    Base class for C++ kernel code generation in PyTorch Inductor.
    This class is responsible for generating C++ code from the intermediate representation.

    Args:
        args: Kernel arguments used for code generation
        num_threads: Number of threads for parallel execution
    """

    overrides = CppOverrides  # type: ignore[assignment]
    sexpr = cexpr
    newvar_prefix = "auto "
    suffix = ";"

    def __init__(self, args, num_threads):
        super().__init__(args)
        # Indicate when this kernel is active, for example
        # {x0, {24, 26}} -> this kernel is active when x0 >= 24 and x0 < 26
        self.active_ranges: dict[sympy.Expr, tuple[sympy.Expr, ...]] = {}
        # Indicate this kernel will be moved under the inner for-loop
        # See move_code_under_inner_loop
        self.inner_itervars: list[sympy.Symbol] = []
        self.call_ranges: Optional[tuple[sympy.Expr, ...]] = None
        self.ranges: list[sympy.Expr] = []
        self.itervars: list[sympy.Symbol] = []
        self.reduction_depth = None
        self.reduction_prefix = IndentedBuffer()
        # We need this because when we run "reduction" nodes here, we lack
        # "loop" information to decide whether we need a scalar init or an array init
        # in the reduction prefix. Meanwhile, we have other information like
        # reduction types and dtype to generate the reduction prefix. We record the information
        # with a callable lambda function, and when we have enough information to finalize
        # the reduction prefix, we can invoke the functions here with additional information.
        self.reduction_prefix_generators: list[Callable] = []  # type: ignore[type-arg]
        self.reduction_suffix = IndentedBuffer()
        self.parallel_reduction_prefix = IndentedBuffer()
        self.parallel_reduction_suffix = IndentedBuffer()
        self.local_reduction_init = IndentedBuffer()
        self.local_reduction_stores = IndentedBuffer()
        self.is_reduction = False
        self.non_parallel_reduction_prefix = IndentedBuffer()
        self.non_parallel_reduction_suffix = IndentedBuffer()
        self.reduction_cse = CSE(self.newvar_prefix, self.suffix, name_prefix="tmp_acc")
        self.welford_helper_cse = CSE(
            self.newvar_prefix, self.suffix, name_prefix="welford_helper"
        )
        self.cascade_helper_cse = CSE(
            self.newvar_prefix, self.suffix, name_prefix="cascade_helper"
        )
        self.preloads = IndentedBuffer()
        self.poststores = IndentedBuffer()
        self.num_threads = num_threads  # num_threads the kernel specialized for
        self.reduction_omp_dec: dict[tuple[str, str], str] = {}
        self.reduction_var_names: list[str] = []

    def _gen_parallel_reduction_buffers(
        self,
        acc,
        acc_type,
        reduction_type,
        dtype,
        reduction_combine_fn=reduction_combine,
        reduction_init_fn=reduction_init,
    ):
        if config.cpp.dynamic_threads and not self.parallel_reduction_prefix:
            self.parallel_reduction_prefix.writeline(
                "int max_threads = omp_get_max_threads();"
            )
        acc_local = f"{acc}_local"
        num_threads = (
            "max_threads" if config.cpp.dynamic_threads else parallel_num_threads()
        )
        acc_local_in_array = f"{acc}_arr[tid]"
        self.local_reduction_init.writeline(
            f"{acc_type} {acc_local} = {reduction_init_fn(reduction_type, dtype)};"
        )
        self.parallel_reduction_prefix.splice(
            reduction_prefix_array(
                acc,
                acc_type,
                reduction_type,
                dtype,
                num_threads,
                reduction_init_fn,
            )
        )
        self.local_reduction_stores.writeline(f"{acc_local_in_array} = {acc_local};")
        self.parallel_reduction_suffix.writelines(
            [
                f"for (int tid = 0; tid < {num_threads}; tid++)",
                "{",
                f"    {acc} = {reduction_combine_fn(reduction_type, acc, acc_local_in_array, src_dtype=dtype)};",
                "}",
            ],
        )

    def update_stores_with_parallel_reduction(self):
        for var_name in self.reduction_var_names:
            replace_acc_name(self.stores, var_name, f"{var_name}_local")

    def gen_body(self, code: Optional[BracesBuffer] = None):
        assert code is None
        code = BracesBuffer()
        with contextlib.ExitStack() as stack:
            if hasattr(self, "codegen_inner_loops"):
                code.splice(self.preloads)
                self.codegen_inner_loops(code)
                stack.enter_context(code.indent())
            code.splice(self.loads)
            code.splice(self.compute)
            code.splice(self.stores)
        if hasattr(self, "codegen_inner_loops"):
            code.splice(self.poststores)

        if self.inner_itervars:
            for idx in self.inner_itervars:
                start, end = self.active_ranges[idx]
                code = move_code_under_inner_loop(code, idx, f"{idx}_tail", start, end)
        return code

    @contextlib.contextmanager
    def masked(self, mask):
        """Context manager to add an additional mask to loads and stores."""
        prior = self._load_mask
        if prior:
            mask = ops.and_(mask, prior)
            if isinstance(mask, OpsValue):
                mask = mask.value
                assert isinstance(mask, CppCSEVariable)
                # see NOTE [dtype of CppCSEVariable]
                # mask's dtype should be bool
                mask.dtype = torch.bool

        self._load_mask = mask
        try:
            yield mask
        finally:
            self._load_mask = prior

    def scale_index_with_offset(
        self, index: sympy.Expr, scale=1, itervar_idx=-1, offset=0
    ):
        var = self.itervars[itervar_idx]
        replacement = {var: var * scale + offset}
        new_index = sympy_subs(index, replacement)
        return new_index

    def index_to_str(self, index: sympy.Expr) -> str:
        """
        Convert an index expr to a string that can be used in cpp code.
        e.g. a sympy expression "s2" may actually appear as "ks1" in the cpp kernel.
        """
        return cexpr(self.rename_indexing(index))

    def index_indirect_depends_on(self, index: sympy.Expr, itervar: sympy.Symbol):
        """
        Check if an index has free symbol CppCSEVariable that depends on `itervar`.
        """
        return any(
            self.cse.varname_map[s.name].depends_on(itervar)  # type: ignore[attr-defined]
            for s in index.free_symbols
            if s.name in self.cse.varname_map  # type: ignore[attr-defined]
            and isinstance(self.cse.varname_map[s.name], CppCSEVariable)  # type: ignore[attr-defined]
        )

    def index_depends_on(self, index: sympy.Expr, itervar: sympy.Symbol):
        return itervar in index.free_symbols or self.index_indirect_depends_on(
            index, itervar
        )

    def var_ranges(self):
        return dict(zip(self.itervars, self.ranges))

    def check_bounds(
        self,
        expr: sympy.Expr,
        size: sympy.Expr,
        lower: bool,
        upper: bool,
    ):
        if not (lower or upper):
            return

        indirect = free_symbol_is_type(expr, SymT.TMP)
        if indirect:
            # indexing in compute
            csevar = ops.index_expr(expr, torch.int64).value
            buffer = V.kernel.compute
        else:
            # indexing in loads
            prior_compute = V.kernel.compute
            try:
                V.kernel.compute = self.loads
                csevar = ops.index_expr(expr, torch.int64).value
            finally:
                V.kernel.compute = prior_compute
            buffer = self.loads

        size_str = V.kernel.sexpr(self.rename_indexing(size)) if upper else None

        line = self.indirect_assert(
            csevar, "0" if lower else None, size_str, self._load_mask
        )
        self.cse.generate(buffer, line, assignment=False)

    def load(self, name: str, index: sympy.Expr):
        var = self.args.input(name)
        index = self.rename_indexing(index)
        line = f"{var}[{cexpr_index(index)}]"
        csevar = self.cse.generate(self.loads, line, dtype=V.graph.get_dtype(name))
        csevar.update_on_args("load", (self, name, index), {})
        return csevar

    def store(self, name, index, value, mode=None):
        assert "buf" in name
        var = self.args.output(name)
        index = self.rename_indexing(index)
        if mode is None:
            line = f"{var}[{cexpr_index(index)}] = {value};"
        elif mode == "atomic_add":
            if not config.cpp.dynamic_threads and self.num_threads == 1:
                line = f"{var}[{cexpr_index(index)}] += {value};"
            else:
                dtype = V.graph.get_dtype(name)
                # mirroring static_cast<float>(...) in load:
                value = f"static_cast<{DTYPE_TO_CPP[dtype]}>({value})"
                line = f"atomic_add(&{var}[{cexpr_index(index)}], {value});"
        else:
            raise NotImplementedError(f"store mode={mode}")
        self.stores.writeline(DeferredLine(name, line))

    def device_assert_async(self, cond, msg):
        self.compute.writeline(
            f'({cond} ? 0 : (throw std::runtime_error("{msg}"), 0));'
        )

    def _gen_reduction_prefix(
        self,
        acc: Union[CSEVariable, str],
        acc_type: str,
        rtype: str,
        dtype: torch.dtype,
        init_fn,
    ):
        # Generate reduction prefix
        # If size is None, we will define and initialize a single reduction variable
        # => float tmp_acc0 = 0;
        # Otherwise, we will define and initialize a reduction array
        # => float tmp_acc0_arr[size];
        # => for (int i = 0; i < size; i++) tmp_acc0_arr[i] = 0;
        def inner(size: Optional[int] = None):
            if size is None:
                return f"{acc_type} {acc} = {init_fn(rtype, dtype)};"
            else:
                return reduction_prefix_array(
                    acc,
                    acc_type,
                    rtype,
                    dtype,
                    size,
                    init_fn,
                )

        return inner

    def finalize_reduction_prefix(self, size: Optional[int] = None):
        for gen_fn in self.reduction_prefix_generators:
            self.reduction_prefix.splice(gen_fn(size))

    def need_use_acc_helper(self, reduction_type, dtype, use_scalar):
        # Check if we need accumulate helper for the reduction operation.
        # using accumulate helper generates the necessary code to improve precision for
        # sum and welford
        # Note: using helper has non-negligible impact on performance

        if reduction_type == "welford_reduce":
            return True

        # TODO add supports for more data types when needed
        if reduction_type == "sum" and dtype == torch.float:
            assert self.call_ranges is not None
            reduction_size = functools.reduce(
                operator.mul, self.call_ranges[self.reduction_depth :]
            )

            # chunk size to balance accuracy and performance
            chunk_size = 4096

            # use acc helper If cannot get size_hint
            try:
                reduction_size_hint = V.graph.sizevars.size_hint(reduction_size)
            except Exception:
                return True

            if reduction_size_hint > chunk_size:
                # use helper if the reduction size is too large
                V.graph.sizevars.check_lt(chunk_size, reduction_size)
                return True
            else:
                V.graph.sizevars.check_leq(reduction_size, chunk_size)
        return False

    def _acc_helper_init(
        self,
        reduction_type,
        helper_val,
        helper_range,
        dtype,
        num_threads=None,
        use_scalar=False,
    ):
        num_range_thread = (
            CeilDiv(helper_range, num_threads) if num_threads else helper_range
        )
        num_range_thread_expr = cexpr_index(num_range_thread)
        assert reduction_type in ["welford_reduce", "sum"]
        chunk_size = 4096
        num_chunks = CeilDiv(num_range_thread, chunk_size)
        helper_type = (
            "WelfordHelper"
            if reduction_type == "welford_reduce"
            else "CascadeSumHelper"
        )
        if use_scalar:
            h_type = DTYPE_TO_CPP[dtype]
        else:
            h_type = (
                self._get_vec_type(dtype)
                if hasattr(self, "_get_vec_type")
                else DTYPE_TO_CPP[dtype]
            )
        helper_init_line = (
            f"{helper_type}<{h_type}, {chunk_size}> {helper_val}"
            f"("
            f"{num_range_thread_expr}"
            f");"
        )
        if reduction_type == "sum":
            return helper_init_line
        if isinstance(num_chunks, sympy.Integer) and num_chunks <= 1:
            # When the number of chunks <= 1, there is no need to use cascade summation to improve
            # reduction accuracy. We can initialize a static WelfordHelper to improve performance.
            return f"static {helper_init_line}"
        else:
            return helper_init_line

    def _use_acc_helper(
        self, reduction_type, acc, helper_val, helper_range, dtype, use_scalar=False
    ):
        num_threads = (
            "max_threads" if config.cpp.dynamic_threads else parallel_num_threads()
        )
        self.non_parallel_reduction_prefix.writeline(
            self._acc_helper_init(
                reduction_type, helper_val, helper_range, dtype, None, use_scalar
            )
        )
        self.local_reduction_init.writeline(
            self._acc_helper_init(
                reduction_type, helper_val, helper_range, dtype, num_threads, use_scalar
            )
        )
        result = acc if use_scalar else f"{acc}_vec"
        if reduction_type == "welford_reduce":
            self.non_parallel_reduction_suffix.writeline(
                f"{result} = welford_combine({result}, &{helper_val});"
            )
            self.local_reduction_stores.writeline(
                f"{result}_local = welford_combine({result}_local, &{helper_val});"
            )
        else:
            self.non_parallel_reduction_suffix.writeline(
                f"{result} = cascade_sum_final(&{helper_val});"
            )
            self.local_reduction_stores.writeline(
                f"{result}_local = cascade_sum_final(&{helper_val});"
            )

    def reduction(self, dtype, src_dtype, reduction_type, value):
        argmax_or_argmin = reduction_type in ("argmax", "argmin")
        reduction_key = src_dtype, reduction_type, value
        if reduction_key in self.reduction_cse.reduction_cache:
            return self.reduction_cse.reduction_cache[reduction_key]

        acc = self.reduction_cse.generate(
            self.loads, f"reduction {reduction_key}", write=False
        )
        self.reduction_var_names.append(f"{acc}")
        self.is_reduction = True
        init_dtype = src_dtype if argmax_or_argmin else dtype
        acc_type = reduction_acc_type(reduction_type, init_dtype)
        self.reduction_prefix_generators.append(
            self._gen_reduction_prefix(
                acc, acc_type, reduction_type, init_dtype, reduction_init
            )
        )

        if self.need_use_acc_helper(reduction_type, dtype, True):
            # use cascade_helper for vec kernel
            reduction_size = functools.reduce(
                operator.mul, self.ranges[self.reduction_depth :]
            )
            # use welford_helper/cascade_helper for vec kernel
            if reduction_type == "welford_reduce":
                helper_val = self.welford_helper_cse.generate(
                    self.compute, f"reduction {reduction_key}", write=False
                )
            else:
                helper_val = self.cascade_helper_cse.generate(
                    self.compute, f"reduction {reduction_key}", write=False
                )
            # rename the helper variable to distinguish it from vectorized version
            scalar_helper_val = f"scalar_{helper_val}"
            self._use_acc_helper(
                reduction_type,
                acc,
                scalar_helper_val,
                reduction_size,
                dtype,
                use_scalar=True,
            )
            self.stores.writeline(
                f"{acc} = {reduction_combine(reduction_type, acc, value, scalar_helper_val)};"
            )
        else:
            assert self.reduction_depth is not None
            index = self.itervars[self.reduction_depth]
            for i in range(self.reduction_depth + 1, len(self.itervars)):
                index = index * self.ranges[i] + self.itervars[i]
            self.stores.writeline(
                f"{acc} = {reduction_combine(reduction_type, acc, value, index=index)};"
            )

        self._gen_parallel_reduction_buffers(acc, acc_type, reduction_type, init_dtype)
        result = reduction_project(reduction_type, acc)
        self.reduction_cse.reduction_cache[reduction_key] = result
        return result

    def store_reduction(self, name, index, value):
        index = self.rename_indexing(index)
        var = self.args.output(name)
        self.reduction_suffix.writeline(
            DeferredLine(name, f"{var}[{cexpr_index(index)}] = {value};")
        )

    def set_ranges(self, lengths, reduction_lengths):
        if self.call_ranges:
            assert self.call_ranges == tuple(lengths) + tuple(reduction_lengths), (
                f"{self.call_ranges} == {tuple(lengths)} + {tuple(reduction_lengths)}"
            )
            assert self.reduction_depth == len(lengths)
        else:
            self.call_ranges = tuple(lengths) + tuple(reduction_lengths)
            self.ranges = [self.rename_indexing(x) for x in self.call_ranges]
            self.itervars = [
                sympy_index_symbol_with_prefix(SymT.XBLOCK, n)
                for n in range(len(self.ranges))
            ]
            # pyrefly: ignore [bad-assignment]
            self.reduction_depth = len(lengths)
        return (
            self.itervars[: self.reduction_depth],
            self.itervars[self.reduction_depth :],
        )

    def size_hint(self):
        assert self.call_ranges is not None
        return V.graph.sizevars.size_hint(
            sympy_product(self.call_ranges), fallback=8192
        )

    def codegen_loops_impl(self, loop_nest, code, worksharing):
        assert isinstance(self, CppKernelProxy)
        threads = parallel_num_threads()
        assert self.call_ranges is not None
        if isinstance(loop_nest.kernel, OuterLoopFusedKernel):
            par_depth = loop_nest.kernel.decide_parallel_depth(
                loop_nest.max_parallel_depth(), threads
            )
        else:
            par_depth = self.decide_parallel_depth(
                loop_nest.max_parallel_depth(), threads
            )

        is_reduction_loop = (
            loop_nest.loops is not None
            and loop_nest.loops[par_depth.start_depth].is_reduction
        )
        with contextlib.ExitStack() as stack:
            if par_depth.parallel_depth:
                if is_reduction_loop:
                    # need to close the worksharing scope to define reduction vars outside it
                    worksharing.close()
                else:
                    worksharing.parallel(threads)
                loop_nest.mark_parallel(par_depth)
            elif threads > 1:
                if worksharing.single():
                    stack.enter_context(code.indent())

            def gen_kernel(_loop_nest: LoopNest):
                def is_parallel_reduction():
                    assert _loop_nest.loops
                    root = _loop_nest.loops[par_depth.start_depth]
                    return root.is_reduction and root.parallel

                kernel = _loop_nest.get_kernel()
                if isinstance(kernel, OuterLoopFusedKernel):
                    for _loop_nest in kernel.inner:
                        gen_loop_nest(_loop_nest)
                else:
                    assert isinstance(kernel, CppKernelProxy)
                    if _loop_nest.loops is not None and is_parallel_reduction():
                        kernel.update_stores_with_parallel_reduction()
                    with contextlib.ExitStack() as stack:
                        stack.enter_context(code.indent())
                        kernel.gen_body(code)

            def get_reduction_prefix_suffix(kernel, parallel=False, is_suffix=False):
                if is_suffix:
                    suffix = kernel.reduction_suffix
                    if parallel:
                        suffix = kernel.parallel_reduction_suffix + suffix
                    else:
                        suffix = kernel.non_parallel_reduction_suffix + suffix
                    return suffix
                else:
                    prefix = kernel.reduction_prefix
                    if parallel:
                        prefix = prefix + kernel.parallel_reduction_prefix
                    else:
                        prefix = prefix + kernel.non_parallel_reduction_prefix
                    return prefix

            def gen_loop_with_reduction(
                _loop_nest: LoopNest, depth: int = 0, in_reduction=False
            ):
                kernel = _loop_nest.get_kernel()
                assert _loop_nest.loops
                loop = _loop_nest.loops[depth]
                with contextlib.ExitStack() as stack_outer:
                    if loop.is_reduction and not in_reduction:
                        reduction_prefix = get_reduction_prefix_suffix(
                            kernel, loop.parallel, is_suffix=False
                        )
                        if reduction_prefix:
                            stack_outer.enter_context(code.indent())
                        code.splice(reduction_prefix)
                    if is_reduction_loop and loop.parallel:
                        worksharing.parallel(threads)
                        if kernel.local_reduction_init:
                            assert kernel.local_reduction_stores
                            code.splice(kernel.local_reduction_init)

                    gen_loop_at(_loop_nest, depth)

                    if is_reduction_loop and loop.parallel:
                        if kernel.local_reduction_stores:
                            code.splice(kernel.local_reduction_stores)
                        worksharing.close()
                    if loop.is_reduction and not in_reduction:
                        code.splice(
                            get_reduction_prefix_suffix(
                                kernel, loop.parallel, is_suffix=True
                            )
                        )

            def gen_loop_at(_loop_nest: LoopNest, depth: int = 0):
                with contextlib.ExitStack() as stack:
                    assert _loop_nest.loops
                    loop = _loop_nest.loops[depth]
                    loop_lines = loop.lines()
                    if loop_lines is None:
                        return
                    code.writelines(loop_lines)
                    stack.enter_context(code.indent())
                    gen_loop_nest(_loop_nest, depth + 1, loop.is_reduction)

            def gen_loop_nest(
                _loop_nest: LoopNest,
                depth: int = 0,
                in_reduction: bool = False,
            ):
                if _loop_nest.loops is None or depth == len(_loop_nest.loops):  # type: ignore[arg-type]
                    gen_kernel(_loop_nest)
                else:
                    gen_loop_with_reduction(_loop_nest, depth, in_reduction)

            stack.enter_context(code.indent())

            if (
                isinstance(loop_nest.kernel, OuterLoopFusedKernel)
                and isinstance(V.local_buffer_context, LocalBufferContext)
                and V.local_buffer_context.local_buffers
            ):
                # Allocate local buffer
                local_buffers = V.local_buffer_context.local_buffers
                for local_buffer in local_buffers.values():
                    # For dynamic size, rename s to ks
                    local_buf_size = sympy_product(
                        [
                            self.rename_indexing(size_val)
                            for size_val in local_buffer.get_layout().size
                        ]
                    )
                    local_buf_dtype = DTYPE_TO_CPP[local_buffer.get_layout().dtype]
                    allocate = f"std::make_unique<{local_buf_dtype} []>({cexpr(local_buf_size)})"
                    local_buffer_name = local_buffer.get_name()
                    code.splice(
                        f"std::unique_ptr<{local_buf_dtype} []> buf_{local_buffer_name} = {allocate};"
                    )
                    code.splice(
                        f"{local_buf_dtype}* {local_buffer_name} = buf_{local_buffer_name}.get();"
                    )
            gen_loop_nest(loop_nest)

    def codegen_loops(self, code, worksharing):
        loop_nest = LoopNest.build(self)
        self.codegen_loops_impl(loop_nest, code, worksharing)

    @property
    def assert_function(self) -> str:
        if V.graph.aot_mode:
            return "AOTI_TORCH_CHECK"
        else:
            return "TORCH_CHECK"

    def decide_parallel_depth(self, max_parallel_depth, threads):
        assert self.call_ranges is not None
        ranges = self.call_ranges[
            max_parallel_depth.start_depth : (
                max_parallel_depth.start_depth + max_parallel_depth.parallel_depth
            )
        ]
        seq = self.size_hint()
        par = 1
        depth = 0
        for expr in ranges:
            hint = V.graph.sizevars.size_hint(expr, fallback=8192)
            if par >= 2 * threads or par == threads:
                break
            if seq // threads < config.cpp.min_chunk_size:
                # not enough work
                break
            depth += 1
            par *= hint
            seq /= hint
        # if we assume thread number is dynamic, make sure we
        # have at least one parallel scope and let OMP runtime
        # to manage the serial vs. parallel.
        if config.cpp.dynamic_threads and depth == 0 and len(ranges) > 0:
            depth = 1
        return ParallelDepth(
            parallel_depth=depth, start_depth=max_parallel_depth.start_depth
        )

    @contextlib.contextmanager
    def write_to_suffix(self):
        prior = (self.loads, self.compute, self.stores, self.cse)
        self.loads = IndentedBuffer()
        self.compute = IndentedBuffer()
        self.stores = IndentedBuffer()
        self.cse = self.cse.clone()
        yield
        self.reduction_suffix.splice(self.loads)
        self.reduction_suffix.splice(self.compute)
        self.reduction_suffix.splice(self.stores)
        (self.loads, self.compute, self.stores, self.cse) = prior

    def create_cse_var(self, *args, **kwargs):
        return CppCSEVariable(*args, **kwargs)

    def get_to_dtype_expr(self, src, dtype, src_dtype):
        return f"c10::convert<{DTYPE_TO_CPP[dtype]}>({src})"

    def cache_dtype_convert(self, dst, dst_dtype, src, src_dtype):
        expr = self.get_to_dtype_expr(src, dst_dtype, src_dtype)
        self.cse.put(expr, dst)

    def codegen_conditions(
        self,
        code: BracesBuffer,
        prefix: Optional[str] = None,
        var: Optional[sympy.Symbol] = None,
    ):
        if prefix is None:
            prefix = ""
        if not self.active_ranges:
            return True
        conditions = []

        def gen(start, end, var):
            if start == end:
                return False
            var_id = None
            for i, _var in enumerate(self.itervars):
                if var == _var:
                    var_id = i
                    break
            if (
                type(self) is CppKernel
                and var_id
                and start == 0
                and end == self.ranges[var_id]
            ):
                end = 1
            # pyrefly: ignore [bad-argument-type]
            conditions.append(f"{var} >= {cexpr_index(start)}")
            # pyrefly: ignore [bad-argument-type]
            conditions.append(f"{var} < {cexpr_index(end)}")
            return True

        if var is not None:
            assert var in self.active_ranges
            start, end = self.active_ranges[var]
            if not gen(start, end, var):
                return False
        else:
            for _var, _range in self.active_ranges.items():
                start, end = _range
                if not gen(start, end, _var):
                    return False
        joined_conditions = " && ".join(conditions)
        if joined_conditions:
            code.writeline(f"if({prefix}({joined_conditions}))")
            return True
        else:
            return False


class CppVecKernel(CppKernel):
    overrides = CppVecOverrides  # type: ignore[assignment]

    def __init__(
        self,
        args,
        num_threads,
        tiling_factor,
        tiling_idx,
        tail_size=None,
    ):
        super().__init__(args, num_threads)
        self.vec_isa = cpu_vec_isa.pick_vec_isa()
        assert self.vec_isa
        assert tiling_factor > 0, "Expect pass in Non-Zero tiling_factor explicitly"
        self.tiling_factor = tiling_factor
        self.tiling_idx = tiling_idx
        self.tail_size = tail_size
        self.num_elems = tail_size if tail_size else tiling_factor

    def _try_get_const_stride(self, index: sympy.Expr, itervar: sympy.Symbol):
        if self.index_indirect_depends_on(index, itervar):
            return None
        for indirect_var in (
            self.cse.varname_map[s.name]  # type: ignore[attr-defined]
            for s in index.free_symbols
            if symbol_is_type(s, SymT.TMP)
        ):
            assert isinstance(indirect_var, CppCSEVariable)
            if indirect_var.is_vec:
                return None
        stride = stride_at_vec_range(index, itervar, self.tiling_factor)
        return stride if stride.is_number else None

    def _get_num_vectors(self, dtype: torch.dtype) -> int:
        num_vectors = math.ceil(
            self.tiling_factor * dtype.itemsize * 8 / self.vec_isa.bit_width()
        )
        assert num_vectors >= 1
        return num_vectors

    def _get_raw_num_vectors(self, dtype: torch.dtype) -> float:
        # This utility function is used to check if the vector lanes has been
        # fully utilized. For example, uint8 will only use 1/4 of the vector lanes.
        return self.tiling_factor * dtype.itemsize * 8 / self.vec_isa.bit_width()

    def _get_vec_type(self, dtype: torch.dtype) -> str:
        num_vectors = self._get_num_vectors(dtype)
        if num_vectors == 1:
            return f"at::vec::Vectorized<{DTYPE_TO_CPP[dtype]}>"
        else:
            return f"at::vec::VectorizedN<{DTYPE_TO_CPP[dtype]},{num_vectors}>"

    def _get_mask_type(self, dtype: torch.dtype = torch.float) -> str:
        if dtype == torch.bool:
            return ""
        num_vectors = self._get_num_vectors(dtype)
        return f"at::vec::VecMask<{DTYPE_TO_CPP[dtype]},{num_vectors}>"

    def _get_mask_cast(self, mask: CppCSEVariable, dtype: torch.dtype) -> str:
        assert mask.dtype == torch.bool, repr(mask)
        num_vectors = self._get_num_vectors(dtype)
        return f"{mask}.template cast<{DTYPE_TO_CPP[dtype]},{num_vectors}>()"

    def _get_vec_load_line(
        self,
        var: str,
        index: sympy.Expr,
        dtype: torch.dtype,
        load_mask: Optional[CppCSEVariable] = None,
    ):
        """
        Get a load line str that loads a vector from `var` at `index` of type `dtype`.
        If `load_mask` is not None, we do a masked load accordingly.
        Notes on the `dtype`:
        1. We always load `self.tiling_factor` number of elements regardless of the `dtype`.
           It means we load half of the vector lanes for 16-bit data types and quarter of the
           vector lanes for 8-bit data types.
        2. `torch.bool` and `torch.uint8` could mean masks and we load them as float mask vectors.
        """
        cpp_type = DTYPE_TO_CPP[dtype]
        num_vectors = self._get_num_vectors(dtype)
        load_mask_str = None
        if load_mask:
            if not load_mask.is_vec:
                # TODO: avoid hard-code torch.float
                load_mask_str = f"{self._get_mask_type(torch.float)}::from({load_mask})"
            else:
                load_mask_str = f"{self._get_mask_cast(load_mask, torch.float)}"
        loadbuf = f"{var} + {cexpr_index(index)}" if index != 0 else var
        if dtype == torch.bool:
            # TODO: should we consider load mask here?
            line = f"{self._get_mask_type()}::from({loadbuf}, {cexpr_index(self.num_elems)})"
        else:
            line = (
                f"{load_mask_str}.template loadu<{cpp_type},{num_vectors}>({loadbuf})"
                if load_mask_str
                else f"{self._get_vec_type(dtype)}::loadu({loadbuf}, {cexpr_index(self.num_elems)})"
            )
        return line

    def _load_or_store_non_contiguous(
        self,
        var: Optional[str],
        index: sympy.Expr,
        dtype: torch.dtype,
        buffer: Optional[IndentedBuffer] = None,
        store_value: Optional[Union[str, CppCSEVariable]] = None,
        accu_store: bool = False,
    ) -> Optional[CppCSEVariable]:
        """
        Load or store a vector in a non-contiguous way. The vector is initialized from an array that is
        filled in an inner loop over the tiling factor.
        :param var: buffer to load from or store to, i.e. `var[transformed(index)]`. If None, we load the index
                    as index expression, i.e. `transformed(index)`.
        :param index: index into the `var` or the index expression by its own if `var` is None.
                      The `index` could contain indirect indexing or the tiling itervar. When used in
                      the inner loop, the index is transformed as follows:
                      1. the index is linearized along the tiling dim.
                      2. the indirect indexing vector variables are transformed into arrays over the tiling dim.
        :param dtype: data type of `var` or `index` if `var` is None.
        :param buffer: the code buffer to write the generated code to. If None, we write to `self.loads`.
        :param store_value: the value to store. If None, we load the vector.
        :param accu_store: whether accumulate the store_value to store_ptr. If True, a store_value should be provided
        :return: a CppCSEVariable that represents the loaded vector or None if it is a store.
        """
        assert not store_value or var is not None, "store var must be provided"
        if accu_store:
            assert store_value
        if buffer is None:
            buffer = self.loads

        def get_result_size(dtype: torch.dtype) -> int:
            if dtype.itemsize < 4:
                return self.num_elems * (4 // dtype.itemsize)
            else:
                return self.num_elems

        def get_tiling_size(dtype: torch.dtype) -> int:
            if dtype.itemsize < 4:
                return self.tiling_factor * (4 // dtype.itemsize)
            else:
                return self.tiling_factor

        def vec_to_array(vec_var: CppCSEVariable) -> CppCSEVariable:
            assert vec_var.is_vec
            code = BracesBuffer()
            code.writeline("[&]")
            with code.indent():
                vec_dtype = vec_var.dtype
                assert vec_dtype is not None
                if vec_dtype == torch.bool:
                    vec_dtype = torch.float
                result_size = get_result_size(vec_dtype)
                tiling_size = get_tiling_size(vec_dtype)
                code.writeline(
                    f"__at_align__ std::array<{DTYPE_TO_CPP[vec_dtype]}, {tiling_size}> tmpbuf;"
                )
                line = f"{vec_var}.store(tmpbuf.data(), {cexpr_index(result_size)});"
                code.writeline(line)
                code.writeline("return tmpbuf;")
            code.writeline("()")
            csevar = self.cse.generate(buffer, code)
            assert isinstance(csevar, CppCSEVariable)
            return csevar

        code = BracesBuffer()
        code.writeline("[&]")
        with code.indent():
            result_size = get_result_size(dtype)
            tiling_size = get_tiling_size(dtype)
            result_declare = (
                f"__at_align__ std::array<{DTYPE_TO_CPP[dtype]}, {tiling_size}> tmpbuf;"
            )
            code.writeline(result_declare)
            if store_value:
                code.writeline(
                    f"{store_value}.store(tmpbuf.data(), {cexpr_index(result_size)});"
                )
            itervar_inner = sympy_index_symbol(
                f"{self.itervars[self.tiling_idx]}_inner"
            )
            replacements = {}
            for indirect_var in (
                self.cse.varname_map[s.name]  # type: ignore[attr-defined]
                for s in index.free_symbols
                if symbol_is_type(s, SymT.TMP)
            ):
                assert isinstance(indirect_var, CppCSEVariable)
                if indirect_var.is_vec:
                    array_var = vec_to_array(indirect_var)
                    replacements[indirect_var] = f"{array_var}[{itervar_inner}]"
            index = self.scale_index_with_offset(
                index, itervar_idx=self.tiling_idx, offset=itervar_inner
            )
            load_mask = None
            if self._load_mask is not None:
                assert not store_value, "unexpected store with load mask"
                assert isinstance(self._load_mask, CppCSEVariable), self._load_mask
                if self._load_mask.is_vec:
                    load_mask = f"{self._load_mask}.is_masked({itervar_inner})"
                else:
                    load_mask = f"{self._load_mask} != 0"
            if cpp_builder.is_gcc():
                code.writeline(f"#pragma GCC unroll {self.tiling_factor}")
            else:
                code.writeline(f"#pragma unroll {self.tiling_factor}")
            code.writeline(
                f"for (long {itervar_inner} = 0; "
                + f"{itervar_inner} < {cexpr_index(self.num_elems)}; "
                + f"{itervar_inner}++)"
            )
            with code.indent(), contextlib.ExitStack() as stack:
                index_c = cexpr_index(index)
                # pyrefly: ignore [bad-assignment]
                for indirect_var in replacements:
                    index_c = re.sub(
                        r"\b" + f"{indirect_var}" + r"\b",
                        replacements[indirect_var],
                        index_c,
                    )
                rhs = f"{var}[{index_c}]" if var is not None else f"{index_c}"
                if load_mask:
                    code.writeline(f"if ({load_mask})")
                    stack.enter_context(code.indent())
                if store_value:
                    conjunction = "+=" if accu_store else "="
                    code.writeline(f"{rhs} {conjunction} tmpbuf[{itervar_inner}];")
                else:
                    code.writeline(f"tmpbuf[{itervar_inner}] = {rhs};")
            if not store_value:
                load_line = self._get_vec_load_line("tmpbuf.data()", 0, dtype)  # type: ignore[arg-type]
                code.writeline(f"return {load_line};")
        code.writeline("()")
        if store_value:
            code.writeline(";")
            buffer.splice(code)
            return None
        else:
            csevar = self.cse.generate(buffer, code, dtype=dtype)
            assert isinstance(csevar, CppCSEVariable)
            csevar.is_vec = True
            return csevar

    def load(self, name: str, index: sympy.Expr):
        var = self.args.input(name)
        index = self.rename_indexing(index)
        dtype = V.graph.get_dtype(name)
        tiling_var = self.itervars[self.tiling_idx]
        stride = self._try_get_const_stride(index, tiling_var)
        if stride == 0:
            # load scalar and lazily broadcast it on demand
            return super().load(name, index)
        elif stride == 1:
            # load contiguously
            line = self._get_vec_load_line(var, index, dtype, self._load_mask)  # type: ignore[arg-type]
            csevar = self.cse.generate(self.loads, line, dtype=dtype)  # type: ignore[assignment]
        else:
            csevar = self._load_or_store_non_contiguous(var, index, dtype)  # type: ignore[assignment]
        assert isinstance(csevar, CppCSEVariable)
        csevar.update_on_args("load", (self, name, index), {})
        csevar.is_vec = True
        return csevar

    def _get_store_line(
        self,
        value: Union[str, CppCSEVariable],
        var: str,
        index: sympy.Expr,
        dtype: torch.dtype,
        accu_store: bool = False,
    ):
        """
        Get a store line buffer that stores `value` into `var` at `index` of `dtype`. It handles
        both contiguous and non-contiguous store cases.
        :param value: Vectorized type templaterized on `dtype`.
        :param var: buffer to store into.
        :index: index into the `var`.
        """
        # when value's type is str (e.g., welford reduction), caller should make sure
        # it is a vector
        assert isinstance(value, str) or (
            isinstance(value, CppCSEVariable) and value.is_vec
        ), value
        tiling_var = self.itervars[self.tiling_idx]
        var_expr = f"{var} + {cexpr_index(index)}"
        stride = self._try_get_const_stride(index, tiling_var)
        code = IndentedBuffer()
        if stride == 1:
            if accu_store:
                load = (
                    f"{self._get_vec_type(dtype)}::loadu({var_expr})"
                    if dtype == torch.float and self.tail_size is None
                    else f"{self._get_vec_type(dtype)}::loadu({var_expr}, {cexpr_index(self.num_elems)})"
                )
                value = f"({value} + {load})"
            if dtype == torch.float and self.tail_size is None:
                code.writeline(f"{value}.store({var_expr});")
            else:
                code.writeline(
                    f"{value}.store({var_expr}, {cexpr_index(self.num_elems)});"
                )
        else:
            self._load_or_store_non_contiguous(
                var, index, dtype, buffer=code, store_value=value, accu_store=accu_store
            )
        return code

    def store(self, name, index, value, mode=None):
        assert "buf" in name
        assert isinstance(value, CppCSEVariable), value
        if not value.is_vec:
            # this happens when we store a scalar into a vectorized buffer like "fill"
            value = self.broadcast(value)
        var = self.args.output(name)
        index = self.rename_indexing(index)
        dtype = V.graph.get_dtype(name)
        if mode is None:
            code = self._get_store_line(value, var, index, dtype)
            self.stores.splice(code.map(lambda x: DeferredLine(name, x)))
        elif mode == "atomic_add":
            if not config.cpp.dynamic_threads and self.num_threads == 1:
                code = self._get_store_line(
                    f"{value}",
                    var,
                    index,
                    dtype,
                    accu_store=True,
                )
                self.stores.splice(code.map(lambda x: DeferredLine(name, x)))
            else:
                n_src = self._get_num_vectors(dtype)
                n_idx = self._get_num_vectors(torch.int64)
                cdtype = DTYPE_TO_CPP[dtype]
                index = ops.index_expr(index, torch.int64).value
                assert isinstance(index, CppCSEVariable) and index.is_vec
                if self.tail_size:
                    line = f"atomic_add_vec<{cdtype}, {n_idx}, {n_src}>({var}, {index}, {value}, {cexpr_index(self.tail_size)});"
                else:
                    line = f"atomic_add_vec<{cdtype}, {n_idx}, {n_src}>({var}, {index}, {value});"
                self.stores.writeline(DeferredLine(name, line))
        else:
            raise NotImplementedError(f"store mode={mode}")

    def reduction(self, dtype, src_dtype, reduction_type, value):
        """
        Perform vectorized reduction operation.

        This method handles vectorized reduction for different reduction types.
        It manages special cases for low-precision floating point types and
        employs precision improvement techniques for certain reduction operations.

        Args:
            dtype: The output data type for the reduction result
            src_dtype: The source data type of the input value
            reduction_type: Type of reduction operation (sum, min, max, etc.)
            value: The input value to reduce

        Returns:
            The result of the reduction operation
        """
        # Note: For argmax and argmin on bool type, we always convert bool to float.
        # Fix issue: https://github.com/pytorch/pytorch/issues/143568
        assert reduction_type in VECTORIZABLE_RTYPES
        argmax_or_argmin = reduction_type in ("argmax", "argmin")
        horizontal_reduction = self.tiling_idx >= self.reduction_depth
        init_dtype = src_dtype if argmax_or_argmin else dtype
        assert isinstance(value, CppCSEVariable), value

        if not value.is_vec:
            value = self.broadcast(value)

        reduction_key = src_dtype, reduction_type, value
        if reduction_key in self.reduction_cse.reduction_cache:
            return self.reduction_cse.reduction_cache[reduction_key]

        vec_ns = "at::vec"
        vec = f"{vec_ns}::Vectorized<{DTYPE_TO_CPP[dtype]}>"
        acc_type = reduction_acc_type(reduction_type, init_dtype)
        acc_type_vec = self.reduction_acc_type_vec(reduction_type, init_dtype)

        acc = self.reduction_cse.generate(
            self.loads, f"reduction {reduction_key}", write=False
        )
        assert isinstance(acc, CppCSEVariable)
        acc_vec = f"{acc}_vec"
        masked_acc = f"masked_{acc}"
        masked_acc_vec = f"masked_{acc_vec}"
        self.reduction_var_names += [f"{acc}", acc_vec, masked_acc_vec]
        self.is_reduction = True
        self.reduction_prefix_generators.append(
            self._gen_reduction_prefix(
                acc, acc_type, reduction_type, init_dtype, reduction_init
            )
        )
        self.reduction_prefix_generators.append(
            self._gen_reduction_prefix(
                acc_vec,
                acc_type_vec,
                reduction_type,
                init_dtype,
                self.reduction_init_vec,
            )
        )

        use_acc_helper = self.need_use_acc_helper(reduction_type, dtype, False)
        if use_acc_helper:
            # use masked acc_vec for tail vec kernel
            self.reduction_prefix_generators.append(
                self._gen_reduction_prefix(
                    masked_acc_vec,
                    acc_type_vec,
                    reduction_type,
                    dtype,
                    self.reduction_init_vec,
                )
            )

            # use welford_helper/cascade_helper for vec kernel
            assert self.reduction_depth is not None
            reduction_size = functools.reduce(
                operator.mul, self.ranges[self.reduction_depth :]
            )
            if reduction_type == "welford_reduce":
                helper_val = self.welford_helper_cse.generate(
                    self.compute, f"reduction {reduction_key}", write=False
                )
            else:
                helper_val = self.cascade_helper_cse.generate(
                    self.compute, f"reduction {reduction_key}", write=False
                )
            masked_helper_val = f"masked_{helper_val}"
            helper_vec_range = (
                (
                    FloorDiv(reduction_size, self.ranges[self.tiling_idx])
                    * FloorDiv(self.ranges[self.tiling_idx], self.tiling_factor)
                    if self.tiling_idx >= self.reduction_depth
                    else reduction_size
                )
                if FloorDiv(self.ranges[self.tiling_idx], self.tiling_factor)
                else sympy.Integer(0)
            )
            masked_helper_vec_range = (
                (
                    FloorDiv(reduction_size, self.ranges[self.tiling_idx])
                    if self.tiling_idx >= self.reduction_depth
                    else reduction_size
                )
                if self.ranges[self.tiling_idx] % self.tiling_factor
                else sympy.Integer(0)
            )
            # scalar helper for scalar welford_reduce/sum is also needed when vec kernel is included
            scalar_helper_val = f"scalar_{helper_val}"
            self._use_acc_helper(
                reduction_type,
                acc,
                scalar_helper_val,
                reduction_size,
                dtype,
                use_scalar=True,
            )
            self._use_acc_helper(
                reduction_type, acc, helper_val, helper_vec_range, dtype
            )
            self._use_acc_helper(
                reduction_type,
                masked_acc,
                masked_helper_val,
                masked_helper_vec_range,
                dtype,
            )

            # use masked acc_vec for tail vec kernel
            acc_vec_ = masked_acc_vec if self.tail_size else acc_vec
            helper_val_ = masked_helper_val if self.tail_size else helper_val
            if reduction_type == "sum":
                self.stores.writeline(
                    f"{acc_vec_} = {self.reduction_combine_vec(reduction_type, acc_vec_, value, helper_val_)};"
                )
            else:
                self.stores.writeline(
                    f"{acc_vec_} = {self.reduction_combine_vec(reduction_type, acc_vec_, value, helper_val_)};"
                )
        else:
            assert self.reduction_depth is not None
            index = self.itervars[self.reduction_depth]
            for i in range(self.reduction_depth + 1, len(self.itervars)):
                index = index * self.ranges[i] + self.itervars[i]
            kwargs = {
                "next_value": value,
                "index": index,
                "horizontal_reduction": horizontal_reduction,
                "src_dtype": src_dtype,
            }
            self.stores.writeline(
                f"{acc_vec} = {self.reduction_combine_vec(reduction_type, acc_vec, **kwargs)};"
            )
        self._gen_parallel_reduction_buffers(
            acc_vec,
            acc_type_vec,
            reduction_type,
            init_dtype,
            reduction_combine_fn=self.reduction_combine_vec,
            reduction_init_fn=self.reduction_init_vec,
        )
        self._gen_parallel_reduction_buffers(
            acc,
            acc_type,
            reduction_type,
            init_dtype,
            reduction_combine_fn=reduction_combine,
            reduction_init_fn=reduction_init,
        )
        if use_acc_helper:
            # use masked acc_vec for tail vec kernel
            self._gen_parallel_reduction_buffers(
                masked_acc_vec,
                acc_type_vec,
                reduction_type,
                dtype,
                reduction_combine_fn=self.reduction_combine_vec,
                reduction_init_fn=self.reduction_init_vec,
            )
        tmpvar: Union[str, CSEVariable]
        is_bool = dtype == torch.bool
        if horizontal_reduction:
            # Horizontal reduction
            if is_welford_reduction(reduction_type):
                assert self._get_num_vectors(dtype) in [
                    1,
                    2,
                ], "Welford reduction does not support VectorizedN (N>2)"
                next_value = f"welford_vec_reduce_all({acc_vec})"
                masked_next_value = f"welford_vec_reduce_all({masked_acc_vec})"
                self.reduction_suffix.writeline(
                    f"{acc} = {reduction_combine(reduction_type, acc, masked_next_value)};"
                )
            elif argmax_or_argmin:
                next_value = f"{reduction_type}_vec_reduce_all({acc_vec})"
            elif is_bool:
                if reduction_type in (
                    "any",
                    "sum",
                    "max",
                ):
                    next_value = f"!{acc_vec}.all_zero()"
                else:
                    assert reduction_type == "min"
                    next_value = f"{acc_vec}.all_masked()"
            else:
                reduce_all_body = (
                    "{ return "
                    + self.reduction_combine_vec(reduction_type, "x", "y")
                    + "; }"
                )
                is_bool = dtype == torch.bool
                # we are using at::vec::VecMask<float, N> for bool
                vec_dtype = torch.float if is_bool else dtype
                vec = f"at::vec::Vectorized<{DTYPE_TO_CPP[vec_dtype]}>"
                vec_reduce_all_func = f"at::vec::vec_reduce_all<{DTYPE_TO_CPP[vec_dtype]}, {self._get_num_vectors(vec_dtype)}>"
                result_vec = f"{acc_vec}"
                if use_acc_helper:
                    assert reduction_type == "sum"
                    result_vec = f"{acc_vec} + {masked_acc_vec}"
                next_value = f"{vec_reduce_all_func}([]({vec}& x, {vec}& y) {reduce_all_body}, {result_vec})"

            self.reduction_suffix.writeline(
                f"{acc} = {reduction_combine(reduction_type, acc, next_value, src_dtype=src_dtype)};"
            )
            tmpvar = acc
        else:
            tmpvar = acc_vec
            if is_welford_reduction(reduction_type):
                masked_tmpvar = f"masked_{tmpvar}"
                self.reduction_suffix.writeline(
                    f"{tmpvar} = {reduction_combine(reduction_type, tmpvar, masked_tmpvar)};"
                )
            elif use_acc_helper:
                assert reduction_type == "sum"
                masked_tmpvar = f"masked_{tmpvar}"
                self.reduction_suffix.writeline(
                    f"{tmpvar} = {tmpvar} + {masked_tmpvar};"
                )

        result = reduction_project(reduction_type, tmpvar)
        self.reduction_cse.reduction_cache[reduction_key] = result
        return result

    def store_reduction(self, name, index, value):
        index = self.rename_indexing(index)
        var = self.args.output(name)
        out_dtype = V.graph.get_dtype(name)
        if out_dtype.is_floating_point and out_dtype != torch.double:
            dtype = torch.float
        else:
            dtype = out_dtype
        out_num_vectors = V.kernel._get_num_vectors(out_dtype)
        src_num_vectors = V.kernel._get_num_vectors(dtype)
        code = IndentedBuffer()
        if self.tiling_idx >= self.reduction_depth:
            # Horizontal reduction
            code.writeline(
                f"{var}[{cexpr_index(index)}] = static_cast<{DTYPE_TO_CPP[out_dtype]}>({value});"
            )
        else:
            # Vertical reduction
            if out_dtype != dtype:
                converted_value = (
                    f"{DTYPE_TO_CPP[out_dtype].replace('::', '_')}_{value}"
                )
                if out_dtype == torch.bool:
                    convert = f"{value}.template cast<bool,{self._get_num_vectors(torch.bool)}>()"
                else:
                    if src_num_vectors == out_num_vectors == 1:
                        convert = (
                            f"at::vec::convert<{DTYPE_TO_CPP[out_dtype]}>({value})"
                        )
                    else:
                        convert = (
                            f"at::vec::convert<{DTYPE_TO_CPP[out_dtype]},"
                            f"{out_num_vectors},{DTYPE_TO_CPP[dtype]},{src_num_vectors}>({value})"
                        )
                code.writeline(f"auto {converted_value} = {convert};")
                value = converted_value
            code.splice(self._get_store_line(value, var, index, out_dtype))
        self.reduction_suffix.splice(code.map(lambda x: DeferredLine(name, x)))

    def broadcast(self, scalar_var: CppCSEVariable) -> CppCSEVariable:
        assert not scalar_var.is_vec
        if scalar_var.dtype == torch.bool:
            vec_var = self.cse.generate(
                self.compute, f"{self._get_mask_type()}::from({scalar_var.name})"
            )
        else:
            assert scalar_var.dtype is not None
            vec_var = self.cse.generate(
                self.compute,
                f"{self._get_vec_type(scalar_var.dtype)}({scalar_var.name})",
            )
        assert isinstance(vec_var, CppCSEVariable)
        vec_var.dtype = scalar_var.dtype
        vec_var.dependent_itervars = scalar_var.dependent_itervars
        vec_var.is_vec = True
        return vec_var

    def arange(self, index: CppCSEVariable, stride: sympy.Symbol) -> CppCSEVariable:
        assert not index.is_vec
        assert index.dtype is not None
        csevar = self.cse.generate(
            self.compute,
            f"{self._get_vec_type(index.dtype)}::arange({index}, {stride})",
        )
        assert isinstance(csevar, CppCSEVariable)
        csevar.dtype = index.dtype
        csevar.is_vec = True
        return csevar

    def reduction_init_vec(self, reduction_type, dtype):
        scalar_type = DTYPE_TO_COMPUTATION_DTYPE[dtype]
        vec_type = self._get_vec_type(scalar_type)

        if is_welford_reduction(reduction_type):
            return f"Welford<{vec_type}>()"

        if reduction_type in ("argmin", "argmax"):
            cdtype = DTYPE_TO_CPP[scalar_type]
            acc_type = self.reduction_acc_type_vec(reduction_type, dtype)
            if reduction_type == "argmin":
                val = (
                    f"std::numeric_limits<{cdtype}>::infinity()"
                    if is_float_dtype(dtype)
                    else f"std::numeric_limits<{cdtype}>::max()"
                )
            else:
                val = (
                    f"-std::numeric_limits<{cdtype}>::infinity()"
                    if is_float_dtype(dtype)
                    else f"std::numeric_limits<{cdtype}>::min()"
                )
            return f"{acc_type}({val})"

        if reduction_type == "any":
            return f"{self._get_mask_type()}::from(0)"

        scalar_init = reduction_init(reduction_type, dtype)
        vec_init = f"{vec_type}({scalar_init})"
        if dtype == torch.bool:
            assert reduction_type in ("min", "max", "sum")
            return f"{self._get_mask_type()}::from({scalar_init})"
        return vec_init

    def reduction_acc_type_vec(self, reduction_type, dtype):
        scalar_type = DTYPE_TO_COMPUTATION_DTYPE[dtype]
        vec_type = self._get_vec_type(scalar_type)
        if is_welford_reduction(reduction_type):
            return f"Welford<{vec_type}>"
        if reduction_type in ("argmin", "argmax"):
            n_src = self._get_num_vectors(scalar_type)
            n_idx = self._get_num_vectors(torch.int64)
            if dtype == torch.bool:
                return f"IndexValueVec<{DTYPE_TO_CPP[torch.float]}, {n_src}, {n_idx}>"
            return f"IndexValueVec<{DTYPE_TO_CPP[scalar_type]}, {n_src}, {n_idx}>"
        if dtype == torch.bool:
            assert reduction_type in ("min", "max", "any", "sum")
            return f"{self._get_mask_type()}"
        return vec_type

    def reduction_combine_vec(
        self,
        reduction_type,
        var,
        next_value,
        helper_val=None,
        index: Optional[sympy.Symbol] = None,
        horizontal_reduction: Optional[bool] = None,
        src_dtype: Optional[torch.dtype] = torch.float32,
    ):
        is_bool = src_dtype == torch.bool
        if reduction_type == "max":
            if self.tail_size:
                return f"max_masked_reduce({var}, {next_value}, {cexpr_index(self.tail_size)})"
            else:
                return (
                    f"{var} | {next_value}"
                    if is_bool
                    else f"at::vec::maximum({var}, {next_value})"
                )
        elif reduction_type == "min":
            if self.tail_size:
                return f"min_masked_reduce({var}, {next_value}, {cexpr_index(self.tail_size)})"
            else:
                return (
                    f"{var} & {next_value}"
                    if is_bool
                    else f"at::vec::minimum({var}, {next_value})"
                )
        elif reduction_type == "sum":
            if helper_val:
                if self.tail_size:
                    return f"cascade_sum_combine({next_value}, {cexpr_index(self.tail_size)}, &{helper_val})"
                else:
                    return f"cascade_sum_combine({next_value}, &{helper_val})"
            else:
                if self.tail_size:
                    return f"sum_masked_reduce({var}, {next_value}, {cexpr_index(self.tail_size)})"
                else:
                    conjunction = "|" if is_bool else "+"
                    return f"{var} {conjunction} {next_value}"
        elif reduction_type == "prod":
            if self.tail_size:
                return f"prod_masked_reduce({var}, {next_value}, {cexpr_index(self.tail_size)})"
            else:
                return f"{var} * {next_value}"
        elif reduction_type == "xor_sum":
            if self.tail_size:
                return f"xor_sum_masked_reduce({var}, {next_value}, {cexpr_index(self.tail_size)})"
            else:
                return f"{var} ^ {next_value}"
        elif reduction_type == "welford_reduce":
            if helper_val:
                if self.tail_size:
                    return f"welford_combine({var}, {next_value}, {cexpr_index(self.tail_size)}, &{helper_val})"
                else:
                    return f"welford_combine({var}, {next_value}, &{helper_val})"
            else:
                if self.tail_size:
                    return f"welford_combine({var}, {next_value}, {cexpr_index(self.tail_size)})"
                else:
                    return f"welford_combine({var}, {next_value})"
        elif reduction_type == "welford_combine":
            if isinstance(next_value, tuple):
                # When reading a value from Inductor IR we have a tuple of variable names
                mean, m2, weight = next_value
            else:
                # When combining intermediate accumulators we have a Welford<T> struct
                mean, m2, weight = reduction_project(reduction_type, next_value)
            if self.tail_size:
                return f"welford_combine({var}, {{{mean}, {m2}, {weight}}}, {cexpr_index(self.tail_size)})"
            else:
                return f"welford_combine({var}, {{{mean}, {m2}, {weight}}})"
        elif reduction_type in ("argmin", "argmax"):
            assert src_dtype is not None
            cdtype = DTYPE_TO_CPP[src_dtype]
            if src_dtype == torch.bool:
                cdtype = DTYPE_TO_CPP[torch.float]
            n_src = self._get_num_vectors(src_dtype)
            n_idx = self._get_num_vectors(torch.int64)
            t_extra = ""
            arg_extra = ""
            if index is not None:
                assert horizontal_reduction is not None
                t_extra = f", {str(horizontal_reduction).lower()}"
                arg_extra = f", {index}"
            if self.tail_size:
                return (
                    f"{reduction_type}_combine_vec<{cdtype}, {n_src}, {n_idx}{t_extra}>"
                    f"({var}, {next_value}{arg_extra}, {cexpr_index(self.tail_size)})"
                )
            else:
                return f"{reduction_type}_combine_vec<{cdtype}, {n_src}, {n_idx}{t_extra}>({var}, {next_value}{arg_extra})"
        elif reduction_type == "any":
            if isinstance(next_value, CppCSEVariable):
                assert next_value.dtype == torch.bool
                (next_value,) = unify_mask_base_type(V.kernel.compute, (next_value,))
            if self.tail_size:
                return f"any_masked_reduce({var}, {next_value}, {cexpr_index(self.tail_size)})"
            else:
                return f"{var} | {next_value}"
        else:
            raise NotImplementedError

    def indirect_assert(self, var, lower, upper, mask=None):
        assert isinstance(var, CppCSEVariable)
        assert var.dtype is not None
        if not var.is_vec:
            if isinstance(mask, CppCSEVariable) and mask.is_vec:
                mask = f"({mask}).all_masked()"
            return super().indirect_assert(var, lower, upper, mask)
        lower_scalar = lower
        upper_scalar = upper
        if lower:
            lower = f"{self._get_vec_type(var.dtype)}({lower})"
        if upper:
            upper = f"{self._get_vec_type(var.dtype)}({upper})"
        if lower and upper:
            cond = f"({lower} <= {var}) & ({var} < {upper})"
            cond_print = f"{lower_scalar} <= {var} < {upper_scalar}"
        elif lower:
            cond = f"{lower} <= {var}"
            cond_print = f"{lower_scalar} <= {var}"
        else:
            assert upper
            cond = f"{var} < {upper}"
            cond_print = f"{var} < {upper_scalar}"
        cond = f"{self._get_mask_type(var.dtype)}({cond})"
        if mask:
            if not mask.is_vec:
                mask = f"{self._get_mask_type(var.dtype)}({mask})"
            # We need not check when the mask is False
            cond = f"({cond}) | ~({mask})"
        if self.tail_size:
            cond = (
                f"{self._get_mask_type(var.dtype)}::set({self._get_mask_type(var.dtype)}::from(1)"
                f", ({cond}), {cexpr_index(self.tail_size)})"
            )
        cond = f"({cond}).all_masked()"
        return f'{self.assert_function}({cond}, "index out of bounds: {cond_print}")'

    def get_to_dtype_expr(self, src, dtype, src_dtype):
        assert isinstance(src, CppCSEVariable)
        if not src.is_vec:
            return super().get_to_dtype_expr(src, dtype, src_dtype)
        src_cpp_type = DTYPE_TO_CPP[src_dtype]
        src_num_vectors = self._get_num_vectors(src_dtype)
        dst_cpp_type = DTYPE_TO_CPP[dtype]
        dst_num_vectors = self._get_num_vectors(dtype)
        expr = f"({src})"
        if src_dtype != torch.bool and dtype == torch.bool:
            expr = f"{self._get_mask_type(src_dtype)}::from<{src_cpp_type},{src_num_vectors}>({src})"
        elif src_dtype == torch.bool and dtype != torch.bool:
            expr = f"{src}.to<{dst_cpp_type},{dst_num_vectors}>()"
        elif src_dtype != dtype:
            if src_num_vectors == dst_num_vectors == 1:
                expr = f"at::vec::convert<{dst_cpp_type}>({src})"
            else:
                expr = f"at::vec::convert<{dst_cpp_type},{dst_num_vectors},{src_cpp_type},{src_num_vectors}>({src})"
        return expr


class CppTile2DKernel(CppVecKernel):
    """
    A vector kernel that handles the 2d tiles with the tile size defined in `tiling_factor` on
    the inner-most loop level and one of the outer loop level (`outer_tiling_idx`). When the data
    tile is accessed in a contiguous way from the outer loop axis, a transposition is applied on the
    tile to make the access contiguous from the inner-most loop axis. Then, the same vectorization
    logic from its parent `CppVecKernel` is leveraged for load/store/compute. The transposed tile load
    and store are generated into kernel.preloads and kernel.poststores buffers.

    The loop structure looks like below:
    for ...
      for i_outer ...
        for ...
          for inner_most ...
            // generated by CppTile2DKernel
            float tmp0[16*16]; at::vec::transpose_mxn<...>(tmp0, in_ptr0 + ..., ...); // into kernel.preloads
            float tmp1[16*16]; // into kernel.preloads
            for i_inner ... { // the kernel inner loop
              vectorized loads/compute/stores (e.g., load tmp0, store tmp1) // into kernel.loads/compute/stores
            }
            at::vec::transpose_mxn(out_ptr0 + ..., tmp1, ...) // into kernel.poststores
          for inner_most ... (tail)
            // generated by CppVecKernel
            ...
      for i_outer ... (tail)
        for ...
          for ...
            // generated by CppKernel
            ...
    """

    overrides = CppTile2DOverrides  # type: ignore[assignment]

    def __init__(
        self,
        args,
        num_threads,
        tiling_factor,
        tiling_indices,
        inner_tail_size=None,
        outer_tail_size=None,
    ):
        super().__init__(
            args,
            num_threads,
            tiling_factor,
            tiling_indices[1],
            inner_tail_size,
        )
        self.tiling_indices = tiling_indices
        self.inner_tail_size = inner_tail_size
        self.outer_tail_size = outer_tail_size
        self.inner_num_elems = inner_tail_size if inner_tail_size else tiling_factor
        self.outer_num_elems = outer_tail_size if outer_tail_size else tiling_factor
        self.inner_is_tiling_idx = True

    def inner_itervar(self):
        return sympy_index_symbol(f"{self.itervars[self.outer_idx]}_inner")

    def need_vec_transpose(self, index):
        outer_var = self.itervars[self.outer_idx]
        inner_var = self.itervars[self.tiling_idx]
        outer_stride = stride_at_vec_range(index, outer_var, self.tiling_factor)
        inner_stride = stride_at_vec_range(index, inner_var, self.tiling_factor)
        return (
            self._load_mask is None  # TODO: support transposition with mask
            and outer_stride == 1
            and index.has(inner_var)
            and not inner_stride.has(inner_var)
            and not inner_stride.has(outer_var)
        )

    def gen_transposed_tile_load_store(
        self, name, var, index, is_store, store_mode=None
    ):
        # transposed tile load/store outside the kernel inner loop
        dtype = V.graph.get_dtype(name)
        factor = self.tiling_factor
        src = f"{var} + {cexpr_index(index)}"
        dst = "__place_holder__"
        ld_src = f"{cexpr_index(stride_at_vec_range(index, self.itervars[self.tiling_idx], self.tiling_factor))}"
        ld_dst = f"{cexpr_index(self.num_elems)}"
        if is_store:
            src, dst = dst, src
            ld_src, ld_dst = ld_dst, ld_src

        need_define = True
        if self.inner_is_tiling_idx ^ is_store:
            M, N = self.inner_num_elems, self.outer_num_elems
        else:
            M, N = (
                self.outer_num_elems,
                self.inner_num_elems,
            )
        atomic_add = "true" if (is_store and (store_mode == "atomic_add")) else "false"
        if (isinstance(M, sympy.Expr) and not M.is_number) or (
            isinstance(N, sympy.Expr) and not N.is_number
        ):
            load_or_store = (
                f"transpose_mxn<{DTYPE_TO_CPP[dtype]},{atomic_add}>"
                f"({src}, {ld_src}, {dst}, {ld_dst}, {cexpr_index(M)}, {cexpr_index(N)});"
            )
        else:
            load_or_store = (
                f"transpose_mxn<{DTYPE_TO_CPP[dtype]},{cexpr_index(M)},{cexpr_index(N)},{atomic_add}>"
                f"({src}, {ld_src}, {dst}, {ld_dst});"
            )
        if is_store:
            tile_var = self.cse.newvar()
        elif not self.cse.contains(load_or_store):
            tile_var = self.cse.generate(self.preloads, load_or_store, write=False)
        else:
            need_define = False
            tile_var = self.cse.get(load_or_store)

        if need_define:
            cpp_dtype = DTYPE_TO_CPP[dtype]
            # tiling_factor might be smaller than the alignment of cpp_dtype, such as
            # with a vector that only holds 4 elements due to NEON 128-bit vectors and
            # cpp_dtype being a 64-bit integer.
            alignas = f"alignas(std::max(std::size_t({factor}), alignof({cpp_dtype})))"
            define_line = f"{alignas} {cpp_dtype} {tile_var}[{factor}*{factor}];"
            self.preloads.writeline(define_line)

        load_or_store = load_or_store.replace("__place_holder__", str(tile_var))
        if is_store:
            self.poststores.writeline(DeferredLine(name, load_or_store))
        else:
            self.preloads.writeline(load_or_store)

        return tile_var

    def load(self, name: str, index: sympy.Expr):
        var = self.args.input(name)
        index = self.rename_indexing(index)

        inner = self.inner_itervar()
        if self.need_vec_transpose(index):
            tile_var = self.gen_transposed_tile_load_store(
                name, var, index, is_store=False
            )
            # vector load inside the kernel inner loop
            loadbuf = f"{tile_var} + {cexpr_index(inner * self.num_elems)}"
            dtype = V.graph.get_dtype(name)
            line = self._get_vec_load_line(loadbuf, 0, dtype)  # type: ignore[arg-type]
            csevar = self.cse.generate(self.loads, line, dtype=dtype)
            csevar.update_on_args("load", (self, name, index), {})
            assert isinstance(csevar, CppCSEVariable)
            csevar.is_vec = True
            return csevar
        else:
            new_index = self.transform_indexing(index)
            return super().load(name, new_index)

    def store(self, name, index, value, mode=None):
        assert "buf" in name
        assert isinstance(value, CppCSEVariable), value
        if not value.is_vec:
            # this happens when we store a scalar into a vectorized buffer like "fill"
            value = self.broadcast(value)

        var = self.args.output(name)

        inner = self.inner_itervar()
        index = self.rename_indexing(index)
        if self.need_vec_transpose(index):
            tile_var = self.gen_transposed_tile_load_store(
                name, var, index, is_store=True, store_mode=mode
            )
            # vector store inside the kernel inner loop
            storebuf = f"{tile_var} + {cexpr_index(inner * self.num_elems)}"
            if self.tail_size or V.graph.get_dtype(name) in DTYPE_LOWP_FP + [
                torch.uint8,
                torch.int8,
                torch.float8_e4m3fn,
                torch.float8_e5m2,
            ]:
                line = f"{value}.store({storebuf}, {cexpr_index(self.num_elems)});"
            else:
                line = f"{value}.store({storebuf});"
            self.stores.writeline(DeferredLine(name, line))
        else:
            new_index = self.transform_indexing(index)
            super().store(name, new_index, value, mode)

    def codegen_inner_loops(self, code):
        inner = self.inner_itervar()
        if self.inner_is_tiling_idx:
            code.writeline(
                f"for (long {inner} = 0; {inner} < {cexpr_index(self.outer_num_elems)}; {inner}++)"
            )
        else:
            code.writeline(
                f"for (long {inner} = 0; {inner} < {cexpr_index(self.inner_num_elems)}; {inner}++)"
            )

    def set_ranges(self, group, reduction_group):
        vars = super().set_ranges(group, reduction_group)
        # do vertical reduction as the tail loop
        self.outer_idx, self.tiling_idx = (
            self.tiling_indices
            if self.tiling_indices[1] < self.reduction_depth
            else reversed(self.tiling_indices)
        )
        if self.tiling_idx == self.tiling_indices[0]:
            self.tail_size = self.outer_tail_size
            self.num_elems = self.outer_num_elems
            self.inner_is_tiling_idx = False
        else:
            self.tail_size = self.inner_tail_size
            self.num_elems = self.inner_num_elems
            self.inner_is_tiling_idx = True
        return vars

    def transform_indexing(self, index: sympy.Expr) -> sympy.Expr:
        return self.scale_index_with_offset(
            index,
            itervar_idx=self.outer_idx,
            offset=self.inner_itervar(),
        )


def get_loop_body_lowp_fp(_body: LoopBody) -> tuple[Optional[torch.dtype], bool]:
    """
    Returns the low precision data type (torch.float16/torch.bfloat16) contained in the nodes
    and if all the nodes can codegen with this data type without converting to float.
    Otherwise returns None and True.
    """
    sub_blocks = [_body.root_block] + list(_body.subblocks.values())

    _lowp_fp_type: Optional[torch.dtype] = None
    _use_fp32 = False
    for sub_block in sub_blocks:
        for _node in sub_block.graph.nodes:
            if _node.op == "placeholder" or _node.target in (
                "get_index",
                "index_expr",
            ):
                continue

            # Fast path if all operations can support bf16/fp16 without converting to fp32
            if _node.target not in [
                "load",
                "store",
                "abs",
                "neg",
                "output",
            ]:
                _use_fp32 = True

            if hasattr(_node, "meta") and _node.meta:
                assert OptimizationContext.key in _node.meta
                opt_ctx: OptimizationContext = _node.meta[OptimizationContext.key]
                if not opt_ctx.dtype or opt_ctx.dtype not in DTYPE_LOWP_FP:
                    _use_fp32 = True
                elif _lowp_fp_type is not None:
                    if _lowp_fp_type != opt_ctx.dtype:
                        warnings.warn("bf16 and fp16 are mixed in the scheduler node.")
                else:
                    _lowp_fp_type = opt_ctx.dtype
            else:
                _use_fp32 = True

    return _lowp_fp_type, _use_fp32


class TilingSelect:
    """
    Implement the heuristic to select the tiling factors and tiling indices.
    In the future, we can implement advanced heuristic in a subclass.
    """

    def select_tiling(
        self,
        fn_list,
        var_sizes_list,
    ) -> tuple[list[int], list[int]]:
        # TODO(jgong5): support alternative tiling factors and data types
        loop_bodies = _get_loop_body(fn_list)
        all_dtypes = _get_dtype_from_loopbodies(loop_bodies)
        assert all_dtypes
        if any(dtype not in VECTORIZABLE_DTYPES for dtype in all_dtypes):
            return [], []
        dtype = torch.float
        _lowp_fp_dtype = get_loop_body_lowp_fp(loop_bodies[0])[0]
        if _lowp_fp_dtype and all(
            (get_loop_body_lowp_fp(loop_body)[0] == _lowp_fp_dtype)
            for loop_body in loop_bodies[1:]
        ):
            dtype = _lowp_fp_dtype

        tiling_factor = cpu_vec_isa.pick_vec_isa().nelements(dtype=dtype)
        tiling_indices = self._select_tiling_indices(
            fn_list, var_sizes_list, tiling_factor
        )

        if tiling_indices:
            group, reduction_group = max(
                var_sizes_list, key=lambda sizes: len(sizes[1])
            )
            call_ranges = tuple(group) + tuple(reduction_group)

            if config.cpp.enable_tiling_heuristics:

                def _try_get_stride(
                    index,
                    itervars,
                    tiling_factor,
                    tiling_indices,
                ):
                    itervar = itervars[tiling_indices[0]]
                    stride = stride_at_vec_range(index, itervar, tiling_factor)
                    return stride if stride.is_number else None

                def _update_negative_op_count(
                    node_name, non_contig_indexing_op_counter
                ):
                    if node_name not in non_contig_indexing_op_counter:
                        non_contig_indexing_op_counter[node_name] = 1
                    else:
                        non_contig_indexing_op_counter[node_name] += 1

                def _is_valid_indices(
                    itervars,
                    tiling_indices,
                ):
                    return (
                        len(tiling_indices) == 1
                        and len(itervars) > 0
                        and (
                            tiling_indices[0]
                            if tiling_indices[0] >= 0
                            else tiling_indices[0] + len(itervars)
                        )
                        < len(itervars)
                    )

                itervars = [
                    sympy_index_symbol_with_prefix(SymT.XBLOCK, n)
                    for n in range(len(call_ranges))
                ]
                reduction_depth = len(group)
                vars, reduction_vars = (
                    itervars[:reduction_depth],
                    itervars[reduction_depth:],
                )
                op_counter: dict[str, int] = {}
                # ops may cause overhead with vectorization, like non-contiguous
                # index_expr, load, store
                non_contig_indexing_op_counter: dict[str, int] = {}
                for _body in loop_bodies:
                    sub_blocks = [_body.root_block] + list(_body.subblocks.values())
                    for sub_block in sub_blocks:
                        for _node in sub_block.graph.nodes:
                            if _node.target in ["index_expr", "load", "store"]:
                                # get the index and replace prefix from z to x
                                arg_idx = 1 if _node.target == "index_expr" else 2
                                index = sub_block.body.indexing_from_args(
                                    (vars, reduction_vars)
                                )[_node.args[arg_idx].args[0]]
                                if _is_valid_indices(itervars, tiling_indices):
                                    stride = _try_get_stride(
                                        index, itervars, tiling_factor, tiling_indices
                                    )
                                    if (
                                        stride is None
                                        if _node.target == "index_expr"
                                        else stride not in [0, 1]
                                    ):
                                        _update_negative_op_count(
                                            _node.target, non_contig_indexing_op_counter
                                        )
                            if isinstance(_node.target, str) and not (
                                _node.target.startswith("masked_subblock")
                                or _node.target
                                in ["ops", "output", "constant", "get_index"]
                            ):
                                if _node.target not in op_counter:
                                    op_counter[_node.target] = 1
                                else:
                                    op_counter[_node.target] += 1

                op_num = sum(op_counter.values())
                non_contig_indexing_op_num = sum(
                    non_contig_indexing_op_counter.values()
                )
                ratio_threshold = 0.12
                quantity_threshold = 35
                if non_contig_indexing_op_num >= quantity_threshold or (
                    op_num > 0
                    and non_contig_indexing_op_num / op_num >= ratio_threshold
                ):
                    # Too many non-contiguous load/store/index_expr which hurts the
                    # vectorization performance. Disable vectorization when exceeding
                    # the thresholds.
                    return [], []

                if (
                    not reduction_group
                    and group
                    and len(tiling_indices) == 1
                    and not has_free_symbols(
                        [
                            group[tiling_indices[0]],
                        ]
                    )
                    and group[tiling_indices[0]] < tiling_factor / 4
                    and op_num < 10
                ):
                    # We found that when the number of elements in the inner loop range is
                    # relatively small(< tiling_factor / 4) and the number of operations is
                    # not large(< 10), vectorization is not efficient.
                    # And found that `#pragma GCC ivdep` has better performance than
                    # `#pragma omp simd simdlen(8)` for these cases.
                    return [], []

            if dtype in DTYPE_LOWP_FP:
                # For lower precision data type, if the call_range is not long enough,
                # use tiling_factor // 2 for better performance
                factor_lowp = cpu_vec_isa.pick_vec_isa().nelements(dtype=dtype)
                for tiling_indice in tiling_indices:
                    if tiling_indice < 0:
                        tiling_indice = tiling_indice + len(call_ranges)
                    if tiling_indice < 0 or tiling_indice >= len(call_ranges):
                        continue
                    if has_free_symbols(call_ranges):
                        call_range = V.graph.sizevars.size_hint(
                            call_ranges[tiling_indice], fallback=0
                        )
                        if call_range < factor_lowp:
                            V.graph.sizevars.check_lt(call_range, factor_lowp)  # type: ignore[arg-type]
                            tiling_factor = factor_lowp // 2
                            break
                    elif call_ranges[tiling_indice] < factor_lowp:
                        tiling_factor = factor_lowp // 2
                        break

            if len(tiling_indices) == 1:
                return [tiling_factor], tiling_indices
            if len(tiling_indices) == 2:
                return [tiling_factor, tiling_factor], tiling_indices
        return [], []

    def _select_tiling_indices(
        self,
        fn_list,
        var_sizes_list,
        tiling_factor,
    ):
        all_index = []
        for fn, var_sizes in zip(fn_list, var_sizes_list):
            rw = dependencies.extract_read_writes(fn, *var_sizes)
            all_index += [dep.index for dep in itertools.chain(rw.reads, rw.writes)]
        contig_vars = OrderedSet[int]()
        contig_vars_list = []
        non_contig_stride_const = OrderedSet[int]()
        non_contig_stride_other = OrderedSet[int]()
        for index in all_index:
            for var in index.free_symbols:
                if not re.search(r"^d\d+$", var.name):
                    continue
                stride = stride_at_vec_range(index, var, tiling_factor)
                if stride == 0:
                    continue
                elif stride == 1:
                    contig_vars.add(int(var.name[1:]))
                    contig_vars_list.append(int(var.name[1:]))
                elif all(symbol_is_type(s, SymT.SIZE) for s in stride.free_symbols):
                    non_contig_stride_const.add(int(var.name[1:]))
                else:
                    non_contig_stride_other.add(int(var.name[1:]))
        contig_only = contig_vars - non_contig_stride_const - non_contig_stride_other
        group, reduction_group = max(var_sizes_list, key=lambda sizes: len(sizes[1]))
        num_itervars = len(group) + len(reduction_group)
        if len(contig_vars) == 0:
            # no contiguous vars
            return [num_itervars - 1]
        if contig_only:
            return sorted(contig_only)[-1:]
        contig_and_const_stride = (
            contig_vars & non_contig_stride_const
        ) - non_contig_stride_other
        contig_vars_sorted = sorted(contig_vars)
        if (
            len(contig_vars_sorted) == 2
            and contig_vars_sorted[-1] in contig_and_const_stride
            and contig_vars_sorted[-1] == num_itervars - 1
        ):
            return contig_vars_sorted
        return sorted(contig_vars_sorted, key=contig_vars_list.count)[-1:]


class CppKernelProxy(CppKernel):
    # Subclass CppKernel, CppVecKernel, etc., to customize code generation.
    # Override CppOverrides or CppVecOverrides to emit custom ops.
    # Earlier, this meant copying codegen_functions() to use your subclasses.
    # Now, use kernel_cls and vec_kernel_cls class attributes instead.
    # This lets CppKernelProxy subclasses inject custom behavior cleanly.
    # No need to duplicate codegen_functions() just to swap kernel classes.
    kernel_cls: type[CppKernel] = CppKernel
    vec_kernel_cls: type[CppVecKernel] = CppVecKernel
    tile2d_kernel_cls: type[CppTile2DKernel] = CppTile2DKernel

    def __init__(self, kernel_group):
        super().__init__(kernel_group.args, kernel_group.ws.num_threads)
        self.kernel_group = kernel_group
        self.loop_nest = None
        self.call_ranges = None
        self.picked_vec_isa: cpu_vec_isa.VecISA = cpu_vec_isa.pick_vec_isa()
        self.kernels: list[CppKernel] = []

    def data_type_propagation(self, nodes):
        for _node in nodes:
            assert isinstance(_node, SchedulerNode)
            DataTypePropagation.propagate_scheduler_node(_node)

    # Check if all the nodes of a given fx graph can support BF16/FP16
    def is_lowp_fp_scheduler(self, scheduler_node: SchedulerNode):
        if not isinstance(scheduler_node._body, LoopBody):
            return True
        # Propagate the dtype to check if all the fx node is bf16/fp16
        DataTypePropagation.propagate_scheduler_node(scheduler_node)
        return (
            get_loop_body_lowp_fp(scheduler_node._body)[0] is not None
            and not get_loop_body_lowp_fp(scheduler_node._body)[1]
        )

    def legalize_lowp_fp_dtype_loopbody(self, loop_body: LoopBody):
        def add_to_dtype(sub_graph: torch.fx.Graph):
            def get_input_dtype(node: torch.fx.Node) -> Optional[torch.dtype]:
                """Get input dtype for nodes that may consumes lowp fp dt"""
                if node.target == "store":
                    return V.graph.get_dtype(node.args[1])  # type: ignore[arg-type]
                elif node.target == "to_dtype_bitcast":
                    return node.args[-1]  # type: ignore[return-value]
                elif node.target == "to_dtype":
                    if len(node.args) > 3:
                        return node.args[3]  # type: ignore[return-value]
                    else:
                        return node.kwargs.get("src_dtype", None)  # type: ignore[return-value]
                else:
                    return None

            def get_output_dtype(node: torch.fx.Node) -> Optional[torch.dtype]:
                """Get output dtype for nodes that may produce lowp fp dt"""
                if node.target == "load":
                    assert len(node.args) == 3
                    return V.graph.get_dtype(node.args[1])  # type: ignore[arg-type]
                elif node.target in ["to_dtype", "constant", "index_expr"]:
                    return node.args[-1]  # type: ignore[return-value]
                elif node.target == "to_dtype_bitcast":
                    return node.args[2]  # type: ignore[return-value]
                else:
                    return None

            def is_lowp_fp_source(node: torch.fx.Node, dt: torch.dtype):
                """Check if the given node produces output with expected low precision floating point data type."""
                assert dt in DTYPE_LOWP_FP
                return get_output_dtype(node) == dt

            def is_lowp_fp_sink(node: torch.fx.Node, dt: torch.dtype):
                """Check if the given node accept input with expected low precision floating point data type."""
                assert dt in DTYPE_LOWP_FP
                if input_dtype := get_input_dtype(node):
                    return input_dtype == dt
                elif node.target == "to_dtype":
                    # The `src_dtype` of a `to_dtype` node might miss, in which case the node accept any input dtype.
                    return True
                else:
                    return False

            def is_lowp_fp_source_no_promote(node: torch.fx.Node, dt: torch.dtype):
                """Check if the node is a lowp fp sources which are all directly fed to ops that accepts lowp fp input
                thus no need to promote to float
                """
                return is_lowp_fp_source(node, dt) and all(
                    is_lowp_fp_sink(user, dt) for user in node.users
                )

            sub_graph_nodes = list(sub_graph.nodes)
            to_lowp_fp_legalized_nodes = []
            for _node in sub_graph_nodes:
                if (
                    _node.target in ["load", "index_expr"]
                    and (dt := get_output_dtype(_node)) in DTYPE_LOWP_FP
                ):
                    # No need to promote to float if all users are ops that accepts lowp fp input
                    # pyrefly: ignore [bad-argument-type]
                    if all(is_lowp_fp_sink(user, dt) for user in _node.users):
                        continue
                    ops = _node.args[0]
                    with sub_graph.inserting_after(_node):
                        to_type_node = sub_graph.call_method(
                            "to_dtype", args=(ops, _node, torch.float)
                        )
                        _node.replace_all_uses_with(
                            to_type_node, lambda n: n is not to_type_node
                        )
                        # pyrefly: ignore [bad-assignment]
                        metrics.cpp_to_dtype_count += 1
                elif (
                    _node.target == "store"
                    and (dt := get_input_dtype(_node)) in DTYPE_LOWP_FP
                ):
                    ops, name, _, value_var, _ = _node.args
                    # pyrefly: ignore [bad-argument-type]
                    if is_lowp_fp_source_no_promote(value_var, dt):
                        continue
                    dtype = V.graph.get_dtype(name)
                    with sub_graph.inserting_before(_node):
                        to_type_node = sub_graph.call_method(
                            "to_dtype", args=(ops, value_var, dtype)
                        )
                        _node.replace_input_with(value_var, to_type_node)
                        # pyrefly: ignore [bad-assignment]
                        metrics.cpp_to_dtype_count += 1
                elif _node.target == "reduction":
                    (
                        ops,
                        dtype,
                        src_dtype,
                        reduction_type,
                        value,
                    ) = _node.args
                    if src_dtype in DTYPE_LOWP_FP:
                        # Since we always convert the load/store value to float if the tensor is bfloat16/float16.
                        # Therefore, the reduction should never work with bfloat16/float16 value. Hence, we update
                        # the bfloat16/float16 reduction by
                        #     1) updating the src_dtype to float
                        # and 2) updating the dtype to float if it is bfloat16/float16.
                        assert dtype in [
                            torch.float,
                            torch.bfloat16,
                            torch.float16,
                            torch.int64,
                        ]
                        _node.args = (
                            ops,
                            torch.float if dtype in DTYPE_LOWP_FP else dtype,
                            torch.float,
                            reduction_type,
                            value,
                        )
                elif _node.target == "constant" and _node.args[-1] in DTYPE_LOWP_FP:
                    # No need to promote to float if all users are ops that accepts lowp fp input
                    (ops, value, dt) = _node.args
                    if all(is_lowp_fp_sink(user, dt) for user in _node.users):  # type: ignore[arg-type]
                        continue
                    _node.args = (ops, value, torch.float)
                elif _node.target == "to_dtype" and _node.args[-1] in DTYPE_LOWP_FP:
                    # No need to promote to float if all users are ops that accepts lowp fp input
                    (ops, x, dt) = _node.args
                    if all(is_lowp_fp_sink(user, dt) for user in _node.users):  # type: ignore[arg-type]
                        continue
                    # The legalization always loads the BF16/FP16 tensor as FP32 for computation
                    # and converts back to BF16/FP16 after the computation.
                    # Hence, there should be no computation w/ BF16/FP16.
                    # Therefore, we update the to_dtype by replacing the bf16/fp16 dtype with fp32.
                    # Save the legalized to_dtype node for the elimination(eliminate_to_dtype step):
                    #  1) Eliminate the redundant to_dtype node if we have a pattern as follows:
                    #     graph():
                    #       %lowp_fp_legalized = call_method[target=to_dtype](args = (%ops, %input, torch.float))
                    #       %to_dtype2 = call_method[target=to_dtype](args = (%ops, %lowp_fp_legalized, torch.bfloat16/float16))
                    # Regarding the first to_dtype, it is redundant because
                    # the second to_type also converts to the torch.bfloat16/torch.float16.
                    # Hence, we remove the first to_type.
                    to_lowp_fp_legalized_nodes.append(_node)
                    _node.args = (ops, x, torch.float)
                elif _node.target == "to_dtype_bitcast":
                    (ops, value_var, dtype, src_dtype) = _node.args

                    # to_dtype_bitcast act as a lowp fp sink:
                    # c10::bit_cast requires the source and target have the same bitwidth. Because the input tensor's
                    # dtype could be promoted, e.g. from float16 to float, we have to cast the tensor to its original
                    # source dtype before invoking bit_cast.
                    if src_dtype in DTYPE_LOWP_FP:
                        # No need to promote to float if it is a user of a lowp fp sources
                        # which are all directly fed to ops that accepts lowp fp input
                        if not is_lowp_fp_source_no_promote(value_var, src_dtype):
                            with sub_graph.inserting_before(_node):
                                to_type_node = sub_graph.call_method(
                                    "to_dtype", args=(ops, value_var, src_dtype)
                                )
                                _node.replace_input_with(value_var, to_type_node)
                                # pyrefly: ignore [bad-assignment]
                                metrics.cpp_to_dtype_count += 1

                    # to_dtype_bitcast act as a lowp fp source:
                    # We also need to convert the bit-casted tensor back to float to make sure we keep using higher
                    # precision values for the rest of the computation.
                    if dtype in DTYPE_LOWP_FP:
                        # No need to promote to float if all users are ops that accepts lowp fp input
                        if not (
                            all(is_lowp_fp_sink(user, dtype) for user in _node.users)
                        ):
                            ops = _node.args[0]
                            with sub_graph.inserting_after(_node):
                                to_type_node = sub_graph.call_method(
                                    "to_dtype", args=(ops, _node, torch.float)
                                )
                                _node.replace_all_uses_with(
                                    to_type_node, lambda n: n is not to_type_node
                                )
                                # pyrefly: ignore [bad-assignment]
                                metrics.cpp_to_dtype_count += 1

            def eliminate_to_dtype(sub_graph: torch.fx.Graph):
                def _eliminate_duplicate_to_node(sub_graph: torch.fx.Graph):
                    # Eliminate the redundant to_dtype node. Let's consider a pattern as follows:
                    #   graph():
                    #     %to_dtype1 = call_method[target=to_dtype](args = (%ops, %input, torch.float), kwargs = {})
                    #     %to_dtype2 = call_method[target=to_dtype](args = (%ops, %to_dtype1, torch.float), kwargs = {})
                    # Regarding the first to_dtype, it is redundant because the second to_type also converts to the
                    # torch.float. Hence, we remove the first to_type
                    def _used_by_to(to_node: torch.fx.Node):
                        return all(usr.target == "to_dtype" for usr in to_node.users)

                    all_to_nodes = [
                        node for node in sub_graph.nodes if node.target == "to_dtype"
                    ]
                    all_to_nodes_and_users = [
                        {node: node.users} for node in all_to_nodes if _used_by_to(node)
                    ]
                    for node_users in all_to_nodes_and_users:
                        for node, users in node_users.items():
                            if node in sub_graph.nodes and (
                                all(usr.args[-1] == node.args[-1] for usr in users)
                                or (
                                    node in to_lowp_fp_legalized_nodes
                                    and all(
                                        usr.args[-1] in DTYPE_LOWP_FP for usr in users
                                    )
                                )
                            ):
                                val_node = node.all_input_nodes[-1]
                                node.replace_all_uses_with(val_node)
                                sub_graph.erase_node(node)

                    # For debug mode, the graph of LoopBody will attach a new GraphModule as
                    # owning_module for debugging while the release mode will not. The lint will
                    # check whether the graph has owning_module to decide if it needs to check
                    # call_module. LoopBody might contain get_index as a module call. But it
                    # is just a function. Hence, it cannot pass the lint check for debug mode.
                    # We bypass the check if the owning_module is None. Eventually, we should call
                    # get_index via call_function but not call_module.
                    if sub_graph.owning_module is None:
                        sub_graph.lint()

                _eliminate_duplicate_to_node(sub_graph)

            eliminate_to_dtype(sub_graph)

        sub_blocks = [loop_body.root_block] + list(loop_body.subblocks.values())
        for sub_block in sub_blocks:
            add_to_dtype(sub_block.graph)

    def legalize_lowp_fp_dtype(self, nodes):
        if all(
            isinstance(_node, SchedulerNode) and self.is_lowp_fp_scheduler(_node)
            for _node in nodes
        ):
            # Mark the load node to load bf16/fp16
            for _node in nodes:
                sub_blocks = [_node._body.root_block] + list(
                    _node._body.subblocks.values()
                )
                for sub_block in sub_blocks:
                    for fx_node in sub_block.graph.nodes:
                        if fx_node.target in ["load", "store"]:
                            assert fx_node.meta
                            assert OptimizationContext.key in fx_node.meta
                            opt_ctx: OptimizationContext = fx_node.meta[
                                OptimizationContext.key
                            ]
                            assert opt_ctx.dtype in DTYPE_LOWP_FP

            # Bypass the legalization as the kernel can run with bf16/fp16 directly
            return

        for _node in nodes:
            assert isinstance(_node, SchedulerNode)
            assert isinstance(_node._body, LoopBody)
            body: LoopBody = _node._body
            if not body.is_memory_copy():
                self.legalize_lowp_fp_dtype_loopbody(body)

    def codegen_functions(self, fn_list, var_sizes_list):
        assert len(fn_list) == len(var_sizes_list)
        kernel_group = self.kernel_group
        group, reduction_group = max(var_sizes_list, key=lambda sizes: len(sizes[1]))

        self.set_ranges(group, reduction_group)

        def codegen_kernel(cls, *args):
            with kernel_group.new_kernel(cls, *args) as kernel:
                # Ugly hack to maintain the metrics kernel count since
                # we only count in CppKernelProxy, not those contained in it
                # pyrefly: ignore [bad-assignment]
                metrics.generated_kernel_count -= 1

                run(kernel)
                return kernel

        def run(kernel):
            vars, reduction_vars = kernel.set_ranges(group, reduction_group)
            in_suffix = False
            for fn, var_sizes in zip(fn_list, var_sizes_list):
                if var_sizes in [
                    (group, reduction_group),
                    (tuple(itertools.chain(group, reduction_group)), ()),
                ]:
                    assert not in_suffix
                    fn(vars, reduction_vars)
                else:
                    in_suffix = True
                    assert var_sizes == (
                        group,
                        (),
                    ), f"unexpected group: {var_sizes} != {group}, {reduction_group}"
                    # we can fuse in some extra pointwise into the suffix
                    with kernel.write_to_suffix():
                        fn(vars, ())

        scalar_kernel = codegen_kernel(self.kernel_cls)
        V.graph.removed_buffers |= scalar_kernel.removed_buffers
        V.graph.inplaced_to_remove |= scalar_kernel.inplaced_to_remove
        self.loop_nest = LoopNest.build(scalar_kernel)

        if not self.picked_vec_isa or not self.itervars:
            self.kernels = [scalar_kernel]
            self.aggregate_reduction_buffers(False, None)
            self.loop_nest.set_kernel(self)
            return

        # Kernels share the same global contexts like V.graph.wrapper_code, V.kernel.args.
        # But the generated scalar kernel has updated these global contexts. Hence, the other kernels
        # should not do this again to avoid context conflict. By now, we only control the
        # config.inplace_buffers. In the future, we could maintain more contexts.
        with torch._inductor.config.patch(inplace_buffers=False):
            tiling_select = TilingSelect()
            tiling_factors, tiling_indices = tiling_select.select_tiling(
                fn_list, var_sizes_list
            )
            assert len(tiling_factors) == len(tiling_indices)
            _inner_loop_reduction_outer_not = False
            _outer_loop = None
            if tiling_indices:
                inner_loop_reduction = False
                outer_loop_level = tiling_indices[0]
                inner_loop_level = outer_loop_level + 1
                if len(self.loop_nest.loops) > inner_loop_level:
                    inner_loop_reduction = self.loop_nest.loops[
                        inner_loop_level
                    ].is_reduction
                    outer_loop_reduction = self.loop_nest.loops[
                        outer_loop_level
                    ].is_reduction
                    _inner_loop_reduction_outer_not = (
                        inner_loop_reduction and not outer_loop_reduction
                    )

            if len(tiling_indices) == 1:
                # pyrefly: ignore [bad-assignment]
                metrics.generated_cpp_vec_kernel_count += 1
                loop = self.loop_nest.tile(tiling_indices[0], factor=tiling_factors[0])
                vec_kernel = codegen_kernel(
                    self.vec_kernel_cls, tiling_factors[0], tiling_indices[0]
                )
                tail_size = loop.size - loop.tiled_size
                vec_kernel.active_ranges = {loop.var: (0, loop.tiled_size)}
                if config.cpp.enable_loop_tail_vec:
                    tail_kernel = codegen_kernel(
                        self.vec_kernel_cls,
                        tiling_factors[0],
                        tiling_indices[0],
                        tail_size,
                    )
                else:
                    tail_kernel = scalar_kernel
                    scalar_kernel.inner_itervars = [loop.var]
                tail_kernel.active_ranges = {loop.var: (loop.tiled_size, loop.size)}
                self.kernels = [vec_kernel, tail_kernel]
                _outer_loop = loop
            elif len(tiling_indices) == 2:
                assert (
                    tiling_indices[1] == len(self.itervars) - 1
                    and tiling_factors[0] == tiling_factors[1]
                )

                # pyrefly: ignore [bad-assignment]
                metrics.generated_cpp_vec_kernel_count += 2
                outer_loop = self.loop_nest.tile(
                    tiling_indices[0], factor=tiling_factors[0]
                )
                outer_ranges = {
                    "main": (0, outer_loop.tiled_size),
                    "tail": (outer_loop.tiled_size, outer_loop.size),
                }
                outer_tail_size = outer_loop.size - outer_loop.tiled_size
                inner_loop = self.loop_nest.tile(
                    tiling_indices[1], factor=tiling_factors[0]
                )
                inner_ranges = {
                    "main": (0, inner_loop.tiled_size),
                    "tail": (inner_loop.tiled_size, inner_loop.size),
                }
                inner_tail_size = inner_loop.size - inner_loop.tiled_size
                tile2d_kernel = codegen_kernel(
                    self.tile2d_kernel_cls,
                    tiling_factors[0],
                    tiling_indices,
                )
                tile2d_kernel.active_ranges = {
                    outer_loop.var: outer_ranges["main"],
                    inner_loop.var: inner_ranges["main"],
                }
                tail_kernel = []
                if config.cpp.enable_loop_tail_vec:
                    for outer_r, inner_r in (
                        ("main", "tail"),
                        ("tail", "main"),
                        ("tail", "tail"),
                    ):
                        _inner_tail_size = (
                            inner_tail_size if inner_r == "tail" else None
                        )
                        _outer_tail_size = (
                            outer_tail_size if outer_r == "tail" else None
                        )
                        kernel = codegen_kernel(
                            self.tile2d_kernel_cls,
                            tiling_factors[0],
                            tiling_indices,
                            _inner_tail_size,
                            _outer_tail_size,
                        )
                        kernel.active_ranges = {
                            outer_loop.var: outer_ranges[outer_r],
                            inner_loop.var: inner_ranges[inner_r],
                        }
                        tail_kernel.append(kernel)
                else:
                    vec_kernel = codegen_kernel(
                        self.vec_kernel_cls, tiling_factors[0], tiling_indices[0]
                    )
                    vec_kernel.active_ranges = {
                        outer_loop.var: outer_ranges["main"],
                        inner_loop.var: inner_ranges["tail"],
                    }
                    vec_kernel.inner_itervars = [inner_loop.var]
                    tail_kernel.append(vec_kernel)
                    scalar_kernel.active_ranges = {
                        outer_loop.var: outer_ranges["tail"],
                        inner_loop.var: (0, inner_loop.size),
                    }
                    scalar_kernel.inner_itervars = [inner_loop.var, outer_loop.var]
                    tail_kernel.append(scalar_kernel)
                self.kernels = [tile2d_kernel] + tail_kernel
                _outer_loop = outer_loop
            else:
                self.kernels = [scalar_kernel]
            self.aggregate_reduction_buffers(
                _inner_loop_reduction_outer_not, _outer_loop
            )
            self.loop_nest.set_kernel(self)

    def codegen_loop_bodies(self, loop_bodies, var_sizes_list):
        for body in loop_bodies:
            self.legalize_lowp_fp_dtype_loopbody(body)
            DataTypePropagation.propagate_loopbody(body)
        self.codegen_functions(loop_bodies, var_sizes_list)

    def codegen_nodes(self, nodes: list[SchedulerNode]):
        # Legalize BF16 node by adding to_dtype explicitly
        self.legalize_lowp_fp_dtype(nodes)
        self.data_type_propagation(nodes)
        assert len(nodes) >= 1

        def fn(node, *index_vars):
            node.decide_inplace_update()
            node.mark_run()
            if isinstance(V.kernel, NullKernelHandler):
                return node._body(*index_vars)
            else:
                return node.codegen(index_vars)

        fn_list = [functools.partial(fn, node) for node in nodes]

        if (
            isinstance(V.local_buffer_context, LocalBufferContext)
            and V.local_buffer_context.local_buffers
        ):

            def wrap_fn(fn):
                wrapped_fn = V.local_buffer_context.localize_function(
                    fn,
                )
                wrapped_fn.original_fn = fn
                return wrapped_fn

            fn_list = [wrap_fn(fn) for fn in fn_list]

        var_sizes_list = [node.group[1] for node in nodes]
        self.codegen_functions(fn_list, var_sizes_list)

    def codegen_loops(self, code, worksharing):
        self.codegen_loops_impl(self.loop_nest, code, worksharing)

    def update_stores_with_parallel_reduction(self):
        for kernel in self.kernels:
            kernel.update_stores_with_parallel_reduction()

    def gen_body(self, code: Optional[BracesBuffer] = None):
        assert code is not None
        if_prefix = "C10_LIKELY"
        for kernel in self.kernels:
            with contextlib.ExitStack() as stack:
                if kernel.codegen_conditions(code, if_prefix):
                    if_prefix = "C10_UNLIKELY"
                    stack.enter_context(code.indent())
                    code.splice(kernel.gen_body())

    def aggregate_reduction_buffers(
        self, inner_loop_reduction_outer_not: bool, outer_loop: Optional["LoopLevel"]
    ):
        """
        CppKernel/CppVecKernel/CppTile2dKernel have reduction buffers themselves.
        Here, we decide how to aggregate them together and place new reduction buffers
        under CppKernelProxy.
        """

        def aggregate_reduction_prefix_suffix(outer_loop: "LoopLevel"):
            assert len(self.kernels) >= 2
            main_loop_kernel = self.kernels[0]
            tail_loop_kernel = self.kernels[-1]
            assert isinstance(main_loop_kernel, self.vec_kernel_cls)

            # Prefix
            if type(tail_loop_kernel) is self.kernel_cls:
                # if tail loop kernel is a scalar kernel, we need to extend tmp_acc -> tmp_acc_arr[] to
                # hold the temporary inner loop acc result for outer tail loop
                tail_loop_kernel.finalize_reduction_prefix(
                    main_loop_kernel.tiling_factor
                )
                main_loop_kernel.finalize_reduction_prefix()
                self.reduction_prefix.splice(
                    tail_loop_kernel.reduction_prefix
                    + main_loop_kernel.reduction_prefix
                )
            else:
                main_loop_kernel.finalize_reduction_prefix()
                self.reduction_prefix.splice(main_loop_kernel.reduction_prefix)

            # Suffix
            suffix_buf = BracesBuffer()
            with contextlib.ExitStack() as stack:
                if main_loop_kernel.codegen_conditions(
                    suffix_buf, "C10_LIKELY", outer_loop.var
                ):
                    stack.enter_context(suffix_buf.indent())
                    suffix_buf.splice(main_loop_kernel.reduction_suffix)
            with contextlib.ExitStack() as stack:
                if tail_loop_kernel.codegen_conditions(
                    suffix_buf, "C10_UNLIKELY", outer_loop.var
                ):
                    stack.enter_context(suffix_buf.indent())
                    if type(tail_loop_kernel) is self.kernel_cls:
                        reduction_vars = tail_loop_kernel.reduction_var_names
                        for name in reduction_vars:
                            new_name = f"{name}_arr[{outer_loop.var}_tail - {cexpr_index(outer_loop.tiled_size)}]"
                            replace_acc_name(tail_loop_kernel.stores, name, new_name)
                            replace_acc_name(
                                tail_loop_kernel.reduction_suffix, name, new_name
                            )
                        # If tail loop kernel is a scalar kernel, use direct sum instead of cascade_sum_combine
                        # as the reduction vars are extended: tmp_acc -> tmp_acc_arr[].
                        replace_cascade_sum_with_add(tail_loop_kernel.stores)
                        suffix_buf.splice(
                            move_code_under_inner_loop(
                                tail_loop_kernel.reduction_suffix,
                                outer_loop.var,
                                f"{outer_loop.var}_tail",
                                outer_loop.tiled_size,
                                outer_loop.size,
                            )
                        )
                    else:
                        suffix_buf.splice(tail_loop_kernel.reduction_suffix)
            self.reduction_suffix = suffix_buf

        main_kernel = self.kernels[0]
        if inner_loop_reduction_outer_not:
            assert outer_loop
            aggregate_reduction_prefix_suffix(outer_loop)
        else:
            main_kernel.finalize_reduction_prefix()
            self.reduction_prefix.splice(main_kernel.reduction_prefix)
            self.reduction_suffix.splice(main_kernel.reduction_suffix)
        self.parallel_reduction_prefix.splice(main_kernel.parallel_reduction_prefix)
        self.parallel_reduction_suffix.splice(main_kernel.parallel_reduction_suffix)
        self.local_reduction_init.splice(main_kernel.local_reduction_init)
        self.local_reduction_stores.splice(main_kernel.local_reduction_stores)
        self.non_parallel_reduction_prefix.splice(
            main_kernel.non_parallel_reduction_prefix
        )
        self.non_parallel_reduction_suffix.splice(
            main_kernel.non_parallel_reduction_suffix
        )


class OuterLoopFusedKernel(CppKernel):
    def __init__(self, kernel_group):
        super().__init__(kernel_group.args, kernel_group.ws.num_threads)
        self.inner: list[LoopNest] = []

    def decide_parallel_depth(self, max_parallel_depth, threads):
        kernels_parallel_depth = []
        nested_kernels: list[CppKernel] = [
            loop_nest.get_kernel() for loop_nest in self.inner
        ]
        # TODO(leslie-fang-intel): only enable parallel within all outer loop levels.
        for kernel in nested_kernels:
            # For any ScalarKernel, VecKernel, or Tile2DKernel,
            # they should all have the same call_ranges
            call_ranges = kernel.call_ranges
            assert call_ranges is not None
            kernels_parallel_depth.append(
                kernel.decide_parallel_depth(
                    ParallelDepth(
                        parallel_depth=(
                            len(call_ranges) - max_parallel_depth.start_depth
                        ),
                        start_depth=max_parallel_depth.start_depth,
                    ),
                    threads,
                ).parallel_depth
            )
        return ParallelDepth(
            parallel_depth=min(
                max_parallel_depth.parallel_depth, max(kernels_parallel_depth)
            ),
            start_depth=max_parallel_depth.start_depth,
        )


class ReasonFusedNodes(Enum):
    SAME_VARS_REDUCE = "same_vars_reduce"
    COMPATIBLE_REDUCTION = "compatible_reduction"
    COMPATIBLE_RANGES_NO_REDUCTION = "compatible_ranges_no_reduction"


class CppScheduling(BaseScheduling):
    # Subclass CppKernelProxy to customize codegen without copying codegen_node().
    # Use kernel_proxy_cls to inject custom proxies in CppScheduling subclasses.
    # Avoid duplicating codegen_node() just to swap in a custom kernel proxy class.
    kernel_proxy_cls: type[CppKernelProxy] = CppKernelProxy
    # ctypes limits the number of args to 1024, refer to:
    # https://github.com/python/cpython/commit/a285af7e626d1b81cf09f8b2bf7656f100bc1237
    # We set a conservative threshold here.
    MAX_FUSED_KERNEL_ARGS_NUM = 500
    backend_features = OrderedSet(
        [
            BackendFeature.INPLACE_BUFFERS,
            BackendFeature.REDUCE_TO_SINGLE_ELEMENT,
        ]
    )

    @classmethod
    def get_backend_features(cls, device: torch.device) -> OrderedSet[BackendFeature]:
        return cls.backend_features

    def __init__(self, scheduler):
        super().__init__(scheduler)
        if scheduler:
            self.reset_kernel_group()
        self._ready_to_flush = False

    def _set_flush_status(self, status: bool):
        self._ready_to_flush = status

    def group_fn(self, sizes):
        return tuple(tuple(map(V.graph.sizevars.simplify, s)) for s in sizes)

    def reset_kernel_group(self):
        self.kernel_group = KernelGroup()

    def fuse(self, node1, node2):
        if node1.is_foreach() or node2.is_foreach():
            return ForeachKernelSchedulerNode.fuse(node1, node2)
        elif node1.is_template():
            assert not node2.is_template()
            return FusedSchedulerNode.fuse(node1, node2)
        else:
            if (
                self._why_fuse_nodes(node1, node2)
                == ReasonFusedNodes.COMPATIBLE_RANGES_NO_REDUCTION
            ):
                assert isinstance(node1, (SchedulerNode, FusedSchedulerNode))
                assert isinstance(node2, (SchedulerNode, FusedSchedulerNode))

                _, (vars1, reduce1) = node1.group
                _, (vars2, reduce2) = node2.group
                assert reduce1 == () and reduce2 == (), (reduce1, reduce2)

                def get_indexing_ranges_exprs(node):
                    if isinstance(node, FusedSchedulerNode):
                        assert len(node.snodes) > 0, node.snodes
                        var_ranges = None
                        indexing_exprs = OrderedSet[Any]()
                        for snode in node.snodes:
                            v, exprs = get_indexing_ranges_exprs(snode)
                            if var_ranges is None:
                                var_ranges = v
                            assert var_ranges == v, (var_ranges, v, node.snodes)
                            indexing_exprs.update(exprs)
                        return var_ranges, list(indexing_exprs)
                    else:
                        assert isinstance(node, SchedulerNode)
                        comp_buffer = node.node
                        assert isinstance(comp_buffer, ir.ComputedBuffer)
                        _, body, _ = comp_buffer.get_default_sizes_body()
                        return body.var_ranges, list(body.indexing_exprs.values())

                node_to_recomp = node1 if len(vars1) < len(vars2) else node2
                assert isinstance(node_to_recomp, SchedulerNode)

                ref_node = node2 if len(vars1) < len(vars2) else node1

                ref_indexing_constraints = get_indexing_ranges_exprs(ref_node)

                node_to_recomp.recompute_size_and_body(
                    extra_indexing_constraints=ref_indexing_constraints
                )

                _, (vars1, _) = node1.group
                _, (vars2, _) = node2.group

                if vars1 == vars2:
                    return FusedSchedulerNode.fuse(node1, node2)

                # recompute ref_node if its ranges are also changed
                node_to_recomp_indexing_constraints = get_indexing_ranges_exprs(
                    node_to_recomp
                )
                if isinstance(ref_node, SchedulerNode):
                    ref_node.recompute_size_and_body(
                        extra_indexing_constraints=node_to_recomp_indexing_constraints
                    )
                else:
                    assert isinstance(ref_node, FusedSchedulerNode)
                    for snode in ref_node.snodes:
                        assert isinstance(snode, SchedulerNode)
                        snode.recompute_size_and_body(
                            extra_indexing_constraints=node_to_recomp_indexing_constraints
                        )
                    ref_node = FusedSchedulerNode(ref_node.scheduler, ref_node.snodes)

                _, (vars1, _) = node1.group
                _, (vars2, _) = node2.group
                assert vars1 == vars2, (vars1, vars2)
                return FusedSchedulerNode.fuse(node1, node2)
            elif self.can_fuse_vertical_outer_loop(node1, node2):
                return OuterLoopFusedSchedulerNode.fuse(
                    node1, node2, self._get_outer_loop_fusion_depth(node1, node2)
                )
            else:
                return FusedSchedulerNode.fuse(node1, node2)

    def _why_fuse_nodes(self, node1, node2) -> Optional[ReasonFusedNodes]:
        _, (vars1, reduce1) = node1.group
        _, (vars2, reduce2) = node2.group

        if vars1 == vars2 and reduce1 == reduce2:
            return ReasonFusedNodes.SAME_VARS_REDUCE
        if reduce1 == () and vars1 == vars2 + reduce2:
            return ReasonFusedNodes.COMPATIBLE_REDUCTION
        if self._can_fuse_nodes_with_compatible_ranges(node1, node2):
            return ReasonFusedNodes.COMPATIBLE_RANGES_NO_REDUCTION
        # TODO(jansel): allow fusion pointwise (vars1, ()) suffix?
        return None

    def _can_fuse_nodes_with_compatible_ranges(self, node1, node2):
        # Here we try to fuse SchedulerNode/FusedSchedulerNode with compatible ranges
        # e.g. (s0, s1, s2) and (s0 * s1 * s2)
        _, (vars1, reduce1) = node1.group
        _, (vars2, reduce2) = node2.group

        c1 = reduce1 == () and reduce2 == ()
        c2 = math.prod(vars1) == math.prod(vars2)
        c3 = len(vars1) == 1 or len(vars2) == 1
        if not (c1 and c2 and c3):
            return False

        node_to_recomp = node1 if len(vars1) < len(vars2) else node2
        ref_node = node2 if len(vars1) < len(vars2) else node1

        # We can not recompute sizes and body for nodes other than SchedulerNode
        # TODO: we can extend fusion support with compatible ranges for FusedSchedulerNode
        if isinstance(node_to_recomp, FusedSchedulerNode):
            return False

        # It may happen that node1 and node2 compatible number of elements
        # but different original ranges, for example:
        # {d0: s0, d1: s1, d2: s2} vs {d0: s0*s1*s2}
        # See https://github.com/pytorch/pytorch/pull/120077/files#r1500427848 for more details
        # TODO: we can fix if it allows us to CSE at least one of the variables

        assert isinstance(node_to_recomp, SchedulerNode)
        if isinstance(node_to_recomp.node, ir.TemplateBuffer):
            return False
        assert isinstance(node_to_recomp.node, ir.ComputedBuffer)
        # node.data.get_size() is a cheaper version of node.get_read_writes().var_ranges
        # but without variable name
        ranges2 = node_to_recomp.node.data.get_size()
        ranges1 = None
        if isinstance(ref_node, FusedSchedulerNode):
            ranges_set = OrderedSet[tuple[Any, ...]]()
            for snode in ref_node.snodes:
                if isinstance(snode.node, ir.TemplateBuffer):
                    break
                assert isinstance(snode.node, ir.ComputedBuffer)
                ranges_set.add(tuple(snode.node.data.get_size()))

            if len(ranges_set) != 1:
                return False

            ranges1 = list(next(iter(ranges_set)))
        else:
            assert isinstance(ref_node, SchedulerNode)
            assert isinstance(ref_node.node, ir.ComputedBuffer)
            ranges1 = ref_node.node.data.get_size()  # type: ignore[assignment]

        if ranges1 != ranges2:
            return False

        return True

    def _can_fuse_horizontal_impl(self, node1, node2):
        assert isinstance(node1, (FusedSchedulerNode, SchedulerNode))
        assert isinstance(node2, (FusedSchedulerNode, SchedulerNode))
        if any(
            isinstance(node, OuterLoopFusedSchedulerNode) for node in (node1, node2)
        ):
            return False
        return self._why_fuse_nodes(node1, node2) is not None

    def can_fuse_horizontal(self, node1, node2):
        if node1.is_template() or node2.is_template():
            return False
        if (
            len(node1.get_nodes()) + len(node2.get_nodes())
            > config.cpp.max_horizontal_fusion_size
        ):
            return False

        return self._can_fuse_horizontal_impl(node1, node2)

    def can_fuse_multi_outputs_template(
        self, node1: BaseSchedulerNode, node2: BaseSchedulerNode
    ) -> bool:
        if template_buf := node1.get_template_node():
            return (
                isinstance(template_buf.layout, ir.MultiOutputLayout)
                and isinstance(node2.node, ir.MultiOutput)
                and len(node2.node.inputs) == 1
                and node2.node.inputs[0].get_name() == template_buf.name  # type: ignore[union-attr]
            )
        return False

    def _get_outer_loop_fusion_depth(self, node1, node2):
        DISABLE_OUTER_LOOP_FUSION = 0
        if not all(
            type(node)
            in (OuterLoopFusedSchedulerNode, FusedSchedulerNode, SchedulerNode)
            for node in (node1, node2)
        ):
            return DISABLE_OUTER_LOOP_FUSION

        _node1 = (
            node1.get_outer_nodes()[-1]
            if isinstance(node1, OuterLoopFusedSchedulerNode)
            else node1
        )
        assert isinstance(_node1, (FusedSchedulerNode, SchedulerNode))
        _node2 = (
            node2.get_outer_nodes()[0]
            if isinstance(node2, OuterLoopFusedSchedulerNode)
            else node2
        )
        assert isinstance(_node2, (FusedSchedulerNode, SchedulerNode))

        _, (vars1, reduce1) = _node1.group
        _, (vars2, reduce2) = _node2.group
        if vars1 == () and vars2 == () and reduce1 != () and reduce2 != ():
            # Reduction only
            return DISABLE_OUTER_LOOP_FUSION
        if all(type(node) is OuterLoopFusedSchedulerNode for node in (node1, node2)):
            return (
                node1.outer_loop_fusion_depth
                if node1.outer_loop_fusion_depth == node2.outer_loop_fusion_depth
                else DISABLE_OUTER_LOOP_FUSION
            )
        outer_loop_fusion_depth = min(len(vars1), len(vars2))
        if (
            outer_loop_fusion_depth >= 1
            and vars1[:outer_loop_fusion_depth] == vars2[:outer_loop_fusion_depth]
        ):
            if any(
                type(node) is OuterLoopFusedSchedulerNode for node in (node1, node2)
            ):
                _compare_node = (
                    node1 if type(node1) is OuterLoopFusedSchedulerNode else node2
                )
                if _compare_node.outer_loop_fusion_depth == outer_loop_fusion_depth:
                    # Same outer loop fusion depth as prev nodes in OuterLoopFusedSchedulerNode
                    return outer_loop_fusion_depth
                else:
                    return DISABLE_OUTER_LOOP_FUSION
            else:
                # First 2 nodes to generate OuterLoopFusedSchedulerNode
                return outer_loop_fusion_depth
        return DISABLE_OUTER_LOOP_FUSION

    def can_fuse_vertical_outer_loop(self, node1, node2):
        return (
            not node1.is_template()
            and not node2.is_template()
            and node1.get_operation_names() & node2.ancestors
            and not (
                self._can_fuse_horizontal_impl(node1, node2)
                and not node1.is_reduction()
            )
            and self._get_outer_loop_fusion_depth(node1, node2) >= 1
        )

    def get_fusion_pair_priority(self, node1, node2):
        if self.can_fuse_vertical_outer_loop(node1, node2):
            # Outer loop fusion with lower priority
            return 1
        else:
            return 0

    def can_fuse_vertical(self, node1, node2):
        if node2.is_template():
            # TODO(jgong5): support pre-op fusion with template
            return False
        if node1.is_template():
            template_fusion_supported, _ = template_fusion_with_epilogues_supported(
                node1, [node2]
            )
            return not node2.is_reduction() and template_fusion_supported
        return (
            self._can_fuse_horizontal_impl(node1, node2) and not node1.is_reduction()
        ) or self.can_fuse_vertical_outer_loop(node1, node2)

    def try_loop_split(self, nodes: list[SchedulerNode]):
        """
        Apply loop split optimization.
        When one of the indexing_exprs contains a division, we eliminate the division by splitting the loop
        to avoid non-contiguous loads, subject to the following conditions:
            1. No reduction and no mudular index for all nodes.
            2. The indexing_exprs of all nodes contain only one (or more, but all the same) division,
               where the divisor is an integer and not too small (the divisor > 8), the dividend is
               one of the iter_vars, and this var, i.e. the dimension that needs to be split, is
               contiguous in all other indexing_exprs.

        For example, if the node's var_ranges: {z0: 2, z1: 9216, z2: 960} and indexing_exprs:
        {'index0': 8847360*z0 + 960*z1 + z2, 'index1': 32*z0 + (z2//30), 'index2': z2},
        we will split z2 -> 30*z2 + z3, then the node's var_ranges will be changed to
        {z0: 2, z1: 9216, z2: 32, z3: 30} and indexing_exprs will be changed to
        {'index0': 8847360*z0 + 960*z1 + 30*z2 + z3, 'index1': 32*z0 + z2, 'index2': 30*z2 + z3}.
        """

        # No reduction and no mudular
        if any(
            len(node.group[1][1]) != 0
            or any(
                expr.has(ModularIndexing) for expr in node._body.indexing_exprs.values()
            )
            for node in nodes
        ):
            return nodes

        split_var = None
        split_number = None
        num_div = 0
        div_expr_ = None
        match_div = False
        matched_node = None

        for node in nodes:
            assert isinstance(node.node, ir.ComputedBuffer)
            _, original_body, _ = node.node.get_default_sizes_body()
            for name, expr in original_body.indexing_exprs.items():
                if not isinstance(expr, sympy.Expr):
                    continue
                for div_expr in expr.find(FloorDiv):
                    if (
                        any(div_expr.has(var) for var in original_body.iter_vars)
                        and div_expr != div_expr_
                    ):
                        div_expr_ = div_expr
                        num_div += 1
                    if num_div > 1:
                        return nodes
                    if (
                        isinstance(div_expr.args[1], sympy.core.numbers.Integer)
                        and div_expr.args[0] in original_body.iter_vars
                        and name is not None
                        and all(
                            stride_at_vec_range(expr_, div_expr.args[0]) in (0, 1)
                            for name_, expr_ in original_body.indexing_exprs.items()
                            if name_ != name
                        )
                        and div_expr.args[1] > 8
                    ):
                        split_var = div_expr.args[0]
                        split_number = div_expr.args[1]
                        match_div = True
                        matched_node = node

        # Only one node contains a division, and the split dimension is contiguous in all other indexing_exprs.
        if not match_div:
            return nodes

        extra_indexing_constraints = None

        def loop_split(sizes, body, vars):
            index_size, reduce_size = sizes
            index_vars, reduce_vars = vars
            split_idx = index_vars.index(split_var)
            new_index_size = index_size.copy()
            new_index_size[split_idx] = index_size[split_idx] // split_number
            new_index_size.insert(split_idx + 1, split_number)
            (new_index_vars, _), var_ranges = dependencies.index_vars_no_squeeze(
                new_index_size, reduce_size, prefix="y"
            )
            iter_vars = new_index_vars.copy()
            divisor_var = iter_vars.pop(split_idx + 1)
            iter_vars[split_idx] = split_number * iter_vars[split_idx] + divisor_var
            body = ir.LoopBody(
                body, [iter_vars, reduce_vars], var_ranges, new_index_vars, reduce_vars
            )
            nonlocal extra_indexing_constraints
            if not extra_indexing_constraints:
                extra_indexing_constraints = (
                    body.var_ranges,
                    list(body.indexing_exprs.values()),
                )
            return (
                (new_index_size, reduce_size),
                body,
                (new_index_vars, reduce_vars),
            )

        # Here decide the final loop order
        for node in nodes:
            if node == matched_node:
                node.recompute_size_and_body(recompute_sizes_body_func=loop_split)
        for node in nodes:
            if node != matched_node:
                node.recompute_size_and_body(
                    extra_indexing_constraints=extra_indexing_constraints,
                    recompute_sizes_body_func=loop_split,
                )

        return nodes

    def codegen_outer_loop_node(
        self,
        node: OuterLoopFusedSchedulerNode,
    ):
        """
        Generate the code for the outer loop fused scheduler node.
        1. Codegen with fused outer loop: depends on the analysis of
            the outer loop fused scheduler node, with or without the local buffer.
        2. If failed, fallback to standard codegen.
        """
        kernel_group = self.kernel_group
        generated_cpp_vec_kernel_count = metrics.generated_cpp_vec_kernel_count
        cpp_kernel_proxy_list: list[self.kernel_proxy_cls] = []  # type: ignore[name-defined]
        nodes_list: list[list[SchedulerNode]] = []
        assert isinstance(node, OuterLoopFusedSchedulerNode)

        def try_outer_loop_fusion_with_local_buf(node: OuterLoopFusedSchedulerNode):
            """
            Codegen code with fused outer loop and local Buffer.
            """
            assert isinstance(node, OuterLoopFusedSchedulerNode)
            cpp_kernel_proxy_list.clear()
            nodes_list.clear()

            def get_call_ranges(node: BaseSchedulerNode):
                assert isinstance(node, (SchedulerNode, FusedSchedulerNode))
                nodes: list[SchedulerNode] = node.get_nodes()  # type: ignore[assignment]
                _, (group, reduction_group) = max(
                    nodes, key=lambda x: int(x.is_reduction())
                ).group
                call_ranges = tuple(group) + tuple(reduction_group)
                return call_ranges

            local_buffers: list[ir.Buffer] = []
            # Map local buffer name to a list of global buffers
            local_to_global_buffers: dict[str, list[ir.Buffer]] = {}
            if all(
                len(get_call_ranges(_node)) == node.outer_loop_fusion_depth + 1
                for _node in node.get_outer_nodes()
            ):
                # Ref to the typical case of local buffer in
                # https://github.com/pytorch/pytorch/blob/1115a25c36340554442f28f9570abd42f0aface2/aten/src/ATen/native/cpu/SoftMaxKernel.cpp#L159 # noqa: B950
                # where the buffer is with size of last dim and contiguous.
                # Only support this typical case at first.
                visited_scheduler_nodes: OrderedSet[str] = OrderedSet()
                for scheduler_node in node.get_nodes():
                    # all users inside same OuterLoopFusedSchedulerNode
                    assert isinstance(scheduler_node, SchedulerNode)
                    visited_scheduler_nodes.add(scheduler_node.get_name())
                    if (
                        scheduler_node.is_reduction()
                        or len(scheduler_node.get_outputs()) != 1
                    ):
                        continue

                    scheduler_buffer = scheduler_node.get_outputs()[0]
                    if all(
                        user.node in node.get_nodes() for user in scheduler_buffer.users
                    ):
                        global_buffer = scheduler_buffer.node
                        assert isinstance(global_buffer, ir.ComputedBuffer)
                        global_buffer_layout = global_buffer.get_layout()
                        size_offset = node.outer_loop_fusion_depth - len(
                            get_call_ranges(scheduler_node)
                        )

                        def is_all_write_read_contiguous():
                            contiguous_index_expr = 0
                            stride = 1
                            for var, range in reversed(
                                # pyrefly: ignore [missing-attribute]
                                scheduler_node._body.var_ranges.items()
                            ):
                                contiguous_index_expr += stride * var
                                stride *= range
                            # pyrefly: ignore [missing-attribute]
                            write_index_expr = scheduler_node._body.get_write_expr(
                                scheduler_buffer.get_name()
                            )

                            def is_contiguous_index(x):
                                return x == contiguous_index_expr

                            return is_contiguous_index(write_index_expr) and all(
                                isinstance(user.node, SchedulerNode)
                                and is_contiguous_index(
                                    user.node._body.get_read_expr(
                                        scheduler_buffer.get_name()
                                    ),
                                )
                                for user in scheduler_buffer.users
                            )

                        if not (
                            global_buffer_layout.is_contiguous()
                            and is_all_write_read_contiguous()
                        ):
                            continue
                        # Local Buffer is a view of global buffer
                        local_buffer_stride: list[int] = []
                        stride = global_buffer_layout.stride[-1]
                        local_buffer_size = get_call_ranges(scheduler_node)[
                            size_offset:
                        ]
                        for sz in reversed(local_buffer_size):
                            local_buffer_stride.insert(0, stride)
                            stride *= sz
                        local_buffer_layout = ir.FixedLayout(
                            global_buffer_layout.device,
                            global_buffer_layout.dtype,
                            local_buffer_size,
                            local_buffer_stride,
                        )

                        def try_share_local_buffer(local_buffer_layout, local_buffers):
                            for local_buf in local_buffers:
                                if local_buffer_layout == local_buf.layout and all(
                                    all(
                                        user.node.get_name() in visited_scheduler_nodes
                                        for user in V.graph.scheduler.name_to_buf[
                                            global_buffer.name
                                        ].users
                                    )
                                    for global_buffer in local_to_global_buffers[
                                        local_buf.name
                                    ]
                                    if global_buffer.name is not None
                                ):
                                    return local_buf
                            return None

                        local_buf_prefix = "local_buffer_data"
                        # Share existing local buffer
                        local_buffer_used = try_share_local_buffer(
                            local_buffer_layout, local_buffers
                        )
                        if not local_buffer_used:
                            # Create new local buffer
                            local_buffer_used = ir.Buffer(
                                name=f"{local_buf_prefix}_{len(local_buffers)}",
                                layout=local_buffer_layout,
                            )
                            local_buffers.append(local_buffer_used)
                            local_to_global_buffers[local_buffer_used.name] = []  # type: ignore[index]

                        local_to_global_buffers[local_buffer_used.name].append(
                            global_buffer,
                        )

            with LocalBufferContext(kernel_group.args) as scope:
                if len(local_buffers) > 0:
                    for local_buffer in local_buffers:
                        assert local_buffer.name is not None
                        scope.add_local_buffer(
                            local_buffer, local_to_global_buffers[local_buffer.name]
                        )
                for _node in node.get_outer_nodes():
                    assert isinstance(_node, (FusedSchedulerNode, SchedulerNode))
                    cpp_kernel_proxy = self.kernel_proxy_cls(kernel_group)
                    cpp_kernel_proxy.codegen_nodes(_node.get_nodes())  # type: ignore[arg-type]
                    cpp_kernel_proxy_list.append(cpp_kernel_proxy)
                    nodes_list.append(_node.get_nodes())  # type: ignore[arg-type]

                if not node.check_outer_fusion_loop_level_attr(
                    cpp_kernel_proxy_list, node.outer_loop_fusion_depth
                ):
                    for removed_buffer in scope.removed_buffers:
                        # Restore the removed buffers by this context before
                        # fallback to codegen without using Local Buffer
                        V.graph.removed_buffers.remove(removed_buffer)
                    return False
                metrics.cpp_outer_loop_fused_inner_counts.append(
                    metrics.CppOuterLoopFusedCount(
                        len(cpp_kernel_proxy_list),
                        local_buffer_number=len(scope.local_buffers),
                    )
                )
                outer_fusion_cpp_kernel_proxy = node.merge_outer_fusion_kernels(
                    cpp_kernel_proxy_list,
                )
                kernel_group.finalize_kernel(
                    outer_fusion_cpp_kernel_proxy,
                    [*itertools.chain.from_iterable(nodes_list)],
                )

            return True

        if not try_outer_loop_fusion_with_local_buf(node):
            # Reset generated_cpp_vec_kernel_count to codegen again
            metrics.generated_cpp_vec_kernel_count = generated_cpp_vec_kernel_count
            cpp_kernel_proxy_list.clear()
            nodes_list.clear()
            # Similar as comment in
            # https://github.com/pytorch/pytorch/blob/469383755fe416eb1c41fa724762ad3eaecdff07/torch/_inductor/codegen/cpp.py#L3269-L3272
            # Kernels share the same global contexts like V.graph.wrapper_code, V.kernel.args.
            with torch._inductor.config.patch(inplace_buffers=False):
                for _node in node.get_outer_nodes():
                    assert isinstance(_node, (FusedSchedulerNode, SchedulerNode))
                    _nodes: list[SchedulerNode] = _node.get_nodes()  # type: ignore[assignment]
                    cpp_kernel_proxy = self.kernel_proxy_cls(kernel_group)
                    cpp_kernel_proxy.codegen_nodes(_nodes)
                    kernel_group.finalize_kernel(cpp_kernel_proxy, _nodes)

    def codegen_node(
        self,
        node: Union[OuterLoopFusedSchedulerNode, FusedSchedulerNode, SchedulerNode],
    ):
        """
        Turn an set of pre-fused nodes into a C++ kernel.
        """
        kernel_group = self.kernel_group

        if isinstance(node, OuterLoopFusedSchedulerNode):
            self.codegen_outer_loop_node(node)
        else:
            nodes: list[SchedulerNode] = node.get_nodes()  # type: ignore[assignment]
            nodes = self.try_loop_split(nodes)
            cpp_kernel_proxy = self.kernel_proxy_cls(kernel_group)
            cpp_kernel_proxy.codegen_nodes(nodes)
            kernel_group.finalize_kernel(cpp_kernel_proxy, nodes)

        args_num = self._get_scheduled_num_args()
        if args_num > CppScheduling.MAX_FUSED_KERNEL_ARGS_NUM:
            self._set_flush_status(True)

    def is_cpp_template(self, node: BaseSchedulerNode) -> bool:
        return isinstance(node, SchedulerNode) and isinstance(
            node.node, ir.CppTemplateBuffer
        )

    def codegen_template(
        self,
        template_node: BaseSchedulerNode,
        epilogue_nodes: Sequence[BaseSchedulerNode],
        prologue_nodes: Sequence[BaseSchedulerNode],
    ):
        """
        Codegen a CPP template, possibly with fused epilogues
        """
        assert not prologue_nodes

        # remove MultiOutput from epilogue_nodes
        epilogue_nodes = [
            epilogue_node
            for epilogue_node in epilogue_nodes
            if isinstance(epilogue_node, (SchedulerNode, FusedSchedulerNode))
        ]
        # The counter cpp_templated_kernel_counter is used for verifying if a
        # a templated kernel was successfully compiled in a UT
        counters["inductor"]["cpp_templated_kernel_counter"] += 1
        counters["inductor"]["cpp_epilogue_fusion_counter"] += len(epilogue_nodes)
        assert self.is_cpp_template(template_node), (
            "Template node passed to CppScheduler.codegen_template must be a SchedulerNode that wraps a CppTemplateBuffer"
        )
        template_node = cast(SchedulerNode, template_node)
        _, (_, rnumel) = template_node.group
        assert rnumel == ()
        ctb: ir.CppTemplateBuffer = cast(ir.CppTemplateBuffer, template_node.node)
        epilogue_ir_nodes: list[Optional[ir.Operation]] = [
            n.node for n in epilogue_nodes
        ]
        assert all(isinstance(n, ir.ComputedBuffer) for n in epilogue_ir_nodes), (
            "Epilogue nodes must all be instances of ir.ComputedBuffer"
        )

        def template_buffer_has_other_users(
            template_buffer, outputs_by_name, epilogue_nodes
        ):
            if not epilogue_nodes:
                return False

            assert template_buffer.get_name() in outputs_by_name
            users = outputs_by_name[template_buffer.get_name()].users
            return not all(
                isinstance(user.node, BaseSchedulerNode)
                and user.node.node in epilogue_nodes
                for user in users
            )

        flag_template_buffer_has_other_users = template_buffer_has_other_users(
            ctb, template_node.outputs_by_name, epilogue_ir_nodes
        )
        kernel, render = ctb.make_kernel_render(  # type: ignore[misc]
            ctb,
            flag_template_buffer_has_other_users=flag_template_buffer_has_other_users,
            epilogue_nodes=epilogue_ir_nodes,
        )
        with kernel:
            if not is_multi_outputs_template(template_node.node):
                template_node.mark_run()  # type: ignore[attr-defined]
            for node in epilogue_nodes:
                node.mark_run()  # type: ignore[attr-defined]
            src_code = render()

        with V.set_kernel_handler(kernel):
            node_schedule = [template_node, *epilogue_nodes]
            kernel_name = self.define_kernel(src_code, node_schedule, kernel.args)

        if is_multi_outputs_template(template_node.node):
            # For multi outputs template, allocate buffers for each output after the epilogue
            # codegen to which determines if the buffer has been removed.
            assert len(template_node.outputs) == 1, (
                "Multi outputs template should be with 1 output template buffer of MultiOutputLayout"
            )
            for user in template_node.outputs[0].users:
                assert isinstance(user.node, ExternKernelSchedulerNode), (
                    "Multi outputs template should be with ExternKernelSchedulerNode"
                )
                assert isinstance(user.node.node, ir.MultiOutput), (
                    "Multi outputs template has multi users with MultiOutput"
                )
                user.node.mark_run()

        self.codegen_comment(node_schedule, kernel_name)
        kernel.call_kernel(kernel_name, ctb)
        V.graph.removed_buffers |= kernel.removed_buffers
        self.free_buffers_in_scheduler()

    def _get_scheduled_num_args(self):
        return self.kernel_group.get_num_args()

    def ready_to_flush(self):
        return self._ready_to_flush

    def codegen_sync(self):
        pass

    def define_kernel(self, src_code, nodes, kernel_args=None):
        wrapper = V.graph.wrapper_code
        if src_code in wrapper.src_to_kernel:
            kernel_name = wrapper.src_to_kernel[src_code]
        else:
            fused_name = (
                get_fused_kernel_name(nodes, config.cpp.descriptive_names)
                if config.cpp.descriptive_names
                else ""
            )
            kernel_name = "_".join(["cpp", fused_name, wrapper.next_kernel_suffix()])
            wrapper.src_to_kernel[src_code] = kernel_name
            kernel_decl_name = kernel_name if V.graph.cpp_wrapper else "kernel"
            src_code = src_code.replace(str(Placeholder.KERNEL_NAME), kernel_decl_name)
            src_code = src_code.replace(str(Placeholder.DESCRIPTIVE_NAME), kernel_name)
            # TODO(voz): Ostensibly, we should not need this. But there are cases where C++ codegen does
            # not use BracesBuffer, so we have no good indicator of a C++ buffer atm.
            src_code = src_code.replace("#pragma CMT", "//")

            # Get the lines in the source code representing the function definition,
            # excluding the first line including cpp_prefix.h.
            first_char = src_code.rfind('extern "C"')
            last_char = src_code.find(")", first_char)
            if _IS_WINDOWS:
                # get_export_declaration introduced one more ')' in Windows
                last_char = src_code.find(")", last_char + 1)
            kernel_definition = f"{src_code[first_char : last_char + 1]};\n"

            compile_wrapper = IndentedBuffer()
            args = self.kernel_group.args if kernel_args is None else kernel_args
            _, _, arg_types = args.cpp_argdefs()
            if not V.graph.cpp_wrapper:
                compile_wrapper.writeline(
                    f"async_compile.cpp_pybinding({arg_types!r}, r'''"
                )
            compile_wrapper.splice(src_code, strip=True)
            if not V.graph.cpp_wrapper:
                compile_wrapper.writeline("''')")
            wrapper.define_kernel(
                kernel_name,
                compile_wrapper.getvalue(),
                gpu=False,
                cpp_definition=kernel_definition,
            )
        return kernel_name

    def flush(self):
        src_code = self.kernel_group.codegen_group()
        if src_code:
            kernel_name = self.define_kernel(
                src_code, self.kernel_group.scheduled_nodes
            )
            self.codegen_comment(self.kernel_group.scheduled_nodes, kernel_name)
            if config.cpp.enable_kernel_profile:
                V.graph.wrapper_code.write_kernel_context_guard_begin()
                V.graph.wrapper_code.write_kernel_context_guard(
                    kernel_name,
                    self.kernel_group.scheduled_nodes,  # type: ignore[arg-type]
                )
            self.kernel_group.call_kernel(V.graph.wrapper_code, kernel_name)
            if config.cpp.enable_kernel_profile:
                V.graph.wrapper_code.write_kernel_context_guard_end()

        self.reset_kernel_group()
        self._set_flush_status(False)

    def codegen_comment(self, node_schedule, kernel_name=None):
        # below add provenance tracing info for cpu CppKernel types
        wrapper = V.graph.wrapper_code
        debug_handle = set_kernel_post_grad_provenance_tracing(
            node_schedule,  # type: ignore[arg-type]
            # pyrefly: ignore [bad-argument-type]
            kernel_name,
        )
        wrapper.write_provenance_debug_handle(kernel_name, debug_handle)


class KernelGroup:
    def __init__(self):
        super().__init__()
        self.args = KernelArgs()
        self.loops_code = BracesBuffer()
        self.ws = WorkSharing(self.loops_code)
        self.stack = contextlib.ExitStack()
        self.stack.enter_context(self.ws)
        self.scheduled_nodes = []

    def new_kernel(self, cls, *args):
        return cls(self.args, parallel_num_threads(), *args)

    def finalize_kernel(self, new_kernel, nodes):
        self.scheduled_nodes += nodes
        code = self.loops_code
        ws = self.ws
        new_kernel.codegen_loops(code, ws)

    def get_num_args(self):
        arg_defs, _call_args, _arg_types = self.args.cpp_argdefs()
        args_num = len(arg_defs)
        return args_num

    def codegen_group(self, name=None) -> str:
        self.stack.close()
        if not self.scheduled_nodes:
            return ""
        code = BracesBuffer()
        # 1. Include header files
        # TODO: support kernel profile on other platforms
        enable_kernel_profile = config.cpp.enable_kernel_profile and sys.platform in [
            "linux",
            "win32",
        ]
        if enable_kernel_profile:
            code.writelines(["#include <torch/csrc/inductor/aoti_runtime/utils.h>"])
        code.writeline("#include <torch/csrc/inductor/cpp_prefix.h>")

        # 2. Function definition
        kernel_decl_name = str(Placeholder.KERNEL_NAME) if name is None else name
        kernel_name = str(Placeholder.DESCRIPTIVE_NAME) if name is None else name
        arg_defs, _, _ = self.args.cpp_argdefs()
        arg_defs = ",\n".ljust(25).join(arg_defs)
        func_export_decl = get_export_declaration()
        inline_attr = (
            "C10_ALWAYS_INLINE_ATTRIBUTE" if config.cpp.force_inline_kernel else ""
        )
        code.writeline(
            f'extern "C" {func_export_decl} void {inline_attr} {kernel_decl_name}({arg_defs})'
        )

        # 3. Function body
        with code.indent():
            if enable_kernel_profile:
                graph_id = V.graph.graph_id
                prefix = "graph_" + str(graph_id) + "_" if graph_id is not None else ""
                code.writelines(
                    [
                        (
                            "torch::aot_inductor::RAIIAtenRecordFunctionHandle "
                            f'record_{prefix + kernel_name}_("{prefix + kernel_name}", nullptr);'
                        )
                    ]
                )
            for old, new in self.args.aliases():
                code.writeline(f"auto {old} = {new};")
            code.splice(self.loops_code)
        return code.getvalue()

    def call_kernel(self, wrapper, kernel_name):
        _, call_args, arg_types = self.args.cpp_argdefs()
        wrapper.generate_kernel_call(
            kernel_name,
            call_args,
            triton=False,
            arg_types=arg_types,
        )


class WorkSharing:
    def __init__(self, code):
        self.code = code
        self.in_parallel = False
        self.num_threads = None
        self.stack = contextlib.ExitStack()

    def parallel(self, threads):
        if self.in_parallel and threads != self.num_threads:
            # wrong number of threads
            self.close()
        if not self.in_parallel:
            self.num_threads = threads
            self.in_parallel = True
            if config.cpp.dynamic_threads:
                self.code.writeline("#pragma omp parallel")
            else:
                self.code.writeline(f"#pragma omp parallel num_threads({threads})")
            self.stack.enter_context(self.code.indent())
            self.code.writeline(
                "int tid = omp_get_thread_num();",
            )

    def single(self):
        if self.in_parallel:
            self.code.writeline("#pragma omp single")
        return self.in_parallel

    def close(self):
        self.stack.close()
        self.in_parallel = False

    def __enter__(self):
        self.stack.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stack.__exit__(exc_type, exc_val, exc_tb)


@dataclasses.dataclass
class LoopLevel:
    var: Optional[sympy.Expr] = None
    size: Optional[sympy.Expr] = None
    offset: sympy.Expr = sympy.S.Zero
    # Note [tiled_size]
    # We may do loop-tiling at this loop level.
    # When var is in [offset, tiled_size), we will perform the vectorization kernel.
    # When var is in [tiled_size, size), we will perform the scalar or masked vectorization kernel.
    # for (var = offset; var < size; var += steps) {
    #     if (var >= offset && var < tiled_size) vec_loop_body();
    #     if (var >= tiled_size && var < size) scalar_or_maskvec_loop_body();
    # }
    tiled_size: sympy.Expr = sympy.S.Zero
    steps: sympy.Expr = sympy.S.One
    parallel: int = 0
    simd_omp: bool = False
    simd_vec: bool = False
    collapsed: bool = False
    is_reduction: bool = False

    def __post_init__(self):
        # Regarding the C++/OpenMP backend, `cpu_vec_isa.pick_vec_isa()` to check
        # vectorization ISA is a time-consuming and one-shot operation. It leads
        # to taking a longer time to import `codegen.cpp` package because the
        # `LoopLevel` of the package is decorated by `@dataclasses.dataclass` while
        # the decorator will invoke `cpu_vec_isa.pick_vec_isa()` to initialize the
        # `simd_nelements` of the `LoopLevel`. It might introduce additional compilation
        # overhead to the Triton backend. Therefore, we moved the `simd_nelements` to
        # `__post_init__`
        picked_vec_isa: cpu_vec_isa.VecISA = cpu_vec_isa.pick_vec_isa()
        self.simd_nelements: int = picked_vec_isa.nelements() if picked_vec_isa else 0

    def tile(self, factor):
        sympy_factor = sympy.Integer(factor)
        loop = LoopLevel(self.var, self.size)
        loop.steps = sympy_factor
        loop.simd_vec = True
        loop.tiled_size = FloorDiv(loop.size, sympy_factor) * sympy_factor
        loop.parallel = self.parallel
        loop.collapsed = False
        loop.is_reduction = self.is_reduction
        return loop

    def lines(self):
        offset_expr = cexpr_index(self.offset)
        size_expr = cexpr_index(self.size)
        if config.cpp.no_redundant_loops and offset_expr == size_expr:
            return None
        simd = (
            f"simd simdlen({self.simd_nelements}) "
            if self.simd_omp and self.simd_nelements > 1
            else ""
        )
        if self.parallel:
            # TODO(jansel): look into chunk size and other schedules
            line1 = "#pragma omp for"
            if self.parallel > 1:
                line1 += f" collapse({self.parallel})"
            if self.simd_omp:
                line1 = line1.replace(" for ", f" for {simd}")
        elif self.simd_vec:
            line1 = ""
        elif self.simd_omp:
            line1 = f"#pragma omp {simd}"
        elif not self.is_reduction and cpp_builder.is_gcc():
            line1 = "#pragma GCC ivdep"
        else:
            line1 = ""
        offset_str = f"{INDEX_TYPE} {self.var}={offset_expr}"
        size_str = f"{self.var}<{size_expr}"
        if self.steps.is_number:
            steps_str = f"{self.var}+={cexpr_index(self.steps)}"
        else:
            # If the step size is 0, change it to 1 because a step size of 0
            # will cause floating point exception (core dump) during parallelization.
            steps_str = (
                f"{self.var}+=({cexpr_index(self.steps)} == 0 ? "
                f"1 : {cexpr_index(self.steps)})"
            )
        line2 = f"for({offset_str}; {size_str}; {steps_str})"
        if self.collapsed or not line1:
            return [line2]
        return [line1, line2]


@dataclasses.dataclass
class LoopNest:
    """
    A loop-nest-like structure. It is built with the `build` method
    as a loop nest and then will perform loop-tiling at some depth.

    A typical case is for vectorization, where we typically do loop-tiling
    at the innermost loop level. A more complicated case is when we do
    2D tiling at both the innermost and outer levels.
    """

    loops: Optional[list[LoopLevel]] = None
    kernel: Optional[CppKernel] = None

    @staticmethod
    def build(kernel: CppKernel):
        """Build a LoopNest with the given `kernel` as the leaf"""
        itervars = kernel.itervars
        ranges = kernel.ranges
        reduction_depth = kernel.reduction_depth
        assert reduction_depth is not None

        loops: Optional[list[LoopLevel]] = None
        for loop_idx, (var, size) in enumerate(zip(itervars, ranges)):
            loop = LoopLevel(var, size)
            if not loops:
                loops = [loop]
            else:
                loops.append(loop)
            if loop_idx >= reduction_depth:
                loop.is_reduction = kernel.is_reduction

        loop_nest = LoopNest(loops)
        return loop_nest

    def __bool__(self):
        return bool(self.loops)

    @cache_on_self
    def max_parallel_depth(self):
        """
        Maximal allowed depth for parallelism: All reduction or non-reduction levels.
        When the range of the first inner loop beyond the maximum parallel depth is much
        larger than the range of all outer loops within the maximum parallel depth,
        change the starting depth of parallelism to the first inner loop and recalculate
        the maximum parallel depth.
        """
        if self.loops is None:
            return ParallelDepth(parallel_depth=0, start_depth=0)

        start_depth = 0
        max_depth = 0
        is_reduction = self.loops[0].is_reduction
        num_steps = sympy.Integer(1)
        for loop in self.loops:
            if loop.is_reduction != is_reduction:
                break
            num_steps = num_steps * FloorDiv(loop.size, loop.steps)
            max_depth += 1

        def get_simd_vec_depth(loops):
            # Return the first loop level which is simd_vec
            for i, loop in enumerate(loops):
                if loop.simd_vec:
                    return i
            return None

        simd_vec_depth = get_simd_vec_depth(self.loops)

        def has_scalar_kernel(loop_nest: LoopNest):
            assert isinstance(loop_nest.kernel, CppKernelProxy)
            return any(
                not isinstance(kernel, CppVecKernel)
                for kernel in loop_nest.kernel.kernels
            )

        # When the number of steps of the first inner loop is much larger than the number of steps of
        # all outer loops, change `start_depth` to the first inner loop and recalculate `max_depth`.
        if (
            max_depth < len(self.loops)
            and isinstance(num_steps, sympy.Integer)
            and isinstance(self.loops[max_depth].size, sympy.Integer)
            and num_steps * 300
            < FloorDiv(self.loops[max_depth].size, self.loops[max_depth].steps)
            and not (
                # Disable parallel reduction under the vec loop
                simd_vec_depth is not None
                and max_depth > simd_vec_depth
                and self.loops[max_depth].is_reduction
                and has_scalar_kernel(self)
            )
        ):
            start_depth = max_depth
            max_depth = 0
            is_reduction = self.loops[start_depth].is_reduction
            for i in range(start_depth, len(self.loops)):
                if self.loops[i].is_reduction != is_reduction:
                    break
                max_depth += 1
        return ParallelDepth(parallel_depth=max_depth, start_depth=start_depth)

    def mark_parallel(self, par_depth):
        assert par_depth.parallel_depth <= self.max_parallel_depth().parallel_depth, (
            "Parallel depth cannot exceed the maximal allowed parallel depth"
        )
        assert self.loops is not None
        assert len(self.loops) >= par_depth.parallel_depth
        loop = self.loops[par_depth.start_depth]
        loop.parallel = par_depth.parallel_depth
        if loop.is_reduction:
            # pyrefly: ignore [bad-assignment]
            metrics.parallel_reduction_count += 1
        for i in range(par_depth.start_depth + 1, par_depth.parallel_depth):
            self.loops[i].collapsed = True

    def tile(self, depth, factor):
        """
        Do loop-tiling at the `depth` level with `factor`.
            for (x0 = 0; x0 < x0_end; x0++)
            ->
            for (x0 = 0; x0 < x0_end; x0 += factor)
        See details in Note [tiled_size].
        """
        assert self.loops
        self.loops[depth] = self.loops[depth].tile(factor)
        return self.loops[depth]

    def get_kernel(self) -> CppKernel:
        assert self.kernel
        return self.kernel

    def set_kernel(self, kernel):
        self.kernel = kernel

    def from_loop_level(self, level: int):
        assert self.loops
        assert len(self.loops) >= level
        loops = None if level == len(self.loops) else self.loops[level:]
        return LoopNest(loops, self.kernel)
