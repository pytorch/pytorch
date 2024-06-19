# mypy: allow-untyped-defs
import contextlib
import dataclasses
import functools
import itertools
import logging
import math
import re
import sys
from copy import copy, deepcopy
from enum import Enum
from typing import Any, cast, Dict, List, Optional, Sequence, Set, Tuple, Union

import sympy

import torch
import torch.fx
from torch._inductor import dependencies
from torch._prims_common import is_float_dtype
from torch.utils import _pytree as pytree
from torch.utils._sympy.functions import CeilDiv, FloorDiv, ModularIndexing
from torch.utils._sympy.symbol import free_symbol_is_type, symbol_is_type, SymT
from torch.utils._sympy.value_ranges import bound_sympy, ValueRanges
from ..._dynamo.utils import counters

from .. import codecache, config, ir, metrics
from ..codegen.wrapper import WrapperCodeGen
from ..optimize_indexing import range_expressable_in_32_bits
from ..scheduler import (
    BaseSchedulerNode,
    BaseScheduling,
    ForeachKernelSchedulerNode,
    FusedSchedulerNode,
    Scheduler,
    SchedulerNode,
)
from ..utils import (
    cache_on_self,
    get_bounds_index_expr,
    get_fused_kernel_name,
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
    CppWrapperKernelArgs,
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
    cexpr,
    cexpr_index,
    DTYPE_TO_CPP,
    INDEX_TYPE,
    unify_mask_base_type,
    value_to_cpp,
)

schedule_log = torch._logging.getArtifactLogger(__name__, "schedule")

NATIVE_OMP_RTYPES = {"+", "*", "^", "||", "min", "max"}
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
VECTORIZABLE_RTYPES = {
    "max",
    "min",
    "sum",
    "prod",
    "xor_sum",
    "welford_reduce",
    "welford_combine",
}

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
    "Optional": "c10::optional",
}

DTYPE_LOWP_FP = [
    torch.bfloat16,
    torch.float16,
]


BIN_CMP_OPS = ["eq", "ne", "le", "ge", "lt", "gt"]


def reduction_init(reduction_type, dtype):
    if dtype in DTYPE_LOWP_FP:
        # Since load promotes all half-precision inputs to float, the initial
        # constant for reduction must be promoted as well
        dtype = torch.float32
    if reduction_type in ("xor_sum", "sum", "any"):
        return 0
    if reduction_type == "prod":
        return 1
    if reduction_type in {"max", "argmax"}:
        return (
            f"-std::numeric_limits<{DTYPE_TO_CPP[dtype]}>::infinity()"
            if is_float_dtype(dtype)
            else f"std::numeric_limits<{DTYPE_TO_CPP[dtype]}>::min()"
        )
    if reduction_type in {"min", "argmin"}:
        return (
            f"std::numeric_limits<{DTYPE_TO_CPP[dtype]}>::infinity()"
            if is_float_dtype(dtype)
            else f"std::numeric_limits<{DTYPE_TO_CPP[dtype]}>::max()"
        )
    if is_welford_reduction(reduction_type):
        return f"Welford<{DTYPE_TO_CPP[dtype]}>()"
    raise AssertionError(reduction_type)


def reduction_acc_type(reduction_type, dtype):
    assert reduction_type not in {"argmin", "argmax"}
    scalar_type = DTYPE_TO_CPP[DTYPE_TO_COMPUTATION_DTYPE[dtype]]
    if is_welford_reduction(reduction_type):
        return f"Welford<{scalar_type}>"

    return scalar_type


def reduction_combine(reduction_type, var, next_value):
    if reduction_type == "sum":
        return f"{var} + {next_value}"
    if reduction_type == "prod":
        return f"{var} * {next_value}"
    if reduction_type == "xor_sum":
        return f"{var} ^ {next_value}"
    if reduction_type == "any":
        return f"{var} || {next_value}"
    if reduction_type in ("min", "max"):
        return f"{reduction_type}_propagate_nan({var}, {next_value})"
    if reduction_type == "welford_reduce":
        return f"welford_combine({var}, {next_value})"
    if reduction_type == "welford_combine":
        if isinstance(next_value, tuple):
            mean, m2, weight = next_value
        else:
            mean, m2, weight = reduction_project(reduction_type, next_value)
        return f"welford_combine({var}, {{{mean}, {m2}, {weight}}})"
    raise AssertionError(reduction_type)


def reduction_project(reduction_type, acc):
    if is_welford_reduction(reduction_type):
        return f"{acc}.mean", f"{acc}.m2", f"{acc}.weight"
    elif reduction_type in {"argmin", "argmax"}:
        return f"{acc}.index"
    return acc


def is_to_lowp_dtype(expr):
    to_exprs = ["convert<half>", "convert<bfloat16>"]
    return any(to_expr in expr for to_expr in to_exprs)


def get_lowp_to_high_prec_expr(lowp_var, dtype, kernel):
    if isinstance(kernel, CppVecKernel):
        return f"at::vec::convert<{DTYPE_TO_CPP[dtype]}>({lowp_var})"
    else:
        assert isinstance(kernel, CppKernel)
        return f"c10::convert<{DTYPE_TO_CPP[dtype]}>({lowp_var})"


index_value_name_counter = 1


def argmax_argmin_prefix(reduction_type, src_dtype, tmpvar):
    global index_value_name_counter
    num_threads = (
        "max_threads" if config.cpp.dynamic_threads else parallel_num_threads()
    )
    struct_name = f"IndexValue_{index_value_name_counter}"
    index_value_name_counter += 1

    # A small annoyance, due to it being a little cumbersome to just throw {} into strings
    prefix = [
        f"struct {struct_name} {{size_t index; {DTYPE_TO_CPP[src_dtype]} value;}};",
        f"{struct_name} {tmpvar}{{0, {reduction_init(reduction_type, src_dtype)}}};",
    ]
    local_init = [
        f"{struct_name} {tmpvar}_local{{0, {reduction_init(reduction_type, src_dtype)}}};",
    ]
    tmpvar_per_thd = f"{tmpvar}_arr[{num_threads}]"
    parallel_prefix = [
        f"{struct_name} {tmpvar_per_thd};",
    ]
    return prefix, parallel_prefix, local_init


@functools.lru_cache
def stride_at(index: sympy.Expr, var: sympy.Symbol):
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
def stride_at_vec_range(index: sympy.Expr, var: sympy.Symbol, vec_length: int):
    index_vec_simplified = simplify_index_in_vec_range(index, var, vec_length)
    return stride_at(index_vec_simplified, var)


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
        outer_fused_nodes: List[Union[FusedSchedulerNode, SchedulerNode]],
        outer_loop_fusion_depth,
    ):
        self.outer_fused_nodes: List[
            Union[FusedSchedulerNode, SchedulerNode]
        ] = outer_fused_nodes
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
            left_loop_level: LoopLevel,
            right_loop_level: LoopLevel,
            loop_fusion_depth: int,
        ) -> bool:
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
                # If the next loop level is expected to undergo outer loop fusion,
                # there should be no kernel present at the current loop level.
                assert (
                    left_loop_level.kernel is None and right_loop_level.kernel is None
                )
                # Check next loop level attr
                if any(
                    # Assume no main/tail loop split at any outer loop fusion depth
                    # Given no clear performance benefit for this complex case
                    len(loop_level.inner) != 1
                    for loop_level in [left_loop_level, right_loop_level]
                ) or not _inner(
                    left_loop_level.inner[0],
                    right_loop_level.inner[0],
                    loop_fusion_depth,
                ):
                    return False

            return True

        for idx in range(len(cpp_kernel_proxy_list) - 1):
            left_loop_nest = cpp_kernel_proxy_list[idx].loop_nest
            right_loop_nest = cpp_kernel_proxy_list[idx + 1].loop_nest
            if any(
                # Assume no main/tail loop split at any outer loop fusion depth
                len(loop_nest.root) != 1
                for loop_nest in [left_loop_nest, right_loop_nest]
            ) or not _inner(
                left_loop_nest.root[0], right_loop_nest.root[0], outer_loop_fusion_depth
            ):
                return False

        return True

    def merge_outer_fusion_kernels(
        self,
        cpp_kernel_proxy_list,
    ):
        loop_nest_list: List[LoopNestWithSplit] = [
            kernel.loop_nest for kernel in cpp_kernel_proxy_list
        ]
        metrics.cpp_outer_loop_fused_inner_counts.append(len(loop_nest_list))

        kernel_group = cpp_kernel_proxy_list[0].kernel_group

        def _merge_outer_fusion_loop_levels(
            loop_level_nested_list: List[List["LoopLevel"]],
            outer_loop_fusion_depth,
        ):
            assert outer_loop_fusion_depth >= 1
            # Assume no main/tail loop split at any outer loop fusion depth
            assert all(
                len(loop_level_list) == 1 for loop_level_list in loop_level_nested_list
            )
            if (outer_loop_fusion_depth := outer_loop_fusion_depth - 1) >= 1:
                # Further merge the next loop level
                next_loop_level_nested_list = [
                    loop_level_list[0].inner
                    for loop_level_list in loop_level_nested_list
                ]
                _merge_outer_fusion_loop_levels(
                    next_loop_level_nested_list,
                    outer_loop_fusion_depth,
                )
            else:
                outer_loop_fused_kernel = OuterLoopFusedKernel(kernel_group)
                loop_level_of_first_kernel = loop_level_nested_list[0][0]
                for kernel_idx in range(len(loop_level_nested_list)):
                    outer_loop_fused_kernel.inner.append(
                        deepcopy(loop_level_nested_list[kernel_idx][0]),
                    )
                loop_level_of_first_kernel.inner = []
                loop_level_of_first_kernel.kernel = outer_loop_fused_kernel

        # Merge the List[LoopNestWithSplit] from cpp_kernel_proxy_list
        # into cpp_kernel_proxy_list[0].loop_nest
        _merge_outer_fusion_loop_levels(
            [_loop_nest.root for _loop_nest in loop_nest_list],  # type: ignore[misc]
            self.outer_loop_fusion_depth,
        )
        return cpp_kernel_proxy_list[0]


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


def get_opt_ctx(node: torch.fx.Node) -> OptimizationContext:
    return node.meta.get(OptimizationContext.key, None)


def get_current_node_opt_ctx() -> OptimizationContext:
    assert V.interpreter.current_node
    return get_opt_ctx(V.interpreter.current_node)


class CppCSEVariable(CSEVariable):
    def __init__(self, name, bounds: ValueRanges[Any]):
        super().__init__(name, bounds)
        self.is_vec = False
        self.dtype: Optional[torch.dtype] = None
        self.dependent_itervars: Set[sympy.Symbol] = set()

    def __repr__(self):
        return (
            f"CppCSEVariable(name: {self.name}, bounds: {self.bounds}, is_vec: {self.is_vec}, dtype: {self.dtype}, "
            f"dependent_itervars: {self.dependent_itervars})"
        )

    def update_on_args(self, name, args, kwargs):
        if name == "load":
            # args[1] is index
            self._set_dependent_itervars(args[1])
        else:
            # propagate relevant itervars and is_vec from args
            self.dependent_itervars.update(
                *[
                    arg.dependent_itervars
                    for arg in args
                    if isinstance(arg, CppCSEVariable)
                ]
            )
            if name == "index_expr":
                self._set_dependent_itervars(args[0])
            if any(arg.is_vec for arg in args if isinstance(arg, CppCSEVariable)):
                self.is_vec = True
        # NOTE [dtype of CppCSEVariable]
        # Deciding dtype according to the current optimization context is not
        # always accurate since the dtypes are initialized during dtype propagation
        # at the beginning of the codegen. It is possible that some ops are invoked
        # during the codegen of the current op and take different dtypes from the
        # current op.
        # TODO(jgong5): A more accurate way of deciding the dtype of the variables is to
        # propagate the dtypes here inside `update_on_args`.
        if (
            hasattr(V.interpreter, "current_node")
            and get_current_node_opt_ctx() is not None
        ):
            self.dtype = get_current_node_opt_ctx().dtype

        if name in BIN_CMP_OPS:
            self.dtype = torch.bool

    def _set_dependent_itervars(self, index: sympy.Expr):
        """
        Set the relevant itervars for this variable based on the `index` expression.
        This includes the itervars directly used in the `index` as well as relevant itervars
        of other cse variables used in the `index`.
        """
        for s in index.free_symbols:
            if s in V.kernel.itervars:
                self.dependent_itervars.add(s)  # type: ignore[arg-type]
            elif s.name in V.kernel.cse.varname_map:  # type: ignore[attr-defined]
                self.dependent_itervars.update(
                    V.kernel.cse.varname_map[s.name].dependent_itervars  # type: ignore[attr-defined]
                )

    def depends_on(self, itervar: sympy.Symbol):
        return itervar in self.dependent_itervars


class CppOverrides(OpOverrides):
    """Map element-wise ops to C++"""

    @staticmethod
    def add(a, b):
        return f"decltype({a})({a} + {b})"

    @staticmethod
    def sub(a, b):
        return f"decltype({a})({a} - {b})"

    @staticmethod
    def mul(a, b):
        return f"decltype({a})({a} * {b})"

    @staticmethod
    def to_dtype(x, dtype, src_dtype=None):
        assert dtype in DTYPE_TO_CPP, f"{dtype} missing from {__name__}.DTYPE_TO_CPP"
        return f"c10::convert<{DTYPE_TO_CPP[dtype]}>({x})"

    @staticmethod
    def to_dtype_bitcast(x, dtype, src_dtype):
        assert dtype in DTYPE_TO_CPP, f"{dtype} missing from {__name__}.DTYPE_TO_CPP"
        if src_dtype in (torch.float16, torch.bfloat16):
            # c10::bit_cast requires the source and target have the bitwidth.
            # Because the input tensor's dtype could be promoted, e.g. from float16 to
            # float, we have to cast the tensor to its original source dtype before
            # invoking bit_cast. We also need to convert the bit-casted tensor
            # back to float to make sure we keep using higher precision values
            # for the rest of the computation.
            cast_x = f"c10::convert<{DTYPE_TO_CPP[src_dtype]}>({x})"
            cast_x = f"c10::bit_cast<{DTYPE_TO_CPP[dtype]}>({cast_x})"
            return f"c10::convert<{DTYPE_TO_CPP[torch.float32]}>({cast_x})"
        else:
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
        return f"std::signbit({x})"

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
        if all(cache_key in V.kernel.cse.cache for cache_key in cache_keys):
            return tuple(V.kernel.cse.cache[cache_key] for cache_key in cache_keys)

        code = BracesBuffer()
        exponent = V.kernel.cse.newvar()
        mantissa = V.kernel.cse.newvar()
        code.writeline(f"int32_t {exponent};")
        code.writeline(f"auto {mantissa} = std::frexp({x}, &{exponent});")
        V.kernel.compute.splice(code)
        cse_vars = (mantissa, exponent)
        for cache_key, cse_var in zip(cache_keys, cse_vars):
            V.kernel.cse.cache[cache_key] = cse_var
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
        opt_ctx: OptimizationContext = get_current_node_opt_ctx()
        assert opt_ctx and opt_ctx.dtype is not None, opt_ctx
        dtype = opt_ctx.dtype
        if dtype in DTYPE_LOWP_FP:
            # Since load promotes all half-precision inputs to float, constants
            # must be promoted as well
            dtype = torch.float32
        return value_to_cpp(val, DTYPE_TO_CPP[dtype])

    @staticmethod
    def index_expr(expr, dtype):
        opt_ctx: OptimizationContext = get_current_node_opt_ctx()
        assert opt_ctx and opt_ctx.dtype is not None
        dtype = opt_ctx.dtype

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
        return f"decltype({a})({a} << {b})"

    @staticmethod
    def bitwise_right_shift(a, b):
        return f"decltype({a})({a} >> {b})"

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
                    # broadcast scalar args to vector if needed
                    new_args = []
                    vec_dtype = vectors[0].dtype
                    for arg in args:
                        if isinstance(arg, (int, sympy.Expr)):
                            arg_dtype = torch.int64
                            opt_ctx: OptimizationContext = get_current_node_opt_ctx()
                            assert opt_ctx
                            if opt_ctx.dtype is not None:
                                arg_dtype = opt_ctx.dtype
                            if isinstance(arg, sympy.Expr) and not arg.is_number:
                                arg = ops.index_expr(arg, arg_dtype)
                            else:
                                arg = ops.constant(arg, arg_dtype)
                            arg = arg.value if isinstance(arg, OpsValue) else arg
                        if isinstance(arg, CppCSEVariable) and not arg.is_vec:
                            assert isinstance(V.kernel, CppVecKernel)
                            # align scalar data type to the vector for binary ops
                            if len(args) == 2 and arg.dtype != vec_dtype:
                                arg = ops.to_dtype(arg, vec_dtype)
                                arg = arg.value if isinstance(arg, OpsValue) else arg
                                # See NOTE [dtype of CppCSEVariable]: we have to fix arg.dtype since
                                # the dtype from optimization context could be wrong.
                                assert isinstance(arg, CppCSEVariable)
                                arg.dtype = vec_dtype
                            new_arg = V.kernel.broadcast(arg)
                            new_args.append(new_arg)
                        else:
                            new_args.append(arg)
                if vectors:
                    return func(*new_args, **kwargs)
                else:
                    # fallback to scalar ops
                    scalar_ops = super(CppVecOverrides, self)
                    scalar_func = getattr(
                        scalar_ops, func.__name__, scalar_ops.__getattr__(func.__name__)  # type: ignore[attr-defined]
                    )
                    assert scalar_func is not None
                    return scalar_func(*args, **kwargs)

            return wrapper

        for name, method in vars(CppVecOverrides).items():
            if getattr(method, "__class__", None) == staticmethod and name not in [
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
        return f"{a} & {b}"

    @staticmethod
    def logical_not(a):
        return f"~{a}"

    @staticmethod
    def logical_or(a, b):
        return f"{a} | {b}"

    @staticmethod
    def logical_xor(a, b):
        return f"{a} ^ {b}"

    @staticmethod
    def tan(a):
        return f"{a}.tan()"

    @staticmethod
    def tanh(a):
        vec_one = f"decltype({a})(1)"
        vec_two = f"decltype({a})(2)"
        vec_minus_two = f"decltype({a})(-2)"
        return f"{vec_two} / ({vec_one} + ({vec_minus_two} * {a}).exp()) - {vec_one}"

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
        # For real x, asinh(x) = log(x + sqrt(1 + x**2))
        vec_one = f"decltype({x})(1)"
        return f"({x} + ({vec_one} + {x}*{x}).sqrt()).log()"

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
        # a and b are integer type
        _t = f"decltype({a})"
        quot = f"{a} / {b}"
        has_rem = f"({a} % {b} != {_t}(0))"
        is_neg = f"(({a} < {_t}(0)) != ({b} < {_t}(0)))"
        return f"{_t}::blendv({quot}, {quot} - {_t}(1), {has_rem} & {is_neg})"

    @staticmethod
    def truncdiv(a, b):
        # a and b are integer type
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
    def to_dtype(x, dtype, src_dtype=None):
        assert dtype in [
            torch.bool,
            torch.float,
            torch.bfloat16,
            torch.float16,
            torch.uint8,
            torch.int8,
            torch.int32,
            torch.int64,
        ], f"{__name__} does not support {dtype}"
        node: torch.fx.Node = V.interpreter.current_node
        assert node and isinstance(node, torch.fx.Node)
        opt_ctx_x = get_opt_ctx(node.args[1])
        assert opt_ctx_x
        assert opt_ctx_x.dtype is not None
        assert isinstance(V.kernel, CppVecKernel)
        src_dtype = opt_ctx_x.dtype
        src_cpp_type = DTYPE_TO_CPP[src_dtype]
        src_num_vectors = V.kernel._get_num_vectors(src_dtype)
        dst_cpp_type = DTYPE_TO_CPP[dtype]
        dst_num_vectors = V.kernel._get_num_vectors(dtype)
        if src_dtype != torch.bool and dtype == torch.bool:
            return f"{V.kernel._get_mask_type(src_dtype)}::from<{src_cpp_type},{src_num_vectors}>({x})"
        if opt_ctx_x.dtype == torch.bool and dtype != torch.bool:
            return f"{x}.to<{dst_cpp_type},{dst_num_vectors}>()"
        if src_dtype != dtype:
            if src_num_vectors == dst_num_vectors == 1:
                return f"at::vec::convert<{dst_cpp_type}>({x})"
            else:
                return f"at::vec::convert<{dst_cpp_type},{dst_num_vectors},{src_cpp_type},{src_num_vectors}>({x})"
        return f"({x})"

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
        body_code_vec = (
            body_code
            if result.is_vec
            else f"{V.kernel._get_vec_type(dtype)}({body_code})"
        )
        other_code = value_to_cpp(other, DTYPE_TO_CPP[dtype])
        # loading bool as VecMask<float, N>
        other_code_vec = (
            f"{V.kernel._get_mask_type()}::from({other_code})"
            if dtype == torch.bool
            else f"{V.kernel._get_vec_type(dtype)}({other_code})"
        )
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
                    code.writeline(
                        f"return {V.kernel.overrides.where(new_mask, body_vec_var, other_vec_var)};"
                    )
            code.writeline("()")
            csevar = V.kernel.cse.generate(
                V.kernel.compute,
                code,
            )
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
        opt_ctx: OptimizationContext = get_current_node_opt_ctx()
        assert opt_ctx and opt_ctx.dtype is not None
        dtype = opt_ctx.dtype
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
        csevar.update_on_args("index_expr", (expr, dtype), {})
        return csevar


CppVecOverrides._initialize_pointwise_overrides("cppvec")


class CppTile2DOverrides(CppVecOverrides):
    @staticmethod
    def index_expr(expr, dtype):
        assert isinstance(V.kernel, CppTile2DKernel)
        expr = V.kernel.transform_indexing(expr)
        return CppVecOverrides.index_expr(expr, dtype)


class CppKernel(Kernel):
    overrides = CppOverrides  # type: ignore[assignment]
    sexpr = cexpr
    newvar_prefix = "auto "
    suffix = ";"

    def __init__(self, args, num_threads):
        super().__init__(args)
        self.call_ranges: Optional[Tuple[sympy.Expr, ...]] = None
        self.ranges: List[sympy.Expr] = []
        self.itervars: List[sympy.Symbol] = []
        self.reduction_depth = None
        self.reduction_prefix = IndentedBuffer()
        self.reduction_suffix = IndentedBuffer()
        self.parallel_reduction_prefix = IndentedBuffer()
        self.parallel_reduction_suffix = IndentedBuffer()
        self.local_reduction_init = IndentedBuffer()
        self.local_reduction_stores = IndentedBuffer()
        self.is_reduction = False
        self.non_parallel_reduction_prefix = IndentedBuffer()
        self.reduction_cse = CSE(self.newvar_prefix, self.suffix, name_prefix="tmp_acc")
        self.preloads = IndentedBuffer()
        self.poststores = IndentedBuffer()
        self.num_threads = num_threads  # num_threads the kernel specialized for
        self.reduction_omp_dec: Dict[Tuple[str, str], str] = {}

    def _gen_parallel_reduction_buffers(
        self,
        acc,
        acc_type,
        reduction_type,
        dtype,
        reduction_combine_fn=reduction_combine,
        reduction_init_fn=reduction_init,
        welford_weight_reciprocal_vec_fn=None,
    ):
        if config.cpp.dynamic_threads and not self.parallel_reduction_prefix:
            self.parallel_reduction_prefix.writeline(
                "int max_threads = omp_get_max_threads();"
            )
        acc_local = f"{acc}_local"
        num_threads = (
            "max_threads" if config.cpp.dynamic_threads else parallel_num_threads()
        )
        acc_per_thread = f"{acc}_arr[{num_threads}]"
        acc_local_in_array = acc_per_thread.replace(f"[{num_threads}]", "[tid]")
        self.local_reduction_init.writeline(
            f"{acc_type} {acc_local} = {reduction_init_fn(reduction_type, dtype)};"
        )
        self.parallel_reduction_prefix.writeline(f"{acc_type} {acc_per_thread};")
        self.parallel_reduction_prefix.writelines(
            [
                f"for (int tid = 0; tid < {num_threads}; tid++)",
                "{",
                f"    {acc_local_in_array} = {reduction_init_fn(reduction_type, dtype)};",
                "}",
            ],
        )
        self.local_reduction_stores.writelines(
            [
                f"{acc_local_in_array} = {acc_local};",
            ]
        )
        self.parallel_reduction_suffix.writelines(
            [
                f"for (int tid = 0; tid < {num_threads}; tid++)",
                "{",
                f"    {acc} = {reduction_combine_fn(reduction_type, acc, acc_local_in_array)};",
                "}",
            ],
        )
        if (
            reduction_type == "welford_reduce"
            and welford_weight_reciprocal_vec_fn
            and hasattr(self, "weight_recp_vec_range")
            and "vec" in f"{acc_type}"
        ):
            self.local_reduction_init.writeline(
                welford_weight_reciprocal_vec_fn(dtype, num_threads)
            )

    def get_reduction_var_pattern(self, line: str):
        return re.search("tmp_acc[0-9]+", line)

    def update_stores_with_parallel_reduction(self):
        for i, line in enumerate(self.stores._lines):
            if isinstance(line, str):
                m = self.get_reduction_var_pattern(line)
                if m:
                    var_name = m.group(0)
                    self.stores._lines[i] = line.replace(var_name, f"{var_name}_local")

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

    def cache_high_prec_cse_var_before_lowp_store(self, var_to_store):
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

        if var_to_store.dtype not in DTYPE_LOWP_FP:
            # only need to cache fp32 cse var while var_to_store is lowp data
            return

        def find_high_prec_var(var, cache):
            high_prec_cse_var = None
            high_prec_cse_var_name = None
            for expr, cse_var in cache.items():
                if cse_var == var:
                    if is_to_lowp_dtype(expr):
                        m = re.search(r"tmp\d+", expr)
                        if m is not None:
                            high_prec_cse_var_name = m.group()
            if high_prec_cse_var_name:
                for cse_var in cache.values():
                    if cse_var.name == high_prec_cse_var_name:
                        high_prec_cse_var = cse_var
                        break
                assert high_prec_cse_var is not None
            return high_prec_cse_var

        high_prec_var = find_high_prec_var(var_to_store, self.cse.cache)
        if high_prec_var and high_prec_var.dtype in DTYPE_TO_CPP:
            cache_key = get_lowp_to_high_prec_expr(
                var_to_store, high_prec_var.dtype, self
            )
            self.cse.cache[cache_key] = high_prec_var

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
            csevar = ops.index_expr(expr, torch.int32).value
            buffer = V.kernel.compute
        else:
            # indexing in loads
            prior_compute = V.kernel.compute
            try:
                V.kernel.compute = self.loads
                csevar = ops.index_expr(expr, torch.int32).value
            finally:
                V.kernel.compute = prior_compute
            buffer = self.loads

        size_str = V.kernel.sexpr(self.rename_indexing(size)) if upper else None

        line = self.indirect_assert(csevar, "0" if lower else None, size_str)
        self.cse.generate(buffer, line, assignment=False)

    def load(self, name: str, index: sympy.Expr):
        var = self.args.input(name)
        index = self.rename_indexing(index)
        line = f"{var}[{cexpr_index(index)}]"
        if V.graph.get_dtype(name) in [torch.float16]:
            line = f"static_cast<float>({line})"
        csevar = self.cse.generate(self.loads, line)
        csevar.update_on_args("load", (name, index), {})
        return csevar

    def store(self, name, index, value, mode=None):
        assert "buf" in name
        var = self.args.output(name)
        self.cache_high_prec_cse_var_before_lowp_store(value)
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

    def reduction(self, dtype, src_dtype, reduction_type, value):
        argmax_or_argmin = reduction_type in {"argmax", "argmin"}

        reduction_key = src_dtype, reduction_type, value
        if reduction_key in self.reduction_cse.reduction_cache:
            return self.reduction_cse.reduction_cache[reduction_key]

        acc = self.reduction_cse.generate(
            self.loads, f"reduction {reduction_key}", write=False
        )
        self.is_reduction = True
        if argmax_or_argmin:
            prefix, parallel_prefix, local_init = argmax_argmin_prefix(
                reduction_type, src_dtype, acc
            )
            self.local_reduction_init.writelines(local_init)
            self.reduction_prefix.writelines(prefix)
            self.parallel_reduction_prefix.writelines(parallel_prefix)
            compare_op = (
                "greater_or_nan" if reduction_type == "argmax" else "less_or_nan"
            )
            assert self.reduction_depth is not None
            index = self.itervars[self.reduction_depth]
            for i in range(self.reduction_depth + 1, len(self.itervars)):
                index = index * self.ranges[i] + self.itervars[i]
            self.stores.writelines(
                [
                    f"if(!({compare_op}({acc}.value, {value}, {acc}.index, {cexpr_index(index)}))) {{",
                    f"    {acc}.index = {cexpr_index(index)}; {acc}.value = {value};",
                    "}",
                ]
            )
            acc_local = f"{acc}_local"
            num_threads = parallel_num_threads()
            acc_per_thread = f"{acc}_arr[{num_threads}]"
            acc_local_in_array = acc_per_thread.replace(f"[{num_threads}]", "[tid]")
            self.parallel_reduction_suffix.writelines(
                [
                    f"for (int tid = 0; tid < {num_threads}; tid++)",
                    "{",
                    f"    if(!({compare_op}({acc}.value, {acc_local_in_array}.value, {acc}.index, {acc_local_in_array}.index))) {{",
                    f"        {acc}.index = {acc_local_in_array}.index; {acc}.value = {acc_local_in_array}.value;",
                    "    }",
                    "}",
                ],
            )
            self.local_reduction_stores.writelines(
                [
                    f"{acc_local_in_array} = {acc_local};",
                ]
            )
        else:
            acc_type = reduction_acc_type(reduction_type, dtype)

            self.reduction_prefix.writeline(
                f"{acc_type} {acc} = {reduction_init(reduction_type, dtype)};"
            )
            self.stores.writeline(
                f"{acc} = {reduction_combine(reduction_type, acc, value)};"
            )
            self._gen_parallel_reduction_buffers(acc, acc_type, reduction_type, dtype)
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
            assert self.call_ranges == tuple(lengths) + tuple(
                reduction_lengths
            ), f"{self.call_ranges} == {tuple(lengths)} + {tuple(reduction_lengths)}"
            assert self.reduction_depth == len(lengths)
        else:
            self.call_ranges = tuple(lengths) + tuple(reduction_lengths)
            self.ranges = [self.rename_indexing(x) for x in self.call_ranges]
            self.itervars = [
                sympy_index_symbol_with_prefix(SymT.XBLOCK, n)
                for n in range(len(self.ranges))
            ]
            self.reduction_depth = len(lengths)
        return (
            self.itervars[: self.reduction_depth],
            self.itervars[self.reduction_depth :],
        )

    def size_hint(self):
        return V.graph.sizevars.size_hint(
            sympy_product(self.call_ranges), fallback=8192
        )

    def codegen_loops_impl(self, loop_nest, code, worksharing):
        threads = parallel_num_threads()
        assert self.call_ranges is not None
        kernels = loop_nest.get_kernels()
        if any(isinstance(kernel, OuterLoopFusedKernel) for kernel in kernels):
            assert len(kernels) == 1
            assert isinstance(kernels[0], OuterLoopFusedKernel)
            par_depth = kernels[0].decide_parallel_depth(
                loop_nest.max_parallel_depth(), threads
            )
        else:
            par_depth = self.decide_parallel_depth(
                loop_nest.max_parallel_depth(), threads
            )

        with contextlib.ExitStack() as stack:
            if par_depth:
                if loop_nest.is_reduction_only():
                    # need to close the worksharing scope to define reduction vars outside it
                    worksharing.close()
                else:
                    worksharing.parallel(threads)
                loop_nest.mark_parallel(par_depth)
            elif threads > 1:
                if worksharing.single():
                    stack.enter_context(code.indent())

            def gen_loop_kernel(loop: LoopLevel):
                def is_parallel_reduction(loop):
                    root = loop.get_root()
                    return root.is_reduction and root.parallel

                kernels = loop.get_kernels()
                assert len(kernels) == 1
                if not isinstance(
                    kernels[0], OuterLoopFusedKernel
                ) and is_parallel_reduction(loop):
                    kernels[0].update_stores_with_parallel_reduction()
                gen_kernel(kernels[0])

            def gen_kernel(kernel):
                if isinstance(kernel, OuterLoopFusedKernel):
                    for loop in kernel.inner:
                        if loop.inner:
                            gen_loops(loop.inner, loop.is_reduction)
                        else:
                            with contextlib.ExitStack() as stack:
                                # If there is any kernel existing at the final outer loop fusion level,
                                # the kernel code should be placed within its respective indent to prevent
                                # the duplication of variable definitions.
                                stack.enter_context(code.indent())
                                gen_loop_kernel(loop)
                else:
                    with contextlib.ExitStack() as stack:
                        assert kernel
                        if hasattr(kernel, "codegen_inner_loops"):
                            code.splice(kernel.preloads)
                            kernel.codegen_inner_loops(code)
                            stack.enter_context(code.indent())
                        code.splice(kernel.loads)
                        code.splice(kernel.compute)
                        code.splice(kernel.stores)
                    if hasattr(kernel, "codegen_inner_loops"):
                        code.splice(kernel.poststores)

            def get_reduction_code_buffer(loops, buffer="prefix"):
                assert buffer in ("prefix", "suffix", "local")
                for loop in loops:
                    for kernel in loop.get_kernels():
                        if buffer == "local":
                            return (
                                kernel.local_reduction_init,
                                kernel.local_reduction_stores,
                            )
                        elif buffer == "suffix":
                            suffix = kernel.reduction_suffix
                            if loop.parallel:
                                suffix = kernel.parallel_reduction_suffix + suffix
                            return suffix
                        else:
                            prefix = kernel.reduction_prefix
                            if loop.parallel:
                                prefix = prefix + kernel.parallel_reduction_prefix
                            else:
                                prefix = prefix + kernel.non_parallel_reduction_prefix
                            return prefix

            def gen_loops(loops: List[LoopLevel], in_reduction=False):
                with contextlib.ExitStack() as stack_outer:
                    local_reduction_init = local_reduction_stores = None
                    if loops:
                        loop = loops[0]
                        if loop.is_reduction and not in_reduction:
                            reduction_prefix = get_reduction_code_buffer(loops)
                            if reduction_prefix:
                                stack_outer.enter_context(code.indent())
                            code.splice(reduction_prefix)
                        if loop_nest.is_reduction_only() and loop.parallel:
                            (
                                local_reduction_init,
                                local_reduction_stores,
                            ) = get_reduction_code_buffer(loops, "local")
                            worksharing.parallel(threads)
                            if local_reduction_init:
                                assert local_reduction_stores
                                code.splice(local_reduction_init)

                    for loop in loops:
                        gen_loop(loop)

                    if loops:
                        loop = loops[0]
                        if loop_nest.is_reduction_only() and loop.parallel:
                            if local_reduction_stores:
                                code.splice(local_reduction_stores)
                            worksharing.close()
                        if loop.is_reduction and not in_reduction:
                            code.splice(get_reduction_code_buffer(loops, "suffix"))

            def gen_loop(loop: LoopLevel):
                with contextlib.ExitStack() as stack:
                    loop_lines = loop.lines()
                    if loop_lines is None:
                        return
                    code.writelines(loop_lines)
                    stack.enter_context(code.indent())
                    # generate inner loops or loop body
                    if loop.inner:
                        gen_loops(loop.inner, loop.is_reduction)
                    else:
                        gen_loop_kernel(loop)

            stack.enter_context(code.indent())
            if loop_nest.root:
                gen_loops(loop_nest.root)
            else:
                gen_kernel(loop_nest.kernel)

    def codegen_loops(self, code, worksharing):
        loop_nest = LoopNestWithSplit.build(self)
        self.codegen_loops_impl(loop_nest, code, worksharing)

    @property
    def assert_function(self) -> str:
        if V.graph.aot_mode:
            # TODO: Using AOTI_TORCH_CHECK is causing performance drop for some models
            # compared with JIT Inductor which uses TORCH_CHECK
            return "AOTI_TORCH_CHECK"
        else:
            return "TORCH_CHECK"

    def decide_parallel_depth(self, max_parallel_depth, threads):
        assert self.call_ranges is not None
        ranges = self.call_ranges[:max_parallel_depth]
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
        return depth

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


class CppVecKernel(CppKernel):
    overrides = CppVecOverrides  # type: ignore[assignment]

    def __init__(
        self,
        args,
        num_threads,
        tiling_factor=0,
        tiling_idx=-1,
        tiling_dtype=torch.float,
    ):
        super().__init__(args, num_threads)
        self.vec_isa = codecache.pick_vec_isa()
        assert self.vec_isa
        if tiling_factor == 0:
            tiling_factor = self.vec_isa.nelements(dtype=tiling_dtype)
        self.tiling_factor = tiling_factor
        self.tiling_idx = tiling_idx

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

    def get_reduction_var_pattern(self, line: str):
        return re.search("tmp_acc[0-9]+_vec", line)

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
        opt_ctx: OptimizationContext = get_current_node_opt_ctx()
        assert opt_ctx is not None
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
            line = f"{self._get_mask_type()}::from({loadbuf})"
        else:
            line = (
                f"{load_mask_str}.template loadu<{cpp_type},{num_vectors}>({loadbuf})"
                if load_mask_str
                else f"{self._get_vec_type(dtype)}::loadu({loadbuf}, {self.tiling_factor})"
            )
        return line

    def _load_or_store_non_contiguous(
        self,
        var: Optional[str],
        index: sympy.Expr,
        dtype: torch.dtype,
        buffer: Optional[IndentedBuffer] = None,
        store_value: Optional[Union[str, CppCSEVariable]] = None,
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
        :return: a CppCSEVariable that represents the loaded vector or None if it is a store.
        """
        assert not store_value or var is not None, "store var must be provided"

        if buffer is None:
            buffer = self.loads

        def get_result_size(dtype: torch.dtype) -> int:
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
                code.writeline(
                    f"__at_align__ std::array<{DTYPE_TO_CPP[vec_dtype]}, {result_size}> tmpbuf;"
                )
                line = f"{vec_var}.store(tmpbuf.data());"
                code.writeline(line)
                code.writeline("return tmpbuf;")
            code.writeline("()")
            csevar = self.cse.generate(buffer, code)
            assert isinstance(csevar, CppCSEVariable)
            return csevar

        opt_ctx: OptimizationContext = get_current_node_opt_ctx()
        assert opt_ctx is not None
        code = BracesBuffer()
        code.writeline("[&]")
        with code.indent():
            result_size = get_result_size(dtype)
            result_declare = (
                f"__at_align__ std::array<{DTYPE_TO_CPP[dtype]}, {result_size}> tmpbuf;"
            )
            code.writeline(result_declare)
            if store_value:
                code.writeline(f"{store_value}.store(tmpbuf.data());")
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
            if codecache.is_gcc():
                code.writeline(f"#pragma GCC unroll {self.tiling_factor}")
            else:
                code.writeline(f"#pragma unroll {self.tiling_factor}")
            code.writeline(
                f"for (long {itervar_inner} = 0; {itervar_inner} < {self.tiling_factor}; {itervar_inner}++)"
            )
            with code.indent(), contextlib.ExitStack() as stack:
                index_c = cexpr_index(index)
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
                    code.writeline(f"{rhs} = tmpbuf[{itervar_inner}];")
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
            csevar = self.cse.generate(buffer, code)
            assert isinstance(csevar, CppCSEVariable)
            csevar.is_vec = True
            return csevar

    def load(self, name: str, index: sympy.Expr):
        opt_ctx: OptimizationContext = get_current_node_opt_ctx()
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
            line = self._get_vec_load_line(var, index, dtype, self._load_mask)
            csevar = self.cse.generate(self.loads, line)  # type: ignore[assignment]
        else:
            csevar = self._load_or_store_non_contiguous(var, index, dtype)  # type: ignore[assignment]
        assert isinstance(csevar, CppCSEVariable)
        csevar.update_on_args("load", (name, index), {})
        csevar.is_vec = True
        return csevar

    def _get_store_line(
        self,
        value: Union[str, CppCSEVariable],
        var: str,
        index: sympy.Expr,
        dtype: torch.dtype,
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
            if dtype == torch.float:
                code.writeline(f"{value}.store({var_expr});")
            else:
                code.writeline(f"{value}.store({var_expr}, {self.tiling_factor});")
        else:
            self._load_or_store_non_contiguous(
                var, index, dtype, buffer=code, store_value=value
            )
        return code

    def store(self, name, index, value, mode=None):
        assert "buf" in name
        assert mode is None
        assert isinstance(value, CppCSEVariable), value
        if not value.is_vec:
            # this happens when we store a scalar into a vectorized buffer like "fill"
            value = self.broadcast(value)
        opt_ctx: OptimizationContext = get_current_node_opt_ctx()
        var = self.args.output(name)
        self.cache_high_prec_cse_var_before_lowp_store(value)
        index = self.rename_indexing(index)
        code = self._get_store_line(value, var, index, V.graph.get_dtype(name))
        self.stores.splice(code.map(lambda x: DeferredLine(name, x)))

    def reduction(self, dtype, src_dtype, reduction_type, value):
        assert reduction_type in {
            "max",
            "min",
            "sum",
            "prod",
            "xor_sum",
            "welford_reduce",
            "welford_combine",
        }
        assert dtype == src_dtype
        assert dtype in [torch.float, torch.int64]
        assert isinstance(value, CppCSEVariable), value

        if not value.is_vec:
            value = self.broadcast(value)

        reduction_key = src_dtype, reduction_type, value
        if reduction_key in self.reduction_cse.reduction_cache:
            return self.reduction_cse.reduction_cache[reduction_key]

        vec_ns = "at::vec"
        vec = f"{vec_ns}::Vectorized<{DTYPE_TO_CPP[dtype]}>"
        acc_type = reduction_acc_type(reduction_type, dtype)
        acc_type_vec = self.reduction_acc_type_vec(reduction_type, dtype)

        acc = self.reduction_cse.generate(
            self.loads, f"reduction {reduction_key}", write=False
        )
        acc_vec = f"{acc}_vec"
        self.is_reduction = True
        self.reduction_prefix.writeline(
            f"{acc_type} {acc} = {reduction_init(reduction_type, dtype)};"
        )
        self.reduction_prefix.writeline(
            f"{acc_type_vec} {acc_vec} = {self.reduction_init_vec(reduction_type, dtype)};"
        )
        # save the reciprocal of weights for welford reduce if using static shape
        reduction_size = functools.reduce(
            lambda x, y: x * y, self.ranges[self.reduction_depth :]
        )
        if reduction_type == "welford_reduce":
            reduction_factor = (
                self.tiling_factor if self.tiling_idx >= self.reduction_depth else 1
            )
            self.weight_recp_vec_range = FloorDiv(reduction_size, reduction_factor)
            self.non_parallel_reduction_prefix.writeline(
                self.welford_weight_reciprocal_vec(dtype, None)
            )
            self.stores.writeline(
                f"{acc_vec} = {self.reduction_combine_vec(reduction_type, acc_vec, value, True)};"
            )
        else:
            self.stores.writeline(
                f"{acc_vec} = {self.reduction_combine_vec(reduction_type, acc_vec, value)};"
            )
        self._gen_parallel_reduction_buffers(
            acc,
            acc_type,
            reduction_type,
            dtype,
        )
        self._gen_parallel_reduction_buffers(
            acc_vec,
            acc_type_vec,
            reduction_type,
            dtype,
            reduction_combine_fn=self.reduction_combine_vec,
            reduction_init_fn=self.reduction_init_vec,
            welford_weight_reciprocal_vec_fn=self.welford_weight_reciprocal_vec,
        )
        tmpvar: Union[str, CSEVariable]
        if self.tiling_idx >= self.reduction_depth:
            # Horizontal reduction
            if is_welford_reduction(reduction_type):
                assert (
                    self._get_num_vectors(dtype) == 1
                ), "Welford reduction does not support VectorizedN (N>1)"
                next_value = f"welford_vec_reduce_all({acc_vec})"
            else:
                reduce_all_body = (
                    "{ return "
                    + self.reduction_combine_vec(reduction_type, "x", "y")
                    + "; }"
                )
                vec = f"at::vec::Vectorized<{DTYPE_TO_CPP[dtype]}>"
                vec_reduce_all_func = f"at::vec::vec_reduce_all<{DTYPE_TO_CPP[dtype]}>"
                next_value = f"{vec_reduce_all_func}([]({vec}& x, {vec}& y) {reduce_all_body}, {acc_vec})"

            self.reduction_suffix.writeline(
                f"{acc} = {reduction_combine(reduction_type, acc, next_value)};"
            )
            tmpvar = acc
        else:
            tmpvar = acc_vec

        result = reduction_project(reduction_type, tmpvar)
        self.reduction_cse.reduction_cache[reduction_key] = result
        return result

    def store_reduction(self, name, index, value):
        index = self.rename_indexing(index)
        var = self.args.output(name)
        out_dtype = V.graph.get_dtype(name)
        dtype = torch.float if out_dtype.is_floating_point else torch.int64
        code = IndentedBuffer()
        if self.tiling_idx >= self.reduction_depth:
            # Horizontal reduction
            code.writeline(
                f"{var}[{cexpr_index(index)}] = static_cast<{DTYPE_TO_CPP[out_dtype]}>({value});"
            )
        else:
            # Vertical reduction
            if out_dtype != dtype:
                converted_value = f"{DTYPE_TO_CPP[out_dtype]}_{value}"
                code.writeline(
                    f"auto {converted_value} = at::vec::convert<{DTYPE_TO_CPP[out_dtype]}>({value});"
                )
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

        scalar_init = reduction_init(reduction_type, dtype)
        return f"{vec_type}({scalar_init})"

    def reduction_acc_type_vec(self, reduction_type, dtype):
        assert reduction_type not in {"argmin", "argmax"}
        scalar_type = DTYPE_TO_COMPUTATION_DTYPE[dtype]
        vec_type = self._get_vec_type(scalar_type)
        if is_welford_reduction(reduction_type):
            return f"Welford<{vec_type}>"

        return vec_type

    def welford_weight_reciprocal_vec(self, dtype, num_threads=None):
        vec_num_range_thread = (
            CeilDiv(self.weight_recp_vec_range, num_threads)
            if num_threads
            else self.weight_recp_vec_range
        )
        vec_num_range_thread_expr = cexpr_index(vec_num_range_thread)
        return f"static WeightRecp<{self._get_vec_type(dtype)}> weight_recps({vec_num_range_thread_expr});"

    def reduction_combine_vec(
        self, reduction_type, var, next_value, use_weight_recps=False
    ):
        if reduction_type == "max":
            return f"at::vec::maximum({var}, {next_value})"
        elif reduction_type == "min":
            return f"at::vec::minimum({var}, {next_value})"
        elif reduction_type == "sum":
            return f"{var} + {next_value}"
        elif reduction_type == "prod":
            return f"{var} * {next_value}"
        elif reduction_type == "xor_sum":
            return f"{var} ^ {next_value}"
        elif reduction_type == "welford_reduce":
            if use_weight_recps:
                return f"welford_combine({var}, {next_value}, &weight_recps)"
            else:
                return f"welford_combine({var}, {next_value})"
        elif reduction_type == "welford_combine":
            if isinstance(next_value, tuple):
                # When reading a value from Inductor IR we have a tuple of variable names
                mean, m2, weight = next_value
            else:
                # When combining intermediate accumulators we have a Welford<T> struct
                mean, m2, weight = reduction_project(reduction_type, next_value)
            return f"welford_combine({var}, {{{mean}, {m2}, {weight}}})"
        else:
            raise NotImplementedError

    def indirect_assert(self, var, lower, upper, mask=None):
        assert not mask, "do not support mask in indirect_indexing assertion"
        assert isinstance(var, CppCSEVariable)
        assert var.dtype is not None
        if not var.is_vec:
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
        cond = f"({self._get_mask_type(var.dtype)}({cond})).all_masked()"
        return f'{self.assert_function}({cond}, "index out of bounds: {cond_print}")'


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

    def __init__(self, args, num_threads, tiling_factor, tiling_indices, tiling_dtype):
        super().__init__(
            args, num_threads, tiling_factor, tiling_indices[1], tiling_dtype
        )
        self.tiling_indices = tiling_indices

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

    def gen_transposed_tile_load_store(self, name, var, index, is_store):
        # transposed tile load/store outside the kernel inner loop
        dtype = V.graph.get_dtype(name)
        factor = self.tiling_factor
        src = f"{var} + {cexpr_index(index)}"
        dst = "__place_holder__"
        ld_src = f"{cexpr_index(stride_at_vec_range(index, self.itervars[self.tiling_idx], self.tiling_factor))}"
        ld_dst = f"{factor}"
        if is_store:
            src, dst = dst, src
            ld_src, ld_dst = ld_dst, ld_src

        need_define = True
        load_or_store = f"at::vec::transpose_mxn<{DTYPE_TO_CPP[dtype]},{factor},{factor}>({src}, {ld_src}, {dst}, {ld_dst});"
        if is_store:
            tile_var = self.cse.newvar()
        elif load_or_store not in self.cse.cache:
            tile_var = self.cse.generate(self.preloads, load_or_store, write=False)
        else:
            need_define = False
            tile_var = self.cse.cache[load_or_store]

        if need_define:
            define_line = f"{DTYPE_TO_CPP[dtype]} {tile_var}[{factor}*{factor}] __attribute__ ((aligned ({factor})));"
            self.preloads.writeline(define_line)

        load_or_store = load_or_store.replace("__place_holder__", str(tile_var))
        if is_store:
            self.poststores.writeline(DeferredLine(name, load_or_store))
        else:
            self.preloads.writeline(load_or_store)

        return tile_var

    def load(self, name: str, index: sympy.Expr):
        opt_ctx: OptimizationContext = get_current_node_opt_ctx()
        var = self.args.input(name)
        index = self.rename_indexing(index)

        inner = self.inner_itervar()
        if self.need_vec_transpose(index):
            tile_var = self.gen_transposed_tile_load_store(
                name, var, index, is_store=False
            )
            # vector load inside the kernel inner loop
            loadbuf = f"{tile_var} + {cexpr_index(inner * self.tiling_factor)}"
            dtype = V.graph.get_dtype(name)
            line = self._get_vec_load_line(loadbuf, 0, dtype)  # type: ignore[arg-type]
            csevar = self.cse.generate(self.loads, line)
            csevar.update_on_args("load", (name, index), {})
            assert isinstance(csevar, CppCSEVariable)
            csevar.is_vec = True
            return csevar
        else:
            new_index = self.transform_indexing(index)
            return super().load(name, new_index)

    def store(self, name, index, value, mode=None):
        assert "buf" in name
        opt_ctx: OptimizationContext = get_current_node_opt_ctx()
        var = self.args.output(name)

        inner = self.inner_itervar()
        index = self.rename_indexing(index)
        assert mode is None
        if self.need_vec_transpose(index):
            tile_var = self.gen_transposed_tile_load_store(
                name, var, index, is_store=True
            )
            # vector store inside the kernel inner loop
            storebuf = f"{tile_var} + {cexpr_index(inner * self.tiling_factor)}"
            if V.graph.get_dtype(name) in DTYPE_LOWP_FP:
                line = f"{value}.store({storebuf}, {self.tiling_factor});"
            elif V.graph.get_dtype(name) in (torch.uint8, torch.int8):
                line = f"{value}.store({storebuf}, {self.tiling_factor});"
            else:
                line = f"{value}.store({storebuf});"
            self.stores.writeline(DeferredLine(name, line))
        else:
            new_index = self.transform_indexing(index)
            super().store(name, new_index, value, mode)

    def codegen_inner_loops(self, code):
        inner = self.inner_itervar()
        code.writeline(
            f"for (long {inner} = 0; {inner} < {self.tiling_factor}; {inner}++)"
        )

    def set_ranges(self, group, reduction_group):
        vars = super().set_ranges(group, reduction_group)
        # do vertical reduction as the tail loop
        self.outer_idx, self.tiling_idx = (
            self.tiling_indices
            if self.tiling_indices[1] < self.reduction_depth
            else reversed(self.tiling_indices)
        )
        return vars

    def transform_indexing(self, index: sympy.Expr) -> sympy.Expr:
        return self.scale_index_with_offset(
            index,
            itervar_idx=self.outer_idx,
            offset=self.inner_itervar(),
        )


class CppVecKernelChecker(CppVecKernel):
    def __init__(self, args, num_threads, tiling_factor, tiling_idx=-1):
        super().__init__(args, num_threads, tiling_factor, tiling_idx)

        # Since this kernel is only for checker but does not generate any
        # code, so we need to decrease the kernel count.
        metrics.generated_kernel_count -= 1

        # Used to record the graph wrapper code as the wrapper_code status could be
        # changed during graph run.
        self._orig_wrapper_code = None

        self.simd_vec = True

        self.fast_vec_list = []
        for k, v in CppVecOverrides.__dict__.items():
            if isinstance(v, staticmethod):
                self.fast_vec_list.append(k)
        self.exit_stack = contextlib.ExitStack()

        # Cache all the load result
        self.supported_dtypes: List[torch.dtype] = [
            torch.float,
            torch.bfloat16,
            torch.float16,
            torch.bool,
            torch.uint8,
            torch.int8,
            torch.int32,
            torch.int64,
        ]

    def disable_vec(self, msg=None):
        if schedule_log.isEnabledFor(logging.DEBUG):
            schedule_log.debug("Disabled vectorization: %s", msg)
        self.simd_vec = False

    def load(self, name: str, index: sympy.Expr):
        with RecordOptimizationContext(__name__) as node_ctx:
            load_dtype = V.graph.get_dtype(name)
            opt_ctx: OptimizationContext = node_ctx.get_opt_ctx()
            assert opt_ctx

            opt_ctx.dtype = load_dtype
            var = self.cse.newvar()

            if len(self.itervars) == 0:
                self.disable_vec("not a loop")
                return var

            if load_dtype not in self.supported_dtypes and (
                index.has(self.itervars[self.tiling_idx])
                or free_symbol_is_type(index, SymT.TMP)
            ):
                self.disable_vec(f"{load_dtype} not supported by load")
                return var

            return var

    def store(self, name, index, value, mode=None):
        with RecordOptimizationContext(__name__) as node_ctx:
            if len(self.itervars) == 0:
                self.disable_vec("not a loop")
                return self.simd_vec

            store_dtype = V.graph.get_dtype(name)

            opt_ctx: OptimizationContext = node_ctx.get_opt_ctx()
            assert opt_ctx
            opt_ctx.dtype = store_dtype

            if store_dtype not in self.supported_dtypes:
                self.disable_vec(f"{store_dtype} not supported by store")
                return self.simd_vec

            assert "buf" in name
            index = self.rename_indexing(index)

            if mode:
                self.disable_vec(f"store mode: {mode}")
                return self.simd_vec

            return self.simd_vec

    def reduction(self, dtype, src_dtype, reduction_type, value):
        if not (
            (dtype == torch.float and src_dtype == torch.float)
            or (dtype == torch.int64 and src_dtype == torch.int64)
            and reduction_type in VECTORIZABLE_RTYPES
        ):
            self.disable_vec(
                f"reduction: dtype {dtype}, src_dtype {src_dtype}, reduction_type {reduction_type}"
            )
        if is_welford_reduction(reduction_type):
            return tuple([self.simd_vec] * 3)
        return self.simd_vec

    def check_bounds(
        self, expr: sympy.Expr, size: sympy.Expr, lower: bool, upper: bool
    ):
        return self.simd_vec

    def store_reduction(self, name, index, value):
        return self.simd_vec

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restore the wrapper_code
        V.graph.wrapper_code = self._orig_wrapper_code  # type: ignore[assignment]
        self.exit_stack.__exit__(exc_type, exc_val, exc_tb)

    def __enter__(self):
        # Record the graph wrapper code. The wrapper_code status could be
        # changed during graph run. Regarding this checker, we also need to
        # run the graph but we don't expect to change any status that would
        # impact the code generation. Hence, we record the graph wrapper code
        # and replace it with a dummy wrapper_code and then restore to the
        # original one as long as the checker is finished.
        self._orig_wrapper_code = V.graph.wrapper_code
        V.graph.wrapper_code = WrapperCodeGen()

        parent_handler = V.MockHandler()

        class VecCheckerProxy:
            @staticmethod
            def __getattr__(name):  # type: ignore[misc]
                def inner(*args, **kwargs):
                    if name not in self.fast_vec_list:
                        self.disable_vec(f"op: {name}")

                    parent_val = getattr(parent_handler, name)(*args, **kwargs)
                    return pytree.tree_map(lambda _: self.simd_vec, parent_val)

                return inner

            @staticmethod
            def load(name: str, index: sympy.Expr):
                return self.load(name, index)

            @staticmethod
            def store(name, index, value, mode=None):
                return self.store(name, index, value, mode=mode)

            @staticmethod
            def reduction(dtype, src_dtype, reduction_type, value):
                return self.reduction(dtype, src_dtype, reduction_type, value)

            @staticmethod
            def store_reduction(name, index, value):
                return self.store_reduction(name, index, value)

            @staticmethod
            def check_bounds(
                expr: sympy.Expr, size: sympy.Expr, lower: bool, upper: bool
            ):
                return self.check_bounds(expr, size, lower, upper)

            @staticmethod
            def constant(val, dtype):
                with RecordOptimizationContext(__name__) as node_ctx:
                    opt_ctx: OptimizationContext = node_ctx.get_opt_ctx()
                    assert opt_ctx
                    # VecKernel override dtype for constant
                    # Vectorization only support int32/fp32 now
                    # So if dtype = int64/fp64, we will cast it to int32/fp32 if possible
                    i32_iinfo = torch.iinfo(torch.int32)
                    if (
                        dtype == torch.int64
                        and val <= i32_iinfo.max
                        and val >= i32_iinfo.min
                        and all(
                            user.target in BIN_CMP_OPS
                            for user in node_ctx.current_node.users
                        )
                    ):
                        opt_ctx.dtype = torch.int32

                    f32_iinfo = torch.finfo(torch.float32)
                    if dtype == torch.double:
                        if (
                            (val <= f32_iinfo.max and val >= f32_iinfo.min)
                            or (val == torch.inf)
                            or (val == -torch.inf)
                        ):
                            opt_ctx.dtype = torch.float32

                    if opt_ctx.dtype not in self.supported_dtypes:
                        self.disable_vec(f"constant dtype: {opt_ctx.dtype}")
                    return val

            @staticmethod
            def index_expr(expr, dtype):
                assert len(self.ranges) == len(self.itervars)

                def can_use_int32():
                    free_symbols = list(expr.free_symbols)
                    sizes = {
                        k: v
                        for k, v in zip(self.itervars, self.ranges)
                        if k in free_symbols
                    }
                    # Trivial case: Range empty
                    if any(v == 0 for v in sizes.values()):
                        return True

                    vars_ranges = {
                        k: ValueRanges(0, v - 1)
                        for k, v in sizes.items()
                        if not isinstance(v, sympy.Expr) or v.is_number
                    }
                    if not vars_ranges or len(vars_ranges) != len(free_symbols):
                        i32_iinfo = torch.iinfo(torch.int32)
                        return (
                            expr.is_number
                            and expr <= i32_iinfo.max
                            and expr >= i32_iinfo.min
                        )
                    expr_ranges = bound_sympy(expr, vars_ranges)
                    if math.isinf(expr_ranges.lower) or math.isinf(expr_ranges.upper):  # type: ignore[arg-type]
                        return False
                    # If something takes the values 0..7, we will compare in the loop
                    # x < 8. As such, for the loop not to overflow in the last iteration, we want
                    # to check that expr_ranges.upper + 1 is representable as well
                    return range_expressable_in_32_bits(
                        ValueRanges(
                            int(expr_ranges.lower), int(expr_ranges.upper) + 1  # type: ignore[arg-type]
                        )
                    )

                with RecordOptimizationContext(__name__) as node_ctx:
                    assert len(self.ranges) == len(self.itervars)
                    opt_ctx: OptimizationContext = node_ctx.get_opt_ctx()
                    assert opt_ctx
                    if (
                        dtype == torch.int64
                        and can_use_int32()
                        and all(
                            user.target in BIN_CMP_OPS
                            for user in node_ctx.current_node.users
                        )
                    ):
                        opt_ctx.dtype = torch.int32
                    else:
                        self.disable_vec(f"index_expr: {expr}, dtype {dtype}")

                    tmp_var = self.cse.newvar()
                    return tmp_var

            @staticmethod
            def indirect_indexing(index_var, size, check=True):
                return sympy_index_symbol(str(index_var))

            @staticmethod
            def masked(mask, body, other):
                body()
                return self.cse.newvar()

            @staticmethod
            def to_dtype(x, dtype, src_dtype=None):
                if dtype not in self.supported_dtypes:
                    self.disable_vec(f"to_dtype: {dtype}")
                return x

        self.exit_stack.enter_context(V.set_ops_handler(VecCheckerProxy()))
        self.exit_stack.enter_context(V.set_kernel_handler(self))
        return self


class CppKernelProxy(CppKernel):
    def __init__(self, kernel_group):
        super().__init__(kernel_group.args, kernel_group.ws.num_threads)
        self.kernel_group = kernel_group
        self.loop_nest = None
        self.call_ranges = None
        self.picked_vec_isa: codecache.VecISA = codecache.pick_vec_isa()

    def data_type_propagation(self, nodes):
        for _node in nodes:
            assert isinstance(_node, SchedulerNode)
            DataTypePropagation.propagate_scheduler_node(_node)

    # Check if all the nodes of a given fx graph can support BF16/FP16
    def is_lowp_fp_scheduler(self, scheduler_node: SchedulerNode):
        if not isinstance(scheduler_node._body, ir.LoopBody):
            return True

        _lowp_fp_type: Optional[torch.dtype] = None

        # Propagate the dtype to check if all the fx node is bf16/fp16
        DataTypePropagation.propagate_scheduler_node(scheduler_node)

        sub_blocks = [scheduler_node._body.root_block] + list(
            scheduler_node._body.subblocks.values()
        )
        for sub_block in sub_blocks:
            for _node in sub_block.graph.nodes:
                # TODO(Eikan): Regarding get_index and index_expr, we should conclude the
                # the data type as well.
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
                    return False

                if hasattr(_node, "meta") and _node.meta:
                    assert OptimizationContext.key in _node.meta
                    opt_ctx: OptimizationContext = _node.meta[OptimizationContext.key]
                    if not opt_ctx.dtype or opt_ctx.dtype not in DTYPE_LOWP_FP:
                        return False
                    if _lowp_fp_type:
                        assert (
                            _lowp_fp_type == opt_ctx.dtype
                        ), "scheduler node do not support bf16/fp16 mix"
                    else:
                        _lowp_fp_type = opt_ctx.dtype
                else:
                    return False

        scheduler_node._lowp_fp_type = _lowp_fp_type  # type: ignore[attr-defined]
        return True

    def legalize_lowp_fp_dtype_loopbody(self, loop_body: ir.LoopBody):
        def add_to_dtype(sub_graph: torch.fx.Graph):
            def is_lowp_fp_load(node: torch.fx.Node):
                if node.target not in ["load"]:
                    return False
                assert len(node.args) == 3
                load_dtype = V.graph.get_dtype(node.args[1])  # type: ignore[arg-type]
                return load_dtype in DTYPE_LOWP_FP

            def is_lowp_fp_store(node: torch.fx.Node):
                if node.target != "store":
                    return False
                _, store_var, _, _, _ = node.args
                store_dtype = V.graph.get_dtype(store_var)  # type: ignore[arg-type]
                return store_dtype in DTYPE_LOWP_FP

            sub_graph_nodes = list(sub_graph.nodes)
            to_lowp_fp_legalized_nodes = []
            for _node in sub_graph_nodes:
                if is_lowp_fp_load(_node):
                    # No need to promote to float if all users are direct stores
                    if all(user.target == "store" for user in _node.users):
                        continue
                    ops = _node.args[0]
                    with sub_graph.inserting_after(_node):
                        to_type_node = sub_graph.call_method(
                            "to_dtype", args=(ops, _node, torch.float)
                        )
                        to_type_node_args = to_type_node.args
                        _node.replace_all_uses_with(to_type_node)
                        to_type_node.args = to_type_node_args
                        metrics.cpp_to_dtype_count += 1
                elif is_lowp_fp_store(_node):
                    ops, name, _, value_var, _ = _node.args
                    # No need to promote to float if it is a user of a load which are all directly stored
                    if value_var.target == "load" and all(
                        user.target == "store" for user in value_var.users
                    ):
                        continue
                    dtype = V.graph.get_dtype(name)
                    with sub_graph.inserting_before(_node):
                        to_type_node = sub_graph.call_method(
                            "to_dtype", args=(ops, value_var, dtype)
                        )
                        _node.replace_input_with(value_var, to_type_node)
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
                elif _node.target == "to_dtype" and _node.args[-1] in DTYPE_LOWP_FP:
                    (ops, x, _) = _node.args
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
                else:
                    pass

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
            assert isinstance(_node._body, ir.LoopBody)
            node: SchedulerNode = _node

            def is_memory_copy_scheduler_node(node: SchedulerNode):
                op_counts = node.read_writes.op_counts
                return (
                    len(op_counts) == 2 and "load" in op_counts and "store" in op_counts
                )

            should_legalize = not is_memory_copy_scheduler_node(node)
            if should_legalize:
                body: ir.LoopBody = node._body
                self.legalize_lowp_fp_dtype_loopbody(body)

    def codegen_functions(self, fn_list, var_sizes_list, vec_dtype=torch.float):
        # TODO(jgong5): remove vec_dtype arg with alternative tiling factors for various dtypes
        assert len(fn_list) == len(var_sizes_list)
        kernel_group = self.kernel_group
        group, reduction_group = max(var_sizes_list, key=lambda sizes: len(sizes[1]))

        self.set_ranges(group, reduction_group)

        def codegen_kernel(cls, *args):
            with kernel_group.new_kernel(cls, *args) as kernel:
                # Ugly hack to maintain the metrics kernel count since
                # we only count in CppKernelProxy, not those contained in it
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

        scalar_kernel = codegen_kernel(CppKernel)
        V.graph.removed_buffers |= scalar_kernel.removed_buffers
        V.graph.inplaced_to_remove |= scalar_kernel.inplaced_to_remove
        self.loop_nest = LoopNestWithSplit.build(scalar_kernel)

        if not self.picked_vec_isa:
            return

        def select_tiling_indices(tiling_factor):
            all_index = []
            for fn, var_sizes in zip(fn_list, var_sizes_list):
                rw = dependencies.extract_read_writes(fn, *var_sizes)
                all_index += [dep.index for dep in itertools.chain(rw.reads, rw.writes)]
            contig_vars = set()
            contig_vars_list = []
            non_contig_stride_const = set()
            non_contig_stride_other = set()
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
            contig_only = (
                contig_vars - non_contig_stride_const - non_contig_stride_other
            )
            if len(contig_vars) == 0:
                # no contiguous vars
                return [len(self.itervars) - 1]
            if contig_only:
                return sorted(contig_only)[-1:]
            contig_and_const_stride = (
                contig_vars & non_contig_stride_const
            ) - non_contig_stride_other
            contig_vars_sorted = sorted(contig_vars)
            if (
                len(contig_vars_sorted) == 2
                and contig_vars_sorted[-1] in contig_and_const_stride
                and contig_vars_sorted[-1] == len(self.itervars) - 1
            ):
                return contig_vars_sorted
            return sorted(contig_vars_sorted, key=contig_vars_list.count)[-1:]

        def select_tiling(dtype: torch.dtype = torch.float):
            # TODO(jgong5): support alternative tiling factors and data types
            tiling_factor = self.picked_vec_isa.nelements(dtype=dtype)
            tiling_indices = select_tiling_indices(tiling_factor)
            if tiling_indices:
                could_vec = True
                for tiling_indice in tiling_indices:
                    with CppVecKernelChecker(
                        deepcopy(self.kernel_group.args),
                        parallel_num_threads(),
                        tiling_factor,
                        tiling_indice,
                    ) as vec_checker:
                        run(vec_checker)
                        could_vec = could_vec and vec_checker.simd_vec
                        if not could_vec:
                            break
                if could_vec:
                    if len(tiling_indices) == 1:
                        return [tiling_factor], tiling_indices
                    if len(tiling_indices) == 2:
                        return [tiling_factor, tiling_factor], tiling_indices
            return [], []

        # Kernels share the same global contexts like V.graph.wrapper_code, V.kernel.args.
        # But the generated scalar kernel has updated these global contexts. Hence, the other kernels
        # should not do this again to avoid context conflict. By now, we only control the
        # config.inplace_buffers. In the future, we could maintain more contexts.
        with torch._inductor.config.patch(inplace_buffers=False):
            tiling_factors, tiling_indices = select_tiling(vec_dtype)
            assert len(tiling_factors) == len(tiling_indices)
            if len(tiling_indices) == 1:
                vec_kernel = codegen_kernel(
                    CppVecKernel, tiling_factors[0], tiling_indices[0], vec_dtype
                )
                metrics.generated_cpp_vec_kernel_count += 1
                main_loop, tail_loop = self.loop_nest.split_with_tiling(
                    tiling_indices[0], factor=tiling_factors[0]
                )
                main_loop.set_kernel(vec_kernel)
                tail_loop.set_kernel(scalar_kernel)
                main_loop.simd_vec = True
                tail_loop.simd_omp = True
                # We chop the loop into two cubes by the nelements - main loop and tail loop.
                # Regarding the main loop, it is straightforward that it could be vectorized with
                # nelements. But for the tail loop, it still could be vectorized. For example,
                # if the nelements is 8(256bits), then the tail loop still could be vectorized
                # as 4(128bits).
                tail_loop.simd_nelements = tiling_factors[0] // 2
            elif len(tiling_indices) == 2:
                assert (
                    tiling_indices[1] == len(self.itervars) - 1
                    and tiling_factors[0] == tiling_factors[1]
                )
                tile2d_kernel = codegen_kernel(
                    CppTile2DKernel, tiling_factors[0], tiling_indices, vec_dtype
                )
                vec_kernel = codegen_kernel(
                    CppVecKernel, tiling_factors[0], tiling_indices[0], vec_dtype
                )
                metrics.generated_cpp_vec_kernel_count += 2
                outer_main_loop, outer_tail_loop = self.loop_nest.split_with_tiling(
                    tiling_indices[0], factor=tiling_factors[0]
                )
                outer_tail_loop.set_kernel(scalar_kernel)
                (
                    inner_main_loop,
                    inner_tail_loop,
                ) = outer_main_loop.split_with_tiling(
                    tiling_indices[1] - tiling_indices[0], factor=tiling_factors[0]
                )
                inner_main_loop.set_kernel(tile2d_kernel)
                inner_tail_loop.set_kernel(vec_kernel)

    def codegen_loop_bodies(self, loop_bodies, var_sizes_list):
        for body in loop_bodies:
            self.legalize_lowp_fp_dtype_loopbody(body)
            DataTypePropagation.propagate_loopbody(body)
        self.codegen_functions(loop_bodies, var_sizes_list)

    def codegen_nodes(self, nodes: List[SchedulerNode]):
        # Legalize BF16 node by adding to_dtype explicitly
        self.legalize_lowp_fp_dtype(nodes)
        self.data_type_propagation(nodes)

        assert len(nodes) >= 1
        first_node = nodes[0]
        vec_dtype = (
            first_node._lowp_fp_type  # type: ignore[attr-defined]
            if all(
                hasattr(_node, "_lowp_fp_type")
                and _node._lowp_fp_type == first_node._lowp_fp_type  # type: ignore[attr-defined]
                for _node in nodes
            )
            else torch.float
        )

        def fn(node, *index_vars):
            node.decide_inplace_update()
            node.mark_run()
            if isinstance(V.kernel, NullKernelHandler):
                return node._body(*index_vars)
            else:
                return node.codegen(index_vars)

        fn_list = [functools.partial(fn, node) for node in nodes]
        var_sizes_list = [node.group[1] for node in nodes]
        self.codegen_functions(fn_list, var_sizes_list, vec_dtype)

    def codegen_loops(self, code, worksharing):
        self.codegen_loops_impl(self.loop_nest, code, worksharing)


class OuterLoopFusedKernel(CppKernel):
    def __init__(self, kernel_group):
        super().__init__(kernel_group.args, kernel_group.ws.num_threads)
        self.inner: List[LoopLevel] = []

    def decide_parallel_depth(self, max_parallel_depth, threads) -> int:
        kernels_parallel_depth = []
        nested_kernels: List[List[CppKernel]] = [
            loop.get_kernels() for loop in self.inner
        ]
        for kernels in nested_kernels:
            # For any ScalarKernel, VecKernel, or Tile2DKernel,
            # they should all have the same call_ranges
            call_ranges = kernels[0].call_ranges
            assert call_ranges is not None
            assert all(kernel.call_ranges == call_ranges for kernel in kernels)
            kernels_parallel_depth.append(
                kernels[0].decide_parallel_depth(len(call_ranges), threads)
            )
        return min(
            max_parallel_depth,
            max(kernels_parallel_depth),
        )


class ReasonFusedNodes(Enum):
    SAME_VARS_REDUCE = "same_vars_reduce"
    COMPATIBLE_REDUCTION = "compatible_reduction"
    COMPATIBLE_RANGES_NO_REDUCTION = "compatible_ranges_no_reduction"


class CppScheduling(BaseScheduling):
    # ctypes limits the number of args to 1024, refer to:
    # https://github.com/python/cpython/commit/a285af7e626d1b81cf09f8b2bf7656f100bc1237
    # We set a conservative threshold here.
    MAX_FUSED_KERNEL_ARGS_NUM = 500
    backend_features = dict.fromkeys(
        [
            BackendFeature.INPLACE_BUFFERS,
        ]
    )

    @classmethod
    def get_backend_features(cls, device: torch.device):
        return cls.backend_features

    def __init__(self, scheduler):
        super().__init__()
        self.scheduler = scheduler
        if scheduler:
            self.reset_kernel_group()
        self._ready_to_flush = False

    def _set_flush_status(self, status: bool):
        self._ready_to_flush = status

    def group_fn(self, sizes):
        return tuple(tuple(map(V.graph.sizevars.simplify, s)) for s in sizes)

    def reset_kernel_group(self):
        from .cpp_wrapper_cpu import CppWrapperCpu

        self.kernel_group: Union[CppWrapperKernelGroup, KernelGroup]
        if isinstance(V.graph.wrapper_code, CppWrapperCpu):
            self.kernel_group = CppWrapperKernelGroup()
        else:
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
                        indexing_exprs = set()
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

                extra_indexing_constraints = get_indexing_ranges_exprs(ref_node)

                node_to_recomp.recompute_size_and_body(
                    extra_indexing_constraints=extra_indexing_constraints
                )

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
            ranges_set = set()
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
            ranges1 = ref_node.node.data.get_size()

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
            and node1.get_names() & node2.ancestors
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
            return not node2.is_reduction()
        return (
            self._can_fuse_horizontal_impl(node1, node2) and not node1.is_reduction()
        ) or self.can_fuse_vertical_outer_loop(node1, node2)

    def codegen_node(
        self,
        node: Union[OuterLoopFusedSchedulerNode, FusedSchedulerNode, SchedulerNode],
    ):
        """
        Turn an set of pre-fused nodes into a C++ kernel.
        """
        kernel_group = self.kernel_group

        if isinstance(node, OuterLoopFusedSchedulerNode):
            cpp_kernel_proxy_list: List[CppKernelProxy] = []
            nodes_list: List[List[SchedulerNode]] = []

            for _node in node.get_outer_nodes():
                assert isinstance(_node, (FusedSchedulerNode, SchedulerNode))
                _nodes: List[SchedulerNode] = _node.get_nodes()  # type: ignore[assignment]
                cpp_kernel_proxy = CppKernelProxy(kernel_group)
                cpp_kernel_proxy.codegen_nodes(_nodes)

                cpp_kernel_proxy_list.append(cpp_kernel_proxy)
                nodes_list.append(_nodes)

            # Note that, in the future, when every kernel can be vectorized,
            # the function select_tiling will be much easier, and we'll be able to lift
            # check_outer_fusion_loop_level_attr to the fusion phase,
            # avoiding grouping kernels at fusion time that "look like we'll be able to fuse them"
            # but then we actually won't.
            if node.check_outer_fusion_loop_level_attr(
                cpp_kernel_proxy_list, node.outer_loop_fusion_depth
            ):
                # Merge the cpp_kernel_proxy_list into cpp_kernel_proxy
                outer_fusion_cpp_kernel_proxy = node.merge_outer_fusion_kernels(
                    cpp_kernel_proxy_list,
                )
                kernel_group.finalize_kernel(
                    outer_fusion_cpp_kernel_proxy,
                    [_node for _nodes in nodes_list for _node in _nodes],
                )
            else:
                # Fall back to standard loop codegen
                for _kernel_proxy, _nodes in zip(cpp_kernel_proxy_list, nodes_list):
                    kernel_group.finalize_kernel(_kernel_proxy, _nodes)
        else:
            nodes: List[SchedulerNode] = node.get_nodes()  # type: ignore[assignment]
            cpp_kernel_proxy = CppKernelProxy(kernel_group)
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
    ):
        """
        Codegen a CPP template, possibly with fused epilogues
        """
        counters["inductor"]["cpp_epilogue_fusion_counter"] += len(epilogue_nodes)
        assert self.is_cpp_template(
            template_node
        ), "Template node passed to CppScheduler.codegen_template must be a SchedulerNode that wraps a CppTemplateBuffer"
        template_node = cast(SchedulerNode, template_node)
        _, (_, rnumel) = template_node.group
        assert rnumel == ()
        ctb: ir.CppTemplateBuffer = cast(ir.CppTemplateBuffer, template_node.node)
        epilogue_ir_nodes: List[Optional[ir.Buffer]] = [n.node for n in epilogue_nodes]
        assert all(
            isinstance(n, ir.ComputedBuffer) for n in epilogue_ir_nodes
        ), "Epilogue nodes must all be instances of ir.ComputedBuffer"
        kernel, render = ctb.make_kernel_render(ctb, epilogue_nodes=epilogue_ir_nodes)
        with kernel:
            for node in [template_node, *epilogue_nodes]:
                node.mark_run()  # type: ignore[attr-defined]
            src_code = render()

        with V.set_kernel_handler(kernel):
            node_schedule = [template_node, *epilogue_nodes]
            kernel_name = self.define_kernel(src_code, node_schedule, kernel.args)
        kernel.call_kernel(kernel_name, ctb)
        V.graph.removed_buffers |= kernel.removed_buffers
        self.scheduler.free_buffers()

    def _get_scheduled_num_args(self):
        return self.kernel_group.get_num_args()

    def ready_to_flush(self):
        return self._ready_to_flush

    def codegen_sync(self):
        pass

    def define_kernel(self, src_code, nodes, kernel_args=None):
        wrapper = V.graph.wrapper_code
        fused_name = (
            get_fused_kernel_name(nodes, config.cpp.descriptive_names)
            if config.cpp.descriptive_names
            else ""
        )
        kernel_name = "_".join(["cpp", fused_name, wrapper.next_kernel_suffix()])
        kernel_decl_name = kernel_name if V.graph.cpp_wrapper else "kernel"
        src_code = src_code.replace(str(Placeholder.KERNEL_NAME), kernel_decl_name)
        src_code = src_code.replace(str(Placeholder.DESCRIPTIVE_NAME), kernel_name)
        # TODO(voz): Ostensibly, we should not need this. But there are cases where C++ codegen does
        # not use BracesBuffer, so we have no good indicator of a C++ buffer atm.
        src_code = src_code.replace("#pragma CMT", "//")

        compile_wrapper = IndentedBuffer()
        args = self.kernel_group.args if kernel_args is None else kernel_args
        _, _, arg_types = args.cpp_argdefs()
        if not V.graph.cpp_wrapper:
            compile_wrapper.writeline(f"async_compile.cpp_pybinding({arg_types!r}, '''")
        compile_wrapper.splice(src_code, strip=True)
        if not V.graph.cpp_wrapper:
            compile_wrapper.writeline("''')")
        wrapper.define_kernel(kernel_name, compile_wrapper.getvalue(), cuda=False)
        return kernel_name

    def flush(self):
        src_code = self.kernel_group.codegen_group()
        if src_code:
            kernel_name = self.define_kernel(
                src_code, self.kernel_group.scheduled_nodes
            )
            self.kernel_group.call_kernel(V.graph.wrapper_code, kernel_name)
        self.reset_kernel_group()
        self._set_flush_status(False)


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
        arg_defs, call_args, arg_types = self.args.cpp_argdefs()
        args_num = len(arg_defs)
        return args_num

    def codegen_group(self, name=None) -> str:
        self.stack.close()
        if not self.scheduled_nodes:
            return ""
        code = BracesBuffer()
        # 1. Include header files
        # TODO: support kernel profile on other platforms
        enable_kernel_profile = (
            config.cpp.enable_kernel_profile and sys.platform == "linux"
        )
        if enable_kernel_profile:
            code.writelines(["#include <ATen/record_function.h>"])
        code.writeline(codecache.cpp_prefix())

        # 2. Function definition
        kernel_decl_name = str(Placeholder.KERNEL_NAME) if name is None else name
        kernel_name = str(Placeholder.DESCRIPTIVE_NAME) if name is None else name
        arg_defs, _, _ = self.args.cpp_argdefs()
        arg_defs = ",\n".ljust(25).join(arg_defs)
        code.writeline(f'extern "C" void {kernel_decl_name}({arg_defs})')

        # 3. Function body
        with code.indent():
            if enable_kernel_profile:
                graph_id = V.graph.graph_id
                prefix = "graph_" + str(graph_id) + "_" if graph_id is not None else ""
                code.writelines(
                    [
                        f'RECORD_FUNCTION("{prefix + kernel_name}", c10::ArrayRef<c10::IValue>({{}}));'
                    ]
                )
            for old, new in self.args.aliases():
                code.writeline(f"auto {old} = {new};")
            code.splice(self.loops_code)
        return code.getvalue()

    def call_kernel(self, wrapper, kernel_name):
        _, call_args, arg_types = self.args.cpp_argdefs()
        wrapper.generate_kernel_call(
            kernel_name, call_args, cuda=False, arg_types=arg_types
        )


class CppWrapperKernelGroup(KernelGroup):
    def __init__(self):
        super().__init__()
        self.args = CppWrapperKernelArgs()


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
    offset: sympy.Expr = sympy.Integer(0)
    steps: sympy.Expr = sympy.Integer(1)
    parallel: int = 0
    simd_omp: bool = False
    simd_vec: bool = False
    collapsed: bool = False
    is_reduction: bool = False
    parent: Optional["LoopLevel"] = None
    # the next inner level of the loop, empty if it is inner-most
    # contains >1 LoopLevel if the inner level of loop is split
    inner: List["LoopLevel"] = dataclasses.field(default_factory=list)
    # kernel assigned to this loop level, only valid when it is a leaf
    kernel: Optional[CppKernel] = None

    def __post_init__(self):
        # Regarding the C++/OpenMP backend, `codecache.pick_vec_isa()` to check
        # vectorization ISA is a time-consuming and one-shot operation. It leads
        # to taking a longer time to import `codegen.cpp` package because the
        # `LoopLevel` of the package is decorated by `@dataclasses.dataclass` while
        # the decorator will invoke `codecache.pick_vec_isa()` to initialize the
        # `simd_nelements` of the `LoopLevel`. It might introduce additional compilation
        # overhead to the Triton backend. Therefore, we moved the `simd_nelements` to
        # `__post_init__`
        picked_vec_isa: codecache.VecISA = codecache.pick_vec_isa()
        self.simd_nelements: int = picked_vec_isa.nelements() if picked_vec_isa else 0

    def get_kernels(self) -> List[CppKernel]:
        """Get all kernel objects under this loop level"""
        if self.kernel:
            return [self.kernel]
        kernels = []
        for loop in self.inner:
            kernels += loop.get_kernels()
        return kernels

    def get_root(self):
        """Get all kernel objects under this loop level"""
        root = self
        while root.parent:
            root = root.parent
        return root

    def set_kernel(self, kernel: CppKernel):
        """
        Set the kernel under this loop level. No split is allowed under
        this loop level.
        """
        if not self.inner:
            self.kernel = kernel
            loop: Optional[LoopLevel] = self
            assert loop is not None
            return
        assert len(self.inner) == 1
        self.inner[0].set_kernel(kernel)

    def get_loops_at(self, depth) -> List["LoopLevel"]:
        if depth == 0:
            return [self]
        else:
            loops = []
            for loop in self.inner:
                loops += loop.get_loops_at(depth - 1)
            return loops

    def split_with_tiling(self, depth, factor):
        def clone_inner():
            inner = []
            if self.inner:
                for loop in self.inner:
                    inner.append(loop.clone())
            return inner

        def do_split_with_tiling():
            sympy_factor = sympy.Integer(factor)

            offset = FloorDiv(self.size, sympy_factor) * sympy_factor
            main_loop = LoopLevel(self.var, offset)
            main_loop.steps = sympy_factor
            main_loop.parallel = self.parallel
            main_loop.collapsed = False
            main_loop.is_reduction = self.is_reduction
            main_loop.inner = clone_inner()
            if main_loop.inner:
                for loop in main_loop.inner:
                    loop.parent = main_loop

            tail_loop = LoopLevel(self.var, self.size)
            tail_loop.offset = offset
            tail_loop.parallel = self.parallel
            tail_loop.collapsed = False
            tail_loop.is_reduction = self.is_reduction
            tail_loop.inner = clone_inner()
            if tail_loop.inner:
                for loop in tail_loop.inner:
                    loop.parent = tail_loop

            return main_loop, tail_loop

        if depth == 0:
            main_loop, tail_loop = do_split_with_tiling()
            parent = self.parent
            if parent:
                parent.inner = [main_loop, tail_loop]
                main_loop.parent = parent
                tail_loop.parent = parent
            return main_loop, tail_loop
        else:
            assert len(self.inner) == 1
            return self.inner[0].split_with_tiling(depth - 1, factor)

    def clone(self):
        loop = copy(self)
        loop.inner = []
        if self.inner:
            for inner_loop in self.inner:
                inner_loop_clone = inner_loop.clone()
                inner_loop_clone.parent = loop
                loop.inner.append(inner_loop_clone)
        loop.kernel = deepcopy(self.kernel)
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
        elif not self.is_reduction and codecache.is_gcc():
            line1 = "#pragma GCC ivdep"
        else:
            line1 = ""
        offset_str = f"{INDEX_TYPE} {self.var}={offset_expr}"
        size_str = f"{self.var}<{size_expr}"
        steps_str = f"{self.var}+={cexpr_index(self.steps)}"
        line2 = f"for({offset_str}; {size_str}; {steps_str})"
        if self.collapsed or not line1:
            return [line2]
        return [line1, line2]


@dataclasses.dataclass
class LoopNestWithSplit:
    """
    A loop-nest like structure but with some loop level split along
    the loop range into the main tiling loop and the tail. It is built
    with the `build` method as a loop nest and then split with
    `split_with_tiling` at some depth.

    A typical case is for vectorization where we typically split at the inner-most
    loop level. A more complicated case is 2D tiling where we split at
    both inner-most and outer levels.
    """

    root: Optional[List[LoopLevel]] = None
    kernel: Optional[CppKernel] = None

    @staticmethod
    def build(kernel: CppKernel):
        """Build a LoopNest with the given `kernel` as the leaf"""
        itervars = kernel.itervars
        ranges = kernel.ranges
        reduction_depth = kernel.reduction_depth
        assert reduction_depth is not None

        root: List[LoopLevel] = []
        levels: List[LoopLevel] = root
        loop: Optional[LoopLevel] = None
        for loop_idx, (var, size) in enumerate(zip(itervars, ranges)):
            loop = LoopLevel(var, size, parent=loop)
            if loop_idx >= reduction_depth:
                loop.is_reduction = kernel.is_reduction
            levels.append(loop)
            levels = loop.inner
        loop_nest = LoopNestWithSplit(root)
        if loop:
            loop.kernel = kernel
        else:
            loop_nest.kernel = kernel
        return loop_nest

    def __bool__(self):
        return bool(self.root)

    def get_loops_at(self, depth) -> List[LoopLevel]:
        """Get all the loop levels at the given `depth` (most outer loop has depth 0)"""
        loops: List[LoopLevel] = []
        assert self.root is not None
        for loop in self.root:
            loops += loop.get_loops_at(depth)
        return loops

    @cache_on_self
    def max_parallel_depth(self):
        """
        Maximal allowed depth for parallelism:
        1) Levels without splitting and
        2) All reduction or non-reduction levels
        When the loop is split at the top level, the max depth is 1.
        """
        max_depth = 0
        assert self.root is not None
        loops = self.root
        if len(loops) > 1:
            return 1
        is_reduction = loops[0].is_reduction if loops else False
        while len(loops) == 1 and loops[0].is_reduction == is_reduction:
            max_depth += 1
            loops = loops[0].inner
        return max_depth

    def is_reduction_only(self):
        """
        Whether all the loops are for reduction. Reduction loops
        are always the inner most ones.
        """
        return (
            self.root is not None and len(self.root) > 0 and self.root[0].is_reduction
        )

    def mark_parallel(self, par_depth):
        assert (
            par_depth <= self.max_parallel_depth()
        ), "Parallel depth cannot exceed the maximal allowed parallel depth"
        assert self.root is not None
        loops = self.root
        for loop in loops:
            loop.parallel = par_depth
        for i in range(1, par_depth):
            loops = loops[0].inner
            loops[0].collapsed = True

    def split_with_tiling(self, depth, factor):
        """
        Split the loop into main and tail loops at given `depth` so that the range
        of the main loop has range `floor_div(range, factor) * factor` and
        the tail loop handles the remainder. The main loop is tiled
        according to the `factor`.
        """
        loops = self.get_loops_at(depth)
        assert len(loops) == 1
        split_loops = loops[0].split_with_tiling(0, factor)
        if depth == 0:
            self.root = split_loops
        return split_loops

    def get_kernels(self) -> List[CppKernel]:
        """Get all kernel objects under this loop nest"""
        if self.kernel:
            return [self.kernel]
        kernels: List[CppKernel] = []
        assert self.root is not None
        for loop in self.root:
            kernels += loop.get_kernels()
        return kernels
