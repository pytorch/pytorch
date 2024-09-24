# mypy: allow-untyped-defs
import contextlib
import copy
import functools
import math
import sys
from collections import namedtuple
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
from unittest.mock import patch

import sympy

import torch
from torch._prims_common import is_integer_dtype
from torch.utils._sympy.symbol import symbol_is_type, SymT
from torch.utils._sympy.value_ranges import ValueRanges

from .. import ir
from ..loop_body import LoopBody
from ..utils import IndentedBuffer, sympy_index_symbol_with_prefix, sympy_subs
from ..virtualized import ops, OpsValue, V
from .common import (
    CSEVariable,
    deduce_output_dtype_by_name,
    ExprPrinter,
    Kernel,
    KernelArgs,
    OptimizationContext,
)


DTYPE_TO_CPP = {
    torch.float32: "float",
    torch.float64: "double",
    torch.float16: "half",
    torch.int64: "int64_t",
    torch.int32: "int32_t",
    torch.int16: "int16_t",
    torch.int8: "int8_t",
    torch.uint64: "uint64_t",
    torch.uint32: "uint32_t",
    torch.uint16: "uint16_t",
    torch.uint8: "uint8_t",
    torch.bool: "bool",
    torch.bfloat16: "bfloat16",
    torch.complex64: "c10::complex<float>",
    torch.float8_e4m3fn: "float8_e4m3fn",
    torch.float8_e5m2: "float8_e5m2",
}

DTYPE_TO_ATEN = {
    torch.float32: "at::kFloat",
    torch.float64: "at::kDouble",
    torch.float16: "at::kHalf",
    torch.int64: "at::kLong",
    torch.int32: "at::kInt",
    torch.int16: "at::kShort",
    torch.int8: "at::kChar",
    torch.uint64: "at::kUInt64",
    torch.uint32: "at::kUInt32",
    torch.uint16: "at::kUInt16",
    torch.uint8: "at::kByte",
    torch.uint32: "at::kUInt32",
    torch.uint64: "at::kUInt64",
    torch.bool: "at::kBool",
    torch.bfloat16: "at::kBFloat16",
    torch.complex32: "at::kComplexHalf",
    torch.complex64: "at::kComplexFloat",
    torch.complex128: "at::kComplexDouble",
    torch.float8_e4m3fn: "at::kFloat8_e4m3fn",
    torch.float8_e5m2: "at::kFloat8_e5m2",
    torch.float8_e4m3fnuz: "at::kFloat8_e4m3fnuz",
    torch.float8_e5m2fnuz: "at::kFloat8_e5m2fnuz",
}

DEVICE_TO_ATEN = {
    "cpu": "at::kCPU",
    "cuda": "at::kCUDA",
}

LAYOUT_TO_ATEN = {
    torch.strided: "at::kStrided",
    torch._mkldnn: "at::kMkldnn",  # type: ignore[attr-defined]
}

_IS_WINDOWS = sys.platform == "win32"

INDEX_TYPE = "int64_t"

GemmBlocking = namedtuple("GemmBlocking", ["block_m", "block_n", "block_k"])


def get_promote_dtype(args):
    return (
        functools.reduce(
            torch.promote_types,  # type: ignore[arg-type]
            [n.dtype for n in args if isinstance(n, CppCSEVariable)],
        )
        if all(n.dtype is not None for n in args if isinstance(n, CppCSEVariable))
        else None  # not enough info to calculate the promote dtype
    )


def promote_args(new_args):
    def promote_arg(arg, promote_type):
        if (
            isinstance(arg, CppCSEVariable)
            and arg.dtype
            and promote_type
            and arg.dtype != promote_type
        ):
            arg = ops.to_dtype(arg, promote_type)
            arg = arg.value if isinstance(arg, OpsValue) else arg
            arg.dtype = promote_type
        return arg

    promote_type = get_promote_dtype(new_args)
    promote_fn = functools.partial(
        promote_arg,
        promote_type=promote_type,
    )
    if (
        all(
            new_arg.dtype is not None
            for new_arg in new_args
            if isinstance(new_arg, CppCSEVariable)
        )
        and promote_type
    ):
        new_args = list(map(promote_fn, new_args))
    return new_args


def get_opt_ctx(node: torch.fx.Node) -> OptimizationContext:
    return node.meta.get(OptimizationContext.key, None)


def get_current_node_opt_ctx() -> OptimizationContext:
    assert V.interpreter.current_node
    return get_opt_ctx(V.interpreter.current_node)


def deduce_dtype_for_cpp_cse_variable(name, *args, **kwargs):
    if (
        output_dtype := deduce_output_dtype_by_name(
            name,
            *args,
            **kwargs,
        )
    ) is not None:
        return output_dtype
    elif name == "masked":
        # <TODO> Leslie: perhaps we can also deduce the masked dtype by
        # inputs' CppCseVariable like other. Let's check it if any
        # unexpected failures.
        assert (
            hasattr(V.interpreter, "current_node")
            and V.interpreter.current_node.target.startswith("masked_subblock")
            and get_current_node_opt_ctx() is not None
        )
        return get_current_node_opt_ctx().dtype
    else:
        # deduce output dtype by inputs' dtype
        assert all(
            arg.dtype is not None for arg in args if isinstance(arg, CppCSEVariable)
        )
        return functools.reduce(
            torch.promote_types,  # type: ignore[arg-type]
            [arg.dtype for arg in args if isinstance(arg, CppCSEVariable)],
        )


class CppCSEVariable(CSEVariable):
    def __init__(self, name, bounds: ValueRanges[Any]) -> None:
        super().__init__(name, bounds)
        self.is_vec = False
        self.dtype: Optional[torch.dtype] = None
        self.dependent_itervars: Set[sympy.Symbol] = set()

    def __repr__(self) -> str:
        return (
            f"CppCSEVariable(name: {self.name}, bounds: {self.bounds}, is_vec: {self.is_vec}, dtype: {self.dtype}, "
            f"dependent_itervars: {self.dependent_itervars})"
        )

    def update_on_args(self, name, args, kwargs):
        if name == "load":
            # args[2] is index
            self._set_dependent_itervars(args[2])
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
        # NOTE [Deduce dtype of CppCSEVariable at runtime]
        self.dtype = deduce_dtype_for_cpp_cse_variable(name, *args, **kwargs)
        assert self.dtype is not None

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


class CppPrinter(ExprPrinter):
    def _print_Integer(self, expr):
        return (
            f"{int(expr)}LL" if sys.platform in ["darwin", "win32"] else f"{int(expr)}L"
        )

    def _print_Where(self, expr):
        c = self.paren(self.doprint(expr.args[0]))
        p = self.paren(self.doprint(expr.args[1]))
        q = self.paren(self.doprint(expr.args[2]))
        return f"{c} ? {p} : {q}"

    def _print_ModularIndexing(self, expr):
        x, div, mod = expr.args
        x = self.paren(self.doprint(x))
        if div != 1:
            div = self.paren(self.doprint(div))
            if expr.is_integer:
                x = f"c10::div_floor_integer(static_cast<int64_t>({x}), static_cast<int64_t>({div}))"
            else:
                x = f"c10::div_floor_floating(static_cast<double>({x}), static_cast<double>({div}))"
        mod = self.paren(self.doprint(mod))
        return f"static_cast<{INDEX_TYPE}>({x}) % static_cast<{INDEX_TYPE}>({mod})"

    def _print_FloorDiv(self, expr):
        x, div = expr.args
        x = self.paren(self.doprint(x))
        div = self.paren(self.doprint(div))
        if expr.is_integer:
            return f"c10::div_floor_integer(static_cast<int64_t>({x}), static_cast<int64_t>({div}))"
        return f"c10::div_floor_floating(static_cast<double>({x}), static_cast<double>({div}))"

    def _print_floor(self, expr):
        assert len(expr.args) == 1
        r = f"std::floor({self._print(expr.args[0])})"
        return f"static_cast<{INDEX_TYPE}>({r})" if expr.is_integer else r

    def _print_FloorToInt(self, expr):
        assert len(expr.args) == 1
        r = f"std::floor({self._print(expr.args[0])})"
        return f"static_cast<{INDEX_TYPE}>({r})" if expr.is_integer else r

    def _print_TruncToInt(self, expr):
        assert len(expr.args) == 1
        r = f"std::trunc({self._print(expr.args[0])})"
        return f"static_cast<{INDEX_TYPE}>({r})"

    def _print_TruncToFloat(self, expr):
        assert len(expr.args) == 1
        return f"std::trunc({self._print(expr.args[0])})"

    def _print_ToFloat(self, expr):
        assert len(expr.args) == 1
        return f"static_cast<double>({self._print(expr.args[0])})"

    # TODO: This is wrong if one of the inputs is negative.  This is hard to
    # tickle though, as the inputs are typically positive (and if we can prove
    # they are positive, we will have used Mod instead, for which this codegen
    # is right).
    def _print_PythonMod(self, expr):
        return " % ".join(map(self.paren, map(self._print, expr.args)))

    def _print_CMod(self, expr):
        return " % ".join(map(self.paren, map(self._print, expr.args)))

    def _print_IntTrueDiv(self, expr):
        lhs, rhs = expr.args
        # TODO: This is only accurate up to 2**53
        return f"static_cast<double>({self._print(lhs)}) / static_cast<double>({self._print(rhs)})"

    # TODO: PowByNatural: we need to implement our own int-int pow.  Do NOT
    # use std::pow, that operates on floats
    def _print_PowByNatural(self, expr):
        raise NotImplementedError(
            f"_print_PowByNatural not implemented for {type(self)}"
        )

    def _print_FloatTrueDiv(self, expr):
        lhs, rhs = expr.args
        return f"{self.paren(self._print(lhs))} / {self.paren(self._print(rhs))}"

    def _print_FloatPow(self, expr):
        base, exp = expr.args
        return f"std::pow({self._print(base)}, {self._print(exp)})"

    def _print_Pow(self, expr):
        # Uses float constants to perform FP div
        base, exp = expr.args
        base = self._print(base)

        if exp == 0.5 or exp == -0.5:
            return f"std::sqrt({base})" if exp == 0.5 else f"1.0/std::sqrt({base})"
        if exp.is_integer:
            exp = int(exp)
            if exp > 0:
                r = "*".join([self.paren(base)] * exp)
            elif exp < 0:
                r = "1.0/" + self.paren("*".join([self.paren(base)] * abs(exp)))
            else:  # exp == 0
                r = "1.0"

            return f"static_cast<{INDEX_TYPE}>({r})" if expr.is_integer else r
        else:
            # TODO: float vs double
            return f"std::pow({base}, {float(exp)})"

    def _print_Rational(self, expr):
        # Uses float constants to perform FP div
        if expr.q == 1:
            r = f"{expr.p}"
        else:
            r = f"{expr.p}.0/{expr.q}.0"
        return f"static_cast<{INDEX_TYPE}>({r})" if expr.is_integer else r

    def _print_ceiling(self, expr):
        assert len(expr.args) == 1
        r = f"std::ceil({self._print(expr.args[0])})"
        return f"static_cast<{INDEX_TYPE}>({r})" if expr.is_integer else r

    def _print_CeilToInt(self, expr):
        assert len(expr.args) == 1
        r = f"std::ceil({self._print(expr.args[0])})"
        return f"static_cast<{INDEX_TYPE}>({r})" if expr.is_integer else r

    def _print_Min(self, expr):
        args = [self._print(a) for a in expr.args]
        if len(args) == 2:
            return f"std::min(static_cast<{INDEX_TYPE}>({args[0]}), static_cast<{INDEX_TYPE}>({args[1]}))"
        else:
            # Initializer list overload
            il = "{" + ", ".join(args) + "}"
            return f"std::min({il})"

    def _print_Max(self, expr):
        args = [self._print(a) for a in expr.args]
        if len(args) == 2:
            return f"std::max(static_cast<{INDEX_TYPE}>({args[0]}), static_cast<{INDEX_TYPE}>({args[1]}))"
        else:
            # Initializer list overload
            il = "{" + ", ".join(args) + "}"
            return f"std::max({il})"

    def _print_Abs(self, expr):
        assert len(expr.args) == 1
        return f"std::abs({self._print(expr.args[0])})"

    def _print_OpaqueUnaryFn_cos(self, expr):
        assert len(expr.args) == 1
        return f"std::cos({self._print(expr.args[0])})"

    def _print_OpaqueUnaryFn_cosh(self, expr):
        assert len(expr.args) == 1
        return f"std::cosh({self._print(expr.args[0])})"

    def _print_OpaqueUnaryFn_acos(self, expr):
        assert len(expr.args) == 1
        return f"std::acos({self._print(expr.args[0])})"

    def _print_OpaqueUnaryFn_sin(self, expr):
        assert len(expr.args) == 1
        return f"std::sin({self._print(expr.args[0])})"

    def _print_OpaqueUnaryFn_sinh(self, expr):
        assert len(expr.args) == 1
        return f"std::sinh({self._print(expr.args[0])})"

    def _print_OpaqueUnaryFn_asin(self, expr):
        assert len(expr.args) == 1
        return f"std::asin({self._print(expr.args[0])})"

    def _print_OpaqueUnaryFn_tan(self, expr):
        assert len(expr.args) == 1
        return f"std::tan({self._print(expr.args[0])})"

    def _print_OpaqueUnaryFn_tanh(self, expr):
        assert len(expr.args) == 1
        return f"std::tanh({self._print(expr.args[0])})"

    def _print_OpaqueUnaryFn_atan(self, expr):
        assert len(expr.args) == 1
        return f"std::atan({self._print(expr.args[0])})"

    def _print_OpaqueUnaryFn_sqrt(self, expr):
        return f"std::sqrt({self._print(expr.args[0])})"

    def _print_RoundToInt(self, expr):
        assert len(expr.args) == 1
        # TODO: dispatch to llrint depending on index type
        return f"std::lrint({self._print(expr.args[0])})"

    def _print_RoundDecimal(self, expr):
        assert len(expr.args) == 2
        number, ndigits = expr.args
        if number.is_integer:
            # ndigits < 0 should have been filtered by the sympy function
            assert ndigits < 0
            raise ValueError(
                f"For integer inputs, only non-negative ndigits are currently supported, but got {ndigits}."
            )
        return f"static_cast<double>(std::nearbyint(1e{ndigits} * {self.paren(self._print(number))}) * 1e{-ndigits})"

    def _print_BooleanTrue(self, expr):
        return "true"

    def _print_BooleanFalse(self, expr):
        return "false"


# A function to print, useful for printing sympy symbols.
cexpr = CppPrinter().doprint


def cexpr_index(index):
    return f"static_cast<{INDEX_TYPE}>({cexpr(index)})"


def value_to_cpp(value, cpp_type):
    if value == float("-inf"):
        return f"-std::numeric_limits<{cpp_type}>::infinity()"
    elif value == float("inf"):
        return f"std::numeric_limits<{cpp_type}>::infinity()"
    elif isinstance(value, bool):
        return f"static_cast<{cpp_type}>({str(value).lower()})"
    elif math.isnan(value):
        return f"std::numeric_limits<{cpp_type}>::quiet_NaN()"
    else:
        return f"static_cast<{cpp_type}>({repr(value)})"


def rewrite_index_for_function(
    localize_buffer_handler: "LocalizeBufferHandler",
    index: sympy.Expr,
    global_buf_name: str,
):
    # Local buffer at the inner dimensions
    snode = V.graph.scheduler.name_to_buf[global_buf_name].defining_op
    local_buf = localize_buffer_handler.global_to_local[global_buf_name]
    scheduler_nodes = snode.get_nodes()
    _, (group, reduction_group) = max(
        scheduler_nodes, key=lambda x: int(x.is_reduction())
    ).group
    call_ranges = tuple(group) + tuple(reduction_group)
    indices_to_keep = [
        f"x{len(call_ranges) - (idx + 1)}"
        for idx in range(len(local_buf.get_layout().size))
    ]
    sorted_symbols = sorted(index.free_symbols, key=lambda s: s.name)  # type: ignore[attr-defined]
    replacements = {}
    for x in sorted_symbols:
        if x.name.startswith("x") and x.name not in indices_to_keep:  # type: ignore[attr-defined]
            # Only keep index used by local buffer
            replacements[x] = sympy.core.numbers.Zero()
    index = sympy_subs(index, replacements)  # type: ignore[arg-type]
    return index


def rewrite_index_for_nodes(
    localize_buffer_handler: "LocalizeBufferHandler",
    index: sympy.Expr,
    global_buf_name: str,
):
    used_vars = {s for s in index.free_symbols if symbol_is_type(s, SymT.INDEX)}
    index_vars = []
    local_buf = localize_buffer_handler.global_to_local[global_buf_name]
    for i in range(len(local_buf.get_size())):
        var = sympy_index_symbol_with_prefix(SymT.INDEX, i)
        index_vars.append(var if var in used_vars else 0)
    index = local_buf.layout.make_indexer()(index_vars)
    return index


class LocalizeBufferHandler(V.WrapperHandler):  # type: ignore[name-defined]
    def __init__(
        self,
        inner,
        global_to_local: Dict[str, ir.Buffer],
        rewrite_index: Callable[["LocalizeBufferHandler", sympy.Expr, str], sympy.Expr],
    ) -> None:
        super().__init__(inner)
        self.global_to_local = global_to_local
        self.rewrite_index = rewrite_index

    def localize(self, name: str, index: sympy.Expr):
        if self.global_to_local and name in self.global_to_local:
            assert self.rewrite_index is not None
            index = self.rewrite_index(self, index, name)
            name = self.global_to_local[name].get_name()
        return name, index

    def load(self, name: str, index: sympy.Expr):
        return self._inner.load(*self.localize(name, index))

    def store(self, name, index, value, mode=None):
        local_buffer_name, local_buffer_index = self.localize(name, index)
        res = self._inner.store(local_buffer_name, local_buffer_index, value, mode)
        if (
            self.global_to_local
            and name in self.global_to_local
            and isinstance(V.kernel, Kernel)
        ):
            # Remove name of local buffer from Kernel.store_buffer_names
            # local_buffer_name is added to Kernel.store_buffer_names in Kernel.CSEProxy.store.
            V.kernel.store_buffer_names.discard(local_buffer_name)
        return res

    def store_reduction(self, name, index, value):
        return self._inner.store_reduction(*self.localize(name, index), value)


class LocalBufferContext:
    """
    This class creates a context that helps to generate code involving Inductor IR with
    function local buffers. These buffers are constructed during the codegen process and
    are used to store intermediate results such as local accumulators. We do not want to
    add them to `V.graph` since they are not global and we do not want to add them as
    function arguments either. So we patch the codegen processes under this scope to support
    these buffers without exposure to the outside world.
    """

    def __init__(self, kernel_args: KernelArgs) -> None:
        self.kernel_args = kernel_args
        self.exit_stack = contextlib.ExitStack()
        # map local buffer name to local buffer
        self.local_buffers: Dict[str, ir.Buffer] = {}
        # map global buffer name to global buffer
        self.global_buffers: Dict[str, ir.Buffer] = {}
        # map global buffer name to local buffer
        self.global_to_local: Dict[str, ir.Buffer] = {}

    def __enter__(self):
        self.exit_stack.__enter__()
        original_get_dtype = V.graph.get_dtype

        def get_dtype(name):
            if name in self.local_buffers:
                return self.local_buffers[name].get_dtype()
            return original_get_dtype(name)

        self.exit_stack.enter_context(patch.object(V.graph, "get_dtype", get_dtype))

        original_input = self.kernel_args.input

        def input(name):
            if name in self.local_buffers:
                return name
            return original_input(name)

        self.exit_stack.enter_context(patch.object(self.kernel_args, "input", input))

        original_output = self.kernel_args.output

        def output(name):
            if name in self.local_buffers:
                return name
            return original_output(name)

        self.exit_stack.enter_context(patch.object(self.kernel_args, "output", output))

        # Set current LocalBufferContext into V
        self.exit_stack.enter_context(V.set_local_buffer_context(self))

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.local_buffers.clear()
        self.exit_stack.__exit__(exc_type, exc_val, exc_tb)

    def add_local_buffer(
        self, local_buffer: ir.Buffer, global_buffers: Optional[List[ir.Buffer]] = None
    ):
        assert local_buffer.get_name() not in self.local_buffers
        self.local_buffers[local_buffer.get_name()] = local_buffer
        if global_buffers:
            for global_buffer in global_buffers:
                global_buffer_name = global_buffer.get_name()
                assert (
                    global_buffer_name not in self.global_buffers
                    and global_buffer_name not in self.global_to_local
                )
                self.global_buffers[global_buffer_name] = global_buffer
                self.global_to_local[global_buffer_name] = local_buffer
                V.graph.removed_buffers.add(global_buffer_name)

    def localize_function(
        self,
        fn: Callable[..., Any],
        rewrite_index: Callable[
            ["LocalizeBufferHandler", sympy.Expr, str], sympy.Expr
        ] = rewrite_index_for_function,
    ):
        def inner(*args, **kwargs):
            with V.set_ops_handler(
                LocalizeBufferHandler(
                    V.get_ops_handler(),
                    global_to_local=self.global_to_local,
                    rewrite_index=rewrite_index,
                )
            ):
                return fn(*args, **kwargs)

        return inner

    def localize_nodes(
        self,
        nodes: List[ir.IRNode],
        rewrite_index: Callable[
            ["LocalizeBufferHandler", sympy.Expr, str], sympy.Expr
        ] = rewrite_index_for_nodes,
    ) -> List[ir.IRNode]:
        """
        Given `local_buf` and `global_buf` registered in current `LocalBufferContext`
        though the method of `add_local_buffer`, localizes the `global_buf` to `local_buf`
        for the given `nodes` and returns a new list of IR nodes that work on `local_buf`
        instead of `global_buf`, i.e., all the loads and stores are redirected to
        `local_buf`. This helps the fused loops to work on smaller-sized local buffers
        for better data locality.

        The the data access of `local_buf` is assumed to be contiguous with the
        same order as the `global_buf`.
        """
        assert len(nodes) > 0

        def wrap_inner_fn_for_node(node: ir.IRNode):
            loops = node.data if isinstance(node, ir.ComputedBuffer) else node
            assert isinstance(loops, ir.Loops)
            new_loops = copy.copy(loops)
            if isinstance(node, ir.ComputedBuffer):
                new_node = ir.ComputedBuffer(
                    node.get_name(), node.get_layout(), new_loops
                )
            else:
                new_node = new_loops  # type: ignore[assignment]

            new_loops.inner_fn = self.localize_function(
                new_loops.inner_fn,
                rewrite_index,
            )
            return new_node

        return [wrap_inner_fn_for_node(node) for node in nodes]


def unify_mask_base_type(
    buffer: IndentedBuffer,
    vars: Tuple[CSEVariable, ...],
    dtype=torch.float,
):
    """
    Given list of cse variables,
    Cast each to new mask base dtype and return casted cse variable.
    """
    new_vars = (
        V.kernel.cse.generate(
            buffer,
            f"{V.kernel._get_mask_cast(var, dtype)}",
        )
        for var in vars
    )
    return new_vars


def codegen_rand(offset, code, rand_function, dst_dtype=torch.float32):
    assert is_integer_dtype(offset.dtype)
    code.writeline("[&]()")
    with code.indent():
        code.writeline(
            f"{DTYPE_TO_CPP[offset.dtype]} offset[{V.kernel.tiling_factor}];"
        )
        code.writeline(f"{DTYPE_TO_CPP[dst_dtype]} result[{V.kernel.tiling_factor}];")
        code.writeline(f"{offset}.store(offset);")
        code.writeline(
            f"for( {DTYPE_TO_CPP[offset.dtype]} offset_idx = 0; offset_idx < {V.kernel.tiling_factor}; offset_idx++ )"
        )
        with code.indent():
            code.writeline(rand_function)
        num_vectors = V.kernel._get_num_vectors(dtype=dst_dtype)
        if num_vectors == 1:
            code.writeline(
                f"return at::vec::Vectorized<{DTYPE_TO_CPP[dst_dtype]}>::loadu(result);"
            )
        else:
            code.writeline(
                f"return at::vec::VectorizedN<{DTYPE_TO_CPP[dst_dtype]}, {num_vectors}>::loadu(result);"
            )
    code.writeline("()")
    return code


def get_gemm_template_output_and_compute_dtype(input_dtype):
    if input_dtype == torch.uint8:
        return (torch.int32, torch.int32)
    else:
        return (torch.float32, torch.float32)


def create_epilogue_with_attr(input_buffer, attr, **kwargs):
    input_loader = input_buffer.make_loader()
    dtype = input_buffer.get_dtype()
    if attr == "relu":

        def inner_fn(index):
            input = input_loader(index)
            zero = ops.constant(0, dtype)
            return ops.maximum(input, zero)

    elif attr == "gelu":
        assert "algorithm" in kwargs
        if kwargs["algorithm"] == "none":

            def inner_fn(index):
                input = input_loader(index)
                if dtype != torch.float:
                    input = ops.to_dtype(input, torch.float)
                half = ops.constant(0.5, torch.float)
                one = ops.constant(1.0, torch.float)
                const = ops.constant(0.7071067811865476, torch.float)
                result = input * half * (ops.erf(input * const) + one)
                if dtype != torch.float:
                    result = ops.to_dtype(result, dtype)
                return result

        else:
            assert kwargs["algorithm"] == "tanh"

            def inner_fn(index):
                input = input_loader(index)
                if dtype != torch.float:
                    input = ops.to_dtype(input, torch.float)
                half = ops.constant(0.5, torch.float)
                one = ops.constant(1.0, torch.float)
                const1 = ops.constant(0.7978845608028654, torch.float)
                const2 = ops.constant(0.044715, torch.float)
                result = (
                    half
                    * input
                    * (
                        one
                        + ops.tanh(const1 * (input + const2 * input * input * input))
                    )
                )
                if dtype != torch.float:
                    result = ops.to_dtype(result, dtype)
                return result

    elif attr == "swish":

        def inner_fn(index):
            input = input_loader(index)
            result = input * ops.sigmoid(input)
            return result

    elif attr == "sigmoid":

        def inner_fn(index):
            return ops.sigmoid(input_loader(index))

    elif attr == "tanh":

        def inner_fn(index):
            return ops.tanh(input_loader(index))

    elif attr == "hardswish" or attr == "hardsigmoid":

        def hardsigmoid_float(input):
            zero = ops.constant(0, torch.float)
            six = ops.constant(6, torch.float)
            three = ops.constant(3, torch.float)
            one_over_six = ops.constant(0.16666666666666666, torch.float)
            max = ops.maximum(input + three, zero)
            min = ops.minimum(max, six)
            return min * one_over_six

        def inner_fn(index):
            input = input_loader(index)
            if dtype != torch.float:
                input = ops.to_dtype(input, torch.float)
            result = hardsigmoid_float(input)
            if attr == "hardswish":
                result = input * result
            if dtype != torch.float:
                result = ops.to_dtype(result, dtype)
            return result

    elif attr == "leaky_relu":
        assert "scalars" in kwargs
        assert len(kwargs["scalars"]) == 1
        negative_slope = kwargs["scalars"][0]

        def inner_fn(index):
            input = input_loader(index)
            if dtype != torch.float:
                input = ops.to_dtype(input, torch.float)
            zero = ops.constant(0, torch.float)
            result = ops.where(
                input > zero, input, input * ops.constant(negative_slope, torch.float)
            )
            if dtype != torch.float:
                result = ops.to_dtype(result, dtype)
            return result

    elif attr == "hardtanh":
        assert "scalars" in kwargs
        assert len(kwargs["scalars"]) == 2
        min_value = kwargs["scalars"][0]
        max_value = kwargs["scalars"][1]

        def inner_fn(index):
            input = input_loader(index)
            if dtype != torch.float:
                input = ops.to_dtype(input, torch.float)
            result = ops.minimum(
                ops.maximum(input, ops.constant(min_value, torch.float)),
                ops.constant(max_value, torch.float),
            )
            if dtype != torch.float:
                result = ops.to_dtype(result, dtype)
            return result

    elif attr in ["add", "sub", "mul"]:
        assert "other" in kwargs
        other = kwargs["other"]
        num_input_dims = len(input_buffer.get_size())
        num_other_dims = len(other.get_size())
        dims_diff = num_input_dims - num_other_dims
        other_loader = other.make_loader()

        def inner_fn(index):
            op = getattr(ops, attr)
            if dims_diff != 0:
                return op(input_loader(index), other_loader(index[dims_diff:]))
            else:
                return op(input_loader(index), other_loader(index))

    elif attr == "bias_add":
        assert "other" in kwargs
        assert "beta" in kwargs
        assert "dtype" in kwargs
        beta = kwargs["beta"]
        other = kwargs["other"]
        dtype = kwargs["dtype"]
        bias_loader = other.make_loader()

        def inner_fn(index):
            bias = bias_loader(index)
            input = input_loader(index)
            if beta != 1:
                result = ops.constant(beta, torch.float) * bias + input
            else:
                result = bias + input
            return result

    else:
        raise ValueError(f"Unsupported epilogue attribute: {attr}")
    return ir.Pointwise(
        device=input_buffer.get_device(),
        dtype=dtype,
        inner_fn=inner_fn,
        ranges=input_buffer.get_size(),
    )


def _get_loop_body(fn_list):
    if all(isinstance(fn, LoopBody) for fn in fn_list):
        loop_bodies = fn_list
    else:
        if hasattr(fn_list[0], "original_fn"):
            # For the case of local buffer, we wrap the fn with localize_function
            assert all(hasattr(fn, "original_fn") for fn in fn_list)
            assert all(
                isinstance(fn.original_fn.args[0]._body, LoopBody) for fn in fn_list
            )
            loop_bodies = [fn.original_fn.args[0]._body for fn in fn_list]
        else:
            assert all(isinstance(fn, functools.partial) for fn in fn_list)
            assert all(isinstance(fn.args[0]._body, LoopBody) for fn in fn_list)
            loop_bodies = [fn.args[0]._body for fn in fn_list]
    assert loop_bodies is not None
    return loop_bodies


def _get_dtype_from_loopbodies(loop_bodies):
    dtypes = set()
    for loop_body in loop_bodies:
        graphs = [loop_body.root_block.graph] + [
            body.graph for body in list(loop_body.subblocks.values())
        ]
        for graph in graphs:
            for node in graph.nodes:
                if node.op != "call_method":
                    continue
                dtypes.add(node.meta[OptimizationContext.key].dtype)
    return dtypes
