# mypy: allow-untyped-defs
import contextlib
import copy
import math

from collections import namedtuple
from typing import Any, Callable, Dict, List, Optional, Tuple
from unittest.mock import patch

import sympy

import torch
from torch.utils._sympy.symbol import symbol_is_type, SymT
from .. import ir

from ..scheduler import SchedulerBuffer
from ..utils import IndentedBuffer, sympy_index_symbol_with_prefix, sympy_subs
from ..virtualized import V

from .common import CSEVariable, ExprPrinter, Kernel, KernelArgs

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
    torch.complex64: "complex64",
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

INDEX_TYPE = "long"

GemmBlocking = namedtuple("GemmBlocking", ["block_m", "block_n", "block_k"])


class CppPrinter(ExprPrinter):
    def _print_Integer(self, expr):
        return f"{int(expr)}L"

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
                x = f"c10::div_floor_integer({x}, {div})"
            else:
                x = f"c10::div_floor_floating(static_cast<double>({x}), static_cast<double>({div}))"
        mod = self.paren(self.doprint(mod))
        return f"static_cast<{INDEX_TYPE}>({x}) % static_cast<{INDEX_TYPE}>({mod})"

    def _print_FloorDiv(self, expr):
        x, div = expr.args
        x = self.paren(self.doprint(x))
        div = self.paren(self.doprint(div))
        if expr.is_integer:
            return f"c10::div_floor_integer({x}, {div})"
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
            return f"std::min({args[0]}, {args[1]})"
        else:
            # Initializer list overload
            il = "{" + ", ".join(args) + "}"
            return f"std::min({il})"

    def _print_Max(self, expr):
        args = [self._print(a) for a in expr.args]
        if len(args) == 2:
            return f"std::max({args[0]}, {args[1]})"
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
    sbuffer = V.graph.scheduler.name_to_buf.get(global_buf_name)
    assert isinstance(sbuffer, SchedulerBuffer)
    snode = sbuffer.defining_op
    local_buf = localize_buffer_handler.global_to_local[global_buf_name]
    assert snode is not None
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
    ):
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

    def __init__(self, kernel_args: KernelArgs):
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
                # Patch the attr of SchedulerNode's outputs and outputs_by_name.
                # Since the name of local buffer has not been added into the outputs
                # of schedulerNode, which fails method of CSEProxy._update_store_cache
                # to get the SchedulerBuffer.
                # We create a new SchedulerBuffer with Local Buffer to substitute the
                # original SchedulerBuffer with global buffer.
                if (
                    V.graph.scheduler
                    and global_buffer.get_name() in V.graph.scheduler.name_to_buf
                ):
                    scheduler_buffer = V.graph.scheduler.name_to_buf.get(
                        global_buffer.get_name()
                    )
                    assert isinstance(scheduler_buffer, SchedulerBuffer)
                    scheduler_node = scheduler_buffer.defining_op
                    new_outputs: List[SchedulerBuffer] = [
                        output
                        for output in scheduler_node.get_outputs()
                        if output.get_name() != global_buffer.get_name()
                    ]
                    new_outputs.append(
                        SchedulerBuffer(
                            scheduler=scheduler_node.scheduler,
                            node=local_buffer,
                            defining_op=scheduler_node,
                        )
                    )
                    new_outputs_by_name: Dict[str, SchedulerBuffer] = {
                        buf.get_name(): buf for buf in new_outputs
                    }
                    self.exit_stack.enter_context(
                        patch.object(scheduler_node, "outputs", new_outputs)
                    )
                    self.exit_stack.enter_context(
                        patch.object(scheduler_node, "outputs_by_name", new_outputs_by_name)
                    )

    def localize_function(
        self,
        fn: Callable[..., Any],
        rewrite_index: Callable[
            ["LocalizeBufferHandler", sympy.Expr, str], sympy.Expr
        ] = rewrite_index_for_function,
    ):
        def inner(node, *index_vars):
            with V.set_ops_handler(
                LocalizeBufferHandler(
                    V.get_ops_handler(),
                    global_to_local=self.global_to_local,
                    rewrite_index=rewrite_index,
                )
            ):
                return fn(node, *index_vars)

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


def get_gemm_template_output_and_compute_dtype(input_dtype):
    if input_dtype == torch.uint8:
        return (torch.int32, torch.int32)
    else:
        return (torch.float32, torch.float32)
