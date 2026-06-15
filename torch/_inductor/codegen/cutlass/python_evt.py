import ast
import itertools
from collections.abc import Callable, Generator, Iterable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from os import linesep
from typing import Any

import sympy

import torch
import torch._inductor.virtualized as virtualized
from torch._inductor.ir import ComputedBuffer, Pointwise
from torch._inductor.ops_handler import DefaultHandler, WrapperHandler
from torch._inductor.scheduler import BaseSchedulerNode
from torch._inductor.utils import DelayReplaceLine, IndentedBuffer, OrderedSet
from torch._inductor.virtualized import OpsValue

from ...virtualized import V


_ACCUMULATOR_ARG_NAME = "accum"

# 1/sqrt(2) (M_SQRT1_2), the constant Inductor's gelu decomposition multiplies
# the input by before applying erf.
_GELU_ERF_INV_SQRT2 = 0.7071067811865476


def _is_close_constant(node: ast.expr, value: float, tol: float = 1e-6) -> bool:
    return (
        isinstance(node, ast.Constant)
        and isinstance(node.value, (int, float))
        and not isinstance(node.value, bool)
        and abs(float(node.value) - value) < tol
    )


def _mult_factors(node: ast.expr) -> list[ast.expr]:
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult):
        return _mult_factors(node.left) + _mult_factors(node.right)
    return [node]


def _match_call(node: ast.expr, name: str) -> "ast.Call | None":
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == name
        and len(node.args) == 1
    ):
        return node
    return None


def _same(a: ast.expr, b: ast.expr) -> bool:
    return ast.dump(a) == ast.dump(b)


def _match_gelu_erf(node: ast.expr) -> "ast.expr | None":
    """Match gelu's erf decomposition ``x * 0.5 * (1 + erf(x / sqrt(2)))`` and
    return the inner argument ``x``. Matching is commutativity-aware for the
    multiplications and the addition.
    """
    factors = _mult_factors(node)
    if len(factors) != 3:
        return None
    half = next((f for f in factors if _is_close_constant(f, 0.5)), None)
    if half is None:
        return None
    rest = [f for f in factors if f is not half]

    def match_one_plus_erf(f: ast.expr) -> "ast.expr | None":
        # Match ``1 + erf(x / sqrt(2))`` and return ``x``.
        if not (isinstance(f, ast.BinOp) and isinstance(f.op, ast.Add)):
            return None
        operands = [f.left, f.right]
        if not any(_is_close_constant(o, 1.0) for o in operands):
            return None
        erf_call = next(
            (c for c in (_match_call(o, "erf") for o in operands) if c is not None),
            None,
        )
        if erf_call is None:
            return None
        erf_factors = _mult_factors(erf_call.args[0])
        if len(erf_factors) != 2 or not any(
            _is_close_constant(g, _GELU_ERF_INV_SQRT2) for g in erf_factors
        ):
            return None
        return next(
            g for g in erf_factors if not _is_close_constant(g, _GELU_ERF_INV_SQRT2)
        )

    # ``x`` may itself be an addition (e.g. gelu(accum + bias)), so identify the
    # ``1 + erf(...)`` factor by structure rather than assuming it is the only Add.
    for i, f in enumerate(rest):
        erf_x = match_one_plus_erf(f)
        if erf_x is not None and _same(erf_x, rest[1 - i]):
            return rest[1 - i]
    return None


@dataclass(frozen=True)
class _ActivationPattern:
    # ``name`` must be a functor the CUTLASS EVT frontend binds natively (see
    # ``ast_op_to_bindings`` in the cutlass python_ast frontend).
    name: str
    # Cheap substring guard so we only parse epilogues that may contain the
    # decomposition (a primitive that the activation expands into).
    trigger: str
    match: Callable[[ast.expr], "ast.expr | None"]


# Activations that Inductor decomposes but CUTLASS can emit as a single functor.
# Add new entries here to re-fuse additional decomposed activations.
_ACTIVATION_PATTERNS: tuple[_ActivationPattern, ...] = (
    _ActivationPattern("gelu", "erf", _match_gelu_erf),
)


def _fuse_activations(code: str) -> str:
    """Re-compose decomposed activations back into single CUTLASS functor calls.

    Inductor lowers activations such as ``aten.gelu`` into
    primitive ops (``erf``, ``sigmoid``, ``mul``, ...) before reaching CUTLASS
    codegen. The CUTLASS EVT frontend natively supports a number of activation
    functors, so we pattern-match each decomposition (``_ACTIVATION_PATTERNS``)
    and fold it back into a single activation node. This enables fusion for
    activations whose primitives are unsupported (e.g. the ``erf`` in gelu) and
    reduces the compute-node count for the rest.
    """
    active = [p for p in _ACTIVATION_PATTERNS if p.trigger in code]
    if not active:
        return code
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code
    if not tree.body or not isinstance(tree.body[0], ast.FunctionDef):
        return code
    func = tree.body[0]

    assigns: dict[str, ast.expr] = {}
    for stmt in func.body:
        if (
            isinstance(stmt, ast.Assign)
            and len(stmt.targets) == 1
            and isinstance(stmt.targets[0], ast.Name)
        ):
            assigns[stmt.targets[0].id] = stmt.value

    def inline(node: ast.expr, seen: OrderedSet[str] = OrderedSet()) -> ast.expr:
        # Fully inline temporary assignments so the expression tree only refers
        # to function parameters (accum / read buffers) and literal constants.
        if isinstance(node, ast.Name) and node.id in assigns:
            if node.id in seen:
                return node
            return inline(assigns[node.id], seen | OrderedSet([node.id]))
        if isinstance(node, ast.BinOp):
            return ast.BinOp(
                left=inline(node.left, seen),
                op=node.op,
                right=inline(node.right, seen),
            )
        if isinstance(node, ast.UnaryOp):
            return ast.UnaryOp(op=node.op, operand=inline(node.operand, seen))
        if isinstance(node, ast.Call):
            return ast.Call(
                func=node.func,
                args=[inline(a, seen) for a in node.args],
                keywords=[],
            )
        return node

    changed = False
    for stmt in func.body:
        if (
            isinstance(stmt, ast.Assign)
            and len(stmt.targets) == 1
            and isinstance(stmt.targets[0], ast.Name)
        ):
            inlined = inline(stmt.value)
            for pattern in active:
                x = pattern.match(inlined)
                if x is not None:
                    stmt.value = ast.Call(
                        func=ast.Name(id=pattern.name, ctx=ast.Load()),
                        args=[x],
                        keywords=[],
                    )
                    changed = True
                    break

    if not changed:
        return code

    # The folded assignment no longer references the decomposition temporaries,
    # so drop the now-dead intermediates (the CUTLASS frontend builds a compute
    # node for every assignment it sees). The frontend emits ``return`` dedented
    # to module level (a sibling of the function def in Python 3.12), so seed
    # liveness from every Return in the tree, not only those inside ``func.body``.
    live: OrderedSet[str] = OrderedSet()
    for node in ast.walk(tree):
        if isinstance(node, ast.Return) and node.value is not None:
            live.update(n.id for n in ast.walk(node.value) if isinstance(n, ast.Name))

    kept: list[ast.stmt] = []
    for stmt in reversed(func.body):
        if (
            isinstance(stmt, ast.Assign)
            and len(stmt.targets) == 1
            and isinstance(stmt.targets[0], ast.Name)
        ):
            if stmt.targets[0].id not in live:
                continue
            live.update(n.id for n in ast.walk(stmt.value) if isinstance(n, ast.Name))
        kept.append(stmt)
    func.body = list(reversed(kept))

    folded = ast.unparse(ast.fix_missing_locations(tree))

    # After dead-code elimination, any activation primitive that served as a
    # trigger (e.g. erf) should no longer appear in the live code -- it was
    # consumed by the pattern match and eliminated.  If one still appears, the
    # fold was incomplete (e.g. a standalone erf not part of a known pattern);
    # bail out and return the original code so the CUTLASS frontend rejects the
    # epilogue cleanly rather than emitting an unsupported functor.
    for p in active:
        if p.trigger in folded:
            return code

    return folded


def scaled_mm_evt(
    scale_A_name: str, scale_B_name: str, bias_name: str | None, output_name: str
) -> tuple[list[str], dict[str, Any], str]:
    evt_read_names = [scale_A_name, scale_B_name]
    var_name_to_buffer_name = {n: n for n in [scale_A_name, scale_B_name]}
    var_name_to_buffer_name["D"] = output_name
    var_name_to_buffer_name[_ACCUMULATOR_ARG_NAME] = output_name
    expr = f"accum * {scale_A_name} * {scale_B_name}{linesep}"
    if bias_name:
        expr = f"({expr}) + {bias_name}"
        evt_read_names.append(bias_name)
        var_name_to_buffer_name[bias_name] = bias_name

    evt_py_code = f"def fn(accum, {','.join(evt_read_names)}):{linesep}\
    D = {expr}{linesep}\
    return D{linesep}"

    return evt_read_names, var_name_to_buffer_name, evt_py_code


class CutlassEVTOpsMixIn:
    @staticmethod
    def _infix_bin_op(op: str, a: str, b: str) -> str:
        return f"{a} {op} {b}"

    @staticmethod
    def _prefix_bin_op(op: str, a: str, b: str) -> str:
        return f"{op}({a}, {b})"

    @staticmethod
    def _prefix_un_op(op: str, a: str) -> str:
        return f"{op}({a})"

    @staticmethod
    def to_dtype(
        x: str,
        dtype: Any,
        src_dtype: torch.dtype | None = None,
        use_compute_types: bool = False,
    ) -> str:
        return x

    @staticmethod
    def constant(value: Any, dtype: Any) -> str:
        # Return the constant as a Python float literal so the EVT python code
        # string can embed it directly (e.g. for the 0.5 and 1/sqrt(2) factors
        # in gelu).  The CUTLASS EVT frontend will render it as a C++ scalar.
        # Only numeric constants appear here (Inductor does not lower non-numeric
        # constants into pointwise epilogues).
        return str(float(value))

    @staticmethod
    def mul(x0: str, x1: str) -> str:
        return CutlassEVTOpsMixIn._infix_bin_op("*", x0, x1)

    @staticmethod
    def truediv(x0: str, x1: str) -> str:
        return CutlassEVTOpsMixIn._infix_bin_op("/", x0, x1)

    @staticmethod
    def ge(x0: str, x1: str) -> str:
        raise NotImplementedError

    @staticmethod
    def add(x0: str, x1: str) -> str:
        return CutlassEVTOpsMixIn._infix_bin_op("+", x0, x1)

    @staticmethod
    def relu(x0: str) -> str:
        return CutlassEVTOpsMixIn._prefix_un_op("relu", x0)

    @staticmethod
    def sigmoid(x0: str) -> str:
        return CutlassEVTOpsMixIn._prefix_un_op("sigmoid", x0)

    @staticmethod
    def sub(x0: str, x1: str) -> str:
        return CutlassEVTOpsMixIn._infix_bin_op("-", x0, x1)

    @staticmethod
    def tanh(x0: str) -> str:
        return CutlassEVTOpsMixIn._prefix_un_op("tanh", x0)

    @staticmethod
    def exp(x0: str) -> str:
        return CutlassEVTOpsMixIn._prefix_un_op("exp", x0)

    @staticmethod
    def erf(x0: str) -> str:
        return CutlassEVTOpsMixIn._prefix_un_op("erf", x0)


class MockCutlassHandler(CutlassEVTOpsMixIn, WrapperHandler):
    """Passthrough handler for cutlass ops, used for running epilogue nodes for memory planning"""


class _AssignmentFormatter(DefaultHandler):
    def __init__(self, parent_handler: "CutlassEVTCodegen"):
        self.parent_handler = parent_handler

    def _default(self, name: str, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
        # Handle op dispatch here
        if hasattr(self.parent_handler, name):
            fn = getattr(self.parent_handler, name)
            line = fn(*args, **kwargs)
            if name in ("load", "store"):
                return OpsValue(line)
            else:
                var = self.parent_handler._tmp_var()
                line = DelayReplaceLine(
                    var,
                    lambda: "D"
                    if var == self.parent_handler.last_stored_var_name
                    else var,
                    f"{var} = {line}",
                )
                self.parent_handler.body.writeline(line)
                return OpsValue(var)
        else:
            raise NotImplementedError(name)


class CutlassEVTCodegen(CutlassEVTOpsMixIn):
    """
    Notes:
        * Used by CUTLASSGemmTemplate.
        * This class should not be instantiated by users, it is intended to be used
            by calling CutlassEVTCodegen.ir_to_evt_python_code(...)
            which instantiates this class as an ops handler for virtualized.V.ops.[op-name]
        * Extend this with more _op_<whatever> nodes to add support for new pointwise operations.
    """

    def __init__(self, accumulator_node_name: str, removed_buffers: OrderedSet[str]):
        """

        Initializes a CutlassEVTEpilogueArgumentFormatter object. Do not instantiate directly.
        Use the CutlassEVTCodegen.ir_to_evt_python_code static method.

        Args:
            accumulator_node_name: The name of the accumulator node which should contain
                                          the Matmul result before fusion according to the IR graph.
            epilogue_nodes: The list of scheduler nodes to be fused into the epilogue
        """
        self.accumulator_node_name: str = accumulator_node_name  #
        self.body: IndentedBuffer = IndentedBuffer(1)  # The body buffer for codegen
        self.var_counter: Iterator[int] = itertools.count()
        self.store_name_to_value: dict[str, OpsValue] = (
            dict()
        )  # Aliases for subexpression functors
        self.reads: OrderedSet[str] = OrderedSet([])
        # Used for creating example tensors
        self.var_name_to_buffer_name: dict[str, str] = {
            _ACCUMULATOR_ARG_NAME: accumulator_node_name
        }
        self.removed_buffers: OrderedSet[str] = removed_buffers
        self.cur_node: ComputedBuffer | None = None
        self.name_to_buffer = V.graph.name_to_buffer | V.graph.graph_inputs
        for name in V.graph.constants:
            # pyrefly: ignore [unsupported-operation]
            self.name_to_buffer[name] = V.graph.add_tensor_constant(
                V.graph.constants[name], name
            )
        self.is_D_assigned = False
        self.D_var_name = None

        if accumulator_node_name not in removed_buffers:
            # cannot return accumulator directly, so alias it
            var = self._tmp_var()
            self.body.writeline(f"{var} = {_ACCUMULATOR_ARG_NAME}")
            self.store(accumulator_node_name, value=OpsValue(var))

    @staticmethod
    def ir_to_evt_python_code(
        cutlass_template_node_name: str,
        epilogue_nodes: list[BaseSchedulerNode],
        removed_buffers: OrderedSet[str],
        fn_name: str = "fn",
        as_standalone_function: bool = False,
    ) -> tuple[list[str], list[str], dict[str, Any], str]:
        codegen = CutlassEVTCodegen(cutlass_template_node_name, removed_buffers)
        handler = _AssignmentFormatter(codegen)

        with virtualized.V.set_ops_handler(handler):
            for s_node in epilogue_nodes:
                node = s_node.node
                if not isinstance(node, ComputedBuffer):
                    raise AssertionError(
                        f"expected node to be a ComputedBuffer, got {type(node)}"
                    )
                with codegen.set_cur_node(node):
                    index_vars = CutlassEVTCodegen.get_index_vars(node)
                    node.get_store_function()(index_vars)

        codegen.finalize()

        return (
            codegen.get_reads(),
            codegen.get_writes(),
            codegen.get_renames(),
            codegen.get_value(
                fn_name=fn_name,
                as_standalone_function=as_standalone_function,
            ),
        )

    def get_value(
        self,
        fn_name: str = "fn",
        as_standalone_function: bool = False,
    ) -> str:
        ret = self._render_return_statement()
        if as_standalone_function:
            ret = "    " + ret
        return _fuse_activations(
            linesep.join(
                [
                    self._render_input_signature(fn_name),
                    self.body.getvalue(),
                    ret,
                ]
            )
        )

    def finalize(self) -> None:
        # Rename the last store to D
        # no other code references this store
        # to workaround https://github.com/NVIDIA/cutlass/issues/2288
        # Note: the delayed line will automatically rewrite the last assignment to
        # be to D
        buffer_name = self.var_name_to_buffer_name[self.last_stored_var_name]
        self.var_name_to_buffer_name.pop(self.last_stored_var_name)
        self.var_name_to_buffer_name["D"] = buffer_name
        self.store_name_to_value[buffer_name] = OpsValue("D")

    @contextmanager
    def set_cur_node(self, node: ComputedBuffer) -> Generator[None, Any, Any]:
        prev_node = self.cur_node
        try:
            self.cur_node = node
            yield
        finally:
            self.cur_node = prev_node

    def get_renames(self) -> dict[str, str]:
        return dict(self.var_name_to_buffer_name)

    def get_reads(self) -> list[str]:
        return list(self.reads.difference(self.store_name_to_value.keys()))

    def get_writes(self) -> list[str]:
        return list(self.store_name_to_value.keys())

    def load(self, name: str, index: Any) -> str:
        self._check_indexing(name, index)
        if name in self.store_name_to_value:
            return self.store_name_to_value[name].value
        elif name == self.accumulator_node_name:
            return _ACCUMULATOR_ARG_NAME
        else:
            self.reads.add(name)
            self.var_name_to_buffer_name[name] = name
            return name

    def store(
        self, name: Any, index: Any = None, value: Any = None, mode: Any = None
    ) -> None:
        if name not in self.removed_buffers:
            if index:
                self._check_indexing(name, index)
            if value.value == _ACCUMULATOR_ARG_NAME:
                raise AssertionError("Cannot store accumulator arg name")
            self.var_name_to_buffer_name[value.value] = name
            self.store_name_to_value[name] = value
            self.last_stored_var_name = value.value
        return None

    def _get_cur_node(self) -> ComputedBuffer:
        if not self.cur_node:
            raise AssertionError("expected cur_node to be set")
        return self.cur_node

    @staticmethod
    def get_index_vars(node: ComputedBuffer) -> Sequence[sympy.Expr]:
        data = node.data
        # TODO mlazos: relax this, cutlass supports reductions and other ops
        if not isinstance(data, Pointwise):
            raise AssertionError(f"expected data to be Pointwise, got {type(data)}")
        return data._index(data.ranges)

    def _get_current_index_vars(self) -> Sequence[sympy.Expr]:
        return self.get_index_vars(self._get_cur_node())

    def _check_indexing(self, name: str, index: sympy.Expr) -> None:
        # We only support indexing that matches the layout today because
        # CUTLASS doesn't support arbitrary indexing
        buffer_name = (
            self.accumulator_node_name if name == _ACCUMULATOR_ARG_NAME else name
        )
        buffer = self.name_to_buffer[buffer_name]
        index_strides = V.graph.sizevars.stride_vars(
            index, self._get_current_index_vars()
        )
        stride = buffer.get_layout().stride
        if not self._stride_compatible(stride, index_strides):
            raise NotImplementedError(
                f"Unsupported indexing for {name} with index {index}, index strides {index_strides}, and layout stride {stride}"
            )

    def _stride_compatible(
        self, left: Iterable[sympy.Expr], right: Iterable[sympy.Expr]
    ) -> bool:
        def _provably_equal_or_zero(a: sympy.Expr, b: sympy.Expr) -> bool:
            # sympy.Eq can return an unevaluated Equality object; only accept
            # cases sympy can prove true.
            if a == b:
                return True
            if V.graph.sizevars.statically_known_equals(a, b):
                return True
            if V.graph.sizevars.statically_known_equals(a, 0):
                return True
            if V.graph.sizevars.statically_known_equals(b, 0):
                return True
            return (
                sympy.Eq(a, b) is sympy.true
                or sympy.Eq(a, 0) is sympy.true
                or sympy.Eq(b, 0) is sympy.true
            )

        left_list = list(left)
        right_list = list(right)
        # Same length: direct comparison
        if len(left_list) == len(right_list):
            return all(
                _provably_equal_or_zero(l, r) for l, r in zip(left_list, right_list)
            )
        # Different lengths: allow compatible reshapes where trailing strides match.
        # This handles view/reshape between template output and consumer, e.g.,
        # buffer strides [3072, 1] vs index strides [1572864, 3072, 1]
        shorter, longer = (
            (left_list, right_list)
            if len(left_list) <= len(right_list)
            else (right_list, left_list)
        )
        n = len(shorter)
        # Check that the trailing strides match
        return all(
            _provably_equal_or_zero(shorter[-(i + 1)], longer[-(i + 1)])
            for i in range(n)
        )

    def _render_input_signature(self, fn_name: str = "fn") -> str:
        arguments = ", ".join(
            [_ACCUMULATOR_ARG_NAME]
            + [name for name in self.reads if name != self.accumulator_node_name]
        )
        return f"def {fn_name}({arguments}):"

    def _render_return_statement(self) -> str:
        return_vars = OrderedSet(
            op_v.value for op_v in self.store_name_to_value.values()
        )
        if "D" not in return_vars:
            raise AssertionError(f"expected 'D' in return_vars, got {return_vars}")
        return f"return {', '.join(return_vars)}"

    def _tmp_var(self) -> str:
        return f"tmp_{next(self.var_counter)}"
