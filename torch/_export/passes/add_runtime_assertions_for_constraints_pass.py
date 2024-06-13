# mypy: allow-untyped-defs
import math
import operator
import traceback
from functools import partial
from typing import Callable, Dict, List, NamedTuple, Set

import sympy

import torch
import torch.fx
from torch.utils._sympy.value_ranges import ValueRanges
from torch.fx.experimental.symbolic_shapes import free_unbacked_symbols
from torch.fx.passes.infra.pass_base import PassBase, PassResult

__all__ = ["InputDim"]


class InputDim(NamedTuple):
    input_name: str
    dim: int


def _convert_to_int(val):
    # Convert simple sympy Integers into concrete int
    if val == sympy.oo:
        return math.inf
    if val == -sympy.oo:
        return -math.inf
    if isinstance(val, sympy.Integer):
        return int(val)
    raise RuntimeError(
        "Export constraints cannot be non-integer expressions"
    )


def _convert_range_to_int(range: ValueRanges):
    assert isinstance(range, ValueRanges)
    min_val = _convert_to_int(range.lower)
    max_val = _convert_to_int(range.upper)
    return min_val, max_val


class _AddRuntimeAssertionsForInlineConstraintsPass(PassBase):
    def __init__(
        self,
        range_constraints: Dict[sympy.Symbol, ValueRanges],
    ):
        super().__init__()
        self.range_constraints: Dict[sympy.Symbol, ValueRanges] = range_constraints
        self._asserts_generated_unbacked_symbols: Set[sympy.Symbol] = set()
        self.counter = 0

    def _assert_range_constraint(self, node, lower, upper, assert_msg):
        last_node = node
        if lower > -math.inf:
            last_node = self._insert_assert_async(last_node, operator.ge, node, lower, assert_msg)

        if upper < math.inf:
            last_node = self._insert_assert_async(last_node, operator.le, node, upper, assert_msg)

    def _insert_assert_async(self, last_node, op, lower, upper, assert_msg):
        """
        Inserts assert_async call_function nodes in the graph. This function is
        called **during** the interpreter-based pass.
        """
        self.counter += 1
        graph = last_node.graph
        with graph.inserting_after(last_node):
            cmp = graph.call_function(op, (lower, upper), {})
        with graph.inserting_after(cmp):
            cmp_tensor = graph.call_function(torch.ops.aten.scalar_tensor.default, (cmp,), {})
        with graph.inserting_after(cmp_tensor):
            assert_async = graph.call_function(
                torch.ops.aten._assert_async.msg,
                (cmp_tensor, assert_msg),
                {},
            )
        return assert_async

    def call(self, graph_module) -> PassResult:
        self.existing_inline_assertions = _get_existing_inline_assertions(
            graph_module, self.range_constraints
        )

        for module in graph_module.modules():
            if not isinstance(module, torch.fx.GraphModule):
                continue
            for node in module.graph.nodes:
                if node.op != "call_function":
                    continue
                if "val" not in node.meta:
                    continue

                val = node.meta["val"]
                # In general, we may have to deal the case such as: ret[1].shape[0].
                # We need first find out what symbols require assertion, then we need to follow the path
                # from ret to the symbol, construct the proxies along the way and construct the messages
                # piece-wise at the same time.
                #
                # We use post-order traversal to collect all the proxies callbacks needed, construct
                # the error message callbacks, and at the top-level traversal tree we execute all the callbacks.
                # We need the callbacks because, in order to call the function to create a proxy for shape[0], we
                # need the proxy for shape, which further requires the proxy for ret[1], etc.

                def add_assertions(val):
                    call_backs: List[Callable] = []
                    messages: List[str] = []
                    if isinstance(val, (torch.SymInt, torch.SymFloat, torch.SymBool)):
                        symbol = val.node.expr
                        if symbol in self.existing_inline_assertions:
                            return call_backs, messages
                        if isinstance(symbol, sympy.Symbol) and free_unbacked_symbols(symbol):
                            if symbol in self._asserts_generated_unbacked_symbols:
                                return call_backs, messages
                            # We only care about unbacked symints for these inline
                            # constraints, which are prefixed with 'u'
                            constraint = self.range_constraints[symbol]
                            min_val, max_val = _convert_range_to_int(constraint)
                            assert_msg = f" is outside of inline constraint [{min_val}, {max_val}]."
                            call_backs.append(
                                partial(self._assert_range_constraint, lower=min_val, upper=max_val)
                            )
                            messages.append(assert_msg)
                            self._asserts_generated_unbacked_symbols.add(symbol)

                    elif isinstance(val, torch.Tensor):
                        for i, sym in enumerate(val.shape):
                            cbs, msgs = add_assertions(sym)
                            for cb, msg in zip(cbs, msgs):
                                def sym_size_cb(node, assert_msg, dim):
                                    with node.graph.inserting_after(node):
                                        dim_node = module.graph.call_function(
                                            torch.ops.aten.sym_size.int,
                                            (node, dim),
                                            {},
                                        )
                                    cb(node=dim_node, assert_msg=assert_msg)
                                call_backs.append(partial(sym_size_cb, dim=i))
                                messages.append(f".shape[{i}]" + msg)
                    return call_backs, messages

                callbacks, messages = add_assertions(val)
                for cb, msg in zip(callbacks, messages):
                    cb(node=node, assert_msg=f"{node}" + msg)

            module.recompile()

        # Sometimes this pass would return a wrong graph where we have mismatched
        # node names in signature. Before we fix it, let's just skip it.
        if self.counter == 0 and type(self) is _AddRuntimeAssertionsForInlineConstraintsPass:
            return PassResult(graph_module, False)

        # Populate the stack trace with dummy vals to respect IR
        for node in graph_module.graph.nodes:
            if not node.meta.get("stack_trace", None) and node.op not in ["placeholder", "output"]:
                node.meta["stack_trace"] = "".join(traceback.format_stack(limit=1))
        return PassResult(graph_module, True)


def _get_existing_inline_assertions(
    graph_module: torch.fx.GraphModule,
    range_constraints: Dict[sympy.Symbol, ValueRanges],
) -> Dict[sympy.Symbol, ValueRanges]:
    existing_inline_assertions: Dict[sympy.Symbol, ValueRanges] = {}

    for module in graph_module.modules():
        if not isinstance(module, torch.fx.GraphModule):
            continue

        # Find all the existing inline assertions. They will look something like:
        # %_local_scalar_dense = call_function[target=torch.ops.aten._local_scalar_dense.default](args = (%arg1_1,), kwargs = {})
        # %ge = call_function[target=operator.ge](args = (%_local_scalar_dense, 0), kwargs = {})
        # %_assert_scalar = call_function[target=torch.ops.aten._assert_scalar.default](args = (%scalar_tensor, "..."), kwargs = {})
        for node in module.graph.nodes:
            if node.target != torch.ops.aten._assert_scalar.default:
                continue

            compare_arg = node.args[0]
            if not (
                isinstance(compare_arg, torch.fx.Node) and
                compare_arg.op == "call_function" and
                compare_arg.target in (operator.le, operator.ge) and
                len(compare_arg.args) == 2
            ):
                continue

            compare_op = compare_arg.target
            maybe_symint_arg, compare_int = compare_arg.args

            # x >= 0 will sometimes be canonicalized to -x <= 0, so in some
            # cases the operation before the comparison is to multiply by -1. We
            # can undo the canonicalization here
            if (
                maybe_symint_arg.op == "call_function" and
                maybe_symint_arg.target == operator.mul and
                maybe_symint_arg.args[0] == -1
            ):
                maybe_symint_arg = maybe_symint_arg.args[1]
                compare_op = operator.ge
                compare_int = -1 * compare_int

                if not (
                    "val" in maybe_symint_arg.meta and
                    isinstance(maybe_symint_arg.meta["val"], torch.SymInt)
                ):
                    continue

                symint = maybe_symint_arg.meta["val"].node.expr
                symint = -1 * symint

            else:
                if not (
                    "val" in maybe_symint_arg.meta and
                    isinstance(maybe_symint_arg.meta["val"], torch.SymInt)
                ):
                    continue

                symint = maybe_symint_arg.meta["val"].node.expr

            if not isinstance(symint, sympy.Symbol):
                continue

            if symint not in range_constraints:
                raise RuntimeError(f"Unable to find symint {symint} in {range_constraints}")

            found_range = existing_inline_assertions.get(symint, ValueRanges(-math.inf, math.inf))

            if compare_arg.target == operator.le:
                existing_inline_assertions[symint] = ValueRanges(
                    lower=found_range.lower, upper=compare_int
                )
            elif compare_arg.target == operator.ge:
                existing_inline_assertions[symint] = ValueRanges(
                    lower=compare_int, upper=found_range.upper
                )

    return existing_inline_assertions
