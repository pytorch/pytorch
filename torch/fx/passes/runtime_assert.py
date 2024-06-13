# mypy: allow-untyped-defs
import contextlib
import logging
import operator
import sys
import sympy
from collections import defaultdict
from typing import Any, Dict, Optional, Set, TYPE_CHECKING

# Import sympy and ShapeEnv during TYPE_CHECKING since importing sympy is slow
if TYPE_CHECKING:
    from torch.fx.experimental.symbolic_shapes import ShapeEnv
else:
    ShapeEnv = Any

import torch
import torch.utils._pytree as pytree
from torch import fx
from torch.fx._compatibility import compatibility
from torch.fx._utils import lazy_format_graph_code
from torch.fx.experimental.proxy_tensor import py_sym_types
from torch.fx.experimental.symbolic_shapes import RuntimeAssert
from torch.fx.experimental.sym_node import SymNode
from torch.fx.graph_module import GraphModule
from torch.fx.passes.utils.hash_cons import SymExprRange, SymRel
from torch.utils._sympy.interp import sympy_interp
from torch.utils._sympy.numbers import int_oo
from torch.utils._sympy.reference import PythonReferenceAnalysis

log = logging.getLogger(__name__)
graph_code_log = torch._logging.getArtifactLogger(__name__, "graph_code")


def _get_example_value(node: fx.Node) -> Optional[str]:
    """
    Get the example value key for a node, since dynamo uses "example_value"
    while non-strict export uses "val.
    """
    if "example_value" in node.meta:
        val = node.meta["example_value"]
    elif "val" in node.meta:
        val = node.meta["val"]
    else:
        return None
    return val


def _get_sym_val(node: fx.Node) -> "Optional[sympy.Expr]":
    if (val := _get_example_value(node)) is not None:
        # dynamo has py_sym_type, export uses sympy.Expr
        if isinstance(val, py_sym_types):
            return val.node.expr
        elif isinstance(val, sympy.Expr):
            return val
    return None


def _get_sym_rel(expr):
    if isinstance(expr, sympy.LessThan):
        return SymRel.LE
    if isinstance(expr, sympy.StrictLessThan):
        return SymRel.LT
    elif isinstance(expr, sympy.GreaterThan):
        return SymRel.GE
    elif isinstance(expr, sympy.StrictGreaterThan):
        return SymRel.GT
    elif isinstance(expr, sympy.Equality):
        return SymRel.EQ
    elif isinstance(expr, sympy.Unequality):
        return SymRel.NE
    return None


def _sym_rel_to_sympy(rel, lhs, rhs):
    if rel == SymRel.EQ:
        return sympy.Equality(lhs, rhs)
    elif rel == SymRel.NE:
        return sympy.Unequality(lhs, rhs)
    if rel == SymRel.LE:
        return sympy.LessThan(lhs, rhs)
    elif rel == SymRel.LT:
        return sympy.StrictLessThan(lhs, rhs)
    elif rel == SymRel.GE:
        return sympy.GreaterThan(lhs, rhs)
    elif rel == SymRel.GT:
        return sympy.StrictGreaterThan(lhs, rhs)


def _maybe_negate_expr(expr, mapping):
    if expr is None or len(expr.args) != 2:
        return expr

    lhs, rhs = expr.args
    to_negate = -lhs in mapping
    if not to_negate:
        return expr

    if isinstance(expr, sympy.LessThan):
        return sympy.GreaterThan(-lhs, -rhs)
    if isinstance(expr, sympy.StrictLessThan):
        return sympy.StrictGreaterThan(-lhs, -rhs)
    elif isinstance(expr, sympy.GreaterThan):
        return sympy.LessThan(-lhs, -rhs)
    elif isinstance(expr, sympy.StrictGreaterThan):
        return sympy.StrictLessThan(-lhs, -rhs)
    elif isinstance(expr, sympy.Equality):
        return sympy.Equality(-lhs, -rhs)
    elif isinstance(expr, sympy.Unequality):
        return sympy.Unequality(-lhs, -rhs)

    return expr


def _get_last_placeholder(graph: fx.Graph) -> fx.Node:
    # find location of last placeholder node, as insert point
    # if not found return first node
    res = None
    for node in graph.nodes:
        if node.op == "placeholder":
            res = node
        else:
            break
    if res is None:
        return next(iter(graph.nodes))
    return res


def _is_size_function_node(node: fx.Node) -> bool:
    return (
        node.op == "call_function"
        and node.target in [torch.ops.aten.sym_size.int, "size"]
    )


def dce_from_node(node: fx.Node):
    # DCE unused node and any unused ancestors
    graph = node.graph
    _args = [node]
    while _args:
        node = _args.pop()
        inputs = node.args
        graph.erase_node(node)
        _args.extend(
            n for n in inputs
            if (
                isinstance(n, fx.Node)
                and not n.users
                and n.op != "placeholder"
            )
        )


@compatibility(is_backward_compatible=True)
def insert_deferred_runtime_asserts(
    gm: GraphModule,
    shape_env: ShapeEnv,
    name: str,
    export: bool = False,
) -> None:

    import sympy
    from torch.fx.experimental.symbolic_shapes import (
        CallMethodKey,
        cast_symbool_to_symint_guardless,
        ConvertIntKey,
        DivideByKey,
        free_symbols,
        InnerTensorKey,
    )

    ops_on_sym_types = [
        torch.ops.aten._assert_async.msg,
        torch.ops.aten._assert_scalar.default,
        torch.ops.aten.sym_constrain_range_for_size.default,
        torch._check,
        torch._check_is_size,
        torch.ops.aten.sym_constrain_range.default,
    ]

    # first pass: collect all range-related assertions
    # track ranges for expressions
    sym_expr_to_ranges: Dict[sympy.Expr, SymExprRange] = defaultdict(SymExprRange)
    sym_expr_to_nodes: Dict[sympy.Expr, Dict[Any, List[fx.Node]]] = defaultdict(lambda: defaultdict(list))

    def _refine_range(sym_rel, lhs, rhs):
        if isinstance(rhs, sympy.Integer):
            rhs = int(rhs)
        # enumerate
        if sym_rel == SymRel.EQ:
            sym_expr_to_ranges[lhs].eq(rhs)
        elif sym_rel == SymRel.NE:
            sym_expr_to_ranges[lhs].ne(rhs)
        elif sym_rel == SymRel.LT:
            sym_expr_to_ranges[lhs].lt(rhs, True)
        elif sym_rel == SymRel.LE:
            sym_expr_to_ranges[lhs].lt(rhs, False)
        elif sym_rel == SymRel.GT:
            sym_expr_to_ranges[lhs].gt(rhs, True)
        elif sym_rel == SymRel.GE:
            sym_expr_to_ranges[lhs].gt(rhs, False)

    for node in gm.graph.nodes:  # maybe we should sort and order check_is_size, for_size nodes first
        if (
            node.op == "call_function"
            and node.target in ops_on_sym_types
        ):
            # get expression, maybe negate
            if node.target == torch.ops.aten._assert_async.msg:
                expr_node = node.args[0].args[0]
            else:
                expr_node = node.args[0]
            is_size = node.target in (
                torch.ops.aten.sym_constrain_range_for_size.default,
                torch._check_is_size,
            )
            expr = _get_sym_val(expr_node)
            if not is_size:
                expr = _maybe_negate_expr(expr, sym_expr_to_ranges)

            # get signature, refine ranges
            if node.target in (
                torch.ops.aten._assert_scalar.default,
                torch._check,
                torch.ops.aten._assert_async.msg,
            ):
                if (sym_rel := _get_sym_rel(expr)) is None:
                    continue
                sym_expr, val = expr.args
                signature = (node.target, sym_rel, val)
                if not isinstance(val, int):
                    # expect shape_env runtime asserts to handle things like s0 < u0
                    continue
                _refine_range(sym_rel, sym_expr, val)

            elif node.target in (
                torch.ops.aten.sym_constrain_range_for_size.default,
                torch.ops.aten.sym_constrain_range.default,
            ):
                _min = node.kwargs.get("min", 0 if node.target == is_size else -int_oo)
                _max = node.kwargs.get("max", int_oo)
                signature = (node.target, _min, _max)
                _refine_range(SymRel.GE, expr, _min)
                _refine_range(SymRel.LE, expr, _max)
                sym_expr = expr

            else:  # torch._check_is_size
                signature = (node.target,)
                sym_expr = expr

            if is_size:  # make size-like
                sym_expr_to_ranges[expr].set_is_size()
            sym_expr_to_nodes[sym_expr][signature].append(node)

        # make size-like for symbols with unbacked bindings
        if unbacked_bindings := node.meta.get("unbacked_bindings"):
            for symbol in unbacked_bindings.keys():
                if symbol in shape_env.size_like:
                    sym_expr_to_ranges[symbol].set_is_size()
                if symbol in shape_env.var_to_range:
                    vr = shape_env.var_to_range[symbol]
                    if not shape_env._default_unspecified_value_range().issubset(vr):
                        # The runtime range is constrained, so add a runtime
                        # assert and also explicitly refine the range
                        # (refinement should not be necessary once runtime
                        # asserts cause refinement, but that's NYI)
                        def convert(s):
                            if s in (int_oo, -int_oo):
                                return None
                            try:
                                return int(s)
                            except TypeError:
                                return None
                        _refine_range(SymRel.GE, symbol, convert(vr.lower))
                        _refine_range(SymRel.LE, symbol, convert(vr.upper))

    # look at exprs in runtime asserts, track ranges
    for ras in shape_env.deferred_runtime_asserts.copy().values():
        ra_exprs = set(ra.expr for ra in ras)
        for ra_expr in ra_exprs:
            expr = _maybe_negate_expr(ra_expr, sym_expr_to_ranges)
            if (sym_rel := _get_sym_rel(expr)) is None:
                continue  # don't track anything for this expr
            # only refine range, no node to hash cons
            _refine_range(sym_rel, *expr.args)

    # hash consing, track created size/stride nodes, insert point, tracked symbols
    hash_cons: Dict[sympy.Expr, fx.Proxy] = {}
    created_placeholder_ops = []
    last_placeholder = _get_last_placeholder(gm.graph)
    tracked_symbols: Set[sympy.Expr] = set()

    # main pass
    graph = gm.graph
    for node in list(gm.graph.nodes)[:-1]:

        # for placeholders, create all size/stride nodes, and delete later
        if (
            node.op == "placeholder"
            and (example_value := _get_example_value(node)) is not None
        ):
            def match_symbol(symint, cb):
                if (
                    isinstance(symint, torch.SymInt)
                    and isinstance(symint.node, SymNode)
                    and isinstance(s := symint.node.expr, sympy.Symbol)
                    and not s in hash_cons
                ):
                    with gm.graph.inserting_before(
                        last_placeholder.next if last_placeholder.op == "placeholder" else last_placeholder
                    ):
                        res = fx.Proxy(cb())
                        hash_cons[s] = res
                        tracked_symbols.add(s)
                        created_placeholder_ops.append(res)
                        log.debug("symbol_to_proxy[%s] = %s", s, res)

            match_symbol(example_value, lambda: node)
            if isinstance(t := example_value, torch.Tensor):
                for i, s in enumerate(t.size()):
                    match_symbol(
                        s,
                        lambda: graph.call_function(
                            torch.ops.aten.sym_size.int, (node, i)
                        ),
                    )
                for i, s in enumerate(t.stride()):
                    match_symbol(
                        s,
                        lambda: graph.call_function(
                            torch.ops.aten.sym_stride.int, (node, i)
                        ),
                    )
                match_symbol(
                    t.storage_offset(),
                    lambda: graph.call_function(
                        torch.ops.aten.sym_storage_offset.default, (node,)
                    ),
                )

        # if call_function node produces a symbolic value, hash cons
        if (
            node.op == "call_function"
            and (sym_val := _get_sym_val(node)) is not None
        ):
            if sym_val in hash_cons:
                hash_node = hash_cons[sym_val].node
                node.replace_all_uses_with(hash_node)
                node.graph.erase_node(node)
                continue
                # I don't think we need to run DCE here...
                # this would only be needed if the graph has multiple ways
                # of constructing the same expr for some reason
                # e.g. x + y + z -> (x + y) + z, or x + (y + z)
                # -> we would DCE some inputs for the newly-hashed branch
                # but I don't think this is needed for us
            elif not _is_size_function_node(node):
                # we don't want size calls on intermediate tensors, so skip this
                '''
                maybe here be careful about data-dep sizes...
                '''
                hash_cons[sym_val] = fx.Proxy(node)

        # if size call on intermediate tensor, convert to compute on input sizes
        # e.g. y = torch.cat([x, x], dim=1), z = y.shape[0]
        # convert from: z = sym_size.int(y, 0)
        # to: x0 = sym_size.int(x, 0), z = x0 * 2
        # so we can clean up intermediate tensors from memory faster
        if (
            _is_size_function_node(node)
            and node.args[0].op != "placeholder"
            and (sym_shape := _get_sym_val(node)) is not None
        ):
            # if this sym expr doesn't exist, or it does exist and it's a size call,
            # then we create/overwrite by calling sympy_interp()
            if (
                sym_shape not in hash_cons
                or _is_size_function_node(hash_cons[sym_shape].node)
            ):
                hash_cons[sym_shape] = sympy_interp(
                    PythonReferenceAnalysis, hash_cons, sym_shape, hash_cons=hash_cons,
                )
            hash_node = hash_cons[sym_shape].node
            node.replace_all_uses_with(hash_node)
            dce_from_node(node)  # this replace call can create dead code

        # look at unbacked bindings, hash cons ops, update tracked symbols
        if unbacked_bindings := node.meta.get("unbacked_bindings"):
            for symbol, keypath in unbacked_bindings.items():

                with graph.inserting_before(node.next):
                    # hash cons / CSE later
                    def go(node, keypath):
                        if keypath == ():
                            return node
                        if (
                            len(keypath) >= 2
                            and isinstance(keypath[0], CallMethodKey)
                            and isinstance(keypath[1], pytree.SequenceKey)
                        ):
                            if keypath[0].name == "size":
                                return go(
                                    graph.call_function(
                                        torch.ops.aten.sym_size.int,
                                        (node, keypath[1].idx),
                                    ),
                                    keypath[2:],
                                )
                            if keypath[0].name == "stride":
                                return go(
                                    graph.call_function(
                                        torch.ops.aten.stride.int,
                                        (node, keypath[1].idx),
                                    ),
                                    keypath[2:],
                                )
                            return go(
                                graph.call_method(
                                    keypath[0].name, (node, keypath[1].idx)
                                ),
                                keypath[2:],
                            )
                        elif isinstance(keypath[0], CallMethodKey):
                            return go(
                                graph.call_method(keypath[0].name, (node,)), keypath[1:]
                            )
                        elif isinstance(keypath[0], pytree.SequenceKey):
                            return go(
                                graph.call_function(
                                    operator.getitem, (node, keypath[0].idx)
                                ),
                                keypath[1:],
                            )
                        elif isinstance(keypath[0], ConvertIntKey):
                            return go(
                                graph.call_function(
                                    cast_symbool_to_symint_guardless, (node,)
                                ),
                                keypath[1:],
                            )
                        elif isinstance(keypath[0], DivideByKey):
                            # TODO: need to assert divisibility
                            return go(
                                graph.call_function(
                                    operator.floordiv, (node, keypath[0].divisor)
                                ),
                                keypath[1:],
                            )
                        elif isinstance(keypath[0], InnerTensorKey):
                            return go(
                                graph.call_function(
                                    getattr, (node, keypath[0].inner_name)
                                ),
                                keypath[1:],
                            )
                        else:
                            raise AssertionError(f"unrecognized keypath {keypath}")

                    res = fx.Proxy(go(node, keypath))
                    sym_expr = _get_sym_val(res.node) if res.node.meta else symbol
                    hash_cons[sym_expr] = res
                    tracked_symbols.update(sym_expr.free_symbols)
                    log.debug("symbol_to_proxy[%s] = %s", sym_expr, hash_cons[sym_expr])

    def add_runtime_assert(node, sym_expr, op, partial_signature):
        signature = (op,) + partial_signature

        if op == torch.ops.aten._assert_scalar.default:
            if signature in sym_expr_to_nodes[sym_expr]:
                res = sym_expr_to_nodes[sym_expr][signature][0]
            elif (torch._check,) + partial_signature in sym_expr_to_nodes[sym_expr]:
                # allow torch._check in place of _assert_scalar if it exists
                res = sym_expr_to_nodes[sym_expr][(torch._check,) + partial_signature][0]
            else:
                sym_rel, rhs = partial_signature
                full_expr = _sym_rel_to_sympy(sym_rel, sym_expr, rhs)
                # be careful about insert point if node already exists,
                # insert after input if it doesn't, otherwise rely on sympy_interp's logic
                with (
                    contextlib.nullcontext()
                    if full_expr in hash_cons
                    else node.graph.inserting_before(node.next)
                ):
                    res0 = sympy_interp(
                        PythonReferenceAnalysis,
                        hash_cons,
                        full_expr,
                        hash_cons=hash_cons,
                    ).node
                with node.graph.inserting_before(res0.next):
                    msg = f"Runtime assertion failed for expression {full_expr} on node '{res0}'"
                    res = node.graph.call_function(op, (res0, msg))

        elif op in (
            torch.ops.aten.sym_constrain_range_for_size.default,
            torch.ops.aten.sym_constrain_range.default,
        ):
            is_size = (op == torch.ops.aten.sym_constrain_range_for_size.default)
            _min, _max = partial_signature
            if signature in sym_expr_to_nodes[sym_expr]:
                res = sym_expr_to_nodes[sym_expr][signature][0]
            else:
                _kwargs = {}
                if not is_size or _min != 0:
                    _kwargs["min"] = _min
                if not is_size or _max != int_oo:
                    _kwargs["max"] = _max
                with node.graph.inserting_before(node.next):
                    res = node.graph.call_function(op, (node,), _kwargs)
        
        return res

    # go through runtime asserts, add appropriate ops
    for sym_expr, sym_ranges in sym_expr_to_ranges.items():

        # check that all symbols exist. if so, call sympy_interp()
        free_symbols = sym_expr.free_symbols
        if len(free_symbols - tracked_symbols) == 0:
            res = sympy_interp(
                PythonReferenceAnalysis, hash_cons, sym_expr, hash_cons=hash_cons,
            ).node

            # track nodes we keep or add
            remaining_assert_nodes: Set[fx.Node] = set()
            _min, is_gt = sym_ranges.min
            _max, is_lt = sym_ranges.max
            
            if sym_ranges.static:
                val = sym_ranges.val
                if sym_ranges.is_size:  # sym_constrain_range_for_size
                    if _max > 2:
                        remaining_assert_nodes.add(
                            add_runtime_assert(
                                res,
                                sym_expr,
                                torch.ops.aten.sym_constrain_range_for_size.default,
                                (val, val),
                            )
                        )
                    else:
                        remaining_assert_nodes.add(
                            add_runtime_assert(
                                res,
                                sym_expr,
                                torch.ops.aten.sym_constrain_range_for_size.default,
                                (_min, int_oo),
                            )
                        )
                        remaining_assert_nodes.add(
                            add_runtime_assert(
                                res,
                                sym_expr,
                                torch.ops.aten._assert_scalar.default,
                                (SymRel.LE, _max),
                            )
                        )
                else:  # _assert_scalar
                    remaining_assert_nodes.add(
                        add_runtime_assert(
                            res,
                            sym_expr,
                            torch.ops.aten._assert_scalar.default,
                            (SymRel.EQ, val),
                        )
                    )
            elif _min is not None and _max is not None:
                '''
                refactor this to try to constrain range
                if max > 2 or not is size, do normal constrain + check
                if non-symbol, torch.check
                '''
                if sym_ranges.is_size:  # sym_constrain_range_for_size
                    if _max > 2:
                        remaining_assert_nodes.add(
                            add_runtime_assert(
                                res,
                                sym_expr,
                                torch.ops.aten.sym_constrain_range_for_size.default,
                                (_min, _max),
                            )
                        )
                    else:
                        remaining_assert_nodes.add(
                            add_runtime_assert(
                                res,
                                sym_expr,
                                torch.ops.aten.sym_constrain_range_for_size.default,
                                (_min, int_oo),
                            )
                        )
                        remaining_assert_nodes.add(
                            add_runtime_assert(
                                res,
                                sym_expr,
                                torch.ops.aten._assert_scalar.default,
                                (SymRel.LE, _max),
                            )
                        )
                elif isinstance(sym_expr, sympy.Symbol):
                    remaining_assert_nodes.add(
                        add_runtime_assert(
                            res,
                            sym_expr,
                            torch.ops.aten.sym_constrain_range.default,
                            (_min, _max),
                        )
                    )
                else:
                    remaining_assert_nodes.add(
                        add_runtime_assert(
                            res,
                            sym_expr,
                            torch.ops.aten._assert_scalar.default,
                            (SymRel.GT if is_gt else SymRel.GE, _min),
                        )
                    )
                    remaining_assert_nodes.add(
                        add_runtime_assert(
                            res,
                            sym_expr,
                            torch.ops.aten._assert_scalar.default,
                            (SymRel.LT if is_lt else SymRel.LE, _max),
                        )
                    )

            elif _min is not None:
                remaining_assert_nodes.add(
                    add_runtime_assert(
                        res,
                        sym_expr,
                        torch.ops.aten._assert_scalar.default,
                        (SymRel.GT if is_gt else SymRel.GE, _min),
                    )
                )

            elif _max is not None:
                remaining_assert_nodes.add(
                    add_runtime_assert(
                        res,
                        sym_expr,
                        torch.ops.aten._assert_scalar.default,
                        (SymRel.LT if is_lt else SymRel.LE, _max),
                    )
                )

            for val in sym_ranges.not_equals:
                remaining_assert_nodes.add(
                    add_runtime_assert(
                        res,
                        sym_expr,
                        torch.ops.aten._assert_scalar.default,
                        (SymRel.NE, val),
                    )
                )

        # clean out remaining nodes
        for nodes in sym_expr_to_nodes[sym_expr].values():
            for node in nodes:
                if node not in remaining_assert_nodes:
                    dce_from_node(node)

    breakpoint()
