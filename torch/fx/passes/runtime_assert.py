# mypy: allow-untyped-defs
import contextlib
import logging
import operator
import sys
import sympy
from collections import defaultdict, OrderedDict
from typing import Any, Dict, List, Optional, Set, TYPE_CHECKING

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
from torch.utils._sympy.expr_ranges import SymExprRange, SymRel
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
        return node.meta["example_value"]
    elif "val" in node.meta:
        return node.meta["val"]
    else:
        return None


def _get_sym_val(node: fx.Node) -> "Optional[sympy.Expr]":
    """
    Returns sympy.Expr from node example value
    """
    if (val := _get_example_value(node)) is not None:
        # dynamo has py_sym_type, export uses sympy.Expr
        if isinstance(val, py_sym_types):
            return val.node.expr
        elif isinstance(val, sympy.Expr):
            return val
    return None


def _get_sym_rel(expr):
    """
    Enumerates supported relations
    """
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
    """
    Sort of inverse of _get_sym_rel()
    """
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


def _canonicalize_expr(expr, mapping):
    """
    https://github.com/pytorch/pytorch/pull/128411 means we won't get expressions like -u0 <= 0
    for calls like torch._check(-u0 <= 0) anymore, but it does seem like we can get 4 <= u0, or u0 < s0,
    which are slightly difficult to handle when we want to track ranges.
    Here we move all non-numeric terms to the LHS, and maybe negate the expression if is_size = False,
    in case we already track the negated value.
    """
    if expr is None or len(expr.args) != 2:
        return expr
    # assume we won't get something like u0 + 2 < 4, so don't call this
    # expr = sympy.simplify(expr)

    # if straightforward we return
    if isinstance(expr.args[1], sympy.Number):
        return expr

    # try flipping, won't hurt
    # can return if original expr is something like 4 <= s0
    lhs, rhs = expr.args
    if isinstance(expr, sympy.LessThan):
        expr = sympy.GreaterThan(rhs, lhs)
    elif isinstance(expr, sympy.StrictLessThan):
        expr = sympy.StrictGreaterThan(rhs, lhs)
    elif isinstance(expr, sympy.GreaterThan):
        expr = sympy.LessThan(rhs, lhs)
    elif isinstance(expr, sympy.StrictGreaterThan):
        expr = sympy.StrictLessThan(rhs, lhs)
    elif isinstance(expr, sympy.Equality):
        expr = sympy.Equality(rhs, lhs)
    elif isinstance(expr, sympy.Unequality):
        expr = sympy.Unequality(rhs, lhs)
    if isinstance(expr.args[1], sympy.Number):
        return expr

    # move all non-numeric values to LHS
    eq = expr.args[0] - expr.args[1]
    non_numeric = tuple(x for x in eq.args if not isinstance(x, sympy.Number))
    numeric = tuple(x for x in eq.args if isinstance(x, sympy.Number))
    assert len(numeric) <= 1
    expr = type(expr)(sympy.Add(*non_numeric), -numeric[0] if numeric else 0)

    # maybe negate
    lhs, rhs = expr.args
    if -lhs in mapping:
        if isinstance(expr, sympy.LessThan):
            expr = sympy.GreaterThan(-lhs, -rhs)
        elif isinstance(expr, sympy.StrictLessThan):
            expr = sympy.StrictGreaterThan(-lhs, -rhs)
        elif isinstance(expr, sympy.GreaterThan):
            expr = sympy.LessThan(-lhs, -rhs)
        elif isinstance(expr, sympy.StrictGreaterThan):
            expr = sympy.StrictLessThan(-lhs, -rhs)
        elif isinstance(expr, sympy.Equality):
            expr = sympy.Equality(-lhs, -rhs)
        elif isinstance(expr, sympy.Unequality):
            expr = sympy.Unequality(-lhs, -rhs)

    return expr


def _get_last_placeholder(graph: fx.Graph) -> fx.Node:
    """
    find location of last placeholder node, as insert point.
    if not found return first node
    """
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


def _dce_from_node(node: fx.Node):
    """
    DCE unused node and any unused ancestors
    """
    graph = node.graph
    _args = [node]
    while _args:
        node = _args.pop()
        inputs = node.args
        log.debug("DCE node %s", node)
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
    """
    During tracing, we may have discovered that some data-dependent values
    had runtime assert on them; e.g., torch.empty(x.item()) induces a runtime
    that x.item() >= 0.  This asserts can happen unpredictably during fake
    tensor propagation, so we cannot conveniently insert them into the FX graph
    when they occur.  Instead, we accumulate them in the ShapeEnv, and in this
    pass insert them into the graph as proper tests.

    This pass also deduplicates size-related computation logically and computationally. Logically, redundant asserts
    (e.g. u0 > 0, u0 > 1) are reduced to equivalent calls, and computationally ops are CSE'd when inserted.
    Additionally, size calls on intermediate tensors are turned into compute on input sizes, allowing us to delete
    intermediate tensors from memory faster. For example, here we can even DCE the cat and repeat calls:

        z = torch.cat([x, x], dim=0)  # 2*s0
        w = z.repeat(y.shape[0])  # 2*s0*s1
        _w = w.shape[0]
        # something with _w, but not w ...

        # turns into ->
        s0 = x.shape[0]
        s1 = y.shape[0]
        _w0 = 2 * s0
        _w = _w0 * s1

    The algorithm works as follows:

    1. First pass through graph, collects current runtime asserts.
        Uses these, along with shape env's runtime asserts and stored ranges for unbacked symbols,
        and computes ranges for expressions.
    2. Main graph pass:
        a) performs CSE for existing and new size-compute nodes, calling sympy_interp(),
           as well as calls on unbacked bindings. Creates size/stride calls to produce shape/unbacked symbols.
        b) turns size calls on intermediate tensors into placeholder-size computation if possible
    3. Create runtime asserts based on collected range information for each expression.
    4. Deletes any unused existing runtime asserts or unused size/stride calls.

    Collecting range information for expressions seems to be a good way to handle runtime assert deduplication.
    If we can _canonicalize_expr() runtime asserts expressions, then asserts are refinements on the LHS expr.
    For example, a sequence of calls like:
    
        torch._check(2 < u0)
        torch._check(u0 >= 4)
        torch._check(u0 != 4)
        torch._check(10 >= u0)
        torch._check(u0 <= 12)
        torch._check_is_size(u0)

    implies u0 has range: (4, 10], and is_size=True.
    This allows us to just generate the following 2 assertions:

        sym_constrain_range_for_size(u0, min=4, max=10)
        _assert_scalar(u0 > 4)

    This information is tracked with SymExprRange objects for each LHS expression.
    """

    import sympy
    from torch.fx.experimental.symbolic_shapes import (
        CallMethodKey,
        cast_symbool_to_symint_guardless,
        ConvertIntKey,
        DivideByKey,
        free_symbols,
        InnerTensorKey,
    )

    RUNTIME_ASSERT_OPS = [
        torch.ops.aten._assert_async.msg,
        torch.ops.aten._assert_scalar.default,
        torch.ops.aten.sym_constrain_range_for_size.default,
        torch._check,
        torch._check_is_size,
        torch.ops.aten.sym_constrain_range.default,
    ]

    ras_by_symbol = shape_env.deferred_runtime_asserts.copy()

    # first pass: collect all range-related assertions
    # track ranges for expressions
    sym_expr_to_ranges: Dict[sympy.Expr, SymExprRange] = defaultdict(SymExprRange)
    sym_expr_to_nodes: Dict[sympy.Expr, Dict[Any, List[fx.Node]]] = defaultdict(lambda: defaultdict(list))

    def _refine_range(sym_rel, lhs, rhs):
        # for working with SymExprRange objects
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
        log.debug("_refine_range(%s, %s, %s)", sym_rel, lhs, rhs)

    graph_code_log.debug(
        "%s",
        lazy_format_graph_code(
            f"pre insert_deferred_runtime_asserts {name}", gm, colored=True
        ),
    )

    # filter out ops on sym types, prioritizing is_size calls first, to handle negative symbols
    # e.g. torch._check(u >= 0) might appear as -u <= 0.
    # because we try negating LHS exprs to see if we already track their ranges,
    # we might end up tracking -u0 instead of u0 if we see the torch._check() call first.
    # is_size calls won't negate their symbol inputs, so we prioritize reading those first.
    is_size_calls = []
    non_is_size_calls = []
    for node in gm.graph.nodes:
        if (
            node.op == "call_function"
            and node.target in RUNTIME_ASSERT_OPS
        ):
            if node.target in (
                torch.ops.aten.sym_constrain_range_for_size.default,
                torch._check_is_size,
                torch.ops.aten.sym_constrain_range.default,  # technically not an is_size call, but doesn't take non-symbols
            ):
                is_size_calls.append(node)
            else:
                non_is_size_calls.append(node)

        # while we're doing this, track ranges for symbols with unbacked bindings
        if unbacked_bindings := node.meta.get("unbacked_bindings"):
            # this below is kind of annoying, with dynamo the unbacked bindings keys match the symbols in the graph,
            # but AOTAutograd seems to produce new unbacked symbols (likely from running call_function() too many times),
            # so we need to read from the example value.
            # NOTE: not so experienced here -> can the unbacked value be a sympy.Expr instead of a single symbol?
            symbols = (
                _get_sym_val(node).free_symbols
                if node.meta and isinstance(_get_sym_val(node), sympy.Symbol)
                else unbacked_bindings.keys()
            )
            for symbol in symbols:
                # NOTE: comment/docstring motion from previous state of this file.
                #
                # Before we perform any asserts, first apply range
                # refinement.  This is important, because if we are going
                # to retrace the graph (and we typically are if we send
                # the graph to AOTAutograd), we need to make sure we apply
                # range refinement (ala _check_is_size) first, BEFORE we
                # run any of the asserts.  Otherwise, we may decide to
                # perform substitutions based on the asserts which we then
                # can't back out, because value ranges can only be applied
                # to asserts.)
                #
                # A perhaps better long term plan is to avoid this order
                # dependence by making it possible to refine ranges on
                # arbitrary expressions, not just symbols.  But it is not
                # so easy to make use of this information, see
                # https://twitter.com/ezyang/status/1745801370299482492
                # We actually made an attempt at this in
                # https://github.com/pytorch/pytorch/pull/119043
                # which didn't work.
                #
                # Another ideas for how to do this:
                # - Have bound_sympy be the source of truth of the ranges of any expression
                # - Cache intermediate results for every subexpression of bound_sympy
                # - This cache should be possible to edit to refine ranges
                #
                # One issue with this proposal is that if
                # we have a bound on 2x, we are not going to be able to
                # apply it for 4x.  Similarly, we may have bounds for an
                # equivalent expression that we are not applying because
                # it's not a perfect match (e.g. x < y vs y > x)".
                #
                # The first issue we already have it and it's impossible
                # to solve in general, so any implementation on a best
                # effort basis should do.
                #
                # The second issue is a preexisting one. It can be mitigated
                # with a normalisation algorithm. In general, it may also
                # be on a best effort basis, but since our grammar is not
                # terribly difficult, chances are we could even fully
                # normalise SymPy expressions... who knows.
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
                                return s
                            try:
                                return int(s)
                            except TypeError:
                                return None
                        _refine_range(SymRel.GE, symbol, convert(vr.lower))
                        _refine_range(SymRel.LE, symbol, convert(vr.upper))

    # now handle existing runtime asserts in prioritized order, looking at is_size calls first
    for node in is_size_calls + non_is_size_calls:
        if node.target == torch.ops.aten._assert_async.msg:
            expr_node = node.args[0].args[0]
        else:
            expr_node = node.args[0]

        # trivial (simplified by dynamo if static), skip
        if expr_node == True:
            continue

        log.debug("read runtime assert %s for node %s", node.target, node)

        # maybe flip/negate expression
        is_size = node.target in (
            torch.ops.aten.sym_constrain_range_for_size.default,
            torch._check_is_size,
        )
        expr = _get_sym_val(expr_node)
        if not is_size:
            old_expr = expr
            expr = _canonicalize_expr(expr, sym_expr_to_ranges)
            log.debug("canonicalize_expr(%s) -> %s", old_expr, expr)

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
            # maybe we get a torch._check(u0 < s0), we expect this to show up in shape_env.deferred_runtime_asserts
            # and handle it there instead.
            if not isinstance(val, sympy.Number):
                continue
            _refine_range(sym_rel, sym_expr, val)

        elif node.target in (
            torch.ops.aten.sym_constrain_range_for_size.default,
            torch.ops.aten.sym_constrain_range.default,
        ):  # refine bounds, maybe mark size-like
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

        # store nodes for each expr
        sym_expr_to_nodes[sym_expr][signature].append(node)

    # look at exprs in runtime asserts, track ranges
    for ras in ras_by_symbol.values():
        ra_exprs = set(ra.expr for ra in ras)
        for ra_expr in ra_exprs:
            log.debug("new runtime assert %s", ra_expr)
            expr = _canonicalize_expr(ra_expr, sym_expr_to_ranges)
            log.debug("canonicalize_expr(%s) -> %s", ra_expr, expr)
            if (sym_rel := _get_sym_rel(expr)) is None:
                continue  # don't know what to do with this expr type, don't track anything for this expr
            # only refine range, no node to CSE yet
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
                    ):  # insert right after placeholders
                        res = fx.Proxy(cb())
                        hash_cons[s] = res
                        tracked_symbols.add(s)
                        if res.node.op == "call_function":  # don't count as deleteable op if placeholder itself is sym type
                            created_placeholder_ops.append(res.node)
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
                log.debug("CSEd node %s -> %s for expr %s", node, hash_node, sym_val)
                continue
                # I don't think we need to run DCE here...
                # this would only be needed if the graph has multiple ways
                # of constructing the same expr for some reason
                # e.g. x + y + z -> (x + y) + z, or x + (y + z)
                # -> we would DCE some inputs for the newly-hashed branch
                # but I don't think this is needed for us
            elif not _is_size_function_node(node):
                # we don't want size calls on intermediate tensors, so skip this
                hash_cons[sym_val] = fx.Proxy(node)

        # if size call on intermediate tensor, convert to compute on input sizes
        # e.g. y = torch.cat([x, x], dim=1), z = y.shape[0]
        # convert from: z = sym_size.int(y, 0)
        # to: x0 = sym_size.int(x, 0), z = x0 * 2
        # so we can clean up intermediate tensors from memory earlier.
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
                    PythonReferenceAnalysis, hash_cons, sym_shape, hash_cons=hash_cons, insert_after_args=True,
                )
            hash_node = hash_cons[sym_shape].node
            node.replace_all_uses_with(hash_node)
            log.debug("CSEd intermediate size node for expr %s", sym_shape)
            _dce_from_node(node)  # here we should DCE

        # look at unbacked bindings, hash cons ops that create them, update tracked symbols
        if unbacked_bindings := node.meta.get("unbacked_bindings"):
            for symbol, keypath in unbacked_bindings.items():
                with graph.inserting_before(node.next):  # insert right after this node
                    def unbacked_interp(node, keypath, signature):
                        if keypath == ():
                            return node
                        if signature in hash_cons:  # CSE
                            hash_node = hash_cons[signature].node
                            log.debug("CSE unbacked_bindings node for keypath %s on node %s", keypath, node)
                            return hash_node
                        if (
                            len(keypath) >= 2
                            and isinstance(keypath[0], CallMethodKey)
                            and isinstance(keypath[1], pytree.SequenceKey)
                        ):
                            signature = (("CallMethod", keypath[0].name, keypath[1].idx), signature)
                            if signature in hash_cons:
                                res = hash_cons[signature].node
                            else:
                                if keypath[0].name == "size":
                                    res = unbacked_interp(
                                        graph.call_function(
                                            torch.ops.aten.sym_size.int,
                                            (node, keypath[1].idx),
                                        ),
                                        keypath[2:],
                                        signature,
                                    )
                                elif keypath[0].name == "stride":
                                    res = unbacked_interp(
                                        graph.call_function(
                                            torch.ops.aten.stride.int,
                                            (node, keypath[1].idx),
                                        ),
                                        keypath[2:],
                                        signature,
                                    )
                                else:
                                    res = unbacked_interp(
                                        graph.call_method(
                                            keypath[0].name, (node, keypath[1].idx)
                                        ),
                                        keypath[2:],
                                        signature,
                                    )
                        elif isinstance(keypath[0], CallMethodKey):
                            signature = (("CallMethod", keypath[0].name), signature)
                            if signature in hash_cons:
                                res = hash_cons[signature].node
                            else:
                                res = unbacked_interp(
                                    graph.call_method(keypath[0].name, (node,)),
                                    keypath[1:],
                                    signature,
                                )
                        elif isinstance(keypath[0], pytree.SequenceKey):
                            signature = (("SequenceKey", keypath[0].idx), signature)
                            if signature in hash_cons:
                                res = hash_cons[signature].node
                            else:
                                res = unbacked_interp(
                                    graph.call_function(
                                        operator.getitem, (node, keypath[0].idx)
                                    ),
                                    keypath[1:],
                                    signature,
                                )
                        elif isinstance(keypath[0], ConvertIntKey):
                            signature = (("ConvertIntKey",), signature)
                            if signature in hash_cons:
                                res = hash_cons[signature].node
                            else:
                                res = unbacked_interp(
                                    graph.call_function(
                                        cast_symbool_to_symint_guardless, (node,)
                                    ),
                                    keypath[1:],
                                    signature,
                                )
                        elif isinstance(keypath[0], DivideByKey):
                            signature = (("DivideByKey", keypath[0].divisor), signature)
                            # TODO: need to assert divisibility
                            if signature in hash_cons:
                                res = hash_cons[signature].node
                            else:
                                res = unbacked_interp(
                                    graph.call_function(
                                        operator.floordiv, (node, keypath[0].divisor)
                                    ),
                                    keypath[1:],
                                    signature,
                                )
                        elif isinstance(keypath[0], InnerTensorKey):
                            signature = (("InnerTensorKey", keypath[0].inner_name), signature)
                            if signature in hash_cons:
                                res = hash_cons[signature].node
                            else:
                                res = unbacked_interp(
                                    graph.call_function(
                                        getattr, (node, keypath[0].inner_name)
                                    ),
                                    keypath[1:],
                                    signature,
                                )
                        else:
                            raise AssertionError(f"unrecognized keypath {keypath}")

                        if signature not in hash_cons:
                            hash_cons[signature] = fx.Proxy(res)
                        return res

                    res = fx.Proxy(unbacked_interp(node, keypath, _get_sym_val(node)))
                    # this below matches the logic mentioned above in the initial collection pass,
                    # about dynamo producing unbacked binding keys that match the symbols in the graph,
                    # but AOTAutograd producing new unbacked symbols, so we need to read the example value instead.
                    sym_expr = (
                        _get_sym_val(res.node)
                        if res.node.meta and isinstance(_get_sym_val(res.node), sympy.Symbol)
                        else symbol
                    )
                    hash_cons[sym_expr] = res
                    tracked_symbols.update(sym_expr.free_symbols)
                    log.debug("symbol_to_proxy[%s] = %s", sym_expr, hash_cons[sym_expr])

    def add_runtime_assert(node, sym_expr, op, partial_signature):
        '''
        From the input node, sym_expr, op, and partial_signature (e.g. min/max),
        add runtime assert to the graph.
        Here we only want to add:
        - torch.ops.aten.sym_constrain_range_for_size.default (anything size-like)
        - torch.ops.aten.sym_constrain_range.default (anything non-size-like)
        - torch.ops.aten._assert_scalar.default (anything not handled by the other two, e.g. strict LT/GT, inequality)
        '''
        nonlocal remaining_assert_nodes
        log.debug("adding runtime assert %s for sym_expr %s", op, sym_expr)
        signature = (op,) + partial_signature  # op is part of signature

        if op == torch.ops.aten._assert_scalar.default:
            if signature in sym_expr_to_nodes[sym_expr]:  # already exists
                res = sym_expr_to_nodes[sym_expr][signature][0]
            elif (torch._check,) + partial_signature in sym_expr_to_nodes[sym_expr]:  # allow torch._check too
                res = sym_expr_to_nodes[sym_expr][(torch._check,) + partial_signature][0]
            else:
                sym_rel, rhs = partial_signature
                full_expr = _sym_rel_to_sympy(sym_rel, sym_expr, rhs)
                # be careful about insert point if node already exists:
                # insert after input if it doesn't, otherwise let sympy_interp() handle insert point.
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
                        insert_after_args=True,
                    ).node
                with node.graph.inserting_before(res0.next):
                    msg = f"Runtime assertion failed for expression {full_expr} on node '{res0}'"
                    res = node.graph.call_function(op, (res0, msg))
                    log.debug("created node %s for runtime assert %s", res, full_expr)

        elif op in (
            torch.ops.aten.sym_constrain_range_for_size.default,
            torch.ops.aten.sym_constrain_range.default,
        ):
            is_size = (op == torch.ops.aten.sym_constrain_range_for_size.default)
            _min, _max = partial_signature
            if signature in sym_expr_to_nodes[sym_expr]:
                res = sym_expr_to_nodes[sym_expr][signature][0]
            else:
                # int_oo doesn't compile in the graph, so we expect numeric inputs to be passed into this function
                _kwargs = {}
                if not is_size or _min != 0:
                    _kwargs["min"] = _min
                if not is_size or _max != sys.maxsize:
                    _kwargs["max"] = _max
                with node.graph.inserting_before(node.next):
                    res = node.graph.call_function(op, (node,), _kwargs)
                    log.debug("created node %s for runtime assert %s, min=%s, max=%s", res, sym_expr, _min, _max)

        remaining_assert_nodes.add(res)
        return res

    def maybe_sym_constrain_range_for_size(node, expr, _min, _max):
        '''
        Try to insert a sym_constrain_range_for_size op if possible, but might not be possible if _max <= 2.
        This can happen when we try to simplify:
            torch._check_is_size(x) & torch._check(x == 2) -> torch.sym_constrain_range_for_size(x, 2, 2)
        If that's the case, add unbounded sym_constrain_range_for_size + bounded sym_constrain_range.
        Expects numeric inputs for _min, _max.
        '''
        log.debug("maybe_sym_constrain_range_for_size(%s), min=%s, max=%s", expr, _min, _max)
        if _max is None or _max > 2:  # doesn't work with max <= 2
            add_runtime_assert(
                res,
                sym_expr,
                torch.ops.aten.sym_constrain_range_for_size.default,
                (_min, _max),
            )
        else:  # sym_constrain_range_for_size(0, inf) + sym_constrain_range(min, max)
            add_runtime_assert(
                res,
                sym_expr,
                torch.ops.aten.sym_constrain_range_for_size.default,
                (0, sys.maxsize),
            )
            add_runtime_assert(
                res,
                sym_expr,
                torch.ops.aten.sym_constrain_range.default,
                (_min, _max),
            )

    # if no asserts, only DCE intermediate shape comp
    if any(ras for ras in ras_by_symbol.values()):

        # go through runtime asserts, add appropriate ops
        for sym_expr, sym_ranges in sym_expr_to_ranges.items():

            # all symbols exist -> sympy_interp()
            free_symbols = sym_expr.free_symbols
            if len(free_symbols - tracked_symbols) == 0:
                res = sympy_interp(
                    PythonReferenceAnalysis, hash_cons, sym_expr, hash_cons=hash_cons, insert_after_args=True,
                ).node

                # track nodes we keep or add
                remaining_assert_nodes: Set[fx.Node] = set()
                _min, is_gt = sym_ranges.min
                _max, is_lt = sym_ranges.max
                _max = sys.maxsize if _max is int_oo else _max  # can't put this in graph

                if sym_ranges.static:
                    # if static & is_size, try sym_constrain_range_for_size
                    # otherwise, _assert_scalar on equality
                    val = sym_ranges.val
                    log.debug("range assert for expr %s = %s", sym_expr, val)
                    if sym_ranges.is_size:
                        maybe_sym_constrain_range_for_size(res, sym_expr, val, val)
                    else:  # _assert_scalar
                        add_runtime_assert(
                            res,
                            sym_expr,
                            torch.ops.aten._assert_scalar.default,
                            (SymRel.EQ, val),
                        )
                elif _min is not None and _max is not None:
                    log.debug("range assert for expr %s, min = %s, max = %s", sym_expr, _min, _max)
                    # if is_size, try sym_constrain_range_for_size
                    # otherwise, we can sym_constrain_range if a single symbol
                    # if an expression, we add two _assert_scalar nodes
                    # also add strict lt, gt nodes onto sym_constrain_range as needed
                    if sym_ranges.is_size:
                        maybe_sym_constrain_range_for_size(res, sym_expr, _min, _max)
                        # add lt/gt
                        if is_lt:
                            add_runtime_assert(
                                res,
                                sym_expr,
                                torch.ops.aten._assert_scalar.default,
                                (SymRel.LT, _max),
                            )
                        if is_gt:
                            add_runtime_assert(
                                res,
                                sym_expr,
                                torch.ops.aten._assert_scalar.default,
                                (SymRel.GT, _min),
                            )
                    elif isinstance(sym_expr, sympy.Symbol):
                        add_runtime_assert(
                            res,
                            sym_expr,
                            torch.ops.aten.sym_constrain_range.default,
                            (_min, _max),
                        )
                        # add lt/gt
                        if is_lt:
                            add_runtime_assert(
                                res,
                                sym_expr,
                                torch.ops.aten._assert_scalar.default,
                                (SymRel.LT, _max),
                            )
                        if is_gt:
                            add_runtime_assert(
                                res,
                                sym_expr,
                                torch.ops.aten._assert_scalar.default,
                                (SymRel.GT, _min),
                            )
                    else:
                        add_runtime_assert(
                            res,
                            sym_expr,
                            torch.ops.aten._assert_scalar.default,
                            (SymRel.GT if is_gt else SymRel.GE, _min),
                        )
                        add_runtime_assert(
                            res,
                            sym_expr,
                            torch.ops.aten._assert_scalar.default,
                            (SymRel.LT if is_lt else SymRel.LE, _max),
                        )

                elif _min is not None:
                    log.debug("range assert for expr %s, min = %s", sym_expr, _min)
                    # one-sided _assert_scalar
                    add_runtime_assert(
                        res,
                        sym_expr,
                        torch.ops.aten._assert_scalar.default,
                        (SymRel.GT if is_gt else SymRel.GE, _min),
                    )

                elif _max is not None:
                    log.debug("range assert for expr %s, max = %s", sym_expr, _max)
                    # same here
                    add_runtime_assert(
                        res,
                        sym_expr,
                        torch.ops.aten._assert_scalar.default,
                        (SymRel.LT if is_lt else SymRel.LE, _max),
                    )

                for val in sym_ranges.not_equals:
                    log.debug("range assert for expr %s != %s", sym_expr, val)
                    # _assert_scalar for any inequalities
                    add_runtime_assert(
                        res,
                        sym_expr,
                        torch.ops.aten._assert_scalar.default,
                        (SymRel.NE, val),
                    )

            # clean out remaining nodes that existed before
            for nodes in sym_expr_to_nodes[sym_expr].values():
                for node in nodes:
                    if node not in remaining_assert_nodes:
                        log.debug("dce node %s for runtime assert %s", node, sym_expr)
                        _dce_from_node(node)

    # clean out unused size/stride nodes that we created.
    # alternative is to track used symbols in initial graph pass so we don't create these.
    for node in created_placeholder_ops:
        if not node.users:
            log.debug("dce created placeholder op %s", node)
            _dce_from_node(node)
