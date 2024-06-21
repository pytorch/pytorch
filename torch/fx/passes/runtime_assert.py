# mypy: allow-untyped-defs
import logging
import operator
from typing import Any, Dict, Optional, Set, TYPE_CHECKING

# Import sympy and ShapeEnv during TYPE_CHECKING since importing sympy is slow
if TYPE_CHECKING:
    import sympy
    from torch.fx.experimental.symbolic_shapes import ShapeEnv
else:
    ShapeEnv = Any

import torch
import torch.utils._pytree as pytree
from torch import fx
from torch.fx._compatibility import compatibility
from torch.fx._utils import lazy_format_graph_code
from torch.fx.experimental.sym_node import SymNode
from torch.fx.graph_module import GraphModule
from torch.fx.experimental.proxy_tensor import py_sym_types

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


def _get_sym_val(node: fx.Node) -> Optional["sympy.Expr"]:
    val = _get_example_value(node)
    if isinstance(val, py_sym_types):
        return val.node.expr
    return None


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

    This pass also deduplicates size-related computation, CSE-ing ops that produce
    symbolic values and/or are involved in runtime asserts. Additionally, shape calls
    (size/stride/storage_offset) are turned into compute on input sizes if possible,
    allowing intermediate tensors to be freed earlier. For example, here dynamo will
    DCE the cat and repeat calls:

        z = torch.cat([x, x], dim=0)  # 2*s0
        w = z.repeat(y.shape[0])  # 2*s0*s1
        _w = w.shape[0]
        # something with _w, but not w ...

        # turns into ->
        s0 = x.shape[0]
        s1 = y.shape[0]
        _w0 = 2 * s0
        _w = _w0 * s1

    Redundant torch._check or torch.ops.aten._assert_scalar.default calls that assert
    the same expression, and redundant constrain_range calls are also deduplicated.
    Additionally, if a constrain_range call exists for an unbacked symbol, then single-symbol
    bound checks (e.g. u0 >= 0, u0 <= 5) will have accumulated information in the ShapeEnv,
    and can be considered redundant and removed.
    """

    # Import sympy locally
    import sympy

    from torch.fx.experimental.symbolic_shapes import (
        CallMethodKey,
        cast_symbool_to_symint_guardless,
        ConvertIntKey,
        DivideByKey,
        free_symbols,
        InnerTensorKey,
    )
    from torch.utils._sympy.numbers import int_oo
    from torch.utils._sympy.reference import PythonReferenceAnalysis

    # TODO: Request simplification on runtime asserts before emitting them
    ras_by_symbol = shape_env.deferred_runtime_asserts.copy()
    graph = gm.graph
    graph_code_log.debug(
        "%s",
        lazy_format_graph_code(
            f"pre insert_deferred_runtime_asserts {name}", gm, colored=True
        ),
    )

    # We are going to mutate the dict
    symbol_to_proxy: Dict[sympy.Symbol, fx.Proxy] = {}
    placeholders = set()
    first_non_placeholder = next(iter(graph.nodes))
    for node in graph.nodes:
        if node.op != "placeholder":
            first_non_placeholder = node
            break
        else:
            placeholders.add(node)

    def _is_intermediate_tensor_shape_call(node: fx.Node) -> bool:
        """
        Returns true if a size/stride/storage offset call on an intermediate tensor.
        If so, we can try to compute this from input shapes instead of this call.
        """
        if (
            node.target in (
                torch.ops.aten.sym_size.int,
                torch.ops.aten.sym_stride.int,
                torch.ops.aten.storage_offset.default,
            )
            and node.args[0].op != "placeholder"
        ):  # export: sym_size.int(node, dim)
            return True
        if (
            node.target == operator.getitem
            and node.args[0].target in (
                "size",
                "stride",
                "storage_offset",
            )
            and node.args[0].args[0].op != "placeholder"
        ):  # dynamo export: getitem(size(node), dim)
            return True
        return False

    # Identify what symbols we need to reify.  This isn't strictly needed
    # but helps reduce churn on the graph
    needed_symbols: Set[sympy.Symbol] = set()
    for ras in ras_by_symbol.values():
        for ra in ras:
            needed_symbols.update(free_symbols(ra.expr))
    for node in gm.graph.nodes:
        if (
            node.op == "call_function"
            and _is_intermediate_tensor_shape_call(node)
            and (sym_expr := _get_sym_val(node)) is not None
            and not isinstance(sym_expr, sympy.Number)
        ):  # use symbols for intermediate shape calls, even if not in runtime asserts
            needed_symbols.update(sym_expr.free_symbols)

    log.debug("needed_symbols = %s", needed_symbols)

    # Track asserts/checks we've added
    added_asserts: Set["sympy.Expr"] = set()
    constrained_unbacked_symbols: Set["sympy.Symbol"] = set()

    def _sympy_interp(symbol_to_proxy, expr):
        # sympy_interp() with hash consing
        from sympy import Integer, Number, Symbol
        from sympy.logic.boolalg import Boolean as SympyBoolean, BooleanAtom
        from torch.utils._sympy.interp import run_sympy_handler, sympy_interp

        # base cases
        if isinstance(expr, (Integer, Number, Symbol, BooleanAtom)):
            return sympy_interp(
                PythonReferenceAnalysis, symbol_to_proxy, expr
            )  # this returns non-proxy object, don't cache
        if expr in symbol_to_proxy:
            return symbol_to_proxy[expr]

        # hash cons on arguments
        fx_args = []
        for arg in expr.args:
            if arg in symbol_to_proxy:
                res = symbol_to_proxy[arg]
            else:
                res = _sympy_interp(symbol_to_proxy, arg)
            fx_args.append(res)
        
        # run expr handler
        symbol_to_proxy[expr] = run_sympy_handler(
            PythonReferenceAnalysis, fx_args, expr
        )
        return symbol_to_proxy[expr]

    def _is_bound_expr_for_symbol(expr: "sympy.Expr") -> bool:
        # This is probably unnecessary, but since torch._check() calls for single-symbol bounds
        # like u0 >= 0, 10 >= u0 accumulate range info in the ShapeEnv, and we insert sym_constrain_range calls
        # anyways, we designate these calls as redundant and remove them.
        if len(expr.args) != 2:
            return False
        if expr.func not in (sympy.LessThan, sympy.GreaterThan):
            return False
        lhs, rhs = expr.args
        return (
            (isinstance(lhs, sympy.Symbol) and isinstance(rhs, sympy.Number))
            or (isinstance(rhs, sympy.Symbol) and isinstance(lhs, sympy.Number))
        )

    def add_runtime_asserts(ras):
        for ra in ras:
            if (
                ra.expr in added_asserts  # redundant
                or (
                    len(ra.expr.free_symbols) == 1
                    and next(iter(ra.expr.free_symbols)) in constrained_unbacked_symbols
                    and _is_bound_expr_for_symbol(ra.expr)
                )
                # if we've already added a constrain_range call for this symbol,
                # then single-symbol bound asserts like u0 >= 0, u0 <= 5 are redundant.
            ):
                continue

            log.debug("inserting runtime assert %s", ra.expr)
            # Need to process ALL free symbols, not just unbacked ones
            fvs = free_symbols(ra.expr)
            missing = fvs - symbol_to_proxy.keys()
            if missing:
                i1 = min(missing, key=str)
                # TODO: Remove relaxing assert on unbacked_symint https://github.com/pytorch/pytorch/issues/119689
                # assert shape_env.is_unbacked_symint(i1), i1
                ras_by_symbol.setdefault(i1, []).append(ra)
            else:
                # Convert the sympy expression into a sequence of FX
                # nodes
                res = _sympy_interp(symbol_to_proxy, ra.expr).node
                graph.call_function(
                    torch.ops.aten._assert_scalar.default,
                    # TODO: use ra.msg here, but it's pretty
                    # useless right now
                    (
                        res,
                        f"Runtime assertion failed for expression {ra.expr} on node '{res}'",
                    ),
                )
                added_asserts.add(ra.expr)

    nodes = list(graph.nodes)
    for i, node in enumerate(nodes[:-1]):
        # Placeholders can match symbols, but when we destructure them
        # with size we have to make sure we insert the nodes after all
        # the placeholders
        with graph.inserting_before(
            nodes[i + 1] if node not in placeholders else first_non_placeholder
        ):
            # Unfortunately, this logic still must remain because manual
            # make_fx calls may not explicitly bind all symbolic ints as
            # arguments to the function, so we must infer it from the other
            # arguments
            if (
                node in placeholders
                and (example_value := _get_example_value(node)) is not None
            ):

                def match_symbol(symint, cb):
                    if (
                        isinstance(symint, torch.SymInt)
                        and isinstance(symint.node, SymNode)
                        and isinstance(s := symint.node.expr, sympy.Symbol)
                        and s not in symbol_to_proxy
                        and s in needed_symbols
                    ):
                        symbol_to_proxy[s] = fx.Proxy(cb())
                        log.debug("symbol_to_proxy[%s] = %s", s, symbol_to_proxy[s])

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

            # Handle asserts that aren't associated with any symbol.  This
            # doesn't really have to be in the loop as it will only run once,
            # it just needs to happen right after the placeholders.
            # insert this after placeholders & added sym nodes, and before non-placeholders.
            if node == first_non_placeholder:
                add_runtime_asserts(ras_by_symbol.pop(None, []))  # type: ignore[call-overload]

            # deduplicate asserts already present in graph
            if node.target in (
                torch._check,
                torch.ops.aten._assert_scalar.default,
            ):
                if (
                    node.args[0] == True  # trivial
                    or (assert_expr := _get_sym_val(node.args[0])) in added_asserts
                ):
                    gm.graph.erase_node(node)
                else:
                    added_asserts.add(assert_expr)

            # hash cons on symbolic values
            if (
                node.op == "call_function"
                and (sym_expr := _get_sym_val(node)) is not None
            ):
                if sym_expr in symbol_to_proxy:  # node produces redundant expression
                    hash_node = symbol_to_proxy[sym_expr].node
                    node.replace_all_uses_with(hash_node)
                    gm.graph.erase_node(node)
                    log.debug("CSE node %s -> %s for expr %s", node, hash_node, sym_expr)
                elif (
                    _is_intermediate_tensor_shape_call(node)
                    and not (sym_expr.free_symbols - symbol_to_proxy.keys())
                    and not isinstance(sym_expr, sympy.Number)
                ):  # shape call on intermediate tensor, turn into computation on input shapes
                    symbol_to_proxy[sym_expr] = _sympy_interp(symbol_to_proxy, sym_expr)
                    hash_node = symbol_to_proxy[sym_expr].node
                    node.replace_all_uses_with(hash_node)
                    gm.graph.erase_node(node)
                    log.debug("CSE node %s -> %s for expr %s", node, hash_node, sym_expr)
                    # won't try DCE-ing tensor compute here
                else:
                    symbol_to_proxy[sym_expr] = fx.Proxy(node)

            if not any(ras for ras in ras_by_symbol.values()):
                continue  # if no runtime asserts are present, we only do CSE, don't add refinement calls

            # If the symbol is used, we'll call sym_constrain_range(_for_size) later when we see it anyways,
            # so delete calls before that
            if node.target in (
                torch._check_is_size,
                torch.ops.aten.sym_constrain_range.default,
                torch.ops.aten.sym_constrain_range_for_size.default,
            ):
                gm.graph.erase_node(node)

            defs = []

            if unbacked_bindings := node.meta.get("unbacked_bindings"):
                for s, keypath in unbacked_bindings.items():
                    defs.append(s)
                    def unbacked_interp(node, keypath, signature):
                        if keypath == ():
                            return node
                        if signature in symbol_to_proxy:  # CSE
                            hash_node = symbol_to_proxy[signature].node
                            log.debug("CSE unbacked_bindings node for keypath %s on node %s", keypath, node)
                            return hash_node
                        if (
                            len(keypath) >= 2
                            and isinstance(keypath[0], CallMethodKey)
                            and isinstance(keypath[1], pytree.SequenceKey)
                        ):
                            signature = (("CallMethod", keypath[0].name, keypath[1].idx), signature)
                            if signature in symbol_to_proxy:
                                res = symbol_to_proxy[signature].node
                            elif keypath[0].name == "size":
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
                            if signature in symbol_to_proxy:
                                res = symbol_to_proxy[signature].node
                            else:
                                res = unbacked_interp(
                                    graph.call_method(keypath[0].name, (node,)),
                                    keypath[1:],
                                    signature,
                                )
                        elif isinstance(keypath[0], pytree.SequenceKey):
                            signature = (("SequenceKey", keypath[0].idx), signature)
                            if signature in symbol_to_proxy:
                                res = symbol_to_proxy[signature].node
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
                            if signature in symbol_to_proxy:
                                res = symbol_to_proxy[signature].node
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
                            if signature in symbol_to_proxy:
                                res = symbol_to_proxy[signature].node
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
                            if signature in symbol_to_proxy:
                                res = symbol_to_proxy[signature].node
                            else:
                                res = unbacked_interp(
                                    graph.call_function(
                                        getattr, (node, keypath[0].inner_name)
                                    ),
                                    keypath[1:],
                                )
                        else:
                            raise AssertionError(f"unrecognized keypath {keypath}")

                        if signature not in symbol_to_proxy:
                            symbol_to_proxy[signature] = fx.Proxy(res)
                        return res

                    if s not in symbol_to_proxy:
                        symbol_to_proxy[s] = fx.Proxy(unbacked_interp(node, keypath, s))
                        log.debug("symbol_to_proxy[%s] = %s", s, symbol_to_proxy[s])

            for i0 in defs:
                ras = ras_by_symbol.pop(i0, [])
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
                if i0 in constrained_unbacked_symbols:
                    continue  # constrain symbol just once

                if i0 in shape_env.size_like:
                    if export:
                        graph.call_function(
                            torch.ops.aten.sym_constrain_range_for_size.default,
                            (symbol_to_proxy[i0].node,),
                        )
                    else:
                        graph.call_function(
                            torch._check_is_size, (symbol_to_proxy[i0].node,)
                        )

                vr = shape_env.var_to_range[i0]
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

                    min_val = convert(vr.lower)
                    max_val = convert(vr.upper)
                    graph.call_function(
                        torch.ops.aten.sym_constrain_range.default,
                        (symbol_to_proxy[i0].node,),
                        {
                            "min": convert(vr.lower),
                            "max": convert(vr.upper),
                        },
                    )

                constrained_unbacked_symbols.add(i0)
                add_runtime_asserts(ras)
