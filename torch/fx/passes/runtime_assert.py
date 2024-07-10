# mypy: allow-untyped-defs
import logging
import operator
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
from torch.fx.experimental.sym_node import SymNode
from torch.fx.graph_module import GraphModule

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
    """

    # We hash (node_name, min_val, max_val)
    nodes_that_already_have_sym_constraint_range = set()

    # We hash only node name here because size don't take min/max
    nodes_that_already_have_sym_constraint_size = set()
    # TODO this only works for top-level nodes today, also
    # we should potentially use it not create duplicate
    # assert_async nodes
    for node in gm.graph.nodes:
        if (
            node.op == "call_function"
            and node.target == torch.ops.aten.sym_constrain_range.default
        ):
            assert len(node.args) == 1
            nodes_that_already_have_sym_constraint_range.add(
                (node.args[0], node.kwargs.get("min"), node.kwargs.get("max"))
            )
        if (
            node.op == "call_function"
            and node.target == torch.ops.aten.sym_constrain_range_for_size.default
        ):
            assert len(node.args) == 1
            nodes_that_already_have_sym_constraint_size.add(node.args[0])

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
    from torch.utils._sympy.interp import sympy_interp
    from torch.utils._sympy.numbers import int_oo
    from torch.utils._sympy.reference import PythonReferenceAnalysis

    # TODO: Request simplification on runtime asserts before emitting them
    ras_by_symbol = shape_env.deferred_runtime_asserts.copy()
    graph = gm.graph

    if not any(ras for ras in ras_by_symbol.values()):
        return

    graph_code_log.debug(
        "%s",
        lazy_format_graph_code(
            f"pre insert_deferred_runtime_asserts {name}", gm, colored=True
        ),
    )

    # deduplicate unassociated runtime assertions
    # we could do better, some guards might be redundant,
    # e.g. Eq(s0, 4) & Eq(2*s0, 8)
    # but unclear how to handle all of that right now.
    # TODO(pianpwk): better way of doing this
    new_ras = []
    ras_exprs: Set[sympy.Expr] = set()
    for ras in ras_by_symbol.pop(None, []):  # type: ignore[call-overload]
        if ras.expr not in ras_exprs:
            new_ras.append(ras)
            ras_exprs.add(ras.expr)
    ras_by_symbol[None] = new_ras  # type: ignore[index]

    # We are going to mutate the dict
    symbol_to_proxy: Dict[sympy.Symbol, fx.Proxy] = {}
    placeholders = set()
    last_placeholder = None
    for node in graph.nodes:
        if node.op != "placeholder":
            break
        last_placeholder = node
        placeholders.add(node)
    if last_placeholder is None:  # no placeholders, just insert before first node
        last_placeholder = next(iter(graph.nodes))

    # Identify what symbols we need to reify.  This isn't strictly needed
    # but helps reduce churn on the graph
    needed_symbols: Set[sympy.Symbol] = set()
    for ras in ras_by_symbol.values():
        for ra in ras:
            needed_symbols.update(free_symbols(ra.expr))

    log.debug("needed_symbols = %s", needed_symbols)

    def add_runtime_asserts(ras):
        for ra in ras:
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
                res = sympy_interp(
                    PythonReferenceAnalysis, symbol_to_proxy, ra.expr
                ).node
                graph.call_function(
                    torch.ops.aten._assert_scalar.default,
                    # TODO: use ra.msg here, but it's pretty
                    # useless right now
                    (
                        res,
                        f"Runtime assertion failed for expression {ra.expr} on node '{res}'",
                    ),
                )

    inserted_sym_nodes = 0  # for inserting unassociated runtime asserts
    nodes = list(graph.nodes)
    for i, node in enumerate(nodes[:-1]):
        # Placeholders can match symbols, but when we destructure them
        # with size we have to make sure we insert the nodes after all
        # the placeholders
        with graph.inserting_before(
            nodes[i + 1] if node not in placeholders else last_placeholder.next
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
                        nonlocal inserted_sym_nodes
                        inserted_sym_nodes += 1

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
            if node not in placeholders:
                last_sym_node = last_placeholder
                for _ in range(inserted_sym_nodes):
                    last_sym_node = last_sym_node.next
                with graph.inserting_before(last_sym_node.next):
                    add_runtime_asserts(ras_by_symbol.pop(None, []))  # type: ignore[call-overload]

            defs = []

            if unbacked_bindings := node.meta.get("unbacked_bindings"):
                for s, keypath in unbacked_bindings.items():
                    defs.append(s)

                    # TODO: some CSE when generating these nodes can probably
                    # help reduce graph size and improve compile itme
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

                    symbol_to_proxy[s] = fx.Proxy(go(node, keypath))
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

                if i0 in shape_env.size_like:
                    if export:
                        if (
                            symbol_to_proxy[i0].node
                            not in nodes_that_already_have_sym_constraint_size
                        ):
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

                    if (
                        symbol_to_proxy[i0].node,
                        min_val,
                        max_val,
                    ) not in nodes_that_already_have_sym_constraint_range:
                        graph.call_function(
                            torch.ops.aten.sym_constrain_range.default,
                            (symbol_to_proxy[i0].node,),
                            {
                                "min": convert(vr.lower),
                                "max": convert(vr.upper),
                            },
                        )

                add_runtime_asserts(ras)
