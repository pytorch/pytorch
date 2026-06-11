import dis
import functools
import logging
import sys
from collections.abc import Callable
from typing import Any, Optional, TYPE_CHECKING


# Import sympy and ShapeEnv during TYPE_CHECKING since importing sympy is slow
if TYPE_CHECKING:
    import sympy

    from torch.fx.experimental.symbolic_shapes import ShapeEnv
else:
    ShapeEnv = Any

import torch
import torch.utils._pytree as pytree
from torch import fx
from torch._subclasses.meta_utils import is_sparse_any
from torch.fx._compatibility import compatibility
from torch.fx._utils import lazy_format_graph_code
from torch.fx.experimental.proxy_tensor import py_sym_types
from torch.fx.experimental.sym_node import SymNode
from torch.fx.graph_module import GraphModule


__all__ = ["insert_deferred_runtime_asserts"]

log = logging.getLogger(__name__)
graph_code_log = torch._logging.getArtifactLogger(__name__, "graph_code_verbose")


def _get_example_value(node: fx.Node) -> str | None:
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
    that x.item() >= 0.  These asserts can happen unpredictably during fake
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
        _w0 = 2 * s0
        _w = _w0 * s1

        # where s0, s1 are either SymInt graph inputs, or the result of added size calls

    Redundant compiler-generated asserts that assert the same expression, and
    redundant constrain_range calls are also deduplicated. Additionally, because
    single-symbol bound checks (e.g. u0 >= 0, u0 <= 5) accumulate information in
    the ShapeEnv, the ShapeEnv contains min/max bounds for each symbol, and we
    add bound checks at the end of this pass when an equivalent explicit assert
    is not already present in the graph.
    """

    # Import sympy locally
    import sympy

    from torch._export.passes._node_metadata_hook import _set_node_metadata_hook
    from torch.fx.experimental.symbolic_shapes import (
        _get_placeholder_expr,
        _has_uninterpretable_sympy_function,
        free_symbols,
        resolve_unbacked_bindings,
        RuntimeAssert,
    )
    from torch.utils._sympy.numbers import int_oo
    from torch.utils._sympy.reference import (
        OptimizedPythonReferenceAnalysis,
        PythonReferenceAnalysis,
    )
    from torch.utils._sympy.value_ranges import ValueRanges

    # TODO: Request simplification on runtime asserts before emitting them
    ras_by_symbol = shape_env.deferred_runtime_asserts.copy()
    graph = gm.graph
    tracer = fx.proxy.GraphAppendingTracer(graph)
    graph_code_log.debug(
        "%s",
        lazy_format_graph_code(
            f"pre insert_deferred_runtime_asserts {name}", gm, colored=True
        ),
    )

    # We are going to mutate the dict
    expr_to_proxy: dict[sympy.Expr, fx.Proxy] = {}
    placeholders = set()
    first_non_placeholder = None
    for node in graph.nodes:
        if node.op != "placeholder":
            first_non_placeholder = node
            break
        else:
            placeholders.add(node)

    def _is_intermediate_tensor_sym_call(node: fx.Node) -> bool:
        """
        If a size/stride/storage offset call on an intermediate tensor,
        we can try to compute the value from input shapes instead.
        """
        return (
            (val := _get_sym_val(node)) is not None
            and not isinstance(val, sympy.Number)
            # this holds back from reifying anything in torch.utils._sympy.functions.py that's unsupported
            and not _has_uninterpretable_sympy_function(val)
            and any(
                isinstance(arg, fx.Node)
                and isinstance(_get_example_value(arg), (torch.Tensor, torch.Size))
                and arg.op != "placeholder"
                for arg in node.args
            )
        )

    # Figure out what key to use, val or example_value
    val_key = "val"
    for node in graph.nodes:
        if "example_value" in node.meta:
            val_key = "example_value"
            break
        elif "val" in node.meta:
            break

    # Note: DO NOT register one _set_node_metadata_hook(_node_metadata_hook)
    # for each nodes in the graph.
    # _set_node_metadata_hook is expensive and this can cause compile
    # time to regress significantly.
    def _node_metadata_hook(
        node: torch.fx.Node,
        stack_trace: str | None = None,
        nn_module_stack: dict[str, Any] | None = None,
        custom: dict[str, Any] | None = None,
        skip_val: bool = False,
    ) -> None:
        if not skip_val:
            fake_args = pytree.tree_map(
                lambda arg: (
                    _get_example_value(arg) if isinstance(arg, torch.fx.Node) else arg
                ),
                node.args,
            )
            try:
                target = node.target
                if node.op == "call_method":
                    if not isinstance(node.target, str):
                        raise AssertionError(
                            f"Expected str target, got {type(node.target)}"
                        )
                    target = getattr(fake_args[0], node.target)
                    fake_args = fake_args[1:]
                node.meta[val_key] = target(*fake_args)  # type: ignore[operator]
            except NotImplementedError:
                # This can happen when attempting to reify a symbol with an unsupported call_function node,
                # e.g. with NestedTensors + sym_size.int via match_symbol().
                # This seems to be fine, as the node gets CSE'd and deleted later in favor of a SymInt graph input.
                pass
        if stack_trace is not None:
            node.meta["stack_trace"] = stack_trace
        if nn_module_stack is not None:
            node.meta["nn_module_stack"] = nn_module_stack
        if custom is not None:
            node.meta["custom"] = custom

    # Track asserts/checks we've added
    added_asserts: set[sympy.Expr] = set()
    constrained_unbacked_symbols: set[sympy.Symbol] = set()

    Analysis = PythonReferenceAnalysis if export else OptimizedPythonReferenceAnalysis

    def _sympy_interp(
        expr_to_proxy: dict[sympy.Expr, fx.Proxy], expr: sympy.Expr
    ) -> fx.Proxy:
        # Lower a sympy expression into the graph and return the resulting
        # Proxy (use .node to get the FX node representing ``expr``).
        # sympy_interp() with hash consing
        from sympy import Integer, Number, Symbol
        from sympy.logic.boolalg import BooleanAtom

        from torch.utils._sympy.interp import _run_sympy_handler, sympy_interp

        # hash cons
        if expr in expr_to_proxy:
            return expr_to_proxy[expr]
        # base cases, don't cache
        if isinstance(expr, (Integer, Number, Symbol, BooleanAtom)):
            return sympy_interp(Analysis, expr_to_proxy, expr)

        # hash cons on arguments, run expr handler
        expr_to_proxy[expr] = _run_sympy_handler(
            Analysis,
            [_sympy_interp(expr_to_proxy, arg) for arg in expr.args],
            expr,
        )
        return expr_to_proxy[expr]

    def _is_bound_expr_for_symbol(expr: "sympy.Expr") -> bool:
        # This is probably unnecessary, but since torch._check() calls for single-symbol bounds
        # like u0 >= 0, 10 >= u0 accumulate range info in the ShapeEnv, we designate these calls as redundant
        # and instead add 2 runtime asserts at the end of this pass, if the min/max bounds are non-trivial.
        if len(expr.args) != 2 or expr.func not in (sympy.LessThan, sympy.GreaterThan):
            return False
        lhs, rhs = expr.args
        return (isinstance(lhs, sympy.Symbol) and isinstance(rhs, sympy.Number)) or (
            isinstance(rhs, sympy.Symbol) and isinstance(lhs, sympy.Number)
        )

    assert_targets = (
        torch._check,
        torch._assert_scalar,
        torch.ops.aten._assert_scalar.default,
    )
    preserve_runtime_assert_key = "_fx_runtime_assert_preserve"

    def _assertion_condition(node: fx.Node) -> Any:
        if node.args:
            return node.args[0]
        return node.kwargs.get("cond", node.kwargs.get("self"))

    def _assertion_message(
        node: fx.Node, cond: Any, assert_expr: Optional["sympy.Expr"]
    ) -> str:
        def static_string_from_callable(fn: Callable[[], object]) -> str | None:
            code = getattr(fn, "__code__", None)
            if (
                code is None
                or code.co_argcount != 0
                or code.co_posonlyargcount != 0
                or code.co_kwonlyargcount != 0
            ):
                return None

            # torch._check message callables are lazy. Preserve common constant
            # string lambdas without executing arbitrary user code at compile time.
            ignored_opnames = {
                "CACHE",
                "COPY_FREE_VARS",
                "EXTENDED_ARG",
                "NOP",
                "RESUME",
            }
            instructions = [
                inst
                for inst in dis.get_instructions(fn)
                if inst.opname not in ignored_opnames
            ]
            if (
                len(instructions) == 1
                and instructions[0].opname == "RETURN_CONST"
                and isinstance(instructions[0].argval, str)
            ):
                return instructions[0].argval
            if (
                len(instructions) == 2
                and instructions[0].opname == "LOAD_CONST"
                and isinstance(instructions[0].argval, str)
                and instructions[1].opname == "RETURN_VALUE"
            ):
                return instructions[0].argval
            return None

        msg = None
        if len(node.args) > 1:
            msg = node.args[1]
        elif node.target is torch._check:
            msg = node.kwargs.get("message")
        else:
            msg = node.kwargs.get("assert_msg")

        if isinstance(msg, str):
            return msg
        if isinstance(msg, fx.Node) and msg.op == "get_attr":
            attr = gm
            if not isinstance(msg.target, str):
                raise AssertionError(f"Expected str target, got {type(msg.target)}")
            for atom in msg.target.split("."):
                attr = getattr(attr, atom)
            if callable(attr):
                static_msg = static_string_from_callable(attr)
                if static_msg is not None:
                    return static_msg
            if isinstance(attr, str):
                return attr
        return f"Runtime assertion failed for expression {assert_expr} on node '{cond}'"

    def _is_user_runtime_assert(node: fx.Node) -> bool:
        if node.target in (torch._check, torch._assert_scalar):
            return True
        return (
            node.target is torch.ops.aten._assert_scalar.default
            and node.meta.get(preserve_runtime_assert_key) is True
        )

    def _replace_with_aten_assert(
        node: fx.Node, cond: Any, assert_expr: Optional["sympy.Expr"]
    ) -> None:
        with graph.inserting_before(node):
            new_node = graph.call_function(
                torch.ops.aten._assert_scalar.default,
                (cond, _assertion_message(node, cond, assert_expr)),
            )
            new_node.meta.update(node.meta)
            new_node.meta[preserve_runtime_assert_key] = True
        node.replace_all_uses_with(new_node)
        graph.erase_node(node)

    user_assert_exprs: set[sympy.Expr] = set()
    for node in graph.nodes:
        if node.target in assert_targets and _is_user_runtime_assert(node):
            cond = _assertion_condition(node)
            if isinstance(cond, fx.Node):
                assert_expr = _get_sym_val(cond)
                if assert_expr is not None:
                    user_assert_exprs.add(assert_expr)

    def add_runtime_asserts(ras: list[RuntimeAssert]) -> None:
        for ra in ras:
            if ra.expr in user_assert_exprs:
                added_asserts.add(ra.expr)
                continue
            if (
                # redundant
                ra.expr in added_asserts
                # if we've already added a constrain_range call for this symbol,
                # then single-symbol bound asserts like u0 >= 0, u0 <= 5 are redundant.
                or (
                    len(ra.expr.free_symbols) == 1
                    and next(iter(ra.expr.free_symbols)) in constrained_unbacked_symbols
                    and _is_bound_expr_for_symbol(ra.expr)
                )
                # don't try to reify sympy functions we can't turn into FX nodes
                or _has_uninterpretable_sympy_function(ra.expr)
            ):
                continue

            log.debug("inserting runtime assert %s", ra.expr)
            # Need to process ALL free symbols, not just unbacked ones
            fvs = free_symbols(ra.expr)
            missing = fvs - expr_to_proxy.keys()
            if missing:
                i1 = min(missing, key=str)
                # TODO: Remove relaxing assert on unbacked_symint https://github.com/pytorch/pytorch/issues/119689
                # assert shape_env.is_unbacked_symint(i1), i1
                ras_by_symbol.setdefault(i1, []).append(ra)
            else:
                # Convert the sympy expression into a sequence of FX
                # nodes
                with _set_node_metadata_hook(
                    gm,
                    functools.partial(
                        _node_metadata_hook,
                        stack_trace=node.meta.get("stack_trace"),
                        nn_module_stack=node.meta.get("nn_module_stack"),
                        # nodes added in `apply_runtime_assertion_pass` will have the same annotation
                        # as the input node to the assertion
                        custom=node.meta.get("custom"),
                    ),
                ):
                    res = _sympy_interp(expr_to_proxy, ra.expr).node

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

                def match_symbol(symint: object, cb: Callable[[], fx.Node]) -> None:
                    if (
                        isinstance(symint, torch.SymInt)
                        and isinstance(symint.node, SymNode)
                        and isinstance(
                            s := _get_placeholder_expr(symint.node), sympy.Symbol
                        )
                        and s not in expr_to_proxy
                    ):
                        with _set_node_metadata_hook(
                            gm,
                            functools.partial(
                                _node_metadata_hook,
                                stack_trace=node.meta.get("stack_trace"),
                                nn_module_stack=node.meta.get("nn_module_stack"),
                                # nodes added in `apply_runtime_assertion_pass` will have the same annotation
                                # as the input node to the assertion
                                custom=node.meta.get("custom"),
                            ),
                        ):
                            expr_to_proxy[s] = fx.Proxy(cb(), tracer=tracer)

                        log.debug("expr_to_proxy[%s] = %s", s, expr_to_proxy[s])

                match_symbol(example_value, lambda: node)

                if isinstance(t := example_value, torch.Tensor):
                    for i, s in enumerate(t.size()):
                        match_symbol(
                            s,
                            lambda: graph.create_size_node(node, i),
                        )
                    if not is_sparse_any(t):
                        for i, s in enumerate(t.stride()):
                            match_symbol(
                                s,
                                lambda: graph.create_stride_node(node, i),
                            )
                        match_symbol(
                            t.storage_offset(),
                            lambda: graph.create_storage_offset_node(node),
                        )

            # Handle asserts that aren't associated with any symbol.  This
            # doesn't really have to be in the loop as it will only run once,
            # it just needs to happen right after the placeholders.
            # insert this after placeholders & added sym nodes, and before non-placeholders.
            if node == first_non_placeholder:
                add_runtime_asserts(ras_by_symbol.pop(None, []))  # type: ignore[call-overload]

            # deduplicate asserts already present in graph, and remove trivial asserts
            if node.target in assert_targets:
                cond = _assertion_condition(node)
                assert_expr = _get_sym_val(cond) if isinstance(cond, fx.Node) else None
                if cond == True:  # noqa: E712
                    arg = cond
                    gm.graph.erase_node(node)
                    if isinstance(arg, fx.Node) and not arg.users:
                        gm.graph.erase_node(arg)
                else:
                    if _is_user_runtime_assert(node):
                        if node.target is not torch.ops.aten._assert_scalar.default:
                            _replace_with_aten_assert(node, cond, assert_expr)
                        if assert_expr is not None:
                            added_asserts.add(assert_expr)
                    elif (
                        assert_expr is not None
                        and assert_expr in expr_to_proxy
                        and assert_expr in added_asserts
                    ):
                        arg = cond
                        gm.graph.erase_node(node)
                        if isinstance(arg, fx.Node) and not arg.users:
                            gm.graph.erase_node(arg)
                    else:
                        if assert_expr is not None:
                            added_asserts.add(assert_expr)

            # hash cons, replace function calls that return torch.SymInts with direct references to
            # FX nodes built up to reify the sympy expression.
            if (
                node.op != "placeholder"
                and (sym_expr := _get_sym_val(node)) is not None
            ):
                # this guards against deleting calls like item() that produce new untracked symbols
                def has_new_untracked_symbols() -> bool:
                    # pyrefly: ignore [missing-attribute]
                    for symbol in sym_expr.free_symbols:
                        if symbol not in expr_to_proxy:
                            return True
                    return False

                # this guards against deleting calls that produce unbacked bindings we haven't yet seen.
                # in this case looking at sym_expr.free_symbols might not be enough, if the example value has a hint
                # (is backed), but produces an unbacked symbol. In this case keep the node alive.
                resolved_unbacked_bindings = resolve_unbacked_bindings(
                    shape_env, node.meta.get("unbacked_bindings", {})
                )

                def has_new_unbacked_bindings() -> bool:
                    if resolved_unbacked_bindings is None:
                        raise AssertionError("resolved_unbacked_bindings is None")
                    for key in resolved_unbacked_bindings:
                        if key not in expr_to_proxy:
                            return True
                    return False

                # maybe re-reify expression, replace current node
                if (
                    sym_expr in expr_to_proxy
                    or (  # example value is redundant
                        _is_intermediate_tensor_sym_call(node)
                        # shape call on intermediate tensor, turn into computation on input shapes
                        and not has_new_untracked_symbols()
                    )
                ) and not has_new_unbacked_bindings():
                    if _is_intermediate_tensor_sym_call(
                        node
                    ):  # reify from input shapes
                        with _set_node_metadata_hook(
                            gm,
                            functools.partial(
                                _node_metadata_hook,
                                stack_trace=node.meta.get("stack_trace"),
                                nn_module_stack=node.meta.get("nn_module_stack"),
                                # nodes added in `apply_runtime_assertion_pass` will have the same annotation
                                # as the input node to the assertion
                                custom=node.meta.get("custom"),
                            ),
                        ):
                            expr_to_proxy[sym_expr] = _sympy_interp(
                                expr_to_proxy,
                                sym_expr,
                            )  # type: ignore[arg-type]
                        # won't try DCE-ing tensor compute here
                    hash_node = expr_to_proxy[sym_expr].node  # type: ignore[arg-type]
                    node.replace_all_uses_with(hash_node)
                    gm.graph.erase_node(node)
                    log.debug(
                        "CSE node %s -> %s for expr %s",
                        node,
                        hash_node,
                        sym_expr,
                    )

                # store node in hash cons, don't delete/replace

                elif sym_expr not in expr_to_proxy and not isinstance(
                    sym_expr,
                    (sympy.Number, sympy.logic.boolalg.BooleanAtom),
                ):  # don't hash cons primitives
                    expr_to_proxy[sym_expr] = fx.Proxy(node, tracer=tracer)  # type: ignore[arg-type]

            # We add sym_constrain_range calls for symbols later in any case if they're size-like or range-constrained,
            # so calls before that are redundant.
            if node.target in (
                torch.ops.aten.sym_constrain_range.default,
                torch.ops.aten.sym_constrain_range_for_size.default,
            ):
                gm.graph.erase_node(node)

            defs = []

            # AOTAutograd will create new symbols as the unbacked_bindings keys, which PropagateSymInts will set as
            # equivalent, but the refinement calls we perform in this pass may struggle with associating the two.
            # More concretely, when re-exporting/tracing, constraining only the new symbol may not communicate enough
            # information about the old symbol when we re-export, raising errors on data-dependent guards.
            # Call resolve_unbacked_bindings() to get the original symbol if present, otherwise we take it as is.
            if unbacked_bindings := resolve_unbacked_bindings(
                shape_env, node.meta.get("unbacked_bindings")
            ):
                for s, keypath in unbacked_bindings.items():
                    defs.append(s)

                    # TODO: some CSE when generating these nodes can probably
                    # help reduce graph size and improve compile time
                    def _lower_symint_divisor(d: torch.SymInt) -> fx.Node | int:
                        try:
                            p = _sympy_interp(expr_to_proxy, d.node.expr)
                        except KeyError:
                            return graph.materialize_symint(d)
                        return p.node if isinstance(p, fx.Proxy) else p

                    if s not in expr_to_proxy:
                        with _set_node_metadata_hook(
                            gm,
                            functools.partial(
                                _node_metadata_hook,
                                stack_trace=node.meta.get("stack_trace"),
                                nn_module_stack=node.meta.get("nn_module_stack"),
                                # nodes added in `apply_runtime_assertion_pass` will have the same annotation
                                # as the input node to the assertion
                                custom=node.meta.get("custom"),
                            ),
                        ):
                            expr_to_proxy[s] = fx.Proxy(
                                graph._resolve_unbacked_binding(
                                    node,
                                    tuple(keypath),
                                    lower_symint=_lower_symint_divisor,
                                ),
                                tracer=tracer,
                            )
                        log.debug("expr_to_proxy[%s] = %s", s, expr_to_proxy[s])

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
                # with a normalization algorithm. In general, it may also
                # be on a best effort basis, but since our grammar is not
                # terribly difficult, chances are we could even fully
                # normalize SymPy expressions... who knows.
                if i0 in constrained_unbacked_symbols:
                    continue  # constrain symbol just once

                vr = shape_env.var_to_range[i0]
                if vr.is_int and vr.upper == sys.maxsize - 1:
                    # treat upper bound == sys.maxsize - 1 for int symbols as +oo
                    # to avoid redundant runtime assert
                    vr = ValueRanges(vr.lower, int_oo)
                if not shape_env._default_unspecified_value_range().issubset(vr):
                    # The runtime range is constrained, so add a runtime
                    # assert and also explicitly refine the range
                    # (refinement should not be necessary once runtime
                    # asserts cause refinement, but that's NYI)
                    def convert(s: Any) -> int | float | None:
                        if s in (sympy.oo, -sympy.oo, int_oo, -int_oo):
                            return None
                        try:
                            if vr.is_int:
                                return int(s)
                            if vr.is_float:
                                return float(s)
                        except (TypeError, OverflowError):
                            return None
                        return None

                    with _set_node_metadata_hook(
                        gm,
                        functools.partial(
                            _node_metadata_hook,
                            stack_trace=node.meta.get("stack_trace"),
                            nn_module_stack=node.meta.get("nn_module_stack"),
                            # nodes added in `apply_runtime_assertion_pass` will have the same annotation
                            # as the input node to the assertion
                            custom=node.meta.get("custom"),
                        ),
                    ):
                        if (min_val := convert(vr.lower)) is not None:
                            ge_expr = i0 >= min_val
                            if ge_expr in user_assert_exprs:
                                added_asserts.add(ge_expr)
                            else:
                                ge = _sympy_interp(expr_to_proxy, ge_expr).node
                                graph.call_function(
                                    torch.ops.aten._assert_scalar.default,
                                    (
                                        ge,
                                        f"Runtime assertion failed for expression {ge_expr} on node '{ge}'",
                                    ),
                                )
                                added_asserts.add(ge_expr)
                        if (max_val := convert(vr.upper)) is not None:
                            le_expr = i0 <= max_val
                            if le_expr in user_assert_exprs:
                                added_asserts.add(le_expr)
                            else:
                                le = _sympy_interp(expr_to_proxy, le_expr).node
                                graph.call_function(
                                    torch.ops.aten._assert_scalar.default,
                                    (
                                        le,
                                        f"Runtime assertion failed for expression {le_expr} on node '{le}'",
                                    ),
                                )
                                added_asserts.add(le_expr)

                constrained_unbacked_symbols.add(i0)
                add_runtime_asserts(ras)

    # delete unused reified symbols
    for expr, proxy in expr_to_proxy.items():
        if (
            isinstance(expr, sympy.Symbol)
            and proxy.node.op != "placeholder"  # keep placeholders intact
            and not proxy.node.users
        ):
            log.debug("deleting unused reified symbol for %s", expr)
            gm.graph.erase_node(proxy.node)
