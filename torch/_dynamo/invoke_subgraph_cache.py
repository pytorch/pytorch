"""Auto-cache logic for invoke_subgraph HOP.

Extracted from InvokeSubgraphHigherOrderVariable so the class can focus on
HOP dispatch while all cache build / lookup / stamp-out logic lives here.
"""

from __future__ import annotations

from typing import Any, NamedTuple, TYPE_CHECKING

import torch
import torch.fx
from torch._dynamo.guards import (
    _extract_tensor_metadata,
    _SKIP_GUARD,
    GUARD_VALUE_DISPATCH,
)
from torch.fx.proxy import Proxy


if TYPE_CHECKING:
    from torch._dynamo.symbolic_convert import InstructionTranslator
    from torch._dynamo.variables.base import VariableTracker
    from torch._guards import AutoCacheCondition


hc_log = torch._logging.getArtifactLogger(__name__, "hierarchical_compile")


# ---------------------------------------------------------------------------
# FlattenedArgs — structured return type for flatten_args_kwargs
# ---------------------------------------------------------------------------


class FlattenedArgs(NamedTuple):
    # (tag, VariableTracker) pairs for each leaf input.
    # Tags: "tensor", "symnode", "constant", "module".
    flat_vts: list[tuple[str, Any]]
    # fx.Node -> flat index, for O(1) deduplication of proxy nodes.
    proxy_node_to_idx: dict[torch.fx.Node, int]
    # Proxy objects in flat index order, one per unique fx.Node.
    flat_proxies: list[Proxy]
    # Source objects collected from leaf VTs that have a source.
    arg_sources: list[Any]


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def flatten_args_kwargs(
    tx: InstructionTranslator,
    fn_args_vt: Any,
    kwargs: dict[str, Any],
) -> FlattenedArgs:
    """Flatten fn_args_vt and kwargs into leaf info via VariableTracker.visit."""
    from torch._dynamo.variables.base import VariableTracker
    from torch._dynamo.variables.constant import ConstantVariable
    from torch._dynamo.variables.nn_module import UnspecializedNNModuleVariable
    from torch._dynamo.variables.tensor import SymNodeVariable, TensorVariable

    flat_vts: list[tuple[str, Any]] = []
    proxy_node_to_idx: dict[torch.fx.Node, int] = {}
    flat_proxies: list[Proxy] = []
    arg_sources: list[Any] = []

    def _collect(vt: VariableTracker) -> None:
        if isinstance(vt, TensorVariable):
            flat_vts.append(("tensor", vt))
        elif isinstance(vt, SymNodeVariable):
            flat_vts.append(("symnode", vt))
        elif isinstance(vt, ConstantVariable):
            flat_vts.append(("constant", vt))
        elif isinstance(vt, UnspecializedNNModuleVariable):
            flat_vts.append(("module", vt))
        else:
            return

        if isinstance(vt, (TensorVariable, SymNodeVariable)):
            proxy = vt.as_proxy()
            if proxy.node not in proxy_node_to_idx:
                proxy_node_to_idx[proxy.node] = len(flat_proxies)
                flat_proxies.append(proxy)

        source = getattr(vt, "source", None)
        if source is not None:
            arg_sources.append(source)

    all_vts = list(fn_args_vt) + list(kwargs.values())
    VariableTracker.visit(_collect, all_vts)
    return FlattenedArgs(flat_vts, proxy_node_to_idx, flat_proxies, arg_sources)


def get_fn_id(fn_var: Any) -> int | None:
    from torch._dynamo.variables.functions import UserFunctionVariable
    from torch._dynamo.variables.nn_module import UnspecializedNNModuleVariable

    if isinstance(fn_var, UserFunctionVariable):
        return id(fn_var.get_function())
    elif isinstance(fn_var, UnspecializedNNModuleVariable):
        return id(fn_var.value.forward.__func__)
    return None


def is_auto_cacheable(body_r: Any, flat_vts: list[tuple[str, Any]]) -> bool:
    """Best-effort check for whether a traced subgraph result can be
    auto-cached.

    It is possible that a subgraph is morally reusable but does not fall
    into the limited support that Dynamo has today. Current limitations:
      - Output must be a single tensor, or a tuple/list of plain tensors.
      - All flattened inputs must be one of: tensor, symnode, constant,
        unspecialized NN module. Pytree-registered or custom VT types
        are not yet supported.
    """
    from torch._dynamo.variables.lists import ListVariable, TupleVariable
    from torch._dynamo.variables.tensor import TensorVariable

    if isinstance(body_r, TensorVariable):
        pass
    elif isinstance(body_r, (TupleVariable, ListVariable)):
        non_tensor = [
            type(item).__name__
            for item in body_r.items
            if not isinstance(item, TensorVariable)
        ]
        if non_tensor:
            hc_log.debug(
                "auto_guard_cache: not cacheable -- output contains non-tensor types: %s",
                non_tensor,
            )
            return False
    else:
        hc_log.debug(
            "auto_guard_cache: not cacheable -- output type %s is not tensor or tuple/list",
            type(body_r).__name__,
        )
        return False

    unknown_vts = [vt for tag, vt in flat_vts if tag == "unknown"]
    if unknown_vts:
        hc_log.debug(
            "auto_guard_cache: not cacheable -- unsupported input VT types: %s",
            [type(vt).__name__ for vt in unknown_vts],
        )
        return False

    return True


def build_auto_cache_condition(
    tx: InstructionTranslator,
    flat_vts: list[tuple[str, Any]],
    arg_sources: list[Any],
    guards_before: Any,
) -> AutoCacheCondition | None:
    """Build an AutoCacheCondition from the guard delta after tracing.

    Computes guards_after - guards_before to find guards introduced by
    subgraph tracing, then filters to only those rooted at the function's
    arg sources and encodes each via the GUARD_VALUE_DISPATCH table.
    Returns None if any guard type is unsupported.
    """
    from torch._guards import AutoCacheCondition

    input_checks: list[tuple[str, Any]] = []
    for tag, vt in flat_vts:
        if tag == "tensor":
            example = vt.proxy.node.meta.get("example_value", None)
            if example is None:
                hc_log.debug(
                    "auto_guard_cache: cannot build condition -- tensor input has no example_value"
                )
                return None
            input_checks.append(("tensor", _extract_tensor_metadata(example)))
        elif tag == "symnode":
            input_checks.append(("symnode", vt.python_type()))
        elif tag == "constant":
            input_checks.append(("constant", vt.value))
        elif tag == "module":
            input_checks.append(("module", None))
        else:
            raise RuntimeError(
                f"Unexpected input tag '{tag}' for {type(vt).__name__} -- "
                f"is_auto_cacheable should have rejected this"
            )

    guards_after = tx.output.guards.inner.copy()
    delta_guards = guards_after - guards_before

    # Collect all guards for arg sources (includes pre-existing ones like
    # TENSOR_MATCH that were installed before the invoke_subgraph call).
    source_guards: set = set()
    for source in arg_sources:
        source_guards.update(tx.output.guards.get_guards_for_source(source))

    all_relevant_guards = set(delta_guards) | source_guards

    guard_tuples: list[tuple[Any, Any, Any]] = []
    for guard in all_relevant_guards:
        source = guard.originating_source
        type_str = guard.create_fn_name()
        handler = GUARD_VALUE_DISPATCH.get(type_str)

        if handler is _SKIP_GUARD:
            continue

        if handler is None:
            hc_log.debug(
                "auto_guard_cache: cannot build condition -- unsupported guard type '%s' on source '%s'",
                type_str,
                source.name,
            )
            return None

        try:
            if handler.resolve_base_only:
                value = tx.output.resolve_source_value(source.base)
            else:
                value = tx.output.resolve_source_value(source)
        except Exception:
            hc_log.debug(
                "auto_guard_cache: cannot build condition -- failed to resolve source '%s' for %s guard",
                source.name,
                type_str,
            )
            return None

        # vLLM workaround: skip CONSTANT_MATCH on strings
        if type_str == "CONSTANT_MATCH" and isinstance(value, str):
            continue

        expected = handler.extract(guard, value)
        guard_tuples.append((source, handler, expected))

    hc_log.debug("Number of guards %s", len(guard_tuples))

    return AutoCacheCondition(
        input_checks=input_checks,
        guards=guard_tuples,
    )


def is_reusable(
    tx: InstructionTranslator,
    condition: AutoCacheCondition,
    flat_vts: list[tuple[str, Any]],
    new_arg_sources: list[Any],
    cached_entry: Any,
) -> bool:
    """Check if a cache entry's conditions match the current call."""
    # Structural check: input count, tags, and metadata must match.
    # Tensor metadata (shape, stride, dtype, device, requires_grad) is checked
    # here because TENSOR_MATCH guards for subgraph inputs typically already
    # exist in the outer graph before tracing and thus won't appear in the
    # guard delta.
    if len(condition.input_checks) != len(flat_vts):
        hc_log.debug(
            "auto_guard_cache: reuse failed -- input count mismatch: cached %d vs current %d",
            len(condition.input_checks),
            len(flat_vts),
        )
        return False

    for i, ((cached_tag, cached_val), (cur_tag, cur_vt)) in enumerate(
        zip(condition.input_checks, flat_vts)
    ):
        if cached_tag != cur_tag:
            hc_log.debug(
                "auto_guard_cache: reuse failed -- input %d tag mismatch: cached '%s' vs current '%s'",
                i,
                cached_tag,
                cur_tag,
            )
            return False
        if cached_tag == "tensor":
            example = cur_vt.proxy.node.meta.get("example_value", None)
            if example is None:
                hc_log.debug(
                    "auto_guard_cache: reuse failed -- input %d tensor has no example_value",
                    i,
                )
                return False
            cur_meta = _extract_tensor_metadata(example)
            if cur_meta != cached_val:
                hc_log.debug(
                    "auto_guard_cache: reuse failed -- input %d tensor metadata mismatch",
                    i,
                )
                return False
        elif cached_tag == "symnode":
            if cur_vt.python_type() != cached_val:
                return False
        elif cached_tag == "constant":
            if cur_vt.value != cached_val:
                return False

    source_replacement = _build_source_replacement(
        cached_entry.arg_sources, new_arg_sources
    )

    def replacement_fn(s: Any) -> Any:
        return source_replacement.get(s, s)

    # Shared resolution context so source.get_value memoizes intermediate
    # results (e.g. common base sources) across all guards in this check.
    resolve_globals = {"G": tx.output.root_tx.f_globals, "L": tx.output.root_tx.f_locals}
    resolve_locals: dict = {}
    resolve_cache: dict = {}

    for source, handler, expected in condition.guards:
        if source_replacement:
            new_source = source.clone(replacement_fn)
        else:
            new_source = source

        try:
            resolve_src = new_source.base if handler.resolve_base_only else new_source
            value = resolve_src.get_value(resolve_globals, resolve_locals, resolve_cache)
        except Exception:
            hc_log.debug(
                "auto_guard_cache: reuse failed -- cannot resolve source '%s'",
                new_source.name,
            )
            return False

        if not handler.check(value, expected):
            hc_log.debug(
                "auto_guard_cache: reuse failed -- guard on '%s': expected %s, got mismatch",
                new_source.name,
                expected,
            )
            return False

    return True


def find_cache_match(
    tx: InstructionTranslator,
    fn_var: Any,
    flat_vts: list[tuple[str, Any]],
    new_arg_sources: list[Any],
) -> Any:
    from torch._guards import InvokeSubgraphCache

    invoke_subgraph_cache = tx.output.tracing_context.hop_dispatch_set_cache.get_cache(
        torch._higher_order_ops.invoke_subgraph
    )
    if not isinstance(invoke_subgraph_cache, InvokeSubgraphCache):
        return None
    fn_id = get_fn_id(fn_var)
    if fn_id is None:
        return None

    return invoke_subgraph_cache.find_auto_cache_entry(
        fn_id,
        lambda cond, entry: is_reusable(tx, cond, flat_vts, new_arg_sources, entry),
    )


def save_cache_entry(
    tx: InstructionTranslator,
    fn_var: Any,
    proxy_node_to_idx: dict[torch.fx.Node, int],
    arg_sources: list[Any],
    body_name: str,
    body_gmod: Any,
    config: Any,
    p_args: tuple,
    body_r: Any,
    example_value: Any,
    condition: AutoCacheCondition,
) -> None:
    from torch._dynamo.variables.tensor import TensorVariable
    from torch._guards import AutoCacheEntry, InvokeSubgraphCache

    invoke_subgraph_cache = tx.output.tracing_context.hop_dispatch_set_cache.get_cache(
        torch._higher_order_ops.invoke_subgraph
    )
    if not isinstance(invoke_subgraph_cache, InvokeSubgraphCache):
        return

    fn_id = get_fn_id(fn_var)
    if fn_id is None:
        return

    freevar_mapping = _build_freevar_mapping(p_args, proxy_node_to_idx)
    single_tensor_output = isinstance(body_r, TensorVariable)

    output_metadata = [
        (t.shape, t.stride(), t.dtype, t.device, t.requires_grad) for t in example_value
    ]

    entry = AutoCacheEntry(
        body_name=body_name,
        body_gmod=body_gmod,
        config=config,
        freevar_mapping=freevar_mapping,
        single_tensor_output=single_tensor_output,
        output_metadata=output_metadata,
        arg_sources=arg_sources,
    )
    invoke_subgraph_cache.add_auto_cache_entry(fn_id, condition, entry)


def stamp_out_cached_subgraph(
    tx: InstructionTranslator,
    flat: FlattenedArgs,
    cached: Any,
) -> Any:
    from torch._dynamo.variables.builder import VariableBuilder
    from torch._dynamo.variables.higher_order_ops import add_call_function, make_attr
    from torch._dynamo.variables.tensor import TensorVariable

    flat_proxies = flat.flat_proxies
    new_arg_sources = flat.arg_sources

    source_replacement = _build_source_replacement(cached.arg_sources, new_arg_sources)

    # TODO: consider extracting this placeholder-by-source lookup into a
    # utility on OutputGraph so other features can reuse it.
    existing_source_to_node: dict = {}
    for node in tx.output.graph.find_nodes(op="placeholder"):
        ga = node.meta.get("grapharg", None)
        if ga is not None and ga.source is not None:
            existing_source_to_node[ga.source] = node

    new_lifted_args = []
    for user_arg_idx, data in cached.freevar_mapping:
        if user_arg_idx >= 0:
            new_lifted_args.append(flat_proxies[user_arg_idx])
        else:
            source = data
            new_source = source
            if source_replacement:
                new_source = source.clone(lambda s: source_replacement.get(s, s))

            if new_source in existing_source_to_node:
                node = existing_source_to_node[new_source]
                new_lifted_args.append(torch.fx.Proxy(node, tx.output.current_tracer))
            else:
                value = tx.output.resolve_source_value(new_source)
                vt = VariableBuilder(tx, new_source)(value)
                new_lifted_args.append(vt.as_proxy())

    assert tx.fake_mode is not None
    with tx.fake_mode:
        example_value = tuple(
            torch.empty_strided(
                shape,
                stride,
                dtype=dtype,
                device=device,
                requires_grad=req_grad,
            )
            for shape, stride, dtype, device, req_grad in cached.output_metadata
        )

    body_node = make_attr(tx, cached.body_name)
    p_args = (body_node, cached.body_name, *new_lifted_args)
    flat_variable = add_call_function(
        tx,
        torch._higher_order_ops.invoke_subgraph,
        tuple(p_args),
        {},
        example_value,
        cached.config,
    )

    # Validate output structure matches what was cached
    if cached.single_tensor_output:
        assert isinstance(flat_variable.items[0], TensorVariable), (
            f"Expected tensor output but got {type(flat_variable.items[0]).__name__}"
        )
        return flat_variable.items[0]
    return flat_variable


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _build_source_replacement(old_sources: list[Any], new_sources: list[Any]) -> dict:
    return {old: new for old, new in zip(old_sources, new_sources) if old != new}


def _build_freevar_mapping(
    p_args: tuple, proxy_node_to_idx: dict[torch.fx.Node, int]
) -> list[tuple[int, Any]]:
    """Map each lifted freevar in p_args to a user arg index or a Source.

    For each lifted arg in p_args[2:] (skipping body_node and body_name),
    if its fx.Node matches a node in proxy_node_to_idx, it came from a
    user argument. Otherwise it's a captured variable and we store its
    Source so we can re-derive the correct proxy on cache hit.
    """
    freevar_mapping: list[tuple[int, Any]] = []
    for outer_proxy in p_args[2:]:
        matched_idx = proxy_node_to_idx.get(outer_proxy.node, -1)
        if matched_idx >= 0:
            freevar_mapping.append((matched_idx, None))
        else:
            grapharg = outer_proxy.node.meta.get("grapharg", None)
            source = grapharg.source if grapharg is not None else None
            assert source is not None, (
                f"Freevar has no source: node.op={outer_proxy.node.op} "
                f"node.name={outer_proxy.node.name} -- this likely means a "
                f"function argument was not included in the proxy matching"
            )
            freevar_mapping.append((-1, source))
    return freevar_mapping
