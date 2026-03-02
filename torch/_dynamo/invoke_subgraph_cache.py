"""Auto-cache logic for invoke_subgraph HOP.

Extracted from InvokeSubgraphHigherOrderVariable so the class can focus on
HOP dispatch while all cache build / lookup / stamp-out logic lives here.
"""

from __future__ import annotations

from typing import Any, NamedTuple, TYPE_CHECKING

import torch
import torch.fx
from torch.fx.proxy import Proxy


if TYPE_CHECKING:
    from torch._dynamo.symbolic_convert import InstructionTranslator
    from torch._dynamo.variables.base import VariableTracker


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


def is_auto_cacheable(
    body_r: Any, flat_vts: list[tuple[str, Any]], has_side_effect: bool
) -> bool:
    """Best-effort check for whether a traced subgraph result can be
    auto-cached.

    It is possible that a subgraph is morally reusable but does not fall
    into the limited support that Dynamo has today. Current limitations:
      - The subgraph must not have side effects.
      - Output must be a single tensor, or a tuple/list of plain tensors.
      - All flattened inputs must be one of: tensor, symnode, constant,
        unspecialized NN module. Pytree-registered or custom VT types
        are not yet supported.
    """
    from torch._dynamo.variables.lists import ListVariable, TupleVariable
    from torch._dynamo.variables.tensor import TensorVariable

    if has_side_effect:
        hc_log.debug(
            "auto_guard_cache: not cacheable -- subgraph has side effects",
        )
        return False

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
