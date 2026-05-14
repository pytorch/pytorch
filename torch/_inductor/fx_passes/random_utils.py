# mypy: allow-untyped-defs
import torch
from torch.nn import functional as F
from torch.utils._ordered_set import OrderedSet

from .control_dependencies import preserve_node_ordering


FALLBACK_RANDOM_FOR_FRACTIONAL_POOL_KEY = "fallback_random_for_fractional_pool"

_FRACTIONAL_MAX_POOL_RANDOM_SAMPLE_OPS = (
    torch.ops.aten.fractional_max_pool2d.default,
    torch.ops.aten.fractional_max_pool3d.default,
)
_FRACTIONAL_MAX_POOL_FUNCTIONAL_OPS = (
    F.fractional_max_pool2d,
    F.fractional_max_pool2d_with_indices,
    F.fractional_max_pool3d,
    F.fractional_max_pool3d_with_indices,
)


def _fractional_pool_random_samples(node: torch.fx.Node):
    if node.op != "call_function":
        return None

    if node.target in _FRACTIONAL_MAX_POOL_RANDOM_SAMPLE_OPS:
        sample = node.args[3] if len(node.args) >= 4 else None
        if sample is None:
            sample = node.kwargs.get("random_samples")
        if sample is None:
            sample = node.kwargs.get("_random_samples")
        return sample

    if node.target in _FRACTIONAL_MAX_POOL_FUNCTIONAL_OPS:
        sample = node.args[5] if len(node.args) >= 6 else None
        if sample is None:
            sample = node.kwargs.get("_random_samples")
        if sample is None:
            sample = node.kwargs.get("random_samples")
        return sample

    return None


def is_rand_node(node: torch.fx.Node) -> bool:
    if node.op != "call_function":
        return False

    if node.target is torch.rand:
        return True

    return (
        isinstance(node.target, torch._ops.OpOverload)
        and node.target.overloadpacket is torch.ops.aten.rand
    )


def is_nondeterministic_seeded_node(node: torch.fx.Node) -> bool:
    return (
        node.op == "call_function"
        and isinstance(node.target, torch._ops.OpOverload)
        and torch.Tag.nondeterministic_seeded in node.target.tags
    )


def has_fractional_pool_sample_rand(graph: torch.fx.Graph) -> bool:
    return any(
        is_rand_node(node)
        and any(_fractional_pool_random_samples(user) is node for user in node.users)
        for node in graph.nodes
    )


def _fractional_pool_has_explicit_samples(node: torch.fx.Node) -> bool:
    return _fractional_pool_random_samples(node) is not None


def has_fractional_pool_implicit_random(graph: torch.fx.Graph) -> bool:
    if has_fractional_pool_sample_rand(graph):
        return True
    return any(
        node.op == "call_function"
        and node.target in _FRACTIONAL_MAX_POOL_FUNCTIONAL_OPS
        and not _fractional_pool_has_explicit_samples(node)
        for node in graph.nodes
    )


def uses_fallback_random_for_fractional_pool(graph: torch.fx.Graph) -> bool:
    return any(
        node.meta.get(FALLBACK_RANDOM_FOR_FRACTIONAL_POOL_KEY) for node in graph.nodes
    )


def mark_fallback_random_for_fractional_pool(node: torch.fx.Node) -> None:
    node.meta[FALLBACK_RANDOM_FOR_FRACTIONAL_POOL_KEY] = True


def chain_random_ops_for_ordering(
    graph: torch.fx.Graph, *, mark_fallback_random: bool = False
) -> None:
    random_nodes = [
        node for node in graph.nodes if is_nondeterministic_seeded_node(node)
    ]
    if mark_fallback_random:
        for node in random_nodes:
            mark_fallback_random_for_fractional_pool(node)

    if len(random_nodes) < 2:
        return

    additional_deps_map: dict[torch.fx.Node, OrderedSet[torch.fx.Node]] = {}
    for i in range(1, len(random_nodes)):
        additional_deps_map[random_nodes[i]] = OrderedSet([random_nodes[i - 1]])

    preserve_node_ordering(graph, additional_deps_map)
