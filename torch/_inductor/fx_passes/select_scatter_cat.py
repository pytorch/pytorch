import torch
from torch._dynamo.utils import counters
from torch._inductor.pattern_matcher import (
    CallFunction,
    FailedMatch,
    Ignored,
    Match,
    MatchContext,
    MatchResult,
    MULTIPLE,
    PatternMatcherPass,
    register_graph_pattern,
)
from torch.utils._ordered_set import OrderedSet

from .reinplace import graph_call_function


def _validate_chain(
    chain: list[torch.fx.Node],
) -> tuple[torch.fx.Node, dict[int, torch.fx.Node]] | None:
    base = chain[-1].args[0]
    if not isinstance(base, torch.fx.Node):
        return None
    value = base.meta.get("val")
    if (
        not isinstance(value, torch.Tensor)
        or value.ndim != 3
        or any(type(size) is not int for size in value.shape)
        or not value.is_contiguous()
    ):
        return None

    sources = {}
    for node in chain:
        if len(node.args) != 4 or node.kwargs:
            return None
        _, source, dim, index = node.args
        if type(dim) is not int or dim != 0 or type(index) is not int:
            return None
        index = index + value.shape[0] if index < 0 else index
        if not 0 <= index < value.shape[0] or index in sources:
            return None
        if not isinstance(source, torch.fx.Node):
            return None
        source_value = source.meta.get("val")
        if (
            not isinstance(source_value, torch.Tensor)
            or any(type(size) is not int for size in source_value.shape)
            or source_value.shape != value.shape[1:]
            or source_value.dtype != value.dtype
            or source_value.device != value.device
        ):
            return None
        sources[index] = source

    forbidden = OrderedSet([*chain, base])
    pending = list(sources.values())
    visited = OrderedSet()
    while pending:
        node = pending.pop()
        if node in forbidden:
            return None
        if node not in visited:
            visited.add(node)
            pending.extend(node.all_input_nodes)

    if len(sources) != value.shape[0]:
        zero_base = base.target is torch.ops.aten.zeros.default or (
            base.target is torch.ops.aten.full.default
            and len(base.args) >= 2
            and isinstance(base.args[1], (int, float))
            and base.args[1] == 0
        )
        if not zero_base:
            return None
    return base, sources


class _SelectScatterChainPattern(CallFunction):
    def __init__(self) -> None:
        super().__init__(
            torch.ops.aten.select_scatter.default,
            Ignored(),
            Ignored(),
            0,
            Ignored(),
            _users=MULTIPLE,
        )

    def _match(self, node: torch.fx.Node, ctx: MatchContext) -> MatchResult:
        match = super()._match(node, ctx)
        if not isinstance(match, Match):
            return match
        aten = torch.ops.aten
        if any(
            user.target is aten.select_scatter.default
            and user.args
            and user.args[0] is node
            for user in node.users
        ):
            return FailedMatch("not a scatter chain tail")

        chain = []
        while node.target is aten.select_scatter.default and node.args:
            chain.append(node)
            base = node.args[0]
            if (
                not isinstance(base, torch.fx.Node)
                or base.target is not aten.select_scatter.default
                or len(base.users) != 1
            ):
                break
            node = base
        if (validated := _validate_chain(chain)) is None:
            return FailedMatch("unsupported scatter chain")

        base, sources = validated
        match.nodes = list(reversed(chain))
        match.kwargs = {"base": base, "sources": sources}
        return match


_patterns = PatternMatcherPass(pass_name="select_scatter_cat")


@register_graph_pattern(_SelectScatterChainPattern(), pass_dict=_patterns)
def _rewrite_chain(
    match: Match, *, base: torch.fx.Node, sources: dict[int, torch.fx.Node]
) -> None:
    graph = match.graph
    aten = torch.ops.aten
    pieces = [
        graph_call_function(graph, aten.unsqueeze.default, sources[index], 0)
        if index in sources
        else graph_call_function(graph, aten.slice.Tensor, base, 0, index, index + 1)
        for index in range(base.meta["val"].shape[0])
    ]
    result = graph_call_function(graph, aten.cat.default, pieces, 0)
    match.output_node().replace_all_uses_with(result)
    match.erase_nodes()


def select_scatter_cat_pass(graph: torch.fx.Graph) -> bool:
    rewrites = _patterns.apply(graph)
    if rewrites:
        graph.eliminate_dead_code()
        graph.lint()
        counters["inductor"]["select_scatter_cat"] += rewrites
    return rewrites != 0
