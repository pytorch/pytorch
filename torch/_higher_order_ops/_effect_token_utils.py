import operator
from collections.abc import Callable
from typing import Any

import torch


InvokeSubgraphTokenCount = Callable[[torch.fx.GraphModule, torch.fx.Node], int]
ExtraProducerTokenCount = Callable[
    ["EffectTokenAnalyzer", torch.fx.GraphModule, torch.fx.Node], int
]


class EffectTokenAnalyzer:
    """Find effect-token prefixes in higher-order-op FX graphs."""

    def __init__(
        self,
        invoke_subgraph_token_count: InvokeSubgraphTokenCount,
        extra_producer_token_count: ExtraProducerTokenCount | None = None,
    ) -> None:
        self.invoke_subgraph_token_count = invoke_subgraph_token_count
        self.extra_producer_token_count = extra_producer_token_count
        self.cond_token_counts: dict[torch.fx.Node, int] = {}

    @staticmethod
    def output_node(module: torch.fx.GraphModule) -> torch.fx.Node:
        output_node = next(reversed(module.graph.find_nodes(op="output")), None)
        if output_node is None:
            raise AssertionError("output node not found in graph")
        return output_node

    @classmethod
    def output_args(cls, module: torch.fx.GraphModule) -> tuple[Any, ...] | list[Any]:
        outs = cls.output_node(module).args[0]
        if not isinstance(outs, (tuple, list)):
            raise AssertionError(f"expected output sequence, got {type(outs)}")
        return outs

    @staticmethod
    def _getitem_source_and_index(
        node: torch.fx.Node,
    ) -> tuple[torch.fx.Node, int] | None:
        if (
            node.op == "call_function"
            and node.target is operator.getitem
            and len(node.args) > 1
            and isinstance(node.args[0], torch.fx.Node)
            and isinstance(node.args[1], int)
        ):
            return node.args[0], node.args[1]
        return None

    def _producer_token_count(
        self, module: torch.fx.GraphModule, producer: torch.fx.Node
    ) -> int:
        if producer.op != "call_function":
            return 0
        if producer.target is torch.ops.higher_order.with_effects:
            return 1
        if producer.target is torch.ops.higher_order.invoke_subgraph:
            return self.invoke_subgraph_token_count(module, producer)
        if producer.target is torch.ops.higher_order.cond:
            return self.cond_token_count(module, producer)
        if self.extra_producer_token_count is not None:
            return self.extra_producer_token_count(self, module, producer)
        return 0

    def is_definite_token_output(
        self, module: torch.fx.GraphModule, node: torch.fx.Node
    ) -> bool:
        getitem = self._getitem_source_and_index(node)
        if getitem is None:
            return False
        producer, index = getitem
        return 0 <= index < self._producer_token_count(module, producer)

    def cond_token_count(
        self, module: torch.fx.GraphModule, node: torch.fx.Node
    ) -> int:
        cached = self.cond_token_counts.get(node)
        if cached is not None:
            return cached

        if (
            len(node.args) < 4
            or not isinstance(node.args[1], torch.fx.Node)
            or not isinstance(node.args[2], torch.fx.Node)
            or node.args[1].op != "get_attr"
            or node.args[2].op != "get_attr"
            or not isinstance(node.args[1].target, str)
            or not isinstance(node.args[2].target, str)
        ):
            raise AssertionError(f"malformed cond node: {node}")

        definite_token_indices: set[int] = set()
        for branch_node in node.args[1:3]:
            if not isinstance(branch_node, torch.fx.Node):
                raise AssertionError(f"expected cond branch node, got {branch_node}")
            if not isinstance(branch_node.target, str):
                raise AssertionError(
                    f"expected cond branch target string, got {branch_node.target}"
                )
            branch = module.get_submodule(branch_node.target)
            if not isinstance(branch, torch.fx.GraphModule):
                raise AssertionError(
                    f"expected cond branch to be a GraphModule, got {type(branch)}"
                )
            for index, out in enumerate(self.output_args(branch)):
                if isinstance(out, torch.fx.Node) and self.is_definite_token_output(
                    branch, out
                ):
                    definite_token_indices.add(index)

        num_tokens = 0
        while num_tokens in definite_token_indices:
            num_tokens += 1
        self.cond_token_counts[node] = num_tokens
        return num_tokens

    def passthrough_cond_tokens(
        self, module: torch.fx.GraphModule, num_tokens: int
    ) -> set[torch.fx.Node]:
        return {
            out
            for out in self.output_args(module)[:num_tokens]
            if isinstance(out, torch.fx.Node) and out.op == "placeholder"
        }
