from typing import Dict, Any

from torch.fx import (
    GraphModule,
    Node,
    map_arg
)

from torch.fx.graph import Graph

from ..utils import (
    get_combined_dict
)

from .pattern_utils import (
    get_default_fusion_patterns,
)

from .match_utils import is_match

from .graph_module import (
    FusedGraphModule
)

from .fusion_patterns import *  # noqa: F401,F403

from .quantization_types import Pattern

from typing import Callable, Tuple


class Fuser:
    def fuse(self, model: GraphModule,
             fuse_custom_config_dict: Dict[str, Any] = None) -> GraphModule:
        if fuse_custom_config_dict is None:
            fuse_custom_config_dict = {}

        input_root = model
        input_graph = model.graph
        self.modules = dict(input_root.named_modules())

        additional_fusion_patterns = \
            fuse_custom_config_dict.get("additional_fusion_pattern", {})
        fusion_patterns = get_combined_dict(
            get_default_fusion_patterns(), additional_fusion_patterns)
        # find fusion
        fusion_pairs = self._find_matches(
            input_root, input_graph, fusion_patterns)
        self.fused_graph = Graph()
        env: Dict[Any, Any] = {}

        def load_arg(a):
            return map_arg(a, lambda node: env[node.name])

        for node in input_graph.nodes:
            root_node, obj = fusion_pairs.get(node.name, (None, None))
            if root_node is node:
                assert obj is not None
                env[node.name] = obj.fuse(self, load_arg)
            elif root_node is None:
                env[node.name] = self.fused_graph.node_copy(node, load_arg)
            # node matched in patterns and is not root is removed here

        preserved_attributes = set(fuse_custom_config_dict.get("preserved_attributes", []))
        model = FusedGraphModule(input_root, self.fused_graph, preserved_attributes)
        return model

    def _find_matches(
            self, root: GraphModule, graph: Graph,
            patterns: Dict[Pattern, Callable]
    ) -> Dict[str, Tuple[Node, FuseHandler]]:
        modules = dict(root.named_modules())
        match_map : Dict[str, Tuple[Node, FuseHandler]] = {}  # node name -> (root_node, match_value)

        def apply_match(pattern, node, match):
            if isinstance(pattern, tuple):
                s, *args = pattern
                apply_match(s, node, match)
                for subpattern, arg in zip(args, node.args):
                    apply_match(subpattern, arg, match)
            else:
                # the first pattern matches will take precedence
                if node.name not in match_map:
                    match_map[node.name] = match

        for node in reversed(graph.nodes):
            if node.name not in match_map:
                for pattern, value in patterns.items():
                    if is_match(modules, node, pattern):
                        apply_match(pattern, node, (node, value(self, node)))

        return match_map
