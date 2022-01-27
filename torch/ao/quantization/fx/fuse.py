from torch.fx import (
    GraphModule,
    Node,
    map_arg
)
from torch.fx.graph import Graph
from ..utils import (
    get_combined_dict
)
from .graph_module import (
    FusedGraphModule
)
from .match_utils import is_match
from .pattern_utils import (
    get_default_fusion_patterns,
)

from .backend_config.utils import get_fusion_pattern_to_fuse_handler_cls
from .backend_config.utils import get_fuser_method_mapping

from .fusion_patterns import *  # noqa: F401,F403

from typing import Callable, Tuple, Dict, Any, Optional, List

from .quantization_types import Pattern, NodePattern

class Fuser:
    def fuse(
        self,
        model: GraphModule,
        is_qat: bool,
        fuse_custom_config_dict: Optional[Dict[str, Any]] = None,
        backend_config_dict: Optional[Dict[str, Any]] = None,
    ) -> GraphModule:
        if fuse_custom_config_dict is None:
            fuse_custom_config_dict = {}

        input_root = model
        input_graph = model.graph
        self.modules = dict(input_root.named_modules())

        if backend_config_dict is None:
            additional_fusion_patterns = \
                fuse_custom_config_dict.get("additional_fusion_pattern", {})
            fusion_pattern_to_fuse_handler_cls = get_combined_dict(
                get_default_fusion_patterns(), additional_fusion_patterns)
            fuser_method_mapping = None
        else:
            fusion_pattern_to_fuse_handler_cls = get_fusion_pattern_to_fuse_handler_cls(backend_config_dict)
            fuser_method_mapping = get_fuser_method_mapping(backend_config_dict)
        # find fusion
        fusion_pairs = self._find_matches(
            input_root, input_graph, fusion_pattern_to_fuse_handler_cls)
        self.fused_graph = Graph()
        env: Dict[Any, Any] = {}

        def load_arg(a):
            return map_arg(a, lambda node: env[node.name])

        def get_root_node(node_pattern):
            while not isinstance(node_pattern[-1], Node):
                node_pattern = node_pattern[-1]
            return node_pattern[-1]

        for node in input_graph.nodes:
            maybe_last_node, pattern, matched_node_pattern, obj = \
                fusion_pairs.get(node.name, (None, None, None, None))
            if maybe_last_node is node:
                assert obj is not None
                # TODO: currently we hard code the root node, which only works for
                # a sequence of ops and assume the root node is the last node,
                # we want to make this more general to support more complex patterns
                root_node = get_root_node(matched_node_pattern)  # type: ignore[index]
                env[node.name] = obj.fuse(
                    self, load_arg, root_node, matched_node_pattern,  # type: ignore[arg-type]
                    fuse_custom_config_dict, fuser_method_mapping, is_qat)
            elif maybe_last_node is None:
                env[node.name] = self.fused_graph.node_copy(node, load_arg)
            # node matched in patterns and is not root is removed here

        preserved_attributes = set(fuse_custom_config_dict.get("preserved_attributes", []))
        model = FusedGraphModule(input_root, self.fused_graph, preserved_attributes)
        return model

    def _find_matches(
            self, root: GraphModule, graph: Graph,
            patterns: Dict[Pattern, Callable]
    ) -> Dict[str, Tuple[Node, Pattern, NodePattern, FuseHandler]]:
        modules = dict(root.named_modules())
        match_map : Dict[str, Tuple[Node, Pattern, NodePattern, FuseHandler]] = {}  # node name -> (root_node, match_value)

        def apply_match(pattern, node, match, matched_node_pattern):
            if isinstance(pattern, tuple):
                s, *args = pattern
                current_node_pattern: List[Node] = []
                apply_match(s, node, match, current_node_pattern)
                for subpattern, arg in zip(args, node.args):
                    apply_match(subpattern, arg, match, current_node_pattern)
                matched_node_pattern.append(tuple(current_node_pattern))
            else:
                # the first pattern matches will take precedence
                if node.name not in match_map:
                    matched_node_pattern.append(node)
                    root_node, pattern, handler = match
                    match_map[node.name] = (root_node, pattern, matched_node_pattern, handler)

        for node in reversed(graph.nodes):
            if node.name not in match_map:
                for pattern, value in patterns.items():
                    matched_node_pattern: List[Node] = []
                    if is_match(modules, node, pattern):
                        apply_match(pattern, node, (node, pattern, value(self, node)), matched_node_pattern)

        return match_map
