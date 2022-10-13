from torch.fx import (
    GraphModule,
    Node,
    map_arg
)
from torch.fx.graph import Graph
from .graph_module import (
    FusedGraphModule
)
from .match_utils import (
    is_match,
    MatchAllNode,
)
from .pattern_utils import (
    sorted_patterns_dict,
)

from ..backend_config import (
    BackendConfig,
    get_native_backend_config,
)
from ..backend_config.utils import (
    get_fuser_method_mapping,
    get_fusion_pattern_to_root_node_getter,
    get_fusion_pattern_to_extra_inputs_getter,
)
from .backend_config_utils import get_fusion_pattern_to_fuse_handler_cls

from .custom_config import FuseCustomConfig

from .fusion_patterns import *  # noqa: F401,F403

from typing import Any, Callable, Dict, List, Tuple, Union
import warnings

from torch.ao.quantization.utils import Pattern, NodePattern


__all__ = [
    "fuse",
]


def fuse(
    model: GraphModule,
    is_qat: bool,
    fuse_custom_config: Union[FuseCustomConfig, Dict[str, Any], None] = None,
    backend_config: Union[BackendConfig, Dict[str, Any], None] = None,
) -> GraphModule:
    if fuse_custom_config is None:
        fuse_custom_config = FuseCustomConfig()

    if isinstance(fuse_custom_config, Dict):
        warnings.warn(
            "Passing a fuse_custom_config_dict to fuse is deprecated and will not be supported "
            "in a future version. Please pass in a FuseCustomConfig instead.")
        fuse_custom_config = FuseCustomConfig.from_dict(fuse_custom_config)

    if isinstance(backend_config, Dict):
        warnings.warn(
            "Passing a backend_config_dict to prepare is deprecated and will not be supported "
            "in a future version. Please pass in a BackendConfig instead.")
        backend_config = BackendConfig.from_dict(backend_config)

    input_root = model
    input_graph = model.graph
    named_modules = dict(input_root.named_modules())

    if backend_config is None:
        backend_config = get_native_backend_config()

    fusion_pattern_to_fuse_handler_cls = sorted_patterns_dict(get_fusion_pattern_to_fuse_handler_cls(backend_config))
    fuser_method_mapping = get_fuser_method_mapping(backend_config)
    fusion_pattern_to_root_node_getter = get_fusion_pattern_to_root_node_getter(backend_config)
    fusion_pattern_to_extra_inputs_getter = get_fusion_pattern_to_extra_inputs_getter(backend_config)

    # find fusion
    fusion_pairs = _find_matches(
        input_root, input_graph, fusion_pattern_to_fuse_handler_cls)
    fused_graph = Graph()
    env: Dict[Any, Any] = {}

    def load_arg(a):
        return map_arg(a, lambda node: env[node.name])

    def default_root_node_getter(node_pattern):
        while not isinstance(node_pattern[-1], Node):
            node_pattern = node_pattern[-1]
        return node_pattern[-1]

    for node in input_graph.nodes:
        maybe_last_node, pattern, matched_node_pattern, obj, node_to_subpattern = \
            fusion_pairs.get(node.name, (None, None, None, None, None))
        # get the corresponding subpattern for the current node
        if node_to_subpattern is not None:
            node_subpattern = node_to_subpattern.get(node, None)
        else:
            node_subpattern = None
        if maybe_last_node is node:
            assert obj is not None
            root_node_getter = fusion_pattern_to_root_node_getter.get(pattern, default_root_node_getter)
            root_node = root_node_getter(matched_node_pattern)  # type: ignore[index]
            extra_inputs_getter = fusion_pattern_to_extra_inputs_getter.get(pattern, None)
            extra_inputs = []
            if extra_inputs_getter is not None:
                extra_inputs = extra_inputs_getter(matched_node_pattern)
            # TODO: add validation that root_node is a module and has the same type
            # as the root_module in the configuration
            env[node.name] = obj.fuse(
                load_arg, named_modules, fused_graph, root_node, extra_inputs, matched_node_pattern,  # type: ignore[arg-type]
                fuse_custom_config, fuser_method_mapping, is_qat)
        elif maybe_last_node is None or node_subpattern is MatchAllNode:
            env[node.name] = fused_graph.node_copy(node, load_arg)
        # node matched in patterns and is not root is removed here

    preserved_attributes = set(fuse_custom_config.preserved_attributes)
    model = FusedGraphModule(input_root, fused_graph, preserved_attributes)
    return model

def _find_matches(
        root: GraphModule, graph: Graph,
        patterns: Dict[Pattern, Callable]
) -> Dict[str, Tuple[Node, Pattern, NodePattern, FuseHandler, Dict[Node, Any]]]:
    modules = dict(root.named_modules())
    # node name -> (root_node, match_value)
    match_map : Dict[
        str, Tuple[Node, Pattern, NodePattern, FuseHandler, Dict[Node, Any]]] = {}
    # a map from node to the matched subpattern
    node_to_subpattern: Dict[Node, Any] = {}

    # TODO: dedup with quantization matching function in match_utils.py
    def apply_match(pattern, node, match, matched_node_pattern, node_to_subpattern):
        if isinstance(pattern, tuple):
            s, *args = pattern
            current_node_pattern: List[Node] = []
            apply_match(s, node, match, current_node_pattern, node_to_subpattern)
            for subpattern, arg in zip(args, node.args):
                apply_match(subpattern, arg, match, current_node_pattern, node_to_subpattern)
            matched_node_pattern.append(tuple(current_node_pattern))
        else:
            # the first pattern matches will take precedence
            if node.name not in match_map:
                matched_node_pattern.append(node)
                # MatchAllNode here is actually MatchAllInputNode which should not
                # be added to match_map
                if pattern is not MatchAllNode:
                    node_to_subpattern[node] = pattern
                    root_node, pattern, handler = match
                    match_map[node.name] = (root_node, pattern, matched_node_pattern, handler, node_to_subpattern)

    for node in reversed(graph.nodes):
        if node.name not in match_map:
            for pattern, value in patterns.items():
                matched_node_pattern: List[Node] = []
                if is_match(modules, node, pattern):
                    apply_match(pattern, node, (node, pattern, value(node)), matched_node_pattern, node_to_subpattern)
                    break

    return match_map
