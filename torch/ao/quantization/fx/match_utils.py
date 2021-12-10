import sys
import torch
from torch.fx.graph import (
    Graph,
    Node,
)
from .quantization_types import Pattern
from .quantization_patterns import (
    QuantizeHandler,
    CustomModuleQuantizeHandler,
    StandaloneModuleQuantizeHandler,
)
from ..qconfig import (
    QConfigAny,
)
from .graph_module import (
    is_observed_standalone_module,
)

from typing import Any, Dict, List, Callable, Optional, Tuple, Set

MatchResult = Tuple[Node, List[Node], Optional[Pattern], QuantizeHandler,
                    QConfigAny]

# TODO: maybe rename this to MatchInputNode
class MatchAllNode:
    """ A node pattern that matches all nodes
    """
    pass

# Note: The order of patterns is important! match function will take whatever is matched first, so we'll
# need to put the fusion patterns before single patterns. For example, add_relu should be registered come before relu.
# decorators are applied in the reverse order we see. Also when we match the nodes in the graph with these patterns,
# we'll start from the last node of the graph and traverse back.
def is_match(modules, node, pattern, max_uses=sys.maxsize):
    """ Matches a node in fx against a pattern
    """
    if isinstance(pattern, tuple):
        self_match, *arg_matches = pattern
        if self_match is getattr:
            assert len(pattern) == 2, 'Expecting getattr pattern to have two elements'
            arg_matches = []
    else:
        self_match = pattern
        arg_matches = []

    print("node:", node.format_node())

    if isinstance(self_match, type) and issubclass(self_match, MatchAllNode):
        return True

    if len(node.users) > max_uses:
        print("returnning false 1")
        return False

    if isinstance(self_match, type) and issubclass(self_match, torch.nn.Module):
        if node.op != 'call_module':
            print("returnning false 2")
            return False
        if not type(modules[node.target]) == self_match:
            print("returnning false 3")
            return False
    elif callable(self_match):
        if node.op != 'call_function' or node.target is not self_match:
            print("returnning false 4")
            return False
        elif node.target is getattr:
            if node.args[1] != pattern[1]:
                print("returnning false 5")
                return False
    elif isinstance(self_match, str):
        if node.op != 'call_method' or node.target != self_match:
            print("returnning false 6")
            return False
    elif node.target != self_match:
        print("returnning false 7")
        return False

    if not arg_matches:
        print("returnning false 8")
        return True

    if len(arg_matches) != len(node.args):
        print("arg matches:", arg_matches)
        print("node args:", list(node.args))
        print("returnning false 9")
        return False

    return all(is_match(modules, node, arg_match, max_uses=1) for node, arg_match in zip(node.args, arg_matches))

def find_matches(
        graph: Graph,
        modules: Dict[str, torch.nn.Module],
        patterns: Dict[Pattern, QuantizeHandler],
        qconfig_map: Dict[str, QConfigAny],
        standalone_module_names: List[str] = None,
        standalone_module_classes: List[Callable] = None,
        custom_module_classes: List[Any] = None) -> Dict[str, MatchResult]:
    """
    Matches the nodes in the input graph to quantization patterns, and
    outputs the information needed to quantize them in future steps.

    Inputs:
      - graph: an fx.Graph object
      - modules: a mapping of fully qualified module name to instance,
          for example, {'foo': ModuleFoo, ...}
      - patterns: a mapping from a tuple of nodes in reverse order to
          uninitialized QuantizeHandler subclass.

    Outputs a map of
      node_name ->
        (node, matched_values, matched_pattern, QuantizeHandler instance,
         qconfig)

    For example, {
      'relu_1': (relu_1, [relu_1], torch.nn.functional.relu,
                 <CopyNodeQuantizeHandler instance>, QConfig(...)),
      ...
    }
    """
    if custom_module_classes is None:
        custom_module_classes = []

    if standalone_module_classes is None:
        standalone_module_classes = []

    if standalone_module_names is None:
        standalone_module_names = []

    match_map: Dict[str, MatchResult] = {}
    all_matched : Set[str] = set()

    def record_match(pattern, node, matched):
        if isinstance(pattern, tuple):
            s, *args = pattern
            record_match(s, node, matched)
            if pattern[0] is not getattr:
                for subpattern, arg in zip(args, node.args):
                    record_match(subpattern, arg, matched)
        else:
            matched.append(node)

    cache_for_no_tensor_check: Dict[Node, bool] = dict()
    for node in reversed(graph.nodes):
        if node.name not in match_map and node.name not in all_matched:
            for pattern, value in patterns.items():
                print("matching:", node, pattern)
                print("is match:", is_match(modules, node, pattern))
                if is_match(modules, node, pattern):
                    matched: List[Any] = []
                    record_match(pattern, node, matched)
                    for n in matched:
                        match_map[n.name] = (
                            node, matched, pattern, value(node, modules),  # type: ignore[operator]
                            qconfig_map[n.name])
                        all_matched.add(n.name)
                    # break after finding the first match
                    break

    # add custom module instances to the match result
    assert modules is not None
    for node in graph.nodes:
        if node.op == 'call_module' and \
           type(modules[node.target]) in custom_module_classes:
            custom_module_qconfig = qconfig_map[node.name]
            match_map[node.name] = (
                node, [node], None, CustomModuleQuantizeHandler(node, modules),
                custom_module_qconfig)

    def is_standalone_module(node_target: str, modules: Dict[str, torch.nn.Module]):
        assert modules is not None
        return (
            node_target in standalone_module_names or  # type: ignore[operator]
            type(modules[node_target]) in standalone_module_classes  # type: ignore[operator]
        )

    # add standalone modules to the match
    for node in graph.nodes:
        if node.op == 'call_module' and \
           (is_standalone_module(node.target, modules) or
                is_observed_standalone_module(modules[node.target])):
            # add node to matched nodes
            custom_module_qconfig = qconfig_map[node.name]
            match_map[node.name] = (
                node, [node], None,
                StandaloneModuleQuantizeHandler(node, modules),
                custom_module_qconfig)

    return match_map
