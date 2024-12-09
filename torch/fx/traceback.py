# mypy: allow-untyped-defs
import copy
import json
import traceback
from contextlib import contextmanager
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from ._compatibility import compatibility
from .graph import Graph
from .node import Node


__all__ = [
    "preserve_node_meta",
    "has_preserved_node_meta",
    "set_stack_trace",
    "set_grad_fn_seq_nr",
    "reset_grad_fn_seq_nr",
    "format_stack",
    "set_current_meta",
    "get_current_meta",
    "NodeSource",
    "NodeSourceAction",
    "get_graph_provenance_json",
]

current_meta: Dict[str, Any] = {}
should_preserve_node_meta = False


@compatibility(is_backward_compatible=False)
class NodeSourceAction(Enum):
    CREATE = "create"
    REPLACE = "replace"


@compatibility(is_backward_compatible=False)
class NodeSource:
    """
    NodeSource is a data structure that contains the provenance information of a node.
    If node `a` is created from node `b`, then `a.meta["from_node"]` may contain NodeSource(b).
    """

    class NodeInfo:
        def __init__(self, name: str, target: str, graph_id: int):
            self.name = name
            self.target = target
            self.graph_id = graph_id

    pass_name: str
    action: List["NodeSourceAction"]
    from_node: List["NodeSource"]
    node_info: Optional["NodeInfo"]

    def __init__(
        self,
        node: Optional[Node],
        pass_name: str = "",
        action: Optional[Union["NodeSourceAction", List["NodeSourceAction"]]] = None,
    ):
        self.pass_name = pass_name

        if action is None:
            action = []
        elif not isinstance(action, list):
            action = [action]
        for a in action:
            assert isinstance(a, NodeSourceAction)
        self.action = action
        if node:
            self.node_info = self.NodeInfo(
                name=node.name, target=str(node.target), graph_id=id(node.graph)
            )
            self.from_node = (
                copy.deepcopy(node.meta["from_node"])
                if "from_node" in node.meta
                else []
            )
        else:
            self.node_info = None
            self.from_node = []

    @property
    def name(self) -> str:
        return self.node_info.name if self.node_info else ""

    @property
    def target(self) -> str:
        return self.node_info.target if self.node_info else ""

    @property
    def graph_id(self) -> int:
        return self.node_info.graph_id if self.node_info else -1

    def __repr__(self):
        return self.print_readable()

    def _get_action_string(self):
        return "+".join([a.name.lower() for a in self.action])

    def print_readable(self, indent=0):
        if indent > 9:
            return ""
        result = ""
        action_string = self._get_action_string()
        result += (
            " " * indent * 4
            + f"(name={self.name}, pass_name={self.pass_name}, action={action_string}, graph_id={self.graph_id})\n"
        )
        for item in self.from_node:
            result += item.print_readable(indent + 1)
        return result

    def to_dict(self) -> dict:
        # Convert the object to a dictionary
        action_string = self._get_action_string()
        return {
            "name": self.name,
            "target": self.target,
            "graph_id": self.graph_id,
            "pass_name": self.pass_name,
            "action": action_string,
            "from_node": [node.to_dict() for node in self.from_node],
        }


@compatibility(is_backward_compatible=False)
@contextmanager
def preserve_node_meta():
    global should_preserve_node_meta
    global current_meta

    saved_should_preserve_node_meta = should_preserve_node_meta
    # Shallow copy is OK since fields of current_meta are not mutated
    saved_current_meta = current_meta.copy()
    try:
        should_preserve_node_meta = True
        yield
    finally:
        should_preserve_node_meta = saved_should_preserve_node_meta
        current_meta = saved_current_meta


@compatibility(is_backward_compatible=False)
def set_stack_trace(stack: List[str]):
    global current_meta

    if should_preserve_node_meta and stack:
        current_meta["stack_trace"] = "".join(stack)


@compatibility(is_backward_compatible=False)
def set_grad_fn_seq_nr(seq_nr):
    global current_meta

    if should_preserve_node_meta:
        # The seq_nr is captured by eager mode in the grad_fn during forward
        current_meta["grad_fn_seq_nr"] = current_meta.get("grad_fn_seq_nr", []) + [
            seq_nr
        ]
        current_meta["in_grad_fn"] = current_meta.get("in_grad_fn", 0) + 1


@compatibility(is_backward_compatible=False)
def reset_grad_fn_seq_nr():
    # NB: reset state properly, this would be helpful towards supporting
    #     reentrant autograd if we actually wanted to do that.
    global current_meta
    if should_preserve_node_meta:
        current_level = current_meta.get("in_grad_fn", 0)
        assert current_level > 0
        if current_level == 1:
            del current_meta["in_grad_fn"]
            del current_meta["grad_fn_seq_nr"]
        else:
            current_meta["in_grad_fn"] = current_level - 1
            current_meta["grad_fn_seq_nr"] = current_meta["grad_fn_seq_nr"][:-1]


@compatibility(is_backward_compatible=False)
def format_stack() -> List[str]:
    if should_preserve_node_meta:
        return [current_meta.get("stack_trace", "")]
    else:
        # fallback to traceback.format_stack()
        return traceback.format_list(traceback.extract_stack()[:-1])


@compatibility(is_backward_compatible=False)
def has_preserved_node_meta() -> bool:
    return should_preserve_node_meta


@compatibility(is_backward_compatible=False)
@contextmanager
def set_current_meta(node, pass_name=""):
    global current_meta
    if should_preserve_node_meta and node.meta:
        saved_meta = current_meta
        try:
            current_meta = node.meta.copy()

            # Update the "from_node" field in current_meta for provenance tracking.
            # Instead of appending, overwrite the "from_node" field because current_meta
            # will be assigned to the new node. The new NodeSource(node, ...) will
            # include the information from the previous current_meta["from_node"].
            current_meta["from_node"] = [
                NodeSource(node, pass_name, NodeSourceAction.CREATE)
            ]
            yield
        finally:
            current_meta = saved_meta
    else:
        yield


@compatibility(is_backward_compatible=False)
def get_current_meta() -> Dict[str, Any]:
    return current_meta


@compatibility(is_backward_compatible=False)
def get_graph_provenance_json(graph: Graph) -> str:
    """
    Given an fx.Graph, return a json string that contains the provenance information of each node.
    """
    provenance_tracking_json = {}
    for node in graph.nodes:
        if node.op == "call_function":
            provenance_tracking_json[node.name] = (
                [source.to_dict() for source in node.meta["from_node"]]
                if "from_node" in node.meta
                else []
            )
    return json.dumps(provenance_tracking_json)
