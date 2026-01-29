# mypy: allow-untyped-defs
import copy
import logging
import traceback
import warnings
from contextlib import contextmanager
from enum import Enum
from typing import Any, Optional, Union

from torch._utils_internal import signpost_event

from ._compatibility import compatibility
from .graph import Graph
from .graph_module import GraphModule
from .node import Node


log = logging.getLogger(__name__)

__all__ = [
    "annotate",
    "annotate_fn",
    "annotate_rqn",
    "annotate_rqn_fn",
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
    "set_current_replay_node",
    "get_current_replay_node",
    "get_current_rqn",
    "reset_current_annotation",
]

current_meta: dict[str, Any] = {}
current_replay_node: Optional[Node] = None
should_preserve_node_meta = False


# If you change the key here, you should also change
# _COPY_META_FIELDS in torch/fx/proxy.py

# RQN stands for relative qualified name. The qualified name is relative to a root.
RQN_ANNOTATION_KEY = "annotated_rqn"
COMPILE_RQN_ANNOTATION_KEY = "_pt2_compiled_region"
RQN_DELIMITER = "."

# List of metadata keys used for annotations
# These keys are propagated through various compilation passes
ANNOTATION_META_KEYS = ["custom", RQN_ANNOTATION_KEY]


GRADIENT_ACC_SPECIAL_STACK = (
    "Gradient addition node due to multiple use of tensor around:"
)
# =============================================================================
# FX Metadata Registry for Memory Profiler
# =============================================================================
# Global in-memory registry for FX metadata
# Maps module_name -> metadata dict containing lineno_map and node_metadata
_FX_METADATA_REGISTRY: dict[str, dict[str, Any]] = {}


def _register_fx_metadata(module_name: str, metadata: dict[str, Any]) -> None:
    """
    Register FX metadata in the global in-memory registry.

    This is called automatically during graph module compilation to store metadata
    for later use by memory profiler augmentation.

    Args:
        module_name: The module identifier (content-addressed filename)
        metadata: Metadata dict containing lineno_map, node_metadata, and source_code
    """
    # TODO: add logging to tlparse
    _FX_METADATA_REGISTRY[module_name] = metadata


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
    action: list["NodeSourceAction"]
    from_node: list["NodeSource"]
    node_info: Optional["NodeInfo"]
    _dict: Optional[dict[str, Any]]
    _action_string: Optional[str]

    def __init__(
        self,
        node: Optional[Node],
        pass_name: str = "",
        action: Optional[Union["NodeSourceAction", list["NodeSourceAction"]]] = None,
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

        # cache the action string and dict representation for performance.
        self._action_string: Optional[str] = None
        self._dict: Optional[dict[str, Any]] = None

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
        if self._action_string is None:
            self._action_string = "+".join([a.name.lower() for a in self.action])
        return self._action_string

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
        if self._dict is None:
            # Convert the object to a dictionary
            action_string = self._get_action_string()
            self._dict = {
                "name": self.name,
                "target": self.target,
                "graph_id": self.graph_id,
                "pass_name": self.pass_name,
                "action": action_string,
                "from_node": [node.to_dict() for node in self.from_node],
            }

        assert self._dict is not None
        return self._dict

    def __eq__(self, other: object):
        if not isinstance(other, NodeSource):
            return False
        return self.to_dict() == other.to_dict()

    def __hash__(self):
        # Create a hash based on the dictionary representation
        # We need to convert the dict to a hashable form
        def _make_hashable(obj):
            if isinstance(obj, dict):
                return tuple(sorted((k, _make_hashable(v)) for k, v in obj.items()))
            elif isinstance(obj, list):
                return tuple(_make_hashable(item) for item in obj)
            else:
                return obj

        return hash(_make_hashable(self.to_dict()))

    @classmethod
    def _from_dict(cls, d: Optional[dict]) -> Optional["NodeSource"]:
        """
        Recursively deserialize from_node metadata from dictionary data.
        It is used to deserialize the from_node field from serialized metadata.
        Please use constructor NodeSource(node, ...) to create a NodeSource object.
        """
        if d is None:
            return None

        assert isinstance(d, dict), f"Expected a dict, got {type(d)}"

        # Create a NodeSource object directly without going through the constructor
        # to avoid issues with graph ID and node creation
        node_source = NodeSource.__new__(NodeSource)

        # Reset the cached properties
        node_source._action_string = None
        node_source._dict = None

        # Set the basic attributes
        node_source.pass_name = d.get("pass_name", "")

        # Parse action string back to NodeSourceAction enum list
        action_str = d.get("action", "")
        actions = []
        if action_str:
            for action_name in action_str.split("+"):
                if action_name.upper() == "CREATE":
                    actions.append(NodeSourceAction.CREATE)
                elif action_name.upper() == "REPLACE":
                    actions.append(NodeSourceAction.REPLACE)
        node_source.action = actions

        # Create the NodeInfo object directly
        if "name" in d and "target" in d and "graph_id" in d:
            node_info = NodeSource.NodeInfo(
                d.get("name", ""), d.get("target", ""), d.get("graph_id", -1)
            )
            node_source.node_info = node_info
        else:
            node_source.node_info = None

        # Recursively deserialize nested from_node
        if d.get("from_node", None) is not None:
            node_source.from_node = [
                result
                for fn in d.get("from_node", [])
                if (result := cls._from_dict(fn)) is not None
            ]
        else:
            node_source.from_node = []
        return node_source


@compatibility(is_backward_compatible=False)
@contextmanager
def preserve_node_meta(enable=True):
    global should_preserve_node_meta
    global current_meta
    saved_should_preserve_node_meta = should_preserve_node_meta
    # Shallow copy is OK since fields of current_meta are not mutated
    saved_current_meta = current_meta.copy()
    try:
        should_preserve_node_meta = enable
        yield
    finally:
        should_preserve_node_meta = saved_should_preserve_node_meta
        current_meta = saved_current_meta


@compatibility(is_backward_compatible=False)
def set_stack_trace(stack: list[str]):
    global current_meta

    if should_preserve_node_meta and stack:
        current_meta["stack_trace"] = "".join(stack)


def _enter_annotation(annotation_dict: dict) -> tuple[bool, dict]:
    """
    Core function to enter an annotation context.
    Returns (had_custom, old_custom) for restoration.
    """
    global current_meta

    has_custom = "custom" in current_meta
    old_custom = copy.copy(current_meta.get("custom", {}))

    if not has_custom:
        current_meta["custom"] = {}

    new_custom = copy.copy(current_meta["custom"])
    new_custom.update(annotation_dict)
    current_meta["custom"] = new_custom

    return (has_custom, old_custom)


def _exit_annotation(saved_state: tuple[bool, dict]) -> None:
    """
    Core function to exit an annotation context.
    """
    global current_meta

    has_custom, old_custom = saved_state
    if has_custom:
        current_meta["custom"] = old_custom
    else:
        del current_meta["custom"]


@compatibility(is_backward_compatible=False)
@contextmanager
def annotate(annotation_dict: dict):
    """
    Temporarily adds custom annotations to the current tracing context.
    The fx_node produced from this tracing context will have the
    custom annotations in node.metadata["custom"] field.

    This context manager allows you to insert arbitrary metadata into the PT2
    tracing system by updating the global `current_meta["custom"]` dictionary.
    The annotations are automatically reverted after the context exits.

    Gradient accumulation nodes will not be annotated.

    This is intended for advanced users who need to attach additional metadata to the fx nodes
    (e.g., for debugging, analysis, or external tooling) during export tracing.

    When nested, annotations with the same keys are overridden by the innermost context,
    while annotations with different keys are merged. Upon exiting each nested context,
    the annotations are restored to their previous state.

    Note:
        This API is **not backward compatible** and may evolve in future releases.

    Note:
        This API is not compatible with fx.symbolic_trace or jit.trace. It's intended
        to be used with PT2 family of tracers, e.g. torch.export and dynamo.

    Args:
        annotation_dict (dict): A dictionary of custom key-value pairs to inject
            into the FX trace metadata.

    Example:
        After exiting the context, custom annotations are removed.

        >>> with annotate({"source": "custom_pass", "tag": 42}):
        ...     pass  # Your computation here
    """
    saved_state = _enter_annotation(annotation_dict)
    try:
        yield
    finally:
        _exit_annotation(saved_state)


@compatibility(is_backward_compatible=False)
@contextmanager
def annotate_rqn(rqn: str):
    """
    Temporarily adds a relatively qualified name (RQN) annotation to the current tracing context.
    FX nodes will have the annotation in node.metadata["annotated_rqn"].

    When nested, RQN annotations are automatically concatenated with dots ('.') to create
    a hierarchical path (e.g., "model.encoder.layer1").

    Note:
        This API is **not backward compatible** and may evolve in future releases.

    Note:
        This API is not compatible with fx.symbolic_trace or jit.trace. It's intended
        to be used with PT2 family of tracers, e.g. torch.export and dynamo.

    Args:
        rqn (str): Relatively qualified name to inject into FX trace metadata. Avoid using dots ('.').

    Example:
        >>> with annotate_rqn("model"):
        ...     with annotate_rqn("encoder"):
        ...         x = y + z  # annotated_rqn="model.encoder"
    """
    if RQN_DELIMITER in rqn:
        warnings.warn(
            f"annotate_rqn {rqn} contains '{RQN_DELIMITER}'. "
            f"'{RQN_DELIMITER}' is used as the delimiter. "
            f"Consider using a different rqn annotation."
        )
    global current_meta

    has_scope_annotation = RQN_ANNOTATION_KEY in current_meta
    old_rqn = current_meta.get(RQN_ANNOTATION_KEY, "")

    try:
        if not has_scope_annotation:
            current_meta[RQN_ANNOTATION_KEY] = rqn
        else:
            current_meta[RQN_ANNOTATION_KEY] = old_rqn + RQN_DELIMITER + rqn
        yield
    finally:
        if has_scope_annotation:
            # Restore the original custom dict
            current_meta[RQN_ANNOTATION_KEY] = old_rqn
        else:
            del current_meta[RQN_ANNOTATION_KEY]


@compatibility(is_backward_compatible=False)
def annotate_fn(annotation_dict: dict):
    """
    A decorator that wraps a function with the annotate context manager.
    Use this when you want to annotate an entire function instead of a specific code block.

    Note:
        This API is **not backward compatible** and may evolve in future releases.

    Note:
        This API is not compatible with fx.symbolic_trace or jit.trace. It's intended
        to be used with PT2 family of tracers, e.g. torch.export and dynamo.

    Args:
        annotation_dict (dict): A dictionary of custom key-value pairs to inject
            into the FX trace metadata for all operations in the function.

    Example:
        All operations in my_function will have {"pp_stage": 1} in their metadata.

        >>> @annotate_fn({"pp_stage": 1})
        ... def my_function(x):
        ...     return x + 1
    """
    from functools import wraps

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            with annotate(annotation_dict):
                return func(*args, **kwargs)

        return wrapper

    return decorator


@compatibility(is_backward_compatible=False)
def annotate_rqn_fn(rqn: str):
    """
    A decorator that wraps a function with the annotate_rqn context manager.
    Use this when you want to annotate an entire function with a relatively qualified name
    instead of a specific code block.

    Note:
        This API is **not backward compatible** and may evolve in future releases.

    Note:
        This API is not compatible with fx.symbolic_trace or jit.trace. It's intended
        to be used with PT2 family of tracers, e.g. torch.export and dynamo.

    Args:
        rqn (str): A relatively qualified name string to inject into the FX trace metadata
            for all operations in the function.

    Example:
        All operations in my_function will have annotated_rqn metadata.

        >>> @annotate_rqn_fn("my_module.my_function")
        ... def my_function(x):
        ...     return x + 1
    """
    from functools import wraps

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            with annotate_rqn(rqn):
                return func(*args, **kwargs)

        return wrapper

    return decorator


@compatibility(is_backward_compatible=False)
@contextmanager
def reset_current_annotation():
    global current_meta
    saved_rqn = current_meta.get(RQN_ANNOTATION_KEY)
    saved_custom = current_meta.get("custom", {})
    try:
        if RQN_ANNOTATION_KEY in current_meta:
            del current_meta[RQN_ANNOTATION_KEY]

        if "custom" in current_meta:
            current_meta["custom"] = {}

        yield
    finally:
        if saved_rqn:
            current_meta[RQN_ANNOTATION_KEY] = saved_rqn

        if saved_custom:
            current_meta["custom"] = saved_custom


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
def format_stack() -> list[str]:
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
def get_current_meta() -> dict[str, Any]:
    return current_meta


@compatibility(is_backward_compatible=False)
def get_current_rqn() -> str:
    return current_meta.get(RQN_ANNOTATION_KEY, "")


@compatibility(is_backward_compatible=False)
@contextmanager
def set_current_replay_node(node):
    """
    Set the currently replay node. If `current_replay_node` is not None,
    then we're re-generating the `current_replay_node` in FunctionalTensorMode.
    """
    # See [Note] annotation for more details.
    global current_replay_node
    saved_current_replay_node = current_replay_node
    try:
        current_replay_node = node
        yield
    finally:
        current_replay_node = saved_current_replay_node


@compatibility(is_backward_compatible=False)
def get_current_replay_node():
    """
    Get the currently replay node
    """
    return current_replay_node


@compatibility(is_backward_compatible=False)
def get_graph_provenance_json(graph: Graph) -> dict[str, Any]:
    """
    Given an fx.Graph, return a json that contains the provenance information of each node.
    """
    try:
        provenance_tracking_json = {}
        for node in graph.nodes:
            if node.op == "call_function":
                provenance_tracking_json[node.name] = (
                    [source.to_dict() for source in node.meta["from_node"]]
                    if "from_node" in node.meta
                    else []
                )
        return provenance_tracking_json
    except Exception as e:
        # Since this is just debugging, it should never interfere with regular
        # program execution, so we use this try-except to guard against any error
        signpost_event(
            "inductor",
            "provenance_tracking_error",
            {
                "function": "get_graph_provenance_json",
                "error_msg": str(e),
                "stack_trace": traceback.format_exc(),
            },
        )
        return {}


def _get_annotation_metadata(
    gm: GraphModule, key: str, ignore_compile_rqn_annotation_key: bool = True
) -> str:
    """
    Generic helper to extract annotation metadata for a given key from a GraphModule.

    Args:
        gm: The GraphModule to extract metadata from
        key: The metadata key to extract (e.g., "custom", "annotated_rqn")
        ignore_compile_rqn_annotation_key: If True (default), exclude the internal
            COMPILE_RQN_ANNOTATION_KEY from the output when key is "custom"

    Returns:
        A string representation of all nodes with the specified metadata key
    """
    assert isinstance(gm, GraphModule)

    def helper(gm: GraphModule):
        metadata = []
        for node in gm.graph.nodes:
            if hasattr(node, "meta") and node.meta.get(key, None):
                node_meta = node.meta[key]
                # Filter out COMPILE_RQN_ANNOTATION_KEY if requested
                if (
                    ignore_compile_rqn_annotation_key
                    and key == "custom"
                    and isinstance(node_meta, dict)
                ):
                    node_meta = {
                        k: v
                        for k, v in node_meta.items()
                        if k != COMPILE_RQN_ANNOTATION_KEY
                    }
                    if not node_meta:
                        continue
                metadata.append((node.op, node.name, node_meta))
            if node.op == "get_attr" and isinstance(
                getattr(gm, node.target), GraphModule
            ):
                metadata.append(helper(getattr(gm, node.target)))
        return metadata

    return "\n".join(str(x) for x in helper(gm))


def _get_custom_metadata(gm: GraphModule) -> str:
    """
    Extract custom annotation metadata from a GraphModule.
    This function maintains backward compatibility.

    Args:
        gm: The GraphModule to extract metadata from

    Returns:
        A string representation of all nodes with custom metadata
    """
    return _get_annotation_metadata(gm, "custom")


def _get_annotated_rqn_metadata(gm: GraphModule) -> str:
    """
    Extract annotated_rqn metadata from a GraphModule.

    Args:
        gm: The GraphModule to extract metadata from

    Returns:
        A string representation of all nodes with annotated_rqn metadata
    """
    return _get_annotation_metadata(gm, "annotated_rqn")


def _remove_rqn_metadata_prefix(subgraph: GraphModule, prefix: str):
    """
    Remove `prefix` from the prefix of node's RQN metadata along with the first delimiter.
    """
    if prefix:
        for node in subgraph.graph.nodes:
            if RQN_ANNOTATION_KEY in node.meta:
                rqn = node.meta[RQN_ANNOTATION_KEY]
                new_rqn = rqn.removeprefix(prefix).removeprefix(RQN_DELIMITER)
                node.meta[RQN_ANNOTATION_KEY] = new_rqn


def _get_compile_rqn_annotation():
    """
    Get the current RQN annotation for the current compilation.
    """
    global current_meta, COMPILE_RQN_ANNOTATION_KEY
    return current_meta.get("custom", {}).get(COMPILE_RQN_ANNOTATION_KEY, "")
