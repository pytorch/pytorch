from __future__ import annotations

import collections
import copy
import functools
import operator

from typing import Any, Dict, Final, Iterator, List, Optional, Sequence, Tuple, Union

import torch
import torch.fx
from torch.onnx._internal import _beartype

from torch.onnx._internal.fx import _pass
from torch.utils import _pytree as pytree

_FX_TRACER_NN_MODULE_STACK_META_TYPE = collections.OrderedDict
"""Legacy type produced by FX symbolic tracer."""

_DYNAMO_NN_MODULE_STACK_META_TYPE = Dict[str, Tuple[str, type]]
"""Type produced by FX dynamo tracer."""


class ModuleMeta:
    """Meta information about a module."""

    _module_class: Final[Optional[type]]
    _module_name: Final[Optional[str]]

    @_beartype.beartype
    def __init__(self, module_name: str, module_class: type):
        self._module_name = module_name
        self._module_class = module_class

    @property
    def display_name(self) -> str:
        # TODO: Make it look nicer.
        # E.g., from 'L__self___h_1_mlp_c_proj' to 'h.1.mlp.c_proj'
        # NOTE: Need to make sure it works well with Netron.
        ...

    @property
    def qualified_class_name(self) -> str:
        if self._module_class is None:
            return ""
        return self._module_class.__module__ + "." + self._module_class.__name__

    @property
    def class_name(self) -> str:
        if self._module_class is None:
            return ""
        return self._module_class.__name__

    @property
    def name(self) -> str:
        if self._module_name is None:
            return ""
        return self._module_name

    def __eq__(self, __value: object) -> bool:
        if not isinstance(__value, ModuleMeta):
            return False
        return (
            self._module_name == __value._module_name
            and self._module_class == __value._module_class
        )


class ModuleStackMeta:
    """Meta information about the module stack.

    This class is used to represent the module stack information in a more
    structured way. It parses raw module stack information from node.meta.
    """

    _module_stack: Final[List[ModuleMeta]]
    _raw_meta: Final[
        Optional[
            Union[
                _FX_TRACER_NN_MODULE_STACK_META_TYPE, _DYNAMO_NN_MODULE_STACK_META_TYPE
            ]
        ]
    ]

    @_beartype.beartype
    def __init__(
        self,
        nn_module_stack_meta: Optional[
            Union[
                _FX_TRACER_NN_MODULE_STACK_META_TYPE, _DYNAMO_NN_MODULE_STACK_META_TYPE
            ]
        ],
    ):
        self._module_stack = []
        self._raw_meta = nn_module_stack_meta
        if nn_module_stack_meta is None:
            return
        if isinstance(nn_module_stack_meta, _FX_TRACER_NN_MODULE_STACK_META_TYPE):
            for module_name, module_class in nn_module_stack_meta.items():
                assert isinstance(
                    module_class, type
                ), f"module_class is not a type: {module_class}"
                self._module_stack.append(ModuleMeta(module_name, module_class))
        elif isinstance(nn_module_stack_meta, dict):
            for module_name, (_, module_class) in nn_module_stack_meta.items():
                self._module_stack.append(ModuleMeta(module_name, module_class))
        else:
            raise AssertionError(
                f"Unknown type of nn_module_stack_meta: {type(nn_module_stack_meta)}"
            )

    def __len__(self) -> int:
        return len(self._module_stack)

    def __getitem__(self, index: int) -> ModuleMeta:
        return self._module_stack[index]

    def __iter__(self) -> Iterator[ModuleMeta]:
        return iter(self._module_stack)

    def pop(self) -> ModuleMeta:
        """Pop the last module from the stack."""
        return self._module_stack.pop()

    def is_empty_or_root(self) -> bool:
        return len(self._module_stack) == 0

    def current(self) -> Optional[ModuleMeta]:
        if self.is_empty_or_root():
            return None
        return self._module_stack[-1]

    @_beartype.beartype
    def is_child_of(
        self,
        module_stack: ModuleStackMeta,
    ) -> bool:
        if self.is_empty_or_root():
            return False

        if module_stack.is_empty_or_root() is None:
            return True

        if len(self) <= len(module_stack):
            return False

        for i, parent_key in enumerate(module_stack):
            if self[i] != parent_key:
                return False

        return True

    @_beartype.beartype
    def is_same(
        self,
        module_stack: ModuleStackMeta,
    ) -> bool:
        return self._module_stack == module_stack._module_stack

    @_beartype.beartype
    def __eq__(self, __value: object) -> bool:
        if not isinstance(__value, ModuleStackMeta):
            return False
        return self.is_same(__value)

    @property
    def raw_meta(self) -> Optional[Dict[str, Tuple[str, type]]]:
        return self._raw_meta


@functools.lru_cache()
def module_stack_meta_from_node(node: torch.fx.Node) -> ModuleStackMeta:
    nn_module_stack_meta = node.meta.get("nn_module_stack", None)
    return ModuleStackMeta(nn_module_stack_meta)


def extract_module_inputs(nodes: Sequence[torch.fx.Node]) -> Sequence[torch.fx.Node]:
    """Extract module inputs from a sequence of nodes.

    All node args that are produced by nodes outside of the module are considered module
    inputs.

    Args:
        nodes: Sequence of nodes to extract module inputs from. The nodes must all
            belong to the same module.
    Returns:
        Sequence of module inputs.
    """
    assert len(nodes) > 0, "Cannot extract module inputs from empty nodes."
    module_stack_meta = module_stack_meta_from_node(nodes[0])
    assert (
        not module_stack_meta.is_empty_or_root()
    ), "Cannot extract module inputs from nodes without nn_module_stack."
    # Need ordered set. Emulate with dict.
    module_inputs: Dict[torch.fx.Node, None] = {}

    # TODO(bowbao): `aten.sym_size` if exist will appear as regular module input. A lot
    # of this is potentially avoidable by checking to extract that particular symbol
    # from existing inputs.
    def _extract_arg_if_node_outside_module(arg: Any):
        if (
            isinstance(arg, torch.fx.Node)
            and module_stack_meta_from_node(arg) != module_stack_meta
        ):
            module_inputs[arg] = None

    for node in nodes:
        pytree.tree_map(_extract_arg_if_node_outside_module, node.args)
        pytree.tree_map(_extract_arg_if_node_outside_module, node.kwargs)
    return list(module_inputs.keys())


def extract_module_outputs(nodes: Sequence[torch.fx.Node]) -> Sequence[torch.fx.Node]:
    """Extract module outputs from a sequence of nodes.

    All nodes that are used by nodes outside of the module are considered module
    outputs.

    Args:
        nodes: Sequence of nodes to extract module outputs from. The nodes must all
            belong to the same module.
    Returns:
        Sequence of module outputs.
    """
    assert len(nodes) > 0, "Cannot extract module inputs from empty nodes."
    module_stack_meta = module_stack_meta_from_node(nodes[0])
    assert (
        not module_stack_meta.is_empty_or_root()
    ), "Cannot extract module inputs from nodes without nn_module_stack."
    # Need ordered set. Emulate with dict.
    module_outputs: Dict[torch.fx.Node, None] = {}

    for node in nodes:
        for user in node.users:
            if module_stack_meta != module_stack_meta_from_node(user):
                module_outputs[node] = None
    return list(module_outputs.keys())


def create_call_module_node(
    root_module: torch.fx.GraphModule,
    module_name: str,
    module_stack_meta: ModuleStackMeta,
    original_inputs: Sequence[torch.fx.Node],
    original_outputs: Sequence[torch.fx.Node],
    stack_trace: Optional[str] = None,
) -> torch.fx.Node:
    """Create a call_module node and insert into root_module at the current insert point.

    This function will insert a call_module node and replace all uses of the old outputs
    with the new module output. It will also populate meta data for the new module output.

    Args:
        root_module: The root module to insert the call_module node into.
        module_name: The name of the module to call. The associated submodule must exist
            in root_module, retrievable by `root_module.get_submodule(module_name)`.
        module_stack_meta: The module stack meta data for the new call_module node.
        original_inputs: Submodule boundary nodes extracted from root_module. They are
            inputs to the new call_module node.
        original_outputs: Submodule boundary nodes extracted from root_module. They are
            replaced by submodule and the new call_module node.
        stack_trace: The stack trace for the new call_module node.

    Returns:
        The new call_module node.
    """

    module_node = root_module.graph.call_module(
        module_name, args=tuple(original_inputs)
    )

    # Copy meta data from old outputs.
    # Replace usage of old outputs with new module output.
    if len(original_outputs) == 1:
        old_output = original_outputs[0]
        old_output.replace_all_uses_with(module_node)
        module_node.meta = copy.copy(old_output.meta)
    else:
        module_node.meta["val"] = tuple(
            old_output.meta.get("val", None) for old_output in original_outputs
        )
        for i, old_output in enumerate(original_outputs):
            module_output = root_module.graph.call_function(
                operator.getitem, args=(module_node, i), type_expr=old_output.type
            )
            old_output.replace_all_uses_with(module_output)
            module_output.meta = copy.copy(old_output.meta)
            # NOTE: We don't want to deepcopy everything in meta, but we do want to
            # copy a new instance of "nn_module_stack". Since we need to pop the
            # current module from the stack later.
            module_output.meta["nn_module_stack"] = copy.copy(
                old_output.meta["nn_module_stack"]
            )
            module_output.meta["nn_module_stack"].pop(module_name)

    # Update stack info for new node.
    raw_module_stack_meta = module_stack_meta.raw_meta
    assert raw_module_stack_meta is not None
    module_node.meta["nn_module_stack"] = copy.copy(raw_module_stack_meta)
    module_node.meta["nn_module_stack"].pop(module_name)

    return module_node


def create_sub_fx_graph_module(
    root_module: torch.fx.GraphModule, nodes: Sequence[torch.fx.Node]
) -> Tuple[Optional[torch.fx.GraphModule], torch.fx.Node]:
    """Create a sub fx graph module from a sequence of nodes.

    This function will replace the nodes with a call_module node that calls the new
    sub fx graph module.

    Args:
        root_module: The root module to insert the call_module node and submodule into.
        nodes: The nodes to create the sub fx graph module from.
    Returns:
        The new sub fx graph module and the call_module node that calls it.
    """
    if len(nodes) == 0:
        raise AssertionError("Cannot create sub fx graph module from empty nodes.")

    module_stack_meta = module_stack_meta_from_node(nodes[0])
    assert (
        not module_stack_meta.is_empty_or_root()
    ), "Cannot create sub fx graph module from nodes without nn_module_stack."
    module_meta = module_stack_meta.current()
    assert (
        module_meta is not None
    ), "Cannot create sub fx graph module from nodes without nn_module_meta."
    module_name = module_meta.name
    module_class_name = module_meta.qualified_class_name

    # Identify module boundaries, i.e., inputs and outputs.
    inputs = extract_module_inputs(nodes)
    outputs = extract_module_outputs(nodes)

    if len(outputs) == 0:
        # TODO: Replace with diagnostic w/ warning.
        # Function with no outputs is invalid. Abort creating this function.
        # It is either ill model code that the computed output is unused, or something
        # is wrong with the fx graph.
        return None, nodes[-1]

    # Create a new fx graph for the new sub module.
    sub_graph = torch.fx.Graph(root_module)
    new_outputs: List[torch.fx.Node] = []
    # Populate `placeholder` nodes for inputs.
    old_to_new_node: Dict[torch.fx.Node, torch.fx.Node] = {}
    for arg in inputs:
        old_to_new_node[arg] = sub_graph.placeholder(arg.name, arg.type)
        old_to_new_node[arg].meta = arg.meta
    # Copy nodes and form a new graph for the new sub module.
    for node in nodes:
        new_node = sub_graph.node_copy(node, lambda n: old_to_new_node[n])
        old_to_new_node[node] = new_node
        if node in outputs:
            new_outputs.append(new_node)
    # Set output nodes.
    sub_graph.output(new_outputs[0] if len(new_outputs) == 1 else new_outputs)

    # Create a new submodule.
    sub_module = torch.fx.GraphModule(root_module, sub_graph, module_class_name)
    root_module.add_submodule(module_name, sub_module)

    module_node_module_stack_meta = copy.copy(module_stack_meta)
    module_node_module_stack_meta.pop()
    # TODO: Figure out a good source code location.
    # Currently set as first node in the subgraph. This is the best we can do with
    # existing meta information. However it is not ideal since it is the location of
    # submodule code, not the location of the module call.
    stack_trace = nodes[0].meta.get("stack_trace", None)
    with root_module.graph.inserting_before(nodes[0]):
        module_node = create_call_module_node(
            root_module,
            module_name,
            module_node_module_stack_meta,
            inputs,
            outputs,
            stack_trace=stack_trace,
        )

    for node in reversed(nodes):
        root_module.graph.erase_node(node)

    return sub_module, module_node


def try_create_sub_fx_graph_module_starting_from_node(
    root_module: torch.fx.GraphModule, start_node: torch.fx.Node
) -> torch.fx.Node:
    """Try to create a sub fx graph module starting from a node.

    If the start node does not belong to a module, or is on root module level, this
    function will return the start node.

    Otherwise, this function will try to find a sequence of nodes that belongs to the
    same module of the start node, and create a sub fx graph module from them. These
    nodes will be removed and replaced with a call_module node that represents the new
    sub fx graph module. The call_module node will be returned.

    If this function finds a sequence of nodes that belongs to the child module of the
    start node, it will recursively call this function to create a sub fx graph module
    for the child module first.

    Args:
        root_module: The root module.
        start_node: The start node to try to create a sub fx graph module from. It is
            expected that all nodes after this node are in flattened form. It is also
            expected that if this node belongs to a module, all nodes after this node
            that belong to the same module should be consecutive.
    Returns:
        The call_module node that represents the new sub fx graph module, or the start
        node if no sub fx graph module is created.
    """
    iter_node = start_node
    current_module_stack_meta = module_stack_meta_from_node(start_node)
    if current_module_stack_meta.is_empty_or_root():
        return iter_node
    nodes_to_form_module: List[torch.fx.Node] = []

    while iter_node:
        module_stack_meta = module_stack_meta_from_node(iter_node)

        if module_stack_meta == current_module_stack_meta:
            nodes_to_form_module.append(iter_node)
            iter_node = iter_node.next
            continue
        elif module_stack_meta.is_child_of(current_module_stack_meta):
            # Need to find nodes for child module and create child module first.
            child_module_node = try_create_sub_fx_graph_module_starting_from_node(
                root_module, iter_node
            )
            # Continue from the new child module node.
            iter_node = child_module_node
            continue
        else:
            # iter_node belongs to another module. And it is not a child module.
            # All nodes for current module are collected.
            _, call_module_node = create_sub_fx_graph_module(
                root_module, nodes_to_form_module
            )
            return call_module_node

    raise AssertionError(f"Should not reach here. Node {start_node}")


class Modularize(_pass.Transform):
    @_beartype.beartype
    def _run(self, *args) -> torch.fx.GraphModule:
        """Modularize a flattened fx graph module.

        This function will try to create sub fx graph modules for all nodes that belong
        to a module. It will recursively create sub fx graph modules for child modules
        first.

        Args:
            *args: Unused.
        Returns:
            The modularized fx graph module.
        """
        iter_node: torch.fx.Node = next(iter(self.module.graph.nodes))
        while iter_node and iter_node != self.module.graph._root:
            maybe_new_module_node = try_create_sub_fx_graph_module_starting_from_node(
                self.module, iter_node
            )
            if maybe_new_module_node == iter_node:
                # No sub fx graph module is created. Continue to next node.
                iter_node = iter_node.next
            else:
                # A sub fx graph module is created. Continue from the new module node,
                # because it may belong to another module higher in the stack.
                iter_node = maybe_new_module_node
        return self.module
