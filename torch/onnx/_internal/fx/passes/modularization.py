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

    def __hash__(self) -> int:
        return hash((self._module_name, self._module_class))


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

    def immediate_child(
        self, module_stack: ModuleStackMeta
    ) -> Optional[ModuleStackMeta]:
        if not self.is_child_of(module_stack):
            return None

        raw_meta_clone = copy.copy(self._raw_meta)
        assert (
            raw_meta_clone is not None
        ), f"`_raw_meta` must not be None since `self` is child of {module_stack}."
        for _ in range(len(self) - len(module_stack) - 1):
            raw_meta_clone.popitem()
        return ModuleStackMeta(raw_meta_clone)

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
) -> Tuple[torch.fx.Node, Sequence[torch.fx.Node]]:
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
    extra_nodes = []
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
            extra_nodes.append(module_output)

    # Update meta for new node.
    if stack_trace is not None:
        module_node.meta["stack_trace"] = stack_trace
    raw_module_stack_meta = module_stack_meta.raw_meta
    assert raw_module_stack_meta is not None
    module_node.meta["nn_module_stack"] = copy.copy(raw_module_stack_meta)
    module_node.meta["nn_module_stack"].pop(module_name)

    return module_node, extra_nodes


def create_sub_fx_graph_module(
    root_module: torch.fx.GraphModule, nodes: Sequence[torch.fx.Node]
) -> Tuple[torch.fx.GraphModule, torch.fx.Node, Sequence[torch.fx.Node]]:
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
        # Function with no outputs is invalid.
        # It is either ill model code that the computed output is unused, or something
        # is wrong with the fx graph.
        raise AssertionError(
            "Cannot create sub fx graph module from nodes with no outputs."
        )

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
        module_node, extra_output_nodes = create_call_module_node(
            root_module,
            module_name,
            module_node_module_stack_meta,
            inputs,
            outputs,
            stack_trace=stack_trace,
        )

    for node in reversed(nodes):
        root_module.graph.erase_node(node)

    return sub_module, module_node, extra_output_nodes


class SubModuleBuilder:
    def __init__(self, root: torch.fx.GraphModule, stack_meta: ModuleStackMeta):
        self._root = root
        self._nodes: List[Union[torch.fx.Node, SubModuleBuilder]] = []
        self._submodules: Dict[ModuleMeta, SubModuleBuilder] = {}
        self._stack_meta: ModuleStackMeta = stack_meta

    @property
    def stack_meta(self) -> ModuleStackMeta:
        return self._stack_meta

    def add_fx_node(self, fx_node: torch.fx.Node):
        node_stack_meta = module_stack_meta_from_node(fx_node)

        if node_stack_meta == self.stack_meta:
            self._nodes.append(fx_node)
        elif node_stack_meta.is_child_of(self.stack_meta):
            child_stack_meta = node_stack_meta.immediate_child(self.stack_meta)
            assert child_stack_meta is not None
            child_meta = child_stack_meta.current()
            assert child_meta is not None
            if child_meta not in self._submodules:
                self._submodules[child_meta] = SubModuleBuilder(
                    self._root, child_stack_meta
                )
                self._nodes.append(self._submodules[child_meta])
            self._submodules[child_meta].add_fx_node(fx_node)
        else:
            raise AssertionError(
                f"Node {fx_node} ({node_stack_meta}) does not belong to module "
                f"{self._stack_meta}."
            )

    def build_submodules(self) -> Sequence[torch.fx.Node]:
        fx_nodes: List[torch.fx.Node] = []

        for node in self._nodes:
            if isinstance(node, SubModuleBuilder):
                _, module_node, extra_output_nodes = node.build()
                fx_nodes.extend([module_node, *extra_output_nodes])
            else:
                fx_nodes.append(node)

        return fx_nodes

    def build(
        self,
    ) -> Tuple[torch.fx.GraphModule, torch.fx.Node, Sequence[torch.fx.Node]]:
        if self._stack_meta.is_empty_or_root():
            raise RuntimeError(
                "`build` should not be called on root module. "
                "Call `build_submodules` instead."
            )
        fx_nodes = self.build_submodules()

        return create_sub_fx_graph_module(self._root, fx_nodes)


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
        builder = SubModuleBuilder(self.module, ModuleStackMeta(None))
        for node in self.module.graph.nodes:
            builder.add_fx_node(node)
        builder.build_submodules()
        return self.module
