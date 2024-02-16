from __future__ import annotations

import abc

import collections
import copy
import operator

from typing import (
    Any,
    Dict,
    Final,
    Generator,
    Iterator,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)

import torch
import torch.fx
from torch.onnx._internal import _beartype

from torch.onnx._internal.fx import _pass, diagnostics
from torch.utils import _pytree as pytree

_FX_TRACER_NN_MODULE_META_TYPE = Tuple[str, type]
"""Legacy type of item from `node.meta["nn_module_stack"].items()` produced by FX symbolic tracer."""
_FX_TRACER_NN_MODULE_STACK_META_TYPE = collections.OrderedDict
"""Legacy type of `node.meta["nn_module_stack"]` produced by FX symbolic tracer."""

_DYNAMO_NN_MODULE_META_TYPE = Tuple[str, Tuple[str, type]]
"""Type of item from `node.meta["nn_module_stack"].items()` produced by FX dynamo tracer."""
_DYNAMO_NN_MODULE_STACK_META_TYPE = Dict[str, _DYNAMO_NN_MODULE_META_TYPE]
"""Type of `node.meta["nn_module_stack"]` produced by FX dynamo tracer."""


class _ModuleMeta:
    """Meta information about a module.

    This class is used to represent the module information in a more structured way.
    It parses raw module information from a single item from
    `node.meta["nn_module_stack"].items()`.

    See the uses of `from_raw_meta`, `from_fx_tracer_produced_raw_meta`, and
    `from_dynamo_produced_raw_meta` for how to create an instance.

    Attributes:
        _module_class: The class of the module. E.g. `torch.nn.module.sparse.Embedding`.
        _module_name: The name of the module. E.g. `L__self___h_1_mlp_c_proj`.
        _raw_meta: The raw meta '(module_name, node.meta["nn_module_stack"][module_name])'.
    """

    _module_class: Final[Optional[type]]
    _module_name: Final[str]
    _raw_meta: Final[Tuple[Any, Any]]

    @_beartype.beartype
    def __init__(
        self, module_name: str, module_class: Optional[type], raw_meta: Tuple[Any, Any]
    ):
        self._module_name = module_name
        self._module_class = module_class
        self._raw_meta = raw_meta

    @property
    def module_display_name(self) -> str:
        """The display name of the module.

        E.g. `h_1_mlp_c_proj`.
        """
        # E.g., from 'L__self___h_1_mlp_c_proj' to 'h_1_mlp_c_proj'.
        name = self.module_name
        if name.startswith("L__self___"):
            name = name[len("L__self___") :]
        return name

    @property
    def qualified_module_class_name(self) -> str:
        """Qualified name of the module class.

        E.g. `torch_nn_module_sparse_Embedding`.
        """
        if self._module_class is None:
            return ""
        return (
            self._module_class.__module__ + "_" + self._module_class.__name__
        ).replace(".", "_")

    @property
    def module_class_name(self) -> str:
        """Name of the module class.

        E.g. `Embedding`.
        """
        if self._module_class is None:
            return ""
        return self._module_class.__name__

    @property
    def module_name(self) -> str:
        """Name of the module.

        E.g. `L__self___h_1_mlp_c_proj`.
        """
        return self._module_name

    @property
    def raw_meta(self) -> Tuple[Any, Any]:
        """Returns the raw module meta data.

        I.e. (module_name, node.meta['nn_module_stack'][module_name]).
        """
        return self._raw_meta

    def __eq__(self, __value: object) -> bool:
        if not isinstance(__value, _ModuleMeta):
            return False
        return (
            self._module_name == __value._module_name
            and self._module_class == __value._module_class
        )

    def __hash__(self) -> int:
        return hash((self._module_name, self._module_class))

    def __repr__(self) -> str:
        return f"ModuleMeta(name={self._module_name}, class={self._module_class})"

    @classmethod
    def create_root(cls) -> _ModuleMeta:
        """Create an empty module meta representing root module."""
        return _ModuleMeta("", None, ("", None))

    @classmethod
    def from_fx_tracer_produced_raw_meta(
        cls, raw_meta: _FX_TRACER_NN_MODULE_META_TYPE
    ) -> _ModuleMeta:
        """Create a module meta from raw meta produced by FX symbolic tracer."""
        module_name, module_class = raw_meta
        return _ModuleMeta(module_name, module_class, raw_meta)

    @classmethod
    def from_dynamo_produced_raw_meta(
        cls, raw_meta: _DYNAMO_NN_MODULE_META_TYPE
    ) -> _ModuleMeta:
        """Create a module meta from raw meta produced by FX dynamo tracer."""
        module_name, (qualified_name, module_class) = raw_meta
        return _ModuleMeta(module_name, module_class, raw_meta)

    @classmethod
    def from_raw_meta(
        cls,
        raw_meta: Union[_FX_TRACER_NN_MODULE_META_TYPE, _DYNAMO_NN_MODULE_META_TYPE],
    ) -> _ModuleMeta:
        if (
            isinstance(raw_meta, tuple)
            and len(raw_meta) == 2
            and isinstance(raw_meta[1], type)
        ):
            # Trying to do `instance(raw_meta, _FX_TRACER_NN_MODULE_META_TYPE)`
            return _ModuleMeta.from_fx_tracer_produced_raw_meta(raw_meta)
        if (
            isinstance(raw_meta, tuple)
            and len(raw_meta) == 2
            and isinstance(raw_meta[1], tuple)
        ):
            # Trying to do `instance(raw_meta, _DYNAMO_NN_MODULE_META_TYPE)`
            return _ModuleMeta.from_dynamo_produced_raw_meta(raw_meta)
        raise TypeError(
            f"Unknown type of raw meta item from node.meta['nn_module_stack'].items(): {type(raw_meta)}"
        )


class _ModuleStackMeta:
    """Meta information about the module call stack.

    This class is used to represent the module call stack information in a more
    structured way. It parses raw module stack information from `node.meta["nn_module_stack"]`.

    Example of raw module stack information:

        If produced by dynamo:

            {
                'L__self___h_1': (
                    "L['self'].h[1]",
                    <class 'transformers.models.gpt2.modeling_gpt2.GPT2Block'>
                ),
                'L__self___h_1_attn': (
                    "L['self'].h[1].attn",
                    <class 'transformers.models.gpt2.modeling_gpt2.GPT2Attention'>
                )
            }

        If produced by fx.symbolic_trace:

            {
                'h.1': <class 'transformers.models.gpt2.modeling_gpt2.GPT2Block'>,
                'h.1.attn': <class 'transformers.models.gpt2.modeling_gpt2.GPT2Attention'>
            }
    """

    _module_stack: Final[List[_ModuleMeta]]

    @_beartype.beartype
    def __init__(
        self,
        nn_module_stack_meta: Optional[
            Union[
                _FX_TRACER_NN_MODULE_STACK_META_TYPE, _DYNAMO_NN_MODULE_STACK_META_TYPE
            ]
        ],
        is_exported_program: bool = True,
    ):
        self._module_stack = []
        if nn_module_stack_meta is None:
            return
        raw_meta = copy.copy(nn_module_stack_meta)
        for item in raw_meta.items():
            # If produced by torch.export.export, there is another call stack layer
            # that we need to skip
            if is_exported_program:
                is_exported_program = False
                continue
            self.push(_ModuleMeta.from_raw_meta(item))

    def __len__(self) -> int:
        return len(self._module_stack)

    def __getitem__(self, index: int) -> _ModuleMeta:
        return self._module_stack[index]

    def __iter__(self) -> Iterator[_ModuleMeta]:
        return iter(self._module_stack)

    def is_empty_or_root(self) -> bool:
        return len(self._module_stack) == 0

    def top(self) -> _ModuleMeta:
        """Returns the top module meta in the stack. I.e., the meta for leaf module.

        Example:

            Consider the following module stack:

            stack = [GPT, block1, Attention_1, MLP]

            stack.top() == MLP
        """
        if self.is_empty_or_root():
            return _ModuleMeta.create_root()
        return self._module_stack[-1]

    @_beartype.beartype
    def is_superset_of(
        self,
        module_stack: _ModuleStackMeta,
    ) -> bool:
        """Determines if self is a superset of the provided module stack.

        I.e., If self includes all elements from the provided module stack, plus additional
        elements on top. If self is empty or root, this method always return False.

        Example:

            Consider the following module stack:

            stack_1 = [GPT, block1, Attention_1, MLP]
            stack_2 = [GPT, block1]

            stack_1.is_superset_of(stack_2) == True
            stack_2.is_superset_of(stack_1) == False

            stack_3 = [GPT, block2, Attention_1]

            stack_1.is_superset_of(stack_3) == False
            stack_3.is_superset_of(stack_1) == False
        """
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

    def push(self, module_meta: _ModuleMeta) -> None:
        """Pushes a module meta to the stack."""
        self._module_stack.append(module_meta)

    @_beartype.beartype
    def __eq__(self, __value: object) -> bool:
        if not isinstance(__value, _ModuleStackMeta):
            return False
        return self._module_stack == __value._module_stack

    @property
    def raw_meta(self) -> Optional[Dict[str, Tuple[str, type]]]:
        """Returns the raw module stack meta data, i.e. node.meta['nn_module_stack']."""
        return {
            module_meta.raw_meta[0]: module_meta.raw_meta[1]
            for module_meta in self._module_stack
        }

    def __repr__(self) -> str:
        return f"ModuleStackMeta({self._module_stack})"

    @property
    def module_display_name(self) -> str:
        """Returns the module display name of the top module."""
        return self.top().module_display_name

    @property
    def qualified_module_class_name(self) -> str:
        """Returns the qualified module class name of the top module."""
        return self.top().qualified_module_class_name

    @property
    def module_class(self) -> Optional[type]:
        """Returns the module class of the top module."""
        return self.top()._module_class


def _module_stack_meta_from_node(
    node: torch.fx.Node, is_exported_program: bool = False
) -> _ModuleStackMeta:
    return _ModuleStackMeta(
        node.meta.get("nn_module_stack"), is_exported_program=is_exported_program
    )


def _get_unique_module_name(module_names: Dict[str, int], module_name: str) -> str:
    module_names.setdefault(module_name, 0)
    module_names[module_name] += 1
    return f"{module_name}_{module_names[module_name]}"


class _IRNode(abc.ABC):
    """Base class for IR nodes.

    IR nodes are used for Modularize pass only. They add a layer of abstraction on top of
    torch.fx.Node.

    [NOTE: Modularize Pass Implementation]
    The main job of the pass is to group `fx.Node`s that belong to the same `nn.Module`
    forward call, and then create `call_module` node and sub `fx.GraphModule` from them.
    Each `fx.Node` possesses an `nn_module_stack` meta data that contains information
    about the module call stack. See `_ModuleStackMeta` for examples.

    Analysis step
    -------------

    Each module call is identified by a set of base stack layers. For each module call,
    the pass creates a `_ModuleNode` and groups the sequence of nodes that shares the
    same base stack layers.

    For example,

        stack_of_node_0 = [GPT, block0]
        stack_of_node_1 = [GPT, block1]
        stack_of_node_2 = [GPT, block1, Attention1, MLP]
        stack_of_node_3 = [GPT, block1, Attention1]
        stack_of_node_4 = [GPT, block2]

    All nodes belong to the `GPT` module call, since they share the base stack layers [GPT].
    [node_1, node_2, node_3] are grouped for `GPT.block1`, because they share the base
    stack layers [GPT, block1]. And [node_2, node_3] for `GPT.block1.Attention1`, [node_0]
    for `GPT.block0`, and [node_4] for `GPT.block2` respectfully.

    After the analysis step, a hierarchical representation is generated.

    For above example, the representation is:

        _ModuleNode(GPT)
            _ModuleNode(block0)
                _LeafNode(node_0)
            _ModuleNode(block1)
                _LeafNode(node_1)
                _ModuleNode(Attention1)
                    _ModuleNode(MLP)
                        _LeafNode(node_2)
                _LeafNode(node_3)
            _ModuleNode(block2)
                _LeafNode(node_4)

    Construction step
    -----------------

    The second step is to build the actual `call_module` node and the sub `fx.GraphModule`.
    This is done recursively from the leaf `_ModuleNode` to the root.

    For example, the first submodule to be built is `GPT.block1.Attention1.MLP`. Below pair
    is generated from `_ModuleNode(MLP)`.

        fx.GraphModule(GPT.block1.Attention1.MLP)
            graph:
                node_2

        new_mlp_node = `call_module[GPT.block1.Attention1.MLP](...)`

    Next, the `GPT.block1.Attention1` submodule is built. Below is generated from
    `_ModuleNode(Attention1)`.

        fx.GraphModule(GPT.block1.Attention1)
            graph:
                new_mlp_node
                node_3

        new_attention1_node = `call_module[GPT.block1.Attention1](...)`

    Until every submodule is built, the new modularized `fx.GraphModule` is generated.

    Alternatives
    ------------

    The current algorithm adopts a top down approach. A bottom up approach is similar.
    In contrast to these two, an alternative flat order approach is also possible, where
    each node is traversed and copied to the corresponding submodule.

    The advantage of the current approach lies in the encapsulation of the fx.GraphModule
    construction for each individual submodule within a single `build_module` method, which
    can be called separately once the analysis phase is completed, making debugging more
    convenient.

    Regarding construction step, an alternative implementation is to utilize `fx.Interpreter`
    for traversing all the nodes under the flattened root module and copying the nodes
    into their respective submodule under construction. This approach is not adopted because

        1. It uses the flat order approach discussed above. This means one cannot individually
    construct a submodule and examine it while debugging.

        2. The graph execution functionality of `fx.Interpreter` is not necessary for the
    purpose of this pass. Ignoring that, `fx.Interpreter.run` achieves the same effect
    as a for loop over all the nodes.
    """

    @property
    @abc.abstractmethod
    def stack_meta(self) -> _ModuleStackMeta:
        """The module stack meta data associated with this node."""
        ...

    @property
    @abc.abstractmethod
    def stack_trace(self) -> Optional[str]:
        """The stack trace associated with this node."""
        ...


class _ModuleNode(_IRNode):
    """Representing a sequence of fx.Nodes to be formed into a fx.GraphModule.

    This class encapsulates metadata and provides building block methods to construct this
    layered abstraction from a sequence of flat fx.Nodes.

    Attributes:
    - _stack_meta: Metadata of the module stack.
    - _nodes: List of IR nodes in the module.
    - _reference_root_module: Reference to the root flat fx.GraphModule instance.
    """

    def __init__(
        self, reference_root_module: torch.fx.GraphModule, stack_meta: _ModuleStackMeta
    ):
        self._stack_meta = stack_meta
        self._nodes: List[_IRNode] = []
        self._reference_module = reference_root_module

    @property
    def stack_meta(self) -> _ModuleStackMeta:
        return self._stack_meta

    @property
    def stack_trace(self) -> Optional[str]:
        assert self._nodes
        return self._nodes[0].stack_trace

    def __str__(self) -> str:
        return f"ModuleNode({self._stack_meta})"

    def is_same_module_as(self, node: _IRNode) -> bool:
        """Determines if the provided node pertains to the same module as this node."""
        return self.stack_meta == node.stack_meta

    def is_parent_module_of(self, node: _IRNode) -> bool:
        """Determines if this node represents a parent module of the provided node."""
        return node.stack_meta.is_superset_of(self.stack_meta)

    def add_leaf_node(self, leaf_node: _LeafNode) -> None:
        """Adds a leaf node to the module.

        The leaf node must belong to the same or a child module. This method will recursively
        construct _ModuleNode instance based on the stack_meta information of the leaf node.
        """
        if self.is_same_module_as(leaf_node) or leaf_node.fx_op == "call_module":
            self._nodes.append(leaf_node)
        elif leaf_node.fx_op == "placeholder":
            # Although the original placeholder has empty nn_module_stack, the placeholder lifted
            # from get_attr nodes by exported program has their original nn_module_stack. Here
            # we need to avoid them building submodule.
            self._nodes.append(leaf_node)
        elif self.is_parent_module_of(leaf_node):
            # This node belongs in a submodule.
            # Check if the last node is a submodule and if it is the parent of this node.
            last_node = self._nodes[-1] if self._nodes else None
            if isinstance(last_node, _ModuleNode) and (
                last_node.is_parent_module_of(leaf_node)
                or last_node.is_same_module_as(leaf_node)
            ):
                # This node belongs to the last_node.
                last_node.add_leaf_node(leaf_node)
            else:
                # Create a new SubmoduleNode for the immediate child module of the current
                # module. The leaf node may be a grandchild of the current module.
                # Example:
                #   self.stack_meta = [A, B, C]
                #   leaf_node.stack_meta = [A, B, C, D, E, F]
                # Create a new ModuleNode with stack_meta = [A, B, C, D] and add leaf_node to it.
                stack_meta = copy.deepcopy(self.stack_meta)
                stack_meta.push(leaf_node.stack_meta[len(self.stack_meta)])
                last_node = _ModuleNode(
                    self._reference_module,
                    stack_meta,
                )
                self._nodes.append(last_node)
                last_node.add_leaf_node(leaf_node)
        else:
            raise AssertionError(
                f"Node {leaf_node} ({leaf_node.stack_meta}) does not belong to module "
                f"{self._stack_meta}."
            )

    def fx_nodes(self) -> Generator[torch.fx.Node, None, None]:
        """Returns an iterator for the sequence of fx nodes this instance holds."""
        for node in self._nodes:
            if isinstance(node, _ModuleNode):
                yield from node.fx_nodes()
            else:
                assert isinstance(node, _LeafNode)
                yield node.fx_node

    def module_inputs(self) -> Sequence[torch.fx.Node]:
        """Extract module inputs from the sequence of fx nodes this instance holds.

        All node args that are produced by nodes outside of the module are considered module
        inputs. The order of returned module inputs is the same as the their use order.

        ### Known limitations

        The original ordering of module inputs is not preserved. There is no meta information
        to be found from the `fx.GraphModule` that can be used to recover the original ordering.

        Returns:
            Sequence of module inputs.
        """
        nodes = list(self.fx_nodes())
        assert len(nodes) > 0, "Cannot extract module inputs from empty nodes."
        module_inputs: Dict[torch.fx.Node, None] = {}
        node_set: Set[torch.fx.Node] = set(nodes)

        def _extract_arg_if_node_outside_module(arg: Any):
            if isinstance(arg, torch.fx.Node) and arg not in node_set:
                module_inputs[arg] = None

        for node in nodes:
            pytree.tree_map(_extract_arg_if_node_outside_module, node.args)
            pytree.tree_map(_extract_arg_if_node_outside_module, node.kwargs)
        return list(module_inputs.keys())

    def module_outputs(self) -> Sequence[torch.fx.Node]:
        """Extract module outputs from the sequence of fx nodes this instance holds.

        All nodes that are used by nodes outside of the module are considered module
        outputs. The order of returned module outputs is the same as the their creation order.

        ### Known limitations

        The original ordering of module outputs is not preserved. There is no meta information
        to be found from the `fx.GraphModule` that can be used to recover the original ordering.

        Returns:
            Sequence of module outputs.
        """
        nodes = list(self.fx_nodes())
        assert len(nodes) > 0, "Cannot extract module inputs from empty nodes."
        # Need ordered set. Emulate with dict.
        module_outputs: Dict[torch.fx.Node, None] = {}
        node_set: Set[torch.fx.Node] = set(nodes)

        for node in nodes:
            if any(user not in node_set for user in node.users):
                module_outputs[node] = None
        return list(module_outputs.keys())

    def build_module(self, module_names: Dict[str, int]) -> torch.fx.GraphModule:
        """
        Constructs the fx.GraphModule for this node, registering submodules as necessary.

        Args:
            module_names: A dictionary of module names and their counts. This is used to
                generate unique module names for submodules. This should be an empty
                dictionary when the method is called on a root module.
        """
        module_class_name = self._stack_meta.qualified_module_class_name
        fx_graph = torch.fx.Graph()
        copy_env: Dict[torch.fx.Node, torch.fx.Node] = {}

        def _arg_transform(node: torch.fx.Node) -> torch.fx.Node:
            return copy_env[node]

        ref_inputs = self.module_inputs()
        for node in ref_inputs:
            copy_env[node] = fx_graph.placeholder(node.name, node.type)
            copy_env[node].meta = copy.copy(node.meta)

        for ir_node in self._nodes:
            if isinstance(ir_node, _LeafNode):
                fx_node = ir_node.fx_node
                copy_env[fx_node] = fx_graph.node_copy(
                    fx_node, arg_transform=_arg_transform
                )
                continue

            assert isinstance(ir_node, _ModuleNode)
            # Create fx.GraphModule for child submodule.
            submodule = ir_node.build_module(module_names)
            ref_submodule_inputs = ir_node.module_inputs()
            ref_submodule_outputs = ir_node.module_outputs()
            unique_submodule_name = _get_unique_module_name(
                module_names, ir_node.stack_meta.module_display_name
            )
            # Link the newly generated sub fx.GraphModule with the root reference module.
            # This step is essential to meet the needs of the subsequent fx.GraphModule initialization
            # for the fx.GraphModule being created by this method.
            # The initialization of fx.GraphModule will replicate all necessary attributes from a reference
            # fx.GraphModule for the fx.Graph. While the root reference module possesses all
            # parameters and buffers, it does not include the newly created sub fx.GraphModule.
            # Therefore, it's necessary to register it under the root reference at this stage.
            self._reference_module.add_submodule(unique_submodule_name, submodule)

            # create call_module fx.Node
            submodule_node = fx_graph.call_module(
                unique_submodule_name,
                tuple(_arg_transform(node) for node in ref_submodule_inputs),
            )
            if len(ref_submodule_outputs) > 1:
                # Module node has multiple output. Create 'getitem' node for each output.
                submodule_node.meta["val"] = tuple(
                    ref_output.meta.get("val") for ref_output in ref_submodule_outputs
                )
                for i, ref_output in enumerate(ref_submodule_outputs):
                    getitem_node = fx_graph.call_function(
                        operator.getitem,
                        args=(submodule_node, i),
                        type_expr=ref_output.type,
                    )
                    getitem_node.meta = copy.copy(ref_output.meta)
                    # Make a copy for "nn_module_stack" since the current module will be
                    # popped from the stack for this 'getitem' node.
                    getitem_node.meta["nn_module_stack"] = copy.copy(
                        ref_output.meta["nn_module_stack"]
                    )
                    # The node is associated with the parent module.
                    getitem_node.meta["nn_module_stack"].popitem()
                    copy_env[ref_output] = getitem_node
            else:
                # Module node has single output. Use module node directly.
                copy_env[ref_submodule_outputs[0]] = submodule_node
                submodule_node.meta = copy.copy(ref_submodule_outputs[0].meta)

            # Update meta for new call_module node.
            if (stack_trace := ir_node.stack_trace) is not None:
                submodule_node.meta["stack_trace"] = stack_trace
            raw_module_stack_meta = ir_node.stack_meta.raw_meta
            assert raw_module_stack_meta is not None
            submodule_node.meta["nn_module_stack"] = copy.copy(raw_module_stack_meta)
            # The node is associated with the parent module.
            submodule_node.meta["nn_module_stack"].popitem()

        new_nodes = fx_graph.nodes
        # Skip if the last node is already 'output'. This is the case for root module.
        # Otherwise create an 'output' node for the inferred outputs.
        if next(iter(reversed(new_nodes))).op != "output":
            ref_submodule_outputs = self.module_outputs()
            new_outputs = [copy_env[ref_output] for ref_output in self.module_outputs()]
            node = fx_graph.output(
                new_outputs[0] if len(new_outputs) == 1 else new_outputs
            )

        graph_module = torch.fx.GraphModule(
            self._reference_module, fx_graph, module_class_name
        )
        if (module_class := self._stack_meta.module_class) is not None:
            graph_module.meta["onnx"] = _pass.GraphModuleOnnxMeta(
                _pass.PackageInfo.from_python_class(module_class)
            )
        return graph_module


class _LeafNode(_IRNode):
    """Representing a single fx.Node."""

    def __init__(self, node: torch.fx.Node, is_exported_program: bool = False):
        self._node = node
        self._stack_meta = _module_stack_meta_from_node(
            node, is_exported_program=is_exported_program
        )

    @property
    def fx_op(self) -> str:
        """Syntax sugar for self.fx_node.op."""
        return self._node.op

    @property
    def fx_node(self) -> torch.fx.Node:
        """Returns the fx.Node this instance represents."""
        return self._node

    @property
    def stack_meta(self) -> _ModuleStackMeta:
        """Returns the module stack meta data associated with this node."""
        return self._stack_meta

    @property
    def stack_trace(self) -> Optional[str]:
        """Returns the stack trace associated with this node."""
        return self.fx_node.meta.get("stack_trace")

    def __str__(self) -> str:
        return f"LeafNode({self._node})"


class Modularize(_pass.Transform):
    """Transforms a flattened `fx.GraphModule` into a modular structure.

    In the flattened `fx.GraphModule`, each `nn.Module` forward call has been traced as
    a sequence of `fx.Node`s. All these `fx.Node`s are flattened and reside in the same
    `fx.GraphModule`. `fx.GraphModule` could be from `torch.export.ExportedProgram` or
    directly generated by `torch._dynamo.export` with torch.nn.Module.

    This pass generates a new `fx.GraphModule`. It groups the flattened `fx.Node`s that belong
    to the same `nn.Module` forward call into a sub `fx.GraphModule`. It then replaces the
    sequence of flattened `fx.Node`s with a single `call_module` node, which is linked with
    the sub `fx.GraphModule` by `node.target`. The sub `fx.GraphModule` is registered as a
    submodule of the new `fx.GraphModule`.

    The process is done based on information from the `nn_module_stack` metadata of each node, i.e.
    `node.meta["nn_module_stack"]`. For more implementation details, see [NOTE: Modularize Pass Implementation].

    An fx submodule under this context can typically be interpreted in three different ways:

        1. As an embodiment of an nn.Module class, which is considered stateless.
        Its execution path can vary depending on the configuration of module initialization,
        which should also be part of the inputs.

        2. As a representation of an nn.Module instance. It maintains the state initialized in the module.
        The execution path can vary based on actual input data.

        3. As a captured call of an nn.Module instance, where the execution path
        is set.

    The generality decreases along this list. Within the scope of this function, the pass
    creates fx submodules according to the third interpretation.

    The first interpretation is the most general case. It requires complex analysis and additional
    metadata and code information to construct its general form. Consider an example nn.Module
    that generates arbitrary submodules based on an initialization configuration file. It's impractical
    to extract this logic for the generated fx submodule to function with arbitrary configuration.

    The second interpretation demands less analysis and is sturdier than the
    first. In most use cases, it's equivalent to the third. It only differs in exceptional situations
    where a complex nn.Module instance is called multiple times, each with a different set of inputs
    leading to a unique execution branching path.

    The third interpretation is the most specific scenario. It necessitates the minimum
    analysis and creates the most stable representation. The drawback is that it
    generates more redundancy than the other two methods. If needed, a subsequent post-processing
    pass can be applied to consolidate completely identical functions and reduce duplication.

    ### Known constraints
    Two successive calls to the same module instance will be conflated. They are indistinguishable.
    This is due to limitations of the current fx metadata "nn_module_stack".

    [NOTE: Modularize pass ordering]
    This pass groups fx nodes into subgraphs that reside within the `call_module` fx node.
    Other fx passes (including some outside the exporter) might not recognize `call_module`.
    They may assume that all nodes are flattened. Hence it is recommended to invoke this pass
    as the last pre onnx export fx pass. If not for this consideration, this operation could
    potentially be relocated anywhere earlier in the pipeline.

    Example:

        >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_ONNX)
        >>> import torch
        >>> from torch.onnx._internal.fx import passes
        >>> from torch.onnx._internal.diagnostics import infra
        >>>
        >>> class CustomModule(torch.nn.Module):
        >>>     def __init__(self):
        >>>         super().__init__()
        >>>         self.embedding = torch.nn.Embedding(10, 32)
        >>>         self.relu = torch.nn.ReLU()
        >>>
        >>>     def forward(self, x):
        >>>         out = self.embedding(x)
        >>>         out = self.relu(out)
        >>>         return out
        >>>
        >>> class TestModule(torch.nn.Module):
        >>>     def __init__(self):
        >>>         super().__init__()
        >>>         self.layer = CustomModule()
        >>>         self.linear = torch.nn.Linear(32, 10)
        >>>
        >>>     def forward(self, x):
        >>>         out = self.layer(x)
        >>>         out = self.linear(out)
        >>>         return out
        >>>
        >>> gm, _ = torch._dynamo.export(TestModule(), aten_graph=True)(torch.tensor([0, 1, 2]))
        >>> gm.print_readable()

        >>> gm = passes.Modularize(infra.DiagnosticContext("test_context", "1.0"), gm).run()
        >>> gm.print_readable()

    """

    @_beartype.beartype
    def __init__(
        self,
        diagnostic_context: diagnostics.DiagnosticContext,
        module: torch.fx.GraphModule,
        is_exported_program: bool = False,
    ):
        super().__init__(diagnostic_context, module)
        self.module = module
        self.is_exported_program = is_exported_program

    @_beartype.beartype
    def _run(self) -> torch.fx.GraphModule:
        # DCE to remove unused nodes.
        # If a submodule is unused, it is hard to analyze which nodes constitutes the submodule
        # outputs. But since it is unused, we can just remove it.
        self.module.graph.eliminate_dead_code()

        reference_module = torch.fx.GraphModule(self.module, self.module.graph)
        root_module_node = _ModuleNode(
            reference_module,
            _ModuleStackMeta(
                nn_module_stack_meta=None, is_exported_program=self.is_exported_program
            ),
        )
        for fx_node in self.module.graph.nodes:
            root_module_node.add_leaf_node(
                _LeafNode(fx_node, is_exported_program=self.is_exported_program)
            )
        return root_module_node.build_module({})
