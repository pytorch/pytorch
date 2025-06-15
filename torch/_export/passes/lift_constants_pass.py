# mypy: allow-untyped-defs
import collections
import logging
from typing import Any, Optional, Union

import torch
from torch._export.verifier import SpecViolationError
from torch._guards import detect_fake_mode
from torch._library.fake_class_registry import FakeScriptObject
from torch._subclasses.fake_tensor import unset_fake_temporarily
from torch.export.exported_program import (
    ArgumentSpec,
    CustomObjArgument,
    ExportGraphSignature,
    InputKind,
    InputSpec,
    TensorArgument,
)
from torch.fx._symbolic_trace import _ConstantAttributeType
from torch.fx.graph_module import _get_attr


log = logging.getLogger(__name__)


class ConstantAttrMap(collections.abc.MutableMapping):
    """A mapping class that understands how to use module constants (tensors,
    ScriptObjects, FakeScriptObjects) as keys. We store tensors and FakeScriptObjects normally,
    but ScriptObjects are stored by hash, because different torch.ScriptObjects can point to
    the same underlying value (but we guarantee that they will `hash()` to the same value
    if that's the case).
    """

    def __init__(self) -> None:
        # Underlying dict that we use to implement this mapping.
        self._constant_attrs: dict[
            Union[int, torch.Tensor, FakeScriptObject, torch.utils._pytree.TreeSpec],
            list[Any],
        ] = {}
        # Map from the hash(ScriptObject) to the ScriptObject itself. Used for
        # APIs like `__iter__` that should look like they're returning the
        # original ScriptObjects.
        self._script_object_map: dict[int, torch.ScriptObject] = {}

    def __getitem__(self, key: _ConstantAttributeType) -> Any:
        real_key = hash(key) if isinstance(key, torch.ScriptObject) else key
        assert isinstance(real_key, (int, torch.Tensor, FakeScriptObject))
        return self._constant_attrs[real_key]

    def __setitem__(self, key: _ConstantAttributeType, value):
        # we shouldn't actually call this, should go to add() instead to handle aliasing
        raise NotImplementedError(
            """Directly setting values for ConstantAttrMap is not supported, please use add(key, value) instead.
The same key can be mapped to multiple values, for handling constant aliasing."""
        )

    def add(self, key: _ConstantAttributeType, value: Any) -> None:
        if isinstance(key, torch.ScriptObject):
            if hash(key) not in self._constant_attrs:
                self._constant_attrs[hash(key)] = []
            self._constant_attrs[hash(key)].append(value)
            self._script_object_map[hash(key)] = key
        elif isinstance(key, (torch.Tensor, FakeScriptObject)):
            if key not in self._constant_attrs:
                self._constant_attrs[key] = []
            self._constant_attrs[key].append(value)
        else:
            raise TypeError(
                f"Expected key to be a tensor or ScriptObject, got {type(key)}"
            )

    def __delitem__(self, key: _ConstantAttributeType):
        real_key = hash(key) if isinstance(key, torch.ScriptObject) else key

        del self._constant_attrs[real_key]

    def __iter__(self):
        for key in self._constant_attrs:
            if isinstance(key, int):
                yield self._script_object_map[key]
            else:
                yield key

    def __len__(self):
        return len(self._constant_attrs)

    def __contains__(self, key: object) -> bool:
        real_key = hash(key) if isinstance(key, torch.ScriptObject) else key
        return real_key in self._constant_attrs


def get_constant_fqn(node: torch.fx.Node, constant_name: str) -> str:
    # The FQN of the constant tensor in the state dict should
    # correspond to the module where the constant tensor was
    # originally used.
    if len(node.meta["nn_module_stack"]) == 0:
        return constant_name
    parent_fqn = list(node.meta["nn_module_stack"].values())[-1][0]
    if len(parent_fqn) > 0:
        return f"{parent_fqn}.{constant_name}"
    else:
        return constant_name


def _get_first_fqn(
    const_attrs: ConstantAttrMap,
    key: _ConstantAttributeType,
) -> Any:
    fqns = const_attrs.get(key)
    return fqns[0] if fqns else None


def _unused_constant(node: torch.fx.Node) -> Optional[list[torch.fx.Node]]:
    """
    If there is a tensor constant created while tracing, here is how the graph
    looks like:

        %_tensor_constant0 : [num_users=1] = get_attr[target=_tensor_constant0]
        %lift_fresh_copy : [num_users=1] = call_function[target=torch.ops.aten.lift_fresh_copy.default](args = (%_tensor_constant0,))
        %detach_ : [num_users=?] = call_function[target=torch.ops.aten.detach_.default](args = (%lift_fresh_copy,))

    To check to see if the tensor constant is being used, we want to traverse to
    the detach node to see if it's actually being used.

    This function returns None if this constant is being used, otherwise it returns the
    lift_fresh and detach node to be removed later.
    """  # noqa: B950
    if len(node.users) > 1:
        return None

    lift_fresh_node = next(iter(node.users.keys()))
    if not (
        lift_fresh_node.op == "call_function"
        and lift_fresh_node.target
        in (
            torch.ops.aten.lift_fresh.default,
            torch.ops.aten.lift_fresh_copy.default,
        )
    ):
        return None

    if len(lift_fresh_node.users) > 1:
        return None

    detach_node = next(iter(lift_fresh_node.users.keys()))
    if not (
        detach_node.op == "call_function"
        and detach_node.target
        in (
            torch.ops.aten.detach_.default,
            torch.ops.aten.detach.default,
        )
    ):
        return None

    if len(detach_node.users) > 0:
        return None
    else:
        return [detach_node, lift_fresh_node, node]


def lift_constants_pass(
    gm: torch.fx.GraphModule,
    graph_signature: ExportGraphSignature,
    constant_attrs: ConstantAttrMap,
) -> dict[str, _ConstantAttributeType]:
    """
    Takes a graph module, graph signature, and modifies them implace to lift any
    constants (tensors or custom classes) as inputs to the graph. Returns a
    dictionary of names to constants.

    Arguments:
        gm (torch.fx.GraphModule): The graph module containing the graph and constants to lift.
        graph_signature (ExportGraphSignature): This graph signature will be
            mutated to add additional CONSTANT_TENSOR and CUSTOM_OBJ inputs.
        constant_attrs (ConstantAttr): A mapping from a constant value to its
            fully-qualified path in `gm`. This is used to maintain consistent
            location of constants between the original module and the exported
            version.

    Returns:
        A dictionary of fqn => constant value.
    """
    all_constants: dict[str, _ConstantAttributeType] = {}

    inputs = graph_signature.input_specs
    num_custom_obj = sum(
        input_specs.kind == InputKind.CUSTOM_OBJ for input_specs in inputs
    )
    num_tensor_constants = sum(
        input_specs.kind == InputKind.CONSTANT_TENSOR for input_specs in inputs
    )

    fake_mode = detect_fake_mode(
        tuple(node.meta["val"] for node in gm.graph.nodes if node.op == "placeholder")
    )

    first_user_input_loc, first_user_input = 0, next(iter(gm.graph.nodes))
    used_target_names = set()
    for node in gm.graph.nodes:
        if node.op == "placeholder":
            if node.name in graph_signature.user_inputs:
                first_user_input = node
                break
            used_target_names.add(inputs[first_user_input_loc].target)
            first_user_input_loc += 1
        # If we ever hit here, it means that
        # there was no user input so the constants
        # should be inserted right before the first
        # non-placeholder node.
        if node.op != "placeholder":
            first_user_input = node
            break

    lifted_objs = ConstantAttrMap()
    renamed_targets = {}
    for node in list(gm.graph.nodes):
        if node.op == "get_attr":
            if nodes_to_remove := _unused_constant(node):
                # Remove the node if it's not being used
                for node_rm in nodes_to_remove:
                    gm.graph.erase_node(node_rm)
                continue

            constant_val = _get_attr(gm, node.target)
            # These are not hashable and not gonna be lifted
            # so we can skip them earlier
            if isinstance(constant_val, torch.fx.GraphModule):
                continue
            if "LoweredBackendModule" in type(constant_val).__name__:
                continue
            if "AOTInductorRunnerWrapper" in type(constant_val).__name__:
                continue
            if isinstance(constant_val, torch.utils._pytree.TreeSpec):
                continue

            if constant_val in lifted_objs:
                # We already lifted this constant elsewhere. Just rewrite uses
                # of this get_attr to point to the already-existing placeholder
                # node.
                const_placeholder_node = _get_first_fqn(lifted_objs, constant_val)
                node.replace_all_uses_with(const_placeholder_node)
                gm.graph.erase_node(node)
                renamed_targets[node.name] = const_placeholder_node.name
                continue

            # For ScriptObject, Tensor and FakeScriptObject constants:
            # First check if the constant was an attribute on some module by
            # consulting `constant_attrs` map. If it is, use the fqn that keeps
            # its location consistent with the eager module.
            #
            # If it's not in the `constant_attrs` map, that means it's an inline
            # constant (e.g. x + torch.tensor(0)), and thus did not have a
            # specific location in the eager module. In that case, just generate
            # some name and attach it to the module in which it was used.
            if isinstance(constant_val, (torch.ScriptObject, FakeScriptObject)):
                constant_kind = InputKind.CUSTOM_OBJ
                constant_fqn = _get_first_fqn(constant_attrs, constant_val)
                if constant_fqn is not None:
                    constant_name = constant_fqn.replace(".", "_")
                else:
                    constant_name = f"lifted_custom_{num_custom_obj}"
                    constant_fqn = get_constant_fqn(node, constant_name)
                    while constant_fqn in used_target_names:
                        num_custom_obj += 1
                        constant_name = f"lifted_custom_{num_custom_obj}"
                        constant_fqn = get_constant_fqn(node, constant_name)
                    num_custom_obj += 1
            elif isinstance(constant_val, torch.Tensor):
                # Remove the parameterness of constant_val
                if isinstance(constant_val, torch.nn.Parameter):
                    log.debug(
                        "%s created when tracing %s is a parameter. But "
                        "it's not registered with register_parameter(). export will treat it as a constant tensor",
                        str(node.target),
                        str(node.meta.get("stack_trace", "<unknown stack>")),
                    )
                    # We get the real data out of the parameter by disabling the surrounding fake mode.
                    with unset_fake_temporarily():
                        constant_val = constant_val.data
                constant_kind = InputKind.CONSTANT_TENSOR
                constant_fqn = _get_first_fqn(constant_attrs, constant_val)
                if constant_fqn is not None:
                    constant_name = constant_fqn.replace(".", "_")
                else:
                    constant_name = f"lifted_tensor_{num_tensor_constants}"
                    constant_fqn = get_constant_fqn(node, constant_name)
                    while constant_fqn in used_target_names:
                        num_tensor_constants += 1
                        constant_name = f"lifted_tensor_{num_tensor_constants}"
                        constant_fqn = get_constant_fqn(node, constant_name)
                    num_tensor_constants += 1
            else:
                raise SpecViolationError(
                    f"getattr node {node} referencing unsupported type {type(constant_val)}"
                )

            with gm.graph.inserting_before(first_user_input):
                # Insert the constant node before the first user input
                const_placeholder_node = gm.graph.placeholder(constant_name)
                # match target name with its node name in case there is name collision
                # and suffix is added to node name in fx
                const_placeholder_node.target = const_placeholder_node.name

                for k, v in node.meta.items():
                    const_placeholder_node.meta[k] = v

                # Once the FQN has been used, remove nn_module_stack, stack_trace
                const_placeholder_node.meta.pop("nn_module_stack")
                const_placeholder_node.meta.pop("stack_trace", None)

                input_spec_arg: ArgumentSpec
                if isinstance(constant_val, torch.Tensor):
                    if fake_mode is not None:
                        const_placeholder_node.meta["val"] = fake_mode.from_tensor(
                            constant_val, static_shapes=True
                        )
                        const_placeholder_node.meta["val"].constant = constant_val
                    else:
                        const_placeholder_node.meta["val"] = constant_val
                    input_spec_arg = TensorArgument(name=const_placeholder_node.name)
                elif isinstance(constant_val, torch._C.ScriptObject):
                    class_fqn = constant_val._type().qualified_name()  # type: ignore[attr-defined]
                    const_placeholder_node.meta["val"] = CustomObjArgument(
                        constant_fqn, class_fqn
                    )
                    input_spec_arg = CustomObjArgument(
                        name=const_placeholder_node.name, class_fqn=class_fqn
                    )
                elif isinstance(constant_val, FakeScriptObject):
                    class_fqn = constant_val.script_class_name
                    const_placeholder_node.meta["val"] = CustomObjArgument(
                        constant_fqn, class_fqn, constant_val
                    )
                    input_spec_arg = CustomObjArgument(
                        name=const_placeholder_node.name,
                        class_fqn=class_fqn,
                        fake_val=constant_val,
                    )
                else:
                    raise SpecViolationError(
                        f"tried to lift unsupported type {type(constant_val)} from node {node.format_node()}"
                    )

                lifted_objs.add(constant_val, const_placeholder_node)
                node.replace_all_uses_with(const_placeholder_node)
                gm.graph.erase_node(node)

                renamed_targets[node.name] = const_placeholder_node.name

                # Add the constant as a buffer to the graph signature
                graph_signature.input_specs.insert(
                    first_user_input_loc,
                    InputSpec(
                        kind=constant_kind,
                        arg=input_spec_arg,
                        target=constant_fqn,
                    ),
                )
                if constant_val in constant_attrs:
                    for fqn in constant_attrs[constant_val]:
                        all_constants[fqn] = constant_val
                else:
                    all_constants[constant_fqn] = constant_val
                first_user_input_loc += 1

    for spec in graph_signature.output_specs:
        if spec.arg.name in renamed_targets:
            spec.arg.name = renamed_targets[spec.arg.name]

    return all_constants


def rewrite_script_object_meta(
    gm: torch.fx.GraphModule,
) -> dict[str, _ConstantAttributeType,]:
    """When tracing, we produce a graph with FakeScriptObject in the
    meta["val"].

    For now, we rewrie meta["val"] to be a placeholder CustomObjArgument
    """
    constants: dict[
        str,
        _ConstantAttributeType,
    ] = {}
    for node in gm.graph.nodes:
        if "val" not in node.meta:
            continue

        old_meta = node.meta["val"]

        if isinstance(old_meta, torch.ScriptObject):
            class_fqn = old_meta._type().qualified_name()  # type: ignore[attr-defined]
            new_meta = CustomObjArgument(node.name, class_fqn)
            constants[node.name] = old_meta
            node.meta["val"] = new_meta

        elif isinstance(old_meta, FakeScriptObject):
            class_fqn = old_meta.script_class_name  # type: ignore[attr-defined]
            new_meta = CustomObjArgument(node.name, class_fqn, old_meta)
            constants[node.name] = old_meta
            node.meta["val"] = new_meta

    return constants


def _materialize_and_lift_constants(
    gm: torch.fx.GraphModule,
    export_graph_signature: ExportGraphSignature,
    constant_attrs: ConstantAttrMap,
) -> dict[str, _ConstantAttributeType]:
    constants = rewrite_script_object_meta(gm)
    constants.update(lift_constants_pass(gm, export_graph_signature, constant_attrs))
    return constants
