import collections
from typing import Any, Dict, Union

import torch
from torch._export.verifier import SpecViolationError
from torch._guards import detect_fake_mode
from torch.export.exported_program import (
    ArgumentSpec,
    CustomObjArgument,
    ExportGraphSignature,
    InputKind,
    InputSpec,
    TensorArgument,
)


class ConstantAttrMap(collections.abc.MutableMapping):
    """A mapping class that understands how to use module constants (tensors and
    ScriptObjects) as keys. We store tensors normally, but ScriptObjects are
    stored by hash, because different torch.ScriptObjects can point to the same
    underlying value (but we guarantee that they will `hash()` to the same value
    if that's the case).
    """

    def __init__(self):
        # Underlying dict that we use to implement this mapping.
        self._constant_attrs: Dict[Union[int, torch.Tensor], Any] = {}
        # Map from the hash(ScriptObject) to the ScriptObject itself. Used for
        # APIs like `__iter__` that should look like they're returning the
        # original ScriptObjects.
        self._script_object_map: Dict[int, torch.ScriptObject] = {}

    def __getitem__(self, key: Union[torch.Tensor, torch.ScriptObject]) -> Any:
        real_key = hash(key) if isinstance(key, torch.ScriptObject) else key
        assert isinstance(real_key, (int, torch.Tensor))
        return self._constant_attrs[real_key]

    def __setitem__(
        self, key: Union[torch.Tensor, torch.ScriptObject], value: Any
    ) -> None:
        if isinstance(key, torch.ScriptObject):
            self._constant_attrs[hash(key)] = value
            self._script_object_map[hash(key)] = key
        elif isinstance(key, torch.Tensor):
            self._constant_attrs[key] = value
        else:
            raise TypeError(
                f"Expected key to be a tensor or ScriptObject, got {type(key)}"
            )

    def __delitem__(self, key):
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
    parent_fqn = list(node.meta["nn_module_stack"].values())[-1][0]
    if len(parent_fqn) > 0:
        return f"{parent_fqn}.{constant_name}"
    else:
        return constant_name


def lift_constants_pass(
    gm: torch.fx.GraphModule,
    graph_signature: ExportGraphSignature,
    constant_attrs: ConstantAttrMap,
) -> Dict[str, Union[torch.Tensor, torch._C.ScriptObject]]:
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
    all_constants: Dict[str, Union[torch.Tensor, torch._C.ScriptObject]] = {}

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

    first_user_input_loc, first_user_input = 0, None
    for node in gm.graph.nodes:
        if node.op == "placeholder" and node.name in graph_signature.user_inputs:
            first_user_input = node
            break
        first_user_input_loc += 1

    lifted_objs = ConstantAttrMap()
    for node in gm.graph.nodes:
        if node.op == "get_attr":
            constant_val = getattr(gm, node.target)
            if constant_val in lifted_objs:
                # We already lifted this constant elsewhere. Just rewrite uses
                # of this get_attr to point to the already-existing placeholder
                # node.
                const_placeholder_node = lifted_objs[constant_val]
                node.replace_all_uses_with(const_placeholder_node)
                gm.graph.erase_node(node)
                continue

            # For ScriptObject and Tensor constants:
            # First check if the constant was an attribute on some module by
            # consulting `constant_attrs` map. If it is, use the fqn that keeps
            # its location consistent with the eager module.
            #
            # If it's not in the `constant_attrs` map, that means it's an inline
            # constant (e.g. x + torch.tensor(0)), and thus did not have a
            # specific location in the eager module. In that case, just generate
            # some name and attach it to the module in which it was used.
            if isinstance(constant_val, torch.ScriptObject):
                constant_kind = InputKind.CUSTOM_OBJ
                constant_fqn = constant_attrs.get(constant_val)
                if constant_fqn is not None:
                    constant_name = constant_fqn.replace(".", "_")
                else:
                    constant_name = f"lifted_custom_{num_custom_obj}"
                    constant_fqn = get_constant_fqn(node, constant_name)
                    num_custom_obj += 1
            elif isinstance(constant_val, torch.Tensor):
                constant_kind = InputKind.CONSTANT_TENSOR
                constant_fqn = constant_attrs.get(constant_val)
                if constant_fqn is not None:
                    constant_name = constant_fqn.replace(".", "_")
                else:
                    constant_name = f"lifted_tensor_{num_tensor_constants}"
                    constant_fqn = get_constant_fqn(node, constant_name)
                    num_tensor_constants += 1
            elif isinstance(constant_val, torch.fx.GraphModule):
                continue
            elif "LoweredBackendModule" in type(constant_val).__name__:
                continue
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
                else:
                    raise SpecViolationError(
                        f"tried to lift unsupported type {type(constant_val)} from node {node.format_node()}"
                    )

                lifted_objs[constant_val] = const_placeholder_node
                node.replace_all_uses_with(const_placeholder_node)
                gm.graph.erase_node(node)

                # Add the constant as a buffer to the graph signature
                graph_signature.input_specs.insert(
                    first_user_input_loc,
                    InputSpec(
                        kind=constant_kind,
                        arg=input_spec_arg,
                        target=constant_fqn,
                    ),
                )
                all_constants[constant_fqn] = constant_val
                first_user_input_loc += 1

    return all_constants


def rewrite_script_object_meta(
    gm: torch.fx.GraphModule,
) -> Dict[str, Union[torch.Tensor, torch.ScriptObject]]:
    """When tracing, we produce a graph with an actual ScriptObject in the
    meta["val"]. Eventually we want to change this behavior, when FakeMode infra
    for ScriptObjects lands.

    For now, we rewrie meta["val"] to be a placeholder CustomObjArgument
    """
    constants: Dict[str, Union[torch.Tensor, torch._C.ScriptObject]] = {}
    for node in gm.graph.nodes:
        if "val" not in node.meta or not isinstance(
            node.meta["val"], torch.ScriptObject
        ):
            continue

        old_meta = node.meta["val"]
        class_fqn = old_meta._type().qualified_name()  # type: ignore[attr-defined]
        new_meta = CustomObjArgument(node.name, class_fqn)
        constants[node.name] = old_meta
        node.meta["val"] = new_meta

    return constants
