from typing import Dict, Union

import torch
from torch._export.verifier import SpecViolationError
from torch._guards import detect_fake_mode
from torch.export.custom_obj import ScriptObjectMeta
from torch.export.exported_program import (
    CustomObjArgument,
    ExportGraphSignature,
    InputKind,
    InputSpec,
    TensorArgument,
)


def lift_constants_pass(
    gm: torch.fx.GraphModule,
    graph_signature: ExportGraphSignature,
) -> Dict[str, Union[torch.Tensor, torch._C.ScriptObject]]:
    """
    Takes a graph module, graph signature, and modifies them implace to lift any
    constants (tensors or custom classes) as inputs to the graph. Returns a
    dictionary of names to constants.
    """
    all_constants: Dict[str, Union[torch.Tensor, torch._C.ScriptObject]] = {}

    if len([node for node in gm.graph.nodes if node.op == "placeholder"]) == 0:
        return {}

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
    assert fake_mode is not None

    first_user_input_loc, first_user_input = 0, None
    for node in gm.graph.nodes:
        if node.op == "placeholder" and node.name in graph_signature.user_inputs:
            first_user_input = node
            break
        first_user_input_loc += 1

    for node in gm.graph.nodes:
        if node.op == "get_attr":
            constant_val = getattr(gm, node.target)

            if isinstance(constant_val, torch.ScriptObject):
                constant_name = f"_lifted_custom_obj{num_custom_obj}"
                constant_kind = InputKind.CUSTOM_OBJ
                constant_arg_cls = CustomObjArgument  # type: ignore[assignment]
                num_custom_obj += 1
            elif isinstance(constant_val, torch.Tensor):
                constant_name = f"_lifted_tensor_constant{num_tensor_constants}"
                constant_kind = InputKind.CONSTANT_TENSOR
                constant_arg_cls = TensorArgument  # type: ignore[assignment]
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

                for k, v in node.meta.items():
                    const_placeholder_node.meta[k] = v

                # The FQN of the constant tensor in the state dict should
                # correspond to the module where the constant tensor was
                # originally used.
                parent_fqn = list(
                    const_placeholder_node.meta["nn_module_stack"].values()
                )[-1][0]
                constant_fqn = f"{parent_fqn}.{constant_name}"

                if isinstance(constant_val, torch.Tensor):
                    const_placeholder_node.meta["val"] = fake_mode.from_tensor(
                        constant_val, static_shapes=True
                    )
                    const_placeholder_node.meta["val"].constant = constant_val
                elif isinstance(constant_val, torch._C.ScriptObject):
                    const_placeholder_node.meta["val"] = ScriptObjectMeta(constant_fqn)
                else:
                    const_placeholder_node.meta["val"] = constant_val

                node.replace_all_uses_with(const_placeholder_node)
                gm.graph.erase_node(node)

                # Add the constant as a buffer to the graph signature
                graph_signature.input_specs.insert(
                    first_user_input_loc,
                    InputSpec(
                        kind=constant_kind,
                        arg=constant_arg_cls(name=const_placeholder_node.name),
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

    For now, we rewrie meta["val"] to be a placeholder ScriptObjectMeta.
    """
    constants: Dict[str, Union[torch.Tensor, torch._C.ScriptObject]] = {}
    for node in gm.graph.nodes:
        if "val" not in node.meta or not isinstance(
            node.meta["val"], torch.ScriptObject
        ):
            continue

        old_meta = node.meta["val"]
        new_meta = ScriptObjectMeta(node.name)
        constants[node.name] = old_meta
        node.meta["val"] = new_meta

    return constants
