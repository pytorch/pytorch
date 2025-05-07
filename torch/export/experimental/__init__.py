import copy
import functools
import types
import typing

import torch
from torch.export.exported_program import _decompose_exported_program


def _copy_graph_module_and_signature(
    ep: torch.fx.GraphModule,
) -> tuple[torch.fx.GraphModule, torch.export.graph_signature.ExportGraphSignature]:
    # copy.deepcopy lets the objects override __deepcopy__ methods with graph_copy() and node_copy(),
    # and this can break placeholder names in some particular cases.
    # For example, node copying will avoid Python keywords like 'input', suffixing and renaming to 'input_1'.
    # So we manually overwrite placeholder names by reading the old graph.
    gm = copy.deepcopy(ep.graph_module)
    new_graph_signature = copy.deepcopy(ep.graph_signature)

    # iterate over old/new graph modules
    for old_gm, new_gm in zip(ep.graph_module.modules(), gm.modules()):  # type: ignore[union-attr]
        old_phs = [node for node in old_gm.graph.nodes if node.op == "placeholder"]
        new_phs = [node for node in new_gm.graph.nodes if node.op == "placeholder"]
        # iterate over placeholders
        assert len(old_phs) == len(new_phs)
        for old_node, new_node in zip(old_phs, new_phs):
            new_node.name = old_node.name

    return gm, new_graph_signature  # type: ignore[return-value]


def _remove_detach_pass(
    gm: torch.fx.GraphModule, sig: torch.export.graph_signature.ExportGraphSignature
) -> None:
    with gm._set_replace_hook(sig.get_replace_hook()):
        for node in list(reversed(gm.graph.nodes)):
            if node.op != "call_function":
                continue
            if (
                node.target == torch.ops.aten.detach.default
                and len(node.users) == 1
                and next(iter(node.users)).target == torch.ops.aten.detach.default
            ):
                next(iter(node.users)).replace_all_uses_with(node)

    gm.graph.eliminate_dead_code()
    gm.recompile()


def _export_forward_backward(
    ep: torch.export.ExportedProgram, joint_loss_index: int = 0
) -> torch.export.ExportedProgram:
    """
    WARNING: This API is highly unstable and will be subject to change in the future.
    """
    from torch._decomp import core_aten_decompositions

    ep = _decompose_exported_program(
        ep,
        cia_to_decomp={},
        python_decomp_table=core_aten_decompositions(),
        joint_loss_index=joint_loss_index,
        # For serialization purpose, we don't want to decompose custom triton ops.
        # If users would like to decompose custom triton ops, they could do it
        # with run_decompositions() API.
        decompose_custom_triton_ops=False,
    )
    gm, new_graph_signature = _copy_graph_module_and_signature(ep)
    _remove_detach_pass(gm, new_graph_signature)

    return ep._update(gm, new_graph_signature)


@typing.no_type_check
def _sticky_export(forward_func, dynamic_shapes_callback=None):
    """
    Lazily export the model on first forward call.
    Usage:
        model.forward = _sticky_export(model.forward, dynamic_shapes_callback=callback)
    """
    model = forward_func.__self__
    original_forward = forward_func.__func__

    @functools.wraps(forward_func)
    def wrapper(*args, **kwargs):
        # Unpatch forward to avoid recursion during export
        model.forward = types.MethodType(original_forward, model)

        dynamic_shapes_spec = None
        if dynamic_shapes_callback:
            dynamic_shapes_spec = dynamic_shapes_callback(*args, **kwargs)

        try:
            exported = torch.export.export(
                model,
                args,
                kwargs,
                dynamic_shapes=dynamic_shapes_spec,
            ).module()
            wrapper._exported_artifact = exported
        finally:
            # Restore the wrapper after export
            model.forward = wrapper

        return exported(*args, **kwargs)

    return wrapper
