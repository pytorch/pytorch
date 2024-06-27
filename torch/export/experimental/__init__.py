import copy

import torch
from torch.export.exported_program import _decompose_exported_program


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
        decomp_table=core_aten_decompositions(),
        _preserve_ops=(),  # type: ignore[arg-type]
        joint_loss_index=joint_loss_index,
    )
    gm = copy.deepcopy(ep.graph_module)
    new_graph_signature = copy.deepcopy(ep.graph_signature)
    _remove_detach_pass(gm, new_graph_signature)

    return ep._update(gm, new_graph_signature)
