from typing import Tuple

import torch
import torch.fx
import torch.utils._pytree as pytree

from .graph_signature import ExportGraphSignature


def _remove_unneccessary_copy_op_pass(
    gm: torch.fx.GraphModule, new_graph_signature: ExportGraphSignature
) -> Tuple[torch.fx.GraphModule, ExportGraphSignature]:
    """
    Removes redundant copy_ node that was introduced due to mutated buffer.
    """
    with gm._set_replace_hook(new_graph_signature.get_replace_hook()):
        for node in gm.graph.nodes:
            if node.op == "output":
                args, _ = pytree.tree_flatten(node.args)
                for out in args:
                    if (
                        isinstance(out, torch.fx.Node)
                        and out.name in new_graph_signature.buffers_to_mutate
                    ):
                        if (
                            out.op == "call_function"
                            and out.target == torch.ops.aten.copy.default
                        ):
                            out.replace_all_uses_with(out.args[1])
                            gm.graph.erase_node(out)
        gm.recompile()
    return gm, new_graph_signature
