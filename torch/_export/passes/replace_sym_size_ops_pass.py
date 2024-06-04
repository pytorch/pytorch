# mypy: allow-untyped-defs
from typing import Dict

import torch

replacements: Dict[torch._ops.OpOverloadPacket, torch._ops.OpOverload] = {
    torch.ops.aten.sym_size: torch.ops.aten.sym_size.int,
    torch.ops.aten.sym_stride: torch.ops.aten.sym_stride.int,
    torch.ops.aten.sym_numel: torch.ops.aten.sym_numel.default,
}


def _replace_sym_size_ops_pass(gm: torch.fx.GraphModule):
    for module in gm.modules():
        if not isinstance(module, torch.fx.GraphModule):
            continue
        for node in module.graph.nodes:
            if node.target in replacements:
                node.target = replacements[node.target]
