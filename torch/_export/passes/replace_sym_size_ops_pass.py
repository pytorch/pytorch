from typing import Dict

import torch
from torch.fx.passes.infra.pass_base import PassBase

replacements: Dict[torch._ops.OpOverloadPacket, torch._ops.OpOverload] = {
    torch.ops.aten.sym_size: torch.ops.aten.sym_size.int,
    torch.ops.aten.sym_stride: torch.ops.aten.sym_stride.int,
    torch.ops.aten.sym_numel: torch.ops.aten.sym_numel.default,
}


class _ReplaceSymSizeOpPass(PassBase):
    """
    Replace torch.ops.aten.sym_size with torch.ops.aten.sym_size.int
    and torch.ops.aten.sym_stride with torch.ops.aten.sym_stride.int
    """

    def call(self, graph_module):
        for module in graph_module.modules():
            if not isinstance(module, torch.fx.GraphModule):
                continue
            for node in module.graph.nodes:
                if node.target in replacements:
                    node.target = replacements[node.target]
