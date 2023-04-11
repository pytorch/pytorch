import dataclasses
from typing import Callable, Tuple

import torch
from torch.fx.passes.pass_manager import PassManager
import torch.utils._pytree as pytree
from torch.utils._pytree import TreeSpec

@dataclasses.dataclass
class ExportedProgram:
    fw_module: torch.fx.GraphModule
    example_inputs: Tuple[torch.Tensor, ...]
    in_spec: TreeSpec
    out_spec: TreeSpec

    def transform(self, *passes: Callable) -> "ExportedProgram":
        res = PassManager(list(passes))(self.fw_module)
        assert res is not None
        transformed = dataclasses.replace(self, fw_module=res.graph_module)
        return transformed

    def __call__(self, *args):
        flat_args, _ = pytree.tree_flatten(args)
        output = self.fw_module(*flat_args)
        return pytree.tree_unflatten(output, self.out_spec)
