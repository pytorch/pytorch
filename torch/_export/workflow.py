import dataclasses
from typing import Callable, Tuple, List, Dict

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
    mutations: List[Tuple[torch.fx.Node, torch.fx.Node]]
    _input_mutation_to_pos: Dict[torch.fx.Node, int] = dataclasses.field(default_factory=dict)

    def __post_init__(self):
        mutations_set_for_placeholders = set([src for src, dest in self.mutations if src.op == "placeholder"])
        for ix, node in enumerate(self.fw_module.graph.nodes):
            if node.op == "placeholder":
                if node in mutations_set_for_placeholders:
                    self._input_mutation_to_pos[node] = ix

    def transform(self, *passes: Callable) -> "ExportedProgram":
        res = PassManager(list(passes))(self.fw_module)
        assert res is not None
        transformed = dataclasses.replace(self, fw_module=res.graph_module)
        return transformed

    def __call__(self, *args):
        flat_args, _ = pytree.tree_flatten(args)
        print(self.fw_module.graph)
        output = self.fw_module(*flat_args)
        # Replay the mutations, we don't want to do this in the graph itself
        # as it would break the "functional" invariant.
        for ix, mutation in enumerate(self.mutations):
            src, dest = mutation
            if src.op == "placeholder":
                assert src in self._input_mutation_to_pos
                flat_args[self._input_mutation_to_pos[src]].copy_(output[ix])
            if src.op == "get_attr":
                assert hasattr(self.fw_module, str(src.target))
                gm_param_or_buffer = getattr(self.fw_module, str(src.target))
                gm_param_or_buffer.copy_(output[ix])
        return pytree.tree_unflatten(output[len(self.mutations):], self.out_spec)
