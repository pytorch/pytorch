import torch
from torch.fx.passes.infra.pass_base import PassBase, PassResult


class _RemoveRuntimeAssertionsPass(PassBase):
    """
    Remove runtime assertions inserted by the
    _AddRuntimeAssertionsForInlineConstraintsPass.
    """

    def call(self, graph_module) -> PassResult:
        modified = False
        for module in graph_module.modules():
            if not isinstance(module, torch.fx.GraphModule):
                continue
            for node in module.graph.nodes:
                if node.target == torch.ops.aten._assert_async.msg:
                    assert_async_node = node
                    if len(assert_async_node.users) > 0:
                        continue
                    module.graph.erase_node(assert_async_node)
                    # the upstream scalar_tensor <- {le, ge} <- sym_size
                    # linear chain of nodes of nodes is removed by the
                    # downstream dead code elimination
                    modified = True
        return PassResult(graph_module, modified)
