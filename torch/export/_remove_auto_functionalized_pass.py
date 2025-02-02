# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import torch
from torch._higher_order_ops.auto_functionalize import (
    auto_functionalized,
    auto_functionalized_v2,
)
from torch._inductor.fx_passes.post_grad import decompose_auto_functionalized
from torch.export import ExportedProgram
from torch.fx import Graph


def remove_self_clone(graph: Graph) -> None:
    for node in graph.nodes:
        if node.target == torch.ops.aten.copy_.default and node.args[0] == node.args[1]:
            node.replace_all_uses_with(node.args[0])
            graph.erase_node(node)


def unsafe_remove_auto_functionalized_pass(
    ep: ExportedProgram,
) -> ExportedProgram:
    """
    This pass removes an instances of the higher order op 'auto_functionalized',
    and modifies the calling EP inplace to have the original mutator op.
    This pass doesn't perform safety checks to make sure that this inplace mutation is safe.
    """

    with ep.graph_module._set_replace_hook(ep.graph_signature.get_replace_hook()):
        for module in ep.graph_module.modules():
            if not isinstance(module, torch.fx.GraphModule):
                continue
            for node in ep.graph.nodes:
                if (
                    node.op == "call_function" and node.target is auto_functionalized
                ) or (
                    node.op == "call_function" and node.target is auto_functionalized_v2
                ):
                    func = node.args[0]
                    assert isinstance(func, torch._ops.OpOverload)
                    # re-inplace everything
                    node.meta["only_clone_these_tensors"] = []
            decompose_auto_functionalized(ep.graph)
            remove_self_clone(ep.graph)
            ep.graph.eliminate_dead_code()

    return ep
