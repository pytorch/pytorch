import functools

import torch
from torch._export.passes._node_metadata_hook import (
    _node_metadata_hook,
    _set_node_metadata_hook,
)


def insert_custom_op_guards(gm: torch.fx.GraphModule, ops_to_guard: list[str]) -> None:
    """
    This is used by draft_export to insert guards in front of calls to custom
    operators which have a generated fake kernel.
    """
    for node in gm.graph.nodes:
        if node.op == "call_function" and str(node.target) in ops_to_guard:
            with _set_node_metadata_hook(
                gm,
                functools.partial(
                    _node_metadata_hook, stack_trace=node.meta.get("stack_trace")
                ),
            ), gm.graph.inserting_before(node):
                for arg in (*node.args, *node.kwargs.values()):
                    if isinstance(arg, torch.fx.Node) and isinstance(
                        arg.meta.get("val"), torch.Tensor
                    ):
                        val = arg.meta["val"]
                        gm.graph.call_function(
                            torch.ops.aten._assert_tensor_metadata.default,
                            args=(arg,),
                            kwargs={
                                "dtype": val.dtype,
                                "device": val.device,
                                "layout": val.layout,
                            },
                        )

    gm.recompile()
