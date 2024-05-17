import torch


def _node_metadata_hook(node: torch.fx.Node, stack_trace: str) -> None:
    """
    Hook for adding the appropriate metadata to nodes that are created during a
    pass using graph.create_node. An example of how to use it:

    ```
    with gm.graph._set_create_node_hook(
        functools.partial(_node_metadata_hook, stack_trace="file")
    ):
        pass(gm)
    ```

    This hook should not work for all generic cases -- specifically it assumes
    that nodes being added are only call_function nodes, and copies over the
    first argument node's nn_module_stack.
    """
    assert node.op == "call_function" and callable(node.target)

    arg_meta = [arg.meta for arg in node.args if isinstance(arg, torch.fx.Node)]
    assert len(arg_meta) >= 1
    arg_meta = arg_meta[0]

    if (
        isinstance(node.target, torch._ops.OpOverload)
        and len(node.target._schema.returns) == 0
    ):
        node.meta["val"] = None
    else:
        fake_args = [
            arg.meta["val"] if isinstance(arg, torch.fx.Node) else arg
            for arg in node.args
        ]
        fake_res = node.target(*fake_args)
        node.meta["val"] = fake_res

    node.meta["stack_trace"] = stack_trace
    node.meta["nn_module_stack"] = arg_meta["nn_module_stack"]
    node.meta["torch_fn"] = (
        f"{node.target.__name__}_0",
        f"{node.target.__class__.__name__}.{node.target.__name__}",
    )
