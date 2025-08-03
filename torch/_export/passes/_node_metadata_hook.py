# mypy: allow-untyped-defs
import contextlib
from typing import Optional

import torch
from torch.fx.graph_module import GraphModule


_EMPTY_NN_MODULE_STACK_KEY = "_empty_nn_module_stack_from_metadata_hook"


def _node_metadata_hook(node: torch.fx.Node, stack_trace: Optional[str] = None) -> None:
    """
    Hook for adding the appropriate metadata to nodes that are created during a
    pass using graph.create_node. An example of how to use it:

    ```
    with _set_node_metadata_hook(gm,
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
    node.meta["nn_module_stack"] = arg_meta.get(
        "nn_module_stack",
        {
            _EMPTY_NN_MODULE_STACK_KEY: (
                _EMPTY_NN_MODULE_STACK_KEY,
                _EMPTY_NN_MODULE_STACK_KEY,
            )
        },
    )

    node.meta["torch_fn"] = (
        f"{node.target.__name__}_0",
        f"{node.target.__class__.__name__}.{node.target.__name__}",
    )


@contextlib.contextmanager
def _set_node_metadata_hook(gm: torch.fx.GraphModule, f):
    """
    Takes a callable which will be called after we create a new node. The
    callable takes the newly created node as input and returns None.
    """
    assert callable(f), "node_metadata_hook must be a callable."

    # Add the hook to all submodules
    for m in gm.modules():
        if isinstance(m, GraphModule):
            m._register_create_node_hook(f)
    try:
        yield
    finally:
        # Restore hook for all submodules
        for m in gm.modules():
            if isinstance(m, GraphModule):
                m._unregister_create_node_hook(f)
