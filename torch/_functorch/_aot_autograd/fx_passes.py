import torch
from torch import fx


def is_primal(node: fx.Node) -> bool:
    return node.op == "placeholder" and "tangents" not in str(node.target)


def update_set_to_write_into_primal(graph: fx.Graph) -> None:
    """
    We replace:
    ```
    set_1 = aten.set_(primal_X, Y1)
    set_2 = aten.set_(set_1, Y2)
    ```
    to:
    ```
    set_1 = aten.set_(primal_X, Y1)
    set_2 = aten.set_(primal_X, Y2)
    ```
    Note that this process is iterative and will handle any number (>=2) of nested `.set_()`.

    Q: Why do we need this pass, if we can just change the proxy_tensor tracing logic to not
    clobber existing tensor proxies (which automatically makes the second `.set_` use `primal_X`)?
    A: We tried that but it turns out to break many existing unit tests, and there is not enough
    confidence that such change will not break production.
    (See details in https://github.com/pytorch/pytorch/pull/130577#issuecomment-2229556623)
    """
    node_list = list(graph.nodes)
    primal_inputs = [*filter(is_primal, node_list)]
    for i, n in enumerate(node_list):
        if n.op == "call_function" and n.target is torch.ops.aten.set_.source_Tensor:
            if (
                n.args[0].target is torch.ops.aten.set_.source_Tensor
                and n.args[0].args[0] in primal_inputs
            ):
                n.args = (n.args[0].args[0], n.args[1])
            assert n.args[0] in primal_inputs, (
                "Violated assumption: every `.set_` node should be setting into the primal input. "
                f"Please report a bug to PyTorch. Violating graph: {graph}"
            )
