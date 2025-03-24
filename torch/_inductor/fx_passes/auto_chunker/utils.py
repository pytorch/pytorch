from torch.fx import Node
from typing import Sequence, Optional
from torch.utils._pytree import tree_flatten
import torch

def get_args_of_node_type(node: Node) -> Sequence[Node]:
    return [x for x in tree_flatten((node.args, node.kwargs))[0]
        if isinstance(x, Node)]

def use_tangent(node: Node) -> bool:
    """
    Whether the fx node uses tangent input.
    """

    return any(
        arg.op == "placeholder" and "tangent" in arg.target
        for arg in get_args_of_node_type(node)
    )

def compute_tensor_size(*args, count_bytes=True, **kwargs):
    """
    Compute total tensor sizes from fx.Node in args & kwargs.
    """
    flat_args, _ = tree_flatten((args, kwargs))
    tot = 0
    for arg in flat_args:
        if (fake_tensor := get_fake_tensor_from_node_arg(arg)) is None:
            continue
        tot += fake_tensor.numel() * (fake_tensor.dtype.itemsize if count_bytes else 1)
    return tot

def get_fake_tensor_from_node_arg(node: torch.fx.node.Argument) -> Optional[torch.Tensor]:
    if (
        not hasattr(node, "meta")
        or ("val" not in node.meta)
        or not isinstance(node.meta["val"], torch.Tensor)
    ):
        return None
    return node.meta["val"]
