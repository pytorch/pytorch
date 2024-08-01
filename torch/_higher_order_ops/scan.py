import torch.utils._pytree as pytree


def scan(
    combine_fn: Callable[[pytree.PyTree, pytree.PyTree], pytree.PyTree],
    init: pytree.PyTree,
    xs: pytree.PyTree,
):
    pass
