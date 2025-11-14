# mypy: allow-untyped-defs


def maybe_view(tensor, size, check_same_size=True):
    if check_same_size and tensor.size() == size:
        return tensor
    return tensor.contiguous().view(size)


def maybe_unexpand(tensor, old_size, check_same_size=True):
    if check_same_size and tensor.size() == old_size:
        return tensor
    num_unsqueezed = tensor.dim() - len(old_size)
    expanded_dims = [
        dim
        for dim, (expanded, original) in enumerate(
            zip(tensor.size()[num_unsqueezed:], old_size)
        )
        if expanded != original
    ]

    for _ in range(num_unsqueezed):
        tensor = tensor.sum(0, keepdim=False)
    for dim in expanded_dims:
        tensor = tensor.sum(dim, keepdim=True)
    return tensor
