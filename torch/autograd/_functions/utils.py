import torch


def maybe_view(variable, size, check_same_size=True):
    if check_same_size and variable.size() == size:
        return variable
    return variable.contiguous().view(size)


def maybe_unexpand(variable, old_size, check_same_size=True):
    if check_same_size and variable.size() == old_size:
        return variable
    num_unsqueezed = variable.dim() - len(old_size)
    expanded_dims = [dim for dim, (expanded, original)
                     in enumerate(zip(variable.size()[num_unsqueezed:], old_size))
                     if expanded != original]

    for _ in range(num_unsqueezed):
        variable = variable.sum(0, keepdim=False)
    for dim in expanded_dims:
        variable = variable.sum(dim, keepdim=True)
    return variable


_SAME_SIZE = 2
_EXPANDABLE = 1
_NOT_EXPANDABLE = 0


def variable_expandable(variable, old_size):
    if variable.size() == old_size:
        return _SAME_SIZE
    try:
        torch._C._infer_size(variable.size(), old_size)
    except RuntimeError:
        return _NOT_EXPANDABLE
    return _EXPANDABLE


def maybe_unexpand_or_view(variable, old_size):
    var_expanded = variable_expandable(variable, old_size)

    if var_expanded == _SAME_SIZE:
        return variable
    elif var_expanded == _EXPANDABLE:
        return maybe_unexpand(variable, old_size, False)
    else:
        return maybe_view(variable, old_size, False)
