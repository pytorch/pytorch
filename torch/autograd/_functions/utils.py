import torch


def maybe_view(variable, size):
    if variable.size() == size:
        return variable
    return variable.contiguous().view(size)


def maybe_unexpand(variable, old_size):
    num_unsqueezed = variable.dim() - len(old_size)
    expanded_dims = [dim for dim, (expanded, original)
                     in enumerate(zip(variable.size()[num_unsqueezed:], old_size))
                     if expanded != original]

    for _ in range(num_unsqueezed):
        variable = variable.sum(0, keepdim=False)
    for dim in expanded_dims:
        variable = variable.sum(dim, keepdim=True)
    return variable


def variable_expandable(variable, old_size):
    try:
        torch._C._infer_size(variable.size(), old_size)
    except RuntimeError:
        return False
    return True


def maybe_unexpand_or_view(variable, old_size):
    var_expanded = True
    if maybe_view:
        var_expanded = variable_expandable(variable, old_size)

    if var_expanded:
        return maybe_unexpand(variable, old_size)
    else:
        return maybe_view(variable, old_size)
