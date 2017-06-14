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
        variable = variable.sum(0)
    for dim in expanded_dims:
        variable = variable.sum(dim, True)
    return variable


def maybe_unexpand_or_view(variable, old_size):
    var_expanded = True
    if maybe_view:
        try:
            torch._C._infer_size(variable.size(), old_size)
        except RuntimeError:
            var_expanded = False

    if var_expanded:
        return maybe_unexpand(variable, old_size)
    else:
        return maybe_view(variable, old_size)
