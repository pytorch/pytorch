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


# Turn the parameter pad in pytorch into paddings in onnx order.
def prepare_paddings(input, pad):
    dim = len(input.type().sizes())
    # The order of paddings is dim_0_begin, dim_0_end, dim_1_begin, ... , dim_n_end.
    # n is the dimension of input.
    assert len(pad) <= dim * 2
    paddings = []
    # pad is guaranteed to have even elements.
    for i, j in zip(pad[0::2], pad[1::2]):
        paddings = [i, j] + paddings
    while len(paddings) < 2 * dim:
        paddings = [0, 0] + paddings
    assert len(paddings) == dim * 2
    return paddings
