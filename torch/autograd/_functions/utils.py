from functools import reduce


def maybe_view(tensor, size, check_same_size=True):
    if check_same_size and tensor.size() == size:
        return tensor
    return tensor.contiguous().view(size)


def maybe_unexpand(tensor, old_size, check_same_size=True):
    if check_same_size and tensor.size() == old_size:
        return tensor
    num_unsqueezed = tensor.dim() - len(old_size)
    expanded_dims = [dim for dim, (expanded, original)
                     in enumerate(zip(tensor.size()[num_unsqueezed:], old_size))
                     if expanded != original]

    for _ in range(num_unsqueezed):
        tensor = tensor.sum(0, keepdim=False)
    for dim in expanded_dims:
        tensor = tensor.sum(dim, keepdim=True)
    return tensor


# Generate paddings in ONNX order based on pad in pytorch.
# Arguments:
#     dim: the dimension of the tensor.
#     pad: the paddings in pytorch.
#          The order is dim_n_begin, dim_n_end, dim_n-1_begin, dim_n-1_end, ...
def prepare_onnx_paddings(dim, pad):
    assert isinstance(dim, int)
    # The desired order of paddings is
    # dim_0_begin, dim_1_begin, ... , dim_0_end, ..., dim_n_end.
    # n is the dimension of input.
    assert len(pad) <= dim * 2
    # assume zero-dimensions in the beginning
    paddings = list(pad[:]) + [0] * (dim * 2 - len(pad))
    # reverse order and collate first beginnings and then ends
    paddings = paddings[-2::-2] + paddings[-1::-2]
    assert len(paddings) == dim * 2
    return paddings


# Check whether the op enable broadcasting, and whether it is supported by ONNX.
# If dims1 and dims2 are different, then broadcast is True.
# We always assume the combination of dims1 and dims2 is broadcastable.
# The following types of broadcasting are supported in ONNX:
#     1) Only one element in dims2, such as dims2 = [1, 1]
#     2) dims2 is suffix of dims1, such as dims1 = [2, 3, 4], and dims2 = [3, 4]
# Details can be found here: https://github.com/onnx/onnx/blob/master/docs/Operators.md#Gemm
def check_onnx_broadcast(dims1, dims2):
    broadcast = False
    supported = True
    len1 = len(dims1)
    len2 = len(dims2)
    numel1 = reduce(lambda x, y: x * y, dims1)
    numel2 = reduce(lambda x, y: x * y, dims2)
    if len1 < len2:
        broadcast = True
        if numel2 != 1:
            supported = False
    elif len1 > len2:
        broadcast = True
        if numel2 != 1 and dims1[len1 - len2:] != dims2:
            supported = False
    else:
        if dims1 != dims2:
            broadcast = True
            if numel2 != 1:
                supported = False

    if not supported:
        raise ValueError("Numpy style broadcasting is not supported in ONNX. "
                         "Input dims are: {}, {}".format(dims1, dims2))
    return broadcast
