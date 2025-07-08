import operator
from collections.abc import Sequence
from functools import reduce

import torch


def maybe_view(
    tensor: torch.Tensor, size: torch.Size, check_same_size: bool = True
) -> torch.Tensor:
    if check_same_size and tensor.size() == size:
        return tensor
    return tensor.contiguous().view(size)


def maybe_unexpand(
    tensor: torch.Tensor, old_size: torch.Size, check_same_size: bool = True
) -> torch.Tensor:
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


# Check whether the op enable broadcasting, and whether it is supported by ONNX.
# If dims1 and dims2 are different, then broadcast is True.
# We always assume the combination of dims1 and dims2 is broadcastable.
# The following types of broadcasting are supported in ONNX:
#     1) Only one element in dims2, such as dims2 = [1, 1]
#     2) dims2 is suffix of dims1, such as dims1 = [2, 3, 4], and dims2 = [3, 4]
# Details can be found here: https://github.com/onnx/onnx/blob/master/docs/Operators.md#Gemm
def check_onnx_broadcast(dims1: Sequence[int], dims2: Sequence[int]) -> bool:
    broadcast = False
    supported = True
    len1 = len(dims1)
    len2 = len(dims2)

    numel2 = reduce(operator.mul, dims2)
    if len1 < len2:
        broadcast = True
        if numel2 != 1:
            supported = False
    elif len1 > len2:
        broadcast = True
        if numel2 != 1 and dims1[len1 - len2 :] != dims2:
            supported = False
    else:
        if dims1 != dims2:
            broadcast = True
            if numel2 != 1:
                supported = False

    if not supported:
        raise ValueError(
            f"Numpy style broadcasting is not supported in ONNX. Input dims are: {dims1}, {dims2}"
        )
    return broadcast
