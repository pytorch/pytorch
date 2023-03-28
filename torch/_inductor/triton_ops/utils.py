import torch


def _extract_strides(shape):
    rank = len(shape)
    ret = [1] * rank
    for i in range(rank - 1, 0, -1):
        ret[i - 1] = ret[i] * shape[i]
    return ret


def _roundup(x, div):
    return (x + div - 1) // div * div


# unpack the given idx given the order of axis of the desired 3-dim tensor
# You could view it as the reverse of flatten the idx of 3 axis in a tensor to 1-dim idx.
# order is the order of axes in tensor, innermost dimension outward
# shape is the 3D tensor's shape
def _unpack(idx, order, shape):
    if torch.is_tensor(idx):
        _12 = torch.div(idx, shape[order[0]], rounding_mode="trunc")
        _0 = idx % shape[order[0]]
        _2 = torch.div(_12, shape[order[1]], rounding_mode="trunc")
        _1 = _12 % shape[order[1]]
    else:
        _12 = idx // shape[order[0]]
        _0 = idx % shape[order[0]]
        _2 = _12 // shape[order[1]]
        _1 = _12 % shape[order[1]]
    return _0, _1, _2
