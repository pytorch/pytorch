import torch
from . import _C

from ._src.vmap import vmap
from ._src.eager_transforms import grad, grad_with_value, vjp, jacrev
from ._src.make_functional import make_functional, make_functional_with_buffers


# Monkeypatching lol
_old_cross_entropy = torch.nn.functional.cross_entropy


def cross_entropy(input, target, weight=None, size_average=None,
                  ignore_index=-100, reduce=None, reduction='mean'):
    if input.dim() == 1 and target.dim() == 0:
        input = input.unsqueeze(0)
        target = target.unsqueeze(0)

    result = _old_cross_entropy(
            input, target, weight, size_average,
            ignore_index, reduce, reduction)
    if reduction == 'none':
        return result.squeeze(0)
    return result


torch.nn.functional.cross_entropy = cross_entropy
