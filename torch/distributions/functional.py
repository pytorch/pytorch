import math
import numbers
import torch

from torch.distributions import constraints
from torch.distributions.utils import _sum_rightmost
from torch.nn.functional import pad

__all__ = [
    'compose_transform_domain',
    'compose_transform_codomain',
    'compose_transform_bijective',
    'compose_transform_sign',
    'compose_transform_event_dim',
    'compose_transform_call_',
    'compose_transform_log_abs_det_jacobian',
    'compose_transform_repr_',
    'exp_transform_call',
    'exp_transform_inverse',
    'exp_transform_log_abs_det_jacobian',
    'power_transform_call',
    'power_transform_inverse',
    'power_transform_log_abs_det_jacobian',
    'sigmoid_transform_call',
    'sigmoid_transform_inverse',
    'sigmoid_transform_log_abs_det_jacobian',
    'abs_transform_call',
    'abs_transform_inverse',
    'affine_transform_sign',
    'affine_transform_call',
    'affine_transform_inverse',
    'affine_transform_log_abs_det_jacobian',
    'soft_max_transform_call',
    'soft_max_transform_inverse',
    'stick_breaking_transform_call',
    'stick_breaking_transform_inverse',
    'stick_breaking_transform_log_abs_det_jacobian',
    'lower_cholesky_transform_call',
    'lower_cholesky_transform_inverse',
    'cat_transform_call',
    'cat_transform_inverse',
    'cat_transform_log_abs_det_jacobian',
    'cat_transform_bijective',
    'cat_transform_domain',
    'cat_transform_codomain',
    'stack_transform_slice',
    'stack_transform_call',
    'stack_transform_inverse',
    'stack_transform_log_abs_det_jacobian',
    'stack_transform_bijective',
    'stack_transform_domain',
    'stack_transform_codomain',
]


def compose_transform_domain(parts):
    if not parts:
        return constraints.real
    return parts[0].domain


def compose_transform_codomain(parts):
    if not parts:
        return constraints.real
    return parts[-1].codomain


def compose_transform_bijective(parts):
    return all(p.bijective for p in parts)


def compose_transform_sign(parts):
    sign = 1
    for p in parts:
        sign = sign * p.sign
    return sign


def compose_transform_event_dim(parts):
    return max(p.event_dim for p in parts) if parts else 0


def compose_transform_call_(parts, x):
    for part in parts:
        x = part(x)
    return x


def compose_transform_log_abs_det_jacobian(parts, event_dim, x, y):
    if not parts:
        return torch.zeros_like(x)
    result = 0
    for part in parts:
        y = part(x)
        result = result + _sum_rightmost(part.log_abs_det_jacobian(x, y),
                                         event_dim - part.event_dim)
        x = y
    return result


def compose_transform_repr_(parts, class_name):
    fmt_string = class_name + '(\n    '
    fmt_string += ',\n    '.join([p.__repr__() for p in parts])
    fmt_string += '\n)'
    return fmt_string


def exp_transform_call(x):
    return x.exp()


def exp_transform_inverse(y):
    return y.log()


def exp_transform_log_abs_det_jacobian(x, y):
    return x


def power_transform_call(exponent, x):
    return x.pow(exponent)


def power_transform_inverse(exponent, y):
    return y.pow(1 / exponent)


def power_transform_log_abs_det_jacobian(exponent, x, y):
    return (exponent * y / x).abs().log()


def sigmoid_transform_call(x):
    return torch.sigmoid(x)


def sigmoid_transform_inverse(y):
    return y.log() - (-y).log1p()


def sigmoid_transform_log_abs_det_jacobian(x, y):
    return -(y.reciprocal() + (1 - y).reciprocal()).log()


def abs_transform_call(x):
    return x.abs()


def abs_transform_inverse(y):
    return y


def affine_transform_sign(scale):
    if isinstance(scale, numbers.Number):
        return 1 if scale > 0 else -1 if scale < 0 else 0
    return scale.sign()


def affine_transform_call(scale, loc, x):
    return loc + scale * x


def affine_transform_inverse(scale, loc, y):
    return (y - loc) / scale


def affine_transform_log_abs_det_jacobian(scale, event_dim, x, y):
    shape = x.shape
    scale = scale
    if isinstance(scale, numbers.Number):
        result = torch.full_like(x, math.log(abs(scale)))
    else:
        result = torch.abs(scale).log()
    if event_dim:
        result_size = result.size()[:-event_dim] + (-1,)
        result = result.view(result_size).sum(-1)
        shape = shape[:-event_dim]
    return result.expand(shape)


def soft_max_transform_call(x):
    logprobs = x
    probs = (logprobs - logprobs.max(-1, True)[0]).exp()
    return probs / probs.sum(-1, True)


def soft_max_transform_inverse(y):
    probs = y
    return probs.log()


def stick_breaking_transform_call(x):
    offset = (x.shape[-1] + 1) - x.new([1]).expand(x.shape).cumsum(-1)
    z = torch.sigmoid(x - offset.log())
    z_cumprod = (1 - z).cumprod(-1)
    y = pad(z, (0, 1), value=1) * pad(z_cumprod, (1, 0), value=1)
    return y


def stick_breaking_transform_inverse(y):
    shape = y.shape[:-1] + (y.shape[-1] - 1,)
    offset = (shape[-1] + 1) - y.new([1]).expand(shape).cumsum(-1)
    sf = (1 - y.cumsum(-1))[..., :-1]
    x = y[..., :-1].log() - sf.log() + offset.log()
    return x


def stick_breaking_transform_log_abs_det_jacobian(x, y):
    offset = (x.shape[-1] + 1) - x.new([1]).expand(x.shape).cumsum(-1)
    z = torch.sigmoid(x - offset.log())
    detJ = ((1 - z).log() + y[..., :-1].log()).sum(-1)
    return detJ


def lower_cholesky_transform_call(x):
    def _call_on_event(x):
        return x.tril(-1) + x.diag().exp().diag()
    flat_x = x.reshape((-1,) + x.shape[-2:])
    return torch.stack([_call_on_event(flat_x[i]) for i in range(flat_x.size(0))]).view(x.shape)


def lower_cholesky_transform_inverse(y):
    def _inverse_on_event(y):
        return y.tril(-1) + y.diag().log().diag()
    flat_y = y.contiguous().view((-1,) + y.shape[-2:])
    return torch.stack([_inverse_on_event(flat_y[i]) for i in range(flat_y.size(0))]).view(y.shape)


def cat_transform_call(dim, length_to_check, lengths, transforms, x):
    assert -x.dim() <= dim < x.dim()
    assert x.size(dim) == length_to_check
    yslices = []
    start = 0
    for trans, length in zip(transforms, lengths):
        xslice = x.narrow(dim, start, length)
        yslices.append(trans(xslice))
        start = start + length  # avoid += for jit compat
    return torch.cat(yslices, dim=dim)


def cat_transform_inverse(dim, length_to_check, lengths, transforms, y):
    assert -y.dim() <= dim < y.dim()
    assert y.size(dim) == length_to_check
    xslices = []
    start = 0
    for trans, length in zip(transforms, lengths):
        yslice = y.narrow(dim, start, length)
        xslices.append(trans.inv(yslice))
        start = start + length  # avoid += for jit compat
    return torch.cat(xslices, dim=dim)


def cat_transform_log_abs_det_jacobian(dim, length_to_check, lengths, transforms, x, y):
    assert -x.dim() <= dim < x.dim()
    assert x.size(dim) == length_to_check
    assert -y.dim() <= dim < y.dim()
    assert y.size(dim) == length_to_check
    logdetjacs = []
    start = 0
    for trans, length in zip(transforms, lengths):
        xslice = x.narrow(dim, start, length)
        yslice = y.narrow(dim, start, length)
        logdetjacs.append(trans.log_abs_det_jacobian(xslice, yslice))
        start = start + length  # avoid += for jit compat
    return torch.cat(logdetjacs, dim=dim)


def cat_transform_bijective(transforms):
    return all(t.bijective for t in transforms)


def cat_transform_domain(dim, lengths, transforms):
    return constraints.cat([t.domain for t in transforms],
                           dim, lengths)


def cat_transform_codomain(dim, lengths, transforms):
    return constraints.cat([t.codomain for t in transforms],
                           dim, lengths)


def stack_transform_slice(dim, z):
    return [z.select(dim, i) for i in range(z.size(dim))]


def stack_transform_call(dim, transforms, x):
    assert -x.dim() <= dim < x.dim()
    assert x.size(dim) == len(transforms)
    yslices = []
    for xslice, trans in zip(stack_transform_slice(dim, x), transforms):
        yslices.append(trans(xslice))
    return torch.stack(yslices, dim=dim)


def stack_transform_inverse(dim, transforms, y):
    assert -y.dim() <= dim < y.dim()
    assert y.size(dim) == len(transforms)
    xslices = []
    for yslice, trans in zip(stack_transform_slice(dim, y), transforms):
        xslices.append(trans.inv(yslice))
    return torch.stack(xslices, dim=dim)


def stack_transform_log_abs_det_jacobian(dim, transforms, x, y):
    assert -x.dim() <= dim < x.dim()
    assert x.size(dim) == len(transforms)
    assert -y.dim() <= dim < y.dim()
    assert y.size(dim) == len(transforms)
    logdetjacs = []
    yslices = stack_transform_slice(dim, y)
    xslices = stack_transform_slice(dim, x)
    for xslice, yslice, trans in zip(xslices, yslices, transforms):
        logdetjacs.append(trans.log_abs_det_jacobian(xslice, yslice))
    return torch.stack(logdetjacs, dim=dim)


def stack_transform_bijective(transforms):
    return all(t.bijective for t in transforms)


def stack_transform_domain(dim, transforms):
    return constraints.stack([t.domain for t in transforms], dim)


def stack_transform_codomain(dim, transforms):
    return constraints.stack([t.codomain for t in transforms], dim)
