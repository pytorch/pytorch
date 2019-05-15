import torch
from torch.distributions import constraints
from torch.distributions.utils import (_sum_rightmost)
__all__ = [

]


def compose_transformation_domain(parts):
    if not parts:
        return constraints.real
    return parts[0].domain


def compose_transformation_codomain(parts):
    if not parts:
        return constraints.real
    return parts[-1].codomain


def compose_transformation_bijective(parts):
    return all(p.bijective for p in parts)


def compose_transformation_sign(parts):
    sign = 1
    for p in parts:
        sign = sign * p.sign
    return sign


def compose_transformation_event_dim(parts):
    return max(p.event_dim for p in parts) if parts else 0


def compose_transformation_call_(parts, x):
    for part in parts:
        x = part(x)
    return x


def compose_transformation_log_abs_det_jacobian(parts, event_dim, x, y):
    if not parts:
        return torch.zeros_like(x)
    result = 0
    for part in parts:
        y = part(x)
        result = result + _sum_rightmost(part.log_abs_det_jacobian(x, y),
                                         event_dim - part.event_dim)
        x = y
    return result


def compose_transformation_repr_(parts, class_name):
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
