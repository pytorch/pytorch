import torch
from torch.distributions import constraints
from torch.distributions.utils import (_sum_rightmost)
__all__ = [

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

def sigmoid_transformation_call(x):
    return torch.sigmoid(x)

def sigmoid_transformation_inverse(y):
    return y.log() - (-y).log1p()

def sigmoid_transformation_log_abs_det_jacobian(x, y):
    return -(y.reciprocal() + (1 - y).reciprocal()).log()
