import inspect
import torch
import torch.nn as nn
from torch._six import inf
from typing import Iterable, Union

_tensor_or_tensors = Union[torch.Tensor, Iterable[torch.Tensor]]


def _patch(fn, newfn):
    xfingerprint = inspect.signature(fn)
    fingerprint = inspect.signature(newfn)
    if xfingerprint != fingerprint:
        raise RuntimeError(
            'Unable to patch {}, signature mismatch: {} vs {}'.format(
                fn, xfingerprint, fingerprint))
    newfn._orig = fn
    return newfn


def clip_grad_norm_(parameters: _tensor_or_tensors,
                    max_norm: float,
                    norm_type: float = 2.0,
                    error_if_nonfinite: bool = False) -> torch.Tensor:
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    max_norm = float(max_norm)
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.)
    device = parameters[0].grad.device
    if norm_type == inf:
        total_norm = max(p.grad.detach().abs().max() for p in parameters)
    else:
        total_norm = torch.norm(
            torch.stack(
                [torch.norm(p.grad.detach(), norm_type) for p in parameters]),
            norm_type)
    if error_if_nonfinite and (total_norm.isnan() or total_norm.isinf()):
        raise RuntimeError(
            f'The norm of order {norm_type} for a gradient from `parameters` '
            'is non-finite, so it cannot be clipped. This error can be '
            'disabled with `error_if_nonfinite=False`')
    clip_coef = torch.tensor(max_norm, device=device) / (total_norm + 1e-6)
    clip_value = torch.where(clip_coef < 1, clip_coef,
                             torch.tensor(1., device=device))
    for p in parameters:
        p.grad.detach().mul_(clip_value)
    return total_norm


def _apply_patches():
    nn.utils.clip_grad_norm_ = _patch(nn.utils.clip_grad_norm_, clip_grad_norm_)
