import warnings
from typing import Union, Iterable, List, Dict, Tuple, Optional

import torch
from torch import Tensor, inf
from torch.utils._foreach_utils import _group_tensors_by_device_and_dtype, _has_foreach_support

_tensor_or_tensors = Union[torch.Tensor, Iterable[torch.Tensor]]

__all__ = ['clip_grad_norm_', 'clip_grad_norm', 'clip_grad_value_']

def clip_grad_norm_(
        parameters: _tensor_or_tensors, max_norm: float, norm_type: float = 2.0,
        error_if_nonfinite: bool = False, foreach: Optional[bool] = None) -> torch.Tensor:
    r"""Clips gradient norm of an iterable of parameters.

    The norm is computed over all gradients together, as if they were
    concatenated into a single vector. Gradients are modified in-place.

    Args:
        parameters (Iterable[Tensor] or Tensor): an iterable of Tensors or a
            single Tensor that will have gradients normalized
        max_norm (float): max norm of the gradients
        norm_type (float): type of the used p-norm. Can be ``'inf'`` for
            infinity norm.
        error_if_nonfinite (bool): if True, an error is thrown if the total
            norm of the gradients from :attr:`parameters` is ``nan``,
            ``inf``, or ``-inf``. Default: False (will switch to True in the future)
        foreach (bool): use the faster foreach-based implementation.
            If ``None``, use the foreach implementation for CUDA and CPU native tensors and silently
            fall back to the slow implementation for other device types.
            Default: ``None``

    Returns:
        Total norm of the parameter gradients (viewed as a single vector).
    """
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    grads = [p.grad for p in parameters if p.grad is not None]
    max_norm = float(max_norm)
    norm_type = float(norm_type)
    if len(grads) == 0:
        return torch.tensor(0.)
    first_device = grads[0].device
    grouped_grads: Dict[Tuple[torch.device, torch.dtype], List[List[Tensor]]] \
        = _group_tensors_by_device_and_dtype([[g.detach() for g in grads]])  # type: ignore[assignment]

    if norm_type == inf:
        norms = [torch.linalg.vector_norm(g.detach(), inf).to(first_device) for g in grads]
        total_norm = norms[0] if len(norms) == 1 else torch.max(torch.stack(norms))
    else:
        norms = []
        for ((device, _), [grads]) in grouped_grads.items():
            if (foreach is None or foreach) and _has_foreach_support(grads, device=device):
                norms.extend(torch._foreach_norm(grads, norm_type))
            elif foreach:
                raise RuntimeError(f'foreach=True was passed, but can\'t use the foreach API on {device.type} tensors')
            else:
                norms.extend([torch.linalg.vector_norm(g, norm_type) for g in grads])

        total_norm = torch.linalg.vector_norm(torch.stack([norm.to(first_device) for norm in norms]), norm_type)

    if error_if_nonfinite and torch.logical_or(total_norm.isnan(), total_norm.isinf()):
        raise RuntimeError(
            f'The total norm of order {norm_type} for gradients from '
            '`parameters` is non-finite, so it cannot be clipped. To disable '
            'this error and scale the gradients by the non-finite norm anyway, '
            'set `error_if_nonfinite=False`')
    clip_coef = max_norm / (total_norm + 1e-6)
    # Note: multiplying by the clamped coef is redundant when the coef is clamped to 1, but doing so
    # avoids a `if clip_coef < 1:` conditional which can require a CPU <=> device synchronization
    # when the gradients do not reside in CPU memory.
    clip_coef_clamped = torch.clamp(clip_coef, max=1.0)
    for ((device, _), [grads]) in grouped_grads.items():
        if (foreach is None or foreach) and _has_foreach_support(grads, device=device):
            torch._foreach_mul_(grads, clip_coef_clamped.to(device))  # type: ignore[call-overload]
        elif foreach:
            raise RuntimeError(f'foreach=True was passed, but can\'t use the foreach API on {device.type} tensors')
        else:
            clip_coef_clamped_device = clip_coef_clamped.to(device)
            for g in grads:
                g.detach().mul_(clip_coef_clamped_device)

    return total_norm


def clip_grad_norm(
        parameters: _tensor_or_tensors, max_norm: float, norm_type: float = 2.,
        error_if_nonfinite: bool = False, foreach: Optional[bool] = None) -> torch.Tensor:
    r"""Clips gradient norm of an iterable of parameters.

    .. warning::
        This method is now deprecated in favor of
        :func:`torch.nn.utils.clip_grad_norm_`.
    """
    warnings.warn("torch.nn.utils.clip_grad_norm is now deprecated in favor "
                  "of torch.nn.utils.clip_grad_norm_.", stacklevel=2)
    return clip_grad_norm_(parameters, max_norm, norm_type, error_if_nonfinite, foreach)


def clip_grad_value_(parameters: _tensor_or_tensors, clip_value: float, foreach: Optional[bool] = None) -> None:
    r"""Clips gradient of an iterable of parameters at specified value.

    Gradients are modified in-place.

    Args:
        parameters (Iterable[Tensor] or Tensor): an iterable of Tensors or a
            single Tensor that will have gradients normalized
        clip_value (float): maximum allowed value of the gradients.
            The gradients are clipped in the range
            :math:`\left[\text{-clip\_value}, \text{clip\_value}\right]`
        foreach (bool): use the faster foreach-based implementation
            If ``None``, use the foreach implementation for CUDA and CPU native tensors and
            silently fall back to the slow implementation for other device types.
            Default: ``None``
    """
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    clip_value = float(clip_value)

    grads = [p.grad for p in parameters if p.grad is not None]
    grouped_grads: Dict[Tuple[torch.device, torch.dtype], List[List[Tensor]]] \
        = _group_tensors_by_device_and_dtype([grads])  # type: ignore[assignment]

    for ((device, _), [grads]) in grouped_grads.items():
        if (foreach is None or foreach) and _has_foreach_support(grads, device=device):
            torch._foreach_clamp_min_(grads, -clip_value)
            torch._foreach_clamp_max_(grads, clip_value)
        elif foreach:
            raise RuntimeError(f'foreach=True was passed, but can\'t use the foreach API on {device.type} tensors')
        else:
            for grad in grads:
                grad.data.clamp_(min=-clip_value, max=clip_value)
