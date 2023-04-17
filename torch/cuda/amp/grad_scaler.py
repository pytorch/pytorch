from collections import abc
from typing import List
import warnings

import torch
from .common import amp_definitely_not_available


__all__ = ["GradScaler"]

class GradScaler(torch.amp.GradScaler):
    r"""
    See :class:`torch.amp.GradScaler`.
    ``torch.cuda.amp.GradScaler(args...)`` is equivalent to ``torch.amp.GradScaler("cuda", args...)``
    """
    def __init__(self, init_scale=2.**16, growth_factor=2.0, backoff_factor=0.5,
                 growth_interval=2000, enabled=True):
        if enabled and amp_definitely_not_available():
            enabled = False
            warnings.warn("torch.cuda.amp.GradScaler is enabled, but CUDA is not available. Disabling.")
        super().__init__("cuda", init_scale=init_scale, growth_factor=growth_factor, backoff_factor=backoff_factor,
                         growth_interval=growth_interval, enabled=enabled)

    def scale(self, outputs):
        """
        Multiplies ('scales') a tensor or list of tensors by the scale factor.
        Returns scaled outputs.  If this instance of :class:`GradScaler` is not enabled, outputs are returned
        unmodified.
        Args:
            outputs (Tensor or iterable of Tensors):  Outputs to scale.
        """
        if not self._enabled:
            return outputs

        # Short-circuit for the common case.
        if isinstance(outputs, torch.Tensor):
            assert outputs.is_cuda or outputs.device.type == 'xla'
            if self._scale is None:
                self._lazy_init_scale_growth_tracker(outputs.device)
            assert self._scale is not None
            return outputs * self._scale.to(device=outputs.device, non_blocking=True)

        # Invoke the more complex machinery only if we're treating multiple outputs.
        # holds a reference that can be overwritten by apply_scale
        stash: List[torch.amp.grad_scaler._MultiDeviceReplicator] = []

        def apply_scale(val):
            if isinstance(val, torch.Tensor):
                assert val.is_cuda or val.device.type == 'xla'
                if len(stash) == 0:
                    if self._scale is None:
                        self._lazy_init_scale_growth_tracker(val.device)
                    assert self._scale is not None
                    stash.append(torch.amp.grad_scaler._MultiDeviceReplicator(self._scale))
                return val * stash[0].get(val.device)
            elif isinstance(val, abc.Iterable):
                iterable = map(apply_scale, val)
                if isinstance(val, (list, tuple)):
                    return type(val)(iterable)
                else:
                    return iterable
            else:
                raise ValueError("outputs must be a Tensor or an iterable of Tensors")

        return apply_scale(outputs)
