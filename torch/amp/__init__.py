"""
``torch.amp`` provides convenience methods for mixed precision training

required when training with mixed precision (IEEE FP16,
which is the type ``.half()`` creates, along with `


The interface for gradient scaling defined here is purely functional (internally stateless).
Persistent values are stored explicitly in user scripts
"""
import torch
import types

from .grad_scaling import scale_outputs, \
                          add_amp_attributes,\
                          set_scale_growth_factor, \
                          get_scale_growth_factor, \
                          set_scale_backoff_factor, \
                          get_scale_backoff_factor

__all__ = ["scale_outputs",
           "add_amp_attributes",
           "get_scale_growth_rate",
           "set_scale_growth_rate"]
