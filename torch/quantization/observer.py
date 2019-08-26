from __future__ import absolute_import, division, print_function, unicode_literals
import torch.nn as nn
import torch
from functools import partial
import warnings

from torch._jit_internal import Optional

class Observer(nn.Module):
    r"""Default Observer Module
    A default implementation of the observer module, only works for
    `per_tensor_affine` quantization scheme.
    The module will record the running average of max and min value of the
    observed Tensor and calulate_qparams will calculate the scale and zero_point

    Other types of Observers should follow the same API, it can take arbitrary
    number of keyward arguments. In forward, it will update the statistics of
    the observed Tensor. And it should provide a `calculate_qparam` function
    that computes the quantization parameters given the collected statistics.
    TODO: Maybe add an abstract Observer class that enforces these rules?
    """
    __annotations__ = {'min_val' : Optional[torch.Tensor], 'max_val' : Optional[torch.Tensor]}

    def __init__(self, dtype=torch.quint8, qscheme=torch.per_tensor_affine):
        super(Observer, self).__init__()
        self.dtype = dtype
        self.qscheme = qscheme
        assert self.qscheme in (torch.per_tensor_affine, torch.per_tensor_symmetric), \
            'Default Observer only works for per_tensor_affine and \
                per_tensor_symmetric quantization scheme'
        assert self.dtype in (torch.qint8, torch.quint8), \
            'Default Observer only works for qint8 and quint data type'
        self.min_val = None
        self.max_val = None
        self.eps = torch.finfo(torch.float32).eps

    def forward(self, x):
        min_val = self.min_val
        max_val = self.max_val
        if min_val is None or max_val is None:
            min_val = torch.min(x)
            max_val = torch.max(x)
        else:
            min_val = torch.min(torch.min(x), min_val)
            max_val = torch.max(torch.max(x), max_val)
        self.min_val = min_val
        self.max_val = max_val
        return x

    @torch.jit.export
    def calculate_qparams(self):
        if self.dtype == torch.qint8:
            qmin, qmax = -128, 127
        else:
            qmin, qmax = 0, 255
        scale = 1.0
        zero_point = 0
        # We pull these out so that TorchScript optional type refinement works.
        # We may be able to remove this in the future if TorchScript supports that
        # feature on attributes
        min_val = self.min_val
        max_val = self.max_val
        if max_val is None or min_val is None:
            warnings.warn("must run observer before calling calculate_qparams")
        else:
            max_val, min_val = self.max_val.item(), self.min_val.item()
            min_val = min(0.0, self.min_val.item())
            max_val = max(0.0, self.max_val.item())
            if self.qscheme == torch.per_tensor_symmetric:
                max_val = max(-min_val, max_val)
                scale = max_val / ((qmax - qmin) / 2)
                scale = max(scale, self.eps)
                zero_point = 0 if self.dtype == torch.qint8 else 128
            else:
                scale = (max_val - min_val) / float(qmax - qmin)
                scale = max(scale, self.eps)
                zero_point = qmin - round(min_val / scale)
                zero_point = max(qmin, zero_point)
                zero_point = min(qmax, zero_point)
                zero_point = int(zero_point)

        return torch.tensor([scale]), torch.tensor([zero_point])

    def extra_repr(self):
        return 'min_val={}, max_val={}'.format(self.min_val, self.max_val)

def observer(observer_cls, **kwargs):
    return partial(observer_cls, **kwargs)

def default_observer(**kwargs):
    return observer(Observer, **kwargs)

def default_weight_observer(**kwargs):
    kwargs.setdefault('dtype', torch.qint8)
    kwargs.setdefault('qscheme', torch.per_tensor_symmetric)
    return observer(Observer, **kwargs)
