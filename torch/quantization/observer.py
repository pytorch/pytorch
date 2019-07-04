from __future__ import absolute_import, division, print_function, unicode_literals
import torch.nn as nn
import torch
from functools import partial
import numpy as np

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
    def __init__(self, dtype=torch.quint8, qscheme=torch.per_tensor_affine, avg_constant=0.9):
        super(Observer, self).__init__()
        self.dtype = dtype
        self.qscheme = qscheme
        assert self.qscheme == torch.per_tensor_affine, \
            'Default Observer only works for per_tensor_affine quantization scheme'
        # Symmetric range for initialization
        # min array and max array
        self.min_history = []
        self.max_history = []
        self.avg_constant = avg_constant

    def forward(self, x):
        self.min_history.append(torch.min(x).float())
        self.max_history.append(torch.max(x).float())

    def calculate_qparams(self):
        n_levels = 255.0
        min_val = np.percentile(self.min_history, 5)
        max_val = np.percentile(self.max_history, 95)
        scale = (max_val - min_val) / n_levels
        if self.dtype == torch.qint8:
            qmin, qmax = -128, 127
        else:
            qmin, qmax = 0, 255
        zero_point = qmin - min_val / scale
        zero_point = max(qmin, zero_point)
        zero_point = min(qmax, zero_point)
        print(zero_point, qmin, qmax)

        return torch.tensor([scale, zero_point])

def observer(observer_cls, **kwargs):
    return partial(observer_cls, **kwargs)

def default_observer(**kwargs):
    return observer(Observer, **kwargs)

def default_weight_observer(**kwargs):
    kwargs.setdefault('dtype', torch.qint8)
    return observer(Observer, **kwargs)
