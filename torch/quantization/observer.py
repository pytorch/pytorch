from __future__ import absolute_import, division, print_function, unicode_literals
import torch.nn as nn
import torch
from functools import partial

class Observer(nn.Module):
    r"""Default Observer Module
    A default implementation of the observer module, only works for
    per_tensor_affine quantization scheme.
    The module will record the running average of max and min value of the
    observed Tensor and calulate_qparams will calculate the scale and zero_point

    Other types of Observers should follow the same API, the first argument of
    the constructor should be qoptions, and it can take arbitrary number of
    arguments after that. and it should provide a calculate_qparam function.
    TODO: Maybe add an abstract Observer class?
    """
    def __init__(self, **kwargs):
        super(Observer, self).__init__()
        self.dtype = kwargs.get('dtype', torch.qint8)
        self.qscheme = kwargs.get('qscheme', torch.per_tensor_affine)
        assert self.qscheme == torch.per_tensor_affine, \
            'Default Observer only works for per_tensor_affine quantization scheme'
        # Symmetric range for initialization
        self.stats = torch.tensor([-6, 6], dtype=torch.float)
        self.avg_constant = kwargs.get('avg_constant', 0.9)

    def forward(self, x):
        self.stats = (1 - self.avg_constant) * self.stats + \
            self.avg_constant * torch.tensor([torch.min(x), torch.max(x)], dtype=torch.float)

    def calculate_qparams(self):
        qparams = torch.zeros(2).float()
        nLevels = 255.0
        if self.dtype == torch.qint8:
            qparams[0] = 2 * torch.max(self.stats[1], -self.stats[0]) / nLevels
            qparams[1] = 0
        else:
            qparams = torch.zeros(2).float()
            nLevels = 255.0
            qparams[0] = 2 * torch.max(self.stats[1], -self.stats[0]) / nLevels
            qparams[1] = 128

        return qparams

class WeightObserver(Observer):
    r""" Forward hook function inserted for wrap and quant modules.
     Observer updates the stats parameter of wrap/quant modules with statistics
     for use later.
    """

    def __init__(self, **kwargs):
        super(WeightObserver, self).__init__(**kwargs)
        self.stats = torch.tensor([-6, 6], dtype=torch.float)

    def forward(self, x):
        self.stats = torch.tensor([torch.min(x), torch.max(x)], dtype=torch.float)
        return x

def observer(observer_cls, kwargs):
    return partial(observer_cls, **kwargs)

def default_observer(kwargs):
    return partial(Observer, **kwargs)

def default_weight_observer(kwargs):
    return partial(WeightObserver, **kwargs)
