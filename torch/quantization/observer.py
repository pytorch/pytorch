from __future__ import absolute_import, division, print_function, unicode_literals

import warnings
from abc import ABCMeta, abstractmethod
from functools import partial

import torch
import torch.nn as nn
from torch._jit_internal import Optional


ABC = ABCMeta(str("ABC"), (object,), {})  # compatible with Python 2 *and* 3:


class ObserverBase(ABC, nn.Module):
    r"""Observer base Module
    Any concrete observer implementation should derive from this class.

    Concrete observers should follow the same API. In forward, they will update
    the statistics of the observed Tensor. And they should provide a
    `calculate_qparams` function that computes the quantization parameters given
    the collected statistics.
    """

    def __init__(self, dtype=torch.quint8, qscheme=torch.per_tensor_affine):
        super(ObserverBase, self).__init__()
        self.dtype = dtype
        self.qscheme = qscheme
        self.eps = torch.finfo(torch.float32).eps
        assert self.qscheme in (
            torch.per_tensor_affine,
            torch.per_tensor_symmetric,
        ), "Default Observer only works for per_tensor_affine and \
                per_tensor_symmetric quantization scheme"
        assert self.dtype in (
            torch.qint8,
            torch.quint8,
        ), "Default Observer only works for qint8 and quint8 data type"

    @abstractmethod
    def forward(self, x):
        pass

    @abstractmethod
    def calculate_qparams(self, **kwargs):
        pass

    def _calculate_qparams(self, min_val, max_val):
        # type: (Optional[Tensor], Optional[Tensor]) -> Tuple[Tensor, Tensor]
        """
        Given min and max values, this function calculates quantization parameters
        """

        if max_val is None or min_val is None:
            warnings.warn(
                "must run observer before calling calculate_qparams.\
                                    Returning default scale and zero point "
            )
            return torch.tensor([1.0]), torch.tensor([0])

        assert min_val <= max_val, "min {} should be less than max {}".format(
            min_val, max_val
        )

        if self.dtype == torch.qint8:
            qmin, qmax = -128, 127
        else:
            qmin, qmax = 0, 255
        max_val, min_val = float(max_val), float(min_val)
        min_val = min(0.0, min_val)
        max_val = max(0.0, max_val)
        if max_val == min_val:
            scale = 1.0
            zero_point = 0
        else:
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


class MinMaxObserver(ObserverBase):
    r"""Default Observer Module
    A default implementation of the observer module, only works for
    `per_tensor_affine` quantization scheme.  The module will record the
    running average of max and min value of the observed Tensor and
    calculate_qparams will calculate scale and zero_point
    """

    __annotations__ = {
        "min_val": Optional[torch.Tensor],
        "max_val": Optional[torch.Tensor],
    }

    def __init__(self, **kwargs):
        super(MinMaxObserver, self).__init__(**kwargs)
        self.min_val = None
        self.max_val = None

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
        return self._calculate_qparams(self.min_val, self.max_val)

    @torch.jit.export
    def extra_repr(self):
        return "min_val={}, max_val={}".format(self.min_val, self.max_val)


class HistogramObserver(ObserverBase):
    r"""
    The module records the running histogram of tensor values along with
    min/max values. calculate_qparams will calculate scale and zero_point
    """

    __annotations__ = {
        "min_val": Optional[torch.Tensor],
        "max_val": Optional[torch.Tensor],
        "relaxed_min": torch.Tensor,
        "relaxed_max": torch.Tensor,
        "histogram": Optional[torch.Tensor],
    }

    def __init__(self, bins=2048, **kwargs):
        super(HistogramObserver, self).__init__(**kwargs)
        self.bins = bins
        self.histogram = None
        self.min_val = None
        self.max_val = None
        self.relaxed_min = torch.tensor([0])
        self.relaxed_max = torch.tensor([0])

    def forward(self, x):
        min_val = self.min_val
        max_val = self.max_val
        histogram = self.histogram
        if min_val is None or max_val is None or histogram is None:
            min_val = torch.min(x)
            max_val = torch.max(x)
            range = max_val - min_val
            self.relaxed_min = min_val - 0.5 * range
            self.relaxed_max = max_val + 0.5 * range
            self.histogram = torch.histc(x, self.bins, min=self.relaxed_min, max=self.relaxed_max)
            self.min_val = self.relaxed_min
            self.max_val = self.relaxed_max
        else:
            new_min = torch.min(x)
            new_max = torch.max(x)
            new_histogram = torch.histc(x, self.bins, min=self.relaxed_min, max=self.relaxed_max)
            self.histogram = new_histogram + histogram

    @torch.jit.export
    def calculate_qparams(self):
        min_val = self.min_val
        max_val = self.max_val
        histogram = self.histogram

        if min_val is None or max_val is None or histogram is None:
            return self._calculate_qparams(None, None)
        else:
            histogram_mask = torch.gt(histogram, 0).to(torch.int8)
            c = torch.cumsum(histogram_mask, 0)
            # Last non-zero bin
            max_bin = torch.argmax(histogram_mask)
            # Only one entry is non-zero, find it.
            min_bin = torch.argmax(torch.eq(c, 1))
            bin_width = (max_val - min_val) / histogram.size()[0]
            new_min = min_val + min_bin * bin_width
            new_max = min_val + (max_bin + 1) * bin_width
            return self._calculate_qparams(new_min, new_max)


def observer(observer_cls, **kwargs):
    return partial(observer_cls, **kwargs)


def default_observer(**kwargs):
    return observer(MinMaxObserver, **kwargs)


def default_weight_observer(**kwargs):
    kwargs.setdefault("dtype", torch.qint8)
    kwargs.setdefault("qscheme", torch.per_tensor_symmetric)
    return observer(MinMaxObserver, **kwargs)
