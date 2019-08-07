from __future__ import absolute_import, division, print_function, unicode_literals

from abc import ABC, abstractmethod
from functools import partial

import torch
import torch.nn as nn


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
        assert self.qscheme in (
            torch.per_tensor_affine,
            torch.per_tensor_symmetric,
        ), "Default Observer only works for per_tensor_affine and \
                per_tensor_symmetric quantization scheme"
        assert self.dtype in (
            torch.qint8,
            torch.quint8,
        ), "Default Observer only works for qint8 and quint data type"

    @abstractmethod
    def forward(self, x):
        pass

    @abstractmethod
    def calculate_qparams(self, **kwargs):
        pass

    def _calculate_qparams(self, min_val, max_val):
        """
        Given min and max values, this function calculates quantization parameters
        """
        assert min_val <= max_val, "min {} should be less than max {}".format(
            min_val, max_val
        )

        if self.dtype == torch.qint8:
            qmin, qmax = -128, 127
        else:
            qmin, qmax = 0, 255
        n_levels = qmax - qmin

        # extend min/max values to include 0 to meet the requirement that 0 is
        # exactly repsentable
        min_val = min(min_val, 0.0)
        max_val = max(max_val, 0.0)
        if max_val == min_val:
            scale = 1.0
            zero_point = 0
        else:
            if self.qscheme == torch.per_tensor_symmetric:
                max_val = max(-min_val, max_val)
                scale = max_val / 127.0
                scale = max(scale, torch.finfo(torch.float32).eps)
                zero_point = 0 if self.dtype == torch.qint8 else 128
            else:
                scale = (max_val - min_val) / n_levels
                scale = max(scale, torch.finfo(torch.float32).eps)
                zero_point = qmin - round(min_val / scale)
                zero_point = max(qmin, zero_point)
                zero_point = min(qmax, zero_point)

        return torch.tensor([scale, zero_point])


class MinMaxObserver(ObserverBase):
    r"""Default Observer Module
    A default implementation of the observer module, only works for
    `per_tensor_affine` quantization scheme.  The module will record the
    running average of max and min value of the observed Tensor and
    calculate_qparams will calculate scale and zero_point
    """

    def __init__(self, **kwargs):
        super(MinMaxObserver, self).__init__(**kwargs)
        self.min_val = None
        self.max_val = None

    def forward(self, x):
        if self.min_val is None or self.max_val is None:
            self.min_val = torch.min(x)
            self.max_val = torch.max(x)
        else:
            self.min_val = torch.min(torch.min(x), self.min_val)
            self.max_val = torch.max(torch.max(x), self.max_val)

    def calculate_qparams(self, **kwargs):
        if self.max_val is None or self.min_val is None:
            raise Exception("must run observer before calling calculate_qparams!")
        return self._calculate_qparams(self.min_val.item(), self.max_val.item())


def observer(observer_cls, **kwargs):
    return partial(observer_cls, **kwargs)


def default_observer(**kwargs):
    return observer(MinMaxObserver, **kwargs)


def default_weight_observer(**kwargs):
    kwargs.setdefault("dtype", torch.qint8)
    kwargs.setdefault("qscheme", torch.per_tensor_symmetric)
    return observer(MinMaxObserver, **kwargs)


class HistogramObserver(ObserverBase):
    r"""
    The module records the running histogram of tensor values along with
    min/max values. calculate_qparams will calculate scale and zero_point
    """

    def __init__(self, bins=2048, **kwargs):
        super(HistogramObserver, self).__init__(**kwargs)
        self.bins = bins
        self.histogram = None
        self.min_val = None
        self.max_val = None

    def _combine_histograms(
        self, dst_histogram, dst_min, dst_max, src_histogram, src_min, src_max
    ):
        bins_dst = dst_histogram.size()[0]
        bins_src = src_histogram.size()[0]

        dst_bin_width = (dst_max - dst_min) / bins_dst
        src_bin_width = (src_max - src_min) / bins_src

        for i in range(bins_src):
            src_bin_count = src_histogram[i].item()
            if src_bin_count == 0:
                continue

            src_bin_begin = src_min + src_bin_width * i
            src_bin_end = src_bin_begin + src_bin_width

            dst_bin = 0
            if dst_bin_width:
                dst_bin = int((src_bin_begin - dst_min) / dst_bin_width)

            dst_bin_begin = dst_min + dst_bin_width * dst_bin
            dst_bin_end = dst_bin_begin + dst_bin_width

            dst_bin2 = 0
            if dst_bin_width:
                dst_bin2 = min(
                    int((src_bin_end - dst_min) / dst_bin_width), bins_dst - 1
                )

            assert dst_bin2 <= dst_bin + 2, "1 src_bin is mapped to at most 2 dst_bins"
            # dst_bin_cnt is the count from src_bin that should go to dst_bin
            # the remainder should go to dst_bin2
            dst_bin_cnt = 0
            if src_bin_width == 0 or dst_bin_width == 0:
                dst_bin_cnt = src_bin_count
            else:
                # We divide counts in src_bin in proportion to range overlap with dst_bin
                dst_bin_cnt = min(
                    round(
                        (dst_bin_end - src_bin_begin) / src_bin_width * src_bin_count
                    ),
                    src_bin_count,
                )

            dst_histogram[dst_bin] += dst_bin_cnt

            # remaining should go to dst_bin2
            if dst_bin_cnt < src_bin_count:
                dst_histogram[dst_bin2] += src_bin_count - dst_bin_cnt

    def forward(self, x):
        if self.min_val is None or self.max_val is None or self.histogram is None:
            self.min_val = torch.min(x)
            self.max_val = torch.max(x)
            self.histogram = torch.histc(x, self.bins)
        else:
            new_min = torch.min(x)
            new_max = torch.max(x)
            new_histogram = torch.histc(x, self.bins)
            # combine the existing histogram and new histogram into 1 histogram
            combined_histogram = torch.zeros_like(self.histogram)
            combined_min = torch.min(new_min, self.min_val)
            combined_max = torch.max(new_max, self.max_val)
            self._combine_histograms(
                combined_histogram,
                combined_min.item(),
                combined_max.item(),
                self.histogram,
                self.min_val.item(),
                self.max_val.item(),
            )
            self._combine_histograms(
                combined_histogram,
                combined_min.item(),
                combined_max.item(),
                new_histogram,
                new_min.item(),
                new_max.item(),
            )
            self.histogram = combined_histogram
            self.min_val = combined_min
            self.max_val = combined_max

    def calculate_qparams(self, **kwargs):
        if self.max_val is None or self.min_val is None:
            raise Exception("must run observer before calling calculate_qparams!")
        min_bin = 0
        max_bin = self.bins - 1
        # find the first bin in histogram with non-zero Value from left
        for i in range(self.histogram.size()[0]):
            if (self.histogram[i].item() > 0):
                min_bin = i
                break
        # find the first bin in histogram with non-zero Value from right
        for i in reversed(range(self.histogram.size()[0])):
            if (self.histogram[i].item() > 0):
                max_bin = i
                break
        bin_width = (self.max_val.item() - self.min_val.item()) / self.histogram.size()[0]
        return self._calculate_qparams(self.min_val.item() + min_bin * bin_width, self.min_val.item() + (max_bin + 1) * bin_width)
