from numbers import Integral

from .module import Module
from .. import functional as F
from .utils import _pair


class _UpsamplingBase(Module):

    def __init__(self, size=None, scale_factor=None):
        super(_UpsamplingBase, self).__init__()
        if size is None and scale_factor is None:
            raise ValueError('either size or scale_factor should be defined')
        if scale_factor is not None and not isinstance(scale_factor, Integral):
            raise ValueError('scale_factor must be of integer type')
        self.size = _pair(size)
        self.scale_factor = scale_factor


class UpsamplingNearest2d(_UpsamplingBase):

    def forward(self, input):
        return F.upsample_nearest(input, self.size, self.scale_factor)


class UpsamplingBillinear2d(_UpsamplingBase):

    def forward(self, input):
        return F.upsample_billinear(input, self.size, self.scale_factor)
