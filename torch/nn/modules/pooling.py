import torch
from torch.autograd import Variable

from .module import Module
from .utils import _pair, _triple


class MaxPool1d(Module):

    def __init__(self, kernel_size, stride=None, padding=0, dilation=1,
            return_indices=False, ceil_mode=False):
        super(MaxPool1d, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride or kernel_size
        self.padding = padding
        self.dilation = dilation
        self.return_indices = return_indices
        self.ceil_mode = ceil_mode

    def forward(self, input):
        return self._backend.MaxPool1d(self.kernel_size, self.stride,
                self.padding, self.dilation, self.ceil_mode,
                self.return_indices)(input)


class MaxPool2d(Module):

    def __init__(self, kernel_size, stride=None, padding=0, dilation=1,
            return_indices=False, ceil_mode=False):
        super(MaxPool2d, self).__init__()
        self.kh, self.kw = _pair(kernel_size)
        self.dh, self.dw = _pair(stride or kernel_size)
        self.padh, self.padw = _pair(padding)
        self.dilh, self.dilw = _pair(dilation)
        self.return_indices = return_indices
        self.ceil_mode = ceil_mode

    def forward(self, input):
        return self._backend.MaxPool2d(self.kw, self.kh, self.dw, self.dh,
                self.padw, self.padh, self.dilh, self.dilw, self.ceil_mode,
                self.return_indices)(input)


class MaxUnpool2d(Module):

    def __init__(self, kernel_size, stride=None, padding=0):
        super(MaxUnpool2d, self).__init__()
        self.kh, self.kw = _pair(kernel_size)
        self.dh, self.dw = _pair(stride or kernel_size)
        self.padh, self.padw = _pair(padding)

    def forward(self, input, indices):
        out_height = (input.size(2) - 1) * self.dh + self.kh - 2*self.padh
        out_width = (input.size(3) - 1) * self.dw + self.kw - 2*self.padw
        return self._backend.MaxUnpool2d(out_width,
                out_height)(input, indices)


class AvgPool2d(Module):

    def __init__(self, kernel_size, stride=None, padding=0, ceil_mode=False,
            count_include_pad=True):
        super(AvgPool2d, self).__init__()
        self.kh, self.kw = _pair(kernel_size)
        self.dh, self.dw = _pair(stride or kernel_size)
        self.padh, self.padw = _pair(padding)
        self.ceil_mode = ceil_mode
        self.count_include_pad = count_include_pad

    def forward(self, input):
        return self._backend.AvgPool2d(self.kw, self.kh, self.dw, self.dh,
                self.padw, self.padh, self.ceil_mode,
                self.count_include_pad)(input)


class MaxPool3d(Module):

    def __init__(self, kernel_size, stride=None, padding=0, dilation=1,
            return_indices=False, ceil_mode=False):
        super(MaxPool3d, self).__init__()
        self.kt, self.kh, self.kw = _triple(kernel_size)
        self.dt, self.dh, self.dw = _triple(stride or kernel_size)
        self.padt, self.padh, self.padw = _triple(padding)
        self.dilt, self.dilh, self.dilw = _triple(dilation)
        self.return_indices = return_indices
        self.ceil_mode = ceil_mode

    def forward(self, input):
        return self._backend.MaxPool3d(self.kt, self.kw, self.kh,
                self.dt, self.dw, self.dh, self.padt, self.padw, self.padh,
                self.dilt, self.dilw, self.dilh,
                self.ceil_mode, self.return_indices)(input)


class AvgPool3d(Module):

    def __init__(self, kernel_size, stride=None):
        super(AvgPool3d, self).__init__()
        self.kt, self.kh, self.kw = _triple(kernel_size)
        self.dt, self.dh, self.dw = _triple(stride or kernel_size)

    def forward(self, input):
        return self._backend.AvgPool3d(self.kt, self.kw, self.kh,
                self.dt, self.dw, self.dh)(input)


class FractionalMaxPool2d(Module):

    def __init__(self, kernel_size, output_size=None, output_ratio=None,
            return_indices=False, _random_samples=None):
        super(FractionalMaxPool2d, self).__init__()
        self.kh, self.kw = _pair(kernel_size)
        self.return_indices = return_indices
        self._random_samples = _random_samples
        if output_size is not None:
            self.outh, self.outw = _pair(output_size)
            self.rh, self.rw = None, None
            assert output_ratio is None
        elif output_ratio is not None:
            self.outh, self.outw = None, None
            self.rh, self.rw = _pair(output_ratio)
            assert output_size is None
            assert 0 < self.rh < 1
            assert 0 < self.rw < 1
        else:
            raise ValueError("FractionalMaxPool2d requires specifying either "
                "an output size, or a pooling ratio")

    def forward(self, input):
        kwargs = {}
        if self.outh is not None:
            kwargs['output_size'] = self.outh, self.outw
        else:
            kwargs['output_ratio'] = self.rh, self.rw
        func = self._backend.FractionalMaxPool2d(self.kw, self.kh,
                return_indices=self.return_indices,
                _random_samples=self._random_samples, **kwargs)
        return func(input)


class MaxUnpool3d(Module):

    def __init__(self, kernel_size, stride=None, padding=0):
        super(MaxUnpool3d, self).__init__()
        self.kt, self.kh, self.kw = _triple(kernel_size)
        self.dt, self.dh, self.dw = _triple(stride or kernel_size)
        self.padt, self.padh, self.padw = _triple(padding)

    def forward(self, input, indices):
        out_depth = (input.size(2) - 1) * self.dt + self.kt - 2*self.padt
        out_height = (input.size(3) - 1) * self.dh + self.kh - 2*self.padh
        out_width = (input.size(4) - 1) * self.dw + self.kw - 2*self.padw
        return self._backend.MaxUnpool3d(out_depth, out_width, out_height,
                self.dt, self.dw, self.dh,
                self.padt, self.padw, self.padh)(input, indices)


class LPPool2d(Module):

    def __init__(self, norm_type, kernel_size, stride=None, ceil_mode=False):
        super(LPPool2d, self).__init__()
        self.norm_type = norm_type
        self.kh, self.kw = _pair(kernel_size)
        self.dh, self.dw = _pair(stride or kernel_size)
        self.ceil_mode = ceil_mode

    def forward(self, input):
        out = input.pow(self.norm_type)
        out = self._backend.AvgPool2d(self.kw, self.kh, self.dw, self.dh,
                0, 0, self.ceil_mode, True)(out)
        return out.mul(self.kw * self.kh).pow(1./self.norm_type)


# TODO: AdaptiveMaxPool2d

