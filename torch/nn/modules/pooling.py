import torch
from torch.autograd import Variable

from .module import Module
from .utils import _pair, _triple


class MaxPool1d(Module):
    """Applies a 1D max pooling over an input signal composed of several input
    planes. 

    ```
    The output value of the layer with input (b x C x W) and output (b x C x oW)
    can be precisely described as:
    output[b_i][c_i][w_i] = max_{k=1, K} input[b_i][c_i][stride_w * w_i + k)]
    ```

    Args:
        kernel_size: the size of the window to take a max over
        stride: the stride of the window
        padding: implicit padding to be added. Default: 0
        dilation: a parameter that controls the stride of elements in the window. Default: kernel_size
        return_indices: if True, will return the indices along with the outputs. Useful when Unpooling later. Default: False
        ceil_mode: when True, will use "ceil" instead of "floor" to compute the output shape
    Input Shape: [ * , * , * ] : Input is minibatch x channels x iW
    Output Shape:[ * , * , * ]  : Output shape = minibatch x channels x floor((iW  + 2*padW - kernel_size) / stride + 1)
    Examples:
        >>> # pool of size=3, stride=2
        >>> m = nn.MaxPool1d(3, stride=2)
        >>> input = autograd.Variable(torch.randn(20, 16, 50))
        >>> output = m.forward(input)
    """
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
    """Applies a 2D max pooling over an input signal composed of several input
    planes. 

    ```
    The output value of the layer with input (b x C x H x W) and output (b x C x oH x oW)
    can be precisely described as:
    output[b_i][c_i][h_i][w_i] = max_{{kh=1, KH}, {kw=1, kW}} input[b_i][c_i][stride_h * h_i + kH)][stride_w * w_i + kW)]
    ```

    Args:
        kernel_size: the size of the window to take a max over. Can be a single number k (for a square kernel of k x k) or a tuple (kh x kw)
        stride: the stride of the window. Can be a single number s or a tuple (sh x sw). Default: kernel_size
        padding: implicit padding to be added. Can be a single number or a tuple. Default: 0
        dilation: a parameter that controls the stride of elements in the window. Can be a single number or a tuple. Default: 1
        return_indices: if True, will return the indices along with the outputs. Useful to pass to nn.MaxUnpool2d . Default: False
        ceil_mode: when True, will use "ceil" instead of "floor" to compute the output shape
    Input Shape: [ * , * , *, * ] : Input is minibatch x channels x iH x iW
    Output Shape:[ * , * , *, * ]  : Output shape = minibatch x channels x floor((iH  + 2*padH - kH) / sH + 1) x floor((iW  + 2*padW - kW) / sW + 1)
    Examples:
        >>> # pool of square window of size=3, stride=2
        >>> m = nn.MaxPool2d(3, stride=2)
        >>> # pool of non-square window
        >>> m = nn.MaxPool2d((3, 2), stride=(2, 1))
        >>> input = autograd.Variable(torch.randn(20, 16, 50, 32))
        >>> output = m.forward(input)
    """
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
    """Computes the inverse operation of MaxPool2d
    MaxPool2d is not invertible, as the locations of the max locations are lost.
    MaxUnpool2d takes in as input the output of MaxPool2d and the indices of the Max locations
    and computes the inverse.

    Args:
        kernel_size: the size of the max window. Can be a single number k (for a square kernel of k x k) or a tuple (kh x kw)
        stride: the stride of the window. Can be a single number s or a tuple (sh x sw). Default: kernel_size
        padding: implicit padding that was added to the input. Can be a single number or a tuple. Default: 0
    Input Shape: [ * , * , *, * ] : Input is minibatch x channels x iH x iW
    Output Shape:[ * , * , *, * ]  : Output shape = minibatch x channels x padH x (iH - 1) * sH + kH x padW x (iW - 1) * sW + kW
    Examples:
        >>> # pool of square window of size=3, stride=2
        >>> m = nn.MaxPool2d(3, stride=2, return_indices = True)
        >>> mu = nn.MaxUnpool2d(3, stride=2)
        >>> input, indices = autograd.Variable(torch.randn(20, 16, 50, 32))
        >>> output = m.forward(input)
        >>> unpooled_output = m2.forward(output, indices)
    """
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
    """Applies a 2D average pooling over an input signal composed of several input
    planes. 

    ```
    The output value of the layer with input (b x C x H x W) and output (b x C x oH x oW)
    can be precisely described as:
    output[b_i][c_i][h_i][w_i] = (1 / K) * sum_{kh=1, KH} sum_{kw=1, kW}  input[b_i][c_i][stride_h * h_i + kh)][stride_w * w_i + kw)]
    ```

    Args:
        kernel_size: the size of the window. Can be a single number k (for a square kernel of k x k) or a tuple (kh x kw)
        stride: the stride of the window. Can be a single number s or a tuple (sh x sw). Default: kernel_size
        padding: implicit padding to be added. Can be a single number or a tuple. Default: 0
        ceil_mode: when True, will use "ceil" instead of "floor" to compute the output shape
    Input Shape: [ * , * , *, * ] : Input is minibatch x channels x iH x iW
    Output Shape:[ * , * , *, * ]  : Output shape = minibatch x channels x floor((iH  + 2*padH - kH) / sH + 1) x floor((iW  + 2*padW - kW) / sW + 1)
    Examples:
        >>> # pool of square window of size=3, stride=2
        >>> m = nn.AvgPool2d(3, stride=2)
        >>> # pool of non-square window
        >>> m = nn.AvgPool2d((3, 2), stride=(2, 1))
        >>> input = autograd.Variable(torch.randn(20, 16, 50, 32))
        >>> output = m.forward(input)
    """
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
    """Applies a 3D max pooling over an input signal composed of several input
    planes. 

    Args:
        kernel_size: the size of the window to take a max over. Can be a single number k (for a square kernel of k x k x k) or a tuple (kt x kh x kw)
        stride: the stride of the window. Can be a single number s or a tuple (st x sh x sw). Default: kernel_size
        padding: implicit padding to be added. Can be a single number or a tuple. Default: 0
        dilation: a parameter that controls the stride of elements in the window. Can be a single number or a tuple. Default: 1
        return_indices: if True, will return the indices along with the outputs. Useful to pass to nn.MaxUnpool3d . Default: False
        ceil_mode: when True, will use "ceil" instead of "floor" to compute the output shape
    Input Shape: [ * , * , *, *, * ] : Input is minibatch x channels x iT x iH x iW
    Output Shape:[ * , * , *, *, * ]  : Output shape = minibatch x channels x floor((iT  + 2*padT - kT) / sT + 1) x floor((iH  + 2*padH - kH) / sH + 1) x floor((iW  + 2*padW - kW) / sW + 1)
    Examples:
        >>> # pool of square window of size=3, stride=2
        >>> m = nn.MaxPool3d(3, stride=2)
        >>> # pool of non-square window
        >>> m = nn.MaxPool3d((3, 2, 2), stride=(2, 1, 2))
        >>> input = autograd.Variable(torch.randn(20, 16, 50,44, 31))
        >>> output = m.forward(input)
    """
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
    """Applies a 3D average pooling over an input signal composed of several input
    planes. 

    Args:
        kernel_size: the size of the window to take a average over. Can be a single number k (for a square kernel of k x k x k) or a tuple (kt x kh x kw)
        stride: the stride of the window. Can be a single number s or a tuple (st x sh x sw). Default: kernel_size
    Input Shape: [ * , * , *, *, * ] : Input is minibatch x channels x iT x iH x iW
    Output Shape:[ * , * , *, *, * ]  : Output shape = minibatch x channels x floor((iT  + 2*padT - kT) / sT + 1) x floor((iH  + 2*padH - kH) / sH + 1) x floor((iW  + 2*padW - kW) / sW + 1)
    Examples:
        >>> # pool of square window of size=3, stride=2
        >>> m = nn.AvgPool3d(3, stride=2)
        >>> # pool of non-square window
        >>> m = nn.AvgPool3d((3, 2, 2), stride=(2, 1, 2))
        >>> input = autograd.Variable(torch.randn(20, 16, 50,44, 31))
        >>> output = m.forward(input)
    """
    def __init__(self, kernel_size, stride=None):
        super(AvgPool3d, self).__init__()
        self.kt, self.kh, self.kw = _triple(kernel_size)
        self.dt, self.dh, self.dw = _triple(stride or kernel_size)

    def forward(self, input):
        return self._backend.AvgPool3d(self.kt, self.kw, self.kh,
                self.dt, self.dw, self.dh)(input)


class FractionalMaxPool2d(Module):
    """Applies a 2D fractional max pooling over an input signal composed of several input
    planes. 

    Fractiona MaxPooling is described in detail in the paper ["Fractional Max-Pooling" by Ben Graham](http://arxiv.org/abs/1412.6071)
    The max-pooling operation is applied in kHxkW regions by a stochastic
    step size determined by the target output size.
    The number of output features is equal to the number of input planes.

    Args:
        kernel_size: the size of the window to take a max over. Can be a single number k (for a square kernel of k x k) or a tuple (kh x kw)
        output_size: the target output size of the image of the form oH x oW. Can be a tuple (oH, oW) or a single number oH for a square image oH x oH
        output_ratio: If one wants to have an output size as a ratio of the input size, this option can be given. This has to be a number or tuple in the range (0, 1)
        return_indices: if True, will return the indices along with the outputs. Useful to pass to nn.MaxUnpool2d . Default: False        
    Input Shape: [ * , * , *, * ] : Input is minibatch x channels x iH x iW
    Output Shape:[ * , * , *, * ]  : Output shape = minibatch x channels x floor((iH  + 2*padH - kH) / sH + 1) x floor((iW  + 2*padW - kW) / sW + 1)
    Examples:
        >>> # pool of square window of size=3, and target output size 13x12
        >>> m = nn.FractionalMaxPool2d(3, output_size=(13, 12))
        >>> # pool of square window and target output size being half of input image size
        >>> m = nn.FractionalMaxPool2d(3, output_ratio=(0.5, 0.5))
        >>> input = autograd.Variable(torch.randn(20, 16, 50, 32))
        >>> output = m.forward(input)
    """
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
    """Computes the inverse operation of MaxPool3d
    MaxPool3d is not invertible, as the locations of the max locations are lost.
    MaxUnpool3d takes in as input the output of MaxPool3d and the indices of the Max locations
    and computes the inverse.

    Args:
        kernel_size: the size of the max window. Can be a single number k (for a square kernel of k x k) or a tuple (kt x kh x kw)
        stride: the stride of the window. Can be a single number s or a tuple (st x sh x sw). Default: kernel_size
        padding: implicit padding that was added to the input. Can be a single number or a tuple. Default: 0
    Input Shape: [ * , * , *, *, * ] : Input is minibatch x channels x iT x iH x iW
    Output Shape:[ * , * , *, *, * ]  : Output shape = minibatch x channels x padT x (iT - 1) * sT + kT x padH x (iH - 1) * sH + kH x padW x (iW - 1) * sW + kW
    Examples:
        >>> # pool of square window of size=3, stride=2
        >>> m = nn.MaxPool3d(3, stride=2, return_indices = True)
        >>> mu = nn.MaxUnpool3d(3, stride=2)
        >>> input, indices = autograd.Variable(torch.randn(20, 16, 50, 32, 15))
        >>> output = m.forward(input)
        >>> unpooled_output = m2.forward(output, indices)
    """
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
    """Applies a 2D power-average pooling over an input signal composed of several input
    planes. 
    On each window, the function computed is: f(X) = pow(sum(pow(X, p)), 1/p)
    At p = infinity, one gets Max Pooling
    At p = 1, one gets Average Pooling
    Args:
        kernel_size: the size of the window. Can be a single number k (for a square kernel of k x k) or a tuple (kh x kw)
        stride: the stride of the window. Can be a single number s or a tuple (sh x sw). Default: kernel_size
        ceil_mode: when True, will use "ceil" instead of "floor" to compute the output shape
    Input Shape: [ * , * , *, * ] : Input is minibatch x channels x iH x iW
    Output Shape:[ * , * , *, * ]  : Output shape = minibatch x channels x floor((iH  + 2*padH - kH) / sH + 1) x floor((iW  + 2*padW - kW) / sW + 1)
    Examples:
        >>> # power-2 pool of square window of size=3, stride=2
        >>> m = nn.LPPool2d(2, 3, stride=2)
        >>> # pool of non-square window of power 1.2
        >>> m = nn.LPPool2d(1.2, (3, 2), stride=(2, 1))
        >>> input = autograd.Variable(torch.randn(20, 16, 50, 32))
        >>> output = m.forward(input)
    """
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

