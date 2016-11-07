import math
import torch
from torch.autograd import Variable

from .module import Module
from .utils import _pair, _triple


class Conv1d(Module):
    """Applies a 1D convolution over an input signal composed of several input
    planes.

    ```
    The output value of the layer with input (b x iC x W) and output (b x oC x oW)
    can be precisely described as:
    output[b_i][oc_i][w_i] = bias[oc_i]
                + sum_iC sum_{ow = 0, oW-1} sum_{kw = 0 to kW-1}
                    weight[oc_i][ic_i][kw] * input[b_i][ic_i][stride_w * ow + kw)]
    ```

    Note that depending of the size of your kernel, several (of the last)
    columns of the input might be lost. It is up to the user
    to add proper padding.

    Args:
        in_channels: The number of expected input channels in the image given as input
        out_channels: The number of output channels the convolution layer will produce
        kernel_size: the size of the convolving kernel.
        stride: the stride of the convolving kernel.
    Input Shape: [ * , in_channels  , * ] : Input is minibatch x in_channels x iW
    Output Shape:[ * , out_channels , * ]  : Output shape is precisely minibatch x out_channels x floor((iW  + 2*padW - kW) / dW + 1)
    Members:
        weight: the learnable weights of the module of shape (out_channels x in_channels x kW)
        bias:   the learnable bias of the module of shape (out_channels)
    Examples:
        >>> m = nn.Conv1d(16, 33, 3, stride=2)
        >>> input = autograd.Variable(torch.randn(20, 16, 50))
        >>> output = m(input)
    """

    def __init__(self, in_features, out_features, kernel_size, stride=1):
        self.in_features = in_features
        self.out_features = out_features
        self.kernel_size = kernel_size
        self.stride = stride

        kernel_elements = self.in_features * self.kernel_size
        super(Conv1d, self).__init__(
            weight = torch.Tensor(out_features, in_features, kernel_size),
            bias = torch.Tensor(out_features)
        )

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.in_features * self.kernel_size)
        self.weight.data.uniform_(-stdv, stdv)
        self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        func = self._backend.Conv2d(
            stride=(1, self.stride),
            pad=(0, 0),
            groups=1)
        input = input.view(input.size(0), input.size(1), 1, input.size(2))
        weight = self.weight.view(self.weight.size(0), self.weight.size(1), 1,
                                  self.weight.size(2))
        return func(input, weight, self.bias)

    def __repr__(self):
        inplace_str=', inplace' if self.inplace else ''
        return self.__class__.__name__ + ' (' \
            + str(self.in_features) + ' -> ' + str(self.out_features) \
            + ', size=' + str(self.kernel_size) \
            + ', stride=' + str(self.stride) + ')'

class Conv2d(Module):
    """Applies a 2D convolution over an input image composed of several input
    planes.

    ```
    The output value of the layer with input (b x iC x H x W) and output (b x oC x oH x oW)
    can be precisely described as:
    output[b_i][oc_i][h_i][w_i] = bias[oc_i]
                + sum_iC sum_{oh = 0, oH-1} sum_{ow = 0, oW-1} sum_{kh = 0 to kH-1} sum_{kw = 0 to kW-1}
                    weight[oc_i][ic_i][kh][kw] * input[b_i][ic_i][stride_h * oh + kh)][stride_w * ow + kw)]
    ```

    Note that depending of the size of your kernel, several (of the last)
    columns or rows of the input image might be lost. It is up to the user
    to add proper padding in images.

    Args:
        in_channels: The number of expected input channels in the image given as input
        out_channels: The number of output channels the convolution layer will produce
        kernel_size: the size of the convolving kernel. Can be a single number k (for a square kernel of k x k) or a tuple (kh x kw)
        stride: the stride of the convolving kernel. Can be a single number s or a tuple (sh x sw). Default: 1
        padding: implicit zero padding on the input. Can be a single number s or a tuple. Default: 0
        dilation: If given, will do dilated (or atrous) convolutions. Can be a single number s or a tuple. Default: None
        bias: If set to False, the layer will not learn an additive bias. Default: True
    Input Shape: [ * , in_channels  , * , * ] : Input is minibatch x in_channels x iH x iW
    Output Shape:[ * , out_channels , * , * ]  : Output shape is precisely minibatch x out_channels x floor((iH  + 2*padH - kH) / dH + 1) x floor((iW  + 2*padW - kW) / dW + 1)
    Members:
        weight: the learnable weights of the module of shape (out_channels x in_channels x kH x kW)
        bias:   the learnable bias of the module of shape (out_channels)
    Examples:
        >>> # With square kernels and equal stride
        >>> m = nn.Conv2d(16, 33, 3, stride=2)
        >>> # non-square kernels and unequal stride and with padding
        >>> m = nn.Conv2d(16, 33, (3, 5), stride=(2, 1), padding=(4, 2))
        >>> # non-square kernels and unequal stride and with padding and dilation
        >>> m = nn.Conv2d(16, 33, (3, 5), stride=(2, 1), padding=(4, 2), dilation=(3, 1))
        >>> input = autograd.Variable(torch.randn(20, 16, 50, 100))
        >>> output = m(input)
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                padding=0, dilation=None, groups=1, bias=True):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kh, self.kw = _pair(kernel_size)
        self.dh, self.dw = _pair(stride)
        self.padh, self.padw = _pair(padding)
        self.is_dilated = dilation is not None
        if self.is_dilated:
            self.dilh, self.dilw = _pair(dilation)
        self.groups = groups

        weight = torch.Tensor(self.out_channels, self.in_channels, self.kh,
                self.kw)
        bias = torch.Tensor(self.out_channels) if bias else None
        super(Conv2d, self).__init__(
            weight=weight,
            bias=bias,
        )

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.kh * self.kw * self.in_channels)
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        if self.is_dilated:
            # TODO: merge this into the Conv2d function
            func = self._backend.DilatedConv2d(
                self.kw, self.kh, self.dw, self.dh, self.padw, self.padh,
                self.dilh, self.dilw)
        else:
            func = self._backend.Conv2d(
                stride=(self.dh, self.dw),
                pad=(self.padh, self.padw),
                groups=self.groups)
        if self.bias is None:
            return func(input, self.weight)
        else:
            return func(input, self.weight, self.bias)
    
    def __repr__(self):
        padding_str=', padding=(' + str(self.padh) + ', ' + str(self.padw) + ')' \
                      if self.padh != 0 and self.padw !=0 else ''
        dilation_str=(', dilation=(' + str(self.dilh) + ', ' \
                        + str(self.dilw) + ')' if self.is_dilated else '')
        groups_str=(', groups=' + str(self.groups) if self.groups != 1 else '')
        bias_str=(', bias=False' if self.bias == None else '')
        return  self.__class__.__name__ + ' (' + str(self.in_channels) \
            + ' -> ' + str(self.out_channels) \
            + ', size=(' + str(self.kh) + ', ' + str(self.kw) + ')' \
            + ', stride=(' + str(self.dh) + ', ' + str(self.dw) + ')' \
            + padding_str + dilation_str + groups_str + bias_str + ')'


class ConvTranspose2d(Conv2d):
    """Applies a 2D deconvolution operator over an input image composed of several input
    planes.
    The deconvolution operator multiplies each input value element-wise by a learnable kernel,
    and sums over the outputs from all input feature planes.
    This module can be seen as the exact reverse of the Conv2d module.

    Args:
        in_channels: The number of expected input channels in the image given as input
        out_channels: The number of output channels the convolution layer will produce
        kernel_size: the size of the convolving kernel. Can be a single number k (for a square kernel of k x k) or a tuple (kh x kw)
        stride: the stride of the convolving kernel. Can be a single number or a tuple (sh x sw). Default: 1
        padding: implicit zero padding on the input. Can be a single number or a tuple. Default: 0
        output_padding: A padding of 0 or 1 pixels that should be added to the output. Can be a single number or a tuple. Default: 0
        bias: If set to False, the layer will not learn an additive bias. Default: True
    Input Shape: [ * , in_channels  , * , * ] : Input is minibatch x in_channels x iH x iW
    Output Shape:[ * , out_channels , * , * ]  : Output shape is minibatch x out_channels x (iH - 1) * sH - 2*padH + kH + output_paddingH x (iW - 1) * sW - 2*padW + kW, or as specified in a second argument to the call.
    Members:
        weight: the learnable weights of the module of shape (in_channels x out_channels x kH x kW)
        bias:   the learnable bias of the module of shape (out_channels)
    Examples:
        >>> # With square kernels and equal stride
        >>> m = nn.ConvTranspose2d(16, 33, 3, stride=2)
        >>> # non-square kernels and unequal stride and with padding
        >>> m = nn.ConvTranspose2d(16, 33, (3, 5), stride=(2, 1), padding=(4, 2))
        >>> input = autograd.Variable(torch.randn(20, 16, 50, 100))
        >>> output = m(input)
        >>> # exact output size can be also specified as an argument
        >>> input = autograd.Variable(torch.randn(1, 16, 12, 12))
        >>> downsample = nn.Conv2d(16, 16, 3, stride=2, padding=1)
        >>> upsample = nn.ConvTranspose2d(16, 16, 3, stride=2, padding=1)
        >>> h = downsample(input)
        >>> output = upsample(h, output_size=input.size())
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, output_padding=0, bias=True):
        super(ConvTranspose2d, self).__init__(in_channels, out_channels, kernel_size,
                                              stride=stride, padding=padding, bias=bias)
        # Conv2d uses a different weight layout than Conv2d
        self.weight.data = self.weight.data.transpose(0, 1).contiguous()
        self.out_padh, self.out_padw = _pair(output_padding)

    def __call__(self, input, output_size=None):
        if output_size:
            self.output_size = list(output_size)
            if len(self.output_size) == 4:
                self.output_size = self.output_size[-2:]
            if len(self.output_size) != 2:
                raise ValueError("output_size should be a sequence containing "
                        "2 or 4 elements, but it has a length of {}".format(
                            len(output_size)))
        else:
            self.output_size = None
        return super(ConvTranspose2d, self).__call__(input)


    def forward(self, input):
        out_padh, out_padw = self.out_padh, self.out_padw
        if self.output_size is not None:
            out_sizew, out_sizeh = self.output_size
            sizew = ((input.size(3) - 1) * self.dw - 2 * self.padw + self.kw)
            sizeh = ((input.size(2) - 1) * self.dh - 2 * self.padh + self.kh)
            out_padw = out_sizew - sizew
            out_padh = out_sizeh - sizeh
            out_padw_ok = 0 <= out_padw < self.dw
            out_padh_ok = 0 <= out_padh < self.dh
            if not out_padw_ok or not out_padh_ok:
                raise ValueError(("requested an output size of {}x{}, but "
                    "valid sizes range from {}x{} to {}x{} (for an input of "
                    "{}x{})").format(out_sizeh, out_sizew, sizeh, sizew,
                        sizeh+self.dh-1, sizew+self.dw-1,
                        input.size(2), input.size(3)))
        func = self._backend.ConvTranspose2d(
            self.kw, self.kh, self.dw, self.dh, self.padw, self.padh,
            out_padw, out_padh)
        if self.bias is None:
            return func(input, self.weight)
        else:
            return func(input, self.weight, self.bias)


class _Conv3dBase(Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
            padding=0):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kt, self.kh, self.kw = _triple(kernel_size)
        self.dt, self.dh, self.dw = _triple(stride)
        self.padt, self.padh, self.padw = _triple(padding)

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.kt * self.kh * self.kw * self.in_channels)
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def __repr__(self):
        padding_str=', padding=(' + str(self.padt) \
                      + ', ' + str(self.padh) + ', ' + str(self.padw) + ')' \
                      if self.padt != 0 and self.padh != 0 and self.padw !=0 else ''
        return  self.__class__.__name__ + ' (' + str(self.in_channels) \
            + ' -> ' + str(self.out_channels) \
            + ', size=(' + str(self.kt) + ', ' + str(self.kh) + ', ' + str(self.kw) + ')' \
            + ', stride=(' + str(self.dt) + ', ' + str(self.dh) + ', ' + str(self.dw) + ')' \
            + padding_str + ')'


class Conv3d(_Conv3dBase):
    """Applies a 3D convolution over an input image composed of several input
    planes.

    Note that depending of the size of your kernel, several (of the last)
    columns or rows of the input image might be lost. It is up to the user
    to add proper padding in images.

    Args:
        in_channels: The number of expected input channels in the image given as input
        out_channels: The number of output channels the convolution layer will produce
        kernel_size: the size of the convolving kernel. Can be a single number k (for a square kernel of k x k x k) or a tuple (kt x kh x kw)
        stride: the stride of the convolving kernel. Can be a single number s or a tuple (kt x sh x sw). Default: 1
        padding: implicit zero padding on the input. Can be a single number s or a tuple. Default: 0
    Input Shape: [ * , in_channels  , * , * , * ] : Input is minibatch x in_channels x iT x iH x iW
    Output Shape:[ * , out_channels , * , * , * ]  : Output shape is precisely minibatch x out_channels x floor((iT  + 2*padT - kT) / dT + 1) x floor((iH  + 2*padH - kH) / dH + 1) x floor((iW  + 2*padW - kW) / dW + 1)
    Members:
        weight: the learnable weights of the module of shape (out_channels x in_channels x kT x kH x kW)
        bias:   the learnable bias of the module of shape (out_channels)
    Examples:
        >>> # With square kernels and equal stride
        >>> m = nn.Conv3d(16, 33, 3, stride=2)
        >>> # non-square kernels and unequal stride and with padding
        >>> m = nn.Conv3d(16, 33, (3, 5, 2), stride=(2, 1, 1), padding=(4, 2, 0))
        >>> input = autograd.Variable(torch.randn(20, 16, 10, 50, 100))
        >>> output = m(input)
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                padding=0):
        super(Conv3d, self).__init__(in_channels, out_channels, kernel_size,
                stride, padding)
        weight = torch.Tensor(self.out_channels, self.in_channels, self.kt,
                self.kh, self.kw)
        bias = torch.Tensor(self.out_channels)
        Module.__init__(self, weight=weight, bias=bias)
        self.reset_parameters()

    def forward(self, input):
        func = self._backend.Conv3d(
            self.kt, self.kw, self.kh, self.dt, self.dw, self.dh, self.padt,
            self.padw, self.padh)
        if self.bias is None:
            return func(input, self.weight)
        else:
            return func(input, self.weight, self.bias)


class ConvTranspose3d(_Conv3dBase):
    """Applies a 3D deconvolution operator over an input image composed of several input
    planes.
    The deconvolution operator multiplies each input value element-wise by a learnable kernel,
    and sums over the outputs from all input feature planes.
    This module can be seen as the exact reverse of the Conv3d module.

    Args:
        in_channels: The number of expected input channels in the image given as input
        out_channels: The number of output channels the convolution layer will produce
        kernel_size: the size of the convolving kernel. Can be a single number k (for a square kernel of k x k x k) or a tuple (kt x kh x kw)
        stride: the stride of the convolving kernel. Can be a single number or a tuple (st x sh x sw). Default: 1
        padding: implicit zero padding on the input. Can be a single number or a tuple. Default: 0
        output_padding: A padding of 0 or 1 pixels that should be added to the output. Can be a single number or a tuple. Default: 0
    Input Shape: [ * , in_channels  , * , * , * ] : Input is minibatch x in_channels x iH x iW
    Output Shape:[ * , out_channels , * , * , * ]  : Output shape is precisely minibatch x out_channels x (iT - 1) * sT - 2*padT + kT + output_paddingT x (iH - 1) * sH - 2*padH + kH + output_paddingH x (iW - 1) * sW - 2*padW + kW
    Members:
        weight: the learnable weights of the module of shape (in_channels x out_channels x kT x kH x kW)
        bias:   the learnable bias of the module of shape (out_channels)
    Examples:
        >>> # With square kernels and equal stride
        >>> m = nn.ConvTranspose3d(16, 33, 3, stride=2)
        >>> # non-square kernels and unequal stride and with padding
        >>> m = nn.Conv3d(16, 33, (3, 5, 2), stride=(2, 1, 1), padding=(0, 4, 2))
        >>> input = autograd.Variable(torch.randn(20, 16, 10, 50, 100))
        >>> output = m(input)
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                padding=0):
        super(ConvTranspose3d, self).__init__(in_channels, out_channels, kernel_size,
                stride, padding)
        weight = torch.Tensor(self.in_channels, self.out_channels, self.kt,
                self.kh, self.kw)
        bias = torch.Tensor(self.out_channels)
        Module.__init__(self, weight=weight, bias=bias)
        self.reset_parameters()

    def forward(self, input):
        func = self._backend.ConvTranspose3d(
            self.kt, self.kw, self.kh, self.dt, self.dw, self.dh, self.padt,
            self.padw, self.padh)
        if self.bias is None:
            return func(input, self.weight)
        else:
            return func(input, self.weight, self.bias)


# TODO: Conv2dLocal
# TODO: Conv2dMap
# TODO: ConvTranspose2dMap
