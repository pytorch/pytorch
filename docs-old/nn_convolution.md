## Convolution Layers
### Conv1d

Applies a 1D convolution over an input signal composed of several input

```python
The output value of the layer with input (b x iC x W) and output (b x oC x oW)
can be precisely described as:
output[b_i][oc_i][w_i] = bias[oc_i]
            + sum_iC sum_{ow = 0, oW-1} sum_{kw = 0 to kW-1}
                weight[oc_i][ic_i][kw] * input[b_i][ic_i][stride_w * ow + kw)]
```

```python
m = nn.Conv1d(16, 33, 3, stride=2)
input = autograd.Variable(torch.randn(20, 16, 50))
output = m(input)
```

planes.


Note that depending of the size of your kernel, several (of the last)
columns of the input might be lost. It is up to the user
to add proper padding.


#### Constructor Arguments

Parameter | Default | Description
--------- | ------- | -----------
in_channels |  | The number of expected input channels in the image given as input
out_channels |  | The number of output channels the convolution layer will produce
kernel_size |  | the size of the convolving kernel.
stride |  | the stride of the convolving kernel.

#### Expected Shape
       | Shape | Description 
------ | ----- | ------------
 input | [ * , in_channels  , * ]  | Input is minibatch x in_channels x iW
output | [ * , out_channels , * ]   | Output shape is precisely minibatch x out_channels x floor((iW  + 2*padW - kW) / dW + 1)

#### Members

Parameter | Description
--------- | -----------
weight | the learnable weights of the module of shape (out_channels x in_channels x kW)
bias | the learnable bias of the module of shape (out_channels)
### Conv2d

Applies a 2D convolution over an input image composed of several input

```python
The output value of the layer with input (b x iC x H x W) and output (b x oC x oH x oW)
can be precisely described as:
output[b_i][oc_i][h_i][w_i] = bias[oc_i]
            + sum_iC sum_{oh = 0, oH-1} sum_{ow = 0, oW-1} sum_{kh = 0 to kH-1} sum_{kw = 0 to kW-1}
                weight[oc_i][ic_i][kh][kw] * input[b_i][ic_i][stride_h * oh + kh)][stride_w * ow + kw)]
```

```python
# With square kernels and equal stride
m = nn.Conv2d(16, 33, 3, stride=2)
# non-square kernels and unequal stride and with padding
m = nn.Conv2d(16, 33, (3, 5), stride=(2, 1), padding=(4, 2))
# non-square kernels and unequal stride and with padding and dilation
m = nn.Conv2d(16, 33, (3, 5), stride=(2, 1), padding=(4, 2), dilation=(3, 1))
input = autograd.Variable(torch.randn(20, 16, 50, 100))
output = m(input)
```

planes.


Note that depending of the size of your kernel, several (of the last)
columns or rows of the input image might be lost. It is up to the user
to add proper padding in images.


#### Constructor Arguments

Parameter | Default | Description
--------- | ------- | -----------
in_channels |  | The number of expected input channels in the image given as input
out_channels |  | The number of output channels the convolution layer will produce
kernel_size |  | the size of the convolving kernel. Can be a single number k (for a square kernel of k x k) or a tuple (kh x kw)
stride | 1 | the stride of the convolving kernel. Can be a single number s or a tuple (sh x sw).
padding | 0 | implicit zero padding on the input. Can be a single number s or a tuple.
dilation | None | If given, will do dilated (or atrous) convolutions. Can be a single number s or a tuple.
bias | True | If set to False, the layer will not learn an additive bias.

#### Expected Shape
       | Shape | Description 
------ | ----- | ------------
 input | [ * , in_channels  , * , * ]  | Input is minibatch x in_channels x iH x iW
output | [ * , out_channels , * , * ]   | Output shape is precisely minibatch x out_channels x floor((iH  + 2*padH - kH) / dH + 1) x floor((iW  + 2*padW - kW) / dW + 1)

#### Members

Parameter | Description
--------- | -----------
weight | the learnable weights of the module of shape (out_channels x in_channels x kH x kW)
bias | the learnable bias of the module of shape (out_channels)
### ConvTranspose2d

Applies a 2D deconvolution operator over an input image composed of several input

```python
# With square kernels and equal stride
m = nn.ConvTranspose2d(16, 33, 3, stride=2)
# non-square kernels and unequal stride and with padding
m = nn.ConvTranspose2d(16, 33, (3, 5), stride=(2, 1), padding=(4, 2))
input = autograd.Variable(torch.randn(20, 16, 50, 100))
output = m(input)
# exact output size can be also specified as an argument
input = autograd.Variable(torch.randn(1, 16, 12, 12))
downsample = nn.Conv2d(16, 16, 3, stride=2, padding=1)
upsample = nn.ConvTranspose2d(16, 16, 3, stride=2, padding=1)
h = downsample(input)
output = upsample(h, output_size=input.size())
```

planes.
The deconvolution operator multiplies each input value element-wise by a learnable kernel,
and sums over the outputs from all input feature planes.
This module can be seen as the exact reverse of the Conv2d module.


#### Constructor Arguments

Parameter | Default | Description
--------- | ------- | -----------
in_channels |  | The number of expected input channels in the image given as input
out_channels |  | The number of output channels the convolution layer will produce
kernel_size |  | the size of the convolving kernel. Can be a single number k (for a square kernel of k x k) or a tuple (kh x kw)
stride | 1 | the stride of the convolving kernel. Can be a single number or a tuple (sh x sw).
padding | 0 | implicit zero padding on the input. Can be a single number or a tuple.
output_padding | 0 | A zero-padding of 0 <= padding < stride that should be added to the output. Can be a single number or a tuple.
bias | True | If set to False, the layer will not learn an additive bias.

#### Expected Shape
       | Shape | Description 
------ | ----- | ------------
 input | [ * , in_channels  , * , * ]  | Input is minibatch x in_channels x iH x iW
output | [ * , out_channels , * , * ]   | Output shape is minibatch x out_channels x (iH - 1) * sH - 2*padH + kH + output_paddingH x (iW - 1) * sW - 2*padW + kW, or as specified in a second argument to the call.

#### Members

Parameter | Description
--------- | -----------
weight | the learnable weights of the module of shape (in_channels x out_channels x kH x kW)
bias | the learnable bias of the module of shape (out_channels)
### Conv3d

Applies a 3D convolution over an input image composed of several input

```python
# With square kernels and equal stride
m = nn.Conv3d(16, 33, 3, stride=2)
# non-square kernels and unequal stride and with padding
m = nn.Conv3d(16, 33, (3, 5, 2), stride=(2, 1, 1), padding=(4, 2, 0))
input = autograd.Variable(torch.randn(20, 16, 10, 50, 100))
output = m(input)
```

planes.

Note that depending of the size of your kernel, several (of the last)
columns or rows of the input image might be lost. It is up to the user
to add proper padding in images.


#### Constructor Arguments

Parameter | Default | Description
--------- | ------- | -----------
in_channels |  | The number of expected input channels in the image given as input
out_channels |  | The number of output channels the convolution layer will produce
kernel_size |  | the size of the convolving kernel. Can be a single number k (for a square kernel of k x k x k) or a tuple (kt x kh x kw)
stride | 1 | the stride of the convolving kernel. Can be a single number s or a tuple (kt x sh x sw).
padding | 0 | implicit zero padding on the input. Can be a single number s or a tuple.

#### Expected Shape
       | Shape | Description 
------ | ----- | ------------
 input | [ * , in_channels  , * , * , * ]  | Input is minibatch x in_channels x iT x iH x iW
output | [ * , out_channels , * , * , * ]   | Output shape is precisely minibatch x out_channels x floor((iT  + 2*padT - kT) / dT + 1) x floor((iH  + 2*padH - kH) / dH + 1) x floor((iW  + 2*padW - kW) / dW + 1)

#### Members

Parameter | Description
--------- | -----------
weight | the learnable weights of the module of shape (out_channels x in_channels x kT x kH x kW)
bias | the learnable bias of the module of shape (out_channels)
### ConvTranspose3d

Applies a 3D deconvolution operator over an input image composed of several input

```python
# With square kernels and equal stride
m = nn.ConvTranspose3d(16, 33, 3, stride=2)
# non-square kernels and unequal stride and with padding
m = nn.Conv3d(16, 33, (3, 5, 2), stride=(2, 1, 1), padding=(0, 4, 2))
input = autograd.Variable(torch.randn(20, 16, 10, 50, 100))
output = m(input)
```

planes.
The deconvolution operator multiplies each input value element-wise by a learnable kernel,
and sums over the outputs from all input feature planes.
This module can be seen as the exact reverse of the Conv3d module.


#### Constructor Arguments

Parameter | Default | Description
--------- | ------- | -----------
in_channels |  | The number of expected input channels in the image given as input
out_channels |  | The number of output channels the convolution layer will produce
kernel_size |  | the size of the convolving kernel. Can be a single number k (for a square kernel of k x k x k) or a tuple (kt x kh x kw)
stride | 1 | the stride of the convolving kernel. Can be a single number or a tuple (st x sh x sw).
padding | 0 | implicit zero padding on the input. Can be a single number or a tuple.
output_padding | 0 | A zero-padding of 0 <= padding < stride that should be added to the output. Can be a single number or a tuple.

#### Expected Shape
       | Shape | Description 
------ | ----- | ------------
 input | [ * , in_channels  , * , * , * ]  | Input is minibatch x in_channels x iH x iW
output | [ * , out_channels , * , * , * ]   | Output shape is precisely minibatch x out_channels x (iT - 1) * sT - 2*padT + kT + output_paddingT x (iH - 1) * sH - 2*padH + kH + output_paddingH x (iW - 1) * sW - 2*padW + kW

#### Members

Parameter | Description
--------- | -----------
weight | the learnable weights of the module of shape (in_channels x out_channels x kT x kH x kW)
bias | the learnable bias of the module of shape (out_channels)
