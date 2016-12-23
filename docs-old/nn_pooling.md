## Pooling Layers
### MaxPool1d

Applies a 1D max pooling over an input signal composed of several input

```python
The output value of the layer with input (b x C x W) and output (b x C x oW)
can be precisely described as:
output[b_i][c_i][w_i] = max_{k=1, K} input[b_i][c_i][stride_w * w_i + k)]
```

```python
# pool of size=3, stride=2
m = nn.MaxPool1d(3, stride=2)
input = autograd.Variable(torch.randn(20, 16, 50))
output = m(input)
```

planes.



#### Constructor Arguments

Parameter | Default | Description
--------- | ------- | -----------
kernel_size |  | the size of the window to take a max over
stride |  | the stride of the window
padding | 0 | implicit padding to be added.
dilation | kernel_size | a parameter that controls the stride of elements in the window.
return_indices | False | if True, will return the indices along with the outputs. Useful when Unpooling later.
ceil_mode |  | when True, will use "ceil" instead of "floor" to compute the output shape

#### Expected Shape
       | Shape | Description 
------ | ----- | ------------
 input | [ * , * , * ]  | Input is minibatch x channels x iW
output | [ * , * , * ]   | Output shape = minibatch x channels x floor((iW  + 2*padW - kernel_size) / stride + 1)
### MaxPool2d

Applies a 2D max pooling over an input signal composed of several input

```python
The output value of the layer with input (b x C x H x W) and output (b x C x oH x oW)
can be precisely described as:
output[b_i][c_i][h_i][w_i] = max_{{kh=1, KH}, {kw=1, kW}} input[b_i][c_i][stride_h * h_i + kH)][stride_w * w_i + kW)]
```

```python
# pool of square window of size=3, stride=2
m = nn.MaxPool2d(3, stride=2)
# pool of non-square window
m = nn.MaxPool2d((3, 2), stride=(2, 1))
input = autograd.Variable(torch.randn(20, 16, 50, 32))
output = m(input)
```

planes.



#### Constructor Arguments

Parameter | Default | Description
--------- | ------- | -----------
kernel_size |  | the size of the window to take a max over. Can be a single number k (for a square kernel of k x k) or a tuple (kh x kw)
stride | kernel_size | the stride of the window. Can be a single number s or a tuple (sh x sw).
padding | 0 | implicit padding to be added. Can be a single number or a tuple.
dilation | 1 | a parameter that controls the stride of elements in the window. Can be a single number or a tuple.
return_indices | False | if True, will return the indices along with the outputs. Useful to pass to nn.MaxUnpool2d .
ceil_mode |  | when True, will use "ceil" instead of "floor" to compute the output shape

#### Expected Shape
       | Shape | Description 
------ | ----- | ------------
 input | [ * , * , *, * ]  | Input is minibatch x channels x iH x iW
output | [ * , * , *, * ]   | Output shape = minibatch x channels x floor((iH  + 2*padH - kH) / sH + 1) x floor((iW  + 2*padW - kW) / sW + 1)
### MaxPool3d

Applies a 3D max pooling over an input signal composed of several input

```python
# pool of square window of size=3, stride=2
m = nn.MaxPool3d(3, stride=2)
# pool of non-square window
m = nn.MaxPool3d((3, 2, 2), stride=(2, 1, 2))
input = autograd.Variable(torch.randn(20, 16, 50,44, 31))
output = m(input)
```

planes.


#### Constructor Arguments

Parameter | Default | Description
--------- | ------- | -----------
kernel_size |  | the size of the window to take a max over. Can be a single number k (for a square kernel of k x k x k) or a tuple (kt x kh x kw)
stride | kernel_size | the stride of the window. Can be a single number s or a tuple (st x sh x sw).
padding | 0 | implicit padding to be added. Can be a single number or a tuple.
dilation | 1 | a parameter that controls the stride of elements in the window. Can be a single number or a tuple.
return_indices | False | if True, will return the indices along with the outputs. Useful to pass to nn.MaxUnpool3d .
ceil_mode |  | when True, will use "ceil" instead of "floor" to compute the output shape

#### Expected Shape
       | Shape | Description 
------ | ----- | ------------
 input | [ * , * , *, *, * ]  | Input is minibatch x channels x iT x iH x iW
output | [ * , * , *, *, * ]   | Output shape = minibatch x channels x floor((iT  + 2*padT - kT) / sT + 1) x floor((iH  + 2*padH - kH) / sH + 1) x floor((iW  + 2*padW - kW) / sW + 1)
### MaxUnpool2d

Computes the inverse operation of MaxPool2d

```python
# pool of square window of size=3, stride=2
m = nn.MaxPool2d(2, stride=2, return_indices = True)
mu = nn.MaxUnpool2d(2, stride=2)
input = autograd.Variable(torch.randn(20, 16, 50, 32))
output, indices = m(input)
unpooled_output = mu.forward(output, indices)
# exact output size can be also specified as an argument
input = autograd.Variable(torch.randn(1, 16, 11, 11))
downsample = nn.MaxPool2d(3, 3, return_indices=True)
upsample = nn.MaxUnpool2d(3, 3)
h, indices = downsample(input)
output = upsample(h, indices, output_size=input.size())
```

MaxPool2d is not invertible, as the locations of the max locations are lost.
MaxUnpool2d takes in as input the output of MaxPool2d and the indices of the Max locations
and computes the inverse.


#### Constructor Arguments

Parameter | Default | Description
--------- | ------- | -----------
kernel_size |  | the size of the max window. Can be a single number k (for a square kernel of k x k) or a tuple (kh x kw)
stride | kernel_size | the stride of the window. Can be a single number s or a tuple (sh x sw).
padding | 0 | implicit padding that was added to the input. Can be a single number or a tuple.

#### Expected Shape
       | Shape | Description 
------ | ----- | ------------
 input | [ * , * , *, * ]  | Input is minibatch x channels x iH x iW
output | [ * , * , *, * ]   | Output shape is minibatch x channels x padH x (iH - 1) * sH + kH x padW x (iW - 1) * sW + kW, or as specified to the call.
### MaxUnpool3d

Computes the inverse operation of MaxPool3d

```python
# pool of square window of size=3, stride=2
m = nn.MaxPool3d(3, stride=2, return_indices = True)
mu = nn.MaxUnpool3d(3, stride=2)
input, indices = autograd.Variable(torch.randn(20, 16, 50, 32, 15))
output = m(input)
unpooled_output = m2.forward(output, indices)
```

MaxPool3d is not invertible, as the locations of the max locations are lost.
MaxUnpool3d takes in as input the output of MaxPool3d and the indices of the Max locations
and computes the inverse.


#### Constructor Arguments

Parameter | Default | Description
--------- | ------- | -----------
kernel_size |  | the size of the max window. Can be a single number k (for a square kernel of k x k) or a tuple (kt x kh x kw)
stride | kernel_size | the stride of the window. Can be a single number s or a tuple (st x sh x sw).
padding | 0 | implicit padding that was added to the input. Can be a single number or a tuple.

#### Expected Shape
       | Shape | Description 
------ | ----- | ------------
 input | [ * , * , *, *, * ]  | Input is minibatch x channels x iT x iH x iW
output | [ * , * , *, *, * ]   | Output shape = minibatch x channels x padT x (iT - 1) * sT + kT x padH x (iH - 1) * sH + kH x padW x (iW - 1) * sW + kW
### AvgPool2d

Applies a 2D average pooling over an input signal composed of several input

```python
The output value of the layer with input (b x C x H x W) and output (b x C x oH x oW)
can be precisely described as:
output[b_i][c_i][h_i][w_i] = (1 / K) * sum_{kh=1, KH} sum_{kw=1, kW}  input[b_i][c_i][stride_h * h_i + kh)][stride_w * w_i + kw)]
```

```python
# pool of square window of size=3, stride=2
m = nn.AvgPool2d(3, stride=2)
# pool of non-square window
m = nn.AvgPool2d((3, 2), stride=(2, 1))
input = autograd.Variable(torch.randn(20, 16, 50, 32))
output = m(input)
```

planes.



#### Constructor Arguments

Parameter | Default | Description
--------- | ------- | -----------
kernel_size |  | the size of the window. Can be a single number k (for a square kernel of k x k) or a tuple (kh x kw)
stride | kernel_size | the stride of the window. Can be a single number s or a tuple (sh x sw).
padding | 0 | implicit padding to be added. Can be a single number or a tuple.
ceil_mode |  | when True, will use "ceil" instead of "floor" to compute the output shape

#### Expected Shape
       | Shape | Description 
------ | ----- | ------------
 input | [ * , * , *, * ]  | Input is minibatch x channels x iH x iW
output | [ * , * , *, * ]   | Output shape = minibatch x channels x floor((iH  + 2*padH - kH) / sH + 1) x floor((iW  + 2*padW - kW) / sW + 1)
### AvgPool3d

Applies a 3D average pooling over an input signal composed of several input

```python
# pool of square window of size=3, stride=2
m = nn.AvgPool3d(3, stride=2)
# pool of non-square window
m = nn.AvgPool3d((3, 2, 2), stride=(2, 1, 2))
input = autograd.Variable(torch.randn(20, 16, 50,44, 31))
output = m(input)
```

planes.


#### Constructor Arguments

Parameter | Default | Description
--------- | ------- | -----------
kernel_size |  | the size of the window to take a average over. Can be a single number k (for a square kernel of k x k x k) or a tuple (kt x kh x kw)
stride | kernel_size | the stride of the window. Can be a single number s or a tuple (st x sh x sw).

#### Expected Shape
       | Shape | Description 
------ | ----- | ------------
 input | [ * , * , *, *, * ]  | Input is minibatch x channels x iT x iH x iW
output | [ * , * , *, *, * ]   | Output shape = minibatch x channels x floor((iT  + 2*padT - kT) / sT + 1) x floor((iH  + 2*padH - kH) / sH + 1) x floor((iW  + 2*padW - kW) / sW + 1)
### FractionalMaxPool2d

Applies a 2D fractional max pooling over an input signal composed of several input

```python
# pool of square window of size=3, and target output size 13x12
m = nn.FractionalMaxPool2d(3, output_size=(13, 12))
# pool of square window and target output size being half of input image size
m = nn.FractionalMaxPool2d(3, output_ratio=(0.5, 0.5))
input = autograd.Variable(torch.randn(20, 16, 50, 32))
output = m(input)
```

planes.

Fractiona MaxPooling is described in detail in the paper ["Fractional Max-Pooling" by Ben Graham](http://arxiv.org/abs/1412.6071)
The max-pooling operation is applied in kHxkW regions by a stochastic
step size determined by the target output size.
The number of output features is equal to the number of input planes.


#### Constructor Arguments

Parameter | Default | Description
--------- | ------- | -----------
kernel_size |  | the size of the window to take a max over. Can be a single number k (for a square kernel of k x k) or a tuple (kh x kw)
output_size |  | the target output size of the image of the form oH x oW. Can be a tuple (oH, oW) or a single number oH for a square image oH x oH
output_ratio |  | If one wants to have an output size as a ratio of the input size, this option can be given. This has to be a number or tuple in the range (0, 1)
return_indices | False | if True, will return the indices along with the outputs. Useful to pass to nn.MaxUnpool2d .

#### Expected Shape
       | Shape | Description 
------ | ----- | ------------
 input | [ * , * , *, * ]  | Input is minibatch x channels x iH x iW
output | [ * , * , *, * ]   | Output shape = minibatch x channels x floor((iH  + 2*padH - kH) / sH + 1) x floor((iW  + 2*padW - kW) / sW + 1)
### LPPool2d

Applies a 2D power-average pooling over an input signal composed of several input

```python
# power-2 pool of square window of size=3, stride=2
m = nn.LPPool2d(2, 3, stride=2)
# pool of non-square window of power 1.2
m = nn.LPPool2d(1.2, (3, 2), stride=(2, 1))
input = autograd.Variable(torch.randn(20, 16, 50, 32))
output = m(input)
```

planes.
On each window, the function computed is: f(X) = pow(sum(pow(X, p)), 1/p)
At p = infinity, one gets Max Pooling
At p = 1, one gets Average Pooling

#### Constructor Arguments

Parameter | Default | Description
--------- | ------- | -----------
kernel_size |  | the size of the window. Can be a single number k (for a square kernel of k x k) or a tuple (kh x kw)
stride | kernel_size | the stride of the window. Can be a single number s or a tuple (sh x sw).
ceil_mode |  | when True, will use "ceil" instead of "floor" to compute the output shape

#### Expected Shape
       | Shape | Description 
------ | ----- | ------------
 input | [ * , * , *, * ]  | Input is minibatch x channels x iH x iW
output | [ * , * , *, * ]   | Output shape = minibatch x channels x floor((iH  + 2*padH - kH) / sH + 1) x floor((iW  + 2*padW - kW) / sW + 1)
