## Dropout layers
### Dropout

Randomly zeroes some of the elements of the input tensor.

```python
m = nn.Dropout(p=0.2)
input = autograd.Variable(torch.randn(20, 16))
output = m(input)
```

The elements to zero are randomized on every forward call.


#### Constructor Arguments

Parameter | Default | Description
--------- | ------- | -----------
p | 0.5 | probability of an element to be zeroed.
inplace | false | If set to True, will do this operation in-place.

#### Expected Shape
       | Shape | Description 
------ | ----- | ------------
 input | Any  | Input can be of any shape
output | Same   | Output is of the same shape as input
### Dropout2d

Randomly zeroes whole channels of the input tensor.

```python
m = nn.Dropout2d(p=0.2)
input = autograd.Variable(torch.randn(20, 16, 32, 32))
output = m(input)
```

The input is 4D (batch x channels, height, width) and each channel
is of size (1, height, width).
The channels to zero are randomized on every forward call.
Usually the input comes from Conv2d modules.

As described in the paper &quot;Efficient Object Localization Using Convolutional
Networks&quot; (http:arxiv.org/abs/1411.4280), if adjacent pixels within
feature maps are strongly correlated (as is normally the case in early
convolution layers) then iid dropout will not regularize the activations
and will otherwise just result in an effective learning rate decrease.
In this case, nn.Dropout2d will help promote independence between
feature maps and should be used instead.


#### Constructor Arguments

Parameter | Default | Description
--------- | ------- | -----------
p | 0.5 | probability of an element to be zeroed.
inplace | false | If set to True, will do this operation in-place.

#### Expected Shape
       | Shape | Description 
------ | ----- | ------------
 input | [*, *, *, *]  | Input can be of any sizes of 4D shape
output | Same   | Output is of the same shape as input
### Dropout3d

Randomly zeroes whole channels of the input tensor.

```python
m = nn.Dropout3d(p=0.2)
input = autograd.Variable(torch.randn(20, 16, 4, 32, 32))
output = m(input)
```

The input is 5D (batch x channels, depth, height, width) and each channel
is of size (1, depth, height, width).
The channels to zero are randomized on every forward call.
Usually the input comes from Conv3d modules.


#### Constructor Arguments

Parameter | Default | Description
--------- | ------- | -----------
p | 0.5 | probability of an element to be zeroed.
inplace | false | If set to True, will do this operation in-place.

#### Expected Shape
       | Shape | Description 
------ | ----- | ------------
 input | [*, *, *, *, *]  | Input can be of any sizes of 5D shape
output | Same   | Output is of the same shape as input
