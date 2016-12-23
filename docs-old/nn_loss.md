## Loss functions
### L1Loss

Creates a criterion that measures the mean absolute value of the 

element-wise difference between input `x` and target `y`:

loss(x, y)  = 1/n \sum |x_i - y_i|

`x` and `y` arbitrary shapes with a total of `n` elements each
the sum operation still operates over all the elements, and divides by `n`.

The division by `n` can be avoided if one sets the internal 
variable `sizeAverage` to `False`
### MSELoss

Creates a criterion that measures the mean squared error between 

`n` elements in the input `x` and target `y`:
    loss(x, y) = 1/n \sum |x_i - y_i|^2
`x` and `y` arbitrary shapes with a total of `n` elements each
the sum operation still operates over all the elements, and divides by `n`.

The division by `n` can be avoided if one sets the internal variable 
`sizeAverage` to `False`
By default, the losses are averaged over observations for each minibatch. 
However, if the field `sizeAverage = False`, the losses are instead summed.
### CrossEntropyLoss

This criterion combines `LogSoftMax` and `ClassNLLLoss` in one single class.


It is useful when training a classification problem with `n` classes.
If provided, the optional argument `weights` should be a 1D `Tensor` 
assigning weight to each of the classes. 
This is particularly useful when you have an unbalanced training set.

The `input` is expected to contain scores for each class: 
      `input` has to be a 2D `Tensor` of size `batch x n`.
This criterion expects a class index (0 to nClasses-1) as the 
`target` for each value of a 1D tensor of size `n`

The loss can be described as:

loss(x, class) = -log(exp(x[class]) / (\sum_j exp(x[j])))
               = -x[class] + log(\sum_j exp(x[j]))

or in the case of the `weights` argument being specified:

loss(x, class) = weights[class] * (-x[class] + log(\sum_j exp(x[j])))

The losses are averaged across observations for each minibatch.
### NLLLoss

The negative log likelihood loss. It is useful to train a classication problem with n classes

```python
m = nn.LogSoftmax()
loss = nn.NLLLoss()
# input is of size nBatch x nClasses = 3 x 5
input = autograd.Variable(torch.randn(3, 5))
# each element in target has to have 0 <= value < nclasses 
target = autograd.Variable(torch.LongTensor([1, 0, 4]))
output = loss(m(input), target)
output.backward()
```


If provided, the optional argument `weights` should be a 1D Tensor assigning
weight to each of the classes.
This is particularly useful when you have an unbalanced training set.

The input given through a forward call is expected to contain log-probabilities
of each class: input has to be a 2D Tensor of size minibatch x n
Obtaining log-probabilities in a neural network is easily achieved by
adding a  `LogSoftmax`  layer in the last layer.
You may use `CrossEntropyLoss`  instead, if you prefer not to
add an extra layer.

The target that this loss expects is a class index (1 to the number of class)

The loss can be described as:
    loss(x, class) = -x[class]

or in the case of the weights argument it is specified as follows:
    loss(x, class) = -weights[class] * x[class]


#### Constructor Arguments

Parameter | Default | Description
--------- | ------- | -----------
weight | None | a manual rescaling weight given to each class. If given, has to be a Tensor of size "nclasses".
size_average | True | By default, the losses are averaged over observations for each minibatch. However, if the field sizeAverage is set to False, the losses are instead summed for each minibatch.
Target Shape: [ * ] : Targets of size [minibatch], each value has to be 1 <= targets[i] <= nClasses

#### Members

Parameter | Description
--------- | -----------
weight | the class-weights given as input to the constructor
### NLLLoss2d

This is negative log likehood loss, but for image inputs. It computes NLL loss per-pixel.

```python
m = nn.Conv2d(16, 32, (3, 3)).float()
loss = nn.NLLLoss2d()
# input is of size nBatch x nClasses x height x width
input = autograd.Variable(torch.randn(3, 16, 10, 10))
# each element in target has to have 0 <= value < nclasses
target = autograd.Variable(torch.LongTensor(3, 8, 8).random_(0, 4))
output = loss(m(input), target)
output.backward()
```

This loss does not support per-class weights

#### Constructor Arguments

Parameter | Default | Description
--------- | ------- | -----------
size_average | True | By default, the losses are averaged over observations for each minibatch. However, if the field sizeAverage is set to False, the losses are instead summed for each minibatch.
Target Shape: [ * , *, *] : Targets of size minibatch x height x width, each value has to be 1 <= targets[i] <= nClasses
### KLDivLoss

The [Kullback-Leibler divergence](http://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence) Loss

KL divergence is a useful distance measure for continuous distributions 
and is often useful when performing direct regression over the space of
(discretely sampled) continuous output distributions.
As with ClassNLLLoss, the `input` given is expected to contain 
_log-probabilities_, however unlike ClassNLLLoss, `input` is not 
restricted to a 2D Tensor, because the criterion is applied element-wise.

This criterion expects a `target` `Tensor` of the same size as the 
`input` `Tensor`.

The loss can be described as:
    loss(x, target) = 1/n \sum(target_i * (log(target_i) - x_i))

By default, the losses are averaged for each minibatch over observations 
*as well as* over dimensions. However, if the field 
`sizeAverage` is set to `False`, the losses are instead summed.
### BCELoss

Creates a criterion that measures the Binary Cross Entropy 

between the target and the output:
    loss(o, t) = - 1/n sum_i (t[i] * log(o[i]) + (1 - t[i]) * log(1 - o[i]))

or in the case of the weights argument being specified:
    loss(o, t) = - 1/n sum_i weights[i] * (t[i] * log(o[i]) + (1 - t[i]) * log(1 - o[i]))

This is used for measuring the error of a reconstruction in for example 
an auto-encoder. Note that the targets `t[i]` should be numbers between 0 and 1, 
for instance, the output of an `nn.Sigmoid` layer.

By default, the losses are averaged for each minibatch over observations 
*as well as* over dimensions. However, if the field `sizeAverage` is set 
to `False`, the losses are instead summed.
### MarginRankingLoss

Creates a criterion that measures the loss given  

inputs `x1`, `x2`, two 1D min-batch `Tensor`s, 
and a label 1D mini-batch tensor `y` with values (`1` or `-1`).

If `y == 1` then it assumed the first input should be ranked higher 
(have a larger value) than the second input, and vice-versa for `y == -1`.

The loss function for each sample in the mini-batch is:

    loss(x, y) = max(0, -y * (x1 - x2) + margin)

if the internal variable `sizeAverage = True`, 
the loss function averages the loss over the batch samples; 
if `sizeAverage = False`, then the loss function sums over the batch samples. 
By default, `sizeAverage` equals to `True`.
### HingeEmbeddingLoss

Measures the loss given an input `x` which is a 2D mini-batch tensor

and a labels `y`, a 1D tensor containg values (`1` or `-1`).
This is usually used for measuring whether two inputs are similar or dissimilar, 
e.g. using the L1 pairwise distance, and is typically used for learning 
nonlinear embeddings or semi-supervised learning.

                     { x_i,                  if y_i ==  1
    loss(x, y) = 1/n {
                     { max(0, margin - x_i), if y_i == -1

`x` and `y` arbitrary shapes with a total of `n` elements each
the sum operation still operates over all the elements, and divides by `n`.
(the division by `n` can be avoided if one sets the internal variable `sizeAverage=False`). 
The `margin` has a default value of `1`, or can be set in the constructor.
### MultiLabelMarginLoss

Creates a criterion that optimizes a multi-class multi-classification 

hinge loss (margin-based loss) between input `x`  (a 2D mini-batch `Tensor`) and 
output `y` (which is a 2D `Tensor` of target class indices).
For each sample in the mini-batch:

    loss(x, y) = sum_ij(max(0, 1 - (x[y[j]] - x[i]))) / x:size(1)

where `i == 0` to `x.size(0)`, `j == 0` to `y.size(0)`, 
      `y[j] != 0`, and `i != y[j]` for all `i` and `j`.

`y` and `x` must have the same size.
The criterion only considers the first non zero `y[j]` targets.
This allows for different samples to have variable amounts of target classes
### SmoothL1Loss

Creates a criterion that uses a squared term if the absolute 

element-wise error falls below 1 and an L1 term otherwise. 
It is less sensitive to outliers than the `MSELoss` and in some cases 
prevents exploding gradients (e.g. see "Fast R-CNN" paper by Ross Girshick).
Also known as the Huber loss.

                          { 0.5 * (x_i - y_i)^2, if |x_i - y_i| < 1
    loss(x, y) = 1/n \sum {
                          { |x_i - y_i| - 0.5,   otherwise

`x` and `y` arbitrary shapes with a total of `n` elements each
the sum operation still operates over all the elements, and divides by `n`.

The division by `n` can be avoided if one sets the internal variable 
`sizeAverage` to `False`
### SoftMarginLoss

Creates a criterion that optimizes a two-class classification 

logistic loss between input `x` (a 2D mini-batch `Tensor`) and 
target `y` (which is a tensor containing either `1`s or `-1`s).

    loss(x, y) = sum_i (log(1 + exp(-y[i]*x[i]))) / x:nElement()

The normalization by the number of elements in the input can be disabled by
setting `self.sizeAverage` to `False`.
### MultiLabelSoftMarginLoss

Creates a criterion that optimizes a multi-label one-versus-all 

loss based on max-entropy, between input `x`  (a 2D mini-batch `Tensor`) and 
target `y` (a binary 2D `Tensor`). For each sample in the minibatch:

   loss(x, y) = - sum_i (y[i] log( exp(x[i]) / (1 + exp(x[i]))) 
                         + (1-y[i]) log(1/(1+exp(x[i])))) / x:nElement()

where `i == 0` to `x.nElement()-1`, `y[i]  in {0,1}`.
`y` and `x` must have the same size.
### CosineEmbeddingLoss

Creates a criterion that measures the loss given  an input tensors x1, x2 

and a `Tensor` label `y` with values 1 or -1.
This is used for measuring whether two inputs are similar or dissimilar, 
using the cosine distance, and is typically used for learning nonlinear 
embeddings or semi-supervised learning.

`margin` should be a number from `-1` to `1`, `0` to `0.5` is suggested.
If `margin` is missing, the default value is `0`.

The loss function for each sample is:

                 { 1 - cos(x1, x2),              if y ==  1
    loss(x, y) = {
                 { max(0, cos(x1, x2) - margin), if y == -1

If the internal variable `sizeAverage` is equal to `True`, 
the loss function averages the loss over the batch samples; 
if `sizeAverage` is `False`, then the loss function sums over the 
batch samples. By default, `sizeAverage = True`.
### MultiMarginLoss

Creates a criterion that optimizes a multi-class classification hinge loss 

(margin-based loss) between input `x` (a 2D mini-batch `Tensor`) and 
output `y` (which is a 1D tensor of target class indices, `0` <= `y` <= `x.size(1)`):

For each mini-batch sample:
    loss(x, y) = sum_i(max(0, (margin - x[y] + x[i]))^p) / x.size(0)
                 where `i == 0` to `x.size(0)` and `i != y`.

Optionally, you can give non-equal weighting on the classes by passing 
a 1D `weights` tensor into the constructor.

The loss function then becomes:
    loss(x, y) = sum_i(max(0, w[y] * (margin - x[y] - x[i]))^p) / x.size(0)

By default, the losses are averaged over observations for each minibatch. 
However, if the field `sizeAverage` is set to `False`, 
the losses are instead summed.
