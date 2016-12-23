## Normalization layers
### BatchNorm1d

Applies Batch Normalization over a 2d input that is seen as a mini-batch of 1d inputs

```python
              x - mean(x)
y =  ----------------------------- * gamma + beta
      standard_deviation(x) + eps
```

```python
# With Learnable Parameters
m = nn.BatchNorm1d(100)
# Without Learnable Parameters
m = nn.BatchNorm1d(100, affine=False)
input = autograd.Variable(torch.randn(20, 100))
output = m(input)
```



The mean and standard-deviation are calculated per-dimension over
the mini-batches and gamma and beta are learnable parameter vectors
of size N (where N is the input size).

During training, this layer keeps a running estimate of its computed mean
and variance. The running sum is kept with a default momentum of 0.1
During evaluation, this running mean/variance is used for normalization.


#### Constructor Arguments

Parameter | Default | Description
--------- | ------- | -----------
num_features |  | the size of each 1D input in the mini-batch
eps | 1e-5 | a value added to the denominator for numerical stability.
momentum | 0.1 | the value used for the running_mean and running_var computation.
affine |  | a boolean value that when set to true, gives the layer learnable affine parameters.

#### Expected Shape
       | Shape | Description 
------ | ----- | ------------
 input | [ * , num_features ]  | 2D Tensor of nBatches x num_features
output | Same  | Output has the same shape as input

#### Returns
    a normalized tensor in the batch dimension
### BatchNorm2d

Applies Batch Normalization over a 4d input that is seen as a mini-batch of 3d inputs

```python
              x - mean(x)
y =  ----------------------------- * gamma + beta
      standard_deviation(x) + eps
```

```python
# With Learnable Parameters
m = nn.BatchNorm2d(100)
# Without Learnable Parameters
m = nn.BatchNorm2d(100, affine=False)
input = autograd.Variable(torch.randn(20, 100, 35, 45))
output = m(input)
```



The mean and standard-deviation are calculated per-dimension over
the mini-batches and gamma and beta are learnable parameter vectors
of size N (where N is the input size).

During training, this layer keeps a running estimate of its computed mean
and variance. The running sum is kept with a default momentum of 0.1
During evaluation, this running mean/variance is used for normalization.


#### Constructor Arguments

Parameter | Default | Description
--------- | ------- | -----------
num_features |  | num_features from an expected input of size batch_size x num_features x height x width
eps | 1e-5 | a value added to the denominator for numerical stability.
momentum | 0.1 | the value used for the running_mean and running_var computation.
affine |  | a boolean value that when set to true, gives the layer learnable affine parameters.

#### Expected Shape
       | Shape | Description 
------ | ----- | ------------
 input | [ * , num_features , *, * ]  | 4D Tensor of batch_size x num_features x height x width
output | Same  | Output has the same shape as input

#### Returns
    a normalized tensor in the batch dimension
### BatchNorm3d

Applies Batch Normalization over a 5d input that is seen as a mini-batch of 4d inputs

```python
              x - mean(x)
y =  ----------------------------- * gamma + beta
      standard_deviation(x) + eps
```

```python
# With Learnable Parameters
m = nn.BatchNorm3d(100)
# Without Learnable Parameters
m = nn.BatchNorm3d(100, affine=False)
input = autograd.Variable(torch.randn(20, 100, 35, 45, 10))
output = m(input)
```



The mean and standard-deviation are calculated per-dimension over
the mini-batches and gamma and beta are learnable parameter vectors
of size N (where N is the input size).

During training, this layer keeps a running estimate of its computed mean
and variance. The running sum is kept with a default momentum of 0.1
During evaluation, this running mean/variance is used for normalization.


#### Constructor Arguments

Parameter | Default | Description
--------- | ------- | -----------
num_features |  | num_features from an expected input of size batch_size x num_features x height x width
eps | 1e-5 | a value added to the denominator for numerical stability.
momentum | 0.1 | the value used for the running_mean and running_var computation.
affine |  | a boolean value that when set to true, gives the layer learnable affine parameters.

#### Expected Shape
       | Shape | Description 
------ | ----- | ------------
 input | [ * , num_features , * , * , * ]  | 5D Tensor of batch_size x num_features x depth x height x width
output | Same  | Output has the same shape as input

#### Returns
    a normalized tensor in the batch dimension
