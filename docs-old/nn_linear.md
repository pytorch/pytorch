## Linear layers
### Linear

Applies a linear transformation to the incoming data, y = Ax + b

```python
m = nn.Linear(20, 30)
input = autograd.Variable(torch.randn(128, 20))
output = m(input)
print(output.size())
```

The input is a 2D mini-batch of samples, each of size in_features
The output will be a 2D Tensor of size mini-batch x out_features


#### Constructor Arguments

Parameter | Default | Description
--------- | ------- | -----------
in_features |  | size of each input sample
out_features |  | size of each output sample
bias | True | If set to False, the layer will not learn an additive bias.

#### Expected Shape
       | Shape | Description 
------ | ----- | ------------
 input | [*, in_features]  | Input can be of shape minibatch x in_features
output | [*, out_features]   | Output is of shape minibatch x out_features

#### Members

Parameter | Description
--------- | -----------
weight | the learnable weights of the module of shape (out_features x in_features)
bias | the learnable bias of the module of shape (out_features)
