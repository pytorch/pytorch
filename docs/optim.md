# torch.optim

The Optim package in Torch is targeted for one to optimize their neural networks
using a wide variety of optimization methods such as SGD, Adam etc.

Currently, the following optimization methods are supported, typically with
options such as weight decay and other bells and whistles.

- SGD          `(params, lr=required, momentum=0, dampening=0)`
- AdaDelta     `(params, rho=0.9, eps=1e-6, weight_decay=0)`
- Adagrad      `(params, lr=1e-2, lr_decay=0, weight_decay=0)`
- Adam         `(params, lr=1e-2, betas=(0.9, 0.999), epsilon=1e-8, weight_decay=0)`
- AdaMax       `(params, lr=1e-2, betas=(0.9, 0.999), eps=1e-38, weight_decay=0)`
- Averaged SGD `(params, lr=1e-2, lambd=1e-4, alpha=0.75, t0=1e6, weight_decay=0)`
- RProp        `(params, lr=1e-2, etas=(0.5, 1.2), step_sizes=(1e-6, 50))`
- RMSProp      `(params, lr=1e-2, alpha=0.99, eps=1e-8, weight_decay=0)`


The usage of the Optim package itself is as follows.

1. Construct an optimizer
2. Use `optimizer.step(...)` to optimize.
   - Call `optimizer.zero_grad()` to zero out the gradient buffers when appropriate

## 1. Constructing the optimizer

One first constructs an `Optimizer` object by giving it a list of parameters
to optimize, as well as the optimizer options,such as learning rate, weight decay, etc.

Examples:

`optimizer = optim.SGD(model.parameters(), lr = 0.01, momentum=0.9)`

`optimizer = optim.Adam([var1, var2], lr = 0.0001)`

### Per-parameter options

In a more advanced usage, one can specify per-layer options by passing each parameter group along with it's custom options.

**__Any parameter group that does not have an attribute defined will use the default attributes.__**

This is very useful when one wants to specify per-layer learning rates for example.

Example:

`optim.SGD([{'params': model1.parameters()}, {'params': model2.parameters(), 'lr': 1e-3}, lr=1e-2, momentum=0.9)`

`model1`'s parameters will use the default learning rate of `1e-2` and momentum of `0.9`
`model2`'s parameters will use a learning rate of `1e-3`, and the default momentum of `0.9`

Then, you can use the optimizer by calling `optimizer.zero_grad()` and `optimizer.step(...)`. Read the next sections.

## 2. Taking an optimization step using `Optimizer.step(...)`

The step function has the following two signatures:

### a. `Optimizer.step(closure)`

The `step` function takes a user-defined closure that computes f(x) and returns the loss.

The closure needs to do the following:
- Optimizer.zero_grad()
- Compute the loss
- Call loss.backward()
- return the loss

Example 1: training a neural network

```python
# Example 1: training a neural network with optimizer.step(closure)
net = MNISTNet()
criterion = ClassNLLLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001)

for data in data_batches:
    input, target = data
	def closure():
	    optimizer.zero_grad()
	    output = net(input)
		loss = criterion(output, target)
		loss.backward()
		return loss
	optimizer.step(closure)
```

Notes: Why is this required? Why cant we simply have the optimizer take the parameters and grads?
       Some optimization algorithms such as Conjugate Gradient and LBFGS need to evaluate their function
	   multiple times. For such optimization methods, the function (i.e. the closure) has to be defined.
      

### b. `Optimizer.step()`

This is a simplified usage that supports most, but not all optimization algorithms. For example, it does not support LBFGS or Conjugate Gradient.

The usage for this is to simply call the function after the backward() is called on your model.

Example 2: training a neural network

```python
# Example 2: training a neural network with optimizer.step()
net = MNISTNet()
criterion = ClassNLLLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001)

for data in data_batches:
    input, target = data
	optimizer.zero_grad()
	output = net(input)
	loss = criterion(output, target)
	loss.backward()
	optimizer.step()
```


