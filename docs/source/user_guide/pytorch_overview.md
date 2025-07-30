---
file_format: mystnb
kernelspec:
  name: python3
mystnb:
  execution_timeout: 30
  merge_streams: True
---

```{code-cell}
:tags: [remove-cell]
import torch
```

(what_is_pytorch)=

# What is PyTorch?

PyTorch, or torch, is an open-source machine learning library written in Python that
provides a platform for deep learning. It features a dynamic computational graph
that allows for flexible model building and debugging.

Here is a quick PyTorch example

```{code-cell}
x = torch.tensor([1.0, 2.0], requires_grad=True)
y = x**2 + 3*x
y.backward(torch.ones_like(x))
print(y)
print(x.grad)
```

The example above shows tensor creation, computation, and automatic gradient calculation - core features that enable neural network training through backpropagation. The code above:

1. Imports `torch`
2. Creates a tensor with gradient tracking enabled
3. Defines a computation using tensor operations
4. Calculates gradients of `y` with respect to `x`
5. After this, `x.grad` will contain the derivatives: `d(x^2 + 3x)/dx = 2x + 3`

At its core, PyTorch uses tensors (multidimensional arrays) that can run on GPUs
for accelerated computation. For example, the gradient descent optimization
process can be represented as:

```{math}
\theta_{\text{new}} = \theta_{\text{old}} - \alpha \nabla_{\theta} J(\theta)
```

* `θ_new` is the updated parameter
* `θ_old` is the current parameter
* `α` is the learning rate
* `∇_θ J(θ)` is the gradient of the cost function with respect to `θ`

PyTorch's autograd engine automatically computes the gradients needed for neural network training.

For a quick tutorial on PyTorch, see the [Learn the Basics tutorial](https://pytorch.org/tutorials/beginner/basics/intro.html).

PyTorch can do so much more beyond the basic alarithmetic operations. It supports complex neural network architectures through
its {mod}`torch.nn` module, provides efficient data loading utilities with 
{mod}`torch.utils.data`, and offers a suite of optimization algorithms in
{mod}`torch.optim`. PyTorch also facilitates distributed training across multiple GPUs and
nodes, making it suitable for large-scale machine learning tasks. For performance optimization,
PyTorch provides {func}`torch.compile`, which allows for compiling PyTorch models to improve execution
speed and efficiency, and {func}`torch.export`, which enables exporting models for deployment in various environments.
compiling PyTorch models to improve execution speed and efficiency.

## GPU Acceleration

PyTorch can run on GPUs for accelerated computation and training.
Herer is a quick example of using GPU acceleration:

```python
# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Run on GPU
torch.randn(1000, 1000, device=device)

# Operations run on GPU
y = x @ x  # Matrix multiplication
```

## Optimizing with `torch.compile`

PyTorch offers performance optimization through compilation
with `torch.compile`. Here is a quick example:

```{code-cell}
@torch.compile
def compute(x):
    return x**2 + 3*x

x = torch.tensor([1.0, 2.0], requires_grad=True)
y = compute(x)
y.backward(torch.ones_like(x))
print(y)
print(x.grad)
```

Learn more about torch.compile in the {ref}`torch.compiler_overview` section.

```{seealso}
* {ref}`torch.compiler_overview`
* {ref}`torch.export`
* [Learn the basics](https://docs.pytorch.org/tutorials/beginner/basics/intro.html)
```
