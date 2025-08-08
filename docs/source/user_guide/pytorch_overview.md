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

# Beyond Basic Operations

PyTorch extends far beyond basic arithmetic operations to provide a complete ecosystem for machine learning development:
* {mod}`torch.nn` module: Offers pre-built neural network layers, activation functions, and loss functions for constructing complex architectures
* {mod}`torch.utils.data`: Provides efficient data loading utilities with support for batching, shuffling, and parallel data loading
* {mod}`torch.optim`: Contains a comprehensive suite of optimization algorithms including SGD, Adam, RMSprop, and many others
* {func}`torch.compile`: Allows for compiling PyTorch models to improve execution speed and efficiency
* {func}`torch.export`: Enables exporting models for deployment in various environments.
* Distributed training: Facilitates training across multiple GPUs and nodes, making it suitable for large-scale machine learning tasks

```{seealso}
* {ref}`torch.compiler_overview`
* {ref}`torch.export`
* [Learn the basics](https://docs.pytorch.org/tutorials/beginner/basics/intro.html)
```
