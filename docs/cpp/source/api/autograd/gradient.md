---
myst:
  html_meta:
    description: PyTorch C++ API for computing gradients — torch::autograd::backward and torch::autograd::grad functions for automatic differentiation.
    keywords: PyTorch, C++, autograd, backward, grad, gradient, automatic differentiation
---

# Gradient Computation

PyTorch provides functions for computing gradients of tensors with respect
to graph leaves.

## Gradient Functions

```{cpp:function} void torch::autograd::backward(const variable_list& tensors, const variable_list& grad_tensors = {}, std::optional<bool> retain_graph = std::nullopt, bool create_graph = false, const variable_list& inputs = {})

Computes the sum of gradients of given tensors with respect to graph leaves.

The graph is differentiated using the chain rule. If any of `tensors`
are non-scalar (i.e. their data has more than one element) and require
gradient, then the Jacobian-vector product would be computed, in this case
the function additionally requires specifying `grad_tensors`. It should be a
sequence of matching length, that contains the "vector" in the
Jacobian-vector product, usually the gradient of the differentiated function
w.r.t. corresponding tensors (`torch::Tensor()` is an acceptable value for
all tensors that don't need gradient tensors).

This function accumulates gradients in the leaves — you might need to zero
them before calling it.

:param tensors: Tensors of which the derivative will be computed.
:param grad_tensors: The "vector" in the Jacobian-vector product, usually
   gradients w.r.t. each element of corresponding tensors.
   `torch::Tensor()` values can be specified for scalar Tensors or ones
   that don't require grad. If a `torch::Tensor()` value would be
   acceptable for all grad_tensors, then this argument is optional.
:param retain_graph: If `false`, the graph used to compute the grad will
   be freed. Note that in nearly all cases setting this option to `true`
   is not needed and often can be worked around in a much more efficient
   way. Defaults to the value of `create_graph`.
:param create_graph: If `true`, graph of the derivative will be
   constructed, allowing to compute higher order derivative products.
   Defaults to `false`.
:param inputs: Inputs w.r.t. which the gradient will be accumulated into
   `at::Tensor::grad`. All other Tensors will be ignored. If not
   provided, the gradient is accumulated into all the leaf Tensors that
   were used to compute `tensors`.
```

```{doxygenfunction} torch::autograd::grad
```

**Example:**

```cpp
#include <torch/torch.h>

auto x = torch::randn({2, 2}, torch::requires_grad());
auto y = x * x;
auto z = y.sum();

// Compute gradients
z.backward();
std::cout << x.grad() << std::endl;

// Or use grad() for specific outputs
auto grads = torch::autograd::grad({z}, {x});
```

## Tensor Gradient Methods

Tensors have built-in methods for gradient computation:

```cpp
// Enable gradient tracking
auto x = torch::randn({2, 2}).requires_grad_(true);

// Check if gradient is required
bool needs_grad = x.requires_grad();

// Access the gradient after backward
auto grad = x.grad();

// Detach from computation graph
auto x_detached = x.detach();
```
