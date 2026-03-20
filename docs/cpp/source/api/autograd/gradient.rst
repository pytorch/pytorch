Gradient Computation
====================

PyTorch provides functions for computing gradients of tensors with respect
to graph leaves.

Gradient Functions
------------------

.. doxygenfunction:: torch::autograd::grad

.. cpp:function:: void torch::autograd::backward(const variable_list& tensors, const variable_list& grad_tensors = {}, std::optional<bool> retain_graph = std::nullopt, bool create_graph = false)

   Computes the sum of gradients of given tensors w.r.t. graph leaves.

**Example:**

.. code-block:: cpp

   #include <torch/torch.h>

   auto x = torch::randn({2, 2}, torch::requires_grad());
   auto y = x * x;
   auto z = y.sum();

   // Compute gradients
   z.backward();
   std::cout << x.grad() << std::endl;

   // Or use grad() for specific outputs
   auto grads = torch::autograd::grad({z}, {x});

Tensor Gradient Methods
-----------------------

Tensors have built-in methods for gradient computation:

.. code-block:: cpp

   // Enable gradient tracking
   auto x = torch::randn({2, 2}).requires_grad_(true);

   // Check if gradient is required
   bool needs_grad = x.requires_grad();

   // Access the gradient after backward
   auto grad = x.grad();

   // Detach from computation graph
   auto x_detached = x.detach();
