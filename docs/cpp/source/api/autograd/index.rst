Autograd: Automatic Differentiation
====================================

PyTorch's autograd system provides automatic differentiation for all operations
on tensors. It records operations on tensors to build a computational graph,
then computes gradients automatically via backpropagation.

**When to use Autograd:**

- When training neural networks (gradients are computed automatically)
- When implementing custom backward passes for specialized operations
- When you need fine-grained control over gradient computation

**Basic usage:**

.. code-block:: cpp

   #include <torch/torch.h>

   // Create tensor with gradient tracking
   auto x = torch::randn({2, 2}, torch::requires_grad());
   auto y = x * x;
   auto z = y.sum();

   // Compute gradients via backpropagation
   z.backward();
   std::cout << x.grad() << std::endl;  // dz/dx = 2x

   // Disable gradient tracking for inference
   {
       torch::NoGradGuard no_grad;
       auto result = model->forward(input);  // No gradients computed
   }

Header Files
------------

- ``torch/csrc/autograd/autograd.h`` - High-level autograd API
- ``torch/csrc/autograd/function.h`` - Custom autograd functions
- ``torch/csrc/autograd/grad_mode.h`` - Gradient computation modes
- ``torch/csrc/api/include/torch/autograd.h`` - C++ Frontend autograd

Autograd Categories
-------------------

.. toctree::
   :maxdepth: 1

   gradient
   custom_functions
   modes
