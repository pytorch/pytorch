Neural Network Modules (torch::nn)
===================================

The ``torch::nn`` namespace provides neural network building blocks that mirror
Python's ``torch.nn`` module. It uses a PIMPL (Pointer to Implementation) pattern
where user-facing classes like ``Conv2d`` wrap internal ``Conv2dImpl`` classes.

**When to use torch::nn:**

- Building neural network models in C++
- Creating custom layers and modules
- Porting Python models to C++ for production inference
- Training models entirely in C++

**Basic usage:**

.. code-block:: cpp

   #include <torch/torch.h>

   // Define a simple model
   struct Net : torch::nn::Module {
       torch::nn::Conv2d conv1{nullptr};
       torch::nn::Linear fc1{nullptr};

       Net() {
           conv1 = register_module("conv1", torch::nn::Conv2d(
               torch::nn::Conv2dOptions(1, 32, 3).stride(1).padding(1)));
           fc1 = register_module("fc1", torch::nn::Linear(32 * 28 * 28, 10));
       }

       torch::Tensor forward(torch::Tensor x) {
           x = torch::relu(conv1->forward(x));
           x = x.view({-1, 32 * 28 * 28});
           return fc1->forward(x);
       }
   };

   // Create and use the model
   auto model = std::make_shared<Net>();
   auto input = torch::randn({1, 1, 28, 28});
   auto output = model->forward(input);

Header Files
------------

- ``torch/nn.h`` - Main neural network header (includes all modules)
- ``torch/nn/module.h`` - Base Module class
- ``torch/nn/modules.h`` - All module implementations
- ``torch/nn/options.h`` - Options structs for modules
- ``torch/nn/functional.h`` - Functional API

Module Base Class
-----------------

All neural network modules inherit from ``torch::nn::Module``.

.. doxygenclass:: torch::nn::Module
   :members:
   :undoc-members:

Module Categories
-----------------

.. toctree::
   :maxdepth: 1

   containers
   convolution
   pooling
   linear
   activation
   normalization
   dropout
   embedding
   recurrent
   transformer
   loss
   utilities
