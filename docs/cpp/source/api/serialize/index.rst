Serialization (torch::serialize)
=================================

The ``torch::serialize`` namespace provides utilities for saving and loading
model weights, tensors, and optimizer state. This enables checkpointing during
training and deployment of trained models.

**When to use torch::serialize:**

- When saving trained models to disk
- When implementing training checkpoints
- When loading pre-trained weights
- When transferring models between C++ and Python

**Basic usage:**

.. code-block:: cpp

   #include <torch/torch.h>

   // Save a model
   auto model = std::make_shared<Net>();
   // ... train the model ...
   torch::save(model, "model.pt");

   // Load a model
   auto loaded_model = std::make_shared<Net>();
   torch::load(loaded_model, "model.pt");

   // Save and load tensors
   torch::Tensor tensor = torch::randn({2, 3});
   torch::save(tensor, "tensor.pt");

   torch::Tensor loaded_tensor;
   torch::load(loaded_tensor, "tensor.pt");

Header Files
------------

- ``torch/csrc/api/include/torch/serialize.h`` - Main serialization header
- ``torch/csrc/api/include/torch/serialize/archive.h`` - Archive classes
- ``torch/csrc/api/include/torch/serialize/input-archive.h`` - Input archive
- ``torch/csrc/api/include/torch/serialize/output-archive.h`` - Output archive

Serialization Categories
------------------------

.. toctree::
   :maxdepth: 1

   save_load
   archives
   checkpoints
