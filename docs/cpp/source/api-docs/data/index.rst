Data Loading (torch::data)
==========================

The ``torch::data`` namespace provides utilities for loading and processing
datasets during training. It includes dataset abstractions, data loaders for
batching and shuffling, samplers for controlling data access patterns, and
transforms for data augmentation.

**When to use torch::data:**

- When loading training data in batches
- When you need parallel data loading with multiple workers
- When implementing custom datasets or transforms

**Components overview:**

- **Dataset**: Defines how to access individual samples (implement ``get()`` and ``size()``)
- **DataLoader**: Batches samples and optionally shuffles/parallelizes loading
- **Sampler**: Controls the order in which samples are accessed
- **Transform**: Applies preprocessing (normalization, augmentation) to samples

**Basic usage:**

.. code-block:: cpp

   #include <torch/torch.h>

   // Load built-in dataset
   auto dataset = torch::data::datasets::MNIST("./data")
       .map(torch::data::transforms::Normalize<>(0.1307, 0.3081))
       .map(torch::data::transforms::Stack<>());

   // Create data loader with batching and shuffling
   auto data_loader = torch::data::make_data_loader(
       std::move(dataset),
       torch::data::DataLoaderOptions().batch_size(64).workers(4));

   // Iterate over batches
   for (auto& batch : *data_loader) {
       auto images = batch.data;   // Shape: [64, 1, 28, 28]
       auto labels = batch.target; // Shape: [64]
   }

Header Files
------------

- ``torch/csrc/api/include/torch/data.h`` - Main data header
- ``torch/csrc/api/include/torch/data/dataloader.h`` - DataLoader
- ``torch/csrc/api/include/torch/data/datasets.h`` - Dataset classes
- ``torch/csrc/api/include/torch/data/samplers.h`` - Samplers

Module Categories
-----------------

.. toctree::
   :maxdepth: 1

   datasets
   dataloader
   samplers
   transforms
