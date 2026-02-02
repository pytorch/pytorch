Optimizers (torch::optim)
=========================

The ``torch::optim`` namespace provides optimization algorithms for
training neural networks. These optimizers update model parameters based
on computed gradients to minimize the loss function.

**When to use torch::optim:**

- When training neural networks with gradient descent
- When you need different optimization strategies (SGD, Adam, etc.)
- When implementing learning rate schedules

**Basic usage:**

.. code-block:: cpp

   #include <torch/torch.h>

   // Create model and optimizer
   auto model = std::make_shared<Net>();
   auto optimizer = torch::optim::Adam(
       model->parameters(),
       torch::optim::AdamOptions(1e-3));

   // Training loop
   for (auto& batch : *data_loader) {
       optimizer.zero_grad();                    // Clear gradients
       auto loss = loss_fn(model->forward(batch.data), batch.target);
       loss.backward();                          // Compute gradients
       optimizer.step();                         // Update parameters
   }

Header Files
------------

- ``torch/csrc/api/include/torch/optim.h`` - Main optim header
- ``torch/csrc/api/include/torch/optim/optimizer.h`` - Optimizer base class
- ``torch/csrc/api/include/torch/optim/sgd.h`` - SGD optimizer
- ``torch/csrc/api/include/torch/optim/adam.h`` - Adam optimizer

Optimizer Base Class
--------------------

All optimizers inherit from the ``Optimizer`` base class, which provides common
functionality for parameter updates, gradient zeroing, and state management.

.. doxygenclass:: torch::optim::Optimizer
   :members:
   :undoc-members:

Choosing an Optimizer
---------------------

Selecting the right optimizer depends on your model architecture, dataset, and
training requirements:

.. list-table::
   :widths: 20 40 40
   :header-rows: 1

   * - Optimizer
     - Best For
     - Trade-offs
   * - **SGD + Momentum**
     - CNNs, well-understood problems, when you can tune hyperparameters
     - Requires careful learning rate tuning; often achieves best final accuracy
   * - **Adam/AdamW**
     - General-purpose, transformers, quick prototyping
     - Works well out-of-the-box; AdamW preferred with weight decay
   * - **RMSprop**
     - RNNs, non-stationary objectives
     - Good for recurrent architectures; handles varying gradient scales
   * - **Adagrad**
     - Sparse data (NLP, embeddings)
     - Learning rate decreases over time; good for infrequent features
   * - **LBFGS**
     - Small models, fine-tuning, convex problems
     - Memory-intensive; requires closure function

Optimizer Categories
--------------------

.. toctree::
   :maxdepth: 1

   gradient_descent
   adaptive
   second_order
   schedulers
