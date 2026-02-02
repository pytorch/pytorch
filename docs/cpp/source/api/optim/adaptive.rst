Adaptive Learning Rate Optimizers
==================================

These optimizers automatically adapt the learning rate for each parameter based
on historical gradient information. They typically require less hyperparameter
tuning and work well across a wide range of problems.

Adam (Adaptive Moment Estimation)
---------------------------------

Adam combines the benefits of RMSprop and momentum, maintaining per-parameter
adaptive learning rates. It's an excellent default choice for most deep learning
tasks, especially when you want fast convergence with minimal tuning.

**When to use:**

- Transformers and attention-based models
- Quick prototyping and experimentation
- When you don't have time for extensive hyperparameter search
- General-purpose deep learning

**Key parameters:**

- ``lr``: Learning rate (typical: 1e-3 to 1e-4)
- ``betas``: Coefficients for running averages (default: {0.9, 0.999})
- ``eps``: Numerical stability term (default: 1e-8)
- ``weight_decay``: L2 regularization (note: applied differently than in SGD)

.. doxygenclass:: torch::optim::Adam
   :members:
   :undoc-members:

**Example:**

.. code-block:: cpp

   // Standard Adam configuration
   auto optimizer = torch::optim::Adam(
       model->parameters(),
       torch::optim::AdamOptions(1e-3)        // learning rate
           .betas({0.9, 0.999})               // momentum terms
           .eps(1e-8)                         // numerical stability
           .weight_decay(0));                 // L2 penalty

   // For transformers, lower learning rate with warmup
   auto optimizer = torch::optim::Adam(
       model->parameters(),
       torch::optim::AdamOptions(1e-4)
           .betas({0.9, 0.98}));              // β2=0.98 for transformers

AdamW (Adam with Decoupled Weight Decay)
----------------------------------------

AdamW fixes a subtle issue with Adam's weight decay implementation. In Adam,
weight decay is coupled with the gradient update, which can lead to suboptimal
regularization. AdamW decouples weight decay, applying it directly to the
weights as in SGD.

**When to use:**

- Always prefer AdamW over Adam when using weight decay
- Training transformers (BERT, GPT, etc.)
- When you want proper L2 regularization behavior

**Key difference from Adam:**

- In Adam: ``weight = weight - lr * (grad + weight_decay * weight)``
- In AdamW: ``weight = weight - lr * grad - lr * weight_decay * weight``

.. doxygenclass:: torch::optim::AdamW
   :members:
   :undoc-members:

**Example:**

.. code-block:: cpp

   // AdamW with decoupled weight decay - preferred for transformers
   auto optimizer = torch::optim::AdamW(
       model->parameters(),
       torch::optim::AdamWOptions(1e-4)
           .betas({0.9, 0.999})
           .weight_decay(0.01));              // Decoupled regularization

RMSprop (Root Mean Square Propagation)
--------------------------------------

RMSprop adapts the learning rate by dividing by a running average of recent
gradient magnitudes. It's particularly effective for recurrent neural networks
and problems with non-stationary objectives.

**When to use:**

- Training RNNs and LSTMs
- Non-stationary problems where gradient scale varies significantly
- Online learning scenarios

**Key parameters:**

- ``lr``: Learning rate (typical: 1e-3 to 1e-2)
- ``alpha``: Smoothing constant (default: 0.99)
- ``momentum``: Optional momentum term
- ``centered``: Use centered RMSprop (normalizes by variance)

.. doxygenclass:: torch::optim::RMSprop
   :members:
   :undoc-members:

**Example:**

.. code-block:: cpp

   // RMSprop for RNN training
   auto optimizer = torch::optim::RMSprop(
       model->parameters(),
       torch::optim::RMSpropOptions(1e-3)
           .alpha(0.99)                       // smoothing constant
           .momentum(0.9)                     // optional momentum
           .centered(true));                  // normalize by variance

Adagrad (Adaptive Gradient)
---------------------------

Adagrad adapts the learning rate based on the accumulated sum of squared
gradients. Parameters with frequent updates get smaller learning rates, while
parameters with infrequent updates get larger rates. This makes it ideal for
sparse data.

**When to use:**

- NLP tasks with sparse features
- Embedding layers with infrequent updates
- Recommendation systems with sparse user/item features

**Limitation:** Learning rate monotonically decreases, which can cause training
to stop prematurely. For long training runs, consider Adam or RMSprop.

.. doxygenclass:: torch::optim::Adagrad
   :members:
   :undoc-members:

**Example:**

.. code-block:: cpp

   // Adagrad for sparse NLP features
   auto optimizer = torch::optim::Adagrad(
       model->parameters(),
       torch::optim::AdagradOptions(0.01)
           .lr_decay(0)                       // learning rate decay
           .weight_decay(0)
           .initial_accumulator_value(0));
