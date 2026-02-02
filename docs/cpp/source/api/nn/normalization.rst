Normalization Layers
====================

Normalization layers stabilize and accelerate training by normalizing intermediate
activations. They help with gradient flow and allow higher learning rates.

- **BatchNorm**: Normalizes across batch dimension; most common in CNNs
- **InstanceNorm**: Normalizes each sample independently; popular in style transfer
- **LayerNorm**: Normalizes across feature dimension; standard in transformers
- **GroupNorm**: Normalizes within groups of channels; works with small batches
- **LocalResponseNorm**: Lateral inhibition inspired by neuroscience (less common today)

BatchNorm1d / BatchNorm2d / BatchNorm3d
---------------------------------------

.. doxygenclass:: torch::nn::BatchNorm1d
   :members:
   :undoc-members:

.. doxygenclass:: torch::nn::BatchNorm1dImpl
   :members:
   :undoc-members:

.. doxygenclass:: torch::nn::BatchNorm2d
   :members:
   :undoc-members:

.. doxygenclass:: torch::nn::BatchNorm2dImpl
   :members:
   :undoc-members:

.. doxygenclass:: torch::nn::BatchNorm3d
   :members:
   :undoc-members:

.. doxygenclass:: torch::nn::BatchNorm3dImpl
   :members:
   :undoc-members:

**Example:**

.. code-block:: cpp

   auto bn = torch::nn::BatchNorm2d(
       torch::nn::BatchNorm2dOptions(64)  // num_features
           .eps(1e-5)
           .momentum(0.1)
           .affine(true)
           .track_running_stats(true));

InstanceNorm1d / InstanceNorm2d / InstanceNorm3d
------------------------------------------------

.. doxygenclass:: torch::nn::InstanceNorm1d
   :members:
   :undoc-members:

.. doxygenclass:: torch::nn::InstanceNorm1dImpl
   :members:
   :undoc-members:

.. doxygenclass:: torch::nn::InstanceNorm2d
   :members:
   :undoc-members:

.. doxygenclass:: torch::nn::InstanceNorm2dImpl
   :members:
   :undoc-members:

.. doxygenclass:: torch::nn::InstanceNorm3d
   :members:
   :undoc-members:

.. doxygenclass:: torch::nn::InstanceNorm3dImpl
   :members:
   :undoc-members:

LayerNorm
---------

.. doxygenclass:: torch::nn::LayerNorm
   :members:
   :undoc-members:

.. doxygenclass:: torch::nn::LayerNormImpl
   :members:
   :undoc-members:

**Example:**

.. code-block:: cpp

   auto ln = torch::nn::LayerNorm(
       torch::nn::LayerNormOptions({768}));  // normalized_shape

GroupNorm
---------

.. doxygenclass:: torch::nn::GroupNorm
   :members:
   :undoc-members:

.. doxygenclass:: torch::nn::GroupNormImpl
   :members:
   :undoc-members:

**Example:**

.. code-block:: cpp

   auto gn = torch::nn::GroupNorm(
       torch::nn::GroupNormOptions(32, 256));  // num_groups, num_channels

LocalResponseNorm
-----------------

.. doxygenclass:: torch::nn::LocalResponseNorm
   :members:
   :undoc-members:

.. doxygenclass:: torch::nn::LocalResponseNormImpl
   :members:
   :undoc-members:
