Linear Layers
=============

Linear layers apply affine transformations to input data: ``y = xW^T + b``.
They are the building blocks of fully-connected networks and are used for
feature transformation, classification heads, and projection layers.

- **Linear**: Standard fully-connected layer
- **Bilinear**: Bilinear transformation of two inputs
- **Identity**: Pass-through layer (useful for skip connections)
- **Flatten/Unflatten**: Reshape tensors between convolutional and linear layers

Linear
------

.. doxygenclass:: torch::nn::Linear
   :members:
   :undoc-members:

.. doxygenclass:: torch::nn::LinearImpl
   :members:
   :undoc-members:

**Example:**

.. code-block:: cpp

   auto linear = torch::nn::Linear(torch::nn::LinearOptions(784, 256).bias(true));
   auto output = linear->forward(input);  // input: [N, 784]

Bilinear
--------

.. doxygenclass:: torch::nn::Bilinear
   :members:
   :undoc-members:

.. doxygenclass:: torch::nn::BilinearImpl
   :members:
   :undoc-members:

Identity
--------

.. doxygenclass:: torch::nn::Identity
   :members:
   :undoc-members:

.. doxygenclass:: torch::nn::IdentityImpl
   :members:
   :undoc-members:

Flatten
-------

.. doxygenclass:: torch::nn::Flatten
   :members:
   :undoc-members:

.. doxygenclass:: torch::nn::FlattenImpl
   :members:
   :undoc-members:

Unflatten
---------

.. doxygenclass:: torch::nn::Unflatten
   :members:
   :undoc-members:

.. doxygenclass:: torch::nn::UnflattenImpl
   :members:
   :undoc-members:
