Transforms
==========

Transforms apply preprocessing to data samples, such as normalization or
augmentation. They can be chained using the ``.map()`` method on datasets.

Normalize
---------

Normalizes tensors with a given mean and standard deviation.

.. doxygenstruct:: torch::data::transforms::Normalize
   :members:
   :undoc-members:

Stack
-----

Stacks a batch of tensors into a single tensor.

.. cpp:class:: torch::data::transforms::Stack

   Transform that stacks a batch of tensors into a single tensor.

   **Example:**

   .. code-block:: cpp

      auto dataset = torch::data::datasets::MNIST("./data")
          .map(torch::data::transforms::Normalize<>(0.5, 0.5))
          .map(torch::data::transforms::Stack<>());

Chaining Transforms
-------------------

Transforms can be chained together using ``.map()``:

.. code-block:: cpp

   auto dataset = torch::data::datasets::MNIST("./data")
       .map(torch::data::transforms::Normalize<>(0.1307, 0.3081))
       .map(torch::data::transforms::Stack<>());
