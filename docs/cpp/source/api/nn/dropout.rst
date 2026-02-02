Dropout Layers
==============

Dropout randomly zeros elements during training as a regularization technique,
preventing overfitting by forcing the network to learn redundant representations.
During evaluation, dropout is disabled and outputs are scaled appropriately.

- **Dropout**: Standard dropout for fully-connected layers
- **Dropout2d/3d**: Spatial dropout that zeros entire channels (better for CNNs)
- **AlphaDropout**: Maintains self-normalizing property (use with SELU activation)

.. note::
   Remember to call ``model->train()`` during training and ``model->eval()`` during
   inference to properly enable/disable dropout behavior.

Dropout
-------

.. doxygenclass:: torch::nn::Dropout
   :members:
   :undoc-members:

.. doxygenclass:: torch::nn::DropoutImpl
   :members:
   :undoc-members:

**Example:**

.. code-block:: cpp

   auto dropout = torch::nn::Dropout(torch::nn::DropoutOptions(0.5));

Dropout2d / Dropout3d
---------------------

.. doxygenclass:: torch::nn::Dropout2d
   :members:
   :undoc-members:

.. doxygenclass:: torch::nn::Dropout2dImpl
   :members:
   :undoc-members:

.. doxygenclass:: torch::nn::Dropout3d
   :members:
   :undoc-members:

.. doxygenclass:: torch::nn::Dropout3dImpl
   :members:
   :undoc-members:

AlphaDropout
------------

.. doxygenclass:: torch::nn::AlphaDropout
   :members:
   :undoc-members:

.. doxygenclass:: torch::nn::AlphaDropoutImpl
   :members:
   :undoc-members:

FeatureAlphaDropout
-------------------

.. doxygenclass:: torch::nn::FeatureAlphaDropout
   :members:
   :undoc-members:

.. doxygenclass:: torch::nn::FeatureAlphaDropoutImpl
   :members:
   :undoc-members:
