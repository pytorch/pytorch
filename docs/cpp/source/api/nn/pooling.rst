Pooling Layers
==============

Pooling layers reduce spatial dimensions by aggregating values in local regions,
providing translation invariance and reducing computational cost in deeper layers.

- **MaxPool**: Takes the maximum value in each pooling window (preserves strong features)
- **AvgPool**: Takes the average value in each pooling window (smoother downsampling)
- **AdaptivePool**: Automatically calculates kernel size to produce a target output size
- **FractionalMaxPool**: Randomized pooling with fractional output size
- **LPPool**: Power-average pooling (generalization of avg/max pooling)

MaxPool1d / MaxPool2d / MaxPool3d
---------------------------------

.. doxygenclass:: torch::nn::MaxPool1d
   :members:
   :undoc-members:

.. doxygenclass:: torch::nn::MaxPool1dImpl
   :members:
   :undoc-members:

.. doxygenclass:: torch::nn::MaxPool2d
   :members:
   :undoc-members:

.. doxygenclass:: torch::nn::MaxPool2dImpl
   :members:
   :undoc-members:

.. doxygenclass:: torch::nn::MaxPool3d
   :members:
   :undoc-members:

.. doxygenclass:: torch::nn::MaxPool3dImpl
   :members:
   :undoc-members:

**Example:**

.. code-block:: cpp

   auto pool = torch::nn::MaxPool2d(
       torch::nn::MaxPool2dOptions(2).stride(2));

AvgPool1d / AvgPool2d / AvgPool3d
---------------------------------

.. doxygenclass:: torch::nn::AvgPool1d
   :members:
   :undoc-members:

.. doxygenclass:: torch::nn::AvgPool1dImpl
   :members:
   :undoc-members:

.. doxygenclass:: torch::nn::AvgPool2d
   :members:
   :undoc-members:

.. doxygenclass:: torch::nn::AvgPool2dImpl
   :members:
   :undoc-members:

.. doxygenclass:: torch::nn::AvgPool3d
   :members:
   :undoc-members:

.. doxygenclass:: torch::nn::AvgPool3dImpl
   :members:
   :undoc-members:

AdaptiveAvgPool1d / AdaptiveAvgPool2d / AdaptiveAvgPool3d
---------------------------------------------------------

.. doxygenclass:: torch::nn::AdaptiveAvgPool1d
   :members:
   :undoc-members:

.. doxygenclass:: torch::nn::AdaptiveAvgPool1dImpl
   :members:
   :undoc-members:

.. doxygenclass:: torch::nn::AdaptiveAvgPool2d
   :members:
   :undoc-members:

.. doxygenclass:: torch::nn::AdaptiveAvgPool2dImpl
   :members:
   :undoc-members:

.. doxygenclass:: torch::nn::AdaptiveAvgPool3d
   :members:
   :undoc-members:

.. doxygenclass:: torch::nn::AdaptiveAvgPool3dImpl
   :members:
   :undoc-members:

**Example:**

.. code-block:: cpp

   // Output will always be 7x7 regardless of input size
   auto adaptive_pool = torch::nn::AdaptiveAvgPool2d(
       torch::nn::AdaptiveAvgPool2dOptions({7, 7}));

AdaptiveMaxPool1d / AdaptiveMaxPool2d / AdaptiveMaxPool3d
---------------------------------------------------------

.. doxygenclass:: torch::nn::AdaptiveMaxPool1d
   :members:
   :undoc-members:

.. doxygenclass:: torch::nn::AdaptiveMaxPool1dImpl
   :members:
   :undoc-members:

.. doxygenclass:: torch::nn::AdaptiveMaxPool2d
   :members:
   :undoc-members:

.. doxygenclass:: torch::nn::AdaptiveMaxPool2dImpl
   :members:
   :undoc-members:

.. doxygenclass:: torch::nn::AdaptiveMaxPool3d
   :members:
   :undoc-members:

.. doxygenclass:: torch::nn::AdaptiveMaxPool3dImpl
   :members:
   :undoc-members:

FractionalMaxPool2d / FractionalMaxPool3d
-----------------------------------------

.. doxygenclass:: torch::nn::FractionalMaxPool2d
   :members:
   :undoc-members:

.. doxygenclass:: torch::nn::FractionalMaxPool2dImpl
   :members:
   :undoc-members:

.. doxygenclass:: torch::nn::FractionalMaxPool3d
   :members:
   :undoc-members:

.. doxygenclass:: torch::nn::FractionalMaxPool3dImpl
   :members:
   :undoc-members:

LPPool1d / LPPool2d / LPPool3d
------------------------------

.. doxygenclass:: torch::nn::LPPool1d
   :members:
   :undoc-members:

.. doxygenclass:: torch::nn::LPPool1dImpl
   :members:
   :undoc-members:

.. doxygenclass:: torch::nn::LPPool2d
   :members:
   :undoc-members:

.. doxygenclass:: torch::nn::LPPool2dImpl
   :members:
   :undoc-members:

.. doxygenclass:: torch::nn::LPPool3d
   :members:
   :undoc-members:

.. doxygenclass:: torch::nn::LPPool3dImpl
   :members:
   :undoc-members:
