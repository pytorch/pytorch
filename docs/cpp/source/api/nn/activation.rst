Activation Functions
====================

Activation functions introduce non-linearity into neural networks, allowing them
to learn complex patterns. Without activations, stacked linear layers would collapse
into a single linear transformation.

**Common choices:**

- **ReLU family** (ReLU, LeakyReLU, PReLU, RReLU): Fast, widely used, good default choice
- **ELU family** (ELU, SELU, CELU): Smoother than ReLU, can produce negative outputs
- **GELU/SiLU/Mish**: Modern activations popular in transformers and advanced architectures
- **Sigmoid/Tanh**: Classic activations, useful for output layers (probabilities, bounded outputs)
- **Softmax**: Converts logits to probability distribution (classification output)

ReLU
----

.. doxygenclass:: torch::nn::ReLU
   :members:
   :undoc-members:

.. doxygenclass:: torch::nn::ReLUImpl
   :members:
   :undoc-members:

**Example:**

.. code-block:: cpp

   auto relu = torch::nn::ReLU(torch::nn::ReLUOptions().inplace(true));

LeakyReLU
---------

.. doxygenclass:: torch::nn::LeakyReLU
   :members:
   :undoc-members:

.. doxygenclass:: torch::nn::LeakyReLUImpl
   :members:
   :undoc-members:

PReLU
-----

.. doxygenclass:: torch::nn::PReLU
   :members:
   :undoc-members:

.. doxygenclass:: torch::nn::PReLUImpl
   :members:
   :undoc-members:

RReLU
-----

.. doxygenclass:: torch::nn::RReLU
   :members:
   :undoc-members:

.. doxygenclass:: torch::nn::RReLUImpl
   :members:
   :undoc-members:

ELU
---

.. doxygenclass:: torch::nn::ELU
   :members:
   :undoc-members:

.. doxygenclass:: torch::nn::ELUImpl
   :members:
   :undoc-members:

SELU
----

.. doxygenclass:: torch::nn::SELU
   :members:
   :undoc-members:

.. doxygenclass:: torch::nn::SELUImpl
   :members:
   :undoc-members:

CELU
----

.. doxygenclass:: torch::nn::CELU
   :members:
   :undoc-members:

.. doxygenclass:: torch::nn::CELUImpl
   :members:
   :undoc-members:

GELU
----

.. doxygenclass:: torch::nn::GELU
   :members:
   :undoc-members:

.. doxygenclass:: torch::nn::GELUImpl
   :members:
   :undoc-members:

SiLU (Swish)
------------

.. doxygenclass:: torch::nn::SiLU
   :members:
   :undoc-members:

.. doxygenclass:: torch::nn::SiLUImpl
   :members:
   :undoc-members:

Mish
----

.. doxygenclass:: torch::nn::Mish
   :members:
   :undoc-members:

.. doxygenclass:: torch::nn::MishImpl
   :members:
   :undoc-members:

Sigmoid
-------

.. doxygenclass:: torch::nn::Sigmoid
   :members:
   :undoc-members:

.. doxygenclass:: torch::nn::SigmoidImpl
   :members:
   :undoc-members:

Tanh
----

.. doxygenclass:: torch::nn::Tanh
   :members:
   :undoc-members:

.. doxygenclass:: torch::nn::TanhImpl
   :members:
   :undoc-members:

Softmax
-------

.. doxygenclass:: torch::nn::Softmax
   :members:
   :undoc-members:

.. doxygenclass:: torch::nn::SoftmaxImpl
   :members:
   :undoc-members:

**Example:**

.. code-block:: cpp

   auto softmax = torch::nn::Softmax(torch::nn::SoftmaxOptions(/*dim=*/1));

LogSoftmax
----------

.. doxygenclass:: torch::nn::LogSoftmax
   :members:
   :undoc-members:

.. doxygenclass:: torch::nn::LogSoftmaxImpl
   :members:
   :undoc-members:

Softmin
-------

.. doxygenclass:: torch::nn::Softmin
   :members:
   :undoc-members:

.. doxygenclass:: torch::nn::SoftminImpl
   :members:
   :undoc-members:

Softplus
--------

.. doxygenclass:: torch::nn::Softplus
   :members:
   :undoc-members:

.. doxygenclass:: torch::nn::SoftplusImpl
   :members:
   :undoc-members:

Softshrink
----------

.. doxygenclass:: torch::nn::Softshrink
   :members:
   :undoc-members:

.. doxygenclass:: torch::nn::SoftshrinkImpl
   :members:
   :undoc-members:

Softsign
--------

.. doxygenclass:: torch::nn::Softsign
   :members:
   :undoc-members:

.. doxygenclass:: torch::nn::SoftsignImpl
   :members:
   :undoc-members:

Hardshrink
----------

.. doxygenclass:: torch::nn::Hardshrink
   :members:
   :undoc-members:

.. doxygenclass:: torch::nn::HardshrinkImpl
   :members:
   :undoc-members:

Hardtanh
--------

.. doxygenclass:: torch::nn::Hardtanh
   :members:
   :undoc-members:

.. doxygenclass:: torch::nn::HardtanhImpl
   :members:
   :undoc-members:

Tanhshrink
----------

.. doxygenclass:: torch::nn::Tanhshrink
   :members:
   :undoc-members:

.. doxygenclass:: torch::nn::TanhshrinkImpl
   :members:
   :undoc-members:

Threshold
---------

.. doxygenclass:: torch::nn::Threshold
   :members:
   :undoc-members:

.. doxygenclass:: torch::nn::ThresholdImpl
   :members:
   :undoc-members:
