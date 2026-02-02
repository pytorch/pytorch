Loss Functions
==============

Loss functions measure how well the model's predictions match the targets.
The choice of loss function depends on your task type and data characteristics.

**Regression losses:**

- **L1Loss/MSELoss**: Basic regression losses (MAE vs MSE)
- **SmoothL1Loss/HuberLoss**: Robust to outliers

**Classification losses:**

- **CrossEntropyLoss**: Multi-class classification (combines LogSoftmax + NLLLoss)
- **NLLLoss**: Negative log likelihood (use with LogSoftmax output)
- **BCELoss/BCEWithLogitsLoss**: Binary classification

**Specialized losses:**

- **CTCLoss**: Sequence-to-sequence without alignment (speech recognition)
- **TripletMarginLoss**: Metric learning (similarity/embedding tasks)
- **CosineEmbeddingLoss**: Similarity learning with cosine distance

L1Loss
------

.. doxygenclass:: torch::nn::L1Loss
   :members:
   :undoc-members:

.. doxygenstruct:: torch::nn::L1LossImpl
   :members:
   :undoc-members:

MSELoss
-------

.. doxygenclass:: torch::nn::MSELoss
   :members:
   :undoc-members:

.. doxygenstruct:: torch::nn::MSELossImpl
   :members:
   :undoc-members:

**Example:**

.. code-block:: cpp

   auto loss_fn = torch::nn::MSELoss();
   auto loss = loss_fn->forward(predictions, targets);

CrossEntropyLoss
----------------

.. doxygenclass:: torch::nn::CrossEntropyLoss
   :members:
   :undoc-members:

.. doxygenstruct:: torch::nn::CrossEntropyLossImpl
   :members:
   :undoc-members:

**Example:**

.. code-block:: cpp

   auto loss_fn = torch::nn::CrossEntropyLoss();
   auto logits = torch::randn({32, 10});  // [batch, num_classes]
   auto targets = torch::randint(0, 10, {32});  // [batch]
   auto loss = loss_fn->forward(logits, targets);

NLLLoss
-------

.. doxygenclass:: torch::nn::NLLLoss
   :members:
   :undoc-members:

.. doxygenstruct:: torch::nn::NLLLossImpl
   :members:
   :undoc-members:

BCELoss
-------

.. doxygenclass:: torch::nn::BCELoss
   :members:
   :undoc-members:

.. doxygenstruct:: torch::nn::BCELossImpl
   :members:
   :undoc-members:

BCEWithLogitsLoss
-----------------

.. doxygenclass:: torch::nn::BCEWithLogitsLoss
   :members:
   :undoc-members:

.. doxygenstruct:: torch::nn::BCEWithLogitsLossImpl
   :members:
   :undoc-members:

HuberLoss
---------

.. doxygenclass:: torch::nn::HuberLoss
   :members:
   :undoc-members:

.. doxygenstruct:: torch::nn::HuberLossImpl
   :members:
   :undoc-members:

SmoothL1Loss
------------

.. doxygenclass:: torch::nn::SmoothL1Loss
   :members:
   :undoc-members:

.. doxygenstruct:: torch::nn::SmoothL1LossImpl
   :members:
   :undoc-members:

KLDivLoss
---------

.. doxygenclass:: torch::nn::KLDivLoss
   :members:
   :undoc-members:

.. doxygenstruct:: torch::nn::KLDivLossImpl
   :members:
   :undoc-members:

CTCLoss
-------

.. doxygenclass:: torch::nn::CTCLoss
   :members:
   :undoc-members:

.. doxygenstruct:: torch::nn::CTCLossImpl
   :members:
   :undoc-members:

PoissonNLLLoss
--------------

.. doxygenclass:: torch::nn::PoissonNLLLoss
   :members:
   :undoc-members:

.. doxygenstruct:: torch::nn::PoissonNLLLossImpl
   :members:
   :undoc-members:

MarginRankingLoss
-----------------

.. doxygenclass:: torch::nn::MarginRankingLoss
   :members:
   :undoc-members:

.. doxygenstruct:: torch::nn::MarginRankingLossImpl
   :members:
   :undoc-members:

HingeEmbeddingLoss
------------------

.. doxygenclass:: torch::nn::HingeEmbeddingLoss
   :members:
   :undoc-members:

.. doxygenstruct:: torch::nn::HingeEmbeddingLossImpl
   :members:
   :undoc-members:

CosineEmbeddingLoss
-------------------

.. doxygenclass:: torch::nn::CosineEmbeddingLoss
   :members:
   :undoc-members:

.. doxygenstruct:: torch::nn::CosineEmbeddingLossImpl
   :members:
   :undoc-members:

MultiMarginLoss
---------------

.. doxygenclass:: torch::nn::MultiMarginLoss
   :members:
   :undoc-members:

.. doxygenstruct:: torch::nn::MultiMarginLossImpl
   :members:
   :undoc-members:

MultiLabelMarginLoss
--------------------

.. doxygenclass:: torch::nn::MultiLabelMarginLoss
   :members:
   :undoc-members:

.. doxygenstruct:: torch::nn::MultiLabelMarginLossImpl
   :members:
   :undoc-members:

MultiLabelSoftMarginLoss
------------------------

.. doxygenclass:: torch::nn::MultiLabelSoftMarginLoss
   :members:
   :undoc-members:

.. doxygenstruct:: torch::nn::MultiLabelSoftMarginLossImpl
   :members:
   :undoc-members:

SoftMarginLoss
--------------

.. doxygenclass:: torch::nn::SoftMarginLoss
   :members:
   :undoc-members:

.. doxygenstruct:: torch::nn::SoftMarginLossImpl
   :members:
   :undoc-members:

TripletMarginLoss
-----------------

.. doxygenclass:: torch::nn::TripletMarginLoss
   :members:
   :undoc-members:

.. doxygenstruct:: torch::nn::TripletMarginLossImpl
   :members:
   :undoc-members:

TripletMarginWithDistanceLoss
-----------------------------

.. doxygenclass:: torch::nn::TripletMarginWithDistanceLoss
   :members:
   :undoc-members:

.. doxygenstruct:: torch::nn::TripletMarginWithDistanceLossImpl
   :members:
   :undoc-members:
