Utilities
=========

Additional utilities for building neural networks: parameter initialization,
module cloning, type-erased containers, padding layers, and vision utilities.

Parameter Initialization
------------------------

The ``torch::nn::init`` namespace provides functions for initializing module parameters:

.. code-block:: cpp

   #include <torch/nn/init.h>

   // Xavier/Glorot initialization
   torch::nn::init::xavier_uniform_(linear->weight);
   torch::nn::init::xavier_normal_(linear->weight);

   // Kaiming/He initialization
   torch::nn::init::kaiming_uniform_(conv->weight, /*a=*/0, torch::kFanIn, torch::kReLU);
   torch::nn::init::kaiming_normal_(conv->weight);

   // Other initializations
   torch::nn::init::zeros_(linear->bias);
   torch::nn::init::ones_(bn->weight);
   torch::nn::init::constant_(linear->bias, 0.1);
   torch::nn::init::normal_(linear->weight, /*mean=*/0, /*std=*/0.01);
   torch::nn::init::uniform_(linear->weight, /*a=*/-0.1, /*b=*/0.1);
   torch::nn::init::orthogonal_(rnn->weight_hh);

Cloneable
---------

.. doxygenclass:: torch::nn::Cloneable
   :members:
   :undoc-members:

All ``torch::nn`` modules inherit from ``Cloneable``, enabling deep copies:

.. code-block:: cpp

   auto model = torch::nn::Linear(10, 5);
   auto model_copy = std::dynamic_pointer_cast<torch::nn::LinearImpl>(model->clone());

AnyModule
---------

.. doxygenclass:: torch::nn::AnyModule
   :members:
   :undoc-members:

``AnyModule`` provides type-erased storage for any module:

.. code-block:: cpp

   torch::nn::AnyModule any_module(torch::nn::Linear(10, 5));
   auto output = any_module.forward(input);

Padding Layers
--------------

ReflectionPad1d / ReflectionPad2d / ReflectionPad3d
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. doxygenclass:: torch::nn::ReflectionPad1d
   :members:
   :undoc-members:

.. doxygenclass:: torch::nn::ReflectionPad2d
   :members:
   :undoc-members:

.. doxygenclass:: torch::nn::ReflectionPad3d
   :members:
   :undoc-members:

ReplicationPad1d / ReplicationPad2d / ReplicationPad3d
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. doxygenclass:: torch::nn::ReplicationPad1d
   :members:
   :undoc-members:

.. doxygenclass:: torch::nn::ReplicationPad2d
   :members:
   :undoc-members:

.. doxygenclass:: torch::nn::ReplicationPad3d
   :members:
   :undoc-members:

ZeroPad1d / ZeroPad2d / ZeroPad3d
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. doxygenclass:: torch::nn::ZeroPad1d
   :members:
   :undoc-members:

.. doxygenclass:: torch::nn::ZeroPad2d
   :members:
   :undoc-members:

.. doxygenclass:: torch::nn::ZeroPad3d
   :members:
   :undoc-members:

ConstantPad1d / ConstantPad2d / ConstantPad3d
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. doxygenclass:: torch::nn::ConstantPad1d
   :members:
   :undoc-members:

.. doxygenclass:: torch::nn::ConstantPad2d
   :members:
   :undoc-members:

.. doxygenclass:: torch::nn::ConstantPad3d
   :members:
   :undoc-members:

Vision Layers
-------------

PixelShuffle
^^^^^^^^^^^^

.. doxygenclass:: torch::nn::PixelShuffle
   :members:
   :undoc-members:

.. doxygenstruct:: torch::nn::PixelShuffleOptions
   :members:
   :undoc-members:

PixelUnshuffle
^^^^^^^^^^^^^^

.. doxygenclass:: torch::nn::PixelUnshuffle
   :members:
   :undoc-members:

.. doxygenstruct:: torch::nn::PixelUnshuffleOptions
   :members:
   :undoc-members:

Upsample
^^^^^^^^

.. doxygenclass:: torch::nn::Upsample
   :members:
   :undoc-members:

.. doxygenstruct:: torch::nn::UpsampleOptions
   :members:
   :undoc-members:

Fold / Unfold
^^^^^^^^^^^^^

.. doxygenclass:: torch::nn::Fold
   :members:
   :undoc-members:

.. doxygenstruct:: torch::nn::FoldOptions
   :members:
   :undoc-members:

.. doxygenclass:: torch::nn::Unfold
   :members:
   :undoc-members:

.. doxygenstruct:: torch::nn::UnfoldOptions
   :members:
   :undoc-members:
