.. role:: hidden
    :class: hidden-section

torch.nn
===================================
.. automodule:: torch.nn
.. automodule:: torch.nn.modules

These are the basic building blocks for graphs:

.. contents:: torch.nn
    :depth: 2
    :local:
    :backlinks: top


.. currentmodule:: torch.nn


.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    ~parameter.Buffer
    ~parameter.Parameter
    ~parameter.UninitializedParameter
    ~parameter.UninitializedBuffer

Containers
----------------------------------

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    Module
    Sequential
    ModuleList
    ModuleDict
    ParameterList
    ParameterDict

Global Hooks For Module

.. currentmodule:: torch.nn.modules.module
.. autosummary::
    :toctree: generated
    :nosignatures:

    register_module_forward_pre_hook
    register_module_forward_hook
    register_module_backward_hook
    register_module_full_backward_pre_hook
    register_module_full_backward_hook
    register_module_buffer_registration_hook
    register_module_module_registration_hook
    register_module_parameter_registration_hook

.. currentmodule:: torch

Convolution Layers
----------------------------------

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    nn.Conv1d
    nn.Conv2d
    nn.Conv3d
    nn.ConvTranspose1d
    nn.ConvTranspose2d
    nn.ConvTranspose3d
    nn.LazyConv1d
    nn.LazyConv2d
    nn.LazyConv3d
    nn.LazyConvTranspose1d
    nn.LazyConvTranspose2d
    nn.LazyConvTranspose3d
    nn.Unfold
    nn.Fold

Pooling layers
----------------------------------

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    nn.MaxPool1d
    nn.MaxPool2d
    nn.MaxPool3d
    nn.MaxUnpool1d
    nn.MaxUnpool2d
    nn.MaxUnpool3d
    nn.AvgPool1d
    nn.AvgPool2d
    nn.AvgPool3d
    nn.FractionalMaxPool2d
    nn.FractionalMaxPool3d
    nn.LPPool1d
    nn.LPPool2d
    nn.LPPool3d
    nn.AdaptiveMaxPool1d
    nn.AdaptiveMaxPool2d
    nn.AdaptiveMaxPool3d
    nn.AdaptiveAvgPool1d
    nn.AdaptiveAvgPool2d
    nn.AdaptiveAvgPool3d

Padding Layers
--------------

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    nn.ReflectionPad1d
    nn.ReflectionPad2d
    nn.ReflectionPad3d
    nn.ReplicationPad1d
    nn.ReplicationPad2d
    nn.ReplicationPad3d
    nn.ZeroPad1d
    nn.ZeroPad2d
    nn.ZeroPad3d
    nn.ConstantPad1d
    nn.ConstantPad2d
    nn.ConstantPad3d
    nn.CircularPad1d
    nn.CircularPad2d
    nn.CircularPad3d

Non-linear Activations (weighted sum, nonlinearity)
---------------------------------------------------

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    nn.ELU
    nn.Hardshrink
    nn.Hardsigmoid
    nn.Hardtanh
    nn.Hardswish
    nn.LeakyReLU
    nn.LogSigmoid
    nn.MultiheadAttention
    nn.PReLU
    nn.ReLU
    nn.ReLU6
    nn.RReLU
    nn.SELU
    nn.CELU
    nn.GELU
    nn.Sigmoid
    nn.SiLU
    nn.Mish
    nn.Softplus
    nn.Softshrink
    nn.Softsign
    nn.Tanh
    nn.Tanhshrink
    nn.Threshold
    nn.GLU

Non-linear Activations (other)
------------------------------

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    nn.Softmin
    nn.Softmax
    nn.Softmax2d
    nn.LogSoftmax
    nn.AdaptiveLogSoftmaxWithLoss

Normalization Layers
----------------------------------

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    nn.BatchNorm1d
    nn.BatchNorm2d
    nn.BatchNorm3d
    nn.LazyBatchNorm1d
    nn.LazyBatchNorm2d
    nn.LazyBatchNorm3d
    nn.GroupNorm
    nn.SyncBatchNorm
    nn.InstanceNorm1d
    nn.InstanceNorm2d
    nn.InstanceNorm3d
    nn.LazyInstanceNorm1d
    nn.LazyInstanceNorm2d
    nn.LazyInstanceNorm3d
    nn.LayerNorm
    nn.LocalResponseNorm
    nn.RMSNorm

Recurrent Layers
----------------

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    nn.RNNBase
    nn.RNN
    nn.LSTM
    nn.GRU
    nn.RNNCell
    nn.LSTMCell
    nn.GRUCell

Transformer Layers
----------------------------------

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    nn.Transformer
    nn.TransformerEncoder
    nn.TransformerDecoder
    nn.TransformerEncoderLayer
    nn.TransformerDecoderLayer

Linear Layers
----------------------------------

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    nn.Identity
    nn.Linear
    nn.Bilinear
    nn.LazyLinear

Dropout Layers
--------------

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    nn.Dropout
    nn.Dropout1d
    nn.Dropout2d
    nn.Dropout3d
    nn.AlphaDropout
    nn.FeatureAlphaDropout

Sparse Layers
-------------

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    nn.Embedding
    nn.EmbeddingBag

Distance Functions
------------------

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    nn.CosineSimilarity
    nn.PairwiseDistance

Loss Functions
--------------

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    nn.L1Loss
    nn.MSELoss
    nn.CrossEntropyLoss
    nn.CTCLoss
    nn.NLLLoss
    nn.PoissonNLLLoss
    nn.GaussianNLLLoss
    nn.KLDivLoss
    nn.BCELoss
    nn.BCEWithLogitsLoss
    nn.MarginRankingLoss
    nn.HingeEmbeddingLoss
    nn.MultiLabelMarginLoss
    nn.HuberLoss
    nn.SmoothL1Loss
    nn.SoftMarginLoss
    nn.MultiLabelSoftMarginLoss
    nn.CosineEmbeddingLoss
    nn.MultiMarginLoss
    nn.TripletMarginLoss
    nn.TripletMarginWithDistanceLoss

Vision Layers
----------------

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    nn.PixelShuffle
    nn.PixelUnshuffle
    nn.Upsample
    nn.UpsamplingNearest2d
    nn.UpsamplingBilinear2d

Shuffle Layers
----------------

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    nn.ChannelShuffle

DataParallel Layers (multi-GPU, distributed)
--------------------------------------------
.. automodule:: torch.nn.parallel
.. currentmodule:: torch

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    nn.DataParallel
    nn.parallel.DistributedDataParallel

Utilities
---------
.. automodule:: torch.nn.utils

From the ``torch.nn.utils`` module:

Utility functions to clip parameter gradients.

.. currentmodule:: torch.nn.utils
.. autosummary::
    :toctree: generated
    :nosignatures:

    clip_grad_norm_
    clip_grad_norm
    clip_grad_value_
    get_total_norm
    clip_grads_with_norm_

Utility functions to flatten and unflatten Module parameters to and from a single vector.

.. autosummary::
    :toctree: generated
    :nosignatures:

    parameters_to_vector
    vector_to_parameters

Utility functions to fuse Modules with BatchNorm modules.

.. autosummary::
    :toctree: generated
    :nosignatures:

    fuse_conv_bn_eval
    fuse_conv_bn_weights
    fuse_linear_bn_eval
    fuse_linear_bn_weights

Utility functions to convert Module parameter memory formats.

.. autosummary::
    :toctree: generated
    :nosignatures:

    convert_conv2d_weight_memory_format
    convert_conv3d_weight_memory_format

Utility functions to apply and remove weight normalization from Module parameters.

.. autosummary::
    :toctree: generated
    :nosignatures:

    weight_norm
    remove_weight_norm
    spectral_norm
    remove_spectral_norm

Utility functions for initializing Module parameters.

.. autosummary::
    :toctree: generated
    :nosignatures:

    skip_init

Utility classes and functions for pruning Module parameters.

.. autosummary::
    :toctree: generated
    :nosignatures:

    prune.BasePruningMethod
    prune.PruningContainer
    prune.Identity
    prune.RandomUnstructured
    prune.L1Unstructured
    prune.RandomStructured
    prune.LnStructured
    prune.CustomFromMask
    prune.identity
    prune.random_unstructured
    prune.l1_unstructured
    prune.random_structured
    prune.ln_structured
    prune.global_unstructured
    prune.custom_from_mask
    prune.remove
    prune.is_pruned

Parametrizations implemented using the new parametrization functionality
in :func:`torch.nn.utils.parameterize.register_parametrization`.

.. autosummary::
    :toctree: generated
    :nosignatures:

    parametrizations.orthogonal
    parametrizations.weight_norm
    parametrizations.spectral_norm

Utility functions to parametrize Tensors on existing Modules.
Note that these functions can be used to parametrize a given Parameter
or Buffer given a specific function that maps from an input space to the
parametrized space. They are not parameterizations that would transform
an object into a parameter. See the
`Parametrizations tutorial <https://pytorch.org/tutorials/intermediate/parametrizations.html>`_
for more information on how to implement your own parametrizations.

.. autosummary::
    :toctree: generated
    :nosignatures:

    parametrize.register_parametrization
    parametrize.remove_parametrizations
    parametrize.cached
    parametrize.is_parametrized

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    parametrize.ParametrizationList

Utility functions to call a given Module in a stateless manner.

.. autosummary::
    :toctree: generated
    :nosignatures:

    stateless.functional_call

Utility functions in other modules

.. currentmodule:: torch
.. autosummary::
    :toctree: generated
    :nosignatures:

    nn.utils.rnn.PackedSequence
    nn.utils.rnn.pack_padded_sequence
    nn.utils.rnn.pad_packed_sequence
    nn.utils.rnn.pad_sequence
    nn.utils.rnn.pack_sequence
    nn.utils.rnn.unpack_sequence
    nn.utils.rnn.unpad_sequence

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    nn.modules.flatten.Flatten
    nn.modules.flatten.Unflatten

Quantized Functions
--------------------

Quantization refers to techniques for performing computations and storing tensors at lower bitwidths than
floating point precision. PyTorch supports both per tensor and per channel asymmetric linear quantization. To learn more how to use quantized functions in PyTorch, please refer to the :ref:`quantization-doc` documentation.

Lazy Modules Initialization
---------------------------

.. currentmodule:: torch
.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    nn.modules.lazy.LazyModuleMixin

Aliases
_______

The following are aliases to their counterparts in ``torch.nn`` in the ``torch.nn.modules`` namespace.

Containers (Aliases)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. currentmodule:: torch.nn.modules
.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    module.Module
    container.Sequential
    container.ModuleList
    container.ModuleDict
    container.ParameterList
    container.ParameterDict

Convolution Layers (Aliases)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    conv.Conv1d
    conv.Conv2d
    conv.Conv3d
    conv.ConvTranspose1d
    conv.ConvTranspose2d
    conv.ConvTranspose3d
    conv.LazyConv1d
    conv.LazyConv2d
    conv.LazyConv3d
    conv.LazyConvTranspose1d
    conv.LazyConvTranspose2d
    conv.LazyConvTranspose3d
    fold.Unfold
    fold.Fold

Pooling layers (Aliases)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    pooling.MaxPool1d
    pooling.MaxPool2d
    pooling.MaxPool3d
    pooling.MaxUnpool1d
    pooling.MaxUnpool2d
    pooling.MaxUnpool3d
    pooling.AvgPool1d
    pooling.AvgPool2d
    pooling.AvgPool3d
    pooling.FractionalMaxPool2d
    pooling.FractionalMaxPool3d
    pooling.LPPool1d
    pooling.LPPool2d
    pooling.LPPool3d
    pooling.AdaptiveMaxPool1d
    pooling.AdaptiveMaxPool2d
    pooling.AdaptiveMaxPool3d
    pooling.AdaptiveAvgPool1d
    pooling.AdaptiveAvgPool2d
    pooling.AdaptiveAvgPool3d

Padding Layers (Aliases)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    padding.ReflectionPad1d
    padding.ReflectionPad2d
    padding.ReflectionPad3d
    padding.ReplicationPad1d
    padding.ReplicationPad2d
    padding.ReplicationPad3d
    padding.ZeroPad1d
    padding.ZeroPad2d
    padding.ZeroPad3d
    padding.ConstantPad1d
    padding.ConstantPad2d
    padding.ConstantPad3d
    padding.CircularPad1d
    padding.CircularPad2d
    padding.CircularPad3d

Non-linear Activations (weighted sum, nonlinearity) (Aliases)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    activation.ELU
    activation.Hardshrink
    activation.Hardsigmoid
    activation.Hardtanh
    activation.Hardswish
    activation.LeakyReLU
    activation.LogSigmoid
    activation.MultiheadAttention
    activation.PReLU
    activation.ReLU
    activation.ReLU6
    activation.RReLU
    activation.SELU
    activation.CELU
    activation.GELU
    activation.Sigmoid
    activation.SiLU
    activation.Mish
    activation.Softplus
    activation.Softshrink
    activation.Softsign
    activation.Tanh
    activation.Tanhshrink
    activation.Threshold
    activation.GLU

Non-linear Activations (other) (Aliases)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    activation.Softmin
    activation.Softmax
    activation.Softmax2d
    activation.LogSoftmax
    adaptive.AdaptiveLogSoftmaxWithLoss

Normalization Layers (Aliases)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    batchnorm.BatchNorm1d
    batchnorm.BatchNorm2d
    batchnorm.BatchNorm3d
    batchnorm.LazyBatchNorm1d
    batchnorm.LazyBatchNorm2d
    batchnorm.LazyBatchNorm3d
    normalization.GroupNorm
    batchnorm.SyncBatchNorm
    instancenorm.InstanceNorm1d
    instancenorm.InstanceNorm2d
    instancenorm.InstanceNorm3d
    instancenorm.LazyInstanceNorm1d
    instancenorm.LazyInstanceNorm2d
    instancenorm.LazyInstanceNorm3d
    normalization.LayerNorm
    normalization.LocalResponseNorm
    normalization.RMSNorm

Recurrent Layers (Aliases)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    rnn.RNNBase
    rnn.RNN
    rnn.LSTM
    rnn.GRU
    rnn.RNNCell
    rnn.LSTMCell
    rnn.GRUCell

Transformer Layers (Aliases)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    transformer.Transformer
    transformer.TransformerEncoder
    transformer.TransformerDecoder
    transformer.TransformerEncoderLayer
    transformer.TransformerDecoderLayer

Linear Layers (Aliases)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    linear.Identity
    linear.Linear
    linear.Bilinear
    linear.LazyLinear

Dropout Layers (Aliases)
^^^^^^^^^^^^^^^^^^^^^^^^
.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    dropout.Dropout
    dropout.Dropout1d
    dropout.Dropout2d
    dropout.Dropout3d
    dropout.AlphaDropout
    dropout.FeatureAlphaDropout

Sparse Layers (Aliases)
^^^^^^^^^^^^^^^^^^^^^^^
.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    sparse.Embedding
    sparse.EmbeddingBag

Distance Functions (Aliases)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    distance.CosineSimilarity
    distance.PairwiseDistance

Loss Functions (Aliases)
^^^^^^^^^^^^^^^^^^^^^^^^
.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    loss.L1Loss
    loss.MSELoss
    loss.CrossEntropyLoss
    loss.CTCLoss
    loss.NLLLoss
    loss.PoissonNLLLoss
    loss.GaussianNLLLoss
    loss.KLDivLoss
    loss.BCELoss
    loss.BCEWithLogitsLoss
    loss.MarginRankingLoss
    loss.HingeEmbeddingLoss
    loss.MultiLabelMarginLoss
    loss.HuberLoss
    loss.SmoothL1Loss
    loss.SoftMarginLoss
    loss.MultiLabelSoftMarginLoss
    loss.CosineEmbeddingLoss
    loss.MultiMarginLoss
    loss.TripletMarginLoss
    loss.TripletMarginWithDistanceLoss

Vision Layers (Aliases)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    pixelshuffle.PixelShuffle
    pixelshuffle.PixelUnshuffle
    upsampling.Upsample
    upsampling.UpsamplingNearest2d
    upsampling.UpsamplingBilinear2d

Shuffle Layers (Aliases)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    channelshuffle.ChannelShuffle


.. This module needs to be documented. Adding here in the meantime
.. for tracking purposes
.. py:module:: torch.nn.backends
.. py:module:: torch.nn.utils.stateless
.. py:module:: torch.nn.backends.thnn
.. py:module:: torch.nn.common_types
.. py:module:: torch.nn.cpp
.. py:module:: torch.nn.functional
.. py:module:: torch.nn.grad
.. py:module:: torch.nn.init
.. py:module:: torch.nn.modules.activation
.. py:module:: torch.nn.modules.adaptive
.. py:module:: torch.nn.modules.batchnorm
.. py:module:: torch.nn.modules.channelshuffle
.. py:module:: torch.nn.modules.container
.. py:module:: torch.nn.modules.conv
.. py:module:: torch.nn.modules.distance
.. py:module:: torch.nn.modules.dropout
.. py:module:: torch.nn.modules.flatten
.. py:module:: torch.nn.modules.fold
.. py:module:: torch.nn.modules.instancenorm
.. py:module:: torch.nn.modules.lazy
.. py:module:: torch.nn.modules.linear
.. py:module:: torch.nn.modules.loss
.. py:module:: torch.nn.modules.module
.. py:module:: torch.nn.modules.normalization
.. py:module:: torch.nn.modules.padding
.. py:module:: torch.nn.modules.pixelshuffle
.. py:module:: torch.nn.modules.pooling
.. py:module:: torch.nn.modules.rnn
.. py:module:: torch.nn.modules.sparse
.. py:module:: torch.nn.modules.transformer
.. py:module:: torch.nn.modules.upsampling
.. py:module:: torch.nn.modules.utils
.. py:module:: torch.nn.parallel.comm
.. py:module:: torch.nn.parallel.distributed
.. py:module:: torch.nn.parallel.parallel_apply
.. py:module:: torch.nn.parallel.replicate
.. py:module:: torch.nn.parallel.scatter_gather
.. py:module:: torch.nn.parameter
.. py:module:: torch.nn.utils.clip_grad
.. py:module:: torch.nn.utils.convert_parameters
.. py:module:: torch.nn.utils.fusion
.. py:module:: torch.nn.utils.init
.. py:module:: torch.nn.utils.memory_format
.. py:module:: torch.nn.utils.parametrizations
.. py:module:: torch.nn.utils.parametrize
.. py:module:: torch.nn.utils.prune
.. py:module:: torch.nn.utils.rnn
