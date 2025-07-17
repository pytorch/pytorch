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

    modules.module.Module
    modules.container.Sequential
    modules.container.ModuleList
    modules.container.ModuleDict
    modules.container.ParameterList
    modules.container.ParameterDict

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

    nn.modules.conv.Conv1d
    nn.modules.conv.Conv2d
    nn.modules.conv.Conv3d
    nn.modules.conv.ConvTranspose1d
    nn.modules.conv.ConvTranspose2d
    nn.modules.conv.ConvTranspose3d
    nn.modules.conv.LazyConv1d
    nn.modules.conv.LazyConv2d
    nn.modules.conv.LazyConv3d
    nn.modules.conv.LazyConvTranspose1d
    nn.modules.conv.LazyConvTranspose2d
    nn.modules.conv.LazyConvTranspose3d
    nn.modules.fold.Unfold
    nn.modules.fold.Fold

Pooling layers
----------------------------------

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    nn.modules.pooling.MaxPool1d
    nn.modules.pooling.MaxPool2d
    nn.modules.pooling.MaxPool3d
    nn.modules.pooling.MaxUnpool1d
    nn.modules.pooling.MaxUnpool2d
    nn.modules.pooling.MaxUnpool3d
    nn.modules.pooling.AvgPool1d
    nn.modules.pooling.AvgPool2d
    nn.modules.pooling.AvgPool3d
    nn.modules.pooling.FractionalMaxPool2d
    nn.modules.pooling.FractionalMaxPool3d
    nn.modules.pooling.LPPool1d
    nn.modules.pooling.LPPool2d
    nn.modules.pooling.LPPool3d
    nn.modules.pooling.AdaptiveMaxPool1d
    nn.modules.pooling.AdaptiveMaxPool2d
    nn.modules.pooling.AdaptiveMaxPool3d
    nn.modules.pooling.AdaptiveAvgPool1d
    nn.modules.pooling.AdaptiveAvgPool2d
    nn.modules.pooling.AdaptiveAvgPool3d

Padding Layers
--------------

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    nn.modules.padding.ReflectionPad1d
    nn.modules.padding.ReflectionPad2d
    nn.modules.padding.ReflectionPad3d
    nn.modules.padding.ReplicationPad1d
    nn.modules.padding.ReplicationPad2d
    nn.modules.padding.ReplicationPad3d
    nn.modules.padding.ZeroPad1d
    nn.modules.padding.ZeroPad2d
    nn.modules.padding.ZeroPad3d
    nn.modules.padding.ConstantPad1d
    nn.modules.padding.ConstantPad2d
    nn.modules.padding.ConstantPad3d
    nn.modules.padding.CircularPad1d
    nn.modules.padding.CircularPad2d
    nn.modules.padding.CircularPad3d

Non-linear Activations (weighted sum, nonlinearity)
---------------------------------------------------

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    nn.modules.activation.ELU
    nn.modules.activation.Hardshrink
    nn.modules.activation.Hardsigmoid
    nn.modules.activation.Hardtanh
    nn.modules.activation.Hardswish
    nn.modules.activation.LeakyReLU
    nn.modules.activation.LogSigmoid
    nn.modules.activation.MultiheadAttention
    nn.modules.activation.PReLU
    nn.modules.activation.ReLU
    nn.modules.activation.ReLU6
    nn.modules.activation.RReLU
    nn.modules.activation.SELU
    nn.modules.activation.CELU
    nn.modules.activation.GELU
    nn.modules.activation.Sigmoid
    nn.modules.activation.SiLU
    nn.modules.activation.Mish
    nn.modules.activation.Softplus
    nn.modules.activation.Softshrink
    nn.modules.activation.Softsign
    nn.modules.activation.Tanh
    nn.modules.activation.Tanhshrink
    nn.modules.activation.Threshold
    nn.modules.activation.GLU

Non-linear Activations (other)
------------------------------

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    nn.modules.activation.Softmin
    nn.modules.activation.Softmax
    nn.modules.activation.Softmax2d
    nn.modules.activation.LogSoftmax
    nn.modules.adaptive.AdaptiveLogSoftmaxWithLoss

Normalization Layers
----------------------------------

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    nn.modules.batchnorm.BatchNorm1d
    nn.modules.batchnorm.BatchNorm2d
    nn.modules.batchnorm.BatchNorm3d
    nn.modules.batchnorm.LazyBatchNorm1d
    nn.modules.batchnorm.LazyBatchNorm2d
    nn.modules.batchnorm.LazyBatchNorm3d
    nn.modules.normalization.GroupNorm
    nn.modules.batchnorm.SyncBatchNorm
    nn.modules.instancenorm.InstanceNorm1d
    nn.modules.instancenorm.InstanceNorm2d
    nn.modules.instancenorm.InstanceNorm3d
    nn.modules.instancenorm.LazyInstanceNorm1d
    nn.modules.instancenorm.LazyInstanceNorm2d
    nn.modules.instancenorm.LazyInstanceNorm3d
    nn.modules.normalization.LayerNorm
    nn.modules.normalization.LocalResponseNorm
    nn.modules.normalization.RMSNorm

Recurrent Layers
----------------

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    nn.modules.rnn.RNNBase
    nn.modules.rnn.RNN
    nn.modules.rnn.LSTM
    nn.modules.rnn.GRU
    nn.modules.rnn.RNNCell
    nn.modules.rnn.LSTMCell
    nn.modules.rnn.GRUCell

Transformer Layers
----------------------------------

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    nn.modules.transformer.Transformer
    nn.modules.transformer.TransformerEncoder
    nn.modules.transformer.TransformerDecoder
    nn.modules.transformer.TransformerEncoderLayer
    nn.modules.transformer.TransformerDecoderLayer

Linear Layers
----------------------------------

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    nn.modules.linear.Identity
    nn.modules.linear.Linear
    nn.modules.linear.Bilinear
    nn.modules.linear.LazyLinear

Dropout Layers
--------------

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    nn.modules.dropout.Dropout
    nn.modules.dropout.Dropout1d
    nn.modules.dropout.Dropout2d
    nn.modules.dropout.Dropout3d
    nn.modules.dropout.AlphaDropout
    nn.modules.dropout.FeatureAlphaDropout

Sparse Layers
-------------

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    nn.modules.sparse.Embedding
    nn.modules.sparse.EmbeddingBag

Distance Functions
------------------

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    nn.modules.distance.CosineSimilarity
    nn.modules.distance.PairwiseDistance

Loss Functions
--------------

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    nn.modules.loss.L1Loss
    nn.modules.loss.MSELoss
    nn.modules.loss.CrossEntropyLoss
    nn.modules.loss.CTCLoss
    nn.modules.loss.NLLLoss
    nn.modules.loss.PoissonNLLLoss
    nn.modules.loss.GaussianNLLLoss
    nn.modules.loss.KLDivLoss
    nn.modules.loss.BCELoss
    nn.modules.loss.BCEWithLogitsLoss
    nn.modules.loss.MarginRankingLoss
    nn.modules.loss.HingeEmbeddingLoss
    nn.modules.loss.MultiLabelMarginLoss
    nn.modules.loss.HuberLoss
    nn.modules.loss.SmoothL1Loss
    nn.modules.loss.SoftMarginLoss
    nn.modules.loss.MultiLabelSoftMarginLoss
    nn.modules.loss.CosineEmbeddingLoss
    nn.modules.loss.MultiMarginLoss
    nn.modules.loss.TripletMarginLoss
    nn.modules.loss.TripletMarginWithDistanceLoss

Vision Layers
----------------

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    nn.modules.pixelshuffle.PixelShuffle
    nn.modules.pixelshuffle.PixelUnshuffle
    nn.modules.upsampling.Upsample
    nn.modules.upsampling.UpsamplingNearest2d
    nn.modules.upsampling.UpsamplingBilinear2d

Shuffle Layers
----------------

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    nn.modules.channelshuffle.ChannelShuffle

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

The following are aliases to their counterparts in ``torch.nn``:

.. currentmodule:: torch
.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    nn.modules.normalization.RMSNorm

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
