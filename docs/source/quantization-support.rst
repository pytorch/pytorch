Quantization API Reference
-------------------------------

torch.ao.quantization
~~~~~~~~~~~~~~~~~~~~~

This module contains Eager mode quantization APIs.

.. currentmodule:: torch.ao.quantization

Top level APIs
^^^^^^^^^^^^^^

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    quantize
    quantize_dynamic
    quantize_qat
    prepare
    prepare_qat
    convert

Preparing model for quantization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    fuse_modules
    QuantStub
    DeQuantStub
    QuantWrapper
    add_quant_dequant

Utility functions
^^^^^^^^^^^^^^^^^

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    swap_module
    propagate_qconfig_
    default_eval_fn

torch.ao.quantization.quantize_fx
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This module contains FX graph mode quantization APIs (prototype).

.. currentmodule:: torch.ao.quantization.quantize_fx

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    prepare_fx
    prepare_qat_fx
    convert_fx
    fuse_fx

torch.ao.quantization.qconfig_mapping
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This module contains QConfigMapping for configuring FX graph mode quantization.

.. currentmodule:: torch.ao.quantization.qconfig_mapping

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    QConfigMapping
    get_default_qconfig_mapping
    get_default_qat_qconfig_mapping

torch.ao.quantization.backend_config
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This module contains BackendConfig, a config object that defines how quantization is supported
in a backend. Currently only used by FX Graph Mode Quantization, but we may extend Eager Mode
Quantization to work with this as well.

.. currentmodule:: torch.ao.quantization.backend_config

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    BackendConfig
    BackendPatternConfig
    DTypeConfig
    DTypeWithConstraints
    ObservationType

torch.ao.quantization.fx.custom_config
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This module contains a few CustomConfig classes that's used in both eager mode and FX graph mode quantization


.. currentmodule:: torch.ao.quantization.fx.custom_config

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    FuseCustomConfig
    PrepareCustomConfig
    ConvertCustomConfig
    StandaloneModuleConfigEntry

torch.ao.quantization.pt2e (quantization in pytorch 2.0 export)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: torch.ao.quantization.pt2e
.. automodule:: torch.ao.quantization.pt2e.quantizer
.. automodule:: torch.ao.quantization.pt2e.representation

torch (quantization related functions)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This describes the quantization related functions of the `torch` namespace.

.. currentmodule:: torch

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    quantize_per_tensor
    quantize_per_channel
    dequantize

torch.Tensor (quantization related methods)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Quantized Tensors support a limited subset of data manipulation methods of the
regular full-precision tensor.

.. currentmodule:: torch.Tensor

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    view
    as_strided
    expand
    flatten
    select
    ne
    eq
    ge
    le
    gt
    lt
    copy_
    clone
    dequantize
    equal
    int_repr
    max
    mean
    min
    q_scale
    q_zero_point
    q_per_channel_scales
    q_per_channel_zero_points
    q_per_channel_axis
    resize_
    sort
    topk


torch.ao.quantization.observer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This module contains observers which are used to collect statistics about
the values observed during calibration (PTQ) or training (QAT).

.. currentmodule:: torch.ao.quantization.observer

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    ObserverBase
    MinMaxObserver
    MovingAverageMinMaxObserver
    PerChannelMinMaxObserver
    MovingAveragePerChannelMinMaxObserver
    HistogramObserver
    PlaceholderObserver
    RecordingObserver
    NoopObserver
    get_observer_state_dict
    load_observer_state_dict
    default_observer
    default_placeholder_observer
    default_debug_observer
    default_weight_observer
    default_histogram_observer
    default_per_channel_weight_observer
    default_dynamic_quant_observer
    default_float_qparams_observer

torch.ao.quantization.fake_quantize
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This module implements modules which are used to perform fake quantization
during QAT.

.. currentmodule:: torch.ao.quantization.fake_quantize

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    FakeQuantizeBase
    FakeQuantize
    FixedQParamsFakeQuantize
    FusedMovingAvgObsFakeQuantize
    default_fake_quant
    default_weight_fake_quant
    default_per_channel_weight_fake_quant
    default_histogram_fake_quant
    default_fused_act_fake_quant
    default_fused_wt_fake_quant
    default_fused_per_channel_wt_fake_quant
    disable_fake_quant
    enable_fake_quant
    disable_observer
    enable_observer

torch.ao.quantization.qconfig
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This module defines `QConfig` objects which are used
to configure quantization settings for individual ops.

.. currentmodule:: torch.ao.quantization.qconfig

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    QConfig
    default_qconfig
    default_debug_qconfig
    default_per_channel_qconfig
    default_dynamic_qconfig
    float16_dynamic_qconfig
    float16_static_qconfig
    per_channel_dynamic_qconfig
    float_qparams_weight_only_qconfig
    default_qat_qconfig
    default_weight_only_qconfig
    default_activation_only_qconfig
    default_qat_qconfig_v2

torch.ao.nn.intrinsic
~~~~~~~~~~~~~~~~~~~~~
.. automodule:: torch.ao.nn.intrinsic
.. automodule:: torch.ao.nn.intrinsic.modules

This module implements the combined (fused) modules conv + relu which can
then be quantized.

.. currentmodule:: torch.ao.nn.intrinsic

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    ConvReLU1d
    ConvReLU2d
    ConvReLU3d
    LinearReLU
    ConvBn1d
    ConvBn2d
    ConvBn3d
    ConvBnReLU1d
    ConvBnReLU2d
    ConvBnReLU3d
    BNReLU2d
    BNReLU3d

torch.ao.nn.intrinsic.qat
~~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: torch.ao.nn.intrinsic.qat
.. automodule:: torch.ao.nn.intrinsic.qat.modules


This module implements the versions of those fused operations needed for
quantization aware training.

.. currentmodule:: torch.ao.nn.intrinsic.qat

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    LinearReLU
    ConvBn1d
    ConvBnReLU1d
    ConvBn2d
    ConvBnReLU2d
    ConvReLU2d
    ConvBn3d
    ConvBnReLU3d
    ConvReLU3d
    update_bn_stats
    freeze_bn_stats

torch.ao.nn.intrinsic.quantized
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: torch.ao.nn.intrinsic.quantized
.. automodule:: torch.ao.nn.intrinsic.quantized.modules


This module implements the quantized implementations of fused operations
like conv + relu. No BatchNorm variants as it's usually folded into convolution
for inference.

.. currentmodule:: torch.ao.nn.intrinsic.quantized

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    BNReLU2d
    BNReLU3d
    ConvReLU1d
    ConvReLU2d
    ConvReLU3d
    LinearReLU

torch.ao.nn.intrinsic.quantized.dynamic
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: torch.ao.nn.intrinsic.quantized.dynamic
.. automodule:: torch.ao.nn.intrinsic.quantized.dynamic.modules

This module implements the quantized dynamic implementations of fused operations
like linear + relu.

.. currentmodule:: torch.ao.nn.intrinsic.quantized.dynamic

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    LinearReLU

torch.ao.nn.qat
~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: torch.ao.nn.qat
.. automodule:: torch.ao.nn.qat.modules

This module implements versions of the key nn modules **Conv2d()** and
**Linear()** which run in FP32 but with rounding applied to simulate the
effect of INT8 quantization.

.. currentmodule:: torch.ao.nn.qat

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    Conv2d
    Conv3d
    Linear

torch.ao.nn.qat.dynamic
~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: torch.ao.nn.qat.dynamic
.. automodule:: torch.ao.nn.qat.dynamic.modules

This module implements versions of the key nn modules such as **Linear()**
which run in FP32 but with rounding applied to simulate the effect of INT8
quantization and will be dynamically quantized during inference.

.. currentmodule:: torch.ao.nn.qat.dynamic

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    Linear

torch.ao.nn.quantized
~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: torch.ao.nn.quantized
   :noindex:
.. automodule:: torch.ao.nn.quantized.modules

This module implements the quantized versions of the nn layers such as
~`torch.nn.Conv2d` and `torch.nn.ReLU`.

.. currentmodule:: torch.ao.nn.quantized

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    ReLU6
    Hardswish
    ELU
    LeakyReLU
    Sigmoid
    BatchNorm2d
    BatchNorm3d
    Conv1d
    Conv2d
    Conv3d
    ConvTranspose1d
    ConvTranspose2d
    ConvTranspose3d
    Embedding
    EmbeddingBag
    FloatFunctional
    FXFloatFunctional
    QFunctional
    Linear
    LayerNorm
    GroupNorm
    InstanceNorm1d
    InstanceNorm2d
    InstanceNorm3d

torch.ao.nn.quantized.functional
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: torch.ao.nn.quantized.functional

This module implements the quantized versions of the functional layers such as
~`torch.nn.functional.conv2d` and `torch.nn.functional.relu`. Note:
:meth:`~torch.nn.functional.relu` supports quantized inputs.

.. currentmodule:: torch.ao.nn.quantized.functional

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    avg_pool2d
    avg_pool3d
    adaptive_avg_pool2d
    adaptive_avg_pool3d
    conv1d
    conv2d
    conv3d
    interpolate
    linear
    max_pool1d
    max_pool2d
    celu
    leaky_relu
    hardtanh
    hardswish
    threshold
    elu
    hardsigmoid
    clamp
    upsample
    upsample_bilinear
    upsample_nearest

torch.ao.nn.quantizable
~~~~~~~~~~~~~~~~~~~~~~~

This module implements the quantizable versions of some of the nn layers.
These modules can be used in conjunction with the custom module mechanism,
by providing the ``custom_module_config`` argument to both prepare and convert.

.. currentmodule:: torch.ao.nn.quantizable

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    LSTM
    MultiheadAttention


torch.ao.nn.quantized.dynamic
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: torch.ao.nn.quantized.dynamic
.. automodule:: torch.ao.nn.quantized.dynamic.modules

Dynamically quantized :class:`~torch.nn.Linear`, :class:`~torch.nn.LSTM`,
:class:`~torch.nn.LSTMCell`, :class:`~torch.nn.GRUCell`, and
:class:`~torch.nn.RNNCell`.

.. currentmodule:: torch.ao.nn.quantized.dynamic

.. autosummary::
    :toctree: generated
    :nosignatures:
    :template: classtemplate.rst

    Linear
    LSTM
    GRU
    RNNCell
    LSTMCell
    GRUCell

Quantized dtypes and quantization schemes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Note that operator implementations currently only
support per channel quantization for weights of the **conv** and **linear**
operators. Furthermore, the input data is
mapped linearly to the quantized data and vice versa
as follows:

    .. math::

        \begin{aligned}
            \text{Quantization:}&\\
            &Q_\text{out} = \text{clamp}(x_\text{input}/s+z, Q_\text{min}, Q_\text{max})\\
            \text{Dequantization:}&\\
            &x_\text{out} = (Q_\text{input}-z)*s
        \end{aligned}

where :math:`\text{clamp}(.)` is the same as :func:`~torch.clamp` while the
scale :math:`s` and zero point :math:`z` are then computed
as described in :class:`~torch.ao.quantization.observer.MinMaxObserver`, specifically:

    .. math::

        \begin{aligned}
            \text{if Symmetric:}&\\
            &s = 2 \max(|x_\text{min}|, x_\text{max}) /
                \left( Q_\text{max} - Q_\text{min} \right) \\
            &z = \begin{cases}
                0 & \text{if dtype is qint8} \\
                128 & \text{otherwise}
            \end{cases}\\
            \text{Otherwise:}&\\
                &s = \left( x_\text{max} - x_\text{min}  \right ) /
                    \left( Q_\text{max} - Q_\text{min} \right ) \\
                &z = Q_\text{min} - \text{round}(x_\text{min} / s)
        \end{aligned}

where :math:`[x_\text{min}, x_\text{max}]` denotes the range of the input data while
:math:`Q_\text{min}` and :math:`Q_\text{max}` are respectively the minimum and maximum values of the quantized dtype.

Note that the choice of :math:`s` and :math:`z` implies that zero is represented with no quantization error whenever zero is within
the range of the input data or symmetric quantization is being used.

Additional data types and quantization schemes can be implemented through
the `custom operator mechanism <https://pytorch.org/tutorials/advanced/torch_script_custom_ops.html>`_.

* :attr:`torch.qscheme` — Type to describe the quantization scheme of a tensor.
  Supported types:

  * :attr:`torch.per_tensor_affine` — per tensor, asymmetric
  * :attr:`torch.per_channel_affine` — per channel, asymmetric
  * :attr:`torch.per_tensor_symmetric` — per tensor, symmetric
  * :attr:`torch.per_channel_symmetric` — per channel, symmetric

* ``torch.dtype`` — Type to describe the data. Supported types:

  * :attr:`torch.quint8` — 8-bit unsigned integer
  * :attr:`torch.qint8` — 8-bit signed integer
  * :attr:`torch.qint32` — 32-bit signed integer


.. These modules are missing docs. Adding them here only for tracking
.. automodule:: torch.ao.nn.quantizable.modules
   :noindex:
.. automodule:: torch.ao.nn.quantized.reference
   :noindex:
.. automodule:: torch.ao.nn.quantized.reference.modules
   :noindex:

.. automodule:: torch.nn.quantizable
.. automodule:: torch.nn.qat.dynamic.modules
.. automodule:: torch.nn.qat.modules
.. automodule:: torch.nn.qat
.. automodule:: torch.nn.intrinsic.qat.modules
.. automodule:: torch.nn.quantized.dynamic
.. automodule:: torch.nn.intrinsic
.. automodule:: torch.nn.intrinsic.quantized.modules
.. automodule:: torch.quantization.fx
.. automodule:: torch.nn.intrinsic.quantized.dynamic
.. automodule:: torch.nn.qat.dynamic
.. automodule:: torch.nn.intrinsic.qat
.. automodule:: torch.nn.quantized.modules
.. automodule:: torch.nn.intrinsic.quantized
.. automodule:: torch.nn.quantizable.modules
.. automodule:: torch.nn.quantized
.. automodule:: torch.nn.intrinsic.quantized.dynamic.modules
.. automodule:: torch.nn.quantized.dynamic.modules
.. automodule:: torch.quantization
.. automodule:: torch.nn.intrinsic.modules
