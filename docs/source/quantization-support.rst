Quantization Operation coverage
-------------------------------

Quantized Tensors support a limited subset of data manipulation methods of the
regular full-precision tensor. For NN operators included in PyTorch, we
restrict support to:

   1. 8 bit weights (data\_type = qint8)
   2. 8 bit activations (data\_type = quint8)

Note that operator implementations currently only
support per channel quantization for weights of the **conv** and **linear**
operators. Furthermore the minimum and the maximum of the input data is
mapped linearly to the minimum and the maximum of the quantized data
type such that zero is represented with no quantization error.

Additional data types and quantization schemes can be implemented through
the `custom operator mechanism <https://pytorch.org/tutorials/advanced/torch_script_custom_ops.html>`_.

Many operations for quantized tensors are available under the same API as full
float version in ``torch`` or ``torch.nn``. Quantized version of NN modules that
perform re-quantization are available in ``torch.nn.quantized``. Those
operations explicitly take output quantization parameters (scale and zero\_point) in
the operation signature.

In addition, we also support fused versions corresponding to common fusion
patterns that impact quantization at: `torch.nn.intrinsic.quantized`.

For quantization aware training, we support modules prepared for quantization
aware training at `torch.nn.qat` and `torch.nn.intrinsic.qat`

.. end-of-part-included-in-quantization.rst

The following operation list is sufficient to cover typical CNN and RNN models


Quantized ``torch.Tensor`` operations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Operations that are available from the ``torch`` namespace or as methods on
Tensor for quantized tensors:

* :func:`~torch.quantize_per_tensor` - Convert float tensor to quantized tensor
  with per-tensor scale and zero point
* :func:`~torch.quantize_per_channel` - Convert float tensor to quantized
  tensor with per-channel scale and zero point
* View-based operations like :meth:`~torch.Tensor.view`,
  :meth:`~torch.Tensor.as_strided`, :meth:`~torch.Tensor.expand`,
  :meth:`~torch.Tensor.flatten`, :meth:`~torch.Tensor.select`, python-style
  indexing, etc - work as on regular tensor (if quantization is not
  per-channel)
* Comparators
    * :meth:`~torch.Tensor.ne` — Not equal
    * :meth:`~torch.Tensor.eq` — Equal
    * :meth:`~torch.Tensor.ge` — Greater or equal
    * :meth:`~torch.Tensor.le` — Less or equal
    * :meth:`~torch.Tensor.gt` — Greater
    * :meth:`~torch.Tensor.lt` — Less
* :meth:`~torch.Tensor.copy_` — Copies src to self in-place
* :meth:`~torch.Tensor.clone` —  Returns a deep copy of the passed-in tensor
* :meth:`~torch.Tensor.dequantize` — Convert quantized tensor to float tensor
* :meth:`~torch.Tensor.equal` — Compares two tensors, returns true if
  quantization parameters and all integer elements are the same
* :meth:`~torch.Tensor.int_repr` — Prints the underlying integer representation
  of the quantized tensor
* :meth:`~torch.Tensor.max` — Returns the maximum value of the tensor (reduction only)
* :meth:`~torch.Tensor.mean` — Mean function. Supported variants: reduction, dim, out
* :meth:`~torch.Tensor.min` — Returns the minimum value of the tensor (reduction only)
* :meth:`~torch.Tensor.q_scale` — Returns the scale of the per-tensor quantized tensor
* :meth:`~torch.Tensor.q_zero_point` — Returns the zero_point of the per-tensor
  quantized zero point
* :meth:`~torch.Tensor.q_per_channel_scales` — Returns the scales of the
  per-channel quantized tensor
* :meth:`~torch.Tensor.q_per_channel_zero_points` — Returns the zero points of
  the per-channel quantized tensor
* :meth:`~torch.Tensor.q_per_channel_axis` — Returns the channel axis of the
  per-channel quantized tensor
* :meth:`~torch.Tensor.resize_` — In-place resize
* :meth:`~torch.Tensor.sort` — Sorts the tensor
* :meth:`~torch.Tensor.topk` — Returns k largest values of a tensor

``torch.nn.functional``
~~~~~~~~~~~~~~~~~~~~~~~

Basic activations are supported.

* :meth:`~torch.nn.functional.relu` — Rectified linear unit (copy)
* :meth:`~torch.nn.functional.relu_` — Rectified linear unit (inplace)
* :meth:`~torch.nn.functional.elu` - ELU
* :meth:`~torch.nn.functional.max_pool2d` - Maximum pooling
* :meth:`~torch.nn.functional.adaptive_avg_pool2d` - Adaptive average pooling
* :meth:`~torch.nn.functional.avg_pool2d` - Average pooling
* :meth:`~torch.nn.functional.interpolate` - Interpolation
* :meth:`~torch.nn.functional.hardsigmoid` - Hardsigmoid
* :meth:`~torch.nn.functional.hardswish` - Hardswish
* :meth:`~torch.nn.functional.hardtanh` - Hardtanh
* :meth:`~torch.nn.functional.upsample` - Upsampling
* :meth:`~torch.nn.functional.upsample_bilinear` - Bilinear Upsampling
* :meth:`~torch.nn.functional.upsample_nearest` - Upsampling Nearest

``torch.nn.intrinsic``
~~~~~~~~~~~~~~~~~~~~~~

Fused modules are provided for common patterns in CNNs. Combining several
operations together (like convolution and relu) allows for better quantization
accuracy


* `torch.nn.intrinsic` — float versions of the modules, can be swapped with
  quantized version 1 to 1:

  * :class:`~torch.nn.intrinsic.ConvBn1d` — Conv1d + BatchNorm1d
  * :class:`~torch.nn.intrinsic.ConvBn2d` — Conv2d + BatchNorm
  * :class:`~torch.nn.intrinsic.ConvBn3d` — Conv3d + BatchNorm3d
  * :class:`~torch.nn.intrinsic.ConvBnReLU1d` — Conv1d + BatchNorm1d + ReLU
  * :class:`~torch.nn.intrinsic.ConvBnReLU2d` — Conv2d + BatchNorm + ReLU
  * :class:`~torch.nn.intrinsic.ConvBnReLU3d` — Conv3d + BatchNorm3d + ReLU
  * :class:`~torch.nn.intrinsic.ConvReLU1d` — Conv1d + ReLU
  * :class:`~torch.nn.intrinsic.ConvReLU2d` — Conv2d + ReLU
  * :class:`~torch.nn.intrinsic.ConvReLU3d` — Conv3d + ReLU
  * :class:`~torch.nn.intrinsic.LinearReLU` — Linear + ReLU

* `torch.nn.intrinsic.qat` — versions of layers for quantization-aware training:

  * :class:`~torch.nn.intrinsic.qat.ConvBn2d` — Conv2d + BatchNorm
  * :class:`~torch.nn.intrinsic.qat.ConvBn3d` — Conv3d + BatchNorm3d
  * :class:`~torch.nn.intrinsic.qat.ConvBnReLU2d` — Conv2d + BatchNorm + ReLU
  * :class:`~torch.nn.intrinsic.qat.ConvBnReLU3d` — Conv3d + BatchNorm3d + ReLU
  * :class:`~torch.nn.intrinsic.qat.ConvReLU2d` — Conv2d + ReLU
  * :class:`~torch.nn.intrinsic.qat.ConvReLU3d` — Conv3d + ReLU
  * :class:`~torch.nn.intrinsic.qat.LinearReLU` — Linear + ReLU

* `torch.nn.intrinsic.quantized` — quantized version of fused layers for
  inference (no BatchNorm variants as it's usually folded into convolution for
  inference):

  * :class:`~torch.nn.intrinsic.quantized.LinearReLU` — Linear + ReLU
  * :class:`~torch.nn.intrinsic.quantized.ConvReLU1d` — 1D Convolution + ReLU
  * :class:`~torch.nn.intrinsic.quantized.ConvReLU2d` — 2D Convolution + ReLU
  * :class:`~torch.nn.intrinsic.quantized.ConvReLU3d` — 3D Convolution + ReLU

`torch.nn.qat`
~~~~~~~~~~~~~~

Layers for the quantization-aware training

* :class:`~torch.nn.qat.Linear` — Linear (fully-connected) layer
* :class:`~torch.nn.qat.Conv2d` — 2D convolution
* :class:`~torch.nn.qat.Conv3d` — 3D convolution

`torch.quantization`
~~~~~~~~~~~~~~~~~~~~

* Functions for eager mode quantization:

  * :func:`~torch.quantization.add_observer_` — Adds observer for the leaf
    modules (if quantization configuration is provided)
  * :func:`~torch.quantization.add_quant_dequant`— Wraps the leaf child module using :class:`~torch.quantization.QuantWrapper`
  * :func:`~torch.quantization.convert` — Converts float module with
    observers into its quantized counterpart. Must have quantization
    configuration
  * :func:`~torch.quantization.get_observer_dict` — Traverses the module
    children and collects all observers into a ``dict``
  * :func:`~torch.quantization.prepare` — Prepares a copy of a model for
    quantization
  * :func:`~torch.quantization.prepare_qat` — Prepares a copy of a model for
    quantization aware training
  * :func:`~torch.quantization.propagate_qconfig_` — Propagates quantization
    configurations through the module hierarchy and assign them to each leaf
    module
  * :func:`~torch.quantization.quantize` — Function for eager mode post training static quantization
  * :func:`~torch.quantization.quantize_dynamic` — Function for eager mode post training dynamic quantization
  * :func:`~torch.quantization.quantize_qat` — Function for eager mode quantization aware training function
  * :func:`~torch.quantization.swap_module` — Swaps the module with its
    quantized counterpart (if quantizable and if it has an observer)
  * :func:`~torch.quantization.default_eval_fn` — Default evaluation function
    used by the :func:`torch.quantization.quantize`
  * :func:`~torch.quantization.fuse_modules`

* Functions for FX graph mode quantization:
  * :func:`~torch.quantization.quantize_fx.prepare_fx` - Function for preparing the model for post training quantization with FX graph mode quantization
  * :func:`~torch.quantization.quantize_fx.prepare_qat_fx` - Function for preparing the model for quantization aware training with FX graph mode quantization
  * :func:`~torch.quantization.quantize_fx.convert_fx` - Function for converting a prepared model to a quantized model with FX graph mode quantization

* Quantization configurations
    * :class:`~torch.quantization.QConfig` — Quantization configuration class
    * :attr:`~torch.quantization.default_qconfig` — Same as
      ``QConfig(activation=default_observer, weight=default_weight_observer)``
      (See :class:`~torch.quantization.qconfig.QConfig`)
    * :attr:`~torch.quantization.default_qat_qconfig` — Same as
      ``QConfig(activation=default_fake_quant,
      weight=default_weight_fake_quant)`` (See
      :class:`~torch.quantization.qconfig.QConfig`)
    * :attr:`~torch.quantization.default_dynamic_qconfig` — Same as
      ``QConfigDynamic(weight=default_weight_observer)`` (See
      :class:`~torch.quantization.qconfig.QConfigDynamic`)
    * :attr:`~torch.quantization.float16_dynamic_qconfig` — Same as
      ``QConfigDynamic(weight=NoopObserver.with_args(dtype=torch.float16))``
      (See :class:`~torch.quantization.qconfig.QConfigDynamic`)

* Stubs
    * :class:`~torch.quantization.DeQuantStub` - placeholder module for
      dequantize() operation in float-valued models
    * :class:`~torch.quantization.QuantStub` - placeholder module for
      quantize() operation in float-valued models
    * :class:`~torch.quantization.QuantWrapper` — wraps the module to be
      quantized. Inserts the :class:`~torch.quantization.QuantStub` and
    * :class:`~torch.quantization.DeQuantStub`

* Observers for computing the quantization parameters

  * Default Observers. The rest of observers are available from
    ``torch.quantization.observer``:

    * :attr:`~torch.quantization.default_observer` — Same as ``MinMaxObserver.with_args(reduce_range=True)``
    * :attr:`~torch.quantization.default_weight_observer` — Same as ``MinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_tensor_symmetric)``

  * :class:`~torch.quantization.Observer` — Abstract base class for observers
  * :class:`~torch.quantization.MinMaxObserver` — Derives the quantization
    parameters from the running minimum and maximum of the observed tensor inputs
    (per tensor variant)
  * :class:`~torch.quantization.MovingAverageMinMaxObserver` — Derives the
    quantization parameters from the running averages of the minimums and
    maximums of the observed tensor inputs (per tensor variant)
  * :class:`~torch.quantization.PerChannelMinMaxObserver` — Derives the
    quantization parameters from the running minimum and maximum of the observed
    tensor inputs (per channel variant)
  * :class:`~torch.quantization.MovingAveragePerChannelMinMaxObserver` — Derives
    the quantization parameters from the running averages of the minimums and
    maximums of the observed tensor inputs (per channel variant)
  * :class:`~torch.quantization.HistogramObserver` — Derives the quantization
    parameters by creating a histogram of running minimums and maximums.

* Observers that do not compute the quantization parameters:
    * :class:`~torch.quantization.RecordingObserver` — Records all incoming
      tensors. Used for debugging only.
    * :class:`~torch.quantization.NoopObserver` — Pass-through observer. Used
      for situation when there are no quantization parameters (i.e.
      quantization to ``float16``)

* FakeQuantize module
    * :class:`~torch.quantization.FakeQuantize` — Module for simulating the
      quantization/dequantization at training time

`torch.nn.quantized`
~~~~~~~~~~~~~~~~~~~~

Quantized version of standard NN layers.

* :class:`~torch.nn.quantized.Quantize` — Quantization layer, used to
  automatically replace :class:`~torch.quantization.QuantStub`
* :class:`~torch.nn.quantized.DeQuantize` — Dequantization layer, used to
  replace :class:`~torch.quantization.DeQuantStub`
* :class:`~torch.nn.quantized.FloatFunctional` — Wrapper class to make
  stateless float operations stateful so that they can be replaced with
  quantized versions
* :class:`~torch.nn.quantized.QFunctional` — Wrapper class for quantized
  versions of stateless operations like ``torch.add``
* :class:`~torch.nn.quantized.Conv1d` — 1D convolution
* :class:`~torch.nn.quantized.Conv2d` — 2D convolution
* :class:`~torch.nn.quantized.Conv3d` — 3D convolution
* :class:`~torch.nn.quantized.Linear` — Linear (fully-connected) layer
* :class:`~torch.nn.MaxPool2d` — 2D max pooling
* :class:`~torch.nn.quantized.ReLU6` — Rectified linear unit with cut-off at
  quantized representation of 6
* :class:`~torch.nn.quantized.ELU` — ELU
* :class:`~torch.nn.quantized.Hardswish` — Hardswish
* :class:`~torch.nn.quantized.BatchNorm2d` — BatchNorm2d. *Note: this module is usually fused with Conv or Linear. Performance on ARM is not optimized*.
* :class:`~torch.nn.quantized.BatchNorm3d` — BatchNorm3d. *Note: this module is usually fused with Conv or Linear. Performance on ARM is not optimized*.
* :class:`~torch.nn.quantized.LayerNorm` — LayerNorm. *Note: performance on ARM is not optimized*.
* :class:`~torch.nn.quantized.GroupNorm` — GroupNorm. *Note: performance on ARM is not optimized*.
* :class:`~torch.nn.quantized.InstanceNorm1d` — InstanceNorm1d. *Note: performance on ARM is not optimized*.
* :class:`~torch.nn.quantized.InstanceNorm2d` — InstanceNorm2d. *Note: performance on ARM is not optimized*.
* :class:`~torch.nn.quantized.InstanceNorm3d` — InstanceNorm3d. *Note: performance on ARM is not optimized*.

`torch.nn.quantized.dynamic`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Layers used in dynamically quantized models (i.e. quantized only on weights)

* :class:`~torch.nn.quantized.dynamic.Linear` — Linear (fully-connected) layer
* :class:`~torch.nn.quantized.dynamic.LSTM` — Long-Short Term Memory RNN module
* :class:`~torch.nn.quantized.dynamic.LSTMCell` — LSTM Cell
* :class:`~torch.nn.quantized.dynamic.GRUCell` — GRU Cell
* :class:`~torch.nn.quantized.dynamic.RNNCell` — RNN Cell

`torch.nn.quantized.functional`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Functional versions of quantized NN layers (many of them accept explicit
quantization output parameters)

* :func:`~torch.nn.quantized.functional.adaptive_avg_pool2d` — 2D adaptive average pooling
* :func:`~torch.nn.quantized.functional.avg_pool2d` — 2D average pooling
* :func:`~torch.nn.quantized.functional.avg_pool3d` — 3D average pooling
* :func:`~torch.nn.quantized.functional.conv1d` — 1D convolution
* :func:`~torch.nn.quantized.functional.conv2d` — 2D convolution
* :func:`~torch.nn.quantized.functional.conv3d` — 3D convolution
* :func:`~torch.nn.quantized.functional.interpolate` — Down-/up- sampler
* :func:`~torch.nn.quantized.functional.linear` — Linear (fully-connected) op
* :func:`~torch.nn.quantized.functional.max_pool2d` — 2D max pooling
* :func:`~torch.nn.quantized.functional.elu` — ELU
* :func:`~torch.nn.quantized.functional.hardsigmoid` — Hardsigmoid
* :func:`~torch.nn.quantized.functional.hardswish` — Hardswish
* :func:`~torch.nn.quantized.functional.hardtanh` — Hardtanh
* :func:`~torch.nn.quantized.functional.upsample` — Upsampler. Will be
  deprecated in favor of :func:`~torch.nn.quantized.functional.interpolate`
* :func:`~torch.nn.quantized.functional.upsample_bilinear` — Bilinear
  upsampler. Will be deprecated in favor of
* :func:`~torch.nn.quantized.functional.interpolate`
* :func:`~torch.nn.quantized.functional.upsample_nearest` — Nearest neighbor
  upsampler. Will be deprecated in favor of
* :func:`~torch.nn.quantized.functional.interpolate`

Quantized dtypes and quantization schemes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* :attr:`torch.qscheme` — Type to describe the quantization scheme of a tensor.
  Supported types:

  * :attr:`torch.per_tensor_affine` — per tensor, asymmetric
  * :attr:`torch.per_channel_affine` — per channel, asymmetric
  * :attr:`torch.per_tensor_symmetric` — per tensor, symmetric
  * :attr:`torch.per_channel_symmetric` — per tensor, symmetric

* ``torch.dtype`` — Type to describe the data. Supported types:

  * :attr:`torch.quint8` — 8-bit unsigned integer
  * :attr:`torch.qint8` — 8-bit signed integer
  * :attr:`torch.qint32` — 32-bit signed integer
