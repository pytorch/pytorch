Quantization
===========================

Introduction to Quantization
----------------------------

Quantization refers to techniques for performing computations and storing tensors at lower bitwidths than
floating point precision. A quantized model executes some or all of the operations on tensors with
integers rather than floating point values. This allows for a more
compact model representation and the use of high performance vectorized
operations on many hardware platforms. PyTorch supports INT8
quantization compared to typical FP32 models allowing for a 4x reduction in the model size and
a 4x reduction in memory bandwidth requirements.  Hardware support for  INT8 computations
is typically 2 to 4 times faster compared to FP32 compute. Quantization is primarily a technique
to speed up inference and only the forward pass is supported for quantized operators.

PyTorch supports multiple approaches to quantizing a deep learning model. In most cases the model is trained
in FP32 and then the model is converted to INT8. In addition, PyTorch also supports quantization aware
training, which models quantization errors in both the forward and backward passes using fake-quantization
modules. Note that the entire computation is carried out in floating point. At the end of quantization aware
training, PyTorch provides conversion functions to convert the trained model into lower precision.

At lower level, PyTorch provides a way to represent quantized tensors and
perform operations with them. They can be used to directly construct models that
perform all or part of the computation in lower precision. Higher-level APIs are
provided that incorporate typical workflows of converting FP32 model to lower
precision with minimal accuracy loss.

Quantized Tensors
---------------------------------------

PyTorch supports both per tensor and per channel asymmetric linear
quantization. Per tensor means that all the values within the tensor are
scaled the same way. Per channel means that for each dimension, typically
the channel dimension of a tensor, the values
in the tensor are scaled and offset by a different value (effectively
the scale and offset become vectors). This allows for lesser error in converting tensors
to quantized values.

The mapping is performed by converting the floating point tensors using

.. image:: math-quantizer-equation.png
   :width: 40%

Note that, we ensure that zero in floating point is represented with no error after quantization,
thereby ensuring that operations like padding do not cause additional quantization error.

In order to do quantization in PyTorch, we need to be able to represent
quantized data in Tensors. A Quantized Tensor allows for storing
quantized data (represented as int8/uint8/int32) along with quantization
parameters like scale and zero\_point. Quantized Tensors allow for many
useful operations making quantized arithmetic easy, in addition to
allowing for serialization of data in a quantized format.

Operation coverage
------------------

Quantized Tensors support a limited subset of data manipulation methods of the regular
full-precision tensor. (see list below)

For NN operators included in PyTorch, we restrict support to:

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

In addition, we also support fused versions corresponding to common fusion patterns that impact quantization at:
torch.nn.intrinsic.quantized.

For quantization aware training, we support modules prepared for quantization aware training at
torch.nn.qat and torch.nn.intrinsic.qat

Current quantized operation list is sufficient to cover typical CNN and RNN
models:


Quantized ``torch.Tensor`` operations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Operations that are available from the ``torch`` namespace or as methods on Tensor for quantized tensors:

* :func:`~torch.quantize_per_tensor` - Convert float tensor to quantized tensor with per-tensor scale and zero point
* :func:`~torch.quantize_per_channel` - Convert float tensor to quantized tensor with per-channel scale and zero point
* View-based operations like :meth:`~torch.Tensor.view`, :meth:`~torch.Tensor.as_strided`, :meth:`~torch.Tensor.expand`, :meth:`~torch.Tensor.flatten`, :meth:`~torch.Tensor.slice`, python-style indexing, etc - work as on regular tensor (if quantization is not per-channel)
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
* :meth:`~torch.Tensor.equal` — Compares two tensors, returns true if quantization parameters and all integer elements are the same
* :meth:`~torch.Tensor.int_repr` — Prints the underlying integer representation of the quantized tensor
* :meth:`~torch.Tensor.max` — Returns the maximum value of the tensor (reduction only)
* :meth:`~torch.Tensor.mean` — Mean function. Supported variants: reduction, dim, out
* :meth:`~torch.Tensor.min` — Returns the minimum value of the tensor (reduction only)
* :meth:`~torch.Tensor.q_scale` — Returns the scale of the per-tensor quantized tensor
* :meth:`~torch.Tensor.q_zero_point` — Returns the zero_point of the per-tensor quantized zero point
* :meth:`~torch.Tensor.q_per_channel_scales` — Returns the scales of the per-channel quantized tensor
* :meth:`~torch.Tensor.q_per_channel_zero_points` — Returns the zero points of the per-channel quantized tensor
* :meth:`~torch.Tensor.q_per_channel_axis` — Returns the channel axis of the per-channel quantized tensor
* :meth:`~torch.Tensor.relu` — Rectified linear unit (copy)
* :meth:`~torch.Tensor.relu_` — Rectified linear unit (inplace)
* :meth:`~torch.Tensor.resize_` — In-place resize
* :meth:`~torch.Tensor.sort` — Sorts the tensor
* :meth:`~torch.Tensor.topk` — Returns k largest values of a tensor


``torch.nn.intrinsic``
~~~~~~~~~~~~~~~~~~~~~~

Fused modules are provided for common patterns in CNNs. Combining several operations together (like convolution and relu) allows for better quantization accuracy

* ``torch.nn.intrinsic`` — float versions of the modules, can be swapped with quantized version 1 to 1
    * :class:`~torch.nn.intrinsic.ConvBn2d` — Conv2d + BatchNorm
    * :class:`~torch.nn.intrinsic.ConvBnReLU2d` — Conv2d + BatchNorm + ReLU
    * :class:`~torch.nn.intrinsic.ConvReLU2d` — Conv2d + Relu
    * :class:`~torch.nn.intrinsic.LinearReLU` — Linear + ReLU
* ``torch.nn.intrinsic.qat`` — versions of layers for quantization-aware training
    * :class:`~torch.nn.intrinsic.qat.ConvBn2d` — Conv2d + BatchNorm
    * :class:`~torch.nn.intrinsic.qat.ConvBnReLU2d` — Conv2d + BatchNorm + ReLU
    * :class:`~torch.nn.intrinsic.qat.ConvReLU2d` — Conv2d + ReLU
    * :class:`~torch.nn.intrinsic.qat.LinearReLU` — Linear + ReLU
* ``torch.nn.intrinsic.quantized`` — quantized version of fused layers for inference (no BatchNorm variants as it's usually folded into convolution for inference)
    * :class:`~torch.nn.intrinsic.quantized.LinearReLU` — Linear + ReLU
    * :class:`~torch.nn.intrinsic.quantized.ConvReLU2d` — 2D Convolution + ReLU

``torch.nn.qat``
~~~~~~~~~~~~~~~~

Layers for the quantization-aware training

* :class:`~torch.nn.qat.Linear` — Linear (fully-connected) layer
* :class:`~torch.nn.qat.Conv2d` — 2D convolution

``torch.quantization``
~~~~~~~~~~~~~~~~~~~~~~

* Functions for quantization
    * :func:`~torch.quantization.add_observer_` — Adds observer for the leaf modules (if quantization configuration is provided)
    * :func:`~torch.quantization.add_quant_dequant`— Wraps the leaf child module using :class:`~torch.quantization.QuantWrapper`
    * :func:`~torch.quantization.convert` — Converts float module with observers into its quantized counterpart. Must have quantization configuration
    * :func:`~torch.quantization.get_observer_dict` — Traverses the module children and collects all observers into a ``dict``
    * :func:`~torch.quantization.prepare` — Prepares a copy of a model for quantization
    * :func:`~torch.quantization.prepare_qat` — Prepares a copy of a model for quantization aware training
    * :func:`~torch.quantization.propagate_qconfig_` — Propagates quantization configurations through the module hierarchy and assign them to each leaf module
    * :func:`~torch.quantization.quantize` — Converts a float module to quantized version
    * :func:`~torch.quantization.quantize_dynamic` — Converts a float module to dynamically quantized version
    * :func:`~torch.quantization.quantize_qat`— Converts a float module to quantized version used in quantization aware training
    * :func:`~torch.quantization.swap_module` — Swaps the module with its quantized counterpart (if quantizable and if it has an observer)
* :func:`~torch.quantization.default_eval_fn` — Default evaluation function used by the :func:`torch.quantization.quantize`
* :func:`~torch.quantization.fuse_modules`
* :class:`~torch.quantization.FakeQuantize` — Module for simulating the quantization/dequantization at training time
* Default Observers. The rest of observers are available from ``torch.quantization.observer``
    * :attr:`~torch.quantization.default_observer` — Same as ``MinMaxObserver.with_args(reduce_range=True)``
    * :attr:`~torch.quantization.default_weight_observer` — Same as ``MinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_tensor_symmetric)``
    * :class:`~torch.quantization.Observer` — Abstract base class for observers
* Quantization configurations
    * :class:`~torch.quantization.QConfig` — Quantization configuration class
    * :attr:`~torch.quantization.default_qconfig` — Same as ``QConfig(activation=default_observer, weight=default_weight_observer)`` (See :class:`~torch.quantization.QConfig.QConfig`)
    * :attr:`~torch.quantization.default_qat_qconfig` — Same as ``QConfig(activation=default_fake_quant, weight=default_weight_fake_quant)`` (See :class:`~torch.quantization.QConfig.QConfig`)
    * :attr:`~torch.quantization.default_dynamic_qconfig` — Same as ``QConfigDynamic(weight=default_weight_observer)`` (See :class:`~torch.quantization.QConfig.QConfigDynamic`)
    * :attr:`~torch.quantization.float16_dynamic_qconfig` — Same as ``QConfigDynamic(weight=NoopObserver.with_args(dtype=torch.float16))`` (See :class:`~torch.quantization.QConfig.QConfigDynamic`)
* Stubs
    * :class:`~torch.quantization.DeQuantStub` - placeholder module for dequantize() operation in float-valued models
    * :class:`~torch.quantization.QuantStub` - placeholder module for quantize() operation in float-valued models
    * :class:`~torch.quantization.QuantWrapper` — wraps the module to be quantized. Inserts the :class:`~torch.quantization.QuantStub` and :class:`~torch.quantization.DeQuantStub`

Observers for computing the quantization parameters

* :class:`~torch.quantization.MinMaxObserver` — Derives the quantization parameters from the running minimum and maximum of the observed tensor inputs (per tensor variant)
* :class:`~torch.quantization.MovingAverageObserver` — Derives the quantization parameters from the running averages of the minimums and maximums of the observed tensor inputs (per tensor variant)
* :class:`~torch.quantization.PerChannelMinMaxObserver`— Derives the quantization parameters from the running minimum and maximum of the observed tensor inputs (per channel variant)
* :class:`~torch.quantization.MovingAveragePerChannelMinMaxObserver` — Derives the quantization parameters from the running averages of the minimums and maximums of the observed tensor inputs (per channel variant)
* :class:`~torch.quantization.HistogramObserver` — Derives the quantization parameters by creating a histogram of running minimums and maximums.
* Observers that do not compute the quantization parameters:
    * :class:`~torch.quantization.RecordingObserver` — Records all incoming tensors. Used for debugging only.
    * :class:`~torch.quantization.NoopObserver` — Pass-through observer. Used for situation when there are no quantization parameters (i.e. quantization to ``float16``)

``torch.nn.quantized``
~~~~~~~~~~~~~~~~~~~~~~

Quantized version of standard NN layers.

* :class:`~torch.nn.quantized.Quantize` — Quantization layer, used to automatically replace :class:`~torch.quantization.QuantStub`
* :class:`~torch.nn.quantized.DeQuantize` — Dequantization layer, used to replace :class:`~torch.quantization.DeQuantStub`
* :class:`~torch.nn.quantized.FloatFunctional` — Wrapper class to make stateless float operations stateful so that they can be replaced with quantized versions
* :class:`~torch.nn.quantized.QFunctional` — Wrapper class for quantized versions of stateless operations like ```torch.add``
* :class:`~torch.nn.quantized.Conv2d` — 2D convolution
* :class:`~torch.nn.quantized.Linear` — Linear (fully-connected) layer
* :class:`~torch.nn.MaxPool2d` — 2D max pooling
* :class:`~torch.nn.quantized.ReLU` — Rectified linear unit
* :class:`~torch.nn.quantized.ReLU6` — Rectified linear unit with cut-off at quantized representation of 6

``torch.nn.quantized.dynamic``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Layers used in dynamically quantized models (i.e. quantized only on weights)

* :class:`~torch.nn.quantized.dynamic.Linear` — Linear (fully-connected) layer
* :class:`~torch.nn.quantized.dynamic.LSTM` — Long-Short Term Memory RNN module

``torch.nn.quantized.functional``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Functional versions of quantized NN layers (many of them accept explicit quantization output parameters)

* :func:`~torch.nn.quantized.functional.adaptive_avg_pool2d` — 2D adaptive average pooling
* :func:`~torch.nn.quantized.functional.avg_pool2d` — 2D average pooling
* :func:`~torch.nn.quantized.functional.conv2d` — 2D convolution
* :func:`~torch.nn.quantized.functional.interpolate` — Down-/up- sampler
* :func:`~torch.nn.quantized.functional.linear` — Linear (fully-connected) op
* :func:`~torch.nn.quantized.functional.max_pool2d` — 2D max pooling
* :func:`~torch.nn.quantized.functional.relu` — Rectified linear unit
* :func:`~torch.nn.quantized.functional.upsample` — Upsampler. Will be deprecated in favor of :func:`~torch.nn.quantized.functional.interpolate`
* :func:`~torch.nn.quantized.functional.upsample_bilinear` — Bilenear upsampler. Will be deprecated in favor of :func:`~torch.nn.quantized.functional.interpolate`
* :func:`~torch.nn.quantized.functional.upsample_nearest` — Nearest neighbor upsampler. Will be deprecated in favor of :func:`~torch.nn.quantized.functional.interpolate`

Quantized dtypes and quantization schemes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* :attr:`torch.qscheme` — Type to describe the quantization scheme of a tensor. Supported types:
    * :attr:`torch.per_tensor_affine` — per tensor, asymmetric
    * :attr:`torch.per_channel_affine` — per channel, asymmetric
    * :attr:`torch.per_tensor_symmetric` — per tensor, symmetric
    * :attr:`torch.per_channel_symmetric` — per tensor, symmetric
* ``torch.dtype`` — Type to describe the data. Supported types:
    * :attr:`torch.quint8` — 8-bit unsigned integer
    * :attr:`torch.qint8` — 8-bit signed integer
    * :attr:`torch.qint32` — 32-bit signed integer



Quantization Workflows
----------------------

PyTorch provides three approaches to quantize models.

1. Post Training Dynamic Quantization: This is the simplest to apply form of
   quantization where the weights are quantized ahead of time but the
   activations are dynamically quantized  during inference. This is used
   for situations where the model execution time is dominated by loading
   weights from memory rather than computing the matrix multiplications.
   This is true for for LSTM and Transformer type models with small
   batch size. Applying dynamic quantization to a whole model can be
   done with a single call to :func:`torch.quantization.quantize_dynamic()`.
   See the `quantization tutorials <https://pytorch.org/tutorials/#quantization-experimental>`_
2. Post Training Static Quantization: This is the most commonly used form of
   quantization where the weights are quantized ahead of time and the
   scale factor and bias for the activation tensors is pre-computed
   based on observing the behavior of the model during a calibration
   process. Post Training Quantization is typically when both memory bandwidth and compute
   savings are important with CNNs being a typical use case.
   The general process for doing post training quantization is:



   1. Prepare the model:
      a. Specify where the activations are quantized and dequantized explicitly by adding QuantStub and DeQuantStub modules.
      b. Ensure that modules are not reused.
      c. Convert any operations that require requantization into modules
   2. Fuse operations like conv + relu or conv+batchnorm + relu together to improve both model accuracy and performance.

   3. Specify the configuration of the quantization methods \'97 such as
      selecting symmetric or asymmetric quantization and MinMax or
      L2Norm calibration techniques.
   4. Use the :func:`torch.quantization.prepare` to insert modules
      that will observe activation tensors during calibration
   5. Calibrate the model by running inference against a calibration
      dataset
   6. Finally, convert the model itself with the
      torch.quantization.convert() method. This does several things: it
      quantizes the weights, computes and stores the scale and bias
      value to be used each activation tensor, and replaces key
      operators quantized implementations.

   See the `quantization tutorials <https://pytorch.org/tutorials/#quantization_experimental>`_


3. Quantization Aware Training: In the rare cases where post training
   quantization does not provide adequate accuracy training can be done
   with simulated quantization using the :class:`torch.quantization.FakeQuantize`. Computations
   will take place in FP32 but with values clamped and rounded to
   simulate the effects of INT8 quantization. The sequence of steps is
   very similar.


   1. Steps (1) and (2) are identical.

   3. Specify the configuration of the fake quantization methods \'97 such as
      selecting symmetric or asymmetric quantization and MinMax or Moving Average
      or L2Norm calibration techniques.
   4. Use the :func:`torch.quantization.prepare_qat` to insert modules
      that will simulate quantization during training.
   5. Train or fine tune the model.
   6. Identical to step (6) for post training quantization

   See the `quantization tutorials <https://pytorch.org/tutorials/#quantization_experimental>`_


While default implementations of observers to select the scale factor and bias
based on observed tensor data are provided, developers can provide their own
quantization functions. Quantization can be applied selectively to different
parts of the model or configured differently for different parts of the model.

We also provide support for per channel quantization for **conv2d()**
and **linear()**

Quantization workflows work by adding (e.g. adding observers as
``.observer`` submodule) or replacing (e.g. converting ``nn.Conv2d`` to
``nn.quantized.Conv2d``) submodules in the model's module hierarchy. It
means that the model stays a regular ``nn.Module``-based instance throughout the
process and thus can work with the rest of PyTorch APIs.


Model Preparation for Quantization
----------------------------------

It is necessary to currently make some modifications to the model definition
prior to quantization. This is because currently quantization works on a module
by module basis. Specifically, for all quantization techniques, the user needs to:

1. Convert any operations that require output requantization (and thus have additional parameters) from functionals to module form.
2. Specify which parts of the model need to be quantized either by assigning ```.qconfig`` attributes on submodules or by specifying ``qconfig_dict``

For static quantization techniques which quantize activations, the user needs to do the following in addition:

1. Specify where activations are quantized and de-quantized. This is done using :class:`~torch.quantization.QuantStub` and :class:`~torch.quantization.DeQuantStub` modules.
2. Use :class:`torch.nn.quantized.FloatFunctional` to wrap tensor operations that require special handling for quantization into modules. Examples
   are operations like ``add`` and ``cat`` which require special handling to determine output quantization parameters.
3. Fuse modules: combine operations/modules into a single module to obtain higher accuracy and performance. This is done using the
   :func:`torch.quantization.fuse_modules` API, which takes in lists of modules to be fused. We currently support the following fusions:
   [Conv, Relu], [Conv, BatchNorm], [Conv, BatchNorm, Relu], [Linear, Relu]


torch.quantization
---------------------------
.. automodule:: torch.quantization

This module implements the functions you call
directly to convert your model from FP32 to quantized form. For
example the :func:`~torch.quantization.prepare` is used in post training
quantization to prepares your model for the calibration step and
:func:`~torch.quantization.convert` actually converts the weights to int8 and
replaces the operations with their quantized counterparts. There are
other helper functions for things like quantizing the input to your
model and performing critical fusions like conv+relu.

Top-level quantization APIs
~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: quantize
.. autofunction:: quantize_dynamic
.. autofunction:: quantize_qat
.. autofunction:: prepare
.. autofunction:: prepare_qat
.. autofunction:: convert
.. autoclass:: QConfig
.. autoclass:: QConfigDynamic
.. autoattr:: default_qconfig

Preparing model for quantization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: fuse_modules
.. autoclass:: QuantStub
.. autoclass:: DeQuantStub
.. autoclass:: QuantWrapper
.. autofunction:: add_quant_dequant

Utility functions
~~~~~~~~~~~~~~~~~
.. autofunction:: add_observer_
.. autofunction:: swap_module
.. autofunction:: propagate_qconfig_
.. autofunction:: default_eval_fn

Observers
~~~~~~~~~~~~~~~
.. autoclass:: Observer
    :members:
.. autoclass:: MinMaxObserver
.. autoclass:: MovingAverageObserver
.. autoclass:: PerChannelMinMaxObserver
.. autoclass:: MovingAveragePerChannelMinMaxObserver
.. autoclass:: HistogramObserver
.. autoclass:: FakeQuantize
.. autoclass:: NoopObserver

Debugging utilities
~~~~~~~~~~~~~~~~~~~
.. autofunction:: get_observer_dict
.. autoclass:: RecordingObserver

torch.nn.instrinsic
--------------------------------

This module implements the combined (fused) modules conv + relu which can be then quantized.

.. automodule:: torch.nn.intrinsic

ConvBn2d
~~~~~~~~~~~~~~~
.. autoclass:: ConvBn2d
    :members:

ConvBnReLU2d
~~~~~~~~~~~~~~~
.. autoclass:: ConvBnReLU2d
    :members:

ConvReLU2d
~~~~~~~~~~~~~~~
.. autoclass:: ConvReLU2d
    :members:

LinearReLU
~~~~~~~~~~~~~~~
.. autoclass:: LinearReLU
    :members:

torch.nn.instrinsic.qat
--------------------------------

This module implements the versions of those fused operations needed for quantization aware training.

.. automodule:: torch.nn.intrinsic.qat

ConvBn2d
~~~~~~~~~~~~~~~
.. autoclass:: ConvBn2d
    :members:

ConvBnReLU2d
~~~~~~~~~~~~~~~
.. autoclass:: ConvBnReLU2d
    :members:

ConvReLU2d
~~~~~~~~~~~~~~~
.. autoclass:: ConvReLU2d
    :members:

LinearReLU
~~~~~~~~~~~~~~~
.. autoclass:: LinearReLU
    :members:

torch.nn.intrinsic.quantized
--------------------------------------

This module implements the quantized implementations of fused operations like conv + relu.

.. automodule:: torch.nn.intrinsic.quantized

ConvReLU2d
~~~~~~~~~~~~~~~
.. autoclass:: ConvReLU2d
    :members:

LinearReLU
~~~~~~~~~~~~~~~
.. autoclass:: LinearReLU
    :members:

torch.nn.qat
---------------------------

This module implements versions of the key nn modules **Conv2d()** and **Linear()** which
run in FP32 but with rounding applied to simulate the effect of INT8 quantization.

.. automodule:: torch.nn.qat

Conv2d
~~~~~~~~~~~~~~~
.. autoclass:: Conv2d
    :members:

Linear
~~~~~~~~~~~~~~~
.. autoclass:: Linear
    :members:


torch.nn.quantized
----------------------------

This module implements the quantized versions of the nn layers such as **Conv2d** and **ReLU**.

Functional interface
~~~~~~~~~~~~~~~~~~~~
.. automodule:: torch.nn.quantized.functional

.. autofunction:: relu
.. autofunction:: linear
.. autofunction:: conv2d
.. autofunction:: max_pool2d

.. automodule:: torch.nn.quantized

ReLU
~~~~~~~~~~~~~~~
.. autoclass:: ReLU
    :members:

ReLU6
~~~~~~~~~~~~~~~
.. autoclass:: ReLU6
    :members:

Conv2d
~~~~~~~~~~~~~~~
.. autoclass:: Conv2d
    :members:

FloatFunctional
~~~~~~~~~~~~~~~
.. autoclass:: FloatFunctional
    :members:

QFunctional
~~~~~~~~~~~~~~~
.. autoclass:: QFunctional
    :members:

Quantize
~~~~~~~~~~~~~~~
.. autoclass:: Quantize
    :members:

DeQuantize
~~~~~~~~~~~~~~~
.. autoclass:: DeQuantize
    :members:

Linear
~~~~~~~~~~~~~~~
.. autoclass:: Linear
    :members:

torch.nn.quantized.dynamic
----------------------------

.. automodule:: torch.nn.quantized.dynamic

Linear
~~~~~~~~~~~~~~~
.. autoclass:: Linear
    :members:

LSTM
~~~~~~~~~~~~~~~
.. autoclass:: LSTM
    :members:
