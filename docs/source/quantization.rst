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

Quantized Representation and Operations
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

Note that, we ensure that zero in floating point is repsresented with no error after quantization,
thereby ensuring that operations like padding do not cause additional quantization error.

Quantized Tensor
~~~~~~~~~~~~~~~~

In order to do quantization in PyTorch, we need to be able to represent
quantized data in Tensors. A Quantized Tensor allows for storing
quantized data (represented as int8/uint8/int32) along with quantization
parameters like scale and zero\\_point. Quantized Tensors allow for many
useful operations making quantized arithmetic easy, in addition to
allowing for serialization of data in a quantized format.

Operation coverage
~~~~~~~~~~~~~~~~~~

Quantized Tensors support a limited subset of data manipulation methods of the regular
full-precision tensor. (see list below)

For NN operators included in PyTorch, we restrict support to:

   1. 8 bit weights (data\\_type = qint8)
   2. 8 bit activations (data\\_type = quint8)

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
operations explicitly take output quantization parameters (scale and bias) in
the operation signature.

In addition, we also support fused versions corresponding to common fusion patterns that impact quantization at:
torch.nn.intrinsic.quantized.

For quantization aware training, we support modules prepared for quantization aware training at
torch.nn.qat and torch.nn.intrinsic.qat

Current quantized operation list is sufficient to cover typical CNN and RNN
models:


*torch*

Operations that are available from the ``torch`` namespace

* **as_strided** — Returns a view of the input tensor with the given size and stride
* Comparators
    * **ne** — Not equal
    * **eq** — Equal
    * **ge** — Greater or equal
    * **le** — Less or equal
    * **gt** — Greater
    * **lt** — Less
* **copy_** — Copies src to self in-place
* **clone** —  Returns a deep copy of the passed-in tensor
* **dequantize** — Convert quantized tensor to float tensor
* **equal** — Compares two tensors, returns true if quantization parameters and all integer elements are the same
* **int_repr** — Prints the underlying integer representation of the quantized tensor
* **max** — Returns the maximum value of the tensor (reduction only)
* **mean** — Mean function. Supported variants: reduction, dim, out
* **min** — Returns the minimum value of the tensor (reduction only)
* **q_scale** — Returns the scale of the per-tensor quantized tensor
* **q_zero_point** — Returns the zero_point of the per-tensor quantized zero point
* **q_per_channel_scales** — Returns the scales of the per-channel quantized tensor
* **q_per_channel_zero_points** — Returns the zero points of the per-channel quantized tensor
* **q_per_channel_axis** — Returns the channel axis of the per-channel quantized tensor
* **quantized_max_pool2d** — Quantized 2D max pooling
* **quantize_per_tensor** - convert float tensor to quantized tensor with per-tensor scale and zero point
* **quantize_per_channel** - convert float tensor to quantized tensor with per-channel scale and zero point
* **relu** — Rectified linear unit (copy)
* **relu_** — Rectified linear unit (inplace)
* **sort** — Sorts the tensor
* **topk** — Returns k largest values of a tensor

**torch.nn.intrinsic**

Fused modules

* **torch.nn.intrinsic** — Post-training quantization module level layers. same as ``torch.nn.intrinsic.modules`` and ``torch.nn.intrinsic.module.fused``
    * **ConvBn2d** — 2D Convolution + BatchNorm
    * **ConvBnReLU2d** — 2D Convolution + BatchNorm + ReLU
    * **ConvReLU2d** — 2D Convolution + Relu
    * **LinearReLU** — Linear + ReLU
* **torch.nn.intrinsic.qat** — Layers for quantization-aware training, same as ``torch.nn.intrinsic.qat.modules``
    * **ConvBn2d** — 2D Convolution + BatchNorm
    * **ConvBnReLU2d** — 2D Convolution + BatchNorm + ReLU
    * **ConvReLU2d** — 2D Convolution + ReLU
    * **LinearReLU** — Linear + ReLU
* **torch.nn.intrinsic.quantized** — Quantized representations of the fused layers
    * **LinearReLU** — Linear + ReLU
    * **ConvReLU2d** — 2D Convolution + ReLU

**torch.nn.qat**

Layers for the quantization-aware training

* **Linear** — Linear (fully-connected) layer
* **Conv2d** — 2D convolution

**torch.quantization**

* Functions for quantization
    * **add_observer_** — Adds observer for the leaf modules (if quantization configuration is provided)
    * **add_quant_dequant**— Wraps the leaf child module using :class:`~torch.quantization.QuantWrapper`
    * **convert** — Converts float module with observers into its quantized counterpart. Must have quantization configuration
    * **get_observer_dict** — Traverses the module children and collects all observers into a ``dict``
    * **prepare** — Prepares a copy of a model for quantization
    * **prepare_qat** — Prepares a copy of a model for quantization aware training
    * **propagate_qconfig_** — Propagates quantization configurations through the module hierarchy and assign them to each leaf module
    * **quantize** — Converts a float module to quantized version
    * **quantize_dynamic** — Converts a float module to dynamically quantized version
    * **quantize_qat**— Converts a float module to quantized version used in quantization aware training
    * **swap_module** — Swaps the module with its quantized counterpart (if quantizable and if it has an observer)
* **default_eval_fn** — Default evaluation function used by the :func:`torch.quantization.quantize`
* **fuse_modules**
* **FakeQuantize** — Module for simulating the quantization/dequantization at training time
* Observers. The rest of observers are available from ``torch.quantization.oobserver``
    * **default_observer** — Same as ``MinMaxObserver.with_args(reduce_range=True)``
    * **default_weight_observer** — Same as ``MinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_tensor_symmetric)``
    * **ObserverBase** — Abstract base class for observers
* Quantization configurations
    * **QConfig** — Quantization configuration class
    * **default_qconfig** — Same as ``QConfig(activation=default_observer, weight=default_weight_observer)`` (See :class:`~torch.quantization.QConfig.QConfig`)
    * **default_qat_qconfig** — Same as ``QConfig(activation=default_fake_quant, weight=default_weight_fake_quant)`` (See :class:`~torch.quantization.QConfig.QConfig`)
    * **default_dynamic_qconfig** — Same as ``QConfigDynamic(weight=default_weight_observer)`` (See :class:`~torch.quantization.QConfig.QConfigDynamic`)
    * **float16_dynamic_qconfig** — Same as ``QConfigDynamic(weight=NoopObserver.with_args(dtype=torch.float16))`` (See :class:`~torch.quantization.QConfig.QConfigDynamic`)
* Stubs
    * **DeQuantStub** - placeholder module for dequantize() operation in float-valued models
    * **QuantStub** - placeholder module for quantize() operation in float-valued models
    * **QuantWrapper** — wraps the module to be quantized. Inserts the :class:`~torch.quantization.QuantStub` and :class:`~torch.quantization.DeQuantStub`

**torch.quantization.observer**

Observers for computing the quantization parameters

* **default_observer** — Same as ``MinMaxObserver.with_args(reduce_range=True)``
* **default_weight_observer** — Same as ``MinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_tensor_symmetric)``
* **ObserverBase** — Abstract base class for observers
* **MinMaxObserver** — Derives the quantization parameters from the running minimum and maximum of the observed tensor inputs (per tensor variant)
* **MovingAverageObserver** — Derives the quantization parameters from the running averages of the minimums and maximums of the observed tensor inputs (per tensor variant)
* **PerChannelMinMaxObserver**— Derives the quantization parameters from the running minimum and maximum of the observed tensor inputs (per channel variant)
* **MovingAveragePerChannelMinMaxObserver** — Derives the quantization parameters from the running averages of the minimums and maximums of the observed tensor inputs (per channel variant)
* **HistogramObserver** — Derives the quantization parameters by creating a histogram of running minimums and maximums.
* Observers that do not compute the quantization parameters:
    * **RecordingObserver** — Records all incoming tensors. Used for debugging only.
    * **NoopObserver** — Pass-through observer. Used for quantization of the modules that do not have quantization paramters (i.e. quantization to ``float16``)

**torch.nn.quantized**

Layers under the ``torch.nn.quantized`` namespace.

* **Quantize** — Quantization layer, used to automatically replace torch.quantization.QuantizeStub
* **DeQuantize** — Dequantization layer, used to replace torch.quantization.QuantizeStub
* **FloatFunctional** — Wrapper class to make stateless float operations stateful so that they can be replaced with quantized versions
* **QFunctional** — Wrapper class for quantized versions of stateless operations like torch.add
* **Conv2d** — 2D convolution
* **Linear** — Linear (fully-connected) layer
* **MaxPool2d** — 2D max pooling 
* **ReLU** — Rectified linear unit
* **ReLU6** — Rectified linear unit with cut-off at quantized representation of 6

**torch.nn.quantized.dynamic**

Layers used in dynamically quantized models

* **Linear** — Linear (fully-connected) layer
* **LSTM** — Long-Short Term Memory RNN module 

**torch.nn.quantized.functional**

Functional operations under ``torch.nn.quantized.functional``

* **adaptive_avg_pool2d** — 2D adaptive average pooling
* **avg_pool2d** — 2D average pooling
* **conv2d** — 2D convolution
* **interpolate** — Down-/up- sampler
* **linear** — Linear (fully-connected) op
* **max_pool2d** — 2D max pooling
* **relu** — Rectified linear unit
* **upsample** — Upsampler. Will be deprecated in favor of :func:`~torch.nn.quantized.functional.interpolate`
* **upsample_bilinear** — Bilenear upsampler. Will be deprecated in favor of :func:`~torch.nn.quantized.functional.interpolate`
* **upsample_nearest** — Nearest neighbor upsampler. Will be deprecated in favor of :func:`~torch.nn.quantized.functional.interpolate`

**Quantized tensor methods**

Member methods as part of quantized tensors

* **as_strided** — Returns a view of the input tensor with the given size and stride
* **clone** — Returns a deep copy of the passed-in tensor
* **copy_** — Copies src to self in-place
* Comparators:
    * **ne** — Not equal
    * **eq** — Equal
    * **ge** — Greater or equal
    * **le** — Less or equal
    * **gt** — Greater
    * **lt** — Less
* **dequantize** — Convert quantized tensor to float tensor
* **equal** — Compares two tensors, returns true if quantization parameters and all integer elements are the same
* **int_repr** — Prints the underlying integer representation of the quantized tensor
* **max** — Returns the maximum value of the tensor (reduction only)
* **mean** (reduction, dim, out) — Mean method. Supported variants: reduction, dim, out
* **min** — Returns the minimum value of the tensor (reduction only)
* **q_scale** — Returns the scale of the per-tensor quantized tensor
* **q_zero_point** — Returns the zero_point of the per-tensor quantized zero point
* **q_per_channel_scales** — Returns the scales of the per-channel quantized tensor
* **q_per_channel_zero_points** — Returns the zero points of the per-channel quantized tensor
* **q_per_channel_axis** — Returns the channel axis of the per-channel quantized tensor
* **qscheme** — Returns the quantization scheme of the quantized tensor
* **relu** — Rectified linear unit (copy)
* **relu_** — Rectified linear unit (inline)
* **resize_** — In-place resize
* **set_** — Sets the underlying storage, size, and strides
* **sort** — Sorts the tensor
* **topk** — Returns k maximum values of a tensor
* **view** — Returns the tensor with the same data but of different shape

**Types in Quantization**

* **torch.qscheme** — Type to describe the quantization scheme of a tensor. Supported types:
    * **torch.per_tensor_affine** — per tensor, asymmetric
    * **torch.per_channel_affine** — per channel, asymmetric
    * **torch.per_tensor_symmetric** — per tensor, symmetric
    * **torch.per_channel_symmetric** — per tensor, symmetric
* torch.dtype — Type to describe the data. Supported types:
    * **torch.quint8** — 8-bit unsigned integer
    * **torch.qint8** — 8-bit signed integer
    * **torch.qint32** — 32-bit signed integer



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
   done with a single call to torch.quantization.quantize\\_dynamic().
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
   4. Use the torch.quantization.prepare() method to insert modules
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
   with simulated quantization using the **FakeQuant**\\ (). Computations
   will take place in FP32 but with values clamped and rounded to
   simulate the effects of INT8 quantization. The sequence of steps is
   very similar.


   1. Steps (1) and (2) are identical.

   3. Specify the configuration of the fake quantization methods \'97 such as
      selecting symmetric or asymmetric quantization and MinMax or Moving Average
      or L2Norm calibration techniques.
   4. Use the torch.quantization.prepare_qat() method to insert modules
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
---------------------------------
It is necessary to currently make some modifications to the model definition
prior to quantization. This is because currently quantization works on a module
by module basis. Specifically, for all quantization techniques, the user needs to:
1. Convert any operations that contain quantizable parameters to modules from functionals
2. Specify which parts of the model need to be quantized.

For static quantization techniques which quantize activations, the user needs to do the following in addition:
1. Specify where activations are quantized and de-quantized. This is done using QuantStub() and DeQuantStub() modules.
2. Use torch.quantization.FloatFunctional() to wrap tensor operations that require special handling for quantization into modules. Examples
   are operations like add and cat which require special handling to determine output quantization parameters.
3. Fuse modules: Combine operations/modules into a single module to obtain higher accuracy and performance. This is done using the
   torch.quantization.fuse_modules() API, which takes in lists of modules to be fused. We currently support the following fusions:
   [Conv,Relu], [Conv,BatchNorm], [Conv, BatchNorm, Relu], [Linear, Relu]


Code Structure
--------------

The code is organized into the following modules

-  ``torch.quantization`` : this module implements the functions you call
   directly to convert your model from FP32 to quantized form. For
   example the **prepare()** method is used in post training
   quantization to prepares your model for the calibration step and the
   **convert()** method actually converts the weights to int8 and
   replaces the operations with their quantized counterparts. There are
   other helper functions for things like quantizing the input to your
   model and performing critical fusions like conv+relu.
-  ``torch.nn.quantized``: This module implements the quantized
   implementations of the nn functions such as **conv2d()** and
   **ReLU()**. It also contains the functions to convert tensors to and
   from quantized form.
-  ``torch.nn.qat.modules``: This module implements versions of the key nn
   modules **Conv2d()** and **Linear()** which will run in FP32
   but with rounding applied to simulate the effect of INT8
   quantization.
-  ``torch.nn.intrinsic.modules``: This module implements the combined (fused)
   modules conv + relu which are later quantized.
-  ``torch.nn.intrinsic.quantized.modules``: This module implements the
   quantized implementations of fused operations like conv + relu.
-  ``torch.nn.intrinsic.qat.modules``: This module implements the versions
   of those fused operations needed for quantization aware training.


torch.quantization
---------------------------
.. automodule:: torch.quantization

Fake Quantize
~~~~~~~~~~~~~~~
.. autoclass:: FakeQuantize

Fuse Modules
~~~~~~~~~~~~~~~
.. autofunction:: fuse_conv_bn
.. autofunction:: fuse_conv_bn_relu
.. autofunction:: _fuse_modules
.. autofunction:: fuse_modules

Observer
~~~~~~~~~~~~~~~
.. autoclass:: Observer

Observer Base
~~~~~~~~~~~~~~~
.. autoclass:: ObserverBase
.. autofunction:: _calculate_qparams

Min Max Observer
~~~~~~~~~~~~~~~
.. autoclass:: MinMaxObserver

Per Channel Min Max Observer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: PerChannelMinMaxObserver

Histogram Observer
~~~~~~~~~~~~~~~
.. autoclass:: HistogramObserver

Recording Observer
~~~~~~~~~~~~~~~
.. autoclass:: RecordingObserver

Noop Observer
~~~~~~~~~~~~~~~
.. autoclass:: NoopObserver

Quant Stub
~~~~~~~~~~~~~~~
.. autoclass:: QuantStub

DeQuant Stub
~~~~~~~~~~~~~~~
.. autoclass:: DeQuantStub

Quant Wrapper
~~~~~~~~~~~~~~~
.. autoclass:: QuantWrapper

Utility Functions
~~~~~~~~~~~~~~~
.. autofunction:: propagate_qconfig_helper
.. autofunction:: propagate_qconfig
.. autofunction:: _observer_forward_hook
.. autofunction:: add_observer
.. autofunction:: add_quant_dequant
.. autofunction:: prepare
.. autofunction:: quantize
.. autofunction:: quantize_dynamic
.. autofunction:: quantize_qat
.. autofunction:: convert
.. autofunction:: swap_module
.. autofunction:: get_observer_dict
.. autofunction:: default_eval_fn

torch.nn.instrinsic.qat.modules
--------------------------------
.. automodule:: torch.nn.intrinsic.qat.modules

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

torch.nn.intrinsic.quantized.modules
--------------------------------------
.. automodule:: torch.nn.intrinsic.quantized.modules

ConvReLU2d
~~~~~~~~~~~~~~~
.. autoclass:: ConvReLU2d
    :members:

LinearReLU
~~~~~~~~~~~~~~~
.. autoclass:: LinearReLU
    :members:

torch.nn.qat.modules
---------------------------
.. automodule:: torch.nn.qat.modules

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
.. automodule:: torch.nn.quantized.functional

Functional interface
~~~~~~~~~~~~~~~
.. autofunction:: relu
.. autofunction:: linear
.. autofunction:: conv2d
.. autofunction:: max_pool2d

.. automodule:: torch.nn.quantized.dynamic.modules

Linear
~~~~~~~~~~~~~~~
.. autoclass:: Linear
    :members:

RNNBase
~~~~~~~~~~~~~~~
.. autoclass:: RNNBase
    :members:

.. automodule:: torch.nn.quantized.modules.activation

ReLU
~~~~~~~~~~~~~~~
.. autoclass:: ReLU
    :members:

ReLU6
~~~~~~~~~~~~~~~
.. autoclass:: ReLU6
    :members:

.. automodule:: torch.nn.quantized.modules.conv

Conv2d
~~~~~~~~~~~~~~~
.. autoclass:: Conv2d
    :members:

.. automodule:: torch.nn.quantized.modules.functional_modules

Float Functional
~~~~~~~~~~~~~~~
.. autoclass:: FloatFunctional
    :members:

QFunctional
~~~~~~~~~~~~~~~
.. autoclass:: QFunctional
    :members:

.. automodule:: torch.nn.quantized.modules.linear

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
