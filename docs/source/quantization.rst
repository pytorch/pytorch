PyTorch Quantization 
================================


Introduction to Quantization
----------------------------

A quantized model executes some or all of the operations on tensors with
integer rather than floating point values. This allows for a more
compact model representation and the use of high performance vectorized
operations on many hardware platforms. PyTorch supports INT8
quantization and the typical tensor uses FP32 so the memory needed to
keep weights and activations is reduced by a factor of up to 4. Hardware
implementations vary but can be anywhere from 2 to 4 times faster at the
level of the vector instructions.

Quantized representation
~~~~~~~~~~~~~~~~~~~~~~~~

PyTorch supports both per tensor and per channel asymmetric linear
quantization. Per tensor means that all the values within the tensor are
scaled the same way. Per channel means that for each channel the values
in the tensor are scaled and offset by a different value (effectively
the scale and offset become vectors). Note that we currently only
support per channel quantization for the **conv** and **linear**
operators. Furthermore the minimum and the maximum of the input data is
mapped linearly to the minimum and the maximum of the quantized data
type such that zero is represented with no quantization error.

The mapping is performed by converting the floating point tensors using

***include image of the formula from the design doc here***

Note that for operators, we restrict support to:

1. 8 bit weights (data\_type = qint8)
2. 8 bit activations (data\_type = quint8)
3. 32 bit, symmetric quantization for bias (zero\_point = 0, data\_type
   = qint32)

Quantized Tensor
~~~~~~~~~~~~~~~~

In order to do quantization in PyTorch, we need to be able to represent
quantized data in Tensors. A quantized Tensor allows for storing
quantized data (represented as int8/uint8/int32) along with quantization
parameters like scale and zero\_point. Quantized tensors allow for many
useful operations making quantized arithmetic easy, in addition to
allowing for serialization of data in a quantized format.

Supported Quantization Techniques
---------------------------------

PyTorch supports three approaches to quantize models.

1. Dynamic Quantization: This is the simplest to apply form of
   quantization where the weights are quantized ahead of time but the
   activations are dynamically scaled during the inference. This is used
   for situations where the model execution time is dominated by loading
   weights from memory rather than computing the matrix multiplications.
   Typically this is in LSTM and Transformer type models with small
   batch size. Applying dynamic quantization to a whole model can be
   done with a single call to torch.quantization.quantize\_dynamic().
   See the ***tutorial-link-here***\ *. *
2. Post Training Quantization: This is the most commonly used form of
   quantization where the weights are quantized ahead of time and the
   scale factor and bias for the activation tensors is pre-computed
   based on observing the behavior of the model during a calibration
   process. Post Training Quantization is typically used for CNNs or any
   other model that is compute bound. The general process for doing post
   training quantization is:

   1. Prepare the model itself by adding quantize and dequantize nodes,
      making sure layers are not reused and performing fusions such as
      conv + relu.
   2. Specify the configuration of the quantization methods — such as
      selecting symmetric or asymmetric quantization and MinMax or
      L2Norm calibration techniques.
   3. Use the torch.quantization.prepare() method to insert functions
      that will observe activation tensors during calibration
   4. Calibrate the model by running inference against a calibration
      dataset
   5. Finally, convert the model itself with the
      torch.quantization.convert() method. This does several things: it
      quantizes the weights, computes and stores the scale and bias
      value to be used each activation tensor, and replaces key
      operators quantized implementations.
   6. See the ***tutorial-link-here ***\ for more information\ *. *

3. Quantization Aware Training: In the rare cases where post training
   quantization does not provide adequate accuracy training can be done
   with simulated quantization using the **FakeQuant**\ (). Computations
   will take place in FP32 but with values clamped and rounded to
   simulate the effects of INT8 quantization. The sequence of steps is
   very similar.

   1. Prepare the model itself by adding quantize and dequantize nodes,
      making sure layers are not reused and performing fusions such as
      conv + relu.
   2. Specify the configuration of the quantization methods — such as
      selecting symmetric or asymmetric quantization and MinMax or
      L2Norm calibration techniques.
   3. Use the torch.quantization.FakeQuant() method to insert functions
      that will simulate quantization during training.
   4. Train or fine tune the model.
   5. Finally, convert the model itself with the
      torch.quantization.convert() method. This does several things: it
      quantizes the weights, computes and stores the scale and bias
      value to be used each activation tensor, and replaces key
      operators quantized implementations.
   6. See the ***tutorial-link-here ***\ for more information\ *. *

While functions to select the scale factor and bias based on observed
tensor data are provided, developers can provide your own quantization
functions. Quantization can be applied selectively to different parts of
the model or configured differently for different portions of the model.

We also provide support for per channel quantization for **conv2d**\ ()
and **linear**\ ()

Code Structure
--------------

The code is organized into the following sections

-  torch.quantization : this module implements the functions you call
   directly to convert your model from FP32 to quantized form. For
   example the **prepare() **\ method is used in post training
   quantization to prepares your model for the calibration step and the
   **convert()** method actually converts the weights to int8 and
   replaces the operations with their quantized counterparts. There are
   other helper functions for things like quantizing the input to your
   model and performing critical fusions like conv+relu.
-  torch.nn.quantized: This module implements the quantized
   implementations of the nn functions such as **conv2d()** and
   **ReLU()**. It also contains the functions to convert tensors to and
   from quantized form.
-  torch.nn.qat.modules: This module implements versions of the key nn
   functions **conv2d**\ () and **linear**\ () which will run in FP32
   but with rounding applied to simulate the effect of INT8
   quantization.
-  torch.nn.intrinsic.quantized.modules: This module implements the
   quantized implementations of fused operations like conv + relu.
-  torch.nn.intrinsic.qat.modules: This module implements the versions
   of those fused operations needed for quantization aware training.


torch.quantization
===========================
.. automodule:: torch.quantization

Initialization
---------------
.. autofunction:: default_eval_fn

Fake Quantize
--------------
.. autoclass:: FakeQuantize

Fuse Modules
-------------
.. autofunction:: fuse_conv_bn
.. autofunction:: fuse_conv_bn_relu
.. autofunction:: _fuse_modules
.. autofunction:: fuse_modules

Observer
---------
.. autoclass:: Observer

Observer Base
--------------
.. autoclass:: ObserverBase
.. autofunction:: _calculate_qparams

Min Max Observer
----------------
.. autoclass:: MinMaxObserver

Per Channel Min Max Observer
-----------------------------
.. autoclass:: MinMaxObserver

Histogram Observer
------------------
.. autoclass:: HistogramObserver

Recording Observer
------------------
.. autoclass:: RecordingObserver

Noop Observer
------------------
.. autoclass:: NoopObserver

Quant Stub
------------------
.. autoclass:: QuantStub

DeQuant Stub
------------------
.. autoclass:: DeQuantStub

Quant Wrapper
------------------
.. autoclass:: QuantWrapper

Utility Functions
------------------
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

torch.nn._instrinsic.qat.modules
================================
.. automodule:: torch.nn._intrinsic.qat.modules

ConvBn2d
--------
.. autoclass:: ConvBn2d
    :members:

ConvBnReLU2d
------------------
.. autoclass:: ConvBnReLU2d
    :members:

ConvReLU2d
------------------
.. autoclass:: ConvReLU2d
    :members:

LinearReLU
------------------
.. autoclass:: LinearReLU
    :members:

torch.nn._intrinsic.quantized.modules
======================================
.. automodule:: torch.nn._intrinsic.quantized.modules

ConvReLU2d
------------------
.. autoclass:: ConvReLU2d
    :members:

LinearReLU
------------------
.. autoclass:: LinearReLU
    :members:

torch.nn.qat.modules
===========================
.. automodule:: torch.nn.qat.modules

Conv2d
------------------
.. autoclass:: Conv2d
    :members:

Linear
------------------
.. autoclass:: Linear
    :members:


torch.nn.quantized
===========================
.. automodule:: torch.nn.quantized.functional

Functional interface
---------------------
.. autofunction:: relu
.. autofunction:: linear
.. autofunction:: conv2d
.. autofunction:: max_pool2d

.. automodule:: torch.nn.quantized.dynamic.modules

Linear
------------------
.. autoclass:: Linear
    :members:

RNNBase
------------------
.. autoclass:: RNNBase
    :members:

.. automodule:: torch.nn.quantized.modules.activation

ReLU
------------------
.. autoclass:: ReLU
    :members:

ReLU6
------------------
.. autoclass:: ReLU6
    :members:

.. automodule:: torch.nn.quantized.modules.conv

Conv2d
------------------
.. autoclass:: Conv2d
    :members:

.. automodule:: torch.nn.quantized.modules.functional_modules

Float Functional
------------------
.. autoclass:: FloatFunctional
    :members:

QFunctional
------------------
.. autoclass:: QFunctional
    :members:

.. automodule:: torch.nn.quantized.modules.linear

Quantize
------------------
.. autoclass:: Quantize
    :members:

DeQuantize
------------------
.. autoclass:: DeQuantize
    :members:

Linear
------------------
.. autoclass: Linear
    :members:
