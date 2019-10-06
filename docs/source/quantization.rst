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
training, pytorch provides conversion functions to convert the trained model into lower precision.

At lower level, PyTorch provides a way to represent quantized tensors and
perform operations with them. They can be used to directly construct models that
perform all or part of the computation in lower precision. Higher-level APIs are
provided that incorporate typical workflows of converting FP32 model to lower
precision with minimal accuracy loss.

Quantized representation and operations
---------------------------------------

PyTorch supports both per tensor and per channel asymmetric linear
quantization. Per tensor means that all the values within the tensor are
scaled the same way. Per channel means that for each dimension, typically
the channel dimension of a tensor, the values
in the tensor are scaled and offset by a different value (effectively
the scale and offset become vectors). This allows for lesser error in converting tensors
to quantized values.

The mapping is performed by converting the floating point tensors using

***include image of the formula from the design doc here***

Note that, we ensure that zero in floating point is repsresented with no error after quantization,
thereby ensuring that operations like padding do not cause additional quantization error.

Quantized Tensor
~~~~~~~~~~~~~~~~

In order to do quantization in PyTorch, we need to be able to represent
quantized data in Tensors. A quantized Tensor allows for storing
quantized data (represented as int8/uint8/int32) along with quantization
parameters like scale and zero\\_point. Quantized tensors allow for many
useful operations making quantized arithmetic easy, in addition to
allowing for serialization of data in a quantized format.

Operation coverage
~~~~~~~~~~~~~~~~~~

Quantized tensors support a limited subset of data manipulation methods of the regular
full-precision tensor. <Link to quantized tensor documentation here>

For NN operators included in PyTorch, we restrict support to:

   1. 8 bit weights (data\\_type = qint8)
   2. 8 bit activations (data\\_type = quint8)

Note that operator implementations currently only
support per channel quantization for weights of the **conv** and **linear**
operators. Furthermore the minimum and the maximum of the input data is
mapped linearly to the minimum and the maximum of the quantized data
type such that zero is represented with no quantization error.

Additional data types and quantization schemes can be implemented through
the custom operator mechanism.
[Note: Will be good to point a link here]

Many operations for quantized tensors are available under the same API as full
float version in ``torch`` or ``torch.nn``. Quantized version of NN modules that
perform re-quantization are available in ``torch.nn.quantized``. Those
operations explicitly take output quantization parameters (scale and bias) in
the operation signature.

Current quantized operation list is sufficient to cover typical CNN and RNN
models:

TODO: explicit list here

Quantization Workflows
----------------------

PyTorch provides three approaches to quantize models.

1. Dynamic Quantization: This is the simplest to apply form of
   quantization where the weights are quantized ahead of time but the
   activations are dynamically quantized  during inference. This is used
   for situations where the model execution time is dominated by loading
   weights from memory rather than computing the matrix multiplications.
   This is true for for LSTM and Transformer type models with small
   batch size. Applying dynamic quantization to a whole model can be
   done with a single call to torch.quantization.quantize\\_dynamic().
   See the ***tutorial-link-here***\\ *. *
2. Post Training Quantization: This is the most commonly used form of
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
   7. See the ***tutorial-link-here ***\\ for more information\\ *. *

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

   7. See the ***tutorial-link-here ***\\ for more information\\ *. *

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

Model preparation for quantization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

TODO:
- fusion
- replacement of modules with FloatFunctional
- QuantStub

Code Structure
--------------

The code is organized into the following modules
[Note: Insert quantization code stack diagram here]

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
.. autoclass:: PerChannelMinMaxObserver

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
.. autofunction:: default_eval_fn

torch.nn.instrinsic.qat.modules
--------------------------------
.. automodule:: torch.nn.intrinsic.qat.modules

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

torch.nn.intrinsic.quantized.modules
--------------------------------------
.. automodule:: torch.nn.intrinsic.quantized.modules

ConvReLU2d
------------------
.. autoclass:: ConvReLU2d
    :members:

LinearReLU
------------------
.. autoclass:: LinearReLU
    :members:

torch.nn.qat.modules
---------------------------
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
----------------------------
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
.. autoclass:: Linear
    :members:
