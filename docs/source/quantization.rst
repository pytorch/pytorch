.. _quantization-doc:

Quantization
============

.. warning ::
     Quantization is in beta and subject to change.

Introduction to Quantization
----------------------------

Quantization refers to techniques for performing computations and storing
tensors at lower bitwidths than floating point precision. A quantized model
executes some or all of the operations on tensors with integers rather than
floating point values. This allows for a more compact model representation and
the use of high performance vectorized operations on many hardware platforms.
PyTorch supports INT8 quantization compared to typical FP32 models allowing for
a 4x reduction in the model size and a 4x reduction in memory bandwidth
requirements.  Hardware support for  INT8 computations is typically 2 to 4
times faster compared to FP32 compute. Quantization is primarily a technique to
speed up inference and only the forward pass is supported for quantized
operators.

PyTorch supports multiple approaches to quantizing a deep learning model. In
most cases the model is trained in FP32 and then the model is converted to
INT8. In addition, PyTorch also supports quantization aware training, which
models quantization errors in both the forward and backward passes using
fake-quantization modules. Note that the entire computation is carried out in
floating point. At the end of quantization aware training, PyTorch provides
conversion functions to convert the trained model into lower precision.

At lower level, PyTorch provides a way to represent quantized tensors and
perform operations with them. They can be used to directly construct models
that perform all or part of the computation in lower precision. Higher-level
APIs are provided that incorporate typical workflows of converting FP32 model
to lower precision with minimal accuracy loss.

Today, PyTorch supports the following backends for running quantized operators efficiently:

* x86 CPUs with AVX2 support or higher (without AVX2 some operations have
  inefficient implementations)
* ARM CPUs (typically found in mobile/embedded devices)

The corresponding implementation is chosen automatically based on the PyTorch build mode.

.. note::

  PyTorch 1.3 doesn't provide quantized operator implementations on CUDA yet -
  this is direction of future work.  Move the model to CPU in order to test the
  quantized functionality.

  Quantization-aware training (through :class:`~torch.quantization.FakeQuantize`)
  supports both CPU and CUDA.


.. note::

    When preparing a quantized model, it is necessary to ensure that qconfig
    and the engine used for quantized computations match the backend on which
    the model will be executed. Quantization currently supports two backends:
    fbgemm (for use on x86, `<https://github.com/pytorch/FBGEMM>`_) and qnnpack
    (for use on the ARM QNNPACK library `<https://github.com/pytorch/QNNPACK>`_).
    For example, if you are interested in quantizing a model to run on ARM, it
    is recommended to set the qconfig by calling:

    ``qconfig = torch.quantization.get_default_qconfig('qnnpack')``

    for post training quantization and

    ``qconfig = torch.quantization.get_default_qat_qconfig('qnnpack')``

    for quantization aware training.

    In addition, the torch.backends.quantized.engine parameter should be set to
    match the backend. For using qnnpack for inference, the backend is set to
    qnnpack as follows

    ``torch.backends.quantized.engine = 'qnnpack'``

Quantization API Summary
---------------------------------------

There are three types of quantization supported in PyTorch:

1. dynamic quantization (weights quantized with activations read/stored in
   floating point and quantized for compute.)
2. static quantization (weights quantized, activations quantized, calibration
   required post training)
3. quantization aware training (weights quantized, activations quantized,
   quantization numerics modeled during training)

Please see our `Introduction to Quantization on Pytorch
<https://pytorch.org/blog/introduction-to-quantization-on-pytorch/>`_ blog post
for a more comprehensive overview of the tradeoffs between these quantization
types.

Dynamic Quantization
^^^^^^^^^^^^^^^^^^^^

Dynamic quantization quantizes the weights of the model.  Activations are
converted to the quantized domain for calculations and converted to floating
point domain for I/O.  It does not require calibration and can be done in a
single function call.  It is commonly used in models where runtime is compute
bound or dominated by memory access of weights, instead of being dominated by
memory accesses of activations dominates the runtime, such as LSTMs/RNNs or
BERT/Transformers.

Diagram::

  # original model
  # all tensors and computations are in floating point
  previous_layer_fp32 -- linear_fp32 -- activation_fp32 -- next_layer_fp32
                   /
  linear_weight_fp32

  # dynamically quantized model
  # linear and conv weights are in int8
  previous_layer_fp32 -- linear_int8_w_fp32_inp -- activation_fp32 -- next_layer_fp32
                       /
     linear_weight_int8

API example::

    import torch

    # define a floating point model
    class M(torch.nn.Module):
        def __init__(self):
            super(M, self).__init__()
            self.fc = torch.nn.Linear(4, 4)

        def forward(self, x):
            x = self.fc(x)
            return x

    # create a model instance
    model_fp32 = M()
    # create a quantized model instance
    model_int8 = torch.quantization.quantize_dynamic(
        model_fp32,  # the original model
        {torch.nn.Linear},  # a set of layers to dynamically quantize
        dtype=torch.qint8)  # the target dtype for quantized weights

    # run the model
    input_fp32 = torch.randn(4, 4, 4, 4)
    res = model_int8(input_fp32)

To learn more about dynamic quantization please see our `end to end tutorial
<https://pytorch.org/tutorials/recipes/recipes/dynamic_quantization.html>`_.

Static Quantization
^^^^^^^^^^^^^^^^^^^^

Static quantization quantizes the weights and activations of the model.  It
fuses activations into preceding layers where possible.  It requires
calibration with a representative dataset to determine optimal quantization
parameters for activations.  It is commonly used in models where activation
throughput dominates the runtime, such as CNNs.  Static quantization is also
known as Post Training Quantization or PTQ.

Diagram::

    # original model
    # all tensors and computations are in floating point
    previous_layer_fp32 -- linear_fp32 -- activation_fp32 -- next_layer_fp32
                        /
        linear_weight_fp32

    # statically quantized model
    # weights and activations are in int8
    previous_layer_int8 -- linear_with_activation_int8 -- next_layer_int8
                        /
      linear_weight_int8

API Example::

  import torch

  # define a floating point model where some layers could be statically quantized
  class M(torch.nn.Module):
      def __init__(self):
          super(M, self).__init__()
          # QuantStub converts tensors from floating point to quantized
          self.quant = torch.quantization.QuantStub()
          self.conv = torch.nn.Conv2d(1, 1, 1)
          self.relu = torch.nn.ReLU()
          # DeQuantStub converts tensors from quantized to floating point
          self.dequant = torch.quantization.DeQuantStub()

      def forward(self, x):
          x = self.quant(x)
          x = self.conv(x)
          x = self.relu(x)
          x = self.dequant(x)
          return x

  # create a model instance
  model_fp32 = M()

  # model must be set to eval mode for static quantization logic to work
  model_fp32.eval()

  # attach a global qconfig, which contains information about what kind
  # of observers to attach. Use 'fbgemm' for server inference and
  # 'qnnpack' for mobile inference.
  model_fp32.qconfig = torch.quantization.get_default_qconfig('fbgemm')

  # fuse the activations to preceding layers, where applicable
  # this needs to be done manually depending on the model architecture
  model_fp32_fused = torch.quantization.fuse_modules(model_fp32, [['conv', 'relu']])

  # prepare the model for static quantization
  model_fp32_prepared = torch.quantization.prepare(model_fp32_fused)

  # calibrate the prepared model to determine quantization parameters for activations
  # in a real world setting, the calibration would be done with a representative dataset
  input_fp32 = torch.randn(4, 1, 4, 4)
  model_fp32_prepared(input_fp32)

  # convert the observed model to a quantized model
  model_int8 = torch.quantization.convert(model_fp32_prepared)

  # run the model, relevant calculations will happen in int8
  res = model_int8(input_fp32)

To learn more about static quantization, please see the `end to end tutorial
<https://pytorch.org/tutorials/advanced/static_quantization_tutorial.html>`_.

Quantization Aware Training
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Quantization Aware Training models the effects of quantization during training
allowing for higher accuracy compared to other quantization methods.  During
training, all calculations are done in floating point, with fake_quant modules
modeling the effects of quantization.  After model conversion, weights and
activations are quantized, and activations are fused into the preceding layer
where possible.  It is commonly used with CNNs and yields a higher accuracy
compared to static quantization.  Quantization Aware Training is also known as
QAT.

Diagram::

  # original model
  # all tensors and computations are in floating point
  previous_layer_fp32 -- linear_fp32 -- activation_fp32 -- next_layer_fp32
                        /
      linear_weight_fp32

  # model with fake_quants for modeling quantization numerics during training
  previous_layer_fp32 -- fq -- linear_fp32 -- activation_fp32 -- fq -- next_layer_fp32
                             /
     linear_weight_fp32 -- fq

  # quantized model
  # weights and activations are in int8
  previous_layer_int8 -- linear_with_activation_int8 -- next_layer_int8
                       /
     linear_weight_int8

API Example::

  import torch

  # define a floating point model where some layers could benefit from QAT
  class M(torch.nn.Module):
      def __init__(self):
          super(M, self).__init__()
          # QuantStub converts tensors from floating point to quantized
          self.quant = torch.quantization.QuantStub()
          self.conv = torch.nn.Conv2d(1, 1, 1)
          self.bn = torch.nn.BatchNorm2d(1)
          self.relu = torch.nn.ReLU()
          # DeQuantStub converts tensors from quantized to floating point
          self.dequant = torch.quantization.DeQuantStub()

      def forward(self, x):
          x = self.quant(x)
          x = self.conv(x)
          x = self.bn(x)
          x = self.relu(x)
          x = self.dequant(x)
          return x

  # create a model instance
  model_fp32 = M()

  # model must be set to train mode for QAT logic to work
  model_fp32.train()

  # attach a global qconfig, which contains information about what kind
  # of observers to attach. Use 'fbgemm' for server inference and
  # 'qnnpack' for mobile inference.
  model_fp32.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')

  # fuse the activations to preceding layers, where applicable
  # this needs to be done manually depending on the model architecture
  model_fp32_fused = torch.quantization.fuse_modules(model_fp32,
      [['conv', 'bn', 'relu']])

  # prepare the model for static quantization
  model_fp32_prepared = torch.quantization.prepare_qat(model_fp32_fused)

  # run the training loop (not shown)
  training_loop(model_fp32_prepared)

  # convert the observed model to a quantized model
  model_fp32_prepared.eval()
  model_int8 = torch.quantization.convert(model_fp32_prepared)

  # run the model, relevant calculations will happen in int8
  res = model_int8(input_fp32)

To learn more about quantization aware training, please see the `end to end
tutorial
<https://pytorch.org/tutorials/advanced/static_quantization_tutorial.html>`_.

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

Note that, we ensure that zero in floating point is represented with no error
after quantization, thereby ensuring that operations like padding do not cause
additional quantization error.

In order to do quantization in PyTorch, we need to be able to represent
quantized data in Tensors. A Quantized Tensor allows for storing
quantized data (represented as int8/uint8/int32) along with quantization
parameters like scale and zero\_point. Quantized Tensors allow for many
useful operations making quantized arithmetic easy, in addition to
allowing for serialization of data in a quantized format.

.. include:: quantization-support.rst
    :end-before: end-of-part-included-in-quantization.rst

The :doc:`list of supported operations <quantization-support>` is sufficient to
cover typical CNN and RNN models

.. toctree::
    :hidden:

    torch.nn.intrinsic
    torch.nn.intrinsic.qat
    torch.nn.intrinsic.quantized
    torch.nn.qat
    torch.quantization
    torch.nn.quantized
    torch.nn.quantized.dynamic

Quantization Workflows
----------------------

PyTorch provides three approaches to quantize models.

.. _quantization tutorials:
   https://pytorch.org/tutorials/#quantization-experimental

1. Post Training Dynamic Quantization: This is the simplest to apply form of
   quantization where the weights are quantized ahead of time but the
   activations are dynamically quantized  during inference. This is used
   for situations where the model execution time is dominated by loading
   weights from memory rather than computing the matrix multiplications.
   This is true for for LSTM and Transformer type models with small
   batch size. Applying dynamic quantization to a whole model can be
   done with a single call to :func:`torch.quantization.quantize_dynamic()`.
   See the `quantization tutorials`_
2. Post Training Static Quantization: This is the most commonly used form of
   quantization where the weights are quantized ahead of time and the
   scale factor and bias for the activation tensors is pre-computed
   based on observing the behavior of the model during a calibration
   process. Post Training Quantization is typically when both memory bandwidth
   and compute savings are important with CNNs being a typical use case.
   The general process for doing post training quantization is:



   1. Prepare the model:

      a. Specify where the activations are quantized and dequantized explicitly
         by adding QuantStub and DeQuantStub modules.
      b. Ensure that modules are not reused.
      c. Convert any operations that require requantization into modules

   2. Fuse operations like conv + relu or conv+batchnorm + relu together to
      improve both model accuracy and performance.

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

   See the `quantization tutorials`_


3. Quantization Aware Training: In the rare cases where post training
   quantization does not provide adequate accuracy training can be done
   with simulated quantization using the
   :class:`torch.quantization.FakeQuantize`. Computations will take place in
   FP32 but with values clamped and rounded to simulate the effects of INT8
   quantization. The sequence of steps is very similar.


   1. Steps (1) and (2) are identical.

   3. Specify the configuration of the fake quantization methods \'97 such as
      selecting symmetric or asymmetric quantization and MinMax or Moving Average
      or L2Norm calibration techniques.
   4. Use the :func:`torch.quantization.prepare_qat` to insert modules
      that will simulate quantization during training.
   5. Train or fine tune the model.
   6. Identical to step (6) for post training quantization

   See the `quantization tutorials`_


While default implementations of observers to select the scale factor and bias
based on observed tensor data are provided, developers can provide their own
quantization functions. Quantization can be applied selectively to different
parts of the model or configured differently for different parts of the model.

We also provide support for per channel quantization for **conv2d()**,
**conv3d()** and **linear()**

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

1. Convert any operations that require output requantization (and thus have
   additional parameters) from functionals to module form.
2. Specify which parts of the model need to be quantized either by assigning
   ```.qconfig`` attributes on submodules or by specifying ``qconfig_dict``

For static quantization techniques which quantize activations, the user needs
to do the following in addition:

1. Specify where activations are quantized and de-quantized. This is done using
   :class:`~torch.quantization.QuantStub` and
   :class:`~torch.quantization.DeQuantStub` modules.
2. Use :class:`torch.nn.quantized.FloatFunctional` to wrap tensor operations
   that require special handling for quantization into modules. Examples
   are operations like ``add`` and ``cat`` which require special handling to
   determine output quantization parameters.
3. Fuse modules: combine operations/modules into a single module to obtain
   higher accuracy and performance. This is done using the
   :func:`torch.quantization.fuse_modules` API, which takes in lists of modules
   to be fused. We currently support the following fusions:
   [Conv, Relu], [Conv, BatchNorm], [Conv, BatchNorm, Relu], [Linear, Relu]


Modules that provide quantization functions and classes
-------------------------------------------------------

.. list-table::

  * - :ref:`torch_quantization`
    - This module implements the functions you call directly to convert your
      model from FP32 to quantized form. For example the
      :func:`~torch.quantization.prepare` is used in post training quantization
      to prepares your model for the calibration step and
      :func:`~torch.quantization.convert` actually converts the weights to int8
      and replaces the operations with their quantized counterparts. There are
      other helper functions for things like quantizing the input to your
      model and performing critical fusions like conv+relu.

  * - :ref:`torch_nn_intrinsic`
    - This module implements the combined (fused) modules conv + relu which can
      then be quantized.
  * - :doc:`torch.nn.intrinsic.qat`
    - This module implements the versions of those fused operations needed for
      quantization aware training.
  * - :doc:`torch.nn.intrinsic.quantized`
    - This module implements the quantized implementations of fused operations
      like conv + relu.
  * - :doc:`torch.nn.qat`
    - This module implements versions of the key nn modules **Conv2d()** and
      **Linear()** which run in FP32 but with rounding applied to simulate the
      effect of INT8 quantization.
  * - :doc:`torch.nn.quantized`
    - This module implements the quantized versions of the nn layers such as
      ~`torch.nn.Conv2d` and `torch.nn.ReLU`.

  * - :doc:`torch.nn.quantized.dynamic`
    - Dynamically quantized :class:`~torch.nn.Linear`, :class:`~torch.nn.LSTM`,
      :class:`~torch.nn.LSTMCell`, :class:`~torch.nn.GRUCell`, and
      :class:`~torch.nn.RNNCell`.
