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

  At the moment PyTorch doesn't provide quantized operator implementations on CUDA -
  this is the direction for future work. Move the model to CPU in order to test the
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

PyTorch provides two different modes of quantization: Eager Mode Quantization and FX Graph Mode Quantization.

Eager Mode Quantization is a beta feature. User needs to do fusion and specify where quantization and dequantization happens manually, also it only supports modules and not functionals.

FX Graph Mode Quantization is a new automated quantization framework in PyTorch, and currently it's a prototype feature. It improves upon Eager Mode Quantization by adding support for functionals and automating the quantization process, although people might need to refactor the model to make the model compatible with FX Graph Mode Quantization (symbolically traceable with ``torch.fx``). Note that FX Graph Mode Quantization is not expected to work on arbitrary models since the model might not be symbolically traceable, we will integrate it into domain libraries like torchvision and users will be able to quantize models similar to the ones in supported domain libraries with FX Graph Mode Quantization. For arbitrary models we'll provide general guidelines, but to actually make it work, users might need to be familiar with ``torch.fx``, especially on how to make a model symbolically traceable.

New users of quantization are encouraged to try out FX Graph Mode Quantization first, if it does not work, user may try to follow the guideline of `using FX Graph Mode Quantization <https://pytorch.org/tutorials/prototype/fx_graph_mode_quant_guide_tutorial.html>`_ or fall back to eager mode quantization.

The following table compares the differences between Eager Mode Quantization and FX Graph Mode Quantization:

+-----------------+-------------------+-------------------+
|                 |Eager Mode         |FX Graph           |
|                 |Quantization       |Mode               |
|                 |                   |Quantization       |
+-----------------+-------------------+-------------------+
|Release          |beta               |prototype          |
|Status           |                   |                   |
+-----------------+-------------------+-------------------+
|Operator         |Manual             |Automatic          |
|Fusion           |                   |                   |
+-----------------+-------------------+-------------------+
|Quant/DeQuant    |Manual             |Automatic          |
|Placement        |                   |                   |
+-----------------+-------------------+-------------------+
|Quantizing       |Supported          |Supported          |
|Modules          |                   |                   |
+-----------------+-------------------+-------------------+
|Quantizing       |Manual             |Automatic          |
|Functionals/Torch|                   |                   |
|Ops              |                   |                   |
+-----------------+-------------------+-------------------+
|Support for      |Limited Support    |Fully              |
|Customization    |                   |Supported          |
+-----------------+-------------------+-------------------+
|Quantization Mode|Post Training      |Post Training      |
|Support          |Quantization:      |Quantization:      |
|                 |Static, Dynamic,   |Static, Dynamic,   |
|                 |Weight Only        |Weight Only        |
|                 |                   |                   |
|                 |Quantiztion Aware  |Quantiztion Aware  |
|                 |Training:          |Training:          |
|                 |Static             |Static             |
+-----------------+-------------------+-------------------+
|Input/Output     |``torch.nn.Module``|``torch.nn.Module``|
|Model Type       |                   |(May need some     |
|                 |                   |refactors to make  |
|                 |                   |the model          |
|                 |                   |compatible with FX |
|                 |                   |Graph Mode         |
|                 |                   |Quantization)      |
+-----------------+-------------------+-------------------+


Eager Mode Quantization
^^^^^^^^^^^^^^^^^^^^^^^

There are three types of quantization supported in Eager Mode Quantization:

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
~~~~~~~~~~~~~~~~~~~~

This is the simplest to apply form of quantization where the weights are
quantized ahead of time but the activations are dynamically quantized
during inference. This is used for situations where the model execution time
is dominated by loading weights from memory rather than computing the matrix
multiplications. This is true for for LSTM and Transformer type models with
small batch size.

Diagram::

  # original model
  # all tensors and computations are in floating point
  previous_layer_fp32 -- linear_fp32 -- activation_fp32 -- next_layer_fp32
                   /
  linear_weight_fp32

  # dynamically quantized model
  # linear and LSTM weights are in int8
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

To learn more about dynamic quantization please see our `dynamic quantization tutorial
<https://pytorch.org/tutorials/recipes/recipes/dynamic_quantization.html>`_.

Static Quantization
~~~~~~~~~~~~~~~~~~~

Static quantization quantizes the weights and activations of the model.  It
fuses activations into preceding layers where possible.  It requires
calibration with a representative dataset to determine optimal quantization
parameters for activations. Post Training Quantization is typically used when
both memory bandwidth and compute savings are important with CNNs being a
typical use case.  Static quantization is also known as Post Training
Quantization or PTQ.

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
          # manually specify where tensors will be converted from floating
          # point to quantized in the quantized model
          x = self.quant(x)
          x = self.conv(x)
          x = self.relu(x)
          # manually specify where tensors will be converted from quantized
          # to floating point in the quantized model
          x = self.dequant(x)
          return x

  # create a model instance
  model_fp32 = M()

  # model must be set to eval mode for static quantization logic to work
  model_fp32.eval()

  # attach a global qconfig, which contains information about what kind
  # of observers to attach. Use 'fbgemm' for server inference and
  # 'qnnpack' for mobile inference. Other quantization configurations such
  # as selecting symmetric or assymetric quantization and MinMax or L2Norm
  # calibration techniques can be specified here.
  model_fp32.qconfig = torch.quantization.get_default_qconfig('fbgemm')

  # Fuse the activations to preceding layers, where applicable.
  # This needs to be done manually depending on the model architecture.
  # Common fusions include `conv + relu` and `conv + batchnorm + relu`
  model_fp32_fused = torch.quantization.fuse_modules(model_fp32, [['conv', 'relu']])

  # Prepare the model for static quantization. This inserts observers in
  # the model that will observe activation tensors during calibration.
  model_fp32_prepared = torch.quantization.prepare(model_fp32_fused)

  # calibrate the prepared model to determine quantization parameters for activations
  # in a real world setting, the calibration would be done with a representative dataset
  input_fp32 = torch.randn(4, 1, 4, 4)
  model_fp32_prepared(input_fp32)

  # Convert the observed model to a quantized model. This does several things:
  # quantizes the weights, computes and stores the scale and bias value to be
  # used with each activation tensor, and replaces key operators with quantized
  # implementations.
  model_int8 = torch.quantization.convert(model_fp32_prepared)

  # run the model, relevant calculations will happen in int8
  res = model_int8(input_fp32)

To learn more about static quantization, please see the `static quantization tutorial
<https://pytorch.org/tutorials/advanced/static_quantization_tutorial.html>`_.

Quantization Aware Training
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Quantization Aware Training models the effects of quantization during training
allowing for higher accuracy compared to other quantization methods.  During
training, all calculations are done in floating point, with fake_quant modules
modeling the effects of quantization by clamping and rounding to simulate the
effects of INT8.  After model conversion, weights and
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
  # 'qnnpack' for mobile inference. Other quantization configurations such
  # as selecting symmetric or assymetric quantization and MinMax or L2Norm
  # calibration techniques can be specified here.
  model_fp32.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')

  # fuse the activations to preceding layers, where applicable
  # this needs to be done manually depending on the model architecture
  model_fp32_fused = torch.quantization.fuse_modules(model_fp32,
      [['conv', 'bn', 'relu']])

  # Prepare the model for QAT. This inserts observers and fake_quants in
  # the model that will observe weight and activation tensors during calibration.
  model_fp32_prepared = torch.quantization.prepare_qat(model_fp32_fused)

  # run the training loop (not shown)
  training_loop(model_fp32_prepared)

  # Convert the observed model to a quantized model. This does several things:
  # quantizes the weights, computes and stores the scale and bias value to be
  # used with each activation tensor, fuses modules where appropriate,
  # and replaces key operators with quantized implementations.
  model_fp32_prepared.eval()
  model_int8 = torch.quantization.convert(model_fp32_prepared)

  # run the model, relevant calculations will happen in int8
  res = model_int8(input_fp32)

To learn more about quantization aware training, please see the `QAT
tutorial
<https://pytorch.org/tutorials/advanced/static_quantization_tutorial.html>`_.

(Prototype) FX Graph Mode Quantization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Quantization types supported by FX Graph Mode can be classified in two ways:

1.
- Post Training Quantization (apply quantization after training, quantization parameters are calculated based on sample calibration data)
- Quantization Aware Training (simulate quantization during training so that the quantization parameters can be learned together with the model using training data)

2.
- Weight Only Quantization (only weight is statically quantized)
- Dynamic Quantization (weight is statically quantized, activation is dynamically quantized)
- Static Quantization (both weight and activations are statically quantized)

These two ways of classification are independent, so theoretically we can have 6 different types of quantization.

The supported quantization types in FX Graph Mode Quantization are:

- Post Training Quantization

  - Weight Only Quantization
  - Dynamic Quantization
  - Static Quantization

- Quantization Aware Training

  - Static Quantization


There are multiple quantization types in post training quantization (weight only, dynamic and static) and the configuration is done through `qconfig_dict` (an argument of the `prepare_fx` function).

API Example::

  import torch.quantization.quantize_fx as quantize_fx
  import copy

  model_fp = UserModel(...)

  #
  # post training dynamic/weight_only quantization
  #

  # we need to deepcopy if we still want to keep model_fp unchanged after quantization since quantization apis change the input model
  model_to_quantize = copy.deepcopy(model_fp)
  model_to_quantize.eval()
  qconfig_dict = {"": torch.quantization.default_dynamic_qconfig}
  # prepare
  model_prepared = quantize_fx.prepare_fx(model_to_quantize, qconfig_dict)
  # no calibration needed when we only have dynamici/weight_only quantization
  # quantize
  model_quantized = quantize_fx.convert_fx(model_prepared)

  #
  # post training static quantization
  #

  model_to_quantize = copy.deepcopy(model_fp)
  qconfig_dict = {"": torch.quantization.get_default_qconfig('qnnpack')}
  model_to_quantize.eval()
  # prepare
  model_prepared = quantize_fx.prepare_fx(model_to_quantize, qconfig_dict)
  # calibrate (not shown)
  # quantize
  model_quantized = quantize_fx.convert_fx(model_prepared)

  #
  # quantization aware training for static quantization
  #

  model_to_quantize = copy.deepcopy(model_fp)
  qconfig_dict = {"": torch.quantization.get_default_qat_qconfig('qnnpack')}
  model_to_quantize.train()
  # prepare
  model_prepared = quantize_fx.prepare_qat_fx(model_to_qunatize, qconfig_dict)
  # training loop (not shown)
  # quantize
  model_quantized = quantize_fx.convert_fx(model_prepared)

  #
  # fusion
  #
  model_to_quantize = copy.deepcopy(model_fp)
  model_fused = quantize_fx.fuse_fx(model_to_quantize)

Please see the following tutorials for more information about FX Graph Mode Quantization:
- `User Guide on Using FX Graph Mode Quantization <https://pytorch.org/tutorials/prototype/fx_graph_mode_quant_guide_tutorial.html>`_
- `FX Graph Mode Post Training Static Quantization <https://pytorch.org/tutorials/prototype/fx_graph_mode_ptq_static_tutorial.html>`_
- `FX Graph Mode Post Training Dynamic Quantization <https://pytorch.org/tutorials/prototype/fx_graph_mode_ptq_dynamic_tutorial.html>`_

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

Quantization Customizations
---------------------------

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
   additional parameters) from functionals to module form (for example,
   using ``torch.nn.ReLU`` instead of ``torch.nn.functional.relu``).
2. Specify which parts of the model need to be quantized either by assigning
   ``.qconfig`` attributes on submodules or by specifying ``qconfig_dict``.
   For example, setting ``model.conv1.qconfig = None`` means that the
   ``model.conv`` layer will not be quantized, and setting
   ``model.linear1.qconfig = custom_qconfig`` means that the quantization
   settings for ``model.linear1`` will be using ``custom_qconfig`` instead
   of the global qconfig.

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

Best Practices
--------------

1. Set the ``reduce_range`` argument on observers to `True` if you are using the
   ``fbgemm`` backend.  This argument prevents overflow on some int8 instructions
   by reducing the range of quantized data type by 1 bit.

Common Errors
---------------------------------------

Passing a non-quantized Tensor into a quantized kernel
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you see an error similar to::

  RuntimeError: Could not run 'quantized::some_operator' with arguments from the 'CPU' backend...

This means that you are trying to pass a non-quantized Tensor to a quantized
kernel. A common workaround is to use ``torch.quantization.QuantStub`` to
quantize the tensor.  This needs to be done manually in Eager mode quantization.
An e2e example::

  class M(torch.nn.Module):
      def __init__(self):
          super().__init__()
          self.quant = torch.quantization.QuantStub()
          self.conv = torch.nn.Conv2d(1, 1, 1)

      def forward(self, x):
          # during the convert step, this will be replaced with a
          # `quantize_per_tensor` call
          x = self.quant(x)
          x = self.conv(x)
          return x

Passing a quantized Tensor into a non-quantized kernel
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you see an error similar to::

  RuntimeError: Could not run 'aten::thnn_conv2d_forward' with arguments from the 'QuantizedCPU' backend.

This means that you are trying to pass a quantized Tensor to a non-quantized
kernel. A common workaround is to use ``torch.quantization.DeQuantStub`` to
dequantize the tensor.  This needs to be done manually in Eager mode quantization.
An e2e example::

  class M(torch.nn.Module):
      def __init__(self):
          super().__init__()
          self.quant = torch.quantization.QuantStub()
          self.conv1 = torch.nn.Conv2d(1, 1, 1)
          # this module will not be quantized (see `qconfig = None` logic below)
          self.conv2 = torch.nn.Conv2d(1, 1, 1)
          self.dequant = torch.quantization.DeQuantStub()

      def forward(self, x):
          # during the convert step, this will be replaced with a
          # `quantize_per_tensor` call
          x = self.quant(x)
          x = self.conv1(x)
          # during the convert step, this will be replaced with a
          # `dequantize` call
          x = self.dequant(x)
          x = self.conv2(x)
          return x

  m = M()
  m.qconfig = some_qconfig
  # turn off quantization for conv2
  m.conv2.qconfig = None


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
