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
