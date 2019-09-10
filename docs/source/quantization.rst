torch.quantization
===========================
.. automodule:: torch.quantization


torch.quantization.__init__
---------------------------
.. autofunction:: default_eval_fn

torch.quantization.fake_quantize
--------------------------------
.. autoclass:: FakeQuantize

torch.quantization.fuse_modules
--------------------------------
.. autofunction:: fuse_conv_bn
.. autofunction:: fuse_conv_bn_relu
.. autofunction:: _fuse_modules
.. autofunction:: fuse_modules

torch.quantization.observer
---------------------------
.. autoclass:: ObserverBase
.. autofunction::  _calculate_qparams
.. autoclass:: MinMaxObserver


torch.quantization.quantize
---------------------------

.. automodule:: torch.quantization

.. autofunction:: propagate_qconfig_helper

.. autofunction:: propagate_qconfig

.. autofunction:: _observer_forward_hook

.. autofunction:: add_observer

.. autoclass:: QuantWrapper

.. autofunction:: add_quant_dequant

.. autofunction:: prepare

.. autoclass:: QuantStub

.. autoclass:: DeQuantStub

.. autofunction:: quantize

.. autofunction:: quantize_dynamic

.. autofunction:: quantize_qat

.. autofunction:: convert

.. autofunction:: swap_module
