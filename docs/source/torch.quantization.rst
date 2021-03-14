.. _torch_quantization:

torch.quantization
------------------
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

.. FIXME: The following doesn't display correctly.
   .. autoattribute:: default_qconfig

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
.. autoclass:: ObserverBase
    :members:
.. autoclass:: MinMaxObserver
.. autoclass:: MovingAverageMinMaxObserver
.. autoclass:: PerChannelMinMaxObserver
.. autoclass:: MovingAveragePerChannelMinMaxObserver
.. autoclass:: HistogramObserver
.. autoclass:: FakeQuantize
.. autoclass:: NoopObserver

Debugging utilities
~~~~~~~~~~~~~~~~~~~
.. autofunction:: get_observer_dict
.. autoclass:: RecordingObserver

.. currentmodule:: torch

.. autosummary::
    :nosignatures:

    nn.intrinsic

