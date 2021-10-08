.. _torch_quantization_quantize_fx:

torch.quantization.quantize_fx
------------------------------
.. automodule:: torch.ao.quantization.quantize_fx

This module implements the functions you can call to convert your model
from FP32 to quantized form, using the FX framework.

Top-level quantization APIs
~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: prepare_fx
.. autofunction:: prepare_qat_fx
.. autofunction:: convert_fx
.. autofunction:: fuse_fx
.. autofunction:: _prepare_standalone_module_fx
.. autofunction:: _convert_standalone_module_fx
.. autoclass:: Scope
.. autoclass:: ScopeContextManager

.. currentmodule:: torch

.. autosummary::
    :nosignatures:
