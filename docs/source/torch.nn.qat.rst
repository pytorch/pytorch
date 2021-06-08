torch.nn.qat
---------------------------

This module implements versions of the key nn modules **Conv2d()** and
**Linear()** which run in FP32 but with rounding applied to simulate the effect
of INT8 quantization.

.. automodule:: torch.nn.qat

Conv2d
~~~~~~~~~~~~~~~
.. autoclass:: Conv2d
    :members:

Linear
~~~~~~~~~~~~~~~
.. autoclass:: Linear
    :members:
