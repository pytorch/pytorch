.. role:: hidden
    :class: hidden-section

Automatic Mixed Precision package - torch.autocast
==================================================

.. Both modules below are missing doc entry. Adding them here for now.
.. This does not add anything to the rendered page
.. py:module:: torch.cpu
.. py:module:: torch.cpu.amp

.. automodule:: torch.amp
.. currentmodule:: torch.amp

:class:`torch.autocast` provides convenience methods for mixed precision,
where some operations use the ``torch.float32`` (``float``) datatype and other operations
use lower precision floating point datatype (``lower_precision_fp``)：``torch.float16`` (``half``) or ``torch.bfloat16``. Some ops, like linear layers and convolutions,
are much faster in ``lower_precision_fp``. Other ops, like reductions, often require the dynamic
range of ``float32``.  Mixed precision tries to match each op to its appropriate datatype.

For CUDA and CPU, :doc:`cuda.amp` and :doc:`cpu.amp` are also provided seperately. ``torch.cuda.amp.autocast(args...)`` is equivalent to ``torch.autocast("cuda", args...)``. ``torch.cpu.amp.autocast(args...)`` is equivalent to ``torch.autocast("cpu", args...)``.

.. contents:: :local:

Autocasting
^^^^^^^^^^^
.. currentmodule:: torch

.. autoclass:: autocast
    :members:
