.. currentmodule:: torch

.. _type-info-doc:

Type Info
=========

The numerical properties of a :class:`torch.dtype` can be accessed through either the :class:`torch.finfo` or the :class:`torch.iinfo`.

.. _finfo-doc:

torch.finfo
-----------

.. class:: torch.finfo

A :class:`torch.finfo` is an object that represents the numerical properties of a floating point
:class:`torch.dtype`, (i.e. ``torch.float32``, ``torch.float64``, ``torch.float16``, and ``torch.bfloat16``). This is similar to `numpy.finfo <https://docs.scipy.org/doc/numpy/reference/generated/numpy.finfo.html>`_.

A :class:`torch.finfo` provides the following attributes:

===============        =====   ==========================================================================
Name                   Type    Description
===============        =====   ==========================================================================
bits                   int     The number of bits occupied by the type.
eps                    float   The smallest representable number such that ``1.0 + eps != 1.0``.
max                    float   The largest representable number.
min                    float   The smallest representable number (typically ``-max``).
tiny                   float   The smallest positive normal number. Equivalent to ``smallest_normal``.
smallest_normal        float   The smallest positive normal number. See notes.
resolution             float   The approximate decimal resolution of this type, i.e., ``10**-precision``.
===============        =====   ==========================================================================

.. note::
  The constructor of :class:`torch.finfo` can be called without argument, in which case the class is created for the pytorch default dtype (as returned by :func:`torch.get_default_dtype`).

.. note::
  `smallest_normal` returns the smallest *normal* number, but there are smaller
  subnormal numbers. See https://en.wikipedia.org/wiki/Denormal_number
  for more information.


.. _iinfo-doc:

torch.iinfo
------------

.. class:: torch.iinfo


A :class:`torch.iinfo` is an object that represents the numerical properties of a integer
:class:`torch.dtype` (i.e. ``torch.uint8``, ``torch.int8``, ``torch.int16``, ``torch.int32``, and ``torch.int64``). This is similar to `numpy.iinfo <https://docs.scipy.org/doc/numpy/reference/generated/numpy.iinfo.html>`_.

A :class:`torch.iinfo` provides the following attributes:

=========   =====   ========================================
Name        Type    Description
=========   =====   ========================================
bits        int     The number of bits occupied by the type.
max         int     The largest representable number.
min         int     The smallest representable number.
=========   =====   ========================================
