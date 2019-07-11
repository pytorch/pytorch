.. currentmodule:: torch

.. _tensor-attributes-doc:

Tensor Attributes
=================

Each ``torch.Tensor`` has a :class:`torch.dtype`, :class:`torch.device`, and :class:`torch.layout`.

.. _dtype-doc:

torch.dtype
-----------

.. class:: torch.dtype

A :class:`torch.dtype` is an object that represents the data type of a
:class:`torch.Tensor`. PyTorch has eight different data types:

========================   ===========================================   ===========================
Data type                  dtype                                         Tensor types
========================   ===========================================   ===========================
32-bit floating point      ``torch.float32`` or ``torch.float``          ``torch.*.FloatTensor``
64-bit floating point      ``torch.float64`` or ``torch.double``         ``torch.*.DoubleTensor``
16-bit floating point      ``torch.float16`` or ``torch.half``           ``torch.*.HalfTensor``
8-bit integer (unsigned)   ``torch.uint8``                               ``torch.*.ByteTensor``
8-bit integer (signed)     ``torch.int8``                                ``torch.*.CharTensor``
16-bit integer (signed)    ``torch.int16`` or ``torch.short``            ``torch.*.ShortTensor``
32-bit integer (signed)    ``torch.int32`` or ``torch.int``              ``torch.*.IntTensor``
64-bit integer (signed)    ``torch.int64`` or ``torch.long``             ``torch.*.LongTensor``
========================   ===========================================   ===========================

To find out if a :class:`torch.dtype` is a floating point data type, the property :attr:`is_floating_point`
can be used, which returns ``True`` if the data type is a floating point data type.

.. _device-doc:

torch.device
------------

.. class:: torch.device

A :class:`torch.device` is an object representing the device on which a :class:`torch.Tensor` is
or will be allocated.

The :class:`torch.device` contains a device type (``'cpu'`` or ``'cuda'``) and optional device ordinal for the
device type.  If the device ordinal is not present, this represents the current device for the device type;
e.g. a :class:`torch.Tensor` constructed with device ``'cuda'`` is equivalent to ``'cuda:X'`` where X is the result of
:func:`torch.cuda.current_device()`.

A :class:`torch.Tensor`'s device can be accessed via the :attr:`Tensor.device` property.

A :class:`torch.device` can be constructed via a string or via a string and device ordinal

Via a string:
::

    >>> torch.device('cuda:0')
    device(type='cuda', index=0)

    >>> torch.device('cpu')
    device(type='cpu')

    >>> torch.device('cuda')  # current cuda device
    device(type='cuda')

Via a string and device ordinal:

::

    >>> torch.device('cuda', 0)
    device(type='cuda', index=0)

    >>> torch.device('cpu', 0)
    device(type='cpu', index=0)

.. note::
   The :class:`torch.device` argument in functions can generally be substituted with a string.
   This allows for fast prototyping of code.

   >>> # Example of a function that takes in a torch.device
   >>> cuda1 = torch.device('cuda:1')
   >>> torch.randn((2,3), device=cuda1)

   >>> # You can substitute the torch.device with a string
   >>> torch.randn((2,3), device='cuda:1')

.. note::
   For legacy reasons, a device can be constructed via a single device ordinal, which is treated
   as a cuda device.  This matches :meth:`Tensor.get_device`, which returns an ordinal for cuda
   tensors and is not supported for cpu tensors.

   >>> torch.device(1)
   device(type='cuda', index=1)

.. note::
   Methods which take a device will generally accept a (properly formatted) string
   or (legacy) integer device ordinal, i.e. the following are all equivalent:

   >>> torch.randn((2,3), device=torch.device('cuda:1'))
   >>> torch.randn((2,3), device='cuda:1')
   >>> torch.randn((2,3), device=1)  # legacy


.. _layout-doc:

torch.layout
------------

.. class:: torch.layout

A :class:`torch.layout` is an object that represents the memory layout of a
:class:`torch.Tensor`. Currently, we support ``torch.strided`` (dense Tensors)
and have experimental support for ``torch.sparse_coo`` (sparse COO Tensors).

``torch.strided`` represents dense Tensors and is the memory layout that
is most commonly used. Each strided tensor has an associated
:class:`torch.Storage`, which holds its data. These tensors provide
multi-dimensional, `strided <https://en.wikipedia.org/wiki/Stride_of_an_array>`_
view of a storage. Strides are a list of integers: the k-th stride
represents the jump in the memory necessary to go from one element to the
next one in the k-th dimension of the Tensor. This concept makes it possible
to perform many tensor operations efficiently.

Example::

    >>> x = torch.Tensor([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
    >>> x.stride()
    (5, 1)

    >>> x.t().stride()
    (1, 5)

For more information on ``torch.sparse_coo`` tensors, see :ref:`sparse-docs`.
