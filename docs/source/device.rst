.. currentmodule:: torch

.. _device-doc:

torch.device
===================================

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
