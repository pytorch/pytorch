.. currentmodule:: torch

.. _device-doc:

torch.device
===================================

A :class:`torch.device` is an object representing the device on which a :class:`torch.Tensor` is
or will be allocated.

The :class:`torch.device` contains a device type (``'cpu'`` or ``'cuda'``) and optional device ordinal for the
device type.  If the device ordinal is not present, this represents the current device for the device type;
e.g. a :class:`torch.Tensor` constructed with device ``'cuda'`` is equivalent to ``'cuda:X'`` where X is the result of
:func:`torch.cuda.get_device()`.

A :class:`torch.device` can be constructed via a string, a string and device ordinal, or a device ordinal.


Via a string:
::

    >>> torch.device('cuda:0')
    device(device_type='cuda', device_index=0)

    >>> torch.device('cpu')
    device(device_type='cpu')

    >>> torch.device('cuda')  # current cuda device
    device(device_type='cuda')


Via a string and device ordinal:

::

    >>> torch.device('cuda', 0)
    device(device_type='cuda', device_index=0)

    >>> torch.device('cpu', 0)
    device(device_type='cpu', device_index=0)

Via a device ordinal (device ordinals by themselves are treated as cuda devices).

::

  >>> torch.device(1)
  device(device_type='cuda', device_index=1)

.. note::
   Methods which take a device will generally accept a (properly formatted) string
   or integer device ordinal, e.g. the following are all equivalent:

   >>> torch.randn((2,3), device=1)
   >>> torch.randn((2,3), device='cuda:1')
   >>> torch.randn((2,3), device=torch.device('cuda:1'))
