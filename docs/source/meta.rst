Meta device
============

The "meta" device is an abstract device which denotes a tensor which records
only metadata, but no actual data.  Meta tensors have two primary use cases:

* Models can be loaded on the meta device, allowing you to load a
  representation of the model without actually loading the actual parameters
  into memory.  This can be helpful if you need to make transformations on
  the model before you load the actual data.

* Most operations can be performed on meta tensors, producing new meta
  tensors that describe what the result would have been if you performed
  the operation on a real tensor.  You can use this to perform abstract
  analysis without needing to spend time on compute or space to represent
  the actual tensors.  Because meta tensors do not have real data, you cannot
  perform data-dependent operations like :func:`torch.nonzero` or
  :meth:`~torch.Tensor.item`.  In some cases, not all device types (e.g., CPU
  and CUDA) have exactly the same output metadata for an operation; we
  typically prefer representing the CUDA behavior faithfully in this
  situation.

.. warning::

    Although in principle meta tensor computation should always be faster than
    an equivalent CPU/CUDA computation, many meta tensor implementations are
    implemented in Python and have not been ported to C++ for speed, so you
    may find that you get lower absolute framework latency with small CPU tensors.

Idioms for working with meta tensors
------------------------------------

An object can be loaded with :func:`torch.load` onto meta device by specifying
``map_location='meta'``::

    >>> torch.save(torch.randn(2), 'foo.pt')
    >>> torch.load('foo.pt', map_location='meta')
    tensor(..., device='meta', size=(2,))

If you have some arbitrary code which performs some tensor construction without
explicitly specifying a device, you can override it to instead construct on meta device by using
the :func:`torch.device` context manager::

    >>> with torch.device('meta'):
    ...     print(torch.randn(30, 30))
    ...
    tensor(..., device='meta', size=(30, 30))

This is especially helpful NN module construction, where you often are not
able to explicitly pass in a device for initialization::

    >>> from torch.nn.modules import Linear
    >>> with torch.device('meta'):
    ...     print(Linear(20, 30))
    ...
    Linear(in_features=20, out_features=30, bias=True)

You cannot convert a meta tensor directly to a CPU/CUDA tensor, because the
meta tensor stores no data and we do not know what the correct data values for
your new tensor are::

    >>> torch.ones(5, device='meta').to("cpu")
    Traceback (most recent call last):
      File "<stdin>", line 1, in <module>
    NotImplementedError: Cannot copy out of meta tensor; no data!

Use a factory function like :func:`torch.empty_like` to explicitly specify how
you would like the missing data to be filled in.

NN modules have a convenience method :meth:`torch.nn.Module.to_empty` that
allow you to the module to another device, leaving all parameters
uninitialized.  You are expected to explicitly reinitialize the parameters
manually::

    >>> from torch.nn.modules import Linear
    >>> with torch.device('meta'):
    ...     m = Linear(20, 30)
    >>> m.to_empty(device="cpu")
    Linear(in_features=20, out_features=30, bias=True)

:mod:`torch._subclasses.meta_utils` contains undocumented utilities for taking
an arbitrary Tensor and constructing an equivalent meta Tensor with high
fidelity.  These APIs are experimental and may be changed in a BC breaking way
at any time.
