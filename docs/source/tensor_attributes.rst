.. currentmodule:: torch

.. _tensor-attributes-doc:

Tensor Attributes
=================

Each ``torch.Tensor`` has a :class:`torch.dtype`, :class:`torch.device`, and :class:`torch.layout`.

.. _dtype-doc:

torch.dtype
-----------

.. class:: dtype

A :class:`torch.dtype` is an object that represents the data type of a
:class:`torch.Tensor`. PyTorch has several different data types:

**Floating point dtypes**

===============================================  ===============================
dtype                                            description
===============================================  ===============================
``torch.float32`` or ``torch.float``             32-bit floating point, as defined in https://en.wikipedia.org/wiki/IEEE_754
``torch.float64`` or ``torch.double``            64-bit floating point, as defined in https://en.wikipedia.org/wiki/IEEE_754
``torch.float16`` or ``torch.half``              16-bit floating point, as defined in https://en.wikipedia.org/wiki/IEEE_754, S-E-M 1-5-10
``torch.bfloat16``                               16-bit floating point, sometimes referred to as Brain floating point, S-E-M 1-8-7
``torch.complex32`` or ``torch.chalf``           32-bit complex with two `float16` components
``torch.complex64`` or ``torch.cfloat``          64-bit complex with two `float32` components
``torch.complex128`` or ``torch.cdouble``        128-bit complex with two `float64` components
``torch.float8_e4m3fn`` [shell]_, [1]_           8-bit floating point, S-E-M 1-4-3, from https://arxiv.org/abs/2209.05433
``torch.float8_e5m2`` [shell]_                   8-bit floating point, S-E-M 1-5-2, from https://arxiv.org/abs/2209.05433
``torch.float8_e4m3fnuz`` [shell]_, [1]_         8-bit floating point, S-E-M 1-4-3, from https://arxiv.org/pdf/2206.02915
``torch.float8_e5m2fnuz`` [shell]_, [1]_         8-bit floating point, S-E-M 1-5-2, from https://arxiv.org/pdf/2206.02915
``torch.float8_e8m0fnu`` [shell]_, [1]_          8-bit floating point, S-E-M 0-8-0, from https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf
``torch.float4_e2m1fn_x2`` [shell]_, [1]_, [3]_  packed 4-bit floating point, S-E-M 1-2-1, from https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf
===============================================  ===============================

**Integer dtypes**

=========================================  ===============================
dtype                                      description
=========================================  ===============================
``torch.uint8``                            8-bit integer (unsigned)
``torch.int8``                             8-bit integer (signed)
``torch.uint16`` [shell]_, [2]_            16-bit integer (unsigned)
``torch.int16`` or ``torch.short``         16-bit integer (signed)
``torch.uint32`` [shell]_, [2]_            32-bit integer (unsigned)
``torch.int32`` or ``torch.int``           32-bit integer (signed)
``torch.uint64`` [shell]_, [2]_            64-bit integer (unsigned)
``torch.int64`` or ``torch.long``          64-bit integer (signed)
``torch.bool``                             Boolean
=========================================  ===============================

.. [shell] a shell dtype is a specialized dtype with limited op and backend support.
  Specifically, ops that support tensor creation (``torch.empty``, ``torch.fill``, ``torch.zeros``)
  and operations which do not peek inside the data elements (``torch.cat``, ``torch.view``, ``torch.reshape``)
  are supported.  Ops that peek inside the data elements such as casting,
  matrix multiplication, nan/inf checks are supported only on a case by
  case basis, depending on maturity and presence of hardware accelerated kernels
  and established use cases.

.. [1] The "fn", "fnu" and "fnuz" dtype suffixes mean:
  "f" - finite value encodings only, no infinity;
  "n" - nan value encodings differ from the IEEE spec;
  "uz" - "unsigned zero" only, i.e. no negative zero encoding

.. [2]
  Unsigned types asides from ``uint8`` are currently planned to only have
  limited support in eager mode (they primarily exist to assist usage with
  torch.compile); if you need eager support and the extra range is not needed,
  we recommend using their signed variants instead.  See
  https://github.com/pytorch/pytorch/issues/58734 for more details.

.. [3]
  The `torch.float4_e2m1fn_x2` dtype represents two 4-bit values packed into a byte.
  Note that PyTorch operations which modify the shape/strides of a tensor (such as transposing)
  operate at the byte boundary and will **not** unpack/repack sub-byte values.


**Note**: legacy constructors such as ``torch.*.FloatTensor``, ``torch.*.DoubleTensor``, ``torch.*.HalfTensor``,
``torch.*.BFloat16Tensor``, ``torch.*.ByteTensor``, ``torch.*.CharTensor``, ``torch.*.ShortTensor``, ``torch.*.IntTensor``,
``torch.*.LongTensor``, ``torch.*.BoolTensor`` only remain for backwards compatibility and should no longer be used.

To find out if a :class:`torch.dtype` is a floating point data type, the property :attr:`is_floating_point`
can be used, which returns ``True`` if the data type is a floating point data type.

To find out if a :class:`torch.dtype` is a complex data type, the property :attr:`is_complex`
can be used, which returns ``True`` if the data type is a complex data type.

.. _type-promotion-doc:

When the dtypes of inputs to an arithmetic operation (`add`, `sub`, `div`, `mul`) differ, we promote
by finding the minimum dtype that satisfies the following rules:

* If the type of a scalar operand is of a higher category than tensor operands
  (where complex > floating > integral > boolean), we promote to a type with sufficient size to hold
  all scalar operands of that category.
* If a zero-dimension tensor operand has a higher category than dimensioned operands,
  we promote to a type with sufficient size and category to hold all zero-dim tensor operands of
  that category.
* If there are no higher-category zero-dim operands, we promote to a type with sufficient size
  and category to hold all dimensioned operands.

A floating point scalar operand has dtype `torch.get_default_dtype()` and an integral
non-boolean scalar operand has dtype `torch.int64`. Unlike numpy, we do not inspect
values when determining the minimum `dtypes` of an operand.  Complex types
are not yet supported. Promotion for shell dtypes is not defined.

Promotion Examples::

    >>> float_tensor = torch.ones(1, dtype=torch.float)
    >>> double_tensor = torch.ones(1, dtype=torch.double)
    >>> complex_float_tensor = torch.ones(1, dtype=torch.complex64)
    >>> complex_double_tensor = torch.ones(1, dtype=torch.complex128)
    >>> int_tensor = torch.ones(1, dtype=torch.int)
    >>> long_tensor = torch.ones(1, dtype=torch.long)
    >>> uint_tensor = torch.ones(1, dtype=torch.uint8)
    >>> bool_tensor = torch.ones(1, dtype=torch.bool)
    # zero-dim tensors
    >>> long_zerodim = torch.tensor(1, dtype=torch.long)
    >>> int_zerodim = torch.tensor(1, dtype=torch.int)

    >>> torch.add(5, 5).dtype
    torch.int64
    # 5 is an int64, but does not have higher category than int_tensor so is not considered.
    >>> (int_tensor + 5).dtype
    torch.int32
    >>> (int_tensor + long_zerodim).dtype
    torch.int32
    >>> (long_tensor + int_tensor).dtype
    torch.int64
    >>> (bool_tensor + long_tensor).dtype
    torch.int64
    >>> (bool_tensor + uint_tensor).dtype
    torch.uint8
    >>> (float_tensor + double_tensor).dtype
    torch.float64
    >>> (complex_float_tensor + complex_double_tensor).dtype
    torch.complex128
    >>> (bool_tensor + int_tensor).dtype
    torch.int32
    # Since long is a different kind than float, result dtype only needs to be large enough
    # to hold the float.
    >>> torch.add(long_tensor, float_tensor).dtype
    torch.float32

When the output tensor of an arithmetic operation is specified, we allow casting to its `dtype` except that:
  * An integral output tensor cannot accept a floating point tensor.
  * A boolean output tensor cannot accept a non-boolean tensor.
  * A non-complex output tensor cannot accept a complex tensor

Casting Examples::

    # allowed:
    >>> float_tensor *= float_tensor
    >>> float_tensor *= int_tensor
    >>> float_tensor *= uint_tensor
    >>> float_tensor *= bool_tensor
    >>> float_tensor *= double_tensor
    >>> int_tensor *= long_tensor
    >>> int_tensor *= uint_tensor
    >>> uint_tensor *= int_tensor

    # disallowed (RuntimeError: result type can't be cast to the desired output type):
    >>> int_tensor *= float_tensor
    >>> bool_tensor *= int_tensor
    >>> bool_tensor *= uint_tensor
    >>> float_tensor *= complex_float_tensor


.. _device-doc:

torch.device
------------

.. class:: device

A :class:`torch.device` is an object representing the device on which a :class:`torch.Tensor` is
or will be allocated.

The :class:`torch.device` contains a device type (most commonly "cpu" or
"cuda", but also potentially :doc:`"mps" <mps>`, :doc:`"xpu" <xpu>`,
`"xla" <https://github.com/pytorch/xla/>`_ or :doc:`"meta" <meta>`) and optional
device ordinal for the device type. If the device ordinal is not present, this object will always represent
the current device for the device type, even after :func:`torch.cuda.set_device()` is called; e.g.,
a :class:`torch.Tensor` constructed with device ``'cuda'`` is equivalent to ``'cuda:X'`` where X is
the result of :func:`torch.cuda.current_device()`.

A :class:`torch.Tensor`'s device can be accessed via the :attr:`Tensor.device` property.

A :class:`torch.device` can be constructed using:

  * A device string, which is a string representation of the device type and optionally the device ordinal.
  * A device type and a device ordinal.
  * A device ordinal, where the current :ref:`accelerator<accelerators>` type will be used.

Via a device string:
::

    >>> torch.device('cuda:0')
    device(type='cuda', index=0)

    >>> torch.device('cpu')
    device(type='cpu')

    >>> torch.device('mps')
    device(type='mps')

    >>> torch.device('cuda')  # implicit index is the "current device index"
    device(type='cuda')

Via a device type and a device ordinal:

::

    >>> torch.device('cuda', 0)
    device(type='cuda', index=0)

    >>> torch.device('mps', 0)
    device(type='mps', index=0)

    >>> torch.device('cpu', 0)
    device(type='cpu', index=0)

Via a device ordinal:

.. note::
   This method will raise a RuntimeError if no accelerator is currently detected.

::

    >>> torch.device(0)  # the current accelerator is cuda
    device(type='cuda', index=0)

    >>> torch.device(1)  # the current accelerator is xpu
    device(type='xpu', index=1)

    >>> torch.device(0)  # no current accelerator detected
    Traceback (most recent call last):
      File "<stdin>", line 1, in <module>
    RuntimeError: Cannot access accelerator device when none is available.

The device object can also be used as a context manager to change the default
device tensors are allocated on:

::

    >>> with torch.device('cuda:1'):
    ...     r = torch.randn(2, 3)
    >>> r.device
    device(type='cuda', index=1)

This context manager has no effect if a factory function is passed an explicit,
non-None device argument.  To globally change the default device, see also
:func:`torch.set_default_device`.

.. warning::

    This function imposes a slight performance cost on every Python
    call to the torch API (not just factory functions).  If this
    is causing problems for you, please comment on
    https://github.com/pytorch/pytorch/issues/92701

.. note::
   The :class:`torch.device` argument in functions can generally be substituted with a string.
   This allows for fast prototyping of code.

   >>> # Example of a function that takes in a torch.device
   >>> cuda1 = torch.device('cuda:1')
   >>> torch.randn((2,3), device=cuda1)

   >>> # You can substitute the torch.device with a string
   >>> torch.randn((2,3), device='cuda:1')

.. note::
   Methods which take a device will generally accept a (properly formatted) string
   or an integer device ordinal, i.e. the following are all equivalent:

   >>> torch.randn((2,3), device=torch.device('cuda:1'))
   >>> torch.randn((2,3), device='cuda:1')
   >>> torch.randn((2,3), device=1)  # equivalent to 'cuda:1' if the current accelerator is cuda

.. note::
   Tensors are never moved automatically between devices and require an explicit call from the user. Scalar Tensors (with tensor.dim()==0) are the only exception to this rule and they are automatically transferred from CPU to GPU when needed as this operation can be done "for free".
   Example:

   >>> # two scalars
   >>> torch.ones(()) + torch.ones(()).cuda()  # OK, scalar auto-transferred from CPU to GPU
   >>> torch.ones(()).cuda() + torch.ones(())  # OK, scalar auto-transferred from CPU to GPU

   >>> # one scalar (CPU), one vector (GPU)
   >>> torch.ones(()) + torch.ones(1).cuda()  # OK, scalar auto-transferred from CPU to GPU
   >>> torch.ones(1).cuda() + torch.ones(())  # OK, scalar auto-transferred from CPU to GPU

   >>> # one scalar (GPU), one vector (CPU)
   >>> torch.ones(()).cuda() + torch.ones(1)  # Fail, scalar not auto-transferred from GPU to CPU and non-scalar not auto-transferred from CPU to GPU
   >>> torch.ones(1) + torch.ones(()).cuda()  # Fail, scalar not auto-transferred from GPU to CPU and non-scalar not auto-transferred from CPU to GPU

.. _layout-doc:

torch.layout
------------

.. class:: layout

.. warning::
  The ``torch.layout`` class is in beta and subject to change.

A :class:`torch.layout` is an object that represents the memory layout of a
:class:`torch.Tensor`. Currently, we support ``torch.strided`` (dense Tensors)
and have beta support for ``torch.sparse_coo`` (sparse COO Tensors).

``torch.strided`` represents dense Tensors and is the memory layout that
is most commonly used. Each strided tensor has an associated
:class:`torch.Storage`, which holds its data. These tensors provide
multi-dimensional, `strided <https://en.wikipedia.org/wiki/Stride_of_an_array>`_
view of a storage. Strides are a list of integers: the k-th stride
represents the jump in the memory necessary to go from one element to the
next one in the k-th dimension of the Tensor. This concept makes it possible
to perform many tensor operations efficiently.

Example::

    >>> x = torch.tensor([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
    >>> x.stride()
    (5, 1)

    >>> x.t().stride()
    (1, 5)

For more information on ``torch.sparse_coo`` tensors, see :ref:`sparse-docs`.

torch.memory_format
-------------------

.. class:: memory_format

A :class:`torch.memory_format` is an object representing the memory format on which a :class:`torch.Tensor` is
or will be allocated.

Possible values are:

- ``torch.contiguous_format``:
  Tensor is or will be allocated in dense non-overlapping memory. Strides represented by values in decreasing order.

- ``torch.channels_last``:
  Tensor is or will be allocated in dense non-overlapping memory. Strides represented by values in
  ``strides[0] > strides[2] > strides[3] > strides[1] == 1`` aka NHWC order.

- ``torch.channels_last_3d``:
  Tensor is or will be allocated in dense non-overlapping memory. Strides represented by values in
  ``strides[0] > strides[2] > strides[3] > strides[4] > strides[1] == 1`` aka NDHWC order.

- ``torch.preserve_format``:
  Used in functions like `clone` to preserve the memory format of the input tensor. If input tensor is
  allocated in dense non-overlapping memory, the output tensor strides will be copied from the input.
  Otherwise output strides will follow ``torch.contiguous_format``
