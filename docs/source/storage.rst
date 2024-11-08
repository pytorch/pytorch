torch.Storage
=============

In PyTorch, a regular tensor is a multi-dimensional array that is defined by the following components:

  - Storage: The actual data of the tensor, stored as a contiguous, one-dimensional array of bytes.
  - ``dtype``: The data type of the elements in the tensor, such as torch.float32 or torch.int64.
  - ``shape``: A tuple indicating the size of the tensor in each dimension.
  - Stride: The step size needed to move from one element to the next in each dimension.
  - Offset: The starting point in the storage from which the tensor data begins.

These components together define the structure and data of a tensor, with the storage holding the
actual data and the rest serving as metadata.

Untyped Storage API
-------------------

An untyped storage is a fundamental concept in PyTorch, representing a contiguous, one-dimensional array of bytes.
It serves as the underlying data container for tensors.
A :class:`torch.UntypedStorage` is a contiguous, one-dimensional array of elements. Its length is equal to the number of
bytes of the tensor.
In general, a tensor created in PyTorch using regular constructors such as :func:`~torch.zeros`, :func:`~torch.zeros_like`
or :func:`~torch.Tensor.new_zeros` will produce tensors where there is a one-to-one correspondence between the tensor
storage and the tensor itself.

But a storage also allows multiple tensors to share the same storage.
For instance, any view of a tensor (obtained through :meth:`~torch.Tensor.view` or some but not all kinds of indexing
like integers and slices) will point to the same underlying tensor as the original one.
When serializing and deserializing tensors that share a common storage, the relationship is preserved, and the tensors
continue to point to the same storage. Interestingly, deserializing multiple tensors that point to a single storage
can be faster than deserializing multiple independent tensors.

A tensor storage can be accessed through the :meth:`~torch.Tensor.untyped_storage` method. This will return an object of
type :class:`torch.UntypedStorage`.
Fortunately, storages have a unique identifier called accessed through the :meth:`torch.UntypedStorage.data_ptr` method.
In regular settings, two tensors with the same data storage will have the same storage ``data_ptr``. However, this is
not true for the tensors themselves: for instance, a tensor with a ``data`` and a ``grad`` value actually points to two
independent storages and hence require a ``data_ptr()`` of its own.

Untyped storages are somewhat independent of the tensors that are built on them. Practically, this means that tensors
with different dtypes or shape can point to the same storage. It also implies that a tensor storage can be changed, as
the following example shows:

    >>> t = torch.ones(3)
    >>> s0 = t.untyped_storage()
    >>> s0
     0
     0
     128
     63
     0
     0
     128
     63
     0
     0
     128
     63
    [torch.storage.UntypedStorage(device=cpu) of size 12]
    >>> s1 = s0.clone()
    >>> s1.fill_(0)
     0
     0
     0
     0
     0
     0
     0
     0
     0
     0
     0
     0
    [torch.storage.UntypedStorage(device=cpu) of size 12]
    >>> # Fill the tensor with a zeroed storage
    >>> t.set_(s1, storage_offset=t.storage_offset(), stride=t.stride(), size=t.size())
    tensor([0., 0., 0.])

Other than their ``data_ptr``, untyped storage also have other attributes such as :attr:`~torch.UntypedStorage.filename`
(in case the storage points to a file on disk), :attr:`~torch.UntypedStorage.device` or
:attr:`~torch.UntypedStorage.is_cuda` for device checks. A storage can also be manipulated in-place or
out-of-place with methods like :attr:`~torch.UntypedStorage.copy_`, :attr:`~torch.UntypedStorage.fill_` or
:attr:`~torch.UntypedStorage.pin_memory`. FOr more information, check the API
reference below. Keep in mind that modifying storages is a low-level API and comes with some risks.

.. autoclass:: torch.UntypedStorage
   :members:
   :undoc-members:
   :inherited-members:

Special cases
-------------

We mentioned that a tensor that has a non-None ``grad`` attribute has actually two pieces of data within it.
In this case, :meth:`~torch.Tensor.untyped_storage` will return the storage of the :attr:`~torch.Tensor.data` attribute,
whereas the storage of the gradient can be obtained through ``tensor.grad.untyped_storage()``.

    >>> t = torch.zeros(3, requires_grad=True)
    >>> t.sum().backward()
    >>> assert list(t.untyped_storage()) == [0] * 12  # the storage of the tensor is just 0s
    >>> assert list(t.grad.untyped_storage()) != [0] * 12  # the storage of the gradient isn't

There are also special cases where tensors do not have a typical storage, or no storage at all:
  - Tensors on ``"meta"`` device: Tensors on the ``"meta"`` device are used for shape inference
    and do not hold actual data.
  - Fake Tensors: Another internal tool used by PyTorch's compiler are
    `FakeTensor <https://pytorch.org/docs/stable/torch.compiler_fake_tensor.html>`_ which are based on a similar idea.

Tensor subclasses or tensor-like object may also display unusual behaviours.

Legacy Typed Storage
--------------------

.. warning::
  For historical context, PyTorch previously used typed storage classes, which are
  now deprecated and should be avoided. The following details this API in case you
  should encounter it, although its usage is highly discouraged.
  All storage classes except for :class:`torch.UntypedStorage` will be removed
  in the future, and :class:`torch.UntypedStorage` will be used in all cases.

:class:`torch.Storage` is an alias for the storage class that corresponds with
the default data type (:func:`torch.get_default_dtype()`). For instance, if the
default data type is :attr:`torch.float`, :class:`torch.Storage` resolves to
:class:`torch.FloatStorage`.

The :class:`torch.<type>Storage` and :class:`torch.cuda.<type>Storage` classes,
like :class:`torch.FloatStorage`, :class:`torch.IntStorage`, etc., are not
actually ever instantiated. Calling their constructors creates
a :class:`torch.TypedStorage` with the appropriate :class:`torch.dtype` and
:class:`torch.device`.  :class:`torch.<type>Storage` classes have all of the
same class methods that :class:`torch.TypedStorage` has.

A :class:`torch.TypedStorage` is a contiguous, one-dimensional array of
elements of a particular :class:`torch.dtype`. It can be given any
:class:`torch.dtype`, and the internal data will be interpreted appropriately.
:class:`torch.TypedStorage` contains a :class:`torch.UntypedStorage` which
holds the data as an untyped array of bytes.

Every strided :class:`torch.Tensor` contains a :class:`torch.TypedStorage`,
which stores all of the data that the :class:`torch.Tensor` views.


.. autoclass:: torch.TypedStorage
   :members:
   :undoc-members:
   :inherited-members:

.. autoclass:: torch.DoubleStorage
   :members:
   :undoc-members:

.. autoclass:: torch.FloatStorage
   :members:
   :undoc-members:

.. autoclass:: torch.HalfStorage
   :members:
   :undoc-members:

.. autoclass:: torch.LongStorage
   :members:
   :undoc-members:

.. autoclass:: torch.IntStorage
   :members:
   :undoc-members:

.. autoclass:: torch.ShortStorage
   :members:
   :undoc-members:

.. autoclass:: torch.CharStorage
   :members:
   :undoc-members:

.. autoclass:: torch.ByteStorage
   :members:
   :undoc-members:

.. autoclass:: torch.BoolStorage
   :members:
   :undoc-members:

.. autoclass:: torch.BFloat16Storage
   :members:
   :undoc-members:

.. autoclass:: torch.ComplexDoubleStorage
   :members:
   :undoc-members:

.. autoclass:: torch.ComplexFloatStorage
   :members:
   :undoc-members:

.. autoclass:: torch.QUInt8Storage
   :members:
   :undoc-members:

.. autoclass:: torch.QInt8Storage
   :members:
   :undoc-members:

.. autoclass:: torch.QInt32Storage
   :members:
   :undoc-members:

.. autoclass:: torch.QUInt4x2Storage
   :members:
   :undoc-members:

.. autoclass:: torch.QUInt2x4Storage
   :members:
   :undoc-members:
