torch.Storage
===================================

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

.. autoclass:: torch.UntypedStorage
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
