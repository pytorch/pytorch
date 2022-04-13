torch.Storage
===================================

A :class:`torch._TypedStorage` is a contiguous, one-dimensional array of
elements of a particular :class:`torch.dtype`. It can be given any
:class:`torch.dtype`, and the internal data will be interpretted appropriately.

Every strided :class:`torch.Tensor` contains a :class:`torch._TypedStorage`,
which stores all of the data that the :class:`torch.Tensor` views.

For backward compatibility, there are also :class:`torch.<type>Storage` classes
(like :class:`torch.FloatStorage`, :class:`torch.IntStorage`, etc). These
classes are not actually instantiated, and calling their constructors creates
a :class:`torch._TypedStorage` with the appropriate :class:`torch.dtype`.
:class:`torch.<type>Storage` classes have all of the same class methods that
:class:`torch._TypedStorage` has.

Also for backward compatibility, :class:`torch.Storage` is an alias for the
storage class that corresponds with the default data type
(:func:`torch.get_default_dtype()`). For instance, if the default data type is
:attr:`torch.float`, :class:`torch.Storage` resolves to
:class:`torch.FloatStorage`.


.. autoclass:: torch._TypedStorage
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
