"""
=================
Structured Arrays
=================

Introduction
============

Structured arrays are ndarrays whose datatype is a composition of simpler
datatypes organized as a sequence of named :term:`fields <field>`. For example,
::

 >>> x = np.array([('Rex', 9, 81.0), ('Fido', 3, 27.0)],
 ...              dtype=[('name', 'U10'), ('age', 'i4'), ('weight', 'f4')])
 >>> x
 array([('Rex', 9, 81.), ('Fido', 3, 27.)],
       dtype=[('name', 'U10'), ('age', '<i4'), ('weight', '<f4')])

Here ``x`` is a one-dimensional array of length two whose datatype is a
structure with three fields: 1. A string of length 10 or less named 'name', 2.
a 32-bit integer named 'age', and 3. a 32-bit float named 'weight'.

If you index ``x`` at position 1 you get a structure::

 >>> x[1]
 ('Fido', 3, 27.0)

You can access and modify individual fields of a structured array by indexing
with the field name::

 >>> x['age']
 array([9, 3], dtype=int32)
 >>> x['age'] = 5
 >>> x
 array([('Rex', 5, 81.), ('Fido', 5, 27.)],
       dtype=[('name', 'U10'), ('age', '<i4'), ('weight', '<f4')])

Structured datatypes are designed to be able to mimic 'structs' in the C
language, and share a similar memory layout. They are meant for interfacing with
C code and for low-level manipulation of structured buffers, for example for
interpreting binary blobs. For these purposes they support specialized features
such as subarrays, nested datatypes, and unions, and allow control over the
memory layout of the structure.

Users looking to manipulate tabular data, such as stored in csv files, may find
other pydata projects more suitable, such as xarray, pandas, or DataArray.
These provide a high-level interface for tabular data analysis and are better
optimized for that use. For instance, the C-struct-like memory layout of
structured arrays in numpy can lead to poor cache behavior in comparison.

.. _defining-structured-types:

Structured Datatypes
====================

A structured datatype can be thought of as a sequence of bytes of a certain
length (the structure's :term:`itemsize`) which is interpreted as a collection
of fields. Each field has a name, a datatype, and a byte offset within the
structure. The datatype of a field may be any numpy datatype including other
structured datatypes, and it may also be a :term:`subarray data type` which
behaves like an ndarray of a specified shape. The offsets of the fields are
arbitrary, and fields may even overlap. These offsets are usually determined
automatically by numpy, but can also be specified.

Structured Datatype Creation
----------------------------

Structured datatypes may be created using the function :func:`numpy.dtype`.
There are 4 alternative forms of specification which vary in flexibility and
conciseness. These are further documented in the
:ref:`Data Type Objects <arrays.dtypes.constructing>` reference page, and in
summary they are:

1.   A list of tuples, one tuple per field

     Each tuple has the form ``(fieldname, datatype, shape)`` where shape is
     optional. ``fieldname`` is a string (or tuple if titles are used, see
     :ref:`Field Titles <titles>` below), ``datatype`` may be any object
     convertible to a datatype, and ``shape`` is a tuple of integers specifying
     subarray shape.

      >>> np.dtype([('x', 'f4'), ('y', np.float32), ('z', 'f4', (2, 2))])
      dtype([('x', '<f4'), ('y', '<f4'), ('z', '<f4', (2, 2))])

     If ``fieldname`` is the empty string ``''``, the field will be given a
     default name of the form ``f#``, where ``#`` is the integer index of the
     field, counting from 0 from the left::

      >>> np.dtype([('x', 'f4'), ('', 'i4'), ('z', 'i8')])
      dtype([('x', '<f4'), ('f1', '<i4'), ('z', '<i8')])

     The byte offsets of the fields within the structure and the total
     structure itemsize are determined automatically.

2.   A string of comma-separated dtype specifications

     In this shorthand notation any of the :ref:`string dtype specifications
     <arrays.dtypes.constructing>` may be used in a string and separated by
     commas. The itemsize and byte offsets of the fields are determined
     automatically, and the field names are given the default names ``f0``,
     ``f1``, etc. ::

      >>> np.dtype('i8, f4, S3')
      dtype([('f0', '<i8'), ('f1', '<f4'), ('f2', 'S3')])
      >>> np.dtype('3int8, float32, (2, 3)float64')
      dtype([('f0', 'i1', (3,)), ('f1', '<f4'), ('f2', '<f8', (2, 3))])

3.   A dictionary of field parameter arrays

     This is the most flexible form of specification since it allows control
     over the byte-offsets of the fields and the itemsize of the structure.

     The dictionary has two required keys, 'names' and 'formats', and four
     optional keys, 'offsets', 'itemsize', 'aligned' and 'titles'. The values
     for 'names' and 'formats' should respectively be a list of field names and
     a list of dtype specifications, of the same length. The optional 'offsets'
     value should be a list of integer byte-offsets, one for each field within
     the structure. If 'offsets' is not given the offsets are determined
     automatically. The optional 'itemsize' value should be an integer
     describing the total size in bytes of the dtype, which must be large
     enough to contain all the fields.
     ::

      >>> np.dtype({'names': ['col1', 'col2'], 'formats': ['i4', 'f4']})
      dtype([('col1', '<i4'), ('col2', '<f4')])
      >>> np.dtype({'names': ['col1', 'col2'],
      ...           'formats': ['i4', 'f4'],
      ...           'offsets': [0, 4],
      ...           'itemsize': 12})
      dtype({'names':['col1','col2'], 'formats':['<i4','<f4'], 'offsets':[0,4], 'itemsize':12})

     Offsets may be chosen such that the fields overlap, though this will mean
     that assigning to one field may clobber any overlapping field's data. As
     an exception, fields of :class:`numpy.object` type cannot overlap with
     other fields, because of the risk of clobbering the internal object
     pointer and then dereferencing it.

     The optional 'aligned' value can be set to ``True`` to make the automatic
     offset computation use aligned offsets (see :ref:`offsets-and-alignment`),
     as if the 'align' keyword argument of :func:`numpy.dtype` had been set to
     True.

     The optional 'titles' value should be a list of titles of the same length
     as 'names', see :ref:`Field Titles <titles>` below.

4.   A dictionary of field names

     The use of this form of specification is discouraged, but documented here
     because older numpy code may use it. The keys of the dictionary are the
     field names and the values are tuples specifying type and offset::

      >>> np.dtype({'col1': ('i1', 0), 'col2': ('f4', 1)})
      dtype([('col1', 'i1'), ('col2', '<f4')])

     This form is discouraged because Python dictionaries do not preserve order
     in Python versions before Python 3.6, and the order of the fields in a
     structured dtype has meaning. :ref:`Field Titles <titles>` may be
     specified by using a 3-tuple, see below.

Manipulating and Displaying Structured Datatypes
------------------------------------------------

The list of field names of a structured datatype can be found in the ``names``
attribute of the dtype object::

 >>> d = np.dtype([('x', 'i8'), ('y', 'f4')])
 >>> d.names
 ('x', 'y')

The field names may be modified by assigning to the ``names`` attribute using a
sequence of strings of the same length.

The dtype object also has a dictionary-like attribute, ``fields``, whose keys
are the field names (and :ref:`Field Titles <titles>`, see below) and whose
values are tuples containing the dtype and byte offset of each field. ::

 >>> d.fields
 mappingproxy({'x': (dtype('int64'), 0), 'y': (dtype('float32'), 8)})

Both the ``names`` and ``fields`` attributes will equal ``None`` for
unstructured arrays. The recommended way to test if a dtype is structured is
with `if dt.names is not None` rather than `if dt.names`, to account for dtypes
with 0 fields.

The string representation of a structured datatype is shown in the "list of
tuples" form if possible, otherwise numpy falls back to using the more general
dictionary form.

.. _offsets-and-alignment:

Automatic Byte Offsets and Alignment
------------------------------------

Numpy uses one of two methods to automatically determine the field byte offsets
and the overall itemsize of a structured datatype, depending on whether
``align=True`` was specified as a keyword argument to :func:`numpy.dtype`.

By default (``align=False``), numpy will pack the fields together such that
each field starts at the byte offset the previous field ended, and the fields
are contiguous in memory. ::

 >>> def print_offsets(d):
 ...     print("offsets:", [d.fields[name][1] for name in d.names])
 ...     print("itemsize:", d.itemsize)
 >>> print_offsets(np.dtype('u1, u1, i4, u1, i8, u2'))
 offsets: [0, 1, 2, 6, 7, 15]
 itemsize: 17

If ``align=True`` is set, numpy will pad the structure in the same way many C
compilers would pad a C-struct. Aligned structures can give a performance
improvement in some cases, at the cost of increased datatype size. Padding
bytes are inserted between fields such that each field's byte offset will be a
multiple of that field's alignment, which is usually equal to the field's size
in bytes for simple datatypes, see :c:member:`PyArray_Descr.alignment`.  The
structure will also have trailing padding added so that its itemsize is a
multiple of the largest field's alignment. ::

 >>> print_offsets(np.dtype('u1, u1, i4, u1, i8, u2', align=True))
 offsets: [0, 1, 4, 8, 16, 24]
 itemsize: 32

Note that although almost all modern C compilers pad in this way by default,
padding in C structs is C-implementation-dependent so this memory layout is not
guaranteed to exactly match that of a corresponding struct in a C program. Some
work may be needed, either on the numpy side or the C side, to obtain exact
correspondence.

If offsets were specified using the optional ``offsets`` key in the
dictionary-based dtype specification, setting ``align=True`` will check that
each field's offset is a multiple of its size and that the itemsize is a
multiple of the largest field size, and raise an exception if not.

If the offsets of the fields and itemsize of a structured array satisfy the
alignment conditions, the array will have the ``ALIGNED`` :attr:`flag
<numpy.ndarray.flags>` set.

A convenience function :func:`numpy.lib.recfunctions.repack_fields` converts an
aligned dtype or array to a packed one and vice versa. It takes either a dtype
or structured ndarray as an argument, and returns a copy with fields re-packed,
with or without padding bytes.

.. _titles:

Field Titles
------------

In addition to field names, fields may also have an associated :term:`title`,
an alternate name, which is sometimes used as an additional description or
alias for the field. The title may be used to index an array, just like a
field name.

To add titles when using the list-of-tuples form of dtype specification, the
field name may be specified as a tuple of two strings instead of a single
string, which will be the field's title and field name respectively. For
example::

 >>> np.dtype([(('my title', 'name'), 'f4')])
 dtype([(('my title', 'name'), '<f4')])

When using the first form of dictionary-based specification, the titles may be
supplied as an extra ``'titles'`` key as described above. When using the second
(discouraged) dictionary-based specification, the title can be supplied by
providing a 3-element tuple ``(datatype, offset, title)`` instead of the usual
2-element tuple::

 >>> np.dtype({'name': ('i4', 0, 'my title')})
 dtype([(('my title', 'name'), '<i4')])

The ``dtype.fields`` dictionary will contain titles as keys, if any
titles are used.  This means effectively that a field with a title will be
represented twice in the fields dictionary. The tuple values for these fields
will also have a third element, the field title. Because of this, and because
the ``names`` attribute preserves the field order while the ``fields``
attribute may not, it is recommended to iterate through the fields of a dtype
using the ``names`` attribute of the dtype, which will not list titles, as
in::

 >>> for name in d.names:
 ...     print(d.fields[name][:2])
 (dtype('int64'), 0)
 (dtype('float32'), 8)

Union types
-----------

Structured datatypes are implemented in numpy to have base type
:class:`numpy.void` by default, but it is possible to interpret other numpy
types as structured types using the ``(base_dtype, dtype)`` form of dtype
specification described in
:ref:`Data Type Objects <arrays.dtypes.constructing>`.  Here, ``base_dtype`` is
the desired underlying dtype, and fields and flags will be copied from
``dtype``. This dtype is similar to a 'union' in C.

Indexing and Assignment to Structured arrays
============================================

Assigning data to a Structured Array
------------------------------------

There are a number of ways to assign values to a structured array: Using python
tuples, using scalar values, or using other structured arrays.

Assignment from Python Native Types (Tuples)
````````````````````````````````````````````

The simplest way to assign values to a structured array is using python tuples.
Each assigned value should be a tuple of length equal to the number of fields
in the array, and not a list or array as these will trigger numpy's
broadcasting rules. The tuple's elements are assigned to the successive fields
of the array, from left to right::

 >>> x = np.array([(1, 2, 3), (4, 5, 6)], dtype='i8, f4, f8')
 >>> x[1] = (7, 8, 9)
 >>> x
 array([(1, 2., 3.), (7, 8., 9.)],
      dtype=[('f0', '<i8'), ('f1', '<f4'), ('f2', '<f8')])

Assignment from Scalars
```````````````````````

A scalar assigned to a structured element will be assigned to all fields. This
happens when a scalar is assigned to a structured array, or when an
unstructured array is assigned to a structured array::

 >>> x = np.zeros(2, dtype='i8, f4, ?, S1')
 >>> x[:] = 3
 >>> x
 array([(3, 3., True, b'3'), (3, 3., True, b'3')],
       dtype=[('f0', '<i8'), ('f1', '<f4'), ('f2', '?'), ('f3', 'S1')])
 >>> x[:] = np.arange(2)
 >>> x
 array([(0, 0., False, b'0'), (1, 1., True, b'1')],
       dtype=[('f0', '<i8'), ('f1', '<f4'), ('f2', '?'), ('f3', 'S1')])

Structured arrays can also be assigned to unstructured arrays, but only if the
structured datatype has just a single field::

 >>> twofield = np.zeros(2, dtype=[('A', 'i4'), ('B', 'i4')])
 >>> onefield = np.zeros(2, dtype=[('A', 'i4')])
 >>> nostruct = np.zeros(2, dtype='i4')
 >>> nostruct[:] = twofield
 Traceback (most recent call last):
 ...
 TypeError: Cannot cast array data from dtype([('A', '<i4'), ('B', '<i4')]) to dtype('int32') according to the rule 'unsafe'

Assignment from other Structured Arrays
```````````````````````````````````````

Assignment between two structured arrays occurs as if the source elements had
been converted to tuples and then assigned to the destination elements. That
is, the first field of the source array is assigned to the first field of the
destination array, and the second field likewise, and so on, regardless of
field names. Structured arrays with a different number of fields cannot be
assigned to each other. Bytes of the destination structure which are not
included in any of the fields are unaffected. ::

 >>> a = np.zeros(3, dtype=[('a', 'i8'), ('b', 'f4'), ('c', 'S3')])
 >>> b = np.ones(3, dtype=[('x', 'f4'), ('y', 'S3'), ('z', 'O')])
 >>> b[:] = a
 >>> b
 array([(0., b'0.0', b''), (0., b'0.0', b''), (0., b'0.0', b'')],
       dtype=[('x', '<f4'), ('y', 'S3'), ('z', 'O')])


Assignment involving subarrays
``````````````````````````````

When assigning to fields which are subarrays, the assigned value will first be
broadcast to the shape of the subarray.

Indexing Structured Arrays
--------------------------

Accessing Individual Fields
```````````````````````````

Individual fields of a structured array may be accessed and modified by indexing
the array with the field name. ::

 >>> x = np.array([(1, 2), (3, 4)], dtype=[('foo', 'i8'), ('bar', 'f4')])
 >>> x['foo']
 array([1, 3])
 >>> x['foo'] = 10
 >>> x
 array([(10, 2.), (10, 4.)],
       dtype=[('foo', '<i8'), ('bar', '<f4')])

The resulting array is a view into the original array. It shares the same
memory locations and writing to the view will modify the original array. ::

 >>> y = x['bar']
 >>> y[:] = 11
 >>> x
 array([(10, 11.), (10, 11.)],
       dtype=[('foo', '<i8'), ('bar', '<f4')])

This view has the same dtype and itemsize as the indexed field, so it is
typically a non-structured array, except in the case of nested structures.

 >>> y.dtype, y.shape, y.strides
 (dtype('float32'), (2,), (12,))

If the accessed field is a subarray, the dimensions of the subarray
are appended to the shape of the result::

   >>> x = np.zeros((2, 2), dtype=[('a', np.int32), ('b', np.float64, (3, 3))])
   >>> x['a'].shape
   (2, 2)
   >>> x['b'].shape
   (2, 2, 3, 3)

Accessing Multiple Fields
```````````````````````````

One can index and assign to a structured array with a multi-field index, where
the index is a list of field names.

.. warning::
    The behavior of multi-field indexes changed from Numpy 1.15 to Numpy 1.16.

The result of indexing with a multi-field index is a view into the original
array, as follows::

 >>> a = np.zeros(3, dtype=[('a', 'i4'), ('b', 'i4'), ('c', 'f4')])
 >>> a[['a', 'c']]
 array([(0, 0.), (0, 0.), (0, 0.)],
      dtype={'names':['a','c'], 'formats':['<i4','<f4'], 'offsets':[0,8], 'itemsize':12})

Assignment to the view modifies the original array. The view's fields will be
in the order they were indexed. Note that unlike for single-field indexing, the
dtype of the view has the same itemsize as the original array, and has fields
at the same offsets as in the original array, and unindexed fields are merely
missing.

.. warning::
    In Numpy 1.15, indexing an array with a multi-field index returned a copy of
    the result above, but with fields packed together in memory as if
    passed through :func:`numpy.lib.recfunctions.repack_fields`.

    The new behavior as of Numpy 1.16 leads to extra "padding" bytes at the
    location of unindexed fields compared to 1.15. You will need to update any
    code which depends on the data having a "packed" layout. For instance code
    such as::

     >>> a[['a', 'c']].view('i8')  # Fails in Numpy 1.16
     Traceback (most recent call last):
        File "<stdin>", line 1, in <module>
     ValueError: When changing to a smaller dtype, its size must be a divisor of the size of original dtype

    will need to be changed. This code has raised a ``FutureWarning`` since
    Numpy 1.12, and similar code has raised ``FutureWarning`` since 1.7.

    In 1.16 a number of functions have been introduced in the
    :mod:`numpy.lib.recfunctions` module to help users account for this
    change. These are
    :func:`numpy.lib.recfunctions.repack_fields`.
    :func:`numpy.lib.recfunctions.structured_to_unstructured`,
    :func:`numpy.lib.recfunctions.unstructured_to_structured`,
    :func:`numpy.lib.recfunctions.apply_along_fields`,
    :func:`numpy.lib.recfunctions.assign_fields_by_name`,  and
    :func:`numpy.lib.recfunctions.require_fields`.

    The function :func:`numpy.lib.recfunctions.repack_fields` can always be
    used to reproduce the old behavior, as it will return a packed copy of the
    structured array. The code above, for example, can be replaced with:

     >>> from numpy.lib.recfunctions import repack_fields
     >>> repack_fields(a[['a', 'c']]).view('i8')  # supported in 1.16
     array([0, 0, 0])

    Furthermore, numpy now provides a new function
    :func:`numpy.lib.recfunctions.structured_to_unstructured` which is a safer
    and more efficient alternative for users who wish to convert structured
    arrays to unstructured arrays, as the view above is often indeded to do.
    This function allows safe conversion to an unstructured type taking into
    account padding, often avoids a copy, and also casts the datatypes
    as needed, unlike the view. Code such as:

     >>> b = np.zeros(3, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
     >>> b[['x', 'z']].view('f4')
     array([0., 0., 0., 0., 0., 0., 0., 0., 0.], dtype=float32)

    can be made safer by replacing with:

     >>> from numpy.lib.recfunctions import structured_to_unstructured
     >>> structured_to_unstructured(b[['x', 'z']])
     array([0, 0, 0])


Assignment to an array with a multi-field index modifies the original array::

 >>> a[['a', 'c']] = (2, 3)
 >>> a
 array([(2, 0, 3.), (2, 0, 3.), (2, 0, 3.)],
       dtype=[('a', '<i4'), ('b', '<i4'), ('c', '<f4')])

This obeys the structured array assignment rules described above. For example,
this means that one can swap the values of two fields using appropriate
multi-field indexes::

 >>> a[['a', 'c']] = a[['c', 'a']]

Indexing with an Integer to get a Structured Scalar
```````````````````````````````````````````````````

Indexing a single element of a structured array (with an integer index) returns
a structured scalar::

 >>> x = np.array([(1, 2., 3.)], dtype='i, f, f')
 >>> scalar = x[0]
 >>> scalar
 (1, 2., 3.)
 >>> type(scalar)
 <class 'numpy.void'>

Unlike other numpy scalars, structured scalars are mutable and act like views
into the original array, such that modifying the scalar will modify the
original array. Structured scalars also support access and assignment by field
name::

 >>> x = np.array([(1, 2), (3, 4)], dtype=[('foo', 'i8'), ('bar', 'f4')])
 >>> s = x[0]
 >>> s['bar'] = 100
 >>> x
 array([(1, 100.), (3, 4.)],
       dtype=[('foo', '<i8'), ('bar', '<f4')])

Similarly to tuples, structured scalars can also be indexed with an integer::

 >>> scalar = np.array([(1, 2., 3.)], dtype='i, f, f')[0]
 >>> scalar[0]
 1
 >>> scalar[1] = 4

Thus, tuples might be thought of as the native Python equivalent to numpy's
structured types, much like native python integers are the equivalent to
numpy's integer types. Structured scalars may be converted to a tuple by
calling :func:`ndarray.item`::

 >>> scalar.item(), type(scalar.item())
 ((1, 4.0, 3.0), <class 'tuple'>)

Viewing Structured Arrays Containing Objects
--------------------------------------------

In order to prevent clobbering object pointers in fields of
:class:`numpy.object` type, numpy currently does not allow views of structured
arrays containing objects.

Structure Comparison
--------------------

If the dtypes of two void structured arrays are equal, testing the equality of
the arrays will result in a boolean array with the dimensions of the original
arrays, with elements set to ``True`` where all fields of the corresponding
structures are equal. Structured dtypes are equal if the field names,
dtypes and titles are the same, ignoring endianness, and the fields are in
the same order::

 >>> a = np.zeros(2, dtype=[('a', 'i4'), ('b', 'i4')])
 >>> b = np.ones(2, dtype=[('a', 'i4'), ('b', 'i4')])
 >>> a == b
 array([False, False])

Currently, if the dtypes of two void structured arrays are not equivalent the
comparison fails, returning the scalar value ``False``. This behavior is
deprecated as of numpy 1.10 and will raise an error or perform elementwise
comparison in the future.

The ``<`` and ``>`` operators always return ``False`` when comparing void
structured arrays, and arithmetic and bitwise operations are not supported.

Record Arrays
=============

As an optional convenience numpy provides an ndarray subclass,
:class:`numpy.recarray`, and associated helper functions in the
:mod:`numpy.rec` submodule, that allows access to fields of structured arrays
by attribute instead of only by index. Record arrays also use a special
datatype, :class:`numpy.record`, that allows field access by attribute on the
structured scalars obtained from the array.

The simplest way to create a record array is with :func:`numpy.rec.array`::

 >>> recordarr = np.rec.array([(1, 2., 'Hello'), (2, 3., "World")],
 ...                    dtype=[('foo', 'i4'),('bar', 'f4'), ('baz', 'S10')])
 >>> recordarr.bar
 array([ 2.,  3.], dtype=float32)
 >>> recordarr[1:2]
 rec.array([(2, 3., b'World')],
       dtype=[('foo', '<i4'), ('bar', '<f4'), ('baz', 'S10')])
 >>> recordarr[1:2].foo
 array([2], dtype=int32)
 >>> recordarr.foo[1:2]
 array([2], dtype=int32)
 >>> recordarr[1].baz
 b'World'

:func:`numpy.rec.array` can convert a wide variety of arguments into record
arrays, including structured arrays::

 >>> arr = np.array([(1, 2., 'Hello'), (2, 3., "World")],
 ...             dtype=[('foo', 'i4'), ('bar', 'f4'), ('baz', 'S10')])
 >>> recordarr = np.rec.array(arr)

The :mod:`numpy.rec` module provides a number of other convenience functions for
creating record arrays, see :ref:`record array creation routines
<routines.array-creation.rec>`.

A record array representation of a structured array can be obtained using the
appropriate `view <numpy-ndarray-view>`_::

 >>> arr = np.array([(1, 2., 'Hello'), (2, 3., "World")],
 ...                dtype=[('foo', 'i4'),('bar', 'f4'), ('baz', 'a10')])
 >>> recordarr = arr.view(dtype=np.dtype((np.record, arr.dtype)),
 ...                      type=np.recarray)

For convenience, viewing an ndarray as type :class:`np.recarray` will
automatically convert to :class:`np.record` datatype, so the dtype can be left
out of the view::

 >>> recordarr = arr.view(np.recarray)
 >>> recordarr.dtype
 dtype((numpy.record, [('foo', '<i4'), ('bar', '<f4'), ('baz', 'S10')]))

To get back to a plain ndarray both the dtype and type must be reset. The
following view does so, taking into account the unusual case that the
recordarr was not a structured type::

 >>> arr2 = recordarr.view(recordarr.dtype.fields or recordarr.dtype, np.ndarray)

Record array fields accessed by index or by attribute are returned as a record
array if the field has a structured type but as a plain ndarray otherwise. ::

 >>> recordarr = np.rec.array([('Hello', (1, 2)), ("World", (3, 4))],
 ...                 dtype=[('foo', 'S6'),('bar', [('A', int), ('B', int)])])
 >>> type(recordarr.foo)
 <class 'numpy.ndarray'>
 >>> type(recordarr.bar)
 <class 'numpy.recarray'>

Note that if a field has the same name as an ndarray attribute, the ndarray
attribute takes precedence. Such fields will be inaccessible by attribute but
will still be accessible by index.

"""
