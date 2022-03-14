.. _numpy_compatibility:

.. testsetup:: *

   import numpy as np
   import torch

NumPy Compatibility
===================

.. note::

   **TL;DR:**

   - Be explicit about obtaining and using NumPy arrays, ``.numpy()`` will always ensure valid arrays
   - `This tutorial`_ shows a more concrete example of interoperability between
     NumPy and PyTorch

PyTorch tensors and NumPy arrays are fundamentally different types, backed by
differing implementations. However, interoperability with NumPy is a top
priority for PyTorch, given that it allows for the bidirectional exchange of
data with external libraries which do not support PyTorch explicitly.

Views, Copies and Data
----------------------

Both PyTorch Tensors and NumPy arrays correspond to blobs of memory on a compute
device. To convert from one object type to another, for any data, we have the
following options:

- We can obtain a **view** of it (reinterpreting an existing blob)
- We can **create** a new blob of memory (allocating new storage)
- We can make a **copy** (also allocates new storage)

From the end-user perspective, it is most desirable to have zero copy methods,
however, for exploratory analyses and their visualizations in particular, it is
often more important to have guaranteed conversions, even if copies are created.

.. doctest::

  >>> a_np = np.ones(3)
  >>> b_fnp = torch.from_numpy(a_np)
  >>> np.shares_memory(a_np, b_torch.numpy())
   False
  >>> np.shares_memory(a_np, b_fnp)
  True

.. warning::

   Do not use ``id`` to check for shared memory, as ``id`` is a CPython only
   implementation detail guaranteed to be unique. Also, for larger arrays,
   ``np.shares_memory`` can be slow.

Basic interoperability
----------------------

When discussing compatibility of data structures and functions, the major considerations to keep in mind are:

- **Where** is the structure allocated?
    * This may be on a compute device (e.g. GPU, TPU) or the CPU.
- **What** does a function return?
    * This will either be a copy, view or a new instance.
- **How** are modifications to the data handled?
    * The data may be read-only, or may be writable

The **computational graph** [#cgdef]_ cannot be represented in ``numpy``, and so
the pre-requisite to have an operation or object without one, i.e. only leaf
nodes are supported.

.. doctest::

   >>> b_torch = torch.ones(3)
   >>> assert(b_torch.requires_grad == False)
   >>> assert(b_torch.is_cuda == False)
   >>> b_torch.numpy()
   array([1., 1., 1.], dtype=float32)

Although this returned object is *equal* to a NumPy array, it is not equivalent
to an existing array, which means that it does not share the same memory as an
existing object. We can obtain a **view** or interpretation of a Tensor from
NumPy array as well.

The conversion to a NumPy array may potentially be a lossy operation (the
computational graph), and so there are also no inbuilt operations to
unconditionally return a NumPy array. Practically speaking, a function can be
created locally to return a NumPy array, for say, visualizations.

.. code-block:: python

   # Possibly lossy function for visualizations
   def uncond_numpy(torch_tensor):
        if torch_tensor.device != torch.device(type="cpu"):
           torch_tensor = torch_tensor.cpu()
        if torch_tensor.is_conj() == True:
           torch_tensor = torch_tensor.resolve_conj()
        if torch_tensor.is_neg() == True:
           torch_tensor = torch_tensor.resolve_neg()
        if torch_tensor.requires_grad == False:
            return torch_tensor.numpy()
        else:
            return torch_tensor.detach().numpy()

Operations
----------

All ``torch`` operators will helpfully fail with a ``TypeError`` if called with
``numpy`` arrays. However, for **numpy operators**, using a ``torch.Tensor``
with an ``np.ndarray`` will return a ``torch.Tensor``.

 - Due to the :meth:`torch.Tensor.__array__()` implementation, a
   ``np.ndarray`` which shares memory with the ``torch.Tensor`` is used for the
   operation.
 - The return type functionality is defined by
   :meth:`torch.Tensor.__array_wrap__()`, and calls ``torch.from_numpy()``
   internally.

As a concrete example, consider the following snippet:

.. doctest::

   >>> a_np = np.ones(3)
   >>> a_np.dtype
   dtype('float64')
   >>> b_torch = torch.ones(3)
   >>> b_torch.dtype
   torch.float32
   >>> torch.add(a_np, b_torch)
   Traceback (most recent call last):
   ...
   TypeError: add(): argument 'input' (position 1) must be Tensor, not numpy.ndarray
   >>> b_torch + a_np
   tensor([2., 2., 2.], dtype=torch.float64)
   >>> a_np + b_torch
   Traceback (most recent call last):
   ...
   TypeError: Concatenation operation is not implemented for NumPy arrays, use np.concatenate() instead. Please do not rely on this error; it may not be given on all Python implementations.
   >>> np.add(a_np, b_torch)
   tensor([2., 2., 2.], dtype=torch.float64)

.. dropdown:: Code path and extended explanation

              - The `Python data model`_ specifies that the ``__radd__`` function is to be
                called when the operands do not both implement compatible ``__add__``, so as a
                Tensor does not support addition with an ``ndarray``, it is the concatenation
                opration which is called instead of addition. This explains the result of
                ``a_np + b_torch``--> ``a_np.__add__(b_torch)``--> **NotImplemented** -->
                ``a_np.__radd__(b_torch)`` which returns a Tensor.

              - For ``b_torch + a_np``, it is ``a_np.__add__`` which is called, and this takes
                an "array-like", so a view of the Tensor is converted to a NumPy array (a
                no-op); subsequently, the returned object is still a Tensor, because of the
                ``__array_wrap__`` and ``__array_priority__``

              Recall that ``torch.Tensor.__array_priority__`` is higher than the NumPy
              default of ``0``, which means in keeping with `NEP 13`_ the returned object
              from a NumPy function will be a PyTorch Tensor.

              .. note::

                    The semantics of this conversion is defined formally in NumPy `NEP 18`_. In
                    particular, the dunder methods are described in `Version 3 of the NumPy Array
                    Interface`_. The exact order in which NumPy attempts to convert a foreign
                    object is described in the `interoperability with NumPy`_ document.

If it is absolutely necessary to write functions where the input objects are not
unconditionally known to be either PyTorch tensors or NumPy arrays, it is
possible to ensure operator functionality by using NumPy functions explicitly as
they are more forgiving than their PyTorch equivalents.

.. csv-table::
   :header: Operator, NumPy Function, Description

   "``+``", "``np.add()``", "Addition"
   "``-``", "``np.subtract()``", "Subtraction"
   "``*``", "``np.multiply()``", "Multiplication"
   "``/``", "``np.divide()``", "Division"

Conversions
-----------

A subset of data type (``dtype``) objects defined in NumPy have
equivalents in PyTorch, namely:

.. csv-table:: $ indicates the sizes supported, e.g. ``uint8``
   :header: ``np.dtype``, ``torch.dtype``, sizes

    "``bool_``", "``bool``", "N/A"
    "``uint$``", "``uint$``", ":math:`8`"
    "``int$``", "``int$``", ":math:`8, 16, 32, 64`"
    "``float$``", "``float$``", ":math:`16, 32, 64`"
    "``complex$``", "``complex$``", ":math:`64, 128`"

To ``numpy``
^^^^^^^^^^^^

The restrictions on a PyTorch tensor becoming a NumPy ``ndarray`` are:

- It must be a strided tensor
- It must be on the CPU
- It must not require gradients
- It must not have the conjugate bit set
- It must not have the negative bit set
- It must not be a tensor-subclass

Essentially these can be expressed as:

.. code-block:: python

   # t is a torch.Tensor
   assert(t.layout == torch.strided) # Dense
   assert(t.is_cuda == False) # CPU
   assert(t.requires_grad == False) # No autograd
   assert(t.is_conj() == False) # Not conjugate
   assert(t.is_neg() == False) # Not negative

The :meth:`torch.numpy()` method  and the :doc:`np.asarray()
<numpy:reference/generated/numpy.asarray>` function returns a **view** of the
underlying tensor as a ``np.ndarray`` object.

.. doctest::

    >>> b_torch = torch.ones(3)
    >>> b_torch.numpy()[2] = 32
    >>> b_torch
    tensor([ 1.,  1., 32.])
    >>> a_np = np.array([1, 1, 32], dtype = np.float32)
    >>> np.array_equal(b_torch.numpy(), a_np) # True
    True
    >>> c_tmp = np.asarray(b_torch, dtype = np.float32) # No copy if same dtype
    >>> c_tmp
    array([ 1.,  1., 32.], dtype=float32)
    >>> c_tmp[2] = 1.
    >>> b_torch
    tensor([1., 1., 1.])

.. note::

   Since ``np.asarray()`` depends on the implementation of
   ``torch.Tensor.__array__()`` which calls ``torch.numpy()``, the **leaf node**
   requirement still needs to be satisfied by the user, i.e., ``requires_grad ==
   False``

From ``numpy``
^^^^^^^^^^^^^^

For a NumPy ``ndarray`` to be convertible to a PyTorch tensor:

- It must have only native byte order
- Array strides must be multiples of the Torch element byte size
- Must have a ``dtype`` which is one of ``float64 float32 float16 complex64
  complex128 int64 int32 int16 int8 uint8 and bool``
- Non-writable arrays will result in undefined behavior, and should be avoided
   + Copies should be made instead

Concretely, these may be expressed as:

.. doctest::

   >>> a_np = np.ones(4).reshape(2, 2)
   >>> b_torch = torch.tensor(a_np)
   >>> assert(a_np.dtype.byteorder == '=') # Native byte order
   True
   >>> assert(a_np.flags.writeable == True) # Not read only
   True
   >>> np.equal([stride % b_torch.element_size() for stride in a_np.strides], np.zeros(len(a_np.strides))) # Multiples of torch element byte size
   array([ True,  True])
   >>> a_np.dtype in ["float64", "float32", "float16", "complex64", "complex128", "int64", "int32", "int16", "int8", "uint8", "bool"] # Supported dtype
   True

To obtain a **view** of the data, :meth:`torch.from_numpy()` can be used.

.. doctest::

   >>> a_np = np.array([1, 2, 3], dtype = np.float64)
   >>> b_torch = torch.from_numpy(a_np)
   >>> # b_torch = torch.as_tensor(a_np) # see note
   >>> b_torch[2] = 23
   >>> b_torch
   tensor([ 1.,  2., 23.], dtype=torch.float64)
   >>> a_np[0] = 22
   >>> b_torch # view, changes with a_np
   tensor([ 22.,  2., 23.], dtype=torch.float64)
   >>> np.array_equal(b_torch.numpy(), a_np)
   True
   >>> np.shares_memory(a_np, b_torch)
   True

- :meth:`torch.from_numpy()` is guaranteed to share memory with NumPy.
- :meth:`torch.as_tensor()` will try to stay away from copy operations, it
  also has the effect of sharing memory. However, ``torch.as_tensor()`` has
  slightly higher overhead as it checks and accepts other iteratable objects as
  well, e.g. ``list`` objects.
- :meth:`torch.from_dlpack()` called with a NumPy array (``np.version.version >=
  1.20``) as its argument will also generate a ``torch.Tensor`` view.

To obtain a **copy** of the ``ndarray`` object, and not share memory, the
:meth:`torch.tensor()` constructor accepts :meth:`np.ndarray` objects as a data
source to construct and return a ``torch.Tensor``.

.. note::

   Recall that, if ``x`` is a tensor, ``torch.tensor(x)`` is equivalent to
    ``x.clone().detach()``.

.. doctest::

   >>> a_np = np.array([1, 2, 3], dtype = np.float64)
   >>> b_torch = torch.tensor(a_np)
   >>> np.shares_memory(a_np, b_torch)
   False

Alternatively, calling ``copy``  after conversion will also make a copy.

.. doctest::

   >>> a_np = np.array([1, 2, 3], dtype = np.float64)
   >>> b_fnp = torch.from_numpy(a_np)
   >>> b_fnp
   tensor([1., 2., 3.], dtype=torch.float64)
   >>> np.shares_memory(a_np, b_fnp)
   True
   >>> np.shares_memory(a_np, b_fnp.numpy())
   True
   >>> np.shares_memori(a_np, b_fnp.numpy().copy())
   False

The DLPack interface
^^^^^^^^^^^^^^^^^^^^

.. note::

   This requires NumPy v1.23


Since both PyTorch tensors and NumPy arrays have the ``__dlpack__`` method
defined, we can use the ``from_dlpack`` methods to obtain a view of the data.

.. doctest::

   >>> b_torch = torch.ones(3)
   >>> np.shares_memory(np.from_dlpack(b_torch), b_torch.numpy())
   True
   >>> a_np = np.ones(3)
   >>> np.shares_memory(torch.from_dlpack(a_np).numpy(), a_np)
   True

.. dropdown:: Fine print and references

   - `DLPack specification`_
   - `NumPy DLPack implementation`_
   - `PyTorch DLPack implementation`_

   .. warning::

      The specification calls for a read-only view, but PyTorch does not support
      immutable arrays (`see issue 44027`_).

Calling NumPy Functions on PyTorch
----------------------------------

Operators aside, **most** NumPy functions can be called on CPU PyTorch tensors as well.
This is because NumPy ``ufuncs`` or universal functions (described fully `in the
NumPy documentation`_), take "array-like" inputs, and return tensor objects due
to the dunder method ``__array_wrap__``.

.. doctest::

   >>> a_np = np.array([1, 2, 3], dtype=np.float64) / 5
   >>> np.arctan2(a_np, 1) # No equivalent torch function
   array([0.19866933, 0.38941834, 0.56464247])
   >>> b_torch = torch.tensor(a_np)
   >>> np.arctan2(b_torch, 1)
   tensor([0.1974, 0.3805, 0.5404], dtype=torch.float64)

.. warning:: Important exceptions

   Calling a NumPy function which calls a method on the object passed will fail.
   This includes: ``np.sum``, ``np.mean``, ``np.min``, ``np.max``, ``np.std``,
   ``np.amin``, ``np.amax``.

Essentially, the code execution path is similar to the operator resolution, that is:

- The PyTorch tensor is converted to a NumPy array
- The NumPy function is executed
- A PyTorch tensor is returned

Conversely, no PyTorch functions will work on any NumPy array without explictly
generating either a tensor `or a subclass`_. This is by design, as the NumPy
array is not equivalent to a Torch tensor without additional guidance (e.g.
Torch tensors may live on non-CPU compute devices).

Indexing
^^^^^^^^

Indexing operations will typically work as expected. This includes both "fancy"
and "simple" indexing operations as defined in `NEP 21`_.

.. doctest::

   >>> b_torch = torch.tensor([1, 2, 3, 4, 5])
   >>> b_torch
   tensor([1, 2, 3, 4, 5])
   >>> b_torch[2]
   tensor(3)
   >>> b_torch[-1]
   tensor(5)
   >>> b_torch[2:-1]
   tensor([3, 4])
   >>> torch.take(b_torch, torch.tensor([3, 2]))
   tensor([4, 3])

NumPy arrays can also be used for indexing.

.. doctest::

   >>> b_torch = torch.tensor([1, 2, 3, 4, 5])
   >>> a_np = np.ones(3)
   >>> b_torch[a_np]
   tensor([2, 2, 2])

Further inter-operability can be found in this `NumPy-PyTorch cheatsheet`_.

.. warning::

   It **is not** recommended to mix objects for indexing either.

Negative strides
~~~~~~~~~~~~~~~~

NumPy arrays may have negative strides, which is not true for PyTorch tensors.

.. doctest::

   >>> a_np = np.array([1, 2, 3])
   >>> b_torch = torch.from_numpy(a_np[::-1]) # doctest: +SKIP
   Traceback (most recent call last):
   ...
   ValueError: At least one stride in the given numpy array is negative, and tensors with negative strides are not currently supported. (You can probably work around this by making a copy of your array  with array.copy().)
   >>> b_torch = torch.from_numpy(np.ascontiguousarray(a_np[::-1]))
   >>> b_torch
   tensor([3, 2, 1])

Conclusions
-----------

NumPy compatibility is a moving target, but aside from the edge cases documented here, the PyTorch project, like Python itself strives to provide the "least surprising" result via implicit conversions.

That said, recall from ``import this``, the Zen of Python:

    Explicit is better than implict

So the recommended solution is to always explicitly convert PyTorch tensors and
NumPy arrays as required.

.. warning::

   This document is for vanilla PyTorch tensors and does not cover `extended tensors`_.

.. dropdown:: Optional details

   **Historical Aside**

   `NEP 13`_ and `NEP 18`_, define ``__array_ufunc__`` and ``__array_function__``
   respectively. Neither of these have been implemented in PyTorch, and since these
   mechanisms have largely been replaced by newer approaches, they are unlikely to
   be included.

   **The Array API**

   The existing NumPy API is far too forgiving about accepting foreign objects
   which can be coerced to an ``array_like``. To address this, `NEP 47`_ defines
   the ``array_api`` namespace and associated functions in keeping with the `Python
   array API standard`_. Eventual adoption of this standard will ensure more usage
   consistency.

   **Tracking provenance**

   Given a NumPy array which is a view of existing data, it may be required to
   determine its provenance. This can be obtained by calling ``base``. ``base``
   will default to returning ``None`` when called on an object which owns its own
   memory, i.e. is not a view.

   .. doctest::

    >>> a_np = np.ones(3)
    >>> b_torch = torch.ones(3)
    >>> np.from_dlpack(b_torch).base
    <capsule object "numpy_dltensor" at ...>
    >>> a_np.base is None
    True

   Note that the results of ``base`` cannot be relied on for more than one level of
   indirection. This means that given a tensor which shares memory with a NumPy
   array, calling ``base`` will return a tensor, not the underlying array.

   .. doctest::

    >>> np.shares_memory(torch.from_numpy(a_np), a_np)
    True
    >>> np.shares_memory(torch.from_numpy(a_np).numpy(), a_np)
    True
    >>> torch.from_numpy(a_np).numpy().base # Unintuitive
    tensor([1., 1., 1.], dtype=torch.float64)

.. rubric:: **Footnotes**

.. [#cgdef] A computational graph is used whenever gradients are to be computed. It consists (roughly) of a series of operations and data in a directed acyclic graph. This is described in more detail in `the introduction to torch.autograd tutorial`_

.. _This tutorial: https://pytorch.org/tutorials/advanced/numpy_extensions_tutorial.html
.. _Version 3 of the NumPy Array Interface: https://numpy.org/doc/stable/reference/arrays.interface.html
.. _NEP 18: https://numpy.org/neps/nep-0018-array-function-protocol.html
.. _Python data model: https://docs.python.org/3/reference/datamodel.html#emulating-numeric-types
.. _NumPy-PyTorch cheatsheet: https://pytorch-for-numpy-users.wkentaro.com/
.. _in the NumPy documentation: https://numpy.org/doc/stable/reference/ufuncs.html
.. _or a subclass: https://pytorch.org/docs/stable/notes/extending.html#subclassing-torch-tensor
.. _NEP 47: https://numpy.org/neps/nep-0047-array-api-standard.html
.. _NEP 13: https://numpy.org/neps/nep-0013-ufunc-overrides.html
.. _NEP 18: https://numpy.org/neps/nep-0018-array-function-protocol.html
.. _NEP 21: https://numpy.org/neps/nep-0021-advanced-indexing.html
.. _NEP 37: https://numpy.org/neps/nep-0037-array-module.html
.. _interoperability with NumPy: https://numpy.org/devdocs/user/basics.interoperability.html
.. _the introduction to torch.autograd tutorial: https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html
.. _Python array API standard: https://data-apis.org/array-api/latest/purpose_and_scope.html#this-api-standard
.. _DLPack specification: https://dmlc.github.io/dlpack/latest/python_spec.html
.. _NumPy DLPack implementation: https://numpy.org/devdocs/reference/generated/numpy.from_dlpack.html
.. _PyTorch DLPack implementation: https://pytorch.org/docs/stable/dlpack.html
.. _see issue 44027: https://github.com/pytorch/pytorch/issues/44027_
.. _extended tensors: https://pytorch.org/docs/stable/notes/extending.html#extending-torch_
