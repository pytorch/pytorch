.. _numpy_compatibility:

.. testsetup:: *

   import numpy as np
   import torch

NumPy Compatibility
===================

.. note::

   **TL;DR:**

   - Be explicit about obtaining and using NumPy arrays
   - `This tutorial`_ shows a more concrete example of interoperability between
     NumPy and PyTorch

At the onset it should be stressed that PyTorch tensors and NumPy arrays are
fundamentally different types. However, in the interests of being accessible to
the wider scientific python ecosystem, there are implicit and explicit
conversion methods. The major considerations to keep in mind are:

- **Where** is the structure allocated?
    * This may be on a compute device (e.g. a GPU) or the CPU
- **What** is the underlying data?
    * In particular, several functions will either return a copy or a view
- **How** are modifications to the data handled?
    * The data returned may be read-only, or may be writable

From the end-user perspective, it is most desirable to have zero copy methods,
however, for exploratory analyses and their visualizations in particular, it is
often more important to have guaranteed conversions, even if copies are created.

Basic interoperability
----------------------

NumPy arrays have no conception of the GPU, and so all conversions require a
``cpu`` tensor. Similarly, the **computational graph** cannot be represented in
```numpy``, and so the pre-requisite to have an operation or object without one,
i.e. only leaf nodes are supported.

.. doctest::

   >>> b_torch = torch.ones(3)
   >>> assert(b_torch.requires_grad == False)
   >>> assert(b_torch.is_cuda == False)
   >>> b_torch.numpy()
   array([1., 1., 1.], dtype=float32)

PyTorch will raise exceptions instead of allowing operations that may lead to
incorrect results. As the conversion to a NumPy array may potentially be a lossy
operation (the computational graph), there are no inbuilt operations to do
something which unconditionally returns a NumPy array. Practically speaking, a
function can be created locally to return a NumPy array, for say, visualizations.

.. code-block:: python

   # Possibly lossy function for visualizations
   def uncond_numpy(torch_tensor):
        if torch_tensor.is_cuda:
           torch_tensor = torch_tensor.cpu()
        if torch_tensor.requires_grad == False:
            return torch_tensor.numpy()
        else:
            return torch_tensor.detach().numpy()

Operations
----------

All ``torch`` operators will helpfully fail with a ``TypeError`` if called with
``numpy`` arrays. However, for **numpy functions**, using a ``torch.Tensor`` with
an ``np.ndarray`` will return a ``torch.Tensor`` without creating any additional
copies.

 - Due to the :meth:`torch.Tensor.__array__()` implementation, a
   ``np.ndarray`` which shares memory with the ``torch.Tensor`` is used for the
   operation.
 - The return type functionality is defined by
   :meth:`torch.Tensor.__array_wrap__()`, and calls ``torch.from_numpy()``
   internally.

.. doctest::

   >>> a_np = np.ones(3) # dtype=float64
   >>> b_torch = torch.ones(3) # dtype=float32
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

Only a small subset of data type (``dtype``) objects defined in NumPy have an
equivalent in PyTorch, namely:

.. csv-table:: $ indicates the sizes supported, e.g. ``uint8``
   :header: ``np.dtype``, ``torch.dtype``, sizes

    "``bool_``", "``bool``", "N/A"
    "``uint$``", "``uint$``", ":math:`8`"
    "``int$``", "``int$``", ":math:`8, 16, 32, 64`"
    "``float$``", "``float$``", ":math:`16, 32, 64`"
    "``complex$``", "``complex$``", ":math:`64, 128`"

To ``numpy``
^^^^^^^^^^^^

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

To obtain a **view** of the data, :meth:`torch.from_numpy()` can be used.

.. doctest::

   >>> a_np = np.array([1, 2, 3], dtype = np.float64)
   >>> b_torch = torch.from_numpy(a_np)
   >>> # b_torch = torch.as_tensor(a_np) # see note
   >>> b_torch[2] = 23
   >>> b_torch
   tensor([ 1.,  2., 23.], dtype=torch.float64)
   >>> np.array_equal(b_torch.numpy(), a_np)
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
   >>> b_torch
   tensor([1., 2., 3.], dtype=torch.float64)
   >>> b_torch[2] = 23
   >>> np.array_equal(b_torch.numpy(), a_np)
   False

Special considerations apply when calling NumPy functions on PyTorch tensors.
Where possible, the equivalent PyTorch function is called.

.. note::

   This is due to the fact that ``torch.Tensor.__array_priority__`` is higher
   than the NumPy default of ``0``.

Indexing
^^^^^^^^

For the most part, both simple and "fancy" indexing work as expected. Edge cases
are enumerated here.

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

.. _`This tutorial`: https://pytorch.org/tutorials/advanced/numpy_extensions_tutorial.html
